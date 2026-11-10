"""Command-line entry point for the Model Headroom Analyzer. Parses CLI arguments, loads model data, runs diagnostic signals, and generates terminal or JSON reports."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, cast

from huggingface_hub import hf_hub_download

from model_headroom.analysis.orchestrator import run_analysis
from model_headroom.analysis.scoring import decision_grade_gaps
from model_headroom.device import resolve_device
from model_headroom.download import download_dataset_files, resolve_data_dir
from model_headroom.models.types import SamplingMode, WeightSample
from model_headroom.reporting.human_report import generate_human_report
from model_headroom.reporting.json_report import generate_json_report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parses command-line arguments for model-headroom. Returns an argparse.Namespace with model config params including model name, model path, model type settings, analysis mode, output format, and optional data file paths."""
    parser = argparse.ArgumentParser(
        prog="model-headroom",
        description="Model Headroom Analyzer — diagnostic signals for training headroom",
    )
    parser.add_argument(
        "--model-name",
        required=False,
        help="Name of the model to analyze (required unless --model-id is given)",
    )
    parser.add_argument(
        "--model-path",
        required=False,
        help="Filesystem path to the model directory (required unless --model-id is given)",
    )
    parser.add_argument(
        "--model-id",
        help="Hugging Face model id (e.g. Qwen/Qwen3-4B). When given, downloads "
        "datasets from {model-id}-headroom on HF Hub (repo_type=dataset) and "
        "runs analysis without needing --model-name/--model-path.",
    )
    parser.add_argument(
        "--dataset-family",
        choices=["dense", "moe", "auto"],
        default="auto",
        help="Dataset family for download: dense (single expert), moe (full expert "
        "data), or auto (detect from config, default: auto)",
    )
    parser.add_argument(
        "--data-dir",
        help="Directory for downloaded datasets (default: ~/.cache/headroom/datasets/, "
        "overridable via HEADROOM_DATA_DIR env var)",
    )
    parser.add_argument(
        "--architecture",
        choices=["auto", "moe", "dense"],
        default="auto",
        help="Model architecture. auto detects from config (default: auto)",
    )
    parser.add_argument(
        "--loss-curve",
        help="Path to loss curve JSON file",
    )
    parser.add_argument(
        "--isoflop",
        help="Path to isoFLOP data JSON file",
    )
    parser.add_argument(
        "--activations",
        help="Path to expert_activations.json from moe_expert_profiler.py",
    )
    parser.add_argument(
        "--weight-pairs-file",
        help="Path to weight_pairs.json — pre-extracted gradient/weight tensor pairs "
        "for effective_rank and grad_weight_ratio signals. "
        'JSON format: {"pairs": [{"gradient": [[...]], "weight": [[...]], "layer_name": "..."}]}',
    )
    parser.add_argument(
        "--mlp-activations-file",
        help="Path to mlp_activations.json — pre-extracted activation tensors "
        "for dead_neurons signal. "
        'JSON format: {"layers": [{"activations": [[[...]]], "layer_name": "..."}]}',
    )
    parser.add_argument(
        "--expert-weights-file",
        help="Path to expert_weights.json — pre-extracted expert weight matrices "
        "for specialization signal. "
        'JSON format: {"layers": [{"expert_weights": [[...]], "layer_name": "..."}]}',
    )
    parser.add_argument(
        "--analysis-mode",
        choices=["diagnostic", "decision_grade"],
        default="diagnostic",
        help="Analysis mode: diagnostic allows partial/synthetic, decision_grade requires full measured (default: diagnostic)",
    )
    parser.add_argument(
        "--sampling-mode",
        choices=[
            "full_matrix",
            "randomized_svd",
            "multi_tile",
            "single_tile",
            "external",
            "unknown",
        ],
        default=None,
        help="Sampling scheme for effective_rank (declared in report; required "
        "for decision_grade beyond unknown)",
    )
    parser.add_argument(
        "--rank-sample-count",
        type=int,
        default=None,
        help="Number of tiles/samples behind the weight-pairs file (feeds the "
        "effective_rank confidence ladder)",
    )
    parser.add_argument(
        "--rank-parent-shapes",
        help="Path to JSON dict {layer_name: [rows, cols]} of full parent "
        "tensor shapes (feeds effective_rank coverage)",
    )
    parser.add_argument(
        "--gradient-provenance-file",
        help="Path to JSON with gradient provenance metadata (model_revision, dataset_hash, etc.)",
    )
    parser.add_argument(
        "--mlp-source-type",
        choices=[
            "measured",
            "synthetic",
            "fixture",
            "estimated",
            "imported",
            "unknown",
        ],
        help="Override MLP activations source_type provenance",
    )
    parser.add_argument(
        "--weight-pairs-source-type",
        choices=[
            "measured",
            "synthetic",
            "fixture",
            "estimated",
            "imported",
            "unknown",
        ],
        help="Override weight pairs source_type provenance",
    )
    parser.add_argument(
        "--gradient-diagnostics",
        action="store_true",
        help="Compute optional gradient diagnostics (effective rank, coherence, module families)",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help="Device for tensor computation: auto uses the first available "
        "accelerator (cuda preferred), else cpu (default: auto)",
    )
    parser.add_argument(
        "--output-format",
        choices=["json", "human"],
        default="human",
        help="Output format (default: human)",
    )
    parser.add_argument(
        "--output-file",
        help="Write report to file instead of stdout",
    )
    return parser.parse_args(argv)


def _data_file_error(kind: str, path: str, flag: str, fmt: str, cause: str) -> str:
    """Build an instructive data-file error carrying the cause, the offending
    path, the flag to pass, and where to find a valid example."""
    return (
        f"Error loading {kind} from {path}: {cause}. "
        f"Use {flag} to provide a {fmt} file; "
        f"see src/model_headroom/download.py for the expected JSON format."
    )


def _gguf_weight_samples(
    model_path: str, sampling_mode: str, max_rows: int | None, device: str
) -> list[WeightSample] | None:
    path = Path(model_path)
    if not path.is_file() or path.suffix.lower() != ".gguf":
        return None
    try:
        from model_headroom.models.gguf_loader import load_gguf_config
        from model_headroom.models.ternary_sampling import (
            sample_weight_for_effective_rank,
        )
    except ImportError:
        print(
            f"warning: cannot sample GGUF weights from {model_path}: "
            "install the [gguf] extra",
            file=sys.stderr,
        )
        return None
    try:
        config = load_gguf_config(path)
    except Exception as e:
        print(
            f"warning: cannot sample GGUF weights from {model_path}: {e}",
            file=sys.stderr,
        )
        return None
    block_count = config.block_count
    layers = sorted({0, block_count // 2, block_count - 1})
    layers = [i for i in layers if 0 <= i < block_count]
    rows = max_rows if isinstance(max_rows, int) and max_rows > 0 else 256
    selected: list[tuple[str, Any, tuple[int, int]]] | None = None
    for proj in ("ffn_gate", "ffn_up", "ffn_down"):
        tensor_names = [f"blk.{i}.{proj}.weight" for i in layers]
        try:
            selected = sample_weight_for_effective_rank(
                model_path, tensor_names, max_rows=rows, device=device
            )
        except Exception as e:
            print(
                f"warning: cannot sample GGUF weights from {model_path}: {e}",
                file=sys.stderr,
            )
            return None
        if selected:
            break
    if not selected:
        print(
            f"warning: no GGUF weight samples for {model_path}: "
            "tried ffn_gate, ffn_up, ffn_down",
            file=sys.stderr,
        )
        return None
    samples: list[WeightSample] = []
    for layer_name, tensor, parent_shape in selected:
        samples.append(
            WeightSample(
                weight=tensor,
                layer_name=layer_name,
                parent_shape=parent_shape,
                truncated=(tensor.shape[0] < parent_shape[0]),
                sampling_mode=cast(
                    "SamplingMode",
                    "single_tile" if sampling_mode == "unknown" else sampling_mode,
                ),
            )
        )
    return samples


def main(argv: list[str] | None = None) -> int:
    # Maps the model type flag to None when auto-detection is selected; otherwise forwards the explicit model type to run_analysis.
    """Drives the Model Headroom analysis pipeline: loads model data from provided paths, runs the analysis signals, checks decision-grade requirements if applicable, and generates terminal or JSON reports. Returns exit code (0 success, 1 error, 2 decision-grade not met)."""
    args = parse_args(argv)

    device = resolve_device(args.device)
    if device is None:
        print(
            f"Error: --device {args.device} requested but no {args.device} "
            "device is available",
            file=sys.stderr,
        )
        return 1

    temp_config_dir: Path | None = None

    if args.model_id is not None:
        if args.model_name is not None or args.model_path is not None:
            print(
                "Error: --model-id cannot be combined with --model-name/--model-path",
                file=sys.stderr,
            )
            return 1
        args.model_name = args.model_id

        config_dir = Path(tempfile.mkdtemp(prefix="headroom-model-"))
        temp_config_dir = config_dir
        try:
            hf_hub_download(
                repo_id=args.model_id,
                filename="config.json",
                local_dir=str(config_dir),
            )
        except Exception as e:
            print(
                f"Warning: config download failed for {args.model_id}: {e}; "
                "proceeding with unknown architecture",
                file=sys.stderr,
            )
        args.model_path = str(config_dir)

        # Resolve "auto" dataset family based on detected architecture so
        # expert data is not downloaded for dense models.
        dataset_family = args.dataset_family
        if dataset_family == "auto":
            from model_headroom.models.loader import detect_architecture

            detected = detect_architecture(args.model_path)
            if detected == "dense":
                dataset_family = "dense"

        data_dir = resolve_data_dir(args.data_dir)
        downloaded = download_dataset_files(
            model_id=args.model_id,
            dataset_family=dataset_family,
            data_dir=str(data_dir),
        )

        args.weight_pairs_file = str(downloaded["weight_pairs"])
        args.mlp_activations_file = str(downloaded["mlp_activations"])
        if "expert_activations" in downloaded:
            args.activations = str(downloaded["expert_activations"])
        if "expert_weights" in downloaded:
            args.expert_weights_file = str(downloaded["expert_weights"])
    elif args.model_name is None or args.model_path is None:
        print(
            "error: model id or model name+path required\n"
            "  Cause:  neither --model-id nor both --model-name and --model-path were provided\n"
            "  Action: Pass --model-id or both --model-name and --model-path\n"
            "  Exit:   1",
            file=sys.stderr,
        )
        return 1

    # Validate model-path: non-empty, exists, and is a directory (F-005, F-007, F-008)
    if not args.model_path:
        print(
            "error: --model-path must not be empty\n"
            "  Cause:  --model-path value is empty\n"
            "  Action: Pass a non-empty model path\n"
            "  Exit:   1",
            file=sys.stderr,
        )
        return 1
    model_path_obj = Path(args.model_path)
    if not model_path_obj.exists():
        print(
            "error: --model-path does not exist\n"
            f"  Cause:  path does not exist: {args.model_path}\n"
            "  Action: Pass an existing model path\n"
            "  Exit:   1",
            file=sys.stderr,
        )
        return 1
    if model_path_obj.suffix != ".gguf" and not model_path_obj.is_dir():
        print(
            "error: --model-path is not a directory\n"
            f"  Cause:  path is not a directory and does not end in .gguf: {args.model_path}\n"
            "  Action: Pass a directory or a .gguf file\n"
            "  Exit:   1",
            file=sys.stderr,
        )
        return 1

    loss_curve = None
    isoflop_points = None
    activation_data = None

    if args.loss_curve:
        from model_headroom.models.types import LossDataPoint

        try:
            with Path(args.loss_curve).open() as f:
                data = json.load(f)
            loss_curve = [LossDataPoint(**pt) for pt in data]
        except Exception as e:
            print(
                "error: loading loss curve\n"
                f"  Cause:  {e}\n"
                "  Action: Check the loss curve JSON file\n"
                "  Exit:   1",
                file=sys.stderr,
            )
            return 1

    if args.isoflop:
        from model_headroom.models.types import IsoFLOPPoint

        try:
            with Path(args.isoflop).open() as f:
                data = json.load(f)
            isoflop_points = [IsoFLOPPoint(**pt) for pt in data]
        except Exception as e:
            print(
                "error: loading isoFLOP data\n"
                f"  Cause:  {e}\n"
                "  Action: Check the isoFLOP JSON file\n"
                "  Exit:   1",
                file=sys.stderr,
            )
            return 1

    if args.activations:
        from model_headroom.models.loader import load_activations_diagnostic

        outcome = load_activations_diagnostic(args.activations)
        if outcome.error is not None:
            print(
                _data_file_error(
                    "activations",
                    args.activations,
                    "--activations",
                    "expert_activations JSON",
                    outcome.error,
                ),
                file=sys.stderr,
            )
            return 1
        activation_data = outcome.data

    weight_pairs = None
    mlp_activations = None
    expert_weight_sets = None

    if args.weight_pairs_file:
        from model_headroom.models.loader import load_weight_pairs_diagnostic

        outcome = load_weight_pairs_diagnostic(args.weight_pairs_file)
        if outcome.error is not None:
            print(
                _data_file_error(
                    "weight pairs",
                    args.weight_pairs_file,
                    "--weight-pairs-file",
                    "weight_pairs JSON",
                    outcome.error,
                ),
                file=sys.stderr,
            )
            return 1
        weight_pairs = outcome.data

    weight_samples = None
    if args.model_path and not args.weight_pairs_file:
        weight_samples = _gguf_weight_samples(
            args.model_path, args.sampling_mode, args.rank_sample_count, device
        )

    if args.mlp_activations_file:
        from model_headroom.models.loader import load_mlp_activations_diagnostic

        outcome = load_mlp_activations_diagnostic(args.mlp_activations_file)
        if outcome.error is not None:
            print(
                _data_file_error(
                    "MLP activations",
                    args.mlp_activations_file,
                    "--mlp-activations-file",
                    "mlp_activations JSON",
                    outcome.error,
                ),
                file=sys.stderr,
            )
            return 1
        mlp_activations = outcome.data

    if args.expert_weights_file:
        from model_headroom.models.loader import load_expert_weights_diagnostic

        outcome = load_expert_weights_diagnostic(args.expert_weights_file)
        if outcome.error is not None:
            print(
                _data_file_error(
                    "expert weights",
                    args.expert_weights_file,
                    "--expert-weights-file",
                    "expert_weights JSON",
                    outcome.error,
                ),
                file=sys.stderr,
            )
            return 1
        expert_weight_sets = outcome.data

    parent_shapes: dict[str, tuple[int, int]] | None = None
    if args.rank_parent_shapes:
        try:
            with Path(args.rank_parent_shapes).open() as f:
                raw: dict[str, Any] = json.load(f)
            parent_shapes = {k: (int(v[0]), int(v[1])) for k, v in raw.items()}
        except Exception as e:
            print(
                "error: loading parent shapes\n"
                f"  Cause:  {e}\n"
                "  Action: Check the parent shapes JSON file\n"
                "  Exit:   1",
                file=sys.stderr,
            )
            return 1

    gradient_provenance = None
    if args.gradient_provenance_file:
        from model_headroom.models.loader import load_gradient_provenance_diagnostic

        outcome = load_gradient_provenance_diagnostic(args.gradient_provenance_file)
        if outcome.error is not None:
            print(
                _data_file_error(
                    "gradient provenance",
                    args.gradient_provenance_file,
                    "--gradient-provenance-file",
                    "gradient provenance JSON",
                    outcome.error,
                ),
                file=sys.stderr,
            )
            return 1
        gradient_provenance = outcome.data
    elif args.weight_pairs_source_type:
        from model_headroom.models.types import GradientProvenance

        gradient_provenance = GradientProvenance(
            source_type=args.weight_pairs_source_type,
            verification_status="verified"
            if args.weight_pairs_source_type == "measured"
            else "unverified",
        )
    elif weight_samples and gradient_provenance is None:
        from model_headroom.models.types import GradientProvenance

        gradient_provenance = GradientProvenance(
            source_type="fixture",
            verification_status="unverified",
        )

    if (
        args.weight_pairs_file
        and args.weight_pairs_source_type is None
        and gradient_provenance is None
    ):
        from model_headroom.models.loader import infer_source_type_from_path

        st, vs = infer_source_type_from_path(args.weight_pairs_file)
        if st != "measured" or vs != "verified":
            from model_headroom.models.types import GradientProvenance

            gradient_provenance = GradientProvenance(
                source_type=st, verification_status=vs
            )

    mlp_st: Any = args.mlp_source_type
    mlp_vs: Any = None
    if mlp_st is not None:
        mlp_vs = "verified" if mlp_st == "measured" else "unverified"
    elif args.mlp_activations_file:
        from model_headroom.models.loader import load_mlp_activations_with_provenance

        _, inferred_st, inferred_vs = load_mlp_activations_with_provenance(
            args.mlp_activations_file
        )
        mlp_st = inferred_st
        mlp_vs = inferred_vs

    try:
        report = run_analysis(
            model_name=args.model_name,
            model_path=args.model_path,
            loss_curve=loss_curve,
            isoflop_points=isoflop_points,
            activation_data=activation_data,
            weight_pairs=weight_pairs,
            mlp_activations=mlp_activations,
            expert_weight_sets=expert_weight_sets,
            architecture=None if args.architecture == "auto" else args.architecture,
            analysis_mode=args.analysis_mode,
            gradient_provenance=gradient_provenance,
            mlp_source_type=mlp_st,
            mlp_verification_status=mlp_vs,
            effective_rank_sampling_mode=args.sampling_mode,
            effective_rank_sample_count=args.rank_sample_count,
            effective_rank_parent_shapes=parent_shapes,
            weight_samples=weight_samples,
            compute_gradient_diagnostics=args.gradient_diagnostics,
            device=device,
        )
    except Exception as e:
        print(
            "error: analysis failed\n"
            f"  Cause:  {e}\n"
            "  Action: Check model path and signal files\n"
            "  Exit:   1",
            file=sys.stderr,
        )
        return 1
    finally:
        if temp_config_dir is not None:
            shutil.rmtree(temp_config_dir, ignore_errors=True)

    if args.analysis_mode == "decision_grade" and not report.decision_grade:
        gaps = decision_grade_gaps(report)
        print("DECISION-GRADE NOT MET:", file=sys.stderr)
        for gap in gaps:
            print(f"  - {gap}", file=sys.stderr)
        return 2

    if args.output_format == "json":
        output = generate_json_report(report)
    else:
        output = generate_human_report(report)

    if args.output_file:
        output_path = Path(args.output_file)
        try:
            # Atomic write (F-012): temp file in same directory, then os.replace
            fd, tmp_path = tempfile.mkstemp(
                dir=output_path.parent, prefix=".sia-output-", suffix=".tmp"
            )
            try:
                with os.fdopen(fd, "w") as f:
                    f.write(output)
                os.replace(tmp_path, output_path)
            except Exception:
                with contextlib.suppress(OSError):
                    os.unlink(tmp_path)
                raise
        except OSError as e:
            # Clean error instead of raw traceback (F-004)
            print(
                "error: writing report\n"
                f"  Cause:  {e}\n"
                "  Action: Check output path permissions\n"
                "  Exit:   1",
                file=sys.stderr,
            )
            return 1
        print(f"Report written to {args.output_file}", file=sys.stderr)
    else:
        print(output)

    return 0


if __name__ == "__main__":
    sys.exit(main())
