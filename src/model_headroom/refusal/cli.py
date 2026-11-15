"""Command-line entry point for refusal-direction extraction.

Three acquisition paths share one spec:

* offline  -- ``--harmful-activations`` / ``--harmless-activations``
* live     -- ``--model-path`` + ``--harmful-prompts`` / ``--harmless-prompts``
* recovery -- ``--weight-deltas``

None of them couples to the headroom scoring pipeline; this entry point never
imports the orchestrator.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    import torch

from model_headroom.device import resolve_device
from model_headroom.models.types import (
    LayerSelection,
    RefusalDirectionReport,
    RefusalDirectionSpec,
    ResidualBatch,
)
from model_headroom.refusal.activations import (
    capture_last_token_residuals,
    capture_residuals_sglang,
    discover_residual_layers,
    load_prompts,
    load_residual_batches_from_json,
)
from model_headroom.refusal.artifact import write_refusal_artifact
from model_headroom.refusal.extract import compute_refusal_direction
from model_headroom.refusal.recover import (
    load_weight_deltas_from_json,
    recover_direction_from_weight_deltas,
)
from model_headroom.refusal.report import (
    generate_refusal_json,
    generate_refusal_report,
)


class _ModelPathError(Exception):
    pass


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse refusal-direction arguments."""
    parser = argparse.ArgumentParser(
        prog="model-headroom-refusal",
        description=(
            "Compute a refusal direction from contrastive harmful/harmless "
            "prompt sets — Arditi-style difference of means over residual "
            "activations, or recovery from an abliterated checkpoint"
        ),
    )
    parser.add_argument(
        "--harmful-activations",
        help="JSON of pre-captured harmful activations: "
        '{"layers": [{"layer_name": "...", "activations": [[...]]}]}',
    )
    parser.add_argument(
        "--harmless-activations",
        help="JSON of pre-captured harmless activations in the same format",
    )
    parser.add_argument(
        "--model-path",
        help="Hugging Face model directory or hub id for live capture",
    )
    parser.add_argument(
        "--harmful-prompts",
        help="Prompt set for live capture (.txt one-per-line, .json, or .jsonl)",
    )
    parser.add_argument(
        "--harmless-prompts",
        help="Harmless prompt set for live capture, same formats",
    )
    parser.add_argument(
        "--sglang-url",
        help="Base URL of a running sglang server started with "
        "--enable-return-hidden-states; captures final-layer residuals over HTTP "
        "without loading the model in-process",
    )
    parser.add_argument(
        "--sglang-model",
        help="Served model name to request from the sglang server "
        "(default: whatever the server reports)",
    )
    parser.add_argument(
        "--weight-deltas",
        help="JSON of W - W' deltas from an abliterated checkpoint "
        '({"deltas": [[...]]}) — recovers r without any activations',
    )
    parser.add_argument(
        "--layer-name",
        help="Select this exact layer instead of the depth heuristic",
    )
    parser.add_argument(
        "--layer-index",
        type=int,
        help="Select this ordinal layer index (0 = shallowest shared layer)",
    )
    parser.add_argument(
        "--depth-fraction",
        type=float,
        default=0.6,
        help="When neither --layer-name nor --layer-index is given, select this "
        "fraction of the way through the shared layer range (default: 0.6)",
    )
    parser.add_argument(
        "--layer-capture",
        default="auto",
        help="Layers to hook during live capture: auto | all | comma list of "
        "module names (default: auto, meaning the depth-fraction layer of the "
        "discovered stack)",
    )
    parser.add_argument(
        "--orthogonalize",
        action="store_true",
        help="Orthogonalize the direction against the harmless mean so that "
        "projecting it does not also remove helpfulness",
    )
    parser.add_argument(
        "--mask-sigma",
        type=float,
        default=None,
        help="Drop rows whose residual norm exceeds mean + sigma*std "
        "(massive-activation masking; off by default)",
    )
    parser.add_argument(
        "--winsorize",
        type=float,
        default=None,
        help="Symmetrically clamp each prompt's residual components to its own "
        "quantile, taming massive activations without discarding the prompt "
        "(off by default; 0.95 is a common choice)",
    )
    parser.add_argument(
        "--min-pairs",
        type=int,
        default=8,
        help="Minimum prompts required per side before a direction is emitted "
        "(default: 8)",
    )
    parser.add_argument(
        "--allow-small-samples",
        action="store_true",
        help="Emit a direction even below --min-pairs",
    )
    parser.add_argument(
        "--position",
        choices=["last", "mean"],
        default="last",
        help="Which residual position to reduce to a vector (default: last, "
        "which forces left padding during live capture)",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help="Device for live capture: auto uses the first available "
        "accelerator (cuda preferred), else cpu (default: auto)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Prompts per forward pass during live capture (default: 8)",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="Tokenisation truncation length during live capture (default: 512)",
    )
    parser.add_argument(
        "--output-format",
        choices=["text", "json", "safetensors", "both"],
        default="text",
        help="Output format (default: text)",
    )
    parser.add_argument(
        "--output-file",
        help="Write the report to a file instead of stdout",
    )
    parser.add_argument(
        "--domain-activations",
        help="JSON of pre-captured in-domain residual activations, same format as --harmful-activations",
    )
    parser.add_argument(
        "--out-of-domain-activations",
        help="JSON of pre-captured out-of-domain residual activations, same format as --harmful-activations",
    )
    parser.add_argument(
        "--domain-prompts",
        help="Prompt file set for the in-domain contrast set (.txt, .json, or .jsonl)",
    )
    parser.add_argument(
        "--out-of-domain-prompts",
        help="Prompt file set for the out-of-domain contrast set (.txt, .json, or .jsonl)",
    )
    parser.add_argument(
        "--domain-label",
        help="Optional label recorded in the artifact metadata, e.g. creative_fiction",
    )
    parser.add_argument(
        "--model-id",
        help="Optional explicit model id recorded in the artifact metadata",
    )
    return parser.parse_args(argv)


def _build_spec(args: argparse.Namespace) -> RefusalDirectionSpec:
    selection: LayerSelection = (
        "explicit" if (args.layer_name or args.layer_index is not None) else "auto"
    )
    return RefusalDirectionSpec(
        layer_selection=selection,
        layer_name=args.layer_name,
        layer_index=args.layer_index,
        depth_fraction=args.depth_fraction,
        orthogonalize=args.orthogonalize,
        massive_activation_sigma=args.mask_sigma,
        winsorize_quantile=args.winsorize,
        min_pairs=1 if args.allow_small_samples else args.min_pairs,
    )


def _load_offline(
    args: argparse.Namespace,
) -> tuple[list[ResidualBatch], list[ResidualBatch]] | None:
    if not (args.harmful_activations and args.harmless_activations):
        return None
    harmful = load_residual_batches_from_json(args.harmful_activations)
    harmless = load_residual_batches_from_json(args.harmless_activations)
    if harmful is None or harmless is None:
        print(
            "error: loading activations from JSON\n"
            "  Cause:  failed to load harmful or harmless activations\n"
            "  Action: Check the JSON files are valid\n"
            "  Exit:   1",
            file=sys.stderr,
        )
        return None
    return (harmful, harmless)


def _is_filesystem_shaped(model_path: str) -> bool:
    return (
        model_path.startswith("/")
        or model_path.startswith("./")
        or model_path.startswith("../")
        or model_path.startswith("~")
    )


def _load_live(
    args: argparse.Namespace,
) -> tuple[list[ResidualBatch], list[ResidualBatch]] | None:
    if not (args.model_path and args.harmful_prompts and args.harmless_prompts):
        return None
    path = Path(args.model_path)
    if path.is_file():
        raise _ModelPathError(
            "--model-path expects a HuggingFace model directory, got a file: "
            f"{path}. transformers cannot load GGUF checkpoints for live capture; "
            "use --harmful-activations/--harmless-activations with pre-captured "
            "residuals, or --sglang-url with --enable-return-hidden-states."
        )
    if path.is_dir() and not (path / "config.json").is_file():
        raise _ModelPathError(
            f"--model-path has no config.json; not a HuggingFace checkpoint directory: {path}"
        )
    if not path.exists() and _is_filesystem_shaped(args.model_path):
        raise _ModelPathError(f"--model-path does not exist: {path}")
    harmful_prompts = load_prompts(args.harmful_prompts)
    harmless_prompts = load_prompts(args.harmless_prompts)
    if harmful_prompts is None or harmless_prompts is None:
        print(
            "error: loading prompt sets\n"
            "  Cause:  failed to load harmful or harmless prompts\n"
            "  Action: Check the prompt files\n"
            "  Exit:   1",
            file=sys.stderr,
        )
        return None

    try:
        from transformers import AutoTokenizer
    except ImportError:
        print(
            "error: live capture requires transformers\n"
            "  Cause:  transformers is not a declared dependency\n"
            "  Action: Install it, or use --harmful-activations with pre-captured tensors\n"
            "  Exit:   1",
            file=sys.stderr,
        )
        return None

    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        dtype = torch_dtype(args.model_path)
        loader_kwargs: dict[str, object] = {"dtype": dtype}
        if str(args.device).startswith("cuda"):
            try:
                import accelerate  # noqa: F401

                loader_kwargs["device_map"] = "auto"
            except ImportError:
                pass

        model = None
        load_errors: list[str] = []
        for loader_name in ("AutoModelForCausalLM", "AutoModelForImageTextToText"):
            try:
                import transformers

                loader = getattr(transformers, loader_name)
                model = loader.from_pretrained(args.model_path, **loader_kwargs)
                break
            except (OSError, ValueError, TypeError, AttributeError) as exc:
                load_errors.append(f"{loader_name}: {exc}")
                model = None
        if model is None:
            print(
                "error: loading model\n"
                "  Cause:  failed to load model with available loaders\n"
                "  Action: Check the model path and try a different loader\n"
                "  Exit:   1",
                file=sys.stderr,
            )
            return None
    except (OSError, ValueError) as exc:
        print(
            "error: loading model\n"
            f"  Cause:  {exc}\n"
            "  Action: Check the model path\n"
            "  Exit:   1",
            file=sys.stderr,
        )
        return None

    discovered = discover_residual_layers(model)
    if not discovered:
        print(
            "error: no decoder layer stack discovered\n"
            "  Cause:  no decoder layer stack found in this model\n"
            "  Action: Check the model architecture\n"
            "  Exit:   1",
            file=sys.stderr,
        )
        return None

    spec = _build_spec(args)
    if args.layer_capture == "all":
        targets = discovered
    elif args.layer_capture == "auto":
        last = len(discovered) - 1
        pick = (
            spec.layer_index
            if spec.layer_index is not None
            else round(spec.depth_fraction * last)
        )
        pick = max(0, min(pick, last))
        targets = [discovered[pick]]
    else:
        targets = [n.strip() for n in args.layer_capture.split(",") if n.strip()]

    harmful = capture_last_token_residuals(
        model,
        tokenizer,
        harmful_prompts,
        targets,
        device=args.device,
        batch_size=args.batch_size,
        max_length=args.max_length,
        position=args.position,
    )
    harmless = capture_last_token_residuals(
        model,
        tokenizer,
        harmless_prompts,
        targets,
        device=args.device,
        batch_size=args.batch_size,
        max_length=args.max_length,
        position=args.position,
    )
    if harmful is None or harmless is None:
        print(
            "error: live capture produced no activations\n"
            "  Cause:  capture returned None for harmful or harmless residuals\n"
            "  Action: Check model, tokenizer, and prompts\n"
            "  Exit:   1",
            file=sys.stderr,
        )
        return None
    return (list(harmful.values()), list(harmless.values()))


def torch_dtype(model_path: str) -> str:
    """Resolve the checkpoint dtype string for from_pretrained."""

    try:
        with (Path(model_path) / "config.json").open() as handle:
            config = json.load(handle)
    except (OSError, json.JSONDecodeError, AttributeError, TypeError):
        return "auto"
    return str(config.get("torch_dtype") or config.get("dtype") or "auto")


def _resolve_device(requested: str) -> str | None:
    """Resolve --device to a usable device string via the shared resolver.

    auto -> first available accelerator (cuda preferred), else cpu
    cuda -> cuda if available else error (None)
    cpu -> cpu
    Returns None only when a requested accelerator is unavailable.
    """
    return resolve_device(requested)


def _domain_requested(args: argparse.Namespace) -> bool:
    return bool(
        args.domain_activations
        or args.out_of_domain_activations
        or args.domain_prompts
        or args.out_of_domain_prompts
    )


def _resolve_model_id(args: argparse.Namespace) -> str | None:
    if args.model_id is not None:
        return cast("str | None", args.model_id)
    if args.model_path is not None:
        return cast("str | None", args.model_path)
    if args.sglang_model is not None:
        return cast("str | None", args.sglang_model)
    return None


def _build_sidecar(
    report: RefusalDirectionReport,
    domain_direction: torch.Tensor | None,
    model_id: str | None,
) -> str:
    payload = json.loads(generate_refusal_json(report))
    if domain_direction is not None:
        payload["domain_direction"] = [float(v) for v in domain_direction.tolist()]
    payload["model_id"] = model_id if model_id is not None else ""
    payload["timestamp"] = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    return json.dumps(payload, indent=2)


def _load_domain_offline(
    args: argparse.Namespace,
) -> tuple[list[ResidualBatch], list[ResidualBatch]] | None:
    in_domain = load_residual_batches_from_json(args.domain_activations)
    out_of_domain = load_residual_batches_from_json(args.out_of_domain_activations)
    if in_domain is None:
        print(
            "error: loading domain activations\n"
            f"  Cause:  failed to load in-domain activations from {args.domain_activations}\n"
            "  Action: Check the JSON file\n"
            "  Exit:   1",
            file=sys.stderr,
        )
        return None
    if out_of_domain is None:
        print(
            "error: loading domain activations\n"
            f"  Cause:  failed to load out-of-domain activations from {args.out_of_domain_activations}\n"
            "  Action: Check the JSON file\n"
            "  Exit:   1",
            file=sys.stderr,
        )
        return None
    return (in_domain, out_of_domain)


def _capture_domain_sglang(
    args: argparse.Namespace, layer_name: str
) -> tuple[list[ResidualBatch], list[ResidualBatch]] | None:
    served = args.sglang_model or "default"
    in_prompts = load_prompts(args.domain_prompts) if args.domain_prompts else None
    out_prompts = (
        load_prompts(args.out_of_domain_prompts) if args.out_of_domain_prompts else None
    )
    if in_prompts is None or out_prompts is None:
        print(
            "error: loading domain prompt sets\n"
            "  Cause:  failed to load domain or out-of-domain prompts\n"
            "  Action: Check the prompt files\n"
            "  Exit:   1",
            file=sys.stderr,
        )
        return None
    in_batches = capture_residuals_sglang(
        args.sglang_url, served, in_prompts, layer_name
    )
    out_batches = capture_residuals_sglang(
        args.sglang_url, served, out_prompts, layer_name
    )
    if in_batches is None or out_batches is None:
        print(
            "error: sglang returned no hidden states\n"
            "  Cause:  sglang server did not return hidden states for the domain contrast\n"
            "  Action: Run sglang server with --enable-return-hidden-states\n"
            "  Exit:   1",
            file=sys.stderr,
        )
        return None
    return (list(in_batches.values()), list(out_batches.values()))


def _load_live_model(args: argparse.Namespace) -> tuple[Any, Any] | None:
    import transformers
    from transformers import AutoTokenizer

    path = Path(args.model_path)
    if path.is_file():
        raise _ModelPathError(
            "--model-path expects a HuggingFace model directory, got a file: "
            f"{path}. transformers cannot load GGUF checkpoints for live capture; "
            "use --harmful-activations/--harmless-activations with pre-captured "
            "residuals, or --sglang-url with --enable-return-hidden-states."
        )
    if path.is_dir() and not (path / "config.json").is_file():
        raise _ModelPathError(
            f"--model-path has no config.json; not a HuggingFace checkpoint directory: {path}"
        )
    if not path.exists() and _is_filesystem_shaped(args.model_path):
        raise _ModelPathError(f"--model-path does not exist: {path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    dtype = torch_dtype(args.model_path)
    loader_kwargs: dict[str, object] = {"dtype": dtype}
    if str(args.device).startswith("cuda"):
        try:
            import accelerate  # noqa: F401

            loader_kwargs["device_map"] = "auto"
        except ImportError:
            pass
    model = None
    for loader_name in ("AutoModelForCausalLM", "AutoModelForImageTextToText"):
        try:
            loader = getattr(transformers, loader_name)
            model = loader.from_pretrained(args.model_path, **loader_kwargs)
            break
        except (OSError, ValueError, TypeError, AttributeError):
            model = None
    if model is None:
        return None, None
    return model, tokenizer


def _capture_domain_live(
    args: argparse.Namespace, layer_name: str
) -> tuple[list[ResidualBatch], list[ResidualBatch]] | None:
    try:
        loaded = _load_live_model(args)
    except _ModelPathError as exc:
        print(
            "error: model path error\n"
            f"  Cause:  {exc}\n"
            "  Action: Check the model path\n"
            "  Exit:   1",
            file=sys.stderr,
        )
        return None
    if loaded is None:
        print(
            "error: loading model\n"
            "  Cause:  failed to load model for domain capture\n"
            "  Action: Check the model path\n"
            "  Exit:   1",
            file=sys.stderr,
        )
        return None
    model, tokenizer = loaded
    modules = dict(model.named_modules())
    if layer_name not in modules:
        print(
            "error: domain layer not found\n"
            f"  Cause:  layer {layer_name!r} is not present in the model\n"
            "  Action: Check the layer selection\n"
            "  Exit:   1",
            file=sys.stderr,
        )
        return None
    in_prompts = load_prompts(args.domain_prompts) if args.domain_prompts else None
    out_prompts = (
        load_prompts(args.out_of_domain_prompts) if args.out_of_domain_prompts else None
    )
    if in_prompts is None or out_prompts is None:
        print(
            "error: loading domain prompt sets\n"
            "  Cause:  failed to load domain or out-of-domain prompts\n"
            "  Action: Check the prompt files\n"
            "  Exit:   1",
            file=sys.stderr,
        )
        return None
    in_batches = capture_last_token_residuals(
        model,
        tokenizer,
        in_prompts,
        [layer_name],
        device=args.device,
        batch_size=args.batch_size,
        max_length=args.max_length,
        position=args.position,
    )
    out_batches = capture_last_token_residuals(
        model,
        tokenizer,
        out_prompts,
        [layer_name],
        device=args.device,
        batch_size=args.batch_size,
        max_length=args.max_length,
        position=args.position,
    )
    if in_batches is None or out_batches is None:
        print(
            "error: live capture produced no activations\n"
            "  Cause:  capture returned None for domain or out-of-domain residuals\n"
            "  Action: Check model, tokenizer, and prompts\n"
            "  Exit:   1",
            file=sys.stderr,
        )
        return None
    return (list(in_batches.values()), list(out_batches.values()))


def _compute_domain_direction(
    args: argparse.Namespace, report: RefusalDirectionReport
) -> torch.Tensor | None:
    if report.layer_index is None:
        print(
            "error: domain contrast requires a resolved layer\n"
            "  Cause:  the refusal direction has no resolved layer_index\n"
            "  Action: Provide --layer-name or --layer-index\n"
            "  Exit:   1",
            file=sys.stderr,
        )
        return None
    layer_name = report.layer_stats[report.layer_index].layer_name
    if args.domain_activations and args.out_of_domain_activations:
        loaded = _load_domain_offline(args)
    elif args.domain_prompts and args.out_of_domain_prompts:
        if args.sglang_url:
            loaded = _capture_domain_sglang(args, layer_name)
        else:
            loaded = _capture_domain_live(args, layer_name)
    else:
        print(
            "error: incomplete domain contrast\n"
            "  Cause:  provide both --domain-activations and --out-of-domain-activations, or both --domain-prompts and --out-of-domain-prompts\n"
            "  Action: Complete the domain pair\n"
            "  Exit:   1",
            file=sys.stderr,
        )
        return None
    if loaded is None:
        return None
    in_domain, out_of_domain = loaded
    domain_spec = RefusalDirectionSpec(
        layer_selection="explicit",
        layer_name=None,
        layer_index=report.layer_index,
        depth_fraction=args.depth_fraction,
        orthogonalize=args.orthogonalize,
        massive_activation_sigma=args.mask_sigma,
        winsorize_quantile=args.winsorize,
        min_pairs=1 if args.allow_small_samples else args.min_pairs,
    )
    domain_report = compute_refusal_direction(in_domain, out_of_domain, domain_spec)
    if domain_report.direction is None:
        print(
            "error: domain contrast computation\n"
            f"  Cause:  {domain_report.error_message}\n"
            "  Action: Check the domain prompt sets and layer selection\n"
            "  Exit:   1",
            file=sys.stderr,
        )
        return None
    return domain_report.direction


def main(argv: list[str] | None = None) -> int:
    """Drive refusal-direction extraction and emit a report."""
    args = parse_args(argv)
    device = _resolve_device(args.device)
    if device is None:
        return 1
    args.device = device
    spec = _build_spec(args)

    if args.weight_deltas and _domain_requested(args):
        print(
            "error: weight-deltas cannot carry a domain contrast\n"
            "  Cause:  --weight-deltas was combined with a domain contrast flag\n"
            "  Action: Use --weight-deltas alone, or use activations/prompts for the domain contrast\n"
            "  Exit:   1",
            file=sys.stderr,
        )
        return 1

    if args.sglang_url:
        if not (args.harmful_prompts and args.harmless_prompts):
            print(
                "error: --sglang-url needs prompt sets\n"
                "  Cause:  --sglang-url requires --harmful-prompts and --harmless-prompts\n"
                "  Action: Provide both prompt files\n"
                "  Exit:   1",
                file=sys.stderr,
            )
            return 1
        harmful_prompts = load_prompts(args.harmful_prompts)
        harmless_prompts = load_prompts(args.harmless_prompts)
        if harmful_prompts is None or harmless_prompts is None:
            print(
                "error: loading prompt sets\n"
                "  Cause:  failed to load harmful or harmless prompts\n"
                "  Action: Check the prompt files\n"
                "  Exit:   1",
                file=sys.stderr,
            )
            return 1
        served = args.sglang_model or "default"
        harmful_batches = capture_residuals_sglang(
            args.sglang_url, served, harmful_prompts
        )
        harmless_batches = capture_residuals_sglang(
            args.sglang_url, served, harmless_prompts
        )
        if harmful_batches is None or harmless_batches is None:
            print(
                "error: sglang returned no hidden states\n"
                "  Cause:  sglang server did not return hidden states\n"
                "  Action: Run sglang server with --enable-return-hidden-states\n"
                "  Exit:   1",
                file=sys.stderr,
            )
            return 1
        report = compute_refusal_direction(
            list(harmful_batches.values()),
            list(harmless_batches.values()),
            spec,
        )
    elif args.weight_deltas:
        deltas = load_weight_deltas_from_json(args.weight_deltas)
        if deltas is None:
            print(
                "error: loading weight deltas\n"
                f"  Cause:  failed to load weight deltas from {args.weight_deltas}\n"
                "  Action: Check the JSON file\n"
                "  Exit:   1",
                file=sys.stderr,
            )
            return 1
        report = recover_direction_from_weight_deltas(deltas, spec, device=args.device)
    else:
        try:
            loaded = _load_offline(args) or _load_live(args)
        except _ModelPathError as exc:
            print(
                "error: model path error\n"
                f"  Cause:  {exc}\n"
                "  Action: Check the model path\n"
                "  Exit:   1",
                file=sys.stderr,
            )
            return 1
        if loaded is None:
            print(
                "error: no acquisition path provided\n"
                "  Cause:  no input source specified\n"
                "  Action: Provide one of:\n"
                "    --harmful-activations + --harmless-activations\n"
                "    --model-path + --harmful-prompts + --harmless-prompts\n"
                "    --sglang-url + --harmful-prompts + --harmless-prompts\n"
                "    --weight-deltas\n"
                "  Exit:   1",
                file=sys.stderr,
            )
            return 1
        harmful, harmless = loaded
        report = compute_refusal_direction(harmful, harmless, spec)

    if report.direction_status == "error":
        print(
            "error: refusal computation\n"
            f"  Cause:  {report.error_message}\n"
            "  Action: Check model and inputs\n"
            "  Exit:   1",
            file=sys.stderr,
        )
        return 1
    if report.direction is None:
        print(
            "error: refusal computation\n"
            f"  Cause:  {report.error_message}\n"
            "  Action: Check model and inputs\n"
            "  Exit:   2",
            file=sys.stderr,
        )
        return 2

    if args.output_format in ("safetensors", "both") and not args.output_file:
        print(
            "error: --output-format safetensors|both requires --output-file\n"
            "  Cause:  --output-file was not provided\n"
            "  Action: Provide --output-file with --output-format safetensors or both\n"
            "  Exit:   1",
            file=sys.stderr,
        )
        return 1

    domain_direction = None
    if _domain_requested(args):
        domain_direction = _compute_domain_direction(args, report)
        if domain_direction is None:
            return 1

    model_id = _resolve_model_id(args)

    if args.output_format == "text":
        output = generate_refusal_report(report)
        if args.output_file:
            Path(args.output_file).write_text(output)
            print(f"Report written to {args.output_file}", file=sys.stderr)
        else:
            print(output)
        return 0
    if args.output_format == "json":
        output = generate_refusal_json(report)
        if args.output_file:
            Path(args.output_file).write_text(output)
            print(f"Report written to {args.output_file}", file=sys.stderr)
        else:
            print(output)
        return 0
    if args.output_format == "safetensors":
        write_refusal_artifact(
            report,
            args.output_file,
            model_id=model_id,
            domain_direction=domain_direction,
            domain_label=args.domain_label,
        )
        return 0
    # both
    write_refusal_artifact(
        report,
        args.output_file,
        model_id=model_id,
        domain_direction=domain_direction,
        domain_label=args.domain_label,
    )
    sidecar = _build_sidecar(report, domain_direction, model_id)
    Path(f"{args.output_file}.json").write_text(sidecar)
    return 0


if __name__ == "__main__":
    sys.exit(main())
