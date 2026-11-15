"""Paired-model recovery diagnostics CLI: compare, decide, verify, schema-version.

decide reads coverage from --coverage when given, otherwise from the coverage
carried by --comparison. --coverage wins when both are supplied.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from model_headroom.paired.schema import SCHEMA_VERSION, MeasurementDocument

if TYPE_CHECKING:
    from model_headroom.analysis.intervention_record import InterventionRecord
    from model_headroom.paired.compare import ComparisonReport

_STRATEGIES: tuple[str, ...] = (
    "continuous_parameters_frozen_codes",
    "targeted_low_rank",
    "rebuild_source_before_quantization",
)

_METRIC_DIRECTIONS: tuple[str, ...] = ("lower_is_better", "higher_is_better")


def _error(cause: str, action: str, *, context: str = "paired diagnostics") -> int:
    print(
        f"error: {context}\n  Cause:  {cause}\n  Action: {action}\n  Exit:   1",
        file=sys.stderr,
    )
    return 1


def _load_document(path: str) -> MeasurementDocument | None:
    try:
        text = Path(path).read_text()
    except OSError as exc:
        _error(
            f"cannot read {path}: {exc}",
            "Check the path exists and is readable",
            context="reading measurement document",
        )
        return None
    try:
        return MeasurementDocument.from_json(text)
    except json.JSONDecodeError as exc:
        _error(
            f"invalid JSON in {path}: {exc}",
            "Check the JSON file",
            context="parsing measurement document",
        )
        return None
    except ValueError as exc:
        _error(
            str(exc),
            f"Regenerate the document with schema_version {SCHEMA_VERSION}",
            context="schema version mismatch",
        )
        return None


def _load_intervention_record(path: str) -> InterventionRecord | None:
    from model_headroom.analysis.intervention_record import InterventionRecord

    try:
        text = Path(path).read_text()
    except OSError as exc:
        _error(
            f"cannot read {path}: {exc}",
            "Check the path exists and is readable",
            context="reading intervention record",
        )
        return None
    try:
        return InterventionRecord.from_json(text)
    except json.JSONDecodeError as exc:
        _error(
            f"invalid JSON in {path}: {exc}",
            "Check the JSON file",
            context="parsing intervention record",
        )
        return None
    except (TypeError, KeyError, ValueError) as exc:
        _error(
            f"{type(exc).__name__}: {exc}",
            "Check the record carries the InterventionRecord fields",
            context="parsing intervention record",
        )
        return None


def _document_to_measurements(doc: MeasurementDocument) -> dict[str, float]:
    measurements: dict[str, float] = {}
    for layer in doc.layers:
        if layer.output_cosine is not None:
            measurements[f"{layer.layer_name}.output_cosine"] = layer.output_cosine
        if layer.normalized_reconstruction_error is not None:
            key = f"{layer.layer_name}.normalized_reconstruction_error"
            measurements[key] = layer.normalized_reconstruction_error
        if layer.output_norm_ratio is not None:
            measurements[f"{layer.layer_name}.output_norm_ratio"] = (
                layer.output_norm_ratio
            )
    if doc.gradients is not None:
        if doc.gradients.grad_norm is not None:
            measurements["gradients.grad_norm"] = doc.gradients.grad_norm
        if doc.gradients.coherence is not None:
            measurements["gradients.coherence"] = doc.gradients.coherence
        if doc.gradients.effective_rank is not None:
            measurements["gradients.effective_rank"] = doc.gradients.effective_rank
    return measurements


def _parse_coverage(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(value)
    except ValueError as exc:
        raise ValueError(f"not a number: {value}") from exc
    if not 0.0 <= parsed <= 1.0:
        raise ValueError(f"outside [0.0, 1.0]: {value}")
    return parsed


def _parse_held_out_keys(value: str | None) -> list[str]:
    if value is None:
        return []
    keys = [key.strip() for key in value.split(",") if key.strip()]
    if not keys:
        raise ValueError("--held-out-keys was empty")
    return keys


def _rebuild(cls: type[Any], data: dict[str, object]) -> Any:
    fields = {field.name for field in dataclasses.fields(cls)}
    missing = sorted(fields - set(data))
    if missing:
        msg = f"{cls.__name__} is missing fields: " + ", ".join(missing)
        raise ValueError(msg)
    return cls(**{key: value for key, value in data.items() if key in fields})


def _load_comparison(path: str) -> ComparisonReport | None:
    from model_headroom.paired.compare import (
        ActivationOutlierReport,
        ComparisonReport,
        GradientComparison,
        LayerComparison,
        ModuleComparison,
    )

    try:
        data = json.loads(Path(path).read_text())
    except OSError as exc:
        _error(
            f"cannot read {path}: {exc}",
            "Check the path exists and is readable",
            context="reading comparison",
        )
        return None
    except json.JSONDecodeError as exc:
        _error(
            f"invalid JSON in {path}: {exc}",
            "Check the JSON file",
            context="parsing comparison",
        )
        return None
    if not isinstance(data, dict):
        _error(
            f"comparison must be a JSON object, got {type(data).__name__}",
            "Regenerate it with the compare subcommand",
            context="parsing comparison",
        )
        return None
    try:
        layers = tuple(
            _rebuild(LayerComparison, layer)
            for layer in _as_list(data.get("layers"), "layers")
        )
        modules = tuple(
            _rebuild(ModuleComparison, module)
            for module in _as_list(data.get("modules"), "modules")
        )
        outliers = tuple(
            _rebuild(ActivationOutlierReport, outlier)
            for outlier in _as_list(data.get("outliers"), "outliers")
        )
        gradients = (
            _rebuild(GradientComparison, data["gradients"])
            if data.get("gradients") is not None
            else None
        )
        payload = dict(data)
        payload["layers"] = layers
        payload["modules"] = modules
        payload["outliers"] = outliers
        payload["gradients"] = gradients
        return cast("ComparisonReport", _rebuild(ComparisonReport, payload))
    except (TypeError, KeyError, ValueError, AttributeError) as exc:
        _error(
            f"{type(exc).__name__}: {exc}",
            "Regenerate it with the compare subcommand",
            context="parsing comparison",
        )
        return None


def _as_list(value: object, field: str) -> list[dict[str, object]]:
    if value is None:
        return []
    if not isinstance(value, list) or not all(isinstance(v, dict) for v in value):
        msg = f"{field} must be a list of objects"
        raise ValueError(msg)
    return value


def _write_payload(payload: str, out: str | None) -> int:
    if out is None:
        print(payload)
        return 0
    try:
        Path(out).write_text(payload + "\n")
    except OSError as exc:
        _error(
            f"cannot write {out}: {exc}",
            "Check the output path is writable",
            context="writing report",
        )
        return 1
    return 0


def _with_coverage(
    comparison: ComparisonReport | None, coverage: float
) -> ComparisonReport:
    from model_headroom.paired.compare import ComparisonReport

    if comparison is None:
        return ComparisonReport(
            reference_model_id="",
            candidate_model_id="",
            input_fingerprint_match=None,
            layers=(),
            modules=(),
            gradients=None,
            outliers=(),
            coverage=coverage,
            measured_layer_count=0,
            comparable_layer_count=0,
            warnings=(),
        )
    return dataclasses.replace(comparison, coverage=coverage)


def _compare_main(args: argparse.Namespace) -> int:
    from model_headroom.paired.compare import ComparisonError, compare_documents

    reference = _load_document(args.reference)
    if reference is None:
        return 1
    candidate = _load_document(args.candidate)
    if candidate is None:
        return 1
    try:
        report = compare_documents(
            reference,
            candidate,
            outlier_sigma=args.outlier_sigma,
            near_inactive_threshold=args.near_inactive_threshold,
        )
    except ComparisonError as exc:
        _error(
            str(exc),
            "Check the measurement documents describe the same model and input",
            context="comparing documents",
        )
        return 1
    payload = json.dumps(dataclasses.asdict(report), indent=2)
    return _write_payload(payload, args.out)


def _build_record(args: argparse.Namespace) -> InterventionRecord | None:
    from model_headroom.analysis.intervention_record import InterventionRecord

    if args.post is None:
        if args.intervention_id is not None:
            _error(
                "--intervention-id needs --post",
                "Provide --post PATH to the post-measurement document",
                context="building intervention record",
            )
            return None
        return _load_intervention_record(args.baseline)

    baseline = _load_document(args.baseline)
    if baseline is None:
        return None
    post = _load_document(args.post)
    if post is None:
        return None
    default_id = f"{baseline.model_id or 'baseline'}->{post.model_id or 'post'}"
    return InterventionRecord(
        intervention_id=args.intervention_id or default_id,
        intervention_type=args.strategy or _STRATEGIES[0],
        checkpoint_hash=baseline.checkpoint_hash or "",
        architecture=baseline.architecture or "",
        runtime_versions={},
        affected_parameters=[],
        baseline_measurements=_document_to_measurements(baseline),
        post_measurements=_document_to_measurements(post),
    )


def _decide_main(args: argparse.Namespace) -> int:
    from model_headroom.analysis.evidence_store import EvidenceStore
    from model_headroom.paired.decision import (
        RecoveryDecision,
        decide,
        evaluate_strategy,
        persist_decision,
    )

    try:
        held_out_keys = _parse_held_out_keys(args.held_out_keys)
    except ValueError as exc:
        _error(
            str(exc),
            "Provide a comma-separated list of measurement keys",
            context="parsing --held-out-keys",
        )
        return 1

    try:
        coverage = _parse_coverage(args.coverage)
    except ValueError as exc:
        _error(
            str(exc),
            "Provide a fraction between 0.0 and 1.0",
            context="parsing --coverage",
        )
        return 1

    record = _build_record(args)
    if record is None:
        return 1
    if args.strategy is not None:
        record.intervention_type = args.strategy

    comparison = None
    if args.comparison is not None:
        comparison = _load_comparison(args.comparison)
        if comparison is None:
            return 1

    if coverage is not None:
        comparison = _with_coverage(comparison, coverage)

    if args.strategy is not None:
        evidence = evaluate_strategy(
            record,
            held_out_keys=held_out_keys,
            coverage=comparison.coverage if comparison is not None else None,
            metric_direction=args.metric_direction,
        )
        decision = RecoveryDecision(
            strategies=(evidence,),
            distinguishable=False,
            summary="Single strategy evaluated.",
            warnings=(),
        )
    else:
        records = [
            dataclasses.replace(record, intervention_type=strategy)
            for strategy in _STRATEGIES
        ]
        decision = decide(
            records,
            comparison=comparison,
            held_out_keys=held_out_keys,
            metric_direction=args.metric_direction,
        )

    payload = json.dumps(dataclasses.asdict(decision), indent=2)
    status = _write_payload(payload, args.out)
    if status != 0:
        return status

    if args.store is not None:
        try:
            store = EvidenceStore(args.store)
            artifact = persist_decision(decision, store)
        except OSError as exc:
            _error(
                f"cannot write evidence store {args.store}: {exc}",
                "Check the store path is writable",
                context="persisting decision",
            )
            return 1
        print(f"artifact_id: {artifact.artifact_id}", file=sys.stderr)
    return 0


def _verify_main(args: argparse.Namespace) -> int:
    from model_headroom.analysis.evidence_store import EvidenceStore

    try:
        store = EvidenceStore(args.store)
        result = store.verify(args.artifact_id)
    except OSError as exc:
        _error(
            f"cannot read evidence store {args.store}: {exc}",
            "Check the store path exists and is readable",
            context="verifying artifact",
        )
        return 1
    print(json.dumps(result, indent=2))
    if result.get("verified") is True:
        return 0
    _error(
        f"artifact {args.artifact_id} did not verify: {result.get('reason', 'unknown')}",
        "Re-run decide --store, or restore the artifact from the evidence store",
        context="verifying artifact",
    )
    return 1


def _schema_version_main(args: argparse.Namespace) -> int:
    print(SCHEMA_VERSION)
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="model-headroom-paired",
        description=(
            "Paired-model recovery diagnostics: compare measurements, decide a "
            "recovery strategy, verify evidence."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_compare = sub.add_parser("compare", help="Compare two measurement documents.")
    p_compare.add_argument("--reference", required=True)
    p_compare.add_argument("--candidate", required=True)
    p_compare.add_argument("--out")
    p_compare.add_argument("--outlier-sigma", type=float, default=3.0)
    p_compare.add_argument("--near-inactive-threshold", type=float, default=1e-3)

    p_decide = sub.add_parser(
        "decide",
        help="Decide recovery strategy from intervention evidence. Coverage "
        "(from --coverage or --comparison) gates a SUPPORTED verdict.",
    )
    p_decide.add_argument("--baseline", required=True)
    p_decide.add_argument("--post")
    p_decide.add_argument("--strategy", choices=list(_STRATEGIES))
    p_decide.add_argument("--held-out-keys")
    p_decide.add_argument(
        "--metric-direction",
        choices=list(_METRIC_DIRECTIONS),
        default="lower_is_better",
    )
    p_decide.add_argument(
        "--coverage",
        help="Measured coverage in [0.0, 1.0] for this intervention. Overrides "
        "the coverage carried by --comparison. Coverage at or above 0.8 is "
        "required for a SUPPORTED verdict.",
    )
    p_decide.add_argument("--comparison")
    p_decide.add_argument("--store")
    p_decide.add_argument("--intervention-id")
    p_decide.add_argument("--out")

    p_verify = sub.add_parser("verify", help="Verify an evidence artifact.")
    p_verify.add_argument("--store", required=True)
    p_verify.add_argument("--artifact-id", required=True)

    sub.add_parser("schema-version", help="Print the measurement schema version.")

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "compare":
        return _compare_main(args)
    if args.command == "decide":
        return _decide_main(args)
    if args.command == "verify":
        return _verify_main(args)
    if args.command == "schema-version":
        return _schema_version_main(args)
    _error(
        f"no handler for {args.command!r}",
        "Use compare, decide, verify, or schema-version",
        context="unknown command",
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
