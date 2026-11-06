"""Runs the 7-signal headroom analysis pipeline. Resolves model type,
dispatches signal functions with timing and error capture, aggregates results
into a single HeadroomReport with composite score, coverage metrics, and recommendations."""

from __future__ import annotations

import time
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, cast

from model_headroom.analysis.scoring import (
    compute_coverage_metrics,
    compute_headroom_score,
    decision_grade_gaps,
    generate_recommendations,
)
from model_headroom.models.loader import (
    detect_architecture,
    detect_architecture_details,
    get_module_family,
)
from model_headroom.models.types import (
    AnalysisMode,
    ArchitectureType,
    ExpertActivationData,
    ExpertWeights,
    GradientProvenance,
    HeadroomReport,
    SamplingMode,
    SignalResult,
    SourceType,
    VerificationStatus,
)
from model_headroom.signals.dead_neurons import analyze_dead_neurons
from model_headroom.signals.effective_rank import compute_effective_rank
from model_headroom.signals.grad_weight_ratio import analyze_grad_weight_ratios
from model_headroom.signals.growth_factor import analyze_growth_factor
from model_headroom.signals.load_balance import analyze_load_balance
from model_headroom.signals.loss_saturation import analyze_loss_saturation
from model_headroom.signals.specialization import analyze_specialization

if TYPE_CHECKING:
    from collections.abc import Callable

    from model_headroom.models.types import GradWeightRatio

_COVERAGE_REFERENCE = 0.2
_COVERAGE_REFERENCE_BASIS = (
    "empirical, synthetic power-law spectra; not a constant of nature"
)


def _tile_coverage(weight: Any, parent_shape: tuple[int, int] | None) -> float | None:
    if parent_shape is None:
        return None
    parent_elements = parent_shape[0] * parent_shape[1]
    if parent_elements <= 0:
        return None
    shape = cast("tuple[int, ...]", weight.shape)
    return (shape[0] * shape[1]) / parent_elements


def _build_r03_comparability(
    pairs: list[Any],
    results: list[Any],
) -> dict[str, Any]:
    coverages: list[float] = []
    layers_considered = 0
    layers_flagged_truncated = 0
    layers_untruncated_but_partial = 0
    best_pct: float | None = None
    coverage_reported: float | None = None

    for pair, result in zip(pairs, results, strict=True):
        parent_shape = getattr(pair, "parent_shape", None)
        if parent_shape is None:
            continue
        layers_considered += 1
        coverage = _tile_coverage(pair.weight, parent_shape)
        if coverage is not None:
            coverages.append(coverage)
            if getattr(pair, "truncated", None) is False and coverage < 1.0:
                layers_untruncated_but_partial += 1
        if getattr(pair, "truncated", None) is True:
            layers_flagged_truncated += 1
        if result.status != "computed" or result.data is None:
            continue
        percentage = result.data.percentage_of_full_rank
        if best_pct is None or percentage < best_pct:
            best_pct = percentage
            coverage_reported = coverage

    coverage_min = min(coverages) if coverages else None
    coverage_max = max(coverages) if coverages else None

    if coverage_min is None or coverage_min <= 0.0:
        state = "absent"
    elif coverage_min >= 1.0:
        state = "full"
    else:
        state = "sampled"

    return {
        "state": state,
        "comparable_to_full_matrix": state == "full",
        "coverage_definition": "tile_elements / parent_elements",
        "coverage_reported": coverage_reported,
        "coverage_min": coverage_min,
        "coverage_max": coverage_max,
        "layers_considered": layers_considered,
        "layers_flagged_truncated": layers_flagged_truncated,
        "layers_untruncated_but_partial": layers_untruncated_but_partial,
        "coverage_reference": _COVERAGE_REFERENCE,
        "coverage_reference_basis": _COVERAGE_REFERENCE_BASIS,
        "below_coverage_reference": (
            coverage_min is not None and coverage_min < _COVERAGE_REFERENCE
        ),
        "measured_inflation_ratio": None,
        "score_source": "effective_rank.percentage_of_full_rank",
    }


def _run_signal(
    name: str, func: Callable[..., Any], *args: Any, **kwargs: Any
) -> SignalResult:
    start = time.monotonic()
    try:
        data = func(*args, **kwargs)
        duration = time.monotonic() - start
        if hasattr(data, "signal_status") and data.signal_status in (
            "error",
            "single_expert",
            "expert_count_unknown",
        ):
            return SignalResult(
                signal_name=name,
                status=data.signal_status,
                data=data,
                duration_seconds=duration,
            )
        return SignalResult(
            signal_name=name,
            status="computed",
            data=data,
            duration_seconds=duration,
        )
    except Exception as e:
        duration = time.monotonic() - start
        return SignalResult(
            signal_name=name,
            status="error",
            data=None,
            duration_seconds=duration,
            error_message=str(e),
        )


def run_analysis(
    model_name: str,
    model_path: str,
    loss_curve: list[Any] | None = None,
    isoflop_points: list[Any] | None = None,
    weight_pairs: list[Any] | None = None,
    mlp_activations: list[Any] | None = None,
    expert_weight_sets: list[Any] | None = None,
    activation_data: Any | None = None,
    architecture: ArchitectureType | None = None,
    metadata: dict[str, Any] | None = None,
    analysis_mode: AnalysisMode | None = None,
    gradient_provenance: GradientProvenance | None = None,
    mlp_source_type: SourceType | None = None,
    mlp_verification_status: VerificationStatus | None = None,
    effective_rank_sampling_mode: SamplingMode | None = None,
    effective_rank_parent_shapes: dict[str, tuple[int, int]] | None = None,
    effective_rank_sample_count: int | None = None,
    weight_pairs_source_type: SourceType | None = None,
    weight_pairs_verification_status: VerificationStatus | None = None,
    compute_gradient_diagnostics: bool = False,
) -> HeadroomReport:
    """Execute all applicable headroom signals and produce a composite report.
    Resolves model type (auto-detect or override), dispatches 7 signal functions
    (loss_saturation, growth_factor, effective_rank, grad_weight_ratio, dead_neurons,
    load_balance, specialization), marks signals as skipped when input data is absent,
    marks load_balance and specialization as single_expert for dense models, selects
    worst-layer results for multi-layer signals, and assembles HeadroomReport with
    composite score, coverage metrics, and recommendations."""
    signals: list[SignalResult] = []

    resolved_architecture = architecture or detect_architecture(model_path)
    details = detect_architecture_details(model_path)
    resolved_mode: AnalysisMode = analysis_mode or "diagnostic"
    report_metadata: dict[str, Any] = {
        **(metadata or {}),
        "architecture": resolved_architecture,
        "architecture_details": {
            "family": details.family,
            "subtype": details.subtype,
            "has_moe": details.has_moe,
            "has_full_attention": details.has_full_attention,
            "has_linear_attention": details.has_linear_attention,
            "has_vision_tower": details.has_vision_tower,
            "has_mtp": details.has_mtp,
            "detection_source": details.detection_source,
        },
        "analysis_mode": resolved_mode,
    }

    if gradient_provenance is not None:
        report_metadata["gradient_provenance"] = {
            "model_revision": gradient_provenance.model_revision,
            "dataset_hash": gradient_provenance.dataset_hash,
            "dataset_type": gradient_provenance.dataset_type,
            "token_count": gradient_provenance.token_count,
            "batch_count": gradient_provenance.batch_count,
            "loss_function": gradient_provenance.loss_function,
            "source_type": gradient_provenance.source_type,
            "verification_status": gradient_provenance.verification_status,
        }

    if weight_pairs_source_type is None and gradient_provenance is not None:
        weight_pairs_source_type = gradient_provenance.source_type
        weight_pairs_verification_status = gradient_provenance.verification_status

    eff_source: SourceType = weight_pairs_source_type or "measured"
    eff_verif: VerificationStatus = weight_pairs_verification_status or "verified"
    grad_source: SourceType = weight_pairs_source_type or "measured"
    grad_verif: VerificationStatus = weight_pairs_verification_status or "verified"
    if gradient_provenance is not None:
        grad_source = gradient_provenance.source_type
        grad_verif = gradient_provenance.verification_status

    mlp_src: SourceType = mlp_source_type or "measured"
    mlp_ver: VerificationStatus = mlp_verification_status or "verified"

    sampling_mode: SamplingMode = effective_rank_sampling_mode or "unknown"

    if loss_curve:
        signals.append(
            _run_signal(
                "loss_saturation",
                analyze_loss_saturation,
                loss_curve,
            )
        )
    else:
        signals.append(
            SignalResult(
                signal_name="loss_saturation",
                status="skipped",
                data=None,
                duration_seconds=0.0,
            )
        )

    if isoflop_points:
        signals.append(
            _run_signal(
                "growth_factor",
                analyze_growth_factor,
                isoflop_points,
            )
        )
    else:
        signals.append(
            SignalResult(
                signal_name="growth_factor",
                status="skipped",
                data=None,
                duration_seconds=0.0,
            )
        )

    if weight_pairs:
        rank_results: list[SignalResult] = []
        pair_provenance: list[dict[str, Any]] = []
        for pair in weight_pairs:
            parent_shape = None
            if effective_rank_parent_shapes is not None:
                parent_shape = effective_rank_parent_shapes.get(pair.layer_name)
            if parent_shape is None:
                parent_shape = getattr(pair, "parent_shape", None)
            pair_sampling = sampling_mode
            if pair_sampling == "unknown":
                pair_sampling = getattr(pair, "sampling_mode", None) or "unknown"
            pair_truncated = getattr(pair, "truncated", None)
            if pair_truncated is not None:
                pair_provenance.append(
                    {
                        "layer_name": pair.layer_name,
                        "truncated": pair_truncated,
                        "parent_shape": (
                            list(parent_shape) if parent_shape is not None else None
                        ),
                        "sampling_mode": pair_sampling,
                    }
                )
            rank_results.append(
                _run_signal(
                    f"effective_rank_{pair.layer_name}",
                    compute_effective_rank,
                    pair.weight,
                    pair_sampling,
                    parent_shape,
                    sample_count=effective_rank_sample_count,
                )
            )
        if pair_provenance:
            report_metadata["weight_pair_provenance"] = pair_provenance
            report_metadata["effective_rank_comparability"] = _build_r03_comparability(
                weight_pairs, rank_results
            )
        computed = [r for r in rank_results if r.status == "computed"]
        if computed:
            worst = min(computed, key=lambda r: r.data.percentage_of_full_rank)
            signals.append(
                SignalResult(
                    signal_name="effective_rank",
                    status="computed",
                    data=worst.data,
                    duration_seconds=sum(r.duration_seconds for r in rank_results),
                    source_type=eff_source,
                    verification_status=eff_verif,
                    sampling_mode=sampling_mode,
                )
            )
        else:
            signals.append(
                SignalResult(
                    signal_name="effective_rank",
                    status="error",
                    data=None,
                    duration_seconds=sum(r.duration_seconds for r in rank_results),
                    source_type=eff_source,
                    verification_status=eff_verif,
                    sampling_mode=sampling_mode,
                )
            )

        rank_families: dict[str, Any] = {}
        for pair, r in zip(weight_pairs, rank_results, strict=True):
            if r.status != "computed" or r.data is None:
                continue
            fam = get_module_family(pair.layer_name)
            entry = rank_families.setdefault(
                fam,
                {"min_percentage_of_full_rank": 1.0, "worst_layer": pair.layer_name},
            )
            if r.data.percentage_of_full_rank < entry["min_percentage_of_full_rank"]:
                entry["min_percentage_of_full_rank"] = r.data.percentage_of_full_rank
                entry["worst_layer"] = pair.layer_name
        if rank_families:
            report_metadata.setdefault("module_family_summary", {})[
                "effective_rank"
            ] = rank_families
    else:
        signals.append(
            SignalResult(
                signal_name="effective_rank",
                status="skipped",
                data=None,
                duration_seconds=0.0,
            )
        )

    if weight_pairs:
        base = _run_signal(
            "grad_weight_ratio",
            analyze_grad_weight_ratios,
            weight_pairs,
        )
        signals.append(
            SignalResult(
                signal_name=base.signal_name,
                status=base.status,
                data=base.data,
                duration_seconds=base.duration_seconds,
                error_message=base.error_message,
                source_type=grad_source,
                verification_status=grad_verif,
            )
        )

        if base.status == "computed" and isinstance(base.data, list):
            ratios = cast("list[GradWeightRatio]", base.data)
            pressure: dict[str, dict[str, int]] = {}
            for ratio in ratios:
                fam = get_module_family(ratio.layer_name)
                entry = pressure.setdefault(fam, {"layers": 0, "high_pressure": 0})
                entry["layers"] += 1
                if ratio.is_high_pressure:
                    entry["high_pressure"] += 1
            if pressure:
                report_metadata.setdefault("module_family_summary", {})[
                    "grad_pressure"
                ] = pressure
    else:
        signals.append(
            SignalResult(
                signal_name="grad_weight_ratio",
                status="skipped",
                data=None,
                duration_seconds=0.0,
            )
        )

    if mlp_activations:
        all_dead: list[SignalResult] = []
        total_duration = 0.0
        for act in mlp_activations:
            result = _run_signal(
                f"dead_neurons_{act.layer_name}",
                analyze_dead_neurons,
                act,
                1e-6,
                False,
                mlp_src,
            )
            total_duration += result.duration_seconds
            if result.status == "computed":
                all_dead.append(result)

        if all_dead:
            worst = max(all_dead, key=lambda r: r.data.dead_fraction)
            signals.append(
                SignalResult(
                    signal_name="dead_neurons",
                    status="computed",
                    data=worst.data,
                    duration_seconds=total_duration,
                    source_type=mlp_src,
                    verification_status=mlp_ver,
                )
            )
        else:
            signals.append(
                SignalResult(
                    signal_name="dead_neurons",
                    status="error",
                    data=None,
                    duration_seconds=total_duration,
                    source_type=mlp_src,
                    verification_status=mlp_ver,
                )
            )
    else:
        signals.append(
            SignalResult(
                signal_name="dead_neurons",
                status="skipped",
                data=None,
                duration_seconds=0.0,
            )
        )

    if resolved_architecture == "dense":
        signals.append(
            _run_signal(
                "load_balance",
                analyze_load_balance,
                ExpertActivationData(
                    metadata={"num_experts": 1, "num_experts_architectural": 1},
                    layers={},
                ),
            )
        )
    elif activation_data:
        signals.append(
            _run_signal(
                "load_balance",
                analyze_load_balance,
                activation_data,
            )
        )
    else:
        signals.append(
            SignalResult(
                signal_name="load_balance",
                status="skipped",
                data=None,
                duration_seconds=0.0,
            )
        )

    if resolved_architecture == "dense":
        signals.append(
            _run_signal(
                "specialization",
                analyze_specialization,
                ExpertWeights(expert_weights=[], layer_name="dense"),
            )
        )
    elif expert_weight_sets:
        collapse_results: list[SignalResult] = []
        single_expert_results: list[SignalResult] = []
        total_duration = 0.0
        for ew in expert_weight_sets:
            result = _run_signal(
                f"specialization_{ew.layer_name}",
                analyze_specialization,
                ew,
            )
            total_duration += result.duration_seconds
            if result.status == "computed":
                collapse_results.append(result)
            elif result.status == "single_expert":
                single_expert_results.append(result)

        if collapse_results:
            worst = max(
                collapse_results, key=lambda r: r.data.mean_inter_expert_similarity
            )
            signals.append(
                SignalResult(
                    signal_name="specialization",
                    status="computed",
                    data=worst.data,
                    duration_seconds=total_duration,
                )
            )
        elif single_expert_results:
            signals.append(
                SignalResult(
                    signal_name="specialization",
                    status="single_expert",
                    data=single_expert_results[0].data,
                    duration_seconds=total_duration,
                )
            )
        else:
            signals.append(
                SignalResult(
                    signal_name="specialization",
                    status="error",
                    data=None,
                    duration_seconds=total_duration,
                )
            )
    else:
        signals.append(
            SignalResult(
                signal_name="specialization",
                status="skipped",
                data=None,
                duration_seconds=0.0,
            )
        )

    if compute_gradient_diagnostics and weight_pairs:
        try:
            from model_headroom.signals.gradient_coherence import (
                compute_coherence_for_layer,
            )
            from model_headroom.signals.gradient_effective_rank import (
                analyze_gradient_effective_ranks_for_pairs,
            )

            sample_pairs = weight_pairs[:8]
            rank_reports = analyze_gradient_effective_ranks_for_pairs(sample_pairs)
            # Cross-batch contract: coherence is only meaningful across >=2
            # batches of the SAME layer/projection. Group pairs by layer name;
            # layers with a single batch yield no coherence (unavailable, never
            # zero), and cross-layer lists must never be fed as batches.
            layer_groups: dict[str, list[Any]] = {}
            for p in weight_pairs:
                layer_groups.setdefault(p.layer_name, []).append(p)
            coherence_by_layer: dict[str, dict[str, Any]] = {}
            for layer_name, group in layer_groups.items():
                if len(group) < 2:
                    continue
                rep = compute_coherence_for_layer(group)
                if rep.signal_status == "computed":
                    coherence_by_layer[layer_name] = {
                        "mean_cosine": rep.mean_cosine_similarity,
                        "subspace_overlap": rep.subspace_overlap,
                        "batches": len(group),
                    }
            coherence: dict[str, Any] | None = None
            coherence_note: str | None = None
            if len(coherence_by_layer) == 1:
                ((only_layer, only_coh),) = coherence_by_layer.items()
                coherence = {"layer": only_layer, **only_coh}
            elif coherence_by_layer:
                coherence_note = (
                    f"coherence computed per layer for "
                    f"{sorted(coherence_by_layer)}; no single-layer summary"
                )
            else:
                coherence_note = (
                    "coherence unavailable: no layer has >=2 gradient "
                    f"batches (got 1 batch each across {len(layer_groups)} "
                    "layer(s)); cross-layer gradients are not batches"
                )
            family_groups: dict[str, list[Any]] = {}
            for p in weight_pairs:
                fam = get_module_family(p.layer_name)
                family_groups.setdefault(fam, []).append(p)
            report_metadata["gradient_diagnostics"] = {
                "gradient_effective_rank": [
                    {
                        "layer": layer_name,
                        "status": r.signal_status,
                        "rank": r.rank,
                        "r50": r.r50,
                        "r90": r.r90,
                        "r95": r.r95,
                    }
                    for layer_name, r in rank_reports
                ],
                "coherence": coherence,
                "coherence_by_layer": coherence_by_layer,
                "coherence_note": coherence_note,
                "module_families": list(family_groups.keys()),
            }
        except Exception:  # noqa: S110
            pass

    report = HeadroomReport(
        model_name=model_name,
        model_path=model_path,
        timestamp=datetime.now(UTC).isoformat(),
        signals=signals,
        overall_headroom_score=0.0,
        recommendations=[],
        metadata=report_metadata,
        analysis_mode=resolved_mode,
    )

    score = compute_headroom_score(report)
    coverage = compute_coverage_metrics(report)
    report_metadata["decision_grade_gaps"] = decision_grade_gaps(report)
    recommendations = generate_recommendations(report)

    return HeadroomReport(
        model_name=model_name,
        model_path=model_path,
        timestamp=report.timestamp,
        signals=signals,
        overall_headroom_score=score,
        recommendations=recommendations,
        metadata=report_metadata,
        coverage_metrics=coverage,
        analysis_mode=resolved_mode,
        decision_grade=coverage.decision_grade,
    )
