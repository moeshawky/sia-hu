"""Computes weighted composite headroom scores from signal results. Provides
coverage analysis distinguishing measured from estimated/synthetic signals,
decision-grade gate evaluation, and recommendation generation based on signal-specific
thresholds."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from model_headroom.models.types import (
        CoverageMetrics,
        DeadNeuronReport,
        EffectiveRankResult,
        GradWeightRatio,
        GrowthFactorReport,
        HeadroomReport,
        LoadBalanceReport,
        SaturationReport,
        SignalResult,
        SpecializationReport,
    )

# Fixed signal weights summing to 1.0. loss_saturation and growth_factor
# carry highest weight (0.20 each) as primary training dynamics indicators; effective_rank
# and grad_weight_ratio moderate (0.15 each); dead_neurons, load_balance, specialization
# lowest (0.10 each).
DEFAULT_SIGNAL_WEIGHTS: dict[str, float] = {
    "loss_saturation": 0.20,
    "growth_factor": 0.20,
    "effective_rank": 0.15,
    "grad_weight_ratio": 0.15,
    "dead_neurons": 0.10,
    "load_balance": 0.10,
    "specialization": 0.10,
}

_MIN_COVERAGE_FOR_DECISION_GRADE = 0.7

# Contribution of a signal that produced no scoreable measurement.
#
# The composite must distinguish "we measured this and it is bad" from "we
# measured nothing". 0.0 is a MEASURED claim of maximal badness; asserting it
# from absent evidence is exactly the failure the mission forbids ("an absence
# of signals is not evidence that a model has zero headroom"). A signal with
# no evidence therefore contributes the neutral prior 0.5 — maximum ignorance
# — which is the behaviour the frozen contract specifies ("Skipped signals
# score 0.5 | Neutral prior, not a measurement", docs/concepts/frozen-contracts.md).
#
# A signal whose computation FAILED, or whose evidence was produced but is
# numerically invalid, contributes 0.0: a claim was attempted and it is not
# usable, which must not be laundered into a neutral "no information".
NEUTRAL_PRIOR = 0.5

# Architecturally impossible: excluded from numerator and denominator alike.
_EXCLUDED_STATUSES: frozenset[str] = frozenset({"not_applicable", "single_expert"})

# No measurement was obtained -> neutral prior.
_NO_EVIDENCE_STATUSES: frozenset[str] = frozenset(
    {"skipped", "insufficient_data", "no_gradient", "expert_count_unknown"}
)

# A measurement was attempted but is unusable -> no contribution (0.0).
_UNUSABLE_EVIDENCE_STATUSES: frozenset[str] = frozenset(
    {"error", "degenerate", "poor_fit", "invalid_evidence", "zero_weight"}
)

# Evidence state of a whole report. Distinguishes "no headroom" from
# "no headroom measurement", which the headline score alone cannot express.
EVIDENCE_STATE_NO_APPLICABLE = "no_applicable_signals"
EVIDENCE_STATE_INSUFFICIENT = "insufficient_evidence"
EVIDENCE_STATE_PARTIAL = "partial_coverage"
EVIDENCE_STATE_FULL = "full_coverage"


def _score_signal(signal: SignalResult) -> float | None:
    # Non-computed payloads carry no scoreable measurement: return the None
    # sentinel so the caller retains the weight but contributes 0.0 to the
    # numerator. not_applicable and single_expert are excluded from the
    # denominator entirely by compute_headroom_score.
    if signal.status != "computed":
        return None

    data = signal.data

    if signal.signal_name == "loss_saturation":
        return 0.0 if cast("SaturationReport", data).is_saturated else 1.0

    if signal.signal_name == "growth_factor":
        r = cast("GrowthFactorReport", data)
        if r.fit_quality_r2 < 0.5:
            return 0.3
        return float(min(1.0, max(0.0, r.alpha)))

    if signal.signal_name == "effective_rank":
        return float(cast("EffectiveRankResult", data).percentage_of_full_rank)

    if signal.signal_name == "grad_weight_ratio":
        ratios = cast("list[GradWeightRatio]", data)
        measured = [r for r in ratios if r.is_high_pressure is not None]
        if not measured:
            return None
        high_pressure_count = sum(1 for r in measured if r.is_high_pressure)
        total = len(measured)
        return 1.0 - (high_pressure_count / total)

    if signal.signal_name == "dead_neurons":
        dnr = cast("DeadNeuronReport", data)
        if dnr.degenerate_criterion:
            # Degenerate criterion: exact-zero amax never fires but
            # near_zero_occupancy is high — do not present constant 1.0
            # as a confident measurement. Return None so the neutral
            # prior (0.5) applies instead of a fabricated score.
            return None
        return float(1.0 - dnr.dead_fraction)

    if signal.signal_name == "load_balance":
        balance_ratio = cast("LoadBalanceReport", data).balance_ratio
        if balance_ratio is None:
            return 0.5
        return float(balance_ratio)

    if signal.signal_name == "specialization":
        return 0.0 if cast("SpecializationReport", data).is_collapsed else 1.0

    return 0.5


def _has_sampling_confidence(signal: SignalResult) -> bool:
    data = signal.data
    if data is None:
        return False
    return (
        getattr(data, "confidence", None) is not None
        or getattr(data, "coverage", None) is not None
    )


def _is_measured(signal: SignalResult) -> bool:
    # A signal is measured only when it was actually computed from measured
    # source. verification_status does NOT gate measurement: an unverified but
    # computed measured signal is still a measurement, just not decision-grade.
    return signal.status == "computed" and signal.source_type == "measured"


def compute_headroom_score(report: HeadroomReport) -> float:
    """Compute weighted composite headroom score from signal results. Iterates
    signals, skips not_applicable entirely (excluded from both numerator and denominator),
    scores each computed/skipped signal via _score_signal, applies DEFAULT_SIGNAL_WEIGHTS.

    The composite is a PRIOR-MIXED EXPECTATION, not a measurement: a signal
    that produced no scoreable value contributes the neutral prior 0.5, so an
    unmeasured signal can never manufacture a 0.0 (or 1.0) headroom claim.
    Signals excluded by architecture contribute neither numerator nor
    denominator. The measured-only score, the coverage fraction, and the
    uncertainty bounds live in compute_coverage_metrics — a composite number
    alone is never a measurement.

    Returns weighted average of scores over applicable signals. Returns 0.0 if no
    signals or no applicable weight."""
    if not report.signals:
        return 0.0

    total_weight = 0.0
    weighted_sum = 0.0

    for signal in report.signals:
        # not_applicable and single_expert are excluded from the denominator
        # entirely (architecture cannot produce the measurement).
        if signal.status in _EXCLUDED_STATUSES:
            continue
        weight = DEFAULT_SIGNAL_WEIGHTS.get(signal.signal_name, 0.1)
        score = _score_signal(signal)
        if score is None:
            score = _unmeasured_contribution(signal.status)
        weighted_sum += weight * score
        total_weight += weight

    return weighted_sum / total_weight if total_weight > 0 else 0.0


def _unmeasured_contribution(status: str) -> float:
    """Return the composite contribution of a signal that produced no scoreable
    measurement. A status that carries no evidence contributes the neutral prior;
    a status that carries unusable evidence contributes 0.0."""
    if status in _UNUSABLE_EVIDENCE_STATUSES:
        return 0.0
    if status in _NO_EVIDENCE_STATUSES:
        return NEUTRAL_PRIOR
    return NEUTRAL_PRIOR


def evidence_state(report: HeadroomReport) -> str:
    """Classify the evidence state of a report.

    Returns one of EVIDENCE_STATE_NO_APPLICABLE (no signal applies to this
    architecture), EVIDENCE_STATE_INSUFFICIENT (signals apply but none was
    measured), EVIDENCE_STATE_PARTIAL (some measured) or EVIDENCE_STATE_FULL
    (every applicable signal measured). The state is what a report must
    communicate when the composite score alone cannot: a report with no
    measurements has no headroom value, only an absence of one."""
    total_weight = 0.0
    measured_weight = 0.0
    for signal in report.signals:
        if signal.status in _EXCLUDED_STATUSES:
            continue
        weight = DEFAULT_SIGNAL_WEIGHTS.get(signal.signal_name, 0.1)
        total_weight += weight
        # A signal counts as measured only when it was actually computed from
        # measured source AND yields a scoreable value — the same definition
        # compute_coverage_metrics uses, so evidence_state and coverage never
        # disagree about what is "measured".
        if _is_measured(signal) and _score_signal(signal) is not None:
            measured_weight += weight
    if total_weight <= 0.0:
        return EVIDENCE_STATE_NO_APPLICABLE
    if measured_weight <= 0.0:
        return EVIDENCE_STATE_INSUFFICIENT
    if measured_weight < total_weight:
        return EVIDENCE_STATE_PARTIAL
    return EVIDENCE_STATE_FULL


def compute_coverage_metrics(report: HeadroomReport) -> CoverageMetrics:
    """Compute coverage and measurement quality metrics for a headroom report.
    Separates measured signals (computed with verified source) from estimated/synthetic/skipped
    signals. Returns CoverageMetrics with measured_score (score over measured signals
    only), coverage (fraction of applicable weight that is measured), lower_bound
    (worst-case if unmeasured signals score 0), upper_bound (best-case if unmeasured
    signals score 1.0), and decision_grade flag."""
    from model_headroom.models.types import CoverageMetrics

    if not report.signals:
        return CoverageMetrics(
            measured_score=0.0,
            coverage=0.0,
            lower_bound=0.0,
            upper_bound=0.0,
            measured_weight=0.0,
            total_applicable_weight=0.0,
            unmeasured_weight=0.0,
            decision_grade=False,
        )

    total_applicable_weight = 0.0
    measured_weight = 0.0
    weighted_sum_measured = 0.0

    for signal in report.signals:
        # not_applicable and single_expert are not applicable to this
        # architecture; exclude them from the applicable-weight denominator.
        if signal.status in _EXCLUDED_STATUSES:
            continue
        weight = DEFAULT_SIGNAL_WEIGHTS.get(signal.signal_name, 0.1)
        total_applicable_weight += weight
        if _is_measured(signal):
            score = _score_signal(signal)
            # A signal whose computation is valid but whose evidence is
            # non-discriminating (e.g. dead_neurons degenerate criterion)
            # contributes no measured information and must not inflate the
            # measured-only score or coverage.
            if score is None:
                continue
            weighted_sum_measured += weight * score
            measured_weight += weight

    unmeasured_weight = total_applicable_weight - measured_weight
    coverage = (
        measured_weight / total_applicable_weight
        if total_applicable_weight > 0
        else 0.0
    )
    measured_score = (
        weighted_sum_measured / measured_weight if measured_weight > 0 else 0.0
    )
    lower_bound = (
        weighted_sum_measured / total_applicable_weight
        if total_applicable_weight > 0
        else 0.0
    )
    upper_bound = (
        (weighted_sum_measured + unmeasured_weight) / total_applicable_weight
        if total_applicable_weight > 0
        else 0.0
    )

    decision_grade = not _coverage_gate_gaps(report, coverage)

    return CoverageMetrics(
        measured_score=measured_score,
        coverage=coverage,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        measured_weight=measured_weight,
        total_applicable_weight=total_applicable_weight,
        unmeasured_weight=unmeasured_weight,
        decision_grade=decision_grade,
    )


_GRADIENT_DERIVED_SIGNALS: tuple[str, ...] = ("effective_rank", "grad_weight_ratio")


def _coverage_gate_gaps(report: HeadroomReport, coverage: float) -> list[str]:
    gaps: list[str] = []
    if coverage < _MIN_COVERAGE_FOR_DECISION_GRADE:
        gaps.append(
            f"coverage {coverage:.0%} below decision-grade threshold "
            f"{_MIN_COVERAGE_FOR_DECISION_GRADE:.0%}"
        )
    unverified = sorted(
        {
            s.signal_name
            for s in report.signals
            if s.status == "computed"
            and (
                s.source_type in ("synthetic", "fixture", "unknown", "estimated")
                or s.verification_status in ("unverified", "provenance_incomplete")
            )
        }
    )
    if unverified:
        gaps.append(
            f"unverified or estimated computed signals: {', '.join(unverified)}"
        )
    if any(
        s.status == "computed" and s.signal_name in _GRADIENT_DERIVED_SIGNALS
        for s in report.signals
    ):
        prov = report.metadata.get("gradient_provenance")
        prov_map: dict[str, Any] = prov if isinstance(prov, dict) else {}
        model_revision = prov_map.get("model_revision")
        dataset_hash = prov_map.get("dataset_hash")
        if model_revision is None or dataset_hash is None:
            gaps.append(
                "gradient provenance incomplete: model_revision and dataset_hash "
                "required for computed gradient-derived signals"
            )
        elif model_revision != report.model_name:
            gaps.append(
                "gradient provenance identity mismatch: model_revision "
                f"{model_revision!r} does not match model being scored "
                f"{report.model_name!r}"
            )
    weak_sampling = sorted(
        {
            s.signal_name
            for s in report.signals
            if s.status == "computed" and s.sampling_mode in ("single_tile", "unknown")
        }
    )
    if weak_sampling:
        gaps.append(
            f"non-full-matrix sampling on computed signals: {', '.join(weak_sampling)}"
        )
    return gaps


def decision_grade_gaps(report: HeadroomReport) -> list[str]:
    """Return list of gap descriptions preventing decision-grade status.
    Checks: coverage below 70% threshold, unverified or estimated computed signals,
    incomplete gradient provenance for gradient-derived signals, non-full-matrix
    sampling modes. Returns empty list if all gates pass."""
    return _coverage_gate_gaps(report, compute_coverage_metrics(report).coverage)


def is_decision_grade(report: HeadroomReport) -> bool:
    """Return True if the report passes all decision-grade gates. Delegates
    to compute_coverage_metrics and returns the decision_grade boolean field."""
    return compute_coverage_metrics(report).decision_grade


def generate_recommendations(report: HeadroomReport) -> list[str]:
    """Generate actionable recommendations based on signal results. Iterates
    computed signals and applies signal-specific threshold checks: saturated loss,
    growth factor crossover, low effective rank (< 0.3), high gradient pressure,
    dead fraction > 0.1, load imbalance (< 0.5), expert collapse. Adds coverage
    and provenance warnings. Pads with generic monitoring advice to ensure at least
    3 recommendations."""
    recommendations: list[str] = []

    for signal in report.signals:
        if signal.status != "computed":
            continue

        data = signal.data

        if signal.signal_name == "loss_saturation" and data.is_saturated:
            recommendations.append(
                "Loss curve has saturated. Consider increasing data diversity "
                "or adjusting learning rate schedule."
            )

        if (
            signal.signal_name == "growth_factor"
            and data.fit_quality_r2 >= 0.5
            and data.crossover_params is not None
        ):
            recommendations.append(
                f"Growth factor crossover at {data.crossover_params:,} params. "
                f"Diminishing returns expected beyond this scale."
            )

        if (
            signal.signal_name == "effective_rank"
            and data.percentage_of_full_rank < 0.3
        ):  # HEURISTIC: 0.3 threshold is uncalibrated; low tile-relative rank
            # may reflect efficient representation, not spare capacity. See
            # mission Phase 2: spectral occupancy != trainability.
            recommendations.append(
                "Low effective rank suggests weight matrices are low-rank. "
                "Consider pruning or low-rank adaptation."
            )

        if signal.signal_name == "grad_weight_ratio":
            high_count = sum(1 for r in data if r.is_high_pressure)
            if high_count > 0:
                recommendations.append(
                    f"{high_count} layers show high gradient-to-weight pressure. "
                    "Consider gradient clipping or learning rate reduction."
                )

        if signal.signal_name == "dead_neurons" and data.dead_fraction > 0.1:
            recommendations.append(
                f"{data.dead_fraction:.1%} of neurons are dead. "
                "Consider adjusting initialization or activation functions."
            )

        if signal.signal_name == "load_balance" and data.balance_ratio < 0.5:
            recommendations.append(
                "Expert load is unbalanced. Consider auxiliary load balancing "
                "loss or adjusting router temperature."
            )

        if signal.signal_name == "specialization" and data.is_collapsed:
            recommendations.append(
                "Experts have collapsed (high inter-expert similarity). "
                "Consider increasing expert diversity or adding noise to routing."
            )

    metrics = compute_coverage_metrics(report)
    state = evidence_state(report)
    unmeasured = sorted(
        s.signal_name
        for s in report.signals
        if s.status not in _EXCLUDED_STATUSES and not _is_measured(s)
    )
    if state == EVIDENCE_STATE_INSUFFICIENT:
        recommendations.append(
            "INSUFFICIENT EVIDENCE: no signal was measured, so no headroom value "
            "can be asserted for this model. The composite carries a neutral "
            f"{NEUTRAL_PRIOR:.1f} prior for every applicable signal "
            f"({', '.join(unmeasured)}) and is NOT a measurement."
        )
    elif metrics.coverage < 0.5:
        recommendations.append(
            f"Low measurement coverage ({metrics.coverage:.0%} measured). "
            "Composite score mixes a neutral 0.5 prior for unmeasured signals "
            f"({', '.join(unmeasured)}); "
            f"measured-only score is {metrics.measured_score:.3f} "
            f"with possible range [{metrics.lower_bound:.3f}, {metrics.upper_bound:.3f}]."
        )
    elif metrics.coverage < _MIN_COVERAGE_FOR_DECISION_GRADE:
        recommendations.append(
            f"Measurement coverage {metrics.coverage:.0%} below decision-grade threshold "
            f"({_MIN_COVERAGE_FOR_DECISION_GRADE:.0%}). Collect missing signals before "
            "finalizing SFT decision."
        )
    if any(
        s.source_type in ("synthetic", "fixture", "estimated", "unknown")
        for s in report.signals
        if s.status == "computed"
    ):
        recommendations.append(
            "Synthetic, fixture, or estimated inputs contributed to computed signals; "
            "replace with measured data for decision-grade report."
        )
    if any(
        s.verification_status in ("unverified", "provenance_incomplete")
        for s in report.signals
        if s.status == "computed"
    ):
        recommendations.append(
            "Unverified or incomplete provenance detected; establish gradient and activation "
            "provenance (model revision, dataset hash, token count) before scoring."
        )

    if len(recommendations) < 3:
        if metrics.coverage == 0.0:
            recommendations.append(
                "No real signals computed; score is a neutral prior, not a measurement."
            )
        else:
            recommendations.append(
                "Overall training headroom appears adequate. "
                "Monitor signals during continued training."
            )
    if len(recommendations) < 3:
        recommendations.append(
            "Run analysis periodically to track changes in headroom signals."
        )
    if len(recommendations) < 3:
        recommendations.append("Compare results across checkpoints to identify trends.")

    return recommendations
