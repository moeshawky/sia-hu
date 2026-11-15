"""Per-strategy recovery verdict engine.

Reads InterventionRecord evidence and emits a SUPPORTED / REFUTED /
INCONCLUSIVE verdict for each candidate recovery strategy. INCONCLUSIVE is
the default; it is returned whenever measurements cannot distinguish the
strategies. No composite headroom score is converted into a derived number,
and no derived-number field exists anywhere in this module.

Polarity: InterventionRecord.compute_delta() returns post - base, so the sign
of a delta alone does not say whether the change is an improvement. A loss or
error-like metric improves when it goes down; a reward-like metric improves
when it goes up. evaluate_strategy() and decide() therefore take an explicit
metric_direction ("lower_is_better" default, or "higher_is_better") that
orients the raw delta before a verdict is assigned. observed_delta keeps the
raw post - base value; metric_direction is recorded on StrategyEvidence so the
report states which polarity produced the verdict. The default is
lower_is_better because every quantity this tool measures (cross-entropy,
token-level KL, normalized reconstruction error, effective-weight
reconstruction error, gradient norm) is error-like and lower is better.

Held-out rule: a verdict other than INCONCLUSIVE requires at least one
held-out key that was actually measured, whatever its sign. When
held_out_keys is None or empty the per-key deltas are still computed over the
shared keys and reported in observed_delta, because the magnitude is evidence,
but the verdict is INCONCLUSIVE with downgrade_reasons naming the missing
held-out set: an in-sample delta is measured on the data the intervention was
fit on, so it cannot separate "the strategy helped" from "the strategy was fit
to this measurement". The rule is symmetric on purpose: REFUTED is a
recommendation too, and one evidence bar is easier to keep than two.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from statistics import mean
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from collections.abc import Sequence

    from model_headroom.analysis.evidence_store import EvidenceArtifact, EvidenceStore
    from model_headroom.analysis.intervention_record import InterventionRecord
    from model_headroom.paired.compare import ComparisonReport

COVERAGE_THRESHOLD: float = 0.8

MetricDirection = Literal["lower_is_better", "higher_is_better"]

RecoveryStrategy = Literal[
    "continuous_parameters_frozen_codes",
    "targeted_low_rank",
    "rebuild_source_before_quantization",
]

Verdict = Literal["SUPPORTED", "REFUTED", "INCONCLUSIVE"]

_STRATEGY_BY_TYPE: dict[str, RecoveryStrategy] = {
    "continuous_parameters_frozen_codes": "continuous_parameters_frozen_codes",
    "continuous_parameters": "continuous_parameters_frozen_codes",
    "frozen_codes": "continuous_parameters_frozen_codes",
    "quantization_recovery": "continuous_parameters_frozen_codes",
    "targeted_low_rank": "targeted_low_rank",
    "low_rank": "targeted_low_rank",
    "lora": "targeted_low_rank",
    "qlora": "targeted_low_rank",
    "rebuild_source_before_quantization": "rebuild_source_before_quantization",
    "rebuild_source": "rebuild_source_before_quantization",
    "rebuild": "rebuild_source_before_quantization",
    "retrain": "rebuild_source_before_quantization",
}

_FAMILY_BY_SUFFIX: tuple[tuple[str, str], ...] = (
    ("attention.output.dense", "attention"),
    ("q_proj", "attention"),
    ("k_proj", "attention"),
    ("v_proj", "attention"),
    ("o_proj", "attention"),
    ("gate_proj", "mlp"),
    ("up_proj", "mlp"),
    ("down_proj", "mlp"),
    ("intermediate", "mlp"),
    ("output.dense", "mlp"),
    ("embedding", "embedding"),
    ("embeddings", "embedding"),
    ("ln", "norm"),
    ("norm", "norm"),
)


@dataclass(frozen=True)
class StrategyEvidence:
    """Frozen evidence and verdict for one recovery strategy."""

    strategy: RecoveryStrategy
    metric_direction: MetricDirection
    intervention_id: str | None
    observed_delta: float | None
    held_out_keys: tuple[str, ...]
    affected_families: tuple[str, ...]
    memory_overhead_gb: float | None
    parameter_overhead: int | None
    uncertainty: float | None
    coverage: float | None
    verdict: Verdict
    verdict_reason: str
    missing_evidence: tuple[str, ...]
    next_experiment: str | None
    downgrade_reasons: tuple[str, ...]


@dataclass(frozen=True)
class RecoveryDecision:
    """Frozen per-strategy verdict and distinguishability result."""

    strategies: tuple[StrategyEvidence, ...]
    distinguishable: bool
    summary: str
    warnings: tuple[str, ...]


def _strategy_for(record: InterventionRecord) -> RecoveryStrategy:
    return _STRATEGY_BY_TYPE.get(
        record.intervention_type, "continuous_parameters_frozen_codes"
    )


def _families_for(affected_parameters: Sequence[str]) -> tuple[str, ...]:
    families: list[str] = []
    for name in affected_parameters:
        matched = False
        for suffix, family in _FAMILY_BY_SUFFIX:
            if name.endswith(suffix):
                families.append(family)
                matched = True
                break
        if not matched:
            families.append("unknown")
    return tuple(families)


def _improved(observed_delta: float, metric_direction: MetricDirection) -> bool:
    if metric_direction == "lower_is_better":
        return observed_delta < 0.0
    return observed_delta > 0.0


def _next_experiment_supported(
    keys: Sequence[str], metric_direction: MetricDirection
) -> str:
    key_list = ", ".join(keys) if keys else "the recorded signals"
    if metric_direction == "lower_is_better":
        condition = "mean delta is not strictly less than 0.0"
    else:
        condition = "mean delta is not strictly greater than 0.0"
    return (
        "Re-run the held-out measurement on "
        + key_list
        + " with metric_direction="
        + metric_direction
        + "; refute the strategy if "
        + condition
        + "."
    )


def _next_experiment_refuted(
    keys: Sequence[str], metric_direction: MetricDirection
) -> str:
    key_list = ", ".join(keys) if keys else "the recorded signals"
    if metric_direction == "lower_is_better":
        condition = "mean delta is not strictly greater than 0.0"
    else:
        condition = "mean delta is not strictly less than 0.0"
    return (
        "Re-run the held-out measurement on "
        + key_list
        + " with metric_direction="
        + metric_direction
        + "; refute the regression if "
        + condition
        + "."
    )


def _next_experiment_inconclusive(
    keys: Sequence[str], coverage: float | None, metric_direction: MetricDirection
) -> str:
    key_list = ", ".join(keys) if keys else "the recorded signals"
    coverage_clause = (
        " with coverage at or above " + str(COVERAGE_THRESHOLD)
        if coverage is not None
        else " and a coverage estimate"
    )
    if metric_direction == "lower_is_better":
        condition = "mean delta is not strictly less than 0.0"
    else:
        condition = "mean delta is not strictly greater than 0.0"
    return (
        "Collect a held-out measurement on "
        + key_list
        + coverage_clause
        + " with metric_direction="
        + metric_direction
        + "; refute the strategy if "
        + condition
        + " or coverage is below "
        + str(COVERAGE_THRESHOLD)
        + "."
    )


def evaluate_strategy(
    record: InterventionRecord,
    *,
    held_out_keys: Sequence[str] | None = None,
    coverage: float | None = None,
    metric_direction: MetricDirection = "lower_is_better",
) -> StrategyEvidence:
    deltas = record.compute_delta()
    keys: list[str] = []
    absent: list[str] = []
    if held_out_keys:
        for key in held_out_keys:
            if key in deltas:
                keys.append(key)
            else:
                absent.append(key)
    else:
        keys = list(deltas.keys())
    observed_delta: float | None = None
    if keys:
        observed_delta = float(mean([deltas[key] for key in keys]))
    held_out_supplied = held_out_keys is not None and len(held_out_keys) > 0
    no_held_out_supplied = not held_out_supplied
    downgrade: list[str] = []
    missing: list[str] = []
    if not keys:
        downgrade.append("no shared numeric keys between baseline and post")
        missing.append("shared numeric measurement between baseline and post")
    if absent:
        downgrade.append("held-out keys not measured: " + ", ".join(absent))
        for key in absent:
            missing.append("missing held-out measurement: " + key)
    if no_held_out_supplied:
        downgrade.append("no held-out keys supplied")
        missing.append("held-out measurement set")
    if coverage is None:
        downgrade.append("coverage unknown")
        missing.append("coverage estimate")
    elif coverage < COVERAGE_THRESHOLD:
        downgrade.append("coverage below threshold")
        missing.append("coverage estimate at or above threshold")
    if record.memory_budget_gb is None:
        missing.append("memory budget")
    uncertainty: float | None = None
    if coverage is not None:
        uncertainty = 1.0 - coverage
    memory_overhead_gb = record.memory_budget_gb
    parameter_overhead: int | None = None
    if record.memory_budget_gb is None:
        missing.append("parameter overhead")
    affected_families = _families_for(record.affected_parameters)
    strategy = _strategy_for(record)
    if observed_delta is None:
        verdict: Verdict = "INCONCLUSIVE"
        verdict_reason = "no shared numeric measurement between baseline and post"
        next_experiment = _next_experiment_inconclusive(
            keys, coverage, metric_direction
        )
    elif observed_delta == 0.0:
        verdict = "INCONCLUSIVE"
        verdict_reason = "zero delta: no change detected"
        next_experiment = _next_experiment_inconclusive(
            keys, coverage, metric_direction
        )
    elif not held_out_supplied:
        verdict = "INCONCLUSIVE"
        verdict_reason = "no held-out keys supplied: delta is in-sample"
        next_experiment = _next_experiment_inconclusive(
            keys, coverage, metric_direction
        )
    elif _improved(observed_delta, metric_direction):
        if coverage is None:
            verdict = "INCONCLUSIVE"
            verdict_reason = "coverage unknown"
            next_experiment = _next_experiment_inconclusive(
                keys, coverage, metric_direction
            )
        elif coverage >= COVERAGE_THRESHOLD:
            verdict = "SUPPORTED"
            verdict_reason = (
                "improvement on held-out measurement (raw delta "
                + str(observed_delta)
                + ", direction="
                + metric_direction
                + ")"
            )
            next_experiment = _next_experiment_supported(keys, metric_direction)
        else:
            verdict = "INCONCLUSIVE"
            verdict_reason = "coverage below threshold"
            next_experiment = _next_experiment_inconclusive(
                keys, coverage, metric_direction
            )
    else:
        verdict = "REFUTED"
        verdict_reason = (
            "regression on held-out measurement (raw delta "
            + str(observed_delta)
            + ", direction="
            + metric_direction
            + ")"
        )
        next_experiment = _next_experiment_refuted(keys, metric_direction)
    return StrategyEvidence(
        strategy=strategy,
        metric_direction=metric_direction,
        intervention_id=record.intervention_id,
        observed_delta=observed_delta,
        held_out_keys=tuple(keys),
        affected_families=affected_families,
        memory_overhead_gb=memory_overhead_gb,
        parameter_overhead=parameter_overhead,
        uncertainty=uncertainty,
        coverage=coverage,
        verdict=verdict,
        verdict_reason=verdict_reason,
        missing_evidence=tuple(missing),
        next_experiment=next_experiment,
        downgrade_reasons=tuple(downgrade),
    )


def next_experiment_for(evidence: StrategyEvidence) -> str:
    if evidence.verdict == "SUPPORTED":
        return _next_experiment_supported(
            evidence.held_out_keys, evidence.metric_direction
        )
    if evidence.verdict == "REFUTED":
        return _next_experiment_refuted(
            evidence.held_out_keys, evidence.metric_direction
        )
    return _next_experiment_inconclusive(
        evidence.held_out_keys, evidence.coverage, evidence.metric_direction
    )


def decide(
    records: Sequence[InterventionRecord],
    comparison: ComparisonReport | None = None,
    *,
    held_out_keys: Sequence[str] | None = None,
    metric_direction: MetricDirection = "lower_is_better",
) -> RecoveryDecision:
    evidence_list: list[StrategyEvidence] = []
    coverage = comparison.coverage if comparison is not None else None
    warnings: list[str] = list(comparison.warnings) if comparison is not None else []
    for record in records:
        evidence_list.append(
            evaluate_strategy(
                record,
                held_out_keys=held_out_keys,
                coverage=coverage,
                metric_direction=metric_direction,
            )
        )
    non_inconclusive = sum(
        1 for evidence in evidence_list if evidence.verdict != "INCONCLUSIVE"
    )
    distinguishable = non_inconclusive >= 2
    if distinguishable:
        summary = "Two or more strategies reached a supported or refuted verdict; the evidence distinguishes them."
    else:
        summary = "Fewer than two strategies reached a supported or refuted verdict, so the result is inconclusive."
    return RecoveryDecision(
        strategies=tuple(evidence_list),
        distinguishable=distinguishable,
        summary=summary,
        warnings=tuple(warnings),
    )


def persist_decision(
    decision: RecoveryDecision,
    store: EvidenceStore,
    *,
    parent_id: str | None = None,
) -> EvidenceArtifact:
    return store.store(
        artifact_type="paired_recovery_decision",
        content=asdict(decision),
        parameters=None,
        parent_id=parent_id,
    )
