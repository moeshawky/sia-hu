from __future__ import annotations

import dataclasses
import inspect
import json
from typing import TYPE_CHECKING

import pytest

from model_headroom.analysis.evidence_store import EvidenceStore
from model_headroom.analysis.intervention_record import InterventionRecord
from model_headroom.paired import decision as decision_module
from model_headroom.paired.compare import ComparisonReport
from model_headroom.paired.decision import (
    COVERAGE_THRESHOLD,
    RecoveryDecision,
    StrategyEvidence,
    decide,
    evaluate_strategy,
    next_experiment_for,
    persist_decision,
)

if TYPE_CHECKING:
    from pathlib import Path


def _record(
    baseline: dict[str, float],
    post: dict[str, float],
    *,
    intervention_type: str = "continuous_parameters_frozen_codes",
) -> InterventionRecord:
    return InterventionRecord(
        intervention_id="irec-1",
        intervention_type=intervention_type,
        checkpoint_hash="c" * 16,
        architecture="bert-base",
        runtime_versions={"torch": "2.0"},
        affected_parameters=["encoder.layer0.q_proj.weight"],
        memory_budget_gb=0.5,
        baseline_measurements=baseline,
        post_measurements=post,
    )


def _comparison(
    coverage: float | None, *, warnings: tuple[str, ...] = ()
) -> ComparisonReport:
    return ComparisonReport(
        reference_model_id="ref",
        candidate_model_id="cand",
        input_fingerprint_match=True,
        layers=(),
        modules=(),
        gradients=None,
        outliers=(),
        coverage=coverage,
        measured_layer_count=0,
        comparable_layer_count=0,
        warnings=warnings,
    )


def _decision(
    *,
    baseline: dict[str, float] | None = None,
    post: dict[str, float] | None = None,
) -> RecoveryDecision:
    record = _record(
        {"loss": 1.0} if baseline is None else baseline,
        {"loss": 0.7} if post is None else post,
    )
    return decide([record], held_out_keys=["loss"])


class TestSignOfTheRawDelta:
    @pytest.mark.parametrize(
        ("baseline", "post", "metric_direction", "expected"),
        [
            (1.0, 0.7, "lower_is_better", "SUPPORTED"),
            (1.0, 1.3, "lower_is_better", "REFUTED"),
            (1.0, 1.0, "lower_is_better", "INCONCLUSIVE"),
            (1.0, 1.3, "higher_is_better", "SUPPORTED"),
            (1.0, 0.7, "higher_is_better", "REFUTED"),
            (1.0, 1.0, "higher_is_better", "INCONCLUSIVE"),
        ],
    )
    def test_verdict_follows_the_sign_of_the_raw_delta_under_each_polarity(
        self,
        baseline: float,
        post: float,
        metric_direction: str,
        expected: str,
    ) -> None:
        record = _record({"loss": baseline}, {"loss": post})
        evidence = evaluate_strategy(
            record,
            held_out_keys=["loss"],
            coverage=0.9,
            metric_direction=metric_direction,  # type: ignore[arg-type]
        )
        assert evidence.verdict == expected
        assert evidence.observed_delta == pytest.approx(post - baseline)
        assert evidence.metric_direction == metric_direction

    def test_observed_delta_is_post_minus_base_not_the_reverse(self) -> None:
        record = _record({"loss": 2.0}, {"loss": 1.0})
        evidence = evaluate_strategy(record, held_out_keys=["loss"], coverage=0.9)
        assert evidence.observed_delta == pytest.approx(-1.0)
        assert evidence.observed_delta != pytest.approx(1.0)
        assert evidence.verdict == "SUPPORTED"
        assert "-1.0" in evidence.verdict_reason

    def test_zero_delta_is_inconclusive_never_supported(self) -> None:
        record = _record({"loss": 1.0}, {"loss": 1.0})
        for metric_direction in ("lower_is_better", "higher_is_better"):
            evidence = evaluate_strategy(
                record,
                held_out_keys=["loss"],
                coverage=0.9,
                metric_direction=metric_direction,  # type: ignore[arg-type]
            )
            assert evidence.observed_delta == 0.0
            assert evidence.verdict == "INCONCLUSIVE"
            assert evidence.verdict != "SUPPORTED"
            assert evidence.verdict != "REFUTED"


class TestCoverageGate:
    def test_coverage_exactly_at_threshold_is_supported(self) -> None:
        assert COVERAGE_THRESHOLD == 0.8
        record = _record({"loss": 1.0}, {"loss": 0.7})
        evidence = evaluate_strategy(
            record, held_out_keys=["loss"], coverage=COVERAGE_THRESHOLD
        )
        assert evidence.verdict == "SUPPORTED"

    def test_coverage_below_threshold_downgrades_an_improvement(self) -> None:
        record = _record({"loss": 1.0}, {"loss": 0.7})
        evidence = evaluate_strategy(record, held_out_keys=["loss"], coverage=0.79)
        assert evidence.observed_delta == pytest.approx(-0.3)
        assert evidence.verdict == "INCONCLUSIVE"
        assert "coverage below threshold" in evidence.verdict_reason
        assert evidence.downgrade_reasons == ("coverage below threshold",)

    def test_missing_coverage_downgrades_an_improvement(self) -> None:
        record = _record({"loss": 1.0}, {"loss": 0.7})
        evidence = evaluate_strategy(record, held_out_keys=["loss"], coverage=None)
        assert evidence.verdict == "INCONCLUSIVE"
        assert evidence.verdict_reason == "coverage unknown"
        assert evidence.downgrade_reasons == ("coverage unknown",)
        assert evidence.coverage is None
        assert evidence.uncertainty is None

    def test_verdict_reason_distinguishes_unknown_coverage_from_below_threshold(
        self,
    ) -> None:
        record = _record({"loss": 1.0}, {"loss": 0.7})
        unknown = evaluate_strategy(record, held_out_keys=["loss"], coverage=None)
        below = evaluate_strategy(record, held_out_keys=["loss"], coverage=0.5)
        assert unknown.verdict_reason == "coverage unknown"
        assert below.verdict_reason == "coverage below threshold"
        assert unknown.verdict_reason != below.verdict_reason

    def test_coverage_does_not_downgrade_a_regression(self) -> None:
        record = _record({"loss": 1.0}, {"loss": 1.3})
        evidence = evaluate_strategy(record, held_out_keys=["loss"], coverage=None)
        assert evidence.verdict == "REFUTED"

    def test_parameter_overhead_guard_keys_on_parameter_overhead_not_memory_budget(
        self,
    ) -> None:
        # C-CAUSE-6: parameter_overhead is never populated (not derivable from the
        # record), so missing_evidence must flag it keyed on parameter_overhead
        # itself — not on memory_budget_gb. memory_budget_gb is set here, so the
        # OLD wrong guard (keying on memory_budget_gb) would silently omit it.
        record = _record({"loss": 1.0}, {"loss": 0.7})
        assert record.memory_budget_gb is not None
        evidence = evaluate_strategy(record, held_out_keys=["loss"], coverage=0.9)
        assert evidence.parameter_overhead is None
        assert "parameter overhead" in evidence.missing_evidence


class TestHeldOutKeys:
    def test_requested_held_out_keys_that_were_never_measured_is_inconclusive(
        self,
    ) -> None:
        record = _record({"loss": 1.0}, {"loss": 0.7})
        evidence = evaluate_strategy(record, held_out_keys=["absent_key"], coverage=0.9)
        assert evidence.observed_delta is None
        assert evidence.verdict == "INCONCLUSIVE"
        assert evidence.held_out_keys == ()
        assert "missing held-out measurement: absent_key" in evidence.missing_evidence

    def test_unmeasured_keys_are_dropped_and_the_measured_key_decides(self) -> None:
        record = _record({"loss": 1.0, "kl": 0.5}, {"loss": 0.7, "kl": 0.4})
        evidence = evaluate_strategy(
            record, held_out_keys=["loss", "absent_key"], coverage=0.9
        )
        assert evidence.held_out_keys == ("loss",)
        assert evidence.observed_delta == pytest.approx(-0.3)
        assert evidence.verdict == "SUPPORTED"
        assert "held-out keys not measured: absent_key" in evidence.downgrade_reasons

    def test_no_shared_numeric_keys_is_inconclusive(self) -> None:
        record = _record({"loss": 1.0}, {"perplexity": 0.7})
        evidence = evaluate_strategy(record, held_out_keys=["loss"], coverage=0.9)
        assert evidence.observed_delta is None
        assert evidence.verdict == "INCONCLUSIVE"
        assert evidence.downgrade_reasons == (
            "no shared numeric keys between baseline and post",
            "held-out keys not measured: loss",
        )

    def test_omitted_held_out_keys_leaves_the_delta_inconclusive(self) -> None:
        record = _record({"loss": 1.0, "kl": 0.5}, {"loss": 0.7, "kl": 0.4})
        evidence = evaluate_strategy(record, held_out_keys=None, coverage=0.9)
        assert evidence.held_out_keys == ("loss", "kl")
        assert evidence.observed_delta == pytest.approx(-0.2)
        assert evidence.verdict == "INCONCLUSIVE"
        assert (
            evidence.verdict_reason == "no held-out keys supplied: delta is in-sample"
        )
        assert evidence.downgrade_reasons == ("no held-out keys supplied",)
        assert "held-out measurement set" in evidence.missing_evidence

    @pytest.mark.parametrize("held_out_keys", [None, []])
    def test_no_held_out_keys_supplied_is_inconclusive_even_when_improving(
        self, held_out_keys: list[str] | None
    ) -> None:
        record = _record({"loss": 1.0}, {"loss": 0.7})
        evidence = evaluate_strategy(record, held_out_keys=held_out_keys, coverage=0.9)
        assert evidence.observed_delta == pytest.approx(-0.3)
        assert evidence.verdict == "INCONCLUSIVE"
        assert evidence.downgrade_reasons == ("no held-out keys supplied",)

    def test_in_sample_regression_without_held_out_keys_is_inconclusive(self) -> None:
        record = _record({"loss": 1.0}, {"loss": 1.3})
        evidence = evaluate_strategy(record, held_out_keys=None, coverage=0.9)
        assert evidence.observed_delta == pytest.approx(0.3)
        assert evidence.verdict == "INCONCLUSIVE"
        assert evidence.downgrade_reasons == ("no held-out keys supplied",)


class TestDistinguishability:
    def test_three_inconclusive_strategies_are_not_distinguishable(self) -> None:
        records = [
            _record({"loss": 1.0}, {"loss": 0.7}),
            _record({"loss": 2.0}, {"loss": 1.9}),
            _record({"loss": 3.0}, {"loss": 3.0}),
        ]
        decision = decide(records, held_out_keys=["absent_key"])
        assert len(decision.strategies) == 3
        assert all(
            evidence.verdict == "INCONCLUSIVE" for evidence in decision.strategies
        )
        assert decision.distinguishable is False
        assert "inconclusive" in decision.summary.lower()

    def test_two_decided_strategies_are_distinguishable(self) -> None:
        records = [
            _record({"loss": 1.0}, {"loss": 0.7}),
            _record({"kl": 0.5}, {"kl": 0.4}),
        ]
        decision = decide(records, _comparison(0.9), held_out_keys=["loss", "kl"])
        assert [evidence.verdict for evidence in decision.strategies] == [
            "SUPPORTED",
            "SUPPORTED",
        ]
        assert decision.distinguishable is True
        assert "inconclusive" not in decision.summary.lower()

    def test_one_decided_strategy_is_not_distinguishable(self) -> None:
        records = [
            _record({"loss": 1.0}, {"loss": 0.7}),
            _record({"kl": 0.5}, {"kl": 0.5}),
        ]
        decision = decide(records, _comparison(0.9), held_out_keys=["loss", "kl"])
        assert [evidence.verdict for evidence in decision.strategies] == [
            "SUPPORTED",
            "INCONCLUSIVE",
        ]
        assert decision.distinguishable is False

    def test_comparison_coverage_reaches_every_strategy(self) -> None:
        records = [_record({"loss": 1.0}, {"loss": 0.7})]
        decision = decide(records, _comparison(0.5), held_out_keys=["loss"])
        assert decision.strategies[0].coverage == 0.5
        assert decision.strategies[0].verdict == "INCONCLUSIVE"

    def test_comparison_warnings_are_carried_into_the_decision(self) -> None:
        records = [_record({"loss": 1.0}, {"loss": 0.7})]
        decision = decide(
            records,
            _comparison(0.9, warnings=("input fingerprint unknown on reference",)),
            held_out_keys=["loss"],
        )
        assert decision.warnings == ("input fingerprint unknown on reference",)

    def test_next_experiment_names_the_recorded_metric_direction(self) -> None:
        record = _record({"loss": 1.0}, {"loss": 1.3})
        evidence = evaluate_strategy(
            record,
            held_out_keys=["loss"],
            coverage=0.9,
            metric_direction="higher_is_better",
        )
        assert evidence.verdict == "SUPPORTED"
        assert "metric_direction=higher_is_better" in next_experiment_for(evidence)


class TestNoDerivedScoreField:
    def test_strategy_evidence_has_no_probability_or_composite_score_field(
        self,
    ) -> None:
        for dataclass_type in (StrategyEvidence, RecoveryDecision):
            for field in dataclasses.fields(dataclass_type):
                lowered = field.name.lower()
                assert "probability" not in lowered, field.name
                assert "score" not in lowered, field.name
                assert "composite" not in lowered, field.name

    def test_decision_module_defines_no_composite_headroom_score(self) -> None:
        source = inspect.getsource(decision_module)
        assert "compute_headroom_score" not in source
        assert hasattr(decision_module, "compute_headroom_score") is False


class TestEvidenceProvenance:
    def test_persist_then_verify_round_trips(self, tmp_path: Path) -> None:
        store = EvidenceStore(tmp_path / "evidence")
        decision = _decision()
        artifact = persist_decision(decision, store)
        assert artifact.artifact_type == "paired_recovery_decision"
        verified = store.verify(artifact.artifact_id)
        assert verified["verified"] is True
        stored = store.retrieve(artifact.artifact_id)
        assert stored is not None
        assert stored["distinguishable"] == decision.distinguishable
        assert stored["summary"] == decision.summary
        assert len(stored["strategies"]) == len(decision.strategies)

    def test_tampering_with_the_stored_artifact_breaks_verification(
        self, tmp_path: Path
    ) -> None:
        store = EvidenceStore(tmp_path / "evidence")
        artifact = persist_decision(_decision(), store)
        assert store.verify(artifact.artifact_id)["verified"] is True
        path = tmp_path / "evidence" / f"{artifact.artifact_id}.json"
        payload = json.loads(path.read_text())
        payload["distinguishable"] = not payload["distinguishable"]
        path.write_text(json.dumps(payload, indent=2))
        result = store.verify(artifact.artifact_id)
        assert result["verified"] is False
        assert result["reason"] == "hash mismatch"

    def test_unknown_artifact_id_does_not_verify(self, tmp_path: Path) -> None:
        store = EvidenceStore(tmp_path / "evidence")
        result = store.verify("paired_recovery_decision_deadbeef")
        assert result["verified"] is False
        assert result["reason"] == "artifact not found"


class TestDeterminism:
    def test_evaluate_strategy_is_independent_of_measurement_key_order(self) -> None:
        forward = _record(
            {"a": 0.0, "b": 0.0, "c": 0.0, "d": 0.0},
            {"a": 1e16, "b": 1.0, "c": -1e16, "d": 1.0},
        )
        reversed_order = _record(
            {"d": 0.0, "c": 0.0, "b": 0.0, "a": 0.0},
            {"d": 1.0, "c": -1e16, "b": 1.0, "a": 1e16},
        )
        first = evaluate_strategy(forward)
        second = evaluate_strategy(reversed_order)
        assert first.observed_delta == pytest.approx(0.5)
        assert second.observed_delta == first.observed_delta
        assert second.verdict == first.verdict

    def test_decide_is_a_pure_function_of_its_records(self) -> None:
        records = [
            _record({"loss": 1.0}, {"loss": 0.7}),
            _record({"kl": 0.5}, {"kl": 0.5}),
        ]
        first = decide(records, _comparison(0.9), held_out_keys=["loss", "kl"])
        second = decide(records, _comparison(0.9), held_out_keys=["loss", "kl"])
        assert first == second
