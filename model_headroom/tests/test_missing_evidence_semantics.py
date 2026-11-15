"""Regression tests for missing-evidence semantics in composite scoring.

The mission constraint under test: an absent measurement must never be
reported as a measured headroom value. A signal that produced no evidence
contributes the neutral prior; a signal whose evidence was attempted but is
unusable contributes nothing; a signal the architecture cannot produce is
excluded from the composite entirely. A report that measured nothing must
communicate INSUFFICIENT EVIDENCE rather than an apparent headroom of zero.
"""

from __future__ import annotations

import json

import pytest

from model_headroom.analysis.scoring import (
    DEFAULT_SIGNAL_WEIGHTS,
    EVIDENCE_STATE_FULL,
    EVIDENCE_STATE_INSUFFICIENT,
    EVIDENCE_STATE_NO_APPLICABLE,
    EVIDENCE_STATE_PARTIAL,
    NEUTRAL_PRIOR,
    compute_coverage_metrics,
    compute_headroom_score,
    evidence_state,
    generate_recommendations,
)
from model_headroom.models.types import (
    DeadNeuronReport,
    EffectiveRankResult,
    HeadroomReport,
    SignalResult,
)
from model_headroom.reporting.human_report import generate_human_report
from model_headroom.reporting.json_report import generate_json_report


def _report(signals: list[SignalResult]) -> HeadroomReport:
    return HeadroomReport(
        model_name="m",
        model_path="/tmp",
        timestamp="t",
        signals=signals,
        overall_headroom_score=0.0,
        recommendations=[],
        metadata={},
    )


def _single(
    name: str, status: str, data: object = None, source_type: str = "unavailable"
) -> SignalResult:
    return SignalResult(
        signal_name=name,
        status=status,
        data=data,
        duration_seconds=0.0,
        source_type=source_type,
    )


def _measured_dead_neurons() -> SignalResult:
    return _single(
        "dead_neurons",
        "computed",
        DeadNeuronReport(
            total_neurons=10,
            dead_count=0,
            dead_fraction=0.0,
            dead_indices=None,
            signal_status="computed",
            error_message=None,
            activation_rms=1.0,
            mean_abs_activation=1.0,
            variance=1.0,
            near_zero_occupancy=0.0,
            cross_token_variance=1.0,
            source_type="measured",
            negative_mean_count=0,
            negative_mean_fraction=0.0,
            never_active_count=0,
            never_active_fraction=0.0,
        ),
        source_type="measured",
    )


class TestMissingEvidenceContributesNeutralPrior:
    @pytest.mark.parametrize(
        "status",
        ["skipped", "insufficient_data", "no_gradient", "expert_count_unknown"],
    )
    def test_no_evidence_status_scores_neutral(self, status: str) -> None:
        report = _report([_single("loss_saturation", status)])
        assert compute_headroom_score(report) == pytest.approx(NEUTRAL_PRIOR)

    @pytest.mark.parametrize(
        "status",
        ["error", "degenerate", "poor_fit", "invalid_evidence", "zero_weight"],
    )
    def test_unusable_evidence_status_scores_zero(self, status: str) -> None:
        report = _report([_single("loss_saturation", status)])
        assert compute_headroom_score(report) == pytest.approx(0.0)

    @pytest.mark.parametrize("status", ["not_applicable", "single_expert"])
    def test_excluded_status_contributes_no_weight(self, status: str) -> None:
        measured = _measured_dead_neurons()
        report = _report([measured, _single("load_balance", status)])
        expected = 1.0
        assert compute_headroom_score(report) == pytest.approx(expected)

    def test_missing_evidence_never_yields_zero_headroom(self) -> None:
        report = _report(
            [
                _single("loss_saturation", "skipped"),
                _single("growth_factor", "skipped"),
                _single("effective_rank", "skipped"),
                _single("grad_weight_ratio", "skipped"),
                _single("dead_neurons", "skipped"),
            ]
        )
        score = compute_headroom_score(report)
        assert score != 0.0
        assert score != 1.0
        assert score == pytest.approx(NEUTRAL_PRIOR)

    def test_missing_evidence_never_yields_full_headroom(self) -> None:
        report = _report([_single("grad_weight_ratio", "no_gradient")])
        assert compute_headroom_score(report) < 1.0

    def test_computed_measured_value_is_preserved(self) -> None:
        data = EffectiveRankResult(
            rank=100.0,
            entropy=1.0,
            percentage_of_full_rank=0.25,
            singular_values_count=100,
            signal_status="computed",
            error_message=None,
            sampling_mode="full",
            coverage=1.0,
            parent_shape=(10, 10),
            sample_count=1,
            confidence=0.9,
            state="full",
        )
        report = _report([_single("effective_rank", "computed", data)])
        assert compute_headroom_score(report) == pytest.approx(0.25)


class TestEvidenceState:
    def test_all_skipped_is_insufficient_evidence(self) -> None:
        report = _report([_single("loss_saturation", "skipped")])
        assert evidence_state(report) == EVIDENCE_STATE_INSUFFICIENT

    def test_no_applicable_signals(self) -> None:
        report = _report([_single("load_balance", "not_applicable")])
        assert evidence_state(report) == EVIDENCE_STATE_NO_APPLICABLE

    def test_partial_coverage(self) -> None:
        report = _report(
            [
                _single("loss_saturation", "skipped"),
                _single("growth_factor", "skipped"),
            ]
        )
        assert evidence_state(report) == EVIDENCE_STATE_INSUFFICIENT

    def test_partial_when_some_measured(self) -> None:
        report = _report(
            [_measured_dead_neurons(), _single("loss_saturation", "skipped")]
        )
        assert evidence_state(report) == EVIDENCE_STATE_PARTIAL

    def test_full_coverage(self) -> None:
        report = _report([_measured_dead_neurons()])
        assert evidence_state(report) == EVIDENCE_STATE_FULL


class TestReportCommunicatesInsufficientEvidence:
    def test_human_report_shows_no_headroom_value(self) -> None:
        report = _report(
            [
                _single("loss_saturation", "skipped"),
                _single("growth_factor", "skipped"),
            ]
        )
        text = generate_human_report(report)
        assert "INSUFFICIENT EVIDENCE" in text
        assert "HEADROOM SCORE" not in text
        assert "no headroom value can be asserted" in text.lower()

    def test_human_report_labels_partial_composite_as_prior_mixed(self) -> None:
        report = _report(
            [_measured_dead_neurons(), _single("loss_saturation", "skipped")]
        )
        text = generate_human_report(report)
        assert "prior-mixed" in text

    def test_json_report_carries_evidence_state(self) -> None:
        report = _report([_single("loss_saturation", "skipped")])
        parsed = json.loads(generate_json_report(report))
        assert parsed["evidence_state"] == EVIDENCE_STATE_INSUFFICIENT
        assert parsed["composite_kind"] == "prior_mixed_expectation"

    def test_recommendation_states_insufficient_evidence(self) -> None:
        report = _report([_single("loss_saturation", "skipped")])
        recs = generate_recommendations(report)
        assert any("INSUFFICIENT EVIDENCE" in rec for rec in recs)

    def test_coverage_metrics_unchanged_by_prior_mixing(self) -> None:
        report = _report(
            [
                _single("loss_saturation", "skipped"),
                _single("growth_factor", "skipped"),
            ]
        )
        metrics = compute_coverage_metrics(report)
        assert metrics.coverage == pytest.approx(0.0)
        assert metrics.measured_weight == pytest.approx(0.0)
        assert metrics.unmeasured_weight == pytest.approx(
            sum(
                DEFAULT_SIGNAL_WEIGHTS[name]
                for name in ("loss_saturation", "growth_factor")
            )
        )
        assert metrics.lower_bound == pytest.approx(0.0)
        assert metrics.upper_bound == pytest.approx(1.0)
        assert metrics.decision_grade is False
