from __future__ import annotations

from model_headroom.analysis.scoring import (
    _score_signal,
    compute_headroom_score,
    generate_recommendations,
)
from model_headroom.models.types import (
    DeadNeuronReport,
    EffectiveRankResult,
    HeadroomReport,
    LoadBalanceReport,
    SaturationReport,
    SignalResult,
    SpecializationReport,
)


class TestScoreSignal:
    def test_skipped_returns_05(self) -> None:
        signal = SignalResult(
            signal_name="loss_saturation",
            status="skipped",
            data=None,
            duration_seconds=0.0,
        )
        assert _score_signal(signal) == 0.5

    def test_error_returns_0(self) -> None:
        signal = SignalResult(
            signal_name="loss_saturation",
            status="error",
            data=None,
            duration_seconds=0.0,
            error_message="bad",
        )
        assert _score_signal(signal) == 0.0

    def test_loss_saturation_not_saturated(self) -> None:
        data = SaturationReport(
            is_saturated=False,
            saturation_point_tokens=None,
            loss_delta_threshold=0.001,
            plateau_slope=0.0,
            data_points_used=5,
            signal_status="computed",
        )
        signal = SignalResult(
            signal_name="loss_saturation",
            status="computed",
            data=data,
            duration_seconds=0.0,
        )
        assert _score_signal(signal) == 1.0

    def test_loss_saturation_saturated(self) -> None:
        data = SaturationReport(
            is_saturated=True,
            saturation_point_tokens=100,
            loss_delta_threshold=0.001,
            plateau_slope=0.0005,
            data_points_used=5,
            signal_status="computed",
        )
        signal = SignalResult(
            signal_name="loss_saturation",
            status="computed",
            data=data,
            duration_seconds=0.0,
        )
        assert _score_signal(signal) == 0.0

    def test_effective_rank(self) -> None:
        data = EffectiveRankResult(
            rank=4.0,
            entropy=1.0,
            percentage_of_full_rank=0.5,
            singular_values_count=8,
            signal_status="computed",
        )
        signal = SignalResult(
            signal_name="effective_rank",
            status="computed",
            data=data,
            duration_seconds=0.0,
        )
        assert _score_signal(signal) == 0.5

    def test_dead_neurons(self) -> None:
        data = DeadNeuronReport(
            total_neurons=100,
            dead_count=10,
            dead_fraction=0.1,
            dead_indices=None,
            signal_status="computed",
        )
        signal = SignalResult(
            signal_name="dead_neurons",
            status="computed",
            data=data,
            duration_seconds=0.0,
        )
        assert abs(_score_signal(signal) - 0.9) < 1e-6

    def test_load_balance(self) -> None:
        data = LoadBalanceReport(
            entropy_per_layer=[1.0],
            min_entropy=1.0,
            max_entropy=1.0,
            ideal_entropy=1.386,
            balance_ratio=0.72,
            signal_status="computed",
        )
        signal = SignalResult(
            signal_name="load_balance",
            status="computed",
            data=data,
            duration_seconds=0.0,
        )
        assert abs(_score_signal(signal) - 0.72) < 1e-6

    def test_specialization_not_collapsed(self) -> None:
        data = SpecializationReport(
            mean_inter_expert_similarity=0.3,
            max_inter_expert_similarity=0.5,
            collapse_threshold=0.85,
            is_collapsed=False,
            signal_status="computed",
        )
        signal = SignalResult(
            signal_name="specialization",
            status="computed",
            data=data,
            duration_seconds=0.0,
        )
        assert _score_signal(signal) == 1.0

    def test_specialization_collapsed(self) -> None:
        data = SpecializationReport(
            mean_inter_expert_similarity=0.9,
            max_inter_expert_similarity=0.95,
            collapse_threshold=0.85,
            is_collapsed=True,
            signal_status="computed",
        )
        signal = SignalResult(
            signal_name="specialization",
            status="computed",
            data=data,
            duration_seconds=0.0,
        )
        assert _score_signal(signal) == 0.0

    def test_grad_weight_ratio_empty(self) -> None:
        signal = SignalResult(
            signal_name="grad_weight_ratio",
            status="computed",
            data=[],
            duration_seconds=0.0,
        )
        assert _score_signal(signal) == 0.5


class TestComputeHeadroomScore:
    def test_all_skipped(self, headroom_report_minimal: HeadroomReport) -> None:
        score = compute_headroom_score(headroom_report_minimal)
        assert score == 0.5

    def test_score_in_range(self, headroom_report_computed: HeadroomReport) -> None:
        score = compute_headroom_score(headroom_report_computed)
        assert 0.0 <= score <= 1.0

    def test_empty_signals(self) -> None:
        report = HeadroomReport(
            model_name="m",
            model_path="/tmp",
            timestamp="2026-01-01T00:00:00+00:00",
            signals=[],
            overall_headroom_score=0.0,
            recommendations=[],
            metadata={},
        )
        assert compute_headroom_score(report) == 0.0


class TestGenerateRecommendations:
    def test_minimum_count(self, headroom_report_minimal: HeadroomReport) -> None:
        recs = generate_recommendations(headroom_report_minimal)
        assert len(recs) >= 3
        assert all(isinstance(r, str) for r in recs)
