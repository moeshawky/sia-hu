from __future__ import annotations

from model_headroom.analysis.scoring import (
    _has_sampling_confidence,
    _is_measured,
    _score_signal,
    compute_coverage_metrics,
    compute_headroom_score,
)
from model_headroom.models.types import (
    CoverageMetrics,
    EffectiveRankResult,
    HeadroomReport,
    SaturationReport,
    SignalResult,
)


class TestScoreSignalNotApplicable:
    def test_not_applicable_returns_none(self) -> None:
        signal = SignalResult(
            signal_name="load_balance",
            status="not_applicable",
            data=None,
            duration_seconds=0.0,
        )
        assert _score_signal(signal) is None


class TestHasSamplingConfidence:
    def test_data_with_confidence_field(self) -> None:
        data = EffectiveRankResult(
            rank=4.0,
            entropy=1.0,
            percentage_of_full_rank=0.5,
            singular_values_count=8,
            signal_status="computed",
            confidence=0.85,
        )
        signal = SignalResult(
            signal_name="effective_rank",
            status="computed",
            data=data,
            duration_seconds=0.0,
        )
        assert _has_sampling_confidence(signal) is True

    def test_data_with_coverage_field(self) -> None:
        data = EffectiveRankResult(
            rank=4.0,
            entropy=1.0,
            percentage_of_full_rank=0.5,
            singular_values_count=8,
            signal_status="computed",
            coverage=0.75,
        )
        signal = SignalResult(
            signal_name="effective_rank",
            status="computed",
            data=data,
            duration_seconds=0.0,
        )
        assert _has_sampling_confidence(signal) is True

    def test_data_without_confidence_or_coverage(self) -> None:
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
        assert _has_sampling_confidence(signal) is False

    def test_none_data_returns_false(self) -> None:
        signal = SignalResult(
            signal_name="loss_saturation",
            status="skipped",
            data=None,
            duration_seconds=0.0,
        )
        assert _has_sampling_confidence(signal) is False


class TestIsMeasured:
    def test_computed_measured_verified(self) -> None:
        signal = SignalResult(
            signal_name="loss_saturation",
            status="computed",
            data=SaturationReport(
                is_saturated=False,
                saturation_point_tokens=None,
                loss_delta_threshold=0.001,
                plateau_slope=0.0,
                data_points_used=5,
                signal_status="computed",
            ),
            duration_seconds=0.0,
            source_type="measured",
            verification_status="verified",
        )
        assert _is_measured(signal) is True

    def test_skipped_is_not_measured(self) -> None:
        signal = SignalResult(
            signal_name="loss_saturation",
            status="skipped",
            data=None,
            duration_seconds=0.0,
        )
        assert _is_measured(signal) is False

    def test_error_is_not_measured(self) -> None:
        signal = SignalResult(
            signal_name="loss_saturation",
            status="error",
            data=None,
            duration_seconds=0.0,
        )
        assert _is_measured(signal) is False

    def test_synthetic_is_not_measured(self) -> None:
        signal = SignalResult(
            signal_name="loss_saturation",
            status="computed",
            data=SaturationReport(
                is_saturated=False,
                saturation_point_tokens=None,
                loss_delta_threshold=0.001,
                plateau_slope=0.0,
                data_points_used=5,
                signal_status="computed",
            ),
            duration_seconds=0.0,
            source_type="synthetic",
            verification_status="verified",
        )
        assert _is_measured(signal) is False

    def test_fixture_is_not_measured(self) -> None:
        signal = SignalResult(
            signal_name="loss_saturation",
            status="computed",
            data=SaturationReport(
                is_saturated=False,
                saturation_point_tokens=None,
                loss_delta_threshold=0.001,
                plateau_slope=0.0,
                data_points_used=5,
                signal_status="computed",
            ),
            duration_seconds=0.0,
            source_type="fixture",
            verification_status="verified",
        )
        assert _is_measured(signal) is False

    def test_unknown_is_not_measured(self) -> None:
        signal = SignalResult(
            signal_name="loss_saturation",
            status="computed",
            data=SaturationReport(
                is_saturated=False,
                saturation_point_tokens=None,
                loss_delta_threshold=0.001,
                plateau_slope=0.0,
                data_points_used=5,
                signal_status="computed",
            ),
            duration_seconds=0.0,
            source_type="unknown",
            verification_status="verified",
        )
        assert _is_measured(signal) is False

    def test_estimated_without_confidence_is_not_measured(self) -> None:
        signal = SignalResult(
            signal_name="loss_saturation",
            status="computed",
            data=SaturationReport(
                is_saturated=False,
                saturation_point_tokens=None,
                loss_delta_threshold=0.001,
                plateau_slope=0.0,
                data_points_used=5,
                signal_status="computed",
            ),
            duration_seconds=0.0,
            source_type="estimated",
            verification_status="verified",
        )
        assert _is_measured(signal) is False

    def test_estimated_with_confidence_is_not_measured(self) -> None:
        data = EffectiveRankResult(
            rank=4.0,
            entropy=1.0,
            percentage_of_full_rank=0.5,
            singular_values_count=8,
            signal_status="computed",
            confidence=0.8,
        )
        signal = SignalResult(
            signal_name="effective_rank",
            status="computed",
            data=data,
            duration_seconds=0.0,
            source_type="estimated",
            verification_status="verified",
        )
        assert _is_measured(signal) is False

    def test_unverified_is_measured(self) -> None:
        signal = SignalResult(
            signal_name="loss_saturation",
            status="computed",
            data=SaturationReport(
                is_saturated=False,
                saturation_point_tokens=None,
                loss_delta_threshold=0.001,
                plateau_slope=0.0,
                data_points_used=5,
                signal_status="computed",
            ),
            duration_seconds=0.0,
            source_type="measured",
            verification_status="unverified",
        )
        assert _is_measured(signal) is True

    def test_provenance_incomplete_is_measured(self) -> None:
        signal = SignalResult(
            signal_name="loss_saturation",
            status="computed",
            data=SaturationReport(
                is_saturated=False,
                saturation_point_tokens=None,
                loss_delta_threshold=0.001,
                plateau_slope=0.0,
                data_points_used=5,
                signal_status="computed",
            ),
            duration_seconds=0.0,
            source_type="measured",
            verification_status="provenance_incomplete",
        )
        assert _is_measured(signal) is True

    def test_imported_is_not_measured(self) -> None:
        signal = SignalResult(
            signal_name="loss_saturation",
            status="computed",
            data=SaturationReport(
                is_saturated=False,
                saturation_point_tokens=None,
                loss_delta_threshold=0.001,
                plateau_slope=0.0,
                data_points_used=5,
                signal_status="computed",
            ),
            duration_seconds=0.0,
            source_type="imported",
            verification_status="verified",
        )
        assert _is_measured(signal) is False


class TestComputeCoverageMetricsEmpty:
    def test_empty_signals_returns_zero_coverage(self) -> None:
        report = HeadroomReport(
            model_name="m",
            model_path="/tmp",
            timestamp="2026-01-01T00:00:00+00:00",
            signals=[],
            overall_headroom_score=0.0,
            recommendations=[],
            metadata={},
        )
        metrics = compute_coverage_metrics(report)
        assert isinstance(metrics, CoverageMetrics)
        assert metrics.coverage == 0.0
        assert metrics.measured_score == 0.0
        assert metrics.lower_bound == 0.0
        assert metrics.upper_bound == 0.0
        assert metrics.measured_weight == 0.0
        assert metrics.total_applicable_weight == 0.0
        assert metrics.unmeasured_weight == 0.0
        assert metrics.decision_grade is False


class TestNotApplicableExcludedFromScore:
    def test_not_applicable_signals_excluded(self) -> None:
        signals = [
            SignalResult(
                signal_name="loss_saturation",
                status="computed",
                data=SaturationReport(
                    is_saturated=False,
                    saturation_point_tokens=None,
                    loss_delta_threshold=0.001,
                    plateau_slope=0.0,
                    data_points_used=5,
                    signal_status="computed",
                ),
                duration_seconds=0.0,
            ),
            SignalResult(
                signal_name="load_balance",
                status="not_applicable",
                data=None,
                duration_seconds=0.0,
            ),
            SignalResult(
                signal_name="specialization",
                status="not_applicable",
                data=None,
                duration_seconds=0.0,
            ),
        ]
        for name in [
            "growth_factor",
            "effective_rank",
            "grad_weight_ratio",
            "dead_neurons",
        ]:
            signals.append(
                SignalResult(
                    signal_name=name,
                    status="skipped",
                    data=None,
                    duration_seconds=0.0,
                )
            )
        report = HeadroomReport(
            model_name="m",
            model_path="/tmp",
            timestamp="2026-01-01T00:00:00+00:00",
            signals=signals,
            overall_headroom_score=0.0,
            recommendations=[],
            metadata={"architecture": "dense"},
        )
        score = compute_headroom_score(report)
        metrics = compute_coverage_metrics(report)
        assert metrics.total_applicable_weight < 1.0
        assert 0.0 <= score <= 1.0
