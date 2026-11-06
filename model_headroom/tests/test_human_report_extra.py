from __future__ import annotations

from model_headroom.models.types import (
    EffectiveRankResult,
    GradWeightRatio,
    HeadroomReport,
    SaturationReport,
    SignalResult,
)
from model_headroom.reporting.human_report import (
    _format_signal_status,
    generate_human_report,
)


class TestFormatSignalStatusEdge:
    def test_error_with_data_message(self) -> None:
        signal = SignalResult(
            signal_name="loss_saturation",
            status="error",
            data=SaturationReport(
                is_saturated=False,
                saturation_point_tokens=None,
                loss_delta_threshold=0.001,
                plateau_slope=0.0,
                data_points_used=0,
                signal_status="error",
                error_message="something broke",
            ),
            duration_seconds=0.0,
        )
        output = _format_signal_status(signal)
        assert "ERROR" in output

    def test_error_no_data(self) -> None:
        signal = SignalResult(
            signal_name="loss_saturation",
            status="error",
            data=None,
            duration_seconds=0.0,
        )
        output = _format_signal_status(signal)
        assert "ERROR" in output

    def test_no_data_signal(self) -> None:
        signal = SignalResult(
            signal_name="loss_saturation",
            status="computed",
            data=None,
            duration_seconds=0.0,
        )
        output = _format_signal_status(signal)
        assert "NO DATA" in output

    def test_effective_rank_formatting(self) -> None:
        signal = SignalResult(
            signal_name="effective_rank",
            status="computed",
            data=EffectiveRankResult(
                rank=6.0,
                entropy=1.5,
                percentage_of_full_rank=0.75,
                singular_values_count=8,
                signal_status="computed",
            ),
            duration_seconds=0.0,
        )
        output = _format_signal_status(signal)
        assert "rank=6.0" in output

    def test_grad_weight_ratio_formatting(self) -> None:
        signal = SignalResult(
            signal_name="grad_weight_ratio",
            status="computed",
            data=[
                GradWeightRatio(
                    ratio=0.5,
                    is_high_pressure=False,
                    z_score=0.0,
                    layer_name="l0",
                    signal_status="computed",
                ),
                GradWeightRatio(
                    ratio=5.0,
                    is_high_pressure=True,
                    z_score=2.5,
                    layer_name="l1",
                    signal_status="computed",
                ),
            ],
            duration_seconds=0.0,
        )
        output = _format_signal_status(signal)
        assert "high pressure" in output

    def test_grad_weight_ratio_empty_data(self) -> None:
        signal = SignalResult(
            signal_name="grad_weight_ratio",
            status="computed",
            data=[],
            duration_seconds=0.0,
        )
        output = _format_signal_status(signal)
        assert "NO DATA" in output


class TestHumanReportRecommendationWrapping:
    def test_long_recommendation_wraps(self) -> None:
        long_rec = "word " * 30
        report = HeadroomReport(
            model_name="m",
            model_path="/tmp",
            timestamp="2026-01-01T00:00:00+00:00",
            signals=[],
            overall_headroom_score=0.5,
            recommendations=[long_rec],
            metadata={},
        )
        output = generate_human_report(report)
        for line in output.split("\n"):
            assert len(line) <= 80


class TestHumanReportTruncation:
    def test_report_max_200_lines(self) -> None:
        recs = ["rec" for _ in range(200)]
        report = HeadroomReport(
            model_name="m",
            model_path="/tmp",
            timestamp="2026-01-01T00:00:00+00:00",
            signals=[],
            overall_headroom_score=0.5,
            recommendations=recs,
            metadata={},
        )
        output = generate_human_report(report)
        assert len(output.split("\n")) <= 200
