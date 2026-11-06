from __future__ import annotations

from model_headroom.models.types import LossDataPoint
from model_headroom.signals.loss_saturation import analyze_loss_saturation


class TestLossSaturationFunctional:
    def test_saturated_curve(self, loss_curve_saturated: list[LossDataPoint]) -> None:
        result = analyze_loss_saturation(loss_curve_saturated, slope_threshold=0.01)
        assert result.is_saturated is True
        assert result.signal_status == "computed"
        assert result.saturation_point_tokens is not None
        assert result.data_points_used == 7

    def test_unsaturated_curve(
        self, loss_curve_unsaturated: list[LossDataPoint]
    ) -> None:
        result = analyze_loss_saturation(loss_curve_unsaturated)
        assert result.is_saturated is False
        assert result.signal_status == "computed"
        assert result.saturation_point_tokens is None

    def test_insufficient_data(self) -> None:
        curve = [LossDataPoint(tokens_trained=100, validation_loss=1.0)]
        result = analyze_loss_saturation(curve)
        assert result.signal_status == "insufficient_data"
        assert result.is_saturated is False
        assert result.data_points_used == 1


class TestLossSaturationEdge:
    def test_empty_curve(self) -> None:
        result = analyze_loss_saturation([])
        assert result.signal_status == "insufficient_data"
        assert result.data_points_used == 0

    def test_single_point(self) -> None:
        result = analyze_loss_saturation(
            [LossDataPoint(tokens_trained=100, validation_loss=1.0)]
        )
        assert result.signal_status == "insufficient_data"

    def test_exactly_min_intervals_plus_one(self) -> None:
        curve = [
            LossDataPoint(tokens_trained=i * 100_000_000, validation_loss=2.0 - i * 0.1)
            for i in range(5)
        ]
        result = analyze_loss_saturation(curve, min_intervals=3)
        assert result.signal_status == "computed"
        assert result.data_points_used == 5


class TestLossSaturationError:
    def test_zero_token_delta(self) -> None:
        curve = [
            LossDataPoint(tokens_trained=100, validation_loss=2.0),
            LossDataPoint(tokens_trained=100, validation_loss=1.5),
            LossDataPoint(tokens_trained=200, validation_loss=1.2),
            LossDataPoint(tokens_trained=300, validation_loss=1.199),
            LossDataPoint(tokens_trained=400, validation_loss=1.198),
        ]
        result = analyze_loss_saturation(curve)
        assert result.signal_status == "computed"

    def test_negative_loss(self) -> None:
        curve = [
            LossDataPoint(tokens_trained=100, validation_loss=-1.0),
            LossDataPoint(tokens_trained=200, validation_loss=-2.0),
            LossDataPoint(tokens_trained=300, validation_loss=-3.0),
            LossDataPoint(tokens_trained=400, validation_loss=-4.0),
            LossDataPoint(tokens_trained=500, validation_loss=-5.0),
        ]
        result = analyze_loss_saturation(curve)
        assert result.signal_status == "computed"
