from __future__ import annotations

from model_headroom.models.types import LossDataPoint
from model_headroom.signals.loss_saturation import analyze_loss_saturation


class TestLossSaturationEmptySlopes:
    def test_all_zero_token_deltas(self) -> None:
        curve = [
            LossDataPoint(tokens_trained=100, validation_loss=2.0),
            LossDataPoint(tokens_trained=100, validation_loss=1.5),
            LossDataPoint(tokens_trained=100, validation_loss=1.2),
            LossDataPoint(tokens_trained=100, validation_loss=1.1),
        ]
        result = analyze_loss_saturation(curve)
        assert result.signal_status == "computed"
        assert result.is_saturated is False
