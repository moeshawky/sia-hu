from __future__ import annotations

from model_headroom.models.types import LossDataPoint
from model_headroom.signals.loss_saturation import analyze_loss_saturation


class TestLossSaturationEmptySlopes:
    def test_all_zero_token_deltas(self) -> None:
        """Duplicate token counts produce no usable slopes -> insufficient_data,
        not computed/1.0 (audit defect: degenerate input must not yield max
        headroom)."""
        curve = [
            LossDataPoint(tokens_trained=100, validation_loss=2.0),
            LossDataPoint(tokens_trained=100, validation_loss=1.5),
            LossDataPoint(tokens_trained=100, validation_loss=1.2),
            LossDataPoint(tokens_trained=100, validation_loss=1.1),
        ]
        result = analyze_loss_saturation(curve)
        assert result.signal_status == "insufficient_data"
        assert result.is_saturated is False
