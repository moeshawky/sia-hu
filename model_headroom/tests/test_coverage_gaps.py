from __future__ import annotations

from typing import Any

from model_headroom.models.loader import (
    extract_layer_weights,
)
from model_headroom.models.types import SignalResult
from model_headroom.reporting.human_report import _format_signal_status


class TestExtractLayerWeightsCorruptBin:
    def test_corrupt_bin_returns_none(self, tmp_path: Any) -> None:
        import torch

        bin_path = tmp_path / "model.bin"
        torch.save({"dummy.key": torch.randn(2, 3)}, bin_path)
        result = extract_layer_weights(str(tmp_path), "nonexistent.weight")
        assert result is None

    def test_safetensors_missing_key(self, tmp_path: Any) -> None:
        from safetensors.torch import save_file

        t = {"model.layers.0.weight": __import__("torch").randn(2, 3)}
        save_file(t, str(tmp_path / "model.safetensors"))
        result = extract_layer_weights(str(tmp_path), "nonexistent.weight")
        assert result is None


class TestHumanReportUnknownSignal:
    def test_unknown_signal_name(self) -> None:
        from model_headroom.models.types import EffectiveRankResult

        signal = SignalResult(
            signal_name="custom_unknown_signal",
            status="computed",
            data=EffectiveRankResult(
                rank=5.0,
                entropy=1.0,
                percentage_of_full_rank=0.625,
                singular_values_count=8,
                signal_status="computed",
            ),
            duration_seconds=0.0,
        )
        output = _format_signal_status(signal)
        assert "UNKNOWN SIGNAL" in output


class TestGrowthFactorExceptionPath:
    def test_exception_in_computation(self) -> None:
        from unittest.mock import patch

        from model_headroom.models.types import IsoFLOPPoint
        from model_headroom.signals.growth_factor import analyze_growth_factor

        points = [
            IsoFLOPPoint(
                compute_flops=1e18, model_params=10_000_000, measured_loss=2.0
            ),
            IsoFLOPPoint(
                compute_flops=1e19, model_params=100_000_000, measured_loss=1.5
            ),
            IsoFLOPPoint(
                compute_flops=1e20, model_params=1_000_000_000, measured_loss=1.2
            ),
        ]

        with patch(
            "model_headroom.signals.growth_factor.math.log",
            side_effect=RuntimeError("math error"),
        ):
            result = analyze_growth_factor(points)
        assert result.signal_status == "error"
