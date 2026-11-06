from __future__ import annotations

import torch

from model_headroom.models.types import GradientWeightPair
from model_headroom.signals.grad_weight_ratio import analyze_grad_weight_ratios


class TestGradWeightRatioFunctional:
    def test_normal_ratios(self) -> None:
        pairs = [
            GradientWeightPair(
                gradient=torch.ones(10, 10) * 0.5,
                weight=torch.ones(10, 10),
                layer_name="layer_0",
            ),
            GradientWeightPair(
                gradient=torch.ones(10, 10) * 0.5,
                weight=torch.ones(10, 10),
                layer_name="layer_1",
            ),
        ]
        result = analyze_grad_weight_ratios(pairs)
        assert len(result) == 2
        assert all(not r.is_high_pressure for r in result)
        assert all(r.signal_status == "computed" for r in result)

    def test_high_pressure_detection(self) -> None:
        pairs = [
            GradientWeightPair(
                gradient=torch.ones(10, 10) * 0.1,
                weight=torch.ones(10, 10),
                layer_name=f"low{i}",
            )
            for i in range(8)
        ] + [
            GradientWeightPair(
                gradient=torch.ones(10, 10) * 50.0,
                weight=torch.ones(10, 10),
                layer_name="high",
            ),
        ]
        result = analyze_grad_weight_ratios(pairs)
        assert any(r.is_high_pressure and r.layer_name == "high" for r in result)


class TestGradWeightRatioEdge:
    def test_empty_list(self) -> None:
        result = analyze_grad_weight_ratios([])
        assert result == []

    def test_zero_weight(self) -> None:
        pairs = [
            GradientWeightPair(
                gradient=torch.ones(10, 10),
                weight=torch.zeros(10, 10),
                layer_name="zero_w",
            ),
        ]
        result = analyze_grad_weight_ratios(pairs)
        assert len(result) == 1
        assert result[0].signal_status == "zero_weight"
        assert result[0].ratio == 0.0

    def test_zero_gradient(self) -> None:
        pairs = [
            GradientWeightPair(
                gradient=torch.zeros(10, 10),
                weight=torch.ones(10, 10),
                layer_name="zero_g",
            ),
        ]
        result = analyze_grad_weight_ratios(pairs)
        assert len(result) == 1
        assert result[0].signal_status == "no_gradient"
        assert result[0].ratio == 0.0

    def test_unicode_layer_name(self) -> None:
        pairs = [
            GradientWeightPair(
                gradient=torch.ones(10, 10) * 0.5,
                weight=torch.ones(10, 10),
                layer_name="层_日本語_test",
            ),
        ]
        result = analyze_grad_weight_ratios(pairs)
        assert result[0].layer_name == "层_日本語_test"

    def test_mixed_valid_zero_weight(self) -> None:
        pairs = [
            GradientWeightPair(
                gradient=torch.ones(10, 10) * 0.5,
                weight=torch.zeros(10, 10),
                layer_name="zero_w",
            ),
            GradientWeightPair(
                gradient=torch.ones(10, 10) * 0.5,
                weight=torch.ones(10, 10),
                layer_name="normal",
            ),
        ]
        result = analyze_grad_weight_ratios(pairs)
        assert result[0].signal_status == "zero_weight"
        assert result[1].signal_status == "computed"
