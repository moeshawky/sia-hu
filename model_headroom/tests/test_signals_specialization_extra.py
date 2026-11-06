from __future__ import annotations

import torch

from model_headroom.models.types import ExpertWeights
from model_headroom.signals.specialization import analyze_specialization


class TestSpecializationCollapseThreshold:
    def test_just_above_threshold(self) -> None:
        w1 = torch.randn(64, 128)
        w2 = w1 + torch.randn(64, 128) * 0.01
        weights = ExpertWeights(
            expert_weights=[w1, w2],
            layer_name="threshold",
        )
        result = analyze_specialization(weights, collapse_threshold=0.85)
        assert result.signal_status == "computed"

    def test_below_threshold(self) -> None:
        w1 = torch.randn(64, 128)
        w2 = torch.randn(64, 128)
        weights = ExpertWeights(
            expert_weights=[w1, w2],
            layer_name="below",
        )
        result = analyze_specialization(weights, collapse_threshold=0.85)
        assert result.is_collapsed is False

    def test_exception_path(self) -> None:
        w1 = torch.randn(2, 3)
        w2 = torch.randn(2, 3)
        weights = ExpertWeights(
            expert_weights=[w1, w2],
            layer_name="exception",
        )
        result = analyze_specialization(weights)
        assert result.signal_status == "computed"
