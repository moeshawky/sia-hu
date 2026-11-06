from __future__ import annotations

import pytest
import torch

from model_headroom.models.types import ExpertWeights
from model_headroom.signals.specialization import analyze_specialization


class TestSpecializationFunctional:
    def test_specialized_experts(
        self, expert_weights_specialized: ExpertWeights
    ) -> None:
        result = analyze_specialization(expert_weights_specialized)
        assert result.is_collapsed is False
        assert result.mean_inter_expert_similarity < 0.85
        assert result.signal_status == "computed"

    def test_collapsed_experts(self, expert_weights_collapsed: ExpertWeights) -> None:
        result = analyze_specialization(expert_weights_collapsed)
        assert result.is_collapsed is True
        assert result.mean_inter_expert_similarity == pytest.approx(1.0, abs=1e-5)

    def test_two_experts(self) -> None:
        weights = ExpertWeights(
            expert_weights=[torch.randn(64, 128), torch.randn(64, 128)],
            layer_name="min",
        )
        result = analyze_specialization(weights)
        assert result.signal_status == "computed"
        assert result.mean_inter_expert_similarity == result.max_inter_expert_similarity


class TestSpecializationEdge:
    def test_single_expert(self) -> None:
        weights = ExpertWeights(
            expert_weights=[torch.ones(64, 128)],
            layer_name="single",
        )
        result = analyze_specialization(weights)
        assert result.signal_status == "single_expert"
        assert result.mean_inter_expert_similarity == 0.0

    def test_zero_tensors(self) -> None:
        weights = ExpertWeights(
            expert_weights=[torch.zeros(64, 128), torch.zeros(64, 128)],
            layer_name="zero",
        )
        result = analyze_specialization(weights)
        assert (
            result.mean_inter_expert_similarity == result.mean_inter_expert_similarity
        )

    def test_two_identical_experts(self) -> None:
        w = torch.ones(64, 128)
        weights = ExpertWeights(
            expert_weights=[w.clone(), w.clone()],
            layer_name="identical",
        )
        result = analyze_specialization(weights)
        assert result.mean_inter_expert_similarity == pytest.approx(1.0, abs=1e-5)
        assert result.is_collapsed is True
