from __future__ import annotations

import torch

from model_headroom.models.types import MLPActivations
from model_headroom.signals.dead_neurons import analyze_dead_neurons


class TestDeadNeuronsFunctional:
    def test_no_dead_neurons(self) -> None:
        activations = MLPActivations(
            activations=torch.ones(2, 10, 64),
            layer_name="test",
        )
        result = analyze_dead_neurons(activations)
        assert result.dead_fraction == 0.0
        assert result.dead_count == 0
        assert result.signal_status == "computed"

    def test_dead_neuron_detection(self, mlp_activations_mixed: MLPActivations) -> None:
        result = analyze_dead_neurons(mlp_activations_mixed)
        assert result.dead_count == 32
        assert result.dead_fraction == 0.5

    def test_all_dead(self) -> None:
        activations = MLPActivations(
            activations=torch.zeros(2, 10, 64),
            layer_name="all_dead",
        )
        result = analyze_dead_neurons(activations)
        assert result.dead_count == 64
        assert result.dead_fraction == 1.0


class TestDeadNeuronsEdge:
    def test_threshold_boundary(self) -> None:
        activations = MLPActivations(
            activations=torch.full((2, 10, 64), 1e-6),
            layer_name="boundary",
        )
        result = analyze_dead_neurons(activations)
        assert result.dead_count == 0

    def test_return_indices(self) -> None:
        activations = MLPActivations(
            activations=torch.cat(
                [torch.ones(2, 10, 32), torch.zeros(2, 10, 32)], dim=2
            ),
            layer_name="test",
        )
        result = analyze_dead_neurons(activations, return_indices=True)
        assert result.dead_indices is not None
        assert len(result.dead_indices) == 32


class TestDeadNeuronsError:
    def test_2d_tensor(self) -> None:
        activations = MLPActivations(
            activations=torch.ones(10, 64),
            layer_name="bad",
        )
        result = analyze_dead_neurons(activations)
        assert result.signal_status == "error"
