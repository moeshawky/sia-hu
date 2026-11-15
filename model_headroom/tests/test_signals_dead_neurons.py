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


class TestDeadNeuronsSignedActivations:
    """Negative-mean neurons are active, not dead. Verified against the Ornith
    regression: a signed MLP intermediate must not misclassify negative-but-active
    channels as dead."""

    def test_negative_mean_not_dead(self) -> None:
        # Signed activations: some neurons have negative mean but are active.
        negative_active = torch.full((2, 10, 16), -0.5)
        positive_active = torch.full((2, 10, 16), 0.5)
        never_active = torch.zeros(2, 10, 16)
        activations = MLPActivations(
            activations=torch.cat(
                [negative_active, positive_active, never_active], dim=2
            ),
            layer_name="signed",
        )
        result = analyze_dead_neurons(activations)

        # 16 never-active (all zeros) are dead; negative-active neurons are NOT dead.
        assert result.dead_count == 16
        assert result.dead_fraction == 16 / 48
        assert result.negative_mean_count == 16
        assert result.negative_mean_fraction == 16 / 48
        assert result.never_active_count == 16
        assert result.never_active_fraction == 16 / 48

    def test_negative_mean_fields_populated(self) -> None:
        activations = MLPActivations(
            activations=torch.full((2, 4, 8), -0.3),
            layer_name="all_negative",
        )
        result = analyze_dead_neurons(activations)
        assert result.dead_count == 0
        assert result.dead_fraction == 0.0
        assert result.negative_mean_count == 8
        assert result.negative_mean_fraction == 1.0
        assert result.never_active_count == 0
        assert result.never_active_fraction == 0.0

    def test_no_negative_mean(self) -> None:
        activations = MLPActivations(
            activations=torch.full((2, 4, 8), 0.3),
            layer_name="all_positive",
        )
        result = analyze_dead_neurons(activations)
        assert result.dead_count == 0
        assert result.dead_fraction == 0.0
        assert result.negative_mean_count == 0
        assert result.negative_mean_fraction == 0.0
        assert result.never_active_count == 0
        assert result.never_active_fraction == 0.0


class TestDeadNeuronsError:
    def test_2d_tensor(self) -> None:
        activations = MLPActivations(
            activations=torch.ones(10, 64),
            layer_name="bad",
        )
        result = analyze_dead_neurons(activations)
        assert result.signal_status == "error"
