"""Dead neuron analysis signal. Computes per-neuron mean activations to identify dead neurons, reports activation statistics including RMS, mean absolute, variance, near-zero occupancy, and cross-token variance."""

from __future__ import annotations

import torch

from model_headroom.models.types import DeadNeuronReport, MLPActivations, SourceType


def analyze_dead_neurons(
    # Guard against division by zero (line 21); cast to float for statistical computations (line 27); near-zero occupancy measures fraction of activations below 1e-5 (line 31); cross-token variance measures how neuron mean activity varies across tokens (line 32-34)
    activations: MLPActivations,
    threshold: float = 1e-6,
    return_indices: bool = False,
    source_type: SourceType = "measured",
    device: str = "cpu",
) -> DeadNeuronReport:
    """Analyze dead neurons from MLP activations. Computes per-neuron mean activations, identifies neurons below threshold, and reports activation statistics."""
    try:
        tensor = activations.activations.to(device)
        neuron_means = tensor.mean(dim=(0, 1))
        total_neurons = neuron_means.shape[0]

        if not torch.isfinite(neuron_means).all() or not torch.isfinite(tensor).all():
            return DeadNeuronReport(
                total_neurons=total_neurons,
                dead_count=0,
                dead_fraction=0.0,
                dead_indices=None,
                signal_status="error",
                error_message="Non-finite (NaN/Inf) values detected in activations",
                source_type=source_type,
            )

        dead_mask = neuron_means < threshold
        dead_count = int(dead_mask.sum().item())
        dead_fraction = dead_count / total_neurons if total_neurons > 0 else 0.0

        dead_indices: list[int] | None = None
        if return_indices:
            dead_indices = torch.where(dead_mask)[0].tolist()

        flat = tensor.float()
        activation_rms = float(torch.sqrt((flat**2).mean()).item())
        mean_abs = float(flat.abs().mean().item())
        variance = float(flat.var(unbiased=False).item())
        near_zero = float((flat.abs() < 1e-5).float().mean().item())
        cross_token_variance = (
            float(neuron_means.var(unbiased=False).item()) if total_neurons > 1 else 0.0
        )

        return DeadNeuronReport(
            total_neurons=total_neurons,
            dead_count=dead_count,
            dead_fraction=dead_fraction,
            dead_indices=dead_indices,
            signal_status="computed",
            activation_rms=activation_rms,
            mean_abs_activation=mean_abs,
            variance=variance,
            near_zero_occupancy=near_zero,
            cross_token_variance=cross_token_variance,
            source_type=source_type,
        )
    except Exception as e:
        return DeadNeuronReport(
            total_neurons=0,
            dead_count=0,
            dead_fraction=0.0,
            dead_indices=None,
            signal_status="error",
            error_message=f"Dead neuron computation failed: {e}",
            source_type=source_type,
        )
