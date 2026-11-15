"""Dead neuron analysis signal. Computes per-neuron activation statistics to identify dead neurons, reports activation statistics including RMS, mean absolute, variance, near-zero occupancy, and cross-token variance. Distinguishes genuinely inactive channels from channels with negative mean activations.

`dead_count`/`dead_fraction` and `never_active_count`/`never_active_fraction`
are aliases: under the activity-based dead criterion (maximum absolute
activation across all batch and token positions is exactly 0), a neuron that
never fired is by definition dead, so the two statistics carry identical
values for a single capture. They are reported separately so a consumer can
read "dead" and "never-active" as distinct concepts without assuming
independence — they are not independent statistics.
"""

from __future__ import annotations

import torch

from model_headroom.models.types import DeadNeuronReport, MLPActivations, SourceType


def analyze_dead_neurons(
    activations: MLPActivations,
    return_indices: bool = False,
    source_type: SourceType = "measured",
    device: str = "cpu",
) -> DeadNeuronReport:
    """Analyze dead neurons from MLP activations. Reports dead/inactive, negative-mean, and never-active channel counts with activation statistics."""
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

        max_abs_per_neuron = torch.amax(tensor.abs(), dim=(0, 1))
        dead_mask = max_abs_per_neuron == 0
        dead_count = int(dead_mask.sum().item())
        dead_fraction = dead_count / total_neurons if total_neurons > 0 else 0.0

        negative_mean_mask = neuron_means < 0
        negative_mean_count = int(negative_mean_mask.sum().item())
        negative_mean_fraction = (
            negative_mean_count / total_neurons if total_neurons > 0 else 0.0
        )

        never_active_count = dead_count
        never_active_fraction = dead_fraction

        dead_indices: list[int] | None = None
        if return_indices:
            dead_indices = torch.where(dead_mask)[0].tolist()

        flat = tensor.float()
        activation_rms = float(torch.sqrt((flat**2).mean()).item())
        mean_abs = float(flat.abs().mean().item())
        variance = float(flat.var(unbiased=False).item())
        near_zero = float((flat.abs() < 1e-5).float().mean().item())

        # Degenerate criterion disclosure: exact-zero amax essentially never
        # fires on real gated-FFN activations, so dead_count == 0 with high
        # near_zero_occupancy means the criterion cannot discriminate — the
        # signal must not present a constant 1.0 as a measurement.
        degenerate_criterion = dead_count == 0 and near_zero > 0.5 and total_neurons > 0

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
            negative_mean_count=negative_mean_count,
            negative_mean_fraction=negative_mean_fraction,
            never_active_count=never_active_count,
            never_active_fraction=never_active_fraction,
            source_type=source_type,
            degenerate_criterion=degenerate_criterion,
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
