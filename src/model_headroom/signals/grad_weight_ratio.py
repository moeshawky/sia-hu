"""Gradient-to-weight norm ratio computation (R-04).

ratio = ||gradient||_2 / ||weight||_2.
Threshold: ratio > mean + 2*std across all layers = high pressure.
No division by zero (returns ratio=0.0 for zero weights).

Dependencies: torch (norm computation), models.types (GradientWeightPair, GradWeightRatio).
Provides: analyze_grad_weight_ratios().
"""

from __future__ import annotations

import math

import torch

from model_headroom.models.types import GradientWeightPair, GradWeightRatio


def analyze_grad_weight_ratios(
    pairs: list[GradientWeightPair],
    device: str = "cpu",
) -> list[GradWeightRatio]:
    """Compute gradient-to-weight norm ratios for a list of layer pairs.

    Each ratio measures how large the gradient is relative to the weight.
    A high ratio indicates the layer is under significant training pressure.

    Input contract:
        pairs: List of GradientWeightPair, each containing gradient and weight
            tensors of the same shape. Empty list returns empty result.

    Output contract (success):
        Returns list of GradWeightRatio with signal_status="computed".
        Each ratio >= 0.0. is_high_pressure=True if ratio > mean + 2*std.

    Output contract (zero weight):
        signal_status="zero_weight", ratio=0.0, z_score=0.0.

    Output contract (no gradient):
        signal_status="no_gradient", ratio=0.0, z_score=0.0.

    Output contract (error):
        signal_status="error", error_message describes the failure.

    Invariants:
        - All ratios >= 0.0
        - z_score is 0.0 when std is zero across layers
        - len(result) == len(pairs)

    Dependencies: Requires torch for norm computation.
    """
    if not pairs:
        return []

    results: list[GradWeightRatio] = []
    ratios: list[float] = []

    for pair in pairs:
        weight_norm = torch.norm(pair.weight.to(device)).item()
        grad_norm = torch.norm(pair.gradient.to(device)).item()

        if weight_norm == 0.0:
            results.append(
                GradWeightRatio(
                    ratio=0.0,
                    is_high_pressure=False,
                    z_score=0.0,
                    layer_name=pair.layer_name,
                    signal_status="zero_weight",
                )
            )
            ratios.append(0.0)
        elif grad_norm == 0.0:
            results.append(
                GradWeightRatio(
                    ratio=0.0,
                    is_high_pressure=False,
                    z_score=0.0,
                    layer_name=pair.layer_name,
                    signal_status="no_gradient",
                )
            )
            ratios.append(0.0)
        elif not math.isfinite(grad_norm):
            results.append(
                GradWeightRatio(
                    ratio=0.0,
                    is_high_pressure=False,
                    z_score=0.0,
                    layer_name=pair.layer_name,
                    signal_status="invalid_evidence",
                )
            )
            ratios.append(0.0)
        else:
            ratio = grad_norm / weight_norm
            ratios.append(ratio)
            results.append(
                GradWeightRatio(
                    ratio=ratio,
                    is_high_pressure=False,  # placeholder, updated below
                    z_score=0.0,  # placeholder, updated below
                    layer_name=pair.layer_name,
                    signal_status="computed",
                )
            )

    # Compute mean and std of all computed ratios
    valid_ratios = [r for r in ratios if r > 0.0]
    if not valid_ratios:
        return results

    mean_r = sum(valid_ratios) / len(valid_ratios)
    variance = sum((r - mean_r) ** 2 for r in valid_ratios) / len(valid_ratios)
    std_r = variance**0.5

    # Update is_high_pressure and z_score for computed results
    for i, result in enumerate(results):
        if result.signal_status == "computed":
            z = (ratios[i] - mean_r) / std_r if std_r > 0 else 0.0
            is_high = ratios[i] > mean_r + 2 * std_r
            results[i] = GradWeightRatio(
                ratio=result.ratio,
                is_high_pressure=is_high,
                z_score=z,
                layer_name=result.layer_name,
                signal_status=result.signal_status,
            )

    return results
