"""Effective rank computation signal. Computes the effective rank of weight matrices via SVD-based spectral entropy, with support for sampled and tiled modes."""

from __future__ import annotations

import math
from typing import Literal

import torch

from model_headroom.models.types import EffectiveRankResult, SamplingMode


def compute_effective_rank(
    # Exact zero check before SVD (line 36); ensure SVs are float dtype (line 66); clamp probabilities to avoid log(0) in entropy (line 84); derive coverage from shape ratio when parent_shape known (line 94-97); confidence score by sampling mode (line 99-109)
    weight: torch.Tensor,
    sampling_mode: SamplingMode = "unknown",
    parent_shape: tuple[int, int] | None = None,
    sample_count: int | None = None,
    coverage: float | None = None,
    device: str = "cpu",
) -> EffectiveRankResult:
    """Compute effective rank of a weight tensor using SVD-based spectral entropy. Supports multiple sampling modes and returns confidence estimates."""
    if weight.ndim < 2:
        rows = weight.shape[0] if weight.ndim == 1 else 1
        cols = 1
        full_rank = min(rows, cols)
        return EffectiveRankResult(
            rank=1.0,
            entropy=0.0,
            percentage_of_full_rank=1.0 / full_rank if full_rank > 0 else 0.0,
            singular_values_count=0,
            signal_status="degenerate",
            sampling_mode=sampling_mode,
            coverage=coverage,
            parent_shape=parent_shape,
            sample_count=sample_count,
        )

    try:
        rows, cols = weight.shape
    except ValueError as e:
        return EffectiveRankResult(
            rank=1.0,
            entropy=0.0,
            percentage_of_full_rank=0.0,
            singular_values_count=0,
            signal_status="error",
            error_message=f"weight shape {list(weight.shape)} (ndim {weight.ndim}) is not 2-D [rows, cols]: {e}",
            sampling_mode=sampling_mode,
            coverage=coverage,
            parent_shape=parent_shape,
            sample_count=sample_count,
        )

    full_rank = min(rows, cols)

    if not torch.isfinite(weight).all():
        return EffectiveRankResult(
            rank=1.0,
            entropy=0.0,
            percentage_of_full_rank=0.0,
            singular_values_count=0,
            signal_status="error",
            error_message="Non-finite (NaN/Inf) values in weight matrix",
            sampling_mode=sampling_mode,
            coverage=coverage,
            parent_shape=parent_shape,
            sample_count=sample_count,
        )

    if weight.numel() == 0 or weight.abs().max() == 0:
        return EffectiveRankResult(
            rank=1.0,
            entropy=0.0,
            percentage_of_full_rank=1.0 / full_rank if full_rank > 0 else 0.0,
            singular_values_count=0,
            signal_status="degenerate",
            sampling_mode=sampling_mode,
            coverage=coverage,
            parent_shape=parent_shape,
            sample_count=sample_count,
        )

    try:
        weight = weight.to(device).float()
        svd_result = torch.linalg.svd(weight, full_matrices=False)
        sv = svd_result[1]
    except Exception as e:
        return EffectiveRankResult(
            rank=1.0,
            entropy=0.0,
            percentage_of_full_rank=0.0,
            singular_values_count=0,
            signal_status="error",
            error_message=f"SVD computation failed: {e}",
            sampling_mode=sampling_mode,
            coverage=coverage,
            parent_shape=parent_shape,
            sample_count=sample_count,
        )

    sv = sv.float()
    n = sv.shape[0]

    total = sv.sum().item()
    if total <= 0:
        return EffectiveRankResult(
            rank=1.0,
            entropy=0.0,
            percentage_of_full_rank=0.0,
            singular_values_count=n,
            signal_status="degenerate",
            sampling_mode=sampling_mode,
            coverage=coverage,
            parent_shape=parent_shape,
            sample_count=sample_count,
        )

    probs = sv / total
    probs = probs.clamp(min=1e-12)

    entropy = -(probs * torch.log(probs)).sum().item()

    rank = math.exp(entropy)

    rank = max(1.0, min(rank, float(full_rank)))

    percentage = rank / full_rank if full_rank > 0 else 0.0

    if parent_shape is not None and coverage is None:
        parent_elements = parent_shape[0] * parent_shape[1]
        tile_elements = rows * cols
        coverage = tile_elements / parent_elements if parent_elements > 0 else None

    state: Literal["full", "sampled"] | None = None
    if coverage is not None:
        state = "full" if coverage >= 1.0 else "sampled"

    confidence: float | None = None
    if sampling_mode == "single_tile" and sample_count == 1:
        confidence = 0.3
    elif (
        sampling_mode == "multi_tile" and sample_count is not None and sample_count > 1
    ):
        confidence = min(0.9, 0.3 + 0.1 * sample_count)
    elif sampling_mode == "full_matrix":
        confidence = 0.95
    elif sampling_mode == "randomized_svd":
        confidence = 0.8

    return EffectiveRankResult(
        rank=rank,
        entropy=entropy,
        percentage_of_full_rank=percentage,
        singular_values_count=n,
        signal_status="computed",
        sampling_mode=sampling_mode,
        coverage=coverage,
        parent_shape=parent_shape,
        sample_count=sample_count,
        confidence=confidence,
        state=state,
    )
