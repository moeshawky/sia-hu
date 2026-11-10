"""Gradient effective rank signal. Computes the effective rank of gradient tensors via SVD spectral entropy, with percentile rank thresholds r50, r90, r95, r99.

Provenance semantics (shared with R-08 gradient coherence): batch helpers
accept one gradient per batch and attribute every result to its named
layer/projection. Scalar norms (0-dim or single-element tensors) and <2-dim
tensors yield ``"degenerate"``, never a fabricated rank.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from model_headroom.models.types import GradientEffectiveRankReport

if TYPE_CHECKING:
    from model_headroom.models.types import GradientWeightPair


def compute_gradient_effective_rank(
    # Singular values are normalized to a probability distribution before entropy;
    # the upper bound on rank is min of rows and cols; rank is clamped to [1.0, full_rank].
    gradient: torch.Tensor,
    device: str = "cpu",
) -> GradientEffectiveRankReport:
    """Compute effective rank of a gradient tensor using SVD spectral entropy. Reports rank, entropy, and percentile thresholds."""
    if gradient.ndim < 2:
        return GradientEffectiveRankReport(
            rank=1.0,
            entropy=0.0,
            r50=1,
            r90=1,
            r95=1,
            r99=1,
            spectral_entropy=0.0,
            signal_status="degenerate",
        )

    if gradient.abs().sum() == 0:
        return GradientEffectiveRankReport(
            rank=1.0,
            entropy=0.0,
            r50=1,
            r90=1,
            r95=1,
            r99=1,
            spectral_entropy=0.0,
            signal_status="degenerate",
        )

    try:
        gradient = gradient.to(device)
        sv = torch.linalg.svd(gradient.float(), full_matrices=False)[1]
    except Exception as e:
        return GradientEffectiveRankReport(
            rank=1.0,
            entropy=0.0,
            r50=0,
            r90=0,
            r95=0,
            r99=0,
            spectral_entropy=0.0,
            signal_status="error",
            error_message=f"SVD failed: {e}",
        )

    sv = sv.float()
    total = sv.sum().item()
    if total <= 0:
        return GradientEffectiveRankReport(
            rank=1.0,
            entropy=0.0,
            r50=0,
            r90=0,
            r95=0,
            r99=0,
            spectral_entropy=0.0,
            signal_status="degenerate",
        )

    probs = sv / total
    probs = probs.clamp(min=1e-12)
    entropy = -(probs * torch.log(probs)).sum().item()
    rank = math.exp(entropy)
    full_rank = min(gradient.shape[0], gradient.shape[1])
    rank = max(1.0, min(rank, float(full_rank)))

    cumsum = torch.cumsum(sv, dim=0)
    total_cum = cumsum[-1].item()

    def rank_for_percent(p: float) -> int:
        thresh = total_cum * p
        return (
            int((cumsum >= thresh).nonzero(as_tuple=False)[0].item()) + 1
            if total_cum > 0
            else full_rank
        )

    r50 = rank_for_percent(0.50)
    r90 = rank_for_percent(0.90)
    r95 = rank_for_percent(0.95)
    r99 = rank_for_percent(0.99)

    return GradientEffectiveRankReport(
        rank=rank,
        entropy=entropy,
        r50=r50,
        r90=r90,
        r95=r95,
        r99=r99,
        spectral_entropy=entropy,
        signal_status="computed",
    )


def analyze_gradient_effective_ranks(
    gradients: list[torch.Tensor],
    *,
    layer_names: list[str] | None = None,
    device: str = "cpu",
) -> list[GradientEffectiveRankReport]:
    """Compute effective rank for each gradient tensor in a list. Returns a list of GradientEffectiveRankReport, one per gradient.

    When ``layer_names`` is supplied it must carry one entry per gradient; a
    length mismatch raises ``ValueError`` so unattributed batches can never
    silently pass as attributed ones. Scalar summaries are rejected upstream
    by :func:`compute_gradient_effective_rank` with ``"degenerate"``.
    """
    if layer_names is not None and len(layer_names) != len(gradients):
        raise ValueError(
            f"layer_names length ({len(layer_names)}) must match "
            f"gradients ({len(gradients)})"
        )
    return [compute_gradient_effective_rank(g, device=device) for g in gradients]


def analyze_gradient_effective_ranks_for_pairs(
    pairs: list[GradientWeightPair],
    device: str = "cpu",
) -> list[tuple[str, GradientEffectiveRankReport]]:
    """Compute effective rank per batch, attributed to its layer/projection.

    Returns ``(layer_name, report)`` tuples in input order so every rank keeps
    its provenance — the same attribution semantics as R-08 coherence, which
    additionally requires all batches to share one layer name.
    """
    return [
        (p.layer_name, compute_gradient_effective_rank(p.gradient, device=device))
        for p in pairs
    ]
