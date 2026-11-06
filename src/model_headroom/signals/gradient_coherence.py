"""Gradient coherence signal. Computes pairwise cosine similarity between gradient batches, a cosine-derived subspace overlap, and variance spectrum of stacked gradients via SVD.

Cross-batch contract: every tensor in ``gradients`` must be the gradient of the
SAME named layer/projection captured on a DIFFERENT batch. Tensors from
different layers are cross-layer, not cross-batch, and never satisfy the
contract. Scalar norms (0-dim or single-element tensors) cannot satisfy a
matrix input and are rejected.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from model_headroom.models.types import GradientCoherenceReport

if TYPE_CHECKING:
    from model_headroom.models.types import GradientWeightPair


def _cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    """Compute cosine similarity between two tensors using flattened float vectors with epsilon stabilization."""
    fa = a.float().flatten()
    fb = b.float().flatten()
    denom = (fa.norm().item() * fb.norm().item()) + 1e-12
    if denom == 0:
        return 0.0
    return float((fa @ fb).item() / denom)


def compute_gradient_coherence(
    gradients: list[torch.Tensor],
    *,
    layer_names: list[str] | None = None,
    device: str = "cpu",
) -> GradientCoherenceReport:
    """Compute gradient coherence metrics from a list of gradient tensors. Reports mean cosine similarity, subspace overlap, and variance spectrum.

    Cross-batch contract: ``gradients`` must hold >=2 gradients of the SAME
    named layer/projection, one per batch. When ``layer_names`` is supplied it
    must have one entry per tensor and all entries must be identical; a list
    mixing different layer names is rejected with ``signal_status="error"``
    because cross-layer similarity is not cross-batch coherence. Tensors that
    are scalar norms (0-dim or single-element) are rejected for the same
    reason. Missing batches yield ``"insufficient_data"``, never zeros
    presented as computed.
    """
    if len(gradients) < 2:
        return GradientCoherenceReport(
            mean_cosine_similarity=0.0,
            subspace_overlap=0.0,
            variance_spectrum=0.0,
            signal_status="insufficient_data",
            error_message="need >=2 gradient batches",
        )

    if layer_names is not None:
        if len(layer_names) != len(gradients):
            return GradientCoherenceReport(
                mean_cosine_similarity=0.0,
                subspace_overlap=0.0,
                variance_spectrum=0.0,
                signal_status="error",
                error_message=(
                    "layer_names length "
                    f"({len(layer_names)}) must match gradients "
                    f"({len(gradients)})"
                ),
            )
        if len(set(layer_names)) > 1:
            return GradientCoherenceReport(
                mean_cosine_similarity=0.0,
                subspace_overlap=0.0,
                variance_spectrum=0.0,
                signal_status="error",
                error_message=(
                    "cross-layer gradients "
                    f"({sorted(set(layer_names))}) do not satisfy the "
                    "cross-batch contract: coherence requires >=2 batches "
                    "of the SAME layer/projection"
                ),
            )

    first_shape = gradients[0].shape
    for g in gradients[1:]:
        if g.shape != first_shape:
            return GradientCoherenceReport(
                mean_cosine_similarity=0.0,
                subspace_overlap=0.0,
                variance_spectrum=0.0,
                signal_status="error",
                error_message=(
                    f"gradient batch shapes differ ({list(first_shape)} vs "
                    f"{list(g.shape)}): batches of one layer share one shape"
                ),
            )
    for g in gradients:
        if g.numel() <= 1:
            return GradientCoherenceReport(
                mean_cosine_similarity=0.0,
                subspace_overlap=0.0,
                variance_spectrum=0.0,
                signal_status="error",
                error_message=(
                    "scalar norms cannot satisfy matrix inputs: each batch "
                    "must be a full gradient tensor, not a scalar summary"
                ),
            )

    try:
        n = len(gradients)
        sims: list[float] = []
        for i in range(n):
            for j in range(i + 1, n):
                sims.append(_cosine_similarity(gradients[i], gradients[j]))

        mean_cos = sum(sims) / len(sims) if sims else 0.0
        var_cos = sum((s - mean_cos) ** 2 for s in sims) / len(sims) if sims else 0.0
        subspace_overlap = max(0.0, min(1.0, (mean_cos + 1.0) / 2.0))

        flat = [g.float().flatten() for g in gradients]
        stacked = torch.stack(flat, dim=0).to(device)
        try:
            sv = torch.linalg.svd(stacked, full_matrices=False)[1].float()
            total = sv.sum().item()
            if total > 0:
                variance_spectrum = float(sv.var(unbiased=False).item())
            else:
                variance_spectrum = 0.0
        except Exception:
            variance_spectrum = var_cos

        return GradientCoherenceReport(
            mean_cosine_similarity=mean_cos,
            subspace_overlap=subspace_overlap,
            variance_spectrum=variance_spectrum,
            signal_status="computed",
        )
    except Exception as e:
        return GradientCoherenceReport(
            mean_cosine_similarity=0.0,
            subspace_overlap=0.0,
            variance_spectrum=0.0,
            signal_status="error",
            error_message=f"coherence failed: {e}",
        )


def compute_coherence_for_layer(
    pairs: list[GradientWeightPair],
) -> GradientCoherenceReport:
    """Compute cross-batch coherence for ONE named layer/projection.

    Accepts multiple batches of the same ``layer_name`` (one
    :class:`GradientWeightPair` per batch) and enforces the cross-batch
    contract: fewer than 2 batches yields ``"insufficient_data"``, and pairs
    spanning different layer names are rejected with ``"error"`` — they must
    never be silently treated as batches of one layer.
    """
    if len(pairs) < 2:
        return GradientCoherenceReport(
            mean_cosine_similarity=0.0,
            subspace_overlap=0.0,
            variance_spectrum=0.0,
            signal_status="insufficient_data",
            error_message="need >=2 gradient batches of the same layer",
        )
    names = [p.layer_name for p in pairs]
    return compute_gradient_coherence([p.gradient for p in pairs], layer_names=names)
