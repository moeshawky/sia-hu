"""Expert specialization collapse detection (R-07).

Pairwise cosine similarity between expert weight vectors.
collapse_threshold = 0.85. is_collapsed = mean_similarity > threshold.

Dependencies: torch (cosine similarity), models.types (ExpertWeights, SpecializationReport).
Provides: analyze_specialization().
"""

from __future__ import annotations

import torch

from model_headroom.models.types import ExpertWeights, SpecializationReport


def analyze_specialization(
    expert_weights: ExpertWeights,
    collapse_threshold: float = 0.85,
    device: str = "cpu",
) -> SpecializationReport:
    """Detect whether experts have collapsed (lost specialization).

    Computes pairwise cosine similarity between flattened expert weight
    vectors. If mean similarity exceeds the threshold, experts are
    considered collapsed — they've converged to similar representations.

    Input contract:
        expert_weights: ExpertWeights with >= 2 expert weight tensors.
            Each tensor has shape (hidden, intermediate).
        collapse_threshold: Similarity threshold for collapse detection.
            Default 0.85.

    Output contract (success):
        Returns SpecializationReport with signal_status="computed".
        mean_inter_expert_similarity in [-1.0, 1.0].

    Output contract (single_expert):
        signal_status="single_expert" if < 2 experts.
        Similarity values set to 0.0.

    Output contract (error):
        signal_status="error", error_message describes the failure.

    Invariants:
        - mean_inter_expert_similarity in [-1.0, 1.0]
        - max_inter_expert_similarity >= mean_inter_expert_similarity
        - If signal_status is "single_expert", similarity is 0.0

    Dependencies: Requires torch for cosine similarity computation.
    """
    weights = expert_weights.expert_weights

    if len(weights) < 2:
        return SpecializationReport(
            mean_inter_expert_similarity=0.0,
            max_inter_expert_similarity=0.0,
            collapse_threshold=collapse_threshold,
            is_collapsed=False,
            signal_status="single_expert",
        )

    try:
        # Flatten each expert weight to a 1D vector and move to device
        flat_weights = [w.flatten().float().to(device) for w in weights]
        stacked = torch.stack(flat_weights)

        # Normalize once, then single batched Gram GEMM
        normalized = torch.nn.functional.normalize(stacked, p=2, dim=1)
        gram = normalized @ normalized.T

        # Extract upper triangle (i < j) in pairwise iteration order
        iu = torch.triu_indices(stacked.shape[0], stacked.shape[0], offset=1)
        similarities: list[float] = gram[iu[0], iu[1]].tolist()

        if not similarities:
            return SpecializationReport(
                mean_inter_expert_similarity=0.0,
                max_inter_expert_similarity=0.0,
                collapse_threshold=collapse_threshold,
                is_collapsed=False,
                signal_status="single_expert",
            )

        mean_sim = sum(similarities) / len(similarities)
        max_sim = max(similarities)
        is_collapsed = mean_sim > collapse_threshold

        return SpecializationReport(
            mean_inter_expert_similarity=mean_sim,
            max_inter_expert_similarity=max_sim,
            collapse_threshold=collapse_threshold,
            is_collapsed=is_collapsed,
            signal_status="computed",
        )
    except Exception as e:
        return SpecializationReport(
            mean_inter_expert_similarity=0.0,
            max_inter_expert_similarity=0.0,
            collapse_threshold=collapse_threshold,
            is_collapsed=False,
            signal_status="error",
            error_message=f"Specialization computation failed: {e}",
        )
