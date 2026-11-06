"""Refusal-direction analysis for model_headroom.

Computes a refusal direction from contrastive harmful/harmless prompt sets, on
residual-stream activations captured live or loaded from disk, or recovered from
the weight deltas of an already-abliterated checkpoint.

This is a standalone capability, not a headroom signal: it never contributes to
``compute_headroom_score`` and is never invoked by ``run_analysis``. It ships
its own console script, ``model-headroom-refusal``.
"""
