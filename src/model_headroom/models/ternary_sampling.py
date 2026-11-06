"""Bounded deterministic sampling of GGUF weight tensors for the R-03 effective-rank analyzer.

Feeds dequantized tensors into :func:`model_headroom.signals.effective_rank.compute_effective_rank`
without manufacturing gradients. Quantized storage is treated as *fixture* evidence
(source_type="fixture"), not measured gradient evidence.

All sampling is deterministic: the same (gguf_path, tensor_name, max_rows, seed) yields the
same rows. For quantized tensors, rows are block-aligned (whole quantized blocks) so the
sample decodes cleanly.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from model_headroom.models.gguf_loader import (
    extract_gguf_weight_sample,
    open_gguf_reader,
)

if TYPE_CHECKING:
    from pathlib import Path

    import torch


def sample_weight_for_effective_rank(
    gguf_path: str | Path,
    tensor_names: list[str],
    max_rows: int = 256,
    seed: int = 42,
) -> list[tuple[str, torch.Tensor, tuple[int, int]]]:
    """Sample deterministic rows from GGUF weight tensors for R-03 effective rank.

    Reads one tensor at a time (never the full model). Each returned weight is a 2-D
    float32 CPU tensor of shape [sampled_rows, cols]; parent_shape is the full
    [rows, cols] tensor shape used for coverage computation.

    Input:
        gguf_path: Path to a .gguf file.
        tensor_names: Full tensor names to sample (e.g. ["blk.0.ffn_gate_exps.0.weight"]).
        max_rows: Upper bound on sampled rows per tensor (default 256).
        seed: Deterministic RNG seed (same seed -> same rows).
    Output:
        List of (layer_name, weight_tensor, parent_shape) tuples, one per tensor found.
        Tensors that cannot be located or dequantized are skipped.
    Dependencies: Requires gguf and torch. Reads file at gguf_path.
    """
    reader = open_gguf_reader(str(gguf_path))

    # Collect shapes for all requested tensors in a single pass over the reader.
    shapes: dict[str, tuple[int, int]] = {}
    for tensor in reader.tensors:
        name = getattr(tensor, "name", None)
        shape = getattr(tensor, "shape", None)
        if name is None or shape is None or name not in tensor_names:
            continue
        try:
            rows = int(shape[0])
            cols = int(shape[1])
        except (TypeError, IndexError, ValueError):
            continue
        if rows > 0 and cols > 0:
            shapes[name] = (rows, cols)

    results: list[tuple[str, torch.Tensor, tuple[int, int]]] = []
    for tensor_name in tensor_names:
        parent_shape = shapes.get(tensor_name)
        if parent_shape is None:
            continue
        sample = extract_gguf_weight_sample(
            gguf_path, tensor_name, max_rows=max_rows, seed=seed
        )
        if sample is None:
            continue
        results.append((tensor_name, sample, parent_shape))

    return results
