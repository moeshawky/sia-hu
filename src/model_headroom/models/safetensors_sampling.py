"""Bounded deterministic sampling of safetensors weight tensors for the R-03 effective-rank analyzer.

Mirrors the GGUF sampler contract in :mod:`model_headroom.models.ternary_sampling`:
the same (model_path, tensor_name, max_rows, seed) yields the same rows, the full
column dimension is preserved, the row count is capped, and one tensor is read at
a time so a multi-shard checkpoint is never materialised in RAM.

Weights read from a safetensors checkpoint are the checkpoint's own tensors. They
are weight evidence only — no gradient is manufactured, exactly as on the GGUF
path — so the caller decides the provenance the report carries.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

_INDEX_FILENAME = "model.safetensors.index.json"


def find_checkpoint_dir(model_path: str | Path) -> Path | None:
    """Locate the directory holding a safetensors checkpoint's shards.

    Input:
        model_path: Path to a checkpoint directory, or to a parent of one.
    Output (found): The shallowest directory under model_path that holds a
        safetensors index or shard files.
    Output (not found): None.
    Invariants:
        - Never raises for a missing, empty, or unreadable path.
        - Shallowest match wins, so a checkpoint's own directory beats a nested
          copy of it.
    """
    root = Path(model_path)
    if not root.is_dir():
        return None
    if _holds_safetensors(root):
        return root
    candidates = sorted(
        (p for p in root.rglob("*") if p.is_dir() and _holds_safetensors(p)),
        key=lambda p: (len(p.parts), str(p)),
    )
    return candidates[0] if candidates else None


def _holds_safetensors(directory: Path) -> bool:
    """Report whether a directory holds a safetensors index or any shard."""
    if (directory / _INDEX_FILENAME).exists():
        return True
    return any(directory.glob("*.safetensors"))


def safetensors_weight_map(model_path: str | Path) -> dict[str, str]:
    """Map every tensor name in a safetensors checkpoint to the shard that holds it.

    Reads the checkpoint's ``model.safetensors.index.json`` when present. A
    checkpoint written as loose shards without an index falls back to reading each
    shard's key list.

    Input:
        model_path: Path to a checkpoint directory, or to a parent of one.
    Output: Mapping of tensor name to shard file name. Empty when the checkpoint
        cannot be read or holds no tensors.
    Invariants:
        - Never raises for a missing, empty, or malformed checkpoint.
        - Shard names are file names relative to the checkpoint directory.
    """
    directory = find_checkpoint_dir(model_path)
    if directory is None:
        return {}

    index_path = directory / _INDEX_FILENAME
    if index_path.exists():
        try:
            data = json.loads(index_path.read_text())
        except (OSError, ValueError):
            return {}
        raw = data.get("weight_map") if isinstance(data, dict) else None
        if isinstance(raw, dict):
            return {str(name): str(shard) for name, shard in raw.items()}
        return {}

    weight_map: dict[str, str] = {}
    for shard in sorted(directory.glob("*.safetensors")):
        for name in _shard_tensor_names(shard):
            weight_map[name] = shard.name
    return weight_map


def _shard_tensor_names(shard_path: Path) -> list[str]:
    """List the tensor names held by a safetensors shard.

    Input:
        shard_path: Path to a .safetensors file.
    Output: The shard's tensor names, or an empty list when the shard cannot be
        read or safetensors is not installed.
    """
    try:
        from safetensors import safe_open
    except ImportError:
        return []
    try:
        with safe_open(str(shard_path), framework="pt") as handle:
            names = handle.keys()
            return list(names)
    except Exception:
        return []


def _read_tensor(shard_path: Path, tensor_name: str) -> torch.Tensor | None:
    """Read one tensor out of a safetensors shard.

    Input:
        shard_path: Path to a .safetensors file.
        tensor_name: Name of the tensor to read.
    Output: The tensor, or None when the shard is unreadable, safetensors is not
        installed, or the shard holds no tensor of that name.
    """
    try:
        from safetensors import safe_open
    except ImportError:
        return None
    try:
        with safe_open(str(shard_path), framework="pt") as handle:
            tensor: torch.Tensor = handle.get_tensor(tensor_name)
            return tensor
    except Exception:
        return None


def sample_safetensors_weight_for_effective_rank(
    model_path: str | Path,
    tensor_names: list[str],
    max_rows: int = 256,
    seed: int = 42,
    device: str = "cpu",
) -> list[tuple[str, torch.Tensor, tuple[int, int]]]:
    """Sample deterministic rows from safetensors weight tensors for R-03 effective rank.

    Reads one tensor at a time (never the whole checkpoint). Each returned weight
    is a 2-D float32 tensor of shape [sampled_rows, cols]; parent_shape is the full
    [rows, cols] tensor shape used for coverage computation.

    Input:
        model_path: Path to a checkpoint directory, or to a parent of one.
        tensor_names: Full tensor names to sample (e.g.
            ["model.layers.0.mlp.gate_proj.weight"]).
        max_rows: Upper bound on sampled rows per tensor (default 256).
        seed: Deterministic RNG seed (same seed -> same rows).
        device: Device the sampled rows are returned on.
    Output:
        List of (layer_name, weight_tensor, parent_shape) tuples, one per tensor
        found. Tensors that are absent, unreadable, or not 2-D are skipped.
    Invariants:
        - Output is 2-D float32.
        - Same (model_path, tensor_name, max_rows, seed) -> same rows.
        - The column dimension is never reduced; only rows are sampled.
    """
    directory = find_checkpoint_dir(model_path)
    if directory is None:
        return []

    weight_map = safetensors_weight_map(directory)
    if not weight_map:
        return []

    results: list[tuple[str, torch.Tensor, tuple[int, int]]] = []
    for tensor_name in tensor_names:
        shard = weight_map.get(tensor_name)
        if shard is None:
            continue
        tensor = _read_tensor(directory / shard, tensor_name)
        if tensor is None:
            continue
        if tensor.ndim != 2:
            continue
        rows = int(tensor.shape[0])
        cols = int(tensor.shape[1])
        if rows <= 0 or cols <= 0:
            continue

        n_take = min(int(max_rows), rows)
        if n_take <= 0:
            continue
        rng = np.random.default_rng(seed)
        candidates = np.arange(rows, dtype=np.int64)
        chosen = rng.choice(candidates, size=n_take, replace=False)
        chosen = np.sort(chosen)
        index = torch.from_numpy(chosen)
        sample = tensor.index_select(0, index).to(device=device, dtype=torch.float32)
        results.append((tensor_name, sample, (rows, cols)))

    return results
