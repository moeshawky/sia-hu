from __future__ import annotations

import importlib.util
from typing import Any

import numpy as np
import pytest
import torch

from model_headroom.models.gguf_loader import _GGUFLoadError
from model_headroom.models.ternary_sampling import sample_weight_for_effective_rank

_GGUF_AVAILABLE = importlib.util.find_spec("gguf") is not None
requires_gguf = pytest.mark.skipif(
    not _GGUF_AVAILABLE, reason="gguf extra not installed"
)


def _create_gguf_file(
    path: Any,
    arch: str = "mixtral",
    expert_count: int = 8,
    block_count: int = 32,
    intermediate_size: int = 14336,
    hidden_size: int = 4096,
    tensors: dict[str, tuple[list[int], list[float]]] | None = None,
) -> Any:
    from gguf import GGUFWriter

    gguf_path = path / "test_model.gguf"
    writer = GGUFWriter(str(gguf_path), arch)
    writer.add_architecture()
    writer.add_expert_count(expert_count)
    writer.add_block_count(block_count)
    writer.add_feed_forward_length(intermediate_size)
    writer.add_embedding_length(hidden_size)

    if tensors:
        for name, (shape, data) in tensors.items():
            arr = np.array(data, dtype=np.float32).reshape(shape)
            writer.add_tensor(name, arr)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    if tensors:
        writer.write_tensors_to_file()
    writer.close()
    return gguf_path


@requires_gguf
class TestSampleWeightForEffectiveRank:
    def test_success(self, tmp_path: Any) -> None:
        shape = [64, 128]
        data = [float(i) for i in range(64 * 128)]
        gguf_path = _create_gguf_file(
            tmp_path,
            tensors={"blk.0.ffn_gate_exp.0.weight": (shape, data)},
        )
        results = sample_weight_for_effective_rank(
            gguf_path, ["blk.0.ffn_gate_exp.0.weight"], max_rows=32, seed=42
        )
        assert len(results) == 1
        name, tensor, parent_shape = results[0]
        assert name == "blk.0.ffn_gate_exp.0.weight"
        assert tensor.ndim == 2
        assert tensor.dtype == torch.float32
        assert tensor.device.type == "cpu"
        assert tensor.shape[0] == 32
        assert tensor.shape[1] == parent_shape[1]
        assert parent_shape[0] * parent_shape[1] == 64 * 128

    def test_deterministic(self, tmp_path: Any) -> None:
        shape = [64, 128]
        data = [float(i) for i in range(64 * 128)]
        gguf_path = _create_gguf_file(
            tmp_path,
            tensors={"blk.0.ffn_gate_exp.0.weight": (shape, data)},
        )
        r1 = sample_weight_for_effective_rank(
            gguf_path, ["blk.0.ffn_gate_exp.0.weight"], max_rows=32, seed=42
        )
        r2 = sample_weight_for_effective_rank(
            gguf_path, ["blk.0.ffn_gate_exp.0.weight"], max_rows=32, seed=42
        )
        assert torch.equal(r1[0][1], r2[0][1])

    def test_different_seed_different_rows(self, tmp_path: Any) -> None:
        shape = [64, 128]
        data = [float(i) for i in range(64 * 128)]
        gguf_path = _create_gguf_file(
            tmp_path,
            tensors={"blk.0.ffn_gate_exp.0.weight": (shape, data)},
        )
        r1 = sample_weight_for_effective_rank(
            gguf_path, ["blk.0.ffn_gate_exp.0.weight"], max_rows=32, seed=42
        )
        r2 = sample_weight_for_effective_rank(
            gguf_path, ["blk.0.ffn_gate_exp.0.weight"], max_rows=32, seed=99
        )
        assert not torch.equal(r1[0][1], r2[0][1])

    def test_max_rows_limits(self, tmp_path: Any) -> None:
        shape = [64, 128]
        data = [float(i) for i in range(64 * 128)]
        gguf_path = _create_gguf_file(
            tmp_path,
            tensors={"blk.0.ffn_gate_exp.0.weight": (shape, data)},
        )
        results = sample_weight_for_effective_rank(
            gguf_path, ["blk.0.ffn_gate_exp.0.weight"], max_rows=10, seed=42
        )
        assert results[0][1].shape[0] == 10

    def test_tensor_not_found(self, tmp_path: Any) -> None:
        gguf_path = _create_gguf_file(tmp_path)
        results = sample_weight_for_effective_rank(
            gguf_path, ["nonexistent.weight"], max_rows=32, seed=42
        )
        assert results == []

    def test_file_not_found(self, tmp_path: Any) -> None:
        with pytest.raises(_GGUFLoadError):
            sample_weight_for_effective_rank(
                tmp_path / "missing.gguf", ["any.weight"], max_rows=32, seed=42
            )

    def test_invalid_file(self, tmp_path: Any) -> None:
        bad_file = tmp_path / "bad.gguf"
        bad_file.write_bytes(b"not gguf")
        with pytest.raises(_GGUFLoadError):
            sample_weight_for_effective_rank(
                bad_file, ["any.weight"], max_rows=32, seed=42
            )

    def test_multiple_tensors(self, tmp_path: Any) -> None:
        shape = [16, 32]
        data = [1.0] * (16 * 32)
        gguf_path = _create_gguf_file(
            tmp_path,
            tensors={
                "blk.0.ffn_gate_exp.0.weight": (shape, data),
                "blk.0.ffn_up_exp.0.weight": (shape, data),
            },
        )
        results = sample_weight_for_effective_rank(
            gguf_path,
            ["blk.0.ffn_gate_exp.0.weight", "blk.0.ffn_up_exp.0.weight"],
            max_rows=8,
            seed=42,
        )
        assert len(results) == 2
        assert results[0][0] == "blk.0.ffn_gate_exp.0.weight"
        assert results[1][0] == "blk.0.ffn_up_exp.0.weight"

    def test_returns_float32_cpu(self, tmp_path: Any) -> None:
        shape = [8, 16]
        data = [0.5] * (8 * 16)
        gguf_path = _create_gguf_file(
            tmp_path, tensors={"blk.0.ffn_gate_exp.0.weight": (shape, data)}
        )
        results = sample_weight_for_effective_rank(
            gguf_path, ["blk.0.ffn_gate_exp.0.weight"], max_rows=4, seed=42
        )
        assert results[0][1].dtype == torch.float32
        assert results[0][1].device.type == "cpu"
