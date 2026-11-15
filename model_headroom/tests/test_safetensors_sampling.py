"""Tests for the safetensors weight sampler used by the R-03 effective-rank path.

GAP-1 (2026-11-12 Ornith handoff): the shipped CLI could only sample GGUF
weights, so R-03 was always skipped for a HuggingFace/safetensors checkpoint.
These tests pin the sampler contract the GGUF sampler already had — seeded row
selection, full column dimension, capped row count, and the full parent shape.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from pathlib import Path


def _write_checkpoint(
    directory: Path, tensors: dict[str, torch.Tensor], with_index: bool = False
) -> Path:
    """Write a safetensors checkpoint, optionally with a shard index."""
    from safetensors.torch import save_file

    directory.mkdir(parents=True, exist_ok=True)
    if with_index:
        shard = "model-00001-of-00002.safetensors"
        save_file(tensors, str(directory / shard))
        (directory / "model.safetensors.index.json").write_text(
            json.dumps(
                {
                    "metadata": {"total_size": 1},
                    "weight_map": dict.fromkeys(tensors, shard),
                }
            )
        )
    else:
        save_file(tensors, str(directory / "model.safetensors"))
    return directory


class TestSafetensorsWeightMap:
    def test_reads_index_file(self, tmp_path: Path) -> None:
        from model_headroom.models.safetensors_sampling import safetensors_weight_map

        _write_checkpoint(
            tmp_path / "ckpt",
            {"model.layers.0.mlp.gate_proj.weight": torch.ones(4, 2)},
            with_index=True,
        )
        weight_map = safetensors_weight_map(tmp_path / "ckpt")
        assert weight_map == {
            "model.layers.0.mlp.gate_proj.weight": "model-00001-of-00002.safetensors"
        }

    def test_falls_back_to_loose_shards(self, tmp_path: Path) -> None:
        from model_headroom.models.safetensors_sampling import safetensors_weight_map

        _write_checkpoint(
            tmp_path / "ckpt", {"model.layers.0.mlp.gate_proj.weight": torch.ones(4, 2)}
        )
        weight_map = safetensors_weight_map(tmp_path / "ckpt")
        assert weight_map == {
            "model.layers.0.mlp.gate_proj.weight": "model.safetensors"
        }

    def test_missing_directory_is_empty(self, tmp_path: Path) -> None:
        from model_headroom.models.safetensors_sampling import safetensors_weight_map

        assert safetensors_weight_map(tmp_path / "nope") == {}

    def test_directory_without_safetensors_is_empty(self, tmp_path: Path) -> None:
        from model_headroom.models.safetensors_sampling import safetensors_weight_map

        (tmp_path / "empty").mkdir()
        assert safetensors_weight_map(tmp_path / "empty") == {}


class TestFindCheckpointDir:
    def test_prefers_the_shallowest_match(self, tmp_path: Path) -> None:
        from model_headroom.models.safetensors_sampling import find_checkpoint_dir

        _write_checkpoint(
            tmp_path / "outer" / "inner",
            {"model.layers.0.mlp.gate_proj.weight": torch.ones(2, 2)},
        )
        found = find_checkpoint_dir(tmp_path / "outer")
        assert found == tmp_path / "outer" / "inner"

    def test_none_when_no_checkpoint(self, tmp_path: Path) -> None:
        from model_headroom.models.safetensors_sampling import find_checkpoint_dir

        (tmp_path / "empty").mkdir()
        assert find_checkpoint_dir(tmp_path / "empty") is None


class TestSampleSafetensorsWeight:
    def _checkpoint(self, tmp_path: Path) -> Path:
        generator = torch.Generator().manual_seed(7)
        weight = torch.randn(64, 12, generator=generator)
        return _write_checkpoint(
            tmp_path / "ckpt", {"model.layers.0.mlp.gate_proj.weight": weight}
        )

    def test_returns_parent_shape_and_capped_rows(self, tmp_path: Path) -> None:
        from model_headroom.models.safetensors_sampling import (
            sample_safetensors_weight_for_effective_rank,
        )

        checkpoint = self._checkpoint(tmp_path)
        results = sample_safetensors_weight_for_effective_rank(
            checkpoint, ["model.layers.0.mlp.gate_proj.weight"], max_rows=8
        )
        assert len(results) == 1
        name, sample, parent_shape = results[0]
        assert name == "model.layers.0.mlp.gate_proj.weight"
        assert parent_shape == (64, 12)
        assert tuple(sample.shape) == (8, 12)
        assert sample.dtype == torch.float32

    def test_preserves_the_full_column_dimension(self, tmp_path: Path) -> None:
        from model_headroom.models.safetensors_sampling import (
            sample_safetensors_weight_for_effective_rank,
        )

        checkpoint = self._checkpoint(tmp_path)
        _, sample, parent_shape = sample_safetensors_weight_for_effective_rank(
            checkpoint, ["model.layers.0.mlp.gate_proj.weight"], max_rows=4
        )[0]
        assert sample.shape[1] == parent_shape[1]

    def test_rows_below_the_cap_are_not_truncated(self, tmp_path: Path) -> None:
        from model_headroom.models.safetensors_sampling import (
            sample_safetensors_weight_for_effective_rank,
        )

        checkpoint = self._checkpoint(tmp_path)
        _, sample, parent_shape = sample_safetensors_weight_for_effective_rank(
            checkpoint, ["model.layers.0.mlp.gate_proj.weight"], max_rows=256
        )[0]
        assert tuple(sample.shape) == parent_shape

    def test_same_seed_selects_the_same_rows(self, tmp_path: Path) -> None:
        from model_headroom.models.safetensors_sampling import (
            sample_safetensors_weight_for_effective_rank,
        )

        checkpoint = self._checkpoint(tmp_path)
        first = sample_safetensors_weight_for_effective_rank(
            checkpoint, ["model.layers.0.mlp.gate_proj.weight"], max_rows=8, seed=42
        )[0][1]
        second = sample_safetensors_weight_for_effective_rank(
            checkpoint, ["model.layers.0.mlp.gate_proj.weight"], max_rows=8, seed=42
        )[0][1]
        assert torch.equal(first, second)

    def test_different_seed_selects_different_rows(self, tmp_path: Path) -> None:
        from model_headroom.models.safetensors_sampling import (
            sample_safetensors_weight_for_effective_rank,
        )

        checkpoint = self._checkpoint(tmp_path)
        first = sample_safetensors_weight_for_effective_rank(
            checkpoint, ["model.layers.0.mlp.gate_proj.weight"], max_rows=8, seed=42
        )[0][1]
        second = sample_safetensors_weight_for_effective_rank(
            checkpoint, ["model.layers.0.mlp.gate_proj.weight"], max_rows=8, seed=43
        )[0][1]
        assert not torch.equal(first, second)

    def test_sampled_rows_match_the_source_tensor(self, tmp_path: Path) -> None:
        from model_headroom.models.safetensors_sampling import (
            sample_safetensors_weight_for_effective_rank,
        )

        generator = torch.Generator().manual_seed(7)
        weight = torch.randn(64, 12, generator=generator)
        checkpoint = _write_checkpoint(
            tmp_path / "ckpt", {"model.layers.0.mlp.gate_proj.weight": weight}
        )
        sample = sample_safetensors_weight_for_effective_rank(
            checkpoint, ["model.layers.0.mlp.gate_proj.weight"], max_rows=8, seed=42
        )[0][1]
        for row in sample:
            assert any(torch.equal(row, source) for source in weight)

    def test_unknown_tensor_is_skipped(self, tmp_path: Path) -> None:
        from model_headroom.models.safetensors_sampling import (
            sample_safetensors_weight_for_effective_rank,
        )

        checkpoint = self._checkpoint(tmp_path)
        assert (
            sample_safetensors_weight_for_effective_rank(
                checkpoint, ["model.layers.9.mlp.gate_proj.weight"]
            )
            == []
        )

    def test_non_2d_tensor_is_skipped(self, tmp_path: Path) -> None:
        from model_headroom.models.safetensors_sampling import (
            sample_safetensors_weight_for_effective_rank,
        )

        checkpoint = _write_checkpoint(
            tmp_path / "ckpt", {"model.layers.0.norm.weight": torch.ones(12)}
        )
        assert (
            sample_safetensors_weight_for_effective_rank(
                checkpoint, ["model.layers.0.norm.weight"]
            )
            == []
        )

    def test_missing_checkpoint_returns_empty(self, tmp_path: Path) -> None:
        from model_headroom.models.safetensors_sampling import (
            sample_safetensors_weight_for_effective_rank,
        )

        assert (
            sample_safetensors_weight_for_effective_rank(
                tmp_path / "nope", ["model.layers.0.mlp.gate_proj.weight"]
            )
            == []
        )


class TestSafetensorsSamplerWithoutDependency:
    def test_import_error_is_reported_not_raised(
        self, tmp_path: Path, monkeypatch: Any
    ) -> None:
        import sys

        from model_headroom.models import safetensors_sampling

        monkeypatch.setitem(sys.modules, "safetensors", None)
        assert (
            safetensors_sampling.sample_safetensors_weight_for_effective_rank(
                tmp_path, ["model.layers.0.mlp.gate_proj.weight"]
            )
            == []
        )
