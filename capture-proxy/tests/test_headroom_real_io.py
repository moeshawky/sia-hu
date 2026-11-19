"""Real-I/O tests for the weight-capture path (DEFER-6).

Every other weight-capture test stubs the TensorLoader, which is how a
100%-green suite sat next to a dead feature. These tests write a tiny real
.safetensors file to tmp_path and run the real default loader against it —
no checkpoint needed. Requires torch + safetensors (the `headroom`
dependency group: `uv sync --group headroom`); the module skips cleanly
without them, and the official bare `uv run pytest` chain stays green.
"""

from __future__ import annotations

import json
import sys
import uuid
from pathlib import Path
from typing import Any

import pytest

from capture_proxy.headroom import (
    HeadroomWriter,
    _default_tensor_loader,
    _default_tensor_loader_all,
    _expert_index_from_name,
    _read_weight_map,
    _shard_tensor_keys,
)


def _require_stack() -> tuple[Any, Any]:
    """Import torch + safetensors.torch, skipping when the headroom group
    is absent.

    Imported lazily — never at module scope. Collection must not pull the
    tensor stack into sys.modules: the off-mode purity test guards exactly
    that, and pytest imports every collected module before running any
    test. A module-level import here would fail that test suite-wide.
    """
    torch = pytest.importorskip("torch")
    st_torch = pytest.importorskip("safetensors.torch")
    return torch, st_torch


@pytest.fixture()
def stack() -> tuple[Any, Any]:
    """Provide (torch, safetensors.torch), then purge safetensors modules.

    The teardown restores the runner to a safetensors-free state so the
    off-mode purity test passes regardless of execution order: it asserts
    on the runner's sys.modules, and a skipped-or-finished real-I/O test
    must leave no trace of the stack behind. (torch itself stays — the
    suite already tolerates it via the authority import.)
    """
    torch, st_torch = _require_stack()
    yield torch, st_torch
    for name in [
        m for m in sys.modules if m == "safetensors" or m.startswith("safetensors.")
    ]:
        del sys.modules[name]


GATE_2D = "model.layers.0.mlp.experts.0.gate_proj.weight"
FUSED_GATE_UP = "model.layers.0.mlp.experts.gate_up_proj.weight"
BIAS_1D = "model.layers.0.mlp.experts.0.bias"


def _write_dir(
    d: Path, torch: Any, st_torch: Any, tensors: dict[str, Any], config: dict[str, Any]
) -> Path:
    """Write config.json + one real shard + its index into d."""
    d.mkdir()
    (d / "config.json").write_text(json.dumps(config))
    st_torch.save_file(tensors, str(d / "model.safetensors"))
    (d / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": dict.fromkeys(tensors, "model.safetensors")})
    )
    return d


@pytest.fixture()
def tiny_moe_dir(tmp_path: Path, stack: tuple[Any, Any]) -> Path:
    """Model dir with one real shard: indexed 2-D + fused 3-D + 1-D."""
    torch, st_torch = stack
    gate = torch.arange(32, dtype=torch.float32).reshape(4, 8)
    fused = torch.arange(4 * 6 * 8, dtype=torch.float32).reshape(4, 6, 8)
    bias = torch.arange(8, dtype=torch.float32)
    return _write_dir(
        tmp_path / "tiny-moe",
        torch,
        st_torch,
        {GATE_2D: gate, FUSED_GATE_UP: fused, BIAS_1D: bias},
        {
            "model_type": "qwen3_moe",
            "num_experts": 4,
            "architectures": ["Qwen3MoeForCausalLM"],
        },
    )


class TestDefaultLoaderRealIO:
    def test_round_trips_values_shape_and_zeros(
        self, tmp_path: Path, tiny_moe_dir: Path, stack: tuple[Any, Any]
    ) -> None:
        torch, _ = stack
        d = tiny_moe_dir
        bundle = _default_tensor_loader(d, GATE_2D)
        assert bundle is not None
        assert (
            bundle.weight
            == torch.arange(32, dtype=torch.float32).reshape(4, 8).tolist()
        )
        assert bundle.gradient == [[0.0] * 8 for _ in range(4)]
        assert bundle.parent_shape == (4, 8)
        assert bundle.truncated is False

    def test_byte_budget_truncates_rows_pre_conversion(
        self, tmp_path: Path, tiny_moe_dir: Path, stack: tuple[Any, Any]
    ) -> None:
        torch, _ = stack
        d = tiny_moe_dir
        bundle = _default_tensor_loader(d, GATE_2D, byte_budget=2 * 8 * 4)
        assert bundle is not None
        assert len(bundle.weight) == 2
        assert (
            bundle.weight
            == torch.arange(16, dtype=torch.float32).reshape(2, 8).tolist()
        )
        assert bundle.parent_shape == (4, 8)
        assert bundle.truncated is True

    def test_missing_tensor_returns_none(
        self, tmp_path: Path, tiny_moe_dir: Path
    ) -> None:
        d = tiny_moe_dir
        assert _default_tensor_loader(d, "model.layers.0.mlp.nope.weight") is None

    def test_singular_loader_returns_first_fanout_bundle(
        self, tmp_path: Path, tiny_moe_dir: Path, stack: tuple[Any, Any]
    ) -> None:
        torch, _ = stack
        d = tiny_moe_dir
        bundle = _default_tensor_loader(d, FUSED_GATE_UP)
        assert bundle is not None
        assert (
            bundle.weight
            == torch.arange(48, dtype=torch.float32).reshape(6, 8).tolist()
        )
        assert bundle.parent_shape == (6, 8)


class TestWeightMapRealIO:
    def test_index_path_resolves(self, tmp_path: Path, tiny_moe_dir: Path) -> None:
        d = tiny_moe_dir
        assert _read_weight_map(d) == {
            GATE_2D: "model.safetensors",
            FUSED_GATE_UP: "model.safetensors",
            BIAS_1D: "model.safetensors",
        }

    def test_header_fallback_without_index(
        self, tmp_path: Path, tiny_moe_dir: Path
    ) -> None:
        d = tiny_moe_dir
        (d / "model.safetensors.index.json").unlink()
        found = _read_weight_map(d)
        assert found is not None
        assert set(found) == {GATE_2D, FUSED_GATE_UP, BIAS_1D}

    def test_shard_keys_real(self, tmp_path: Path, tiny_moe_dir: Path) -> None:
        d = tiny_moe_dir
        assert _shard_tensor_keys(d / "model.safetensors") == {
            GATE_2D,
            FUSED_GATE_UP,
            BIAS_1D,
        }


class TestFusedFanoutRealIO:
    def test_fused_3d_fans_out_per_expert(
        self, tmp_path: Path, tiny_moe_dir: Path, stack: tuple[Any, Any]
    ) -> None:
        torch, _ = stack
        d = tiny_moe_dir
        bundles, stopped = _default_tensor_loader_all(d, FUSED_GATE_UP)
        assert stopped is False
        assert [idx for idx, _ in bundles] == [0, 1, 2, 3]
        for i, (_, bundle) in enumerate(bundles):
            expected = torch.arange(4 * 6 * 8, dtype=torch.float32).reshape(4, 6, 8)[i]
            assert bundle.weight == expected.tolist()
            assert bundle.parent_shape == (6, 8)
            assert bundle.truncated is False

    def test_fanout_budget_stops_early_with_flag(
        self, tmp_path: Path, tiny_moe_dir: Path
    ) -> None:
        d = tiny_moe_dir
        budget = 6 * 8 * 4 + 2 * 8 * 4
        bundles, stopped = _default_tensor_loader_all(
            d, FUSED_GATE_UP, byte_budget=budget
        )
        assert stopped is True
        assert [idx for idx, _ in bundles] == [0, 1]
        assert bundles[0][1].truncated is False
        assert len(bundles[1][1].weight) == 2
        assert bundles[1][1].truncated is True

    def test_unsupported_rank_skips_without_raising(
        self, tmp_path: Path, tiny_moe_dir: Path
    ) -> None:
        d = tiny_moe_dir
        bundles, stopped = _default_tensor_loader_all(d, BIAS_1D)
        assert bundles == []
        assert stopped is False
        assert _default_tensor_loader(d, BIAS_1D) is None


class TestExpertIndexParsing:
    def test_indexed_expert_templates(self) -> None:
        assert _expert_index_from_name(GATE_2D) == 0
        assert (
            _expert_index_from_name("model.layers.7.mlp.experts.12.gate_proj.weight")
            == 12
        )
        assert (
            _expert_index_from_name(
                "model.layers.0.block_sparse_moe.experts.3.w1.weight"
            )
            == 3
        )

    def test_fused_and_dense_names_yield_none(self) -> None:
        assert _expert_index_from_name(FUSED_GATE_UP) is None
        assert (
            _expert_index_from_name("bert.encoder.layer.0.attention.self.query.weight")
            is None
        )


class TestCapturePairsRealIO:
    def test_indexed_pair_carries_expert_id(
        self, tmp_path: Path, tiny_moe_dir: Path
    ) -> None:
        d = tiny_moe_dir
        w = HeadroomWriter(tmp_path / "headroom", weights="gate")
        out = tmp_path / "pairs.json"
        pairs, _, _, _ = w._capture_weight_pairs(d, out)
        assert pairs
        first = pairs[0]
        assert first["layer_name"] == GATE_2D
        assert first["expert_id"] == 0

    def test_fused_only_dir_emits_per_expert_ids(
        self, tmp_path: Path, stack: tuple[Any, Any]
    ) -> None:
        torch, st_torch = stack
        fused = torch.arange(4 * 6 * 8, dtype=torch.float32).reshape(4, 6, 8)
        d = _write_dir(
            tmp_path / "fused-only",
            torch,
            st_torch,
            {FUSED_GATE_UP: fused},
            {"model_type": "mixtral", "num_local_experts": 8},
        )
        w = HeadroomWriter(tmp_path / "headroom", weights="gate")
        out = tmp_path / "pairs.json"
        pairs, _, _, _ = w._capture_weight_pairs(d, out)
        assert [p["expert_id"] for p in pairs] == [0, 1, 2, 3]
        assert all(p["layer_name"] == FUSED_GATE_UP for p in pairs)
        assert all(len(p["weight"]) == 6 and len(p["weight"][0]) == 8 for p in pairs)


class TestSpecializationArtifactRealIO:
    def test_fused_dir_emits_authority_readable_layers(
        self, tmp_path: Path, stack: tuple[Any, Any]
    ) -> None:
        torch, st_torch = stack
        fused = torch.arange(4 * 6 * 8, dtype=torch.float32).reshape(4, 6, 8)
        d = _write_dir(
            tmp_path / "fused-only",
            torch,
            st_torch,
            {FUSED_GATE_UP: fused},
            {"model_type": "mixtral", "num_local_experts": 8},
        )
        w = HeadroomWriter(tmp_path / "headroom", weights="gate")
        out = tmp_path / "ew.json"
        entries, used, trunc, _note = w._capture_expert_weights(
            d, out, budget_remaining=None
        )
        assert entries is not None and len(entries) == 1
        assert entries[0]["layer_name"] == FUSED_GATE_UP
        assert len(entries[0]["expert_weights"]) == 4
        assert trunc is False
        assert used == 4 * 6 * 8 * 4
        payload = json.loads(out.read_text())
        assert set(payload) == {"layers"}
        try:
            from model_headroom.models.loader import load_expert_weights_from_json
        except ImportError:
            pytest.skip("authority loader unavailable")
        loaded = load_expert_weights_from_json(str(out))
        assert loaded is not None and len(loaded) == 1
        assert len(loaded[0].expert_weights) == 4
        assert [tuple(t.shape) for t in loaded[0].expert_weights] == [(6, 8)] * 4

    def test_single_expert_yields_note_not_file(
        self, tmp_path: Path, tiny_moe_dir: Path
    ) -> None:
        d = tiny_moe_dir
        w = HeadroomWriter(tmp_path / "headroom", weights="gate")
        out = tmp_path / "ew.json"
        entries, _, _, note = w._capture_expert_weights(d, out, budget_remaining=None)
        assert entries == []
        assert note is not None and "fewer than 2 experts" in note
        assert not out.exists()

    def test_stub_loader_single_bundle_skips(
        self, tmp_path: Path, tiny_moe_dir: Path
    ) -> None:
        from capture_proxy.headroom import TensorBundle

        def stub(model_path: Path, tensor_name: str) -> TensorBundle | None:
            return TensorBundle(weight=[[1.0]], gradient=[[0.0]])

        w = HeadroomWriter(tmp_path / "headroom", weights="gate", tensor_loader=stub)
        out = tmp_path / "ew.json"
        entries, _, _, note = w._capture_expert_weights(
            tiny_moe_dir, out, budget_remaining=None
        )
        assert entries == []
        assert note is not None

    def test_budget_exhaustion_stops_with_note(
        self, tmp_path: Path, stack: tuple[Any, Any]
    ) -> None:
        torch, st_torch = stack
        fused = torch.arange(4 * 6 * 8, dtype=torch.float32).reshape(4, 6, 8)
        d = _write_dir(
            tmp_path / "fused-only",
            torch,
            st_torch,
            {FUSED_GATE_UP: fused},
            {"model_type": "mixtral", "num_local_experts": 8},
        )
        w = HeadroomWriter(tmp_path / "headroom", weights="gate")
        out = tmp_path / "ew.json"
        entries, _, trunc, note = w._capture_expert_weights(
            d, out, budget_remaining=100
        )
        assert entries == []
        assert trunc is True
        assert note is not None

    def test_planned_path_and_write_sync_agree(
        self, tmp_path: Path, stack: tuple[Any, Any]
    ) -> None:
        torch, st_torch = stack
        fused = torch.arange(4 * 6 * 8, dtype=torch.float32).reshape(4, 6, 8)
        d = _write_dir(
            tmp_path / "fused-only",
            torch,
            st_torch,
            {FUSED_GATE_UP: fused},
            {"model_type": "mixtral", "num_local_experts": 8},
        )
        w = HeadroomWriter(
            tmp_path / "headroom",
            weights="gate",
            signals=frozenset({"specialization"}),
        )
        capture_id = str(uuid.uuid4())
        _sidecar, weight, activation, ew = w.planned_artifacts(
            capture_id, "2026-10-12T00:00:00+00:00"
        )
        assert weight is None and activation is None
        assert ew is not None and ew.endswith("-expert_weights.json")
        out = w._write_sync(
            capture_id,
            tmp_path / "raw.jsonl",
            "m",
            d,
            "2026-10-12T00:00:00+00:00",
        )
        assert out is not None
        sidecar_payload = json.loads(out.read_text())
        assert sidecar_payload["signals"]["specialization"] == ew
        assert Path(ew).is_file()
