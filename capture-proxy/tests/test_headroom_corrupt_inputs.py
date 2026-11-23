"""Corrupt-input and loader-failure tests for the headroom capture path.

Covers the swallow-all safety nets that a green suite previously sat next
to: a corrupt safetensors shard whose header is unreadable, and a
TensorLoader that raises. Both must degrade to a named note / None return,
never propagate out of the sidecar writer.

Requires torch + safetensors (the ``headroom`` dependency group); the
module skips cleanly without them. The ``stack`` fixture purges
safetensors modules on teardown so the off-mode purity test in
test_headroom.py stays green regardless of execution order.
"""

from __future__ import annotations

import asyncio
import json
import sys
import uuid
from typing import TYPE_CHECKING, Any

import pytest

from capture_proxy.headroom import HeadroomWriter, TensorBundle, _shard_tensor_keys

if TYPE_CHECKING:
    from pathlib import Path


def _require_stack() -> tuple[Any, Any]:
    """Import torch + safetensors.torch, skipping when the group is absent.

    Imported lazily — never at module scope. Collection must not pull the
    tensor stack into sys.modules: the off-mode purity test guards exactly
    that, and pytest imports every collected module before running any
    test.
    """
    torch = pytest.importorskip("torch")
    st_torch = pytest.importorskip("safetensors.torch")
    return torch, st_torch


@pytest.fixture()
def stack() -> tuple[Any, Any]:
    """Provide (torch, safetensors.torch), then purge safetensors modules.

    The teardown restores the runner to a safetensors-free state so the
    off-mode purity test passes regardless of execution order.
    """
    torch, st_torch = _require_stack()
    yield torch, st_torch
    for name in [
        m for m in sys.modules if m == "safetensors" or m.startswith("safetensors.")
    ]:
        del sys.modules[name]


def _make_model_dir(tmp_path: Path) -> Path:
    """Create a fake model dir: config.json + safetensors index (1 tensor)."""
    d = tmp_path / "model"
    d.mkdir()
    (d / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen3_5",
                "architectures": ["Qwen3_5ForConditionalGeneration"],
            }
        )
    )
    name = "model.layers.0.mlp.gate_proj.weight"
    (d / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {name: "model-00001-of-00002.safetensors"}})
    )
    return d


class TestShardKeysCorrupt:
    """H4: _shard_tensor_keys swallows Exception -> None for a corrupt shard."""

    def test_shard_keys_corrupt_shard_returns_none(self, tmp_path, stack):
        """A directory with one good + one truncated .safetensors shard:
        _shard_tensor_keys must return None for the truncated one.

        The swallow-all at headroom.py:370-374 is the loader-contract
        safety net: a corrupt shard must yield None, never propagate.
        """
        torch, st_torch = stack
        d = tmp_path / "shards"
        d.mkdir()
        good = d / "good.safetensors"
        corrupt = d / "corrupt.safetensors"
        st_torch.save_file(
            {"model.layers.0.mlp.gate_proj.weight": torch.zeros(2, 2)},
            str(good),
        )
        st_torch.save_file(
            {"model.layers.0.mlp.up_proj.weight": torch.zeros(2, 2)},
            str(corrupt),
        )
        # Truncate the second shard to ~1/3 of its bytes so the header
        # read fails inside safe_open.
        data = corrupt.read_bytes()
        corrupt.write_bytes(data[: len(data) // 3])

        assert _shard_tensor_keys(good) is not None
        assert _shard_tensor_keys(corrupt) is None


class TestTensorLoaderRaises:
    """H5: a raising TensorLoader is caught and noted, never propagated."""

    def test_tensor_loader_raises_is_caught_as_unreadable_note(self, tmp_path):
        """Stub the loader raising RuntimeError; assert the sidecar note
        says the tensor was unreadable and no exception propagates.

        The swallow-all at headroom.py:1102-1106 (and the twin at
        :1251-1259 for the specialization path) is the loader-contract
        safety net: a raising loader must degrade to a named note, not
        crash the sidecar writer.
        """

        def _raising_loader(model_path: Path, tensor_name: str) -> TensorBundle:
            raise RuntimeError(f"stub loader failure for {tensor_name}")

        model_dir = _make_model_dir(tmp_path)
        w = HeadroomWriter(
            tmp_path / "h",
            weights="gate",
            signals=frozenset({"effective_rank"}),
            tensor_loader=_raising_loader,
        )
        capture_id = str(uuid.uuid4())
        out = asyncio.run(
            w.write_async(
                capture_id=capture_id,
                parent_raw=tmp_path / "raw.jsonl",
                model="m",
                model_path=model_dir,
                created_at="2026-08-21T19:40:00+00:00",
            )
        )
        assert out is not None
        assert out.name == f"{capture_id}.headroom.json"
        sidecar = json.loads(out.read_text())
        # The loader raised; the note names the tensor as unreadable and
        # the sidecar was still written (no exception propagated).
        assert sidecar["error_body"] is not None
        assert "unreadable" in sidecar["error_body"]
        assert sidecar["signals"]["effective_rank"] is None
