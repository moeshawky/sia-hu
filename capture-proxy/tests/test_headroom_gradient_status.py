"""Tests for the gradient_status contract on headroom weight-pair artifacts.

Covers: a bundle carrying a real gradient reports "captured" alongside it,
a bundle with no gradient reports "not_captured" with gradient None, and the
artifact key set is pinned exactly (7 base keys + gradient_status).
"""

import asyncio
import json
import uuid
from pathlib import Path

import pytest

from capture_proxy.headroom import HeadroomWriter, TensorBundle
from capture_proxy.names import HEADROOM_SIGNALS


def _make_model_dir(tmp_path: Path, prefix: str = "model") -> Path:
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
    name = f"{prefix}.layers.0.mlp.gate_proj.weight"
    (d / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {name: "model-00001-of-00002.safetensors"}})
    )
    return d


def _fake_loader_with_gradient(rows: int = 2, cols: int = 3):
    """Loader returning a weight plus a matching real gradient."""

    def loader(model_path: Path, tensor_name: str) -> TensorBundle:
        weight = [[float(i * cols + j) for j in range(cols)] for i in range(rows)]
        gradient = [[float(i * cols + j + 1) for j in range(cols)] for i in range(rows)]
        return TensorBundle(weight=weight, gradient=gradient)

    return loader


def _fake_loader_no_gradient(rows: int = 2, cols: int = 3):
    """Loader returning a weight with no gradient (gradient is None)."""

    def loader(model_path: Path, tensor_name: str) -> TensorBundle:
        weight = [[float(i * cols + j) for j in range(cols)] for i in range(rows)]
        return TensorBundle(weight=weight, gradient=None)

    return loader


_EXPECTED_PAIR_KEYS = {
    "gradient",
    "gradient_status",
    "weight",
    "layer_name",
    "expert_id",
    "parent_shape",
    "truncated",
    "sampling_mode",
}


def test_real_gradient_reports_captured(tmp_path):
    """A bundle carrying a real gradient writes it with status "captured".

    This is the branch the hardcoded status hid: with gradient_status pinned
    to "not_captured" the artifact contradicted the value beside it. The test
    fails if the writer ever hardcodes the status again.
    """
    model_dir = _make_model_dir(tmp_path)
    w = HeadroomWriter(
        tmp_path / "h",
        weights="gate",
        signals=frozenset(HEADROOM_SIGNALS),
        tensor_loader=_fake_loader_with_gradient(rows=2, cols=3),
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
    sidecar = json.loads(out.read_text())
    pairs = json.loads(Path(sidecar["signals"]["effective_rank"]).read_text())["pairs"]
    assert len(pairs) == 1
    pair = pairs[0]
    assert pair["gradient"] == [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    assert pair["gradient_status"] == "captured"


def test_absent_gradient_reports_not_captured(tmp_path):
    """A bundle with no gradient writes gradient None with status "not_captured"."""
    model_dir = _make_model_dir(tmp_path)
    w = HeadroomWriter(
        tmp_path / "h",
        weights="gate",
        signals=frozenset(HEADROOM_SIGNALS),
        tensor_loader=_fake_loader_no_gradient(rows=2, cols=3),
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
    sidecar = json.loads(out.read_text())
    pairs = json.loads(Path(sidecar["signals"]["effective_rank"]).read_text())["pairs"]
    assert len(pairs) == 1
    pair = pairs[0]
    assert pair["gradient"] is None
    assert pair["gradient_status"] == "not_captured"


def test_pair_key_set_is_pinned_exactly(tmp_path):
    """The artifact key set is exactly the documented 8 keys, no more, no less.

    Pinned in one place so the schema cannot drift by accident. Fails if a
    key is dropped, added, or made optional.
    """
    model_dir = _make_model_dir(tmp_path)
    w = HeadroomWriter(
        tmp_path / "h",
        weights="gate",
        signals=frozenset(HEADROOM_SIGNALS),
        tensor_loader=_fake_loader_no_gradient(rows=2, cols=3),
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
    sidecar = json.loads(out.read_text())
    pairs = json.loads(Path(sidecar["signals"]["effective_rank"]).read_text())["pairs"]
    assert len(pairs) == 1
    assert set(pairs[0].keys()) == _EXPECTED_PAIR_KEYS


@pytest.mark.parametrize("gradient", [None, [[1.0, 2.0], [3.0, 4.0]]])
def test_gradient_status_derives_from_value(tmp_path, gradient):
    """gradient_status is a function of the gradient value, never hardcoded.

    None -> "not_captured"; any real gradient -> "captured". The key is always
    present; only the value changes.
    """
    model_dir = _make_model_dir(tmp_path)

    def loader(model_path: Path, tensor_name: str) -> TensorBundle:
        weight = [[0.0, 0.0], [0.0, 0.0]]
        return TensorBundle(weight=weight, gradient=gradient)

    w = HeadroomWriter(
        tmp_path / "h",
        weights="gate",
        signals=frozenset(HEADROOM_SIGNALS),
        tensor_loader=loader,
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
    sidecar = json.loads(out.read_text())
    pairs = json.loads(Path(sidecar["signals"]["effective_rank"]).read_text())["pairs"]
    assert len(pairs) == 1
    pair = pairs[0]
    assert pair["gradient"] == gradient
    expected = "not_captured" if gradient is None else "captured"
    assert pair["gradient_status"] == expected
    assert "gradient_status" in pair
