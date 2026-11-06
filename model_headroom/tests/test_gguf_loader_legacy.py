"""Tests for the legacy GGUF fallback reader orientation contract."""

from __future__ import annotations

from pathlib import Path

import pytest

from model_headroom.models.gguf_loader import (
    _PRISM_QUANT_SIZES,
    _gguf_shape_from_dims,
    extract_gguf_weight_sample,
    open_gguf_reader,
)

_MODEL_PATH = Path(
    "/kaggle/input/models/enagic/ternary-bonsai-27b/gguf/default/1/"
    "Ternary-Bonsai-27B-Q2_0.gguf"
)

skip_model_absent = pytest.mark.skipif(
    not _MODEL_PATH.exists(), reason="real GGUF model not present"
)


def test_gguf_shape_from_dims_two_dim() -> None:
    assert _gguf_shape_from_dims((5120, 48)) == (48, 5120)


def test_gguf_shape_from_dims_large() -> None:
    assert _gguf_shape_from_dims((17408, 5120)) == (5120, 17408)


def test_gguf_shape_from_dims_returns_python_ints() -> None:
    result = _gguf_shape_from_dims((5120, 48))
    assert all(isinstance(v, int) for v in result)


def test_gguf_shape_from_dims_reverses_not_sorts() -> None:
    assert _gguf_shape_from_dims((3, 7, 11)) == (11, 7, 3)


@skip_model_absent
def test_real_model_every_quant_tensor_satisfies_block_contract() -> None:
    reader = open_gguf_reader(_MODEL_PATH)
    for tensor in reader.tensors:
        qtype_name = getattr(tensor, "tensor_type", None)
        name = getattr(qtype_name, "name", None)
        if name not in _PRISM_QUANT_SIZES:
            continue
        block_size, _type_size = _PRISM_QUANT_SIZES[name]
        assert int(tensor.shape[-1]) % block_size == 0, tensor.name


@skip_model_absent
def test_real_model_ssm_alpha_shape_and_sample() -> None:
    reader = open_gguf_reader(_MODEL_PATH)
    target = next(t for t in reader.tensors if t.name == "blk.0.ssm_alpha.weight")
    assert tuple(int(v) for v in target.shape) == (48, 5120)
    sample = extract_gguf_weight_sample(
        _MODEL_PATH, "blk.0.ssm_alpha.weight", max_rows=8
    )
    assert sample is not None
    assert sample.shape[1] == 5120


@skip_model_absent
def test_real_model_ffn_gate_shape() -> None:
    reader = open_gguf_reader(_MODEL_PATH)
    target = next(t for t in reader.tensors if t.name == "blk.0.ffn_gate.weight")
    assert tuple(int(v) for v in target.shape) == (17408, 5120)
