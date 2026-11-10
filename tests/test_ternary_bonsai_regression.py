"""Skip-if-absent regression test for the ternary functions against the real
Ternary-Bonsai-27B GGUF (legacy Prism Q2_0, group 128 stored as type id 42).

The model file (~6.7 GiB) is not part of the repo, so every test is skipped
when it is absent. When present, this asserts the loader's fallback inference
resolves PQ2_0 unambiguously for all 498 type-42 tensors, buffer validation
passes, dequantization round-trips the stored fp16 block scale exactly (trit
error 0.0), and deterministic row sampling is reproducible.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from model_headroom.models.gguf_loader import (
    _dequantize_gguf_tensor,
    _tensor_qtype_name,
    extract_gguf_weight_sample,
    open_gguf_reader,
)
from model_headroom.models.ternary_quants import (
    dequantize_ternary,
    prism_quant_size,
)

_MODEL_PATH = Path(
    "/kaggle/input/models/enagic/ternary-bonsai-27b/gguf/default/1/"
    "Ternary-Bonsai-27B-Q2_0.gguf"
)

skip_model_absent = pytest.mark.skipif(
    not _MODEL_PATH.exists(), reason="real Ternary-Bonsai-27B GGUF not present"
)

_BLOCK, _TYPE_SIZE = prism_quant_size("PQ2_0")  # (128, 34)


@skip_model_absent
def test_reader_is_fallback_and_resolves_pq2_0() -> None:
    """Installed gguf tops out at type id 41 → fallback reader; infers PQ2_0
    for all 498 unknown tensors, unambiguously."""
    reader = open_gguf_reader(_MODEL_PATH)
    assert len(reader.tensors) == 851

    qnames = Counter(_tensor_qtype_name(t) for t in reader.tensors)
    assert qnames == {"PQ2_0": 498, "F32": 353}

    pq = [t for t in reader.tensors if _tensor_qtype_name(t) == "PQ2_0"]
    assert len(pq) == 498


@skip_model_absent
def test_buffer_validation_all_pq2_0_tensors() -> None:
    """Every type-42 tensor satisfies the PQ2_0 byte-length contract."""
    reader = open_gguf_reader(_MODEL_PATH)
    failures = 0
    for t in reader.tensors:
        if _tensor_qtype_name(t) != "PQ2_0":
            continue
        rows, cols = int(t.shape[0]), int(t.shape[1])
        size = np.ascontiguousarray(t.data).size
        if not (
            cols % _BLOCK == 0
            and rows * cols % _BLOCK == 0
            and size % _TYPE_SIZE == 0
            and size // _TYPE_SIZE == rows * cols // _BLOCK
        ):
            failures += 1
    assert failures == 0


@skip_model_absent
def test_dequantize_matches_stored_scale_exactly() -> None:
    """Dequantized max|value| per block equals the stored fp16 scale; trit
    reconstruction error is 0.0; direct adapter == loader output."""
    reader = open_gguf_reader(_MODEL_PATH)
    by_name = {t.name: t for t in reader.tensors}

    for name in (
        "blk.0.ssm_alpha.weight",
        "blk.0.attn_qkv.weight",
        "blk.0.ssm_out.weight",
        "blk.0.ffn_gate.weight",
        "blk.31.ffn_down.weight",
    ):
        t = by_name[name]
        rows, cols = int(t.shape[0]), int(t.shape[1])
        raw = np.ascontiguousarray(t.data).reshape(-1)

        blocks = raw.reshape((-1, _TYPE_SIZE))
        scales = blocks[:, :2].copy().view(np.float16).astype(np.float32).reshape(-1)

        out = _dequantize_gguf_tensor(t, device="cpu")
        assert out is not None
        arr = out.numpy()
        assert np.isfinite(arr).all()

        decoded = arr.reshape(-1, _BLOCK)
        absmax = np.abs(decoded).max(axis=1)
        assert np.allclose(absmax, np.abs(scales), rtol=0, atol=1e-6), name

        nz = scales != 0
        ratios = np.zeros_like(decoded)
        ratios[nz] = decoded[nz] / scales[nz, None]
        nearest = np.rint(ratios)
        assert np.isin(nearest, (-1.0, 0.0, 1.0)).all(), name
        assert float(np.abs(ratios - nearest).max()) < 1e-3, name

        direct = dequantize_ternary(raw.reshape((-1, _TYPE_SIZE)), "PQ2_0")
        assert np.array_equal(direct.reshape(rows, cols), arr), name


@skip_model_absent
def test_deterministic_sampling_reproducible() -> None:
    """extract_gguf_weight_sample is deterministic for a fixed seed."""
    a = extract_gguf_weight_sample(
        _MODEL_PATH, "blk.0.ffn_gate.weight", max_rows=64, seed=7
    )
    b = extract_gguf_weight_sample(
        _MODEL_PATH, "blk.0.ffn_gate.weight", max_rows=64, seed=7
    )
    assert a is not None and b is not None
    assert np.array_equal(a.numpy(), b.numpy())
