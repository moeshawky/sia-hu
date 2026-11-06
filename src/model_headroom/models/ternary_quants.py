"""Python adapters for packed ternary GGUF types reusing PrismML's existing Q2_0/TQ1_0 codec logic.

PQ2_0 = Q2_0 2-bit codec at group-128 block size.
PTQ1_0 = TQ1_0 base-3 trit packing at group-128 block size.

These adapters do NOT create independent codec implementations; they adapt Prism's existing
Q2_0 and TQ1_0 dequantize_blocks algorithms to the group-128 block size used by PQ2_0 and
PTQ1_0. Q2_0/TQ1_0/TQ2_0 at their native block sizes are handled by the installed gguf
package when available, with Prism-adapted fallbacks when the installed package lacks the
type (e.g. gguf 0.19.0 has no Q2_0/PQ2_0/PTQ1_0 constants).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

TernaryQType = Literal["PQ2_0", "PTQ1_0", "Q2_0", "TQ1_0", "TQ2_0"]

# Prism GGML_QUANT_SIZES for ternary types (block_size, type_size).
# Q2_0: group 64, type_size 2 + 16 = 18
# PQ2_0: group 128, type_size 2 + 32 = 34
# TQ1_0: group 256, type_size 2 + 4*13 = 54
# TQ2_0: group 256, type_size 2 + 64 = 66
# PTQ1_0: group 128, type_size 2 + 24 + 2 = 28
_PRISM_QUANT_SIZES: dict[str, tuple[int, int]] = {
    "Q2_0": (64, 18),
    "PQ2_0": (128, 34),
    "TQ1_0": (256, 54),
    "TQ2_0": (256, 66),
    "PTQ1_0": (128, 28),
}

_POW3_5: NDArray[np.uint16] = np.array([1, 3, 9, 27, 81], dtype=np.uint16)
_POW3_4: NDArray[np.uint16] = np.array([1, 3, 9, 27], dtype=np.uint16)
_BITS_2BIT: NDArray[np.uint8] = np.array([0, 2, 4, 6], dtype=np.uint8)

_U8_MASK_03: NDArray[np.uint8] = np.array([0x03], dtype=np.uint8)


class _TernaryAdapter:
    """Base class for ternary dequantization adapters reusing Prism codec logic."""

    qtype_name: str = ""
    block_size: int = 0
    type_size: int = 0

    @classmethod
    def dequantize_blocks(cls, blocks: NDArray[np.uint8]) -> NDArray[np.float32]:
        raise NotImplementedError


class Q2_0Adapter(_TernaryAdapter):
    """Q2_0 dequantization at native group-64 block size (Prism codec).

    Packs 4 two-bit trits per byte (bits 0-1, 2-3, 4-5, 6-7), each trit value in
    {-1, 0, 1} stored as {0, 1, 2, 3} minus 1. One fp16 scale `d` per block of 64.

    type_size = 2 (scale) + 64 / 4 (16 bytes of qs) = 18.
    """

    qtype_name = "Q2_0"
    block_size = 64
    type_size = 18

    @classmethod
    def dequantize_blocks(cls, blocks: NDArray[np.uint8]) -> NDArray[np.float32]:
        n_blocks = blocks.shape[0]
        d, qs = np.hsplit(blocks, [2])
        d = np.asarray(d, dtype=np.uint8).view(np.float16).astype(np.float32)
        qs = np.asarray(qs, dtype=np.uint8)
        packed = qs.reshape((n_blocks, -1, 1)) >> _BITS_2BIT
        qvals = (packed & _U8_MASK_03).reshape((n_blocks, -1)).astype(
            np.int8
        ) - np.int8(1)
        result = (d * qvals.astype(np.float32)).reshape((n_blocks, cls.block_size))
        return result.astype(np.float32, copy=False)


class PQ2_0Adapter(_TernaryAdapter):
    """PQ2_0 dequantization: Q2_0 codec at group-128 block size.

    Identical 2-bit packing to Q2_0 but one fp16 scale covers 128 weights instead of 64.

    type_size = 2 (scale) + 128 / 4 (32 bytes of qs) = 34.
    """

    qtype_name = "PQ2_0"
    block_size = 128
    type_size = 34

    @classmethod
    def dequantize_blocks(cls, blocks: NDArray[np.uint8]) -> NDArray[np.float32]:
        n_blocks = blocks.shape[0]
        d, qs = np.hsplit(blocks, [2])
        d = np.asarray(d, dtype=np.uint8).view(np.float16).astype(np.float32)
        qs = np.asarray(qs, dtype=np.uint8)
        packed = qs.reshape((n_blocks, -1, 1)) >> _BITS_2BIT
        qvals = (packed & _U8_MASK_03).reshape((n_blocks, -1)).astype(
            np.int8
        ) - np.int8(1)
        result = (d * qvals.astype(np.float32)).reshape((n_blocks, cls.block_size))
        return result.astype(np.float32, copy=False)


class TQ1_0Adapter(_TernaryAdapter):
    """TQ1_0 dequantization at native group-256 block size (Prism codec).

    Base-3 trit packing in groups of 5, ceiling-divided by 243 into bytes. Stages of
    32 + 16 qs bytes plus 4 qh bytes and one fp16 scale.

    type_size = 2 (scale) + 48 (qs) + 4 (qh) = 54.
    """

    qtype_name = "TQ1_0"
    block_size = 256
    type_size = 54

    @classmethod
    def dequantize_blocks(cls, blocks: NDArray[np.uint8]) -> NDArray[np.float32]:
        n_blocks = blocks.shape[0]
        qs_len = (cls.block_size - 4 * cls.block_size // 64) // 5
        qs, rest = np.hsplit(blocks, [qs_len])
        qh, d = np.hsplit(rest, [cls.block_size // 64])
        qs = np.asarray(qs, dtype=np.uint8)
        qh = np.asarray(qh, dtype=np.uint8)
        d = np.asarray(d, dtype=np.uint8).view(np.float16).astype(np.float32)
        qs0 = qs[..., :32]
        qs1 = qs[..., 32:]
        qs0 = qs0.reshape((n_blocks, -1, 1, 32)) * _POW3_5.reshape((1, 1, 5, 1))
        qs0 = qs0.reshape((n_blocks, -1))
        qs1 = qs1.reshape((n_blocks, -1, 1, 16)) * _POW3_5.reshape((1, 1, 5, 1))
        qs1 = qs1.reshape((n_blocks, -1))
        qh = qh.reshape((n_blocks, -1, 1, 4)) * _POW3_4.reshape((1, 1, 4, 1))
        qh = qh.reshape((n_blocks, -1))
        packed = np.concatenate([qs0, qs1, qh], axis=-1)
        qvals: NDArray[np.int8] = ((packed.astype(np.uint16) * 3) >> 8).astype(
            np.int8
        ) - np.int8(1)
        result = (d * qvals.astype(np.float32)).reshape((n_blocks, cls.block_size))
        return result.astype(np.float32, copy=False)


class TQ2_0Adapter(_TernaryAdapter):
    """TQ2_0 dequantization at native group-256 block size (Prism codec).

    Packs 4 two-bit trits per byte at group 256. One fp16 scale per block.

    type_size = 2 (scale) + 256 / 4 (64 bytes of qs) = 66.
    """

    qtype_name = "TQ2_0"
    block_size = 256
    type_size = 66

    @classmethod
    def dequantize_blocks(cls, blocks: NDArray[np.uint8]) -> NDArray[np.float32]:
        n_blocks = blocks.shape[0]
        qs, d = np.hsplit(blocks, [cls.block_size // 4])
        qs = np.asarray(qs, dtype=np.uint8)
        d = np.asarray(d, dtype=np.uint8).view(np.float16).astype(np.float32)
        packed = qs.reshape((n_blocks, -1, 1, 32)) >> np.array(
            [0, 2, 4, 6], dtype=np.uint8
        ).reshape((1, 1, 4, 1))
        qvals = (packed & np.uint8(0x03)).reshape((n_blocks, -1)).astype(
            np.int8
        ) - np.int8(1)
        result = (d * qvals.astype(np.float32)).reshape((n_blocks, cls.block_size))
        return result.astype(np.float32, copy=False)


class PTQ1_0Adapter(_TernaryAdapter):
    """PTQ1_0 dequantization: TQ1_0 base-3 codec at group-128 block size.

    Identical base-3 packing to TQ1_0 but qs split into 16 + 8 byte stages
    (24 qs bytes total) plus 2 qh bytes and one fp16 scale per 128 weights.

    type_size = 2 (scale) + 24 (qs stages) + 2 (qh) = 28.
    """

    qtype_name = "PTQ1_0"
    block_size = 128
    type_size = 28

    @classmethod
    def dequantize_blocks(cls, blocks: NDArray[np.uint8]) -> NDArray[np.float32]:
        n_blocks = blocks.shape[0]
        qs_len = (cls.block_size - 4 * cls.block_size // 64) // 5
        qs, rest = np.hsplit(blocks, [qs_len])
        qh, d = np.hsplit(rest, [cls.block_size // 64])
        qs = np.asarray(qs, dtype=np.uint8)
        qh = np.asarray(qh, dtype=np.uint8)
        d = np.asarray(d, dtype=np.uint8).view(np.float16).astype(np.float32)

        qs0 = qs[..., :16]
        qs1 = qs[..., 16:24]
        qs0 = qs0.reshape((n_blocks, -1, 1, 16)) * _POW3_5.reshape((1, 1, 5, 1))
        qs0 = qs0.reshape((n_blocks, -1))
        qs1 = qs1.reshape((n_blocks, -1, 1, 8)) * _POW3_5.reshape((1, 1, 5, 1))
        qs1 = qs1.reshape((n_blocks, -1))
        qh = qh.reshape((n_blocks, -1, 1, 2)) * _POW3_4.reshape((1, 1, 4, 1))
        qh = qh.reshape((n_blocks, -1))
        packed = np.concatenate([qs0, qs1, qh], axis=-1)
        qvals: NDArray[np.int8] = ((packed.astype(np.uint16) * 3) >> 8).astype(
            np.int8
        ) - np.int8(1)
        result = (d * qvals.astype(np.float32)).reshape((n_blocks, cls.block_size))
        return result.astype(np.float32, copy=False)


_TERNARY_ADAPTERS: dict[str, type[_TernaryAdapter]] = {
    "Q2_0": Q2_0Adapter,
    "PQ2_0": PQ2_0Adapter,
    "TQ1_0": TQ1_0Adapter,
    "TQ2_0": TQ2_0Adapter,
    "PTQ1_0": PTQ1_0Adapter,
}


def prism_quant_size(qtype_name: str) -> tuple[int, int] | None:
    """Return Prism's (block_size, type_size) for a ternary qtype name, or None if unknown."""
    return _PRISM_QUANT_SIZES.get(qtype_name)


def _gguf_dequantize(
    data: NDArray[np.uint8],
    qtype_name: str,
) -> NDArray[np.float32]:
    """Dequantize via the installed gguf package. Raises if the type is unavailable."""
    from gguf import GGMLQuantizationType
    from gguf import dequantize as gguf_dequantize

    gguf_qtype = getattr(GGMLQuantizationType, qtype_name)
    raw = gguf_dequantize(data.reshape(-1), gguf_qtype)
    result = np.asarray(raw, dtype=np.float32)
    prism_size = prism_quant_size(qtype_name)
    if prism_size is not None:
        block_size = prism_size[0]
        result = result.reshape((-1, block_size))
    return result


def dequantize_ternary(
    data: NDArray[np.uint8],
    qtype_name: str,
) -> NDArray[np.float32]:
    """Dispatch dequantization for ternary GGUF types.

    PQ2_0 and PTQ1_0 are decoded by the in-package adapters reusing Prism's Q2_0/TQ1_0
    codec logic at group-128 block size (the installed gguf package lacks these types).
    Q2_0, TQ1_0, TQ2_0 are delegated to the installed gguf package's dequantize when the
    type is available, falling back to Prism-adapted adapters otherwise.

    Input:
        data: uint8 byte buffer of quantized blocks, shape (n_blocks, type_size).
        qtype_name: one of "PQ2_0", "PTQ1_0", "Q2_0", "TQ1_0", "TQ2_0".
    Output: float32 dequantized weights, shape (n_blocks, block_size).
    Errors:
        ValueError if qtype_name is unknown or blocks are malformed
        (wrong ndim/type_size/size).
    """
    if qtype_name not in _TERNARY_ADAPTERS:
        raise ValueError(f"Unknown or unsupported ternary qtype: {qtype_name}")

    adapter = _TERNARY_ADAPTERS[qtype_name]
    _block_size, type_size = _PRISM_QUANT_SIZES[qtype_name]

    if data.ndim != 2 or data.shape[1] != type_size:
        raise ValueError(
            f"Expected 2-D blocks with type_size={type_size} for {qtype_name}, "
            f"got shape {data.shape}"
        )
    if data.size % type_size != 0:
        raise ValueError(
            f"{qtype_name} buffer size {data.size} is not a multiple of type_size "
            f"{type_size}"
        )

    blocks = data.reshape((-1, type_size))

    if qtype_name in ("PQ2_0", "PTQ1_0"):
        return adapter.dequantize_blocks(blocks)

    # Q2_0 / TQ1_0 / TQ2_0: prefer installed gguf, fall back to Prism adapter.
    try:
        return _gguf_dequantize(blocks, qtype_name)
    except (AttributeError, NotImplementedError):
        return adapter.dequantize_blocks(blocks)
