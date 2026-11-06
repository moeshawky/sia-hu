from __future__ import annotations

import numpy as np
import pytest

from model_headroom.models.ternary_quants import (
    PQ2_0Adapter,
    PTQ1_0Adapter,
    Q2_0Adapter,
    TQ1_0Adapter,
    TQ2_0Adapter,
    dequantize_ternary,
    prism_quant_size,
)


class TestPrismQuantSize:
    def test_q2_0(self) -> None:
        assert prism_quant_size("Q2_0") == (64, 18)

    def test_pq2_0(self) -> None:
        assert prism_quant_size("PQ2_0") == (128, 34)

    def test_tq1_0(self) -> None:
        assert prism_quant_size("TQ1_0") == (256, 54)

    def test_tq2_0(self) -> None:
        assert prism_quant_size("TQ2_0") == (256, 66)

    def test_ptq1_0(self) -> None:
        assert prism_quant_size("PTQ1_0") == (128, 28)

    def test_unknown(self) -> None:
        assert prism_quant_size("Q4_0") is None


class TestQ2_0Adapter:
    def test_dequantize_blocks(self) -> None:
        scale = np.array([0x00, 0x3C], dtype=np.uint8)
        qs = np.array([0x56] + [0x55] * 15, dtype=np.uint8)
        block = np.concatenate([scale, qs]).reshape(1, 18)
        result = Q2_0Adapter.dequantize_blocks(block)
        assert result.shape == (1, 64)
        assert result.dtype == np.float32
        assert result[0, 0] == pytest.approx(1.0)
        assert np.all(result[0, 1:] == 0.0)


class TestPQ2_0Adapter:
    def test_dequantize_blocks(self) -> None:
        scale = np.array([0x00, 0x3C], dtype=np.uint8)
        qs = np.array([0x56] + [0x55] * 31, dtype=np.uint8)
        block = np.concatenate([scale, qs]).reshape(1, 34)
        result = PQ2_0Adapter.dequantize_blocks(block)
        assert result.shape == (1, 128)
        assert result.dtype == np.float32
        assert result[0, 0] == pytest.approx(1.0)
        assert np.all(result[0, 1:] == 0.0)


class TestTQ2_0Adapter:
    def test_dequantize_blocks(self) -> None:
        qs = np.full(64, 0x55, dtype=np.uint8)  # every byte trit value 1 -> qval 0
        d = np.array([0x00, 0x3C], dtype=np.uint8)  # fp16 scale 1.0
        block = np.concatenate([qs, d]).reshape(1, 66)
        result = TQ2_0Adapter.dequantize_blocks(block)
        assert result.shape == (1, 256)
        assert result.dtype == np.float32
        assert np.all(result == 0.0)

        # Set position 0 trit to 2 -> qval +1
        qs2 = np.full(64, 0x55, dtype=np.uint8)
        qs2[0] = 0x02
        block2 = np.concatenate([qs2, d]).reshape(1, 66)
        result2 = TQ2_0Adapter.dequantize_blocks(block2)
        assert result2[0, 0] == pytest.approx(1.0)


class TestTQ1_0Adapter:
    def test_zero_block(self) -> None:
        block = np.zeros((1, 54), dtype=np.uint8)
        result = TQ1_0Adapter.dequantize_blocks(block)
        assert result.shape == (1, 256)
        assert result.dtype == np.float32
        assert np.all(result == 0.0)


class TestPTQ1_0Adapter:
    def test_zero_block(self) -> None:
        block = np.zeros((1, 28), dtype=np.uint8)
        result = PTQ1_0Adapter.dequantize_blocks(block)
        assert result.shape == (1, 128)
        assert result.dtype == np.float32
        assert np.all(result == 0.0)


class TestDequantizeTernaryDispatch:
    def test_q2_0(self) -> None:
        scale = np.array([0x00, 0x3C], dtype=np.uint8)
        qs = np.array([0x56] + [0x55] * 15, dtype=np.uint8)
        block = np.concatenate([scale, qs]).reshape(1, 18)
        result = dequantize_ternary(block, "Q2_0")
        assert result.shape == (1, 64)
        assert result[0, 0] == pytest.approx(1.0)

    def test_pq2_0(self) -> None:
        scale = np.array([0x00, 0x3C], dtype=np.uint8)
        qs = np.array([0x56] + [0x55] * 31, dtype=np.uint8)
        block = np.concatenate([scale, qs]).reshape(1, 34)
        result = dequantize_ternary(block, "PQ2_0")
        assert result.shape == (1, 128)
        assert result[0, 0] == pytest.approx(1.0)

    def test_tq1_0(self) -> None:
        block = np.zeros((1, 54), dtype=np.uint8)
        result = dequantize_ternary(block, "TQ1_0")
        assert result.shape == (1, 256)
        assert np.all(result == 0.0)

    def test_tq2_0(self) -> None:
        qs = np.full(64, 0x55, dtype=np.uint8)
        d = np.array([0x00, 0x3C], dtype=np.uint8)
        block = np.concatenate([qs, d]).reshape(1, 66)
        result = dequantize_ternary(block, "TQ2_0")
        assert result.shape == (1, 256)
        assert np.all(result == 0.0)

        qs2 = np.full(64, 0x55, dtype=np.uint8)
        qs2[0] = 0x02
        block2 = np.concatenate([qs2, d]).reshape(1, 66)
        result2 = dequantize_ternary(block2, "TQ2_0")
        assert result2[0, 0] == pytest.approx(1.0)

    def test_ptq1_0(self) -> None:
        block = np.zeros((1, 28), dtype=np.uint8)
        result = dequantize_ternary(block, "PTQ1_0")
        assert result.shape == (1, 128)
        assert np.all(result == 0.0)

    def test_unknown_qtype(self) -> None:
        block = np.zeros((1, 18), dtype=np.uint8)
        with pytest.raises(ValueError, match="Unknown or unsupported ternary qtype"):
            dequantize_ternary(block, "Q4_0")

    def test_wrong_ndim(self) -> None:
        block = np.zeros((18,), dtype=np.uint8)
        with pytest.raises(ValueError, match="Expected 2-D blocks"):
            dequantize_ternary(block, "Q2_0")

    def test_wrong_type_size(self) -> None:
        block = np.zeros((1, 17), dtype=np.uint8)
        with pytest.raises(ValueError, match="type_size=18"):
            dequantize_ternary(block, "Q2_0")

    def test_wrong_size(self) -> None:
        block = np.zeros((1, 36), dtype=np.uint8)
        with pytest.raises(ValueError, match="type_size=18"):
            dequantize_ternary(block, "Q2_0")


class TestMultiBlock:
    def test_q2_0_multiple_blocks(self) -> None:
        scale = np.array([0x00, 0x3C], dtype=np.uint8)
        qs = np.array([0x55] * 16, dtype=np.uint8)
        block = np.concatenate([scale, qs])
        data = np.tile(block, (3, 1))
        result = dequantize_ternary(data, "Q2_0")
        assert result.shape == (3, 64)
        assert np.all(result == 0.0)
