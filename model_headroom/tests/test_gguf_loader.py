from __future__ import annotations

import importlib.util
from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from model_headroom.models.gguf_loader import (
    _GGUFLoadError,
    _GGUFModelConfig,
    extract_expert_weight_from_gguf,
    extract_gguf_weight_sample,
    load_gguf_config,
)
from model_headroom.models.loader import _detect_model_format

if TYPE_CHECKING:
    from pathlib import Path

_GGUF_AVAILABLE = importlib.util.find_spec("gguf") is not None
requires_gguf = pytest.mark.skipif(
    not _GGUF_AVAILABLE, reason="gguf extra not installed"
)


def _create_gguf_file(
    path: Path,
    arch: str = "mixtral",
    expert_count: int = 8,
    block_count: int = 32,
    intermediate_size: int = 14336,
    hidden_size: int = 4096,
    tensors: dict[str, tuple[list[int], list[float]]] | None = None,
) -> Path:
    """Create a minimal GGUF file for testing.

    Input:
        path: Directory to create the file in.
        arch: Architecture string.
        expert_count, block_count, intermediate_size, hidden_size: Metadata.
        tensors: Optional dict of tensor_name -> (shape, flat_data).
    Output: Path to the created .gguf file.
    """
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


class TestGGUFModelConfig:
    def test_frozen_dataclass(self) -> None:
        config = _GGUFModelConfig(
            architecture="mixtral",
            expert_count=8,
            block_count=32,
            intermediate_size=14336,
            hidden_size=4096,
        )
        assert config.architecture == "mixtral"
        assert config.expert_count == 8

    def test_immutable(self) -> None:
        config = _GGUFModelConfig(
            architecture="mixtral",
            expert_count=8,
            block_count=32,
            intermediate_size=14336,
            hidden_size=4096,
        )
        with pytest.raises(AttributeError):
            config.architecture = "qwen2moe"  # type: ignore[misc]


class TestGGUFLoadError:
    def test_exception_fields(self) -> None:
        err = _GGUFLoadError(
            error_type="file_not_found",
            message="not here",
            recoverable=True,
        )
        assert err.error_type == "file_not_found"
        assert err.message == "not here"
        assert err.recoverable is True

    def test_is_exception(self) -> None:
        err = _GGUFLoadError(
            error_type="invalid_gguf",
            message="bad",
            recoverable=False,
        )
        assert isinstance(err, Exception)

    def test_frozen(self) -> None:
        err = _GGUFLoadError(
            error_type="missing_metadata",
            message="x",
            recoverable=False,
        )
        with pytest.raises(AttributeError):
            err.message = "y"  # type: ignore[misc]


@requires_gguf
class TestLoadGgufConfig:
    def test_success(self, tmp_path: Any) -> None:
        gguf_path = _create_gguf_file(tmp_path)
        config = load_gguf_config(gguf_path)
        assert config.architecture == "mixtral"
        assert config.expert_count == 8
        assert config.block_count == 32
        assert config.intermediate_size == 14336
        assert config.hidden_size == 4096

    def test_file_not_found(self, tmp_path: Any) -> None:
        with pytest.raises(_GGUFLoadError) as exc_info:
            load_gguf_config(tmp_path / "nonexistent.gguf")
        assert exc_info.value.error_type == "file_not_found"
        assert exc_info.value.recoverable is True

    def test_invalid_gguf_file(self, tmp_path: Any) -> None:
        bad_file = tmp_path / "bad.gguf"
        bad_file.write_bytes(b"not a gguf file")
        with pytest.raises(_GGUFLoadError) as exc_info:
            load_gguf_config(bad_file)
        assert exc_info.value.error_type == "invalid_gguf"

    def test_missing_metadata(self, tmp_path: Any) -> None:
        from gguf import GGUFWriter

        gguf_path = tmp_path / "no_meta.gguf"
        writer = GGUFWriter(str(gguf_path), "mixtral")
        # Don't add any metadata fields
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.close()
        with pytest.raises(_GGUFLoadError) as exc_info:
            load_gguf_config(gguf_path)
        assert exc_info.value.error_type == "missing_metadata"

    def test_qwen2moe_architecture(self, tmp_path: Any) -> None:
        gguf_path = _create_gguf_file(tmp_path, arch="qwen2moe", expert_count=4)
        config = load_gguf_config(gguf_path)
        assert config.architecture == "qwen2moe"
        assert config.expert_count == 4

    def test_qwen3_5_text_missing_expert_count_defaults_to_one(
        self, tmp_path: Any
    ) -> None:
        from gguf import GGUFWriter

        gguf_path = tmp_path / "dense.gguf"
        writer = GGUFWriter(str(gguf_path), "qwen3_5_text")
        writer.add_architecture()
        writer.add_block_count(32)
        writer.add_feed_forward_length(14336)
        writer.add_embedding_length(4096)
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.close()
        config = load_gguf_config(gguf_path)
        assert config.architecture == "qwen3_5_text"
        assert config.expert_count == 1


@requires_gguf
class TestExtractExpertWeightFromGguf:
    def test_success(self, tmp_path: Any) -> None:
        shape = [64, 128]
        data = list(range(64 * 128))
        gguf_path = _create_gguf_file(
            tmp_path,
            tensors={"blk.0.ffn_gate_exp.0.weight": (shape, [float(x) for x in data])},
        )
        result = extract_expert_weight_from_gguf(
            gguf_path, "blk.0.ffn_gate_exp.0.weight"
        )
        assert result is not None
        assert result.ndim == 2
        assert result.dtype == torch.float32
        assert result.numel() == 64 * 128

    def test_tensor_not_found(self, tmp_path: Any) -> None:
        gguf_path = _create_gguf_file(tmp_path)
        result = extract_expert_weight_from_gguf(gguf_path, "nonexistent.tensor.weight")
        assert result is None

    def test_file_not_found(self, tmp_path: Any) -> None:
        result = extract_expert_weight_from_gguf(
            tmp_path / "missing.gguf", "any.weight"
        )
        assert result is None

    def test_invalid_file(self, tmp_path: Any) -> None:
        bad_file = tmp_path / "bad.gguf"
        bad_file.write_bytes(b"not gguf")
        result = extract_expert_weight_from_gguf(bad_file, "any.weight")
        assert result is None

    def test_on_cpu_device(self, tmp_path: Any) -> None:
        gguf_path = _create_gguf_file(
            tmp_path,
            tensors={"blk.0.ffn_gate_exp.0.weight": ([4, 8], [1.0] * 32)},
        )
        result = extract_expert_weight_from_gguf(
            gguf_path, "blk.0.ffn_gate_exp.0.weight", device="cpu"
        )
        assert result is not None
        assert result.device.type == "cpu"


class TestDetectModelFormat:
    def test_gguf_file(self, tmp_path: Any) -> None:
        gguf_path = tmp_path / "model.gguf"
        gguf_path.write_bytes(b"")
        assert _detect_model_format(str(gguf_path)) == "gguf"

    def test_huggingface_directory(self, tmp_path: Any) -> None:
        (tmp_path / "config.json").write_text("{}")
        assert _detect_model_format(str(tmp_path)) == "huggingface"

    def test_unknown_format_raises(self, tmp_path: Any) -> None:
        with pytest.raises(ValueError, match="Cannot determine model format"):
            _detect_model_format(str(tmp_path / "unknown.bin"))

    def test_file_without_gguf_extension(self, tmp_path: Any) -> None:
        bad = tmp_path / "model.bin"
        bad.write_bytes(b"")
        with pytest.raises(ValueError, match="Cannot determine model format"):
            _detect_model_format(str(bad))


@requires_gguf
class TestExtractGGUFWeightSample:
    def _fake_reader(
        self,
        qtype_name: str,
        rows: int,
        cols: Any,
        tensor_name: str = "blk.0.ffn_gate.weight",
    ) -> MagicMock:
        from model_headroom.models.ternary_quants import prism_quant_size

        prism_size = prism_quant_size(qtype_name)
        if prism_size is not None:
            block_size, type_size = prism_size
            n_blocks = rows * (cols // block_size)
            data = np.zeros(n_blocks * type_size, dtype=np.uint8)
        else:
            block_size = 1
            data = np.zeros(rows * cols, dtype=np.float32).tobytes()
        tensor = MagicMock()
        qtype = MagicMock()
        qtype.name = qtype_name
        tensor.tensor_type = qtype
        tensor.name = tensor_name
        tensor.shape = (rows, cols)
        tensor.data = data
        reader = MagicMock()
        reader.tensors = [tensor]
        return reader

    def test_quantized_honors_max_rows_not_block_stride(self, tmp_path: Any) -> None:
        # Q2_0: block_size 64 over 1024 rows. Old stride -> 16 candidates; fix -> all rows.
        gguf_path = tmp_path / "model.gguf"
        gguf_path.write_bytes(b"fake")
        reader = self._fake_reader("Q2_0", 1024, 512)
        with patch(
            "model_headroom.models.gguf_loader.open_gguf_reader",
            return_value=reader,
        ):
            sample = extract_gguf_weight_sample(
                gguf_path, "blk.0.ffn_gate.weight", max_rows=256, seed=42
            )
        assert sample is not None
        assert sample.shape == (256, 512)
        assert sample.dtype == torch.float32

    def test_max_rows_larger_than_rows(self, tmp_path: Any) -> None:
        gguf_path = tmp_path / "model.gguf"
        gguf_path.write_bytes(b"fake")
        reader = self._fake_reader("Q2_0", 48, 512, "blk.0.ssm_alpha.weight")
        with patch(
            "model_headroom.models.gguf_loader.open_gguf_reader",
            return_value=reader,
        ):
            sample = extract_gguf_weight_sample(
                gguf_path, "blk.0.ssm_alpha.weight", max_rows=256, seed=42
            )
        assert sample is not None
        assert sample.shape == (48, 512)

    def test_max_rows_smaller_than_rows(self, tmp_path: Any) -> None:
        gguf_path = tmp_path / "model.gguf"
        gguf_path.write_bytes(b"fake")
        reader = self._fake_reader("Q2_0", 1024, 512)
        with patch(
            "model_headroom.models.gguf_loader.open_gguf_reader",
            return_value=reader,
        ):
            sample = extract_gguf_weight_sample(
                gguf_path, "blk.0.ffn_gate.weight", max_rows=40, seed=42
            )
        assert sample is not None
        assert sample.shape == (40, 512)

    def test_f32_path_unchanged(self, tmp_path: Any) -> None:
        gguf_path = tmp_path / "model.gguf"
        gguf_path.write_bytes(b"fake")
        reader = self._fake_reader("F32", 100, 50, "dense.weight")
        with patch(
            "model_headroom.models.gguf_loader.open_gguf_reader",
            return_value=reader,
        ):
            sample = extract_gguf_weight_sample(
                gguf_path, "dense.weight", max_rows=256, seed=42
            )
        assert sample is not None
        assert sample.shape == (100, 50)
        with patch(
            "model_headroom.models.gguf_loader.open_gguf_reader",
            return_value=reader,
        ):
            sample = extract_gguf_weight_sample(
                gguf_path, "dense.weight", max_rows=10, seed=42
            )
        assert sample is not None
        assert sample.shape == (10, 50)
