from __future__ import annotations

import json
from typing import Any

import pytest

from model_headroom.models.loader import _detect_model_format


class TestDetectModelFormat:
    def test_gguf_file(self, tmp_path: Any) -> None:
        p = tmp_path / "model.gguf"
        p.write_bytes(b"GGUF")
        assert _detect_model_format(str(p)) == "gguf"

    def test_gguf_with_path_prefix(self, tmp_path: Any) -> None:
        p = tmp_path / "subdir" / "model.gguf"
        p.parent.mkdir()
        p.write_bytes(b"GGUF")
        assert _detect_model_format(str(p)) == "gguf"

    def test_hf_directory_with_config(self, tmp_path: Any) -> None:
        (tmp_path / "config.json").write_text(json.dumps({}))
        assert _detect_model_format(str(tmp_path)) == "huggingface"

    def test_hf_directory_without_config(self, tmp_path: Any) -> None:
        with pytest.raises(ValueError, match="Cannot determine model format"):
            _detect_model_format(str(tmp_path))

    def test_hf_directory_with_encoder_config(self, tmp_path: Any) -> None:
        """laya format: encoder/config.json without root config.json."""
        (tmp_path / "encoder").mkdir()
        (tmp_path / "encoder" / "config.json").write_text(json.dumps({}))
        (tmp_path / "model.safetensors").write_bytes(b"")
        assert _detect_model_format(str(tmp_path)) == "huggingface"

    def test_nonexistent_path(self) -> None:
        with pytest.raises(ValueError, match="Cannot determine model format"):
            _detect_model_format("/nonexistent/path")

    def test_regular_file_not_gguf(self, tmp_path: Any) -> None:
        p = tmp_path / "data.txt"
        p.write_text("not a model")
        with pytest.raises(ValueError, match="Cannot determine model format"):
            _detect_model_format(str(p))
