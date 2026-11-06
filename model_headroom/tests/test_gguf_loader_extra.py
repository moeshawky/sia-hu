from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from model_headroom.models.gguf_loader import (
    _GGUFLoadError,
    _read_metadata_int,
    _read_metadata_string,
)


class TestReadMetadataStringErrorPaths:
    def test_none_field(self) -> None:
        with pytest.raises(_GGUFLoadError) as exc_info:
            _read_metadata_string(None)
        assert exc_info.value.error_type == "missing_metadata"

    def test_no_contents_method(self) -> None:
        obj = MagicMock(spec=[])
        with pytest.raises(_GGUFLoadError) as exc_info:
            _read_metadata_string(obj)
        assert exc_info.value.error_type == "missing_metadata"

    def test_contents_returns_none(self) -> None:
        obj = MagicMock()
        obj.contents.return_value = None
        with pytest.raises(_GGUFLoadError) as exc_info:
            _read_metadata_string(obj)
        assert exc_info.value.error_type == "missing_metadata"

    def test_contents_returns_string(self) -> None:
        obj = MagicMock()
        obj.contents.return_value = "mixtral"
        result = _read_metadata_string(obj)
        assert result == "mixtral"


class TestReadMetadataIntErrorPaths:
    def test_none_field(self) -> None:
        with pytest.raises(_GGUFLoadError) as exc_info:
            _read_metadata_int(None)
        assert exc_info.value.error_type == "missing_metadata"

    def test_no_contents_method(self) -> None:
        obj = MagicMock(spec=[])
        with pytest.raises(_GGUFLoadError) as exc_info:
            _read_metadata_int(obj)
        assert exc_info.value.error_type == "missing_metadata"

    def test_contents_returns_none(self) -> None:
        obj = MagicMock()
        obj.contents.return_value = None
        with pytest.raises(_GGUFLoadError) as exc_info:
            _read_metadata_int(obj)
        assert exc_info.value.error_type == "missing_metadata"

    def test_contents_returns_int(self) -> None:
        obj = MagicMock()
        obj.contents.return_value = 42
        result = _read_metadata_int(obj)
        assert result == 42

    def test_contents_returns_string_int(self) -> None:
        obj = MagicMock()
        obj.contents.return_value = "8"
        result = _read_metadata_int(obj)
        assert result == 8
