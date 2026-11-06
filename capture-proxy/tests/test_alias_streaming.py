"""Tests for alias streaming sanitize (per-event upstream -> public)."""

from __future__ import annotations

import glob
import json
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from capture_proxy.app import CaptureProxyApp
from capture_proxy.upstream import StreamingUpstreamResponse

PUBLIC = "qwen3.8-27b-fp8"
UPSTREAM = "davidau-qwen3.8-27b-twin-turbo-fable"
PAIR_JSON = json.dumps({PUBLIC: UPSTREAM})


def _streaming_response(chunks: list[bytes]) -> StreamingUpstreamResponse:
    async def _aiter():
        for chunk in chunks:
            yield chunk

    return StreamingUpstreamResponse(
        status_code=200,
        headers={"content-type": "text/event-stream"},
        content_type="text/event-stream",
        stream=_aiter(),
    )


def _chunks_with_model(model: str, content_note: str = "hello") -> list[bytes]:
    return [
        json.dumps(
            {
                "id": "chatcmpl-9",
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"role": "assistant", "content": content_note},
                        "finish_reason": None,
                    }
                ],
            }
        ).encode(),
        json.dumps(
            {
                "id": "chatcmpl-9",
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": " world"},
                        "finish_reason": "stop",
                    }
                ],
            }
        ).encode(),
    ]


def _sse(chunks: list[bytes]) -> list[bytes]:
    return [b"data: " + c + b"\n\n" for c in chunks] + [b"data: [DONE]\n\n"]


@pytest.fixture
def alias_client(tmp_path):
    env = {
        "CAPTURE_BASE_PATH": str(tmp_path / "captures"),
        "CAPTURE_MODEL_ALIASES": PAIR_JSON,
        "CAPTURE_UPSTREAM_URL": "http://127.0.0.1:8080",
    }
    with patch.dict(os.environ, env, clear=False):
        os.environ.pop("CAPTURE_BACKEND", None)
        proxy = CaptureProxyApp()
        mock_upstream = MagicMock()
        mock_upstream.health_check = AsyncMock(return_value=True)
        mock_upstream.proxy = AsyncMock()
        proxy._upstream_client = mock_upstream
        yield TestClient(proxy.app), mock_upstream, tmp_path


@pytest.fixture
def off_client(tmp_path):
    env = {
        "CAPTURE_BASE_PATH": str(tmp_path / "captures"),
        "CAPTURE_UPSTREAM_URL": "http://127.0.0.1:8080",
    }
    with patch.dict(os.environ, env, clear=False):
        os.environ.pop("CAPTURE_MODEL_ALIASES", None)
        os.environ.pop("CAPTURE_BACKEND", None)
        proxy = CaptureProxyApp()
        mock_upstream = MagicMock()
        mock_upstream.health_check = AsyncMock(return_value=True)
        mock_upstream.proxy = AsyncMock()
        proxy._upstream_client = mock_upstream
        yield TestClient(proxy.app), mock_upstream, tmp_path


def _read_record(tmp_path) -> dict:
    files = glob.glob(
        str(tmp_path / "captures" / "raw" / "**" / "*.jsonl"), recursive=True
    )
    summary = [f for f in files if ".stream.jsonl" not in f]
    assert summary, "expected a capture JSONL line"
    with open(summary[0]) as f:
        return json.loads(f.readline())


def _read_stream_terminal(tmp_path) -> dict:
    files = glob.glob(
        str(tmp_path / "captures" / "raw" / "**" / "*.stream.jsonl"), recursive=True
    )
    assert files, "expected a stream.jsonl file"
    last: dict = {}
    with open(files[0]) as f:
        for line in f:
            row = json.loads(line)
            if row.get("type") == "terminal":
                last = row
    assert last, "expected a terminal record"
    return last


class TestStreamingSanitize:
    def test_chunk_model_remapped_downstream(self, alias_client) -> None:
        c, mock, _ = alias_client
        mock.proxy.return_value = _streaming_response(
            _sse(_chunks_with_model(UPSTREAM))
        )
        resp = c.post(
            "/v1/chat/completions",
            json={
                "model": PUBLIC,
                "messages": [{"role": "user", "content": "hi"}],
                "stream": True,
            },
        )
        assert resp.status_code == 200
        text = resp.text
        assert PUBLIC in text
        assert UPSTREAM not in text.replace(f"mention {UPSTREAM}", "")

    def test_delta_content_with_model_name_untouched(self, alias_client) -> None:
        c, mock, _ = alias_client
        note = f"the model {UPSTREAM} says hi"
        mock.proxy.return_value = _streaming_response(
            _sse(_chunks_with_model(UPSTREAM, content_note=note))
        )
        resp = c.post(
            "/v1/chat/completions",
            json={
                "model": PUBLIC,
                "messages": [{"role": "user", "content": "hi"}],
                "stream": True,
            },
        )
        assert note in resp.text

    def test_done_verbatim(self, alias_client) -> None:
        c, mock, _ = alias_client
        mock.proxy.return_value = _streaming_response(
            _sse(_chunks_with_model(UPSTREAM))
        )
        resp = c.post(
            "/v1/chat/completions",
            json={
                "model": PUBLIC,
                "messages": [{"role": "user", "content": "hi"}],
                "stream": True,
            },
        )
        assert "data: [DONE]" in resp.text

    def test_reconstructed_response_model_public(self, alias_client) -> None:
        c, mock, tmp = alias_client
        mock.proxy.return_value = _streaming_response(
            _sse(_chunks_with_model(UPSTREAM))
        )
        c.post(
            "/v1/chat/completions",
            json={
                "model": PUBLIC,
                "messages": [{"role": "user", "content": "hi"}],
                "stream": True,
            },
        )
        record = _read_record(tmp)
        assert record["response"]["model"] == PUBLIC
        assert record["model"] == PUBLIC

    def test_off_mode_identical_true(self, off_client) -> None:
        c, mock, tmp = off_client
        chunks = _sse(_chunks_with_model("m"))
        mock.proxy.return_value = _streaming_response(chunks)
        resp = c.post(
            "/v1/chat/completions",
            json={
                "model": "m",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": True,
            },
        )
        assert resp.status_code == 200
        terminal = _read_stream_terminal(tmp)
        assert terminal["identical"] is True
