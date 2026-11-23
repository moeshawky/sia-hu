"""Tests for unexercised error paths in src/capture_proxy/app.py.

Pins the except → 502 branches of the five non-chat handlers, the streaming
pre-flight connect failure, and the seven native gpu-arm routes. Source pinned:
src/capture_proxy/app.py.

Reuses the existing `app_and_mock` pattern from capture-proxy/tests/test_app.py
(CaptureProxyApp + MagicMock upstream + TestClient) via a plain helper — no new
fixtures invented, no existing test files edited, no fixture imported across
modules (which ruff cannot validate as used).
"""

from __future__ import annotations

import glob
import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from capture_proxy.app import CaptureProxyApp

EMBEDDING_BODY = {"model": "m", "input": "hello"}
RERANK_BODY = {"model": "m", "query": "q", "documents": ["d1"]}
CLASSIFICATION_BODY = {"model": "m", "inputs": ["i1"]}
IMAGE_BODY = {"model": "m", "prompt": "p", "n": 1}
NATIVE_BODY = {"arbitrary": "body", "numbers": [1, 2, 3]}

NATIVE_ROUTES = [
    "/v1/systemone",
    "/v1/classify",
    "/v1/batch-classify",
    "/v1/ner",
    "/v1/batch-ner",
    "/v1/timeseries",
    "/v1/batch-timeseries",
]


def _build_app(tmp_path, monkeypatch):
    """Replicate the app_and_mock pattern: app + mocked upstream + TestClient.

    Mirrors capture-proxy/tests/test_app.py::app_and_mock without importing a
    cross-module fixture (ruff cannot validate fixture imports used as params).
    """
    monkeypatch.setenv("CAPTURE_BASE_PATH", str(tmp_path / "captures"))
    proxy = CaptureProxyApp()
    mock_upstream = MagicMock()
    mock_upstream.health_check = AsyncMock(return_value=True)
    mock_upstream.proxy = AsyncMock()
    proxy._upstream_client = mock_upstream
    return TestClient(proxy.app), mock_upstream, proxy, tmp_path


def _read_errors_record(tmp_path):
    """Return the first parsed record from errors/ for the current capture."""
    files = glob.glob(
        str(tmp_path / "captures" / "errors" / "**" / "*.jsonl"), recursive=True
    )
    assert files, "no errors/ capture record written"
    with open(files[0]) as fh:
        return json.loads(fh.readline())


def _assert_502_and_error_body(resp, tmp_path, exc_message="refused"):
    """Shared assertion: 502 response and an errors/ record carrying error_body."""
    assert resp.status_code == 502
    assert resp.json()["error"] == "upstream unavailable"
    record = _read_errors_record(tmp_path)
    assert record["upstream_status"] == 502
    assert record["error_body"] == exc_message


class TestEmbeddingErrorPath:
    def test_embedding_upstream_refused_returns_502(self, tmp_path, monkeypatch):
        """POST /v1/embeddings: upstream Exception → 502 + error_body (app.py:1324-1336)."""
        c, mock, _, tmp_path = _build_app(tmp_path, monkeypatch)
        mock.proxy.side_effect = Exception("refused")
        resp = c.post("/v1/embeddings", json=EMBEDDING_BODY)
        _assert_502_and_error_body(resp, tmp_path)


class TestRerankErrorPath:
    def test_rerank_upstream_refused_returns_502(self, tmp_path, monkeypatch):
        """POST /v1/rerank: upstream Exception → 502 + error_body (app.py:1499-1508).

        The default backend is ollama, which short-circuits rerank to 501
        before reaching the proxy, so the backend is switched to a non-ollama
        value on the built proxy.
        """
        c, mock, proxy, tmp_path = _build_app(tmp_path, monkeypatch)
        proxy._backend = "llamacpp"
        mock.proxy.side_effect = Exception("refused")
        resp = c.post("/v1/rerank", json=RERANK_BODY)
        _assert_502_and_error_body(resp, tmp_path)


class TestClassificationErrorPath:
    def test_classification_upstream_refused_returns_502(self, tmp_path, monkeypatch):
        """POST /v1/classification: upstream Exception → 502 + error_body
        (app.py:1664-1673)."""
        c, mock, _, tmp_path = _build_app(tmp_path, monkeypatch)
        mock.proxy.side_effect = Exception("refused")
        resp = c.post("/v1/classification", json=CLASSIFICATION_BODY)
        _assert_502_and_error_body(resp, tmp_path)


class TestImageErrorPath:
    def test_image_upstream_refused_returns_502(self, tmp_path, monkeypatch):
        """POST /v1/images/generations: upstream Exception → 502 + error_body
        (app.py:1825-1834)."""
        c, mock, _, tmp_path = _build_app(tmp_path, monkeypatch)
        mock.proxy.side_effect = Exception("refused")
        resp = c.post("/v1/images/generations", json=IMAGE_BODY)
        _assert_502_and_error_body(resp, tmp_path)


class TestNativePassthroughErrorPath:
    def test_native_passthrough_upstream_refused_returns_502(
        self, tmp_path, monkeypatch
    ):
        """POST /v1/classify: upstream Exception → 502 + error_body
        (app.py:1977-1986)."""
        c, mock, _, tmp_path = _build_app(tmp_path, monkeypatch)
        mock.proxy.side_effect = Exception("refused")
        resp = c.post("/v1/classify", json=NATIVE_BODY)
        _assert_502_and_error_body(resp, tmp_path)


class TestStreamingPreflightError:
    def test_streaming_preflight_connect_error_returns_502(self, tmp_path, monkeypatch):
        """POST /v1/chat/completions stream=True, proxy raises → 502 JSON,
        not a 200 SSE stream (app.py:2093-2116)."""
        c, mock, _, tmp_path = _build_app(tmp_path, monkeypatch)
        mock.proxy.side_effect = Exception("refused")
        resp = c.post(
            "/v1/chat/completions",
            json={
                "model": "m",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": True,
            },
        )
        assert resp.status_code == 502
        assert resp.json() == {"error": "upstream unavailable"}
        assert "text/event-stream" not in resp.headers.get("content-type", "")
        record = _read_errors_record(tmp_path)
        assert record["request_type"] == "chat"
        assert record["error_body"] == "refused"


class TestNativeRoutes:
    @pytest.mark.parametrize("path", NATIVE_ROUTES)
    def test_native_route_body_verbatim_and_captured(self, path, tmp_path, monkeypatch):
        """POST <native route> with an arbitrary body: upstream receives
        identical bytes and a capture record lands (app.py:724-778,
        app.py:1948-1986)."""
        c, mock, _, tmp_path = _build_app(tmp_path, monkeypatch)
        mock.proxy.return_value = MagicMock(status_code=200, body={"ok": True})
        resp = c.post(path, json=NATIVE_BODY)
        assert resp.status_code == 200
        # Upstream received the identical parsed body (aliases off = verbatim).
        sent_body = mock.proxy.call_args.kwargs["body"]
        assert sent_body == NATIVE_BODY
        # A capture record landed in raw/.
        files = glob.glob(
            str(tmp_path / "captures" / "raw" / "**" / "*.jsonl"), recursive=True
        )
        assert files, "no raw/ capture record written"
        with open(files[0]) as fh:
            record = json.loads(fh.readline())
        assert record["request_type"] == path.removeprefix("/v1/").replace("-", "_")
        assert record["request"] == NATIVE_BODY
