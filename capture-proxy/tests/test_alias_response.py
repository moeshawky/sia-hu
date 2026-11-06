"""Tests for alias response sanitize (upstream -> public)."""

from __future__ import annotations

import glob
import json
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from capture_proxy.app import CaptureProxyApp

PUBLIC = "qwen3.8-27b-fp8"
UPSTREAM = "davidau-qwen3.8-27b-twin-turbo-fable"
PAIR_JSON = json.dumps({PUBLIC: UPSTREAM})


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


def _read_record(tmp_path, subdir="raw") -> dict:
    files = glob.glob(
        str(tmp_path / "captures" / subdir / "**" / "*.jsonl"), recursive=True
    )
    summary = [f for f in files if ".stream.jsonl" not in f]
    assert summary, "expected a capture JSONL line"
    with open(summary[0]) as f:
        return json.loads(f.readline())


class TestNonStreamingSanitize:
    def test_client_sees_public(self, alias_client) -> None:
        c, mock, _ = alias_client
        mock.proxy.return_value = MagicMock(
            status_code=200,
            body={
                "id": "chatcmpl-1",
                "model": UPSTREAM,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "hi"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            },
            headers={},
        )
        resp = c.post(
            "/v1/chat/completions",
            json={"model": PUBLIC, "messages": [{"role": "user", "content": "hi"}]},
        )
        assert resp.status_code == 200
        assert resp.json()["model"] == PUBLIC

    def test_record_response_model_public(self, alias_client) -> None:
        c, mock, tmp = alias_client
        mock.proxy.return_value = MagicMock(
            status_code=200,
            body={
                "id": "chatcmpl-1",
                "model": UPSTREAM,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "hi"},
                        "finish_reason": "stop",
                    }
                ],
            },
            headers={},
        )
        c.post(
            "/v1/chat/completions",
            json={"model": PUBLIC, "messages": [{"role": "user", "content": "hi"}]},
        )
        record = _read_record(tmp)
        assert record["response"]["model"] == PUBLIC
        assert record["model"] == PUBLIC

    def test_off_byte_identical(self, off_client) -> None:
        c, mock, tmp = off_client
        body = {
            "id": "chatcmpl-1",
            "model": "m",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "hi"},
                    "finish_reason": "stop",
                }
            ],
        }
        mock.proxy.return_value = MagicMock(
            status_code=200, body=dict(body), headers={}
        )
        resp = c.post(
            "/v1/chat/completions",
            json={"model": "m", "messages": [{"role": "user", "content": "hi"}]},
        )
        assert resp.json() == body
        record = _read_record(tmp)
        assert record["response"] == body


class TestEmbeddingSanitize:
    def test_embedding_client_sees_public(self, alias_client) -> None:
        c, mock, _ = alias_client
        mock.proxy.return_value = MagicMock(
            status_code=200,
            body={
                "object": "list",
                "data": [{"object": "embedding", "index": 0, "embedding": [0.1]}],
                "model": UPSTREAM,
                "usage": {"prompt_tokens": 1, "total_tokens": 1},
            },
            headers={},
        )
        resp = c.post("/v1/embeddings", json={"model": PUBLIC, "input": "hello"})
        assert resp.status_code == 200
        assert resp.json()["model"] == PUBLIC


class TestModelsList:
    def test_ids_remapped_root_absent_order_len(self, alias_client) -> None:
        c, mock, _ = alias_client
        upstream_body = {
            "object": "list",
            "data": [
                {"id": UPSTREAM, "object": "model", "root": UPSTREAM},
                {"id": "b-model", "object": "model", "root": "b-model"},
                {"id": "a-model", "object": "model", "root": "a-model"},
            ],
        }
        mock.proxy.return_value = MagicMock(
            status_code=200, body=upstream_body, headers={}
        )
        resp = c.get("/v1/models")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert len(data) == 3
        assert [e["id"] for e in data] == [PUBLIC, "b-model", "a-model"]
        for entry in data:
            assert "root" not in entry

    def test_empty_passthrough(self, alias_client) -> None:
        c, mock, _ = alias_client
        mock.proxy.return_value = MagicMock(
            status_code=200, body={"object": "list", "data": []}, headers={}
        )
        resp = c.get("/v1/models")
        assert resp.json() == {"object": "list", "data": []}

    def test_non_list_passthrough(self, alias_client) -> None:
        c, mock, _ = alias_client
        mock.proxy.return_value = MagicMock(
            status_code=200, body={"object": "list"}, headers={}
        )
        resp = c.get("/v1/models")
        assert resp.json() == {"object": "list"}

    def test_off_verbatim_with_root(self, off_client) -> None:
        c, mock, _ = off_client
        upstream_body = {
            "object": "list",
            "data": [{"id": "m", "object": "model", "root": "m"}],
        }
        mock.proxy.return_value = MagicMock(
            status_code=200, body=upstream_body, headers={}
        )
        resp = c.get("/v1/models")
        assert resp.json() == upstream_body
