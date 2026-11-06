"""Tests for alias request rewrite (public -> upstream) with audit fields."""

from __future__ import annotations

import glob
import json
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from capture_proxy.app import CaptureProxyApp, _compute_sha256

PUBLIC = "qwen3.8-27b-fp8"
UPSTREAM = "davidau-qwen3.8-27b-twin-turbo-fable"
PAIR_JSON = json.dumps({PUBLIC: UPSTREAM})


@pytest.fixture
def alias_client(tmp_path):
    """App with aliases on, upstream mocked."""
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
    """App with aliases off, upstream mocked."""
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


class TestChatRewrite:
    def test_upstream_gets_upstream_id(self, alias_client) -> None:
        c, mock, _ = alias_client
        mock.proxy.return_value = MagicMock(
            status_code=200,
            body={"id": "x", "model": UPSTREAM, "choices": []},
            headers={},
        )
        resp = c.post(
            "/v1/chat/completions",
            json={"model": PUBLIC, "messages": [{"role": "user", "content": "hi"}]},
        )
        assert resp.status_code == 200
        sent_body = mock.proxy.call_args.kwargs["body"]
        assert sent_body["model"] == UPSTREAM

    def test_record_audit_and_hash(self, alias_client) -> None:
        c, mock, tmp = alias_client
        mock.proxy.return_value = MagicMock(
            status_code=200,
            body={"id": "x", "model": UPSTREAM, "choices": []},
            headers={},
        )
        c.post(
            "/v1/chat/completions",
            json={"model": PUBLIC, "messages": [{"role": "user", "content": "hi"}]},
        )
        record = _read_record(tmp)
        assert record["model"] == PUBLIC
        assert record["request"]["model"] == UPSTREAM
        assert record["request_model_original"] == PUBLIC
        assert record["request_model_forwarded"] == UPSTREAM
        assert record["routing"] is None
        assert record["request_body_sha256"] == _compute_sha256(record["request"])

    def test_unknown_verbatim(self, alias_client) -> None:
        c, mock, tmp = alias_client
        mock.proxy.return_value = MagicMock(
            status_code=200,
            body={"id": "x", "model": "other", "choices": []},
            headers={},
        )
        c.post(
            "/v1/chat/completions",
            json={"model": "other", "messages": [{"role": "user", "content": "hi"}]},
        )
        sent_body = mock.proxy.call_args.kwargs["body"]
        assert sent_body["model"] == "other"
        record = _read_record(tmp)
        assert record["model"] == "other"
        # No rewrite happened, so the alias pair is NOT RECORDED rather
        # than "not applicable". schema.json declares these optional, and
        # an optional-unrecorded field has to be absent from the line;
        # null is reserved for the nullable class, or the two states
        # become indistinguishable to a consumer.
        for key in ("request_model_original", "request_model_forwarded"):
            assert key not in record, (
                f"{key} should be absent when no alias rewrite occurred"
            )

    def test_off_byte_identical(self, off_client) -> None:
        c, mock, tmp = off_client
        mock.proxy.return_value = MagicMock(
            status_code=200,
            body={"id": "x", "model": "m", "choices": []},
            headers={},
        )
        body = {"model": "m", "messages": [{"role": "user", "content": "hi"}]}
        c.post("/v1/chat/completions", json=body)
        sent_body = mock.proxy.call_args.kwargs["body"]
        assert sent_body == body
        record = _read_record(tmp)
        assert record["request"] == body
        # No rewrite happened, so the alias pair is NOT RECORDED rather
        # than "not applicable". schema.json declares these optional, and
        # an optional-unrecorded field has to be absent from the line;
        # null is reserved for the nullable class, or the two states
        # become indistinguishable to a consumer.
        for key in ("request_model_original", "request_model_forwarded"):
            assert key not in record, (
                f"{key} should be absent when no alias rewrite occurred"
            )


class TestEmbeddingRewrite:
    def test_passthrough_upstream_gets_upstream(self, alias_client) -> None:
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
        sent_body = mock.proxy.call_args.kwargs["body"]
        assert sent_body["model"] == UPSTREAM

    def test_embedding_audit_and_hash(self, alias_client) -> None:
        c, mock, tmp = alias_client
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
        c.post("/v1/embeddings", json={"model": PUBLIC, "input": "hello"})
        record = _read_record(tmp)
        assert record["model"] == PUBLIC
        assert record["request"]["model"] == UPSTREAM
        assert record["request_model_original"] == PUBLIC
        assert record["request_model_forwarded"] == UPSTREAM
        assert record["request_body_sha256"] == _compute_sha256(record["request"])

    def test_ollama_branch_uses_forwarded(self, tmp_path) -> None:
        env = {
            "CAPTURE_BASE_PATH": str(tmp_path / "captures"),
            "CAPTURE_MODEL_ALIASES": PAIR_JSON,
            "CAPTURE_UPSTREAM_URL": "http://127.0.0.1:11434",
            "CAPTURE_BACKEND": "ollama",
        }
        with patch.dict(os.environ, env, clear=False):
            proxy = CaptureProxyApp()
            assert proxy._backend == "ollama"
            mock_upstream = MagicMock()
            mock_upstream.proxy = AsyncMock(
                return_value=MagicMock(
                    status_code=200,
                    body={"embeddings": [[0.1, 0.2]]},
                    headers={},
                )
            )
            proxy._upstream_client = mock_upstream
            client = TestClient(proxy.app)
            resp = client.post(
                "/v1/embeddings", json={"model": PUBLIC, "input": "hello"}
            )
            assert resp.status_code == 200
            sent_body = mock_upstream.proxy.call_args.kwargs["body"]
            assert sent_body["model"] == UPSTREAM
            assert mock_upstream.proxy.call_args.kwargs["path"] == "/api/embed"


class TestRerankRewrite:
    def test_rerank_upstream_gets_upstream(self, alias_client) -> None:
        c, mock, tmp = alias_client
        mock.proxy.return_value = MagicMock(
            status_code=200, body={"results": []}, headers={}
        )
        resp = c.post(
            "/v1/rerank",
            json={"model": PUBLIC, "query": "q", "documents": ["d1", "d2"]},
        )
        assert resp.status_code == 200
        sent_body = mock.proxy.call_args.kwargs["body"]
        assert sent_body["model"] == UPSTREAM
        record = _read_record(tmp)
        assert record["model"] == PUBLIC
        assert record["request"]["model"] == UPSTREAM
        assert record["request_model_original"] == PUBLIC
        assert record["request_model_forwarded"] == UPSTREAM
        assert record["request_body_sha256"] == _compute_sha256(record["request"])
