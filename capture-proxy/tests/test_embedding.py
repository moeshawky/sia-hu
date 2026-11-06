"""Tests for POST /v1/embeddings endpoint."""

import glob
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from capture_proxy.app import CaptureProxyApp


@pytest.fixture
def app_and_mock(tmp_path):
    """Create app with mocked upstream, bypassing /data mount check."""
    with patch.dict(
        "os.environ",
        {
            "CAPTURE_BASE_PATH": str(tmp_path / "captures"),
            "CAPTURE_UPSTREAM_URL": "http://127.0.0.1:8001",
        },
    ):
        proxy = CaptureProxyApp()

        mock_upstream = MagicMock()
        mock_upstream.health_check = AsyncMock(return_value=True)
        mock_upstream.proxy = AsyncMock()
        proxy._upstream_client = mock_upstream

        yield proxy.app, mock_upstream, tmp_path, proxy


@pytest.fixture
def client(app_and_mock):
    """Provide TestClient for the capture proxy app."""
    app, mock, _, _proxy = app_and_mock
    return TestClient(app), mock


def _get_error_records(tmp_path):
    """Read all summary JSONL records from the errors/ category."""
    jsonl_files = glob.glob(
        str(tmp_path / "captures" / "errors" / "**" / "*.jsonl"),
        recursive=True,
    )
    records = []
    for f in jsonl_files:
        with open(f) as fh:
            records.append(json.loads(fh.readline()))
    return records


def _get_summary_records(tmp_path):
    """Read all summary JSONL records from captures."""
    jsonl_files = glob.glob(
        str(tmp_path / "captures" / "raw" / "**" / "*.jsonl"),
        recursive=True,
    )
    summary_files = [f for f in jsonl_files if ".stream.jsonl" not in f]
    records = []
    for f in summary_files:
        with open(f) as fh:
            records.append(json.loads(fh.readline()))
    return records


class TestEmbeddingLlamacpp:
    """Tests for embedding requests against llama.cpp backend."""

    def test_single_embedding_returns_200(self, client):
        c, mock = client
        mock.proxy.return_value = MagicMock(
            status_code=200,
            body={
                "object": "list",
                "data": [
                    {"object": "embedding", "index": 0, "embedding": [0.1, 0.2, 0.3]}
                ],
                "model": "test-model",
                "usage": {"prompt_tokens": 5, "total_tokens": 5},
            },
            elapsed_ms=5.0,
        )
        resp = c.post(
            "/v1/embeddings",
            json={"input": "hello world", "model": "test-model"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["object"] == "list"
        assert len(data["data"]) == 1
        assert data["data"][0]["object"] == "embedding"
        assert data["data"][0]["index"] == 0
        assert data["data"][0]["embedding"] == [0.1, 0.2, 0.3]

    def test_batch_embedding_returns_200(self, client):
        c, mock = client
        mock.proxy.return_value = MagicMock(
            status_code=200,
            body={
                "object": "list",
                "data": [
                    {"object": "embedding", "index": 0, "embedding": [0.1, 0.2]},
                    {"object": "embedding", "index": 1, "embedding": [0.3, 0.4]},
                    {"object": "embedding", "index": 2, "embedding": [0.5, 0.6]},
                ],
                "model": "test-model",
                "usage": {"prompt_tokens": 15, "total_tokens": 15},
            },
            elapsed_ms=5.0,
        )
        resp = c.post(
            "/v1/embeddings",
            json={"input": ["hello", "world", "test"], "model": "test-model"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["data"]) == 3
        for i, entry in enumerate(data["data"]):
            assert entry["object"] == "embedding"
            assert entry["index"] == i

    def test_batch_embedding_body_unchanged(self, client):
        c, mock = client
        upstream_body = {
            "object": "list",
            "data": [{"object": "embedding", "index": 0, "embedding": [0.1, 0.2]}],
            "model": "m",
            "usage": {"prompt_tokens": 3, "total_tokens": 3},
        }
        mock.proxy.return_value = MagicMock(
            status_code=200,
            body=upstream_body,
            elapsed_ms=5.0,
        )
        resp = c.post(
            "/v1/embeddings",
            json={"input": "test", "model": "m"},
        )
        assert resp.json() == upstream_body


class TestEmbeddingOllama:
    """Tests for embedding requests against ollama backend."""

    def _make_ollama_app(self, tmp_path):
        """Create app configured for ollama backend."""
        with patch.dict(
            "os.environ",
            {
                "CAPTURE_BASE_PATH": str(tmp_path / "captures"),
                "CAPTURE_UPSTREAM_URL": "http://127.0.0.1:11434",
            },
        ):
            proxy = CaptureProxyApp()
            mock_upstream = MagicMock()
            mock_upstream.health_check = AsyncMock(return_value=True)
            mock_upstream.proxy = AsyncMock()
            proxy._upstream_client = mock_upstream
            return proxy.app, mock_upstream

    def test_single_embedding_ollama(self, tmp_path):
        app, mock = self._make_ollama_app(tmp_path)
        mock.proxy.return_value = MagicMock(
            status_code=200,
            body={"embeddings": [[0.1, 0.2, 0.3]]},
            elapsed_ms=5.0,
        )
        c = TestClient(app)
        resp = c.post(
            "/v1/embeddings",
            json={"input": "hello", "model": "nomic-embed"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["object"] == "list"
        assert len(data["data"]) == 1
        assert data["data"][0]["embedding"] == [0.1, 0.2, 0.3]

        # Verify upstream was called with ollama format
        call_args = mock.proxy.call_args
        assert call_args[1]["path"] == "/api/embed"
        assert call_args[1]["body"]["input"] == "hello"

    def test_batch_embedding_ollama(self, tmp_path):
        app, mock = self._make_ollama_app(tmp_path)
        mock.proxy.return_value = MagicMock(
            status_code=200,
            body={"embeddings": [[0.1, 0.2], [0.3, 0.4]]},
            elapsed_ms=5.0,
        )
        c = TestClient(app)
        resp = c.post(
            "/v1/embeddings",
            json={"input": ["hello", "world"], "model": "nomic-embed"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["data"]) == 2
        assert data["data"][0]["embedding"] == [0.1, 0.2]
        assert data["data"][1]["embedding"] == [0.3, 0.4]

        call_args = mock.proxy.call_args
        assert call_args[1]["body"]["input"] == ["hello", "world"]

    def test_ollama_upstream_error_returns_status(self, tmp_path):
        app, mock = self._make_ollama_app(tmp_path)
        mock.proxy.return_value = MagicMock(
            status_code=404,
            body={"error": "model not found"},
            elapsed_ms=5.0,
        )
        c = TestClient(app)
        resp = c.post(
            "/v1/embeddings",
            json={"input": "test", "model": "missing-model"},
        )
        assert resp.status_code == 404


class TestEmbeddingVllm:
    """Tests for embedding requests against vllm backend — OpenAI passthrough.

    vLLM serves native OpenAI /v1/embeddings, so the body forwards unchanged
    (same path as llama.cpp) and the response passes through.
    """

    def _make_vllm_app(self, tmp_path):
        """Create app configured for vllm backend."""
        with patch.dict(
            "os.environ",
            {
                "CAPTURE_BASE_PATH": str(tmp_path / "captures"),
                "CAPTURE_UPSTREAM_URL": "http://127.0.0.1:8000",
                "CAPTURE_BACKEND": "vllm",
            },
        ):
            proxy = CaptureProxyApp()
            mock_upstream = MagicMock()
            mock_upstream.health_check = AsyncMock(return_value=True)
            mock_upstream.proxy = AsyncMock()
            proxy._upstream_client = mock_upstream
            return proxy.app, mock_upstream

    def test_vllm_embedding_proxies_v1_embeddings(self, tmp_path):
        app, mock = self._make_vllm_app(tmp_path)
        mock.proxy.return_value = MagicMock(
            status_code=200,
            body={
                "object": "list",
                "data": [{"object": "embedding", "index": 0, "embedding": [0.9, 0.8]}],
                "model": "m",
                "usage": {"prompt_tokens": 4, "total_tokens": 4},
            },
            elapsed_ms=5.0,
        )
        c = TestClient(app)
        resp = c.post("/v1/embeddings", json={"input": "hello", "model": "m"})
        assert resp.status_code == 200
        assert resp.json()["data"][0]["embedding"] == [0.9, 0.8]

        call_args = mock.proxy.call_args
        assert call_args[1]["path"] == "/v1/embeddings"
        assert call_args[1]["body"]["input"] == "hello"
        assert call_args[1]["body"]["model"] == "m"

    def test_vllm_embedding_batch_body_unchanged(self, tmp_path):
        app, mock = self._make_vllm_app(tmp_path)
        mock.proxy.return_value = MagicMock(
            status_code=200,
            body={
                "object": "list",
                "data": [
                    {"object": "embedding", "index": 0, "embedding": [0.1, 0.2]},
                    {"object": "embedding", "index": 1, "embedding": [0.3, 0.4]},
                ],
                "model": "m",
                "usage": {"prompt_tokens": 8, "total_tokens": 8},
            },
            elapsed_ms=5.0,
        )
        c = TestClient(app)
        resp = c.post("/v1/embeddings", json={"input": ["a", "b"], "model": "m"})
        assert resp.status_code == 200
        assert len(resp.json()["data"]) == 2
        assert mock.proxy.call_args[1]["body"]["input"] == ["a", "b"]


class TestEmbeddingValidation:
    """Tests for input validation on embedding requests."""

    def test_empty_string_input_returns_422(self, client):
        c, _ = client
        resp = c.post(
            "/v1/embeddings",
            json={"input": "", "model": "test"},
        )
        assert resp.status_code == 422
        assert "input" in resp.json()["error"]

    def test_empty_list_input_returns_422(self, client):
        c, _ = client
        resp = c.post(
            "/v1/embeddings",
            json={"input": [], "model": "test"},
        )
        assert resp.status_code == 422
        assert "input" in resp.json()["error"]

    def test_missing_model_returns_422(self, client):
        c, _ = client
        resp = c.post(
            "/v1/embeddings",
            json={"input": "hello"},
        )
        assert resp.status_code == 422

    def test_empty_model_returns_422(self, client):
        c, _ = client
        resp = c.post(
            "/v1/embeddings",
            json={"input": "hello", "model": "   "},
        )
        assert resp.status_code == 422
        assert "model" in resp.json()["error"]

    def test_stream_true_returns_400(self, client):
        c, _ = client
        resp = c.post(
            "/v1/embeddings",
            json={"input": "hello", "model": "test", "stream": True},
        )
        assert resp.status_code == 400
        assert "streaming" in resp.json()["error"]

    def test_invalid_json_returns_422(self, client):
        c, _ = client
        resp = c.post(
            "/v1/embeddings",
            content=b"not json",
            headers={"Content-Type": "application/json"},
        )
        assert resp.status_code == 422


class TestEmbeddingCaptureRecord:
    """Tests for CaptureRecord fields populated by embedding requests."""

    def test_record_has_embedding_fields(self, client, tmp_path):
        c, mock = client
        mock.proxy.return_value = MagicMock(
            status_code=200,
            body={
                "object": "list",
                "data": [
                    {"object": "embedding", "index": 0, "embedding": [0.1, 0.2, 0.3]}
                ],
                "model": "test",
                "usage": {"prompt_tokens": 5, "total_tokens": 5},
            },
            elapsed_ms=5.0,
        )
        c.post(
            "/v1/embeddings",
            json={"input": "hello", "model": "test"},
        )

        records = _get_summary_records(tmp_path)
        assert len(records) >= 1
        record = records[-1]

        assert record["request_type"] == "embedding"
        assert record["embedding_count"] == 1
        assert record["embedding_dimensions"] == 3
        assert record["input_sha256"] is not None
        assert len(record["input_sha256"]) == 64  # SHA-256 hex digest

    def test_batch_record_has_correct_count(self, client, tmp_path):
        c, mock = client
        mock.proxy.return_value = MagicMock(
            status_code=200,
            body={
                "object": "list",
                "data": [
                    {"object": "embedding", "index": 0, "embedding": [0.1]},
                    {"object": "embedding", "index": 1, "embedding": [0.2]},
                    {"object": "embedding", "index": 2, "embedding": [0.3]},
                ],
                "model": "test",
                "usage": {"prompt_tokens": 9, "total_tokens": 9},
            },
            elapsed_ms=5.0,
        )
        c.post(
            "/v1/embeddings",
            json={"input": ["a", "b", "c"], "model": "test"},
        )

        records = _get_summary_records(tmp_path)
        record = records[-1]

        assert record["request_type"] == "embedding"
        assert record["embedding_count"] == 3
        assert record["embedding_dimensions"] == 1

    def test_input_sha256_deterministic(self, client, tmp_path):
        c, mock = client
        mock.proxy.return_value = MagicMock(
            status_code=200,
            body={
                "object": "list",
                "data": [{"object": "embedding", "index": 0, "embedding": [0.1]}],
                "model": "test",
                "usage": {"prompt_tokens": 1, "total_tokens": 1},
            },
            elapsed_ms=5.0,
        )
        c.post(
            "/v1/embeddings",
            json={"input": "hello", "model": "test"},
        )
        c.post(
            "/v1/embeddings",
            json={"input": "hello", "model": "test"},
        )

        records = _get_summary_records(tmp_path)
        assert len(records) >= 2
        assert records[-1]["input_sha256"] == records[-2]["input_sha256"]

    def test_different_inputs_different_hash(self, client, tmp_path):
        c, mock = client
        mock.proxy.return_value = MagicMock(
            status_code=200,
            body={
                "object": "list",
                "data": [{"object": "embedding", "index": 0, "embedding": [0.1]}],
                "model": "test",
                "usage": {"prompt_tokens": 1, "total_tokens": 1},
            },
            elapsed_ms=5.0,
        )
        c.post(
            "/v1/embeddings",
            json={"input": "hello", "model": "test"},
        )
        c.post(
            "/v1/embeddings",
            json={"input": "world", "model": "test"},
        )

        records = _get_summary_records(tmp_path)
        assert len(records) >= 2
        assert records[-1]["input_sha256"] != records[-2]["input_sha256"]

    def test_record_upstream_error_logged(self, client, tmp_path):
        """An upstream 500 on the embedding route is captured in errors/.

        The embeddings handler used to call the capture logger directly,
        which always wrote to raw/ and so misfiled upstream failures as
        successful captures. Routing through _log_capture applies the
        documented central policy (status >= 400 → errors/) that chat,
        rerank, classification and image already followed.
        """
        c, mock = client
        mock.proxy.return_value = MagicMock(
            status_code=500,
            body={"error": "internal"},
            elapsed_ms=5.0,
        )
        c.post(
            "/v1/embeddings",
            json={"input": "hello", "model": "test"},
        )

        assert _get_summary_records(tmp_path) == [], (
            "an upstream 500 must not be filed under raw/"
        )
        error_records = _get_error_records(tmp_path)
        record = error_records[-1]
        assert record["upstream_status"] == 500
        assert record["request_type"] == "embedding"
        assert record["task"] == "embedding.single"

    def test_successful_embedding_still_lands_in_raw(self, client, tmp_path):
        """The routing fix must not push successes out of raw/."""
        c, mock = client
        mock.proxy.return_value = MagicMock(
            status_code=200,
            body={
                "object": "list",
                "model": "test",
                "data": [{"object": "embedding", "index": 0, "embedding": [0.1]}],
                "usage": {"prompt_tokens": 1, "total_tokens": 1},
            },
            elapsed_ms=5.0,
        )
        c.post("/v1/embeddings", json={"input": "hello", "model": "test"})

        records = _get_summary_records(tmp_path)
        assert len(records) == 1
        assert records[-1]["upstream_status"] == 200
        assert _get_error_records(tmp_path) == []


class TestEmbeddingErrors:
    """Error handling — upstream failures and OSError resilience."""

    def test_os_error_during_validation_log_still_returns_422(self, app_and_mock):
        """A disk fault while logging an embedding validation failure still
        returns 422 — the validation-error log is not routed through
        _log_capture and needs its own OSError suppression."""
        app, _, _, proxy = app_and_mock
        with patch.object(
            proxy._capture_logger, "log_error", side_effect=OSError("disk full")
        ):
            resp = TestClient(app).post("/v1/embeddings", json={"model": "m"})
        assert resp.status_code == 422
        assert "embedding" in resp.json()["error"]
