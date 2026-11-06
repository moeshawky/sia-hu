"""Tests for POST /v1/rerank endpoint.

Covers: llama.cpp passthrough, ollama rejection, validation, CaptureRecord fields,
error handling, and OSError resilience during logging.
"""

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
    return TestClient(app), mock, _proxy


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


def _get_error_records(tmp_path):
    """Read all summary JSONL records from the errors/ directory."""
    jsonl_files = glob.glob(
        str(tmp_path / "captures" / "errors" / "**" / "*.jsonl"),
        recursive=True,
    )
    summary_files = [f for f in jsonl_files if ".stream.jsonl" not in f]
    records = []
    for f in summary_files:
        with open(f) as fh:
            records.append(json.loads(fh.readline()))
    return records


def _make_ollama_app(tmp_path):
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


def _make_vllm_app(tmp_path):
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


class TestRerankLlamacpp:
    """Tests for rerank requests against llama.cpp backend — proxy passthrough."""

    def test_single_document_returns_200(self, client):
        c, mock, _ = client
        upstream_body = {"results": [{"index": 0, "relevance_score": 0.95}]}
        mock.proxy.return_value = MagicMock(
            status_code=200,
            body=upstream_body,
            elapsed_ms=5.0,
        )
        resp = c.post(
            "/v1/rerank",
            json={
                "model": "rerank-model",
                "query": "test query",
                "documents": ["document content"],
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["results"][0]["relevance_score"] == 0.95

    def test_multiple_documents_returns_200(self, client):
        c, mock, _ = client
        upstream_body = {
            "results": [
                {"index": 0, "relevance_score": 0.9},
                {"index": 1, "relevance_score": 0.8},
                {"index": 2, "relevance_score": 0.7},
            ]
        }
        mock.proxy.return_value = MagicMock(
            status_code=200,
            body=upstream_body,
            elapsed_ms=5.0,
        )
        resp = c.post(
            "/v1/rerank",
            json={
                "model": "rerank-model",
                "query": "test query",
                "documents": ["doc1", "doc2", "doc3"],
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["results"]) == 3
        assert data["results"][0]["index"] == 0
        assert data["results"][1]["index"] == 1
        assert data["results"][2]["index"] == 2

    def test_body_unchanged(self, client):
        c, mock, _ = client
        upstream_body = {
            "results": [{"index": 0, "relevance_score": 0.95, "text": "doc"}]
        }
        mock.proxy.return_value = MagicMock(
            status_code=200,
            body=upstream_body,
            elapsed_ms=5.0,
        )
        resp = c.post(
            "/v1/rerank",
            json={
                "model": "m",
                "query": "q",
                "documents": ["doc"],
            },
        )
        assert resp.json() == upstream_body

    def test_top_n_handling(self, client):
        c, mock, _ = client
        upstream_body = {"results": [{"index": 0, "relevance_score": 0.95}]}
        mock.proxy.return_value = MagicMock(
            status_code=200,
            body=upstream_body,
            elapsed_ms=5.0,
        )
        resp = c.post(
            "/v1/rerank",
            json={
                "model": "m",
                "query": "q",
                "documents": ["doc1", "doc2"],
                "top_n": 1,
            },
        )
        assert resp.status_code == 200

        call_body = mock.proxy.call_args[1]["body"]
        assert call_body["top_n"] == 1


class TestRerankOllama:
    """Tests for rerank requests against ollama backend — always 501."""

    def test_ollama_returns_501(self, tmp_path):
        app, _mock = _make_ollama_app(tmp_path)
        c = TestClient(app)
        resp = c.post(
            "/v1/rerank",
            json={
                "model": "test",
                "query": "q",
                "documents": ["doc"],
            },
        )
        assert resp.status_code == 501
        assert "rerank" in resp.json()["error"].lower()

    def test_ollama_no_proxy_call(self, tmp_path):
        app, mock = _make_ollama_app(tmp_path)
        c = TestClient(app)
        c.post(
            "/v1/rerank",
            json={
                "model": "test",
                "query": "q",
                "documents": ["doc"],
            },
        )
        mock.proxy.assert_not_called()


class TestRerankVllm:
    """Tests for rerank requests against vllm backend — proxy passthrough.

    vLLM (>= 0.7) exposes Jina/Cohere-compatible /v1/rerank, so the request
    body (model, query, documents, top_n) is forwarded verbatim and the
    response is passed through unchanged.
    """

    def test_vllm_rerank_proxies_to_v1_rerank(self, tmp_path):
        app, mock = _make_vllm_app(tmp_path)
        mock.proxy.return_value = MagicMock(
            status_code=200,
            body={"results": [{"index": 0, "relevance_score": 0.97}]},
            elapsed_ms=5.0,
        )
        c = TestClient(app)
        resp = c.post(
            "/v1/rerank",
            json={
                "model": "rerank-model",
                "query": "test query",
                "documents": ["document content"],
            },
        )
        assert resp.status_code == 200
        assert resp.json()["results"][0]["relevance_score"] == 0.97
        call_args = mock.proxy.call_args
        assert call_args[1]["path"] == "/v1/rerank"

    def test_vllm_rerank_forwards_top_n(self, tmp_path):
        app, mock = _make_vllm_app(tmp_path)
        mock.proxy.return_value = MagicMock(
            status_code=200,
            body={"results": [{"index": 0, "relevance_score": 0.9}]},
            elapsed_ms=5.0,
        )
        c = TestClient(app)
        resp = c.post(
            "/v1/rerank",
            json={
                "model": "m",
                "query": "q",
                "documents": ["doc1", "doc2"],
                "top_n": 2,
            },
        )
        assert resp.status_code == 200
        assert mock.proxy.call_args[1]["body"]["top_n"] == 2

    def test_vllm_rerank_upstream_error_passthrough(self, tmp_path):
        app, mock = _make_vllm_app(tmp_path)
        mock.proxy.return_value = MagicMock(
            status_code=404,
            body={"error": "model not found"},
            elapsed_ms=5.0,
        )
        c = TestClient(app)
        resp = c.post(
            "/v1/rerank",
            json={"model": "m", "query": "q", "documents": ["doc"]},
        )
        assert resp.status_code == 404
        assert resp.json()["error"] == "model not found"


class TestRerankValidation:
    """Tests for input validation on rerank requests — all 4xx error paths."""

    def test_empty_model_returns_422(self, client):
        c, _, _ = client
        resp = c.post(
            "/v1/rerank",
            json={"model": "", "query": "q", "documents": ["doc"]},
        )
        assert resp.status_code == 422
        assert "model" in resp.json()["error"]

    def test_missing_model_returns_422(self, client):
        c, _, _ = client
        resp = c.post(
            "/v1/rerank",
            json={"query": "q", "documents": ["doc"]},
        )
        assert resp.status_code == 422

    def test_empty_query_returns_422(self, client):
        c, _, _ = client
        resp = c.post(
            "/v1/rerank",
            json={"model": "m", "query": "", "documents": ["doc"]},
        )
        assert resp.status_code == 422
        assert "query" in resp.json()["error"]

    def test_missing_query_returns_422(self, client):
        c, _, _ = client
        resp = c.post(
            "/v1/rerank",
            json={"model": "m", "documents": ["doc"]},
        )
        assert resp.status_code == 422

    def test_empty_documents_returns_422(self, client):
        c, _, _ = client
        resp = c.post(
            "/v1/rerank",
            json={"model": "m", "query": "q", "documents": []},
        )
        assert resp.status_code == 422
        assert "documents" in resp.json()["error"]

    def test_missing_documents_returns_422(self, client):
        c, _, _ = client
        resp = c.post(
            "/v1/rerank",
            json={"model": "m", "query": "q"},
        )
        assert resp.status_code == 422

    def test_top_n_less_than_one_returns_422(self, client):
        c, _, _ = client
        resp = c.post(
            "/v1/rerank",
            json={"model": "m", "query": "q", "documents": ["doc"], "top_n": 0},
        )
        assert resp.status_code == 422
        assert "top_n" in resp.json()["error"]

    def test_negative_top_n_returns_422(self, client):
        c, _, _ = client
        resp = c.post(
            "/v1/rerank",
            json={"model": "m", "query": "q", "documents": ["doc"], "top_n": -1},
        )
        assert resp.status_code == 422

    def test_stream_true_returns_400(self, client):
        c, _, _ = client
        resp = c.post(
            "/v1/rerank",
            json={
                "model": "m",
                "query": "q",
                "documents": ["doc"],
                "stream": True,
            },
        )
        assert resp.status_code == 400
        assert "streaming" in resp.json()["error"]

    def test_invalid_json_returns_422(self, client):
        c, _, _ = client
        resp = c.post(
            "/v1/rerank",
            content=b"not json",
            headers={"Content-Type": "application/json"},
        )
        assert resp.status_code == 422


class TestRerankCaptureRecord:
    """Tests for CaptureRecord fields populated by rerank requests."""

    def test_request_type_is_rerank(self, client, tmp_path):
        c, mock, _ = client
        mock.proxy.return_value = MagicMock(
            status_code=200,
            body={"results": [{"index": 0, "relevance_score": 0.95}]},
            elapsed_ms=5.0,
        )
        c.post(
            "/v1/rerank",
            json={"model": "m", "query": "q", "documents": ["doc"]},
        )

        records = _get_summary_records(tmp_path)
        assert len(records) >= 1
        assert records[-1]["request_type"] == "rerank"

    def test_record_has_rerank_request_body(self, client, tmp_path):
        c, mock, _ = client
        mock.proxy.return_value = MagicMock(
            status_code=200,
            body={"results": [{"index": 0, "relevance_score": 0.95}]},
            elapsed_ms=5.0,
        )
        body = {"model": "m", "query": "q", "documents": ["doc"]}
        c.post("/v1/rerank", json=body)

        records = _get_summary_records(tmp_path)
        record = records[-1]
        assert record["request"]["model"] == "m"
        assert record["request"]["query"] == "q"
        assert record["request"]["documents"] == ["doc"]

    def test_record_has_upstream_response(self, client, tmp_path):
        c, mock, _ = client
        mock.proxy.return_value = MagicMock(
            status_code=200,
            body={"results": [{"index": 0, "relevance_score": 0.95}]},
            elapsed_ms=5.0,
        )
        c.post(
            "/v1/rerank",
            json={"model": "m", "query": "q", "documents": ["doc"]},
        )

        records = _get_summary_records(tmp_path)
        record = records[-1]
        assert record["response"]["results"][0]["relevance_score"] == 0.95
        assert record["upstream_status"] == 200

    def test_record_upstream_error_logged(self, client, tmp_path):
        c, mock, _ = client
        mock.proxy.return_value = MagicMock(
            status_code=500,
            body={"error": "internal"},
            elapsed_ms=5.0,
        )
        c.post(
            "/v1/rerank",
            json={"model": "m", "query": "q", "documents": ["doc"]},
        )

        records = _get_error_records(tmp_path)
        record = records[-1]
        assert record["upstream_status"] == 500
        assert record["request_type"] == "rerank"


class TestRerankErrors:
    """Tests for error handling on rerank requests — upstream failures and OSError resilience."""

    def test_upstream_down_returns_502(self, client):
        c, mock, _ = client
        mock.proxy.side_effect = Exception("connection refused")
        resp = c.post(
            "/v1/rerank",
            json={"model": "m", "query": "q", "documents": ["doc"]},
        )
        assert resp.status_code == 502
        assert resp.json()["error"] == "upstream unavailable"

    def test_upstream_non_200_passthrough(self, client):
        c, mock, _ = client
        mock.proxy.return_value = MagicMock(
            status_code=404,
            body={"error": "model not found"},
            elapsed_ms=5.0,
        )
        resp = c.post(
            "/v1/rerank",
            json={"model": "m", "query": "q", "documents": ["doc"]},
        )
        assert resp.status_code == 404
        assert resp.json()["error"] == "model not found"

    def test_upstream_400_passthrough(self, client):
        c, mock, _ = client
        mock.proxy.return_value = MagicMock(
            status_code=400,
            body={"error": "bad request"},
            elapsed_ms=5.0,
        )
        resp = c.post(
            "/v1/rerank",
            json={"model": "m", "query": "q", "documents": ["doc"]},
        )
        assert resp.status_code == 400

    def test_os_error_during_logging_does_not_break_response(self, client, tmp_path):
        c, mock, proxy = client
        mock.proxy.return_value = MagicMock(
            status_code=200,
            body={"results": [{"index": 0, "relevance_score": 0.95}]},
            elapsed_ms=5.0,
        )
        with patch.object(
            proxy._capture_logger, "log", side_effect=OSError("disk full")
        ):
            resp = c.post(
                "/v1/rerank",
                json={"model": "m", "query": "q", "documents": ["doc"]},
            )
        assert resp.status_code == 200
        assert resp.json()["results"][0]["relevance_score"] == 0.95

    def test_os_error_during_error_logging_does_not_break_502(self, client, tmp_path):
        c, mock, proxy = client
        mock.proxy.side_effect = Exception("connection refused")
        with patch.object(
            proxy._capture_logger, "log_error", side_effect=OSError("disk full")
        ):
            resp = c.post(
                "/v1/rerank",
                json={"model": "m", "query": "q", "documents": ["doc"]},
            )
        assert resp.status_code == 502
        assert resp.json()["error"] == "upstream unavailable"

    def test_os_error_during_upstream_error_log_does_not_break_404(
        self, client, tmp_path
    ):
        c, mock, proxy = client
        mock.proxy.return_value = MagicMock(
            status_code=404,
            body={"error": "not found"},
            elapsed_ms=5.0,
        )
        with patch.object(
            proxy._capture_logger, "log", side_effect=OSError("disk full")
        ):
            resp = c.post(
                "/v1/rerank",
                json={"model": "m", "query": "q", "documents": ["doc"]},
            )
        assert resp.status_code == 404

    def test_os_error_during_validation_log_still_returns_422(self, client):
        """A disk fault while logging a rerank validation failure still
        returns 422 — the validation-error log is not routed through
        _log_capture and needs its own OSError suppression."""
        c, _, proxy = client
        with patch.object(
            proxy._capture_logger, "log_error", side_effect=OSError("disk full")
        ):
            resp = c.post("/v1/rerank", json={"model": "m"})
        assert resp.status_code == 422
        assert "rerank" in resp.json()["error"]
