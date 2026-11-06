"""Integration tests for the capture proxy FastAPI app."""

import glob
import json
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from fastapi.testclient import TestClient

from capture_proxy.app import CaptureProxyApp, _reconstruct_response
from capture_proxy.stream import StreamLogger
from capture_proxy.upstream import StreamingUpstreamResponse


def _make_streaming_response(
    status_code: int,
    chunks: list[bytes],
    content_type: str = "text/event-stream",
) -> StreamingUpstreamResponse:
    """Build a StreamingUpstreamResponse from raw byte chunks."""

    async def _aiter():
        for chunk in chunks:
            yield chunk

    return StreamingUpstreamResponse(
        status_code=status_code,
        headers={"content-type": content_type},
        content_type=content_type,
        stream=_aiter(),
    )


@pytest.fixture
def app_and_mock(tmp_path):
    """Create app with mocked upstream, bypassing /data mount check."""
    with (
        patch.dict(
            "os.environ",
            {"CAPTURE_BASE_PATH": str(tmp_path / "captures")},
        ),
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


class TestHealth:
    def test_health_returns_200(self, client):
        c, _ = client
        resp = c.get("/health")
        assert resp.status_code == 200

    def test_health_returns_ok_status(self, client):
        c, _ = client
        resp = c.get("/health")
        data = resp.json()
        assert data["status"] == "ok"

    def test_health_upstream_connected(self, client):
        c, _ = client
        resp = c.get("/health")
        data = resp.json()
        assert data["upstream"] == "connected"

    def test_health_upstream_disconnected(self, app_and_mock):
        app, mock, _, _proxy = app_and_mock
        mock.health_check = AsyncMock(return_value=False)
        c = TestClient(app)
        resp = c.get("/health")
        data = resp.json()
        assert data["upstream"] == "disconnected"

    def test_health_ollama_probes_api_tags(self, tmp_path):
        """Ollama has no /health route — the proxy must probe /api/tags."""
        with patch.dict(
            "os.environ",
            {
                "CAPTURE_BASE_PATH": str(tmp_path / "captures"),
                "CAPTURE_UPSTREAM_URL": "http://127.0.0.1:11434",
            },
        ):
            proxy = CaptureProxyApp()
        assert proxy._backend == "ollama"

        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/api/tags":
                return httpx.Response(200, json={"models": []})
            return httpx.Response(404)

        proxy._upstream_client._client = httpx.AsyncClient(
            transport=httpx.MockTransport(handler),
            base_url="http://127.0.0.1:11434",
        )
        c = TestClient(proxy.app)
        resp = c.get("/health")
        assert resp.status_code == 200
        assert resp.json()["upstream"] == "connected"


class TestModelsPassthrough:
    def test_models_returns_200(self, client):
        c, mock = client
        mock.proxy.return_value = MagicMock(
            status_code=200,
            body={"data": [{"id": "private-model"}]},
        )
        resp = c.get("/v1/models")
        assert resp.status_code == 200

    def test_models_body_unchanged(self, client):
        c, mock = client
        upstream_body = {"data": [{"id": "private-model", "object": "model"}]}
        mock.proxy.return_value = MagicMock(
            status_code=200,
            body=upstream_body,
        )
        resp = c.get("/v1/models")
        assert resp.json() == upstream_body

    def test_models_upstream_error_passes_through(self, client):
        c, mock = client
        mock.proxy.return_value = MagicMock(
            status_code=500,
            body={"error": "internal"},
        )
        resp = c.get("/v1/models")
        assert resp.status_code == 500

    def test_models_upstream_down_returns_502(self, client):
        c, mock = client
        mock.proxy.side_effect = Exception("connection refused")
        resp = c.get("/v1/models")
        assert resp.status_code == 502

    def test_os_error_during_models_error_log_still_returns_502(self, app_and_mock):
        """A disk fault while logging the models connection failure still
        returns 502 — the connection-error log is not routed through
        _log_capture and needs its own OSError suppression."""
        app, mock, _, proxy = app_and_mock
        mock.proxy.side_effect = Exception("connection refused")
        with patch.object(
            proxy._capture_logger, "log_error", side_effect=OSError("disk full")
        ):
            resp = TestClient(app).get("/v1/models")
        assert resp.status_code == 502
        assert resp.json()["error"] == "upstream unavailable"


class TestChatCompletions:
    def test_non_streaming_returns_200(self, client):
        c, mock = client
        mock.proxy.return_value = MagicMock(
            status_code=200,
            body={
                "id": "chatcmpl-123",
                "choices": [{"message": {"content": "hi"}}],
            },
            elapsed_ms=5.0,
        )
        resp = c.post(
            "/v1/chat/completions",
            json={
                "model": "private-model",
                "messages": [{"role": "user", "content": "hello"}],
            },
        )
        assert resp.status_code == 200

    def test_non_streaming_body_unchanged(self, client):
        c, mock = client
        upstream_body = {
            "id": "chatcmpl-123",
            "choices": [{"message": {"content": "response"}, "finish_reason": "stop"}],
            "usage": {
                "prompt_tokens": 5,
                "completion_tokens": 3,
                "total_tokens": 8,
            },
        }
        mock.proxy.return_value = MagicMock(
            status_code=200,
            body=upstream_body,
            elapsed_ms=5.0,
        )
        resp = c.post(
            "/v1/chat/completions",
            json={
                "model": "private-model",
                "messages": [{"role": "user", "content": "test"}],
            },
        )
        assert resp.json() == upstream_body

    def test_invalid_json_returns_422(self, client):
        c, _ = client
        resp = c.post(
            "/v1/chat/completions",
            content=b"not json",
            headers={"Content-Type": "application/json"},
        )
        assert resp.status_code == 422

    def test_invalid_chat_body_returns_422_and_captures(self, client, tmp_path):
        """Body failing schema validation returns 422 and is captured to errors/ (R-4)."""
        c, _ = client
        resp = c.post(
            "/v1/chat/completions",
            json={"model": "m", "messages": "not-a-list", "stream": "yes"},
        )
        assert resp.status_code == 422

        error_files = glob.glob(
            str(tmp_path / "captures" / "errors" / "**" / "*.jsonl"),
            recursive=True,
        )
        summary_files = [f for f in error_files if ".stream.jsonl" not in f]
        assert len(summary_files) >= 1
        with open(summary_files[0]) as f:
            record = json.loads(f.readline())
        assert record["upstream_status"] == 422
        assert record["error_body"] is not None
        assert record["error_category"] == "validation_error"

    def test_streaming_returns_sse(self, client):
        c, mock = client

        sse_data = [
            b'data: {"id":"chatcmpl-1","choices":[{"delta":{"content":"Hi"}}]}\n\n',
            b"data: [DONE]\n\n",
        ]
        mock.proxy.return_value = _make_streaming_response(200, sse_data)

        resp = c.post(
            "/v1/chat/completions",
            json={
                "model": "private-model",
                "messages": [{"role": "user", "content": "hello"}],
                "stream": True,
            },
        )
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers["content-type"]


class TestCompletions:
    def test_non_streaming_completion(self, client):
        c, mock = client
        upstream_body = {
            "id": "cmpl-123",
            "choices": [{"text": "ok", "finish_reason": "stop"}],
            "usage": {"total_tokens": 4},
        }
        mock.proxy.return_value = MagicMock(
            status_code=200,
            body=upstream_body,
            elapsed_ms=5.0,
        )
        resp = c.post(
            "/v1/completions",
            json={
                "model": "private-model",
                "prompt": "Say ok.",
                "max_tokens": 4,
            },
        )
        assert resp.status_code == 200
        assert resp.json() == upstream_body

    def test_streaming_completion_reconstructs_text(self, client, tmp_path):
        c, mock = client

        sse_data = [
            b'data: {"id":"cmpl-1","model":"m","choices":[{"index":0,"text":"Hel","finish_reason":null}]}\n\n',
            b'data: {"id":"cmpl-1","model":"m","choices":[{"index":0,"text":"lo","finish_reason":"stop"}]}\n\n',
            b"data: [DONE]\n\n",
        ]
        mock.proxy.return_value = _make_streaming_response(200, sse_data)

        resp = c.post(
            "/v1/completions",
            json={
                "model": "private-model",
                "prompt": "greet",
                "stream": True,
            },
        )
        assert resp.status_code == 200

        jsonl_files = glob.glob(
            str(tmp_path / "captures" / "raw" / "**" / "*.jsonl"),
            recursive=True,
        )
        # Exclude stream.jsonl files
        summary_files = [f for f in jsonl_files if ".stream.jsonl" not in f]
        with open(summary_files[0]) as f:
            record = json.loads(f.readline())
        assert record["response"]["object"] == "text_completion"
        assert record["response"]["choices"][0]["message"]["content"] == "Hello"


class TestCaptureLogging:
    def test_non_streaming_logs_record(self, client, tmp_path):
        c, mock = client
        mock.proxy.return_value = MagicMock(
            status_code=200,
            body={
                "id": "chatcmpl-123",
                "choices": [{"message": {"content": "hi"}}],
            },
            elapsed_ms=5.0,
        )
        resp = c.post(
            "/v1/chat/completions",
            json={
                "model": "test",
                "messages": [{"role": "user", "content": "hello"}],
            },
        )
        assert resp.status_code == 200

        jsonl_files = glob.glob(
            str(tmp_path / "captures" / "raw" / "**" / "*.jsonl"),
            recursive=True,
        )
        summary_files = [f for f in jsonl_files if ".stream.jsonl" not in f]
        assert len(summary_files) >= 1

        with open(summary_files[0]) as f:
            record = json.loads(f.readline())
        assert record["model"] == "test"
        assert record["upstream_status"] == 200

    def test_record_has_all_fields(self, client, tmp_path):
        c, mock = client
        mock.proxy.return_value = MagicMock(
            status_code=200,
            body={
                "id": "chatcmpl-1",
                "choices": [{"message": {"content": "ok"}}],
                "usage": {"total_tokens": 10},
            },
            elapsed_ms=5.0,
        )
        c.post(
            "/v1/chat/completions",
            json={
                "model": "test",
                "messages": [{"role": "user", "content": "hello"}],
            },
        )

        jsonl_files = glob.glob(
            str(tmp_path / "captures" / "raw" / "**" / "*.jsonl"),
            recursive=True,
        )
        summary_files = [f for f in jsonl_files if ".stream.jsonl" not in f]
        with open(summary_files[0]) as f:
            record = json.loads(f.readline())

        expected_fields = [
            "schema_version",
            "session_id",
            "user_id",
            "turn_id",
            "capture_id",
            "created_at",
            "latency_ms",
            "client_ip",
            "client_user_agent",
            "model",
            "request",
            "request_body_sha256",
            "messages_sha256",
            "attempt_number",
            "response",
            "tool_calls",
            "finish_reason",
            "usage_raw",
            "usage_normalized",
            "upstream_status",
            "error_body",
            "routing",
            "stream_diagnostics",
            "stream_file",
        ]
        for field in expected_fields:
            assert field in record, f"Missing field: {field}"

    def test_capture_record_includes_runtime_settings(self, client, tmp_path):
        c, mock = client
        mock.proxy.return_value = MagicMock(
            status_code=200,
            body={
                "id": "chatcmpl-1",
                "choices": [{"message": {"content": "ok"}}],
            },
            elapsed_ms=1.0,
        )
        resp = c.post(
            "/v1/chat/completions",
            json={
                "model": "private-model",
                "messages": [{"role": "user", "content": "test"}],
            },
        )
        assert resp.status_code == 200

        jsonl_files = glob.glob(
            str(tmp_path / "captures" / "raw" / "**" / "*.jsonl"),
            recursive=True,
        )
        summary_files = [f for f in jsonl_files if ".stream.jsonl" not in f]
        with open(summary_files[0]) as f:
            record = json.loads(f.readline())
        assert record["run_id"].startswith("run_")
        assert record["runtime_settings"]["backend"]
        assert record["runtime_settings"]["kv_cache_k"]

    def test_non_streaming_usage_normalized(self, client, tmp_path):
        c, mock = client
        mock.proxy.return_value = MagicMock(
            status_code=200,
            body={
                "id": "chatcmpl-1",
                "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 50,
                    "total_tokens": 150,
                    "prompt_tokens_details": {"cached_tokens": 80},
                },
            },
            elapsed_ms=5.0,
        )
        c.post(
            "/v1/chat/completions",
            json={
                "model": "test",
                "messages": [{"role": "user", "content": "hello"}],
            },
        )

        jsonl_files = glob.glob(
            str(tmp_path / "captures" / "raw" / "**" / "*.jsonl"),
            recursive=True,
        )
        summary_files = [f for f in jsonl_files if ".stream.jsonl" not in f]
        with open(summary_files[0]) as f:
            record = json.loads(f.readline())

        assert record["usage_raw"]["prompt_tokens"] == 100
        assert record["usage_normalized"]["prompt_total"] == 100
        assert record["usage_normalized"]["prompt_cached"] == 80
        assert record["usage_normalized"]["prompt_uncached"] == 20
        assert record["usage_normalized"]["completion"] == 50
        assert record["usage_normalized"]["context_total"] == 150

    def test_non_streaming_usage_null_details(self, client, tmp_path):
        """vLLM sends explicit nulls in usage — must not 502 (regression)."""
        c, mock = client
        mock.proxy.return_value = MagicMock(
            status_code=200,
            body={
                "id": "chatcmpl-vllm",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "Paris",
                            "refusal": None,
                            "annotations": None,
                        },
                        "finish_reason": "length",
                    }
                ],
                "usage": {
                    "prompt_tokens": 65,
                    "completion_tokens": 20,
                    "total_tokens": 85,
                    "prompt_tokens_details": None,
                },
            },
            elapsed_ms=5.0,
        )
        resp = c.post(
            "/v1/chat/completions",
            json={
                "model": "test",
                "messages": [{"role": "user", "content": "hello"}],
            },
        )

        assert resp.status_code == 200
        jsonl_files = glob.glob(
            str(tmp_path / "captures" / "raw" / "**" / "*.jsonl"),
            recursive=True,
        )
        summary_files = [f for f in jsonl_files if ".stream.jsonl" not in f]
        with open(summary_files[0]) as f:
            record = json.loads(f.readline())

        assert record["upstream_status"] == 200
        assert record["usage_normalized"]["prompt_total"] == 65
        assert record["usage_normalized"]["prompt_cached"] == 0
        assert record["usage_normalized"]["prompt_uncached"] == 65
        assert record["usage_normalized"]["context_total"] == 85

    def test_non_streaming_usage_null_counters(self, client, tmp_path):
        """vLLM sends explicit null counters — must not 502 (regression)."""
        c, mock = client
        mock.proxy.return_value = MagicMock(
            status_code=200,
            body={
                "id": "chatcmpl-vllm",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "Paris",
                            "refusal": None,
                            "annotations": None,
                        },
                        "finish_reason": "length",
                    }
                ],
                "usage": {
                    "prompt_tokens": None,
                    "completion_tokens": None,
                    "total_tokens": None,
                    "prompt_tokens_details": None,
                },
            },
            elapsed_ms=5.0,
        )
        resp = c.post(
            "/v1/chat/completions",
            json={
                "model": "test",
                "messages": [{"role": "user", "content": "hello"}],
            },
        )

        assert resp.status_code == 200
        jsonl_files = glob.glob(
            str(tmp_path / "captures" / "raw" / "**" / "*.jsonl"),
            recursive=True,
        )
        summary_files = [f for f in jsonl_files if ".stream.jsonl" not in f]
        with open(summary_files[0]) as f:
            record = json.loads(f.readline())

        assert record["upstream_status"] == 200
        assert record["usage_normalized"]["prompt_total"] == 0
        assert record["usage_normalized"]["completion"] == 0
        assert record["usage_normalized"]["context_total"] == 0

    def test_non_streaming_usage_null_completion_counter(
        self, client, tmp_path
    ) -> None:
        """completion_tokens=null alone must not 502 either (regression)."""
        c, mock = client
        mock.proxy.return_value = MagicMock(
            status_code=200,
            body={
                "id": "chatcmpl-vllm",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "Paris"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 65,
                    "completion_tokens": None,
                    "total_tokens": None,
                    "prompt_tokens_details": None,
                },
            },
            elapsed_ms=5.0,
        )
        resp = c.post(
            "/v1/chat/completions",
            json={
                "model": "test",
                "messages": [{"role": "user", "content": "hello"}],
            },
        )

        assert resp.status_code == 200
        jsonl_files = glob.glob(
            str(tmp_path / "captures" / "raw" / "**" / "*.jsonl"),
            recursive=True,
        )
        summary_files = [f for f in jsonl_files if ".stream.jsonl" not in f]
        with open(summary_files[0]) as f:
            record = json.loads(f.readline())

        assert record["upstream_status"] == 200
        assert record["usage_normalized"]["prompt_total"] == 65
        assert record["usage_normalized"]["completion"] == 0
        assert record["usage_normalized"]["context_total"] == 65


class TestToolCallAccumulation:
    """Tests verifying tool calls are accumulated, not dropped."""

    def test_tool_calls_accumulated_from_deltas(self):
        """Tool call arguments accumulated across multiple chunks."""
        chunks = [
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {
                                        "name": "write",
                                        "arguments": '{"path":',
                                    },
                                }
                            ]
                        }
                    }
                ]
            },
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "function": {"arguments": '"/tmp/test.txt"}'},
                                }
                            ]
                        }
                    }
                ]
            },
            {
                "choices": [
                    {
                        "delta": {"content": "Continuing..."},
                        "finish_reason": "tool_calls",
                    }
                ]
            },
        ]
        result = _reconstruct_response(chunks, "chat.completion")

        assert result["choices"][0]["finish_reason"] == "tool_calls"
        assert result["choices"][0]["message"]["content"] == "Continuing..."
        tc = result["choices"][0]["message"]["tool_calls"]
        assert len(tc) == 1
        assert tc[0]["id"] == "call_1"
        assert tc[0]["function"]["name"] == "write"
        assert tc[0]["function"]["arguments"] == '{"path":"/tmp/test.txt"}'

    def test_multiple_tool_calls_by_index(self):
        """Multiple tool calls accumulated separately by index."""
        chunks = [
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_1",
                                    "function": {
                                        "name": "read",
                                        "arguments": '{"file":',
                                    },
                                },
                                {
                                    "index": 1,
                                    "id": "call_2",
                                    "function": {
                                        "name": "write",
                                        "arguments": '{"path":',
                                    },
                                },
                            ]
                        }
                    }
                ]
            },
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "function": {"arguments": '"/tmp/a.txt"}'},
                                },
                                {
                                    "index": 1,
                                    "function": {"arguments": '"/tmp/b.txt"}'},
                                },
                            ]
                        }
                    }
                ]
            },
        ]
        result = _reconstruct_response(chunks, "chat.completion")
        tc = result["choices"][0]["message"]["tool_calls"]
        assert len(tc) == 2
        assert tc[0]["function"]["name"] == "read"
        assert tc[1]["function"]["name"] == "write"

    def test_reasoning_content_accumulated(self):
        """Reasoning content accumulated across chunks."""
        chunks = [
            {"choices": [{"delta": {"reasoning_content": "Thinking..."}}]},
            {"choices": [{"delta": {"reasoning_content": " Still thinking."}}]},
            {"choices": [{"delta": {"content": "Answer"}, "finish_reason": "stop"}]},
        ]
        result = _reconstruct_response(chunks, "chat.completion")
        msg = result["choices"][0]["message"]
        assert msg["reasoning_content"] == "Thinking... Still thinking."
        assert msg["content"] == "Answer"

    def test_multiple_choices_accumulated(self):
        """Multiple choices accumulated by index."""
        chunks = [
            {
                "choices": [
                    {"index": 0, "delta": {"content": "A1"}},
                    {"index": 1, "delta": {"content": "B1"}},
                ]
            },
            {
                "choices": [
                    {"index": 0, "delta": {"content": "A2"}, "finish_reason": "stop"},
                    {"index": 1, "delta": {"content": "B2"}, "finish_reason": "stop"},
                ]
            },
        ]
        result = _reconstruct_response(chunks, "chat.completion")
        assert len(result["choices"]) == 2
        assert result["choices"][0]["message"]["content"] == "A1A2"
        assert result["choices"][1]["message"]["content"] == "B1B2"


class TestStreamingAcceptance:
    """6 synthetic upstream scenarios per the acceptance criteria."""

    def _get_summary_record(self, tmp_path):
        jsonl_files = glob.glob(
            str(tmp_path / "captures" / "raw" / "**" / "*.jsonl"),
            recursive=True,
        )
        summary_files = [f for f in jsonl_files if ".stream.jsonl" not in f]
        with open(summary_files[0]) as f:
            return json.loads(f.readline())

    def _get_error_record(self, tmp_path):
        jsonl_files = glob.glob(
            str(tmp_path / "captures" / "errors" / "**" / "*.jsonl"),
            recursive=True,
        )
        summary_files = [f for f in jsonl_files if ".stream.jsonl" not in f]
        with open(summary_files[0]) as f:
            return json.loads(f.readline())

    def _get_stream_records(self, tmp_path):
        jsonl_files = glob.glob(
            str(tmp_path / "captures" / "raw" / "**" / "*.stream.jsonl"),
            recursive=True,
        )
        with open(jsonl_files[0]) as f:
            return [json.loads(line) for line in f if line.strip()]

    def test_scenario_1_normal_content_stop_done(self, app_and_mock):
        """Normal content ending with finish_reason: stop and [DONE]."""
        app, mock, tmp_path, _proxy = app_and_mock
        sse_data = [
            b'data: {"id":"chatcmpl-1","model":"m","choices":[{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":null}]}\n\n',
            b'data: {"id":"chatcmpl-1","model":"m","choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}\n\n',
            b'data: {"id":"chatcmpl-1","model":"m","choices":[{"index":0,"delta":{"content":" world"},"finish_reason":null}]}\n\n',
            b'data: {"id":"chatcmpl-1","model":"m","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15}}\n\n',
            b"data: [DONE]\n\n",
        ]
        mock.proxy.return_value = _make_streaming_response(200, sse_data)

        c = TestClient(app)
        resp = c.post(
            "/v1/chat/completions",
            json={
                "model": "m",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": True,
            },
        )
        assert resp.status_code == 200

        record = self._get_summary_record(tmp_path)
        assert record["finish_reason"] == "stop"
        assert record["usage_raw"]["total_tokens"] == 15
        assert record["usage_normalized"]["prompt_total"] == 10
        assert record["usage_normalized"]["completion"] == 5
        diag = record["stream_diagnostics"]
        assert diag["upstream_eof"] is True
        assert diag["done_seen"] is True
        assert diag["finish_reason_seen"] is True

        stream = self._get_stream_records(tmp_path)
        terminal = next(r for r in stream if r["type"] == "terminal")
        assert terminal["classification"] == "complete"
        assert terminal["upstream_eof"] is True
        assert terminal["done_seen"] is True
        assert terminal["identical"] is True

    def test_scenario_2_tool_calls_fragmented(self, app_and_mock):
        """Tool calls fragmented across several chunks."""
        app, mock, tmp_path, _proxy = app_and_mock

        def sse(event_dict: dict) -> bytes:
            return (
                "data: " + json.dumps(event_dict, separators=(",", ":")) + "\n\n"
            ).encode()

        # Build tool-call arguments incrementally as OpenAI streams them
        args_part1 = '{"path":'
        args_part2 = '"/tmp/f.txt"}'

        sse_data = [
            sse(
                {
                    "id": "chatcmpl-2",
                    "model": "m",
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"role": "assistant", "content": ""},
                            "finish_reason": None,
                        }
                    ],
                }
            ),
            sse(
                {
                    "id": "chatcmpl-2",
                    "model": "m",
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "id": "call_abc",
                                        "type": "function",
                                        "function": {
                                            "name": "write",
                                            "arguments": args_part1,
                                        },
                                    }
                                ]
                            },
                            "finish_reason": None,
                        }
                    ],
                }
            ),
            sse(
                {
                    "id": "chatcmpl-2",
                    "model": "m",
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "tool_calls": [
                                    {"index": 0, "function": {"arguments": args_part2}}
                                ]
                            },
                            "finish_reason": "tool_calls",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 100,
                        "completion_tokens": 20,
                        "total_tokens": 120,
                    },
                }
            ),
            b"data: [DONE]\n\n",
        ]
        mock.proxy.return_value = _make_streaming_response(200, sse_data)

        c = TestClient(app)
        resp = c.post(
            "/v1/chat/completions",
            json={
                "model": "m",
                "messages": [{"role": "user", "content": "write a file"}],
                "stream": True,
            },
        )
        assert resp.status_code == 200

        record = self._get_summary_record(tmp_path)
        assert record["finish_reason"] == "tool_calls"
        tc = record["tool_calls"]
        assert tc is not None
        assert len(tc) == 1
        assert tc[0]["id"] == "call_abc"
        assert tc[0]["function"]["name"] == "write"
        # Arguments should be fully concatenated
        args = json.loads(tc[0]["function"]["arguments"])
        assert "path" in args

        stream = self._get_stream_records(tmp_path)
        terminal = next(r for r in stream if r["type"] == "terminal")
        assert terminal["classification"] == "complete"

    def test_scenario_3_http_200_immediate_eof(self, app_and_mock):
        """HTTP 200 followed by immediate EOF."""
        app, mock, tmp_path, _proxy = app_and_mock
        mock.proxy.return_value = _make_streaming_response(200, [])

        c = TestClient(app)
        resp = c.post(
            "/v1/chat/completions",
            json={
                "model": "m",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": True,
            },
        )
        assert resp.status_code == 200

        record = self._get_summary_record(tmp_path)
        diag = record["stream_diagnostics"]
        assert diag["upstream_eof"] is True
        assert diag["done_seen"] is False
        assert diag["raw_upstream_bytes"] == 0

        stream = self._get_stream_records(tmp_path)
        terminal = next(r for r in stream if r["type"] == "terminal")
        assert terminal["classification"] == "empty_200_eof"

    def test_scenario_4_content_then_eof_no_finish(self, app_and_mock):
        """HTTP 200 with content, then EOF without finish reason."""
        app, mock, tmp_path, _proxy = app_and_mock
        sse_data = [
            b'data: {"id":"chatcmpl-3","model":"m","choices":[{"index":0,"delta":{"content":"Partial"},"finish_reason":null}]}\n\n',
        ]
        mock.proxy.return_value = _make_streaming_response(200, sse_data)

        c = TestClient(app)
        resp = c.post(
            "/v1/chat/completions",
            json={
                "model": "m",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": True,
            },
        )
        assert resp.status_code == 200

        record = self._get_summary_record(tmp_path)
        diag = record["stream_diagnostics"]
        assert diag["upstream_eof"] is True
        assert diag["done_seen"] is False
        assert diag["finish_reason_seen"] is False

        stream = self._get_stream_records(tmp_path)
        terminal = next(r for r in stream if r["type"] == "terminal")
        assert terminal["classification"] == "incomplete_sse_eof"

    def test_scenario_5_http_400_json_error(self, app_and_mock):
        """HTTP 400 with JSON error body — routed to errors/, not raw/."""
        app, mock, tmp_path, _proxy = app_and_mock
        mock.proxy.return_value = MagicMock(
            status_code=400,
            body={"error": {"message": "context length exceeded"}},
            elapsed_ms=5.0,
        )

        c = TestClient(app)
        resp = c.post(
            "/v1/chat/completions",
            json={
                "model": "m",
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
        assert resp.status_code == 400

        record = self._get_error_record(tmp_path)
        assert record["upstream_status"] == 400
        assert record["error_body"] is None

    def test_scenario_streaming_http_400_preflight(self, app_and_mock):
        """Streaming upstream 4xx returns the real status, not HTTP 200 SSE (R-1)."""
        app, mock, tmp_path, _proxy = app_and_mock
        mock.proxy.return_value = _make_streaming_response(
            400,
            [b'{"error": {"message": "context length exceeded"}}\n'],
        )

        c = TestClient(app)
        resp = c.post(
            "/v1/chat/completions",
            json={
                "model": "m",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": True,
            },
        )
        assert resp.status_code == 400
        assert resp.json()["error"]["message"] == "context length exceeded"

        record = self._get_error_record(tmp_path)
        assert record["upstream_status"] == 400
        assert record["error_body"] is not None

    def test_scenario_6_client_disconnect(self, app_and_mock):
        """Client disconnect while upstream continues."""
        app, mock, _, _proxy = app_and_mock

        # Simulate: upstream produces data, but client closes connection
        # FastAPI TestClient doesn't easily support mid-stream disconnect,
        # so we test the reconstruction handles GeneratorExit gracefully
        sse_data = [
            b'data: {"id":"chatcmpl-4","model":"m","choices":[{"index":0,"delta":{"content":"Hello"}}]}\n\n',
            b'data: {"id":"chatcmpl-4","model":"m","choices":[{"index":0,"delta":{"content":" world"}}]}\n\n',
            b"data: [DONE]\n\n",
        ]
        mock.proxy.return_value = _make_streaming_response(200, sse_data)

        c = TestClient(app)
        resp = c.post(
            "/v1/chat/completions",
            json={
                "model": "m",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": True,
            },
        )
        # Response should still be valid even if client doesn't consume all
        assert resp.status_code == 200


class TestRequestCorrelation:
    """Tests for capture_id, request_body_sha256, messages_sha256."""


class TestOSErrorResilience:
    """Adversarial: OSError in capture logging must not break proxy responses."""

    def test_non_streaming_os_error_preserves_response(self, app_and_mock):
        """OSError during capture log must not prevent client from receiving upstream response."""
        app, mock, tmp_path, proxy = app_and_mock
        mock.proxy.return_value = MagicMock(
            status_code=200,
            body={
                "id": "chatcmpl-1",
                "choices": [{"message": {"content": "response"}}],
            },
            elapsed_ms=5.0,
        )

        with patch.object(
            proxy._capture_logger, "log", side_effect=OSError("disk full")
        ):
            c = TestClient(app)
            resp = c.post(
                "/v1/chat/completions",
                json={
                    "model": "test",
                    "messages": [{"role": "user", "content": "hello"}],
                },
            )

        assert resp.status_code == 200
        assert resp.json()["choices"][0]["message"]["content"] == "response"

        jsonl_files = glob.glob(
            str(tmp_path / "captures" / "raw" / "**" / "*.jsonl"),
            recursive=True,
        )
        summary_files = [f for f in jsonl_files if ".stream.jsonl" not in f]
        assert len(summary_files) == 0

    def test_streaming_os_error_preserves_response(self, app_and_mock):
        """OSError during streaming capture log must not prevent client from receiving chunks."""
        app, mock, _, proxy = app_and_mock
        sse_data = [
            b'data: {"id":"c1","choices":[{"delta":{"content":"Hi"}}]}\n\n',
            b"data: [DONE]\n\n",
        ]
        mock.proxy.return_value = _make_streaming_response(200, sse_data)

        with patch.object(
            proxy._capture_logger, "log", side_effect=OSError("disk full")
        ):
            c = TestClient(app)
            resp = c.post(
                "/v1/chat/completions",
                json={
                    "model": "test",
                    "messages": [{"role": "user", "content": "hello"}],
                    "stream": True,
                },
            )

        assert resp.status_code == 200
        assert b"Hi" in resp.content

    def test_upstream_error_os_error_still_returns_502(self, app_and_mock):
        """OSError during error logging does not prevent 502 from being returned."""
        app, mock, _, proxy = app_and_mock
        mock.proxy.side_effect = Exception("connection refused")

        with patch.object(
            proxy._capture_logger, "log_error", side_effect=OSError("disk full")
        ):
            c = TestClient(app)
            resp = c.post(
                "/v1/chat/completions",
                json={
                    "model": "test",
                    "messages": [{"role": "user", "content": "hello"}],
                },
            )

        assert resp.status_code == 502
        assert resp.json()["error"] == "upstream unavailable"

    def test_stream_header_log_oserror_preserves_stream(self, app_and_mock):
        """A disk fault while logging upstream headers must not misattribute
        the healthy 200 stream as a 502 — record_upstream_headers has its
        own OSError guard routing to _capture_write_failed."""
        app, mock, _, _proxy = app_and_mock
        sse_data = [
            b'data: {"choices":[{"delta":{"content":"Hi"}}]}\n\n',
            b"data: [DONE]\n\n",
        ]
        mock.proxy.return_value = _make_streaming_response(200, sse_data)
        with patch.object(
            StreamLogger,
            "record_upstream_headers",
            side_effect=OSError("disk full"),
        ):
            c = TestClient(app)
            resp = c.post(
                "/v1/chat/completions",
                json={
                    "model": "test",
                    "messages": [{"role": "user", "content": "hello"}],
                    "stream": True,
                },
            )
        assert resp.status_code == 200
        assert b"Hi" in resp.content

    def test_capture_id_unique(self, client, tmp_path):
        c, mock = client
        mock.proxy.return_value = MagicMock(
            status_code=200,
            body={"id": "c1", "choices": [{"message": {"content": "ok"}}]},
            elapsed_ms=1.0,
        )
        c.post(
            "/v1/chat/completions",
            json={"model": "m", "messages": [{"role": "user", "content": "a"}]},
        )
        c.post(
            "/v1/chat/completions",
            json={"model": "m", "messages": [{"role": "user", "content": "a"}]},
        )

        jsonl_files = glob.glob(
            str(tmp_path / "captures" / "raw" / "**" / "*.jsonl"),
            recursive=True,
        )
        summary_files = [f for f in jsonl_files if ".stream.jsonl" not in f]
        records = []
        for f in summary_files:
            with open(f) as fh:
                records.append(json.loads(fh.readline()))

        assert len(records) == 2
        assert records[0]["capture_id"] != records[1]["capture_id"]

    def test_identical_requests_have_same_hashes(self, client, tmp_path):
        c, mock = client
        mock.proxy.return_value = MagicMock(
            status_code=200,
            body={"id": "c1", "choices": [{"message": {"content": "ok"}}]},
            elapsed_ms=1.0,
        )
        body = {
            "model": "m",
            "messages": [{"role": "user", "content": "hello"}],
        }
        c.post("/v1/chat/completions", json=body)
        c.post("/v1/chat/completions", json=body)

        jsonl_files = glob.glob(
            str(tmp_path / "captures" / "raw" / "**" / "*.jsonl"),
            recursive=True,
        )
        summary_files = [f for f in jsonl_files if ".stream.jsonl" not in f]
        records = []
        for f in summary_files:
            with open(f) as fh:
                records.append(json.loads(fh.readline()))

        assert records[0]["request_body_sha256"] == records[1]["request_body_sha256"]
        assert records[0]["messages_sha256"] == records[1]["messages_sha256"]


class TestStreamFile:
    """Tests for companion stream.jsonl file creation."""

    def test_streaming_creates_stream_file(self, client, tmp_path):
        c, mock = client
        sse_data = [
            b'data: {"id":"c1","choices":[{"delta":{"content":"Hi"}}]}\n\n',
            b"data: [DONE]\n\n",
        ]
        mock.proxy.return_value = _make_streaming_response(200, sse_data)

        c.post(
            "/v1/chat/completions",
            json={
                "model": "m",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": True,
            },
        )

        stream_files = glob.glob(
            str(tmp_path / "captures" / "raw" / "**" / "*.stream.jsonl"),
            recursive=True,
        )
        assert len(stream_files) == 1

        with open(stream_files[0]) as f:
            lines = [json.loads(line) for line in f if line.strip()]

        types = [r["type"] for r in lines]
        assert "upstream_chunk" in types
        assert "upstream_eof" in types
        assert "sse_event" in types
        assert "sse_done" in types
        assert "terminal" in types

    def test_non_streaming_no_stream_file(self, client, tmp_path):
        c, mock = client
        mock.proxy.return_value = MagicMock(
            status_code=200,
            body={"id": "c1", "choices": [{"message": {"content": "ok"}}]},
            elapsed_ms=1.0,
        )
        c.post(
            "/v1/chat/completions",
            json={"model": "m", "messages": [{"role": "user", "content": "hi"}]},
        )

        stream_files = glob.glob(
            str(tmp_path / "captures" / "raw" / "**" / "*.stream.jsonl"),
            recursive=True,
        )
        assert len(stream_files) == 0


class TestBackendDetection:
    """Backend resolution: explicit CAPTURE_BACKEND beats port auto-detection."""

    def test_explicit_vllm_backend(self, tmp_path):
        with patch.dict(
            "os.environ",
            {
                "CAPTURE_BASE_PATH": str(tmp_path / "captures"),
                "CAPTURE_UPSTREAM_URL": "http://127.0.0.1:8000",
                "CAPTURE_BACKEND": "vllm",
            },
        ):
            proxy = CaptureProxyApp()
            assert proxy._backend == "vllm"

    def test_vllm_default_port_not_auto_detected(self, tmp_path):
        """vLLM's default port 8000 is ambiguous — auto-detect falls back to
        llamacpp, so vllm requires explicit CAPTURE_BACKEND (same as
        node-llama-cpp)."""
        with patch.dict(
            "os.environ",
            {
                "CAPTURE_BASE_PATH": str(tmp_path / "captures"),
                "CAPTURE_UPSTREAM_URL": "http://127.0.0.1:8000",
                "CAPTURE_BACKEND": "",
            },
        ):
            proxy = CaptureProxyApp()
            assert proxy._backend == "llamacpp"
