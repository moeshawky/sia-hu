"""
End-to-end capture contract tests: task, stream body, blobs, index, schema.

Each test drives a real HTTP request through the FastAPI app with a mocked
upstream and then reads what actually landed on disk. The point is to
assert the persisted artifact, not an in-memory field, because every
defect in this file's scope was a persistence defect.

Invariants exercised:
  - ``task`` is non-null on every capture written with the model-task map
    OFF (the configuration that produced null in 100% of the corpus)
  - a streamed capture's ``request_start`` line carries the request body,
    so the stream file is analyzable standalone
  - every hash on a new record resolves against the blob store
  - index lines carry the task label and the stream path, and declare
    their dedup key
  - schema.json declares the version and the tri-state field classes
"""

from __future__ import annotations

import glob
import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from capture_proxy.app import CaptureProxyApp
from capture_proxy.upstream import StreamingUpstreamResponse

CHAT_RESPONSE = {
    "id": "chatcmpl-1",
    "object": "chat.completion",
    "created": 1700000000,
    "model": "test-model",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "ok"},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
}

EMBED_RESPONSE = {
    "object": "list",
    "model": "test-model",
    "data": [{"object": "embedding", "index": 0, "embedding": [0.1, 0.2]}],
    "usage": {"prompt_tokens": 2, "total_tokens": 2},
}

RERANK_RESPONSE = {
    "id": "rerank-1",
    "model": "test-model",
    "results": [{"index": 0, "relevance_score": 0.9, "document": {"text": "Paris..."}}],
}


@pytest.fixture
def proxy(tmp_path):
    """Build an app with a mocked upstream and an isolated capture root."""
    with patch.dict(
        "os.environ",
        {
            "CAPTURE_BASE_PATH": str(tmp_path / "captures"),
            "CAPTURE_UPSTREAM_URL": "http://127.0.0.1:8001",
            "CAPTURE_BACKEND": "llamacpp",
        },
    ):
        app = CaptureProxyApp()
        mock_upstream = MagicMock()
        mock_upstream.health_check = AsyncMock(return_value=True)
        mock_upstream.proxy = AsyncMock()
        app._upstream_client = mock_upstream
        yield app, mock_upstream, tmp_path / "captures"


def _records(base: str, category: str = "raw") -> list[dict]:
    """Load every summary record (stream sidecars excluded)."""
    out: list[dict] = []
    for path in sorted(glob.glob(f"{base}/{category}/**/*.jsonl", recursive=True)):
        if path.endswith(".stream.jsonl"):
            continue
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    out.append(json.loads(line))
    return out


class TestTaskIsPopulated:
    """P0: `task` must be non-null with the model-task map OFF."""

    def test_chat_task_is_populated_without_model_tasks(self, proxy) -> None:
        app, mock, base = proxy
        mock.proxy = AsyncMock(
            return_value=MagicMock(
                status_code=200,
                headers={"content-type": "application/json"},
                content_type="application/json",
                body=CHAT_RESPONSE,
                stream=None,
            )
        )
        with TestClient(app.app) as client:
            client.post(
                "/v1/chat/completions",
                json={
                    "model": "test-model",
                    "messages": [{"role": "user", "content": "Hello!"}],
                },
            )
        records = _records(str(base))
        assert len(records) == 1
        assert records[0]["task"] is not None
        assert records[0]["task"] == "chat.single_shot"
        assert records[0]["task_rule"] == "short_no_tools_chat"

    def test_agent_session_is_agentic(self, proxy) -> None:
        app, mock, base = proxy
        mock.proxy = AsyncMock(
            return_value=MagicMock(
                status_code=200,
                headers={"content-type": "application/json"},
                content_type="application/json",
                body=CHAT_RESPONSE,
                stream=None,
            )
        )
        with TestClient(app.app) as client:
            client.post(
                "/v1/chat/completions",
                json={
                    "model": "test-model",
                    "messages": [{"role": "user", "content": "Hello!"}],
                    "tools": [{"type": "function", "function": {"name": "searx"}}],
                },
            )
        assert _records(str(base))[0]["task"] == "chat.agentic"

    def test_embedding_capture_is_labelled(self, proxy) -> None:
        app, mock, base = proxy
        mock.proxy = AsyncMock(
            return_value=MagicMock(
                status_code=200,
                headers={"content-type": "application/json"},
                content_type="application/json",
                body=EMBED_RESPONSE,
                stream=None,
            )
        )
        with TestClient(app.app) as client:
            client.post(
                "/v1/embeddings",
                json={"model": "test-model", "input": ["a", "b"]},
            )
        records = _records(str(base))
        assert records[0]["task"] == "embedding.batch"
        assert records[0]["request_type"] == "embedding"

    def test_rerank_capture_is_labelled(self, proxy) -> None:
        app, mock, base = proxy
        mock.proxy = AsyncMock(
            return_value=MagicMock(
                status_code=200,
                headers={"content-type": "application/json"},
                content_type="application/json",
                body=RERANK_RESPONSE,
                stream=None,
            )
        )
        with TestClient(app.app) as client:
            client.post(
                "/v1/rerank",
                json={
                    "model": "test-model",
                    "query": "capital of France?",
                    "documents": ["Paris is the capital.", "London is the capital."],
                },
            )
        records = _records(str(base))
        assert records[0]["task"] == "rerank"
        assert records[0]["request_type"] == "rerank"

    def test_operator_map_outranks_structural(self, proxy) -> None:
        """An explicit operator statement beats inference."""
        app, mock, base = proxy
        with patch.dict(
            "os.environ",
            {"CAPTURE_MODEL_TASKS": '{"test-model": "operator-declared"}'},
        ):
            mock.proxy = AsyncMock(
                return_value=MagicMock(
                    status_code=200,
                    headers={"content-type": "application/json"},
                    content_type="application/json",
                    body=CHAT_RESPONSE,
                    stream=None,
                )
            )
            with TestClient(app.app) as client:
                client.post(
                    "/v1/chat/completions",
                    json={
                        "model": "test-model",
                        "messages": [{"role": "user", "content": "Hello!"}],
                    },
                )
        record = _records(str(base))[0]
        assert record["task"] == "operator-declared"
        assert record["task_rule"] == "operator_model_task_map"


class TestStreamCaptureIsSelfDescribing:
    """P0: a stream capture must carry its request body."""

    @staticmethod
    def _streaming_upstream(chunks: list[bytes]) -> StreamingUpstreamResponse:
        async def _aiter():
            for chunk in chunks:
                yield chunk

        return StreamingUpstreamResponse(
            status_code=200,
            headers={"content-type": "text/event-stream"},
            content_type="text/event-stream",
            stream=_aiter(),
        )

    def test_request_start_carries_the_body(self, proxy) -> None:
        app, mock, base = proxy
        sse = [
            b'data: {"choices":[{"index":0,"delta":{"content":"ok"},'
            b'"finish_reason":"stop"}]}\n\n',
            b"data: [DONE]\n\n",
        ]
        mock.proxy = AsyncMock(return_value=self._streaming_upstream(sse))
        body = {
            "model": "test-model",
            "messages": [{"role": "user", "content": "stream me"}],
            "stream": True,
        }
        with TestClient(app.app) as client:
            client.post("/v1/chat/completions", json=body)

        stream_files = glob.glob(f"{base}/raw/**/*.stream.jsonl", recursive=True)
        assert len(stream_files) == 1
        with open(stream_files[0], encoding="utf-8") as handle:
            first = json.loads(handle.readline())

        assert first["type"] == "request_start"
        # The defect: this key was absent, leaving only a hash and a count.
        assert "request_body" in first
        assert first["request_body"]["messages"][0]["content"] == "stream me"
        assert first["task"] == "chat.single_shot"

    def test_stream_record_is_indexed_with_its_path(self, proxy) -> None:
        app, mock, base = proxy
        sse = [
            b'data: {"choices":[{"index":0,"delta":{"content":"ok"},'
            b'"finish_reason":"stop"}]}\n\n',
            b"data: [DONE]\n\n",
        ]
        mock.proxy = AsyncMock(return_value=self._streaming_upstream(sse))
        with TestClient(app.app) as client:
            client.post(
                "/v1/chat/completions",
                json={
                    "model": "test-model",
                    "messages": [{"role": "user", "content": "hi"}],
                    "stream": True,
                },
            )
        record = _records(str(base))[0]
        assert record["stream_file"]
        assert record["stream_file"].endswith(".stream.jsonl")


class TestHashesResolve:
    """P1: an emitted hash must resolve to bytes."""

    def test_both_hashes_resolve_against_the_blob_store(self, proxy) -> None:
        app, mock, base = proxy
        mock.proxy = AsyncMock(
            return_value=MagicMock(
                status_code=200,
                headers={"content-type": "application/json"},
                content_type="application/json",
                body=CHAT_RESPONSE,
                stream=None,
            )
        )
        with TestClient(app.app) as client:
            client.post(
                "/v1/chat/completions",
                json={
                    "model": "test-model",
                    "messages": [{"role": "user", "content": "Hello!"}],
                },
            )
        record = _records(str(base))[0]
        for field in ("request_body_sha256", "messages_sha256"):
            digest = record[field]
            blob = base / "blobs" / "sha256" / digest[:2] / digest
            assert blob.is_file(), f"{field} does not resolve: {digest}"
            assert blob.stat().st_size > 0

    def test_embedding_hashes_resolve_too(self, proxy) -> None:
        """Embeddings carry no messages array; their digests are the input.

        ``messages_sha256`` and ``input_sha256`` are both the digest of
        ``request["input"]`` for this request type. Writing only the request
        body left both handles dangling on every embedding capture — the
        same defect the blob store exists to close, applied one request type
        over. Chat coverage cannot catch it because chat has a messages array.
        """
        app, mock, base = proxy
        mock.proxy = AsyncMock(
            return_value=MagicMock(
                status_code=200,
                headers={"content-type": "application/json"},
                content_type="application/json",
                body=EMBED_RESPONSE,
                stream=None,
            )
        )
        with TestClient(app.app) as client:
            resp = client.post(
                "/v1/embeddings",
                json={"model": "test-model", "input": ["a", "b"]},
            )
        assert resp.status_code == 200
        record = _records(str(base))[0]
        assert record["request_type"] == "embedding"
        for field in ("request_body_sha256", "messages_sha256", "input_sha256"):
            digest = record[field]
            blob = base / "blobs" / "sha256" / digest[:2] / digest
            assert blob.is_file(), (
                f"{field} does not resolve: {digest} — embedding inputs are "
                "not reaching the blob store"
            )
            assert blob.stat().st_size > 0
        # Both digests must be the same document, or one of them is lying.
        assert record["messages_sha256"] == record["input_sha256"]

    def test_blob_store_can_be_switched_off(self) -> None:
        """CAPTURE_BLOB_STORE=off disables writes without losing captures."""
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp) / "captures"
            with patch.dict(
                "os.environ",
                {
                    "CAPTURE_BASE_PATH": str(base),
                    "CAPTURE_UPSTREAM_URL": "http://127.0.0.1:8001",
                    "CAPTURE_BACKEND": "llamacpp",
                    "CAPTURE_BLOB_STORE": "off",
                },
            ):
                app = CaptureProxyApp()
                assert app._blob_store.enabled is False
                mock_upstream = MagicMock()
                mock_upstream.health_check = AsyncMock(return_value=True)
                mock_upstream.proxy = AsyncMock(
                    return_value=MagicMock(
                        status_code=200,
                        headers={"content-type": "application/json"},
                        content_type="application/json",
                        body=CHAT_RESPONSE,
                        stream=None,
                    )
                )
                app._upstream_client = mock_upstream
                with TestClient(app.app) as client:
                    client.post(
                        "/v1/chat/completions",
                        json={
                            "model": "test-model",
                            "messages": [{"role": "user", "content": "Hello!"}],
                        },
                    )
            assert not glob.glob(f"{base}/blobs/**", recursive=True)
            # Losing redundancy must lose no capture.
            assert len(_records(str(base))) == 1


class TestIndexCoverage:
    """P1: the index must carry what the proxy captures."""

    @staticmethod
    def _index_lines(base: str, kind: str = "sessions") -> list[dict]:
        lines: list[dict] = []
        for path in sorted(glob.glob(f"{base}/index/{kind}/*.jsonl")):
            with open(path, encoding="utf-8") as handle:
                for line in handle:
                    if line.strip():
                        lines.append(json.loads(line))
        return lines

    def test_index_carries_task_and_stream_path(self, proxy) -> None:
        app, mock, base = proxy
        sse = [
            b'data: {"choices":[{"index":0,"delta":{"content":"ok"},'
            b'"finish_reason":"stop"}]}\n\n',
            b"data: [DONE]\n\n",
        ]
        mock.proxy = AsyncMock(return_value=TestStreamHelper.make(sse))
        with TestClient(app.app) as client:
            client.post(
                "/v1/chat/completions",
                json={
                    "model": "test-model",
                    "messages": [{"role": "user", "content": "hi"}],
                    "stream": True,
                },
            )
        lines = self._index_lines(str(base))
        assert lines, "no session index line written"
        assert lines[0]["task"] == "chat.single_shot"
        assert lines[0]["stream_file"], "stream path missing from index"
        assert lines[0]["index_key"] == lines[0]["capture_id"]

    def test_embedding_capture_appears_in_the_index(self, proxy) -> None:
        """Non-chat captures must be reachable through the index."""
        app, mock, base = proxy
        mock.proxy = AsyncMock(
            return_value=MagicMock(
                status_code=200,
                headers={"content-type": "application/json"},
                content_type="application/json",
                body=EMBED_RESPONSE,
                stream=None,
            )
        )
        with TestClient(app.app) as client:
            client.post(
                "/v1/embeddings", json={"model": "test-model", "input": "hello"}
            )
        lines = self._index_lines(str(base))
        assert any(line["request_type"] == "embedding" for line in lines)

    def test_both_shards_carry_the_same_declared_key(self, proxy) -> None:
        """One turn, two shards, one declared dedup key."""
        app, mock, base = proxy
        mock.proxy = AsyncMock(
            return_value=MagicMock(
                status_code=200,
                headers={"content-type": "application/json"},
                content_type="application/json",
                body=CHAT_RESPONSE,
                stream=None,
            )
        )
        with TestClient(app.app) as client:
            client.post(
                "/v1/chat/completions",
                headers={"x-session-id": "s1", "x-user-id": "u1"},
                json={
                    "model": "test-model",
                    "messages": [{"role": "user", "content": "hi"}],
                },
            )
        session_lines = self._index_lines(str(base), "sessions")
        user_lines = self._index_lines(str(base), "users")
        assert len(session_lines) == 1
        assert len(user_lines) == 1
        # Same turn seen through two views — the key is what makes the
        # double-count correctable rather than invisible.
        assert session_lines[0]["index_key"] == user_lines[0]["index_key"]
        assert session_lines[0]["shard"] == "sessions"
        assert user_lines[0]["shard"] == "users"

    def test_session_shard_exists_even_for_anonymous_traffic(self, proxy) -> None:
        """Every capture carries a session_id, so every capture indexes."""
        app, mock, base = proxy
        mock.proxy = AsyncMock(
            return_value=MagicMock(
                status_code=200,
                headers={"content-type": "application/json"},
                content_type="application/json",
                body=CHAT_RESPONSE,
                stream=None,
            )
        )
        with TestClient(app.app) as client:
            client.post(
                "/v1/chat/completions",
                json={
                    "model": "test-model",
                    "messages": [{"role": "user", "content": "hi"}],
                },
            )
        assert self._index_lines(str(base), "sessions")


class TestSchemaManifest:
    """P2: the corpus declares its own shape."""

    def test_schema_json_is_written_at_startup(self, proxy) -> None:
        _app, _mock, base = proxy
        schema_path = base / "schema.json"
        assert schema_path.is_file()
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        assert schema["schema_version"] == 6
        assert "request_body_sha256" in schema["required_fields"]

    def test_schema_declares_tri_state_classes(self, proxy) -> None:
        _app, _mock, base = proxy
        schema = json.loads((base / "schema.json").read_text(encoding="utf-8"))
        required = set(schema["required_fields"])
        nullable = set(schema["nullable_not_applicable_fields"])
        optional = set(schema["optional_unrecorded_fields"])
        # The three classes must not overlap: overlapping membership would
        # make the tri-state claim meaningless.
        assert not required & nullable
        assert not required & optional
        assert not nullable & optional
        # The alias fields are the ones that were merely unrecorded.
        assert "request_model_original" in optional
        assert "task" in nullable

    def test_schema_declares_the_dedup_key(self, proxy) -> None:
        _app, _mock, base = proxy
        schema = json.loads((base / "schema.json").read_text(encoding="utf-8"))
        assert schema["index_key"] == "capture_id"

    def test_schema_declares_task_vocabulary(self, proxy) -> None:
        _app, _mock, base = proxy
        schema = json.loads((base / "schema.json").read_text(encoding="utf-8"))
        assert "rerank" in schema["task_vocabulary"]
        assert "code_summarize" in schema["task_vocabulary"]

    def test_schema_is_declared_per_run(self, proxy) -> None:
        """A corpus-level schema is overwritten by the next run.

        That makes it unattributable: a reader cannot tell which run
        produced which records. The per-run copy sits beside the run
        manifest and survives the next boot.
        """
        _app, _mock, base = proxy
        manifests = sorted(glob.glob(f"{base}/runs/*.json"))
        schemas = sorted(glob.glob(f"{base}/runs/*.schema.json"))
        assert manifests, "run manifest missing"
        assert schemas, "no per-run schema declaration was written"
        run_id = json.loads((base / "schema.json").read_text(encoding="utf-8"))[
            "run_id"
        ]
        assert run_id, "corpus-level manifest does not record a run id"
        per_run = json.loads((base / "runs" / f"{run_id}.schema.json").read_text())
        assert per_run["run_id"] == run_id
        # The per-run file must describe the same contract, not a variant.
        assert (
            per_run["task_vocabulary"]
            == json.loads((base / "schema.json").read_text(encoding="utf-8"))[
                "task_vocabulary"
            ]
        )


class TestTriStateOnDisk:
    """null = not applicable, absent = not recorded. Must stay distinct."""

    def test_alias_fields_absent_when_aliasing_off(self, proxy) -> None:
        app, mock, base = proxy
        mock.proxy = AsyncMock(
            return_value=MagicMock(
                status_code=200,
                headers={"content-type": "application/json"},
                content_type="application/json",
                body=CHAT_RESPONSE,
                stream=None,
            )
        )
        with TestClient(app.app) as client:
            client.post(
                "/v1/chat/completions",
                json={
                    "model": "test-model",
                    "messages": [{"role": "user", "content": "hi"}],
                },
            )
        record = _records(str(base))[0]
        # Aliasing is off, so the original/forwarded pair is not RECORDED,
        # which is a different claim from "not applicable". The key must
        # be absent. Asserting `is None` here once encoded the bug: a
        # Pydantic model_dump emits an unset optional field as null, the
        # same value the nullable class uses, so the two states became
        # indistinguishable and the schema declaration was untrue.
        for key in ("request_model_original", "request_model_forwarded"):
            assert key not in record, (
                f"{key} landed on disk as {record.get(key)!r}; "
                "optional-unrecorded must be absent, not null"
            )
        # The nullable class must still serialize as null; dropping the
        # optional keys must not have collapsed the two classes.
        assert record["embedding_count"] is None


class TestStreamHelper:
    """Namespace for the streaming response factory used above."""

    @staticmethod
    def make(chunks: list[bytes]) -> StreamingUpstreamResponse:
        async def _aiter():
            for chunk in chunks:
                yield chunk

        return StreamingUpstreamResponse(
            status_code=200,
            headers={"content-type": "text/event-stream"},
            content_type="text/event-stream",
            stream=_aiter(),
        )
