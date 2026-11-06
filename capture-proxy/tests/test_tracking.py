"""Session/user tracking and capture index tests.

Covers the end-to-end tracking contract:
  headers → SessionResolver.resolve_user_id → CaptureRecord.user_id
          → CaptureLogger index append → /v1/sessions|users query endpoints.

Edge cases under test:
  EC1: empty x-session-id / x-user-id values fall back (never stored empty)
  EC2: Authorization hash is deterministic; raw key bytes never hit disk
  EC3: header name case variations all match
  EC4: unknown ids return 200 + empty list; index failures never break captures
"""

from __future__ import annotations

import glob
import json
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from capture_proxy.app import CaptureProxyApp
from capture_proxy.logger import CaptureLogger
from capture_proxy.session import ANONYMOUS_USER_ID, SessionResolver
from capture_proxy.upstream import StreamingUpstreamResponse

SECRET_KEY = "Bearer sk-super-secret-value-42"  # noqa: S105  # fake test value


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
    """Create app with mocked upstream on a llamacpp-style URL."""
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
    """Provide TestClient bound to the mocked app."""
    app, mock, _, _proxy = app_and_mock
    return TestClient(app), mock


def _summary_records(tmp_path, category="raw"):
    """Load every summary JSONL record from a capture category."""
    files = glob.glob(
        str(tmp_path / "captures" / category / "**" / "*.jsonl"),
        recursive=True,
    )
    records = []
    for filepath in sorted(files):
        if filepath.endswith(".stream.jsonl"):
            continue
        with open(filepath, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
    return records


def _chat_success_body() -> dict:
    """Minimal OpenAI-compatible chat completion response body."""
    return {
        "choices": [
            {
                "message": {"role": "assistant", "content": "hi"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }


# ---------------------------------------------------------------------------
# Unit: SessionResolver.resolve_user_id
# ---------------------------------------------------------------------------


class TestResolveUserId:
    """Tests for user identity resolution (WHO)."""

    def setup_method(self) -> None:
        self.resolver = SessionResolver()

    def test_header_verbatim_stripped(self) -> None:
        assert self.resolver.resolve_user_id({"x-user-id": "  alice "}) == "alice"

    @pytest.mark.parametrize("name", ["X-User-Id", "x-user-id", "X-USER-ID"])
    def test_header_case_variants(self, name: str) -> None:
        """EC3: all case spellings of the header match."""
        assert self.resolver.resolve_user_id({name: "bob"}) == "bob"

    def test_empty_header_falls_back_to_auth_hash(self) -> None:
        """EC1: empty x-user-id must not be stored as the user id."""
        result = self.resolver.resolve_user_id(
            {"x-user-id": "", "Authorization": SECRET_KEY}
        )
        assert result.startswith("user-")
        assert result != ""

    def test_whitespace_header_falls_back(self) -> None:
        result = self.resolver.resolve_user_id(
            {"x-user-id": "   ", "Authorization": SECRET_KEY}
        )
        assert result.startswith("user-")

    def test_auth_hash_deterministic(self) -> None:
        """EC2: same Authorization → same user id."""
        r1 = self.resolver.resolve_user_id({"Authorization": SECRET_KEY})
        r2 = self.resolver.resolve_user_id({"Authorization": SECRET_KEY})
        assert r1 == r2
        assert r1.startswith("user-")

    def test_auth_hash_differs_from_session_resolution(self) -> None:
        """WHO != WHICH: auth-only clients get distinct session/user ids."""
        headers = {"Authorization": SECRET_KEY}
        assert self.resolver.resolve(headers) != self.resolver.resolve_user_id(headers)

    def test_different_keys_different_hashes(self) -> None:
        h1 = self.resolver.resolve_user_id({"Authorization": "Bearer key-one"})
        h2 = self.resolver.resolve_user_id({"Authorization": "Bearer key-two"})
        assert h1 != h2

    def test_anonymous_fallback(self) -> None:
        assert self.resolver.resolve_user_id({}) == ANONYMOUS_USER_ID

    def test_header_priority_over_auth(self) -> None:
        headers = {"x-user-id": "carol", "Authorization": SECRET_KEY}
        assert self.resolver.resolve_user_id(headers) == "carol"


# ---------------------------------------------------------------------------
# Unit: CaptureLogger index maintenance
# ---------------------------------------------------------------------------


class TestCaptureIndex:
    """Tests for the per-session/per-user index files."""

    @pytest.fixture
    def logger(self, tmp_path):
        return CaptureLogger(base_path=str(tmp_path / "captures"))

    @staticmethod
    def _record(**overrides):
        base = {
            "schema_version": 6,
            "session_id": "sess-1",
            "user_id": "user-alice",
            "turn_id": "turn-1",
            "capture_id": "cap-1",
            "created_at": "2026-08-26T10:00:00+00:00",
            "latency_ms": 1.0,
            "client_ip": "127.0.0.1",
            "client_user_agent": "test",
            "model": "m",
            "request": {},
            "upstream_status": 200,
        }
        base.update(overrides)
        return base

    def test_index_filename_deterministic_and_traversal_safe(self) -> None:
        """Hostile ids produce flat, deterministic, single-component names."""
        hostile = "../../etc/passwd"
        n1 = CaptureLogger.index_filename(hostile)
        n2 = CaptureLogger.index_filename(hostile)
        assert n1 == n2
        # No path separators → the name cannot traverse directories.
        assert "/" not in n1
        assert n1 not in ("..", ".")
        assert n1.endswith(".jsonl")

    def test_log_writes_session_and_user_index_lines(self, logger, tmp_path) -> None:
        path = logger.log(self._record())
        sess_entries = logger.read_index("sessions", "sess-1")
        user_entries = logger.read_index("users", "user-alice")
        assert len(sess_entries) == 1
        assert len(user_entries) == 1
        assert sess_entries[0]["capture_file"] == str(
            path.relative_to(tmp_path / "captures")
        )
        assert user_entries[0]["session_id"] == "sess-1"

    def test_read_index_exact_filter_skips_foreign_lines(
        self, logger, tmp_path
    ) -> None:
        """A foreign line manually placed in the file is filtered out."""
        logger.log(self._record())
        index_file = next((tmp_path / "captures" / "index" / "sessions").iterdir())
        with open(index_file, "a", encoding="utf-8") as f:
            f.write(json.dumps({"session_id": "sess-IMPOSTOR", "ts": "t"}) + "\n")
        entries = logger.read_index("sessions", "sess-1")
        assert len(entries) == 1
        assert entries[0]["session_id"] == "sess-1"

    def test_read_index_missing_id_returns_empty(self, logger) -> None:
        """EC4: unknown id → [] (not an exception)."""
        assert logger.read_index("sessions", "nope") == []
        assert logger.list_index_ids("sessions") == []

    def test_corrupt_index_line_skipped(self, logger, tmp_path) -> None:
        logger.log(self._record())
        index_file = next((tmp_path / "captures" / "index" / "sessions").iterdir())
        with open(index_file, "a", encoding="utf-8") as f:
            f.write('{"session_id": "sess-1", "ts": "ok"}\n')
            f.write("{torn append\n")
        entries = logger.read_index("sessions", "sess-1")
        assert len(entries) == 2

    def test_index_failure_does_not_break_capture(self, tmp_path) -> None:
        """EC4/G-SEC-adjacent: blocked index tree never breaks log()."""
        base = tmp_path / "captures"
        base.mkdir()
        (base / "index").write_text("blocker")  # index dir path is a FILE
        logger = CaptureLogger(base_path=str(base))
        path = logger.log(self._record())
        assert path.exists()

    def test_minimal_error_record_skips_indexing(self, logger) -> None:
        """/v1/models-style error dicts without ids must not raise."""
        path = logger.log_error(
            {
                "turn_id": "t",
                "created_at": "2026-08-26T10:00:00+00:00",
                "error_body": "boom",
            }
        )
        assert path.exists()
        assert logger.read_index("sessions", "t") == []

    def test_list_index_ids_recency_order(self, logger) -> None:
        logger.log(self._record(session_id="s-old", user_id="u-old"))
        time.sleep(0.02)  # distinct mtimes — recency sort must be observable
        logger.log(self._record(session_id="s-new", user_id="u-new"))
        sessions = logger.list_index_ids("sessions")
        users = logger.list_index_ids("users")
        assert [s["id"] for s in sessions] == ["s-new", "s-old"]
        assert [u["id"] for u in users] == ["u-new", "u-old"]
        assert sessions[0]["captures"] == 1


# ---------------------------------------------------------------------------
# Integration: handlers wire user_id end-to-end; endpoints query the index
# ---------------------------------------------------------------------------


class TestTrackingIntegration:
    """End-to-end: request headers → record → index → query endpoint."""

    def test_chat_capture_queryable_by_session_and_user(
        self, client, app_and_mock, tmp_path
    ) -> None:
        c, mock = client
        mock.proxy.return_value = MagicMock(status_code=200, body=_chat_success_body())
        resp = c.post(
            "/v1/chat/completions",
            json={"model": "m", "messages": [{"role": "user", "content": "hi"}]},
            headers={"x-session-id": "sess-abc", "x-user-id": "alice"},
        )
        assert resp.status_code == 200

        records = _summary_records(tmp_path)
        assert len(records) == 1
        record = records[0]
        assert record["session_id"] == "sess-abc"
        assert record["user_id"] == "alice"
        assert record["schema_version"] == 6

        by_session = c.get("/v1/sessions/sess-abc/captures")
        assert by_session.status_code == 200
        body = by_session.json()
        assert body["count"] == 1
        entry = body["captures"][0]
        assert entry["user_id"] == "alice"
        capture_file = tmp_path / "captures" / entry["capture_file"]
        assert capture_file.exists()

        by_user = c.get("/v1/users/alice/captures")
        assert by_user.status_code == 200
        assert by_user.json()["count"] == 1
        assert by_user.json()["captures"][0]["session_id"] == "sess-abc"

    def test_streaming_capture_queryable(self, client, app_and_mock, tmp_path) -> None:
        c, mock = client
        chunks = [
            b'data: {"id":"c1","object":"chat.completion.chunk","choices":'
            b'[{"index":0,"delta":{"role":"assistant","content":"He"}}]}\n\n',
            b'data: {"id":"c1","object":"chat.completion.chunk","choices":'
            b'[{"index":0,"delta":{"content":"y"},"finish_reason":"stop"}]}\n\n',
            b"data: [DONE]\n\n",
        ]
        mock.proxy.return_value = _make_streaming_response(200, chunks)
        resp = c.post(
            "/v1/chat/completions",
            json={
                "model": "m",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": True,
            },
            headers={"x-session-id": "sess-stream", "x-user-id": "streamer"},
        )
        assert resp.status_code == 200

        records = _summary_records(tmp_path)
        assert len(records) == 1
        assert records[0]["user_id"] == "streamer"
        assert records[0]["response"]["choices"][0]["message"]["content"] == "Hey"

        body = c.get("/v1/sessions/sess-stream/captures").json()
        assert body["count"] == 1
        assert body["captures"][0]["category"] == "raw"

    @pytest.mark.parametrize(
        ("endpoint", "payload", "upstream_body"),
        [
            (
                "/v1/embeddings",
                {"model": "m", "input": "hello"},
                {"data": [{"object": "embedding", "index": 0, "embedding": [0.1]}]},
            ),
            ("/v1/rerank", {"model": "m", "query": "q", "documents": ["d"]}, {}),
        ],
    )
    def test_embeddings_and_rerank_carry_user_id(
        self, client, app_and_mock, tmp_path, endpoint, payload, upstream_body
    ) -> None:
        c, mock = client
        mock.proxy.return_value = MagicMock(status_code=200, body=upstream_body or None)
        resp = c.post(
            endpoint, json=payload, headers={"x-session-id": "s9", "x-user-id": "u9"}
        )
        assert resp.status_code == 200
        records = _summary_records(tmp_path)
        assert records
        assert all(r["user_id"] == "u9" for r in records)
        assert c.get("/v1/users/u9/captures").json()["count"] >= 1

    def test_validation_error_record_indexed(
        self, client, app_and_mock, tmp_path
    ) -> None:
        """422 path still persists identity and lands in errors/ + index."""
        c, _mock = client
        resp = c.post(
            "/v1/chat/completions",
            json={"model": "m", "messages": "not-a-list"},
            headers={"x-session-id": "sess-bad", "x-user-id": "careless"},
        )
        assert resp.status_code == 422
        records = _summary_records(tmp_path, "errors")
        assert len(records) == 1
        assert records[0]["user_id"] == "careless"
        body = c.get("/v1/sessions/sess-bad/captures").json()
        assert body["count"] == 1
        assert body["captures"][0]["category"] == "errors"

    def test_unknown_session_returns_empty_200(self, client) -> None:
        """EC4: query for nonexistent ids is 200 + count 0, never 500."""
        c, _mock = client
        assert c.get("/v1/sessions/ghost/captures").json() == {
            "kind": "session",
            "id": "ghost",
            "count": 0,
            "captures": [],
        }
        assert c.get("/v1/users/ghost/captures").json()["count"] == 0

    def test_traversal_session_id_safe(self, client, app_and_mock, tmp_path) -> None:
        """Path-traversal ids cannot escape the index directory."""
        c, _mock = client
        resp = c.get("/v1/sessions/..%2F..%2Fetc%2Fpasswd/captures")
        assert resp.status_code in (200, 404)
        escaped = [
            p
            for p in tmp_path.rglob("*")
            if "passwd" in p.name and "captures" not in p.parts
        ]
        assert escaped == []

    def test_no_raw_authorization_on_disk(self, client, app_and_mock, tmp_path) -> None:
        """G-SEC: raw Authorization bytes never reach any captured file."""
        c, mock = client
        mock.proxy.return_value = MagicMock(status_code=200, body=_chat_success_body())
        resp = c.post(
            "/v1/chat/completions",
            json={"model": "m", "messages": [{"role": "user", "content": "hi"}]},
            headers={"Authorization": SECRET_KEY},
        )
        assert resp.status_code == 200
        secret_bytes = SECRET_KEY.encode()
        leaked = [
            p
            for p in (tmp_path / "captures").rglob("*")
            if p.is_file() and secret_bytes in p.read_bytes()
        ]
        assert leaked == []

        record = _summary_records(tmp_path)[0]
        assert record["user_id"].startswith("user-")
        assert SECRET_KEY not in record["user_id"]

    def test_listing_endpoints_show_ids(self, client, app_and_mock) -> None:
        c, mock = client
        mock.proxy.return_value = MagicMock(status_code=200, body=_chat_success_body())
        c.post(
            "/v1/chat/completions",
            json={"model": "m", "messages": [{"role": "user", "content": "hi"}]},
            headers={"x-session-id": "sess-ls", "x-user-id": "listed"},
        )
        sessions = c.get("/v1/sessions").json()
        assert sessions["kind"] == "session"
        assert "sess-ls" in [s["id"] for s in sessions["ids"]]
        users = c.get("/v1/users").json()
        assert "listed" in [u["id"] for u in users["ids"]]
        assert users["ids"][0]["captures"] >= 1

    def test_anonymous_grouping_without_headers(
        self, client, app_and_mock, tmp_path
    ) -> None:
        c, mock = client
        mock.proxy.return_value = MagicMock(status_code=200, body=_chat_success_body())
        for _ in range(2):
            c.post(
                "/v1/chat/completions",
                json={"model": "m", "messages": [{"role": "user", "content": "hi"}]},
            )
        records = _summary_records(tmp_path)
        assert len(records) == 2
        assert all(r["user_id"] == ANONYMOUS_USER_ID for r in records)
        users = c.get("/v1/users").json()
        assert ANONYMOUS_USER_ID in [u["id"] for u in users["ids"]]
