"""Tests for SseParser, StreamLogger, and classify_terminal."""

import base64
import glob
import hashlib
import json
import os
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.testclient import TestClient

from capture_proxy.stream import SseParser, StreamLogger, classify_terminal


class TestSseParser:
    """Tests for incremental SSE parsing from raw bytes."""

    def test_single_complete_event(self) -> None:
        parser = SseParser()
        raw = b'data: {"choices":[{"delta":{"content":"Hi"}}]}\n\n'
        events = parser.feed(raw)
        assert len(events) == 1
        assert events[0]["json_valid"] is True
        assert events[0]["done"] is False
        assert "Hi" in events[0]["raw_data"]

    def test_done_event(self) -> None:
        parser = SseParser()
        events = parser.feed(b"data: [DONE]\n\n")
        assert len(events) == 1
        assert events[0]["done"] is True
        assert events[0]["raw_data"] == "[DONE]"

    def test_multiple_events_in_one_chunk(self) -> None:
        parser = SseParser()
        raw = (
            b'data: {"choices":[{"delta":{"content":"A"}}]}\n\n'
            b'data: {"choices":[{"delta":{"content":"B"}}]}\n\n'
        )
        events = parser.feed(raw)
        assert len(events) == 2

    def test_split_across_chunks(self) -> None:
        parser = SseParser()
        events1 = parser.feed(b'data: {"choices":[{"delta":{"content":')
        assert len(events1) == 0
        events2 = parser.feed(b'"Hi"}}]}\n\n')
        assert len(events2) == 1
        assert events2[0]["json_valid"] is True

    def test_half_utf8_sequence(self) -> None:
        parser = SseParser()
        # Send first byte of a 2-byte UTF-8 character
        partial = b"data: " + bytes([0xC3])
        parser.feed(partial)
        # Should buffer, not crash
        events2 = parser.feed(b"\xa9\n\n")
        assert len(events2) == 1
        # The split character must decode intact, not U+FFFD U+FFFD
        assert events2[0]["raw_data"] == "\u00e9"
        assert "\ufffd" not in events2[0]["raw_data"]

    def test_split_utf8_inside_quoted_string(self) -> None:
        """Split char inside a JSON string must keep json_valid and content."""
        parser = SseParser()
        parser.feed(b'data: {"conte\xc3')
        events = parser.feed(b'\xa9nt": "x"}\n\n')
        assert len(events) == 1
        assert events[0]["json_valid"] is True
        assert events[0]["raw_data"] == '{"conte\u00e9nt": "x"}'

    def test_flush_finalizes_split_sequence(self) -> None:
        """A character completed before EOF but without a newline is decoded."""
        parser = SseParser()
        parser.feed(b"data: hi \xc3")
        parser.feed(b"\xa9")
        events = parser.flush()
        assert len(events) == 1
        assert events[0]["raw_data"] == "hi \u00e9"

    def test_flush_remaining_buffer(self) -> None:
        parser = SseParser()
        parser.feed(b'data: {"choices":[]}')
        events = parser.flush()
        assert len(events) == 1
        assert events[0]["json_valid"] is True

    def test_flush_empty_buffer(self) -> None:
        parser = SseParser()
        events = parser.flush()
        assert events == []

    def test_invalid_json(self) -> None:
        parser = SseParser()
        events = parser.feed(b"data: not json at all\n\n")
        assert len(events) == 1
        assert events[0]["json_valid"] is False

    def test_finish_reason_detected(self) -> None:
        parser = SseParser()
        raw = b'data: {"choices":[{"finish_reason":"stop"}]}\n\n'
        events = parser.feed(raw)
        assert events[0]["finish_reasons"] == ["stop"]

    def test_usage_detected(self) -> None:
        parser = SseParser()
        raw = b'data: {"usage":{"prompt_tokens":10}}\n\n'
        events = parser.feed(raw)
        assert events[0]["has_usage"] is True

    def test_non_data_lines_ignored(self) -> None:
        parser = SseParser()
        events = parser.feed(b"event: message\ndata: test\n\n")
        # Only "data: test" should be parsed
        assert len(events) == 1

    def test_empty_lines_ignored(self) -> None:
        parser = SseParser()
        events = parser.feed(b"\n\n\ndata: test\n\n")
        assert len(events) == 1

    def test_invalid_utf8_does_not_raise(self) -> None:
        """Corrupt upstream bytes are replaced with U+FFFD, not raised —
        the SSE line structure survives and json_valid flags the loss."""
        parser = SseParser()
        events = parser.feed(b'data: {"content":"\xff\xfe"}\n\n')
        assert len(events) == 1
        assert (
            events[0]["json_valid"] is True
        )  # replacement chars are valid JSON string content
        assert "�" in events[0]["raw_data"]

    def test_invalid_utf8_json_structure_survives(self) -> None:
        """Invalid bytes outside a JSON string break the payload but not
        the parser — the event is yielded with json_valid=False."""
        parser = SseParser()
        events = parser.feed(b"data: \xff\n\n")
        assert len(events) == 1
        assert events[0]["json_valid"] is False

    def test_flush_invalid_utf8_does_not_raise(self) -> None:
        """Undecodable trailing bytes at EOF yield a replacement event."""
        parser = SseParser()
        parser.feed(b"data: \xff")
        events = parser.flush()
        assert len(events) == 1
        assert events[0]["json_valid"] is False

    def test_valid_multibyte_split_across_chunks(self) -> None:
        """A valid multi-byte char split across chunks decodes intact —
        no regression from the invalid-byte fault path."""
        parser = SseParser()
        parser.feed(b'data: {"content":"' + b"\xe2\x82")  # first 2 bytes of €
        events = parser.feed(b'\xac"}\n\n')  # final byte closes string + object
        assert len(events) == 1
        assert events[0]["json_valid"] is True
        assert "€" in events[0]["raw_data"]
        assert "�" not in events[0]["raw_data"]

    def test_decoder_state_clean_after_fault(self) -> None:
        """After an invalid-byte fault the parser resets its decoder — a
        following valid stream decodes cleanly with no stale state."""
        parser = SseParser()
        parser.feed(b"data: \xff\n\n")  # fault — replacement path
        events = parser.feed(b'data: {"content":"ok"}\n\n')
        assert len(events) == 1
        assert events[0]["json_valid"] is True
        assert "ok" in events[0]["raw_data"]


class TestClassifyTerminal:
    """Tests for terminal state classification."""

    def test_complete(self) -> None:
        result = classify_terminal(
            headers_status=200,
            upstream_eof=True,
            done_seen=True,
            finish_reason_seen=True,
            usage_seen=True,
        )
        assert result == "complete"

    def test_empty_200_eof(self) -> None:
        result = classify_terminal(
            headers_status=200,
            upstream_eof=True,
            done_seen=False,
            finish_reason_seen=False,
        )
        assert result == "empty_200_eof"

    def test_http_error(self) -> None:
        result = classify_terminal(
            headers_status=400,
            upstream_eof=True,
        )
        assert result == "http_error_before_stream"

    def test_incomplete_sse_eof(self) -> None:
        result = classify_terminal(
            headers_status=200,
            upstream_eof=False,
            done_seen=False,
            sse_events=[{"json_valid": True, "done": False}],
        )
        assert result == "incomplete_sse_eof"

    def test_upstream_exception(self) -> None:
        result = classify_terminal(
            upstream_exception="connection refused",
        )
        assert result == "upstream_stream_exception"

    def test_downstream_exception(self) -> None:
        result = classify_terminal(
            downstream_exception="client closed",
        )
        assert result == "downstream_forward_exception"

    def test_client_disconnect(self) -> None:
        result = classify_terminal(
            client_disconnected=True,
        )
        assert result == "client_disconnect"

    def test_invalid_sse(self) -> None:
        result = classify_terminal(
            headers_status=200,
            upstream_eof=True,
            sse_events=[
                {"json_valid": False, "done": False},
            ],
        )
        assert result == "invalid_sse"

    def test_valid_finish_missing_done(self) -> None:
        result = classify_terminal(
            headers_status=200,
            upstream_eof=True,
            done_seen=False,
            finish_reason_seen=True,
            sse_events=[{"json_valid": True, "done": False}],
        )
        assert result == "valid_finish_missing_done"

    def test_done_missing_finish(self) -> None:
        result = classify_terminal(
            headers_status=200,
            upstream_eof=True,
            done_seen=True,
            finish_reason_seen=False,
        )
        assert result == "done_missing_finish"


class TestStreamLogger:
    """Tests for companion stream.jsonl writer."""

    def test_creates_file_on_write(self, tmp_path) -> None:
        logger = StreamLogger(
            base_path=str(tmp_path),
            turn_id="test-turn-123",
            created_at="2026-07-23T11:01:32.000000+00:00",
        )
        assert not logger.filepath.exists()
        logger.record_upstream_eof(0.0)
        assert logger.filepath.exists()
        assert logger.filepath.suffix == ".jsonl"
        assert "stream" in logger.filepath.name

    def test_records_upstream_chunk(self, tmp_path) -> None:
        logger = StreamLogger(
            base_path=str(tmp_path),
            turn_id="test-turn",
            created_at="2026-07-23T11:01:32.000000+00:00",
        )
        data = b"hello"
        b64 = base64.b64encode(data).decode()
        logger.record_upstream_chunk(b64, 10.0)

        assert logger.upstream_bytes == 5
        lines = logger.filepath.read_text().strip().split("\n")
        assert len(lines) == 1
        record = json.loads(lines[0])
        assert record["type"] == "upstream_chunk"
        assert record["data_b64"] == b64

    def test_records_upstream_eof(self, tmp_path) -> None:
        logger = StreamLogger(
            base_path=str(tmp_path),
            turn_id="test-turn",
            created_at="2026-07-23T11:01:32.000000+00:00",
        )
        logger.record_upstream_eof(100.0)
        lines = logger.filepath.read_text().strip().split("\n")
        record = json.loads(lines[0])
        assert record["type"] == "upstream_eof"

    def test_records_request_start(self, tmp_path) -> None:
        logger = StreamLogger(
            base_path=str(tmp_path),
            turn_id="test-turn",
            created_at="2026-07-23T11:01:32.000000+00:00",
        )
        logger.record_request_start(
            capture_id="cap-123",
            request_body_sha256="abc123",
            messages_sha256="def456",
            model="qwen",
            messages_count=10,
            stream=True,
            max_completion_tokens=4096,
            request_bytes=525999,
        )
        lines = logger.filepath.read_text().strip().split("\n")
        record = json.loads(lines[0])
        assert record["type"] == "request_start"
        assert record["capture_id"] == "cap-123"
        assert record["request_body_sha256"] == "abc123"
        assert record["messages_sha256"] == "def456"
        assert record["model"] == "qwen"
        assert record["messages"] == 10
        assert record["stream"] is True
        assert record["max_completion_tokens"] == 4096
        assert record["request_bytes"] == 525999

    def test_records_upstream_headers(self, tmp_path) -> None:
        logger = StreamLogger(
            base_path=str(tmp_path),
            turn_id="test-turn",
            created_at="2026-07-23T11:01:32.000000+00:00",
        )
        logger.record_upstream_headers(
            elapsed_ms=15.2,
            status=200,
            content_type="text/event-stream",
            headers={"x-request-id": "req-1"},
        )
        lines = logger.filepath.read_text().strip().split("\n")
        record = json.loads(lines[0])
        assert record["type"] == "upstream_headers"
        assert record["elapsed_ms"] == 15.2
        assert record["status"] == 200
        assert record["content_type"] == "text/event-stream"
        assert record["headers"] == {"x-request-id": "req-1"}

    def test_records_forward_checkpoint(self, tmp_path) -> None:
        logger = StreamLogger(
            base_path=str(tmp_path),
            turn_id="test-turn",
            created_at="2026-07-23T11:01:32.000000+00:00",
        )
        logger.record_forward_checkpoint(
            elapsed_ms=50.0,
            upstream_bytes=100,
            downstream_bytes=100,
            upstream_sha256_partial="aaa",
            downstream_sha256_partial="aaa",
            identical_so_far=True,
        )
        lines = logger.filepath.read_text().strip().split("\n")
        record = json.loads(lines[0])
        assert record["type"] == "forward_checkpoint"
        assert record["identical_so_far"] is True
        assert record["upstream_bytes"] == 100
        assert record["downstream_bytes"] == 100

    def test_records_sse_event(self, tmp_path) -> None:
        logger = StreamLogger(
            base_path=str(tmp_path),
            turn_id="test-turn",
            created_at="2026-07-23T11:01:32.000000+00:00",
        )
        event = {
            "raw_data": '{"choices":[{"finish_reason":"stop"}]}',
            "json_valid": True,
            "finish_reasons": ["stop"],
            "has_usage": False,
            "done": False,
        }
        logger.record_sse_event(event, 50.0)
        assert logger.finish_reasons_seen == ["stop"]
        lines = logger.filepath.read_text().strip().split("\n")
        record = json.loads(lines[0])
        assert record["type"] == "sse_event"

    def test_records_sse_done(self, tmp_path) -> None:
        logger = StreamLogger(
            base_path=str(tmp_path),
            turn_id="test-turn",
            created_at="2026-07-23T11:01:32.000000+00:00",
        )
        event = {
            "raw_data": "[DONE]",
            "json_valid": True,
            "finish_reasons": [],
            "has_usage": False,
            "done": True,
        }
        logger.record_sse_event(event, 60.0)
        assert logger.done_seen is True
        lines = logger.filepath.read_text().strip().split("\n")
        record = json.loads(lines[0])
        assert record["type"] == "sse_done"

    def test_sha256_tracking(self, tmp_path) -> None:
        logger = StreamLogger(
            base_path=str(tmp_path),
            turn_id="test-turn",
            created_at="2026-07-23T11:01:32.000000+00:00",
        )
        data = b"test data"
        b64 = base64.b64encode(data).decode()
        logger.record_upstream_chunk(b64, 1.0)
        logger.record_downstream_chunk(b64, 2.0)

        assert logger.upstream_bytes == len(data)
        assert logger.downstream_bytes == len(data)
        assert logger.upstream_sha256_hex == logger.downstream_sha256_hex

    def test_terminal_record(self, tmp_path) -> None:
        logger = StreamLogger(
            base_path=str(tmp_path),
            turn_id="test-turn",
            created_at="2026-07-23T11:01:32.000000+00:00",
        )
        # Simulate a complete stream: record an SSE event, [DONE], then EOF
        event = {
            "raw_data": '{"choices":[{"finish_reason":"stop"}]}',
            "json_valid": True,
            "finish_reasons": ["stop"],
            "has_usage": False,
            "done": False,
        }
        logger.record_sse_event(event, 50.0)
        done_event = {
            "raw_data": "[DONE]",
            "json_valid": True,
            "finish_reasons": [],
            "has_usage": False,
            "done": True,
        }
        logger.record_sse_event(done_event, 60.0)
        logger.record_upstream_eof(100.0)
        logger.write_terminal(
            headers_status=200,
            upstream_eof=True,
            elapsed_ms=100.0,
        )
        lines = logger.filepath.read_text().strip().split("\n")
        # terminal + earlier records
        terminal_line = lines[-1]
        record = json.loads(terminal_line)
        assert record["type"] == "terminal"
        assert record["classification"] == "complete"
        assert record["headers_status"] == 200
        assert record["upstream_eof"] is True


def _first_json_line(path: str) -> dict[str, Any]:
    """Read and parse the first JSONL line of ``path``."""
    with open(path, encoding="utf-8") as handle:
        line: dict[str, Any] = json.loads(handle.readline())
    return line


class TestStreamBodyHashIntegrity:
    """The persisted body must hash to the recorded request_body_sha256.

    ``request_body_sha256`` is computed from the POST-alias body — the one
    actually forwarded upstream. A stream file that stores the PRE-alias
    body therefore stores a different document than the hash describes,
    and the hash stops verifying: a consumer who re-hashes the embedded
    request gets a mismatch and cannot tell whether the body or the hash
    is lying.

    Aliasing is what makes the two differ. With no alias map the two
    variables are identical and the bug is invisible, so this test drives a
    real PUBLIC->UPSTREAM rewrite.
    """

    PUBLIC = "qwen3.8-27b-fp8"
    UPSTREAM = "davidau-qwen3.8-27b-twin-turbo-fable"

    def _stream_request(self, tmp_path: Path) -> Path:
        """Drive one streaming chat request through the app with aliasing on."""

        from capture_proxy.app import CaptureProxyApp
        from capture_proxy.upstream import StreamingUpstreamResponse

        async def _aiter() -> AsyncIterator[bytes]:
            yield b'data: {"choices":[{"delta":{"content":"hi"}}]}\n\n'
            yield b"data: [DONE]\n\n"

        env = {
            "CAPTURE_BASE_PATH": str(tmp_path / "captures"),
            "CAPTURE_MODEL_ALIASES": json.dumps({self.PUBLIC: self.UPSTREAM}),
            "CAPTURE_UPSTREAM_URL": "http://127.0.0.1:8080",
        }
        with patch.dict(os.environ, env, clear=False):
            os.environ.pop("CAPTURE_BACKEND", None)
            app = CaptureProxyApp()
            mock_upstream = MagicMock()
            mock_upstream.health_check = AsyncMock(return_value=True)
            mock_upstream.proxy = AsyncMock(
                return_value=StreamingUpstreamResponse(
                    status_code=200,
                    headers={"content-type": "text/event-stream"},
                    content_type="text/event-stream",
                    stream=_aiter(),
                )
            )
            app._upstream_client = mock_upstream
            client = TestClient(app.app)
            client.post(
                "/v1/chat/completions",
                json={
                    "model": self.PUBLIC,
                    "messages": [{"role": "user", "content": "hi"}],
                    "stream": True,
                },
            )
        return tmp_path

    def test_persisted_body_hashes_to_request_body_sha256(self, tmp_path: Path) -> None:
        from capture_proxy.blobs import canonical_bytes

        base = self._stream_request(tmp_path)
        files = glob.glob(
            str(base / "captures" / "raw" / "**" / "*.stream.jsonl"), recursive=True
        )
        assert files, "no stream file written"
        start = _first_json_line(files[0])
        assert start["type"] == "request_start"

        body = start["request_body"]
        # The alias rewrite must have happened, or this test proves nothing:
        # with no rewrite the pre- and post-alias bodies are byte-identical.
        assert body["model"] != self.PUBLIC, "alias rewrite did not occur"

        digest = hashlib.sha256(canonical_bytes(body)).hexdigest()
        assert digest == start["request_body_sha256"], (
            f"persisted body hashes to {digest[:16]}..., "
            f"request_body_sha256 is {start['request_body_sha256'][:16]}...; "
            "the stream file stores a different document than the hash describes"
        )

    def test_stream_body_matches_the_raw_record_body(self, tmp_path: Path) -> None:
        """Stream file and raw record must agree, and both must be post-alias.

        The blob store is written from ``record.request``, which is the
        post-alias body. If the stream file carried the pre-alias body the
        two artifacts of the same capture would describe different requests.
        """
        from capture_proxy.blobs import canonical_bytes

        base = self._stream_request(tmp_path)
        streams = glob.glob(
            str(base / "captures" / "raw" / "**" / "*.stream.jsonl"), recursive=True
        )
        raws = [
            p
            for p in glob.glob(
                str(base / "captures" / "raw" / "**" / "*.jsonl"), recursive=True
            )
            if ".stream.jsonl" not in p
        ]
        assert streams and raws, "expected both a stream file and a raw record"
        start = _first_json_line(streams[0])
        raw = _first_json_line(raws[0])
        assert start["request_body"] == raw["request"], (
            "stream request_body and raw record request describe different documents"
        )
        assert (
            hashlib.sha256(canonical_bytes(raw["request"])).hexdigest()
            == raw["request_body_sha256"]
        )

    def test_round_trip_yields_original_messages(self, tmp_path: Path) -> None:
        """The DoD: a stream file alone must yield what was asked.

        No cross-referencing against the paired raw record, no blob lookup.
        If this fails, the capture is the unrecoverable shape — data loss.
        """
        base = self._stream_request(tmp_path)
        files = glob.glob(
            str(base / "captures" / "raw" / "**" / "*.stream.jsonl"), recursive=True
        )
        start = _first_json_line(files[0])
        assert start["request_body"]["messages"] == [{"role": "user", "content": "hi"}]

    def test_body_write_failure_degrades_without_raising(self, tmp_path: Path) -> None:
        """A capture-tree fault must cost diagnostics, never the response.

        The contract is that a fault writing the stream file degrades
        diagnostics for that one request and still serves the client. This
        pins it, because the whole P0-2 change adds a second write on the
        streaming hot path and a failure there must stay non-fatal.
        """
        from unittest.mock import AsyncMock, MagicMock, patch

        from capture_proxy.app import CaptureProxyApp
        from capture_proxy.upstream import StreamingUpstreamResponse

        async def _aiter() -> AsyncIterator[bytes]:
            yield b'data: {"choices":[{"delta":{"content":"hi"}}]}\n\n'
            yield b"data: [DONE]\n\n"

        env = {
            "CAPTURE_BASE_PATH": str(tmp_path / "captures"),
            "CAPTURE_UPSTREAM_URL": "http://127.0.0.1:8080",
        }
        with patch.dict(os.environ, env, clear=False):
            os.environ.pop("CAPTURE_BACKEND", None)
            os.environ.pop("CAPTURE_MODEL_ALIASES", None)
            app = CaptureProxyApp()
            mock_upstream = MagicMock()
            mock_upstream.health_check = AsyncMock(return_value=True)
            mock_upstream.proxy = AsyncMock(
                return_value=StreamingUpstreamResponse(
                    status_code=200,
                    headers={"content-type": "text/event-stream"},
                    content_type="text/event-stream",
                    stream=_aiter(),
                )
            )
            app._upstream_client = mock_upstream

            def _boom(*_a: object, **_k: object) -> None:
                raise OSError(28, "No space left on device")

            with (
                patch(
                    "capture_proxy.stream.StreamLogger._write",
                    side_effect=_boom,
                ),
                patch("capture_proxy.app._capture_write_failed") as warned,
            ):
                client = TestClient(app.app)
                resp = client.post(
                    "/v1/chat/completions",
                    json={
                        "model": "m",
                        "messages": [{"role": "user", "content": "hi"}],
                        "stream": True,
                    },
                )

        assert resp.status_code == 200, "capture fault must not fail the response"
        assert resp.content, "client received no body"
        assert warned.called, "the one-time write-failure warning never fired"
