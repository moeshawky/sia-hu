"""
Stream Parser and Logger

Owns: Incremental SSE parser, raw chunk recording, terminal state classification
Depends: json (stdlib), base64 (stdlib), hashlib (stdlib), pathlib (stdlib)
Provides: SseParser, StreamLogger, classify_terminal

Invariants:
  - Raw bytes forwarded BEFORE parse copy is inspected — INV-1
  - Each stream.jsonl line is a self-contained JSON object — INV-2
  - Terminal classification answers one of 10 failure modes — INV-3
  - SHA-256 of upstream bytes equals SHA-256 of downstream bytes in transparent mode — INV-4
"""

from __future__ import annotations

import base64
import codecs
import contextlib
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


class SseParser:
    """Incremental SSE parser that handles chunk boundaries correctly.

    One network chunk can contain:
    - Half an SSE event
    - Multiple SSE events
    - [DONE] plus earlier JSON
    - A JSON error not formatted as SSE

    This parser buffers incomplete lines and yields complete events.
    """

    def __init__(self) -> None:
        self._buffer = ""
        self._decoder = codecs.getincrementaldecoder("utf-8")()

    def feed(self, raw_bytes: bytes) -> list[dict[str, Any]]:
        """Feed raw network bytes, return completed SSE events.

        Decodes bytes as UTF-8 with an incremental decoder, so a multi-byte
        character split across network chunks decodes intact (flush()
        finalizes any pending bytes). Undecodable bytes (corrupt or truncated
        upstream UTF-8) are replaced with U+FFFD instead of raising — the
        event is still yielded with the replacement visible in raw_data
        (json_valid=False when the payload no longer parses) instead of
        escaping into the stream path.
        Accepts both ``data:`` and ``data: <payload>`` forms, and a bare
        ``[DONE]`` line.

        Each event dict contains:
          - raw_data: str (the data payload, e.g. '{"choices":[...]}')
          - json_valid: bool
          - finish_reasons: list[str]
          - has_usage: bool
          - done: bool (True if data == "[DONE]")
        """
        try:
            self._buffer += self._decoder.decode(raw_bytes)
        except UnicodeDecodeError:
            # The stateful decoder leaves undefined internal state after a
            # strict-decode fault — reset it and append the chunk lossily so
            # SSE line structure survives and the replacement is visible in
            # the event's raw_data.
            self._buffer += raw_bytes.decode("utf-8", errors="replace")
            self._decoder = codecs.getincrementaldecoder("utf-8")()

        events: list[dict[str, Any]] = []
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            line = line.strip()
            if not line:
                continue
            if line == "[DONE]":
                events.append(self._parse_data_line("[DONE]"))
            elif line.startswith("data:"):
                data = line[len("data:") :].lstrip()
                events.append(self._parse_data_line(data))
        return events

    def flush(self) -> list[dict[str, Any]]:
        """Flush remaining buffer — call after upstream EOF.

        Undecodable trailing bytes are replaced with U+FFFD instead of
        raising, matching feed()'s lossy fallback.
        """
        try:
            self._buffer += self._decoder.decode(b"", True)
        except UnicodeDecodeError:
            # Pending bytes are undecodable and unreachable after the fault —
            # emit one replacement char and reset the decoder.
            self._buffer += "\ufffd"
            self._decoder = codecs.getincrementaldecoder("utf-8")()
        remaining = self._buffer.strip()
        self._buffer = ""
        if not remaining:
            return []
        if remaining == "[DONE]":
            return [self._parse_data_line("[DONE]")]
        if remaining.startswith("data:"):
            data = remaining[len("data:") :].lstrip()
            return [self._parse_data_line(data)]
        return [self._parse_data_line(remaining)]

    @staticmethod
    def _parse_data_line(data: str) -> dict[str, Any]:
        """Parse a single SSE data payload."""
        if data == "[DONE]":
            return {
                "raw_data": "[DONE]",
                "json_valid": True,
                "finish_reasons": [],
                "has_usage": False,
                "done": True,
            }

        json_valid = False
        finish_reasons: list[str] = []
        has_usage = False
        with contextlib.suppress(json.JSONDecodeError, TypeError):
            parsed = json.loads(data)
            json_valid = True
            for choice in parsed.get("choices", []):
                fr = choice.get("finish_reason")
                if fr:
                    finish_reasons.append(fr)
            if parsed.get("usage"):
                has_usage = True

        return {
            "raw_data": data,
            "json_valid": json_valid,
            "finish_reasons": finish_reasons,
            "has_usage": has_usage,
            "done": False,
        }


class StreamLogger:
    """Writes companion stream.jsonl for a single request.

    Produces these record types:
      1. request_start — metadata before contacting upstream
      2. upstream_headers — upstream HTTP response headers
      3. upstream_chunk — raw network bytes with sequence and timing
      4. forward_checkpoint — proof of downstream forwarding
      5. upstream_eof — marks end of upstream data
      6. sse_event — parsed SSE event with diagnostics
      7. sse_done — marks [DONE] received
      8. terminal — final state classification

    File naming matches the summary capture:
      {timestamp}-{turn_id}.stream.jsonl
    """

    def __init__(self, base_path: str, turn_id: str, created_at: str) -> None:
        self._base_path = Path(base_path)
        self._turn_id = turn_id
        self._created_at = created_at
        self._seq = 0
        self._filepath = self._build_filepath()
        self._upstream_bytes = 0
        self._upstream_sha256 = hashlib.sha256()
        self._downstream_bytes = 0
        self._downstream_sha256 = hashlib.sha256()
        self._done_seen = False
        self._finish_reasons_seen: list[str] = []
        self._usage_seen = False
        self._sse_events: list[dict[str, Any]] = []
        self._first_event: dict[str, Any] | None = None
        self._last_event: dict[str, Any] | None = None

    def record_request_start(
        self,
        *,
        capture_id: str,
        request_body_sha256: str,
        messages_sha256: str,
        model: str,
        messages_count: int,
        stream: bool,
        max_completion_tokens: int | None,
        request_bytes: int,
        request_body: dict[str, Any] | None = None,
        task: str | None = None,
    ) -> None:
        """Write request_start record before contacting upstream.

        The record carries the request body itself when ``request_body`` is
        supplied, not only its hash and size. A stream capture whose first
        line holds just ``messages_sha256`` and a count is not
        analyzable on its own: it says a request happened and how big it
        was, never what was asked. Embedding the body makes a stream file
        self-describing without a second lookup.

        The body is written once here and never again — the remaining
        lines are the response side of the exchange. Storage duplication
        against the sibling ``.jsonl`` line is the deliberate trade for
        self-containment; the hash lets a reader dedupe if they prefer.

        Args:
            capture_id: This capture's identifier.
            request_body_sha256: Digest of the forwarded body.
            messages_sha256: Digest of the messages array.
            model: Public model id.
            messages_count: Number of messages in the request.
            stream: Whether the request asked for streaming.
            max_completion_tokens: The cap, when the body set one.
            request_bytes: Serialized byte length of the body.
            request_body: The forwarded request body, when available.
                None omits the key entirely (absent = not recorded, which
                is distinct from an empty dict = recorded as empty).
            task: Task label derived structurally at capture time.

        Returns: None
        """
        self._seq += 1
        record: dict[str, Any] = {
            "type": "request_start",
            "seq": self._seq,
            "capture_id": capture_id,
            "created_at": self._created_at,
            "request_body_sha256": request_body_sha256,
            "messages_sha256": messages_sha256,
            "model": model,
            "messages": messages_count,
            "stream": stream,
            "max_completion_tokens": max_completion_tokens,
            "request_bytes": request_bytes,
            "task": task,
        }
        if request_body is not None:
            record["request_body"] = request_body
        self._write(record)

    def record_upstream_headers(
        self,
        *,
        elapsed_ms: float,
        status: int,
        content_type: str | None,
        headers: dict[str, str],
    ) -> None:
        """Write upstream_headers record on header receipt."""
        self._seq += 1
        self._write(
            {
                "type": "upstream_headers",
                "seq": self._seq,
                "elapsed_ms": round(elapsed_ms, 3),
                "status": status,
                "content_type": content_type,
                "headers": headers,
            }
        )

    def record_forward_checkpoint(
        self,
        *,
        elapsed_ms: float,
        upstream_bytes: int,
        downstream_bytes: int,
        upstream_sha256_partial: str,
        downstream_sha256_partial: str,
        identical_so_far: bool,
    ) -> None:
        """Write forward_checkpoint record proving downstream forwarding."""
        self._seq += 1
        self._write(
            {
                "type": "forward_checkpoint",
                "seq": self._seq,
                "elapsed_ms": round(elapsed_ms, 3),
                "upstream_bytes": upstream_bytes,
                "downstream_bytes": downstream_bytes,
                "upstream_sha256_partial": upstream_sha256_partial,
                "downstream_sha256_partial": downstream_sha256_partial,
                "identical_so_far": identical_so_far,
            }
        )

    @property
    def filepath(self) -> Path:
        return self._filepath

    @property
    def upstream_bytes(self) -> int:
        return self._upstream_bytes

    @property
    def upstream_sha256_hex(self) -> str:
        return self._upstream_sha256.hexdigest()

    @property
    def downstream_bytes(self) -> int:
        return self._downstream_bytes

    @property
    def downstream_sha256_hex(self) -> str:
        return self._downstream_sha256.hexdigest()

    @property
    def done_seen(self) -> bool:
        return self._done_seen

    @property
    def finish_reasons_seen(self) -> list[str]:
        return list(self._finish_reasons_seen)

    @property
    def usage_seen(self) -> bool:
        return self._usage_seen

    def record_upstream_chunk(self, data_b64: str, elapsed_ms: float) -> None:
        """Record a raw upstream network chunk."""
        self._seq += 1
        raw_bytes = base64.b64decode(data_b64)
        self._upstream_bytes += len(raw_bytes)
        self._upstream_sha256.update(raw_bytes)
        self._write(
            {
                "type": "upstream_chunk",
                "seq": self._seq,
                "elapsed_ms": round(elapsed_ms, 3),
                "data_b64": data_b64,
            }
        )

    def record_upstream_eof(self, elapsed_ms: float) -> None:
        """Record upstream EOF."""
        self._seq += 1
        self._write(
            {
                "type": "upstream_eof",
                "seq": self._seq,
                "elapsed_ms": round(elapsed_ms, 3),
            }
        )

    def record_sse_event(self, event: dict[str, Any], elapsed_ms: float) -> None:
        """Record a parsed SSE event."""
        self._seq += 1
        if self._first_event is None:
            self._first_event = event
        self._last_event = event

        if event.get("done"):
            self._done_seen = True
            self._write(
                {
                    "type": "sse_done",
                    "seq": self._seq,
                    "elapsed_ms": round(elapsed_ms, 3),
                }
            )
            return

        if event.get("finish_reasons"):
            self._finish_reasons_seen.extend(event["finish_reasons"])
        if event.get("has_usage"):
            self._usage_seen = True

        self._sse_events.append(event)
        self._write(
            {
                "type": "sse_event",
                "seq": self._seq,
                "elapsed_ms": round(elapsed_ms, 3),
                "raw_data": event.get("raw_data", ""),
                "json_valid": event.get("json_valid", False),
                "finish_reasons": event.get("finish_reasons", []),
                "has_usage": event.get("has_usage", False),
                "done": event.get("done", False),
            }
        )

    def record_downstream_chunk(self, data_b64: str, elapsed_ms: float) -> None:
        """Record downstream bytes and write forward_checkpoint proving delivery."""
        raw_bytes = base64.b64decode(data_b64)
        self._downstream_bytes += len(raw_bytes)
        self._downstream_sha256.update(raw_bytes)
        identical = (
            self._downstream_sha256.hexdigest() == self._upstream_sha256.hexdigest()
            and self._downstream_bytes == self._upstream_bytes
        )
        self.record_forward_checkpoint(
            elapsed_ms=elapsed_ms,
            upstream_bytes=self._upstream_bytes,
            downstream_bytes=self._downstream_bytes,
            upstream_sha256_partial=self._upstream_sha256.hexdigest(),
            downstream_sha256_partial=self._downstream_sha256.hexdigest(),
            identical_so_far=identical,
        )

    def write_terminal(
        self,
        *,
        headers_status: int | None = None,
        upstream_eof: bool = False,
        client_disconnected: bool = False,
        upstream_exception: str | None = None,
        downstream_exception: str | None = None,
        elapsed_ms: float = 0.0,
    ) -> None:
        """Write terminal state record with classification."""
        final_finish_reason = (
            self._finish_reasons_seen[-1] if self._finish_reasons_seen else None
        )
        classification = classify_terminal(
            headers_status=headers_status,
            upstream_eof=upstream_eof,
            done_seen=self._done_seen,
            finish_reason_seen=final_finish_reason is not None,
            usage_seen=self._usage_seen,
            upstream_exception=upstream_exception,
            downstream_exception=downstream_exception,
            client_disconnected=client_disconnected,
            sse_events=self._sse_events,
        )
        hashes_identical = (
            self.upstream_sha256_hex == self.downstream_sha256_hex
            and self._upstream_bytes == self._downstream_bytes
        )

        self._seq += 1
        self._write(
            {
                "type": "terminal",
                "seq": self._seq,
                "elapsed_ms": round(elapsed_ms, 3),
                "classification": classification,
                "headers_status": headers_status,
                "upstream_eof": upstream_eof,
                "done_seen": self._done_seen,
                "final_finish_reason": final_finish_reason,
                "usage_seen": self._usage_seen,
                "upstream_exception": upstream_exception,
                "downstream_exception": downstream_exception,
                "client_disconnected": client_disconnected,
                "upstream": {
                    "bytes": self._upstream_bytes,
                    "sha256": self.upstream_sha256_hex,
                },
                "downstream": {
                    "bytes": self._downstream_bytes,
                    "sha256": self.downstream_sha256_hex,
                },
                "identical": hashes_identical,
            }
        )

    def _write(self, record: dict[str, Any]) -> None:
        """Append one JSONL line to the stream file."""
        jsonl_line = json.dumps(record, ensure_ascii=False, default=str) + "\n"
        with open(self._filepath, "a", encoding="utf-8") as f:
            f.write(jsonl_line)

    def _build_filepath(self) -> Path:
        """Build stream.jsonl file path mirroring summary capture layout."""
        try:
            dt = datetime.fromisoformat(self._created_at.replace("Z", "+00:00"))
        except (ValueError, AttributeError):
            dt = datetime.now(UTC)

        date_str = dt.strftime("%Y-%m-%d")
        timestamp = dt.strftime("%Y%m%dT%H%M%S.%f")
        category_path = self._base_path / "raw" / date_str
        category_path.mkdir(parents=True, exist_ok=True)
        return category_path / f"{timestamp}-{self._turn_id}.stream.jsonl"


def classify_terminal(
    *,
    headers_status: int | None = None,
    upstream_eof: bool = False,
    done_seen: bool = False,
    finish_reason_seen: bool = False,
    usage_seen: bool = False,
    upstream_exception: str | None = None,
    downstream_exception: str | None = None,
    client_disconnected: bool = False,
    sse_events: list[dict[str, Any]] | None = None,
) -> str:
    """Classify terminal state into one of 10 failure modes.

    Returns one of:
      complete
      http_error_before_stream
      empty_200_eof
      incomplete_sse_eof
      upstream_stream_exception
      downstream_forward_exception
      client_disconnect
      invalid_sse
      valid_finish_missing_done
      done_missing_finish
    """
    if client_disconnected:
        return "client_disconnect"
    if upstream_exception:
        return "upstream_stream_exception"
    if downstream_exception:
        return "downstream_forward_exception"
    if headers_status is not None and headers_status >= 400:
        return "http_error_before_stream"
    if (
        headers_status == 200
        and upstream_eof
        and not done_seen
        and not (sse_events or [])
    ):
        return "empty_200_eof"
    if not upstream_eof and not done_seen:
        return "incomplete_sse_eof"

    has_valid_sse = any(e.get("json_valid") for e in (sse_events or []))
    has_invalid_sse = any(
        not e.get("json_valid") and not e.get("done") for e in (sse_events or [])
    )
    if has_invalid_sse and not has_valid_sse:
        return "invalid_sse"

    if finish_reason_seen and not done_seen:
        return "valid_finish_missing_done"
    if done_seen:
        if finish_reason_seen:
            return "complete"
        return "done_missing_finish"

    if upstream_eof:
        if sse_events:
            return "incomplete_sse_eof"
        return "empty_200_eof"

    # Total-case fallback: with done_seen False the branches above already
    # returned, so this line is unreachable — it keeps the function total
    # for the remaining flag combination (same value as the not-upstream_eof
    # and not done_seen branch above).
    return "incomplete_sse_eof"
