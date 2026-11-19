"""
Capture Logger

Owns: JSONL file creation, directory management, record writing, per-id
capture index maintenance
Depends on: json (stdlib), pathlib (stdlib), datetime (stdlib), hashlib
(stdlib), re (stdlib)
Provides: CaptureLogger.log(), CaptureLogger.log_error(),
CaptureLogger.plan_path(), CaptureLogger.read_index(),
CaptureLogger.list_index_ids()

Invariants:
  - 1 record per request, 0 lost records — INV-1
  - Files are append-only, never truncated — INV-2
  - Directory structure created on first write — INV-3
  - Index appends happen AFTER the main capture line is durable and never
    raise: an OSError degrades to a one-time named warning, the capture
    itself is unaffected
  - Index files live under {base_path}/index/{sessions|users}/ and are
    keyed by a sanitized-prefix + SHA-256(filename hash) name, so hostile
    ids cannot escape the index directory (no path traversal)
  - read_index() filters lines by exact id equality inside each line, so a
    filename-level collision can never cross-contaminate query results
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# schema.py is import-light and imports taskclass lazily inside a
# function, so this does not create a cycle at module load.
from capture_proxy.schema import OPTIONAL_UNRECORDED

# Characters allowed in an index filename component. Everything else is
# collapsed to "_" before the disambiguating hash suffix is appended.
_INDEX_SAFE_RE = re.compile(r"[^A-Za-z0-9._-]")

# Longest sanitized prefix kept before the hash suffix (filename budget).
_INDEX_PREFIX_LEN = 80

_index_write_warned = False


def _index_write_failed(site: str, exc: OSError) -> None:
    """One-time named warning for a suppressed index write failure.

    Index loss is diagnostic-only (the raw/errors capture line was already
    written); naming the site once per process keeps the hot path cheap
    while the fault stays visible.
    """
    global _index_write_warned
    if _index_write_warned:
        return
    _index_write_warned = True
    print(
        f"warning: capture index write failed at {site} "
        f"(file={getattr(exc, 'filename', None) or 'unknown'}, "
        f"errno={exc.errno}); continuing without that index entry",
        file=sys.stderr,
    )


class CaptureLogger:
    """
    Writes CaptureRecord to the configured capture store as JSONL and
    maintains the per-session / per-user capture index.

    Purpose: Persists every proxied request/response as a single JSONL line
    for later analysis, and appends one O(1) index line per capture so any
    session_id or user_id is retrievable without scanning raw/ JSONL files.
    Dependencies: pathlib for file operations, json for serialization,
    hashlib/re for traversal-safe index filenames

    Invariants:
      - Files created on first write, not at startup (lazy directory creation)
      - File naming: {timestamp}-{turn_id}.jsonl
      - Directory structure: {base_path}/{raw|errors}/YYYY-MM-DD/
      - Index structure: {base_path}/index/{sessions|users}/{safe}-{hash}.jsonl
      - Append-only writes — never truncate existing files
      - Index write failure never propagates (one-time warning instead)
    """

    def __init__(self, base_path: str = "/root/captures") -> None:
        """
        Initialize logger with base storage path.

        Args:
            base_path: Root directory for capture files

        Returns: None

        Raises: None

        Invariants:
          - Before: No directories created
          - After: Base path stored, directories created on first write
        """
        self._base_path = Path(base_path)

    def log(self, record: dict[str, Any]) -> Path:
        """
        Append 1 JSONL line to raw/YYYY-MM-DD/{timestamp}-{turn_id}.jsonl.

        Args:
            record: CaptureRecord as dict (fully populated before call)

        Returns:
            Path: Path to the JSONL file written

        Raises:
            OSError: If file write fails (disk full, permissions)

        Invariants:
          - Before: Record dict must contain 'turn_id' and 'created_at' keys
          - After: 1 JSONL line appended to file, file exists
          - File is append-only — existing content not modified
          - Directory created on first write if missing
        """
        return self._write_record(record, "raw")

    def log_error(self, record: dict[str, Any]) -> Path:
        """
        Append 1 JSONL line to errors/YYYY-MM-DD/{timestamp}-{turn_id}.jsonl.

        Args:
            record: CaptureRecord as dict with error_body populated

        Returns:
            Path: Path to the error JSONL file written

        Raises:
            OSError: If file write fails

        Invariants:
          - Before: Record dict must contain 'turn_id' and 'created_at' keys
          - After: 1 JSONL line appended to errors/ file
          - Separate from raw/ — errors are isolated for filtering
        """
        return self._write_record(record, "errors")

    def plan_path(self, record: dict[str, Any], category: str) -> Path:
        """
        Compute the JSONL file path for a record without creating anything.

        Single source of the {timestamp}-{turn_id}.jsonl naming so callers
        that reference the file before it is written (e.g. the headroom
        sidecar ref) agree with the path log() produces.

        Args:
            record: CaptureRecord as dict (uses 'turn_id' and 'created_at')
            category: "raw" or "errors"

        Returns:
            Path: {base_path}/{category}/YYYY-MM-DD/{timestamp}-{turn_id}.jsonl

        Raises: None

        Invariants:
          - Deterministic for identical turn_id/created_at inputs
          - No directories are created by this call
          - Unparseable created_at falls back to the current UTC date/time
        """
        turn_id = record.get("turn_id", "unknown")
        created_at = record.get("created_at", datetime.now(UTC).isoformat())

        # Parse date from ISO 8601
        dt = None
        try:
            dt = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
            date_str = dt.strftime("%Y-%m-%d")
        except (ValueError, AttributeError):
            date_str = datetime.now(UTC).strftime("%Y-%m-%d")

        if dt is not None:
            timestamp = dt.strftime("%Y%m%dT%H%M%S.%f")
        else:
            timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%f")
        return self._base_path / category / date_str / f"{timestamp}-{turn_id}.jsonl"

    def _write_record(self, record: dict[str, Any], category: str) -> Path:
        """
        Write a single JSONL record to the appropriate file.

        Args:
            record: CaptureRecord as dict
            category: "raw" or "errors"

        Returns:
            Path: Path to the written file

        Raises:
            OSError: If file write fails

        Invariants:
          - File path computed by plan_path() (single source of naming)
          - Directory created on first write (mkdir -p behavior)
          - JSON serialized with ensure_ascii=False for unicode support
        """
        filepath = self.plan_path(record, category)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # Tri-state, third arm: schema.json classifies every field as
        # required, nullable-not-applicable, or optional-unrecorded. The
        # optional class is only meaningful if those keys are ABSENT when
        # nothing was recorded. A Pydantic model_dump() emits them as null,
        # which collapses "not applicable" and "not recorded" into one
        # indistinguishable value and makes the declaration a lie. Drop
        # them here, where the record becomes bytes, rather than touching
        # the nullable class that legitimately serializes as null.
        for key in OPTIONAL_UNRECORDED:
            if key in record and record[key] is None:
                del record[key]

        # Append JSONL line
        jsonl_line = json.dumps(record, ensure_ascii=False, default=str) + "\n"
        with open(filepath, "a", encoding="utf-8") as f:
            f.write(jsonl_line)

        self._append_indexes(record, category, filepath)

        return filepath

    @staticmethod
    def index_filename(identifier: str) -> str:
        """
        Build a traversal-safe, collision-resistant index filename for an id.

        Args:
            identifier: Raw session_id or user_id value (may contain any
                characters, including path separators).

        Returns:
            str: "{sanitized_prefix}-{sha256[:12]}.jsonl" where sanitized
                prefix keeps only [A-Za-z0-9._-] truncated to 80 chars.

        Raises: None

        Invariants:
          - Deterministic: same identifier → same filename
          - Output contains no "/" or ".." sequence — safe to join under
            the index directory
          - Distinct identifiers map to distinct filenames with
            overwhelming probability; residual collisions are neutralized
            by read_index()'s exact-id line filter
        """
        prefix = _INDEX_SAFE_RE.sub("_", identifier)[:_INDEX_PREFIX_LEN] or "empty"
        digest = hashlib.sha256(identifier.encode("utf-8")).hexdigest()[:12]
        return f"{prefix}-{digest}.jsonl"

    def _index_path(self, kind: str, identifier: str) -> Path:
        """Return the index file path for one id under index/{kind}/."""
        return self._base_path / "index" / kind / self.index_filename(identifier)

    def _append_indexes(
        self, record: dict[str, Any], category: str, filepath: Path
    ) -> None:
        """
        Append one index line per session/user id found in the record.

        Called after the main capture line is durable. Never raises: every
        OSError is suppressed through the one-time warning so a broken or
        blocked index tree cannot affect captures or responses.

        Args:
            record: CaptureRecord as dict (session_id/user_id read via .get,
                so minimal error records without ids skip indexing cleanly).
            category: "raw" or "errors" — stored on each index line.
            filepath: The main JSONL path just written.

        Returns: None

        Raises: None (OSError suppressed)
        """
        try:
            rel = filepath.relative_to(self._base_path)
        except ValueError:
            rel = filepath

        def _clean_id(value: Any) -> str | None:
            return value if isinstance(value, str) and value.strip() else None

        # Both ids ride on every line so a user lookup shows which session
        # each capture came from and vice versa. A capture with both ids is
        # therefore written to BOTH shards with one identical capture_id:
        # two views of one turn, not two turns. `index_key` names that
        # dedup key so a naive row count is visibly wrong rather than
        # silently inflated.
        base_entry: dict[str, Any] = {
            "ts": datetime.now(UTC).isoformat(),
            "capture_file": str(rel),
            "category": category,
            "index_key": record.get("capture_id") or record.get("turn_id"),
            "capture_id": record.get("capture_id") or None,
            "turn_id": record.get("turn_id") or None,
            "model": record.get("model") or None,
            "request_type": record.get("request_type") or None,
            "task": record.get("task") or None,
            "task_rule": record.get("task_rule") or None,
            "stream_file": record.get("stream_file") or None,
            "upstream_status": record.get("upstream_status"),
            "session_id": _clean_id(record.get("session_id")),
            "user_id": _clean_id(record.get("user_id")),
        }
        for kind, id_field in (("sessions", "session_id"), ("users", "user_id")):
            if base_entry[id_field] is None:
                continue
            base_entry["shard"] = kind
            try:
                path = self._index_path(kind, base_entry[id_field])
                path.parent.mkdir(parents=True, exist_ok=True)
                with open(path, "a", encoding="utf-8") as f:
                    f.write(
                        json.dumps(base_entry, ensure_ascii=False, default=str) + "\n"
                    )
            except OSError as exc:
                _index_write_failed(f"index/{kind}", exc)

    def read_index(self, kind: str, identifier: str) -> list[dict[str, Any]]:
        """
        Read all index entries for one session_id or user_id.

        Reads ONLY the single index file for this id — never scans raw/.
        Lines that fail to parse (torn append) are skipped. Entries whose
        embedded id does not exactly equal ``identifier`` are filtered out,
        neutralizing any filename-level collision.

        Args:
            kind: "sessions" or "users".
            identifier: Exact session_id or user_id to look up.

        Returns:
            list[dict]: Index entries in chronological append order;
                empty list when the id is unknown or the index is unreadable.

        Raises: None (OSError suppressed → [])
        """
        id_field = "session_id" if kind == "sessions" else "user_id"
        entries: list[dict[str, Any]] = []
        try:
            with open(self._index_path(kind, identifier), encoding="utf-8") as f:
                for line in f:
                    stripped = line.strip()
                    if not stripped:
                        continue
                    try:
                        entry = json.loads(stripped)
                    except json.JSONDecodeError:
                        continue  # torn append — skip, never fail the query
                    if isinstance(entry, dict) and entry.get(id_field) == identifier:
                        entries.append(entry)
        except OSError:
            return []
        return entries

    def list_index_ids(self, kind: str, limit: int = 100) -> list[dict[str, Any]]:
        """
        List known session/user ids with activity bounds, most recent first.

        Reads only index files under {base_path}/index/{kind}/ — at most
        ``limit`` files (newest mtime first), each fully parsed to count
        captures and extract first/last-seen timestamps. Never touches
        raw/ or errors/ JSONL.

        Args:
            kind: "sessions" or "users".
            limit: Maximum number of ids returned (clamped to >= 0).

        Returns:
            list[dict]: {"id", "captures", "first_seen", "last_seen"}
                sorted by last activity descending; empty when the index
                directory is absent or unreadable.

        Raises: None (OSError suppressed → [])
        """
        id_field = "session_id" if kind == "sessions" else "user_id"
        directory = self._base_path / "index" / kind
        try:
            candidates: list[tuple[int, Path]] = [
                (p.stat().st_mtime_ns, p) for p in directory.iterdir() if p.is_file()
            ]
        except OSError:
            return []
        candidates.sort(reverse=True)

        summaries: list[dict[str, Any]] = []
        for _, path in candidates[: max(0, limit)]:
            ident: str | None = None
            first_seen: str | None = None
            last_seen: str | None = None
            count = 0
            try:
                with open(path, encoding="utf-8") as f:
                    for line in f:
                        stripped = line.strip()
                        if not stripped:
                            continue
                        try:
                            entry = json.loads(stripped)
                        except json.JSONDecodeError:
                            continue
                        if not isinstance(entry, dict):
                            continue
                        count += 1
                        if ident is None:
                            found = entry.get(id_field)
                            ident = found if isinstance(found, str) else None
                        ts = entry.get("ts")
                        if isinstance(ts, str):
                            if first_seen is None:
                                first_seen = ts
                            last_seen = ts
            except OSError:
                continue
            summaries.append(
                {
                    "id": ident if ident is not None else path.stem,
                    "captures": count,
                    "first_seen": first_seen,
                    "last_seen": last_seen,
                }
            )
        return summaries
