"""
Capture Schema Manifest

Owns: The ``schema.json`` declaration written once per capture base path
Depends: json (stdlib), pathlib (stdlib), datetime (stdlib)
Provides: SCHEMA_VERSION, REQUIRED_FIELDS, OPTIONAL_FIELDS, NULLABLE_NOT_APPLICABLE, write_schema_manifest, read_schema_manifest

Invariants:
  - the manifest is the contract: it names the version every NEW record is
    written at, so a reader never has to infer the shape from a sample
  - tri-state is declared, not implied. Every field is classified as
    REQUIRED (always present), NULLABLE_NOT_APPLICABLE (present, may be
    null, null means "does not apply to this request type"), or
    OPTIONAL_UNRECORDED (may be absent from the line entirely; absence
    means "not recorded", which is a different claim from null)
  - a manifest write failure never fails a proxy start — the manifest is
    documentation of a layout that already exists on disk
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

#: Schema version every record written by this proxy declares.
SCHEMA_VERSION = 6

#: Always present on a v6 record. Absence means a malformed or pre-v6 line.
REQUIRED_FIELDS: tuple[str, ...] = (
    "schema_version",
    "session_id",
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
    "upstream_status",
    "attempt_number",
)

#: Present on a v6 record, may be null. Null means NOT APPLICABLE to this
#: request type (a chat turn has no embedding_count; an embedding turn has
#: no finish_reason) — it does not mean "recording failed".
NULLABLE_NOT_APPLICABLE: tuple[str, ...] = (
    "user_id",
    "task",
    "task_rule",
    "task_evidence",
    "run_id",
    "runtime_settings",
    "response",
    "tool_calls",
    "finish_reason",
    "usage_raw",
    "usage_normalized",
    "error_body",
    "routing",
    "stream_diagnostics",
    "stream_file",
    "request_type",
    "embedding_count",
    "embedding_dimensions",
    "input_sha256",
    "headroom",
)

#: May be absent from the line entirely. Absent means NOT RECORDED — the
#: proxy did not observe the fact (aliasing off means no original/forwarded
#: model id was resolvable). Distinct from null, which asserts
#: non-applicability.
OPTIONAL_UNRECORDED: tuple[str, ...] = (
    "request_model_original",
    "request_model_forwarded",
)

#: Artifact families the manifest describes, so a reader knows where to
#: look for a capture without guessing the directory layout.
ARTIFACT_LAYOUT: dict[str, str] = {
    "raw": "{base_path}/raw/YYYY-MM-DD/{timestamp}-{turn_id}.jsonl",
    "errors": "{base_path}/errors/YYYY-MM-DD/{timestamp}-{turn_id}.jsonl",
    "stream": "{base_path}/raw/YYYY-MM-DD/{timestamp}-{turn_id}.stream.jsonl",
    "index_session": "{base_path}/index/sessions/{safe}-{sha256[:12]}.jsonl",
    "index_user": "{base_path}/index/users/{safe}-{sha256[:12]}.jsonl",
    "blob": "{base_path}/blobs/sha256/{digest[:2]}/{digest}",
    "run_manifest": "{base_path}/runs/{run_id}.json",
    "schema": "{base_path}/schema.json",
}


def build_schema_manifest(
    blobs_enabled: bool = True, run_id: str | None = None
) -> dict[str, Any]:
    """Build the schema manifest document.

    Args:
        blobs_enabled: Whether the content-addressed blob store is active.
            Declared so a reader knows whether an unresolvable hash means
            "write failed" or "store disabled by configuration".
        run_id: The RuntimeSnapshot run id this manifest describes. Recorded
            so a per-run declaration is attributable to the records that
            run wrote.

    Returns:
        dict: The manifest, ready for JSON serialization.
    """
    from capture_proxy.taskclass import TASK_VOCABULARY

    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(UTC).isoformat(),
        # Present only in a per-run declaration; null in the corpus-level
        # file, which aggregates runs rather than describing one.
        "run_id": run_id,
        "required_fields": list(REQUIRED_FIELDS),
        "nullable_not_applicable_fields": list(NULLABLE_NOT_APPLICABLE),
        "optional_unrecorded_fields": list(OPTIONAL_UNRECORDED),
        "task_vocabulary": list(TASK_VOCABULARY),
        "index_key": "capture_id",
        "index_key_note": (
            "A capture appears in BOTH index/sessions/ and index/users/ "
            "when it carries both ids. Dedupe on capture_id before counting; "
            "the two shards are two views of one turn, not two turns."
        ),
        "hash_resolution": {
            "blob_store_enabled": blobs_enabled,
            "layout": ARTIFACT_LAYOUT["blob"],
            "preimage": "json.dumps(value, sort_keys=True, default=str).encode()",
            "covers": ["request_body_sha256", "messages_sha256"],
        },
        "artifact_layout": dict(ARTIFACT_LAYOUT),
    }


def write_schema_manifest(
    base_path: str | Path, blobs_enabled: bool = True, run_id: str | None = None
) -> Path | None:
    """Write the schema manifest for one proxy run.

    Writes ``{base_path}/schema.json`` as the corpus-level declaration and,
    when ``run_id`` is known, ``{base_path}/runs/{run_id}.schema.json``
    as the per-run declaration. The per-run copy is the durable one: the
    corpus-level file is overwritten by the next run, so a reader who
    arrives later cannot tell which run produced which records. The
    per-run file sits beside the run manifest it describes.

    Args:
        base_path: The capture root.
        blobs_enabled: Whether the blob store is active.
        run_id: The RuntimeSnapshot run id, when the caller has it.

    Returns:
        Path | None: The corpus-level manifest path, or None when the write
            failed (OSError suppressed — a missing manifest never blocks
            capture).
    """
    manifest = build_schema_manifest(blobs_enabled, run_id)
    path = Path(base_path) / "schema.json"
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(manifest, indent=2) + "\n"
        path.write_text(payload, encoding="utf-8")
        if run_id:
            per_run = Path(base_path) / "runs" / f"{run_id}.schema.json"
            per_run.parent.mkdir(parents=True, exist_ok=True)
            per_run.write_text(payload, encoding="utf-8")
    except OSError:
        return None
    return path


def read_schema_manifest(base_path: str | Path) -> dict[str, Any] | None:
    """Read the schema manifest for a capture root.

    Args:
        base_path: The capture root.

    Returns:
        dict | None: The parsed manifest, or None when absent or
            unreadable — an undeclared version is reported, never guessed.
    """
    path = Path(base_path) / "schema.json"
    try:
        loaded: Any = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return loaded if isinstance(loaded, dict) else None
