#!/usr/bin/env python3
"""capture_query.py — answer corpus questions from the capture store alone.

The proxy used to be a wire-level recorder: it persisted bytes and
transport metadata and never recorded what a request was *for*, so the
question "show me the reranking captures" had no supported answer. This
tool is that answer, reading only what is on disk.

It works on any capture store, new or legacy:

  - ``task`` is read from the record when present (labelled at capture
    time by the structural classifier)
  - a legacy record with a null/absent ``task`` is labelled on the fly by
    running the SAME structural classifier over its stored request body,
    so a pre-fix corpus answers identically to a post-fix one
  - turns are deduped on ``capture_id``, the key declared in schema.json,
    because a capture carrying both a session and a user id is written to
    BOTH index shards and a raw row count double-counts it
  - stream captures are reachable through the index because index lines
    now carry ``stream_file``

Classification is structural. The classifier never reads topic
vocabulary to decide what a request is; see ``capture_proxy.taskclass``
and ``tests/test_taskclass_adversarial.py``, which feeds the classifier
the deployed cannon-gate and symbiose skill bodies and asserts zero
false positives.

Usage:
  capture_query.py --base /root/captures tasks
  capture_query.py --base /root/captures captures --task rerank
  capture_query.py --base /root/captures captures --task code_summarize
  capture_query.py --base /root/captures captures --stream
  capture_query.py --base /root/captures stats

Exit codes:
  0  query ran (an empty result is a valid answer, not an error)
  1  usage error
  2  capture store unreadable
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import Counter
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from capture_proxy.schema import read_schema_manifest
from capture_proxy.taskclass import classify_capture_record


def _iter_capture_files(base: str) -> list[str]:
    """Return every summary capture file, excluding stream sidecars."""
    out: list[str] = []
    for category in ("raw", "errors"):
        for path in glob.glob(f"{base}/{category}/**/*.jsonl", recursive=True):
            if path.endswith(".stream.jsonl"):
                continue
            out.append(path)
    return sorted(out)


def _iter_index_files(base: str) -> list[str]:
    """Return every index shard file across both kinds."""
    return sorted(glob.glob(f"{base}/index/sessions/*.jsonl")) + sorted(
        glob.glob(f"{base}/index/users/*.jsonl")
    )


def _read_jsonl(path: str) -> list[dict[str, Any]]:
    """Read a JSONL file, skipping unparseable lines."""
    rows: list[dict[str, Any]] = []
    try:
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    row = json.loads(stripped)
                except json.JSONDecodeError:
                    continue  # torn append — never fail a query on it
                if isinstance(row, dict):
                    rows.append(row)
    except OSError:
        return []
    return rows


def _label(record: dict[str, Any]) -> dict[str, Any]:
    """Return a record annotated with its effective task and rule."""
    classification = classify_capture_record(record)
    annotated = dict(record)
    annotated["_task"] = classification.task
    annotated["_task_rule"] = classification.rule
    annotated["_task_source"] = (
        "capture" if classification.rule == "pre_populated" else "backfill"
    )
    annotated["_file"] = record.get("_file", "")
    return annotated


def _dedupe(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Collapse records sharing a capture_id, keeping the first.

    The index writes one line per shard; the raw store writes one line
    per capture. A turn reachable through both paths is ONE turn.
    """
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for record in records:
        key = record.get("capture_id") or record.get("turn_id")
        if not key:
            continue
        if key in seen:
            continue
        seen.add(key)
        out.append(record)
    return out


def _relative(base: str, path: str) -> str:
    """Return a path relative to the capture root when possible."""
    try:
        return os.path.relpath(path, base)
    except ValueError:
        return path


def load_records(base: str) -> list[dict[str, Any]]:
    """Load and label every capture in the store, deduplicated.

    Args:
        base: Capture root.

    Returns:
        list[dict]: Annotated capture records, one per distinct turn.
    """
    records: list[dict[str, Any]] = []
    for path in _iter_capture_files(base):
        for row in _read_jsonl(path):
            row["_file"] = _relative(base, path)
            records.append(row)
    if not records:
        # Index-only store (raw pruned): the index still names the turns.
        for path in _iter_index_files(base):
            for row in _read_jsonl(path):
                row["_file"] = row.get("capture_file", _relative(base, path))
                records.append(row)
    labelled = [_label(row) for row in records]
    return _dedupe(labelled)


def cmd_tasks(args: argparse.Namespace) -> int:
    """Print the task distribution, with capture-time vs backfill split."""
    records = load_records(args.base)
    if not records:
        print(f"no captures found under {args.base}")
        return 0
    tasks = Counter(str(r.get("_task")) for r in records)
    sources = Counter(str(r.get("_task_source")) for r in records)
    schema = read_schema_manifest(args.base)
    print(f"distinct turns: {len(records)}")
    version = schema["schema_version"] if schema else None
    declared = f"declared v{version}" if schema else "UNDECLARED (no schema.json)"
    print(f"schema: {declared}")
    print(f"labelling: {dict(sources)}")
    print("\ntask distribution:")
    for task, count in tasks.most_common():
        print(f"  {task:22} {count:>8}")
    return 0


def cmd_captures(args: argparse.Namespace) -> int:
    """Print the captures matching a task and/or stream filter."""
    records = load_records(args.base)
    if not records:
        print(f"no captures found under {args.base}")
        return 0
    matched = records
    if args.task:
        matched = [r for r in matched if r.get("_task") == args.task]
    if args.stream:
        matched = [r for r in matched if r.get("stream_file")]
    if args.request_type:
        matched = [r for r in matched if r.get("request_type") == args.request_type]

    if args.json:
        print(json.dumps(matched, indent=2, default=str))
        return 0

    print(f"{len(matched)} capture(s) match")
    for record in matched[: args.limit]:
        stream = " [stream]" if record.get("stream_file") else ""
        print(
            f"  {record.get('created_at', '?')} "
            f"{record.get('capture_id', '?')} "
            f"task={record.get('_task')} "
            f"rule={record.get('_task_rule')} "
            f"type={record.get('request_type')}{stream}"
        )
        print(f"    {record.get('_file')}")
        if record.get("stream_file"):
            print(f"    stream: {record['stream_file']}")
    if len(matched) > args.limit:
        print(f"  ... {len(matched) - args.limit} more (use --limit)")
    return 0


def cmd_stats(args: argparse.Namespace) -> int:
    """Print corpus-level integrity statistics."""
    base = args.base
    records = load_records(base)
    capture_files = _iter_capture_files(base)
    stream_files = glob.glob(f"{base}/raw/**/*.stream.jsonl", recursive=True)
    index_files = _iter_index_files(base)
    index_rows = [row for path in index_files for row in _read_jsonl(path)]
    distinct_index_keys = {
        row.get("index_key") or row.get("capture_id")
        for row in index_rows
        if (row.get("index_key") or row.get("capture_id"))
    }

    schema = read_schema_manifest(base)
    with_task = sum(1 for r in records if r.get("task"))
    unresolved = sum(
        1
        for r in records
        if r.get("request_body_sha256")
        and not os.path.isfile(
            os.path.join(
                base,
                "blobs",
                "sha256",
                str(r.get("request_body_sha256"))[:2],
                str(r.get("request_body_sha256")),
            )
        )
    )

    print(f"capture root: {base}")
    version = schema["schema_version"] if schema else None
    declared = f"v{version} declared" if schema else "UNDECLARED"
    print(f"schema:       {declared}")
    print(f"distinct turns:            {len(records)}")
    print(f"raw capture files:         {len(capture_files)}")
    print(f"stream sidecar files:      {len(stream_files)}")
    print(f"index shard files:         {len(index_files)}")
    print(f"index rows (raw count):    {len(index_rows)}")
    print(f"index rows (distinct key): {len(distinct_index_keys)}")
    if index_rows and len(index_rows) != len(distinct_index_keys):
        print(
            f"  NOTE: {len(index_rows) / max(1, len(distinct_index_keys)):.2f}x "
            "inflation from multi-shard writes — dedupe on capture_id"
        )
    print(f"records with task label:   {with_task}/{len(records)}")
    print(f"records with unresolved body hash: {unresolved}")
    unindexed_streams = [
        p
        for p in stream_files
        if not any(p.endswith(str(r.get("stream_file", ""))) for r in records)
    ]
    print(f"stream files not reachable via index: {len(unindexed_streams)}")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Parse arguments and dispatch to the sub-command."""
    parser = argparse.ArgumentParser(
        description="Answer capture-corpus questions from disk.",
    )
    parser.add_argument(
        "--base",
        default=os.environ.get("CAPTURE_BASE_PATH", "/root/captures"),
        help="capture root (default: $CAPTURE_BASE_PATH or /root/captures)",
    )
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("tasks", help="task distribution").set_defaults(func=cmd_tasks)

    captures = sub.add_parser("captures", help="list matching captures")
    captures.add_argument("--task", help="filter by task label")
    captures.add_argument(
        "--stream", action="store_true", help="only streamed captures"
    )
    captures.add_argument("--request-type", help="filter by request_type")
    captures.add_argument("--limit", type=int, default=20)
    captures.add_argument("--json", action="store_true")
    captures.set_defaults(func=cmd_captures)

    sub.add_parser("stats", help="corpus integrity stats").set_defaults(func=cmd_stats)

    args = parser.parse_args(argv)
    if not getattr(args, "func", None):
        parser.print_help()
        return 1
    if not os.path.isdir(args.base):
        print(f"capture root does not exist: {args.base}", file=sys.stderr)
        return 2
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
