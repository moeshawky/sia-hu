"""
Routed-Experts Routing Capture

Owns: routing/ sidecar artifacts under {base_path}/routing/YYYY-MM-DD/ —
    one {capture_id}.routing.json per capture whose response carried
    routed_experts — under the proven carrier key ``sglext`` or, as
    forward-compat fallbacks, ``meta_info`` (SGLang
    --enable-return-routed-experts)
Depends: std (array, asyncio, base64, binascii, contextlib, json,
    datetime, pathlib)
Provides: decode_routed_experts, extract_routed_experts_b64,
    parse_routing_shape, DecodedRoutedExperts, RoutedExpertsLogger,
    RoutedExpertsWriter, ROUTING_ARTIFACT_SUFFIX, ROUTING_FILENAME,
    ROUTED_EXPERTS_FLAG

Invariants:
  - off mode (flag unset/invalid): no writer built, no per-request work
    beyond one None check at the spawn site; raw/ and errors/ bytes are
    never modified by this module
  - decode mirrors SGLang's extract_routed_experts_from_meta_info
    (srt/state_capturer/routed_experts.py:153-161): base64 -> raw int32
    bytes -> native-order int32 array, row-major
    (token, layer, topk) flattening; the width is the decoded tensor's,
    never a hardcoded 8/10/48
  - write_async never raises; returns None when nothing was written
  - artifacts are keyed by the existing capture_id (UUID) and follow the
    headroom sidecar file layout ({dir}/YYYY-MM-DD/{capture_id}.*.json);
    the original base64 stays inside the raw JSONL capture for audit
  - expert ids carry no PII (routing ids only); prompt text is never
    read or stored here
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import contextlib
import json
import re
from array import array
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, NamedTuple

# Suffix of the routing sidecar artifact written beside the raw capture.
ROUTING_ARTIFACT_SUFFIX = ".routing.json"

ROUTING_FILENAME = "{capture_id}" + ROUTING_ARTIFACT_SUFFIX

# Request flag the proxy injects upstream when the routing sidecar is
# armed. SGLang attaches the routed-experts blob to the
# response only when the request carries this key as true
# (protocol.py:896, default False); passive capture alone therefore
# sees zero routing data on normal traffic. The proxy never overrides
# a client-set value — a flag explicitly present in the client body
# (true or false) wins over injection.
ROUTED_EXPERTS_FLAG = "return_routed_experts"

# {capture_id}/LxK shape config, e.g. "48x10" for the CYBER-FROST
# qwen4_exp artifact. The parser accepts no hardcoded default shape:
# decode width comes from deployment config or nowhere.
_SHAPE_RE = re.compile(r"^(\d+)\s*[xX]\s*(\d+)$")


class DecodedRoutedExperts(NamedTuple):
    """Routed expert ids decoded from one response's routed-experts blob.

    Attributes:
        expert_ids: Flat row-major expert ids; index =
            (token * num_layers + layer) * topk + slot.
        num_tokens: Token rows derived from the decoded tensor length
            (len(expert_ids) // (num_layers * topk)); never from
            client-supplied metadata.
        num_layers: Layers per token as configured for this deployment.
        topk: Expert slots per layer as configured for this deployment.
    """

    expert_ids: list[int]
    num_tokens: int
    num_layers: int
    topk: int


def decode_routed_experts(
    raw_base64: str, *, num_layers: int, topk: int
) -> DecodedRoutedExperts:
    """Decode SGLang's base64 int32 routed-experts blob into int ids.

    Mirrors sglang.srt.state_capturer.routed_experts.extract_routed_experts_from_meta_info:
    base64-decode the meta_info string, read it back as native-order
    int32 (array('i') is the stdlib twin of SGLang's np.frombuffer
    dtype=np.int32 — both are native-order int32), then derive the token
    dimension from the tensor length. Layer and top-k widths come from
    the caller's config; nothing about them is hardcoded here.

    Args:
        raw_base64: base64 string from sglext.routed_experts (primary carrier) or meta_info.routed_experts (legacy fallback).
        num_layers: Model layer count (deployment config).
        topk: Experts per token (deployment config).

    Returns:
        DecodedRoutedExperts: ids plus derived shape.

    Raises:
        ValueError: num_layers/topk < 1, invalid base64, payload not a
            multiple of 4 bytes, itemsize mismatch, or tensor length not
            a multiple of num_layers * topk.
    """
    if num_layers < 1 or topk < 1:
        raise ValueError(f"num_layers and topk must be >= 1 (got {num_layers}, {topk})")
    try:
        blob = base64.b64decode(raw_base64, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("routed_experts is not valid base64") from exc
    if len(blob) % 4 != 0:
        raise ValueError(
            f"routed_experts payload is not int32-aligned ({len(blob)} bytes)"
        )
    ids = array("i")
    ids.frombytes(blob)
    if ids.itemsize != 4:
        raise ValueError(
            f"platform int32 is {ids.itemsize} bytes; cannot decode int32 blob"
        )
    unit = num_layers * topk
    if len(ids) % unit != 0:
        raise ValueError(
            f"routed_experts tensor length {len(ids)} is not a multiple of "
            f"num_layers*topk = {unit}"
        )
    return DecodedRoutedExperts(
        expert_ids=list(ids),
        num_tokens=len(ids) // unit,
        num_layers=num_layers,
        topk=topk,
    )


def extract_routed_experts_b64(
    response_body: dict[str, Any] | None,
) -> str | None:
    """Find the routed_experts base64 string in an upstream response.

    Check order (proven launch 2026-10-13): top-level
    ``sglext.routed_experts`` FIRST — the carrier SGLang actually uses
    for both chat and completions — then top-level
    ``meta_info.routed_experts`` and each choice's ``meta_info``
    (harmless forward-compat fallbacks; the proven server carries
    nothing in meta_info). Anything non-string (including null) is a
    miss, and a malformed sglext falls through to the fallback scan
    instead of aborting — extraction never raises on the response path.

    Args:
        response_body: Sanitized upstream response body, or None.

    Returns:
        str | None: The base64 string; None when the response carries no
            routed_experts (the default-off, zero-work signature).

    Raises: None
    """
    if not isinstance(response_body, dict):
        return None
    sglext = response_body.get("sglext")
    if isinstance(sglext, dict):
        raw = sglext.get("routed_experts")
        if isinstance(raw, str) and raw:
            return raw
    meta = response_body.get("meta_info")
    if isinstance(meta, dict):
        raw = meta.get("routed_experts")
        if isinstance(raw, str) and raw:
            return raw
    choices = response_body.get("choices")
    if isinstance(choices, list):
        for choice in choices:
            if not isinstance(choice, dict):
                continue
            choice_meta = choice.get("meta_info")
            if isinstance(choice_meta, dict):
                raw = choice_meta.get("routed_experts")
                if isinstance(raw, str) and raw:
                    return raw
    return None


def parse_routing_shape(raw: str) -> tuple[int, int] | None:
    """Parse an "LxK" routed-experts shape value into (layers, topk).

    Args:
        raw: Shape string such as "48x10"; "x" is case-insensitive.

    Returns:
        tuple[int, int] | None: (num_layers, topk) when valid and both
            dimensions >= 1; None for empty, malformed, or degenerate
            values (callers fail open to off).

    Raises: None
    """
    match = _SHAPE_RE.match(raw.strip()) if raw else None
    if match is None:
        return None
    layers, topk = int(match.group(1)), int(match.group(2))
    if layers < 1 or topk < 1:
        return None
    return layers, topk


class RoutedExpertsLogger:
    """Writes routing sidecar JSON files under the routing/ tree.

    A CaptureLogger fork for the routing/ directory: same per-call
    open/close style, day directories created on first write, OSError
    raised (never swallowed) so the writer's caller can suppress it.

    Invariants:
      - Files are per-capture (unique capture_id), written once each
      - Directory structure: {base_dir}/YYYY-MM-DD/
      - OSError propagates to the caller
    """

    def __init__(self, base_dir: Path) -> None:
        """Store the routing root directory.

        Args:
            base_dir: Root of the routing/ tree (default
                {capture_base_path}/routing).

        Returns: None

        Raises: None

        Invariants:
          - Before: No directories created
          - After: Base dir stored; directories created on first write
        """
        self._base_dir = Path(base_dir)

    def day_dir(self, created_at: str) -> Path:
        """Return the {base_dir}/YYYY-MM-DD directory for a record timestamp.

        Args:
            created_at: ISO 8601 timestamp; unparseable values fall back
                to the current UTC date.

        Returns:
            Path: The day directory (not created by this call).

        Raises: None
        """
        try:
            dt = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
            date_str = dt.strftime("%Y-%m-%d")
        except ValueError:
            date_str = datetime.now(UTC).strftime("%Y-%m-%d")
        return self._base_dir / date_str

    def write_json(self, path: Path, payload: dict[str, Any]) -> Path:
        """Write one routing JSON file, creating parent dirs as needed.

        Args:
            path: Destination file (under a routing day directory).
            payload: JSON-serializable dict.

        Returns:
            Path: The written file path.

        Raises:
            OSError: If directory creation or the file write fails.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        text = json.dumps(
            payload, ensure_ascii=False, separators=(",", ":"), default=str
        )
        path.write_text(text + "\n", encoding="utf-8")
        return path


class RoutedExpertsWriter:
    """Decodes and persists routed_experts for one capture, off-thread.

    Spawned fire-and-forget by CaptureProxyApp (same shape as
    HeadroomWriter): the decode and file write run in a worker thread
    via asyncio.to_thread, the event loop is never blocked, and
    write_async never raises.

    Invariants:
      - write_async never raises; returns None when nothing was written
      - the file is keyed by capture_id and never touches raw/ bytes
      - decode failures (malformed tensor) yield None, not exceptions;
        the original base64 remains in the raw capture for audit
    """

    def __init__(self, base_dir: Path, *, num_layers: int, topk: int) -> None:
        """Configure a routed-experts sidecar writer.

        Args:
            base_dir: Root of the routing/ tree.
            num_layers: Layer count for reshaping the decoded tensor.
            topk: Expert slots per layer for reshaping.

        Returns: None

        Raises:
            ValueError: num_layers or topk < 1.
        """
        if num_layers < 1 or topk < 1:
            raise ValueError("num_layers and topk must be >= 1")
        self._base_dir = Path(base_dir)
        self._num_layers = num_layers
        self._topk = topk
        self._logger = RoutedExpertsLogger(self._base_dir)
        self._pending: set[asyncio.Task[Any]] = set()

    async def write_async(
        self,
        *,
        capture_id: str,
        session_id: str,
        turn_id: str,
        model: str,
        created_at: str,
        raw_base64: str,
    ) -> Path | None:
        """Decode and persist one routing artifact in a worker thread.

        Args:
            capture_id: UUID capture identifier (validated upstream).
            session_id: Conversation id, copied for correlation.
            turn_id: Turn id, copied for correlation.
            model: Served model name.
            created_at: ISO 8601 record timestamp; selects the day dir.
            raw_base64: The sglext-or-meta_info routed_experts base64 string.

        Returns:
            Path | None: The written file path; None when decoding or
                writing failed.

        Raises: None (all failures return None).
        """
        try:
            return await asyncio.to_thread(
                self._write_sync,
                capture_id,
                session_id,
                turn_id,
                model,
                created_at,
                raw_base64,
            )
        except Exception:
            return None

    def _write_sync(
        self,
        capture_id: str,
        session_id: str,
        turn_id: str,
        model: str,
        created_at: str,
        raw_base64: str,
    ) -> Path | None:
        """Decode and write the routing artifact (runs in a worker thread).

        The decode happens HERE, in the worker thread — never on the
        event loop — so the response path stays free of tensor work.

        Args:
            capture_id: UUID capture identifier.
            session_id: Conversation id.
            turn_id: Turn id.
            model: Served model name.
            created_at: ISO 8601 record timestamp.
            raw_base64: The sglext-or-meta_info routed_experts base64 string.

        Returns:
            Path | None: The written file path; None when the tensor
                fails to decode (original base64 remains in raw/ for
                audit).

        Raises:
            OSError: If directory creation or the file write fails.
        """
        try:
            decoded = decode_routed_experts(
                raw_base64, num_layers=self._num_layers, topk=self._topk
            )
        except ValueError:
            return None
        path = self._logger.day_dir(created_at) / ROUTING_FILENAME.format(
            capture_id=capture_id
        )
        payload: dict[str, Any] = {
            "schema_version": 1,
            "capture_id": capture_id,
            "session_id": session_id,
            "turn_id": turn_id,
            "model": model,
            "created_at": created_at,
            "num_layers": decoded.num_layers,
            "topk": decoded.topk,
            "num_tokens": decoded.num_tokens,
            "layout": "flat row-major int32; index = "
            "(token * num_layers + layer) * topk + slot",
            "expert_ids": decoded.expert_ids,
        }
        return self._logger.write_json(path, payload)

    def keep(self, task: asyncio.Task[Any]) -> None:
        """Retain a reference to a running write task until it completes.

        Args:
            task: The asyncio task running write_async.

        Returns: None

        Raises: None

        Invariants:
          - The task is removed from the pending set when it finishes
          - Task exceptions are retrieved in the done callback (no
            'exception was never retrieved' warnings at GC)
        """
        self._pending.add(task)
        task.add_done_callback(self._on_task_done)

    def _on_task_done(self, task: asyncio.Task[Any]) -> None:
        """Drop a finished write task and retrieve its exception, if any."""
        self._pending.discard(task)
        with contextlib.suppress(Exception):
            task.exception()
