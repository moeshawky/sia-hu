"""
Capture Proxy Application

Owns: Route registration, middleware lifecycle, startup/shutdown, streaming tee
Depends: fastapi, httpx, capture_proxy modules
Provides: FastAPI app with /health, /v1/models, /v1/chat/completions, /v1/completions, /v1/embeddings, /v1/rerank, /v1/classification, /v1/images/generations, /v1/systemone, /v1/classify, /v1/batch-classify, /v1/ner, /v1/batch-ner, /v1/timeseries, /v1/batch-timeseries, /v1/sessions, /v1/sessions/{id}/captures, /v1/users, /v1/users/{id}/captures

Invariants:
  - Non-streaming responses logged BEFORE return to client; streaming responses logged in generator finally block — INV-2
  - Raw bytes forwarded BEFORE parse copy inspected — INV-3
  - SHA-256 of upstream bytes equals downstream bytes in transparent mode — INV-4 (llama.cpp: true; ollama embeddings: computed on ollama-translated body before forwarding)
  - Tool calls accumulated by index, not dropped — INV-5
  - Streaming failure (upstream_exception) results in upstream_status=502 — not 200 — in the capture record
  - Streaming upstream status is propagated to the client (4xx/5xx returned as JSONResponse, not HTTP 200)
  - Client disconnect during streaming sets client_disconnected (capture + terminal classification)
  - Headroom sidecar writes are scheduled after response finalization and never awaited on the response path
  - Headroom files live under {base_path}/headroom/ (or CAPTURE_HEADROOM_DIR) and never touch raw/ or errors/ bytes
  - Routed-experts routing sidecars ({capture_id}.routing.json under {base_path}/routing/) are scheduled beside the headroom spawn, fire-and-forget, and never modify raw/ or errors/ bytes; off by default (CAPTURE_ROUTED_EXPERTS unset)
  - Every new CaptureRecord carries a non-None user_id (x-user-id > Authorization hash > "anonymous"); session/user index appends are O(1), never raise, and query endpoints read ONLY {base_path}/index/ files — never raw/ JSONL scans
  - /v1/classification and /v1/images/generations are non-streaming OpenAI-passthrough routes: no backend translation branch, no ollama rejection, no stream field — the upstream body is forwarded and returned unchanged
  - /v1/systemone, /v1/classify, /v1/batch-classify, /v1/ner, /v1/batch-ner, /v1/timeseries, /v1/batch-timeseries are gpu-arm native JSON routes: body-verbatim passthrough with no declared request schema and no required-field validation — the body forwards as sent, the upstream response returns unchanged, and both sides are captured. Multipart /v1/transcribe is deliberately NOT registered
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import logging
import os
import sys
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import ValidationError

from capture_proxy.alias import (
    AliasConfigError,
    AliasMap,
    resolve_aliases,
    rewrite_request_model,
    sanitize_models_list,
    sanitize_response_model,
    sanitize_sse_event,
)
from capture_proxy.blobs import BlobStore, content_digest
from capture_proxy.headroom import HeadroomWriter, parse_byte_cap
from capture_proxy.logger import CaptureLogger
from capture_proxy.models import (
    CaptureIndexResponse,
    CaptureRecord,
    ChatCompletionRequest,
    ClassificationRequest,
    EmbeddingData,
    EmbeddingRequest,
    EmbeddingResponse,
    EmbeddingUsage,
    HeadroomRef,
    HealthResponse,
    IdListResponse,
    IdSummary,
    ImageRequest,
    IndexEntry,
    RerankRequest,
    StreamDiagnostics,
    UsageNormalized,
)
from capture_proxy.names import HEADROOM_SIGNALS, SIGNAL_KEYS
from capture_proxy.routing import (
    ROUTED_EXPERTS_FLAG,
    RoutedExpertsWriter,
    extract_routed_experts_b64,
    parse_routing_shape,
)
from capture_proxy.runtime import RuntimeSnapshot
from capture_proxy.schema import write_schema_manifest
from capture_proxy.session import SessionResolver
from capture_proxy.stream import SseParser, StreamLogger
from capture_proxy.taskclass import TaskClassification, classify_request
from capture_proxy.upstream import UpstreamClient

if TYPE_CHECKING:
    from collections.abc import AsyncIterator


def _compute_sha256(data: dict[str, Any] | list[Any] | str) -> str:
    """Compute deterministic SHA-256 hex digest of JSON-serializable data.

    Delegates to ``blobs.content_digest`` so the digest recorded on a
    CaptureRecord and the bytes written into the content-addressed blob
    store come from ONE serialization definition. Two implementations
    would let a stored blob and its recorded hash drift apart, which
    reproduces the exact dangling-handle defect this store exists to
    close.

    Args:
        data: dict or list to serialize and hash.

    Returns:
        Lowercase hex SHA-256 digest string.
    """
    return content_digest(data)


_logger = logging.getLogger(__name__)
_capture_write_warned = False
# Distinct raw GGUF path strings already warned about, so the GGUF refusal is
# logged once per path (once per distinct model deployment) rather than once per
# eligible capture. The set lives for the process lifetime and is never reset;
# tests use distinct paths to stay independent of it.
_gguf_warned_paths: set[str] = set()


def _blob_store_enabled() -> bool:
    """Return whether the content-addressed blob store is switched on.

    On by default. An emitted hash with no store behind it is a handle
    that dangles — that was the defect this store closes, so the default
    keeps it closed and an operator must opt OUT deliberately.

    Args:
        None. Reads ``CAPTURE_BLOB_STORE``.

    Returns:
        bool: False only when the env value is blank, ``"off"``, ``"0"``,
            ``"false"``, or ``"no"`` (case-insensitive); True otherwise.
    """
    raw = os.environ.get("CAPTURE_BLOB_STORE")
    if raw is None:
        return True
    return raw.strip().lower() not in {"", "off", "0", "false", "no"}


def _capture_write_failed(site: str, exc: OSError) -> None:
    """One-time named warning for a suppressed capture-disk write failure.

    The failure is diagnostic-only (the client response and the lossless
    forward are unaffected); naming the file and errno once per process
    keeps the hot path cheap while the fault stays visible.
    """
    global _capture_write_warned
    if _capture_write_warned:
        return
    _capture_write_warned = True
    _logger.warning(
        "capture write failed at %s (file=%s, errno=%s); "
        "continuing without that diagnostic",
        site,
        getattr(exc, "filename", None) or "unknown",
        exc.errno,
    )


_task_map_warned = False


def _task_map_warned_once(raw: str) -> None:
    """Emit one process-wide warning for an unusable CAPTURE_MODEL_TASKS value.

    _resolve_task runs per request, so an unparseable map would otherwise
    write to stderr on every captured call. The label degrades to None for
    the rest of the process (fail-open) and the fault stays visible once.
    The raw value is truncated before printing so an oversized env string
    cannot flood the log.

    Args:
        raw: The offending environment value, already known to be
            non-JSON or to decode to a non-object.

    Returns: None

    Raises: None

    Invariants:
        - Called at most once per process (guarded by ``_task_map_warned``).
        - Side effects: one stderr warning line; no request outcome changes.
    """
    global _task_map_warned
    if _task_map_warned:
        return
    _task_map_warned = True
    print(
        f"warning: invalid CAPTURE_MODEL_TASKS {raw[:120]!r}; task labels off",
        file=sys.stderr,
    )


class CaptureProxyApp:
    """FastAPI application — lossless streaming tee with diagnostics."""

    def __init__(self) -> None:
        """Initialize capture proxy application.

        Reads configuration from environment variables, sets up upstream client,
        logger, and session resolver. Detects backend type from upstream URL
        or explicit CAPTURE_BACKEND env var.

        Args: None

        Returns: None

        Raises: None

        Invariants:
            - Before: No state initialized
            - After: All clients ready, routes registered
            - _backend is the explicit CAPTURE_BACKEND value when set,
              "ollama" if port 11434 in URL, else "llamacpp" (fallback)
            - CAPTURE_BACKEND env var overrides auto-detection
            - vllm, node-llama-cpp, and sglang are OpenAI-compatible passthrough
              backends: chat/completions/embeddings/rerank proxy verbatim
            - _aliases is a validated AliasMap built once here
              (proxy-restart-only); invalid config warns to stderr and
              falls back to off (None), except duplicate-upstream which
              exits the process with code 2 naming the collision
        """
        self._upstream_url = os.environ.get(
            "CAPTURE_UPSTREAM_URL", "http://127.0.0.1:11434"
        )
        explicit_backend = os.environ.get("CAPTURE_BACKEND")
        if explicit_backend:
            self._backend = explicit_backend
        else:
            self._backend = "ollama" if "11434" in self._upstream_url else "llamacpp"
            if self._backend == "llamacpp" and ":8000" in self._upstream_url:
                print(
                    "warning: backend auto-detected as 'llamacpp'; if the "
                    "upstream is vLLM set CAPTURE_BACKEND=vllm (port 8000 is "
                    "ambiguous by design)",
                    file=sys.stderr,
                )
            if self._backend == "llamacpp" and ":30000" in self._upstream_url:
                print(
                    "warning: backend auto-detected as 'llamacpp'; if the "
                    "upstream is SGLang set CAPTURE_BACKEND=sglang (port "
                    "30000 is the SGLang default and is not auto-detected)",
                    file=sys.stderr,
                )
        self._session_resolver = SessionResolver()
        capture_base_path = os.environ.get("CAPTURE_BASE_PATH", "/root/captures")
        self._capture_base_path = capture_base_path
        self._capture_logger = CaptureLogger(base_path=capture_base_path)
        self._blob_store = BlobStore(capture_base_path, enabled=_blob_store_enabled())
        # Declare the schema before serving: a reader that arrives at a
        # corpus must be able to learn its version and field classes
        # without inferring them from a sample of records. The manifest is
        # also written per run, because the corpus-level file is
        # overwritten by the next run and cannot say which run produced
        # which records.
        try:
            boot_run_id = RuntimeSnapshot.get(
                capture_base_path=capture_base_path,
                backend=self._backend,
            ).run_id
        except OSError:
            boot_run_id = None
        write_schema_manifest(
            capture_base_path,
            blobs_enabled=self._blob_store.enabled,
            run_id=boot_run_id,
        )
        self._aliases: AliasMap | None = self._build_alias_map()
        self._headroom_writer = self._build_headroom_writer(capture_base_path)
        self._routed_experts_writer = self._build_routed_experts_writer(
            capture_base_path
        )
        self._upstream_client = UpstreamClient(upstream_url=self._upstream_url)

        self.app = FastAPI(title="capture-proxy", version="0.3.0")
        self._register_routes()

    @staticmethod
    def _build_alias_map() -> AliasMap | None:
        """Build the model-alias map once from CAPTURE_MODEL_ALIASES.

        Args: None (reads ``CAPTURE_MODEL_ALIASES`` env JSON string).

        Returns:
            AliasMap when configured and valid; None when off
                (unset, blank, ``"off"``) or when invalid (warning
                on stderr, fail-open to off).

        Raises:
            SystemExit: Code 2 on duplicate-upstream collision (fatal
                startup; message names the colliding upstream id).

        Invariants:
            - Off/invalid never raises; duplicate-upstream always exits.
            - Side effects: stderr warning on invalid; no I/O otherwise.
        """
        raw = os.environ.get("CAPTURE_MODEL_ALIASES")
        if raw is None or raw.strip() == "":
            return None
        try:
            return resolve_aliases(raw)
        except AliasConfigError as err:
            if "duplicate upstream" in str(err).lower():
                print(f"error: duplicate model alias upstream: {err}", file=sys.stderr)
                sys.exit(2)
            print(
                f"warning: invalid CAPTURE_MODEL_ALIASES {raw!r} ({err}); aliases off",
                file=sys.stderr,
            )
            return None

    def _resolve_task(self, model: str) -> str | None:
        """Resolve the task label configured for a public model id.

        Args:
            model: Public model id as sent by the client (the same value
                stored in CaptureRecord.model). Never the forwarded
                upstream id — the map is keyed on what the client asks for.

        Returns:
            str | None: The configured task string on an exact
                case-sensitive key hit; None when the map is off (unset,
                blank, ``"off"``), the JSON is malformed, the key is
                absent, or the configured value is not a string.

        Raises:
            None — never. A malformed map degrades to None (fail-open) so
            a bad env value cannot fail a request.

        Invariants:
            - Pre: ``model`` is any string; no lookup runs for it.
            - Post: return is either None or a non-empty stripped string.
            - Side effects: reads ``CAPTURE_MODEL_TASKS`` on every call
              (an env edit takes effect without a proxy restart) and
              emits at most one stderr warning per process for a
              malformed map, reusing the ``_capture_write_warned``
              one-time-flag pattern.
            - Exact case-sensitive full-string match only; never a
              substring, prefix, or case-folded lookup.
        """
        raw = os.environ.get("CAPTURE_MODEL_TASKS")
        if raw is None or raw.strip() == "" or raw.strip().lower() == "off":
            return None
        try:
            decoded: Any = json.loads(raw)
        except json.JSONDecodeError:
            _task_map_warned_once(raw)
            return None
        if not isinstance(decoded, dict):
            _task_map_warned_once(raw)
            return None
        value = decoded.get(model)
        if not isinstance(value, str):
            return None
        cleaned = value.strip()
        return cleaned or None

    def _derive_task(
        self, public_model: str, request: dict[str, Any], request_type: str | None
    ) -> TaskClassification:
        """Label a capture, preferring the operator map over inference.

        Precedence:
          1. ``CAPTURE_MODEL_TASKS`` — an explicit operator statement
             about the served model outranks any inference, and it keeps
             its existing behavior and warning semantics
          2. structural classification of the request shape

        The structural path is what makes ``task`` non-null in practice.
        The map is off by default, so on a default deployment the field
        used to be null in every single capture; inference from request
        shape is what closes that.

        Args:
            public_model: The PUBLIC model id — the value in
                ``CaptureRecord.model``, i.e. what the client asked for.
                Passed separately from ``request`` because under aliasing
                ``request["model"]`` is the FORWARDED upstream id. The
                documented contract keys the map on the public id only;
                reading it off the forwarded body silently breaks every
                aliased deployment.
            request: Forwarded request body, used for structural shape.
            request_type: Endpoint family, when the handler knows it.

        Returns:
            TaskClassification: The operator label wrapped as a
                classification with ``rule="operator_model_task_map"`` and
                empty evidence (the map asserts a fact, it does not
                measure one), else the structural classification.

        Raises: None
        """
        configured = self._resolve_task(public_model)
        if configured is not None:
            return TaskClassification(
                task=configured,
                rule="operator_model_task_map",
                evidence={},
            )
        return classify_request(request, request_type)

    def _apply_task(self, record: CaptureRecord, request_type: str | None) -> None:
        """Populate ``task``/``task_rule``/``task_evidence`` on a record.

        Args:
            record: The capture record to label, in place.
            request_type: Endpoint family for the structural classifier.
                None when the handler does not declare one; the classifier
                then infers from body shape alone.

        Returns: None

        Raises: None
        """
        classification = self._derive_task(record.model, record.request, request_type)
        record.task = classification.task
        record.task_rule = classification.rule
        record.task_evidence = classification.evidence or None

    def _store_request_blobs(self, record: CaptureRecord) -> None:
        """Persist the request body and messages array into the blob store.

        Makes ``request_body_sha256`` and ``messages_sha256`` resolvable
        handles instead of dangling ones — the defect was that every
        sampled capture carried both hashes with no store anywhere behind
        them. Writing both is deduplicated by the store: identical bodies
        across turns cost one stat, not a second file.

        Args:
            record: Capture record whose hashes should resolve.

        Returns: None

        Raises: None (OSError suppressed inside BlobStore)
        """
        if not self._blob_store.enabled:
            return
        self._blob_store.put(record.request)
        messages = record.request.get("messages")
        if isinstance(messages, list):
            self._blob_store.put(messages)
        # Embeddings carry no messages array. Their `messages_sha256` and
        # `input_sha256` are both the digest of `request["input"]` (set in
        # the embedding handler), so writing only `request` leaves two
        # dangling handles on every embedding capture — the same defect
        # this store exists to close, one request type over. The preimage
        # is `request["input"]` verbatim: re-hashing it reproduces
        # `input_sha256` exactly, because both go through
        # `blobs.canonical_bytes`.
        embedding_input = record.request.get("input")
        if embedding_input is not None:
            self._blob_store.put(embedding_input)

    @staticmethod
    def _parse_headroom_weights(
        raw: str,
    ) -> Literal["off", "gate", "all"]:
        """Validate a CAPTURE_HEADROOM_WEIGHTS value.

        Args:
            raw: Raw env value.

        Returns:
            Literal: The value when it is one of "off"/"gate"/"all",
                else "off" (invalid input falls back to off).

        Raises: None
        """
        if raw == "gate":
            return "gate"
        if raw == "all":
            return "all"
        return "off"

    def _build_headroom_writer(self, capture_base_path: str) -> HeadroomWriter | None:
        """Build the headroom sidecar writer from CAPTURE_HEADROOM* env vars.

        Args:
            capture_base_path: Capture base path; the default headroom dir
                is {capture_base_path}/headroom.

        Returns:
            HeadroomWriter | None: A configured writer, or None when
                CAPTURE_HEADROOM is "off" (the default) or a configured
                value is invalid (warning on stderr, treated as off).

        Raises: None

        Invariants:
          - Returns None without importing torch/safetensors (off means
            zero overhead: no files, no imports)
          - CAPTURE_HEADROOM_DIR overrides the base_path-derived default
        """
        mode = os.environ.get("CAPTURE_HEADROOM", "off")
        if mode == "off":
            return None
        if mode == "all":
            signals = frozenset(HEADROOM_SIGNALS)
        else:
            requested = {name.strip() for name in mode.split(",") if name.strip()}
            unknown = requested.difference(HEADROOM_SIGNALS)
            if not requested or unknown:
                print(
                    f"warning: invalid CAPTURE_HEADROOM {mode!r}; headroom off",
                    file=sys.stderr,
                )
                return None
            signals = frozenset(requested)
        weights_raw = os.environ.get("CAPTURE_HEADROOM_WEIGHTS", "off")
        weights = self._parse_headroom_weights(weights_raw)
        if weights != weights_raw:
            print(
                f"warning: invalid CAPTURE_HEADROOM_WEIGHTS {weights_raw!r}; using off",
                file=sys.stderr,
            )
        max_bytes_raw = os.environ.get("CAPTURE_HEADROOM_MAX_BYTES", "0")
        try:
            max_bytes = parse_byte_cap(max_bytes_raw)
        except ValueError as err:
            print(f"warning: {err}; using 0", file=sys.stderr)
            max_bytes = 0
        interval_raw = os.environ.get("CAPTURE_HEADROOM_INTERVAL", "0")
        try:
            interval = int(interval_raw)
            if interval < 0:
                raise ValueError(f"invalid CAPTURE_HEADROOM_INTERVAL {interval_raw!r}")
        except ValueError as err:
            print(f"warning: {err}; using 0", file=sys.stderr)
            interval = 0
        headroom_dir = os.environ.get("CAPTURE_HEADROOM_DIR", "")
        base_dir = (
            Path(headroom_dir) if headroom_dir else Path(capture_base_path) / "headroom"
        )
        return HeadroomWriter(
            base_dir,
            weights=weights,
            max_bytes=max_bytes,
            interval=interval,
            signals=signals,
        )

    @staticmethod
    def _build_routed_experts_writer(
        capture_base_path: str,
    ) -> RoutedExpertsWriter | None:
        """Build the routing sidecar writer from CAPTURE_ROUTED_EXPERTS* env vars.

        Args:
            capture_base_path: Capture base path; the default routing dir
                is {capture_base_path}/routing.

        Returns:
            RoutedExpertsWriter | None: A configured writer, or None when
                CAPTURE_ROUTED_EXPERTS is "off" (the default), blank, or
                CAPTURE_ROUTED_EXPERTS_SHAPE is missing/invalid (warning on
                stderr, fail-open to off).

        Raises: None

        Invariants:
          - Returns None without any per-request cost (one None check at
            the spawn site is the entire off-mode overhead)
          - The LxK tensor shape is deployment config; the artifact
            (48x10 for qwen4_exp) is never hardcoded here
          - CAPTURE_ROUTED_EXPERTS_DIR overrides the base_path-derived default
        """
        mode = os.environ.get("CAPTURE_ROUTED_EXPERTS", "off")
        if mode.strip() == "" or mode.strip().lower() == "off":
            return None
        shape_raw = os.environ.get("CAPTURE_ROUTED_EXPERTS_SHAPE", "")
        shape = parse_routing_shape(shape_raw)
        if shape is None:
            print(
                f"warning: invalid CAPTURE_ROUTED_EXPERTS_SHAPE {shape_raw!r}; "
                "routed-experts capture off",
                file=sys.stderr,
            )
            return None
        num_layers, topk = shape
        routing_dir = os.environ.get("CAPTURE_ROUTED_EXPERTS_DIR", "")
        base_dir = (
            Path(routing_dir) if routing_dir else Path(capture_base_path) / "routing"
        )
        return RoutedExpertsWriter(
            base_dir,
            num_layers=num_layers,
            topk=topk,
        )

    def _spawn_routing_capture(self, record: CaptureRecord) -> None:
        """Schedule a routed-experts sidecar write when the response carries one.

        Fire-and-forget like _spawn_headroom: the decode runs in a worker
        thread and the raw JSONL line is never modified, so the capture
        bytes are identical whether or not routed_experts is present.

        Eligibility: writer built (CAPTURE_ROUTED_EXPERTS=on plus a valid
        shape), request_type in (chat, completion), upstream success,
        valid-UUID capture_id, and a string routed_experts payload in
        the captured response body — primary carrier top-level
        sglext.routed_experts (base64 int32); meta_info paths are
        legacy fallback only.

        Args:
            record: Finalized CaptureRecord (response populated).

        Returns: None

        Raises: None

        Invariants:
          - Returns immediately when the writer is None (flag off) or the
            response carries no routed_experts — zero new decode/IO work
          - The base64 payload itself stays inside the raw capture; this
            sidecar only adds decoded ids keyed by capture_id
        """
        writer = self._routed_experts_writer
        if writer is None:
            return
        if record.request_type not in ("chat", "completion"):
            return
        if record.upstream_status >= 400 or record.error_body is not None:
            return
        if not record.capture_id or record.response is None:
            return
        raw = extract_routed_experts_b64(record.response)
        if raw is None:
            return
        try:
            uuid.UUID(record.capture_id)
        except (ValueError, AttributeError, TypeError):
            return
        task = asyncio.create_task(
            writer.write_async(
                capture_id=record.capture_id,
                session_id=record.session_id,
                turn_id=record.turn_id,
                model=record.model,
                created_at=record.created_at,
                raw_base64=raw,
            )
        )
        writer.keep(task)

    def _register_routes(self) -> None:
        @self.app.get("/health")
        async def health() -> HealthResponse:
            upstream_ok = await self._upstream_client.health_check(
                path="/api/tags" if self._backend == "ollama" else "/health"
            )
            return HealthResponse(
                status="ok",
                upstream="connected" if upstream_ok else "disconnected",
            )

        @self.app.get("/v1/models")
        async def models(request: Request) -> Response:
            headers = dict(request.headers)
            try:
                upstream_resp = await self._upstream_client.proxy(
                    method="GET",
                    path="/v1/models",
                    headers=headers,
                )
                # Invariant: sanitize-before-serialize — remap ids via
                # reverse map and drop root before returning to client.
                sanitized = sanitize_models_list(upstream_resp.body, self._aliases)
                return Response(
                    content=json.dumps(sanitized).encode() if sanitized else b"",
                    status_code=upstream_resp.status_code,
                    media_type="application/json",
                )
            except Exception as e:
                with contextlib.suppress(OSError):
                    self._capture_logger.log_error(
                        {
                            "turn_id": str(uuid.uuid4()),
                            "created_at": datetime.now(UTC).isoformat(),
                            "error_body": str(e),
                            "request_type": "models",
                            "error_category": "connection_error",
                        }
                    )
                return JSONResponse(
                    status_code=502,
                    content={"error": "upstream unavailable"},
                )

        @self.app.post("/v1/chat/completions", response_model=None)
        async def chat_completions(request: Request) -> Response | StreamingResponse:
            return await self._handle_completion_request(
                request=request,
                upstream_path="/v1/chat/completions",
                response_object="chat.completion",
            )

        @self.app.post("/v1/completions", response_model=None)
        async def completions(request: Request) -> Response | StreamingResponse:
            return await self._handle_completion_request(
                request=request,
                upstream_path="/v1/completions",
                response_object="text_completion",
            )

        @self.app.post("/v1/embeddings", response_model=None)
        async def embeddings(request: Request) -> Response:
            return await self._handle_embedding_request(request=request)

        @self.app.post("/v1/rerank", response_model=None)
        async def rerank(request: Request) -> Response:
            return await self._handle_rerank_request(request=request)

        @self.app.post("/v1/classification", response_model=None)
        async def classification(request: Request) -> Response:
            return await self._handle_classification_request(request=request)

        @self.app.post("/v1/images/generations", response_model=None)
        async def image_generations(request: Request) -> Response:
            return await self._handle_image_request(request=request)

        # gpu-arm native JSON routes: body-verbatim passthrough.
        # No declared schema, no required-field validation — the body forwards
        # as sent and the upstream response returns unchanged.
        @self.app.post("/v1/systemone", response_model=None)
        async def systemone(request: Request) -> Response:
            return await self._handle_native_passthrough(
                request=request,
                upstream_path="/v1/systemone",
                request_type="systemone",
            )

        @self.app.post("/v1/classify", response_model=None)
        async def classify(request: Request) -> Response:
            return await self._handle_native_passthrough(
                request=request,
                upstream_path="/v1/classify",
                request_type="classify",
            )

        @self.app.post("/v1/batch-classify", response_model=None)
        async def batch_classify(request: Request) -> Response:
            return await self._handle_native_passthrough(
                request=request,
                upstream_path="/v1/batch-classify",
                request_type="batch_classify",
            )

        @self.app.post("/v1/ner", response_model=None)
        async def ner(request: Request) -> Response:
            return await self._handle_native_passthrough(
                request=request,
                upstream_path="/v1/ner",
                request_type="ner",
            )

        @self.app.post("/v1/batch-ner", response_model=None)
        async def batch_ner(request: Request) -> Response:
            return await self._handle_native_passthrough(
                request=request,
                upstream_path="/v1/batch-ner",
                request_type="batch_ner",
            )

        @self.app.post("/v1/timeseries", response_model=None)
        async def timeseries(request: Request) -> Response:
            return await self._handle_native_passthrough(
                request=request,
                upstream_path="/v1/timeseries",
                request_type="timeseries",
            )

        @self.app.post("/v1/batch-timeseries", response_model=None)
        async def batch_timeseries(request: Request) -> Response:
            return await self._handle_native_passthrough(
                request=request,
                upstream_path="/v1/batch-timeseries",
                request_type="batch_timeseries",
            )

        @self.app.get("/v1/sessions", response_model=None)
        async def list_sessions(limit: int = 100) -> IdListResponse:
            """List known session ids, most recently active first.

            Args:
                limit: Max ids returned (clamped to 0..500).

            Returns:
                IdListResponse: ids with capture counts and activity bounds.
            """
            return self._list_ids("sessions", "session", limit)

        @self.app.get("/v1/sessions/{session_id}/captures", response_model=None)
        async def session_captures(session_id: str) -> CaptureIndexResponse:
            """Return every capture indexed under one session_id.

            Reads only the session's index file — O(captures of this
            session), never a raw/ scan. Unknown ids yield count 0 with an
            empty captures list (200, not 404/500).
            """
            return self._query_index("sessions", "session", session_id)

        @self.app.get("/v1/users", response_model=None)
        async def list_users(limit: int = 100) -> IdListResponse:
            """List known user ids, most recently active first.

            Args:
                limit: Max ids returned (clamped to 0..500).

            Returns:
                IdListResponse: ids with capture counts and activity bounds.
            """
            return self._list_ids("users", "user", limit)

        @self.app.get("/v1/users/{user_id}/captures", response_model=None)
        async def user_captures(user_id: str) -> CaptureIndexResponse:
            """Return every capture indexed under one user_id.

            Reads only the user's index file — O(captures of this user),
            never a raw/ scan. Unknown ids yield count 0 with an empty
            captures list (200, not 404/500).
            """
            return self._query_index("users", "user", user_id)

    async def _handle_completion_request(
        self,
        request: Request,
        upstream_path: str,
        response_object: str,
    ) -> Response | StreamingResponse:
        """Handle chat/completions with alias rewrite and routing injection.

        Invariants:
            - rewrite-before-hash — the public model id is replaced with
              the upstream id after validation and before SHA-256
            - routing-injection — when the routing sidecar
              is armed and the client did not set the flag, the FORWARDED
              body gains return_routed_experts=true; a client-set flag
              (true or false) is never overridden. The raw capture keeps
              the client-sent body and request_body_sha256 covers the
              forwarded body — the injected flag is the one canonical
              diff. Streaming and
              non-streaming share this path.
        """
        body_bytes = await request.body()
        try:
            body_dict: dict[str, Any] = json.loads(body_bytes) if body_bytes else {}
        except json.JSONDecodeError:
            return JSONResponse(status_code=422, content={"error": "invalid JSON"})

        headers = dict(request.headers)
        try:
            request_data = ChatCompletionRequest(**body_dict)
        except (ValidationError, TypeError) as e:
            session_id = self._session_resolver.resolve(headers)
            turn_id = str(uuid.uuid4())
            capture_id = str(uuid.uuid4())
            created_at = datetime.now(UTC).isoformat()
            client_ip = request.client.host if request.client else "unknown"
            client_ua = headers.get("user-agent", "")
            record = CaptureRecord(
                session_id=session_id,
                user_id=self._session_resolver.resolve_user_id(headers),
                turn_id=turn_id,
                capture_id=capture_id,
                created_at=created_at,
                latency_ms=0.0,
                client_ip=client_ip,
                client_user_agent=client_ua,
                model=str(body_dict.get("model") or ""),
                run_id="unknown",
                runtime_settings=None,
                request=body_dict,
                request_body_sha256=_compute_sha256(body_dict),
                messages_sha256=_compute_sha256(body_dict.get("messages", [])),
                attempt_number=1,
                upstream_status=422,
                error_body=str(e),
                request_type="chat"
                if response_object.startswith("chat")
                else "completion",
                task=self._resolve_task(str(body_dict.get("model") or "")),
                routing=None,
            )
            self._log_capture(record, extra={"error_category": "validation_error"})
            return JSONResponse(status_code=422, content={"error": str(e)})

        is_stream = request_data.stream

        session_id = self._session_resolver.resolve(headers)
        turn_id = str(uuid.uuid4())
        capture_id = str(uuid.uuid4())
        created_at = datetime.now(UTC).isoformat()
        client_ip = request.client.host if request.client else "unknown"
        client_ua = headers.get("user-agent", "")

        runtime_snapshot = await asyncio.to_thread(
            RuntimeSnapshot.get,
            capture_base_path=self._capture_base_path,
            backend=self._backend,
        )

        forwarded_body, alias_original, alias_forwarded = rewrite_request_model(
            body_dict, self._aliases
        )
        # Routing injection: an armed sidecar requires the
        # upstream to see return_routed_experts=true, or the response
        # carries no routing blob and passive capture sees nothing.
        # Off mode costs one None check; a client-set flag (true or
        # false) is never overridden.
        upstream_body = forwarded_body
        if (
            self._routed_experts_writer is not None
            and ROUTED_EXPERTS_FLAG not in forwarded_body
        ):
            upstream_body = {**forwarded_body, ROUTED_EXPERTS_FLAG: True}
            # request_sha256 below covers upstream_body; store its
            # preimage so the handle resolves:
            # the injected flag is the one canonical diff against the
            # client-sent body kept in record.request.
            self._blob_store.put(upstream_body)
        request_sha256 = _compute_sha256(upstream_body)
        messages_list = forwarded_body.get("messages", [])
        messages_sha256 = _compute_sha256(messages_list)
        public_model = (
            alias_original if alias_original is not None else request_data.model
        )

        record = CaptureRecord(
            session_id=session_id,
            user_id=self._session_resolver.resolve_user_id(headers),
            turn_id=turn_id,
            capture_id=capture_id,
            created_at=created_at,
            latency_ms=0.0,
            client_ip=client_ip,
            client_user_agent=client_ua,
            model=public_model,
            request_model_original=alias_original,
            request_model_forwarded=alias_forwarded,
            run_id=runtime_snapshot.run_id,
            runtime_settings=runtime_snapshot.settings,
            request=forwarded_body,
            request_body_sha256=request_sha256,
            messages_sha256=messages_sha256,
            attempt_number=1,
            upstream_status=200,
            request_type="chat" if response_object.startswith("chat") else "completion",
            task=self._resolve_task(public_model),
            routing=None,
        )

        start_time = time.monotonic()
        try:
            if is_stream:
                return await self._handle_streaming(
                    headers,
                    upstream_body,
                    record,
                    start_time,
                    upstream_path,
                    response_object,
                )
            return await self._handle_non_streaming(
                headers,
                upstream_body,
                record,
                start_time,
                upstream_path,
            )
        except Exception as e:
            elapsed_ms = (time.monotonic() - start_time) * 1000
            record.latency_ms = elapsed_ms
            record.upstream_status = 502
            record.error_body = str(e)
            record.stream_diagnostics = StreamDiagnostics(
                stream_exception=str(e),
            )
            self._log_capture(record, extra={"error_category": "connection_error"})
            return JSONResponse(
                status_code=502,
                content={"error": "upstream unavailable"},
            )

    async def _handle_non_streaming(
        self,
        headers: dict[str, str],
        body_dict: dict[str, Any],
        record: CaptureRecord,
        start_time: float,
        upstream_path: str,
    ) -> Response:
        """Handle non-streaming: forward, log, return.

        Invariant: sanitize-before-log-and-serialize — the upstream
            model id is remapped to public before capture and client.
        """
        upstream_resp = await self._upstream_client.proxy(
            method="POST",
            path=upstream_path,
            headers=headers,
            body=body_dict,
            stream=False,
        )

        elapsed_ms = (time.monotonic() - start_time) * 1000
        record.latency_ms = elapsed_ms
        sanitized = sanitize_response_model(upstream_resp.body, self._aliases)
        record.response = sanitized
        record.upstream_status = upstream_resp.status_code

        if sanitized:
            choices = sanitized.get("choices", [])
            if choices:
                choice = choices[0]
                msg = choice.get("message", {})
                record.tool_calls = msg.get("tool_calls")
                record.finish_reason = choice.get("finish_reason")
            usage = sanitized.get("usage")
            if usage:
                record.usage_raw = usage
                record.usage_normalized = _normalize_usage(usage)

        self._spawn_headroom(record)
        self._spawn_routing_capture(record)
        self._log_capture(record)

        content = json.dumps(sanitized).encode() if sanitized else b""
        return Response(
            content=content,
            status_code=upstream_resp.status_code,
            media_type="application/json",
        )

    def _translate_ollama_response(
        self, raw: dict[str, Any], model: str
    ) -> dict[str, Any]:
        """Translate ollama /api/embed response to OpenAI /v1/embeddings format.

        Args:
            raw: Raw ollama response dict containing "embeddings" list of float lists.
            model: Model name from the original request.

        Returns:
            dict: OpenAI-compatible EmbeddingResponse as dict with:
                - object: "list"
                - data: [{object: "embedding", index: i, embedding: v}]
                - model: model name
                - usage: {prompt_tokens: 0, total_tokens: 0}

        Raises: None

        Invariants:
            - len(data) == len(raw["embeddings"])
            - Each embedding entry has object="embedding" and sequential index
            - Usage is zeroed (ollama does not report token counts)
        """
        ollama_embeddings = raw.get("embeddings", [])
        data = [
            EmbeddingData(index=i, embedding=vec)
            for i, vec in enumerate(ollama_embeddings)
        ]
        response = EmbeddingResponse(
            data=data,
            model=model,
            usage=EmbeddingUsage(prompt_tokens=0, total_tokens=0),
        )
        return response.model_dump()

    async def _handle_embedding_request(self, request: Request) -> Response:
        """Handle POST /v1/embeddings — forward to upstream and log.

        Args:
            request: FastAPI Request with embedding body.

        Returns:
            Response: JSON response from upstream (OpenAI-compatible format).
                - 200: Successful embedding response
                - 400: Stream=True (embeddings don't stream)
                - 422: Invalid JSON or empty input/model
                - 502: Upstream unavailable

        Raises: None

        Invariants:
            - Before: Request body not yet parsed
            - After: CaptureRecord written with request_type="embedding"
              (early-validation 4xx returns exit before capture and are
              not logged)
            - input_sha256 is deterministic for identical inputs
            - stream=True always returns 400
            - embedding_count matches len(input) from request
            - For llamacpp/node-llama-cpp/vllm: proxied to /v1/embeddings unchanged
            - For ollama: body translated to /api/embed, response translated back
            - INV-4: request_body_sha256 is computed on the body actually forwarded to upstream (ollama_body for ollama, body_dict for llamacpp)
            - INV-5: record.request mirrors the body actually forwarded to upstream (ollama_body for ollama)
            - Alias rewrite-before-hash: public model replaced with upstream
              id after validation and before SHA-256 and branch dispatch.
            - Alias sanitize-before-log-and-serialize: upstream model
              remapped to public before capture and client.
            - A capture-disk OSError while logging the validation failure is
              suppressed — the 422 response is unaffected (diagnostic record
              lost, request served)
        """
        body_bytes = await request.body()
        try:
            body_dict: dict[str, Any] = json.loads(body_bytes) if body_bytes else {}
        except json.JSONDecodeError:
            return JSONResponse(status_code=422, content={"error": "invalid JSON"})

        try:
            request_data = EmbeddingRequest(**body_dict)
        except Exception as e:
            with contextlib.suppress(OSError):
                self._capture_logger.log_error(
                    {
                        "turn_id": str(uuid.uuid4()),
                        "created_at": datetime.now(UTC).isoformat(),
                        "error_body": str(e),
                        "request_type": "embedding",
                        "error_category": "validation_error",
                    }
                )
            return JSONResponse(
                status_code=422, content={"error": "invalid embedding request"}
            )

        # Validate input is non-empty
        input_val = request_data.input
        if isinstance(input_val, str) and not input_val.strip():
            return JSONResponse(
                status_code=422, content={"error": "input must not be empty"}
            )
        if isinstance(input_val, list) and not input_val:
            return JSONResponse(
                status_code=422, content={"error": "input list must not be empty"}
            )

        # Validate model is non-empty
        if not request_data.model.strip():
            return JSONResponse(
                status_code=422, content={"error": "model must not be empty"}
            )

        # Embeddings don't stream
        if request_data.stream:
            return JSONResponse(
                status_code=400,
                content={"error": "streaming is not supported for embeddings"},
            )

        headers = dict(request.headers)
        session_id = self._session_resolver.resolve(headers)
        turn_id = str(uuid.uuid4())
        capture_id = str(uuid.uuid4())
        created_at = datetime.now(UTC).isoformat()
        client_ip = request.client.host if request.client else "unknown"
        client_ua = headers.get("user-agent", "")

        runtime_snapshot = await asyncio.to_thread(
            RuntimeSnapshot.get,
            capture_base_path=self._capture_base_path,
            backend=self._backend,
        )

        # Compute input_sha256 for embedding inputs
        input_sha256 = _compute_sha256(input_val)

        # request_body_sha256 is computed on the body actually forwarded to upstream;
        # see per-branch blocks below (RC-3 fix).
        request_sha256 = ""

        # Compute embedding count
        embedding_count = len(input_val) if isinstance(input_val, list) else 1

        forwarded_body, alias_original, alias_forwarded = rewrite_request_model(
            body_dict, self._aliases
        )
        forwarded_model = (
            alias_forwarded if alias_forwarded is not None else request_data.model
        )

        record = CaptureRecord(
            session_id=session_id,
            user_id=self._session_resolver.resolve_user_id(headers),
            turn_id=turn_id,
            capture_id=capture_id,
            created_at=created_at,
            latency_ms=0.0,
            client_ip=client_ip,
            client_user_agent=client_ua,
            model=alias_original if alias_original is not None else request_data.model,
            request_model_original=alias_original,
            request_model_forwarded=alias_forwarded,
            run_id=runtime_snapshot.run_id,
            runtime_settings=runtime_snapshot.settings,
            request=forwarded_body,
            request_body_sha256=request_sha256,
            messages_sha256=input_sha256,
            attempt_number=1,
            upstream_status=200,
            routing=None,
            request_type="embedding",
            task=self._resolve_task(
                alias_original if alias_original is not None else request_data.model
            ),
            embedding_count=embedding_count,
            input_sha256=input_sha256,
        )

        start_time = time.monotonic()
        try:
            if self._backend == "ollama":
                # Translate to ollama format and proxy to /api/embed
                ollama_body: dict[str, Any] = {
                    "model": forwarded_model,
                }
                ollama_body["input"] = input_val

                # Compute SHA-256 on the actual forwarded body (ollama format) — RC-3 fix
                request_sha256 = _compute_sha256(ollama_body)
                record.request_body_sha256 = request_sha256
                # Record the body actually forwarded upstream (ollama format)
                record.request = ollama_body

                upstream_resp = await self._upstream_client.proxy(
                    method="POST",
                    path="/api/embed",
                    headers=headers,
                    body=ollama_body,
                    stream=False,
                )

                elapsed_ms = (time.monotonic() - start_time) * 1000
                record.latency_ms = elapsed_ms
                record.upstream_status = upstream_resp.status_code

                if upstream_resp.status_code != 200:
                    sanitized_error = sanitize_response_model(
                        upstream_resp.body, self._aliases
                    )
                    record.response = sanitized_error
                    record.error_body = (
                        json.dumps(sanitized_error) if sanitized_error else None
                    )
                    self._log_capture(record)
                    content = (
                        json.dumps(sanitized_error).encode() if sanitized_error else b""
                    )
                    return Response(
                        content=content,
                        status_code=upstream_resp.status_code,
                        media_type="application/json",
                    )

                # Translate ollama response to OpenAI format
                openai_response = self._translate_ollama_response(
                    upstream_resp.body or {}, forwarded_model
                )
                sanitized_embed = sanitize_response_model(
                    openai_response, self._aliases
                )
                if sanitized_embed is None:
                    sanitized_embed = openai_response
                record.response = sanitized_embed

                # Extract embedding dimensions from first embedding
                if sanitized_embed.get("data"):
                    first_emb = sanitized_embed["data"][0].get("embedding", [])
                    record.embedding_dimensions = len(first_emb)

                self._log_capture(record)

                content = json.dumps(sanitized_embed).encode()
                return Response(
                    content=content,
                    status_code=200,
                    media_type="application/json",
                )

            # OpenAI-compatible backend (llamacpp/node-llama-cpp/vllm):
            # proxy to /v1/embeddings with same body — RC-3 fix
            # Compute SHA-256 on the actual forwarded body (OpenAI format, unchanged)
            request_sha256 = _compute_sha256(forwarded_body)
            record.request_body_sha256 = request_sha256

            upstream_resp = await self._upstream_client.proxy(
                method="POST",
                path="/v1/embeddings",
                headers=headers,
                body=forwarded_body,
                stream=False,
            )

            elapsed_ms = (time.monotonic() - start_time) * 1000
            record.latency_ms = elapsed_ms
            sanitized_passthrough = sanitize_response_model(
                upstream_resp.body, self._aliases
            )
            record.response = sanitized_passthrough
            record.upstream_status = upstream_resp.status_code

            # Extract embedding dimensions from response
            if sanitized_passthrough and sanitized_passthrough.get("data"):
                first_emb = sanitized_passthrough["data"][0].get("embedding", [])
                record.embedding_dimensions = len(first_emb)

            # Route through _log_capture rather than the logger directly:
            # the choke point is where task labelling and blob storage
            # happen. Calling the logger here skipped both, which is why
            # embedding captures carried a null task and unresolvable
            # hashes while chat captures did not.
            self._log_capture(record)

            content = (
                json.dumps(sanitized_passthrough).encode()
                if sanitized_passthrough
                else b""
            )
            return Response(
                content=content,
                status_code=upstream_resp.status_code,
                media_type="application/json",
            )
        except Exception as e:
            elapsed_ms = (time.monotonic() - start_time) * 1000
            record.latency_ms = elapsed_ms
            record.upstream_status = 502
            record.error_body = str(e)
            record.stream_diagnostics = StreamDiagnostics(
                stream_exception=str(e),
            )
            self._log_capture(record)
            return JSONResponse(
                status_code=502,
                content={"error": "upstream unavailable"},
            )

    async def _handle_rerank_request(self, request: Request) -> Response:
        """Handle POST /v1/rerank — forward to upstream and log.

        Args:
            request: FastAPI Request with rerank body containing query, documents,
                     model, optional top_n.

        Returns:
            Response: JSON response from upstream (passthrough).
                - 200: Successful rerank response
                - 400: Stream=True (rerank does not stream)
                - 422: Invalid JSON or empty model/query/documents
                - 501: Ollama backend (not supported)
                - 502: Upstream unavailable

        Raises: None

        Invariants:
            - Before: Request body not yet parsed
            - After: CaptureRecord written with request_type="rerank"
              (early-validation 4xx returns exit before capture and are
              not logged)
            - stream=True always returns 400
            - ollama backend always returns 501
            - Upstream response body is passed through unchanged (no translation)
            - Alias rewrite-before-hash: public model replaced with upstream
              id after validation and before SHA-256 and proxy call.
            - Alias sanitize-before-log-and-serialize: upstream model
              remapped to public before capture and client.
            - A capture-disk OSError while logging the validation failure is
              suppressed — the 422 response is unaffected (diagnostic record
              lost, request served)
        """
        body_bytes = await request.body()
        try:
            body_dict: dict[str, Any] = json.loads(body_bytes) if body_bytes else {}
        except json.JSONDecodeError:
            return JSONResponse(status_code=422, content={"error": "invalid JSON"})

        try:
            request_data = RerankRequest(**body_dict)
        except Exception as e:
            with contextlib.suppress(OSError):
                self._capture_logger.log_error(
                    {
                        "turn_id": str(uuid.uuid4()),
                        "created_at": datetime.now(UTC).isoformat(),
                        "error_body": str(e),
                        "request_type": "rerank",
                        "error_category": "validation_error",
                    }
                )
            return JSONResponse(
                status_code=422, content={"error": "invalid rerank request"}
            )

        # Validate required fields are non-empty
        if not request_data.model.strip():
            return JSONResponse(
                status_code=422, content={"error": "model must not be empty"}
            )
        if not request_data.query.strip():
            return JSONResponse(
                status_code=422, content={"error": "query must not be empty"}
            )
        if not request_data.documents:
            return JSONResponse(
                status_code=422, content={"error": "documents list must not be empty"}
            )

        # Validate top_n >= 1 if provided
        if request_data.top_n is not None and request_data.top_n < 1:
            return JSONResponse(
                status_code=422, content={"error": "top_n must be >= 1"}
            )

        # Rerank does not support streaming
        if request_data.stream:
            return JSONResponse(
                status_code=400,
                content={"error": "streaming is not supported for rerank"},
            )

        # Ollama does not support rerank
        if self._backend == "ollama":
            return JSONResponse(
                status_code=501,
                content={"error": "rerank is not supported for ollama backend"},
            )

        headers = dict(request.headers)
        session_id = self._session_resolver.resolve(headers)
        turn_id = str(uuid.uuid4())
        capture_id = str(uuid.uuid4())
        created_at = datetime.now(UTC).isoformat()
        client_ip = request.client.host if request.client else "unknown"
        client_ua = headers.get("user-agent", "")

        runtime_snapshot = await asyncio.to_thread(
            RuntimeSnapshot.get,
            capture_base_path=self._capture_base_path,
            backend=self._backend,
        )

        forwarded_body, alias_original, alias_forwarded = rewrite_request_model(
            body_dict, self._aliases
        )
        request_sha256 = _compute_sha256(forwarded_body)

        record = CaptureRecord(
            session_id=session_id,
            user_id=self._session_resolver.resolve_user_id(headers),
            turn_id=turn_id,
            capture_id=capture_id,
            created_at=created_at,
            latency_ms=0.0,
            client_ip=client_ip,
            client_user_agent=client_ua,
            model=alias_original if alias_original is not None else request_data.model,
            request_model_original=alias_original,
            request_model_forwarded=alias_forwarded,
            run_id=runtime_snapshot.run_id,
            runtime_settings=runtime_snapshot.settings,
            request=forwarded_body,
            request_body_sha256=request_sha256,
            messages_sha256="",
            attempt_number=1,
            upstream_status=200,
            routing=None,
            request_type="rerank",
            task=self._resolve_task(
                alias_original if alias_original is not None else request_data.model
            ),
        )

        start_time = time.monotonic()
        try:
            upstream_resp = await self._upstream_client.proxy(
                method="POST",
                path="/v1/rerank",
                headers=headers,
                body=forwarded_body,
                stream=False,
            )

            elapsed_ms = (time.monotonic() - start_time) * 1000
            record.latency_ms = elapsed_ms
            sanitized_rerank = sanitize_response_model(
                upstream_resp.body, self._aliases
            )
            record.response = sanitized_rerank
            record.upstream_status = upstream_resp.status_code

            self._log_capture(record)

            content = json.dumps(sanitized_rerank).encode() if sanitized_rerank else b""
            return Response(
                content=content,
                status_code=upstream_resp.status_code,
                media_type="application/json",
            )
        except Exception as e:
            elapsed_ms = (time.monotonic() - start_time) * 1000
            record.latency_ms = elapsed_ms
            record.upstream_status = 502
            record.error_body = str(e)
            self._log_capture(record)
            return JSONResponse(
                status_code=502,
                content={"error": "upstream unavailable"},
            )

    async def _handle_classification_request(self, request: Request) -> Response:
        """Handle POST /v1/classification — forward to upstream and log.

        Args:
            request: FastAPI Request with a classification body containing
                model, inputs, and optional parameters.

        Returns:
            Response: JSON response from upstream (passthrough).
                - 200: Successful classification response
                - 422: Invalid JSON, failed schema validation, or empty
                  model/inputs
                - 4xx/5xx: Upstream status propagated verbatim with its body
                - 502: Upstream unavailable (connection failure)

        Raises: None

        Invariants:
            - Before: Request body not yet parsed
            - After: CaptureRecord written with request_type="classification"
              (early-validation 4xx returns exit before capture and are
              not logged)
            - Upstream response body is passed through unchanged; the
              declared ClassificationResponse contract is not enforced, so
              an upstream with a wider shape is still forwarded
            - No backend branch: every backend forwards to /v1/classification
              unchanged (classification does not stream, so there is no
              ollama translation and no ollama rejection)
            - Alias rewrite-before-hash: public model replaced with upstream
              id after validation and before SHA-256 and proxy call
            - Alias sanitize-before-log-and-serialize: upstream model
              remapped to public before capture and client
            - INV-4: request_body_sha256 hashes the forwarded body, and
              record.request mirrors that same body
            - task is resolved from the public model id and is None when
              CAPTURE_MODEL_TASKS is off or has no entry
            - A capture-disk OSError while logging the validation failure is
              suppressed — the 422 response is unaffected (diagnostic record
              lost, request served)
        """
        body_bytes = await request.body()
        try:
            body_dict: dict[str, Any] = json.loads(body_bytes) if body_bytes else {}
        except json.JSONDecodeError:
            return JSONResponse(status_code=422, content={"error": "invalid JSON"})

        try:
            request_data = ClassificationRequest(**body_dict)
        except Exception as e:
            with contextlib.suppress(OSError):
                self._capture_logger.log_error(
                    {
                        "turn_id": str(uuid.uuid4()),
                        "created_at": datetime.now(UTC).isoformat(),
                        "error_body": str(e),
                        "request_type": "classification",
                        "error_category": "validation_error",
                    }
                )
            return JSONResponse(
                status_code=422, content={"error": "invalid classification request"}
            )

        # Validate required fields are non-empty
        if not request_data.model.strip():
            return JSONResponse(
                status_code=422, content={"error": "model must not be empty"}
            )
        if isinstance(request_data.inputs, str) and not request_data.inputs.strip():
            return JSONResponse(
                status_code=422, content={"error": "inputs must not be empty"}
            )
        if isinstance(request_data.inputs, list) and not request_data.inputs:
            return JSONResponse(
                status_code=422, content={"error": "inputs list must not be empty"}
            )

        headers = dict(request.headers)
        session_id = self._session_resolver.resolve(headers)
        turn_id = str(uuid.uuid4())
        capture_id = str(uuid.uuid4())
        created_at = datetime.now(UTC).isoformat()
        client_ip = request.client.host if request.client else "unknown"
        client_ua = headers.get("user-agent", "")

        runtime_snapshot = await asyncio.to_thread(
            RuntimeSnapshot.get,
            capture_base_path=self._capture_base_path,
            backend=self._backend,
        )

        forwarded_body, alias_original, alias_forwarded = rewrite_request_model(
            body_dict, self._aliases
        )
        request_sha256 = _compute_sha256(forwarded_body)
        public_model = (
            alias_original if alias_original is not None else request_data.model
        )

        record = CaptureRecord(
            session_id=session_id,
            user_id=self._session_resolver.resolve_user_id(headers),
            turn_id=turn_id,
            capture_id=capture_id,
            created_at=created_at,
            latency_ms=0.0,
            client_ip=client_ip,
            client_user_agent=client_ua,
            model=public_model,
            request_model_original=alias_original,
            request_model_forwarded=alias_forwarded,
            run_id=runtime_snapshot.run_id,
            runtime_settings=runtime_snapshot.settings,
            request=forwarded_body,
            request_body_sha256=request_sha256,
            messages_sha256="",
            attempt_number=1,
            upstream_status=200,
            routing=None,
            request_type="classification",
            task=self._resolve_task(public_model),
        )

        start_time = time.monotonic()
        try:
            upstream_resp = await self._upstream_client.proxy(
                method="POST",
                path="/v1/classification",
                headers=headers,
                body=forwarded_body,
                stream=False,
            )

            elapsed_ms = (time.monotonic() - start_time) * 1000
            record.latency_ms = elapsed_ms
            sanitized_classification = sanitize_response_model(
                upstream_resp.body, self._aliases
            )
            record.response = sanitized_classification
            record.upstream_status = upstream_resp.status_code

            self._spawn_headroom(record)
            self._log_capture(record)

            content = (
                json.dumps(sanitized_classification).encode()
                if sanitized_classification
                else b""
            )
            return Response(
                content=content,
                status_code=upstream_resp.status_code,
                media_type="application/json",
            )
        except Exception as e:
            elapsed_ms = (time.monotonic() - start_time) * 1000
            record.latency_ms = elapsed_ms
            record.upstream_status = 502
            record.error_body = str(e)
            self._log_capture(record)
            return JSONResponse(
                status_code=502,
                content={"error": "upstream unavailable"},
            )

    async def _handle_image_request(self, request: Request) -> Response:
        """Handle POST /v1/images/generations — forward to upstream and log.

        Args:
            request: FastAPI Request with an image body containing model,
                prompt, and optional n/size/response_format.

        Returns:
            Response: JSON response from upstream (passthrough).
                - 200: Successful image generation response
                - 422: Invalid JSON, failed schema validation, empty
                  model/prompt, or n < 1
                - 4xx/5xx: Upstream status propagated verbatim with its body
                - 502: Upstream unavailable (connection failure)

        Raises: None

        Invariants:
            - Before: Request body not yet parsed
            - After: CaptureRecord written with request_type="image"
              (early-validation 4xx returns exit before capture and are
              not logged)
            - Upstream response body is passed through unchanged; the
              declared ImageResponse contract is not enforced, so an
              upstream with a wider shape is still forwarded
            - n/size/response_format are forwarded exactly as sent; the
              proxy never resizes or re-encodes a generated image
            - No backend branch: every backend forwards to
              /v1/images/generations unchanged (image generation does not
              stream, so there is no ollama translation and no rejection)
            - Alias rewrite-before-hash: public model replaced with upstream
              id after validation and before SHA-256 and proxy call
            - Alias sanitize-before-log-and-serialize: upstream model
              remapped to public before capture and client
            - INV-4: request_body_sha256 hashes the forwarded body, and
              record.request mirrors that same body
            - task is resolved from the public model id and is None when
              CAPTURE_MODEL_TASKS is off or has no entry
            - A capture-disk OSError while logging the validation failure is
              suppressed — the 422 response is unaffected (diagnostic record
              lost, request served)
        """
        body_bytes = await request.body()
        try:
            body_dict: dict[str, Any] = json.loads(body_bytes) if body_bytes else {}
        except json.JSONDecodeError:
            return JSONResponse(status_code=422, content={"error": "invalid JSON"})

        try:
            request_data = ImageRequest(**body_dict)
        except Exception as e:
            with contextlib.suppress(OSError):
                self._capture_logger.log_error(
                    {
                        "turn_id": str(uuid.uuid4()),
                        "created_at": datetime.now(UTC).isoformat(),
                        "error_body": str(e),
                        "request_type": "image",
                        "error_category": "validation_error",
                    }
                )
            return JSONResponse(
                status_code=422, content={"error": "invalid image request"}
            )

        # Validate required fields are non-empty
        if not request_data.model.strip():
            return JSONResponse(
                status_code=422, content={"error": "model must not be empty"}
            )
        if not request_data.prompt.strip():
            return JSONResponse(
                status_code=422, content={"error": "prompt must not be empty"}
            )

        # Validate n >= 1
        if request_data.n < 1:
            return JSONResponse(status_code=422, content={"error": "n must be >= 1"})

        headers = dict(request.headers)
        session_id = self._session_resolver.resolve(headers)
        turn_id = str(uuid.uuid4())
        capture_id = str(uuid.uuid4())
        created_at = datetime.now(UTC).isoformat()
        client_ip = request.client.host if request.client else "unknown"
        client_ua = headers.get("user-agent", "")

        runtime_snapshot = await asyncio.to_thread(
            RuntimeSnapshot.get,
            capture_base_path=self._capture_base_path,
            backend=self._backend,
        )

        forwarded_body, alias_original, alias_forwarded = rewrite_request_model(
            body_dict, self._aliases
        )
        request_sha256 = _compute_sha256(forwarded_body)
        public_model = (
            alias_original if alias_original is not None else request_data.model
        )

        record = CaptureRecord(
            session_id=session_id,
            user_id=self._session_resolver.resolve_user_id(headers),
            turn_id=turn_id,
            capture_id=capture_id,
            created_at=created_at,
            latency_ms=0.0,
            client_ip=client_ip,
            client_user_agent=client_ua,
            model=public_model,
            request_model_original=alias_original,
            request_model_forwarded=alias_forwarded,
            run_id=runtime_snapshot.run_id,
            runtime_settings=runtime_snapshot.settings,
            request=forwarded_body,
            request_body_sha256=request_sha256,
            messages_sha256="",
            attempt_number=1,
            upstream_status=200,
            routing=None,
            request_type="image",
            task=self._resolve_task(public_model),
        )

        start_time = time.monotonic()
        try:
            upstream_resp = await self._upstream_client.proxy(
                method="POST",
                path="/v1/images/generations",
                headers=headers,
                body=forwarded_body,
                stream=False,
            )

            elapsed_ms = (time.monotonic() - start_time) * 1000
            record.latency_ms = elapsed_ms
            sanitized_image = sanitize_response_model(upstream_resp.body, self._aliases)
            record.response = sanitized_image
            record.upstream_status = upstream_resp.status_code

            self._spawn_headroom(record)
            self._log_capture(record)

            content = json.dumps(sanitized_image).encode() if sanitized_image else b""
            return Response(
                content=content,
                status_code=upstream_resp.status_code,
                media_type="application/json",
            )
        except Exception as e:
            elapsed_ms = (time.monotonic() - start_time) * 1000
            record.latency_ms = elapsed_ms
            record.upstream_status = 502
            record.error_body = str(e)
            self._log_capture(record)
            return JSONResponse(
                status_code=502,
                content={"error": "upstream unavailable"},
            )

    async def _handle_native_passthrough(
        self, request: Request, upstream_path: str, request_type: str
    ) -> Response:
        """Forward a gpu-arm native JSON route verbatim and capture both sides.

        Shared passthrough for the seven native routes gpu-arm serves
        (/v1/systemone, /v1/classify, /v1/batch-classify, /v1/ner,
        /v1/batch-ner, /v1/timeseries, /v1/batch-timeseries). Unlike the
        OpenAI-shaped /v1/classification handler there is no declared request
        schema and no required-field validation: the body is forwarded as sent
        and the upstream response is returned unchanged, so the proxy never
        invents a shape the checkpoint never had.

        Args:
            request: FastAPI Request carrying a native JSON body (a model id
                plus route-specific fields such as text, texts, or state).
            upstream_path: Upstream route to forward to — the same path the
                client called.
            request_type: Endpoint-family label stored on the capture (e.g.
                "classify", "batch_classify"). A label for task inference and
                headroom eligibility, never a routing key.

        Returns:
            Response: The upstream JSON response passed through unchanged —
                status code, body, and content type.
                - 200: Successful native response
                - 422: Invalid JSON, or a JSON body that is not an object
                - 4xx/5xx: Upstream status propagated verbatim with its body
                - 502: Upstream unavailable (connection failure)

        Raises: None

        Invariants:
            - Before: Request body not yet parsed
            - After: CaptureRecord written with request_type=<request_type>;
              early-validation 4xx returns exit before capture and are not
              logged
            - Body forwarded verbatim: only the top-level model field is
              rewritten when an alias maps it, and request_body_sha256 hashes
              that forwarded body (INV-4); record.request mirrors it
            - Upstream response body passed through unchanged; no declared
              response contract is enforced, so a wider native shape forwards
            - Alias sanitize-before-log-and-serialize: the upstream model id
              is remapped to the public id before capture and client
            - No backend branch: every backend forwards to upstream_path
              unchanged (native routes do not stream)
            - task is resolved from the public model id and is None when
              CAPTURE_MODEL_TASKS is off or has no entry; headroom is a
              fire-and-forget no-op here because the native request types are
              outside the eligible (chat/completion/classification/image) set
            - A capture-disk OSError is suppressed — the client response is
              unaffected
        """
        body_bytes = await request.body()
        try:
            body_dict: dict[str, Any] = json.loads(body_bytes) if body_bytes else {}
        except json.JSONDecodeError:
            return JSONResponse(status_code=422, content={"error": "invalid JSON"})
        if not isinstance(body_dict, dict):
            return JSONResponse(
                status_code=422, content={"error": "body must be a JSON object"}
            )

        headers = dict(request.headers)
        session_id = self._session_resolver.resolve(headers)
        turn_id = str(uuid.uuid4())
        capture_id = str(uuid.uuid4())
        created_at = datetime.now(UTC).isoformat()
        client_ip = request.client.host if request.client else "unknown"
        client_ua = headers.get("user-agent", "")

        runtime_snapshot = await asyncio.to_thread(
            RuntimeSnapshot.get,
            capture_base_path=self._capture_base_path,
            backend=self._backend,
        )

        forwarded_body, alias_original, alias_forwarded = rewrite_request_model(
            body_dict, self._aliases
        )
        request_sha256 = _compute_sha256(forwarded_body)
        raw_model = body_dict.get("model")
        public_model = (
            alias_original
            if alias_original is not None
            else (raw_model if isinstance(raw_model, str) else "")
        )

        record = CaptureRecord(
            session_id=session_id,
            user_id=self._session_resolver.resolve_user_id(headers),
            turn_id=turn_id,
            capture_id=capture_id,
            created_at=created_at,
            latency_ms=0.0,
            client_ip=client_ip,
            client_user_agent=client_ua,
            model=public_model,
            request_model_original=alias_original,
            request_model_forwarded=alias_forwarded,
            run_id=runtime_snapshot.run_id,
            runtime_settings=runtime_snapshot.settings,
            request=forwarded_body,
            request_body_sha256=request_sha256,
            messages_sha256="",
            attempt_number=1,
            upstream_status=200,
            routing=None,
            request_type=request_type,
            task=self._resolve_task(public_model),
        )

        start_time = time.monotonic()
        try:
            upstream_resp = await self._upstream_client.proxy(
                method="POST",
                path=upstream_path,
                headers=headers,
                body=forwarded_body,
                stream=False,
            )

            elapsed_ms = (time.monotonic() - start_time) * 1000
            record.latency_ms = elapsed_ms
            sanitized_response = sanitize_response_model(
                upstream_resp.body, self._aliases
            )
            record.response = sanitized_response
            record.upstream_status = upstream_resp.status_code

            self._spawn_headroom(record)
            self._log_capture(record)

            content = (
                json.dumps(sanitized_response).encode() if sanitized_response else b""
            )
            return Response(
                content=content,
                status_code=upstream_resp.status_code,
                media_type="application/json",
            )
        except Exception as e:
            elapsed_ms = (time.monotonic() - start_time) * 1000
            record.latency_ms = elapsed_ms
            record.upstream_status = 502
            record.error_body = str(e)
            self._log_capture(record)
            return JSONResponse(
                status_code=502,
                content={"error": "upstream unavailable"},
            )

    async def _handle_streaming(
        self,
        headers: dict[str, str],
        body_dict: dict[str, Any],
        record: CaptureRecord,
        start_time: float,
        upstream_path: str,
        response_object: str,
    ) -> Response | StreamingResponse:
        """Handle streaming — lossless tee pattern with upstream status propagation.

        The upstream stream is opened before the StreamingResponse is
        constructed so the client receives the real upstream status: an
        upstream 4xx/5xx is returned as a JSONResponse with that status
        instead of an HTTP 200 SSE stream. Connect failures return 502. Client
        disconnect (GeneratorExit / CancelledError) is detected and recorded
        as client_disconnected. Disk errors during capture are suppressed to
        avoid masking upstream exceptions — a capture-tree fault disables
        stream logging for that request (named one-time warning) instead of
        failing it. Upstream failure is captured as
        upstream_status=502 in the record. Runtime snapshot is refreshed on
        upstream failure to capture restarted process metadata.

        Args:
            headers: Request headers forwarded to upstream.
            body_dict: Parsed request body; ``stream`` forced True upstream.
            record: CaptureRecord finalized in the generator ``finally`` block
                (or in the pre-flight error paths below).
            start_time: monotonic timestamp from request start.
            upstream_path: Upstream path to forward to.
            response_object: Object name for reconstructed responses
                ("chat.completion" or "text_completion").

        Returns:
            JSONResponse with the upstream status for 4xx/5xx and connect
                failures; StreamingResponse with the upstream status for
                successful SSE streams.

        Raises:
            The generator re-raises upstream and cancellation exceptions after
            finalizing the capture record; connect failures are converted to a
            502 JSONResponse in the pre-flight path.

        Invariants:
            - SSE parse-copy rewrite — when aliases are on, each parsed
              event is sanitized post-parse and reframed pre-forward;
              the reconstructed response is sanitized before capture.
        """
        upstream_body = dict(body_dict)
        upstream_body["stream"] = True

        # Label before the first byte: the stream file's request_start
        # carries the task, and _log_capture would not run until the
        # response finished. Re-applied there idempotently.
        self._apply_task(record, record.request_type)

        stream_logger: StreamLogger | None = None
        try:
            stream_logger = StreamLogger(
                base_path=self._capture_base_path,
                turn_id=record.turn_id,
                created_at=record.created_at,
            )
            record.stream_file = str(stream_logger.filepath)

            # Write request_start immediately — survives upstream crash
            # Hashes computed from forwarded body (upstream model name if aliasing);
            # bytes recomputed here from the same body for request_bytes field.
            request_body_bytes = json.dumps(body_dict, ensure_ascii=False).encode()
            stream_logger.record_request_start(
                capture_id=record.capture_id,
                request_body_sha256=record.request_body_sha256,
                messages_sha256=record.messages_sha256,
                model=record.model,
                messages_count=len(upstream_body.get("messages", [])),
                stream=True,
                max_completion_tokens=upstream_body.get("max_completion_tokens"),
                request_bytes=len(request_body_bytes),
                request_body=body_dict,
                task=record.task,
            )
        except OSError as e:
            # Capture-tree fault before upstream — degrade diagnostics for
            # this request instead of misattributing it as an upstream error
            stream_logger = None
            _capture_write_failed("stream setup", e)

        sse_parser = SseParser()
        collected_chunks: list[dict[str, Any]] = []
        diag = StreamDiagnostics()
        first_byte_recorded = False
        client_disconnected = False
        upstream_exception: str | None = None
        downstream_exception: str | None = None

        # Pre-flight: open the upstream stream so the real status is known
        # before the StreamingResponse is constructed.
        try:
            upstream_resp = await self._upstream_client.proxy(
                method="POST",
                path=upstream_path,
                headers=headers,
                body=upstream_body,
                stream=True,
            )
        except Exception as e:
            upstream_exception = str(e)
            diag.stream_exception = str(e)
            final_elapsed = (time.monotonic() - start_time) * 1000
            record.latency_ms = final_elapsed
            record.upstream_status = 502
            record.error_body = str(e)
            record.stream_diagnostics = diag
            await self._refresh_runtime_on_upstream_failure(record)
            if stream_logger is not None:
                with contextlib.suppress(OSError):
                    stream_logger.write_terminal(
                        headers_status=None,
                        upstream_eof=False,
                        client_disconnected=False,
                        upstream_exception=upstream_exception,
                        downstream_exception=downstream_exception,
                        elapsed_ms=final_elapsed,
                    )
            self._log_capture(record)
            return JSONResponse(
                status_code=502,
                content={"error": "upstream unavailable"},
            )

        diag.upstream_headers_received = True
        diag.upstream_status = upstream_resp.status_code
        diag.upstream_content_type = upstream_resp.content_type

        headers_elapsed = (time.monotonic() - start_time) * 1000
        if stream_logger is not None:
            try:
                stream_logger.record_upstream_headers(
                    elapsed_ms=headers_elapsed,
                    status=upstream_resp.status_code,
                    content_type=upstream_resp.content_type,
                    headers=dict(upstream_resp.headers),
                )
            except OSError as e:
                _capture_write_failed("record_upstream_headers", e)

        if upstream_resp.status_code >= 400:
            # Upstream rejected before streaming — drain the error body and
            # return the real status instead of an HTTP 200 SSE stream.
            error_bytes = b""
            async for error_chunk in upstream_resp.stream:
                error_bytes += error_chunk
            error_payload: Any = None
            if error_bytes:
                try:
                    error_payload = json.loads(error_bytes.decode("utf-8"))
                except (json.JSONDecodeError, TypeError):
                    error_payload = {
                        "error": error_bytes.decode("utf-8", errors="replace")
                    }
            final_elapsed = (time.monotonic() - start_time) * 1000
            record.latency_ms = final_elapsed
            record.response = error_payload if isinstance(error_payload, dict) else None
            record.error_body = (
                json.dumps(error_payload) if error_payload is not None else None
            )
            record.upstream_status = upstream_resp.status_code
            record.stream_diagnostics = diag
            if stream_logger is not None:
                with contextlib.suppress(OSError):
                    stream_logger.write_terminal(
                        headers_status=upstream_resp.status_code,
                        upstream_eof=False,
                        client_disconnected=False,
                        upstream_exception=None,
                        downstream_exception=downstream_exception,
                        elapsed_ms=final_elapsed,
                    )
            self._log_capture(record)
            return JSONResponse(
                status_code=upstream_resp.status_code,
                content=(
                    error_payload
                    if error_payload is not None
                    else {"error": "upstream error"}
                ),
            )

        async def stream_generator() -> AsyncIterator[bytes]:
            nonlocal first_byte_recorded, client_disconnected
            nonlocal upstream_exception, downstream_exception

            alias_on = self._aliases is not None

            def _note_event(event: dict[str, Any], parse_elapsed: float) -> None:
                if stream_logger is not None:
                    try:
                        stream_logger.record_sse_event(event, parse_elapsed)
                    except OSError as e:
                        _capture_write_failed("record_sse_event", e)
                if event.get("done"):
                    diag.done_seen = True
                if event.get("finish_reasons"):
                    diag.finish_reason_seen = True
                if event.get("has_usage"):
                    diag.usage_seen = True

            try:
                async for raw_chunk in upstream_resp.stream:
                    chunk_start = time.monotonic()
                    elapsed_ms = (chunk_start - start_time) * 1000

                    if not first_byte_recorded:
                        diag.first_byte_ms = elapsed_ms
                        first_byte_recorded = True

                    diag.raw_upstream_bytes += len(raw_chunk)
                    if not alias_on:
                        b64_chunk = base64.b64encode(raw_chunk).decode("ascii")
                        if stream_logger is not None:
                            try:
                                stream_logger.record_upstream_chunk(
                                    b64_chunk, elapsed_ms
                                )
                            except OSError as e:
                                _capture_write_failed("record_upstream_chunk", e)

                        # Forward raw bytes to client IMMEDIATELY
                        yield raw_chunk
                        diag.forwarded_downstream_bytes += len(raw_chunk)
                        if stream_logger is not None:
                            try:
                                stream_logger.record_downstream_chunk(
                                    b64_chunk, elapsed_ms
                                )
                            except OSError as e:
                                _capture_write_failed("record_downstream_chunk", e)

                        # Parse a copy for diagnostics (does not block forwarding)
                        events = sse_parser.feed(raw_chunk)
                        for event in events:
                            parse_elapsed = (time.monotonic() - start_time) * 1000
                            _note_event(event, parse_elapsed)
                            if event.get("json_valid") and not event.get("done"):
                                with contextlib.suppress(
                                    json.JSONDecodeError, TypeError
                                ):
                                    parsed = json.loads(event["raw_data"])
                                    collected_chunks.append(parsed)
                        continue

                    b64_up = base64.b64encode(raw_chunk).decode("ascii")
                    if stream_logger is not None:
                        try:
                            stream_logger.record_upstream_chunk(b64_up, elapsed_ms)
                        except OSError as e:
                            _capture_write_failed("record_upstream_chunk", e)

                    # Parse first, then forward the sanitized reframe.
                    events = sse_parser.feed(raw_chunk)
                    for event in events:
                        parse_elapsed = (time.monotonic() - start_time) * 1000
                        _note_event(event, parse_elapsed)
                        sanitized_data = sanitize_sse_event(
                            str(event.get("raw_data", "")), self._aliases
                        )
                        if event.get("json_valid") and not event.get("done"):
                            with contextlib.suppress(json.JSONDecodeError, TypeError):
                                collected_chunks.append(json.loads(sanitized_data))
                        frame = f"data: {sanitized_data}\n\n".encode()
                        yield frame
                        diag.forwarded_downstream_bytes += len(frame)
                        if stream_logger is not None:
                            try:
                                stream_logger.record_downstream_chunk(
                                    base64.b64encode(frame).decode("ascii"),
                                    elapsed_ms,
                                )
                            except OSError as e:
                                _capture_write_failed("record_downstream_chunk", e)

                # Upstream stream ended — flush parser buffer
                remaining = sse_parser.flush()
                for event in remaining:
                    parse_elapsed = (time.monotonic() - start_time) * 1000
                    _note_event(event, parse_elapsed)
                    if alias_on:
                        sanitized_data = sanitize_sse_event(
                            str(event.get("raw_data", "")), self._aliases
                        )
                        if event.get("json_valid") and not event.get("done"):
                            with contextlib.suppress(json.JSONDecodeError, TypeError):
                                collected_chunks.append(json.loads(sanitized_data))
                            frame = f"data: {sanitized_data}\n\n".encode()
                            yield frame
                            diag.forwarded_downstream_bytes += len(frame)
                            if stream_logger is not None:
                                try:
                                    stream_logger.record_downstream_chunk(
                                        base64.b64encode(frame).decode("ascii"),
                                        parse_elapsed,
                                    )
                                except OSError as e:
                                    _capture_write_failed("record_downstream_chunk", e)
                    elif event.get("json_valid") and not event.get("done"):
                        with contextlib.suppress(json.JSONDecodeError, TypeError):
                            parsed = json.loads(event["raw_data"])
                            collected_chunks.append(parsed)

                diag.upstream_eof = True
                eof_elapsed = (time.monotonic() - start_time) * 1000
                if stream_logger is not None:
                    try:
                        stream_logger.record_upstream_eof(eof_elapsed)
                    except OSError as e:
                        _capture_write_failed("record_upstream_eof", e)

            except (GeneratorExit, asyncio.CancelledError):
                client_disconnected = True
                diag.client_disconnected = True
                raise
            except Exception as e:
                upstream_exception = str(e)
                diag.stream_exception = str(e)
                raise
            finally:
                final_elapsed = (time.monotonic() - start_time) * 1000
                record.latency_ms = final_elapsed
                record.upstream_status = (
                    502
                    if upstream_exception
                    else (diag.upstream_status if diag.upstream_status else 200)
                )
                record.stream_diagnostics = diag

                if collected_chunks:
                    reconstructed = _reconstruct_response(
                        collected_chunks, response_object
                    )
                    sanitized_final = sanitize_response_model(
                        reconstructed, self._aliases
                    )
                    if sanitized_final is None:
                        sanitized_final = reconstructed
                    record.response = sanitized_final
                    choices = sanitized_final.get("choices", [])
                    if choices:
                        choice = choices[0]
                        record.tool_calls = choice.get("message", {}).get("tool_calls")
                        record.finish_reason = choice.get("finish_reason")
                    usage = reconstructed.get("usage")
                    if usage:
                        record.usage_raw = usage
                        record.usage_normalized = _normalize_usage(usage)

                if upstream_exception:
                    await self._refresh_runtime_on_upstream_failure(record)

                if stream_logger is not None:
                    with contextlib.suppress(OSError):
                        stream_logger.write_terminal(
                            headers_status=diag.upstream_status,
                            upstream_eof=diag.upstream_eof,
                            client_disconnected=client_disconnected,
                            upstream_exception=upstream_exception,
                            downstream_exception=downstream_exception,
                            elapsed_ms=final_elapsed,
                        )

                self._spawn_headroom(record)
                self._spawn_routing_capture(record)
                self._log_capture(record)

        return StreamingResponse(
            stream_generator(),
            status_code=upstream_resp.status_code,
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    def _spawn_headroom(self, record: CaptureRecord) -> None:
        """Schedule a headroom sidecar write for an eligible capture.

        Sets record.headroom to the precomputed HeadroomRef and spawns the
        writer as a fire-and-forget asyncio task (never awaited here, so the
        response path is not blocked by tensor I/O). Call before
        _log_capture so the JSONL line carries the ref.

        Eligibility: headroom enabled (writer built), request_type in
        (chat, completion, classification, image), valid-UUID capture_id,
        upstream success (status < 400 and no error_body),
        runtime_settings.model_path pointing at an existing directory,
        and the interval samples this capture. A ``.gguf`` model_path is refused
        before the directory gate with a one-time warning naming the path; the
        sidecar reads safetensors checkpoints only and emits nothing for GGUF.

        Args:
            record: Finalized CaptureRecord (status/error fields complete).

        Returns: None

        Raises: None

        Invariants:
          - record.headroom is set only when the task was spawned
          - Ref paths are precomputed; a missing sidecar file means the
            background write did not complete (sidecar is authoritative)
          - No tensor work or torch/safetensors imports happen here
          - Classifier and image requests are eligible alongside chat
            and completion
        """
        writer = self._headroom_writer
        if writer is None:
            return
        if record.request_type not in ("chat", "completion", "classification", "image"):
            return
        if not record.capture_id:
            return
        if record.upstream_status >= 400 or record.error_body is not None:
            return
        if not writer.should_sample():
            return
        runtime = record.runtime_settings or {}
        model_path_raw = (
            runtime.get("model_path") if isinstance(runtime, dict) else None
        )
        if not isinstance(model_path_raw, str) or not model_path_raw:
            return
        model_path = Path(model_path_raw)
        if model_path_raw.lower().endswith(".gguf"):
            # GGUF is a non-goal for the sidecar (safetensors-only by contract).
            # Refuse loudly and deduplicated: one warning per distinct raw path,
            # then return with record.headroom still None — no silent inertness.
            if model_path_raw not in _gguf_warned_paths:
                _gguf_warned_paths.add(model_path_raw)
                _logger.warning(
                    "Headroom sidecar refused GGUF model path %r: the sidecar reads "
                    "safetensors checkpoints only, so no sidecar will be emitted "
                    "for this GGUF model.",
                    model_path_raw,
                )
            return
        if not model_path.is_dir():
            return
        try:
            uuid.UUID(record.capture_id)
        except (ValueError, AttributeError, TypeError):
            return
        _, weight_path, activation_path, expert_weights_path = writer.planned_artifacts(
            record.capture_id, record.created_at
        )
        planned = {
            "effective_rank": weight_path,
            "load_balance": activation_path,
            "specialization": expert_weights_path,
        }
        signals = {key: planned.get(key) for key in SIGNAL_KEYS}
        record.headroom = HeadroomRef(
            capture_id=record.capture_id,
            parent_raw=self._capture_logger.plan_path(record.model_dump(), "raw"),
            model=record.model,
            model_path=model_path_raw,
            task=record.task,
            signals=signals,
        )
        task = asyncio.create_task(
            writer.write_async(
                capture_id=record.capture_id,
                parent_raw=record.headroom.parent_raw,
                model=record.model,
                model_path=model_path,
                created_at=record.created_at,
                task=record.task,
            )
        )
        writer.keep(task)

    def _query_index(
        self,
        kind_dir: str,
        kind_label: Literal["session", "user"],
        identifier: str,
    ) -> CaptureIndexResponse:
        """Serve captures for one session/user id from its index file.

        Args:
            kind_dir: Index directory name — "sessions" or "users".
            kind_label: Response kind label — "session" or "user".
            identifier: Exact session_id or user_id to look up.

        Returns:
            CaptureIndexResponse: Chronological index entries; count 0 with
                an empty list when the id is unknown (never a 404/500).

        Raises: None

        Invariants:
          - Reads ONLY {base_path}/index/{kind_dir}/{id}.jsonl — never
            scans raw/ or errors/ JSONL
          - Unparsable index lines are skipped, not surfaced as errors
        """
        captures: list[IndexEntry] = []
        for entry in self._capture_logger.read_index(kind_dir, identifier):
            try:
                captures.append(IndexEntry(**entry))
            except ValidationError:
                continue
        return CaptureIndexResponse(
            kind=kind_label,
            id=identifier,
            count=len(captures),
            captures=captures,
        )

    def _list_ids(
        self, kind_dir: str, kind_label: Literal["session", "user"], limit: int
    ) -> IdListResponse:
        """List known session/user ids from index directory metadata.

        Args:
            kind_dir: Index directory name — "sessions" or "users".
            kind_label: Response kind label — "session" or "user".
            limit: Requested max entries; clamped to 0..500.

        Returns:
            IdListResponse: Most recently active ids first.

        Raises: None
        """
        clamped = max(0, min(limit, 500))
        summaries = [
            IdSummary(**item)
            for item in self._capture_logger.list_index_ids(kind_dir, clamped)
        ]
        return IdListResponse(
            kind=kind_label,
            count=len(summaries),
            ids=summaries,
        )

    def _log_capture(
        self, record: CaptureRecord, extra: dict[str, Any] | None = None
    ) -> Path | None:
        """Persist a capture record, routing failures to errors/.

        Central routing policy: upstream_status >= 400 or error_body set →
        errors/, else raw/. ``extra`` is merged into the written record dict.
        OSError during write is suppressed — capture failure must not break
        the client response.

        Task labelling and blob storage happen HERE rather than in each
        handler's record construction. Every capture — chat, completion,
        embedding, rerank, classification, image, plus every validation-
        error, connection-error and streaming-failure path — funnels
        through this one method, so labelling at the choke point is
        structurally impossible to miss a site, whereas six scattered
        constructor kwargs are six chances to forget one.

        Args:
            record: CaptureRecord to persist.
            extra: Optional extra fields merged into the JSONL line.

        Returns:
            Path | None: The JSONL file path written (raw/ or errors/), or
                None when the write failed (OSError suppressed).

        Raises: None (OSError suppressed)
        """
        self._apply_task(record, record.request_type)
        self._store_request_blobs(record)
        log_record = record.model_dump()
        if extra:
            log_record.update(extra)
        try:
            if record.upstream_status >= 400 or record.error_body is not None:
                return self._capture_logger.log_error(log_record)
            return self._capture_logger.log(log_record)
        except OSError:
            return None

    async def _refresh_runtime_on_upstream_failure(self, record: CaptureRecord) -> None:
        """Rebuild the runtime snapshot after an upstream failure.

        Rebuilds with ``force=True`` to capture restarted llama-server process
        metadata; ``run_id`` may change if the serving configuration changed.
        The rebuild runs off the event loop (blocking advisory subprocesses).

        Args:
            record: CaptureRecord whose run_id/runtime_settings are updated in place.

        Returns: None

        Raises: None
        """
        refresh_snapshot = await asyncio.to_thread(
            RuntimeSnapshot.get,
            capture_base_path=self._capture_base_path,
            backend=self._backend,
            force=True,
        )
        record.run_id = refresh_snapshot.run_id
        record.runtime_settings = refresh_snapshot.settings


def _reconstruct_response(
    chunks: list[dict[str, Any]], response_object: str
) -> dict[str, Any]:
    """Merge SSE delta chunks into a complete response.

    Accumulates tool_calls by index, content, reasoning_content,
    and legacy function_call. Handles multiple choices by choice.index.
    """
    if not chunks:
        return {}

    first = chunks[0]
    result: dict[str, Any] = {
        "id": first.get("id", ""),
        "object": response_object,
        "created": first.get("created", 0),
        "model": first.get("model", ""),
        "choices": [],
        "usage": None,
    }

    # Per-choice accumulators
    choices_map: dict[int, dict[str, Any]] = {}
    tool_calls_map: dict[int, dict[int, dict[str, Any]]] = {}

    for chunk in chunks:
        for choice in chunk.get("choices", []):
            idx = choice.get("index", 0)

            if idx not in choices_map:
                choices_map[idx] = {
                    "index": idx,
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "reasoning_content": "",
                    },
                    "finish_reason": None,
                }
                tool_calls_map[idx] = {}

            delta = choice.get("delta", {})
            msg = choices_map[idx]["message"]

            # Content accumulation
            if delta.get("content"):
                msg["content"] += delta["content"]

            # Reasoning content accumulation
            if delta.get("reasoning_content"):
                msg["reasoning_content"] += delta["reasoning_content"]

            # Legacy text accumulation (for /v1/completions)
            if choice.get("text"):
                msg["content"] += choice["text"]

            # Role
            if "role" in delta:
                msg["role"] = delta["role"]

            # Finish reason
            if choice.get("finish_reason"):
                choices_map[idx]["finish_reason"] = choice["finish_reason"]

            # Tool calls accumulation by index
            if delta.get("tool_calls"):
                for tc in delta["tool_calls"]:
                    tc_index = tc.get("index", 0)
                    if tc_index not in tool_calls_map[idx]:
                        tool_calls_map[idx][tc_index] = {
                            "id": "",
                            "type": "function",
                            "function": {"name": "", "arguments": ""},
                        }
                    existing = tool_calls_map[idx][tc_index]
                    if tc.get("id"):
                        existing["id"] = tc["id"]
                    if tc.get("type"):
                        existing["type"] = tc["type"]
                    func = tc.get("function", {})
                    if func.get("name"):
                        existing["function"]["name"] += func["name"]
                    if func.get("arguments"):
                        existing["function"]["arguments"] += func["arguments"]

            # Legacy function_call accumulation
            if delta.get("function_call"):
                fc = delta["function_call"]
                if "function_call" not in msg:
                    msg["function_call"] = {"name": "", "arguments": ""}
                if fc.get("name"):
                    msg["function_call"]["name"] += fc["name"]
                if fc.get("arguments"):
                    msg["function_call"]["arguments"] += fc["arguments"]

    # Assemble choices
    for idx in sorted(choices_map.keys()):
        choice_data = choices_map[idx]
        msg = choice_data["message"]

        # Attach accumulated tool calls
        if tool_calls_map.get(idx):
            tc_list = [
                tool_calls_map[idx][k] for k in sorted(tool_calls_map[idx].keys())
            ]
            msg["tool_calls"] = tc_list

        # Clean empty reasoning_content
        if not msg.get("reasoning_content"):
            msg.pop("reasoning_content", None)

        # Clean empty function_call
        if msg.get("function_call") and not msg["function_call"].get("name"):
            msg.pop("function_call", None)

        result["choices"].append(choice_data)

    # Usage from last chunk that has it
    for chunk in reversed(chunks):
        if chunk.get("usage"):
            result["usage"] = chunk["usage"]
            break

    return result


def _normalize_usage(usage: dict[str, Any]) -> UsageNormalized:
    """Create diagnostic interpretation of token usage.

    Never substitutes into the live response — purely observational.
    """
    # vLLM sends explicit nulls (counters and "prompt_tokens_details") — the
    # key-exists-default pattern returns None there, so use truthy fallbacks.
    prompt_total = usage.get("prompt_tokens") or 0
    details = usage.get("prompt_tokens_details") or {}
    prompt_cached = details.get("cached_tokens") or 0
    prompt_uncached = max(0, prompt_total - prompt_cached)
    completion = usage.get("completion_tokens") or 0
    return UsageNormalized(
        prompt_total=prompt_total,
        prompt_cached=prompt_cached,
        prompt_uncached=prompt_uncached,
        completion=completion,
        context_total=prompt_total + completion,
    )


def create_app() -> FastAPI:
    """Factory function to create the FastAPI application."""
    proxy = CaptureProxyApp()
    return proxy.app
