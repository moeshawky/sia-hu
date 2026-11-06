"""
Data Models

Owns: Pydantic models for capture proxy request/response schema
Depends on: pydantic>=2.14, pathlib (std), uuid (std)
Provides: UsageNormalized, StreamDiagnostics, HeadroomRef, CapturedHeadroom, CaptureRecord, ChatCompletionRequest, EmbeddingRequest, EmbeddingResponse, RerankRequest, ClassificationRequest, ClassificationResponse, ImageRequest, ImageResponse, HealthResponse

Invariants:
  - CaptureRecord includes request/response plus runtime provenance
  - routing field is always None (placeholder, never populated)
  - usage_raw is never substituted into the live response
  - StreamDiagnostics tracks every observable state of the streaming pipeline
  - CaptureRecord.schema_version 6 is additive over 5: v5 lines without the
    user_id key still validate (field defaults to None); every NEW record
    written by the proxy carries a non-None user_id resolved from request
    headers (x-user-id > Authorization hash > "anonymous")
  - CaptureRecord.task is additive within schema_version 6 (same precedent as
    request_type/embedding_count): a v6 line written before the task field
    existed still validates (field defaults to None)
  - HeadroomRef.schema_version 6 is additive over 5: v5 lines without the
    task key still validate (field defaults to None); every new sidecar
    record written by the proxy carries the task label when available
  - CapturedHeadroom.schema_version 7 is additive over 6: it is the version
    of the headroom artifact family this sidecar indexes, and at 7 the
    *weight_pairs.json it points at carries parent_shape, truncated, and
    sampling_mode on every pair; a v6 sidecar line still validates
    (schema_version is read from the file, its default is only for new writes)
  - task is None when CAPTURE_MODEL_TASKS is off or has no entry for the
    public model id
"""

from __future__ import annotations

import uuid
from pathlib import Path  # noqa: TC003  # pydantic resolves annotations at runtime
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


class UsageNormalized(BaseModel):
    """Diagnostic interpretation of token usage.

    Never substituted into the live response. Purely observational.
    """

    prompt_total: int = 0
    prompt_cached: int = 0
    prompt_uncached: int = 0
    completion: int = 0
    context_total: int = 0


class StreamDiagnostics(BaseModel):
    """Observable state of the streaming pipeline for a single request.

    Tracks every decision point from upstream connection to client delivery.
    """

    upstream_headers_received: bool = False
    upstream_status: int | None = None
    upstream_content_type: str | None = None
    first_byte_ms: float | None = None
    upstream_eof: bool = False
    client_disconnected: bool = False
    done_seen: bool = False
    finish_reason_seen: bool = False
    usage_seen: bool = False
    stream_exception: str | None = None
    raw_upstream_bytes: int = 0
    forwarded_downstream_bytes: int = 0


def _check_capture_uuid(value: str) -> str:
    """Validate that a capture identifier parses as a UUID.

    Args:
        value: Candidate capture_id string.

    Returns:
        The input string unchanged when it parses as a UUID.

    Raises:
        ValueError: When the string is not a valid UUID (any version).
    """
    try:
        uuid.UUID(value)
    except (ValueError, AttributeError, TypeError) as exc:
        raise ValueError("capture_id must be a valid UUID string") from exc
    return value


class HeadroomRef(BaseModel):
    """Reference to a headroom capture sidecar, written into a CaptureRecord.

    Paths are precomputed from capture_id before the JSONL line is written,
    so the append-only raw line does not need rewriting when the background
    write completes. The sidecar file is the authoritative artifact; this
    reference only carries its location and provenance.

    Attributes:
        schema_version: Schema version (currently 6).
        capture_id: Capture identifier; must parse as a UUID.
        parent_raw: Planned path of the raw JSONL line this sidecar pairs with.
        model: Served model name from the request.
        model_path: Model directory read for weight tensors (runtime_settings).
        architecture: Detected model class ("dense", "moe", or "unknown").
        task: Task label for the served model, resolved by the handler from
            CAPTURE_MODEL_TASKS (model id to task type). None when the map is
            off, the id is absent, or the configured value is not a string.
        signals: Signal name to artifact path (None when not captured).

    Invariants:
        - capture_id is a valid UUID (field validator; ValidationError otherwise)
        - parent_raw points at the raw/ JSONL line, computed by
          CaptureLogger.plan_path from the same record
        - a file listed in signals or at parent_raw may be absent when the
          background write did not complete
    """

    schema_version: int = 6
    capture_id: str
    parent_raw: Path
    model: str
    model_path: str
    architecture: Literal["dense", "moe", "unknown"] = "unknown"
    task: str | None = None
    signals: dict[str, str | None] = Field(default_factory=dict)

    @field_validator("capture_id")
    @classmethod
    def _validate_capture_id(cls, value: str) -> str:
        return _check_capture_uuid(value)


class CapturedHeadroom(BaseModel):
    """Schema of the headroom sidecar file (*.headroom.json).

    Validated before the file is written by HeadroomWriter. Extends the
    HeadroomRef fields with write diagnostics.

    Attributes:
        schema_version: Schema version (currently 7). Also the version of the
            headroom artifact family this sidecar indexes — at 7 the
            *weight_pairs.json named in signals carries per-pair
            parent_shape/truncated/sampling_mode provenance.
        capture_id: Capture identifier; must parse as a UUID.
        parent_raw: Path of the raw JSONL line this sidecar pairs with.
        model: Served model name from the request.
        model_path: Model directory read for weight tensors.
        architecture: Detected model class ("dense", "moe", or "unknown").
        task: Task label for the served model, resolved by the handler from
            CAPTURE_MODEL_TASKS (model id to task type). None when the map is
            off, the id is absent, or the configured value is not a string.
        collected_at: ISO 8601 timestamp of the sidecar write.
        signals: Signal name to artifact path (None when not captured).
        bytes_written: Float32 weight payload bytes actually written.
        bytes_cap: Float32 byte cap applied (0 = unlimited).
        truncated: True when the byte cap reduced the payload.
        error_body: Why a capture step was skipped or truncated; None when clean.

    Invariants:
        - capture_id is a valid UUID (field validator; ValidationError otherwise)
        - bytes_written <= bytes_cap when bytes_cap > 0
        - error_body is None on a clean, untruncated write
    """

    schema_version: int = 7
    capture_id: str
    parent_raw: Path
    model: str
    model_path: str
    architecture: Literal["dense", "moe", "unknown"] = "unknown"
    task: str | None = None
    collected_at: str
    signals: dict[str, str | None] = Field(default_factory=dict)
    bytes_written: int = 0
    bytes_cap: int = 0
    truncated: bool = False
    error_body: str | None = None

    @field_validator("capture_id")
    @classmethod
    def _validate_capture_id(cls, value: str) -> str:
        return _check_capture_uuid(value)


class CaptureRecord(BaseModel):
    """Schema for raw request/response capture.

    Version 6 — adds user identity for per-user retrieval (additive over
    version 5; v5 lines without ``user_id`` still validate). The ``task``
    field is additive within the same version, following the precedent of
    ``request_type``/``embedding_count``.

    Tri-state field classes (declared in ``schema.py`` and written to
    ``{base_path}/schema.json``): a field listed in ``REQUIRED_FIELDS`` is
    always present; a field listed in ``NULLABLE_NOT_APPLICABLE`` is
    present and may be null, where null asserts *not applicable to this
    request type*; a field listed in ``OPTIONAL_UNRECORDED`` may be absent
    entirely, where absence asserts *not recorded*. The distinction is
    load-bearing: "this chat turn has no embedding count" and "the proxy
    did not record an embedding count" are different facts about a
    capture, and collapsing them into one loses provenance.

    Attributes:
        schema_version: Schema version (currently 6).
        session_id: Stable session identifier (UUID v4 or client-supplied
            x-session-id value) — WHICH conversation.
        user_id: Stable user identifier (x-user-id value, Authorization-hash
            prefix, or "anonymous") — WHO sent the request. None only for
            records written before schema v6.
        turn_id: Unique turn identifier (UUID v4).
        capture_id: Unique capture identifier (UUID v4).
        created_at: ISO 8601 timestamp of request creation.
        latency_ms: Round-trip latency in milliseconds.
        client_ip: Client IP address.
        client_user_agent: Client User-Agent header.
        model: Model identifier from request.
        request_model_original: Public model id as sent by the client
            (alias mode only; None when aliases off or id unknown).
        request_model_forwarded: Upstream model id actually forwarded
            (alias mode only; None when aliases off or id unknown).
        run_id: Runtime run identifier.
        runtime_settings: Runtime configuration snapshot.
        request: Request body actually forwarded upstream (original OpenAI
            format for chat/completion/rerank; ollama /api/embed format for
            ollama embeddings).
        request_body_sha256: SHA-256 of serialized request body.
        messages_sha256: SHA-256 of messages array (chat) or input (embedding).
        attempt_number: Request attempt number (1-based).
        response: Upstream response body as dict.
        tool_calls: Tool calls accumulated from response.
        finish_reason: Finish reason from response choices.
        usage_raw: Raw usage dict from upstream.
        usage_normalized: Diagnostic usage interpretation.
        upstream_status: HTTP status code from upstream.
        error_body: Error body if upstream failed.
        routing: Routing info (always None — placeholder).
        stream_diagnostics: Streaming pipeline diagnostics.
        stream_file: Path to companion stream.jsonl file.
        request_type: Request type — "chat", "completion", "embedding",
            "rerank", "classification", or "image"; None for legacy lines
            written before request_type existed.
        task: Task label for the served model. Resolved by the handler from
            CAPTURE_MODEL_TASKS (model id to task type) when the operator
            configured one; otherwise derived structurally from request
            shape by ``taskclass.classify_request``. None only when both
            paths decline.
        task_rule: Which classification rule produced ``task``
            ("toolbelt_present", "payload_keys.query+documents", ...).
            Makes every label auditable without re-running the classifier.
        task_evidence: The structural measurements the rule read (message
            count, body bytes, key presence, code-line ratio). Empty when
            the label came from the operator's model-task map.
        embedding_count: Number of embeddings in request input.
        embedding_dimensions: Dimension of each embedding vector (0 if unknown).
        input_sha256: SHA-256 of the raw input text(s) for embeddings.
        headroom: Reference to the headroom sidecar (None when headroom
            capture is off, unsampled, or not eligible for this record).

    Invariants:
        - routing field is always None (placeholder, never populated)
        - task is None only when the operator map is off AND structural
          classification declined; it is a label, never a routing key, and
          is derived from request SHAPE rather than prompt vocabulary
        - user_id is non-None for every record written by schema v6 handlers
          and is the join key for {base_path}/index/users/ lookups
        - usage_raw is never substituted into the live response
        - StreamDiagnostics tracks every observable state of the streaming pipeline
        - request_type is None for legacy requests, populated for new request types
        - task is None whenever CAPTURE_MODEL_TASKS is off or has no exact
          entry for the public model id; it is a label, never a routing key
        - request always reflects the body actually sent upstream, and
          request_body_sha256 hashes that same forwarded body
        - embedding_count is None unless request_type == "embedding"
        - embedding_dimensions is None unless response is available
        - INV-4 alias audit: model is the public id; request[model] is the
          forwarded upstream id; request_model_original/forwarded are both
          None when aliases are off or the id is unknown
    """

    schema_version: int = 6
    session_id: str
    user_id: str | None = None
    turn_id: str
    capture_id: str = ""
    created_at: str
    latency_ms: float
    client_ip: str
    client_user_agent: str
    model: str
    run_id: str | None = None
    runtime_settings: dict[str, Any] | None = None
    request: dict[str, Any]
    request_body_sha256: str = ""
    messages_sha256: str = (
        ""  # For embedding requests, carries input_sha256 (semantic alias)
    )
    attempt_number: int = 1
    response: dict[str, Any] | None = None
    tool_calls: list[dict[str, Any]] | None = None
    finish_reason: str | None = None
    usage_raw: dict[str, Any] | None = None
    usage_normalized: UsageNormalized | None = None
    upstream_status: int
    error_body: str | None = None
    routing: str | None = None
    stream_diagnostics: StreamDiagnostics | None = None
    stream_file: str | None = None
    request_type: str | None = None
    task: str | None = None
    task_rule: str | None = None
    task_evidence: dict[str, Any] | None = None
    embedding_count: int | None = None
    embedding_dimensions: int | None = None
    input_sha256: str | None = None
    headroom: HeadroomRef | None = None
    request_model_original: str | None = None
    request_model_forwarded: str | None = None

    @field_validator("capture_id")
    @classmethod
    def _validate_capture_id(cls, value: str) -> str:
        return _check_capture_uuid(value)


class ChatCompletionRequest(BaseModel):
    """OpenAI-compatible chat completion request body."""

    model: str = "unknown"
    messages: list[dict[str, Any]] = Field(default_factory=list)
    stream: bool = False
    tools: list[dict[str, Any]] | None = None
    temperature: float | None = None
    max_tokens: int | None = None


class EmbeddingRequest(BaseModel):
    """OpenAI-compatible embedding request body.

    Attributes:
        input: Text(s) to embed. Single string or list of strings.
        model: Model identifier (non-empty).
        encoding_format: Encoding format — "float" (default) or "base64".
        stream: Streaming flag. Embeddings do not support streaming.

    Invariants:
        - input MUST be non-empty (str or list) — validated at handler level
        - model MUST be non-empty — validated at handler level
        - stream=True is rejected with 400 — embeddings never stream
    """

    input: str | list[str]
    model: str
    encoding_format: Literal["float", "base64"] = "float"
    stream: bool = False


class EmbeddingData(BaseModel):
    """Single embedding vector entry in OpenAI-compatible response.

    Attributes:
        object: Always "embedding".
        index: Position index within the batch (0-based).
        embedding: Float vector of the computed embedding.

    Invariants:
        - object is always "embedding" — never varies
        - index matches position in the data list
    """

    object: Literal["embedding"] = "embedding"
    index: int
    embedding: list[float]


class EmbeddingUsage(BaseModel):
    """Token usage for an embedding request.

    Attributes:
        prompt_tokens: Number of tokens in the input.
        total_tokens: Total tokens consumed (= prompt_tokens for embeddings).

    Invariants:
        - total_tokens == prompt_tokens for embedding requests
        - Both are non-negative integers
    """

    prompt_tokens: int = 0
    total_tokens: int = 0


class EmbeddingResponse(BaseModel):
    """OpenAI-compatible embedding response body.

    Attributes:
        object: Always "list".
        data: List of EmbeddingData entries (one per input).
        model: Model identifier used for embedding.
        usage: Token usage summary.

    Invariants:
        - object is always "list" — never varies
        - len(data) == len(input) from request
    """

    object: Literal["list"] = "list"
    data: list[EmbeddingData]
    model: str
    usage: EmbeddingUsage


class RerankRequest(BaseModel):
    """OpenAI-compatible rerank request body.

    Attributes:
        model: Model identifier (non-empty string).
        query: Search query text (non-empty string).
        documents: List of documents to rerank (non-empty list).
        top_n: Maximum number of top results to return (optional, >= 1).
        stream: Streaming flag. Rerank does not support streaming.

    Invariants:
        - model MUST be non-empty — validated at handler level
        - query MUST be non-empty — validated at handler level
        - documents MUST be non-empty — validated at handler level
        - top_n MUST be >= 1 when provided — validated at handler level
        - stream=True is rejected with 400 — rerank never streams
    """

    model: str
    query: str
    documents: list[str]
    top_n: int | None = None
    stream: bool = False


class ClassificationRequest(BaseModel):
    """Classifier request body for POST /v1/classification.

    Attributes:
        model: Model identifier (non-empty string).
        inputs: Text to classify. Single string or list of strings.
        parameters: Optional model parameters passed through untouched
            (e.g. {"truncate": true}); None when the client sent no block.

    Invariants:
        - model MUST be non-empty — validated at handler level
        - inputs MUST be non-empty (str or list) — validated at handler level
        - classification does not stream; no ``stream`` field exists, so a
          client-sent ``stream`` key is neither rejected here nor stripped
          (the handler forwards the parsed client body, not this model)
    """

    model: str
    inputs: str | list[str]
    parameters: dict[str, Any] | None = None


class ClassificationResponse(BaseModel):
    """Classifier response body forwarded from upstream.

    Declares the wire shape the proxy returns verbatim; the handler does not
    validate against it, so an upstream that returns extra keys or a wider
    shape is still forwarded unchanged.

    Attributes:
        labels: One label per input, in input order.
        scores: One score per input, parallel to ``labels``.
        model: Model identifier reported by the upstream.

    Invariants:
        - len(scores) == len(labels) for a conforming upstream response
        - scores are not clamped or reordered by the proxy
    """

    labels: list[str]
    scores: list[float]
    model: str


class ImageRequest(BaseModel):
    """Image generation request body for POST /v1/images/generations.

    Attributes:
        model: Model identifier (non-empty string).
        prompt: Generation prompt (non-empty string).
        n: Number of images requested (>= 1 — validated at handler level).
        size: Output size as "WIDTHxHEIGHT"; forwarded to upstream verbatim.
        response_format: "url" (default) or "b64_json"; forwarded verbatim.

    Invariants:
        - model MUST be non-empty — validated at handler level
        - prompt MUST be non-empty — validated at handler level
        - n MUST be >= 1 — validated at handler level
        - image generation does not stream; no ``stream`` field exists, so a
          client-sent ``stream`` key is neither rejected here nor stripped
          (the handler forwards the parsed client body, not this model)
    """

    model: str
    prompt: str
    n: int = 1
    size: str = "1024x1024"
    response_format: Literal["url", "b64_json"] = "url"


class ImageResponse(BaseModel):
    """Image generation response body forwarded from upstream.

    Declares the wire shape the proxy returns verbatim; the handler does not
    validate against it, so an upstream that returns extra keys or a wider
    shape is still forwarded unchanged.

    Attributes:
        data: One dict per generated image (url or b64_json payload plus
            upstream keys).
        model: Model identifier reported by the upstream.
        object: Always "list".

    Invariants:
        - object is always "list" — never varies
        - len(data) matches the request's n when the upstream honors it
    """

    data: list[dict[str, Any]]
    model: str
    object: Literal["list"] = "list"


class HealthResponse(BaseModel):
    """Response schema for GET /health endpoint."""

    status: str = "ok"
    upstream: str = "disconnected"


class IndexEntry(BaseModel):
    """One capture reference line from a session/user index file.

    Attributes:
        ts: ISO 8601 timestamp of the index append.
        capture_file: Capture JSONL path relative to CAPTURE_BASE_PATH.
        category: Capture category — "raw" or "errors".
        capture_id: Capture identifier (None when absent in the source line).
        index_key: The declared dedup key — always equal to capture_id
            when present. A capture carrying both a session and a user id
            is written to BOTH index shards under this one key, so a count
            taken over raw rows double-counts; dedupe on this field.
        shard: Which index this line lives in — "sessions" or "users".
            Present so a reader spanning both shards can tell a duplicate
            row from a genuinely distinct turn.
        turn_id: Turn identifier (None when absent).
        model: Model name from the request (None when absent).
        request_type: "chat" | "completion" | "embedding" | "rerank" |
            "classification" | "image" (None when absent in the source line).
        task: Structural task label from CaptureRecord.task (None when the
            line predates structural labelling). Carrying it here is what
            lets a task query run off the index instead of re-deriving
            from every raw body.
        task_rule: Which classifier rule produced ``task``.
        stream_file: Relative path of the companion .stream.jsonl, when the
            capture streamed. Present so a stream capture is reachable from
            the index without opening the raw line first.
        upstream_status: Upstream HTTP status (None when absent).
        session_id: Session the capture belongs to (present on both index
            kinds so a user query shows which session each capture came from).
        user_id: User the capture belongs to.

    Invariants:
        - capture_file joined with CAPTURE_BASE_PATH points at the exact
          JSONL file holding the full CaptureRecord
        - (index_key, shard) is unique per line; the same index_key may
          appear under two different shards, and those are one turn
    """

    ts: str
    capture_file: str
    category: str
    index_key: str | None = None
    shard: str | None = None
    capture_id: str | None = None
    turn_id: str | None = None
    model: str | None = None
    request_type: str | None = None
    task: str | None = None
    task_rule: str | None = None
    stream_file: str | None = None
    upstream_status: int | None = None
    session_id: str | None = None
    user_id: str | None = None


class CaptureIndexResponse(BaseModel):
    """Response for GET /v1/sessions/{id}/captures and GET /v1/users/{id}/captures."""

    kind: Literal["session", "user"]
    id: str
    count: int
    captures: list[IndexEntry]


class IdSummary(BaseModel):
    """One known session/user id with activity bounds.

    Attributes:
        id: The raw session_id or user_id value.
        captures: Number of indexed captures for this id.
        first_seen: ISO 8601 timestamp of the earliest indexed capture.
        last_seen: ISO 8601 timestamp of the latest indexed capture.
    """

    id: str
    captures: int
    first_seen: str | None = None
    last_seen: str | None = None


class IdListResponse(BaseModel):
    """Response for GET /v1/sessions and GET /v1/users listings."""

    kind: Literal["session", "user"]
    count: int
    ids: list[IdSummary]
