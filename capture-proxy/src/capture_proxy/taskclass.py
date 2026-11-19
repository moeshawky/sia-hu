"""
Structural Task Classification

Owns: Derivation of a capture's ``task`` label from request *shape*
Depends: json (stdlib) only — no regex over topic vocabulary
Provides: TaskClassification, classify_request, TASK_VOCABULARY,
classify_capture_record

Invariants:
  - CLASSIFICATION IS STRUCTURAL. Every rule reads payload keys, key
    presence/absence, message counts, or byte size. No rule inspects a
    topic word to decide *what the request is about*; the single lexical
    component (SUMMARY_INSTRUCTION_TOKENS) is reachable ONLY from inside
    the code-bearing subset, which is itself reached only by a body-size
    and code-density measurement
  - The session trap is closed by the toolbelt gate, not by keyword
    filtering: an agent session carries ``tools`` and is therefore
    ``chat.agentic`` before any second pass can run, so session text that
    *discusses* reranking or summarization cannot reach a task label
  - A body with no tools and no messages is still ``chat.task_api``, never
    ``chat.agentic`` — absence of a toolbelt is the discriminator, so an
    unlabeled programmatic caller is classified, not discarded
  - classification never raises: a malformed body degrades to the widest
    label (``chat.agentic``) with ``rule="fallback"``
  - evidence is always populated with the measurements the rule actually
    used, so a label can be audited without re-running the classifier
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

#: Body size (serialized UTF-8 bytes) at or above which a no-tools chat is
#: treated as a session rather than a single task call. 4096 = 4 KiB.
SINGLE_SHOT_MAX_BYTES = 4096

#: Message-count band for a single-shot call (inclusive on both ends).
SINGLE_SHOT_MIN_MESSAGES = 1
SINGLE_SHOT_MAX_MESSAGES = 4

#: Minimum share of payload lines that must look like source code before a
#: prompt is treated as code-bearing. Measured per line, not by topic word.
CODE_LINE_RATIO_THRESHOLD = 0.12

#: Minimum absolute number of code-bearing lines, so a 3-line request that
#: happens to contain two brackets cannot be mistaken for a code payload.
CODE_LINE_MIN_COUNT = 6

#: Maximum share of DOCUMENT SCAFFOLDING lines a code-bearing payload may
#: contain. This is the guard that separates a pasted source payload from a
#: Markdown document that merely *contains* source.
#:
#: Measured 2026-10-15 over 251 deployed skill documents and 69 Python
#: source files. Document scaffolding (fence markers, ATX headings, table
#: rows, list bullets, blockquote markers) as a fraction of non-blank
#: lines: skill documents min 0.050, median 0.487; source files median
#: 0.000. At a threshold of 0.03 this admits 0/251 skill documents and
#: 64/69 source files.
#:
#: The margin is deliberately asymmetric. A false positive here is the
#: exact defect the corpus measurement found — an agent session labelled
#: as a task call — while a false negative is merely a missed label. The
#: threshold sits 40% below the lowest observed skill document so an
#: unusual document does not drift over the line.
DOC_SCAFFOLD_MAX_RATIO = 0.03

#: The ONLY lexical component in this module. Reachable only inside the
#: code-bearing subset; it names an *instruction form*, not a subject
#: matter. It cannot see "rerank", "embedding", "agent", or a skill name.
SUMMARY_INSTRUCTION_TOKENS = frozenset(
    {
        "summarize",
        "summarise",
        "summary",
        "condense",
        "shorten",
        "tl;dr",
        "tldr",
        "digest",
        "abstract",
        "recap",
        "overview",
    }
)

#: Closed vocabulary written into schema.json. Adding a value is a schema
#: change, not a runtime decision.
TASK_VOCABULARY: tuple[str, ...] = (
    "chat.task_api",
    "chat.single_shot",
    "chat.agentic",
    "embedding.batch",
    "embedding.single",
    "rerank",
    "code_summarize",
    "classification.single",
    "classification.batch",
    "image.generate",
    "completion.text",
)

# Tokens that mark a line as source code. Deliberately generic lexical
# syntax (punctuation and reserved words), never domain vocabulary: these
# fire on any programming language and cannot be tripped by prose that
# merely discusses code.
_CODE_SIGNALS = frozenset(
    {
        "def ",
        "class ",
        "return",
        "import ",
        "from ",
        "function ",
        "const ",
        "let ",
        "var ",
        "async ",
        "await ",
        "public ",
        "private ",
        "static ",
        "void ",
        "struct ",
        "impl ",
        "fn ",
        "end",
        "elif ",
        "else:",
        "try:",
        "except",
        "catch",
        "#include",
        "package ",
        "func ",
        "interface ",
        "=>",
    }
)

# Markdown document scaffolding patterns. This is a FORMAT test — the same
# class of measurement as "is this JSON" — and not a topic test: none of
# these patterns names a subject matter, so no document that merely
# discusses reranking or summarization can be pulled through by wording.
_FENCE_RE = re.compile(r"^\s*```")
_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s")
_TABLE_ROW_RE = re.compile(r"^\s*\|")
_BULLET_RE = re.compile(r"^\s{0,4}[-*+]\s")
_BLOCKQUOTE_RE = re.compile(r"^\s*>")

_SCAFFOLD_PATTERNS = (
    _FENCE_RE,
    _HEADING_RE,
    _TABLE_ROW_RE,
    _BULLET_RE,
    _BLOCKQUOTE_RE,
)


def _scaffold_ratio(lines: list[str]) -> float:
    """Measure what share of lines are document scaffolding.

    Args:
        lines: Non-blank lines of the payload.

    Returns:
        float: Fraction of lines carrying a Markdown scaffolding marker;
            0.0 when there are no lines.
    """
    if not lines:
        return 0.0
    marked = sum(1 for line in lines if any(p.match(line) for p in _SCAFFOLD_PATTERNS))
    return marked / len(lines)


@dataclass(frozen=True)
class TaskClassification:
    """A task label plus the structural measurements that produced it.

    Attributes:
        task: One of ``TASK_VOCABULARY``. Never None — an unlabeled body
            degrades to ``chat.agentic`` rather than going unclassified.
        rule: Identifier of the rule that fired, e.g. ``"no_tools"``.
        evidence: The measurements the rule read (key presence, counts,
            sizes, densities). Always populated, never empty.
    """

    task: str
    rule: str
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-safe form written into the capture record.

        Returns:
            dict: ``{"task", "task_rule", "task_evidence"}`` — flat keys so
                a consumer can read them without unpacking a sub-object.
        """
        return {
            "task": self.task,
            "task_rule": self.rule,
            "task_evidence": self.evidence,
        }


def _has_tools(request: dict[str, Any]) -> bool:
    """Return True when the request carries a non-empty ``tools`` array.

    Args:
        request: Parsed request body.

    Returns:
        bool: True when ``tools`` is a non-empty list. A ``tools`` key that
            is null, absent, or an empty list counts as *no* toolbelt —
            an agent harness that offers zero tools is indistinguishable
            from a programmatic caller on the wire.
    """
    tools = request.get("tools")
    return isinstance(tools, list) and len(tools) > 0


def _serialized_size(request: dict[str, Any]) -> int:
    """Return the serialized UTF-8 byte size of a request body.

    Args:
        request: Parsed request body.

    Returns:
        int: Byte length of the canonical JSON encoding; 0 when the body
            cannot be serialized.
    """
    try:
        return len(json.dumps(request, ensure_ascii=False, default=str).encode())
    except (TypeError, ValueError):
        return 0


def _code_stats(text: str) -> tuple[int, int]:
    """Measure code-bearing lines in a text block.

    Counts a line as code-bearing when it carries a structural marker:
    balanced braces/brackets, a trailing semicolon, or a reserved-word
    token from ``_CODE_SIGNALS``. This is a *shape* measurement over
    punctuation, not a topic measurement — prose about programming has no
    braces, semicolons, or ``def``/``import`` keywords.

    Args:
        text: Concatenated message content.

    Returns:
        tuple[int, int]: ``(code_lines, total_lines)``. ``total_lines`` is
            0 when ``text`` has no non-blank lines.
    """
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        return 0, 0
    code_lines = 0
    for line in lines:
        stripped = line.strip()
        if (
            "{" in stripped
            or "}" in stripped
            or ";" in stripped
            or ("(" in stripped and ")" in stripped)
            or any(sig in stripped.lower() for sig in _CODE_SIGNALS)
        ):
            code_lines += 1
    return code_lines, len(lines)


def _message_text(request: dict[str, Any]) -> str:
    """Concatenate the text of every message in a chat body.

    Handles the OpenAI content-block form (``content`` as a list of
    ``{"type": "text", "text": ...}`` dicts) as well as the plain string
    form, so an agent harness using blocks is measured the same way.

    Args:
        request: Parsed request body.

    Returns:
        str: Concatenated message text; empty string when there is none.
    """
    messages = request.get("messages")
    if not isinstance(messages, list):
        return ""
    parts: list[str] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and isinstance(block.get("text"), str):
                    parts.append(block["text"])
    return "\n".join(parts)


def _classify_code_summarize(request: dict[str, Any]) -> TaskClassification | None:
    """Second-pass structural test for a code-summarization task call.

    Runs ONLY for a body already labelled ``chat.task_api``. Three gates,
    in cost order:

    1. **Document scaffold** — the body must not be a Markdown document.
       This is the gate that closes the measured false positive. An agent
       session whose system prompt carries a skill body is text with
       fences, headings, tables and bullets; a summarization call carries
       raw source. Format, not wording, decides.
    2. **Code density** — enough lines must carry code punctuation.
    3. **Summarization instruction** — consulted ONLY after 1 and 2 pass,
       so the one lexical component can never see prose that merely uses
       the word.

    Args:
        request: Parsed request body (no toolbelt by precondition).

    Returns:
        TaskClassification | None: The classification when all three gates
            pass, else None so the caller keeps its ladder label.
    """
    text = _message_text(request)
    code_lines, total_lines = _code_stats(text)
    if total_lines == 0 or code_lines < CODE_LINE_MIN_COUNT:
        return None

    lines = [ln for ln in text.splitlines() if ln.strip()]
    scaffold = _scaffold_ratio(lines)
    if scaffold > DOC_SCAFFOLD_MAX_RATIO:
        return None

    ratio = code_lines / total_lines
    if ratio < CODE_LINE_RATIO_THRESHOLD:
        return None

    lowered = text.lower()
    if not any(token in lowered for token in SUMMARY_INSTRUCTION_TOKENS):
        return None

    return TaskClassification(
        task="code_summarize",
        rule="not_a_document+code_density+summarize_instruction",
        evidence={
            "code_lines": code_lines,
            "total_lines": total_lines,
            "code_ratio": round(ratio, 4),
            "code_ratio_threshold": CODE_LINE_RATIO_THRESHOLD,
            "doc_scaffold_ratio": round(scaffold, 4),
            "doc_scaffold_max": DOC_SCAFFOLD_MAX_RATIO,
            "instruction_matched": True,
        },
    )


def _classify_rerank(request: dict[str, Any]) -> TaskClassification | None:
    """Second-pass structural test for a rerank task call.

    Keys on payload structure alone: a rerank body carries a ``query``
    string alongside a ``documents`` array. A chat body that merely uses
    the words "query" and "documents" in prose has neither key at the top
    level and cannot satisfy this rule.

    Args:
        request: Parsed request body (no toolbelt by precondition).

    Returns:
        TaskClassification | None: The classification when both keys are
            present with the right types, else None.
    """
    query = request.get("query")
    documents = request.get("documents")
    if not isinstance(query, str) or not isinstance(documents, list):
        return None
    return TaskClassification(
        task="rerank",
        rule="payload_keys.query+documents",
        evidence={
            "query_type": "str",
            "documents_count": len(documents),
            "top_n": request.get("top_n"),
        },
    )


def classify_request(
    request: dict[str, Any],
    request_type: str | None = None,
) -> TaskClassification:
    """Derive a task label from request shape.

    Rule order is fixed and short-circuiting:

    1. ``request_type`` in a transport-declared family wins outright
       (embedding/rerank/classification/image/completion bodies are
       self-identifying by endpoint, not by inference)
    2. a non-empty ``tools`` array → ``chat.agentic`` (toolbelt gate)
    3. ``request.input`` is a list of strings → ``embedding.batch``
    4. no toolbelt → ``chat.task_api``, refined by the second pass
    5. 1-4 messages, no tools, body < 4 KiB → ``chat.single_shot``
    6. otherwise → ``chat.agentic``

    Args:
        request: Parsed request body as forwarded upstream.
        request_type: Endpoint family when the handler knows it
            ("chat", "completion", "embedding", ...). None for a body read
            off disk without an endpoint label.

    Returns:
        TaskClassification: Never raises. A body that is not a dict
            degrades to ``chat.agentic`` with ``rule="fallback"``.

    Raises: None
    """
    if not isinstance(request, dict):
        return TaskClassification(
            task="chat.agentic",
            rule="fallback",
            evidence={"error": "body_not_object"},
        )

    size_bytes = _serialized_size(request)
    input_value = request.get("input")
    messages = request.get("messages")
    message_count = len(messages) if isinstance(messages, list) else 0

    base_evidence: dict[str, Any] = {
        "request_type": request_type,
        "has_tools": _has_tools(request),
        "message_count": message_count,
        "body_bytes": size_bytes,
    }

    # 1. Transport-declared families.
    if request_type == "embedding":
        is_batch = (
            isinstance(input_value, list)
            and bool(input_value)
            and all(isinstance(item, str) for item in input_value)
        )
        if is_batch and isinstance(input_value, list):
            return TaskClassification(
                task="embedding.batch",
                rule="request_type.embedding+input_str_list",
                evidence={
                    **base_evidence,
                    "input_count": len(input_value),
                },
            )
        return TaskClassification(
            task="embedding.single",
            rule="request_type.embedding",
            evidence=base_evidence,
        )
    if request_type == "rerank":
        return TaskClassification(
            task="rerank", rule="request_type.rerank", evidence=base_evidence
        )
    if request_type == "classification":
        is_batch = isinstance(request.get("inputs"), list)
        return TaskClassification(
            task="classification.batch" if is_batch else "classification.single",
            rule="request_type.classification",
            evidence=base_evidence,
        )
    if request_type == "image":
        return TaskClassification(
            task="image.generate",
            rule="request_type.image",
            evidence=base_evidence,
        )
    if request_type == "completion":
        return TaskClassification(
            task="completion.text",
            rule="request_type.completion",
            evidence=base_evidence,
        )

    # 2. Toolbelt gate — an agent harness never becomes a task call.
    if _has_tools(request):
        tools = request.get("tools")
        tool_count = len(tools) if isinstance(tools, list) else 0
        return TaskClassification(
            task="chat.agentic",
            rule="toolbelt_present",
            evidence={**base_evidence, "tool_count": tool_count},
        )

    # 3. Batch embedding arriving without an endpoint label.
    if (
        isinstance(input_value, list)
        and input_value
        and all(isinstance(item, str) for item in input_value)
    ):
        return TaskClassification(
            task="embedding.batch",
            rule="input_str_list",
            evidence={**base_evidence, "input_count": len(input_value)},
        )

    # 4. No toolbelt — a candidate task call, refined by the second pass.
    rerank = _classify_rerank(request)
    if rerank is not None:
        return TaskClassification(
            task=rerank.task,
            rule=rerank.rule,
            evidence={**base_evidence, **rerank.evidence},
        )
    code_summarize = _classify_code_summarize(request)
    if code_summarize is not None:
        return TaskClassification(
            task=code_summarize.task,
            rule=code_summarize.rule,
            evidence={**base_evidence, **code_summarize.evidence},
        )

    # 5/6. Short no-tools chat — a task call that is not otherwise special.
    if (
        SINGLE_SHOT_MIN_MESSAGES <= message_count <= SINGLE_SHOT_MAX_MESSAGES
        and size_bytes < SINGLE_SHOT_MAX_BYTES
    ):
        return TaskClassification(
            task="chat.single_shot",
            rule="short_no_tools_chat",
            evidence=base_evidence,
        )
    return TaskClassification(
        task="chat.agentic",
        rule="long_or_tooled_conversation",
        evidence=base_evidence,
    )


def classify_capture_record(record: dict[str, Any]) -> TaskClassification:
    """Derive a task label for an already-written capture record.

    Used by the backfill/query paths to label a legacy corpus. Reads the
    stored ``request`` body and ``request_type``; a record whose task was
    already populated by the capture-time classifier keeps that label,
    because it is the label taken with full context at capture time.

    Args:
        record: One parsed CaptureRecord line.

    Returns:
        TaskClassification: The record's existing task when present,
            else a fresh structural classification of its ``request``.
    """
    existing = record.get("task")
    if isinstance(existing, str) and existing.strip():
        return TaskClassification(
            task=existing,
            rule="pre_populated",
            evidence={"source": "capture_record.task"},
        )
    return classify_request(record.get("request") or {}, record.get("request_type"))
