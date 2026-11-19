"""Model alias map and per-request rewrite/sanitize helpers.

Owns: Public-to-upstream model name mapping and exact-match field rewrites.
Depends: json (stdlib) only; no I/O, no subprocess, no network.
Provides: AliasConfigError, AliasMap, resolve_aliases, rewrite_request_model,
    sanitize_response_model, sanitize_sse_event, sanitize_models_list.

Invariants:
    - Exact case-sensitive full-string match only; never substring or regex.
    - None aliases means off: every helper returns the input unchanged
      (same object for dict payloads, verbatim string for SSE).
    - Duplicate upstream values are fatal at startup (caller exits 2).
    - Only parsed ``model``/``id`` fields are rewritten; message content,
      tool calls, usage, and SSE ``delta.content`` bytes are never touched.
    - Models-list sanitize drops the ``root`` key from each entry whenever
      aliases are on (literal root-strip).
    - Alias mode intentionally forks byte-identity to field-identity: when
      on, downstream bytes differ from upstream bytes exactly in the
      remapped ``model``/``id`` fields. Off-mode preserves byte-identity.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


class AliasConfigError(ValueError):
    """Raised for invalid model-alias configuration at startup only.

    Covers empty maps, self-maps (a name appearing as both public and
    upstream), duplicate upstream targets, non-string entries, oversized
    maps, and malformed JSON. Per-request helpers never raise this;
    they take ``AliasMap | None`` and treat ``None`` as infallible off.
    """


_MAX_ALIASES = 64


def _clean_name(value: object, kind: str) -> str:
    """Strip and validate a single alias name.

    Args:
        value: Candidate name; must be a string.
        kind: Label used in error text ("key" or "value").

    Returns:
        The whitespace-stripped name.

    Raises:
        AliasConfigError: When ``value`` is not a string or is empty
            after stripping whitespace.
    """
    if not isinstance(value, str):
        raise AliasConfigError(
            f"alias {kind} must be a string, got {type(value).__name__}"
        )
    cleaned = value.strip()
    if not cleaned:
        raise AliasConfigError(f"alias {kind} must be non-empty after stripping")
    return cleaned


@dataclass(frozen=True)
class AliasMap:
    """Frozen public-name to upstream-name value object.

    Attributes:
        forward: Mapping of public model id to upstream model id.
            Keys and values are stripped non-empty strings, 1..64 entries.

    Invariants:
        - Pre: ``forward`` has 1..64 entries; every key/value is a
          non-empty stripped string; stripped keys are unique; values
          are unique; no name appears as both a key and a value.
        - Post: ``forward`` is a defensive copy; ``reverse`` is the exact
          inverse mapping (upstream to public).
        - Frozen: attribute reassignment raises after construction.
        - Side effects: none (pure value object, no I/O).

    Raises:
        AliasConfigError: On empty map, oversized map (>64), non-string
            or blank entry, self-map/chain (keys intersect values), or
            duplicate upstream target.
    """

    forward: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate entries and store a defensive copy.

        Args: None (operates on ``self.forward``).

        Returns: None.

        Raises:
            AliasConfigError: On any contract violation listed in the
                class docstring.

        Invariants:
            - Before: ``self.forward`` holds caller-supplied dict.
            - After: ``self.forward`` holds stripped defensive copy.
            - Side effects: none beyond ``object.__setattr__`` copy.
        """
        raw = self.forward
        if not isinstance(raw, dict):
            raise AliasConfigError("aliases must be a mapping of str to str")
        if len(raw) == 0:
            raise AliasConfigError("aliases must contain 1..64 entries (got 0)")
        if len(raw) > _MAX_ALIASES:
            raise AliasConfigError(
                f"aliases must contain 1..64 entries (got {len(raw)})"
            )
        cleaned: dict[str, str] = {}
        for raw_key, raw_value in raw.items():
            key = _clean_name(raw_key, "key")
            value = _clean_name(raw_value, "value")
            if key in cleaned:
                raise AliasConfigError(f"duplicate alias key after stripping: {key!r}")
            cleaned[key] = value
        keys = set(cleaned.keys())
        values = list(cleaned.values())
        if len(set(values)) != len(values):
            seen: set[str] = set()
            dup = next(v for v in values if v in seen or seen.add(v))  # type: ignore[func-returns-value]
            raise AliasConfigError(f"duplicate upstream target: {dup!r}")
        overlap = keys.intersection(cleaned.values())
        if overlap:
            name = sorted(overlap)[0]
            raise AliasConfigError(
                f"alias name appears as both public and upstream: {name!r}"
            )
        object.__setattr__(self, "forward", dict(cleaned))

    @property
    def reverse(self) -> dict[str, str]:
        """Derive the upstream-to-public inverse mapping.

        Args: None.

        Returns:
            New dict mapping each upstream id back to its public id.
                Empty only when unreachable (constructor requires >=1).

        Raises: None (infallible; derived from validated forward map).

        Invariants:
            - Post: ``reverse[forward[k]] == k`` for every key ``k``.
            - Side effects: none; fresh dict per call so callers cannot
              mutate stored state.
        """
        return {v: k for k, v in self.forward.items()}


def resolve_aliases(raw: str | dict[str, object] | None) -> AliasMap | None:
    """Parse alias configuration into an AliasMap or off.

    Args:
        raw: JSON-string mapping, TOML-table dict mapping, ``None``,
            or an off-sentinel string (``""``, whitespace-only,
            ``"off"`` case-insensitive, ``"..."``). JSON must decode
            to an object mapping strings to strings. Dict values must
            already be strings (TOML tables).

    Returns:
        AliasMap when a non-empty mapping is supplied; None when off
            (``None``, empty/blank string, ``"off"``, ``"..."``).

    Raises:
        AliasConfigError: On malformed JSON, JSON decoding to a
            non-object, empty map, self-map/chain, duplicate upstream
            target, non-string entry, or oversized map. No I/O is
            performed; caller warns and falls back to off (fail-open),
            except duplicate-upstream which the startup caller treats
            as fatal.

    Invariants:
        - Pre: ``raw`` is None, a string, or a string-to-string dict.
        - Post: return is None (off) or a validated frozen AliasMap.
        - Side effects: none (pure parse, no I/O, no env access).
    """
    if raw is None:
        return None
    if isinstance(raw, dict):
        if len(raw) == 0:
            raise AliasConfigError("aliases must contain 1..64 entries (got 0)")
        forward: dict[str, str] = {}
        for k, v in raw.items():
            if not isinstance(k, str) or not isinstance(v, str):
                raise AliasConfigError("aliases must map str to str")
            forward[k] = v
        return AliasMap(forward)
    if isinstance(raw, str):
        stripped = raw.strip()
        if stripped == "" or stripped.lower() == "off" or stripped == "...":
            return None
        try:
            decoded: Any = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise AliasConfigError(f"invalid model-aliases JSON: {exc}") from exc
        if not isinstance(decoded, dict):
            raise AliasConfigError("model-aliases JSON must decode to an object")
        forward_json: dict[str, str] = {}
        for k, v in decoded.items():
            if not isinstance(k, str) or not isinstance(v, str):
                raise AliasConfigError("aliases must map str to str")
            forward_json[k] = v
        return AliasMap(forward_json)
    raise AliasConfigError("aliases must be a JSON string, table, or None")


def rewrite_request_model(
    body: dict[str, Any], aliases: AliasMap | None
) -> tuple[dict[str, Any], str | None, str | None]:
    """Rewrite the top-level ``model`` field from public to upstream id.

    Args:
        body: Parsed request body; never mutated.
        aliases: AliasMap or None (None means off).

    Returns:
        Tuple of ``(new_body, original, forwarded)`` where:
            - Hit: ``(shallow copy with model replaced, public, upstream)``.
            - Off / missing / non-string model / unknown id:
              ``(body UNCOPIED same object, None, None)``.

    Raises: None (infallible; O(1) exact dict lookup).

    Invariants:
        - Pre: ``body`` is a dict; ``aliases`` is None or validated.
        - Post: input ``body`` is never mutated; hit returns a new
          shallow copy, miss returns the identical object.
        - Side effects: none.
    """
    if aliases is None:
        return (body, None, None)
    model = body.get("model")
    if not isinstance(model, str):
        return (body, None, None)
    target = aliases.forward.get(model)
    if target is None:
        return (body, None, None)
    new_body = dict(body)
    new_body["model"] = target
    return (new_body, model, target)


def sanitize_response_model(
    payload: dict[str, Any] | None, aliases: AliasMap | None
) -> dict[str, Any] | None:
    """Remap the top-level ``model`` field from upstream to public id.

    Args:
        payload: Upstream response body dict or None.
        aliases: AliasMap or None (None means off).

    Returns:
        - None when ``payload`` is None.
        - The same object UNCOPIED when off, when ``model`` is missing
          or non-string, or when the id has no reverse hit.
        - A shallow copy with ``model`` replaced via the reverse map
          on an exact hit. ``choices`` content and ``usage`` are never
          touched.

    Raises: None (infallible; O(1) exact dict lookup).

    Invariants:
        - Pre: ``payload`` is None or a dict.
        - Post: input is never mutated; only the top-level ``model``
          key may differ in the copy.
        - Side effects: none.
    """
    if payload is None or aliases is None:
        return payload
    model = payload.get("model")
    if not isinstance(model, str):
        return payload
    public = aliases.reverse.get(model)
    if public is None:
        return payload
    new_payload = dict(payload)
    new_payload["model"] = public
    return new_payload


def sanitize_sse_event(raw_data: str, aliases: AliasMap | None) -> str:
    """Remap the ``model`` field inside one SSE data payload.

    Args:
        raw_data: SSE ``data:`` payload text (without the ``data:``
            prefix); ``"[DONE]"`` is a verbatim sentinel.
        aliases: AliasMap or None (None means off).

    Returns:
        - ``"[DONE]"`` verbatim (never parsed).
        - Input verbatim when off, when the payload is non-JSON, when
          it decodes to a non-object, when ``model`` is missing or
          non-string, or when the id has no reverse hit (no
          re-serialization, preserving byte-identity).
        - Re-serialized JSON string with ``model`` replaced via the
          reverse map on an exact hit. ``delta.content`` string values
          are byte-identical (only the ``model`` key changes).

    Raises: None (infallible; corrupt payloads pass through).

    Invariants:
        - Pre: ``raw_data`` is a string.
        - Post: miss returns the identical string value unmodified;
          hit returns JSON text decoding to the same object except
          for the ``model`` field.
        - Side effects: none; caller reframes as ``data: {out}\\n\\n``.
    """
    if aliases is None:
        return raw_data
    if raw_data == "[DONE]":
        return raw_data
    try:
        parsed: Any = json.loads(raw_data)
    except json.JSONDecodeError:
        return raw_data
    if not isinstance(parsed, dict):
        return raw_data
    model = parsed.get("model")
    if not isinstance(model, str):
        return raw_data
    public = aliases.reverse.get(model)
    if public is None:
        return raw_data
    parsed["model"] = public
    return json.dumps(parsed, ensure_ascii=False)


def sanitize_models_list(
    payload: dict[str, Any] | None, aliases: AliasMap | None
) -> dict[str, Any] | None:
    """Remap ``data[].id`` entries from upstream to public ids.

    Args:
        payload: Upstream ``/v1/models`` body (``{"data": [...]}``)
            or None.
        aliases: AliasMap or None (None means off).

    Returns:
        - None when ``payload`` is None (verbatim).
        - The same object UNCHANGED when off.
        - The same object verbatim when ``data`` is missing or not
          a list.
        - Otherwise a shallow copy of the payload with a new ``data``
          list: each dict entry is shallow-copied, its ``id`` remapped
          via exact reverse lookup (unknown ids verbatim), its ``root``
          key dropped, all other keys preserved; order and length
          preserved. Non-dict entries pass through verbatim.

    Raises: None (infallible; O(E) over entries).

    Invariants:
        - Pre: ``payload`` is None or a dict.
        - Post: input is never mutated; when on with list data the
          ``root`` key is absent from every dict entry in the copy.
        - Side effects: none.
    """
    if payload is None or aliases is None:
        return payload
    data = payload.get("data")
    if not isinstance(data, list):
        return payload
    reverse = aliases.reverse
    new_data: list[Any] = []
    for entry in data:
        if not isinstance(entry, dict):
            new_data.append(entry)
            continue
        new_entry = dict(entry)
        entry_id = new_entry.get("id")
        if isinstance(entry_id, str):
            public = reverse.get(entry_id)
            if public is not None:
                new_entry["id"] = public
        new_entry.pop("root", None)
        new_data.append(new_entry)
    new_payload = dict(payload)
    new_payload["data"] = new_data
    return new_payload
