"""
Session Resolver

Owns: Session and user identity resolution from request headers
Depends on: hashlib (stdlib), uuid (stdlib)
Provides: SessionResolver.resolve(), SessionResolver.resolve_user_id()

Invariants:
  - 100% of calls return non-null session_id — INV-1
  - Same input → same output (deterministic) — INV-2
  - user_id and session_id are DISTINCT namespaces: even when both fall
    back from the same Authorization header, the derived strings differ
    (user ids carry a "user-" prefix; session fallback is a UUID v4)
  - Raw Authorization values are never returned by resolve_user_id —
    only the SHA-256-derived hash prefix leaves this module
"""

from __future__ import annotations

import hashlib
import uuid

# Domain-separated hash input for the Authorization→user_id fallback so a
# user id can never collide with (or equal) any session id derivation.
_USER_HASH_DOMAIN = "capture-proxy-user|"

# Grouping bucket for requests carrying neither x-user-id nor Authorization.
ANONYMOUS_USER_ID = "anonymous"


class SessionResolver:
    """
    Resolves session_id from x-session-id header or Authorization key hash.

    Purpose: Ensures every captured request has a stable session identifier.
    Dependencies: hashlib (stdlib) for API key hashing, uuid (stdlib) for UUID generation

    Invariants:
      - Returns non-null string in all cases
      - x-session-id header takes priority over Authorization hash
      - Falls back to random UUID v4 when neither header present
    """

    def resolve(self, headers: dict[str, str]) -> str:
        """
        Return non-null session_id. x-session-id > Authorization hash > random UUID.

        Args:
            headers: Request headers dict (keys are case-insensitive per HTTP spec)

        Returns:
            str: Raw x-session-id header value when present (verbatim, not
                validated); otherwise a deterministic UUID v4 hashed from the
                Authorization header; otherwise a random UUID v4 string.

        Raises: None

        Invariants:
          - Before: headers dict may be empty
          - After: Always returns a non-null string (x-session-id verbatim,
            auth hash UUID, or random UUID)
          - Deterministic: same headers → same session_id (except random-UUID path)
        """
        session_id_header = self._find_header(headers, "x-session-id")
        if session_id_header:
            return session_id_header

        auth_header = self._find_header(headers, "authorization")
        if auth_header:
            return self._hash_to_uuid(auth_header)

        return str(uuid.uuid4())

    def resolve_user_id(self, headers: dict[str, str]) -> str:
        """
        Return non-null user_id. x-user-id > Authorization hash > "anonymous".

        User identity answers WHO sent the request — a different question from
        session_id (WHICH conversation). The derivations are deliberately
        namespaced apart so an auth-only client gets a stable user bucket
        without collapsing into any single session.

        Args:
            headers: Request headers dict (keys are case-insensitive per HTTP
                spec)

        Returns:
            str: The stripped x-user-id header value when present and
                non-blank; otherwise ``"user-"`` + first 16 hex chars of
                SHA-256 over a domain-separated Authorization value (the raw
                key is never returned or stored); otherwise the constant
                ``"anonymous"`` so headerless traffic still groups into one
                queryable bucket.

        Raises: None

        Invariants:
          - Before: headers dict may be empty
          - After: Always returns a non-empty string
          - Deterministic: same headers → same user_id in all branches
            (including the anonymous fallback)
          - Empty/whitespace x-user-id is treated as absent (falls through)
          - Output never contains the raw Authorization value
        """
        user_header = self._find_header(headers, "x-user-id")
        if user_header and user_header.strip():
            return user_header.strip()

        auth_header = self._find_header(headers, "authorization")
        if auth_header and auth_header.strip():
            digest = hashlib.sha256(
                (_USER_HASH_DOMAIN + auth_header).encode("utf-8")
            ).hexdigest()
            return f"user-{digest[:16]}"

        return ANONYMOUS_USER_ID

    def _find_header(self, headers: dict[str, str], name: str) -> str | None:
        """
        Case-insensitive header lookup.

        Args:
            headers: Request headers dict
            name: Header name to find (case-insensitive)

        Returns:
            str | None: Header value if found, None otherwise

        Raises: None

        Invariants:
          - Case-insensitive matching per HTTP specification
        """
        for key, value in headers.items():
            if key.lower() == name.lower():
                return value
        return None

    def _hash_to_uuid(self, value: str) -> str:
        """
        Hash a string value to a deterministic UUID v4.

        Args:
            value: String to hash (e.g., API key from Authorization header)

        Returns:
            str: UUID v4 string derived from SHA-256 hash of input

        Raises: None

        Invariants:
          - Deterministic: same input → same UUID
          - Output is always a valid UUID v4 format
        """
        hash_bytes = hashlib.sha256(value.encode()).digest()[:16]
        return str(uuid.UUID(bytes=hash_bytes, version=4))
