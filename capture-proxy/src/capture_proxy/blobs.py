"""
Content-Addressed Blob Store

Owns: Making ``request_body_sha256`` / ``messages_sha256`` resolvable
Depends: hashlib (stdlib), json (stdlib), pathlib (stdlib)
Provides: canonical_bytes, content_digest, BlobStore

Invariants:
  - ONE canonical serialization. ``canonical_bytes`` is the sole definition
    of the bytes a capture hash covers; ``app._compute_sha256`` delegates
    here so a stored blob and its recorded hash can never diverge
  - a hash that resolves is a hash you can read: every digest written to
    a CaptureRecord has a corresponding file under {base}/blobs/sha256/
  - writes are idempotent — an existing blob is left untouched, so a
    repeated turn costs one stat and no rewrite
  - layout is {base_path}/blobs/sha256/{digest[:2]}/{digest}; the 2-char
    fan-out keeps directory cardinality near-constant at corpus scale
  - blob write failure never propagates to the capture path — a blob is
    a redundancy, the capture line is the record of truth
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

#: Directory name under ``{base_path}/`` holding the store.
BLOB_ROOT = "blobs"

#: Subdirectory holding sha256-addressed blobs.
BLOB_ALGORITHM_DIR = "sha256"


def canonical_bytes(data: dict[str, Any] | list[Any] | str) -> bytes:
    """Serialize a value to the canonical bytes a capture hash covers.

    Single source of truth for the hash preimage. ``sort_keys=True`` makes
    the digest independent of dict insertion order; ``default=str`` keeps
    an exotic value from raising inside the capture path; ``ensure_ascii``
    is left at its default so the encoding is byte-identical to the
    historical ``_compute_sha256`` behavior.

    Args:
        data: Value to serialize (dict, list, or str).

    Returns:
        bytes: UTF-8 encoded canonical JSON.
    """
    return json.dumps(data, sort_keys=True, default=str).encode()


def content_digest(data: dict[str, Any] | list[Any] | str) -> str:
    """Return the lowercase hex SHA-256 of a value's canonical bytes.

    Args:
        data: Value to hash.

    Returns:
        str: 64-character lowercase hex digest.
    """
    return hashlib.sha256(canonical_bytes(data)).hexdigest()


class BlobStore:
    """Content-addressed store for request and message payloads.

    Purpose: Give every hash recorded on a CaptureRecord a resolvable
    target. A hash with no store behind it is a handle that dangles; this
    class makes the handle resolve to bytes.
    Dependencies: pathlib for file layout, json for serialization

    Invariants:
      - Layout: {base_path}/blobs/sha256/{digest[:2]}/{digest}
      - put() is idempotent and never overwrites an existing blob
      - path_for() is pure — it creates nothing
      - Every method degrades to None/False on OSError rather than raising
    """

    def __init__(self, base_path: str | Path, enabled: bool = True) -> None:
        """Initialize the store against a capture base path.

        Args:
            base_path: The capture root (``CAPTURE_BASE_PATH``).
            enabled: When False, ``put`` is a no-op returning the digest
                without writing, so an operator can keep hashing (cheap,
                useful for dedup) while declining the disk cost.

        Returns: None
        """
        self._root = Path(base_path) / BLOB_ROOT / BLOB_ALGORITHM_DIR
        self._enabled = enabled

    @property
    def root(self) -> Path:
        """Return the sha256 blob directory (not created by this call)."""
        return self._root

    @property
    def enabled(self) -> bool:
        """Return whether this store writes blobs.

        Read by the schema manifest so a reader can tell "the hash did not
        resolve because the write failed" apart from "the hash did not
        resolve because the store is switched off".
        """
        return self._enabled

    def path_for(self, digest: str) -> Path:
        """Return the on-disk path a digest resolves to.

        Args:
            digest: 64-character lowercase hex digest.

        Returns:
            Path: ``{root}/{digest[:2]}/{digest}``. Pure — no directory is
                created, so a caller can test existence cheaply.
        """
        return self._root / digest[:2] / digest

    def has(self, digest: str) -> bool:
        """Return True when a digest resolves to a stored blob.

        Args:
            digest: 64-character lowercase hex digest.

        Returns:
            bool: True when the blob file exists and is non-empty.
        """
        if not self._enabled or not digest:
            return False
        try:
            return self.path_for(digest).stat().st_size > 0
        except OSError:
            return False

    def put(self, data: dict[str, Any] | list[Any] | str) -> str | None:
        """Store a value's canonical bytes and return its digest.

        Args:
            data: Value whose canonical bytes are stored (typically the
                forwarded request body, or the messages array).

        Returns:
            str | None: The 64-char hex digest, or None when the store is
                disabled or the write failed. The digest is returned even
                on write failure so the caller can still record what the
                handle *would* resolve to.

        Raises: None (OSError suppressed)
        """
        if not self._enabled:
            return None
        try:
            payload = canonical_bytes(data)
        except (TypeError, ValueError):
            return None
        digest = hashlib.sha256(payload).hexdigest()
        target = self.path_for(digest)
        try:
            if target.exists() and target.stat().st_size > 0:
                return digest
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(payload)
        except OSError:
            return digest
        return digest

    def get(self, digest: str) -> bytes | None:
        """Read back a stored blob.

        Args:
            digest: 64-character lowercase hex digest.

        Returns:
            bytes | None: The stored canonical bytes, or None when the
                store is disabled or the digest does not resolve.
        """
        if not self._enabled or not digest:
            return None
        try:
            return self.path_for(digest).read_bytes()
        except OSError:
            return None
