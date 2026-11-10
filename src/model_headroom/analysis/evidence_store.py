"""Durable evidence store — persists measurement artifacts with hashes and metadata.

Ensures diagnostic reports remain independently auditable. Records artifact
hashes, software versions, sampling parameters, and numerical precision.
"""

from __future__ import annotations

import hashlib
import json
import platform
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast


@dataclass
class EvidenceArtifact:
    """A persisted measurement artifact with provenance."""

    artifact_id: str
    artifact_type: str  # headroom_report, effective_rank, intervention_record, etc.
    content_hash: str
    created_at: str
    software_version: str
    python_version: str
    parameters: dict[str, Any] = field(default_factory=dict)
    file_path: str | None = None
    parent_artifact_id: str | None = None


class EvidenceStore:
    """File-based evidence store with hashing and versioning."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._index_path = self.root / "index.json"
        self._index: list[dict[str, Any]] = self._load_index()

    def _load_index(self) -> list[dict[str, Any]]:
        if self._index_path.exists():
            try:
                return cast(
                    "list[dict[str, Any]]", json.loads(self._index_path.read_text())
                )
            except (json.JSONDecodeError, OSError):
                return []
        return []

    def _save_index(self) -> None:
        self._index_path.write_text(json.dumps(self._index, indent=2))

    def store(
        self,
        artifact_type: str,
        content: Any,
        parameters: dict[str, Any] | None = None,
        parent_id: str | None = None,
    ) -> EvidenceArtifact:
        """Store an artifact and return its metadata."""
        content_str = json.dumps(content, sort_keys=True, default=str)
        content_hash = hashlib.sha256(content_str.encode()).hexdigest()

        artifact_id = f"{artifact_type}_{content_hash[:12]}"
        timestamp = datetime.now(UTC).isoformat()

        # Write content file
        content_path = self.root / f"{artifact_id}.json"
        content_path.write_text(json.dumps(content, indent=2, default=str))

        artifact = EvidenceArtifact(
            artifact_id=artifact_id,
            artifact_type=artifact_type,
            content_hash=content_hash,
            created_at=timestamp,
            software_version="0.3.0",
            python_version=platform.python_version(),
            parameters=parameters or {},
            file_path=str(content_path),
            parent_artifact_id=parent_id,
        )

        self._index.append(asdict(artifact))
        self._save_index()
        return artifact

    def retrieve(self, artifact_id: str) -> dict[str, Any] | None:
        """Retrieve an artifact by ID."""
        path = self.root / f"{artifact_id}.json"
        if path.exists():
            try:
                return cast("dict[str, Any]", json.loads(path.read_text()))
            except (json.JSONDecodeError, OSError):
                return None
        return None

    def list_artifacts(self, artifact_type: str | None = None) -> list[dict[str, Any]]:
        """List stored artifacts, optionally filtered by type."""
        if artifact_type:
            return [a for a in self._index if a["artifact_type"] == artifact_type]
        return list(self._index)

    def verify(self, artifact_id: str) -> dict[str, Any]:
        """Verify an artifact's integrity (hash match)."""
        content = self.retrieve(artifact_id)
        if content is None:
            return {"verified": False, "reason": "artifact not found"}
        content_str = json.dumps(content, sort_keys=True, default=str)
        actual_hash = hashlib.sha256(content_str.encode()).hexdigest()
        # Find recorded hash
        recorded = None
        for a in self._index:
            if a["artifact_id"] == artifact_id:
                recorded = a["content_hash"]
                break
        if recorded is None:
            return {"verified": False, "reason": "not in index"}
        if actual_hash == recorded:
            return {"verified": True, "hash": actual_hash}
        return {
            "verified": False,
            "reason": "hash mismatch",
            "recorded": recorded,
            "actual": actual_hash,
        }
