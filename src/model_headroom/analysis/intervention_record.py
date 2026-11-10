"""Intervention validation record — tracks baseline→intervention→post-measurement.

Supports measurement and validation of external interventions (LoRA, QLoRA,
quantization recovery, weight edits) without implementing training itself.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


@dataclass
class InterventionRecord:
    """Record of a single intervention experiment."""

    intervention_id: str
    intervention_type: str  # lora, qlora, quantization_recovery, weight_edit, etc.
    checkpoint_hash: str
    architecture: str
    runtime_versions: dict[str, str]
    affected_parameters: list[str]
    adapter_config: dict[str, Any] | None = None
    measurement_methods: list[str] = field(default_factory=list)
    training_data_provenance: str | None = None
    compute_budget: str | None = None
    memory_budget_gb: float | None = None
    baseline_measurements: dict[str, Any] = field(default_factory=dict)
    post_measurements: dict[str, Any] = field(default_factory=dict)
    behavioral_results: dict[str, Any] = field(default_factory=dict)
    failure_cases: list[str] = field(default_factory=list)
    rollback_path: str | None = None
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    status: str = "planned"  # planned, running, completed, failed, rolled_back

    def compute_delta(self) -> dict[str, Any]:
        """Compute before/after deltas for all measured signals."""
        deltas: dict[str, Any] = {}
        for key in self.baseline_measurements:
            if key in self.post_measurements:
                base = self.baseline_measurements[key]
                post = self.post_measurements[key]
                if isinstance(base, (int, float)) and isinstance(post, (int, float)):
                    deltas[key] = post - base
        return deltas

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, default=str)

    @classmethod
    def from_json(cls, data: str) -> InterventionRecord:
        return cls(**json.loads(data))

    def save(self, path: str | Path) -> None:
        Path(path).write_text(self.to_json())

    @classmethod
    def load(cls, path: str | Path) -> InterventionRecord:
        return cls.from_json(Path(path).read_text())


def hash_checkpoint(path: str | Path) -> str:
    """Compute SHA-256 hash of a checkpoint directory or file."""
    p = Path(path)
    if p.is_file():
        h = hashlib.sha256()
        with p.open("rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()
    if p.is_dir():
        h = hashlib.sha256()
        for child in sorted(p.rglob("*")):
            if child.is_file():
                h.update(child.name.encode())
                with child.open("rb") as fh:
                    for chunk in iter(lambda: fh.read(8192), b""):
                        h.update(chunk)
        return h.hexdigest()
    return ""
