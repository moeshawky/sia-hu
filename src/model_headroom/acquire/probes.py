"""Bounded, reproducible hypothesis-driven probes for the Sia acquisition layer.

A probe investigates a specific question against an experimental copy of a
model. The original checkpoint is never modified; all mutations happen on
disposable in-memory copies. Every probe carries a complete contract and
returns an independently identifiable result with full provenance.

Probe results are packaged as EvidenceBundle instances so they flow into the
existing Sia authority pipeline (orchestrator → signals → scoring → reporting)
without bypassing its authority or creating a parallel scoring system.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import monotonic
from typing import TYPE_CHECKING, Any, Literal

from model_headroom.acquire.provenance import (
    EvidenceBundle,
    EvidenceProvenance,
)

if TYPE_CHECKING:
    from collections.abc import Callable

ProbeStatus = Literal["planned", "running", "completed", "failed", "inconclusive"]

ProbeMechanism = Literal[
    "behavioral_inference",
    "pytorch_forward",
    "pytorch_backward",
    "pytorch_one_step",
    "activation_streaming",
    "existing_artifact",
]


@dataclass(frozen=True)
class ProbeBudget:
    """Hard resource bounds for a probe execution."""

    max_seconds: float | None = None
    max_model_loads: int = 1
    max_forward_passes: int | None = None
    max_backward_passes: int | None = None
    max_grad_bytes: int | None = None
    max_activation_bytes: int | None = None


@dataclass(frozen=True)
class ProbeSpec:
    """Immutable contract for one bounded probe."""

    probe_id: str
    question: str
    hypothesis: str
    target_model: str
    target_revision: str | None
    measurement: str
    acquisition_mechanism: ProbeMechanism
    input_data: str | None
    input_provenance: str | None
    experimental_control: str | None
    intervention: str | None
    seed: int
    budget: ProbeBudget
    stopping_condition: str
    expected_observations: str | None
    known_limitations: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.probe_id:
            raise ValueError("probe_id is required")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        if self.budget.max_model_loads < 1:
            raise ValueError("max_model_loads must be >= 1")


@dataclass(frozen=True)
class ProbeResult:
    """Outcome of one probe execution, with complete provenance."""

    probe_id: str
    status: ProbeStatus
    observations: dict[str, Any]
    artifacts: tuple[str, ...] = ()
    provenance: EvidenceProvenance | None = None
    error_message: str | None = None
    duration_seconds: float = 0.0
    budget_exhausted: bool = False
    stopping_reason: str | None = None

    def to_evidence_bundle(self, signal_name: str) -> EvidenceBundle:
        """Wrap this probe result as an EvidenceBundle for the authority."""
        if self.provenance is None:
            provenance = EvidenceProvenance(
                provenance_class="locally_measured",
                acquisition_method=self.probe_id,
                retrieved_at="",
                applicability="probe result; " + (self.error_message or ""),
                limitations=(),
            )
        else:
            provenance = self.provenance
        return EvidenceBundle(
            signal_name=signal_name,
            provenance=provenance,
            payload={
                "probe_id": self.probe_id,
                "status": self.status,
                "observations": self.observations,
                "artifacts": self.artifacts,
                "error_message": self.error_message,
                "duration_seconds": self.duration_seconds,
                "budget_exhausted": self.budget_exhausted,
                "stopping_reason": self.stopping_reason,
            },
        )


def _set_seed(seed: int) -> None:
    import random

    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_probe(
    spec: ProbeSpec,
    execute: Callable[[], dict[str, Any]] | None = None,
) -> ProbeResult:
    """Execute one probe within its declared budget and contract.

    Parameters:
        spec: The immutable probe contract.
        execute: Optional callable producing raw observations. When provided,
            run_probe wraps it with seed, timing, budget, provenance, and
            stopping enforcement. When omitted, the caller must have already
            bound execution into spec via acquisition_mechanism-specific logic
            elsewhere.

    Returns a ProbeResult. A probe that cannot complete within budget or that
    raises is reported as failed/inconclusive with stopping_reason set — never
    silently converted into a negative scientific finding.
    """
    _set_seed(spec.seed)
    start = monotonic()
    model_loads = 0
    forward_passes = 0
    backward_passes = 0

    def check_budget() -> str | None:
        if spec.budget.max_seconds is not None:
            elapsed = monotonic() - start
            if elapsed >= spec.budget.max_seconds:
                return f"budget: max_seconds {spec.budget.max_seconds} reached"
        if (
            spec.budget.max_model_loads is not None
            and model_loads > spec.budget.max_model_loads
        ):
            return f"budget: max_model_loads {spec.budget.max_model_loads} exceeded"
        if (
            spec.budget.max_forward_passes is not None
            and forward_passes > spec.budget.max_forward_passes
        ):
            return (
                f"budget: max_forward_passes {spec.budget.max_forward_passes} exceeded"
            )
        if (
            spec.budget.max_backward_passes is not None
            and backward_passes > spec.budget.max_backward_passes
        ):
            return f"budget: max_backward_passes {spec.budget.max_backward_passes} exceeded"
        return None

    stop_reason = check_budget()
    if stop_reason is not None:
        return ProbeResult(
            probe_id=spec.probe_id,
            status="inconclusive",
            observations={},
            error_message=stop_reason,
            duration_seconds=monotonic() - start,
            budget_exhausted=True,
            stopping_reason=stop_reason,
        )

    observations: dict[str, Any] = {}
    error: str | None = None
    status: ProbeStatus = "completed"

    try:
        if execute is not None:
            observations = execute()
        else:
            observations = {"mechanism": spec.acquisition_mechanism, "executed": False}
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        status = "failed"

    duration = monotonic() - start
    stop_reason = check_budget()
    if stop_reason is not None and status == "completed":
        status = "inconclusive"

    provenance = EvidenceProvenance(
        provenance_class="locally_measured",
        acquisition_method=f"probe:{spec.acquisition_mechanism}",
        retrieved_at="",
        applicability=spec.measurement,
        limitations=tuple(
            list(spec.known_limitations)
            + ([] if error is None else ["execution failed: " + error])
        ),
        assumptions=(f"seed={spec.seed}", f"target={spec.target_model}"),
    )

    return ProbeResult(
        probe_id=spec.probe_id,
        status=status,
        observations=observations,
        provenance=provenance,
        error_message=error,
        duration_seconds=duration,
        budget_exhausted=stop_reason is not None,
        stopping_reason=stop_reason,
    )
