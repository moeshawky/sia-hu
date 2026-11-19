"""Integration of probe results into the Sia authority pipeline.

Probe results are packaged as EvidenceBundle instances (provenance_class
"locally_measured", acquisition_method "probe:<mechanism>") and carried into
run_analysis via report metadata. This does not bypass the authority: the
orchestrator, signals, and scoring interpret the evidence; probes only supply
it. No frozen dataclass is mutated.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from model_headroom.acquire.provenance import EvidenceBundle, EvidenceProvenance

if TYPE_CHECKING:
    from model_headroom.acquire.behavioral import BehavioralProbeResult
    from model_headroom.acquire.optimization import OptimizationProbeResult
    from model_headroom.acquire.probes import ProbeResult


def probe_to_evidence_bundle(
    probe_result: ProbeResult,
    signal_name: str,
) -> EvidenceBundle:
    """Wrap any ProbeResult as an EvidenceBundle for the authority pipeline.

    The bundle's payload carries the full probe result (status, observations,
    provenance, error, budget exhaustion). The consuming signal decides how to
    interpret it; an inconclusive or failed probe remains distinguishable from
    a computed measurement.
    """
    prov = probe_result.provenance or EvidenceProvenance(
        provenance_class="locally_measured",
        acquisition_method=f"probe:{probe_result.probe_id}",
        retrieved_at="",
        applicability=signal_name,
        limitations=(),
    )
    return EvidenceBundle(
        signal_name=signal_name,
        provenance=prov,
        payload={
            "probe_id": probe_result.probe_id,
            "status": probe_result.status,
            "observations": probe_result.observations,
            "artifacts": probe_result.artifacts,
            "error_message": probe_result.error_message,
            "duration_seconds": probe_result.duration_seconds,
            "budget_exhausted": probe_result.budget_exhausted,
            "stopping_reason": probe_result.stopping_reason,
        },
    )


def optimization_result_to_evidence_bundles(
    result: OptimizationProbeResult,
) -> list[EvidenceBundle]:
    """Convert an optimization probe result into signal-scoped EvidenceBundles.

    Gradient measurements feed R-04 (grad_weight_ratio) as computed evidence
    when gradients are present; absent gradients are reported as unavailable
    evidence, never fabricated as zero.
    """
    bundles: list[EvidenceBundle] = []
    grad_bundle = probe_to_evidence_bundle(result.probe_result, "grad_weight_ratio")
    bundles.append(grad_bundle)
    if result.component_attribution:
        comp_bundle = probe_to_evidence_bundle(
            result.probe_result, "component_gradient_attribution"
        )
        bundles.append(comp_bundle)
    return bundles


def behavioral_result_to_evidence_bundle(
    result: BehavioralProbeResult,
) -> EvidenceBundle:
    """Convert a behavioral probe result into an EvidenceBundle."""
    return probe_to_evidence_bundle(result.probe_result, "behavioral_consistency")


def probe_evidence_to_metadata(
    bundles: list[EvidenceBundle],
) -> dict[str, Any]:
    """Flatten EvidenceBundles into a metadata dict for run_analysis.

    Each bundle is stored under "probe_evidence:<signal_name>" as a serializable
    dict (provenance fields + payload). No torch tensors are included, so the
    metadata remains JSON-serializable.
    """
    metadata: dict[str, Any] = {}
    for bundle in bundles:
        key = f"probe_evidence:{bundle.signal_name}"
        metadata[key] = {
            "signal_name": bundle.signal_name,
            "provenance_class": bundle.provenance.provenance_class,
            "acquisition_method": bundle.provenance.acquisition_method,
            "applicability": bundle.provenance.applicability,
            "limitations": list(bundle.provenance.limitations),
            "assumptions": list(bundle.provenance.assumptions),
            "payload": bundle.payload,
        }
    return metadata
