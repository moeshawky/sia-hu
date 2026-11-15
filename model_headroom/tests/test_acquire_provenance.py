"""Tests for model_headroom.acquire.provenance."""

from __future__ import annotations

import dataclasses

from model_headroom.acquire.provenance import (
    EvidenceBundle,
    EvidenceProvenance,
    ProvenanceClass,
)

EXPECTED_CLASSES = {
    "exact_checkpoint",
    "authenticated_ancestor",
    "related_comparison",
    "published_context",
    "locally_measured",
    "estimated",
    "unavailable",
}


def test_provenance_class_has_exactly_seven_mission_values() -> None:
    assert set(ProvenanceClass.__args__) == EXPECTED_CLASSES
    assert len(ProvenanceClass.__args__) == 7


def test_evidence_provenance_defaults() -> None:
    p = EvidenceProvenance(provenance_class="unavailable")
    assert p.source_url is None
    assert p.source_revision is None
    assert p.source_sha256 is None
    assert p.acquisition_method == ""
    assert p.artifact_path is None
    assert p.retrieved_at is None
    assert p.applicability == ""
    assert p.limitations == ()
    assert p.assumptions == ()


def test_evidence_provenance_is_frozen() -> None:
    p = EvidenceProvenance(provenance_class="published_context", source_url="u")
    try:
        p.source_url = "other"
        raise AssertionError("mutation of a frozen dataclass must raise")
    except dataclasses.FrozenInstanceError:
        pass


def test_evidence_provenance_full() -> None:
    p = EvidenceProvenance(
        provenance_class="authenticated_ancestor",
        source_url="https://example.co/datasets/x/resolve/main/run_record.json",
        source_revision="abc123",
        source_sha256="0" * 64,
        acquisition_method="hub_metadata_tree",
        artifact_path="training_runs/control_off/run_record.json",
        retrieved_at="2026-11-15T00:00:00+00:00",
        applicability="what it does and does not establish",
        limitations=("limit one", "limit two"),
        assumptions=("assumption one",),
    )
    assert p.limitations == ("limit one", "limit two")
    assert p.assumptions == ("assumption one",)
    assert p.provenance_class == "authenticated_ancestor"


def test_evidence_bundle_is_frozen_and_carries_payload() -> None:
    p = EvidenceProvenance(
        provenance_class="locally_measured",
        acquisition_method="local_safetensors",
    )
    payload = [1, 2, 3]
    b = EvidenceBundle(signal_name="effective_rank", provenance=p, payload=payload)
    assert b.signal_name == "effective_rank"
    assert b.payload == [1, 2, 3]
    assert b.provenance is p
    try:
        b.signal_name = "other"
        raise AssertionError("mutation of a frozen dataclass must raise")
    except dataclasses.FrozenInstanceError:
        pass


def test_unavailable_is_a_valid_class_not_an_error() -> None:
    b = EvidenceBundle(
        signal_name="loss_saturation",
        provenance=EvidenceProvenance(
            provenance_class="unavailable",
            limitations=("network unreachable",),
        ),
        payload=None,
    )
    assert b.provenance.provenance_class == "unavailable"
    assert b.payload is None
