"""Provenance types for evidence acquired from external or local sources.

Defines the authority-side provenance vocabulary for the evidence-acquisition
package. The seven-value ``ProvenanceClass`` taxonomy is the binding contract:
it separates evidence that *is* the checkpoint (``exact_checkpoint``) from
evidence about a known training ancestor (``authenticated_ancestor``), from
evidence that merely shares context (``published_context``,
``related_comparison``), from measurements made locally on the checkpoint
(``locally_measured``), from estimates (``estimated``), and from the honest
absence of evidence (``unavailable``).

``EvidenceProvenance`` is a frozen dataclass — immutable, exactly like the
other authority type contracts in :mod:`model_headroom.models.types`. It is
deliberately a *new* dataclass: existing frozen dataclasses in
``models/types.py`` are never edited.

``EvidenceBundle`` pairs a signal name, its provenance, and an opaque
``payload``. The payload is typed ``Any`` by design — each signal consumes a
different evidence shape (a list of ``LossDataPoint``, a weight matrix, a
per-channel statistic table) and the bundle must not force a single shape on
all of them. The signal that reads the bundle is responsible for validating
the payload; the bundle only asserts *where the evidence came from*.

This module is dependency-free (stdlib only) so it can be imported without the
``[headroom]`` extra.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

ProvenanceClass = Literal[
    "exact_checkpoint",
    "authenticated_ancestor",
    "related_comparison",
    "published_context",
    "locally_measured",
    "estimated",
    "unavailable",
]


@dataclass(frozen=True)
class EvidenceProvenance:
    """Provenance record for one piece of acquired evidence.

    Attributes:
        provenance_class: The seven-value taxonomy class. ``unavailable`` is
            the honest answer when no evidence could be obtained — it is a
            valid, non-fabricated state, not an error to hide.
        source_url: URL the evidence was read from, when it came from a
            remote source (e.g. a Hugging Face Hub resolve URL).
        source_revision: The repo revision/commit the source was pinned to.
        source_sha256: SHA-256 of the exact artifact bytes, when recorded.
        acquisition_method: How the evidence was obtained, e.g.
            ``"local_safetensors"``, ``"hub_metadata_tree"``,
            ``"transformers_forward_hook"``.
        artifact_path: Local path or in-repo path of the artifact, when any.
        retrieved_at: ISO-8601 UTC timestamp of when the evidence was read.
        applicability: Human-readable statement of what this evidence DOES and
            DOES NOT establish. A training loss is not a validation loss; this
            field is where that distinction is said out loud.
        limitations: Exact, machine-readable list of the limits of this
            evidence (e.g. which quantity was NOT available).
        assumptions: The assumptions the acquisition made (e.g. "token count
            derived as steps*batch*max_len because tokens were not logged").
    """

    provenance_class: ProvenanceClass
    source_url: str | None = None
    source_revision: str | None = None
    source_sha256: str | None = None
    acquisition_method: str = ""
    artifact_path: str | None = None
    retrieved_at: str | None = None
    applicability: str = ""
    limitations: tuple[str, ...] = ()
    assumptions: tuple[str, ...] = ()


@dataclass(frozen=True)
class EvidenceBundle:
    """A unit of acquired evidence: signal name + provenance + payload.

    Attributes:
        signal_name: The headroom signal this evidence feeds (e.g.
            ``"loss_saturation"`` for R-01).
        provenance: Where the evidence came from and what it does/does not
            establish.
        payload: The evidence itself, in whatever shape the target signal
            consumes. Typed ``Any`` because the payload shape is
            signal-specific; the consuming signal validates it. ``None`` when
            ``provenance.provenance_class`` is ``"unavailable"``.
    """

    signal_name: str
    provenance: EvidenceProvenance
    payload: Any
