"""Evidence acquisition for the model_headroom authority.

Makes the authority able to obtain real evidence from a real checkpoint —
weight layouts, sampled weight tiles, per-channel FFN activation
statistics, and published training-history artifacts — without the
operator handcrafting JSON input files.

Boundary: this package lives ONLY under ``src/model_headroom/``. It never
imports ``capture_proxy`` (the carrier), and it never scores — it acquires
evidence and hands it to the existing signals, which judge. Heavy
dependencies (torch, transformers, safetensors) are imported lazily inside
functions, so importing this package works without the ``[headroom]`` extra.
"""

from model_headroom.acquire.activations import (
    BranchName,
    FFNChannelStats,
    capture_ffn_channel_stats,
    dead_neuron_report_from_channel_stats,
    tokenize_prompt,
)
from model_headroom.acquire.behavioral import (
    BehavioralCondition,
    BehavioralProbeResult,
    InferenceClient,
    MockInferenceClient,
    run_behavioral_probe,
)
from model_headroom.acquire.layout import (
    WeightLayout,
    resolve_weight_layout,
    sample_layout_weights,
)
from model_headroom.acquire.optimization import (
    GradientMeasurement,
    OptimizationProbeResult,
    run_optimization_probe,
)
from model_headroom.acquire.probe_integration import (
    behavioral_result_to_evidence_bundle,
    optimization_result_to_evidence_bundles,
    probe_evidence_to_metadata,
    probe_to_evidence_bundle,
)
from model_headroom.acquire.probes import (
    ProbeBudget,
    ProbeResult,
    ProbeSpec,
    ProbeStatus,
    run_probe,
)
from model_headroom.acquire.provenance import (
    EvidenceBundle,
    EvidenceProvenance,
    ProvenanceClass,
)
from model_headroom.acquire.published import (
    HubFile,
    PublishedAcquisitionError,
    acquire_published_loss_evidence,
    download_artifact,
    extract_loss_trajectory,
    identify_training_history_artifacts,
    list_repo_tree,
)
from model_headroom.acquire.serving import (
    ServingEngineClient,
    health_check,
)

__all__ = [
    "BehavioralCondition",
    "BehavioralProbeResult",
    "BranchName",
    "EvidenceBundle",
    "EvidenceProvenance",
    "FFNChannelStats",
    "GradientMeasurement",
    "HubFile",
    "InferenceClient",
    "MockInferenceClient",
    "OptimizationProbeResult",
    "ProbeBudget",
    "ProbeResult",
    "ProbeSpec",
    "ProbeStatus",
    "PublishedAcquisitionError",
    "WeightLayout",
    "acquire_published_loss_evidence",
    "behavioral_result_to_evidence_bundle",
    "capture_ffn_channel_stats",
    "dead_neuron_report_from_channel_stats",
    "download_artifact",
    "extract_loss_trajectory",
    "health_check",
    "identify_training_history_artifacts",
    "list_repo_tree",
    "optimization_result_to_evidence_bundles",
    "probe_evidence_to_metadata",
    "probe_to_evidence_bundle",
    "resolve_weight_layout",
    "run_behavioral_probe",
    "run_optimization_probe",
    "run_probe",
    "sample_layout_weights",
    "tokenize_prompt",
]
