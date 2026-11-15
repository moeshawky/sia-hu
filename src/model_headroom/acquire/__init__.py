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
from model_headroom.acquire.layout import (
    WeightLayout,
    resolve_weight_layout,
    sample_layout_weights,
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

__all__ = [
    "BranchName",
    "EvidenceBundle",
    "EvidenceProvenance",
    "FFNChannelStats",
    "HubFile",
    "ProvenanceClass",
    "PublishedAcquisitionError",
    "WeightLayout",
    "acquire_published_loss_evidence",
    "capture_ffn_channel_stats",
    "dead_neuron_report_from_channel_stats",
    "download_artifact",
    "extract_loss_trajectory",
    "identify_training_history_artifacts",
    "list_repo_tree",
    "resolve_weight_layout",
    "sample_layout_weights",
    "tokenize_prompt",
]
