# Type alias restricting values to a fixed set for model config categorization.
"""Data types and model loading utilities. Defines frozen dataclasses for diagnostic signals, reports, and provenance tracking."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    import torch

ArchitectureType = Literal["moe", "dense", "unknown"]

SourceType = Literal[
    "measured", "synthetic", "fixture", "estimated", "imported", "unknown"
]

VerificationStatus = Literal["verified", "provenance_incomplete", "unverified"]

SamplingMode = Literal[
    "full_matrix", "randomized_svd", "multi_tile", "single_tile", "external", "unknown"
]

ModuleFamily = Literal[
    "ffn",
    "attention",
    "linear_attention",
    "state",
    "gating",
    "embedding",
    "lm_head",
    "vision",
    "mtp",
    "expert",
    "router",
    "other",
    "unknown",
]

ArchitectureSubtype = Literal["hybrid_linear_attention", "dense_standard", "unknown"]

AnalysisMode = Literal["diagnostic", "decision_grade"]

ResidualPosition = Literal["last", "mean"]

LayerSelection = Literal["auto", "explicit"]

RefusalAxis = Literal["mean_difference", "weight_delta_svd"]

RefusalDirectionStatus = Literal["computed", "degenerate", "insufficient_data", "error"]


@dataclass(frozen=True)
class LossDataPoint:
    """Frozen dataclass holding tokens_trained, validation_loss."""

    tokens_trained: int
    validation_loss: float


@dataclass(frozen=True)
class IsoFLOPPoint:
    """Frozen dataclass holding compute_flops, model_params, measured_loss."""

    compute_flops: float
    model_params: int
    measured_loss: float


@dataclass(frozen=True)
class ExpertActivationData:
    """Frozen dataclass holding metadata, layers."""

    metadata: dict[str, Any]
    layers: dict[str, dict[str, int]]


@dataclass(frozen=True)
class GradientWeightPair:
    """Frozen dataclass holding gradient, weight, layer_name, parent_shape, truncated, sampling_mode."""

    gradient: torch.Tensor
    weight: torch.Tensor
    layer_name: str
    parent_shape: tuple[int, int] | None = None
    truncated: bool | None = None
    sampling_mode: SamplingMode | None = None


@dataclass(frozen=True)
class WeightSample:
    weight: torch.Tensor
    layer_name: str
    parent_shape: tuple[int, int] | None = None
    truncated: bool | None = None
    sampling_mode: SamplingMode | None = None


@dataclass(frozen=True)
class MLPActivations:
    """Frozen dataclass holding activations, layer_name."""

    activations: torch.Tensor
    layer_name: str


@dataclass(frozen=True)
class ExpertWeights:
    """Frozen dataclass holding expert_weights, layer_name."""

    expert_weights: list[torch.Tensor]
    layer_name: str


@dataclass(frozen=True)
class GradientProvenance:
    """Frozen dataclass holding model_revision, dataset_hash, dataset_type, token_count, batch_count, loss_function, forward_dtype, backward_dtype, module_name, parameter_name, gradient_aggregation, seed, source_type, verification_status."""

    model_revision: str | None = None
    dataset_hash: str | None = None
    dataset_type: str | None = None
    token_count: int | None = None
    batch_count: int | None = None
    loss_function: str | None = None
    forward_dtype: str | None = None
    backward_dtype: str | None = None
    module_name: str | None = None
    parameter_name: str | None = None
    gradient_aggregation: str | None = None
    seed: int | None = None
    source_type: SourceType = "measured"
    verification_status: VerificationStatus = "verified"


@dataclass(frozen=True)
class ArchitectureDetails:
    """Frozen dataclass holding family, subtype, has_moe, has_full_attention, has_linear_attention, has_vision_tower, has_mtp, detection_source."""

    family: ArchitectureType
    subtype: ArchitectureSubtype
    has_moe: bool
    has_full_attention: bool
    has_linear_attention: bool
    has_vision_tower: bool
    has_mtp: bool
    detection_source: str


@dataclass(frozen=True)
class CoverageMetrics:
    """Frozen dataclass holding measured_score, coverage, lower_bound, upper_bound, measured_weight, total_applicable_weight, unmeasured_weight, decision_grade."""

    measured_score: float
    coverage: float
    lower_bound: float
    upper_bound: float
    measured_weight: float
    total_applicable_weight: float
    unmeasured_weight: float
    decision_grade: bool


@dataclass(frozen=True)
class SaturationReport:
    """Frozen dataclass holding is_saturated, saturation_point_tokens, loss_delta_threshold, plateau_slope, data_points_used, signal_status, error_message."""

    is_saturated: bool
    saturation_point_tokens: int | None
    loss_delta_threshold: float
    plateau_slope: float
    data_points_used: int
    signal_status: Literal["computed", "insufficient_data", "error"]
    error_message: str | None = None


@dataclass(frozen=True)
class GrowthFactorReport:
    """Frozen dataclass holding alpha, crossover_params, crossover_loss, fit_quality_r2, data_points_used, signal_status, error_message."""

    alpha: float
    crossover_params: int | None
    crossover_loss: float | None
    fit_quality_r2: float
    data_points_used: int
    signal_status: Literal["computed", "insufficient_data", "poor_fit", "error"]
    error_message: str | None = None


@dataclass(frozen=True)
class EffectiveRankResult:
    """Frozen dataclass holding rank, entropy, percentage_of_full_rank, singular_values_count, signal_status, error_message, sampling_mode, coverage, parent_shape, sample_count, confidence, state."""

    rank: float
    entropy: float
    percentage_of_full_rank: float
    singular_values_count: int
    signal_status: Literal["computed", "degenerate", "error"]
    error_message: str | None = None
    sampling_mode: SamplingMode = "unknown"
    coverage: float | None = None
    parent_shape: tuple[int, int] | None = None
    sample_count: int | None = None
    confidence: float | None = None
    state: Literal["full", "sampled"] | None = None


@dataclass(frozen=True)
class GradWeightRatio:
    """Frozen dataclass holding ratio, is_high_pressure, z_score, layer_name, signal_status, error_message."""

    ratio: float
    is_high_pressure: bool
    z_score: float
    layer_name: str
    signal_status: Literal["computed", "zero_weight", "no_gradient", "error"]
    error_message: str | None = None


@dataclass(frozen=True)
class DeadNeuronReport:
    """Frozen dataclass holding total_neurons, dead_count, dead_fraction, dead_indices, signal_status, error_message, activation_rms, mean_abs_activation, variance, near_zero_occupancy, cross_token_variance, source_type."""

    total_neurons: int
    dead_count: int
    dead_fraction: float
    dead_indices: list[int] | None
    signal_status: Literal["computed", "error"]
    error_message: str | None = None
    activation_rms: float | None = None
    mean_abs_activation: float | None = None
    variance: float | None = None
    near_zero_occupancy: float | None = None
    cross_token_variance: float | None = None
    source_type: SourceType = "measured"


@dataclass(frozen=True)
class LoadBalanceReport:
    """Frozen dataclass holding entropy_per_layer, min_entropy, max_entropy, ideal_entropy, balance_ratio, signal_status, error_message."""

    entropy_per_layer: list[float]
    min_entropy: float
    max_entropy: float
    ideal_entropy: float
    balance_ratio: float | None
    signal_status: Literal["computed", "single_expert", "expert_count_unknown", "error"]
    error_message: str | None = None


@dataclass(frozen=True)
class SpecializationReport:
    """Frozen dataclass holding mean_inter_expert_similarity, max_inter_expert_similarity, collapse_threshold, is_collapsed, signal_status, error_message."""

    mean_inter_expert_similarity: float
    max_inter_expert_similarity: float
    collapse_threshold: float
    is_collapsed: bool
    signal_status: Literal["computed", "single_expert", "error"]
    error_message: str | None = None


@dataclass(frozen=True)
class GradientEffectiveRankReport:
    """Frozen dataclass holding rank, entropy, r50, r90, r95, r99, spectral_entropy, signal_status, error_message."""

    rank: float
    entropy: float
    r50: int
    r90: int
    r95: int
    r99: int
    spectral_entropy: float
    signal_status: Literal["computed", "degenerate", "error"]
    error_message: str | None = None


@dataclass(frozen=True)
class GradientCoherenceReport:
    """Frozen dataclass holding mean_cosine_similarity, subspace_overlap, variance_spectrum, signal_status, error_message."""

    mean_cosine_similarity: float
    subspace_overlap: float
    variance_spectrum: float
    signal_status: Literal["computed", "insufficient_data", "error"]
    error_message: str | None = None


@dataclass(frozen=True)
class SignalResult:
    """Frozen dataclass holding signal_name, status, data, duration_seconds, error_message, source_type, verification_status, sampling_mode."""

    signal_name: str
    status: Literal[
        "computed",
        "skipped",
        "not_applicable",
        "single_expert",
        "expert_count_unknown",
        "error",
    ]
    data: Any
    duration_seconds: float
    error_message: str | None = None
    source_type: SourceType = "measured"
    verification_status: VerificationStatus = "verified"
    sampling_mode: SamplingMode | None = None


@dataclass(frozen=True)
class HeadroomReport:
    """Frozen dataclass holding model_name, model_path, timestamp, signals, overall_headroom_score, recommendations, metadata, coverage_metrics, analysis_mode, decision_grade."""

    model_name: str
    model_path: str
    timestamp: str
    signals: list[SignalResult]
    overall_headroom_score: float
    recommendations: list[str]
    metadata: dict[str, Any]
    coverage_metrics: CoverageMetrics | None = None
    analysis_mode: AnalysisMode = "diagnostic"
    decision_grade: bool = False


@dataclass(frozen=True)
class ResidualBatch:
    """Frozen dataclass holding activations, layer_name, position, source_type, prompt_count."""

    activations: torch.Tensor
    layer_name: str
    position: ResidualPosition = "last"
    source_type: SourceType = "measured"
    prompt_count: int | None = None


@dataclass(frozen=True)
class RefusalLayerStat:
    """Frozen dataclass holding layer_name, layer_index, harmful_count, harmless_count, cohen_d, mean_norm, masked_count, direction_norm, orthogonality_with_harmless_mean."""

    layer_name: str
    layer_index: int
    harmful_count: int
    harmless_count: int
    cohen_d: float
    mean_norm: float
    masked_count: int
    direction_norm: float
    orthogonality_with_harmless_mean: float


@dataclass(frozen=True)
class RefusalDirectionSpec:
    """Frozen dataclass holding layer_selection, layer_name, layer_index, depth_fraction, orthogonalize, massive_activation_sigma, winsorize_quantile, min_pairs."""

    layer_selection: LayerSelection = "auto"
    layer_name: str | None = None
    layer_index: int | None = None
    depth_fraction: float = 0.6
    orthogonalize: bool = False
    massive_activation_sigma: float | None = None
    winsorize_quantile: float | None = None
    min_pairs: int = 8


@dataclass(frozen=True)
class RefusalDirectionReport:
    """Frozen dataclass holding direction, axis, layer_name, layer_index, hidden_size, harmful_count, harmless_count, cohen_d, orthogonalized, massive_activation_sigma, winsorize_quantile, masked_count, layer_stats, selection, direction_status, error_message."""

    axis: RefusalAxis
    hidden_size: int
    harmful_count: int
    harmless_count: int
    layer_stats: list[RefusalLayerStat]
    direction: torch.Tensor | None = None
    layer_name: str | None = None
    layer_index: int | None = None
    cohen_d: float | None = None
    orthogonalized: bool = False
    massive_activation_sigma: float | None = None
    winsorize_quantile: float | None = None
    masked_count: int = 0
    selection: str = "none"
    direction_status: RefusalDirectionStatus = "computed"
    error_message: str | None = None
