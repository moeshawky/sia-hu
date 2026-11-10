"""Training feasibility estimation from model and training configuration.

Provides bounded resource estimates accounting for weight storage, gradients,
optimizer states, activations, and framework overhead. Distinguishes inference
from training parallelism. Returns lower-bound estimates with uncertainty ranges.

No GPU execution required — static analysis only.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class TrainingFeasibilityEstimate:
    """Bounded resource estimate for a training configuration."""

    model_params: int
    trainable_params: int
    weight_bytes: int
    gradient_bytes: int
    optimizer_state_bytes: int
    activation_bytes_estimate: int
    total_training_bytes: int
    total_training_gb: float
    lower_bound_gb: float
    upper_bound_gb: float
    assumptions: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


_BYTES_PER_PARAM = {
    "fp32": 4,
    "fp16": 2,
    "bf16": 2,
    "int8": 1,
    "int4": 0.5,
}

_OPTIMIZER_MULTIPLIER = {
    "adam": 2,  # m + v
    "adamw": 2,
    "sgd": 1,  # momentum
    "adafactor": 1,  # row/col factors (approx)
    "none": 0,
}


def estimate_training_feasibility(
    model_params: int,
    trainable_params: int | None = None,
    weight_dtype: str = "bf16",
    optimizer: str = "adamw",
    sequence_length: int = 2048,
    microbatch_size: int = 1,
    gradient_accumulation: int = 1,
    activation_checkpointing: bool = False,
    attention_impl: str = "flash_attention_2",
    num_layers: int = 32,
    hidden_size: int = 4096,
    num_attention_heads: int = 32,
    framework_overhead_factor: float = 1.3,
) -> TrainingFeasibilityEstimate:
    """Estimate training memory requirements.

    Accounts for:
    - Weight storage (quantized or full precision)
    - Gradients (trainable params only)
    - Optimizer states (adam: 2x trainable params)
    - Activations (sequence_length * hidden_size * layers * batch)
    - Framework/CUDA overhead (1.3x multiplier)

    Returns lower-bound and upper-bound estimates with stated assumptions.
    """
    if trainable_params is None:
        trainable_params = model_params

    assumptions: list[str] = []
    warnings: list[str] = []

    bytes_per_weight = _BYTES_PER_PARAM.get(weight_dtype, 4)
    opt_mult = _OPTIMIZER_MULTIPLIER.get(optimizer, 2)

    # Weight storage
    weight_bytes = int(model_params * bytes_per_weight)

    # Gradients (trainable params, fp32 for stability)
    gradient_bytes = int(trainable_params * 4)

    # Optimizer states
    optimizer_state_bytes = int(trainable_params * 4 * opt_mult)

    # Activations estimate
    # Per layer: seq_len * hidden * (1 for attention + 1 for ffn) * batch
    # With activation checkpointing: sqrt(layers) factor
    tokens_per_step = sequence_length * microbatch_size
    per_layer_act = tokens_per_step * hidden_size * 2  # attention + ffn
    if activation_checkpointing:
        act_layers = int(num_layers**0.5) + 1
        assumptions.append("Activation checkpointing: sqrt(layers) activation storage")
    else:
        act_layers = num_layers
    activation_bytes = int(per_layer_act * act_layers * 4)  # fp32

    # Attention-specific: QKV projections
    attn_bytes = int(tokens_per_step * hidden_size * num_attention_heads * 2 * 4)
    activation_bytes += attn_bytes

    # Total
    subtotal = weight_bytes + gradient_bytes + optimizer_state_bytes + activation_bytes
    total = int(subtotal * framework_overhead_factor)

    total_gb = total / (1024**3)
    lower_bound_gb = subtotal / (1024**3)
    upper_bound_gb = total_gb * 1.2  # 20% uncertainty

    assumptions.extend(
        [
            f"Weight dtype: {weight_dtype} ({bytes_per_weight} bytes/param)",
            f"Optimizer: {optimizer} (state multiplier: {opt_mult}x)",
            f"Framework overhead factor: {framework_overhead_factor}x",
            f"Attention implementation: {attention_impl}",
            "Activation estimate is approximate (depends on model architecture)",
            "Does not account for: data loading buffers, communication buffers, "
            "CUDA context, or driver overhead",
        ]
    )

    if weight_dtype in ("int4", "int8") and optimizer != "none":
        warnings.append(
            "Quantized weights with full-precision optimizer states: "
            "master weights may be needed, increasing memory."
        )
    if not activation_checkpointing and num_layers > 24:
        warnings.append(
            "Deep model without activation checkpointing: activation memory "
            "may dominate. Consider enabling checkpointing."
        )
    if microbatch_size > 1 and sequence_length > 4096:
        warnings.append(
            "Large microbatch + long sequence: activation memory scales "
            "as O(batch * seq_len * hidden * layers)."
        )

    return TrainingFeasibilityEstimate(
        model_params=model_params,
        trainable_params=trainable_params,
        weight_bytes=weight_bytes,
        gradient_bytes=gradient_bytes,
        optimizer_state_bytes=optimizer_state_bytes,
        activation_bytes_estimate=activation_bytes,
        total_training_bytes=total,
        total_training_gb=round(total_gb, 2),
        lower_bound_gb=round(lower_bound_gb, 2),
        upper_bound_gb=round(upper_bound_gb, 2),
        assumptions=assumptions,
        warnings=warnings,
    )
