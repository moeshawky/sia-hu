# Phase 8 — Research Findings

## Adopted techniques

### 1. Effective rank (Roy & Vetterli 2007)
- **Source**: "The effective rank: a measure of complexity for matrix-valued random variables" (Roy & Vetterli, IEEE TSP 2007)
- **What it establishes**: Spectral occupancy — how concentrated the singular values are.
- **Applicability**: Universal (any weight matrix). Tile-relative percentage only.
- **Assumptions**: SVD is numerically stable; matrix is representative of the full layer.
- **Complexity**: O(min(m,n)^2 * max(m,n)) for full SVD. Expensive for large matrices.
- **Integration cost**: Already implemented. Fixed NaN handling and SVD precision.
- **Failure modes**: NaN propagation, quantization artifacts, tied weights counted multiple times.
- **Decision improvement**: None alone. Spectral occupancy ≠ trainability. Used as one signal among many.

### 2. Difference-of-means refusal direction (Arditi et al.)
- **Source**: "Refusal in Language Models Is Mediated by a Single Direction" (Arditi et al., 2024)
- **What it establishes**: A rank-1 approximation of the refusal direction in residual stream.
- **Applicability**: Models with linear residual streams. Requires contrastive prompt sets.
- **Assumptions**: Refusal is mediated by a single direction; harmful/harmless sets are representative.
- **Complexity**: O(n_prompts * seq_len * hidden * n_layers) for capture + O(hidden^2 * n_layers) for SVD.
- **Integration cost**: Already implemented as standalone capability.
- **Failure modes**: Small sample sizes, massive activations, non-representative prompts.
- **Decision improvement**: Provides a measurable intervention target. Requires behavioral validation.

### 3. Load balance entropy (MoE routing)
- **Source**: Standard information theory (Shannon entropy of expert routing distribution).
- **What it establishes**: How evenly tokens are distributed across experts.
- **Applicability**: MoE models only. Dense models: not_applicable.
- **Assumptions**: Router entropy reflects load balance. Architectural expert count is the denominator.
- **Complexity**: O(n_tokens * n_experts) for histogram + O(n_experts) for entropy.
- **Integration cost**: Already implemented.
- **Failure modes**: Collapsed routers (all tokens to 1 expert) score low but are "computed" not "single_expert".
- **Decision improvement**: Identifies routing collapse. Does not predict training outcomes.

## Rejected / deferred candidates

### 1. Randomized SVD (Halko et al. 2011)
- **Source**: "Finding structure with randomness" (Halko, Martinsson, Tropp 2011)
- **What it establishes**: Approximate singular values/vectors in O(mn log k) instead of O(mn^2).
- **Why deferred**: Effective rank needs ALL singular values for entropy computation. Randomized SVD approximates top-k, which biases entropy. Would need full SVD or a different metric (stable rank).
- **When to revisit**: If stable rank (||W||_F^2/||W||_2^2) is adopted as a complementary metric.

### 2. Gradient covariance / Fisher information
- **Source**: Standard (Kirkpatrick et al. 2017, EWC)
- **What it establishes**: Parameter importance for continual learning.
- **Why deferred**: Requires gradient computation (expensive, needs training data). Not available from static weights.
- **When to revisit**: Phase 4 (training feasibility) if runtime probes are implemented.

### 3. Activation-aware compression (AWQ, SmoothQuant)
- **Source**: AWQ (Lin et al. 2023), SmoothQuant (Xiao et al. 2023)
- **What it establishes**: Which channels are salient for quantization.
- **Why deferred**: Requires activation data (available from capture-proxy sidecar). Could be integrated as a signal but needs producer (capture-proxy routing histogram).
- **When to revisit**: When CAPTURE_ROUTED_EXPERTS sidecar is operational.

### 4. Intrinsic dimensionality (Aghajanyan et al. 2020)
- **Source**: "Intrinsic Dimensionality Explains the Effectiveness of Language Model Fine-Tuning" (Aghajanyan et al. 2020)
- **What it establishes**: Fine-tuning operates in a low-dimensional subspace.
- **Why deferred**: Requires training experiments to measure. Not a static diagnostic.
- **When to revisit**: Phase 6 (intervention validation) with controlled experiments.

## Underused techniques worth investigating

1. **Stable rank** (||W||_F^2 / ||W||_2^2): Cheaper than effective rank (no SVD needed, just Frobenius and spectral norms). Less sensitive to small singular values. Could replace effective rank as the default.

2. **Participation ratio** ((Σσ_i²)² / Σσ_i⁴): Measures how many singular values contribute significantly. Complementary to effective rank.

3. **Layer-wise sensitivity to quantization**: Compare fp16 vs quantized SVD. Could detect quantization-degraded layers.

4. **Gradient subspace alignment**: Measure angle between gradient subspace and weight subspace. Requires gradients (Phase 4).

## No universal thresholds

Per mission constraints: no numerical threshold is introduced without calibration or explicit documentation that it is heuristic. The existing 0.3 effective-rank threshold is documented as heuristic (uncalibrated).
