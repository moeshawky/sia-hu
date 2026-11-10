# Phase 2 — Effective Rank Audit

## What effective rank actually establishes

### Current implementation (signals/effective_rank.py)

- **Definition**: Effective rank = exp(spectral_entropy) where spectral_entropy = -Σ p_i log(p_i), p_i = σ_i / Σσ_j. This is the standard "effective rank" (Roy & Vetterli 2007) measuring spectral occupancy.
- **Normalization**: `percentage_of_full_rank = rank / min(rows, cols)` — tile-relative, never parent-level.
- **SVD**: `torch.linalg.svd(weight, full_matrices=False)` — economy SVD. Now casts to float32 before SVD (fixed).
- **Sampling**: Sidecar byte cap truncates rows. Coverage = tile_elements / parent_elements. State = "full" if coverage >= 1.0 else "sampled".
- **Confidence**: Heuristic by sampling_mode (single_tile=0.3, multi_tile=0.3+0.1*n, full_matrix=0.95, randomized_svd=0.8). Not calibrated.

### Critical examination: does low effective rank indicate spare capacity?

**NO.** Spectral occupancy measures how concentrated the singular values are, not trainability. A weight matrix with low effective rank may be:
1. **Efficiently compressed** — the model learned a compact representation (good, not underutilized).
2. **Underutilized** — there is room for additional structure (possible capacity).
3. **Quantization artifact** — dequantized weights may have artificially low rank due to quantization grid effects.
4. **Architecturally constrained** — some layers are inherently low-rank (e.g., bottleneck projections).

The signal measures spectral structure of existing weights. It does NOT establish:
- Local sensitivity to parameter changes (gradient accessibility)
- Expressivity of a proposed adapter
- Optimization stability
- Behavioral improvement after intervention
- Retention of previously learned capabilities

### Fixes applied
1. NaN/Inf weights → error status (was: NaN propagation through SVD)
2. Zero-check uses `max()` instead of `sum()` (more robust for tiny values)
3. SVD casts to float32 before computation (numerical stability)
4. Recommendation threshold 0.3 documented as heuristic (uncalibrated)

### Known limitations (not fixed — require operator decision or new data)
- No randomized SVD implementation (randomized_svd is an enum value but code always uses full SVD)
- Recommendation threshold (0.3) is uncalibrated
- Tied weights/embeddings analyzed multiple times (no dedup)
- Quantization impact on SVD not measured
- Confidence scores are heuristic, not calibrated
- No complementary metrics (gradient covariance, stable rank, participation ratio)

### Complementary metrics to investigate (Phase 8 research)
- Stable rank: ||W||_F^2 / ||W||_2^2 (cheaper than effective rank, less sensitive to small SVs)
- Participation ratio: (Σσ_i^2)^2 / Σσ_i^4 (measures how many SVs contribute)
- Gradient covariance alignment (requires gradients, more expensive)
