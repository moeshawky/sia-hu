# Headroom signals — R-01 through R-09

**Purpose:** Reference for every headroom signal: what each measures, its scope
(universal vs MoE-only vs diagnostic), how it is computed, and where it lives in the
code. Cross-references `architecture/authority-carrier.md` (signal ownership) and
`concepts/frozen-contracts.md` (frozen dataclass contracts). No Python code changes
are proposed here.

---

## Signal overview

| ID | Name | Scope | Measures | Computed by |
|----|------|-------|----------|-------------|
| R-01 | Loss Saturation | Universal | Training loss plateau detection | `signals/loss_saturation.py:analyze_loss_saturation()` |
| R-02 | Growth Factor | Universal | Quadratic loss fit, crossover detection | `signals/growth_factor.py:analyze_growth_factor()` |
| R-03 | Effective Rank | Universal | SVD-based spectral entropy of weight matrices | `signals/effective_rank.py:compute_effective_rank()` |
| R-04 | Gradient/Weight Ratio | Universal | Norm ratio, z-score computation | `signals/grad_weight_ratio.py:analyze_grad_weight_ratios()` |
| R-05 | Dead Neurons | Universal | Near-zero activation channel detection | `signals/dead_neurons.py:analyze_dead_neurons()` |
| R-06 | Load Balance | MoE only | Router entropy / balance ratio | `signals/load_balance.py:analyze_load_balance()` |
| R-07 | Specialization | MoE only | Pairwise expert cosine similarity | `signals/specialization.py:analyze_specialization()` |
| R-08 | Gradient Coherence | Diagnostic (opt-in) | Cross-batch gradient cosine similarity | `signals/gradient_coherence.py:compute_gradient_coherence()` |
| R-09 | Gradient Effective Rank | Diagnostic (opt-in) | SVD rank analysis of gradient matrices | `signals/gradient_effective_rank.py:compute_gradient_effective_rank()` |

**7 core signals** always run. **2 diagnostic signals** (R-08, R-09) run only with
`--gradient-diagnostics` and never affect the headroom composite score.

---

## R-01 — Loss Saturation

**Scope:** Universal (requires training dynamics data: `tokens_trained` +
`validation_loss` pairs).

**What it measures:** Whether the training loss curve has plateaued — slope
computation and plateau detection over a loss curve. Handles insufficient data
gracefully.

**Data source:** `--loss-curve loss.json` (LossDataPoint: `{tokens_trained: int,
validation_loss: float}`). Inference traffic carries neither `tokens_trained` nor
`validation_loss` — this signal is training-telemetry only.

**Sidecar key:** `loss_saturation` (reserved null in the sidecar; not emitted by
capture-proxy).

---

## R-02 — Growth Factor

**Scope:** Universal (requires training dynamics data: `compute_flops` +
`model_params` + `measured_loss` triplets).

**What it measures:** Quadratic loss fit over isoFLOP points, crossover detection,
alpha computation. Uses Cramer's rule — numerically unstable for large datasets
(documented gotcha).

**Data source:** `--isoflop isoflop.json` (IsoFLOPPoint: `{compute_flops: float,
model_params: int, measured_loss: float}`). Training-telemetry only.

**Sidecar key:** `growth_factor` (reserved null in the sidecar; not emitted by
capture-proxy).

---

## R-03 — Effective Rank

**Scope:** Universal (requires weight tensors).

**What it measures:** SVD-based spectral entropy of weight matrices — how much of
the full rank is actually used. Computes `rank_ratio` (≤1.0, never exceeds 1.0)
and `pct_ratio` (percentage of full rank, unbounded as cap shrinks). `scoring.py`
returns `percentage_of_full_rank` for R-03.

**Data source:** `--weight-pairs-file weights.json` (weight pairs with `gradient`,
`weight`, `layer_name`). Can also be produced by the capture-proxy headroom sidecar
from a checkpoint.

**Sidecar key:** `effective_rank` — the only signal key emitted by the MVP sidecar.
Weight tensors come from the checkpoint via safetensors.

**Byte cap interaction:** `CAPTURE_HEADROOM_MAX_BYTES` truncates rows of wide
tensors (e.g. 256M cap on 5120×17408 → 3855-row truncation). The sidecar emits
`parent_shape` + `sampling_mode` so the authority can report coverage rather than
silently re-denominating.

---

## R-04 — Gradient/Weight Ratio

**Scope:** Universal (requires weight + gradient pairs).

**What it measures:** Gradient norm / weight norm ratio, z-score computation,
high-pressure thresholding. `grad_weight_ratio` returns a list (not a single
report); `scoring.py` treats `data` as iterable for this signal.

**Data source:** `--weight-pairs-file weights.json` with gradient arrays.

**Sidecar key:** `grad_weight_ratio` — weight-only pairs carry a real zero-gradient
array, so this scores 0.0 by design without a training hook.

---

## R-05 — Dead Neurons

**Scope:** Universal (requires activation data).

**What it measures:** Near-zero activation channel detection — fraction of neurons
whose activations stay below threshold (1e-6). Threshold application and index
extraction. Floating-point false positives possible at the float16/float32 boundary.

**Data source:** `--mlp-activations-file activations.json` or `--activations
expert_activations.json` (ExpertActivationData: `layers: dict[str, dict[str, int]]`
plus metadata).

**Sidecar key:** `dead_neurons` (reserved null; needs forward hooks on inference
traffic — not yet produced by capture-proxy).

---

## R-06 — Load Balance

**Scope:** MoE only. On dense models, marked `not_applicable` and excluded from the
composite score (weights renormalize over remaining signals).

**What it measures:** Router entropy / balance ratio — per-expert integer counts
per layer. Entropy computation, balance ratio, ideal entropy.

**Data source:** `--activations expert_activations.json` — requires a per-expert
routing histogram, observable only from inference traffic with routing visibility.
`moe_expert_profiler.py` does not exist in either repo (documented producer gap).

**Sidecar key:** `load_balance` (reserved null; needs routing histogram —
`CAPTURE_ROUTED_EXPERTS` sidecar is the planned producer path).

**Denominator:** Architectural expert count (not fed count), per fix D4/D5 — a
collapsed router scoring 1-of-8 experts reports `computed` with low ratio, not
`single_expert`.

---

## R-07 — Specialization

**Scope:** MoE only. On dense models, marked `not_applicable` and excluded from the
composite score.

**What it measures:** Pairwise expert cosine similarity, collapse detection. Uses
worst-case as representative (same as dead_neurons).

**Data source:** `--expert-weights-file experts.json` (expert weight vectors). Weak
offline — activation-weighted specialization needs inference.

**Sidecar key:** `specialization` (reserved null).

---

## R-08 — Gradient Coherence (diagnostic)

**Scope:** Diagnostic, opt-in with `--gradient-diagnostics`. Never affects the
headroom score.

**What it measures:** Cross-batch gradient cosine similarity.

**Data source:** `--gradient-provenance-file grad_provenance.json` with weight-pair
files carrying gradients.

---

## R-09 — Gradient Effective Rank (diagnostic)

**Scope:** Diagnostic, opt-in with `--gradient-diagnostics`. Never affects the
headroom score.

**What it measures:** SVD rank analysis of gradient matrices.

**Data source:** `--gradient-provenance-file grad_provenance.json` with weight-pair
files carrying gradients.

---

## Signal status values

| Status | Meaning | Scored as |
|--------|---------|-----------|
| `computed` | Signal successfully computed | Real value in [0.0, 1.0] |
| `skipped` | No data provided | 0.5 (neutral) in composite |
| `error` | Signal computation failed | 0.0 |
| `not_applicable` | Signal doesn't apply to this architecture | Excluded from numerator and denominator |
| `single_expert` | Architectural expert count == 1 | Treated as neutral (balance_ratio = None) |

**Skipped ≠ not_applicable.** Skipped signals score 0.5 (neutral). `not_applicable`
signals are excluded from the composite entirely. Dense models mark R-06/R-07 as
`not_applicable`; the composite is computed over the 5 universal signals only.

---

## Cross-references

- `architecture/authority-carrier.md` — which repo owns each signal
- `architecture/contract.md` — sidecar signal keys, reserved keys
- `concepts/frozen-contracts.md` — frozen dataclass types for each signal result
- `concepts/provenance.md` — source_type / verification_status for measured vs synthetic
- `guides/model-headroom.md` — how to run analysis with each signal
- `reference/env-vars.md` -- `CAPTURE_HEADROOM*` env vars controlling sidecar emission
