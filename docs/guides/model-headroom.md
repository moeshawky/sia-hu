# Authority usage guide — model-headroom

**Purpose:** How to use model-headroom (the authority, scores Sia): run analysis,
understand signals, read output formats. Cross-references
`architecture/authority-carrier.md` (authority role), `concepts/headroom-signals.md`
(R-01 through R-09 reference), `concepts/provenance.md` (measured vs synthetic),
`concepts/frozen-contracts.md` (frozen dataclasses, NFR-01), and
`reference/cli-flags.md` (all 20 flags). No Python code changes are proposed here.

---

## What model-headroom is

Model Headroom Analyzer — 7 core signals + 2 opt-in gradient diagnostics for
training headroom on MoE and dense models. Python package, strict types, 21 frozen
dataclasses, 95% test coverage.

**Version:** 0.3.0 — the authority half of sia-hu. Scores; never proxies.

---

## Install

From the repo root:

```bash
pip install -e .                 # base package (carrier-free)
pip install -e ".[headroom]"     # + torch / numpy / safetensors (scoring)
pip install -e ".[gguf]"         # optional: GGUF model support
pip install -e ".[dev]"          # pytest / mypy / ruff / pre-commit
```

Entry points: `model-headroom`, `model-headroom-refusal`. `python -m model_headroom`
works identically to the `model-headroom` entry point.

---

## Run analysis

```bash
model-headroom --model-name NAME --model-path /path/to/model [OPTIONS]
```

Architecture is auto-detected from the model config. Override with `--architecture`:

```bash
model-headroom --model-name llama-7b --model-path /models/llama-7b --architecture dense
model-headroom --model-name mixtral-8x7b --model-path /models/mixtral --architecture moe
```

### Optional data files (provenance-aware scoring)

```bash
# Training dynamics
model-headroom --model-name NAME --model-path PATH \
  --loss-curve loss.json \
  --isoflop isoflop.json

# Pre-extracted tensors
model-headroom --model-name NAME --model-path PATH \
  --weight-pairs-file weights.json \
  --mlp-activations-file activations.json \
  --expert-weights-file experts.json \
  --activations expert_activations.json

# Rank confidence + provenance
model-headroom --model-name NAME --model-path PATH \
  --weight-pairs-file weights.json \
  --rank-parent-shapes shapes.json \
  --rank-sample-count 64 \
  --sampling-mode randomized_svd \
  --gradient-provenance-file grad_provenance.json
```

### Analysis modes

```bash
# Strict decision-grade gate (exit 2 if <70% measured coverage)
model-headroom --model-name NAME --model-path PATH --analysis-mode decision_grade

# Opt-in gradient diagnostics (R-08/R-09, never affect score)
model-headroom --model-name NAME --model-path PATH \
  --weight-pairs-file weights.json --gradient-diagnostics

# Provenance overrides + machine-readable output
model-headroom --model-name NAME --model-path PATH \
  --weight-pairs-file weights.json \
  --weight-pairs-source-type measured \
  --mlp-source-type synthetic \
  --output-format json --output-file report.json
```

**Exit codes:** 0 success, 1 error (bad input, unreadable file, analysis failure),
2 decision-grade not met.

All 20 flags are declared in `src/model_headroom/cli.py:parse_args`.

---

## Signals

| ID | Name | Scope | Measures |
|----|------|-------|----------|
| R-01 | Loss Saturation | Universal | Training loss plateau |
| R-02 | Growth Factor | Universal | Quadratic loss fit, crossover |
| R-03 | Effective Rank | Universal | SVD spectral entropy of weights |
| R-04 | Grad/Weight Ratio | Universal | Norm ratio, z-score |
| R-05 | Dead Neurons | Universal | Near-zero activation channels |
| R-06 | Load Balance | MoE only | Router entropy, balance ratio |
| R-07 | Specialization | MoE only | Expert cosine similarity |
| R-08 | Gradient Coherence | Diagnostic | Cross-batch gradient cosine |
| R-09 | Gradient Effective Rank | Diagnostic | SVD rank of gradients |

On dense models, R-06 and R-07 are `not_applicable` and excluded from the composite.
Skipped signals score 0.5 (neutral). R-08/R-09 never affect the score.

---

## Output formats

| Format | Description |
|--------|-------------|
| `human` (default) | Terminal report, 78-col width, ≤200 lines |
| `json` | Full `HeadroomReport` with per-signal `source_type`/`verification_status`/`sampling_mode`, `analysis_mode`, `decision_grade`, coverage fields (`measurement_coverage`, `headroom_score_measured`, `score_lower_bound`, `score_upper_bound`) |

Use `--output-file` to write to a file instead of stdout.

---

## Scoring

Composite score is a weighted average over applicable signals with fixed weights:
loss_saturation 0.20, growth_factor 0.20, effective_rank 0.15, grad_weight_ratio
0.15, dead_neurons/load_balance/specialization 0.10 each.

The **measured headroom score** separates verified-measured signals from
synthetic/fixture/estimated/unverified ones and reports coverage plus lower/upper
bounds. Decision-grade requires ≥70% measured coverage.

---

## Model formats

`--model-path` accepts a HuggingFace directory (with `config.json`) or a `.gguf`
file. Format is auto-detected from the path. GGUF configs are read via
`models/gguf_loader.py` (requires the `gguf` extra). Weights resolve through
`src/model_headroom/device.py` — accelerator preferred, CPU as fallback (NFR-01).

---

## Architecture detection

Auto-detection reads the HuggingFace config and classifies:
- `dense` — no expert keys, `num_experts=0`
- `moe` — expert keys present, `num_experts > 0`

Subtype detection identifies `hybrid_linear_attention`, `has_vision_tower`, and
`has_mtp` (multi-token prediction).

---

## Cross-references

- `architecture/authority-carrier.md` — authority role, what it never does
- `architecture/contract.md` — 2-D pair contract, frozen dataclass invariants
- `concepts/headroom-signals.md` — R-01 through R-09 signal reference
- `concepts/provenance.md` — source_type / verification_status
- `concepts/frozen-contracts.md` — frozen dataclasses, NFR-01
- `guides/quickstart.md` — install and bootstrap
- `reference/cli-flags.md` — all 19 model-headroom flags
- `reference/env-vars.md` — model_headroom env vars
