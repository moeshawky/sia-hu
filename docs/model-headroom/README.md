# model_headroom

Model Headroom Analyzer — 7 core signals + 2 opt-in gradient diagnostics for training headroom on MoE and dense models. Python package, strict types, 21 frozen dataclasses, 95% test coverage.

## Installation

```bash
# From the repo root (sia-hu) — the package has no pyproject.toml of its own
pip install -e .
pip install -e ".[dev]"   # pytest / mypy / ruff / pre-commit
pip install -e ".[gguf]"  # optional: GGUF model support
pip install -e ".[headroom]"   # torch / numpy / safetensors
pip install -e ".[dev,headroom,gguf]"   # everything
```

Requires Python ≥3.12, `torch>=2.15`, `numpy>=2.5.4`. Package version 0.3.0.
`pip install -e .` alone installs the carrier-free base package; the heavy
scoring deps live in the `headroom` extra, which is what keeps a carrier-only
install from pulling torch. Entry points: `model-headroom`, `model-headroom-refusal`.

## Usage

```bash
model-headroom --model-name NAME --model-path /path/to/model [OPTIONS]
```

Architecture is auto-detected from the model config. Override with `--architecture {auto,moe,dense}`:

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

### Analysis modes, diagnostics, output

```bash
# Strict decision-grade gate (exit 2 with gaps listed if unmet)
model-headroom --model-name NAME --model-path PATH \
  --analysis-mode decision_grade

# Opt-in gradient diagnostics (R-08/R-09, module families)
model-headroom --model-name NAME --model-path PATH \
  --weight-pairs-file weights.json --gradient-diagnostics

# Provenance overrides + machine-readable output
model-headroom --model-name NAME --model-path PATH \
  --weight-pairs-file weights.json \
  --weight-pairs-source-type measured \
  --mlp-source-type synthetic \
  --output-format json --output-file report.json
```

Exit codes: `0` success, `1` error (bad input, unreadable file, analysis failure), `2` decision-grade not met.

All 20 flags are declared in `src/model_headroom/cli.py:parse_args` — the list above is grouped, not truncated.

## Signals

| Signal | Name | Scope | Description |
|--------|------|-------|-------------|
| R-01 | Loss Saturation | Universal | Training loss plateau detection |
| R-02 | Growth Factor | Universal | Quadratic loss fit, crossover detection |
| R-03 | Effective Rank | Universal | SVD-based spectral entropy of weight matrices |
| R-04 | Gradient/Weight Ratio | Universal | Norm ratio, z-score computation |
| R-05 | Dead Neurons | Universal | Near-zero activation channel detection |
| R-06 | Load Balance | MoE only | Router entropy / balance ratio |
| R-07 | Specialization | MoE only | Pairwise expert cosine similarity |
| R-08 | Gradient Coherence | Diagnostic (opt-in `--gradient-diagnostics`) | Cross-batch gradient cosine similarity |
| R-09 | Gradient Effective Rank | Diagnostic (opt-in `--gradient-diagnostics`) | SVD rank analysis of gradient matrices |

On dense models, R-06 and R-07 are marked `not_applicable` and excluded from the composite score (computed over the 5 universal signals only). On MoE models all 7 signals participate. R-08 and R-09 are computed only with `--gradient-diagnostics` and never affect the headroom score.

## Architecture Detection

Auto-detection reads the HuggingFace config and classifies:
- `dense` — no expert keys, `num_experts=0`
- `moe` — expert keys present, `num_experts > 0`

Architecture subtype detection identifies `hybrid_linear_attention` (mixed full/linear attention layers), `has_vision_tower`, and `has_mtp` (multi-token prediction) capabilities.

## Model Formats

`--model-path` accepts a HuggingFace directory (with `config.json`) or a `.gguf` file. Format is auto-detected from the path (`models/loader.py:_detect_model_format`); GGUF configs are read via `models/gguf_loader.py` (requires the `gguf` extra). Weights resolve through `src/model_headroom/device.py` — accelerator preferred, CPU as fallback (NFR-01).

### GGUF scope and limits

Supported: dense and hybrid-attention GGUF; legacy Prism ternary Q2_0 via the ternary dequant adapters (`models/gguf_loader.py:634-635`, `:705-706`); config from KV metadata (`load_gguf_config`, `models/loader.py:141`); family from `expert_count` (`models/loader.py:142`); subtype/attention/vision/MTP flags derived from the tensor-name table with `detection_source="gguf_tensor_names"` (`models/loader.py:135-173`).

Not supported: MoE expert extraction end-to-end — there is no architecture→tensor-name resolution; `extract_expert_weight_from_gguf` (`models/gguf_loader.py:595`) is a single-tensor reader keyed by literal tensor name. Headroom sidecar capture from a `.gguf` — safetensors-only, one warning, skipped by design (`src/capture_proxy/app.py:2384`, `:2421`). GGUF weight samples default to provenance `fixture`, which yields 0% measured coverage by construction; `--weight-pairs-source-type measured` opts in (`src/model_headroom/cli.py:426-441`). Sampling: `--rank-sample-count` sets the tile rows (default `None`, `cli.py:118`); samples come from layers `0`, `block_count//2`, `block_count-1` across `ffn_gate`/`ffn_up`/`ffn_down` (`cli.py:200-205`); the printed rank percentage is tile-relative and the parent ceiling + coverage are printed beside it.

## Scoring

Composite score is a weighted average over applicable signals (`analysis/scoring.py:compute_headroom_score`). Fixed weights: loss_saturation 0.20, growth_factor 0.20, effective_rank 0.15, grad_weight_ratio 0.15, dead_neurons / load_balance / specialization 0.10 each. Skipped signals score 0.5 (neutral), errors score 0.0, `not_applicable` signals are excluded from numerator and denominator.

The measured score (`compute_coverage_metrics`) separates verified-measured signals from synthetic/fixture/estimated/unverified ones and reports coverage plus lower/upper bounds. Decision-grade requires ≥70% measured coverage (`--analysis-mode decision_grade` enforces it, exit 2 otherwise). Source types: `measured`, `synthetic`, `fixture`, `estimated`, `imported` — inferred from file paths unless overridden with `--weight-pairs-source-type` / `--mlp-source-type`.

## Output Formats

- `human` (default): terminal report, lines wrapped at 78 columns, truncated at 200 lines.
- `json`: full `HeadroomReport` including per-signal `source_type`/`verification_status`/`sampling_mode`, `analysis_mode`, `decision_grade`, and flattened coverage fields (`measurement_coverage`, `headroom_score_measured`, `score_lower_bound`, `score_upper_bound`).

Use `--output-file` to write to a file instead of stdout. `python -m model_headroom` works identically to the `model-headroom` entry point.

## Refusal direction analysis

Entry point `model-headroom-refusal` (`src/model_headroom/refusal/cli.py:48`).

Live capture: `--model-path` must be a HuggingFace directory containing
`config.json`, or a Hub id. A file path — including a `.gguf` — is rejected
before `AutoTokenizer` is constructed (`refusal/cli.py:_ModelPathError`,
`_is_filesystem_shaped`, lines 44 and 218), exit 1, because transformers cannot
load GGUF checkpoints (`refusal/cli.py:233-239`, `:253`). A directory without
`config.json` is rejected (`refusal/cli.py:240-243`); a nonexistent
filesystem-shaped path is rejected (`refusal/cli.py:244-245`); a Hub id is
passed through for transformers to resolve.

Alternatives named by the guard: `--harmful-activations`/`--harmless-activations`
with pre-captured residuals, or `--sglang-url` with `--enable-return-hidden-states`
(`refusal/cli.py:236-238`, `:388-414`).

Prompt files are newline-delimited `.txt` (exactly one prompt per physical line),
`.json`, or `.jsonl` (`refusal/activations.py:load_prompts`, lines 79-120).
`scripts/fetch_refusal_prompts.py` builds both sets with the standard library
only (AdvBench CSV + the HF datasets-server rows API).

## Further Docs

- `PROJECT-ROADMAP.md` — living plan, completed phases 1–4, evidence-gated phases 5–9.
- `TEST_PROTOCOL.md` — adversarial test protocol, scope, success/failure criteria.
- Contributor constraints: frozen dataclasses, lint→typecheck→test order (see repo README).

## Project Structure

Package source lives at the root as `src/model_headroom/`, under the single
`sia-hu` project (one `pyproject.toml`, one `uv.lock`, one ruff/mypy/pytest
config, all verified from the repo root).

Measured by `ix` across the codebase:

| Metric | Value |
|--------|-------|
| Source files | 23 Python files, 3,977 lines |
| Test files | 42 files, 6,109 lines |
| Frozen dataclasses | 21 (all in `models/types.py`) |
| Signal modules | 9 signals + `__init__` |
| Reporting modules | 2 (`human_report.py`, `json_report.py`) |
| Analysis modules | `orchestrator.py` (522 lines), `scoring.py` (381 lines) |
| Model types | `types.py` (306 lines), `loader.py` (585 lines) |
| CLI entry | `cli.py` — `model-headroom` command |
| Test assertions | 422 test functions |
| `py.typed` marker | Present |

## Key Conventions

- **mypy strict = true** — no `Any` without justification
- **21 frozen dataclasses** in `models/types.py` — all immutable
- **Weights resolve through `device.py`** — accelerator preferred (cuda, then xpu, then mps), CPU as fallback (NFR-01).
- **Skipped signals score 0.5** (neutral) in composite score; not_applicable signals are excluded entirely
- **Provenance-aware scoring** — measured, synthetic, fixture, estimated, imported data classified and weighted separately
- **Measured headroom score** separated from legacy composite with coverage and uncertainty bounds
- **One project**: this package ships inside the root `sia-hu` project; install, config, and verify commands are root-level (see `../../README.md`)

## Verification

```bash
# Full verification chain (never skip steps) — root commands
ruff check . && ruff format --check . && mypy src/ && pytest

# Test suite
pytest --tb=short -ra -v

# Coverage (both src/ packages)
pytest --cov="capture_proxy" --cov="model_headroom" --cov-report=term-missing -v
```

`mypy src/` is the authoritative typecheck — the authority's own config lives in
the root `pyproject.toml`, and `mypy .` sweeps `tests/` out of scope and yields
pre-existing errors. `mypy src/` takes 1m20s–3m03s; a timeout means *unmeasured*,
not green. `pytest` runs both packages' suites (root `addopts`: `-v --tb=short`,
`testpaths` = `capture-proxy/tests` + `model_headroom/tests`).
