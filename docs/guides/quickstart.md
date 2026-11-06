# Quickstart — model-headroom end-to-end

**Purpose:** Install `sia-hu`, run `model-headroom` against an accessible model,
read the human report, and understand what the score means — and what it does not.
No prior knowledge of the codebase required.

This quickstart covers the authority package (`model_headroom`, the scoring
authority). It does not cover the carrier (`capture-proxy`, which emits sidecar
artifacts but never scores — see the README).

## 1. Install

From the repo root (`/kaggle/working/sia-hu`):

```bash
pip install -e ".[headroom]"
```

This installs the `sia-hu` package (version 0.2.0, `pyproject.toml:5-7`) in
editable mode plus the `headroom` extra (`pyproject.toml:30-37`): `torch>=2.15`,
`numpy>=2.5.4`, `safetensors>=0.8.0`, `huggingface_hub>=2.4`, `transformers>=5.22.1`,
`accelerate>=1.15`. The base install (`pip install -e .`, `pyproject.toml:14-19`)
pulls only FastAPI/uvicorn/httpx/pydantic — no torch — so `.[headroom]` is required
for scoring (`docs/model-headroom/README.md:17-19`).

The `model-headroom` entry point is declared at `pyproject.toml:44`:
`model-headroom = "model_headroom.cli:main"`.

Verify the install:

```bash
model-headroom --help
```

## 2. Run — base analysis (model files only)

Point `model-headroom` at an accessible HuggingFace model directory. The Qwen3-0.6B
model is available on this machine:

```bash
model-headroom \
  --model-name qwen3-0.6b \
  --model-path /kaggle/input/models/qwen-lm/qwen-3/transformers/0.6b/1
```

`--model-path` must be a directory containing `config.json` (`cli.py:239-247`,
`models/loader.py:59-74`). The architecture is auto-detected from the config
(`models/loader.py:126-241`); Qwen3-0.6B has no `num_experts`, so it is classified `dense`
(`models/loader.py:157-160`).

### What the base report shows

With only `--model-path` and no optional data files, every signal that needs
training data is skipped:

| Signal | Status | Why |
|--------|--------|-----|
| Loss Saturation | SKIPPED | no `--loss-curve` (`analysis/orchestrator.py:237-253`) |
| Growth Factor | SKIPPED | no `--isoflop` (`analysis/orchestrator.py:255-271`) |
| Effective Rank | SKIPPED | no `--weight-pairs-file` (`analysis/orchestrator.py:273-363`) |
| Grad Weight Ratio | SKIPPED | no `--weight-pairs-file` (`analysis/orchestrator.py:365-404`) |
| Dead Neurons | SKIPPED | no `--mlp-activations-file` (`analysis/orchestrator.py:406-453`) |
| Load Balance | SINGLE EXPERT | dense model, one expert (`analysis/orchestrator.py:455-482`) |
| Specialization | SINGLE EXPERT | dense model, no experts (`analysis/orchestrator.py:484-546`) |

Composite score: **0.50** (neutral prior — every applicable signal scores 0.5,
`analysis/scoring.py:41-49`). Measured coverage: **0%** (`analysis/scoring.py:133-199`,
`reporting/human_report.py:168-178`). Decision-grade: **False** (coverage below 70%,
`analysis/scoring.py:205-252`).

**This is where a stranger gets stuck.** The base run proves the tool works, but
it produces no *measured* result. To get measured signals you must supply training
data files (step 3).

## 3. Run — measured analysis (with training data files)

The seven core signals each need a corresponding data file:

| Signal | Flag | Data file format |
|--------|------|------------------|
| Loss Saturation | `--loss-curve` | JSON list of `{tokens_trained, validation_loss}` (`cli.py:65-67`, `models/types.py:53-58`) |
| Growth Factor | `--isoflop` | JSON list of isoFLOP points (`cli.py:69-71`) |
| Effective Rank | `--weight-pairs-file` | `{"pairs": [{"gradient": [[...]], "weight": [[...]], "layer_name": "..."}]}` (`cli.py:77-81`, `models/loader.py:328-381`) |
| Grad Weight Ratio | `--weight-pairs-file` | same file as effective_rank (`cli.py:77-81`) |
| Dead Neurons | `--mlp-activations-file` | `{"layers": [{"activations": [[[...]]], "layer_name": "..."}]}` (`cli.py:83-87`, `models/loader.py:473-498`) |
| Load Balance | `--activations` | expert activations JSON (MoE only) (`cli.py:73-75`) |
| Specialization | `--expert-weights-file` | expert weights JSON (MoE only) (`cli.py:89-93`) |

Weight pairs must be 2-D tensors (`models/loader.py:370-381`); a 3-D fused expert stack is
rejected, not sliced. MLP activations must be 3-D (`models/loader.py:493-494`).

### Example data files

Save these two files to `/tmp/quickstart-demo/`. They are **example/demonstration
data** — replace them with your own measured training data before drawing
conclusions.

**weight_pairs.json** (gradient/weight pairs for effective_rank and grad_weight_ratio):

```json
{
  "pairs": [
    {
      "layer_name": "model.layers.0.self_attn.q_proj.weight",
      "weight": [[0.12, -0.05, 0.31, 0.08], [-0.22, 0.17, -0.09, 0.14], [0.05, 0.28, -0.16, -0.03], [-0.11, 0.04, 0.20, -0.19]],
      "gradient": [[0.01, -0.005, 0.03, 0.008], [-0.02, 0.017, -0.009, 0.014], [0.005, 0.028, -0.016, -0.003], [-0.011, 0.004, 0.020, -0.019]],
      "parent_shape": [1024, 1024],
      "truncated": false,
      "sampling_mode": "full_matrix"
    },
    {
      "layer_name": "model.layers.0.mlp.gate_proj.weight",
      "weight": [[0.5, -0.2, 0.1], [-0.3, 0.4, -0.1], [0.2, 0.1, 0.6], [-0.1, 0.3, -0.4]],
      "gradient": [[0.05, -0.02, 0.01], [-0.03, 0.04, -0.01], [0.02, 0.01, 0.06], [-0.01, 0.03, -0.04]],
      "parent_shape": [3072, 1024],
      "truncated": false,
      "sampling_mode": "full_matrix"
    }
  ]
}
```

**mlp_activations.json** (3-D activation tensors for dead_neurons):

```json
{
  "layers": [
    {
      "layer_name": "model.layers.0.mlp.down_proj",
      "activations": [
        [[0.5, 0.0, 1.2, -0.3, 0.0], [0.1, 0.0, 0.8, 0.0, 0.4], [0.0, 0.0, 0.0, -0.1, 0.0]],
        [[0.3, 0.0, 0.9, 0.0, 0.0], [0.0, 0.0, 1.1, -0.2, 0.0], [0.2, 0.0, 0.0, 0.5, 0.0]]
      ]
    }
  ]
}
```

Then run:

```bash
model-headroom \
  --model-name qwen3-0.6b \
  --model-path /kaggle/input/models/qwen-lm/qwen-3/transformers/0.6b/1 \
  --weight-pairs-file /tmp/quickstart-demo/weight_pairs.json \
  --mlp-activations-file /tmp/quickstart-demo/mlp_activations.json
```

### What the measured report shows

Three signals are now computed; four remain skipped or single_expert:

| Signal | Status | Result |
|--------|--------|--------|
| Effective Rank | computed | rank=2.8 (91.8% of full rank), `full_matrix` sampling |
| Grad Weight Ratio | computed | 0/2 layers high pressure |
| Dead Neurons | computed | 40.0% dead (2/5 neurons) |
| Loss Saturation | SKIPPED | no `--loss-curve` |
| Growth Factor | SKIPPED | no `--isoflop` |
| Load Balance | SINGLE EXPERT | dense model |
| Specialization | SINGLE EXPERT | dense model |

Composite score: **0.65**. Measured score: **0.959** at **30% coverage**
(`reporting/human_report.py:168-178`). Score range: **[0.288 .. 0.988]** — the unmeasured
signals could push the true score anywhere in that band (`analysis/scoring.py:177-186`).
Decision-grade: **False** — 30% coverage is below the 70% threshold
(`analysis/scoring.py:205-252`), and the computed signals lack full-matrix sampling and
gradient provenance (`analysis/scoring.py:241-252`).

## 4. How to read the result

- **HEADROOM SCORE** (composite, `analysis/scoring.py:110-130`): weighted average over
  applicable signals using fixed weights (`analysis/scoring.py:28-36`):
  loss_saturation 0.20, growth_factor 0.20, effective_rank 0.15, grad_weight_ratio
  0.15, dead_neurons/load_balance/specialization 0.10 each. Skipped and
  single_expert signals score 0.5 (neutral, `analysis/scoring.py:41-49`); not_applicable
  signals are excluded entirely (`analysis/scoring.py:123-124`).
- **MEASURED SCORE** (`reporting/human_report.py:168-178`): score computed only over signals
  that were actually measured from data. If this differs from the composite, the
  composite is mixing in neutral priors for missing signals.
- **SCORE RANGE** (`analysis/scoring.py:177-186`): lower bound = measured signals score 0
  for everything unmeasured; upper bound = measured signals score 1.0 for
  everything unmeasured. A wide range means the composite is not trustworthy.
- **R-03 BOUND** (`reporting/human_report.py:79-126`): effective_rank is a tile score, not a
  full-matrix score. Do not compare it to full-matrix effective_rank unless tile
  coverage is above ~20% (`analysis/orchestrator.py:63-121`).
- **Decision-grade** (`analysis/scoring.py:255-260`): requires ≥70% measured coverage, no
  unverified/estimated computed signals, complete gradient provenance, and
  full-matrix sampling on gradient-derived signals.

## 5. Value and limitations

**Value:**
- Seven core signals (R-01–R-07) plus two opt-in gradient diagnostics (R-08,
  R-09) give a structured read on training headroom
  (`docs/model-headroom/README.md:83-95`).
- Provenance-aware scoring separates measured from synthetic/fixture/estimated
  inputs (`docs/model-headroom/README.md:113`, `analysis/scoring.py:100-107`).
- Decision-grade gate prevents acting on incomplete evidence
  (`analysis/scoring.py:255-260`, exit code 2 with `--analysis-mode decision_grade`,
  `cli.py:415-420`).
- Weights load to CPU RAM by design (NFR-01) — no GPU required
  (`docs/model-headroom/README.md`).

**Limitations:**
- No signal is computed from raw model weights alone. Every measured signal needs
  an external data file (loss curve, weight pairs, activations, expert weights).
  The base run with only `--model-path` yields all-skipped or single-expert
  signals and a neutral 0.50 score.
- On dense models, R-06 (load_balance) and R-07 (specialization) are single_expert
  and score 0.5 neutral — they cannot be meaningfully measured
  (`analysis/orchestrator.py:455-546`, `analysis/scoring.py:44-45`).
- R-08/R-09 gradient diagnostics require `--gradient-diagnostics` and weight pairs;
  they never affect the composite score (`analysis/orchestrator.py:548-615`,
  `docs/model-headroom/README.md:92-95`).
- The composite score mixes neutral priors for missing signals; use the measured
  score and score range to judge reliability.
- Single-sample or synthetic data should not be quoted as model-level claims
  (`docs/model-headroom/README.md`).
- DeBERTa naming is unverified (`docs/model-headroom/README.md`).

## 6. Troubleshooting

**1. `model-headroom: command not found`**
The entry point is not installed or not on PATH. Fix: run `pip install -e
".[headroom]"` from the repo root (`pyproject.toml:44`), then confirm
`which model-headroom` resolves.

**2. `Error: --model-path does not exist` or `is not a directory`**
The path is wrong or points at a file. Fix: point at the model directory that
contains `config.json` (e.g.
`/kaggle/input/models/qwen-lm/qwen-3/transformers/0.6b/1`). Validation is at
`cli.py:239-247`; format detection at `models/loader.py:59-74`.

**3. Data file rejected (weight pairs not 2-D, activations not 3-D, malformed JSON)**
Weight pairs must be 2-D tensors (`models/loader.py:370-381`); MLP activations must be
3-D (`models/loader.py:493-494`). A 3-D fused expert stack is rejected, not sliced
(`models/loader.py:376-379`). Fix: export one expert per pair, or match the documented
JSON formats (`cli.py:77-93`).

## 7. Verified end-to-end

The commands in this quickstart were run against the accessible Qwen3-0.6B model
at `/kaggle/input/models/qwen-lm/qwen-3/transformers/0.6b/1`:

- `pip install -e ".[headroom]"` — deps already satisfied (torch 2.11.0+cpu,
  numpy 2.1.3, safetensors 0.8.0, transformers 5.16.1, huggingface_hub 1.33.0,
  accelerate 1.14.0); sia-hu 0.2.0 installed editable at
  `/kaggle/working/sia-hu`.
- Base run (`--model-name qwen3-0.6b --model-path ...`) — exit 0, all signals
  skipped/single_expert, composite 0.50, measured coverage 0%.
- Extended run (with `/tmp/quickstart-demo/weight_pairs.json` and
  `/tmp/quickstart-demo/mlp_activations.json`) — exit 0, effective_rank,
  grad_weight_ratio, dead_neurons computed; composite 0.65, measured score 0.959
  at 30% coverage, decision-grade False.

File:line references cited throughout are from:
`pyproject.toml`, `src/model_headroom/cli.py`,
`src/model_headroom/analysis/orchestrator.py`,
`src/model_headroom/analysis/scoring.py`,
`src/model_headroom/reporting/human_report.py`,
`src/model_headroom/models/loader.py`,
`src/model_headroom/models/types.py`,
`docs/model-headroom/README.md`.

---

See also: `docs/model-headroom/README.md` (full CLI flag reference, 19 flags,
`cli.py:parse_args`).
