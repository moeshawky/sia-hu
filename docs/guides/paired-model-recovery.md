# Paired-Model Recovery Diagnostics

This guide documents the `model-headroom-paired` CLI: `compare`, `decide`,
`verify`, `schema-version`. It is written for an operator who runs an external
GPU notebook or training script and needs to supply real measurements to the
CLI, then read the decision.

Every command in this guide was executed by me while writing it, except the
GPU capture steps in section 4, which require a GPU and a loaded model and are
labelled `NOT EXECUTED HERE`. Every number in `docs/examples/paired-recovery-report.json`
is output from the real code, produced by the commands in section 6.

## 1. What this answers

The three operator questions, in their words:

1. "Did the compressed model change, and by how much on each layer?" → `compare`.
2. "Is my recovery intervention supported by held-out measurements?" → `decide`.
3. "Can I prove, later, that this decision artifact was not tampered with?" → `verify`.

One-sentence boundary statement: this toolkit MEASURES RECORDED RESULTS and
never executes an intervention. It reads measurement documents you produce,
compares them, and assigns a verdict. It does not train, quantize, repack
checkpoints, serve models, or edit weights (section 9).

## 2. Install and verify

From the repo root (`/kaggle/working/sia-hu`):

```bash
pip install -e .
```

Confirm the CLI is present and the schema version:

```bash
which model-headroom-paired
# /usr/local/bin/model-headroom-paired  (executed here)

model-headroom-paired schema-version
# 1  (executed here; exit 0)
```

`schema-version` prints `SCHEMA_VERSION`, which is `"1"` in
`src/model_headroom/paired/schema.py:27`.

## 3. The measurement contract

This is the conceptual core. Read it before running anything.

A measurement document (`MeasurementDocument`, `src/model_headroom/paired/schema.py:274`)
carries layers, each with `output_cosine`, `normalized_reconstruction_error`,
`output_norm_ratio`, and a tri-state `status`:

| Status | Meaning |
|--------|---------|
| `measured` | A value is present and trustworthy. |
| `not_measured` | The quantity was not measured; it is `null`, never `0.0`. |
| `not_applicable` | The quantity does not apply to this kind of tensor. |

Key rules, enforced by construction in `src/model_headroom/paired/schema.py`:

- An unmeasured quantity is a named state in the type system, never a number.
  Construction rejects a `value` under any status other than `measured`
  (`src/model_headroom/paired/schema.py:56-70`).
- `TensorKind` distinguishes `packed_buffer` from `effective_weight`
  (`src/model_headroom/paired/schema.py:31-38`). A `packed_buffer` module is
  `not_applicable` for effective-weight error — never a number
  (`src/model_headroom/paired/compare.py:223-242`).
- `compare` pairs layers by `layer_name`, never by position. Coverage is
  `measured_layer_count / comparable_layer_count`, with `None` meaning unknown
  (`src/model_headroom/paired/compare.py:186-188`).

`delta` is always raw `post - base`
(`src/model_headroom/analysis/intervention_record.py:48`), so a loss that fell
is negative. Polarity is set explicitly with `--metric-direction`; the default
is `lower_is_better` because every quantity this tool measures is error-like
(`src/model_headroom/paired/decision.py:17-19`).

## 4. Producing measurements on a GPU

On a GPU runtime that already has a model loaded, wrap the forward (and
backward) pass in `PairedActivationRecorder` (`src/model_headroom/paired/instrument.py:102`):

```python
from model_headroom.paired.instrument import (
    PairedActivationRecorder,
    InstrumentationBudget,
)

budget = InstrumentationBudget(
    max_bytes_per_module=1 << 20,  # 1 MiB cap per module
    max_modules=8,  # at most 8 tensors recorded
    max_tokens_per_module=64,
    sample_every_n_tokens=1,
)

recorder = PairedActivationRecorder(model, budget=budget)

# NOT EXECUTED HERE — requires a GPU, a loaded torch model, and a
# forward/backward pass. The import itself is torch-free (verified in section 7,
# claim 12): `import model_headroom.paired.instrument` adds no torch to
# sys.modules.
with recorder.capture_forward(kind="activation"):
    _ = model(input_ids)  # your training/forward step; hooks sample outputs

manifest = recorder.last_manifest
# manifest.captured holds CapturedTensor records (module_path, kind, shape,
# dtype, sampled_rows, sampling_strategy, status, values, byte_size).
# manifest.skipped lists paths declined because max_modules was reached.
# manifest.dropped_bytes is a scalar side-channel for budget-excluded modules.
```

Byte budget behavior (`src/model_headroom/paired/instrument.py:209-271`): a
module whose tensor fits within `max_bytes_per_module` is sampled `full`;
larger tensors are row-sliced. A module whose minimum row still exceeds the cap
is recorded with `status="not_measured"`, `sampled_rows=0`, `values=()` — never
a fabricated number.

To target QTEA-style buffer-backed modules instead of the default
`.mlp` / `.input_layernorm` suffixes, use
`discover_buffer_backed_modules(model)`
(`src/model_headroom/paired/instrument.py:436`). It matches buffer names
against the keywords `{"scale", "codes", "codebook", "residual"}` discovered
through `named_modules()` + `named_buffers()` and returns the module paths that
own them.

Every hook handle is removed unconditionally in `finally` blocks;
`release()` is idempotent and returns `False` if a handle is still present
after removal (`src/model_headroom/paired/instrument.py:415-426`).

Serialize the captured manifest to JSON and feed it to the CLI as a
measurement document (section 5 shows the document shape).

## 5. Producing measurements without a GPU

This path IS executable here — no GPU, no torch model, no training step. Build
a `MeasurementDocument` in Python and serialize it to JSON:

```python
import sys

sys.path.insert(0, "/kaggle/working/sia-hu/src")

from model_headroom.paired.schema import (
    GradientMeasurement,
    LayerMeasurement,
    MeasurementDocument,
)

doc = MeasurementDocument(
    model_id="ref-qwen-dense",
    role="reference",  # "reference" | "compressed" | "recovered"
    input_fingerprint=None,
    captured_at="2026-11-12T00:00:00+00:00",
    architecture="bert-base-uncased",
    torch_version="2.11.0",
    device="cpu",
    layers=(
        LayerMeasurement(
            layer_name="encoder.layer0.mlp",
            layer_index=0,
            family="mlp",
            output_cosine=0.97,
            normalized_reconstruction_error=0.04,
            output_norm_ratio=1.01,
            status="measured",
        ),
    ),
    gradients=GradientMeasurement(
        status="measured",
        grad_norm=4.2,
        coherence=0.91,
        effective_rank=12.0,
        parameter_count=12_000_000,
        source_type="measured",
        verification_status="unverified",
    ),
    notes=("synthetic fixture",),
)

print(doc.to_json())
```

I ran the equivalent of this snippet via the scratch script
`/tmp/paired-fixture/gen_fixtures.py` (scratch lives under /tmp, never in the
repo), which wrote `reference.json`, `compressed.json`, `baseline.json`, and
`post.json` under `/tmp/paired-fixture/`. Those are the fixtures used in
section 6.

The CLI flattens a document into measurement keys
(`src/model_headroom/paired/cli.py:97-116`): each layer field becomes
`<layer_name>.<field>` (e.g. `encoder.layer0.mlp.normalized_reconstruction_error`),
plus `gradients.grad_norm`, `gradients.coherence`, `gradients.effective_rank`
when gradients are present. Those keys are what `--held-out-keys` addresses.

## 6. The workflow

Ingest reference, ingest compressed, compare, associate an intervention,
decide, persist, verify. All commands below were executed here; outputs are
shown exactly as produced.

Step 1 — schema version:

```bash
model-headroom-paired schema-version
# 1
```

Step 2 — compare reference vs compressed:

```bash
model-headroom-paired compare \
  --reference /tmp/paired-fixture/reference.json \
  --candidate /tmp/paired-fixture/compressed.json \
  --out /tmp/paired-fixture/comparison.json
# exit 0
```

`/tmp/paired-fixture/comparison.json` (real output):

```json
{
  "reference_model_id": "ref-qwen-dense",
  "candidate_model_id": "cmp-qwen-dense",
  "input_fingerprint_match": null,
  "layers": [
    {
      "layer_name": "encoder.layer0.mlp",
      "family": "mlp",
      "cosine": 1.0,
      "normalized_reconstruction_error": 0.0,
      "norm_ratio": 1.0,
      "reference_norm": null,
      "candidate_norm": null,
      "status": "measured",
      "reason": null
    }
  ],
  "modules": [],
  "gradients": {
    "status": "not_measured",
    "reference_grad_norm": null,
    "candidate_grad_norm": null,
    "grad_norm_ratio": null,
    "coherence_delta": null,
    "effective_rank_ratio": null,
    "reason": "gradients absent on reference, candidate"
  },
  "outliers": [],
  "coverage": 1.0,
  "measured_layer_count": 1,
  "comparable_layer_count": 1,
  "warnings": [
    "input fingerprint unknown on reference, candidate",
    "gradient measurement absent on reference, candidate"
  ]
}
```

`coverage` is 1.0 because 1 layer was measured out of 1 comparable. Both
documents lack `input_fingerprint`, so the match is `null` with a warning —
absence is not reported as agreement.

Step 3 — decide on the intervention. Baseline = pre-intervention document,
post = post-intervention document. `--held-out-keys` names measurement keys
that were measured on held-out data (the keys the CLI flattens from the
documents). `--coverage 0.9` is the measured coverage of this intervention.

```bash
model-headroom-paired decide \
  --baseline /tmp/paired-fixture/baseline.json \
  --post /tmp/paired-fixture/post.json \
  --strategy targeted_low_rank \
  --held-out-keys "encoder.layer0.mlp.normalized_reconstruction_error" \
  --coverage 0.9 \
  --store /tmp/paired-fixture/store \
  --out /tmp/paired-fixture/decision.json
# exit 0
# stderr: artifact_id: paired_recovery_decision_cf2d1207028d
```

`/tmp/paired-fixture/decision.json` (real output; this is also the sample
report at `docs/examples/paired-recovery-report.json`):

```json
{
  "strategies": [
    {
      "strategy": "targeted_low_rank",
      "metric_direction": "lower_is_better",
      "intervention_id": "base-qwen-dense->post-qwen-dense",
      "observed_delta": -0.22999999999999998,
      "held_out_keys": ["encoder.layer0.mlp.normalized_reconstruction_error"],
      "affected_families": [],
      "memory_overhead_gb": null,
      "parameter_overhead": null,
      "uncertainty": 0.09999999999999998,
      "coverage": 0.9,
      "verdict": "SUPPORTED",
      "verdict_reason": "improvement on held-out measurement (raw delta -0.22999999999999998, direction=lower_is_better)",
      "missing_evidence": ["memory budget", "parameter overhead"],
      "next_experiment": "Re-run the held-out measurement on encoder.layer0.mlp.normalized_reconstruction_error with metric_direction=lower_is_better; refute the strategy if mean delta is not strictly less than 0.0.",
      "downgrade_reasons": []
    }
  ],
  "distinguishable": false,
  "summary": "Single strategy evaluated.",
  "warnings": []
}
```

How to read it: `observed_delta` is `post - base` = `0.08 - 0.31` = `-0.23`.
With `lower_is_better`, negative means improvement. Coverage `0.9` is at or
above the `0.8` threshold (`COVERAGE_THRESHOLD`,
`src/model_headroom/paired/decision.py:45`), so the verdict is `SUPPORTED`.
`distinguishable` is `false` because only one strategy was evaluated
(`--strategy` given); two or more non-INCONCLUSIVE strategies are required
(`src/model_headroom/paired/decision.py:367-370`).

Step 4 — persist. The `--store` flag already wrote the artifact to the
evidence store and printed `artifact_id` on stderr:

```
artifact_id: paired_recovery_decision_cf2d1207028d
```

Step 5 — verify the persisted artifact:

```bash
model-headroom-paired verify \
  --store /tmp/paired-fixture/store \
  --artifact-id paired_recovery_decision_cf2d1207028d
# exit 0
```

Real output:

```json
{
  "verified": true,
  "hash": "cf2d1207028d5d3e3c81d9a6ff7cce7cd45bfc51ac46d9a4c7948706ac182067"
}
```

`verify` recomputes the SHA-256 of the stored content and compares it to the
hash recorded in the store index (`src/model_headroom/analysis/evidence_store.py:105-127`).

## 7. Reading the decision

Three verdicts (`Verdict`, `src/model_headroom/paired/decision.py:55`):

| Verdict | Requires |
|---------|----------|
| `SUPPORTED` | Held-out keys supplied, at least one measured, mean delta strictly improved in the chosen `metric_direction`, and coverage at or above 0.8. |
| `REFUTED` | Held-out keys supplied and measured, but mean delta regressed (not an improvement) in the chosen direction. Coverage is not a gate for REFUTED. |
| `INCONCLUSIVE` | Default. Returned when measurements cannot distinguish: no shared numeric keys, `delta == 0.0`, no held-out keys supplied (in-sample), coverage unknown, or coverage below 0.8. |

Polarity rule (read this twice): `observed_delta` is raw `post - base`. A loss
or error-like metric that fell is negative. `--metric-direction` defaults to
`lower_is_better`. Getting the polarity backwards inverts every verdict
(`src/model_headroom/paired/decision.py:141-144`).

INCONCLUSIVE-first design: `decide` with no `--coverage` and no
`--comparison` yields `INCONCLUSIVE` for every strategy, because coverage is
unknown (`src/model_headroom/paired/cli.py:356-362`). The fix is `--coverage 0.9`
or a real `--comparison` file carrying coverage. `held_out_keys` is required
for a non-INCONCLUSIVE verdict: omitting it yields `INCONCLUSIVE` even with a
good delta and good coverage, because the delta is in-sample
(`src/model_headroom/paired/decision.py:276-281`). `delta == 0.0` is
`INCONCLUSIVE`, never `SUPPORTED`
(`src/model_headroom/paired/decision.py:270-272`).

I verified each of these by execution (negative controls in section 6, claim
13-15 in the ledger).

## 8. Refusal comparison

The refusal path compares reference vs compressed vs recovered on paired
prompt sets. It lives in `src/model_headroom/paired/refusal_compare.py` and is
a separate concern from `compare`/`decide`; it is not part of the
`model-headroom-paired` CLI subcommands.

Two levels are kept separate and never combined into a single score:

- Response-level outcomes (`RefusalOutcome`,
  `src/model_headroom/paired/refusal_compare.py:49`): refusal counts and rates
  on harmful and harmless prompt sets. `compare_outcomes`
  (`src/model_headroom/paired/refusal_compare.py:161`) returns
  `candidate.refusal_rate - reference.refusal_rate`, or `None` when either is
  unmeasured.
- Direction-level geometry (`DirectionShift`,
  `src/model_headroom/paired/refusal_compare.py:62`): cosine to the reference
  direction, Cohen's d, norm ratio, orthogonalization residual per layer
  (`compare_refusal_directions`,
  `src/model_headroom/paired/refusal_compare.py:104`).

`RefusalComparison` (`src/model_headroom/paired/refusal_compare.py:77`)
carries both, plus `response_level_change`, `overrefusal_change`,
`prompt_fingerprint_match` (tri-state: True/False/None), and
`geometry_without_behaviour`.

Two explicit statements the operator must not invert:

- A shifted direction (geometry) is not proof of lost safety.
- Cosine to reference is not a quality objective.

`geometry_without_behaviour` is `True` when at least one `DirectionShift` is
measured while the response-level rate for that side is absent, unmeasured, or
`None` (`src/model_headroom/paired/refusal_compare.py:237-248`). Geometry
without behaviour is never interpreted as evidence of safety change.

## 9. Limitations

What this tool cannot do:

- No training. It never runs an optimizer step or a training loop.
- No quantization. It reads weights; it does not quantize or dequantize.
- No checkpoint repacking. It does not write model files.
- No serving. It is not an inference engine and exposes no server.
- No weight editing. It compares measurements; it does not modify parameters.
- No refusal removal. The refusal comparison observes outcomes and geometry; it
  does not alter safety behaviour.

Honest statement: no live-engine behaviour was exercised in producing this
guide. The GPU capture path (section 4) was not run here — it requires a GPU
and a loaded model. Every command that was run ran on CPU against synthetic
JSON fixtures under `/tmp`, and every artifact shown is real output from those
runs.

## 10. Verified / unverified ledger

| # | Claim | Status | Evidence (command or file:line) |
|---|-------|--------|---------------------------------|
| 1 | `model-headroom-paired schema-version` prints `1` and exits 0 | executed-here | `model-headroom-paired schema-version` → `1`, exit 0 |
| 2 | `which model-headroom-paired` resolves to `/usr/local/bin/model-headroom-paired` | executed-here | `which model-headroom-paired` → `/usr/local/bin/model-headroom-paired` |
| 3 | `SCHEMA_VERSION = "1"` | read-from-source | `src/model_headroom/paired/schema.py:27` |
| 4 | Tri-state `MeasurementStatus`: `measured` / `not_measured` / `not_applicable` | read-from-source | `src/model_headroom/paired/schema.py:29` |
| 5 | Unmeasured quantity is `null`, never `0.0`; value/rejection rules | read-from-source | `src/model_headroom/paired/schema.py:56-70` |
| 6 | `TensorKind` includes `packed_buffer` vs `effective_weight` | read-from-source | `src/model_headroom/paired/schema.py:31-38` |
| 7 | `packed_buffer` module is `not_applicable` for effective-weight error | read-from-source | `src/model_headroom/paired/compare.py:223-242` |
| 8 | Coverage = measured / comparable, `None` when unknown | read-from-source | `src/model_headroom/paired/compare.py:186-188` |
| 9 | `compare` pairs layers by `layer_name`, not position | read-from-source | `src/model_headroom/paired/compare.py:124-126` |
| 10 | `delta = post - base` | read-from-source | `src/model_headroom/analysis/intervention_record.py:48` |
| 11 | `--metric-direction` defaults to `lower_is_better`; polarity orients delta | read-from-source | `src/model_headroom/paired/cli.py:464-468`; `src/model_headroom/paired/decision.py:141-144` |
| 12 | `import model_headroom.paired.instrument` adds no torch to sys.modules | executed-here | `python -c "import model_headroom.paired.instrument; ..."` → `torch in sys.modules: False` |
| 13 | `decide` with no `--coverage`/`--comparison` → `INCONCLUSIVE` for every strategy | executed-here | `model-headroom-paired decide --baseline ... --post ... --held-out-keys ...` (no coverage) → 3 × `INCONCLUSIVE`, `coverage unknown` |
| 14 | `decide` without `--held-out-keys` → `INCONCLUSIVE` "delta is in-sample" | executed-here | `model-headroom-paired decide ... --coverage 0.9` (no held-out-keys) → `INCONCLUSIVE`, `no held-out keys supplied: delta is in-sample` |
| 15 | `decide` with `delta == 0.0` → `INCONCLUSIVE` "zero delta" | executed-here | identical baseline/post → 3 × `INCONCLUSIVE`, `zero delta: no change detected` |
| 16 | Coverage ≥ 0.8 required for `SUPPORTED` (`COVERAGE_THRESHOLD = 0.8`) | read-from-source | `src/model_headroom/paired/decision.py:45` |
| 17 | Real end-to-end run: compare → coverage 1.0; decide → `SUPPORTED`, delta -0.23 | executed-here | section 6 commands; `/tmp/paired-fixture/comparison.json` and `decision.json` |
| 18 | `verify` recomputes SHA-256 against store index; real artifact verified `true` | executed-here | `model-headroom-paired verify --store /tmp/paired-fixture/store --artifact-id paired_recovery_decision_cf2d1207028d` → `"verified": true` |
| 19 | Sample report `docs/examples/paired-recovery-report.json` is the real `decide` output; sha256 `7ef166060f09a90b06d4a3e6c1c8a8be6584f9c3b43152c1f1e6142f83832e18` | executed-here | `sha256sum docs/examples/paired-recovery-report.json` |
| 20 | No GPU or live-engine behaviour was exercised in this guide | executed-here | all commands ran on CPU against `/tmp` fixtures; section 4 steps labelled `NOT EXECUTED HERE` |

Unverified items (none claimed as fact in this guide):

- GPU capture (`PairedActivationRecorder.capture_forward` on a live model): not executed here; requires a GPU and a loaded torch model. Verify with the section 4 snippet on a GPU runtime.
- `discover_buffer_backed_modules` on a real QTEA model: read from source only (`src/model_headroom/paired/instrument.py:436`); not executed here.
- Refusal comparison end-to-end via `compare_refusal`: read from source only (`src/model_headroom/paired/refusal_compare.py`); not executed here — no prompt sets or `RefusalDirectionReport` objects were constructed.
