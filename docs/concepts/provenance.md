# Provenance — measured vs synthetic data

**Purpose:** Define the source_type / verification_status taxonomy used by
model_headroom's scoring, how measured data differs from synthetic/fixture/estimated
data, and how the measured headroom score is separated from the legacy composite.
Cross-references `architecture/contract.md` (2-D pair contract) and
`concepts/headroom-signals.md` (signal reference). No Python code changes are
proposed here.

---

## Source types

Every data input to scoring carries a `source_type` that classifies its provenance:

| Source type | Meaning | How inferred | Can be overridden |
|-------------|---------|--------------|-------------------|
| `measured` | Data from real measurement (real checkpoint, real routing) | Inferred from file paths / real-I/O loaders | `--weight-pairs-source-type measured` |
| `synthetic` | Algorithmically generated test data | Inferred from file paths | `--weight-pairs-source-type synthetic` |
| `fixture` | Test fixture data | Inferred from file paths | `--weight-pairs-source-type fixture` |
| `estimated` | Estimated/derived, not directly measured | Inferred from file paths | `--weight-pairs-source-type estimated` |
| `imported` | Imported from another system | Inferred from file paths | `--weight-pairs-source-type imported` |

**Verification status** is orthogonal: it records whether the data has been
independently verified against a real artifact.

---

## Measured headroom score vs legacy composite

model_headroom produces two scores:

1. **Legacy composite** — weighted average over applicable signals with fixed weights:
   - loss_saturation 0.20, growth_factor 0.20, effective_rank 0.15,
   - grad_weight_ratio 0.15, dead_neurons/load_balance/specialization 0.10 each
   - Skipped signals score 0.5 (neutral); errors score 0.0; `not_applicable` excluded

2. **Measured headroom score** (`compute_coverage_metrics`) — separates
   verified-measured signals from synthetic/fixture/estimated/unverified ones and
   reports:
   - `measurement_coverage` — fraction of signals that are measured
   - `headroom_score_measured` — score computed only from measured signals
   - `score_lower_bound` / `score_upper_bound` — uncertainty bounds

**Decision-grade** (`--analysis-mode decision_grade`) requires ≥70% measured coverage
and exits with code 2 if unmet.

---

## Why provenance matters

A `0.50` composite with 0% coverage is the neutral prior over zero measured signals
— not a headroom finding. This was the EMO-001 failure mode: the sidecar wrote no
weight pairs, the downstream analyzer printed a confident-looking `0.50`, and
nothing in the shell caught it. The `MEASURED SCORE ... (0% coverage)` line and the
per-signal `[SKIPPED]` markers are the load-bearing parts of that output.

**Rule:** Never report a bare composite without coverage. Always report `MEASURED
SCORE` and coverage, and use `--analysis-mode decision_grade` when a decision-grade
answer is required.

---

## Cross-references

- `architecture/contract.md` — 2-D pair contract, signal names
- `concepts/headroom-signals.md` — R-01 through R-09 signal reference
- `concepts/frozen-contracts.md` — frozen dataclasses carrying source_type
- `guides/model-headroom.md` — `--weight-pairs-source-type`, `--analysis-mode`
- `reference/cli-flags.md` — `--analysis-mode`, `--weight-pairs-source-type`,
  `--mlp-source-type`
