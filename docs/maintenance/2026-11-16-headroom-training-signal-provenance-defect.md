# Ticket: headroom R-01/R-02 computed from real training telemetry are never marked measured — decision-grade unreachable for dense models

| Field | Value |
|---|---|
| Status | **FIXED** — committed as `bf38f54`, pushed to `origin/master` 2026-11-16 |
| Severity | **HIGH** — silently caps dense-model measured coverage at 50%; `--analysis-mode decision_grade` can never pass through the CLI |
| Component | `src/model_headroom/analysis/orchestrator.py` (authority), with CLI input path in `src/model_headroom/cli.py` |
| Signals affected | R-01 `loss_saturation`, R-02 `growth_factor` |
| Found by | RockAndEgg-v3 pre-release headroom analysis (workspace `/kaggle/working/rockandegg-v3`) |
| Repository state at filing | `d079954`; pre-existing uncommitted carrier changes left untouched |
| Blocks | a faithful, decision-grade headroom measurement from training telemetry for any dense model |

## Intent this ticket serves

The operator needs a **faithful headroom analysis to decide a final finetune before release**.
`model-headroom` accepts real training telemetry via `--loss-curve` and `--isoflop`.
This ticket records that those two signals are computed but can never be counted as
*measured* evidence: `source_type` stays `unavailable`. The consequence is not a
cosmetic label — it makes the documented decision-grade gate mathematically
unreachable for a dense model, regardless of how good the supplied evidence is.

## Reproduction (executed on this machine, 2026-11-16)

Instrument-characterization probe only. The synthetic inputs below are **not**
headroom evidence and must never be quoted as such; they exist solely to make the
instrument reveal its own provenance behavior.

Files:

```json
// /tmp/opencode/sia_probe/loss.json
[{"tokens_trained":0,"validation_loss":2.0},{"tokens_trained":1000,"validation_loss":1.5},
 {"tokens_trained":2000,"validation_loss":1.2},{"tokens_trained":3000,"validation_loss":1.1},
 {"tokens_trained":4000,"validation_loss":1.05}]
```

```json
// /tmp/opencode/sia_probe/isoflop.json
[{"compute_flops":1e15,"model_params":313000000,"measured_loss":2.0},
 {"compute_flops":2e15,"model_params":313000000,"measured_loss":1.7},
 {"compute_flops":4e15,"model_params":313000000,"measured_loss":1.5}]
```

Command:

```bash
cd /kaggle/working/sia-hu
sia model-headroom --model-name PROBE \
  --model-path /kaggle/working/rockandegg-v3/export/rockandegg-313m-v3 \
  --architecture dense \
  --loss-curve /tmp/opencode/sia_probe/loss.json \
  --isoflop /tmp/opencode/sia_probe/isoflop.json \
  --output-format json --output-file /tmp/opencode/sia_probe/report.json
# exit 0
```

Per-signal fields read from the emitted report (verbatim):

```
loss_saturation          status=computed         source=unavailable  verif=unavailable
growth_factor            status=poor_fit         source=unavailable  verif=unavailable
effective_rank           status=computed         source=fixture      verif=unverified
grad_weight_ratio        status=skipped          source=unavailable  verif=unavailable
dead_neurons             status=skipped          source=unavailable  verif=unavailable
load_balance             status=single_expert    source=unavailable  verif=unavailable
specialization           status=single_expert    source=unavailable  verif=unavailable
coverage: 0.0
```

`loss_saturation.status == "computed"` proves the data was actually ingested and
the signal ran. `source == "unavailable"` proves the provenance was not carried
through. (`growth_factor` being `poor_fit` for these three synthetic points is a
separate signal-quality matter, not the defect filed here; the load-bearing
observation is the `source`/`verif` pair on both rows.)

## Root cause — two gaps on the same path

### D1 — no provenance channel in the CLI input loaders

`src/model_headroom/cli.py:634-660` parses both files into bare dataclass lists:

```python
data = json.load(f)
loss_curve = [LossDataPoint(**pt) for pt in data]
...
isoflop_points = [IsoFLOPPoint(**pt) for pt in data]
```

`LossDataPoint` / `IsoFLOPPoint` (`models/types.py:65-78`) contain only their
numeric fields. There is no nested `provenance` block, no
`--training-source-type` flag, and no equivalent of
`--weight-pairs-source-type` / `--mlp-source-type` for training telemetry.

### D2 — `_run_signal` never stamps provenance on R-01/R-02

`src/model_headroom/analysis/orchestrator.py:261-297` builds both signals through
the generic helper without provenance arguments:

```python
signals.append(_run_signal("loss_saturation", analyze_loss_saturation, loss_curve))
signals.append(_run_signal("growth_factor", analyze_growth_factor, isoflop_points))
```

`_run_signal` (`:127-176`) returns, on the success path at `:161-166`:

```python
return SignalResult(
    signal_name=name,
    status="computed",
    data=data,
    duration_seconds=duration,
)
```

`SignalResult` defaults `source_type="unavailable"`,
`verification_status="unavailable"` (`models/types.py:333-334`). D2 fires for
every caller of `_run_signal` that does not explicitly override these fields —
R-01 and R-02 are the two affected scored signals (the MLP/gradient paths at
`:355-517` do override them, which is why R-03/R-04/R-05 can reach `measured`).

D1 and D2 are independent: fixing only D1 (a provenance file) leaves D2 unable to
stamp it; fixing only D2 gives the helper nothing real to stamp from a data file.

## Impact — arithmetic, not an adjective

For a dense model, R-06/R-07 are `not_applicable` and excluded from the
applicable-weight denominator (`scoring.py:250-255`). Applicable weights
(`docs/concepts/provenance.md:33-35`):

| Signal | Weight | Can be `measured` today? |
|---|---|---|
| R-01 loss_saturation | 0.20 | **No** — D1+D2 |
| R-02 growth_factor | 0.20 | **No** — D1+D2 |
| R-03 effective_rank | 0.15 | Yes (`--weight-pairs-file` + source-type) |
| R-04 grad_weight_ratio | 0.15 | Yes (real gradients) |
| R-05 dead_neurons | 0.10 | Yes (`--mlp-activations-file` + `--mlp-source-type measured`) |
| **Total applicable** | **0.80** | |

Maximum attainable measured coverage =
`(0.15 + 0.15 + 0.10) / 0.80 = 0.50` (50%).

`_coverage_gate_gaps` (`scoring.py:305-352`) requires coverage ≥ 0.70 for
decision-grade and `cli.py` exits 2 when it is unmet. Therefore:

> For a dense model, no combination of real loss curves, real isoFLOP points,
> real gradients and real activations can produce a decision-grade report while
> D1+D2 remain. The gate is not strict — it is arithmetically unsatisfiable.

This is distinct from the correctly-refused 19%-coverage case already recorded
for the RockAndEgg-v3 checkpoint. That refusal was honest about missing
evidence; this defect refuses even after the evidence is supplied.

## Claim-proof record (anti-idiocy six-field form)

- **claim:** Supplying real loss-curve and isoFLOP telemetry yields measured
  R-01/R-02 signals and can raise dense-model coverage above the 70% gate.
- **evidence:** executed CLI probe above; `status=computed` for R-01 with
  `source_type=unavailable`. Source read of `cli.py:634-660`,
  `orchestrator.py:127-176,261-297`, `types.py:333-334`, `scoring.py:139-143`.
- **proof_adequacy:** claim property = provenance class of the emitted R-01/R-02
  signal; measured property = the `source_type` field in the emitted JSON.
  Raw data was ingested, but the provenance class stayed `unavailable`.
  `entails_claim: false` for "measured".
- **innocent hypothesis:** the withheld label is content-based — synthetic
  inputs are deliberately downgraded, real ones would be stamped `measured`.
  **Refuted** structurally: the loaders carry no provenance field at all and
  `_run_signal`'s success return has no code path that sets `source_type`, so
  the label cannot depend on input content.
- **falsification:** a data file with a nested provenance block cannot even be
  parsed by the current loader (`[LossDataPoint(**pt) for pt in data]` expects a
  list), confirming there is no channel for a real declaration.
- **verdict:** **REJECTED** as implemented. The instrument cannot attest to the
  provenance of training telemetry it has already scored.

## Proposed fix (scope: authority only)

1. Add an explicit training-telemetry provenance input to `run_analysis` —
   e.g. `training_source_type: SourceType`, `training_verification_status:
   VerificationStatus` — **defaulting to the current `unavailable` behavior**
   so this change cannot silently reclassify historical reports.
2. Extend `_run_signal` (or build R-01/R-02 explicitly) to stamp those values
   onto the `SignalResult` for `loss_saturation` and `growth_factor`, matching
   how R-03/R-04/R-05 are already stamped at `orchestrator.py:355-517`.
3. Expose a CLI route with a **non-laundering default**:
   - preferred: a `--training-provenance-file` using the same nested
     `{"provenance": {...}}` shape as `load_gradient_provenance_from_json`
     (`loader.py:459-500`), so `model_revision` / `dataset_hash` / token count
     travel with the curve; or
   - minimal: explicit `--training-source-type` / `--training-verification-status`
     flags.
   **Do not** set `measured` merely because a path was supplied — that is the
   EMO-001 class of error (fixture data laundered into a measurement).
4. Tests in `model_headroom/tests/`:
   - real-format loss/isoflop JSON **with** provenance → R-01/R-02
     `source_type == "measured"`, `verification_status == "verified"`;
   - same JSON **without** provenance → both stay `unavailable` (negative
     control);
   - with R-01…R-05 all measured, computed coverage ≥ 0.70 and
     `--analysis-mode decision_grade` exits 0;
   - no regression to the existing skipped-signal semantics.

## Acceptance criteria

1. The reproduction command above, re-run with a provenance-bearing loss curve,
   reports `source=measured verif=verified` for `loss_saturation`.
2. `growth_factor` carries the same provenance when its data is provenance-bearing
   (a `poor_fit` verdict may remain — status and provenance are orthogonal).
3. Dense-model coverage can reach ≥ 0.70 from five genuinely measured signals;
   a report with only three measured signals still reports 50% and exits 2.
4. `ruff check`, `mypy`, and the `model_headroom` test suite pass.

## Dedup and scope notes

- **Not the same as D-1** in `docs/model-headroom/PROJECT-ROADMAP.md:198`: D-1 is
  "0.00 composite with no evidence supplied". This ticket is "evidence supplied
  and scored, but its provenance is discarded". Different code path, different
  fix.
- **Not Phase 8** of the roadmap (`:962-965`): that phase targets R-01 returning
  `computed` from real loss data. That part works; what is missing is the
  provenance stamp.
- **Not fixed here:** `--acquire-prompts` activation capture via
  `AutoModelForCausalLM` (fails on headless backbones); the safetensors template
  name-resolution defect D-4; gradient-provenance identity checks. Each is a
  separate need.
- The synthetic probe files live in `/tmp/opencode/sia_probe/` and are diagnostic
  instrumentation only. No claim in this ticket rests on their numeric values.

## References

- `src/model_headroom/analysis/orchestrator.py:127-176,261-297,355-517`
- `src/model_headroom/cli.py:634-660,812-870`
- `src/model_headroom/models/types.py:65-78,312-335`
- `src/model_headroom/analysis/scoring.py:139-143,224-297,305-352`
- `docs/concepts/provenance.md:14-46`
- `docs/concepts/headroom-signals.md:30-58`
- `docs/maintenance/2026-11-13-ornith-headroom-pitfalls.md` (prior headroom blocker record)
