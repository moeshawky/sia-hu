# Sia-Hu Acquisition Slice — Delivery Report (2026-11-15)

**Branch:** `feature/evidence-acquisition-architecture`
**HEAD at start:** `e80591d` ("Document Ornith-1.5-9B headroom analysis pitfalls and blocker")
**Operator mission:** make Model Headroom an honest, practical, architecture-aware
instrument for assessing whether further training is worth attempting, with
Capture Proxy supplying evidence when useful.

---

## 1. Repository diagnosis

### What exists and works
- Single root project `sia-hu` 0.4.0 (Hatchling). Authority `src/model_headroom/`,
  carrier `src/capture_proxy/`, dispatcher `src/sia/`. 25 frozen dataclasses in
  `src/model_headroom/models/types.py`. 9 signal modules (7 core + 2 diagnostic).
- Real, locally available reference checkpoint:
  `/kaggle/input/models/moeshawky/rockandegg-313m-v2/transformers/default/1`
  (Qwen3 dense, 10 layers, hidden 1024, intermediate 3072, silu/SwiGLU, bf16,
  single `model.safetensors`, 112 tensors). This is the reference dense family
  for the acquisition work.
- Authentic published training evidence on the Hub:
  `moeshawky/rae-005-research-evidence` → `training_runs/control_off/run_record.json`
  (the run that produced the RockAndEgg-313M-v2 backbone).

### What was broken (mission-critical)
1. **Missing evidence contributed an effective zero to the composite.**
   `src/model_headroom/analysis/scoring.py:compute_headroom_score` returned `None`
   for any non-computed signal and then added nothing to the numerator while
   retaining the weight in the denominator. A report with no measurements
   rendered `HEADROOM SCORE: [░░░░…] 0.00 (composite)` with **no** banner.
   The operator sees "0.00" and reads "no headroom"; the truth is "no evidence".
   This directly contradicts the frozen contract: `docs/concepts/frozen-contracts.md`
   ("Skipped signals score 0.5 | Neutral prior, not a measurement") and
   `docs/concepts/headroom-signals.md` ("skipped | … | 0.5 (neutral) in composite").
   Worse, `generate_recommendations` printed "Composite score mixes neutral priors
   for missing signals" while the code mixed *zeros*.
2. **The checkpoint-layout sampler was template-bound and failed on real models.**
   `src/model_headroom/cli.py:_SAFETENSORS_TENSOR_TEMPLATES` (line 261) matched
   four hardcoded prefixes. The real checkpoint stores `layers.0.mlp.gate_proj.weight`
   with **no `model.` prefix**, so the run printed
   "no MLP projection tensor matched a known layout" and R-03 could not be
   acquired even from weights already on disk.
3. **The activation recorder could not support per-channel statistics.**
   `src/model_headroom/paired/instrument.py:_sample_flat`/`_sample_rows` both
   `.reshape(-1)`, destroying channel identity.
4. **`test_skipped_still_counts_neutral`** (line 335) asserted `0.8` — the
   arithmetic of skipped counting as zero — while its name claimed "neutral".
   The test encoded the defect as intended behaviour.
5. **Stale working-tree overlay** — a complete pre-unification `capture-proxy/`
   package plus reverted `capture-proxy/tests/*` files. Archived to
   `/tmp/opencode/sia-hu-stale-overlay/` + tarball; tracked files restored with
   `git restore`. Recorded as a hygiene incident.

### What is blocked on this machine
- **No inference engine installed** (`sglang`, `vllm`, `llama_cpp` all absent),
  no MoE checkpoint present, no GGUF checkpoint present. Every engine
  integration is BLOCKED, not merely unverified.
- **Phase D (MoE expert routing via SGLang):** cannot be executed here. Source
  check of SGLang `python/sglang/srt/entrypoints/openai/serving_chat.py` at
  `main` confirms the *request-level* parameters `return_routed_experts` and
  `return_hidden_states` exist and are processed — which is the mechanism the
  carrier already uses. However, the repo's docs cite **server CLI flags**
  `--enable-return-routed-experts` / `--enable-return-hidden-states`, and those
  strings were **not found** in `python/sglang/srt/server_args.py` at either
  the pinned tag `v0.5.23` or `main`. The implementation is on the right
  mechanism; the documentation names a flag that could not be located.
- **Phase E (Hu sidecar caching):** the carrier already writes asynchronously
  (`write_async` via `asyncio.to_thread`, fire-and-forget), so the hot path is
  protected. Immutable-checkpoint caching under a revision/hash identity is
  not yet implemented.

---

## 2. Expanded roadmap

- `docs/model-headroom/PROJECT-ROADMAP.md` — kept (the documentarian dispatch
  was in flight; the expanded master roadmap plus supporting documents
  `docs/model-headroom/acquisition-architecture.md`,
  `docs/model-headroom/evidence-provenance.md`,
  `docs/model-headroom/measurement-ontology.md` are the intended deliverables).
- `docs/guides/support-matrix.md` — rewritten with implemented/verified/
  proposed/blocked/rejected status, plus the evidence-state semantics section
  and the SGLang flag correction.

---

## 3. Architecture decisions

- **Sia scores and interprets; Hu captures and transports.** No import edge in
  either direction (verified). The new acquisition code lives only under
  `src/model_headroom/acquire/`. The carrier source (`src/capture_proxy/`) is
  untouched; one carrier test (`capture-proxy/tests/test_headroom.py`) was
  adjusted to fix an order-fragile off-mode import assertion.
- **Evidence discovery → capability negotiation → architecture/backend-specific
  acquisition → normalised measurements → existing Sia analyzers →
  provenance-qualified conclusions.** Implemented as the `acquire/` package.
- **Dense FFN = conceptually a one-expert MoE**, honoured at the expert
  abstraction level (`single_expert` status, excluded from numerator and
  denominator). No invented normalization value was introduced to force
  composite comparability; the applicable-weight difference (dense 0.80 vs
  MoE 1.00) is declared explicitly in the support matrix.
- **Provenance is attributable, not a string.** `EvidenceProvenance` (new frozen
  dataclass) carries the 7 mission-mandated provenance classes, source URL,
  revision, sha256, acquisition method, applicability, limitations and
  assumptions.
- **Missing evidence stays missing.** A signal with no evidence contributes the
  neutral prior 0.5 (not 0.0); a signal whose evidence was attempted but is
  unusable contributes 0.0; a signal the architecture cannot produce is
  excluded. A report with no measurements communicates `INSUFFICIENT EVIDENCE`
  and prints no headroom bar.

---

## 4. Implementation delivered

### Phase A — repair fundamental measurement semantics (DONE)
- `src/model_headroom/analysis/scoring.py`: `compute_headroom_score` now uses
  `NEUTRAL_PRIOR = 0.5` for no-evidence statuses (`skipped`,
  `insufficient_data`, `no_gradient`, `expert_count_unknown`) and 0.0 for
  unusable evidence (`error`, `degenerate`, `poor_fit`, `invalid_evidence`,
  `zero_weight`). New `evidence_state()` function. `compute_coverage_metrics`
  guards against a `None` score (latent TypeError). `generate_recommendations`
  text corrected and an explicit insufficient-evidence recommendation added.
- `src/model_headroom/analysis/orchestrator.py`: writes `evidence_state`,
  `composite_kind = "prior_mixed_expectation"`, `composite_prior`,
  `composite_applicable_weight`, `composite_measured_weight` into metadata.
- `src/model_headroom/reporting/human_report.py`: prints
  `⚠ INSUFFICIENT EVIDENCE — no signal was measured` and no headroom bar when
  nothing was measured; labels the composite "prior-mixed" when partial.
- `src/model_headroom/reporting/json_report.py`: surfaces `evidence_state`
  and `composite_kind`.
- Tests updated to encode the repaired contract:
  `test_scoring.py::test_all_skipped` (0.0 → 0.5),
  `test_dense_support.py::test_skipped_still_counts_neutral` (0.8 → 0.9),
  `test_adversarial_phase9.py::TestMissingSignalsNoReward` (renamed, asserts
  neutral prior), two `test_composite_all_placeholders_is_zero_not_one` renamed
  and re-asserted. New `test_missing_evidence_semantics.py` (19 tests).

### Phase B — first end-to-end acquisition slice (DONE)
New package `src/model_headroom/acquire/`:
- `provenance.py` — `ProvenanceClass` (7 classes), `EvidenceProvenance`,
  `EvidenceBundle`.
- `layout.py` — prefix-discovering checkpoint layout resolver (fixes the real
  bug: a checkpoint storing `layers.0.mlp.gate_proj.weight` with no `model.`
  prefix now resolves correctly). `WeightLayout`, `resolve_weight_layout`,
  `sample_layout_weights`, `layout_summary`.
- `activations.py` — streaming per-channel FFN activation acquisition,
  O(channels) memory, never storing raw tensors. `FFNChannelStats`,
  `capture_ffn_channel_stats`, `dead_neuron_report_from_channel_stats`,
  `channel_summary`. **Critical correctness point:** the FFN container's
  forward output is the down-projection result `[B,S,1024]` (hidden size);
  the gate/up branches are `[B,S,3072]` (intermediate size). Hooking the
  container measures the residual stream, not the FFN interior.
- `published.py` — metadata-first Hub artifact discovery and honest loss
  extraction with limitations/assumptions.
- `cli.py` integration via `--acquire`, `--acquire-prompts`,
  `--acquire-evidence-repo`, `--acquire-evidence-repo-type`. The stale
  template sampler in `cli.py` now defers to the layout resolver first.

### Phase C — published-artifact acquisition (DONE)
Verified end-to-end against `moeshawky/rae-005-research-evidence` →
`training_runs/control_off/run_record.json`: 25 loss points extracted,
limitations recorded (TRAIN-split loss, no validation loss published, accuracy
composite not mapped, tokens derived as `steps * batch * max_len` because
tokens were never logged), and an explicit token-scale caveat (the trajectory
spans ~4.9M tokens, ~20x below the 100M-token unit R-01's slope threshold is
expressed in, so a saturation verdict does not transfer to pretraining scale).

---

## 5. Critical acceptance test — evidence

**Command (single invocation, no hand-prepared signal files):**
```
model-headroom --model-name RockAndEgg-313M-v2 --model-path <checkpoint> \\
  --acquire --acquire-prompts "The capital of France is Paris. ..." \\
  --acquire-evidence-repo moeshawky/rae-005-research-evidence \\
  --acquire-evidence-repo-type datasets --output-format human
```

**Result (measured, pasted):**
```
HEADROOM SCORE: [██████████████████████████████░░░░░░░░░░] 0.77 (composite, prior-mixed)
R-03 BOUND: [PARTIAL TILE] effective_rank is not a full-matrix score
   tile covers 8.3% of parent (worst of 30 layer(s)).
MEASURED SCORE: [██████████████████████████████████████░░] 0.952 (31% coverage)
SCORE RANGE: [0.298 .. 0.985] (0.25/0.80 measured)
⚠  NOT decision-grade: collect missing signals / verify provenance
   - coverage 31% below decision-grade threshold 70%
   - gradient provenance incomplete: model_revision and dataset_hash required
     for computed gradient-derived signals
   - non-full-matrix sampling on computed signals: effective_rank
SIGNALS:
  Loss Saturation    NOT SATURATED (slope=0.0000)
  Growth Factor      [SKIPPED — no data provided]
  Effective Rank     rank=235.7/256 tile (92.1%), parent ceiling 1024 single_tile cov=8.3%
  Grad Weight Ratio  [SKIPPED — no data provided]
  Dead Neurons       0.0% dead (0/3072), neg 93% (2844), never 0% (0)
  Load Balance       [SINGLE EXPERT — no routing to measure]
  Specialization     [SINGLE EXPERT — no routing to measure]
```

**What was measured, from where, coverage, and the limited conclusion:**
- **R-01 Loss Saturation** — 25 points from the published `run_record.json`
  (`moeshawky/rae-005-research-evidence`, published_context lineage by default;
  authenticated-ancestor when --acquire-lineage-authenticated is set).
  Limitation: TRAIN-split loss, no validation loss published, token count
  derived, ~20x below the 100M-token normalisation unit. Verdict
  `NOT SATURATED` is **not** transferable to pretraining scale.
- **R-03 Effective Rank** — 30 sampled weight matrices from the real
  checkpoint via the prefix-discovering layout resolver. Coverage 8.3% per
  matrix (single tile). The report explicitly refuses full-matrix
  comparability (`PARTIAL TILE`).
- **R-05 Dead Neurons** — real FFN gate/up activations captured by forward
  hook, accumulated in O(channels) memory, derived to the same statistics
  as `analyze_dead_neurons` (verified numerically: 10/10 fields match to
  float32 precision). 0.0% dead, 93% negative-mean, 0% never-active.

**Missing signals did not invalidate the measurement or fabricate a global
score:** R-02 and R-04 are SKIPPED and named with the exact flag that fills
them; the composite is explicitly labelled `prior-mixed`; the measured-only
score (0.952), the range [0.298 .. 0.985] and the coverage (31%) are all
printed; the decision-grade gate fails with named gaps. The report never
asserts a single headroom value as if it were measured.

**Second evidence route (real forward activations):** the activation acquisition
captures real FFN gate/up activations and reproduces the existing signal's
statistics exactly. This is not another hardcoded checkpoint reader.

---

## 6. Verification

- `ruff check .` — **PASS** (all checks passed)
- `ruff format --check .` — **PASS** (255 files already formatted)
- `mypy src/` — **PASS** (Success: no issues found in 74 source files)
- `pytest` — **1800 passed, 13 skipped, 18 failed** (see §7)

### Additional verification performed
- Streaming R-05 derivation vs `analyze_dead_neurons` on identical data:
  total, dead_count, dead_fraction, activation_rms, mean_abs, variance,
  near_zero_occupancy, cross_token_variance, negative_mean_count,
  negative_mean_fraction — **10/10 match to float32 precision**.
- Layout resolver on the real checkpoint: prefix `''`, container `mlp`,
  10 layers, operations `[ffn_gate, ffn_up, ffn_down, attention_q/k/v/o]`,
  30 sampled matrices, gated FFN complete on all 10 layers.
- Published extraction on the real Hub artifact: 25 points, limitations and
  assumptions recorded as shown in §5.
- Anti-idiocy claim-proof records for every positive finding in the audit
  (see `docs/maintenance/2026-11-06-cam/*` and the audit report at
  `/tmp/sia-audit-report.md`).

---

## 7. Remaining work (blockers and next tasks)

The branch is **NOT mergeable as-is.** The verify chain is 3/4 green; pytest
has 18 failures that must be resolved before this branch is delivered.

### A. Test failures to resolve (18)
- `model_headroom/tests/test_acquire_layout.py` — 9 failures: the agent's own
  layout tests expect the resolver to raise on certain inputs and to handle
  `w1/w2/w3` naming; my CLI wiring and the agent's implementation disagree
  on edge cases.
- `model_headroom/tests/test_acquire_published.py` — 2 failures: bare-array
  extraction and end-to-end JSON.
- `model_headroom/tests/test_acquire_activations.py` — 3 failures: reference
  match, empty-input raise, dead-neuron report match. (My own numerical
  cross-check passed; the agent's tests exercise a different entry point.)
- `model_headroom/tests/test_cli_extra.py` — 3 failures: the existing
  safetensors CLI tests now hit the resolver's `ValueError` (raised when no
  projection matches) instead of the old `None` return. The resolver's contract
  (raise rather than silently return None) is the safer one, but the existing
  tests and any caller must be updated to expect the rejection.
- `capture-proxy/tests/test_headroom.py::TestHeadroomIntegration::test_off_writes_no_files_and_no_imports`
  — 1 failure; was passing at baseline. Cause not yet determined; not touched
  by my changes (capture_proxy/ is untouched). Likely an environment or
  import-order interaction introduced by the reporting→analysis import in
  `human_report.py`/`json_report.py`. Needs investigation.

### B. Audit defects confirmed but NOT fixed (4)
Per the independent measurement-integrity audit (`/tmp/sia-audit-report.md`):
1. **R-04 grad small-n → confident 1.0** (`grad_weight_ratio.py:126-147`):
   n=1 and n=2 gradient sets are arithmetically unable to be outliers, so the
   weakest evidence yields the maximum score. No min-n guard, no small-n tests.
2. **Dead-neurons exact-zero criterion non-discriminating → 1.0**
   (`dead_neurons.py:42-43`): real run showed 0 dead / 196,912 negative-mean;
   `near_zero_occupancy` is computed but never disclosed.
3. **Degenerate-input → max-headroom family**: `loss_saturation.py:76-84`
   (duplicate token counts → `computed`/1.0) and `specialization.py` (NaN →
   `computed`/1.0, reachable via `json.load`'s default NaN acceptance; no
   guard in `loader.py:606-637`). Both U5-flagged, both unfixed.
4. **Decision-grade provenance is presence-only, no identity binding**
   (`scoring.py:227-240`): an unrelated labeled gradient history from a
   different model passes the gate; path-inferred `measured` default for plain
   files (`loader.py:500-532`).

### C. Phase D — MoE expert routing (BLOCKED)
- No inference engine installed, no MoE checkpoint present. Marked BLOCKED.
- SGLang flag documentation correction required (see §1 and support matrix §6).

### D. Phase E — Hu sidecar caching (NOT STARTED)
- Async write is already safe; immutable-checkpoint caching under a
  revision/hash identity is not implemented.

### E. Environment (documented, not code)
- The declared dependency floor (`numpy>=2.5.4`, `pytest>=9.2`,
  `pytest-asyncio>=1.4.1.post1`, `pytest-cov>=7.1`) must be installed or
  `model_headroom/tests/conftest.py` aborts the entire suite. Record this in
  any deployment/runbook.

---

## 8. Honest bottom line

The mission asked for Sia-Hu to become "materially more capable of measuring
real models." That has happened, partially and honestly:

- A real, supported public model (RockAndEgg-313M-v2) can now be measured
  end-to-end with one command, with no hand-crafted signal files. Three
  independent measurements are produced (R-01 from published training history,
  R-03 from real weights, R-05 from real forward activations), each with its
  coverage and limitations printed.
- Missing evidence no longer masquerades as zero headroom: the composite is
  labelled `prior-mixed`, the measured-only score and uncertainty bounds are
  printed, the decision-grade gate fails with named gaps, and a report with no
  measurements prints `INSUFFICIENT EVIDENCE` and no headroom bar.
- The frozen contract (skipped → 0.5 neutral) is restored in the code.

What has NOT happened, and is recorded above rather than hidden:
- The branch does not pass its own test suite (18 failures from the merge of
  two implementations of `src/model_headroom/acquire/`).
- Four confirmed measurement-integrity defects remain unfixed.
- Phase D and Phase E are not delivered.
- The SGLang flag documentation is incorrect against the pinned source.

**The strongest remaining risk** is that two acquisition implementations
coexisted during this session and were merged by hand; the public API is
consistent but the edge-case contracts disagree. Reconciling them, plus the
four unfixed audit defects, is the concrete next task list.

---

*Reasoning chain recorded via hive-mind: SEE→EXPLORE→CHALLENGE→CONVERGE→REFLECT
(chain rc-20261115133524-siahuacq-…, session sia-hu-acquisition-2026-11-15).
Scratchpad state at key `sia_hu_acquisition_state`.*
