# PROJECT-ROADMAP.md — model_headroom (Sia) — Master Roadmap

**Rewritten 2026-11-15 on branch `feature/evidence-acquisition-architecture`.
Baseline: HEAD `e80591d` on that branch.**

**History policy (operator-mandated):** this is an *extended restructure*, not a
rewrite of history. Completed Phases 1–5 and 11, Future Phases 6–10 and 12, the
A-1..A-5 decision record, the dead-surface notes, Scope-Creep Defenses,
Controlled-Evolution Path, and Drift Checkpoints are preserved below (§ 21–26).
Stale numbers in the old "Current State" block are marked **SUPERSEDED** with the
measured replacement.

**Status of concurrent work:** at the time of writing, a concurrent agent is
modifying `src/model_headroom/` (scoring, orchestrator, CLI, reporters) and
adding `src/model_headroom/acquire/` on this same branch. Everything below that
depends on that work is labeled **IN FLIGHT / UNVERIFIED** and cites HEAD
`e80591d` line numbers, not the moving working tree.

---

## Table of contents

| § | Section | Covers operator-mandated item |
|---|---------|-------------------------------|
| 1 | Mission & non-goals | 1 |
| 2 | Status legend & verification discipline | 27 |
| 3 | Current state — measured 2026-11-15 | baseline, env requirement, hygiene incident |
| 4 | Confirmed defects D-1..D-7 | the mission-critical record |
| 5 | Historical architectural decisions A-1..A-5 | 2 |
| 6 | Measurement ontology | 3 |
| 7 | Signal-by-signal evidence contracts (R-01..R-09) | 4 |
| 8 | Evidence provenance & applicability | 5 |
| 9 | Acquisition architecture (9.1–9.11) | 6, 7, 8, 9, 10, 11, 12, 16, 17, 18, 19 |
| 10 | Dense as a single-expert MoE | 13 |
| 11 | MoE routing instrumentation | 14 |
| 12 | Hybrid / recurrent considerations | 15 |
| 13 | Error semantics & insufficient-evidence behaviour | 21 |
| 14 | Scientific calibration & predictive validation | 20 |
| 15 | Backward compatibility & migration | 22 |
| 16 | Real-model integration coverage | 23 |
| 17 | Milestones, dependencies, falsifiable exit criteria | 24 |
| 18 | Deferred research & rejected alternatives | 25 |
| 19 | Known limitations & resource requirements | 26 |
| 20 | Documentation drift corrections (measured) | honesty rules |
| 21 | Completed Phases (preserved) | 28 |
| 22 | Future Phases 6–10, 11, 12 (preserved, status-annotated) | 28 |
| 23 | Evidence base — Qwen3.8-27B (preserved) | 28 |
| 24 | Scope-Creep Defenses (preserved) | 28 |
| 25 | Controlled-Evolution Path (preserved) | 28 |
| 26 | Drift Checkpoints (preserved) | 28 |
| A | Appendix — measurement log | audit trail |

Supporting documents (created with this roadmap):

- `model-headroom/acquisition-architecture.md` — family adapters, backend
  adapters, capability negotiation, published-evidence discovery, activation
  extraction & replay, streaming statistics, budget, sidecar caching.
- `model-headroom/evidence-provenance.md` — the 7 provenance classes,
  attribution rules, "what this evidence does NOT establish", revision
  identity, calibration.
- `model-headroom/measurement-ontology.md` — the five ontology classes,
  never-conflate rules, dense-as-single-expert math, routing instrumentation,
  hybrid/recurrent.

---

## 1. Mission & non-goals

**Sia is a diagnostic instrument for training headroom.** It reads what a
checkpoint is (structure), what it does on declared inputs (activity), and what
its published or operator-supplied telemetry says about its optimization
trajectory — and assembles a *prior-weighted, coverage-declared* headroom
estimate with honest error semantics. The authority (`model_headroom`) scores;
the carrier (`capture-proxy`) emits; the carrier never scores and the authority
never proxies (root `AGENTS.md` Iron Law).

Explicit non-goals (restated so the mission cannot drift into them):

1. **Not a universal estimator of "unused intelligence."** No such observable
   is defined; a headroom score is a diagnostic ordering over declared
   evidence, not a probability of a latent quantity. An *absence of signals*
   is not evidence of zero headroom — it is insufficient evidence (defect D-1
   is the measured instance of this conflation).
2. **Not a training framework.** No optimizer, no training loop. Training
   telemetry is *input*, never produced. The authority runs forward (and, for
   R-04 evidence, backward) passes for measurement; it does not train.
3. **Not an inference-engine competitor.** It does not serve, does not beat
   vLLM/SGLang/llama.cpp to anything. Serving engines are *evidence sources*,
   subject to their hard constraints (§ 9.2).
4. **Not an unbounded interpretability platform.** The signal set is the fixed
   R-01..R-09 plus the standalone refusal-direction capability (Phase 11
   precedent: standalone, non-scoring, zero coupling to the composite). New
   signals require operator request + coverage proof + adversarial test design
   (§ 25).

---

## 2. Status legend & verification discipline

Every status claim in this document set is one of:

| Label | Meaning |
|-------|---------|
| **IMPLEMENTED** | Code exists at HEAD `e80591d` and runs. *A branch in the code is not verification.* |
| **VERIFIED** | IMPLEMENTED **and** executed against a real model / real engine on this machine, with the measurement recorded (date + artifact or doc reference). |
| **PROPOSED** | Designed, not yet implemented or not yet executed. |
| **BLOCKED** | Cannot proceed here for a stated physical or environmental reason. |
| **REJECTED** | Considered and declined; the reason is recorded (§ 18). |
| **IN FLIGHT** | Uncommitted concurrent work on this branch; treated as UNVERIFIED until committed and the full chain is green. |
| **UNKNOWN** | Not measured; the command that would measure it is stated. |

Discipline rules:

- Never write that a backend integration works unless it was **executed**.
- Carrier-side verifications (e.g. SGLang routing emission verified 2026-10-09
  *by the carrier*) do not transfer to authority-side consumption.
- Single-sample inference telemetry is never quoted as a model-level claim
  (root `AGENTS.md` BANNED list).
- Technique names (participation ratio, randomized SVD, sketching) are cited
  by role only; no paper citation is asserted in this document set unless
  verified, and none currently is.

---

## 3. Current state — measured 2026-11-15

### 3.1 Verified baseline (HEAD `e80591d`, branch `feature/evidence-acquisition-architecture`)

| Gate | Result |
|------|--------|
| `ruff check .` | PASS |
| `ruff format --check .` | PASS (242 files) |
| `mypy src/` | PASS (69 source files) |
| `python3 -m pytest -q` | **1753 passed, 13 skipped**, 0 failed |

Single root project `sia-hu` (one `pyproject.toml`, one toolchain;
`testpaths = ["capture-proxy/tests", "model_headroom/tests", "tests"]`).
Verify chain from repo root: `ruff check . && ruff format --check . && mypy src/ && pytest`.

### 3.2 Environment requirement (measured)

The baseline passes only after installing the **declared dependency floor**:
`numpy>=2.5.4`, `pytest>=9.2`, `pytest-asyncio>=1.4.1.post1`, `pytest-cov>=7.1`.
The image shipped `numpy 2.1.3` and **no `pytest-asyncio`**, which **aborts the
suite at collection** via `model_headroom/tests/conftest.py` (the guard exists
so async `UpstreamClient` tests cannot silently drop from coverage). Record any
environment without the floor as *unmeasured*, not green.

### 3.3 Measured surface (HEAD `e80591d`)

| Item | Measured value |
|------|----------------|
| Package version (`pyproject.toml`) | `0.4.0` |
| Frozen dataclasses in `src/model_headroom/models/types.py` (407 lines) | **25** (`@dataclass` count = `frozen=True` count = 25) |
| CLI flags in `src/model_headroom/cli.py` | **23** `add_argument` calls |
| `SignalResult.status` literal (`types.py:309-322`) | 12 values: `computed, skipped, not_applicable, single_expert, expert_count_unknown, error, insufficient_data, degenerate, poor_fit, zero_weight, no_gradient, invalid_evidence` |
| `SourceType` literal (`types.py:14-25`) / loader acceptance set (`loader.py:445-448`) | 10 type values / **6 accepted externally** (`measured, synthetic, fixture, estimated, imported, unknown`) |
| Carrier `CaptureRecord.schema_version` / `CapturedHeadroom.schema_version` (`src/capture_proxy/models.py:121` / `:168`) | **6** / **7** |
| Signal-name vocabulary | Carrier `SIGNAL_KEYS` (6 names, `names.py:118`, stdlib-only) vs authority's 7 scored names as **scattered string literals** (scoring.py weights table, orchestrator.py, human_report.py) — **no compile-time link** (the seam at risk, § 9.1 of `acquisition-architecture.md`) |
| Authority ↔ carrier import boundary | **CLEAN** by executed probe 2026-11-15: no import edge either direction; the carrier computes no score, ratio, rank, entropy, cosine, or normalized quantity — raw weight matrices, raw routing counts, byte accounting, structural facts only; carrier torch/safetensors imports are lazy |

### 3.4 SUPERSEDED metrics (old "Current State (Verified 2026-09-26)" block)

| Old claim | Measured replacement | Note |
|-----------|----------------------|------|
| "sia-hu 0.2.0" | `0.4.0` | `pyproject.toml` |
| "21 frozen dataclasses" | **25** | `types.py` count at HEAD |
| "pytest: 426 passed, 16 skipped" | **1753 passed, 13 skipped** | single root suite; the 426 figure predates unification and the carrier/authority test growth |
| "CLI entry (19 args)" | **23** flags | `cli.py` at HEAD |
| "23 source files, 42 test files, ~3,977 LOC" | 69 source files under `src/` (mypy); suites as above | the per-package metric block no longer matches the unified tree; the support matrix and this roadmap are the current accounting |

### 3.5 Hygiene incident (recorded) — stale pre-unification overlay

The working tree of this branch arrived carrying a stale pre-unification
overlay: a complete old copy of the `capture-proxy/` package (untracked) plus
modifications to tracked `capture-proxy/tests/*` files that reverted them to
pre-`gradient_status` behavior. Evidence it was debris, not operator work: the
overlay's `capture_proxy/headroom.py` **lacks** `gradient_status`, which HEAD's
`src/capture_proxy/headroom.py` **has**. The overlay was archived to
`/tmp/opencode/sia-hu-stale-overlay/` (+ tarball) and the tracked files were
restored with `git restore`.

**Branch-start precaution (now standing):** before starting work on a branch,
`git status --short` and diff any unexpected untracked package copies against
HEAD; an overlay that lags HEAD's invariants is debris until proven otherwise.

---

## 4. Confirmed defects (measured by execution, 2026-11-15)

Reproduced against the real checkpoint
`/kaggle/input/models/moeshawky/rockandegg-313m-v2/transformers/default/1`
(Qwen3 dense, 10 layers, hidden 1024, intermediate 3072, SwiGLU/silu, bf16,
single `model.safetensors`, 112 tensors, `config.architectures: ["Qwen3Model"]`).

| ID | Defect | Location at HEAD | Consequence |
|----|--------|------------------|-------------|
| **D-1** | The analyzer produces a headline `0.00` with a real checkpoint present | `model-headroom --model-name … --model-path …` → `HEADROOM SCORE: 0.00 (composite)`, `MEASURED SCORE: 0.000 (0% coverage)`, all 7 signals SKIPPED | A reader sees "0.00" and concludes "no headroom" when the truth is "**no evidence**". **Mission-critical.** R-01/R-02/R-03/R-04/R-05 all require hand-prepared JSON inputs the CLI does not synthesize from the checkpoint it was pointed at |
| **D-2** | Missing signals contribute an **effective zero** to the composite | `analysis/scoring.py:116-128` (`compute_headroom_score`): `_score_signal` returns `None` for any status ≠ `computed`; the numerator adds nothing but the **denominator retains the weight**, so each skipped signal pushes the composite toward 0 | **Contradicts the documented frozen contract** — `concepts/frozen-contracts.md` ("Skipped signals score 0.5 \| Neutral prior, not a measurement") and `concepts/headroom-signals.md` ("skipped \| … \| 0.5 (neutral)") both specify 0.5. The test that should catch it is itself wrong at HEAD: `model_headroom/tests/test_dense_support.py:335` (HEAD) asserts `approx(0.8)` — the zero-prior arithmetic — under the name `test_skipped_still_counts_neutral` (name contradicts assertion) |
| **D-3** | The recommendation text is factually false | `analysis/scoring.py:341-343`: "Composite score mixes neutral priors for missing signals" | The code mixes zeros, not neutral priors. A report claiming a semantic the code does not have is worse than no report |
| **D-4** | The checkpoint-layout resolver is template-bound and fails on a real model | `cli.py:261-277` `_SAFETENSORS_TENSOR_TEMPLATES`: four hardcoded prefixes (`model.language_model.layers.{layer}.mlp.{proj}_proj.weight`, `model.layers.…`, `model.layers.…mlp.experts.0.…`, `block_sparse_moe…w1/w2/w3`) | The real RockAndEgg checkpoint stores `layers.0.mlp.gate_proj.weight` (no `model.` prefix) → run prints `warning: no safetensors weight samples … no MLP projection tensor matched a known layout`; **R-03 cannot be acquired even from weights already on disk** |
| **D-5** | Architecture misidentification | same run reports `Subtype: hybrid_linear_attention [full_attn]` | for a Qwen3 model whose `config.json` lists `layer_types` = 10× `full_attention`. The hybrid detector (VERIFIED on ornith-1.5-9b) is misbehaving on this config |
| **D-6** | The activation recorder cannot support per-channel statistics | `paired/instrument.py:171-207` — `_sample_flat` and `_sample_rows` both `.reshape(-1)` their tensors | Channel identity is destroyed at record time. Per-channel RMS, near-zero occupancy, and covariance are **not computable** from that representation (drives § 9.9) |
| **D-7** | `single_expert` vs `not_applicable` semantics undocumented in the report | code marks R-06/R-07 `single_expert` on dense and excludes **both** statuses from numerator and denominator (`scoring.py:116-119`), giving dense applicable weight **0.80** vs MoE **1.00** | Dense and MoE composites are **not on a comparable scale**, and nothing in the report says so. Docs say `not_applicable` (`headroom-signals.md`), code says `single_expert` — same exclusion, two labels |

**Measured live instance of "names do not imply identical mathematics"
(accompanying D-4/D-6):** for this Qwen3 checkpoint the FFN **container's**
forward output is the **down-projection result**, shape `[B, S, 1024]`
(hidden_size); the gate/up branches are `[B, S, 3072]` (intermediate_size).
Hooking the container measures the residual stream, not the FFN interior.

**Fix status:** D-1/D-2/D-3 (and part of D-4 via `acquire/layout.py`) have an
**IN FLIGHT** uncommitted implementation on this branch (concurrent agent —
working tree diff of `analysis/scoring.py` introduces `NEUTRAL_PRIOR = 0.5`,
status-category sets, and evidence-state strings; `src/model_headroom/acquire/`
is a new untracked module; `test_dense_support.py` is being rewritten to assert
`approx(0.9)`). **UNVERIFIED** until committed and the full chain is green
(Phase 13/14 exit criteria, § 17).

---

## 5. Historical architectural decisions & unresolved contradictions

A-1..A-5 are preserved from the 2026-11-07 ornith-1.5-9b end-to-end run
(measured against the emitted report, line references to the then-HEAD `da3e903`
as recorded there). Current status is updated below.

| ID | Decision / finding | Then-status | Now (2026-11-15) |
|----|--------------------|-------------|------------------|
| **A-1** | `grad_weight_ratio` scored 1.0 from a zero gradient | RESOLVED in code (orchestrator list-status handling + `_score_signal` None sentinel; zero gradient → `no_gradient`, neutral, not inverted) | **RESOLVED — standing.** The null-gradient contract is invariant 5 in `architecture/contract.md` ("a zero-filled placeholder is not a measured training gradient") |
| **A-2** | Provenance defaults were optimistic | RESOLVED in code (defaults `unavailable`; GGUF/safetensors sampled path reports `fixture`/`unverified`; weight-pairs `measured`/`verified` only when explicitly declared) | **RESOLVED — standing.** No optimistic defaults exist at HEAD |
| **A-3** | `effective_rank` accepted from a single truncated tensor and scored it as a model-level property | REQUIRED (the artifact is honest about being a tile — 4096×4096 of a 12288×4096 parent, coverage 0.3333 — but the composite gave it no penalty or gate) | **STILL OPEN at the composite level.** The tile machinery (`parent_shape`, `coverage`, `sampling_mode`, `PARTIAL TILE` printing) is IMPLEMENTED; the *gate* — the composite must not present a one-tensor top-left tile as the model's rank — is part of Phase 6 (full-matrix + stratification) and Phase 13 (coverage-declared headline). The tensor was also an MLP gate, not the attention projection the weight plan intended (carrier defect C-3) |
| **A-4** | Dense scoring contract unresolved against the operator's rule | DECISION BLOCKED (operator ruling required before any dense score is compared to a MoE score; arithmetic in SEAMS.md: Option A 0.80, Option B 0.90, B-variant 1.00) | **RESOLVED by operator ruling (2026-11-15):** a dense FFN *is* conceptually a one-expert MoE — **honor the unification at the expert-abstraction level, do NOT invent a normalization value to force numeric comparability, declare the comparability limit instead.** Implemented treatment: `single_expert` status with `balance_ratio=None` (excluded from numerator and denominator, as at HEAD) + an explicit comparability statement in the report (Phase 18). The explicit math — `log(1) = 0` is well-defined, the degenerate distribution is real, `None` is the honest value — is in `measurement-ontology.md` § 4 |
| **A-5** | Refusal-direction extraction cannot run against a served vLLM | UNVERIFIED AT SCALE (all three acquisition paths fail against live vLLM 0.32.0: no `output_hidden_states` in `SamplingParams`; the SGLang route needs `--enable-return-hidden-states`; the live `--model-path` route needs the GPU/RAM the server holds; `--weight-deltas` needs an already-abliterated checkpoint. A behavioural probe ran instead: 100% refusal on AdvBench n=100, 0% on `harmless_alpaca` n=100, 2/30 on constructed borderline) | **SGLang route EXECUTED 2026-11-19** (in `/root/.venvs/inference`, env-qualified — sglang 0.5.2, T4; `capture_residuals_sglang` via `/generate`+boolean, cohen_d=3.83); vLLM production-server route still **UNKNOWN**; in-process domain contrast executed (cohen_d=4.97). Production/174G extraction remains UNKNOWN (`docs/guides/runtime-refusal-ablation.md`, commit 160aca4) |

**Unresolved contradictions carried forward** (each now has a home):

- Code vs frozen-contract docs on skipped-signal scoring (D-2/D-3) — Phase 13.
- Docs vs code on the dense status label `not_applicable` vs `single_expert`
  (D-7) — Phase 18 + M-doc.
- Docs vs measured counts (21 vs 25 dataclasses; v6 vs v7; 20 vs 23 flags;
  0.3.0 vs 0.4.0) — M-doc, § 20.
- The roadmap's own "Observed Shortcomings" line references (written for
  `da3e903`) vs current HEAD line numbers — this rewrite pins every reference
  to `e80591d` and states the convention.

---

## 6. Measurement ontology

Five classes; **never conflated** (full definitions, the never-conflate table
with measured instances, and the signal mapping in
`measurement-ontology.md`):

1. **Structure** — static facts about the stored weights (shapes, spectra,
   entropy, layout). No execution. → R-03, R-07 (weak proxy), architecture
   detection.
2. **Activity** — what the model does on specific inputs (activations, routing
   counts). Needs a forward pass. → R-05, R-06.
3. **Optimization** — the trajectory of a training process (loss curves,
   gradients, isoFLOP points). Needs training telemetry. → R-01, R-02, R-04,
   R-08/R-09.
4. **Plasticity** — the capacity to change under future training. **Never
   directly measured**; the headroom score is a prior-weighted estimate *about*
   it, with declared coverage.
5. **Observed intervention response** — baseline → intervention → post. The
   only evidence that can support "X changed Y by Z." →
   `analysis/intervention_record.py` (IMPLEMENTED at HEAD).

The rule: a class-A statistic never establishes a class-B claim. A weight rank
is not a learning-speed claim. A negative mean is not a dead neuron (GAP-2).
One 24-token prompt is not a model-level fact (AGENTS.md BANNED). A train-split
loss is not a validation trajectory. A representation statistic is never a
direct measurement of future learning gains (§ 14).

---

## 7. Signal-by-signal evidence contracts (R-01..R-09)

The contract for each signal: what it scientifically is, what evidence it
legitimately requires, the feasible acquisition routes (status per § 16), and
what a **missing input legitimately yields** — which is *absence*, handled by
status and neutral prior, never a fabricated zero.

| Signal | Scientific definition | Required evidence (ontology / provenance class) | Feasible acquisition routes | Missing input legitimately yields |
|--------|----------------------|--------------------------------------------------|------------------------------|-----------------------------------|
| **R-01** loss saturation | Has the **validation** loss plateaued: slope and plateau detection over (tokens_trained, validation_loss) | Optimization / class 1 (measured-direct) or 3 (imported-published). **Train-split loss is not a substitute** (`evidence-provenance.md` § 4) | Published run records (Phase 17, with stated limits); operator training logs | `skipped` / `insufficient_data`. The signal is simply absent; the composite carries a neutral prior (post-Phase 13) and the report names the gap. **Not** a 0.0, **not** a "converged" reading |
| **R-02** growth factor | Quadratic loss fit over isoFLOP triplets (compute_flops, model_params, measured_loss); crossover, alpha | Optimization / class 2/3 across **multiple** compute/size points | Published multi-run evidence only; operator isoFLOP runs (out of scope — the authority does not train) | `skipped`. **Structurally, one checkpoint yields nothing** — it is one point, and a scaling law is not manufactured from one point (§ 8, anti-manufacturing rule) |
| **R-03** effective rank | SVD-based spectral entropy of weight matrices; `percentage_of_full_rank` is **tile-relative** by contract | Structure / class 1 (or sidecar weight pairs) | Direct safetensors/GGUF sampling (IMPLEMENTED; layout fix Phase 14); Hu sidecar weight-pair path (IMPLEMENTED); full-matrix randomized SVD (Phase 6, needs stronger hardware) | `skipped` with a **stated reason** post-Phase 14 ("layout unresolved: <what the header actually contains>"), not a silent warning |
| **R-04** gradient/weight ratio | ‖∂L/W‖ / ‖W‖ with z-score and high-pressure thresholding; returns a list | Optimization / class 1, `measured` + `verified` provenance (model_revision, dataset_hash, token_count) | In-process backward over a declared corpus (Phase 7); published gradient pairs (rare) | `no_gradient` with `ratio: null` when weight-only (A-1; invariant 5). **Never** a zero-filled placeholder; **never** a maximum score from absence |
| **R-05** dead neurons | Activity-based: a channel is dead iff max\|x\| == 0 across all batch/token positions (never fires); signed breakdown reported separately (GAP-2 fix) | Activity / class 1 | In-process forward hooks (IMPLEMENTED; per-channel recorder Phase 16); SGLang hidden-state residuals (Phase 19 — a residual is **not** an FFN interior) | `skipped`. A checkpoint with no activation evidence yields no dead-neuron claim — and no "0 dead" claim |
| **R-06** load balance | Router entropy vs ideal; balance ratio with the **architectural** expert count as denominator (fix D4/D5) | Activity / class 1 — per-layer per-expert token counts | SGLang `--enable-return-routed-experts` via the carrier routing sidecar (carrier IMPLEMENTED/VERIFIED 2026-10-09; authority consumption Phase 19, PROPOSED) | On non-SGLang engines: legitimately nothing (no routing visibility — report `routed_experts: false`, never a guess). On dense: `single_expert` (§ 10) |
| **R-07** specialization | Pairwise expert cosine similarity; collapse detection | Structure (weak: weight vectors) / Activity (strong: activation-weighted) | `--expert-weights-file`; carrier `specialization` key **reserved null** (weak offline by design) | `skipped`; dense → `single_expert` (pairwise cosine over one vector is undefined — the honest value is the status, not a number) |
| **R-08** gradient coherence | Cross-batch gradient cosine similarity — **diagnostic, opt-in** (`--gradient-diagnostics`); never affects the composite | Optimization / class 1, multi-batch gradient pairs | Same as R-04, multiple batches | Not run unless the flag is set; when run without evidence, `insufficient_data` |
| **R-09** gradient effective rank | SVD rank analysis of gradient matrices — **diagnostic, opt-in**; never affects the composite | Optimization / class 1 | Same as R-08 | As R-08 |

Two structural notes:

- **R-08/R-09 isolation** is enforced, not just documented: they are opt-in,
  never imported by `analysis/orchestrator.py`'s scoring path, and never
  contribute to the composite (`frozen-contracts.md`).
- **The status vocabulary is 12 values at HEAD** (`types.py:309-322`); the
  5-value table in `headroom-signals.md` is an incomplete subset (M-doc).

---

## 8. Evidence provenance & applicability

Full treatment in `evidence-provenance.md`. The load-bearing rules:

1. **Seven provenance classes** (PROPOSED evidence-level taxonomy mapping onto
   the frozen `SourceType` values — no silent contract mutation):
   `measured-direct`, `measured-transferred`, `imported-published`, `claimed`,
   `estimated-derived`, `synthetic`, `fixture`. Absence is handled by status,
   never by a source type (`null` ≠ healthy-zero, contract invariant 3).
2. **Attributable, not just tagged.** Every evidence record carries checkpoint
   identity (name **and** revision), split (train/eval), producer identity
   (script sha, engine version), derivation chain (direct or named formula), and
   stated limitations. A tag without the chain is decoration.
3. **A model card's claimed loss is not a validated trajectory for a different
   derivative** — that is the `measured-transferred` vs `claimed` distinction,
   and it is load-bearing for the operator's derivative-checkpoint line
   (RockAndEgg-313M → -v2).
4. **R-02 scaling evidence must never be manufactured from one checkpoint** —
   structural impossibility, not a policy preference (§ 7, R-02 row).
5. **"What this evidence does NOT establish"** is a standing section of the
   acquisition design: the `rae-005` published record (train-split loss, eval
   accuracy + `pres_mse`, derived tokens, duplicate final point) is real and
   usable **with its limits stated on the artifact it feeds** — a mapping that
   launders a limit into a clean curve is a bug.

---

## 9. Acquisition architecture

Full design in `acquisition-architecture.md`. The spine:

### 9.1 Family adapters (checkpoint layouts) — operator item 6

Compose feature-specific adapters; **do not build one gigantic class**. Names
do not imply identical mathematics (measured live instance: the FFN container
output is the down-projection `[B,S,1024]`, not the FFN interior — § 4).
Adapters: dense gated FFN (SwiGLU) — IMPLEMENTED, template-bound (D-4);
standard dense blocks — IMPLEMENTED (Phase 5); sparse MoE — detection
IMPLEMENTED, **never verified end-to-end**; hybrid attention/recurrent —
IMPLEMENTED, VERIFIED on ornith-1.5-9b, D-5 on the Qwen3 case; encoder/
classifier — IMPLEMENTED (DeBERTa P6 UNKNOWN); multimodal (vision + mtp) —
GAP-3 resolved; quantized/custom projections — ternary path IMPLEMENTED
(§ 9.7).

### 9.2 Backend adapters — operator item 7

Verify the **actual** API/version/supported families/output semantics before
relying on an integration; measure, don't assume. safetensors direct —
IMPLEMENTED (D-4 caveat); GGUF direct — IMPLEMENTED, never end-to-end on a real
GGUF model; HF Transformers + instrumentation — IMPLEMENTED, VERIFIED on
ornith-1.5-9b (32 hooks, shapes verified), D-6 per-channel defect; **vLLM —
PROPOSED (nothing executed; serving engines expose no gradients, no
routed_experts, no hidden states)**; **SGLang — PROPOSED (the only engine
exposing `routed_experts` and hidden-state residuals; hard constraints per
`guides/engine-selection.md`; refusal residual capture IMPLEMENTED + executed
2026-11-19 in `/root/.venvs/inference` (sglang 0.5.2); R-05/R-06 consumption
PROPOSED)**; **llama.cpp — PROPOSED (exposes none of the
measurement signals)**; published artifacts (Hub) — PROPOSED, route real (§ 9.4);
Hu capture replay — IMPLEMENTED for the weight-pair path (2-D guard at
`models/loader.py:432-443` rejects 3-D fused stacks rather than slicing).

### 9.3 Capability negotiation — operator item 8

At run start the authority assembles a capability manifest: what the checkpoint
provides (layout resolution — the role of the IN FLIGHT
`src/model_headroom/acquire/layout.py`: read the header, classify tensors by
the mathematical operation their name denotes, report the prefix actually
found), what the runtime provides (device via NFR-01, memory, installed
libraries/versions), and what is reachable (local by default; published only on
explicit operator opt-in with a pinned revision). Output: per-signal
**obtainable** (route + limits) or **not obtainable** (reason). **The tool
decides which evidence it can obtain, and declares what it cannot** — the
declaration is part of the report, not a log line (the reporting half of D-1).

### 9.4 Published-evidence discovery & acquisition — operator item 9

**Metadata-first, revision-pinned, selective download.** Metadata-first means:
read the Hub file listing at a pinned revision and the small metadata files
first; decide from metadata whether the repo contains the needed artifact class;
a repo with metadata but no run record is answered from metadata without
downloading shards. Revision-pinned means: every read names a revision (§ 9.5 —
the drift is measured, not hypothetical). Selective download means: fetch only
what the manifest identifies, verify hashes. Status: PROPOSED (the
`moeshawky/rae-005-research-evidence` route is measured real; authority-side
acquisition not yet executed).

### 9.5 Checkpoint / model-revision identity — operator item 10

**Measured revision drift:** the local `PROVENANCE.md` of
`moeshawky/RockAndEgg-313M-v2` records revision
`ac8bb272d74c1d7ed6879e53802d857fefb555ac`; the current Hub HEAD of that repo is
`7928cdfeda71fe3bf040cd584686d2e42cbe5c21`. Different revisions, same name.
Rules: revision-pinning mandatory for published acquisition; a local
`PROVENANCE.md` is an operator-recorded fact, not a live check — verify against
the Hub when it matters; absence of an artifact is a measured fact
(`RockAndEgg-313M-v2` publishes no `trainer_state.json` — "training state: not
published at this revision," and stop); cache identity includes the revision
(§ 9.11). Full treatment in `evidence-provenance.md` § 5.

### 9.6 Exact vs sampled measurement semantics — operator item 11

**A 2.1% tile is never presented as a full-matrix number.** The machinery:
`parent_shape` + `coverage` + `sampling_mode` + `truncated` on every sampled
artifact; `percentage_of_full_rank` tile-relative by contract; the human report
prints `PARTIAL TILE` and states that inflation against the full matrix is not
measured by the tool. The label extends to every new statistic class
(activations, routing counts, sketches) via the same fields. Composite-level
coverage is `compute_coverage_metrics` (measured-only score, bounds,
decision-grade gate ≥70%, `--analysis-mode decision_grade`, exit 2 — note: the
flag is `--analysis-mode {diagnostic,decision_grade}`; **there is no
`--decision-grade` flag**, and documents that name one are drifted, § 20).

### 9.7 Quantization & effective-weight handling — operator item 12

- GGUF extraction is IMPLEMENTED (`models/gguf_loader.py`); the ternary path
  (`models/ternary_quants.py`, `models/ternary_sampling.py`) serves the
  operator's known ternary/GGUF work (Ternary-Bonsai-8B-Q2_0 lineage). Stock
  llama.cpp build 11395 **cannot read** that GGUF (`gguf_init_from_reader`
  offset error — legacy ternary quant); a PrismML llama.cpp is being built; the
  authority path handles it separately via the ternary dequant adapters — do
  not conflate the two.
- Known issue (preserved from the old roadmap): `dequantize_ternary` documents
  a Prism fallback, but `_gguf_dequantize` catches only
  `AttributeError`/`NotImplementedError`; with the `gguf` extra absent the call
  raises `ModuleNotFoundError` instead of falling back (measured 2026-11-06,
  CI run 42260689947). No-`gguf` test paths skip; fallback semantics deferred.
- **Effective weights:** for quantized checkpoints the *measured quantity* is
  the dequantized (effective) weight; the report must name the quant format and
  the dequantization applied. A fp8 checkpoint on sm_75 dequantizes to bf16 and
  never executes fp8 kernels (hard constraint) — so "the fp8 weight" and "the
  bf16 computation" must not be conflated in any statistic.

### 9.8 Activation extraction & replay — operator item 16

In-process forward with `forward_pre_hook` on declared module paths from the
layout resolution (never hand-written prefixes — GAP-4). The recorder must
preserve **channel identity** (D-6 fix, Phase 16). **A replay is not identical
to a production execution unless evidence supports it**: identity requires same
batching, KV-cache state, sampling parameters, engine and version; a captured
activation is a fact about one production request, not a reproducible property
of the checkpoint. The artifact declares replay conditions (engine, version,
batch, prompt set, seed) or says "replay conditions not declared."

### 9.9 Streaming numerical statistics — operator item 17

O(channels) memory requirement (no `[B,S,C]` retention): per-channel RMS
(O(C) running sums of squares); near-zero occupancy (O(C) running counters);
second-moment matrix O(C²) — valid at C=3072 (≈151 MB fp32), must fall back to
the sketch or declare infeasible at C=12288 (≈585 MB) on this box; effective
dimensionality via participation ratio PR = (Σλ)²/Σλ² (technique name only;
citation not verified); randomized NNL (randomized SVD / sketching) for
spectrum estimates where valid — a sketch recovers the top-r spectrum when
reasonably conditioned, it does not certify the tail (technique name only;
citation not verified). Threshold τ and sketch size k are declared parameters
of the artifact, never implicit; a capped statistic carries `sampling_mode` +
coverage like a weight tile.

### 9.10 Compute & memory budget — operator item 18

**Measured (this machine):** 2× Tesla T4 sm_75 (cc 7.5), 14911 MiB VRAM each,
**~11.7 GB already in use** at the 2026-11-15 baseline, ~31 GB RAM.
Consequences (each measured, not preferred): fp8 kernels impossible (sm_75 <
sm_89; fp8 dequantizes to bf16); bf16 emulated on sm_75 (native needs sm_80+) —
slower, VRAM-hungry; an 18 GB checkpoint does not fit 15 GB VRAM (CPU-RAM load +
GPU compute is the correct architecture); forward+backward on a 9.65 B bf16
model OOMs even at seq_len 4 (2026-11-13 ticket); full-matrix SVD over ~700
large matrices does not complete in reasonable time (same ticket).
Consequences for the acquisition layer: per-tensor reads only (`safe_open`,
shard never materialised); row-capped seeded sampling with full `parent_shape`;
streaming O(C) activation statistics; device through `device.py` (NFR-01:
accelerator preferred cuda→xpu→mps, CPU fallback, explicit unavailable
accelerator = error, never a silent downgrade).

### 9.11 Hu sidecar integration & caching — operator item 19

The carrier (opt-in `CAPTURE_HEADROOM`) emits `{capture_id}.headroom.json` +
`{capture_id}-weight_pairs.json` post-response, async, never on the response
path (ADR-001). Only `effective_rank` produces an artifact today; the other five
keys are **reserved null — unobservable at the proxy layer** (training loss,
compute-loss points, serving-time activations are not things the proxy can see).
Describe them that way, never as missing features.

**Caching design (PROPOSED):** immutable checkpoint-derived diagnostics (weight
rank, layout facts, architecture facts) are cacheable under
`(model revision, file sha256, sampling parameters, software version)` — same
identity, same result. Genuinely dynamic observations (activations, routing,
refusal probes) are keyed by **capture id** — they depend on the input and the
serving state, not only the checkpoint; a cache keyed on the checkpoint alone
would serve a stale "measurement" of a different request. **Hot-path isolation
(invariant):** sidecar and cache failures never touch the inference hot path —
a cache bug may cost a recompute; it may never cost a proxied response.

---

## 10. Dense as a single-expert MoE — operator item 13

Operator ruling (resolves A-4): honor the unification at the expert-abstraction
level; do not invent a normalization; declare the comparability limit.

- **The math** (`measurement-ontology.md` § 4): the one-expert router emits the
  degenerate distribution p = (1.0); H = −1·log(1) = 0 **exactly** — `log(1)=0`
  is well-defined, not a singularity. The one-expert router is simultaneously
  maximally imbalanced and trivially "balanced"; the numeric ratio is
  undefined-as-a-measurement, so the honest value is `None` (status
  `single_expert`), **not** 0.0 or 1.0 — either invented value would fabricate
  a measurement from an architectural fact. Pairwise expert cosine over one
  vector is undefined (no pair); same treatment for R-07.
- **The exclusion stands:** `single_expert` and `not_applicable` are excluded
  from numerator and denominator (`scoring.py:116-119` at HEAD) → dense
  applicable weight 0.80 vs MoE 1.00 → **composites not numerically
  comparable** → the report must say so (Phase 18). No invented normalization.
- **Status:** code behavior IMPLEMENTED at HEAD; report declaration PROPOSED
  (Phase 18); docs/code label drift recorded (D-7, M-doc).

---

## 11. MoE routing instrumentation — operator item 14

Evidence: per-layer per-expert integer token counts + architectural expert
count. Source (hard constraint): SGLang `--enable-return-routed-experts`;
carrier routing sidecar decodes `sglext.routed_experts` (base64 int32, shape
`LxK`) to `{base_path}/routing/YYYY-MM-DD/{capture_id}.routing.json`
(`src/capture_proxy/routing.py`); llama.cpp/vLLM/ollama report
`routed_experts: false`, never a guess. Counts derive from expanding the
per-token per-layer K-expert records; R-06's denominator is the architectural
count (fix D4/D5) — a collapsed router on 8 experts is `computed` with a low
ratio, **not** `single_expert`. Status: carrier side IMPLEMENTED/VERIFIED
(2026-10-09, SGLang 0.5.21 — the carrier's verification); authority-side
consumption PROPOSED (Phase 19). Full treatment in
`measurement-ontology.md` § 5.

---

## 12. Hybrid / recurrent considerations — operator item 15

- `config.json` `layer_types` is the source of truth for per-layer identity;
  derive, never hand-infer (GAP-4: a worker died on
  `layers.15.linear_attn.in_proj_qkv.weight` because layer 15 is
  `full_attention`).
- Measured hybrid instance: ornith-1.5-9b — 32 layers, 3× `linear_attention` +
  1× `full_attention` repeating (Gated DeltaNet), `model.visual.*` tower,
  `mtp.layers.0.*`; linear-attention layers run on transformers' pure-torch
  fallbacks without `fla`/`causal_conv1d`/`mamba_ssm` (executed 2026-11-12).
- Detection is IMPLEMENTED and VERIFIED on ornith-1.5-9b; **it is misbehaving on
  the Qwen3 case (D-5)** — a detector verified on one architecture is not
  verified on the next. Linear-attention tensor names are absent from the CLI
  sampling templates (D-4 scope). Full treatment in
  `measurement-ontology.md` § 6.

---

## 13. Error semantics & insufficient-evidence behaviour — operator item 21

**The target semantic:** a report with no valid measurements communicates
**INSUFFICIENT EVIDENCE** — never an apparent headroom value of zero. The three
defects measured at HEAD that violate it:

- D-1: headline `0.00` on a real checkpoint with zero computed signals.
- D-2: skipped signals contribute effective zero (contract says 0.5).
- D-3: the recommendation text claims the code's nonexistent semantic.

**Design (matches the IN FLIGHT implementation, which remains UNVERIFIED):**

1. The composite is a **prior-mixed expectation**, not a measurement: a signal
   with no evidence contributes the neutral prior 0.5 (maximum ignorance); a
   signal whose computation *failed* or whose evidence was produced but is
   *numerically invalid* contributes 0.0 (a claim was attempted and is not
   usable — that must not be laundered into neutral "no information");
   architecturally-excluded signals (`not_applicable`, `single_expert`)
   contribute to neither numerator nor denominator.
2. The report carries an **evidence state** for the whole run
   (`no_applicable_signals` / `insufficient_evidence` / `partial_coverage` /
   `full_coverage`) so the headline and the evidence cannot be read apart.
3. `null` ≠ healthy-zero (contract invariant 3) and weight-only pairs carry
   `gradient: null` + `gradient_status: "not_captured"` (invariant 5).
4. The capability manifest (§ 9.3) explains each absence by reason.

**Falsification (Phase 13 exit criteria, § 17):** any run that prints a numeric
headline while zero signals are computed, or any test that passes under
zero-prior arithmetic while named for the neutral prior, fails.

---

## 14. Scientific calibration & predictive validation — operator item 20

1. **A representation statistic is never a direct measurement of future
   learning gains.** Rank, RMS, entropy, balance ratios describe the checkpoint
   as stored or as it behaved on sampled inputs; what a model *would* learn is
   a property of (model, intervention, data, budget) that none of them
   observes.
2. **The composite is a diagnostic ordering, not a probability** — "neither is
   a percentage of anything" (Ornith handoff). `--analysis-mode decision_grade`
   gates on measured coverage (≥70%), not on score calibration.
3. **The only validating evidence is observed intervention response** (fifth
   ontology class): baseline → intervention (LoRA, QLoRA, quantization
   recovery, weight edit) → post, in `InterventionRecord` (IMPLEMENTED at HEAD
   — the record structure exists; no realized intervention-with-outcome has
   been executed on this machine).
4. **Predictive-validation protocol (PROPOSED):** held-out checkpoints with
   pre-intervention signals and realized outcomes; measure whether the signal
   ordering predicted the outcome; report base rates honestly — if every
   intervened checkpoint improved, the ordering is *untested*, and the report
   says "calibration: untested," not "calibration: good." Trigger: the first
   `InterventionRecord` with a realized outcome (Phase 20, deferred).

---

## 15. Backward compatibility & migration — operator item 22

- **25 frozen dataclasses** (`models/types.py` at HEAD; measured — the
  "21" in three documents is drifted, § 20). All `frozen=True`; never mutated;
  new instances only. Adding a field to an existing dataclass is an additive
  contract change and must be versioned like the schema bumps; no silent
  mutation of a frozen contract to fit the sidecar (root `AGENTS.md` BANNED).
- **The 2-D pair contract guard** (`models/loader.py:432-443`): weight/gradient
  pairs must be 2-D; a 3-D fused expert stack is **rejected, not sliced** —
  slicing would invent a shape the checkpoint never had. A producer with 3-D
  stacks exports per-expert, or the contract is versioned additively; it is
  never bent silently.
- **Schema versions are additive** (measured): carrier `CaptureRecord` v6,
  `CapturedHeadroom` v7 — older records without new keys still validate. Any
  new version follows the same rule.
- **`SignalResult.status` is a 12-value literal at HEAD** (`types.py:309-322`);
  the docs' 5-value table is a subset (M-doc). The IN FLIGHT scoring change
  re-partitions the *existing* 12 values into three treatment categories
  (excluded / no-evidence→0.5 / unusable→0.0) without adding values —
  UNVERIFIED until committed.
- **`EvidenceStore` hardcodes `software_version="0.3.0"`**
  (`analysis/evidence_store.py:78`, still present 2026-11-23) while the package is
  `0.6.0` (`pyproject.toml:7`) — an audit-identity defect: artifacts claim the wrong
  software version. Fix belongs with Phase 13/M-doc (read from package
  metadata, never a literal).
- **Signal-name seam:** the 6-name carrier `SIGNAL_KEYS` vs the 7 scored
  authority names as scattered literals have no compile-time link. A single
  shared vocabulary module (authority-owned, carrier imports the names or a
  generated artifact) is the proposed hardening — PROPOSED, and it must respect
  R-01 purity (the carrier's `names.py` must stay stdlib-only importable in off
  mode).

---

## 16. Real-model integration coverage — operator item 23

IMPLEMENTED vs VERIFIED vs PROPOSED vs BLOCKED vs REJECTED. "A branch in the
loader is not a verified model" (`guides/support-matrix.md`).

### 16.1 Model families (authority-side)

| Family | Backends / layouts | Status | Evidence |
|--------|--------------------|--------|----------|
| Dense transformers (standard naming, gated FFN) | safetensors direct | **VERIFIED** | qwen3-0.6b, mimo-v2.6 analyzed against real checkpoints (support matrix); qwen3-0.6b refusal direction `computed`, cohen_d=4.36 (roadmap history) |
| Dense transformers (no `model.` prefix — RockAndEgg layout) | safetensors direct | **IMPLEMENTED, DEFECTIVE (D-4)** — layout unresolved by the CLI; layout-resolution module IN FLIGHT | measured 2026-11-15: RockAndEgg-313M-v2 run prints "no MLP projection tensor matched a known layout" and all 7 signals skip (D-1/D-4). The same pattern was measured earlier on ornith-1.5-9b (GAP-1): the CLI reports R-03 `skipped` while `run_analysis()` with samples built directly from the tensors computes it — the tensors are on disk; the CLI's name resolution fails to reach them |
| Hybrid attention/recurrent (Gated DeltaNet) | safetensors direct + in-process forward | **VERIFIED (one model)** | ornith-1.5-9b, 2026-11-12 run: R-03 computed (20 tensors, seed 42), R-05 computed (32 text-stack MLPs, real forward); detection D-5 on the Qwen3 case |
| MoE (any) | safetensors/GGUF | **IMPLEMENTED, never VERIFIED** | detection + templates exist; no real MoE tested end-to-end (support matrix); R-06/R-07 need routing/expert evidence no carrier source currently produces end-to-end |
| GGUF dense | GGUF reader | **IMPLEMENTED, never VERIFIED** | no real GGUF model tested end-to-end (support matrix) |
| GGUF ternary (operator work) | ternary quants/sampling | **IMPLEMENTED** | `ternary_quants.py`/`ternary_sampling.py`; stock llama.cpp 11395 cannot read the Ternary-Bonsai GGUF (serving-side, separate path); Prism fallback gap recorded (§ 9.7) |
| Encoder/classifier (BERT family) | safetensors direct | **IMPLEMENTED** (Phase 5) | BERT/ModernBERT/laya/RockAndEgg naming; **DeBERTa P6 UNKNOWN** — implemented, never verified on a real model |
| Multimodal (vision + mtp) | in-process forward | **VERIFIED (one model)** | ornith-1.5-9b tower handled (`"visual"` substring, GAP-3); sampling covers MLP projections only |
| qwen3-0.6b / mimo-v2.6 refusal direction | in-process forward | **VERIFIED (small models)** | Phase 11: synthetic-axis recovery to the analytic noise floor; in-process tiny `LlamaModel` only — **no 174G checkpoint loaded** (production-scale extraction still UNKNOWN, A-5); model-scale in-process extraction VALIDATED 2026-11-19 (qwen3-0.6b cohen_d=4.97; Qwen2.5-Coder-1.5B-Instruct 4/4 refuse + ablation 4/4→0/4), TIER A — see `docs/guides/runtime-refusal-ablation.md` header, commit 160aca4 |

### 16.2 Serving-engine routes (authority consumption)

| Engine | Capability needed | Status |
|--------|-------------------|--------|
| SGLang | `routed_experts` (R-06), hidden-state residuals (R-05/refusal A-5) | **PROPOSED** — hard-constraint engine; carrier emission VERIFIED 2026-10-09 (carrier's, not the authority's); R-06 carrier-artifact consumption UNEXECUTED; REFUSAL residual capture EXECUTED authority-side 2026-11-19 in `/root/.venvs/inference` (sglang 0.5.2; cohen_d=3.83); sglang absent from default env/PATH (default env unchanged) — `docs/guides/runtime-refusal-ablation.md`, commit 160aca4 |
| vLLM | none of the measurement signals | **PROPOSED** for any future consumption; vLLM 0.32.0 present in a local venv (environment fact, 2026-11-10) |
| llama.cpp | none of the measurement signals | **PROPOSED**; engine running (build 11395, 2026-11-10); cannot read the ternary GGUF (serving-side limitation) |
| ollama | none of the measurement signals | **PROPOSED**; not on PATH |
| Published artifacts (Hub) | R-01/R-02 telemetry, run provenance | **PROPOSED** — route measured real (`rae-005-research-evidence`); authority-side acquisition not executed |
| Hu sidecar replay | R-03 (weight pairs), R-04 when gradients exist | **IMPLEMENTED** (weight-pair path, 2-D guard); other keys reserved null (unobservable at proxy layer) |

### 16.3 BLOCKED / REJECTED roll-up

- **BLOCKED (hardware):** full-matrix SVD at scale, forward+backward on 9B+
  bf16, decision-grade runs for large checkpoints on 2× T4 (measured OOMs /
  infeasible runtimes — § 9.10). Move to sm_80+ hardware with ≥24 GB to unblock.
- **REJECTED:** the alternatives and their reasons live in § 18.

---

## 17. Milestones, dependencies, falsifiable exit criteria — operator item 24

Phase numbering continues the historical record (1–12 exist; 13+ are new).
"Chain green" = `ruff check . && ruff format --check . && mypy src/ && pytest`
from the repo root, 0 failures, on the declaring branch.

### Phase 13 — Insufficient-evidence error semantics (fixes D-1, D-2, D-3)

- **Status:** IN FLIGHT (concurrent work; UNVERIFIED).
- **Dependencies:** none.
- **Falsifiable exit criteria:**
  1. `model-headroom --model-name RockAndEgg-313M-v2 --model-path <dir>` with no
     JSON inputs produces a report whose headline communicates
     INSUFFICIENT EVIDENCE (evidence state + neutral-prior composite), not a
     bare `0.00` presented as a measurement.
  2. Skipped signals contribute 0.5 to the composite per
     `concepts/frozen-contracts.md`; `test_dense_support.py::test_skipped_still_
     counts_neutral` asserts `approx(0.9)` and name and assertion agree.
  3. No recommendation text claims a mixing semantic the code does not have.
  4. `EvidenceStore` `software_version` reads from package metadata (0.4.0),
     not a hardcoded literal.
  5. Chain green.
- **Falsification:** any report printing a numeric headline with zero computed
  signals; any test named for the neutral prior that passes under zero-prior
  arithmetic.

### Phase 14 — Checkpoint layout resolution (fixes D-4)

- **Status:** IN FLIGHT (`src/model_headroom/acquire/layout.py` +
  `test_acquire.py`; UNVERIFIED).
- **Dependencies:** none.
- **Exit criteria:**
  1. RockAndEgg-313M-v2 (`layers.0.mlp.gate_proj.weight`, no `model.` prefix)
     resolves its MLP projections; R-03 computes from the on-disk checkpoint via
     the CLI.
  2. All four legacy layouts (Qwen3_5 conditional wrapper; standard
     `model.layers`; MoE `experts.0`; `block_sparse_moe` w1/w2/w3) still
     resolve — regression tests.
  3. No new hardcoded family prefix without a layout-derived fallback; the
     unresolved case reports what the header actually contains.
  4. Chain green.
- **Falsification:** a real checkpoint whose tensors are on disk still yields
  "no MLP projection tensor matched a known layout."

### Phase 15 — Architecture identification correctness (fixes D-5)

- **Status:** PROPOSED.
- **Dependencies:** none.
- **Exit criteria:**
  1. RockAndEgg-313M-v2 (`layer_types` = 10× `full_attention`) reports dense /
     full attention — not `hybrid_linear_attention`.
  2. ornith-1.5-9b still reports
     `hybrid_linear_attention [linear_attn, full_attn, vision, mtp]` —
     regression.
  3. Subtype derives from `layer_types` + tower presence only; no pattern
     inference.
  4. Chain green.
- **Falsification:** any config whose reported subtype contradicts its own
  `layer_types`.

### Phase 16 — Per-channel activation instrumentation (fixes D-6)

- **Status:** PROPOSED.
- **Dependencies:** Phase 14 (module paths from layout resolution).
- **Exit criteria:**
  1. The recorder preserves channel identity; per-channel RMS and near-zero
     occupancy are computed from recorded data for a declared `[B, S, C]` hook
     output.
  2. Memory is O(C) in the token loop — measured: recording a C=3072 layer over
     24 tokens without materialising the full tensor.
  3. Every activation artifact carries declared replay conditions (engine,
     version, batch, prompt set, seed) or an explicit "not declared."
  4. Adversarial regression test for the flatten defect; chain green.
- **Falsification:** per-channel statistics require re-reading the full tensor,
  or an artifact ships without replay conditions.

### Phase 17 — Published-evidence acquisition

- **Status:** PROPOSED.
- **Dependencies:** none (revision-pinning utility; Phase 14 helpful).
- **Exit criteria:**
  1. Metadata-first probe of `moeshawky/rae-005-research-evidence` at a pinned
     revision lists files and reads the run record without downloading weight
     shards; file hashes verified.
  2. R-01 input constructed from that record carries the stated labels:
     **train-split** (not validation), **derived tokens** (formula
     steps × batch × max_len named), **duplicate final point** excluded or
     flagged, eval `composite` accuracy marked **not-a-loss**.
  3. A repo with metadata but no run record is answered "training telemetry:
     not published at this revision" from metadata alone.
  4. Chain green.
- **Falsification:** a mapping that emits a clean (tokens, validation_loss)
  curve without the stated labels.

### Phase 18 — Dense single-expert reporting (A-4 ruling)

- **Status:** PROPOSED.
- **Dependencies:** Phase 13 (evidence-state reporting).
- **Exit criteria:**
  1. Dense reports declare the applicable weight (0.80) and state that dense
     and MoE composites are not numerically comparable; excluded signals named
     with reason (`single_expert`).
  2. `balance_ratio` remains `None`; no normalization value enters the score.
  3. Docs and code agree on the label (M-doc applied to the two drifted docs).
  4. Chain green.
- **Falsification:** a dense-vs-MoE comparison presented without the
  comparability statement, or any invented balance value.

### Phase 19 — SGLang authority-side integration (R-05/R-06 activity route)

- **Status:** PROPOSED (environment-gated: SGLang must be installed and served
  with the two flags; absent from the default repo env/PATH at measurement
  time — present in `/root/.venvs/inference` 2026-11-19, served with
  `--enable-return-hidden-states`, executed for refusal residuals; the R-06
  routing-histogram exit criterion remains unmet, so PROPOSED stands).
- **Dependencies:** Phase 16 (per-channel recorder).
- **Exit criteria:**
  1. Executed: SGLang served with `--enable-return-routed-experts` (and
     `--enable-return-hidden-states` for residuals); carrier routing sidecar
     artifact consumed by the authority; R-06 computed from a real routing
     histogram with the architectural-expert denominator.
  2. vLLM/llama.cpp/ollama report `routed_experts: false` /
     `hidden_states: false` in the capability manifest — never true on a guess.
  3. Chain green.
- **Falsification:** R-06 computed from non-routing data, or a capability claim
  for a non-SGLang engine.

### Phase 20 — Calibration on the first observed intervention response (deferred)

- **Status:** DEFERRED — trigger: the first `InterventionRecord` with a realized
  outcome on this line.
- **Exit criteria:** the predictive-validation protocol of § 14 executed with
  honest base rates; "calibration: untested" is the expected initial result and
  a valid, reported one.

### M-doc — Documentation drift corrections

- **Status:** PROPOSED (independent; small).
- **Exit criteria:** every correction in § 20 applied; `rg` no longer matches
  the stale values in living docs (history quotes preserved in the
  "SUPERSEDED" tables); `docs/index.md` navigation lists the three supporting
  documents.

**Dependency graph:** 13 ∥ 14 ∥ 15 ∥ 17 ∥ M-doc (independent); 16 ← 14; 18 ← 13;
19 ← 16 (+ environment); 20 ← first realized intervention.

---

## 18. Deferred research & rejected alternatives — operator item 25

**REJECTED (with reason):**

| Alternative | Rejected because |
|-------------|------------------|
| A universal estimator of "unused intelligence" | No defined observable; the mission is diagnostic instrumentation over declared evidence (mission non-goal 1). A score presented as such an estimate would be the D-1 conflation at the philosophy level |
| Forcing dense/MoE comparability with a normalization value (SEAMS.md Option B 0.90 / B-variant 1.00) | Operator ruling (A-4): inventing the value manufactures a measurement from an architectural fact; the comparability limit is declared instead (Phase 18) |
| Manufacturing R-02 from one checkpoint | Structural impossibility — one checkpoint is one isoFLOP point; a scaling law is not a resampling of one point (§ 8) |
| Measurement from serving engines (weight matrices, gradients, module introspection, optimizer state) | Physically unavailable from serving engines — they expose tokens and, at most, SGLang extras; measurement is in-process (`guides/engine-selection.md` hard constraints) |
| Synchronous sidecar await on the response path | Violates hot-path isolation (ADR-001; R-06 in the sidecar design: `_spawn_headroom` wall time < 5 ms p99); a capture bug must never cost a proxied response |
| The hardcoded template table as the permanent layout mechanism | A template is a guess about the checkpoint; the checkpoint header is the fact (D-4 → Phase 14). Templates survive only as regression cases |
| Signed mean as the dead-neuron criterion | Measured false: 99.994% of flagged channels were active (GAP-2); the activity-based criterion (max\|x\| == 0) replaced it |
| Zero-filled gradient placeholders for weight-only pairs | A fabricated zero is not a measured gradient (A-1; contract invariant 5); `gradient: null` + `no_gradient` is the honest encoding |
| Port-based backend identification changes | Frozen by decision (`app.py` ~line 230; `guides/engine-selection.md`); the probe is the decision-time artifact; "this is not a bug to fix" |
| Unifying the torch/safetensors environments | The separate-venv constraint is recorded in the carrier roadmap; unification is explicitly out of scope for this line |

**DEFERRED (with trigger):**

| Item | Deferred because | Trigger to resume |
|------|------------------|-------------------|
| GGUF MoE end-to-end | No real GGUF MoE model available; single-tensor literal-name extraction cannot do architecture→tensor resolution (`ARCHITECTURE_TENSOR_NAMES` was removed as dead, unconsumed code — do not restore it as a guess) | A real GGUF MoE checkpoint on the machine |
| Full-matrix SVD at scale (Phase 6 completion) | Measured infeasible on 2× T4 (no completion after minutes, 86% CPU) | sm_80+ hardware, ≥24 GB; randomized SVD is the interim path |
| Decision-grade runs for 9B+ checkpoints on this box | Measured OOM (forward+backward at seq_len 4) | Hardware move per the 2026-11-13 ticket's recommended next steps |
| DeBERTa verification (P6 UNKNOWN) | No DeBERTa model on the machine | Model availability |
| Calibration harness (Phase 20) | No realized intervention-with-outcome yet | First `InterventionRecord` with outcome |
| `ModuleNotFoundError` fallback in `_gguf_dequantize` | No-`gguf` paths skip; measured 2026-11-06 (CI run 42260689947) | A no-`gguf` install that needs the Prism fallback |
| Production-scale refusal-direction extraction (A-5) | All three paths fail against live vLLM 0.32.0; measured cause recorded | SGLang route live (Phase 19 environment) or an abliterated checkpoint for `--weight-deltas` — trigger FIRED at small-model scale 2026-11-19 (`/root/.venvs/inference`; `docs/guides/runtime-refusal-ablation.md`, commit 160aca4); the 174G / production-scale bar remains the gate |

---

## 19. Known limitations & resource requirements — operator item 26

1. **Hardware:** 2× T4 sm_75, 14911 MiB each with ~11.7 GB already in use at
   baseline; fp8 kernels impossible; bf16 emulated; 18 GB checkpoints do not fit
   (§ 9.10). bf16 forward+backward and full-matrix SVD are BLOCKED here.
2. **Environment floor:** numpy≥2.5.4, pytest≥9.2, pytest-asyncio≥1.4.1.post1,
   pytest-cov≥7.1 — without it the suite aborts at collection
   (`model_headroom/tests/conftest.py`), § 3.2.
3. **Evidence poverty of released checkpoints:** a weights-only checkpoint
   contains no training trajectory (R-01/R-02), no gradients (R-04), no
   activations (R-05/R-06). Absence is the normal state for released models and
   is a finding to report, not a gap to paper over (the Ornith run: 5 of 7
   signals legitimately absent).
4. **Single-sample telemetry:** every activation/refusal measurement on this
   line so far is small-prompt, small-n; the reports say so, and AGENTS.md bans
   scaling such telemetry into model-level claims.
5. **No CI, no pre-commit** at the root (the pre-unification authority note
   stands: `pre-commit` is a dev dep with no config file). The drift
   checkpoints (§ 26) are manual discipline.
6. **Version drift between `EvidenceStore` and the package** (0.3.0 vs 0.4.0)
   degrades audit identity until Phase 13/M-doc.
7. **The signal-name seam** (6 carrier names vs 7 authority literals, no
   compile-time link) is the most likely drift point for any new evidence
   source (§ 15).
8. **Documented but unverified surfaces:** DeBERTa (P6 UNKNOWN), all MoE
   families, all GGUF families, all serving-engine authority-side routes
   (§ 16).

---

## 20. Documentation drift corrections (measured 2026-11-15)

Each row: the drifted claim, where it sits, and the measured value. Corrected by
M-doc; preserved here so the drift record survives even if the edits slip.

| # | Drifted claim | Location | Measured value |
|---|---------------|----------|----------------|
| 1 | "21 frozen dataclasses" | `docs/concepts/frozen-contracts.md:13`; `docs/architecture/authority-carrier.md:29,79`; root `AGENTS.md` (model_headroom section); `docs/model-headroom/README.md` (header) | **25** (`types.py`, `@dataclass` = `frozen=True` = 25 at HEAD) |
| 2 | Sidecar "schema_version currently 6" | `docs/architecture/contract.md:35,105` | `CaptureRecord` **6** / `CapturedHeadroom` (the sidecar file schema) **7** (`src/capture_proxy/models.py:121,168`) |
| 3 | "CLI entry (20 flags)" | `docs/architecture/authority-carrier.md:35`; old roadmap "19 args" | **31** `add_argument` calls in `cli.py` on the current tree (a463b16, measured 2026-11-23); the e80591d baseline had 23 |
| 4 | `EvidenceStore` claims 0.3.0 | `analysis/evidence_store.py:78` | package is **0.4.0** (`pyproject.toml`) |
| 5 | "`--decision-grade`" flag | old roadmap Phase 13-era text; `docs/maintenance/2026-11-13-ornith-headroom-pitfalls.md:153`; `docs/model-headroom/README.md` | the flag is **`--analysis-mode {diagnostic,decision_grade}`** (`cli.py:98-101`); there is no `--decision-grade` |
| 6 | "dense models mark R-06/R-07 `not_applicable`" | `docs/concepts/headroom-signals.md:141-142,162-163,209-211`; `docs/architecture/authority-carrier.md:51-52`; `docs/guides/support-matrix.md:31` | code emits **`single_expert`** (`signals/load_balance.py:84`; `signals/specialization.py:60,82`); the scoring treatment is identical (both statuses excluded from numerator and denominator), so scores are unaffected — the label is what drifts |
| 7 | "pytest: 426 passed, 16 skipped" / "422 test functions" / "0.2.0" | old roadmap Current State; root `AGENTS.md` metrics block | **1753 passed, 13 skipped** at HEAD (single root suite); **0.4.0** |
| 8 | Signal status table (5 values) | `docs/concepts/headroom-signals.md:199-211` | `SignalResult.status` is a **12-value literal** at HEAD (`types.py:309-322`) |

---

## 21. Completed Phases (preserved — history, not re-litigated)

### Phase 1: GGUF Test Environment Fix ✅ COMPLETED
- **Result**: GGUF tests either pass (gguf installed) or skip cleanly. Loader coverage improved from 43% to ≥80%.
- **Evidence**: `pytest` shows 0 failures for GGUF tests.

### Phase 2: Adversarial Test Audit ✅ COMPLETED
- **Result**: Every functional surface element has at least one falsifying test. Anti-idiocy review completed for all test modules.
- **Evidence**: TEST_PROTOCOL.md 88 cases mapped against actual coverage. New test files: test_module_family.py, test_loader_provenance.py, test_cli_provenance.py, test_human_report_diagnostics.py, test_scoring_coverage.py, test_loader_format_detection.py, test_orchestrator_errors.py, test_signal_correctness.py.

### Phase 3: Coverage Gap Closure ✅ COMPLETED
- **Result**: All source modules ≥80% coverage. Total coverage: 95%.
- **Evidence**: cli.py, human_report.py, models/loader.py, gguf_loader.py all now ≥80%.

### Phase 4: Gap Testing ✅ COMPLETED
- **Result**: get_module_family, infer_source_type_from_path, load_gradient_provenance_from_json all have dedicated test coverage.
- **Evidence**: 8 new test files added (118 new tests) covering previously untested functions.

### Phase 5: Classifier First-Class Support ✅ COMPLETED
- **Result**: Full classifier model family detection and classification. BERT-style naming ("attention.self.query", "dense.weight") now correctly routed through get_module_family. Encoder-based models (laya with encoder/config.json) detect as dense. Full greenfield chain completed (DESIGN → OPTIMIZE → IMPLEMENT) with all gates passing.
- **Code changes** (all in `src/model_headroom/models/loader.py`):
  1. `get_module_family` FFN branch: added "dense" keyword + nested "attention" exclusion + nested "pooler" exclusion
  2. `get_module_family` attention branch: added "attention" keyword (fixes BERT/ModernBERT "attention.self.query" naming)
  3. `_detect_model_format`: added encoder/config.json fallback (enables laya composite checkpoint)
  4. `detect_architecture_details`: added "attention." substring to has_full_attention check (fixes BERT encoder detection)
- **What did not change**: types.py (no new ModuleFamily value — Rule of Three), scoring.py (dense path already correct), cli.py, orchestrator.py, signals — all prior art reused as-is.
- **Tests added**: 16 new unit tests (15 in `tests/test_module_family.py` TestGetModuleFamilyClassifier class, 1 in `tests/test_loader_format_detection.py` test_hf_directory_with_encoder_config) + 4 new integration tests in `tests/test_integration_real_models.py` (TestBERTFamilyDense class: test_bert_base_uncased, test_deberta_base, test_rockandegg_313m, test_laya — all skip when models absent).
- **Gate chain result** (verified 2026-09-26): ruff check ✅ | ruff format ✅ (84 files) | mypy ✅ (64 source files) | pytest ✅ (426 passed, 16 skipped, 0 failed). *(The 426/16 figure is the pre-unification suite count, preserved as history; the current baseline is § 3.1.)*
- **Key ADRs**: ADR-001 extend existing keyword-chain pattern (not new abstraction); ADR-002 targeted encoder/config.json fallback (not generic config search); ADR-003 extend has_full_attention substring check; ADR-004 no new ModuleFamily Literal value (Rule of Three); ADR-005 no scoring changes; ADR-006 inherit existing test fixture patterns.
- **Deviation**: Implementer added "pooler" nested exclusion to T-01 because test contract C-04 specified pooler.dense.weight → "other" but T-01's exact code would have returned "ffn" (since "pooler.dense" contains "dense"). Correctly resolved by following test contract, using same nested-exclusion pattern.
- **Doctrine-gate labels**: P1 BERT naming CANONICAL, P2 ModernBERT STANDARD, P3 classifier-head-family INFERRED (model_headroom-local), P4 laya-no-root-config CANONICAL, P5 BERT FFN CANONICAL, P6 DeBERTa naming UNKNOWN (scoped out, not verified).

### Phase 11: Refusal-Direction Extraction ✅ IMPLEMENTED (see § 22 for the scale caveat)

---

## 22. Future Phases 6–10, 11, 12 (preserved, status-annotated)

Status annotations (2026-11-15) are added in brackets; the original text is
preserved.

### Phase 6: Full-Matrix R-03 + Family Stratification [PENDING — BLOCKED on this box for the full-matrix part (measured infeasible, § 9.10); randomized-SVD path open on stronger hardware; the A-3 composite gate is carried into Phase 13]
- **Goal**: Replace tile-only effective rank with full-matrix randomized_svd; stratify by module family (ffn/attention/linear_attention/vision/mtp) and layer depth
- **Success criterion**: R-03 reports coverage>1% with parent_shape; per-family rank breakdown available
- **Evidence**: `qwen27b.md:140` — "Next: randomized_svd on full [17408,5120] for layers 0,8,16,24,32,40,48,56,63 × families"; `chatgpt_suggestions.md:174` — R-03 should distinguish tile vs full matrix rank; `chatgpt_suggestions.md:257` — module-family reporting
- **Scope**: New CLI flags for full-matrix mode, family stratification in reporting, expanded test matrix for coverage>1% cases

### Phase 7: Domain Gradient Verification (R-04 real) [PENDING — requires in-process backward on a domain corpus; BLOCKED on this box for 9B+ bf16 (measured OOM)]
- **Goal**: R-04 with verified provenance — actual ∂L/∂W from backward pass on domain corpus (50k-500k tokens, 512-2048 channels/layer)
- **Success criterion**: source_type="measured", verification_status="verified" for gradient pairs; decision-grade achievable
- **Evidence**: `qwen27b.md:344` — "Decision-grade requires secure server with 50k-500k domain tokens"; `chatgpt_suggestions.md:132` — GradientWeightPair should carry model_revision, dataset_hash, token_count provenance; `chatgpt_suggestions.md:511` — establish "these numbers really are ∂L/W"
- **Scope**: Domain corpus loader, verified provenance chain, scoring exclusion test for unverified gradients

### Phase 8: R-01 Loss Curve Probe [PENDING — the published-evidence route (Phase 17) is now the cheaper first input; a real LoRA probe remains the gold standard]
- **Goal**: Real adaptation probe — LoRA training loss curve (100-500 optimizer steps) for loss saturation detection
- **Success criterion**: R-01 signal_status="computed" with real loss_curve data; is_saturated determination with ground truth
- **Evidence**: `qwen27b.md:342` — "short R-01 adaptation probe"; `chatgpt_suggestions.md:682` — "Obtain a real R-01 loss-saturation curve"; `signals/loss_saturation.py:53` — currently returns insufficient_data for all real models
- **Scope**: Loss curve data loader, real-model integration test, is_saturated correctness validation

### Phase 9: Real Activation Data (R-05 real) [PENDING — Phase 16 (per-channel recorder) is the prerequisite instrument; the Ornith run is the executed small-scale template]
- **Goal**: R-05 with real domain activations instead of synthetic fixtures; distribution statistics (RMS, mean_abs, variance, near_zero_occupancy, cross_token_variance) on real data
- **Success criterion**: source_type="measured" for activations; distribution stats computed on real model data
- **Evidence**: `qwen27b.md:342` — "50k-200k domain tokens 512-2048 channels/layer"; `chatgpt_suggestions.md:290` — improve R-05 definition beyond mean<1e-6; `qwen27b.md:179` — "R-05 synthetic valid for code test, invalid for model inference"
- **Scope**: Real activation loader, distribution statistic validation, synthetic vs real discrimination test

### Phase 10: Historical Control Model [PENDING — unchanged]
- **Goal**: Run one historical control model (27B) for comparison baseline; establishes whether signal changes are real or noise
- **Success criterion**: Full analysis on control model with comparable architecture; results stored as reference baseline
- **Evidence**: `qwen27b.md:421` — "historical control 27B"; `chatgpt_suggestions.md:847` — "Run one historical control model"
- **Scope**: Control model measurement pipeline, baseline storage, comparison reporting

### Phase 12: GPU/CPU Device Selection (explicit operator request 2026-11-04) [IMPLEMENTED at HEAD — `--device {auto,cuda,cpu}` (default `auto`) in `cli.py`, resolved through `device.py`; NFR-01 rewritten accordingly]
- **Goal**: Use GPU when an accelerator is available for GPU-eligible operations; CPU as fallback when no accelerator exists or `--device cpu` is requested. `--device {auto,cuda,cpu}` (default `auto`) is implemented in `src/model_headroom/cli.py` and resolved through `src/model_headroom/device.py` (`available_accelerators()`, `default_device()`, `resolve_device()`). Preference order: cuda, then xpu, then mps, else cpu. An explicit unavailable accelerator returns `None`, which the CLI turns into an error + exit 1, never a silent downgrade.
- **Success criterion**: `--device cuda` with `torch.cuda.is_available()==True` executes tensor operations on GPU tensors; `--device cpu` forces CPU; `--device auto` (default) resolves to first available accelerator else cpu; all existing tests pass; scores bit-for-bit identical on CPU default.
- **Scope**: Limited to device placement only. No signal result changes. No new signal categories. No refactoring unrelated to device.
- **NFR-01**: Rewritten — accelerator preferred (cuda, then xpu, then mps), CPU as fallback. CPU is not "the default" when an accelerator exists; `auto` is the default. Explicitly requesting an unavailable accelerator is an error (exit 1).

### Phase 11: Refusal-Direction Extraction ✅ IMPLEMENTED [scale caveat preserved below]
- **Goal**: Compute a model-agnostic refusal direction on the fly from contrastive harmful/harmless prompt sets, without coupling to the headroom scoring pipeline
- **Trigger**: Explicit operator request ("add this to model headroom… not just for this model's case, I want the abstraction, on the fly calculation of refusal direction")
- **Success criterion**: Direction recovered from a planted synthetic axis to the analytic noise floor; zero change to the 9-signal pipeline or `compute_headroom_score`
- **Scope**: `refusal/` package (activations, extract, recover, report, cli) + `model-headroom-refusal` console script
- **Explicitly NOT a signal**: does not appear in `signals/`, is never imported by `analysis/orchestrator.py` or `analysis/scoring.py`, and does not contribute to the composite score. Enforced by `tests/test_refusal_cli.py::TestIsolation`
- **Three acquisition paths**: offline JSON activations; live transformers forward hook; recovery from abliterated weight deltas (top right singular vector of `W r rᵀ`)
- **Scope-defence amendment**: Phase 11 amends the Phase 6-10 scope-creep line below. It adds a standalone capability, not a signal category, so it does not reopen the fixed R-01…R-07 set
- **Unverified at production scale**: the live hook is verified against in-process models (tiny `LlamaModel`; Qwen2.5-Coder-1.5B-Instruct — refusal RATE now measured 2026-11-19: chat-templated depth gradient cohen_d 1.63→9.76, causal ablation 4/4→0/4 refuse). No 174G checkpoint was loaded and no KL measurement was made, so end-to-end extraction quality on a production model remains UNKNOWN (`docs/guides/runtime-refusal-ablation.md`, commit 160aca4)

### Known issues (preserved)

- Sidecar weight-capture silent-failure history and DEFER-2, DEFER-3, DEFER-4,
  DEFER-6 live in the carrier roadmap (`docs/capture-proxy/PROJECT-ROADMAP.md`)
  — read before touching the sidecar path.
- **Dead surface (2026-11-06 audit):** `read_log_scales` and
  `extract_expert_weight_from_gguf` in `models/gguf_loader.py` have no
  production caller — only tests. Kept deliberately: both are accurate
  primitives, unlike the consumer-less `ARCHITECTURE_TENSOR_NAMES` table removed
  in the same audit. Remove only when a phase needs the space or a real caller
  appears.
- The `dequantize_ternary` / `_gguf_dequantize` fallback gap — see § 9.7.

---

## 23. Evidence base — Qwen3.8-27B (preserved)

- **Composite**: 0.6919 (legacy, mixes neutral priors for skipped signals — *pre-fix characterization; see D-2/D-3 and Phase 13 for the current semantic*)
- **Measured**: 0.8838 at 50% coverage (strict: 0.876 at 37.5%)
- **Decision-grade**: false (coverage <70% + estimated/synthetic provenance)
- **Architecture**: dense / hybrid_linear_attention (0 expert keys, vision+mtp present)
- **Signals**: 3 computed (effective_rank@0.0046% coverage, grad_weight_ratio@estimated, dead_neurons@synthetic), 2 skipped (R-01, R-02), 2 not_applicable (R-06, R-07)
- **Gap analysis** (per chatgpt_suggestions.md): R-03 weakly sampled (64×64 tiles ≠ full matrix), R-04 estimated (not ∂L/W), R-05 synthetic (discard as model evidence), R-01/R-02 correctly absent
- **Next measurements needed**: randomized_svd full-matrix, real domain gradients, real activations, R-01 loss curve, historical control 27B

---

## 24. Scope-Creep Defenses (preserved, updated 2026-11-15)

- IN scope: Fixing failing tests, adding missing coverage, adversarial audit of
  existing tests, measurement quality improvements (full-matrix, real data,
  provenance verification), phase-gated measurement expansion per future
  phases 6-10, classifier family support (BERT/DeBERTa/ModernBERT/laya/RockAndEgg),
  standalone non-scoring capabilities added under explicit operator request with
  zero coupling to the composite score (Phase 11 precedent), GPU/CPU device
  placement for GPU-eligible operations (Phase 12, explicit operator request
  2026-11-04), **and the evidence-acquisition line of this branch (Phases
  13–19, M-doc): error semantics, layout resolution, identification
  correctness, per-channel instrumentation, published-evidence acquisition,
  single-expert reporting, SGLang integration, documentation drift** — each
  phase above carries its own falsifiable exit criteria and none adds a signal
  category.
- OUT of scope: Refactoring working code unrelated to the authorized phases,
  adding new signal categories (R-01 through R-07 are fixed; R-08/R-09 are
  opt-in diagnostics, not composite inputs), adding non-measurement CLI features,
  security fuzzing, training, serving.
- Performance optimization was previously OUT of scope; expanded to IN scope
  for Phase 12 only by explicit operator request (GPU/CPU device selection).
  No other performance work is authorized.
- Adding new phase requires explicit operator request OR gap discovered during
  audit.
- Measurement phases (6-10) are gated: each requires the preceding phase's
  evidence to be collected before proceeding.
- Phase 12 is gated: device placement changes must not alter signal results
  (bit-for-bit identical scores on CPU), must preserve NFR-01 (accelerator
  preferred, CPU fallback), must add `--device auto|cuda|cpu` (default auto).

---

## 25. Controlled-Evolution Path (preserved)

- New signals require: operator request + coverage proof + adversarial test design
- Coverage below 80% in any module triggers immediate investigation
- Test failures trigger immediate investigation (no accumulation)
- **Added 2026-11-15:** any change touching the authority↔carrier boundary
  (imports, the signal-name vocabulary, the sidecar schema, the 2-D pair
  contract) re-runs the boundary probe (§ 3.3) and, for schema changes, the
  additive-validation test. A frozen dataclass is never mutated to fit a new
  evidence source.

---

## 26. Drift Checkpoints (preserved)

- After every phase: re-verify ruff → format → mypy → pytest in order
- After each phase: compare to baseline
- Before delivery: reconcile against full functional surface census
- **Added 2026-11-15:** before starting work on a branch, run `git status
  --short` and diff unexpected untracked package copies against HEAD (the
  stale-overlay incident, § 3.5). Before claiming any documentation number,
  measure it (`rg`, `wc -l`, or the verify chain) — a date-stamped claim with
  no command behind it is decoration.

---

## Appendix A — Measurement log (2026-11-15)

| Measurement | Command / source | Result |
|-------------|------------------|--------|
| Verify chain at HEAD `e80591d` | `ruff check . && ruff format --check . && mypy src/ && pytest` (after installing the § 3.2 dependency floor) | PASS / PASS (242 files) / PASS (69 source files) / **1753 passed, 13 skipped** |
| Frozen dataclass count | `git show HEAD:src/model_headroom/models/types.py \| grep -c "@dataclass"` and `grep -c "frozen=True"` | 25 / 25 |
| CLI flag count | `git show HEAD:src/model_headroom/cli.py \| grep -c "add_argument"` | 23 |
| Zero-prior composite at HEAD | isolated HEAD worktree (`git worktree add`), test's own report builder vs HEAD `compute_headroom_score` | 0.8 (test at HEAD asserts `approx(0.8)` — name contradicts assertion; working-tree rewrite to `approx(0.9)` is IN FLIGHT concurrent work) |
| D-1..D-5 reproduction | `model-headroom --model-name RockAndEgg-313M-v2 --model-path /kaggle/input/models/moeshawky/rockandegg-313m-v2/transformers/default/1` | `HEADROOM SCORE: 0.00`, 0% coverage, all 7 skipped; "no MLP projection tensor matched a known layout"; `Subtype: hybrid_linear_attention [full_attn]` on a 10× `full_attention` config |
| D-6 | read of `paired/instrument.py:171-207` at HEAD | `.reshape(-1)` in both `_sample_flat` and `_sample_rows` |
| FFN container output shape | forward pass on the RockAndEgg checkpoint | container output `[B,S,1024]` (down-projection), gate/up `[B,S,3072]` |
| Boundary probe | import-graph + computation audit across `src/model_headroom` and `src/capture_proxy` | no import edge either direction; carrier computes no score/ratio/rank/entropy/cosine/normalized quantity |
| SGLang/vLLM/ollama presence | `command -v`, venv import probe (2026-11-10, `docs/guides/engine-selection.md`) | vLLM 0.32.0 in `/root/.venvs/vllm`; llama-server build 11395 running; ollama/sglang not on PATH |
| Stale overlay | diff of untracked `capture-proxy/` overlay vs HEAD `src/capture_proxy/` | overlay's `headroom.py` lacks `gradient_status` (HEAD has it) → archived to `/tmp/opencode/sia-hu-stale-overlay/`, tracked files `git restore`d |

*Derived from measured repository state 2026-11-15 (HEAD e80591d). Phases 1–5,
11, 12 complete/implemented; 6–10 pending with status annotations; 13–20 and
M-doc are this roadmap's plan. Update at every phase transition; correct
stale numbers at the same time, marking them SUPERSEDED rather than deleting
them.*
