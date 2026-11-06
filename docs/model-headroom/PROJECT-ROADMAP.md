# PROJECT-ROADMAP.md — model_headroom_tool

## Intent Statement
Model Headroom Analyzer — 7 core signals + 2 opt-in gradient diagnostics for training headroom on MoE and dense models. Strict types, frozen dataclasses, adversarial-tested.

## Current State (Verified 2026-09-26)

- **Package status**: authority half of the single root project `sia-hu` 0.2.0 —
  one `pyproject.toml`, one `uv.lock`, one ruff/mypy/pytest config at the root;
  install `pip install -e .` (`.[dev]`, `.[gguf]`, `.[dev,headroom,gguf]`) and
  verify `ruff check . && ruff format --check . && mypy src/ && pytest`. The
  carrier never scores, and the authority wins any score dispute.
- **Scope**: 23 source files, 42 test files, ~3,977 LOC, 21 frozen dataclasses, 7 core signals + 2 diagnostic signals
- **Verification baseline**: ruff ✅ | ruff format ✅ | mypy ✅ | pytest: 426 passed, 16 skipped, 0 failed
- **Coverage**: 95% (above 80% threshold), all modules ≥80%
- **Functional surface**: CLI entry (19 args), orchestrator, scoring, 7 signals, 2 diagnostic signals, model loader (HF+GGUF), 2 reporters, architecture detection, classifier naming support (BERT/DeBERTa/ModernBERT/laya/RockAndEgg), provenance-aware scoring, coverage metrics, gradient diagnostics
- **Known issues**: None blocking. DeBERTa naming unverified (P6 UNKNOWN — scoped out, requires separate verification when model available). All TEST_PROTOCOL.md issues addressed by existing tests.
- **Dead surface (2026-11-06 audit)**: `read_log_scales` and `extract_expert_weight_from_gguf` in `models/gguf_loader.py` have no production caller — only tests. Kept deliberately: both are accurate primitives, unlike the consumer-less `ARCHITECTURE_TENSOR_NAMES` table removed in the same audit. Remove only when a phase needs the space or a real caller appears.

## Completed Phases

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
- **Gate chain result** (verified 2026-09-26): ruff check ✅ | ruff format ✅ (84 files) | mypy ✅ (64 source files) | pytest ✅ (426 passed, 16 skipped, 0 failed).
- **Key ADRs**: ADR-001 extend existing keyword-chain pattern (not new abstraction); ADR-002 targeted encoder/config.json fallback (not generic config search); ADR-003 extend has_full_attention substring check; ADR-004 no new ModuleFamily Literal value (Rule of Three); ADR-005 no scoring changes; ADR-006 inherit existing test fixture patterns.
- **Deviation**: Implementer added "pooler" nested exclusion to T-01 because test contract C-04 specified pooler.dense.weight → "other" but T-01's exact code would have returned "ffn" (since "pooler.dense" contains "dense"). Correctly resolved by following test contract, using same nested-exclusion pattern.
- **Doctrine-gate labels**: P1 BERT naming CANONICAL, P2 ModernBERT STANDARD, P3 classifier-head-family INFERRED (model_headroom-local), P4 laya-no-root-config CANONICAL, P5 BERT FFN CANONICAL, P6 DeBERTa naming UNKNOWN (scoped out, not verified).

## Evidence Base (Qwen3.8-27B — qwen27b.md)
- **Composite**: 0.6919 (legacy, mixes neutral priors for skipped signals)
- **Measured**: 0.8838 at 50% coverage (strict: 0.876 at 37.5%)
- **Decision-grade**: false (coverage <70% + estimated/synthetic provenance)
- **Architecture**: dense / hybrid_linear_attention (0 expert keys, vision+mtp present)
- **Signals**: 3 computed (effective_rank@0.0046% coverage, grad_weight_ratio@estimated, dead_neurons@synthetic), 2 skipped (R-01, R-02), 2 not_applicable (R-06, R-07)
- **Gap analysis** (per chatgpt_suggestions.md): R-03 weakly sampled (64×64 tiles ≠ full matrix), R-04 estimated (not ∂L/∂W), R-05 synthetic (discard as model evidence), R-01/R-02 correctly absent
- **Next measurements needed**: randomized_svd full-matrix, real domain gradients, real activations, R-01 loss curve, historical control 27B

## Future Phases (Evidence-Driven)

### Phase 6: Full-Matrix R-03 + Family Stratification
- **Goal**: Replace tile-only effective rank with full-matrix randomized_svd; stratify by module family (ffn/attention/linear_attention/vision/mtp) and layer depth
- **Success criterion**: R-03 reports coverage>1% with parent_shape; per-family rank breakdown available
- **Evidence**: `qwen27b.md:140` — "Next: randomized_svd on full [17408,5120] for layers 0,8,16,24,32,40,48,56,63 × families"; `chatgpt_suggestions.md:174` — R-03 should distinguish tile vs full matrix rank; `chatgpt_suggestions.md:257` — module-family reporting
- **Scope**: New CLI flags for full-matrix mode, family stratification in reporting, expanded test matrix for coverage>1% cases

### Phase 7: Domain Gradient Verification (R-04 real)
- **Goal**: R-04 with verified provenance — actual ∂L/∂W from backward pass on domain corpus (50k-500k tokens, 512-2048 channels/layer)
- **Success criterion**: source_type="measured", verification_status="verified" for gradient pairs; decision-grade achievable
- **Evidence**: `qwen27b.md:344` — "Decision-grade requires secure server with 50k-500k domain tokens"; `chatgpt_suggestions.md:132` — GradientWeightPair should carry model_revision, dataset_hash, token_count provenance; `chatgpt_suggestions.md:511` — establish "these numbers really are ∂L/∂W"
- **Scope**: Domain corpus loader, verified provenance chain, scoring exclusion test for unverified gradients

### Phase 8: R-01 Loss Curve Probe
- **Goal**: Real adaptation probe — LoRA training loss curve (100-500 optimizer steps) for loss saturation detection
- **Success criterion**: R-01 signal_status="computed" with real loss_curve data; is_saturated determination with ground truth
- **Evidence**: `qwen27b.md:342` — "short R-01 adaptation probe"; `chatgpt_suggestions.md:682` — "Obtain a real R-01 loss-saturation curve"; `signals/loss_saturation.py:53` — currently returns insufficient_data for all real models
- **Scope**: Loss curve data loader, real-model integration test, is_saturated correctness validation

### Phase 9: Real Activation Data (R-05 real)
- **Goal**: R-05 with real domain activations instead of synthetic fixtures; distribution statistics (RMS, mean_abs, variance, near_zero_occupancy, cross_token_variance) on real data
- **Success criterion**: source_type="measured" for activations; distribution stats computed on real model data
- **Evidence**: `qwen27b.md:342` — "50k-200k domain tokens 512-2048 channels/layer"; `chatgpt_suggestions.md:290` — improve R-05 definition beyond mean<1e-6; `qwen27b.md:179` — "R-05 synthetic valid for code test, invalid for model inference"
- **Scope**: Real activation loader, distribution statistic validation, synthetic vs real discrimination test

### Phase 10: Historical Control Model
- **Goal**: Run one historical control model (27B) for comparison baseline; establishes whether signal changes are real or noise
- **Success criterion**: Full analysis on control model with comparable architecture; results stored as reference baseline
- **Evidence**: `qwen27b.md:421` — "historical control 27B"; `chatgpt_suggestions.md:847` — "Run one historical control model"
- **Scope**: Control model measurement pipeline, baseline storage, comparison reporting

### Phase 12: GPU/CPU Device Selection (explicit operator request 2026-11-04)

- **Goal**: Use GPU when an accelerator is available for GPU-eligible operations; keep CPU as fallback and default (NFR-01 preserved). Add `--device auto|cuda` opt-in to `model-headroom-refusal` CLI. Fix `_capture_hook` to not force CPU transfer mid-forward-pass. Add optional `device` param to GPU-eligible signals (effective_rank SVD, gradient_effective_rank SVD, gradient_coherence SVD). Restructure `specialization` to batched Gram GEMM before GPU enablement. Restructure `dead_neurons` to fuse reduction passes before GPU enablement. Keep CPU for `grad_weight_ratio` norms (transfer cost > compute saved), `load_balance` (pure Python), loss/growth signals (pure Python), JSON→tensor construction, reporting serialization.
- **Success criterion**: `--device cuda` with `torch.cuda.is_available()==True` keeps activations on GPU through `_capture_hook` and executes SVD/norm/cosine on GPU tensors; `--device cpu` (default) unchanged from current behavior; all existing tests pass; scores bit-for-bit identical on CPU default.
- **Scope**: Limited to device placement only. No signal result changes. No new signal categories. No refactoring unrelated to device.
- **NFR-01**: Preserved — CPU remains the default. GPU is opt-in only.

### Phase 11: Refusal-Direction Extraction ✅ IMPLEMENTED
- **Goal**: Compute a model-agnostic refusal direction on the fly from contrastive harmful/harmless prompt sets, without coupling to the headroom scoring pipeline
- **Trigger**: Explicit operator request ("add this to model headroom… not just for this model's case, I want the abstraction, on the fly calculation of refusal direction")
- **Success criterion**: Direction recovered from a planted synthetic axis to the analytic noise floor; zero change to the 9-signal pipeline or `compute_headroom_score`
- **Scope**: `refusal/` package (activations, extract, recover, report, cli) + `model-headroom-refusal` console script
- **Explicitly NOT a signal**: does not appear in `signals/`, is never imported by `analysis/orchestrator.py` or `analysis/scoring.py`, and does not contribute to the composite score. Enforced by `tests/test_refusal_cli.py::TestIsolation`
- **Three acquisition paths**: offline JSON activations; live transformers forward hook; recovery from abliterated weight deltas (top right singular vector of `W r rᵀ`)
- **Scope-defence amendment**: Phase 11 amends the Phase 6-10 scope-creep line below. It adds a standalone capability, not a signal category, so it does not reopen the fixed R-01…R-07 set
- **Unverified at model scale**: the live hook is verified only against an in-process tiny `LlamaModel`. No 174G checkpoint was loaded and no refusal-rate or KL measurement was made, so end-to-end extraction quality on a production model remains UNKNOWN

## Scope-Creep Defenses

- IN scope: Fixing failing tests, adding missing coverage, adversarial audit of existing tests, measurement quality improvements (full-matrix, real data, provenance verification), phase-gated measurement expansion per future phases 6-10, classifier family support (BERT/DeBERTa/ModernBERT/laya/RockAndEgg), standalone non-scoring capabilities added under explicit operator request with zero coupling to the composite score (Phase 11 precedent), GPU/CPU device placement for GPU-eligible operations (Phase 12, explicit operator request 2026-11-04)
- OUT of scope: Refactoring working code unrelated to device placement, adding new signal categories (R-01 through R-07 are fixed), adding non-measurement CLI features to `model-headroom` unrelated to device selection, security fuzzing
- Performance optimization was previously OUT of scope; expanded to IN scope for Phase 12 only by explicit operator request (GPU/CPU device selection). No other performance work is authorized.
- Adding new phase requires explicit operator request OR gap discovered during audit
- Measurement phases (6-10) are gated: each requires the preceding phase's evidence to be collected before proceeding
- Phase 12 is gated: device placement changes must not alter signal results (bit-for-bit identical scores on CPU), must preserve NFR-01 CPU default, must add `--device auto|cuda` opt-in only

## Controlled-Evolution Path
- New signals require: operator request + coverage proof + adversarial test design
- Coverage below 80% in any module triggers immediate investigation
- Test failures trigger immediate investigation (no accumulation)

## Drift Checkpoints
- After every phase: re-verify ruff → format → mypy → pytest in order
- After each phase: compare to baseline
- Before delivery: reconcile against full functional surface census

---

*Derived from measured repository state 2026-09-26; Phase 11 added by operator request 2026-10-15. Phases 1-5 and 11 complete, 6-10 pending operator request. Update at every phase transition.*
