# Session Handoff — 2026-11-08 — Full Phase Implementation

## Status

All 9 phases of the Sia-Hu mission have been implemented and pushed to `public-repo`.
Verify chain is fully green: ruff check ✓, ruff format ✓, mypy 0 errors, pytest 1375 passed/21 skipped/0 failed.

## Commits (this session)

| Commit | Description |
|---|---|
| `5e12ff0` | Reconstruct measurement-integrity fixes (Phases 0-1.3) |
| `fa0e37c` | Phase 0/1 corrections: gguf import guard, NaN activations, growth_factor, classifier detection, schema, logger |
| `f6e1de1` | U6/U7 audit: refusal module + integration/meta + 4 fixes |
| `1104117` | Phases 2-9: effective rank, module discovery, training feasibility, template compat, intervention records, evidence store, adversarial tests, research |

## What was done

### Phase 0 — Implementation inventory
- Reconstructed from 2026-11-06 CAM audit (9/11 units) + direct code reading
- 45 production modules, 16,874 lines, 9 signals (7 core + 2 diagnostic)

### Phase 1 — Measurement integrity (complete)
- Provenance axis: 10 source types (measured, synthetic, fixture, estimated, imported, unknown, inferred, unavailable, invalid, not_applicable)
- Verification status: 4 values (verified, provenance_incomplete, unverified, unavailable)
- Denominator rule: cat 1 (not_applicable) + cat 2 (single_expert) excluded; cat 3 (evidence absent/invalid) → 0.0 numerator, weight retained
- `_NOT_SCOREABLE = None` (not NaN)
- `_is_measured` = provenance-only (verification gates decision-grade, not coverage)
- Human report: truthful non-computed rendering, 3 guidance sections
- Dense ≡ 1-expert-MoE: equal score, equal weight, neither rewarded

### Phase 2 — Effective rank
- NaN/Inf weights → error status (was: NaN propagation through SVD)
- SVD casts to float32 before computation (numerical stability)
- Zero-check uses max() instead of sum() (robust for tiny values)
- Recommendation threshold 0.3 documented as heuristic (uncalibrated)
- Audit doc: `docs/maintenance/2026-11-08-audit/Phase2-effective-rank.md`

### Phase 3 — Module discovery
- `discover_modules(model)` — introspects named_modules, classifies by family
- `generate_adapter_candidates(modules)` — 5 strategies (attention-only, ffn-only, attention+ffn, all-linear, custom)
- Each candidate: target/excluded paths, matched count, trainable/frozen params, adapter bytes, family coverage, assumptions, incompatibilities

### Phase 4 — Training feasibility
- `estimate_training_feasibility()` — bounded estimates with uncertainty ranges
- Accounts for: weight storage, gradients, optimizer states, activations, framework overhead
- Lower/upper bounds, assumptions list, warnings

### Phase 5 — Template compatibility
- `check_template_compat(tokenizer, example_messages)` — chat template validation
- Checks: template presence, rendering, role detection, EOS/BOS, tool role support

### Phase 6 — Intervention records
- `InterventionRecord` dataclass — tracks baseline→intervention→post-measurement
- `hash_checkpoint(path)` — SHA-256 of checkpoint files/dirs
- Compute delta, save/load, JSON serialization

### Phase 7 — Evidence store
- `EvidenceStore` — file-based with hashing, versioning, verification
- `store()`, `retrieve()`, `list_artifacts()`, `verify()`
- Distinguishes schema validation, arithmetic reproduction, structural verification, measurement reproduction

### Phase 8 — Research findings
- Adopted: effective rank (Roy & Vetterli), refusal direction (Arditi), load balance entropy
- Rejected/deferred: randomized SVD, gradient covariance, AWQ/SmoothQuant, intrinsic dimensionality
- Underused: stable rank, participation ratio, quantization sensitivity, gradient subspace
- Doc: `docs/maintenance/2026-11-08-audit/Phase8-research.md`

### Phase 9 — Adversarial tests
- 14 tests in `model_headroom/tests/test_adversarial_phase9.py`
- Covers: synthetic gradients, valid zero gradients, missing signals, sparse coverage, dense/MoE, rank sampling, adapter counts, memory estimates, verification levels

### U6/U7 audit
- U6 (refusal/*): 5 LOW/INFO findings, 3 fixes applied
- U7 (integration/meta): 10 findings (2 MEDIUM, 5 LOW, 3 INFO), 3 fixes applied
- Audit reports: `docs/maintenance/2026-11-08-audit/U6.md`, `U7.md`

## Current state

- Branch: `public-repo`, pushed to `origin/public-repo` (Gogs)
- Working tree: clean (except `capture-proxy/.pi/` untracked dev artifacts)
- Verify chain: all green
- 1375 tests passed, 21 skipped, 0 failed

## New modules (importable)

```python
from model_headroom.analysis.module_discovery import (
    discover_modules,
    generate_adapter_candidates,
)
from model_headroom.analysis.training_feasibility import estimate_training_feasibility
from model_headroom.analysis.template_compat import check_template_compat
from model_headroom.analysis.intervention_record import (
    InterventionRecord,
    hash_checkpoint,
)
from model_headroom.analysis.evidence_store import EvidenceStore
```

## GPU machine next steps

1. Clone from Gogs: `git clone <gogs-url> && cd sia-hu && git checkout public-repo`
2. Install: `pip install -e ".[headroom,gguf,dev]"`
3. Run verify chain: `ruff check . && ruff format --check . && mypy src/ && pytest`
4. GPU-specific work:
   - Run runtime probes (Phase 4) against real models
   - Test module discovery on real transformer checkpoints
   - Validate chat templates with real tokenizers
   - Run effective rank on real weight matrices (not synthetic)
   - Test intervention recording with real checkpoint hashes
   - Calibrate heuristic thresholds against observed outcomes
5. Remaining carrier defects (lower priority):
   - U2b-02: `--stop` PID ownership check
   - U2b-01: blob dedupe race (temp+rename)
   - U1a2-02: unbounded runtime refresh
   - U1a2-03: usage zeros
   - U1a1-01/02/03: upstream error handling
   - U1b2-01/02: session/user id handling
   - U1a2-01: rerank ollama 501 not logged
   - U1a2-04: malformed SSE chunks
   - U2a-01: sidecar write failure silent

## Environment notes

- Python 3.13/3.14 (requires-python >=3.12, recommend >=3.13)
- torch, transformers, accelerate, safetensors, gguf under [headroom]/[gguf] extras
- No venv in repo — install globally or create venv
- mypy: 0 errors with gguf installed; 8 pre-existing errors without gguf
- pytest: 1375 passed, 21 skipped (transformers-dependent tests skip without it)
- Codegraph: run `ix --build` after checkout (ixd daemon auto-syncs)

## Key files for GPU work

| File | Purpose |
|---|---|
| `src/model_headroom/analysis/module_discovery.py` | Module discovery for real models |
| `src/model_headroom/analysis/training_feasibility.py` | Memory estimation (validate with real GPU) |
| `src/model_headroom/analysis/template_compat.py` | Template validation (test with real tokenizers) |
| `src/model_headroom/analysis/intervention_record.py` | Intervention tracking (use with real checkpoints) |
| `src/model_headroom/analysis/evidence_store.py` | Artifact persistence (test with real runs) |
| `src/model_headroom/signals/effective_rank.py` | SVD (validate numerical stability on GPU) |

## Unresolved issues

1. Effective rank heuristic threshold (0.3) uncalibrated — needs GPU validation
2. Training feasibility estimates are static — runtime probes not yet implemented
3. Module discovery relies on naming conventions — may need adjustment
4. No universal thresholds introduced without documentation
5. Intervention validation is a data structure, not an execution engine
6. Refusal module (U6) has 5 LOW findings not yet fixed
7. Carrier robustness (U7) has 7 remaining findings not yet fixed
