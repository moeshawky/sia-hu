# Model-Family Support Matrix

Honest accounting of which architectures are verified against real models versus
merely implemented in code. A branch in the loader is not a verified model.

Source of truth: code read at file:line, then matched to real-model test evidence.
No claim here rests on "the code has a branch for it."

## Detection mechanism (authority)

`src/model_headroom/models/loader.py:detect_architecture_details` (lines 126-241)
classifies every model into one of three families:

- **dense** — `num_local_experts`/`num_experts` absent or 0 (loader.py:157-161)
- **moe** — `num_local_experts`/`num_experts` >= 2 (loader.py:160)
- **unknown** — any exception falls back to family="unknown" (loader.py:231-241)

GGUF format is detected by `.gguf` suffix (loader.py:64) and routed through
`load_gguf_config` (loader.py:134-137), which reads `expert_count` from GGUF
metadata (gguf_loader.py:246-250) and defaults dense architectures to
`expert_count = 1` (gguf_loader.py:247-248).

## Verified families

| Family | Members | Real-model evidence | Detection code |
|--------|---------|---------------------|----------------|
| Dense (transformers) | qwen3-0.6b, mimo-v2.6 | Analyzed against real checkpoints; SIA-DIAGNOSTIC-CONTRACT.md:42 marks both with ✅ and lists them under "Verified against real model?"; PROJECT-ROADMAP.md:48 records Qwen3-0.6b analysis (`direction_status=computed, cohen_d=4.36, layer model.layers.16`) | `num_experts` absent/0 → `family="dense"` (loader.py:157-161) |

- qwen3-0.6b path used in report specimens: `/kaggle/input/models/qwen-lm/qwen-3/transformers/0.6b/1` (SIA-DIAGNOSTIC-CONTRACT.md:71-78, 129-131, 180-181)
- mimo-v2.6 listed as verified in SIA-DIAGNOSTIC-CONTRACT.md:42 and SIA-RELEASE-DECISION.md:29 ("only dense transformers verified against real models (qwen3-0.6b, mimo-v2.6)")
- On dense models, R-06 load_balance and R-07 specialization are `not_applicable` (excluded from composite); R-01..R-05 run (SIA-DIAGNOSTIC-CONTRACT.md:42)

## Unverified families (code path exists, no real model tested end-to-end)

| Family | Members | Code evidence | Why unverified |
|--------|---------|---------------|----------------|
| MoE | qwen3_moe, qwen3_5_moe, qwen2_moe, mixtral, olmoe, phimoe | `num_experts >= 2` → `family="moe"` (loader.py:160); MoE tensor-name templates in loader.py:52 (`qwen3_moe`, `qwen3_5_moe`, `qwen3_5_text`, `qwen2_moe`, `olmoe`); capture-proxy `MOE_MODEL_TYPES` set (names.py:34-36) | SIA-DIAGNOSTIC-CONTRACT.md:43: "❌ Not tested end-to-end"; SIA-RELEASE-DECISION.md:29: MoE not in the verified set; requires `--activations` + `--expert-weights-file` which no carrier routing source currently produces (SIA-DIAGNOSTIC-CONTRACT.md:43) |
| GGUF MoE | mixtral (GGUF), qwen3_5_moe (GGUF) | `extract_expert_weight_from_gguf` extracts a single tensor by literal name only; no architecture→tensor-name resolution exists — the `ARCHITECTURE_TENSOR_NAMES` table was removed as dead, unconsumed code (gguf_loader.py) | Not supported end-to-end; no real GGUF MoE model tested (SIA-RELEASE-DECISION.md:115-121) |

## GGUF status

| Item | Value | Evidence |
|------|-------|----------|
| Format detected | `.gguf` suffix | loader.py:64-65 |
| `qwen35` in `_DENSE_ARCHITECTURES` | Yes | gguf_loader.py:81 |
| Dense default `expert_count = 1` for `qwen35` | Yes | gguf_loader.py:247-248 |
| Real GGUF model tested | **No** | SIA-DIAGNOSTIC-CONTRACT.md:44: "❌ Not tested (gguf extra not installed)"; SIA-RELEASE-DECISION.md:115-121: no real qwen35 GGUF available for validation |

## P6 UNKNOWN — DeBERTa

| Item | Value | Evidence |
|------|-------|----------|
| Status | **P6 UNKNOWN** — implemented, not verified against a real model | DeBERTa naming unverified — support is implemented but not verified against a real model. Requires separate verification when model available. |
| Detection code | `get_module_family` supports BERT-family naming including deberta (loader.py:84-89); `deberta-v2` in capture-proxy `ARCHITECTURE_TENSOR_NAMES` attention/intermediate/output entries (names.py:96-108) | loader.py:84-89; names.py:96-108 |
| Integration test exists but skips | `test_deberta_base` (test_integration_real_models.py:154-165) skips when `/models/moe_test_models/deberta-base` absent (fixture at lines 131-139) | test_integration_real_models.py:128-139 |

## Honesty check

Does the matrix overclaim anything?

- **MoE as "supported"?** No. Not marked supported. Listed as unverified: detection code exists (loader.py:160), tensor-name templates exist (loader.py:52, gguf_loader.py:35-65), but no real MoE model has been tested end-to-end (SIA-DIAGNOSTIC-CONTRACT.md:43). MoE requires `--activations` + `--expert-weights-file` which no carrier routing source currently produces.
- **GGUF as "supported"?** No. `qwen35` is registered (gguf_loader.py:71, 81) but no real GGUF model has been tested (SIA-DIAGNOSTIC-CONTRACT.md:44; SIA-RELEASE-DECISION.md:115-121).
- **DeBERTa as "supported"?** No. Explicitly P6 UNKNOWN. Code implements deberta naming (loader.py:84-89; names.py:96-108) but the integration test skips when the model is absent (test_integration_real_models.py:154-165).
- **Only dense transformers verified:** qwen3-0.6b and mimo-v2.6, per SIA-DIAGNOSTIC-CONTRACT.md:42 and SIA-RELEASE-DECISION.md:29.

**Result: No overclaims.** The matrix marks exactly two families as verified (both dense), three as unverified (MoE, GGUF MoE, GGUF generally), and DeBERTa as P6 UNKNOWN.

## File:line reference index

| Claim | File:line |
|-------|-----------|
| dense/moe detection via `num_experts` | loader.py:157-161 |
| GGUF format detection | loader.py:64-65 |
| GGUF path in `detect_architecture_details` | loader.py:134-148 |
| `unknown` fallback | loader.py:231-241 |
| MoE tensor-name templates (HF) | loader.py:52 |
| `get_module_family` BERT/deberta support | loader.py:84-89 |
| `qwen35` entry | removed (was gguf_loader.py:71-75) |
| `_DENSE_ARCHITECTURES` | gguf_loader.py:81 |
| dense default `expert_count = 1` | gguf_loader.py:247-248 |
| `load_gguf_config` expert_count read | gguf_loader.py:246-250 |
| Dense verified: qwen3-0.6b, mimo-v2.6 | SIA-DIAGNOSTIC-CONTRACT.md:42 |
| MoE not tested end-to-end | SIA-DIAGNOSTIC-CONTRACT.md:43 |
| GGUF not tested | SIA-DIAGNOSTIC-CONTRACT.md:44 |
| Only dense transformers verified | SIA-RELEASE-DECISION.md:29 |
| Qwen3-0.6b real analysis | PROJECT-ROADMAP.md:48 |
| qwen35 GGUF registration, no real model | SIA-RELEASE-DECISION.md:115-121 |
| DeBERTa P6 UNKNOWN | Known issue (see Authority Gotchas) |
| DeBERTa test skips when model absent | test_integration_real_models.py:128-139, 154-165 |
| Capture-proxy MOE_MODEL_TYPES | src/capture_proxy/names.py:34-36 |
| Capture-proxy deberta tensor names | src/capture_proxy/names.py:96-108 |
