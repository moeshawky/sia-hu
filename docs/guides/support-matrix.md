# Model-Family and Backend Support Matrix

Honest accounting of which architectures, evidence routes, and acquisition
backends are **verified against real models**, which are merely **implemented
in code**, and which are **blocked** on this machine.

Source of truth: code read at file:line, then matched to executed real-model
evidence. No claim here rests on "the code has a branch for it." A skipped test
cannot establish support. A CLI exit code 0 is not evidence that a scientific
measurement succeeded.

Status vocabulary used throughout:

| Status | Meaning |
|--------|---------|
| **VERIFIED** | An executed operation measured the claimed property against a real artifact, and the result is pasted in the report that established it |
| **IMPLEMENTED** | The code path exists and is exercised by unit tests, but no real-model evidence exists |
| **PROPOSED** | Designed and specified, not yet built |
| **BLOCKED** | Cannot be executed here (missing dependency, missing model, or missing hardware) |
| **REJECTED** | Considered and deliberately not done, with the reason recorded |

---

## 0. Environment of record

Measured on the machine that produced this matrix (2026-11-15):

| Item | Value | Consequence |
|------|-------|-------------|
| GPUs | 2x Tesla T4, 15,360 MiB each, compute capability 7.5 (sm_75) | No fp8 kernels (needs sm_89+); no native bf16 tensor cores (needs sm_80+) |
| GPU memory in use at measurement time | ~11.7 GiB of 15.4 GiB per GPU | Tight headroom for a 9B-class forward+backward; a 313M model fits comfortably |
| RAM | 31 GiB total, ~23 GiB available | CPU-RAM loading plus GPU compute is a legitimate architecture for large checkpoints |
| Disk | 19 GiB free on `/kaggle/working` | Large weight downloads are discouraged; metadata-first retrieval is the default |
| Inference engines installed | **none** — `sglang`, `vllm`, `llama_cpp` all absent | Every engine integration is BLOCKED here, not merely unverified |
| Declared dependency floor | `numpy>=2.5.4`, `pytest>=9.2`, `pytest-asyncio>=1.4.1.post1`, `pytest-cov>=7.1` | The pre-existing image shipped numpy 2.1.3 and **no** `pytest-asyncio`; `model_headroom/tests/conftest.py` **aborts the whole suite** when `pytest-asyncio` is missing. Installing the declared floor is a prerequisite, not an optimisation |
| Baseline verify chain | ruff PASS, ruff format PASS, mypy PASS, pytest **1753 passed / 13 skipped** at `e80591d` | Post-Phase-A baseline is 1777 passed / 13 skipped |

---

## 1. Detection mechanism (authority)

`src/model_headroom/models/loader.py:detect_architecture_details` (line 138) and
`:detect_architecture` (line 340) classify every model into one of three families:

- **dense** — `num_local_experts`/`num_experts` absent or 0
- **moe** — `num_local_experts`/`num_experts` >= 2
- **unknown** — any exception falls back to `family="unknown"`

GGUF format is detected by `.gguf` suffix and routed through `load_gguf_config`,
which reads `expert_count` from GGUF metadata and defaults dense architectures
to `expert_count = 1`.

**Known misidentification (VERIFIED defect, open):** a Qwen3 checkpoint whose
`config.json` lists `layer_types` = 10x `full_attention` is reported as
`Subtype: hybrid_linear_attention [full_attn]`. The subtype label is not
trustworthy for hybrid/recurrent claims until the `layer_types` contract is
read directly rather than inferred.

---

## 2. Verified families

| Family | Members | Real-model evidence | Detection code |
|--------|---------|---------------------|----------------|
| Dense (transformers) | qwen3-0.6b, mimo-v2.6 | Analyzed against real checkpoints; recorded in the historical `SIA-DIAGNOSTIC-CONTRACT.md:42` and `SIA-RELEASE-DECISION.md:29` | `num_experts` absent/0 -> `family="dense"` |
| Dense (transformers) | **RockAndEgg-313M-v2** (Qwen3, 10 layers, hidden 1024, intermediate 3072, SwiGLU, bf16) | **VERIFIED 2026-11-15 on this machine.** Checkpoint `/kaggle/input/models/moeshawky/rockandegg-313m-v2/transformers/default/1` (112 tensors, single `model.safetensors`). Loaded with `transformers` 5.16.1 in 0.8 s; `model.layers.N.mlp.gate_proj` measured as `Linear[3072, 1024]`, `up_proj` as `Linear[3072, 1024]`, `down_proj` as `Linear[1024, 3072]`; 10 FFN modules discovered | `num_experts` absent -> dense |

- The RockAndEgg-313M-v2 checkpoint is the **reference dense family** for the
  acquisition work because it is small enough to measure end-to-end on a T4 and
  carries authentic published training provenance (see section 6).
- On dense models R-06 load_balance and R-07 specialization are emitted with
  status `single_expert` and excluded from the composite denominator, giving a
  dense applicable weight of **0.80** against MoE's **1.00**.

---

## 3. Unverified families (code path exists, no real model tested end-to-end)

| Family | Members | Code evidence | Why unverified |
|--------|---------|---------------|----------------|
| MoE | qwen3_moe, qwen3_5_moe, qwen2_moe, mixtral, olmoe, phimoe | `num_experts >= 2` -> `family="moe"`; MoE tensor-name templates in `loader.py`; carrier `MOE_MODEL_TYPES` in `src/capture_proxy/names.py` | No MoE checkpoint is present on this machine and no inference engine is installed, so R-06/R-07 remain **BLOCKED** for real measurement. The carrier CAN emit routing counts, but no MoE model has been driven through it here |
| GGUF MoE | mixtral (GGUF), qwen3_5_moe (GGUF) | `extract_expert_weight_from_gguf` extracts a single tensor by literal name only; no architecture->tensor-name resolution exists (the `ARCHITECTURE_TENSOR_NAMES` table was removed as dead, unconsumed code) | Not supported end-to-end; no real GGUF MoE model available |
| Hybrid / recurrent | qwen3_next, mamba-style, linear-attention hybrids | Subtype detection exists; `layer_types` handling is not verified | Subtype is currently MISIDENTIFIED (section 1) and no such checkpoint is present locally |
| Encoder / classifier | BERT, DeBERTa, ModernBERT, laya, RockAndEgg | `get_module_family` supports BERT-family naming; `detect_architecture_details` has an encoder/config.json fallback | **DeBERTa remains P6 UNKNOWN** — implemented, never validated against a real DeBERTa checkpoint. RockAndEgg-v2 in section 2 is a *causal* Qwen3 backbone, not the classifier-head variant the naming support targets |
| Multimodal | vision towers, MTP | `has_vision_tower` / `has_mtp` flags exist | No multimodal checkpoint present; unverified |

---

## 4. GGUF status

| Item | Value | Evidence |
|------|-------|----------|
| Format detected | `.gguf` suffix | `loader.py` `_detect_model_format` |
| Dense default `expert_count = 1` | Yes | `gguf_loader.py` `load_gguf_config` |
| Real GGUF model tested | **No** | No GGUF checkpoint present on this machine |
| Ternary / custom quant support | IMPLEMENTED | `models/ternary_quants.py`, `models/ternary_sampling.py`; `tests/test_ternary_bonsai_regression.py` skips when the model is absent |

---

## 5. Evidence routes (how the evidence is actually obtained)

This is the layer that was historically missing. "The analyzer requires a
hand-prepared JSON file" is not a model limitation, it is an acquisition gap.

| Route | Status | Evidence / limitation |
|-------|--------|------------------------|
| safetensors weight sampling (R-03) | **IMPLEMENTED, with a VERIFIED layout defect** | `src/model_headroom/cli.py:261` `_SAFETENSORS_TENSOR_TEMPLATES` hardcodes prefixes (`model.language_model.layers.{layer}.mlp.{proj}_proj.weight`, `model.layers.{layer}.mlp.…`, `model.layers.{layer}.mlp.experts.0.…`). The real RockAndEgg-313M-v2 checkpoint stores `layers.0.mlp.gate_proj.weight` with **no `model.` prefix**, so the run prints `no MLP projection tensor matched a known layout` and R-03 cannot be acquired even from weights already on disk. A layout resolver that discovers the actual prefix is the required fix |
| GGUF weight sampling (R-03) | IMPLEMENTED | `models/gguf_loader.py`; no real GGUF checkpoint on this machine |
| HF Transformers forward hooks (R-05) | **VERIFIED reachable, not yet wired into the signal path** | Executed 2026-11-15: the real checkpoint loads and `model.layers.N.mlp.gate_proj` / `up_proj` / `down_proj` are hookable. **Measured correction:** the FFN *container* output is the DOWN-PROJECTION result, shape `[B,S,1024]` (= hidden_size); the gate/up branches are `[B,S,3072]` (= intermediate_size). Hooking the container measures the residual stream, not the FFN interior. Any implementation that targets the container is measuring the wrong tensor |
| Per-channel activation statistics (R-05) | **IMPLEMENTED (existing recorder) but INSUFFICIENT for R-05** | `src/model_headroom/paired/instrument.py:171` `_sample_flat` and `:186` `_sample_rows` both `.reshape(-1)` their tensors, destroying channel identity. Per-channel RMS, near-zero occupancy, and covariance are **not computable** from that representation. A streaming O(channels) accumulator is required; storing raw activation tensors is not |
| Published training artifacts (R-01) | **VERIFIED reachable** | The operator publishes authentic evidence datasets on the Hub. `moeshawky/rae-005-research-evidence` contains `training_runs/control_off/run_record.json` with `history` (6 eval points, step/composite/pres_mse), `losses` (25 logged training-loss values), `hyperparams`, `params`, `provenance` (git_head, script_sha256, released_rev, cuda attestation), `data.manifest_sha256`, and `checkpoint`. Fetched successfully over the Hub REST API. **Limitations that must travel with any mapping:** the published series is a TRAIN-split loss log and an eval-split `pres_mse` (a preservation-anchor loss) plus an eval-split accuracy `composite` — an accuracy is not a loss; the token count is DERIVED (`steps x batch x max_len`) because tokens were never logged; the history contains a DUPLICATE final point; and the `moeshawky/RockAndEgg-313M-v2` model repo itself publishes no `trainer_state.json` |
| Hub metadata-first discovery | VERIFIED reachable | `GET /api/{models\|datasets}/{repo}/tree/{rev}?recursive=true` lists candidate artifacts without downloading weights; only the chosen small artifact is fetched |
| Gradient pairs (R-04, R-08, R-09) | BLOCKED | Requires a forward+backward pass on real domain data. The 9.65B Ornith-1.5-9B checkpoint OOM'd on this hardware (15 GiB T4, sm_75); the JSON weight-pair format is ~70 GB at full-tensor scale and infeasible. A bounded differentiable probe over sampled parameters is the only viable route here |
| Hu capture replay | IMPLEMENTED (carrier side), unverified for Sia | The carrier emits captures and a headroom sidecar; no replay path feeds Sia yet |

---

## 6. Engine backends

| Backend | Installed here | Supports | Verified? |
|---------|----------------|----------|-----------|
| safetensors | yes (0.8.0) | HF transformer checkpoints | **VERIFIED** against RockAndEgg-313M-v2 |
| HF Transformers | yes (5.16.1) | forward hooks, module discovery, real gradients | **VERIFIED** for load + forward-hook reachability against RockAndEgg-313M-v2; **not** verified for a full backward pass on this GPU |
| GGUF | yes (0.19.0) | quantized checkpoints | IMPLEMENTED; no real GGUF checkpoint present |
| Hugging Face Hub (REST) | yes (stdlib urllib) | metadata-first artifact discovery | **VERIFIED** against two real repos |
| SGLang | **no** | `return_routed_experts`, `return_hidden_states` request params | **BLOCKED** (not installed, no MoE checkpoint). Source-level check of `python/sglang/srt/entrypoints/openai/serving_chat.py` at `main` confirms both request-level parameters exist and are processed |
| vLLM | **no** | OpenAI-compatible serving | **BLOCKED**. Per the operator's own engine-selection gate: `SamplingParams` exposes no `output_hidden_states` |
| llama.cpp | **no** | GGUF serving | **BLOCKED**. The operator's gate: llama.cpp and vLLM expose neither routed-expert nor hidden-state residual capture |

### Correction to the SGLang flag documentation (VERIFIED 2026-11-15)

`docs/guides/engine-selection.md` and
`docs/guides/runtime-refusal-ablation.md` state that routed-expert capture
requires the **server CLI flag** `--enable-return-routed-experts` and hidden
states require `--enable-return-hidden-states`. Those flag strings were **not
found** in `python/sglang/srt/server_args.py` at either the pinned tag
`v0.5.23` or at `main` (both files fetch 200 and contain no
`--enable-return-*` registration).

What the carrier and the authority actually use is the **request-level**
parameter, and that form **is** present in
`python/sglang/srt/entrypoints/openai/serving_chat.py` at `main`
(`return_routed_experts`, `return_hidden_states`, and the
`process_routed_experts_from_ret` / `process_hidden_states_from_ret` handlers).
The implementation is therefore on the right mechanism; the **documentation
names a server flag that could not be located**. The flag spelling should be
re-verified against a pinned SGLang checkout before it is relied on again.

---

## 7. Evidence-state semantics (contract, VERIFIED)

`src/model_headroom/analysis/scoring.py` now distinguishes three things a
single composite number cannot:

| Concept | Field | Meaning |
|---------|-------|---------|
| Composite | `overall_headroom_score` | A **prior-mixed expectation**. A signal with no evidence contributes the neutral prior 0.5; a signal whose evidence was attempted but is unusable contributes 0.0; a signal the architecture cannot produce is excluded from both numerator and denominator. It is never a measurement |
| Measured-only score | `coverage_metrics.measured_score` | Weighted score over measured signals only |
| Uncertainty bounds | `coverage_metrics.lower_bound` / `.upper_bound` | Worst case (unmeasured score 0) to best case (unmeasured score 1) |
| Evidence state | `metadata["evidence_state"]` | `full_coverage`, `partial_coverage`, `insufficient_evidence`, `no_applicable_signals` |

When `evidence_state == insufficient_evidence`, the human report prints
`INSUFFICIENT EVIDENCE — no signal was measured` and **does not print a
headroom bar at all**; the JSON report carries `evidence_state` and
`composite_kind: prior_mixed_expectation`.

This closes a defect that was live at `e80591d`: a skipped signal previously
contributed an **effective zero** to the composite while the denominator kept
its weight, so a report with nothing measured rendered as
`HEADROOM SCORE: 0.00 (composite)`. A reader would conclude "no headroom"
where the truth was "no evidence". The frozen contract already specified the
correct behaviour (`docs/concepts/frozen-contracts.md`: "Skipped signals score
0.5 | Neutral prior, not a measurement"), so the repair restores the contract
rather than inventing one.

---

## 8. Honesty check

Does this matrix overclaim anything?

- **MoE as "supported"?** No. Listed as **BLOCKED** here: no MoE checkpoint and
  no inference engine on this machine.
- **GGUF as "supported"?** No. **IMPLEMENTED**, unverified against a real
  checkpoint.
- **DeBERTa as "supported"?** No. Still **P6 UNKNOWN**.
- **Hybrid/recurrent as "supported"?** No. The subtype label is currently
  **misidentified** on a real Qwen3 checkpoint, and no hybrid checkpoint is
  present.
- **The acquisition layer as "shipped"?** No. Status per route is in section 5;
  anything that has not been executed against a real artifact is not marked
  VERIFIED.
- **R-01 from published logs?** The *route* is VERIFIED (real artifact
  fetched and parsed). The *mapping* is a separate question and its limits are
  recorded in section 5 rather than being silently absorbed.

**Result: no overclaims.** Two families are VERIFIED (both dense, one of them
measured today), five are unverified or blocked, one is P6 UNKNOWN, and every
engine row states the hardware reason it cannot be exercised here.

---

## 9. Document/code drift corrections

Measured 2026-11-15 by contract audit:

| Claim in docs | Measured value | Where the stale number lives |
|---------------|----------------|-------------------------------|
| "21 frozen dataclasses" | **25** | `docs/concepts/frozen-contracts.md`, `docs/architecture/authority-carrier.md`, root `AGENTS.md` |
| `CapturedHeadroom` schema version 6 | **7** | `docs/architecture/contract.md` |
| "20 CLI flags" | **23** | `docs/architecture/contract.md`, `docs/reference/cli-flags.md` |
| `EvidenceStore.software_version = "0.3.0"` | package is **0.4.0** | `src/model_headroom/analysis/evidence_store.py` |
| `--decision-grade` flag | does not exist; it is `--analysis-mode {diagnostic,decision_grade}` | operator task framing; `src/model_headroom/cli.py` |
| `grad_weight_ratio` as a carrier sidecar key | the carrier's `SIGNAL_KEYS` has **6** names and does NOT include it; the authority scores **7** | `src/capture_proxy/names.py` vs `src/model_headroom/analysis/scoring.py` |
