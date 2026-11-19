# Runtime Refusal Ablation — Operator Guide

**Capability:** refusal-direction extraction and runtime residual ablation (`model-headroom-refusal`).

**Status:** fringe. Not part of the core headroom verify chain (`src/model_headroom/refusal/__init__.py:7-9`). No new dependency: `safetensors` is already declared in `pyproject.toml` `[project.optional-dependencies] headroom` (`pyproject.toml:42`). The refusal tests DO run inside the root pytest suite because `pyproject.toml` `testpaths` includes `model_headroom/tests` (`pyproject.toml:97`) — a pre-existing convention, NOT a promotion into the core verify chain.

## Validation tiers used in this document

Every claim in this document carries one of three tiers:

- **[TIER A] VERIFIED IN THIS REPO** — executed probe in this environment; basis is file:line in this repo plus measured output.
- **[TIER B] READ FROM UPSTREAM SOURCE** — read against upstream source at the stated tag, not executed in this repo.
- **[TIER C] NOT VERIFIED** — explicitly unknown, or specified in the contract but not present / not executed in this repo.

**Environment measured 2026-11-12:** `import vllm` → `ModuleNotFoundError: No module named 'vllm'`; `import sglang` → `ModuleNotFoundError: No module named 'sglang'`. torch `2.11.0+cu128` and safetensors `0.8.0` are installed. [TIER A — measured]

Because no inference engine is installed, **no claim about live vLLM/SGLang behaviour is tier A**. Engine registration points below are tier B (source locations read against upstream) for the locations themselves and tier C for any claim that a hook fires on a live server.

### Live execution update — 2026-11-19 (separate inference venv, Tesla T4)

The 2026-11-12 measurement above holds for the repo's default environment. On **2026-11-19** live execution was performed in the separate inference venv `/root/.venvs/inference` (torch `2.14.0+cu130`, transformers `5.22.2`, SGLang `0.5.2`) on a **Tesla T4 (SM75, 2×15 GB)**. This upgrades the engine-touching claims below to **[TIER A]** *for that environment only*; it does not change the repo default-env measurement. Key measured results:

- **Path (a) live in-process domain-contrast now works on T4** (`--model-path` + `--domain-prompts`/`--out-of-domain-prompts`, no `--sglang-url`). Required the `_load_live_model` T4 fix (landed in this change): `dtype` now resolves through `_sm_aware_dtype` (bf16→fp16 on SM<80) and `model.to(device)` is applied when `device_map` was not, mirroring `_load_live`. Measured on `qwen3 0.6b-base`: exit 0, `refusal_direction` len 1024 norm 1.0000 finite, `domain_direction` len 1024 norm 1.0000 finite, `cohen_d = 4.97`, layer `model.layers.16`, `hidden_size` 1024. Before the fix this path requested bf16 on SM75, left the model on CPU without `accelerate`, and raised `AttributeError` on load failure. [TIER A — executed]
- **Path (b) SGLang refusal over HTTP works on SGLang 0.5.2** via `capture_residuals_sglang` (`--sglang-url`). SGLang 0.5.2 serves hidden states through `POST /generate` with **boolean** `return_hidden_states: true`, read from `meta_info.hidden_states[0][-1]` (shape `[[[…hidden…]]]` = batch×token×hidden). The `/v1/chat/completions` path is **broken on 0.5.2** for hidden states (string `"last"` fails `bool_parsing`; boolean hits an `input_ids should be a list of lists` batch bug) and is retained only as a fallback. Measured: `direction_status = computed`, `cohen_d = 3.83`, `refusal_direction` len 1024 norm 1.0000 finite (12 harmful / 12 harmless, final-layer residual). [TIER A — executed]
- **Behavioral probes via `ServingEngineClient(engine="sglang")`** reach `/generate` and return outputs (temp-0 repeat `consistency_score = 1.0`). [TIER A — executed]
- **SGLang launch on T4/SM75** required three operator-side patches (no repo change): `SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1` (the `t4_shim` stubs the import but not `importlib.metadata.version`), moving the `torchao_config == ""` early-return above the lazy `import torchao` in `sglang/srt/layers/torchao_utils.py` (it imported `torchao` before the empty-config return), and the `t4_shim` `sitecustomize.py` on `PYTHONPATH`. Launch: `--model-impl transformers --dtype float16 --attention-backend triton --sampling-backend pytorch --disable-cuda-graph --mem-fraction-static 0.85 --enable-return-hidden-states`. [TIER A — executed]
- **Instruct-model refusal behavior: VALIDATED (2026-11-19).** A chat/instruct checkpoint is now present (`qwen2.5-coder 1.5b-instruct`, `Qwen2ForCausalLM`, hidden 1536, 28 layers, bf16, chat template present) and exhibits **true refusal**: all 4 sampled harmful prompts return "I'm sorry, but I can't assist with that." while harmless prompts comply normally. The refusal direction was extracted **properly** by pre-applying the chat template (`apply_chat_template(..., add_generation_prompt=True)`) before `capture_last_token_residuals`, then computing the difference of means per layer. Separation grows with depth — `cohen_d` = 1.63 (L0) → 2.41 (L7) → 4.42 (L14) → 9.32 (L21) → 9.76 (L27, last) — the expected late-layer refusal signature. **Gold-standard causal check:** ablating the L27 direction at strength 1.0 (`install_refusal_ablation`) flips the model from 4/4 refusals to **0/4 refusals** (it answers bomb / email-hack / meth / lock-pick prompts), while the no-ablation control still refuses 4/4. The direction captures refusal causally, not just text-difference correlation. [TIER A — executed; note `capture_last_token_residuals` does NOT auto-apply the chat template, so instruct-model callers must pre-template prompts]

**Line-reference caveat:** `cli.py:XXX` anchors in this document are against the pre-2026-11-19 tree. The `_load_live_model` T4 fix added lines around `cli.py:585`, so anchors *after* that region shifted by a few lines; resolve by function name (`_load_live`, `_load_live_model`, `_capture_domain_live`, `_compute_domain_direction`, `_build_sidecar`) when an anchor lands slightly off.

## 1. What this is / what it is NOT

**What it is:**
- A refusal direction `r` is the normalised difference of means between harmful and harmless residual-stream activation sets at one layer (`extract.py:267-278`, `extract.py:70-77`). [TIER A]
- Runtime ablation applies `residual - strength * component * direction` per forward through `extract.project_out` (`extract.py:130-143`). [TIER A]
- Weights stay unmodified; nothing is written to the checkpoint. The effect exists only while the hook is installed. [TIER C — contract principle; the hook is installed in-process and removed with `handle.remove()` (`runtime.py:63-66`); not measured on a live model]
- Selective: you choose which modules to hook via target suffixes. `DEFAULT_TARGET_SUFFIXES = ("o_proj","out_proj","down_proj","shared_expert")` (`runtime.py:21-26`); `EXTENDED_TARGET_SUFFIXES` adds `"experts"` and `"linear_attn"` (`runtime.py:27-31`). The runtime module IS present in this repo: `src/model_headroom/refusal/runtime.py` (`runtime.py:1-257`), and its public functions plus `set_strength` are exercised by `model_headroom/tests/test_refusal_runtime.py` (28 passed). [TIER A]

**What it is NOT:**
- Not permanent ablation. Removing the hook restores the base model exactly. [TIER A — `test_lambda_zero_bit_exact_no_op` confirms `torch.equal(before, after)` at `test_refusal_runtime.py:119-128`]
- Not a headroom signal. It never contributes to `compute_headroom_score` and is never invoked by `run_analysis` (`refusal/__init__.py:7-9`). [TIER A]
- Not guaranteed to work on a live vLLM/SGLang server. [TIER C — no engine installed]
- Not a domain gate. The domain vector is optional; the domain GATE is not implemented here (roadmap only). [TIER C]

## 2. Producing the artifact — four acquisition paths

You produce a `RefusalDirectionReport` first, then serialize it to a safetensors artifact with `write_refusal_artifact` (`artifact.py:21-98`).

**Prerequisite (all paths):** safetensors `0.8.0` is installed and declared in the `headroom` extra (`pyproject.toml:42`). torch `2.11.0+cu128` is installed, though the headroom extra pins `torch>=2.15` (`pyproject.toml:40`) — check your torch version before running. [TIER A — measured]

Paths (c) and (d) require no inference engine and no GPU, so they complete the artifact-production half end to end in this environment. [TIER A]

### Path (a): live in-process capture

```
model-headroom-refusal \
  --model-path /path/to/model \
  --harmful-prompts harmful.txt \
  --harmless-prompts harmless.txt \
  --output-format json \
  --output-file report.json
```

`--model-path` is a Hugging Face model directory or hub id (`cli.py:75-77`); `--harmful-prompts`/`--harmless-prompts` accept `.txt`, `.json`, or `.jsonl` (`cli.py:78-85`). Live capture hooks the discovered decoder-layer stack and reads last-token residuals (`activations.py:174 capture_last_token_residuals`, `activations.py:123 discover_residual_layers`). `transformers` is not a declared dependency and is imported lazily (`activations.py:10-11`, `activations.py:193-196`). [TIER A for the CLI and capture functions]

**Instruct/chat models — pass `--chat-template`:** `capture_last_token_residuals` tokenizes prompts raw and does NOT auto-apply the tokenizer chat template, so on an instruct checkpoint the last-token residual reflects raw text, not the refusal decision. `--chat-template` wraps each prompt via `apply_chat_template(..., add_generation_prompt=True)` before capture (off by default; rejected with `--sglang-url`, which sends raw text — pre-template those prompt files instead). [TIER A — flag `cli.py:187-194`; wraps in `_load_live` and `_capture_domain_live`]

Domain-contrast variant: supply `--domain-prompts` and `--out-of-domain-prompts` (prompt files in `.txt`, `.json`, or `.jsonl` format, `cli.py:203-209`) together with `--sglang-url` for the SGLang route (`_capture_domain_sglang`, `cli.py:550`) or without it for the live in-process route (`_capture_domain_live`, `cli.py:630`). The CLI computes `domain_direction = mean(in_domain) - mean(out_of_domain)` at the resolved layer (`_compute_domain_direction`, `cli.py:723`) and passes it to `write_refusal_artifact`. [TIER A — flags present in `cli.py:195-209`, logic in `cli.py:550-750`; end-to-end live execution now performed on T4 2026-11-19 (see "Live execution update" in the header)]

### Path (b): SGLang over HTTP with `--enable-return-hidden-states`

```
model-headroom-refusal \
  --sglang-url http://127.0.0.1:30000 \
  --sglang-model <served-model-name> \
  --harmful-prompts harmful.txt \
  --harmless-prompts harmless.txt \
  --output-format json \
  --output-file report.json
```

`--sglang-url` captures final-layer residuals over HTTP without loading the model in-process (`cli.py:87-91`, `cli.py:718-759`). It requires the sglang server to be started with `--enable-return-hidden-states`; without it the server silently omits the field and capture returns `None` (`activations.py:275-277`). [TIER A for the CLI flag and capture function; tier B for server behaviour]

**SGLang 0.5.2 (measured 2026-11-19):** `capture_residuals_sglang` now uses `POST /generate` with **boolean** `return_hidden_states: true` as the primary path, reading `meta_info.hidden_states[0][-1]` (batch×token×hidden). The `/v1/chat/completions` path is broken on 0.5.2 for hidden states (string `"last"` fails `bool_parsing`; boolean hits an `input_ids should be a list of lists` batch bug) and is kept only as a fallback. [TIER A — executed against live SGLang 0.5.2]

The hidden-states switch maps to `features.enable_return_hidden_states` (`forward_batch_info.py:329`, read against upstream source at SGLang v0.5.23). [TIER B]

Only the final pre-lm_head layer is available through this route (`activations.py:397-400`). [TIER A]

Domain-contrast variant: `--domain-prompts` + `--out-of-domain-prompts` with `--sglang-url` routes to `_capture_domain_sglang` (`cli.py:550`), which calls `capture_residuals_sglang` per prompt set and computes the domain direction at the resolved layer (`_compute_domain_direction`, `cli.py:723`). [TIER A for the CLI wiring; tier B/C for server behaviour]

### Path (c): offline pre-captured activation JSON

```
model-headroom-refusal \
  --harmful-activations harmful_acts.json \
  --harmless-activations harmless_acts.json \
  --output-format json \
  --output-file report.json
```

JSON format: `{"layers": [{"layer_name": "...", "activations": [[...]]}]}` (`cli.py:67-69`). Loaded via `load_residual_batches_from_json` (`activations.py:33-76`). [TIER A]

Domain-contrast variant: supply `--domain-activations` and `--out-of-domain-activations` (same format as `--harmful-activations`, `cli.py:195-201`) and the CLI computes `domain_direction` at the resolved layer (`cli.py:657-662` → `_load_domain_offline`). The artifact API accepts `domain_direction` (`artifact.py:26`). [TIER A]

### Path (d): weight-delta recovery from an abliterated checkpoint

```
model-headroom-refusal \
  --weight-deltas deltas.json \
  --output-format json \
  --output-file report.json
```

`deltas.json`: `{"deltas": [[...]]}` — W - W' deltas from an already-abliterated checkpoint (`cli.py:98-101`). The direction is recovered as the top left singular vector of the horizontally-stacked deltas (`recover.py:34-48`, `recover.py:98-151`). No activations and no forward pass are required (`recover.py:1-14`). [TIER A]

On this path `capture_point = "weight_delta"` and `source = "weight_delta_recovery"` (`artifact.py:57-61`). [TIER A]

Domain-contrast variant: not meaningful here — the weight-delta path has no in-domain/out-of-domain residual sets; `domain_direction` is omitted unless you supply one externally. The CLI REJECTS `--weight-deltas` combined with any domain contrast flag (`--domain-activations`/`--out-of-domain-activations`/`--domain-prompts`/`--out-of-domain-prompts`) with exit 1 and the message "weight-deltas cannot carry a domain contrast" (`cli.py:708-716`). [TIER A — verified by probe]

### Serializing the artifact (all paths)

The CLI writes the safetensors artifact directly via `--output-format safetensors` or `--output-format both`; `--output-file` is required for those formats (`cli.py:819-827`):

```
model-headroom-refusal \
  --harmful-activations harmful_acts.json \
  --harmless-activations harmless_acts.json \
  --output-format safetensors \
  --output-file /path/to/refusal_artifact.safetensors \
  --model-id qwen3-0.6b \
  --domain-label fiction-vs-nonfiction
```

`--output-format` accepts `text`, `json`, `safetensors`, `both` (default `text`) (`cli.py:185-189`). With `safetensors`, `write_refusal_artifact` writes the artifact and returns rc 0 with empty stdout (`cli.py:853-861`). With `both`, it writes the artifact plus a JSON sidecar at `<output-file>.json` (`cli.py:862-872`). [TIER A — verified by probe]

The Python API remains the programmatic route:

```python
from model_headroom.refusal.artifact import write_refusal_artifact
from model_headroom.models.types import RefusalDirectionReport

path = write_refusal_artifact(
    report,
    "/path/to/refusal_artifact.safetensors",
    model_id="qwen3-0.6b",
    domain_label="fiction-vs-nonfiction",
)
```

[`write_refusal_artifact` — `artifact.py:21-98`; `RefusalDirectionReport` — `types.py:395`] [TIER A]

The artifact always contains tensor `refusal_direction` (float32, shape `(hidden_size,)`, L2 unit) and optionally `domain_direction` (same shape, L2 unit, only when a domain contrast is supplied) (`artifact.py:13-14, 86-88`). [TIER A]

## 3. Artifact schema

**Tensors:**

| Tensor | dtype | shape | When present |
|--------|-------|-------|--------------|
| `refusal_direction` | float32 | `(hidden_size,)` | always |
| `domain_direction` | float32 | `(hidden_size,)` | only when `domain_direction` supplied to `write_refusal_artifact` |

[`artifact.py:13-14, 86-88`] [TIER A]

**Metadata (every value a string):**

| Key | Meaning | Derivation |
|-----|---------|------------|
| `schema_version` | artifact schema version | `"1"` (`artifact.py:15, 50`) |
| `model_id` | model identifier | `model_id` argument or `""` (`artifact.py:51`) |
| `layer_name` | selected layer name | `report.layer_name` or `""` (`artifact.py:52`) |
| `layer_index` | selected layer index | `str(report.layer_index)` or `""` (`artifact.py:53-55`) |
| `hidden_size` | direction width | `str(report.hidden_size)` (`artifact.py:56`) |
| `capture_point` | where residuals were read | `"weight_delta"` if `axis=="weight_delta_svd"` else `"last_token_residual"` (`artifact.py:57-61`) |
| `source` | how the direction was derived | `"weight_delta_recovery"` if `axis=="weight_delta_svd"` else `"base_model_activation_difference"` (`artifact.py:62-66`) |
| `normalized` | direction normalized | `"true"` (`artifact.py:67`) |
| `orthogonalized` | orthogonalized against harmless mean | `"true"`/`"false"` from `report.orthogonalized` (`artifact.py:68`) |
| `cohen_d` | effect size | `"nan"` when unmeasurable, else `repr(float)`, round-trippable (`artifact.py:69-73`) |
| `cohen_d_kind` | effect-size kind | `"svd_dominance"` if weight-delta else `"effect_size"` (`artifact.py:74-76`) |
| `harmful_count` / `harmless_count` | prompt counts per side | `str(report.harmful_count)` / `str(report.harmless_count)` (`artifact.py:77-78`) |
| `selection` | how the layer was selected | `report.selection` (`artifact.py:79`) |
| `direction_status` | computation status | `report.direction_status` (`artifact.py:80`) |
| `axis` | acquisition axis | `report.axis` (`artifact.py:81`) |
| `direction_norm` | pre-normalization norm | `repr(float(pre_norm))` (`artifact.py:82`) |
| `has_domain_direction` | domain tensor present | `"true"`/`"false"` (`artifact.py:83`) |
| `domain_label` | optional domain label | `domain_label` or `""` (`artifact.py:84`) |
| `timestamp` | ISO-8601 UTC | provided or `now()` (`artifact.py:85-89`) |

[`artifact.py:49-90`] [TIER A]

### Disagreement with the JSON report

`generate_refusal_json` hard-codes `capture_point: "last_token_residual"` and `source: "base_model_activation_difference"` for **every** axis (`report.py:133-134`). On the weight-delta recovery path the artifact says `capture_point: "weight_delta"` and `source: "weight_delta_recovery"` (`artifact.py:57-61`). So the JSON report and the safetensors artifact **disagree** on `capture_point`/`source` for weight-delta recovery — the artifact is authoritative for its own origin; do not reconcile them by editing the artifact. [TIER A — read from `artifact.py` and `report.py`, both in repo]

## 4. Runtime use

The runtime module IS present in this repo: `src/model_headroom/refusal/runtime.py` (`runtime.py:1-257`). Refusal/ contains `activations.py`, `artifact.py`, `cli.py`, `extract.py`, `report.py`, `recover.py`, and `runtime.py`. Every runtime claim below is [TIER A] unless noted, backed by `model_headroom/tests/test_refusal_runtime.py` (28 passed).

- `load_artifact(path) -> RefusalArtifact` — load a safetensors artifact written by `write_refusal_artifact`. Validates the `refusal_direction` tensor is present, 1-D, unit-norm, and matches `hidden_size` metadata (`runtime.py:69-116`).
- `resolve_lambda(explicit=None) -> float` — resolves lambda; reads `SIA_REFUSAL_LAMBDA` at call time (not import time). `0.0` = bit-exact base model; `1.0` = full ablation. Rejects non-finite and unparseable values (`runtime.py:119-136`).
- `find_residual_writers(model, suffixes=DEFAULT_TARGET_SUFFIXES) -> list[tuple[str, nn.Module]]` — `DEFAULT_TARGET_SUFFIXES = ("o_proj","out_proj","down_proj","shared_expert")` (`runtime.py:21-26`); `EXTENDED_TARGET_SUFFIXES` adds `"experts"` and `"linear_attn"` (`runtime.py:27-31`). Keeps the outmost matching module and drops descendants (`runtime.py:139-156`).
- `install_refusal_ablation(model, artifact, *, strength=None, targets=None) -> AblationHandle` — installs a per-forward projection hook on the discovered residual writers; returns a handle with `.remove()` and `.set_strength()`. When `strength == 0.0` (or `DEFAULT_STRENGTH == 0.0`) the hook is a no-op returning `None` and `strength_tensor is None` (`runtime.py:203-211`); otherwise it installs an arithmetic hook (`runtime.py:204-206`).
- `verify_lambda_zero_identity(model, artifact, *, inputs, targets=None) -> bool` — compares outputs with `torch.equal` (bit-exact), not `allclose` (`runtime.py:221-245`).

**Env vars:** `SIA_REFUSAL_LAMBDA` (the lambda), `SIA_REFUSAL_ARTIFACT` (artifact path). Read at call time by `resolve_lambda` (`runtime.py:17, 122`). [TIER A]

**One-line identity verification:** with `SIA_REFUSAL_LAMBDA=0`, `verify_lambda_zero_identity` returns `True` iff outputs are bit-identical to the base model (`test_refusal_runtime.py:251-255`). [TIER A]

[All runtime: verified against `src/model_headroom/refusal/runtime.py` and `model_headroom/tests/test_refusal_runtime.py` (28 passed), TIER A]

## 5. Mathematics

**Projection:** `extract.project_out(residual, direction, strength=1.0)` computes
`component = (residual * direction).sum(dim=-1, keepdim=True)` then
`residual - strength * component * direction`, reducing over the last dimension so it applies to a vector, `[batch, hidden]`, or `[batch, seq, hidden]` (`extract.py:130-143`). [TIER A]

**Equivalence to permanent weight edit:** Arditi abliteration is `W' = W - r (r^T W)`, i.e. `delta = W' - W = -r (W^T r)^T` — rank one with left factor `r` (`recover.py:8-9`). The runtime projection applies the same direction removal to the residual stream at each hooked writer instead of rewriting `W`. The algebra is derived from the repo's own `extract.project_out` (`extract.py:130-143`) and `recover.py` formula (`recover.py:8-9`). [TIER A for the formula from `recover.py` and `extract.py`. What is NOT measured: that the per-module runtime projection is numerically equivalent to the permanent edit `W'` on a live model end to end — that equivalence is algebraic, not an executed probe.]

**SET-vs-whole-model caveat:** Hooking `o_proj`/`down_proj` ablates only those writers' contributions to `r` — not the embedding's, not earlier layers', and not a fused expert GEMM's output (`qwen3_moe.py:1171` shows the expert GEMM is fused, not per-expert `nn.Linear`, read against SGLang v0.5.23). [TIER C for the runtime claim; TIER B for the fused-GEMM source]

**Why runtime:** preferred for testing and reversibility — you can sweep `lambda`, verify bit-exactness at `lambda=0`, and remove the hook without touching weights. [TIER C]

## 6. vLLM — step-by-step

All source locations below are [TIER B]: read against upstream source at vLLM v0.33.0, not executed in this repo. Any claim that the hook fires on a live vLLM server is [TIER C] — no engine is installed (`import vllm` raises `ModuleNotFoundError`, measured).

1. The model is loaded in `GPUModelRunner.load_model` (`gpu_model_runner.py:4967`), which assigns `self.model = model or model_loader.load_model(...)` (`gpu_model_runner.py:4996`).
2. Register the refusal hook on `self.model` after load and before wrapping.
3. The model is then wrapped: `BreakableCUDAGraphWrapper(..., runtime_mode=CUDAGraphMode.PIECEWISE)` (`gpu_model_runner.py:5139`), or `CUDAGraphWrapper(..., runtime_mode=CUDAGraphMode.FULL)` (`gpu_model_runner.py:5154`), then `UBatchWrapper(...)` (`gpu_model_runner.py:5159` and `:5163`).
4. CUDA-graph wrapper ordering matters: register the hook **before** graph capture so the wrapped forward passes through your hook; if you register after capture, the captured graph bypasses it. [TIER C — ordering constraint from the contract; not executed]
5. Plugins: vLLM discovers plugins via entry points in group `vllm.general_plugins` (`plugins/__init__.py:18`, `plugins/__init__.py:42`); `load_general_plugins` carries the upstream warning that plugins may be loaded multiple times in different processes and must be idempotent (`plugins/__init__.py:77`); `allowed_plugins = envs.VLLM_PLUGINS` filters which load (`plugins/__init__.py:40`). A refusal plugin must therefore be idempotent. [TIER B]

**First-validation recipe:** start vLLM with CUDA graphs disabled, install the hook, run `verify_lambda_zero_identity` and confirm `torch.equal` returns `True` at `SIA_REFUSAL_LAMBDA=0`. [TIER C — not executed]

**Production recipe:** install the hook before graph capture, keep lambda in a device tensor (see §8), sweep lambda, remove the hook to restore the base model. [TIER C]

## 7. SGLang — step-by-step

All source locations below are [TIER B]: read against upstream source at SGLang v0.5.23, not executed in this repo. This repo's own AGENTS.md records SGLang 0.5.21 verified live on 2026-10-09 against `gfx942`; the v0.5.23 anchors are read from upstream, not executed here. Any claim that the hook fires on a live SGLang server is [TIER C] — no engine is installed (`import sglang` raises `ModuleNotFoundError`, measured).

1. The model class is resolved before engine init. Patch `resolve_model_cls` (`registry.py:93`), which reads `self.models` via `_try_load_model_cls` (`registry.py:69-72`) — this is the model-class patch point.
2. Install the refusal hook on the model **before** `init_decode_cuda_graph()` — upstream comment "Must be called BEFORE init_decode_cuda_graph() so CUDA graph capture ..." (`model_runner.py:1107`) and "canary to monkey-patch, and BEFORE init_decode_cuda_graph so warmup" (`model_runner.py:1005`). [TIER B for the source comments; TIER C for the hook-placement claim]
3. Residual writers to target (extended suffixes `"experts"` and `"linear_attn"`):
   - `qwen2.py:198` `self.o_proj = RowParallelLinear(...)` with tuple return `output, _ = self.o_proj(attn_output)` (`qwen2.py:235`) — hook `o_proj`.
   - `qwen2.py:92` `self.down_proj = RowParallelLinear(...)` with `x, _ = self.down_proj(x, forward_batch=forward_batch)` (`qwen2.py:126`) — hook `down_proj`.
   - `qwen3_moe.py:265` `self.experts = get_moe_impl_class(quant_config)(...)`; `:1171` `ckpt_down_proj_name="down_proj"` — the expert GEMM is fused, so the residual writer is the `experts` module itself (`qwen3_moe.py:265`, `:1171`). [TIER B]
   - `qwen4_exp.py:1787` `hidden_states = self.linear_attn(hidden_states, forward_batch)` — the Gated-DeltaNet path is a separate residual writer; `:1973` `output, _ = self.o_proj(attn_output)`. [TIER B]
4. Hidden states: enable with `--enable-return-hidden-states` (the old boolean `--disable-cuda-graph` spelling is deprecated, `no_cli=True`, `server_args.py:408`; `cuda_graph_config.py` `parse_cuda_graph_config_arg`). The hidden-states mode is `get_server_return_hidden_states_mode()` reading `features.enable_return_hidden_states` (`forward_batch_info.py:324`, `:329`). [TIER B]

**First-validation recipe:** start SGLang with `--enable-return-hidden-states` and CUDA graphs disabled, install the hook before `init_decode_cuda_graph()`, confirm `torch.equal` bit-exactness at `SIA_REFUSAL_LAMBDA=0`. [TIER C — not executed]

## 8. The lambda / CUDA-graph hazard

A python `float` lambda is constant-folded into a captured CUDA graph at capture time, so replaying the graph reuses the stale value — a silent wrong lambda, not an error. [TIER C — contract; not executed, no engine installed]

**Mitigation:** lambda lives in a persistent 0-dim device tensor, so a captured CUDA graph reads its memory at replay instead of constant-folding a stale value. Update the tensor in place to change lambda at runtime. [TIER C — contract]

**`set_strength` after graph capture:** `set_strength` calls `tensor.fill_(float(value))` on the persistent 0-dim device tensor (`runtime.py:60`), which IS the device-tensor update that takes effect at replay because the hook reads `strength_tensor` from device memory on every forward (`runtime.py:159-179`). It does NOT read a captured constant. What `set_strength` CANNOT do: convert a handle installed in no-op mode (lambda == 0 at install, so `strength_tensor is None`, `runtime.py:203-211`) into an arithmetic hook — that raises `RuntimeError` by design ("cannot set a non-zero strength on a no-op handle...", `runtime.py:53-57`). Install the arithmetic hook (strength != 0.0) **before** capturing a CUDA graph if you intend to change lambda after capture. [TIER A — corrected against `runtime.py:50-61`, `runtime.py:203-211`, `runtime.py:159-179`; `test_set_strength_noop_nonzero_raises` at `test_refusal_runtime.py:241-248`]

## 9. Version tracking

**Pins:** vLLM v0.33.0 and SGLang v0.5.23 are the upstream anchors read for this document (`forward_batch_info.py:329`, `gpu_model_runner.py:4967`, etc.). [TIER B]

**Re-validate on each major/minor bump:** hook registration point, graph compatibility, residual points unchanged, and `lambda=0` remains bit-exact (`torch.equal`). Record last-validated versions in this file. Any change to residual-writing modules or the execution model may require hook updates. [TIER C]

**Last-validated record (this document):** executed in this repo — artifact schema, CLI text/json/safetensors/both paths, `project_out`, `generate_refusal_json`, `discover_residual_layers`, `capture_last_token_residuals`, weight-delta recovery, runtime install/set_strength/verify_lambda_zero_identity (all [TIER A]); not executed here — any live vLLM/SGLang behaviour ([TIER B/C], no engine installed). [TIER A for the record itself]

LAST EXECUTED ENGINE VALIDATION: SGLang 0.5.2 + in-process transformers live on Tesla T4 (SM75), 2026-11-19 — see "Live execution update" in the header. (Repo default env: no engine installed; `import vllm` and `import sglang` both raise `ModuleNotFoundError`, measured 2026-11-12.)
LAST READ-AGAINST-UPSTREAM: vLLM v0.33.0 and SGLang v0.5.23 (source locations read against upstream tags, not executed in this repo).

## 10. Limitations

- Single direction: one `refusal_direction` per artifact; the residual is not perfectly 1-D, so a rank-1 projection is an approximation. [TIER C — contract principle]
- Graph-capture constraints: hook must be installed before CUDA graph capture; a python-float lambda is constant-folded (§8). [TIER C]
- Per-forward hook cost: the hook runs on every forward of every targeted module — measure it before production use. Concretely: `_make_arithmetic_hook` casts the output to float32, runs `project_out`, and casts the result back to the module output's dtype (`runtime.py:159-179`). That is one float32 temporary of 2x the activation size per hooked call, plus one result cast. There is no host sync in the hook (no `.item()`, no `float()`), so the device-tensor lambda is still read from device memory at replay. Returned dtype equals input dtype for bfloat16, float16 and float32 (`test_refusal_runtime.py:270-278`). On a bf16 model the hook computes the projection in float32 and casts the result back to bf16; effective-lambda drift from bf16 rounding of the result is ~0.006 at lambda=0.3 vs a float32 reference (the direction and the lambda are never quantised). [TIER A — measured from `runtime.py` and the dtype test]
- Fused-MoE writers: for fused expert GEMMs the residual writer is the `experts` module itself, not per-expert `nn.Linear` (`qwen3_moe.py:1171`, SGLang v0.5.23). [TIER B for the source; TIER C for the runtime impact]
- The domain vector is optional and the domain GATE is not implemented here (roadmap only). [TIER C]

## 11. Selective intervention

Domain gating is the recommended way to keep this selective: enable the residual projection only when both (i) domain match `cos(h, c_domain) >= tau_domain` and (ii) refusal-pressure `z(h · r) >= tau_refusal` fire, for a request-scoped intervention. **This gate is ROADMAP ONLY — not implemented.** Reference the roadmap item "DOMAIN-GATED SELECTIVE INTERVENTION". sia-hu's responsibility stops at emitting the unit vectors (`refusal_direction` + optional `domain_direction`); the gate and the decision to enable the hook live in the serving runtime. [TIER C — roadmap item; not implemented]

## 12. Validation tiers

| Tier | Meaning | Basis |
|------|---------|-------|
| [TIER A] VERIFIED IN THIS REPO | Executed probe in this environment | File:line in this repo + measured output |
| [TIER B] READ FROM UPSTREAM SOURCE | Read against upstream source at the stated tag | Upstream tag + file:line, not executed here |
| [TIER C] NOT VERIFIED | Explicitly unknown / contract-specified, not present or executed here | No probe exists in this repo |

No inference engine is installed (`import vllm` and `import sglang` both raise `ModuleNotFoundError`, measured 2026-11-12), so **nothing about live engine behaviour is tier A**. Engine registration points are tier B (source locations) / tier C (hook-firing claims). [TIER A — measured]

Every engine-specific and behaviour-specific claim in this document is labelled inline with its tier. The complete mapping is the **TIER LEDGER** in Appendix A below (shipped inside this document; no separate ledger file is maintained).

## Appendix A — Tier ledger

This appendix is the complete mapping of every claim in this document to its tier and its evidence. It is shipped inside this document; there is no separate "TIER LEDGER maintained alongside this document" file.

| § | Claim | Tier | Evidence |
|---|-------|------|----------|
| Header | `import vllm` / `import sglang` raise ModuleNotFoundError | A | measured, 2026-11-12 |
| Header | refusal tests run in root pytest suite via testpaths | A | `pyproject.toml:97` |
| §1 | refusal direction = normalised difference of means | A | `extract.py:267-278`, `extract.py:70-77` |
| §1 | runtime ablation via `extract.project_out` | A | `extract.py:130-143` |
| §1 | weights unmodified; hook installed/removed | C | `runtime.py:63-66`; not live-measured |
| §1 | `DEFAULT_TARGET_SUFFIXES` / `EXTENDED_TARGET_SUFFIXES` | A | `runtime.py:21-31`; `test_refusal_runtime.py` 28 passed |
| §1 | not permanent; `handle.remove()` restores base | A | `test_refusal_runtime.py:119-128` |
| §1 | not a headroom signal | A | `refusal/__init__.py:7-9` |
| §1 | not guaranteed on live vLLM/SGLang | C | no engine installed |
| §1 | not a domain gate | C | roadmap only |
| §2 (a) | `--model-path`, `--harmful-prompts`, `--harmless-prompts` | A | `cli.py:75-85` |
| §2 (a) | `transformers` lazy import | A | `activations.py:10-11`, `activations.py:193-196` |
| §2 (a) | `--domain-prompts`/`--out-of-domain-prompts` present | A | `cli.py:203-209` |
| §2 (a) | domain contrast wiring (sglang + live) | A | `cli.py:550`, `cli.py:723` |
| §2 (b) | `--sglang-url` over HTTP | A | `cli.py:87-91`, `cli.py:718-759` |
| §2 (b) | `--enable-return-hidden-states` required | A | `activations.py:275-277` |
| §2 (b) | hidden-states → `features.enable_return_hidden_states` | B | `forward_batch_info.py:329`, SGLang v0.5.23 |
| §2 (b) | final pre-lm_head layer only | A | `activations.py:397-400` |
| §2 (b) | domain contrast via `_capture_domain_sglang` | A | `cli.py:550`, `cli.py:723` |
| §2 (c) | `--harmful-activations`/`--harmless-activations` JSON | A | `cli.py:67-69`, `activations.py:33-76` |
| §2 (c) | `--domain-activations`/`--out-of-domain-activations` | A | `cli.py:195-201` |
| §2 (d) | `--weight-deltas` recovery, no forward pass | A | `cli.py:98-101`, `recover.py:1-14`, `recover.py:34-48` |
| §2 (d) | `capture_point`/`source` for weight-delta | A | `artifact.py:57-61` |
| §2 (d) | CLI rejects domain flags with `--weight-deltas` (rc 1) | A | `cli.py:708-716`; verified by probe |
| §2 | `--output-format {text,json,safetensors,both}` | A | `cli.py:185-189` |
| §2 | `--output-file` required for safetensors/both | A | `cli.py:819-827`; verified by probe |
| §2 | `both` writes artifact + `<file>.json` sidecar | A | `cli.py:862-872`; verified by probe |
| §2 | `write_refusal_artifact` Python API | A | `artifact.py:21-98` |
| §3 | artifact tensor schema | A | `artifact.py:13-14, 86-88` |
| §3 | metadata table derivations | A | `artifact.py:49-90` |
| §3 | JSON/artifact `capture_point`/`source` disagreement | A | `report.py:133-134`, `artifact.py:57-61` |
| §4 | `runtime.py` present; five public functions | A | `runtime.py:1-257`; `test_refusal_runtime.py` 28 passed |
| §4 | `load_artifact` validation | A | `runtime.py:69-116` |
| §4 | `resolve_lambda` reads env at call time | A | `runtime.py:119-136` |
| §4 | `find_residual_writers` suffixes | A | `runtime.py:21-31`, `runtime.py:139-156` |
| §4 | `install_refusal_ablation` no-op at strength 0 | A | `runtime.py:203-211` |
| §4 | `verify_lambda_zero_identity` uses `torch.equal` | A | `runtime.py:221-245` |
| §5 | `project_out` formula | A | `extract.py:130-143` |
| §5 | Arditi `W' = W - r(r^T W)` | A | `recover.py:8-9` |
| §5 | per-module equivalence not live-measured | C | algebraic only |
| §5 | fused expert GEMM writer | B/C | `qwen3_moe.py:1171`, SGLang v0.5.23 |
| §6 | vLLM registration points | B | `gpu_model_runner.py:4967,4996,5139,5154,5159,5163`; v0.33.0 |
| §6 | vLLM plugin idempotency | B | `plugins/__init__.py:18,40,42,77`; v0.33.0 |
| §6 | hook before graph capture | C | contract; not executed |
| §7 | SGLang registration points | B | `registry.py:69-72,93`; `model_runner.py:1005,1107`; v0.5.23 |
| §7 | Qwen2/Qwen3-MoE/Qwen4-Exp writers | B | `qwen2.py:92,198,235`; `qwen3_moe.py:265,1171`; `qwen4_exp.py:1787,1973`; v0.5.23 |
| §7 | hidden-states flag/mode | B | `server_args.py:408`; `cuda_graph_config.py`; `forward_batch_info.py:324,329`; v0.5.23 |
| §7 | hook before `init_decode_cuda_graph` | C | contract; not executed |
| §8 | float lambda constant-folded into graph | C | contract; not executed |
| §8 | lambda in persistent 0-dim device tensor | C | contract |
| §8 | `set_strength` = `fill_` on device tensor; takes effect at replay | A | `runtime.py:50-61`, `runtime.py:159-179`; `test_refusal_runtime.py:241-248` |
| §8 | `set_strength` on no-op handle raises RuntimeError | A | `runtime.py:53-57`; `test_refusal_runtime.py:241-248` |
| §9 | vLLM v0.33.0 / SGLang v0.5.23 anchors | B | read against upstream tags |
| §9 | last-validated record: NONE executed engine | A | measured 2026-11-12 |
| §10 | hook dtype behaviour + cost | A | `runtime.py:159-179`; `test_refusal_runtime.py:270-278` |
| §10 | single direction rank-1 approximation | C | contract principle |
| §10 | graph-capture constraints | C | contract |
| §10 | fused-MoE writer impact | B/C | `qwen3_moe.py:1171` |
| §10 | domain gate not implemented | C | roadmap only |
| §11 | selective intervention gate | C | roadmap only |
| §12 | tier definitions | A | this document |
| Appendix | tier ledger shipped inside document | A | this document |
