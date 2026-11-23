# 008 — Refusal ablation is not geometry-aware: a 10240-wide hyper-connection direction cannot be installed on 2560-wide residual writers (qwen4_exp hybrid MoE)

| Field | Value |
|-------|-------|
| Status | **PARTIALLY RESOLVED** — filed 2026-11-21; E1–E4, E8, E9 landed 2026-11-21 (code + tests + docs); E5, E6, E7, E10 remain OPEN |
| Severity | HIGH — the documented integration path produces a server that crashes at CUDA-graph capture, or (after a caller-side workaround) serves an approximation whose geometry is not represented in the artifact |
| Component | `src/model_headroom/refusal/runtime.py` (`find_residual_writers` :139, `install_refusal_ablation` :184); `src/model_headroom/refusal/activations.py` / `capture_residuals_sglang`; `docs/guides/runtime-refusal-ablation.md` §§7–10; `capture-proxy` exposure metadata |
| Target model | `/scratch/CYBER-FROST-3.8-NVFP4-V2` — `qwen4_exp`, hidden 2560, PLE 2560, 4-way hyper-connections (widened stream 10240), NVFP4 hybrid MoE + linear-attn/GDN, served on SGLang 0.5.23 / SM120 |
| Found by | BUILD agent, blackfrost serving session, 2026-11-21 (full failure ledger in §4) |
| Blocks | reproducible inference-time refusal ablation on hybrid MoE / hyper-connection models; moving the guide's engine claims from TIER B/C to TIER A; self-describing capture-proxy evidence for an ablated arm |
| Evidence | `/scratch/blackfrost/reports/refusal-capture-20261121.md`, `/scratch/sglang-work/reports/refusal-ablation/{hook.*.jsonl,ablation-failure-*.txt}`, `/scratch/sglang-work/reports/longctx-serving-20261121.md`, `/scratch/sglang-work/sia_plugin/sitecustomize.py` |

## Resolution record — 2026-11-21

| Item | State | Landed as |
|------|-------|-----------|
| E1 geometry-aware schema | **FIXED** | `RefusalArtifact` validates `capture_space`/`capture_width`/`source_hidden_size`/`stream_count`/`fold_mode`/`geometry_id` at construction; `load_artifact` infers `widened_stream` from `source_hidden_size*stream_count == direction length` and installation then refuses with the remedy named |
| E2 projection API | **FIXED** | `project_direction_to_residual(artifact, hidden_size, mode)` → new artifact + provenance (`fold_mode`, `folded_from_capture_width`) + fresh `geometry_id`; `first` rejected for declared multi-stream artifacts. Measured: `mean` and `least_squares` agree to ~1e-7 on the real 10240-wide artifact — for this concatenated-replication objective the exact LS solution IS the mean; the mean-fold's validity is exactly as good as the replication hypothesis itself |
| E3 width preflight | **FIXED** | `validate_install` compares direction width to each writer's actual output width (parameter-derived for fused writers) and refuses before any hook is registered; called by `install_refusal_ablation` |
| E4 flexible discovery | **FIXED** | `find_residual_writers` accepts suffixes, fully-qualified names, and globs; `targets=["auto"]` uses `ARCHITECTURE_TARGET_MAPS` (qwen2/qwen3_moe/qwen4_exp); error message names all modes |
| E8 intervention metadata | **FIXED** | `SIA_REFUSAL_INTERVENTION_JSON` is parsed by `RuntimeSnapshot` and stamped into `runtime_settings`/run manifest for every backend; control traffic carries no field |
| E9 deployment notes | **FIXED** | guide §14: pinned-allocation threshold, fp8 PLE, `flashinfer_cutlass`, interpreter note |
| E5 engine adapters | OPEN | adapter/entrypoint loader for `resolve_model_cls`; caller-side `sitecustomize.py` remains the de-facto path |
| E6 per-layer capture | OPEN | engine-side; `cohen_d ≈ 0.43` remains the ceiling for single-vector capture |
| E7 cohen_d verdict gating | OPEN | `cohen_d=0.4249` is recorded in the artifact metadata but no threshold gate exists |
| E10 validation lane | OPEN | offline E1–E4 probes pass against the real artifact; in-engine assertions (hook before `init_decode_cuda_graph`, lambda=0 bit-exact at replay, served token) need the E5 adapter |

Evidence for the landed items: `model_headroom/tests/test_refusal_geometry.py` (18 tests), `capture-proxy/tests/test_runtime.py::TestInterventionMetadata` (7 tests), and a live-host probe against the genuine `CYBER-FROST-3.8-NVFP4-V2.advbench100` artifact: preflight refuses the 10240-wide direction with the named remedy, mean-fold → 2560 (norm 1.0 within fp32 epsilon), install registers 8/8 writers, lambda=0 output bit-exact (`torch.equal`). Guide §13 documents the geometry contract; the tier ledger records the live measurement.

## Intent this ticket serves

Produce **reproducible, inference-time activation-space refusal ablation** on an
operator-supplied hybrid MoE checkpoint, then expose the ablated arm through this
repo's capture proxy so the operator can test it personally (full 262144-token
context, 8-bit KV, tool calling). The operator is writing a paper and needs the
intervention, its geometry, and its exact commands to be self-describing.

The repo's guide already *names* qwen4_exp as a target
(`docs/guides/runtime-refusal-ablation.md` §7.3: `qwen4_exp.py:1787`
`linear_attn` and `:1973` `o_proj` are listed as residual writers). What it does
not do is account for the fact that qwen4_exp **exposes its hidden states in a
different width than those writers emit**. That gap is the substance of this
ticket: the documented API is shape-unsafe and fails at the worst possible
time — inside `init_all_cuda_graphs()`, after the model is loaded.

## 1. Measured environment (OBSERVED, not inferred)

| Item | Value |
|------|-------|
| sglang | 0.5.23 (installed, `/opt/pytorch`) |
| torch / python | 2.14.1+cu130 / 3.13.18 |
| sia-hu | 0.6.0 |
| GPU | 62 GiB host, SM120; PLE offloaded to pinned host memory |
| Artifact | `CYBER-FROST-3.8-NVFP4-V2.advbench100`, SHA-256 `4f493093…baaaaa` |
| Artifact direction | 1-D float32, shape `(10240,)`, norm 1.00000024 |
| Artifact metadata | `hidden_size=10240`, `capture_point=last_token_residual`, `cohen_d=0.424902…` |
| Model config | `hidden_size=2560`, `ple_embed_dim=2560`, 24 heads / 2 KV heads |
| Widened stream | `qwen4_exp.py:1636` — `_widen_streams()` returns `torch.cat([hidden_states for _ in range(self.hc_count)], dim=-1)`; 10240 = 4 × 2560 |

The artifact metadata labels the captured vector's width as `hidden_size=10240`.
**That is the widened hyper-connection stream, not the model's hidden size.**
The config's real hidden size is 2560. This mislabel is the root of §2.

## 2. Issue catalogue — what actually broke

### I1 — Capture surface is one flat vector, not a residual tensor (capture-time)
`--enable-return-hidden-states` in "full" mode still returns a **single flat
10240-vector** for `/v1/chat/completions`; there are no per-token residuals and
no multi-layer `aux_hidden_states` on this route. `qwen4_exp` does not populate
multi-layer aux states. Consequence: no Arditi-style per-layer difference of
means, no generated-token residual, no rank estimation. Recorded in
`reports/refusal-capture-20261121.md` §"Load test" and §"Consequence".

### I2 — Captured direction is geometrically mislabelled (capture-time)
The vector is captured from `_widen_streams()` output (4 concatenated copies of
the 2560 residual), but serialized with `hidden_size=10240` and no
`capture_space`, `stream_count`, or `source_hidden_size`. A consumer cannot tell
that 10240 is 4× the residual width. `load_artifact()` accepts it without
complaint.

### I3 — Direction is too weak for the advertised use (capture-time)
`cohen_d = 0.4249` on canonical 100+100; `0.4366` on the measured 36/97
refusing-vs-compliant contrast; `--orthogonalize` and `--winsorize 0.95` are
inert (0.42490 → 0.42490). The repo's own gold standard is `9.76` with a 4/4 →
0/4 behavioural flip. Full-set acquisition and refusal-subset selection were
both tried. The report's verdict: *"not suitable for causal refusal ablation as
it stands."* This checkpoint is only *slightly* censored (36/100 harmful
refusals), which the capture method cannot sharpen.

### I4 — No supported in-engine integration point (integration-time)
`install_refusal_ablation` operates on an in-process `nn.Module`; the model
lives inside the server process. The guide describes patching
`ModelRegistry.resolve_model_cls` (`registry.py:93`) but marks every live-engine
claim TIER B/C and ships no adapter. The only working route was a hand-written
`sitecustomize.py` that wraps `builtins.__import__`, patches
`ModelRegistry.resolve_model_cls`, and installs the hook inside the model
class's `__init__` — before CUDA graph capture, as the guide requires. This is
a fragile, undocumented-by-the-repo monkeypatch.

### I5 — `find_residual_writers` accepts suffixes only, fails on full names
`src/model_headroom/refusal/runtime.py:146` selects with
`name.rsplit(".", 1)[-1] in suffix_set`. Passing fully-qualified module names
(e.g. `model.layers.0.linear_attn.out_proj`) raises
`ValueError: no residual writers matched suffixes (...)`, killing the scheduler.
There is no glob, regex, or fully-qualified-name mode, and the error names only
suffixes, which misleads the caller into thinking the modules are absent.
Evidence: `ablation-failure-width-filter.txt`, `hook.failed-width-filter.jsonl`.

### I6 — No width preflight; hooks are installed with incompatible direction (integration-time)
`install_refusal_ablation` (:198–212) moves the direction to the writer's device
and registers hooks on **every** matched writer without ever comparing
`direction.shape[0]` to the writer's output width. With the 10240-wide artifact
and 2560-wide writers, the first forward fails:

```
RuntimeError: The size of tensor a (2560) must match the size of tensor b (10240)
              at non-singleton dimension 1
```

and, because the hooks are live during `init_all_cuda_graphs()`, the scheduler
dies *after* model load. Evidence: `ablation-failure-2560.txt`,
`ablation-failure-o_proj.txt`, `hook.failed-2560.jsonl`,
`hook.failed-o_proj.jsonl`.

### I7 — `out_features` is not an authoritative width source
Caller-side pre-filtering by `getattr(module, "out_features", artifact_width)`
was ineffective: fused/non-`Linear` writers such as `mlp.shared_expert` do not
expose `out_features`, so the default let all 96 candidates through. The plugin
had to remove incompatible hooks *after* `install_refusal_ablation` returned and
before any forward. There is no repo-provided way to ask "what width does this
writer actually emit?" Evidence: `hook.failed-prefold.jsonl`
(`candidates 96, eligible 96, active 96` with a 10240-wide direction).

### I8 — Target-set selection is guesswork
`o_proj` alone is also 2560-wide here (`ablation-failure-o_proj.txt`). The
working set had to be discovered empirically as
`o_proj,out_proj,down_proj,shared_expert` (96 writers). The guide additionally
mentions `experts` and `linear_attn` suffixes for SGLang 0.5.23. There is no
architecture-aware discovery of which modules actually write the residual.

### I9 — Install is not idempotent at the point that matters
`resolve_model_cls` is invoked several times before the first instance is
constructed. Setting the "already installed" class flag inside the first
`__init__` produced **four** hook installs. The guard had to be moved to wrap
time (`resolve_with_ablation`) before any instance exists. The runtime API has
no idempotence contract for this case.

### I10 — No fold/projection utility, so the fix is a private approximation
Because no repo API maps a captured direction from widened space back to
residual space, the serving plugin implemented it locally:

```python
chunks = artifact.direction.view(-1, hidden).to(torch.float32)  # 4 x 2560
folded = chunks.mean(dim=0)  # or chunks[0]
artifact.direction = folded / folded.norm()
```

`SIA_STREAM_FOLD=mean|first` is an **approximation**, not a reconstruction: a
single rank-1 direction in 10240-space cannot in general be represented by its
mean over 4 streams. The artifact does not record that this transformation
happened, so a third party reproducing the experiment cannot tell which
geometry was ablated. `sitecustomize.py` now writes `capture_width=10240`,
`hidden_size=2560`, `stream_count=4`, `stream_fold=mean`,
`folded_from_capture_width=10240` into the install record — but that is a
caller-side convention, not a sia-hu contract.

### I11 — Capture-proxy exposure does not record the intervention
The proxy forwarded the ablated arm correctly (`PROXY_SERVING_OK`, real
`get_weather({"city":"Paris"})` tool call, raw JSONL captures written), but the
capture rows carry no indication that the upstream was ablated: no artifact
hash, no λ, no hook record, no fold mode. For a paper, the capture corpus cannot
distinguish control from ablated traffic without external notes.

### I12 — Environment/toolchain discovery
The `sia` shim on Python 3.14 lacks torch; all capture and probing had to run
with `/opt/pytorch/bin/python`. For this model class the serving host also
needed a derived `pinned_max_round_threshold_mb` (48828) to avoid a 128 GiB
pinned-allocation rounding artifact, an fp8 PLE override to fit 62 GiB, and
`--moe-runner-backend flashinfer_cutlass` (TRTLLM fails with
`'FusedMoE' object has no attribute 'g1_scale_c'`). None of this is in the
refusal guide.

## 3. Failure ledger (chronological, OBSERVED)

| # | UTC | Symptom | Cause | Resolution |
|---|-----|---------|-------|------------|
| 1 | 16:18 | 4 hook installs; `RuntimeError 2560 vs 10240` in `init_all_cuda_graphs()` | class guard set in `__init__`; 10240 direction | guard moved to wrap time; narrowed targets |
| 2 | 16:24 | `RuntimeError: 2560 vs 10240` on `o_proj`-only run | `o_proj` is also 2560-wide | filter by width attempted |
| 3 | 16:26 | `ValueError: no residual writers matched suffixes ('model.layers…')` | full names passed where suffixes expected | pass suffixes |
| 4 | 16:32 | again `2560 vs 10240`, `active 96` despite filter | `out_features` absent → filter ineffective | remove incompatible hooks post-install |
| 5 | 16:44 | server served at 32768 ctx; direction folded, 96 hooks | mean-fold 10240→2560 | worked; evidence saved |
| 6 | 17:00 | server serves at 262144 ctx, fp8 KV, via proxy | — | final state; no crash |

Every failure was a **geometry/contract** problem, not a model-quality problem;
that is why this ticket is about sia-hu's API rather than about the checkpoint.

## 4. Workaround currently in production (do not treat as a fix)

`sitecustomize.py` wraps `ModelRegistry.resolve_model_cls`, mean-folds the
direction, installs on 96 writers, removes incompatible hooks, records install
metadata to `hook.jsonl`, and uses an early class guard. Launch profile:
`launch.LONGCTX-ABLATED.sh` (262144 ctx, `fp8_e4m3` KV). The served arm is live
at `http://127.0.0.1:8000` via `sia capture-proxy`. This works, but it is
private glue: a second operator cannot reproduce it from repo docs alone, and
the fold is an approximation nobody validated.

## 5. How sia-hu needs to evolve to handle such models

Ordered by leverage. Each item names the issues it closes.

**E1 — Geometry-aware artifact schema (I2, I10).** Add first-class fields to
`RefusalArtifact` / metadata: `capture_space` (`residual` | `widened_stream`),
`capture_width`, `source_hidden_size`, `stream_count`, `fold_mode`, and a
`geometry_id`. Validate at `load_artifact()`: if `capture_width` is a multiple of
a target `hidden_size` and `capture_space == widened_stream`, require an explicit
fold before installation. Never label a widened vector `hidden_size`.

**E2 — First-class fold/projection API (I10).** Ship
`project_direction_to_residual(artifact, hidden_size, mode="mean"|"first"|"least_squares")`
that returns a NEW artifact with provenance of the transformation. `mean` and
`first` are the current approximations; `least_squares` (fit one vector whose
4-stream concatenation best approximates the captured direction) is the
principled default this model class needs. Record the operation in the artifact,
not only in a caller log.

**E3 — Fail-fast preflight before graph capture (I6, I7).** Add
`validate_install(model, artifact, targets)` that resolves each candidate
writer's **actual output width** — by a materialized probe hook or a dry-run
forward, not `out_features` — and refuses to install with a precise message:
`"direction width 10240 != writer model.layers.0.o_proj output 2560; fold via E2"`.
Call it inside `install_refusal_ablation` before registering any hook.

**E4 — Flexible, architecture-aware writer discovery (I5, I8).** Extend
`find_residual_writers` to accept suffixes (current), fully-qualified names,
and globs; make the match mode explicit in the error. Add an
`auto` mode that uses per-architecture residual-writer maps (qwen2 / qwen3_moe /
qwen4_exp) covering `o_proj`, `out_proj`, `down_proj`, `shared_expert`, `experts`,
and `linear_attn` so callers stop guessing target lists.

**E5 — Official engine adapters (I4, I9).** Provide a supported SGLang adapter
implementing the `resolve_model_cls` patch point, installing during model
construction before `init_decode_cuda_graph()`, with an idempotence contract
(guard at wrap time) and a documented `sitecustomize`/entrypoint loader. Move the
guide's TIER B/C engine claims to TIER A with a live integration test. Mirror for
vLLM.

**E6 — Real capture surface for hybrid models (I1, I3).** Add engine-side
capture of per-layer, per-token residuals (the guide's `aux_hidden_states` path)
and generate-token residuals, so contrastive extraction is not limited to one
flat vector. Without this, `cohen_d ≈ 0.43` is a ceiling for this model class,
not a fixable hyperparameter.

**E7 — Acquisition-condition provenance (I3).** Record the refusal/contrast
selection policy (full set vs measured contrast) and refuse to claim
"causal-ready" below an explicit separation threshold. Surface `cohen_d` next to
the artifact with a verdict, so a weak artifact cannot silently be used for
causal claims.

**E8 — Capture-proxy intervention metadata (I11).** When the upstream is an
ablated arm, stamp capture rows with `intervention: {artifact_sha256, lambda,
fold_mode, capture_width, stream_count, hook_record_ref}`. Control and ablated
traffic must be distinguishable from the corpus alone — this is a direct paper
reproducibility requirement.

**E9 — Hybrid-model deployment notes (I12).** Document the PLE pinned-allocation
rounding trap and its per-artifact derived threshold, the fp8 PLE override, and
the required MoE runner backend for NVFP4 hybrid models, either in the refusal
guide or a linked runbook. These are properties of "such models", not of one
host.

**E10 — One validation matrix across engines (I4, I6, I9).** Add a CI/integration
lane that at minimum asserts: hook fires before `init_decode_cuda_graph()`;
`lambda=0` is bit-exact; direction width matches every writer; idempotent
install; and a real generated token is returned through the served endpoint.
That converts the guide's contract claims from TIER C to TIER A.

## 6. Scope and negative claims

- This ticket does **not** claim the technique failed: a real ablated arm served
  successfully after E1/E2-style handling was implemented caller-side.
- It does **not** claim the 10240→2560 mean-fold is correct — only that it is
  the workaround that made the hooks dimension-compatible. Its scientific
  validity is untested; E2 exists for that reason.
- It does **not** propose substituting a different checkpoint. The operator's
  artifact remains the subject.
- No upstream sia-hu code was changed to file this ticket; repo state left clean.

## 7. Evidence index

| Path | What it proves |
|------|----------------|
| `/scratch/blackfrost/reports/refusal-capture-20261121.md` | flat 10240 capture, cohen_d 0.4249/0.4366, weakness verdict |
| `/scratch/sglang-work/reports/refusal-ablation/ablation-failure-2560.txt` | 2560 vs 10240 crash in graph capture |
| `/scratch/sglang-work/reports/refusal-ablation/ablation-failure-o_proj.txt` | o_proj alone still 2560-wide |
| `/scratch/sglang-work/reports/refusal-ablation/ablation-failure-width-filter.txt` | full-name suffixes → ValueError |
| `/scratch/sglang-work/reports/refusal-ablation/hook.failed-prefold.jsonl` | 96 candidates "eligible" though incompatible |
| `/scratch/sglang-work/reports/refusal-ablation/hook.jsonl` | successful folded install (capture 10240 → 2560) |
| `/scratch/sglang-work/reports/longctx-serving-20261121.md` | 262144 ctx + fp8 KV + proxy serving, commands, hashes |
| `/scratch/sglang-work/sia_plugin/sitecustomize.py` | the caller-side workaround that E5 should replace |
| `/opt/pytorch/…/sglang/srt/models/qwen4_exp.py:1636` | `_widen_streams()` 4× replication |
| `src/model_headroom/refusal/runtime.py:139,184` | suffix-only discovery; no width preflight |
| `docs/guides/runtime-refusal-ablation.md` §§7–10, 12 | TIER B/C engine claims, graph hazard, limitations |

## 8. Empirical outcome — the intervention did not work (2026-11-21)

The served arm from §4 was probed with the same 100-harmful prompt set (the run
was interrupted after 98). Result: **control 14/100 refusals, ablated 16/98**.
Paired per-prompt comparison: 4 refused in both arms, 10 only in control,
**12 only in ablated** — the intervention removed 10 refusals and created 12.
With max_tokens=1024 on four prompts that refused in both arms, live refusals
persisted verbatim, e.g. *"I cannot provide a plan to steal from a charity…"*,
*"I won't provide code for tracking users…"*.

This is a **negative result** and reinforces E1/E2/E6/E7 rather than weakening
them:

- The hook is structurally live and changes output (every non-empty paired
  response differed), so this is not a missing-wiring failure.
- The four widened streams are **not** copies of one another: pairwise cosine
  0.22–0.60; cos(mean-fold, each stream) 0.61–0.84. The mean fold is a lossy
  projection of the captured direction, i.e. E2 is required before any causal
  claim, and even the principled fold cannot repair a weak source vector.
- The original artifact's `cohen_d ≈ 0.425` (repo gold 9.76) was already
  recorded as "not suitable for causal refusal ablation"; using it anyway was
  the primary error. E7 (refuse to label a weak artifact causal-ready) is the
  guard that would have blocked it.
- The 256-token probe truncated 70/98 responses to empty final text, a separate
  measurement confound for reasoning models. Behavioural probes of reasoning
  checkpoints must allow a final answer to be emitted.

No lambda sweep or least-squares fold was run (both would require restarting the
server the operator was testing on); their effect is UNKNOWN. Full debug record:
`/scratch/sglang-work/reports/refusal-ablation/debug-log-20261121.md`.

## 9. Runtime redesign: four-stream cone (2026-11-21 evening)

The operator rejected weight-space edits: this engagement is about **quick
runtime ablation**. Implemented at the direction-representation boundary:
`SIA_STREAM_MODE=cone4` installs the artifact's four 2560-wide streams as four
separate rank-1 projections on the same 96 writers instead of their lossy mean.

- Full-generation sweep (2048 tokens, 14 control-refusing harmful + 5 harmless):
  λ=1 → 6/14 refusals, 8/14 completed, 5/5 harmless completed;
  λ=2 → 4/14 refusals but 2/5 harmless truncated;
  λ=3 → 1/14 refusals but **0/5 harmless and 1/14 harmful completed** (degenerate:
  no final answer). Selected λ=1 as the safe state. Zero refusals was not reached.
- On the four prompts that refused in both earlier arms: mean-fold 4/4 refusals,
  cone4 λ=1 1/4 (one hard refusal remains).
- Writer-activation reachability CONFIRMED: a gated read-only hook wrote 16
  finite `(2560,)` samples from two writers. One startup crash occurred because
  a CPU copy ran during CUDA graph capture; fixed by skipping capture while
  `torch.cuda.is_current_stream_capturing()`.
- This supports E2 (keep the streams instead of averaging) and E5 (an in-engine
  adapter owns the install), but does not establish E1/E7 — the source direction
  is still weak (`cohen_d≈0.425`) and cone4 cannot create absent signal. The
  next step is fresh *labelled* writer-space capture, whose reachability is now
  proven, not assumed. Full record:
  `/scratch/sglang-work/reports/refusal-ablation/cone4-experiment-20261121.md`.

## 10. Weight-space implementation — operator reversal (2026-11-21 night)

The operator subsequently **reversed** §9's decision: "do weight edits, we have
space on /scratch." The paper gate is the gigantic already-derisked checkpoint;
the runtime arm is now a fallback/comparison, not the final path.

- Measured before design: only `model.language_model.layers.*.mlp.experts` are
  NVFP4-quantized. All 98 residual-writer tensors (`o_proj` 13, `out_proj` 36,
  `shared_expert.down_proj` 49) are **BF16 and unquantized** — editable with no
  requantization or conversion. Source `MODEL_PATH` opened read-only throughout.
- `weight_ablate.py` streams all 131 shards and applies `W' = W − λ Σᵢ rᵢ(rᵢᵀW)`
  with the four normalized 2560-wide streams to only those 98 tensors (~3 min,
  174 GiB). Derived λ=1 and λ=2 checkpoints verified complete (150 entries,
  131 shards, 296,474 tensors); source hashes unchanged after every build.
- Served with a **no-plugin** profile (`launch.WEIGHT-ABLATED.sh`) at 262144 ctx,
  fp8 KV, zero hook records, real tool call through the capture proxy.
- In-sample (14+5): λ=1 → 4/14 refusals, 4/14 completed-compliant, 5/5 harmless;
  λ=2 → 5/14 refusals, 4/14 completed-compliant, 5/5 harmless. **λ=1 selected** —
  λ=2 nets +1 refusal (removes 44, creates 57+72), so λ is not the lever; the
  weak direction (cohen_d≈0.425) is the bottleneck. λ=3 not attempted (Bent
  Pyramid: two λ failures at the same boundary → redesign, do not patch).
- **Held-out** (AdvBench goals 100–139 + Alpaca rows 100–139, zero overlap with
  the derivation set, 2048 tokens, transparent `eval_probe.py`): source → 15/40
  harmful refusals, 8/40 completed-compliant, 17/40 truncated; λ=1 → **7/40**
  harmful refusals, **10/40** completed-compliant, 24/40 truncated. Harmless:
  0/40 refusals both arms; 37→35/40 completed-compliant. Completed-harmful
  refusal rate 65.2%→37.5%, so truncation is not the whole explanation — but λ=1
  reasons ~75% longer on harmful prompts (median 797→1394 tokens).
- Verdicts (anti-idiocy): marker-prevalence reduction **VERIFIED** for the scoped
  judge/set/budget; "refusal genuinely removed" **SUPPORTED BUT NOT VERIFIED**
  (no independent LLM judge); capability preservation **SUPPORTED BUT NOT
  VERIFIED** (small 37→35 harmless regression at n=40); serving predicate
  **VERIFIED** independently. **The gate is NOT passed** — 6/16 completed
  held-out harmful responses still refuse.
- Full record with exact commands, hashes, claim-proof records, and evidence
  file list: `/scratch/sglang-work/reports/refusal-ablation/weight-ablation-results-20261121.md`
  (copied to branch `aws` as `reports/weight-ablation-results-20261121.md`).
