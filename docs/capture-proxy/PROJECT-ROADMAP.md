# PROJECT-ROADMAP — capture-proxy

## Intent Statement

Capture proxy for llama.cpp/ollama/node-llama-cpp/vLLM — logs raw JSONL of all requests/responses before forwarding. Exposes an **OpenAI-compatible API** so any agent using OpenAI client libraries can route through it transparently.

## Current State (as of measurement)

- **Version**: 0.2.0 (pyproject.toml)
- **Codebase**: ~12,111 lines across 10 source modules + 18 test files
- **Tests**: 361 collected, 361 passing (was 246 at last maintenance; +115 from alias + headroom tests)
- **Subsystem count**: 10 source modules (see Functional Surface Ledger below)
- **Uncommitted work**: Model alias feature partially done (app.py, __main__.py, models.py modified; alias.py + 6 test files untracked)
- **Tests passing**: ✅ Yes (361/361)
- **Tests existing**: ✅ Yes (18 test files)
- **Tests meaningful**: Need adversarial audit (not yet done)

## Scope

The proxy serves as an observability and routing layer between agent workloads and LLM backends. It must:
1. Never lose or mutate request/response data (lossless forwarding)
2. Log every interaction for audit and replay
3. Present a unified OpenAI API across heterogeneous backends
4. Provide diagnostic telemetry (sessions, users, headroom, runtime metadata)

## Planned Phases

### Phase 1: Foundation Measurement & Stabilization (CURRENT)
- [ ] Functional surface census complete
- [ ] CodeGraph indexed
- [ ] Graph-assisted analysis (structural/semantic/historical)
- [ ] CAM audit findings triaged
- [ ] Adversarial test-surface audit
- [ ] Uncommitted alias feature assessed for commit/merge
- [ ] Stale documentation updated (README test count)
- [ ] PROJECT-ROADMAP.md created

### Phase 2: Code Quality Gates
- [ ] Run full CAM audit via subagents
- [ ] Run CBP boundary sweep on all module interfaces
- [ ] AD root-cause analysis on any failures found in Phase 1
- [ ] Apply FIX-NOW items from prior maintenance session (17 items)
- [x] Verify DEFER items are still valid (verified 2026-10-12: all four reproduced as live defects, all four fixed below with tests + full chain green)

### Phase 3: Deep Analysis
- [ ] Structural combo analysis (imports → modules → methods → dependencies)
- [ ] Semantic combo analysis (concepts → schemas → contracts → state)
- [ ] Historical combo analysis (change concentration → risk hotspots)
- [ ] Adversarial test coverage mapping against functional surface
- [ ] Headroom feature completion (torch/safetensors dependency, dangling refs)

### Phase 4: Refactoring & Hardening
- [ ] Address all CAM BLOCKER/REQUIRED findings
- [ ] Resolve S-DEAD dead exports and S-WIRE missing wiring
- [ ] Harden G-EDGE boundary conditions (upstream 4xx/5xx, disconnect, empty input)
- [ ] Resolve G-SEM semantic mismatches between docstrings and behavior
- [ ] Fix any G-DRIFT documentation/code drift

### Phase 5: Verification & Delivery
- [x] All gates green: ruff, mypy strict, pytest 472+
- [x] Functional surface verified against implementation
- [x] Test surface verified against functional surface
- [x] No drift from original intent
- [x] Documentation updated and accurate

### Classifier & Image Extension (COMPLETED)
- [x] `/v1/classification` endpoint for BERT/DeBERTa classifier models
- [x] `/v1/images/generations` endpoint for image generation models
- [x] `CAPTURE_MODEL_TASKS` config: model-id → task-type mapping (CLI/env/TOML)
- [x] `_resolve_task()` per-request task resolution from env var
- [x] `ClassificationRequest`, `ClassificationResponse`, `ImageRequest`, `ImageResponse` Pydantic schemas
- [x] `CaptureRecord.task` field extended for task labels
- [x] HuggingFace model ID support (full `https://` URLs accepted as model identifiers)
- [x] 104 new tests across `test_classification.py` and `test_image.py`
- [x] Out of Scope updated: classifier/image endpoints now in scope

### Headroom Model Integration
- [x] Headroom sidecar covers the four eligible request types: chat, completion, classification, image (embeddings/rerank excluded)
- [x] `task` field propagated to headroom sidecar (HeadroomRef v6, CapturedHeadroom v6)
- [x] `_spawn_headroom` eligibility extended to include classification and image
- [ ] Loss saturation capture via model_headroom `--loss-curve` (proxy emits inference data only; model_headroom scores R-01)
- [ ] Growth factor capture via model_headroom `--isoflop` (proxy emits latency data only; model_headroom scores R-02)
- [ ] Dead neurons capture via model_headroom `--mlp-activations-file` (proxy does not capture activations; model_headroom scores R-05)
- [ ] Load balance capture for MoE variants via model_headroom `--activations` (requires vLLM `routed_experts` exposure)
- [ ] Specialization capture via model_headroom `--expert-weights-file` (proxy does not capture expert weights; model_headroom scores R-07)

## Observed Shortcomings — measured 2026-11-07 (end-to-end, ornith-1.5-9b)

Measured on a dense model (`ornith-1.5-9b`) served by vLLM and captured through this proxy with
`CAPTURE_HEADROOM=all`, `CAPTURE_HEADROOM_WEIGHTS=gate`, `CAPTURE_HEADROOM_MAX_BYTES=64M`.
Seam inventory: `docs/maintenance/2026-11-07-seams/SEAMS.md`. Items already ranked there are
cross-referenced, not restated. **Line references are to `HEAD` (`da3e903`)** — the working tree
carries uncommitted changes that shift them.

### C-1 — `HeadroomRef` advertises artifacts that were never written (REQUIRED — severity corrected)

Measured ref for one capture:

| Ref key | Path | Exists |
|---|---|---|
| `effective_rank` | `{capture_id}-weight_pairs.json` | yes |
| `load_balance` | `{capture_id}.activations.json` | **no** |
| `specialization` | `{capture_id}-expert_weights.json` | **no** |

The ref is built from `planned_artifacts()` (`headroom.py:973-1016`), which returns a path for every
**enabled** signal. `app.py:2440-2456` copies those planned paths into `HeadroomRef.signals` before
the background write is scheduled. The sidecar is built differently — `_write_sync`
(`headroom.py:1366`) starts from `dict.fromkeys(SIGNAL_KEYS)` and fills a key only when the artifact
is actually written — so the sidecar says `null` while the ref says a path.

SEAMS.md S-06 rates this SUGGESTION on the assumption that the only failure is a write race
("sidecar is authoritative, ref may point at a missing file"). The measurement shows a **permanent**
dangling pointer, not a race: `load_balance` and `specialization` have no producer at all, so those
files never appear. Severity should be REQUIRED, and the ref should publish only artifacts that
exist, or mark the rest as planned-but-absent.

### C-2 — Ref `architecture` is structurally always `"unknown"` (REQUIRED)

`HeadroomRef.architecture` defaults to `"unknown"` (`models.py:126`) and the call site
(`app.py:2449-2456`) never passes it. The sidecar carries the detected value. Measured pair for one
capture: ref `architecture: "unknown"`, sidecar `architecture: "dense"`. Any consumer reading the
ref — the append-only record, the one that survives — gets the wrong answer on every capture. `task`
is `null` on both sides, so architecture is the only field where they disagree.

### C-3 — `CAPTURE_HEADROOM_WEIGHTS=gate` captures an MLP gate under an attention plan (REQUIRED)

`names.py` "attention" templates are BERT/DeBERTa/RoBERTa-only
(`model.bert.encoder.layer…`, `bert.encoder.layer…`, `deberta.encoder.layer…`,
`roberta.encoder.layer…`). A decoder-only dense model exposes `model.layers.{i}.self_attn.q_proj`,
which matches none of them. `_capture_projs` (`headroom.py:89-119`) resolves dense/`gate` as
`("attention", "gate")` with `fallback_start = 1` — attention is primary, the MoE gate is fallback —
and its own docstring notes that primary misses are **silent** (`headroom.py:105`). So the miss is
not reported, and resolution lands on `model.layers.{layer}.mlp.gate_proj.weight`, which is in the
**gate** family (`names.py:75-76`).

Measured artifact: `layer_name: model.language_model.layers.0.mlp.gate_proj.weight`,
`parent_shape [12288, 4096]`. The plan said attention, the emitted tensor is an MLP gate. The
`layer_name` is truthful, so nothing lies — but the *selection* is silently a different module family
than intended. This is SEAMS.md R-4 / S-02, now with a measured instance.

### C-4 — Byte cap reduces the rank input to a single tile (REQUIRED)

`_tensor_to_bundle` (`headroom.py:646-658`) slices rows to fit `CAPTURE_HEADROOM_MAX_BYTES` before
conversion and sets `truncated`. At a 64 MB cap a `[12288, 4096]` fp32 tensor fits 4096 of 12288 rows,
so the authority receives a `4096 × 4096` top-left tile with `sampling_mode=single_tile`. The
artifact declares this honestly; the defect is that the pipeline accepts a one-tile, one-tensor input
as a model-level rank. Authority-side consequence is A-3 in the authority roadmap.

### C-5 — Daemon lifecycle does not survive a restart cleanly (OBSERVED, operational)

`--stop` read a PID file that did not match the running daemon, so a restart left two proxies bound
to the same port and the second exited silently; the daemon's cmdline is
`python3 -m capture_proxy` (underscore), which `pgrep -f capture-proxy` does not match. Restart
correctness currently depends on the operator enumerating `ss -lptn` and `/proc/*/cmdline`. Recorded
because it produced a false "headroom off" state for one restart cycle — an armed `headroom=all`
proxy stayed alive while the operator believed it had been replaced.

### EMO-001 — Headroom capture for `allenai/Emo_1b14b_1T` (FIXED 2026-09-27, `80b447f`)

**Ticket:** EMO-001 (ticket doc removed in pre-release sanitization)

- [x] **F1 — MoE detection gate.** `model_type: "emo"` was absent from `_MOE_MODEL_TYPES`, and
      `if model_type: return "dense"` fired *before* the `architectures` substring check, making
      that fallback dead code for every model with a named `model_type`. Shipped as a guarded
      structural `num_experts`/`num_local_experts >= 2` gate **plus** moving the `architectures`
      check above the early return. The name list was **kept, demoted to lag-cover only** — it now
      fires solely when no expert count is parseable at all, which is exactly where
      `model_headroom`'s `get("num_experts", 0)` default is uninformed rather than authoritative.
      See *Sidecar scope* below for why that is the correct shape.
- [x] **F2 — expert projection candidates.** Added `mlp.experts.{E}.{gate,up,down}_proj` ahead of the
      dense `mlp.*_proj` templates, plus legacy `block_sparse_moe.experts.{E}.w{1,3,2}`. Expert
      templates precede dense ones, so a checkpoint exposing both resolves the expert.
      **This also fixed Qwen3-MoE, OLMoE, Mixtral and PhiMoE**, which were broken in EMO's exact
      shape (`None` for all six projections) — the ticket had framed this as a one-model bug.
- [x] **Regression tests** — `tests/test_headroom_emo_support.py`, incl. the F1 ordering case, the
      expert-over-dense precedence, malformed-config no-raise cases, and a **differential test
      against the real `model_headroom.models.loader`** (not a paraphrase). The original
      acceptance criterion 5 was unsatisfiable as written — it demanded "the same tensor names as
      before" for families that were `None` before — and was rewritten to the satisfiable form:
      a projection that was a real name must still be that same name; one that was `None` is a
      labelled FIX. Measured result: **0 regressions, 6 fixes.**
- [x] **Repo-wide loader bug (found during verification, unrelated to EMO).**
      `_default_tensor_loader` probed the fp8 scale tensor with `scale_name in shard:`, but
      `safetensors.safe_open` has no `__contains__` or `__iter__` — so that raised `TypeError`
      *inside* `except Exception: return None`, and the loader returned `None` for **every tensor
      of every model**. Headroom weight capture was dead repo-wide, not just for EMO; EMO's
      resolution miss fired first and masked it completely. This is the **third** distinct
      silent-failure mode in this one code path (the others: the D2 resolution miss, and the
      torch/safetensors-unavailable failure recorded in the session notes). Now a `try/except` around
      `get_tensor`. **Systemic cause: every weight-capture test stubs the `TensorLoader`, so the
      real-I/O contract is untested** — see below.
- [x] **DEFER-6 — FIXED 2026-10-12: real-I/O test added (`tests/test_headroom_real_io.py`, 14 tests).** Tiny real
      `.safetensors` fixture in `tmp_path` (indexed 2-D + fused 3-D + 1-D tensors) run through the real
      `_default_tensor_loader`: value round-trip, zero gradients, parent_shape, byte-budget truncation,
      index + header-fallback weight maps, shard keys. Skips cleanly without the `headroom` group
      (`uv sync --group headroom` to execute); the official bare chain stays green either way.
- [x] **DEFER-4 — FIXED 2026-10-12: every pair carries `expert_id`.** Parsed from indexed templates
      (`experts.N`), per-slice for fused 3-D fan-out, `None` for dense/non-expert tensors — never invented.
      Additive key only; the authority's loader requires just gradient/weight/layer_name and reads the
      rest via `.get()`, so the 2-D pair contract is untouched. Pinned by pair key-set assertions
      (stub + real-I/O paths).
- [x] **DEFER-2 — FIXED 2026-10-12: fused 3-D layouts fan out to per-expert 2-D pairs.**
      `mlp.experts.gate_up_proj` / `mlp.experts.down_proj` templates added AFTER the indexed ones
      (first-match behavior unchanged on per-expert indexes); a 3-D [E,*,*] match emits one 2-D
      bundle per expert slice sharing the byte budget cumulatively, each with its expert index.
      The fused tensor is NOT split into gate/up halves (would invent layout semantics). Unsupported
      ranks yield empty, never raise (also closes a latent 1-D TypeError out of the singular loader).
- [x] **DEFER-3 — FIXED 2026-10-12: DeBERTa-v2 / RoBERTa resolve.** Dense templates gained
      `deberta.encoder.layer.{i}.attention.self.query_proj` (+ intermediate/output `.dense`) and the
      `roberta.` equivalents — read off the libraries' own module structure
      (DisentangledSelfAttention.query_proj, RobertaSelfAttention.query), not recalled. BERT entries
      keep precedence (locked by test). Resolution pinned by synthetic-map tests, not just the
      `dense` verdict.
- [x] **R-06 producer — SOLVED, prototype verified (not yet landed).** `model_headroom`
      `cli.py:48` and `signals/load_balance.py:43` name `moe_expert_profiler.py`, which does not
      exist anywhere on this box. A working substitute was built and run against real EMO forward
      passes: hook `model.layers.{i}.mlp.gate` for router logits, or simply count each
      `EmoSparseMoeBlock.experts[j]` input batch size — that IS the observed per-expert token
      count, no recomputation needed. `EmoSparseMoeBlock.forward` already returns `router_logits`
      as its second value (`modeling_emo.py:408`), so the data is exposed by design. Measured over
      **1024 real tokens**: routing volume is exactly `8192 = 1024 × top_k 8` on all 16 layers.
      **Still open:** the script lives in ephemeral `/tmp/opencode/`, not in either repo, and needs
      a durable home plus the `layers: dict[str, dict[str, int]]` adapter.
- [ ] **R-06 first real signal — expert usage decays sharply with depth (measured).** Over 1024
      tokens of real text: layer 0 uses **114/128** experts, layer 11 uses 64, layer 15 uses only
      **46/128** — i.e. **82 experts receive zero tokens** at the deepest layer. The shared expert
      (index 127) takes every token by construction (`num_shared_experts=1`, legacy split path at
      `modeling_emo.py:327-357`), so it must be reported separately from the 127 routed experts.
      **Caveat: do not quote a single load-balance scalar from this yet.** A CV computed across all
      128 experts is contaminated by expert 127 holding 1024 tokens, and one 1024-token corpus is
      not a sample. It is a signal worth chasing, not a number to report.
- [x] **Serving — safetensors path SOLVED; the GGUF converter is NOT required for telemetry.**
      The earlier "separate ticket needed — serving/conversion" framing was wrong: a GGUF converter
      is only needed for llama.cpp serving, not for producing inference-time headroom signals. EMO
      runs directly from its safetensors via `trust_remote_code`. Verified: **13,568,641,024 params**
      in bf16 on 2× T4
      (`device_map="auto"`, layers 14–15 + `lm_head` CPU-offloaded by accelerate), vocab 100352,
      and `"The capital of France is"` → **`" Paris"`** (id 12366), generation `" Paris. Paris is"`.
      fp32 (54.3 GB) remains impossible; bf16 (27.1 GB) against 30.7 GB VRAM leaves little headroom.
- [ ] **`transformers` version ceiling for EMO (measured, hard constraint).** EMO's remote code uses
      the pre-5.0 rope API: `modeling_emo.py:548` does `ROPE_INIT_FUNCTIONS[self.rope_type]` with
      `rope_type="default"`, and **`transformers` 5.0.0 raises `KeyError: 'default'`** — the key was
      removed in 5.x. `transformers==4.57.6` imports and runs it. All 16 of EMO's `transformers`
      imports resolve on 4.57.6. Pin **<5** for anything that loads EMO.
- [ ] **Shared-env hygiene rule (learned the hard way).** The 4.57.6 fix lives in a throwaway venv
      built with `--system-site-packages`, **not** in `/usr/local/lib/python3.12/dist-packages`,
      which capture-proxy's own headroom tests depend on. System `transformers` 5.0.0 there is
      still broken by a `huggingface-hub==2.0.0` versus `<2.0` pin conflict — that breakage is
      **pre-existing**, not a regression from this work. Anything fixing it must not touch the
      shared torch/safetensors/numpy.
- [ ] **R-06 cross-check: 13/16 layers match exactly, 3 differ by 1–2 low-probability experts.**
      Recomputing top-k from observed gate logits and comparing against observed expert traffic
      agrees exactly on 13 layers. Layers 4, 11, and 15 each differ by one or two experts, all with
      mean selection probability ~0.002 — i.e. right at the rank-7/rank-8 boundary. **Cause not
      isolated**: the prime suspect is bf16 rounding in the gate output changing the order of
      near-ties, but that is a hypothesis, not a diagnosis. Routing *volume* matches exactly
      (`8192` on all 16 layers), so the mechanism is right; only the identity of a few
      lowest-probability experts is in question. Resolve before treating any per-expert identity as
      exact — the counts are trustworthy, a specific expert's membership may not be.
- [ ] **Instrumentation hazard, recorded because it produced a false result.** Leaving hooks
      registered across `generate()` unions the routing of every call, so it cannot be compared
      against a single-call recomputed top-k. The first prototype did exactly this and reported
      16/16 layers "MISMATCH". The real state was 15/16 agreeing. Routed volume is a far stronger
      check than set equality, because it survives this class of error. Measure on one forward pass
      with hooks removed before generating.
- [ ] **R-05 dead-neuron metric is undefined and the naive version is wrong.** A never-fed expert
      emits an empty `[0, hidden]` output, so *all* of its neurons read as "never activated" **by
      construction**. The first prototype reported 75.05% dead neurons across the model — that
      number was measuring prompt length, not EMO. Restricted to experts that actually received
      tokens, the same run gave **0.0%**. Any R-05 implementation must split fed from starved
      experts or it will report a corpus-length artifact as a model property.
- [ ] **Still blocked, unchanged: llama.cpp serving.** `libllama.so` has no EMO architecture
      string, and the official repo ships **0 `.gguf` files** (24 files, safetensors only). So
      llama.cpp/llama-server cannot serve EMO at any quant until either a GGUF converter class for
      `EmoForCausalLM` exists or llama.cpp gains the architecture. Watch `/dev/shm`: 14 GB, too
      small for any EMO precision above int4, so the earlier staging pattern does not transfer.
- [ ] **GGUF asymmetry (verified, not assumed).** `model_headroom` reads GGUF via
      `models/gguf_loader.py`, but its `gguf` extra (`pyproject.toml:25-26`) is **not installed** here —
      `import gguf` fails; `pip install -e "/kaggle/working/model_headroom[gguf]"` fixes it.
      `capture-proxy` is **safetensors-only**, so the sidecar cannot extract weights from a GGUF path.
      **GGUF is an explicit NON-GOAL for the headroom sidecar:** the carrier reads the `.gguf`
      suffix before the directory gate and logs one deduplicated `warning` naming the path,
      then returns with `record.headroom` still None — no sidecar, no silent inertness,
      no GGUF reader in the carrier. Not a blocker — the weight-side headroom runs in Mode A
      from the HF directory, and the inference-time signals come from telemetry, not weight reads.
- [ ] **`model_headroom` lags on name resolution, not detection.** Its `_get_expert_weight_name`
      (`models/loader.py:30-44`) is the same brittle `model_type` name-list shape, and for
      `model_type: "emo"` it returns a Mixtral-style name that does not exist in EMO. Its
      classification is generic (`int(num_experts) >= 2`) so it does not lag there, but it reads only
      `num_experts`/`num_local_experts` while the real EMO config also carries a (null)
      `num_experts_per_layer`. Its only caller is reached from tests, not the CLI path, so no action
      needed unless a direct model-dir extraction path is added.

### Sidecar scope — the two roles, and what falls outside them

The sidecar exists to do exactly two things: **cover cases where `model_headroom` might lag**, and
**offer healthy redundancy** so agreement between two implementations is a real cross-check.
Anything outside those two roles is out of scope.

Applied to detection, that resolves what would otherwise look like a contradiction between two
locally-reasonable implementations: the structural rule follows the authority; the `model_type`
name list survives only as lag-cover for configs carrying no parseable count; and it can never
override a measured count. The one place the sidecar deliberately **exceeds** the authority is
that `model_headroom` raises on a truthy non-numeric count and the capture path must not.

Candidates for a future scope review under "anything out of that scope goes": the 5 sidecar signal
keys with no producer (reserved slots, not abandoned ones), and the safetensors-only limitation.

### Which of the six sidecar signals inference traffic can actually supply

`headroom.py:804` allocates all six keys; `headroom.py:813` fills one. The other five are
reserved slots (`docs/headroom-capture-design.md:128-129`), not abandoned ones. Sorting
them by where the input physically comes from — the split that decides what this proxy is
for:

| Signal | Key | Needs | Source |
|--------|-----|-------|--------|
| R-03 effective rank | `effective_rank` | weight tensors | checkpoint — **implemented** |
| R-06 load balance | `load_balance` | per-expert integer counts | **producer prototyped + verified**; adapter not landed |
| R-05 dead neurons | `dead_neurons` | per-channel activations | hookable; **metric must exclude never-fed experts** (see below) |
| R-07 specialization | `specialization` | expert vectors | weak offline / needs inference |
| R-01 loss saturation | `loss` | `tokens_trained` + `validation_loss` | training run only — not this proxy |
| R-02 growth factor | `isoflop` | `compute_flops` + `model_params` + `measured_loss` | training run only — not this proxy |

R-01/R-02 are hard-blocked on training telemetry — no volume of inference capture
substitutes for a tokens-trained counter and a validation loss. R-06 asks for a routing
histogram, which is the most ordinary thing an instrumented serving path produces and
something no offline checkpoint read can recover. **R-06 is the strategic case for this
proxy, and it is unbuilt.**

Symptom to check first on any headroom investigation: sidecar `error_body` reads
`"no gate_proj weight found for layer 0"`. See the session notes for the
prior silent failure with a different `error_body`.

## Measured: R-06 fed-vs-starved on a real MoE (2026-09-29)

> **Correction (2026-09-29, later the same day).** The measurement below was
> taken on the checkpoint at `/root/Emo_1b14b_1T_EmoAnnealed`, whose README
> identifies it as **`StdMoE_1b14b_1T_EmoAnnealed`** — the paper's Appendix B.4
> *ablation*, not the main `Emo_1b14b_1T` release. The directory name and
> `config.json` both say "emo", and every other discriminator (config, index,
> `configuration_emo.py`, tensor names, shard sizes, parameter count) is
> identical between the two, so the README H1 is the only thing that
> distinguishes them. It was also probed with **one synthetic document**, which
> is the worst possible instrument for an objective whose unit of specialisation
> is the document. Measured across 5 domain-separated documents the same
> checkpoint ranges 0.38–0.78 coverage, not 0.74–0.80. The fed-vs-starved
> *mechanism* below still holds; the *magnitude* does not generalise. Full
> two-checkpoint measurement: `EMO-CHECKPOINT-REPORT.md` at the workspace root.

MODEL_SLOT-1 is unblocked. `StdMoE_1b14b_1T_EmoAnnealed` (16 layers, 128 experts, top-8)
loaded across 2× Tesla T4 in fp16 — 13.02 GiB on cuda:0, 12.25 GiB on cuda:1,
2048 experts tracked, **0 straddling a device**. Probe: `tools/emo_runner/` at the
workspace root. Reproduces in 255 s.

The finding that matters is not a balance number. It is that **expert liveness is
corpus-length dependent**, so a balance verdict from a short corpus is an artifact
of corpus length rather than a property of the router:

| Corpus | never-fired (of 2048) | coverage |
|---|---|---|
| 256 tok | 541 (26.4%) | 0.7358 |
| 512 tok | 442 (21.6%) | 0.7842 |
| 1024 tok | 404 (19.7%) | 0.8027 |

**99 experts that read as dead at 256 tokens are alive at 512.** Monotone violations
across 256→512→1024: 0. 404 never fired at any length, across 16/16 layers.

The authority's R-06 is **entropy-based, not min/max**, and it is stable:
`balance_ratio` = 0.7457 / 0.7460 / 0.7438 — a 0.09% spread across a 4× corpus
while coverage moves 9.1%. That is a well-behaved signal, and it is why a
min/max reading of the same data is wrong.

**Open gap, measured not asserted.** An equal-mass construction shows a router
genuinely collapsed onto 8 experts scores 0.8619 — *lower* than the real
unvisited case at 0.8734. So `balance_ratio` ranks concentration correctly but
**cannot separate "unvisited because the corpus was short" from "collapsed because
the router is broken."** Coverage is the only discriminating quantity, and
`LoadBalanceReport` has no coverage field. Adding one is a defaulted field on a
frozen dataclass (permitted), but it changes a report contract, so it needs an
operator ruling before the authority is touched.

Two operational notes for whoever runs the checkpoint:

- **Placement must be layer-granular.** Experts are 97.9% of every EMO layer, so
  `device_map="auto"` cuts mid-layer and top-8-of-128 then makes nearly every
  token select a straddling expert across a PHB link (no NVLink on this box).
- **Two device failures, both from per-forward derived tensors, not weights.**
  RoPE `cos`/`sin` are computed once at model level and consumed by all layers;
  and the decoder layer's own `residual` is not covered by accelerate's
  per-submodule `AlignDevicesHook`. Both are handled by a per-layer
  `forward_pre_hook`. This is the concrete form of "MoE moves layers and weights
  differently than dense models."

Random token IDs are not a substitute for a corpus: they routed only 64.6/128
experts per layer versus 94.2/128 for real text. Out-of-distribution input
collapses the router, so single-sample telemetry must never be quoted as a
model-level claim.

## Gate mechanics — three traps that re-break a green chain silently

Recorded 2026-10-15 after a G2/G6 repair pass in which each of these had to be
diagnosed rather than read off a report. All three are recurrences, not one-offs.

- **`uv lock` drift recurs on every sibling `model_headroom` edit.** `uv.lock`
  carries a snapshot of the editable sibling `model-headroom`
  (`source = { editable = "../model_headroom" }`), so its `requires-dist` is
  frozen into *this* repo's lockfile. Measured instance: sibling commit `76de73e`
  added `safetensors>=0.4` to its MAIN deps, which left capture-proxy's lock
  stale even though nothing here changed. **After any `model_headroom`
  `pyproject.toml` edit, re-run `uv lock` in *this* directory and confirm
  `uv lock --check` exits 0** — G6 cannot detect the drift on its own, and the
  failure surfaces later as an unrelated-looking install error. Fix is `uv lock`,
  never a manifest edit: dropping the dep from a local manifest "fixes" G6 by
  deleting the requirement rather than tracking it.

- **`ruff format .` rewrites python fences inside `.md`, with no flag.** Measured
  on ruff 0.16.x: a ````python` block in a markdown file is reformatted exactly
  like a `.py` file, so `ruff format --check .` (G2) covers `docs/` too. The
  consequence is specific and permanent: **hand-aligned inline comment columns in
  ticket docs cannot survive G2.** The EMO-001 ticket doc lost
  that padding to a formatter pass. The comment *content* is intact — only column
  alignment is gone — so this is accepted cost, not damage. Leave it alone: do NOT
  hand-realign the columns (the next `ruff format .` re-strips them, producing an
  endless G2 loop) and do NOT `extend-exclude` the doc (that hides the doc from G2
  entirely, which is gate-weakening, not gate-fixing). The doc now sits at the
  formatter's fixed point — `ruff format --check` on it alone exits 0 — so it will
  not re-break on its own.

- **A bare `uv run` re-locks, and it mutates `uv.lock` mid-command.** `uv run`
  performs an *implicit* lock update, so a command that reads as read-only —
  `uv run ruff format`, `uv run python -c ...` — can rewrite `uv.lock` as a side
  effect. Measured contract: `uv run --locked` is defined as "Assert that the
  `uv.lock` will remain unchanged". **Use `uv run --locked` for every uv
  invocation in this repo**, including read-only probes. For a pure
  non-mutating lock check use `uv lock --check` ("Check if the lockfile is
  up-to-date").
  **Do NOT reflex `git checkout -- uv.lock` when it looks dirty.** That is a
  symptom reflex that re-breaks G6 by restoring the stale sibling snapshot from
  bullet 1 — it turns a lockfile that is correct-but-uncommitted into one that is
  wrong. If `uv.lock` genuinely needs to change, the fix is `uv lock`, and the
  change belongs in the commit.
  *Reproducibility note:* this cannot be re-demonstrated right now. The 2026-10-15
  repair already ran `uv lock` and consumed the mutation window, so a bare
  `uv run` today has nothing left to change. That is the fix working, not the
  mechanism being fictional — do not conclude otherwise from a failed repro.

## Out of Scope

- Backend model training/fine-tuning (handled by vLLM/llama.cpp)
- Changing the core lossless-forwarding invariants (INV-2..INV-5)
- Production deployment infrastructure

## Scope-Creed Defenses

1. Every new feature must map to a functional surface entry
2. Every change must pass ruff + mypy strict + pytest before merge
3. No modification of invariants without an AD dossier and invariant migration plan
4. Documentation updates are required with every feature, not deferred
5. If a change spans >3 modules, escalate to DESIGNER agent before proceeding

---

*Living document — update at each phase boundary.*
