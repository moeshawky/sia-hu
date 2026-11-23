# Acquisition architecture — how the authority obtains evidence

**Purpose:** Design layer for evidence acquisition in `model_headroom`: family
adapters (checkpoint layouts), backend adapters (where evidence physically
comes from), capability negotiation (what the tool declares it can and cannot
obtain), published-evidence discovery, activation extraction and replay,
streaming statistics, exact-vs-sampled semantics, the compute budget, and
sidecar caching. Cross-references `model-headroom/PROJECT-ROADMAP.md` (defects,
milestones), `model-headroom/evidence-provenance.md`, and
`model-headroom/measurement-ontology.md`.

**Boundary (never inverted):** the authority never proxies; the carrier never
scores. The carrier sidecar is *one* acquisition backend, not the only one.

---

## 1. The seam at risk

The signal-name vocabulary is **duplicated** across the boundary with no
compile-time link:

- Carrier: `SIGNAL_KEYS` (6 names) in `src/capture_proxy/names.py:118`,
  stdlib-only so it imports in `CAPTURE_HEADROOM=off` mode.
- Authority: the 7 scored signal names as **scattered string literals** across
  `analysis/scoring.py` (weights table, `_score_signal`), `analysis/orchestrator.py`,
  and `reporting/human_report.py`.

A new evidence source is most likely to drift at this seam. Boundary status
**measured clean 2026-11-15** by executed probe: no import edge in either
direction between `src/model_headroom` and `src/capture_proxy`; the carrier
computes no score, ratio, rank, entropy, cosine, or normalized quantity — only
raw weight matrices, raw routing counts, byte accounting, and structural facts
(architecture, expert count). Treat "clean" as a probe result with a date, not a
permanent property: re-run the probe after any change that touches either side's
imports or the vocabulary.

## 2. Family adapters (checkpoint layouts)

**Rules.**

1. **Compose feature-specific adapters; do not build one gigantic class.** Each
   layout feature (gated FFN, plain MLP, MoE experts, linear attention, vision
   tower, quantized blocks) is a small adapter; a model is a composition of them.
2. **Names do not imply identical mathematics.** `gate_proj`/`up_proj`/`down_proj`
   are naming conventions, not a mathematical contract.

   **Live measured instance (2026-11-15, RockAndEgg-313M-v2, Qwen3 dense):** the
   FFN *container's* forward output is the **down-projection result**, shape
   `[B, S, 1024]` (hidden_size). The gate/up branches are `[B, S, 3072]`
   (intermediate_size). Hooking the container measures the residual stream, not
   the FFN interior. An adapter that assumes "hook the container, read the FFN"
   is reading the wrong tensor without any error.

**Adapter inventory (status per master roadmap's integration matrix):**

| Adapter | Handles | Status |
|---------|---------|--------|
| Dense gated FFN (SwiGLU/silu) | gate/up/down, SiLU gating | IMPLEMENTED (module families in `models/loader.py`; safetensors sampling) — but CLI name resolution is template-bound (defect D-4) |
| Standard dense block | BERT-style `dense.weight`, plain MLPs | IMPLEMENTED (Phase 5) |
| Sparse MoE | experts + router; `num_experts ≥ 2` | Detection IMPLEMENTED; expert extraction IMPLEMENTED (GGUF single-tensor by literal name; HF templates); **never verified end-to-end on a real MoE** (support matrix) |
| Hybrid attention/recurrent | `layer_types` dispatch; `linear_attn` vs `self_attn` tensor names | Detection IMPLEMENTED, VERIFIED on ornith-1.5-9b; **D-5 defect on the Qwen3 case**; linear-attention names absent from the CLI sampling templates |
| Encoder / classifier | BERT/DeBERTa/ModernBERT/laya/RockAndEgg naming | IMPLEMENTED (Phase 5); DeBERTa P6 UNKNOWN (never verified on a real model) |
| Multimodal (vision + mtp) | `model.visual.*`, `mtp.layers.*` | `"visual"` substring added (GAP-3 resolved); exercised on ornith-1.5-9b by the run harness (excluded from sampling); authority sampler covers MLP projections only |
| Quantized / custom projections | GGUF quants, ternary blocks | `models/ternary_quants.py` + `models/ternary_sampling.py` IMPLEMENTED; `dequantize_ternary` Prism fallback partially documented (a missing `gguf` extra raises `ModuleNotFoundError`, not the caught `AttributeError`/`NotImplementedError` — known issue, no-`gguf` paths skip for 0.4.0); the operator's ternary GGUF work (Ternary-Bonsai-8B-Q2_0) is unreadable by stock llama.cpp build 11395 — the authority path handles it separately, do not conflate |

## 3. Backend adapters (where evidence physically comes from)

**Verify-before-relying rule:** before an integration is relied on, verify the
*actual* API, version, supported families, and output semantics of the
dependency — measure, don't assume. Each row's status reflects what was
**executed on this machine** at the date given.

| Backend | Role for the authority | Status | Hard constraints / notes |
|---------|------------------------|--------|--------------------------|
| safetensors (direct, in-process) | Weight matrices for R-03/R-04 structure; layout facts | IMPLEMENTED (`models/loader.py`, `models/safetensors_sampling.py`; `safe_open`, one tensor at a time, shard never materialised) | **D-4:** name resolution is the hardcoded template table `_SAFETENSORS_TENSOR_TEMPLATES` (`cli.py:324-341`, four prefixes: `model.language_model.…`, `model.layers.…`, `model.layers.…mlp.experts.0.…`, `block_sparse_moe…w1/w2/w3`); a real checkpoint storing `layers.0.mlp.gate_proj.weight` (no `model.` prefix) matches nothing → "no MLP projection tensor matched a known layout" and R-03 cannot be acquired from weights already on disk |
| GGUF (direct) | Quantized weight extraction | IMPLEMENTED (`models/gguf_loader.py`, ternary path) | Never end-to-end on a real GGUF model (support matrix) |
| HF Transformers + instrumentation (in-process forward) | Activations for R-05; refusal-direction paths | IMPLEMENTED (`paired/instrument.py`; executed 2026-11-12: 32 `down_proj` `forward_pre_hook`s, all outputs verified `[1, 24, 12288]` before write) | **D-6:** `_sample_flat`/`_sample_rows` (`instrument.py:171-204`) `.reshape(-1)` their tensors, destroying channel identity — per-channel statistics are not computable from that representation. Replay semantics undeclared (§ 6) |
| vLLM (serving) | — (no measurement signals exposed) | PROPOSED for any future consumption; nothing executed | Serving engines expose no weight matrices, no `named_modules()`, no optimizer state, **no gradients**, no routed_experts, no hidden states. Environment fact: vLLM 0.32.0 present in a local venv (measured 2026-11-10 probe) — an environment fact, not an integration |
| SGLang (serving) | **The only** source of `routed_experts` and hidden-state residuals | PROPOSED; nothing executed from the authority | `routed_experts` requires `--enable-return-routed-experts`; hidden-state residual capture requires `--enable-return-hidden-states` (`refusal/activations.py:capture_residuals_sglang`). Carrier-side routing emission verified against SGLang 0.5.21 on 2026-10-09 (carrier's verification, not the authority's). sglang is not on this machine's PATH (measured) |
| llama.cpp (serving) | — | PROPOSED for any future consumption; nothing executed | Exposes none of the measurement signals; engine running (build 11395, measured 2026-11-10) |
| Published artifacts (Hub) | Training telemetry (R-01/R-02 inputs), run provenance | PROPOSED; the route is real and exists | `moeshawky/rae-005-research-evidence` carries `training_runs/control_off/run_record.json` with `history` (6 eval points), `losses` (25 train-loss values), `hyperparams`, `params`, `provenance` (git_head, script_sha256, released_repo, released_rev, versions, cuda attestation), `data` (manifest_sha256), `checkpoint` (best_validated step 4800). Limits per `evidence-provenance.md` § 4. Discovery mechanics below |
| Hu capture replay (carrier sidecar → authority) | Weight-pair evidence (R-03, and R-04 when gradients exist) | IMPLEMENTED for the weight-pair path (consumed by `load_weight_pairs_from_json`; the 2-D guard at `models/loader.py:433-443` rejects 3-D fused expert stacks rather than slicing) | The other 5 sidecar keys are reserved null — `loss`, `isoflop`, `dead_neurons` are **unobservable at the proxy layer** (training loss, compute-loss points, and serving-time activations are not things the proxy can see); that is reserved-and-unobservable, not a missing feature. A replay is not identical to a production execution unless evidence supports it (§ 6) |

## 4. Capability negotiation

**Design (PROPOSED).** At run start, before any signal is scheduled, the
authority assembles a capability manifest by probing three things:

1. **What the checkpoint provides** — layout resolution: read the safetensors
   header (or GGUF metadata), classify each tensor by the mathematical operation
   its name denotes, and report the prefix actually found. This is what
   `src/model_headroom/acquire/layout.py` does (IN FLIGHT, unverified — see
   master roadmap). The layout answer replaces the template table: instead of
   "does your name match one of four prefixes," the answer is "this checkpoint
   stores its MLP projections at `layers.{i}.mlp.{gate,up,down}_proj`."
2. **What the runtime provides** — device (`device.py`: accelerator preferred
   cuda→xpu→mps, CPU fallback, NFR-01), available memory against the
   checkpoint's size, installed libraries and versions (transformers, the
   `gguf` extra, `fla`/`causal_conv1d`/`mamba_ssm` for recurrent fallbacks).
3. **What is reachable** — local files only by default; published artifacts
   only on explicit operator opt-in with a pinned revision. No network
   assumption is ever implicit.

The manifest's output is per-signal: **obtainable** (with the route and its
limits) or **not obtainable** (with the reason — no input file, no runtime
capability, no publishing at this revision, structurally impossible as with
R-02 from one checkpoint). The report renders this manifest, so an
insufficient-evidence result is *explained*, not merely asserted (the reporting
half of defect D-1).

The rule: **the tool decides which evidence it can obtain, and declares what it
cannot.** The declaration is part of the report, not a log line.

## 5. Published-evidence discovery & acquisition

**Metadata-first, revision-pinned, selective download.** Operationally:

1. **Metadata-first** means: before downloading bytes, read the repo's file
   listing at a pinned revision (the Hub tree-listing route) and the
   small metadata files (record manifests, config, `PROVENANCE.md`). Decide
   from metadata whether the repo contains the needed artifact class (a
   run record with a `history` block? a `trainer_state.json`? weight shards?).
   A repo with metadata but no run record is answered "no training telemetry
   published" without downloading gigabytes.
2. **Revision-pinned** means: every read names a revision (see
   `evidence-provenance.md` § 5 — the measured `ac8bb272…` vs `7928cdf…`
   drift makes this load-bearing, not pedantic).
3. **Selective download** means: fetch only the files the manifest identifies
   as needed (the run record is small; the checkpoint shards are not), and
   verify file hashes against the listing.

Status: PROPOSED. The published evidence route is real (the `rae-005` repo is
measured to exist with the described record); the authority-side acquisition
code has not been executed.

## 6. Activation extraction & replay

- **Extraction.** In-process forward pass with `forward_pre_hook` on declared
  module paths (the ornith-1.5-9b run is the executed template: 32 hooks, shape
  verification of every captured tensor before write, single prompt, 24 tokens).
  Module paths come from the family adapter's layout resolution, never from
  hand-written prefixes (GAP-4: derive from `layer_types`, never by hand).
- **The D-6 fix requirement.** The recorder must preserve **channel identity**:
  per-channel statistics must be computable without retaining the full
  `[B, S, C]` tensor. The current `.reshape(-1)` flattening
  (`instrument.py:171-204`) makes per-channel RMS, near-zero occupancy, and
  covariance impossible from what is recorded.
- **Replay is not production.** A forward pass over stored inputs is a *replay*.
  It is identical to the production execution only if the evidence supports
  identity: same batching, same KV-cache state, same sampling parameters, same
  engine and engine version. A weight snapshot replayed through transformers is
  not the vLLM/SGLang production run, and a captured activation is a fact about
  one production request, not a reproducible property of the checkpoint.
  The report declares the replay's conditions (engine, version, batch, prompt
  set, seed) or says "replay conditions not declared," and single-prompt
  replays never scale into model-level claims (AGENTS.md BANNED list).

## 7. Streaming numerical statistics (O(channels) requirement)

Statistics the activation path must support, each in one pass over tokens with
memory bounded by the channel count C (or an explicit sketch budget):

| Statistic | Definition | Memory |
|-----------|------------|--------|
| Per-channel RMS | √(mean over tokens of x² per channel) | O(C) running sums of squares |
| Near-zero occupancy | Fraction of (channel, token) samples with |x| < τ | O(C) running counters (or a per-channel threshold histogram) |
| Second-moment / covariance | M = (1/n) Σ xᵀx per layer | O(C²) — valid at C = 3072 (≈151 MB fp32); at C = 12288 (≈585 MB fp32) it must fall back to the sketch, or be declared infeasible on this box |
| Effective dimensionality | Participation ratio PR = (Σλ)²/Σλ² over the second-moment spectrum (technique name only; citation not verified) | inherits the M or sketch budget |
| Rank-family sketches | Randomized numerical linear algebra (randomized SVD / sketching) for spectrum estimates where valid — a sketch recovers the top-r spectrum when the matrix is reasonably conditioned; it does not certify the tail (technique name only; citation not verified) | O(C·k), k ≪ C |

Rules:

1. The threshold τ and the sketch size k are **declared parameters of the
   artifact**, never implicit.
2. A statistic computed from a sketch or a capped window carries
   `sampling_mode` + coverage, exactly as weight tiles do (§ 8).
3. Where O(C²) is infeasible on the budget (§ 9), the acquisition layer says so
   in the capability manifest rather than silently degrading.

## 8. Exact vs sampled measurement semantics

- **A 2.1% tile is never presented as a full-matrix number.** The machinery
  exists and must stay: `parent_shape` + `coverage` + `sampling_mode` +
  `truncated` on every sampled weight artifact; `percentage_of_full_rank` is
  **tile-relative** by contract (`concepts/headroom-signals.md` R-03 — a 40-row
  tile of a rank-40 parent reads 80.6%, of a rank-5120 parent 99.97%); the
  human report prints `PARTIAL TILE` and states that inflation against the full
  matrix is not measured by the tool.
- The sampled-vs-exact label extends to every new statistic class (activations,
  routing counts, sketches) via the same fields.
- **Coverage reporting** at the composite level is `compute_coverage_metrics`
  (`analysis/scoring.py`): measurement coverage, measured-only score,
  lower/upper bounds, and the decision-grade gate (≥70% measured coverage,
  `--analysis-mode decision_grade`, exit 2).

## 9. Compute & memory budget (measured, this machine)

| Item | Value | Evidence |
|------|-------|----------|
| GPUs | 2× Tesla T4, sm_75 (compute capability 7.5) | nvidia-smi / `llama-server --list-devices`, 2026-11-10 |
| VRAM per device | 14911 MiB | same |
| VRAM already in use at baseline | ~11.7 GB | measured 2026-11-15 |
| RAM | ~31 GB | measured 2026-11-15 |
| fp8 | **impossible as kernels** — sm_75 < sm_89; fp8 checkpoints dequantize to bf16 | hard constraint, `docs/guides/engine-selection.md` |
| bf16 | emulated on sm_75 (native bf16 tensor cores need sm_80+) — slower, more VRAM | measured OOMs, 2026-11-13 pitfall ticket |
| 18 GB checkpoint vs 15 GB VRAM | does not fit; CPU-RAM load + GPU compute is the correct architecture, not a workaround | measured |
| forward+backward on a 9.65 B bf16 model | OOM even at seq_len 4 | measured 2026-11-13 |
| full-matrix SVD over ~700 large matrices | no completion after minutes at 86% CPU | measured 2026-11-13 |

Consequences for the acquisition layer (all derived from the table, not
preferences):

1. **Per-tensor reads only** — `safe_open`, one matrix at a time; a multi-shard
   checkpoint is never materialised.
2. **Row-capped, seeded sampling** for weight spectra (seed 42 default, full
   column dimension kept, full `parent_shape` reported) — the
   `ternary_sampling.py` / `safetensors_sampling.py` contract.
3. **Streaming O(C) statistics** for activations (§ 7); O(C²) only below the
   declared channel threshold.
4. **Device through `device.py`** — accelerator preferred, CPU fallback, an
   explicit unavailable accelerator is an error, never a silent downgrade (NFR-01).

## 10. Hu sidecar integration & caching

**What the sidecar provides today.** The carrier (opt-in `CAPTURE_HEADROOM`)
emits `{capture_id}.headroom.json` + `{capture_id}-weight_pairs.json`
post-response, async, never on the response path (ADR-001). Three keys
produce artifacts: `effective_rank` (weight tensors from the checkpoint),
`specialization` (expert-weight tensors, when ≥2 same-shape experts resolve),
and `load_balance` (metadata-only activation artifact). The remaining keys
are reserved null: `dead_neurons` (needs forward hooks the proxy does not
run), `loss_saturation` and `growth_factor` (reserved null — **training
telemetry is unobservable at the proxy layer**). Describe reserved-null
keys as *reserved and unobservable at the proxy layer*, never as missing
features.

**Caching design (PROPOSED).**

1. **Immutable checkpoint-derived diagnostics** (weight rank, layout facts,
   architecture facts, streaming weight statistics) are cacheable under the
   identity `(model revision, file sha256, sampling parameters, software
   version)` — same identity, same result, cache hit. The revision component is
   load-bearing per `evidence-provenance.md` § 5.
2. **Genuinely dynamic observations** (activations, routing counts, refusal
   probes) are keyed by **capture id** — they depend on the input and the
   serving state, not only the checkpoint, and a cache keyed on the checkpoint
   alone would serve a stale "measurement" of a different request.
3. **Hot-path isolation (invariant).** Sidecar failures never touch the
   inference hot path: the carrier writes post-response via fire-and-forget
   tasks; the authority's cache reads/writes are offline. A cache bug may cost
   a recompute; it may never cost a proxied response.

## Cross-references

- `model-headroom/PROJECT-ROADMAP.md` — defects D-1..D-7, milestones, integration matrix
- `model-headroom/evidence-provenance.md` — provenance classes, revision identity, calibration
- `model-headroom/measurement-ontology.md` — ontology classes, dense-as-single-expert, routing instrumentation
- `docs/capture-proxy/headroom-capture-design.md` — ADR-001..ADR-004 (carrier side)
- `docs/guides/engine-selection.md` — serving vs measurement, hard constraints, probe
- `docs/guides/support-matrix.md` — verified vs implemented families
