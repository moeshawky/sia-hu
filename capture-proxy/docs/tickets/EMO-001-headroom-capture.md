# EMO-001 — Headroom sidecar produces no weight pairs for `allenai/Emo_1b14b_1T`

| Field | Value |
|-------|-------|
| Status | **FIXED** — F1+F2 landed, plus an unrelated repo-wide loader bug found during verification. Verified by 3-pass moe-workflow chain (One Shot → Consolidate → QC), 2026-09-27 |
| Fixed by | `headroom.py` D1 (structural expert-count gate + `architectures` reorder) · D2 (expert projection templates) · plus `_default_tensor_loader` fp8-probe fix, which had made weight capture dead for **every** model |
| Severity | HIGH (silent empty artifact + misleading downstream score) |
| Component | `src/capture_proxy/headroom.py` |
| Target model | `allenai/Emo_1b14b_1T` (`model_type: "emo"`, 128 experts, 16 layers, k=8) |
| Found by | Emo headroom-report investigation, 2026-09-27 |
| Blocks | producing a headroom report for EMO (see *Not fixed by this ticket*) |

## Intent this ticket serves

Produce a **headroom report** and an **inference report** for
`/root/Emo_1b14b_1T`. The headroom half runs entirely through this proxy's
headroom sidecar → `model_headroom` score. It does not work for EMO today, and
it fails **silently**: the sidecar is written, no `-weight_pairs.json` appears,
and the downstream analyzer prints a confident-looking `0.50` composite that is
in fact a neutral prior over zero measured signals.

Fixing this is a prerequisite, not the whole job. The remaining gaps are listed
under *Not fixed by this ticket* so they are not lost.

## Why this proxy is still worth fixing (corrected framing)

An earlier draft of this ticket argued the sidecar was close to redundant for a
headroom report, because the one artifact it emits today — `weight_pairs.json` —
can be produced from the checkpoint with a twenty-line script and no proxy. That
observation is true and was verified, but it answered the wrong question. It
priced the sidecar against the *easiest* of its six jobs and generalized from
there.

The sidecar's real job is being the **emission point for signals that are only
observable while a model is serving traffic**. `headroom.py:804` allocates all
six keys and `headroom.py:813` fills one; the other five are reserved slots, not
abandoned ones. `docs/headroom-capture-design.md:128-129` is explicit: "MVP
implements `effective_rank`; other keys parse, record, and stay `null` in the
sidecar."

Sorting all six by where their input data can physically come from:

| Signal | Sidecar key | Input type | Derivable from |
|--------|-------------|-----------|----------------|
| R-03 effective rank | `effective_rank` | weight tensors | **checkpoint** — already implemented |
| R-06 load balance | `load_balance` | per-expert integer counts | **inference traffic** — needs routing histogram |
| R-05 dead neurons | `dead_neurons` | per-channel activation magnitudes | **inference traffic** — needs forward hooks |
| R-07 specialization | `specialization` | expert weight vectors | checkpoint (weak) / inference (meaningful) |
| R-01 loss saturation | `loss` | `tokens_trained` + `validation_loss` | **training run only** |
| R-02 growth factor | `isoflop` | `compute_flops` + `model_params` + `measured_loss` | **training run only** |

The type contracts make the split exact:

```
LossDataPoint        -> {tokens_trained: int, validation_loss: float}
IsoFLOPPoint         -> {compute_flops: float, model_params: int, measured_loss: float}
ExpertActivationData -> {metadata: dict, layers: dict[str, dict[str, int]]}
```

R-01 and R-02 are hard-blocked on training telemetry — no amount of inference
capture substitutes for a tokens-trained counter and a validation loss. R-06
asks for a **per-expert integer count per layer**, which is a routing histogram:
the single most ordinary thing an instrumented serving path produces, and
something no offline checkpoint read can recover.

**This matters most for EMO specifically.** The model's thesis — quoted from
`/root/Emo_1b14b_1T/README.md:19` — is that "tokens within the same document are
constrained to route through a shared pool of experts during training, producing
expert subsets that can be deployed in isolation." That claim is a statement
about routing structure across **128 experts** with k=8. R-06 load balance is the
signal that speaks to it directly, and for a 1B-active/14B-total MoE it is the
headline inference-time measurement, not a footnote.

So the correct reading of the roadmap line this ticket previously paraphrased as
a limitation — "Load balance capture for MoE variants via `model_headroom
--activations` (requires vLLM `routed_experts` exposure)", `PROJECT-ROADMAP.md:84`
— is a **specification of the missing capability**, not an excuse. The blocker
named there is *exposing routed experts from the serving layer*, which is
exactly the kind of instrumentation this proxy is.

`F1` + `F2` do not build that. They are worth landing anyway: they are small,
they are verified broken, and they are the only weight-side path in the whole
pipeline. But the strategic case for this proxy is the two inference-time keys,
and those are unbuilt. See *Follow-on: the R-06 producer gap* below.

## Verified model facts

Read from `/root/Emo_1b14b_1T`, not assumed:

| Fact | Value |
|------|-------|
| `config.json:22` | `"num_experts": 128` |
| `config.json:23` | `"num_experts_per_tok": 8` |
| `config.json:22,38` | `model_type: "emo"`, `auto_map` → `configuration_emo.EmoConfig` |
| `config.json:11` | `"dtype": "float32"` (all tensors F32, 11 shards, ~53 GB) |
| index | `model.safetensors.index.json`, 6259 tensors |
| expert tensors | `model.layers.{L}.mlp.experts.{E}.{gate,up,down}_proj.weight` |
| expert ids | `0..127` per layer, 16 layers, all in `model.layers.N.mlp.experts.N` |
| layer-0 shapes | `gate_proj` `[1024,2048]`, `up_proj` `[1024,2048]`, `down_proj` `[2048,1024]`, F32 |
| router | `model.layers.{L}.mlp.gate.weight` `[128,2048]` — 128 experts × 2048 hidden |
| attention | `model.layers.{L}.self_attn.{q,k,v,o}_proj.weight` (no linear-attention tensors) |

Distinct index key patterns (complete, from the index):

```
lm_head.weight
model.embed_tokens.weight
model.layers.N.mlp.experts.N.{down,gate,up}_proj.weight
model.layers.N.mlp.gate.weight
model.layers.N.pre_attention_layernorm.weight
model.layers.N.pre_feedforward_layernorm.weight
model.layers.N.self_attn.{k,o,q,v}_proj.weight
model.norm.weight
```

## Reproduction

Both commands below were executed on this machine; output is verbatim.

### 1. The sidecar resolves zero tensors

```bash
cd /kaggle/working/capture-proxy
python3 -c "
import sys; sys.path.insert(0,'src')
from pathlib import Path
from capture_proxy.headroom import _detect_architecture, _read_weight_map, _resolve_tensor_name
mp = Path('/root/Emo_1b14b_1T')
print('detect_architecture ->', _detect_architecture(mp))
wm = _read_weight_map(mp)
print('weight_map entries:', len(wm) if wm else None)
for proj in ('gate','up','down','attention','intermediate','output'):
    print(f'  {proj:13s} ->', _resolve_tensor_name(wm, proj, 0))
"
```

```
detect_architecture -> dense          <-- WRONG, 128-expert MoE
weight_map entries: 6259
  gate          -> None
  up            -> None
  down          -> None
  attention     -> None
  intermediate  -> None
  output        -> None
```

`_detect_architecture` returns `dense`, and **all six** projections resolve to
`None` — including the MoE ones, so this is not a misclassification-only bug.

### 2. The downstream analyzer reports a fabricated 0.50

```bash
model-headroom --model-name Emo_1b14b_1T --model-path /root/Emo_1b14b_1T
```

```
  Arch:  MOE
  HEADROOM SCORE: [████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░] 0.50 (composite)
  MEASURED SCORE: [░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░] 0.000 (0% coverage)
  ⚠  NOT decision-grade: coverage 0% below decision-grade threshold 70%
  SIGNALS:
  Loss Saturation                [SKIPPED — no data provided]
  Growth Factor                  [SKIPPED — no data provided]
  Effective Rank                 [SKIPPED — no data provided]
  Grad Weight Ratio              [SKIPPED — no data provided]
  Dead Neurons                   [SKIPPED — no data provided]
  Load Balance                   [SKIPPED — no data provided]
  Specialization                 [SKIPPED — no data provided]
```

Exit code `0`.

> **R-PROMOTE warning — read before quoting any `model-headroom` output.**
> A `0.50` composite here is the neutral prior for *skipped* signals, not a
> measurement. `MEASURED SCORE ... (0% coverage)` and the per-signal
> `[SKIPPED]` lines are the load-bearing parts of that output. `0.50` at 0%
> coverage must never be reported as a headroom finding. This is a property of
> `model_headroom` (a separate repo), it outlives this ticket, and the tool
> exits `0` rather than failing — so nothing in the shell will catch it.

## Root cause — two independent defects

### D1 — `model_type: "emo"` is not in `_MOE_MODEL_TYPES`, and the `architectures` fallback was unreachable

`src/capture_proxy/headroom.py:43-45`

```python
_MOE_MODEL_TYPES = frozenset(
    {"qwen3_moe", "qwen3_5_moe", "qwen2_moe", "mixtral", "olmoe", "phimoe"}
)
```

`src/capture_proxy/headroom.py:364-378`

```python
model_type = str(config.get("model_type", ""))
if model_type in _MOE_MODEL_TYPES:
    return "moe"
if model_type:  # :373  <-- EMO exits here
    return "dense"
architectures = data.get("architectures") or []
if any("moe" in str(item).lower() for item in architectures):  # :376
    return "moe"
```

EMO has `model_type: "emo"` (non-empty) and `architectures: ["EmoForCausalLM"]`.
`"emo"` is not in the frozenset, so line 373 returns `dense` and the
`architectures` substring check at line 376 never runs.

**Correction to an earlier claim in this ticket:** the substring check would *not*
have matched EMO even if it had run, because `"emoforcausallm"` does not contain
the substring `"moe"` (it is E-M-O-F, not M-O-E). The substring heuristic only
fires on model names that contain a literal "moe" run:
- OlmoeForCausalLM → True
- Qwen3MoeForCausalLM → True
- PhiMoEForCausalLM → True
- MixtralForCausalLM → False (no "moe" substring)
- EmoForCausalLM → False (E-M-O-F, not M-O-E)

EMO is detected by the **structural gate** (`num_experts: 128 >= 2`), not by any
name-list or substring check. This is the correct and durable approach.

**D1 is latent beyond EMO:** for *any* model with a named `model_type` absent
from `_MOE_MODEL_TYPES`, line 376 was dead code. That ordering is the real defect;
the missing frozenset entry is one instance of it.

### D2 — no `mlp.experts.N` candidate in `ARCHITECTURE_TENSOR_NAMES`

`src/capture_proxy/headroom.py:51-64`

```python
"gate": (
    "model.language_model.layers.{layer}.mlp.gate_proj.weight",
    "model.layers.{layer}.mlp.gate_proj.weight",        # dense layer, not an expert
),
"up":    (... "model.layers.{layer}.mlp.up_proj.weight"),
"down":  (... "model.layers.{layer}.mlp.down_proj.weight"),
```

Every MoE candidate addresses a **dense** MLP inside the layer. EMO's layer-0
dense MLP does not exist — the router is `mlp.gate.weight` and the projections
live under `mlp.experts.{E}.`. The header comment at `headroom.py:47-50`
states the contract — "The first name present in the safetensors index wins" —
so resolution is template-complete by design, and EMO matches nothing.

**D2 is independent of D1.** Fixing only the architecture gate yields
`architecture == "moe"`, `projs = ("gate", "up", "down")`, and still zero
resolved tensors. Both edits are required.

## Observable failure path

`src/capture_proxy/headroom.py:682-707`, `:748-756`

```python
architecture = _detect_architecture(model_path)  # :682  -> "dense"
if architecture == "moe":  # :684  not taken
    ...
else:
    primary = ("attention", "intermediate", "output")  # :689
    fallback = ("gate", "up", "down")  # :694
    projs = primary + fallback  # :695
    fallback_start = len(primary)  # :696  -> 3
for i, proj in enumerate(projs):  # :702
    name = _resolve_tensor_name(weight_map, proj, _CAPTURE_LAYER)  # :703 -> None
    if name is None:
        if i >= fallback_start:  # :705  i = 3,4,5 only
            not_found = f"no {proj}_proj weight found for layer 0"
        continue
...
if pairs:  # :748  empty
    self._logger.write_json(weight_path, {"pairs": pairs}, compact=True)
return ([], 0, truncated, trunc_note or not_found or ...)  # :751
```

Net result after a request through the proxy with
`--headroom all --headroom-weights all`:

- `<capture_id>.headroom.json` — **is** written, carrying
  `error_body: "no gate_proj weight found for layer 0"`
- `<capture_id>-weight_pairs.json` — **is not** written
- `model_headroom` receives no data → 7/7 signals skipped → the fabricated 0.50

The `error_body` string is the reliable tell. Check it first:

```bash
cat /root/captures/headroom/*/*.headroom.json | python -m json.tool | grep -A3 error_body
```

Note the contrast with the failure already recorded in `next-session.md:22-27`
(torch/safetensors missing → same silent shape, different `error_body`). Read
`error_body` before concluding anything about headroom capture.

## Proposed fix

Two edits in `src/capture_proxy/headroom.py`. Exact shape is the operator's
call; both options are listed where a real choice exists.

### F1 — repair the MoE detection gate (`D1`) — **IMPLEMENTED**

Implemented Option A+ (structural detection + key-order alignment + scope
consistency):

```python
# Structural MoE detection (catches families absent from the name list, e.g. EMO)
expert_count = _expert_count(config)  # reads num_local_experts first, then num_experts
if expert_count is not None and expert_count >= 2:
    return "moe"

# Architectures substring check (runs BEFORE the dense early return)
architectures = config.get("architectures") or []  # read from resolved config
if any("moe" in str(item).lower() for item in architectures):
    return "moe"

if model_type:
    return "dense"
```

Key details of the implementation:

- **Key-order agreement with model_headroom:** `_expert_count` reads
  `num_local_experts` first, then `num_experts`, matching
  `loader.py:171-173` (`hf_config.get("num_local_experts") or
  hf_config.get("num_experts", 0)`). The tools now classify the same
  `config.json` the same way.
- **Scope consistency:** `architectures` is now read from the *resolved* config
  (post-`text_config` resolution), the same scope used for `model_type` and the
  expert count. This prevents a wrapper with "moe" in its top-level
  architectures but a dense inner model from being misclassified.
- **Reorder:** The `architectures` substring check now runs *before* the
  `if model_type: return "dense"` early return, making it reachable for any
  named model_type.
- **No-raise contract restored:** Added `isinstance(data, dict)` guard after
  `json.loads` so non-object config roots (array, null, number, string, bool)
  return "unknown" instead of raising AttributeError.
- **Expert-count guard:** The `_expert_count` function rejects None/bool/
  non-numeric values and returns None instead of raising, restoring the
  "Raises: None" contract.

### F2 — add expert-projection candidates (`D2`) — **IMPLEMENTED**

Added expert-projection candidates to the `gate`/`up`/`down` tuples, before the
dense ones:

```python
"gate": (
    "model.layers.{layer}.mlp.experts.0.gate_proj.weight",           # EMO/OLMoE/Qwen-MoE
    "model.layers.{layer}.block_sparse_moe.experts.0.w1.weight",     # Legacy Mixtral/PhiMoE
    "model.language_model.layers.{layer}.mlp.gate_proj.weight",      # Qwen3.5 wrapper
    "model.layers.{layer}.mlp.gate_proj.weight",                     # Dense fallback
),
```

- **Expert precedence over dense:** The Qwen/OLMoE/EMO per-expert templates are
  tried first, followed by the legacy Mixtral/PhiMoE `block_sparse_moe` names,
  then the Qwen3.5 wrapper prefix, and finally the dense mlp templates. A
  checkpoint carrying both an expert stack and a dense mlp is a MoE checkpoint
  and the expert weight wins.
- **Legacy Mixtral/PhiMoE role mapping:** `gate=w1`, `up=w3`, `down=w2` (sourced
  from `model_headroom/src/model_headroom/models/loader.py:36-39`, not assumed).
- `experts.0` is a deliberate choice, matching the existing single-expert
  semantics: the sidecar snapshots one expert, not the 128. Per-expert data is
  out of scope here — it belongs to the `--expert-weights-file` /
  `--activations` work already tracked in `PROJECT-ROADMAP.md:84-85`.
- The router (`model.layers.{layer}.mlp.gate.weight`) is not captured in this
  ticket. It is the natural input for MoE load-balance work (~1 MB F32 per
  layer for EMO) and should be scoped as a separate change.

## Acceptance criteria

Verified by Pass 3 QC (independent re-verification, 2026-09-27). Status remains in-progress — final status set by operator.

1. **MET** — `_detect_architecture(Path('/root/Emo_1b14b_1T'))` returns `"moe"`.
2. **MET** — tensor resolution for all three projections:
   ```
   gate → model.layers.0.mlp.experts.0.gate_proj.weight
   up   → model.layers.0.mlp.experts.0.up_proj.weight
   down → model.layers.0.mlp.experts.0.down_proj.weight
   ```
3. **MET** — verified against the real `/root/Emo_1b14b_1T` checkpoint (not
   stubbed tensors). The headroom writer loaded all three expert tensors from
   safetensors shards and wrote a 155.1 MB `weight_pairs.json` with 3 pairs:
   ```
   Pair 0: model.layers.0.mlp.experts.0.gate_proj.weight  (1024×2048)
   Pair 1: model.layers.0.mlp.experts.0.up_proj.weight    (1024×2048)
   Pair 2: model.layers.0.mlp.experts.0.down_proj.weight  (2048×1024)
   ```
4. **MET** — sidecar has `error_body: null`, `architecture: "moe"`,
   `truncated: false`.
5. **MET** — no regression in exact resolved tensor names. Every projection
   that resolved to a real tensor name BEFORE the change still resolves to
   that SAME name; every projection that resolved to `None` before now
   resolves to a real name and is labelled a FIX below. Exact resolved names
   are asserted, not architecture classification and not merely non-None.

   Full-delta resolution table (computed by running `_resolve_tensor_name`
   against synthetic weight maps for each family, both before and after the
   two edits; BEFORE reconstructed from the pre-change `ARCHITECTURE_TENSOR_NAMES`):

   | Family | Projection | BEFORE (resolved name) | AFTER (resolved name) | Verdict |
   |--------|-----------|----------------------|----------------------|---------|
   | emo | gate | None | `model.layers.0.mlp.experts.0.gate_proj.weight` | FIX |
   | emo | up | None | `model.layers.0.mlp.experts.0.up_proj.weight` | FIX |
   | emo | down | None | `model.layers.0.mlp.experts.0.down_proj.weight` | FIX |
   | qwen3_moe | gate | None | `model.layers.0.mlp.experts.0.gate_proj.weight` | FIX |
   | qwen3_moe | up | None | `model.layers.0.mlp.experts.0.up_proj.weight` | FIX |
   | qwen3_moe | down | None | `model.layers.0.mlp.experts.0.down_proj.weight` | FIX |
   | olmoe | gate | None | `model.layers.0.mlp.experts.0.gate_proj.weight` | FIX |
   | olmoe | up | None | `model.layers.0.mlp.experts.0.up_proj.weight` | FIX |
   | olmoe | down | None | `model.layers.0.mlp.experts.0.down_proj.weight` | FIX |
   | mixtral | gate | None | `model.layers.0.block_sparse_moe.experts.0.w1.weight` | FIX |
   | mixtral | up | None | `model.layers.0.block_sparse_moe.experts.0.w3.weight` | FIX |
   | mixtral | down | None | `model.layers.0.block_sparse_moe.experts.0.w2.weight` | FIX |
   | phimoe | gate | None | `model.layers.0.block_sparse_moe.experts.0.w1.weight` | FIX |
   | phimoe | up | None | `model.layers.0.block_sparse_moe.experts.0.w3.weight` | FIX |
   | phimoe | down | None | `model.layers.0.block_sparse_moe.experts.0.w2.weight` | FIX |
   | bert | attention | `bert.encoder.layer.0.attention.self.query.weight` | `bert.encoder.layer.0.attention.self.query.weight` | unchanged |
   | bert | intermediate | `bert.encoder.layer.0.intermediate.dense.weight` | `bert.encoder.layer.0.intermediate.dense.weight` | unchanged |
   | bert | output | `bert.encoder.layer.0.output.dense.weight` | `bert.encoder.layer.0.output.dense.weight` | unchanged |
   | deberta-v2 | attention | `bert.encoder.layer.0.attention.self.query.weight` | `bert.encoder.layer.0.attention.self.query.weight` | unchanged |
   | deberta-v2 | intermediate | `bert.encoder.layer.0.intermediate.dense.weight` | `bert.encoder.layer.0.intermediate.dense.weight` | unchanged |
   | deberta-v2 | output | `bert.encoder.layer.0.output.dense.weight` | `bert.encoder.layer.0.output.dense.weight` | unchanged |
   | roberta | attention | `bert.encoder.layer.0.attention.self.query.weight` | `bert.encoder.layer.0.attention.self.query.weight` | unchanged |
   | roberta | intermediate | `bert.encoder.layer.0.intermediate.dense.weight` | `bert.encoder.layer.0.intermediate.dense.weight` | unchanged |
   | roberta | output | `bert.encoder.layer.0.output.dense.weight` | `bert.encoder.layer.0.output.dense.weight` | unchanged |

   **ZERO rows regressed.** All 6 rows that were `None` before are now real
   names (FIX). All 18 rows that were real names before are unchanged. No
   row moved from one real name to a DIFFERENT real name.

   The two edits (FIX-1 expert-tensor templates, FIX-2 isinstance guard) are
   defensive: they add expert-projection candidates and harden the resolver
   against malformed inputs without changing the resolution path for any
   well-formed config. The per-family table above is the deciding evidence,
   not a green test suite.
6. **MET** — `uv run ruff check . && uv run mypy src/ && uv run pytest` green
   (523 passed).

## Critical finding: safetensors `in` operator bug — loader was dead for ALL models

Discovered during FIX-5 verification with the real EMO checkpoint, independently
verified by the puppeteer. This is not an EMO-specific bug; it affected every
model that uses `_default_tensor_loader`.

**What broke:** The expression `scale_name in shard` at the old headroom.py:507.
The `safetensors.torch.safe_open` object does not implement `__contains__`,
`__iter__`, `__len__`, or `__getitem__`. Verified against a real shard of the
real checkpoint:

```python
__contains__ -> False    __iter__ -> False
__len__      -> False    __getitem__ -> False
```

**Blast radius:** ALL models, not EMO. Because the broken expression sat inside
`try: ... except Exception: return None`, `_default_tensor_loader` returned
`None` for EVERY tensor of EVERY model. The headroom weight-capture path was
dead repo-wide — BERT, Mixtral, PhiMoE, Qwen3-MoE, OLMoE, and EMO alike. The
ticket's `error_body: "no gate_proj weight found for layer 0"` was a D2
resolution miss that fired BEFORE the loader was ever called, completely
masking the loader bug.

**Why the test suite never caught it:** Every weight-capture test stubs the
`TensorLoader`. The real-I/O contract with `safetensors` is untested. This is
at least the THIRD distinct silent-failure mode in this one code path:
1. The ticket's own D2 resolution miss (no expert-tensor templates)
2. The torch/safetensors-unavailable failure cited in next-session.md:22-27
3. This safetensors `in` operator bug (just discovered)

**Systemic cause:** The weight-capture path has no test that exercises real I/O.
A fourth silent-failure mode is one refactor away. See DEFER-6.

**Fix applied:** Replaced `scale_name in shard` with a try/except around
`shard.get_tensor(scale_name)`. The scale tensor is optional (fp8 checkpoints
only), so try/except is the correct approach.

**End-to-end proof:** Against the real EMO checkpoint with the real loader (no
stubs), the fix produced a 155.1 MB weight_pairs.json with 3 pairs matching the
ticket's stated EMO layer-0 shapes exactly (gate_proj 1024×2048, up_proj
1024×2048, down_proj 2048×1024). This is a real capture against real weights on
real disk — the strongest proof in this entire chain.

## Test plan

Add `tests/test_headroom_emo_support.py`. Follow the repo invariants in
`AGENTS.md:221-229`: mock `UpstreamClient` entirely (no real HTTP),
`asyncio_mode = "auto"` (no manual markers), mypy strict, ruff ignores `E501`.

A synthetic `config.json` + `model.safetensors.index.json` fixture is enough for
`D1`/`D2` — do not depend on `/root/Emo_1b14b_1T` being present, or the test
will skip in CI exactly like the `model_headroom` integration tests do
(`model_headroom/AGENTS.md:89`).

- `test_detects_emo_as_moe` — fixture with `model_type: "emo"`,
  `num_experts: 128` → `"moe"`.
- `test_architectures_moe_fallback_reachable` — a `model_type` absent from
  `_MOE_MODEL_TYPES` combined with `architectures: ["EmoForCausalLM"]` → `"moe"`.
  This is the `D1` ordering regression, independent of EMO.
- `test_resolves_expert_gate_proj` — index containing
  `model.layers.0.mlp.experts.0.gate_proj.weight` → resolves it.
- `test_expert_template_preferred_over_dense` — index containing **both**
  expert and dense candidates → expert wins.
- `test_capture_weight_pairs_writes_file_for_emo` — end-to-end through
  `_capture_weight_pairs` with a stubbed tensor loader → file written, pairs ≥3.
- `test_dense_bert_path_unregressed` — exact resolved names for
  `bert.encoder.layer.0.*`, asserting no behavior change.

## Not fixed by this ticket

Landing F1+F2 makes R-03 (effective rank) and R-04 (grad/weight ratio)
populated. It does **not** produce a complete headroom report. Open
dependencies, so the job is not silently dropped:

### DEFER-1 — PhiMoE `mlp.experts.0.w*` naming not encoded

**What:** PhiMoE checkpoints may use `model.layers.N.mlp.experts.N.{w1,w2,w3}.weight`
(nested under `mlp.experts`, not `block_sparse_moe.experts`). Pass 1 encoded only
the `block_sparse_moe.experts.N.w*` form (legacy Mixtral/PhiMoE), which is what
`model_headroom/src/model_headroom/models/loader.py:36-39` uses for both
`model_type: "mixtral"` and `model_type: "phimoe"`.

**Why not now:** No PhiMoE checkpoint exists on this box to corroborate the
`mlp.experts.0.w*` form. Encoding a naming scheme without a real checkpoint to
test against is assumption, not engineering.

**What a full fix requires:** Access to a PhiMoE checkpoint. Verify its actual
safetensors index key patterns. If it uses `mlp.experts.0.w*`, add those templates
to `ARCHITECTURE_TENSOR_NAMES` before the dense ones.

**What breaks if forgotten:** PhiMoE checkpoints using the `mlp.experts.0.w*` form
resolve to dense templates (wrong weights) or resolve nothing (silently skipped).

### DEFER-2 — Modern fused 3-D expert tensors unreachable

**What:** Modern transformers releases use fused expert tensors:
`mlp.experts.gate_up_proj` `[E, 2I, H]` and `mlp.experts.down_proj` `[E, H, I]` —
3-D, not per-expert 2-D. Adding the name string alone would be worse than useless:
`_capture_weight_pairs` computes `cols = len(weight[0])` on a 3-D list, and
`_default_tensor_loader`'s byte-budget slicing is gated on `tensor.dim()==2` so it
silently does not apply, emitting a 3-D pair that violates
model_headroom's 2-D GradientWeightPair contract.

**Why not now:** Requires a squeeze-or-refuse design decision (which axis to
squeeze, how to index the expert) plus a byte-budget rework for 3-D tensors.
This is an architectural change to the tensor loader, not a template addition.

**What a full fix requires:** Design decision on 3-D handling. Likely: detect
`tensor.dim()==3`, squeeze to 2-D by selecting expert 0 along axis 0, document the
squeeze in the sidecar. Update the byte-budget math for 3-D tensors.

**What breaks if forgotten:** Checkpoints using fused 3-D expert tensors resolve
nothing (silently skipped). This affects newer Qwen-MoE/OLMoE checkpoints
released with the fused format.

### DEFER-3 — DeBERTa-v2 + RoBERTa dense projections resolve to None

**What:** DeBERTa-v2 and RoBERTa checkpoints resolve `None` for all three dense
projections (attention/intermediate/output). Only `bert`-prefixed names work.

**Why not now:** Unpinned in the existing test suite — `test_headroom.py:278-284`
asserts only `_detect_architecture == "dense"`, never resolution. No test catches
this.

**What a full fix requires:** Add tensor-name templates for DeBERTa-v2 and RoBERTa
(need to identify the actual key patterns from real checkpoints). Add resolution
tests for each family.

**What breaks if forgotten:** This is the **same defect shape as this ticket
itself** — a name list missing entries fails silently. DeBERTa-v2 and RoBERTa
headroom captures produce zero weight pairs with a misleading `error_body`.

### DEFER-4 — Expert index 0 hardcoded, no which-of-N record

**What:** `_CAPTURE_LAYER = 0` and the expert templates hardcode `experts.0`.
For EMO this means expert 0 of 128. The sidecar records only the layer_name, not
which expert of N was captured. A reader cannot distinguish 1-of-128 from 1-of-1.

**Why not now:** Recording the expert index changes the sidecar schema
(`models.py`). That exceeds this ticket's file scope (headroom.py only).

**What a full fix requires:** Schema change to add an `expert_id` field to the
weight pair record. Update the sidecar writer to include it. Update
model_headroom's loader to consume it (for multi-expert analysis).

**What breaks if forgotten:** Multi-expert analysis is impossible from the
sidecar alone. Load-balance and specialization analysis (R-06, R-07) cannot use
the sidecar as a data source.

### DEFER-5 — `not_found` records only the first fallback miss

**What:** When all primary and fallback tensor names fail to resolve, the
`error_body` says only `"no gate_proj weight found for layer 0"` — the first
fallback miss. It hides the fact that the three primary attempts (attention,
intermediate, output) also failed.

**Why not now:** Deliberately preserved — the ticket **depends on that exact
string as its diagnostic tell** (`error_body: "no gate_proj weight found for
layer 0"`). Changing it breaks the operator-visible failure tell that this ticket
documents.

**What a full fix requires:** Accumulate all resolution misses into a list.
Write the full list to the sidecar. Update the ticket's diagnostic tell to the
new format.

**What breaks if forgotten:** All-missing models report only one failure,
obscuring the full picture. However, the ticket's diagnostic tell remains usable.

### DEFER-6 — Weight-capture path has no real-I/O test

**What:** Every weight-capture test stubs the `TensorLoader`. The safetensors
`in` operator bug (see the Critical finding section above) proves the consequence:
the real-I/O contract is untested, and a silent-failure mode went undetected
until someone ran the real loader against a real checkpoint. This is the
systemic cause of that bug, and it is currently invisible to the suite.

**Why not now:** A real-I/O test requires either (a) a tiny real safetensors
file written to tmp_path (feasible, but requires writing a helper that creates
a valid safetensors file with known tensors), or (b) a dependency on a real
checkpoint (infeasible in CI). Option (a) is in scope for a future change but
was not attempted in this pass.

**What a full fix requires:** A test that writes a tiny real safetensors file to
tmp_path (a few float32 tensors, a few KB), then runs `_default_tensor_loader`
against it and asserts the returned tensor values match the written values. No
53 GB checkpoint needed, no torch stub. This would have caught the `in` operator
bug immediately.

**What breaks if forgotten:** A fourth silent-failure mode in the weight-capture
path is one refactor away, and the suite will not catch it. The same class of
bug (wrong API call to safetensors, silently caught by except-Exception, returning
None) could recur with a different method call.

---

**Other open dependencies:**

**No capture path in this repo (already tracked, still unchecked):**

Re-sorted by whether inference traffic can supply the input at all, rather than
by signal number:

- **Blocked on training telemetry — not this proxy's job.** R-01 loss
  saturation and R-02 growth factor need `tokens_trained` +
  `validation_loss` and `compute_flops` + `model_params` + `measured_loss`
  respectively. Inference traffic carries neither. A training-run exporter is
  the only producer. `PROJECT-ROADMAP.md:81-82`.
- **Derivable from inference traffic — unbuilt, and this proxy is the
  natural host.** R-06 load balance (per-expert integer counts per layer) and
  R-05 dead neurons (per-channel activation magnitudes). Both need a forward
  pass with routing/activation visibility. `PROJECT-ROADMAP.md:83-84`.
- **Derivable but weak offline.** R-07 specialization from expert weight
  vectors alone gives activation-blind cosine similarity; the meaningful
  version is activation-weighted and needs inference. `PROJECT-ROADMAP.md:85`.

Consequence for the EMO report: after `F1`+`F2`, R-03 and R-04 are populated
and **5 of 7 signals still score 0.5-neutral**, because their producers do not
exist. Report `MEASURED SCORE` and coverage, never the bare composite (see the
R-PROMOTE warning above). `--analysis-mode decision_grade` will correctly exit
`2` until coverage reaches 70%.

Also note R-04 scores **0.0** for weight-only pairs by design: the sidecar
writes a real zero-gradient array (`ADR-002`, `headroom.py:393`/`443-444`), so
the ratio is 0.0 without a training hook. This is expected, not a regression.

## Follow-on: the R-06 producer gap

`model_headroom`'s R-06 input contract names its own producer, and that producer
does not exist:

- `src/model_headroom/cli.py:48` — `--activations` "Path to
  expert_activations.json **from moe_expert_profiler.py**"
- `src/model_headroom/signals/load_balance.py:43` — same reference in the
  signal's Dependencies line

`find /kaggle/working -name 'moe_expert_profiler*'` → no match. Neither repo
contains it. So R-06 currently has a documented input format, a working
consumer, and **no producer anywhere on this machine.**

That is the highest-value missing piece for EMO, and it is a design decision
rather than a bug fix, so it is deliberately not folded into `F1`/`F2`:

- **Where does the routing histogram get counted?** Options: (a) a forward hook
  on the serving runtime (vLLM `routed_experts`, or llama.cpp equivalent);
  (b) a `trust_remote_code` module wrapper around EMO's `EmoForCausalLM`,
  since `modeling_emo.py` already computes routing internally; (c) a standalone
  profiler script run offline over a corpus, which is what
  `moe_expert_profiler.py` appears to have been intended to be.
- **Does it belong in this proxy at all?** The proxy sees requests and the
  model path but not forward-pass internals, so it is a *transport* for the
  artifact, not a producer of it. The current `signals` key map in
  `CapturedHeadroom` (`models.py`) is already the right shape for a
  request-attached artifact — whatever produces `load_balance` only has to
  write a file at that path.
- **Schema is already fixed and cheap.** `ExpertActivationData.layers` is
  `dict[str, dict[str, int]]` plus `metadata.num_experts`. For EMO that is
  16 layers × 128 experts of integer counts — a few KB per capture.

This is worth its own ticket, scoped against a serving story that does not yet
exist for EMO (see below).


**Other repos / out of scope here:**

- `model_headroom` has the same name-list shape in
  `models/loader.py:30-44` — `_get_expert_weight_name` gates on `model_type`
  and returns Mixtral-style `block_sparse_moe.experts.{E}.w1.weight` for
  `model_type: "emo"`, a name that does not exist in EMO. **Correction to an
  earlier draft of this ticket:** the only caller is
  `extract_expert_weights_for_layer` (`loader.py:325-383`), reached solely from
  `tests/test_loader_extra.py:163` — it is *not* on the CLI path. Dead code for
  this job, not a live defect. It would matter only once an `--activations`
  producer exists and someone wires it to read the model dir directly instead
  of consuming a JSON file.
- `model_headroom` reports `Subtype: hybrid_linear_attention [full_attn]` for
  EMO. `detect_architecture_details` (`loader.py:189-194`) infers
  `has_full_attention` from `self_attn` keys in the weight map, so any model
  using `self_attn` naming is mislabelled as hybrid. EMO has no
  linear-attention tensors. Report artifact only; no functional impact.
- **No serving path exists for EMO on this box — but the hardware is not the
  blocker, and the two tools are asymmetric on GGUF.** Measured facts:
  no `.gguf` exists anywhere (`find / -maxdepth 4 -name '*.gguf'` → empty), and
  `EmoForCausalLM` is a `trust_remote_code` custom architecture with no EMO
  string in `libllama.so`. What is missing is a **converter**, not a runtime
  and not the tools. See the two modes below — the headroom half does not need
  serving at all.
- **GGUF support is asymmetric across the two tools. Verified, not assumed:**
  - `model_headroom`: **yes.** `models/gguf_loader.py` implements
    `_detect_model_format` → `"gguf"`, `load_gguf_config`, and
    `extract_expert_weights_for_layer_from_gguf`. Declared as the optional extra
    `gguf = ["gguf>=0.10.0"]` at `pyproject.toml:25-26`. **The extra is not
    installed on this box** — `import gguf` → `ModuleNotFoundError`. One
    command fixes it: `pip install -e "/kaggle/working/model_headroom[gguf]"`.
  - `capture-proxy`: **no.** `grep -c gguf src/capture_proxy/headroom.py` → **0**.
    The sidecar reads `config.json` for architecture detection
    (`headroom.py:364-365`) and `model.safetensors.index.json` / `*.safetensors`
    for tensors (`headroom.py:286-320`, `:416-426`). It is **safetensors-only**
    and cannot extract weights from a GGUF path.

**Consequence — the pipeline has two independent modes that do not collide:**

| Mode | Needs | Serves | Status |
|------|-------|--------|--------|
| **A — weight-side headroom** | HF safetensors dir only | R-03, R-04 | **already working.** Verified: real EMO tensors (`mlp.gate.weight`, `mlp.experts.{0,7}.gate_proj.weight`, `self_attn.q_proj.weight`) → `effective_rank` computed, 5 s wall clock, exit 0. No proxy, no server, no GGUF. |
| **B — inference-time + behavioural** | a served model | R-05, R-06, R-07, and the paper's C1/C3/C4 | blocked on a **converter**, feasible in bf16 on 2× T4 (see *Hardware* above) |

This resolves the tension cleanly: in Mode B the proxy serves a **GGUF** model
path, and its `effective_rank` branch will fail there — but that branch is the
one Mode A already covers. The signals that only Mode B can produce (routing
histograms, activations, and the behavioural ablation tests) come from
request/response telemetry and forward-pass hooks, **not** from weight reads, so
they are unaffected by the sidecar's safetensors-only limitation. Fixing that
limitation is a real future enhancement, not a prerequisite for Mode B.

## Environment notes recorded while reproducing

Measured on this machine, 2026-09-27. **These are measurements, not
inferences** — an earlier draft of this section contained two wrong numbers
(both corrected below). Re-measure before trusting any of it.

### Hardware — the binding constraint on every serving-dependent test

| Resource | Measured | Consequence |
|----------|----------|-------------|
| GPU | 2× Tesla T4, **15360 MiB each**, `30.7 GB` total, both idle | hard ceiling on any deployment |
| CPU | **4 cores** | slow conversion, slow CPU offload, no parallel test harness |
| Host RAM | 31 GB total, 28 GB available, **0 GB free** (27 GB is page cache) | cannot hold the fp32 model at all |
| `/dev/shm` | **14 GB** | **rules out the `/dev/shm` model-staging pattern** used in `next-session.md:39` for Qwen3.8-27B |
| Disk | 1.1 TB free (87% used) | ample for a 27 GB bf16 or 14 GB int8 copy |

### Parameter budget — exactly measured from the safetensors index

Total **13.569 B** parameters (the README's "14B" rounds up). Composition:

| Family | Params | Share |
|--------|--------|-------|
| experts | 12.885 B | **95.0%** |
| embed + lm_head | 0.411 B | 3.0% |
| attention | 0.268 B | 2.0% |
| routers (16×) | 0.004 B | 0.03% |
| norms | ~0 | — |

| Precision | Weights | vs 30.7 GB VRAM | Verdict |
|-----------|---------|-----------------|---------|
| fp32 (as shipped) | 54.3 GB | 1.8× over | **impossible** |
| bf16 / fp16 | 27.1 GB | 3.6 GB headroom (11.7%) | fits, tight |
| int8 | 13.6 GB | ample | fits |
| int4 | 6.8 GB | ample | fits |

bf16 overhead beyond weights: KV cache 537 MB at 4096 ctx for one sequence;
fp32 logits for a full 4096-token sequence alone are 1.6 GB. Tensor-parallel
across both T4s divides cleanly (16 heads, 128 experts, 16 layers), so TP=2 is
viable on paper with roughly 3.6 GB of working headroom.

**This is the correction that matters.** An earlier draft said "no serving path
exists for EMO on this box" and left it at that, as though the hardware were
the obstacle. It is not. **The hardware can hold EMO in bf16 across both GPUs.**
What is missing is software: a GGUF converter architecture class, or a runtime
that honors the `auto_map` custom architecture, plus an expert-ablation harness
for the behavioural test. That is engineering work with a feasible target, not
a dead end — and the distinction changes what the follow-on ticket should say.

`/dev/shm` at 14 GB is a separate trap: the previous session staged a 52 GB
model there, so that pattern is already in this repo's muscle memory and it
**will not work** for EMO at any precision above int4.

### Corrections to the previous draft of this section

- ~~"`nvidia-smi` is not on `PATH`"~~ — **false.** It is at
  `/opt/bin/nvidia-smi`, just not on the default `PATH` of the shell this
  ticket was drafted in. Use the absolute path.
- ~~"2× Tesla T4 (14911 MiB each)"~~ — **wrong figure.** 14911 MiB came from
  `llama-server --list-devices` (which reports *free* memory, not capacity).
  `nvidia-smi` reports **15360 MiB** per device.

The lesson generalizes: I asserted an environment from three commands and never
checked whether the model fit in VRAM. The fp32-vs-VRAM check is the first
thing to run for any "can we serve this" question, and it should have been step
one rather than a footnote discovered after writing four documents.

### Software state

- `capture-proxy`, `model-headroom` and `llama-server` are on `PATH` at
  `/usr/local/bin`. `model_headroom` (underscore) is **not** a command; the
  entry point is `model-headroom` (hyphen).
- `model_headroom` 0.2.0 is installed **editable** from
  `/kaggle/working/model_headroom` into `/usr/local/lib/python3.12/dist-packages`.
- `capture-proxy` runs under the system Python 3.12, which has `torch 2.10.0+cu128`
  and `safetensors` — so the torch/safetensors blocker in `next-session.md:49-71`
  is already resolved for this install.
- `llama-server` is build 11200, commit 81bc6b83, 0.5.0-dev, CUDA-enabled.
  No EMO architecture string is present in `libllama.so`.
- **`model_headroom`'s `gguf` extra is declared but NOT installed** —
  `import gguf` → `ModuleNotFoundError`. Its GGUF code paths are therefore
  dead on this box until `pip install -e "/kaggle/working/model_headroom[gguf]"`
  runs. Verified, not inferred.
- `capture-proxy --status` → not running. `/root/captures` does not exist.
  No live upstream for EMO. No `.gguf` anywhere on the box
  (`find / -maxdepth 4 -name '*.gguf'` → empty).

## References

- `docs/headroom-capture-design.md` — ADR-001/ADR-002, the sidecar design
- `docs/example-headroom.md` — the end-to-end loop this ticket breaks at step 1
- `next-session.md:22-27` — the prior silent headroom failure (`error_body` tell)
- `PROJECT-ROADMAP.md:77-85` — the Headroom Model Integration gap list
- `AGENTS.md:185-202` — `CAPTURE_HEADROOM*` env vars and eligibility rules
