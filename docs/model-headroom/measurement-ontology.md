# Measurement ontology — structure / activity / optimization / plasticity / observed intervention response

**Purpose:** Define the five classes of thing a headroom measurement can be, which
signal and which evidence live in each class, and the rule that these classes must
never be conflated. Cross-references `concepts/headroom-signals.md` (R-01..R-09
reference), `concepts/frozen-contracts.md` (frozen dataclasses), and
`model-headroom/evidence-provenance.md` (provenance classes and the
"what this evidence does NOT establish" discipline).

Status of the taxonomy itself: PROPOSED design layer (this document). The code
behind each mapping below carries its own IMPLEMENTED/VERIFIED status in the
master roadmap's integration matrix.

---

## 1. The five classes

| Class | What it is | Requires | Example evidence |
|-------|-----------|----------|------------------|
| **Structure** | Static facts about the weight matrices as stored | A checkpoint file, no execution | tensor shapes, effective rank of weight matrices, spectral entropy, expert count, quantization state, layout (prefixes, shard map) |
| **Activity** | What the model does on specific inputs | A forward pass over real or declared inputs | per-channel activations, routing histograms, never-active channels, per-layer output statistics |
| **Optimization** | The trajectory of a training process | Training telemetry (loss log, gradients, compute points) | loss curves, ∂L/W pairs, isoFLOP points, optimizer-state arithmetic |
| **Plasticity** | The capacity to change under future training | Nothing directly — inferable only | — (this is the quantity headroom is *about*; it is never itself measured) |
| **Observed intervention response** | What actually changed when an intervention was applied | A baseline→intervention→post measurement chain | `InterventionRecord` (`analysis/intervention_record.py`: intervention_type, checkpoint_hash, affected_parameters, measurement_methods) |

Plasticity is the only class with no direct measurement. Every headroom score is
a prior-weighted estimate *about* plasticity, assembled from structure, activity,
and optimization evidence with declared coverage. It is never a measurement of
plasticity, and a report that presents it as one has conflated classes.

## 2. The never-conflate rule

Evidence from one class cannot establish a claim in another. The load-bearing
prohibitions, each with the measured instance that motivates it:

| Prohibited conflation | Why it is false | Measured instance |
|-----------------------|-----------------|-------------------|
| Structure ⇒ optimization ("low rank ⇒ learns slowly") | Weight rank is a spectral fact about the stored matrix; it says nothing about the loss landscape a future optimizer would traverse | Ornith 2.1% tile rank reported beside a skipped R-01 — the report must not read the rank as a learning-speed claim |
| Structure ⇒ activity ("negative weight mean ⇒ dead neuron") | A stored value is not an activation | GAP-2 (2026-11-12 handoff): 99.994% of "dead" channels were negative-mean but active; fixed by the activity-based dead criterion (`signals/dead_neurons.py`) |
| Activity (single sample) ⇒ model-level activity | One prompt, 24 tokens, is low-sample telemetry | Ornith R-05 ran on one 24-token prompt; the handoff marks those figures non-generalizable; AGENTS.md bans quoting single-sample inference telemetry as model-level claims |
| Activity ⇒ plasticity ("dead neurons ⇒ wasted capacity") | A channel dormant on the sampled inputs may be active on others; dormancy is not a capacity verdict | Same single-prompt caveat |
| Optimization (train split) ⇒ validation trajectory | Train-split loss is not validation loss; accuracy is not a loss | The published `rae-005-research-evidence` run record carries a train-split loss log and eval-split `pres_mse`/`composite` — see `evidence-provenance.md` § limits |
| Claimed ⇒ measured ("the model card says loss X") | A claim without a reproducible measurement chain is the `claimed` provenance class | `evidence-provenance.md` § attribution rules |
| Sampled ⇒ exact ("a 2.1% tile is the matrix rank") | A tile's statistics are relative to the tile's own ceiling | `percentage_of_full_rank` is tile-relative by contract (`concepts/headroom-signals.md` R-03); the human report prints `PARTIAL TILE` |
| Representation statistic ⇒ future learning gain | No representation statistic is a direct measurement of future learning | Calibrated only by observed intervention response (§ 5); `evidence-provenance.md` § calibration |

## 3. Signal-to-ontology mapping

| Signal | Ontology class of its evidence | Notes |
|--------|-------------------------------|-------|
| R-01 loss saturation | Optimization | Needs (tokens_trained, validation_loss); train-split loss is NOT a substitute |
| R-02 growth factor | Optimization | Needs isoFLOP triplets; structurally impossible from one checkpoint (see `evidence-provenance.md`) |
| R-03 effective rank | Structure | Weight matrices only; sampled vs exact must stay labeled |
| R-04 gradient/weight ratio | Optimization | Needs real ∂L/∂W; a zero gradient is evidence of no training pressure, scored as `no_gradient`, never as a maximum (A-1) |
| R-05 dead neurons | Activity | Needs forward-pass activations; activity-based criterion (never fires = max |x| is 0) |
| R-06 load balance | Activity | Needs a routing histogram; SGLang-only source (hard constraint) |
| R-07 specialization | Structure (weak) / Activity (strong) | Pairwise expert cosine from weight vectors is a structure proxy; activation-weighted specialization needs inference |
| R-08 gradient coherence | Optimization (diagnostic) | Opt-in; never affects the composite |
| R-09 gradient effective rank | Optimization (diagnostic) | Opt-in; never affects the composite |
| `InterventionRecord` | Observed intervention response | The only evidence that can support "intervention X changed Y by Z" |

## 4. Dense as a single-expert MoE — explicit mathematical treatment

Operator rule (resolves A-4 in the master roadmap): **a dense FFN is conceptually
a one-expert MoE.** Honor the unification at the expert-abstraction level; do not
invent a normalization value to force numeric comparability; declare the
comparability limit instead.

### 4.1 The degenerate-case math

A router over E experts emits a distribution p over the experts.

- **E ≥ 2 (MoE):** entropy H = −Σ pᵢ log pᵢ is a real quantity in (0, log E].
  `load_balance` scores the balance ratio against this; the denominator is the
  **architectural** expert count (fix D4/D5, `concepts/headroom-signals.md` R-06),
  so a collapsed router scoring 1-of-8 experts reports `computed` with a low ratio,
  not `single_expert`.
- **E = 1 (dense):** the only possible distribution is p = (1.0). Then
  H = −1·log(1) = 0 exactly — `log(1) = 0` is well-defined, not a singularity.
  The one-expert router is simultaneously maximally imbalanced (every token goes
  to the only expert) and trivially "balanced" (there is nowhere else for tokens
  to go). Both readings are true; that is precisely why the numeric ratio is
  undefined-as-a-measurement and the honest value is `None`, not 0.0 or 1.0.
  Reporting `balance_ratio = 1.0` would manufacture "perfect balance" out of an
  architectural fact; reporting `0.0` would manufacture "total collapse." Both
  are fabricated measurements. The code emits `signal_status="single_expert"`
  with `balance_ratio=None` (`signals/load_balance.py:84`;
  `signals/specialization.py:60,82`) — IMPLEMENTED at HEAD.
- **Specialization with E = 1:** pairwise cosine over a single vector is
  undefined (no pair exists). Status `single_expert`, no score — same treatment.

### 4.2 The comparability limit

Because `single_expert` (and `not_applicable`) are excluded from both numerator
and denominator (`analysis/scoring.py:116-119` at HEAD e80591d):

- dense applicable weight = 0.20 + 0.20 + 0.15 + 0.15 + 0.10 = **0.80**
- MoE applicable weight = **1.00**

Dense and MoE composites are therefore computed over different denominators and
are **not numerically comparable**. Nothing in the current report states this —
that is defect D-7 in the master roadmap. The authorized treatment:

1. Keep the exclusion (no invented normalization value).
2. Declare the limit in the report: applicable weight, excluded signals and why,
   and the statement that dense vs MoE composites are not on a common scale.
3. Honor the unification where it is real: the expert abstraction treats a dense
   FFN as one expert, so the *status vocabulary* (`single_expert`) applies to
   dense, and the structural facts (expert count = 1) are recorded, not hidden.

### 4.3 Documentation drift on this point

`concepts/headroom-signals.md` and `architecture/authority-carrier.md` still say
dense models mark R-06/R-07 `not_applicable`; the code at HEAD emits
`single_expert`. The scoring treatment is identical (both statuses are excluded
from numerator and denominator), so scores do not change — but the docs and code
disagree on the label. Recorded in the master roadmap's drift section; corrected
by milestone M-doc, not here.

## 5. MoE routing instrumentation

- **Evidence required:** per-layer, per-expert integer token counts (the routing
  histogram) plus the architectural expert count.
- **Source (hard constraint):** SGLang with `--enable-return-routed-experts`.
  The carrier's routing sidecar decodes `sglext.routed_experts` (base64 int32,
  shape `LxK` = layers × routed-experts-per-token) to
  `{base_path}/routing/YYYY-MM-DD/{capture_id}.routing.json`
  (`src/capture_proxy/routing.py`). llama.cpp, vLLM, and ollama expose no routing
  visibility — report `routed_experts: false` for them, never mark true on a guess
  (`docs/guides/engine-selection.md`).
- **Derived quantities:** per-expert counts come from expanding the per-token
  per-layer K-expert records; load balance then runs on the counts with the
  architectural count as denominator.
- **Status:** carrier-side emission is IMPLEMENTED and was verified against
  SGLang 0.5.21 on 2026-10-09 *by the carrier*. Authority-side consumption of the
  routing artifact (histogram → R-06) has never been executed on this machine:
  PROPOSED.

## 6. Hybrid / recurrent architecture considerations

- **`layer_types` is the source of truth.** Layer numbers and per-layer tensor
  names must be derived from `config.json` `layer_types`, never by hand. Measured
  cost of the hand-inference trap: a worker died with
  `KeyError: ...layers.15.linear_attn.in_proj_qkv.weight` because layer 15 is
  `full_attention` and has no `linear_attn` tensors (GAP-4, 2026-11-12 handoff).
  `detect_architecture_details` reads `layer_types` correctly; run harnesses must
  not re-derive it.
- **Measured hybrid instance:** ornith-1.5-9b — 32 text layers, 3×
  `linear_attention` + 1× `full_attention` repeating (Gated DeltaNet), plus a
  `model.visual.*` tower and `mtp.layers.0.*`. The linear-attention layers run
  without `fla`/`causal_conv1d`/`mamba_ssm` via transformers' pure-torch
  fallbacks (`torch_chunk_gated_delta_rule` / `torch_recurrent_gated_delta_rule`)
  — IMPLEMENTED path, executed in the 2026-11-12 run.
- **Naming is not a mathematical contract.** `self_attn.q_proj` and
  `linear_attn.in_proj_qkv` are different operations with different shapes; the
  family adapters must dispatch on the operation the layer declares, not on a
  prefix assumption. Linear-attention tensor names are not covered by the CLI's
  safetensors sampling templates (see D-4, master roadmap).
- **Detection status:** hybrid subtype detection is IMPLEMENTED in
  `detect_architecture_details` and VERIFIED on ornith-1.5-9b (2026-11-12 run).
  It is misbehaving on the Qwen3 case — a model whose `layer_types` is 10×
  `full_attention` reports `hybrid_linear_attention [full_attn]` (defect D-5,
  CONFIRMED by execution 2026-11-15). A detector verified on one architecture is
  not verified on the next.
- **Encoder/classifier and multimodal:** `model.visual.*` recognition landed
  (GAP-3 resolved, `"visual"` substring); BERT-family naming is Phase 5
  (COMPLETED); DeBERTa naming is P6 UNKNOWN — implemented, never verified against
  a real model.

## Cross-references

- `model-headroom/PROJECT-ROADMAP.md` — master roadmap, defects D-1..D-7, milestones
- `model-headroom/acquisition-architecture.md` — family and backend adapters, activation extraction
- `model-headroom/evidence-provenance.md` — provenance classes, "what this does NOT establish"
- `concepts/headroom-signals.md` — per-signal reference
- `docs/maintenance/2026-11-12-handoff-ornith-1.5-9b.md` — measured GAP-1..GAP-4
- `docs/guides/engine-selection.md` — serving vs measurement hard constraints
