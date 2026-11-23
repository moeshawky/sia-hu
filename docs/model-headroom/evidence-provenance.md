# Evidence provenance — classes, attribution, and the "what this does NOT establish" discipline

**Purpose:** Define the provenance classes evidence can fall into, the rule that
evidence must be *attributable* (not merely tagged), the discipline of stating
what a piece of evidence does not establish, checkpoint revision identity, and
the calibration rule that separates a representation statistic from a prediction
of future learning. Cross-references `concepts/provenance.md` (the code-level
source_type / verification_status contract), `architecture/contract.md` (invariant
3: `null` ≠ healthy-zero), and `model-headroom/measurement-ontology.md`.

---

## 1. Code-level contract (frozen — do not mutate)

The authority's frozen contract carries the `SourceType` literal with **ten**
values (`src/model_headroom/models/types.py:14-25` at HEAD e80591d):

```
measured | synthetic | fixture | estimated | imported | unknown | inferred | unavailable | invalid | not_applicable
```

while the loader's acceptance set is the **six-value** frozenset
(`src/model_headroom/models/loader.py:446-448`):

```
measured | synthetic | fixture | estimated | imported | unknown
```

The loader boundary is the effective contract for external input; the wider
type literal covers internal states. The `VerificationStatus` literal carries
four values (`loader.py:449-451`):

```
verified | provenance_incomplete | unverified | unavailable
```

These are the values the signal code and the 25 frozen dataclasses accept. The
taxonomy below is an **evidence-level refinement** of this contract: it proposes
how the *acquisition layer* should classify evidence before it maps onto the
frozen values. It is PROPOSED. It changes no code value; it adds no enum member
without a versioned, additive contract change.

## 2. The seven provenance classes (PROPOSED)

| # | Class | Meaning | Maps to frozen `source_type` |
|---|-------|---------|------------------------------|
| 1 | **measured-direct** | Measured by this run, on this checkpoint: weights read from the checkpoint files on disk, activations from this run's forward pass, routing from this run's traffic | `measured` (+ `verified` when the provenance chain is complete) |
| 2 | **measured-transferred** | Measured on a *different* checkpoint or derivative: an upstream base model's training log, a sister checkpoint's eval, a quantization's source-model run | `measured` or `imported` with `provenance_incomplete`, applicability restricted to the checkpoint actually measured |
| 3 | **imported-published** | Imported from a published evidence artifact with a pinned identity: a Hub run record carrying `git_head`, `script_sha256`, `released_rev`, `manifest_sha256` | `imported` |
| 4 | **claimed** | Asserted by a producer without a reproducible measurement chain: a model card's loss curve, a README accuracy number | `estimated` or `imported` with `unverified`; diagnostic use only, never decision-grade |
| 5 | **estimated-derived** | Derived by computation from logged quantities: tokens = steps × batch × max_len (tokens were never logged), FLOPs from parameter counts | `estimated` |
| 6 | **synthetic** | Algorithmically generated test data | `synthetic` |
| 7 | **fixture** | Test fixture data | `fixture` |

**Absence is not an eighth class.** A missing input is handled by status
(`skipped`, `insufficient_data`, `no_gradient`, `unavailable`), never by a source
type. Invariant 3 stands: `null` means unproduced; it is never a healthy zero and
never a measurement.

### Why the split of `measured` into direct vs transferred

The frozen contract has one word for both "we measured this checkpoint" and "the
base model's card says this about a related checkpoint." Those are different
applicability claims. The measured instance: the operator publishes derivative
checkpoints (e.g. `RockAndEgg-313M-v2` derived from the `RockAndEgg-313M` line).
A training log published for the parent is `measured-transferred` for the
derivative — real measurement, wrong checkpoint. A model card's *claimed* loss is
`claimed` for every checkpoint, including the one that produced it, unless the
measurement chain is reproducible.

## 3. Attribution rules — attributable, not just tagged

A tag without an attribution chain is decoration. Every evidence record the
acquisition layer hands to a signal must carry:

1. **Checkpoint identity** — repo name **and** revision (see § 5). "RockAndEgg-313M-v2"
   is not an identity; "RockAndEgg-313M-v2 @ `7928cdf…`" is.
2. **Split** — which data split the number came from (train / eval / test). A
   number without a split label is unusable for R-01.
3. **Producer** — what produced it and its identity (script sha, engine version,
   run id). For published artifacts: `script_sha256`, `git_head`, engine
   versions from the record's `provenance` block.
4. **Derivation chain** — direct measurement, or derived (name the formula:
   `tokens = steps × batch × max_len`). Derived quantities carry the formula and
   the assumptions it bakes in (no padding variance, no truncation).
5. **Stated limitations** — duplicate points, single-sample telemetry, tile
   coverage, dequantization applied. Limitations are part of the record, not a
   footnote.

The `provenance_incomplete` verification status exists exactly for the case where
a record carries some but not all of (1)–(5).

## 4. "What this evidence does NOT establish" — the standing discipline

For every acquisition route, the report must be able to say what the evidence
does *not* support. The standing list, each item measured:

| Evidence | Does NOT establish |
|----------|--------------------|
| Published train-split loss log (e.g. `rae-005-research-evidence` `losses`, 25 points) | A **validation**-loss trajectory. R-01 wants (tokens_trained, **validation_loss**); train-split loss is a different quantity on a different split |
| Eval-split `composite` accuracy (6 points at steps 2400–4800) | A loss of any kind. Accuracy is not a loss; it cannot be fitted as a saturation curve for R-01 without an explicit, labeled conversion that is itself `estimated-derived` |
| Eval-split `pres_mse` | Generalization loss; it is a preservation-anchor loss on one eval split |
| Token counts computed as steps × batch 8 × max_len 128 | Logged token counts — tokens were never logged in the record; the product assumes exact utilization |
| The final duplicate history point (step 4800 appears twice) | A seventh independent evaluation; it is one evaluation recorded twice |
| A resume-from-step-2400 loss log (25 logged values at log_every=100) | Loss history before step 2400 |
| Single-sample inference telemetry (one 24-token prompt, one forward pass) | Any model-level activity claim (AGENTS.md BANNED list) |
| A 2.1% tile of a weight matrix | The full-matrix rank (tile statistics are tile-relative) |
| A representation statistic of any kind (rank, RMS, entropy, balance ratio) | Future learning gains. No representation statistic is a direct measurement of what a model *would* learn |
| A model card's claimed loss/accuracy | Anything, until the measurement chain is reproducible (`claimed` class) |
| One checkpoint at one size | Any scaling law (R-02 is structurally impossible from one checkpoint — see § 6) |

The rule of use: **a mapping that launders a limit into a clean curve is a bug
in the acquisition layer.** The published `rae-005` record is a *real* and
*usable* evidence source — with the above limits stated on the artifact it feeds,
not silently absorbed.

## 5. Checkpoint / model-revision identity

The same repo name on the Hub can point at different revisions. A measurement
attributed to a name without a revision is attributable to nothing.

**Measured instance (2026-11-15):** the local checkout of
`moeshawky/RockAndEgg-313M-v2` carries a `PROVENANCE.md` recording revision
`ac8bb272d74c1d7ed6879e53802d857fefb555ac`; the current Hub HEAD of that repo is
`7928cdfeda71fe3bf040cd584686d2e42cbe5c21`. These are different revisions under
the same name. Evidence taken from the local files is evidence about
`ac8bb272…`-era contents, not about the current Hub HEAD, and vice versa.

Rules:

1. **Revision-pinning is mandatory** for published-evidence acquisition. The Hub
   tree-listing route exposes per-revision file listings and file hashes; a
   download without a pinned revision is a race against a re-push.
2. **A local `PROVENANCE.md` is an operator-recorded fact, not a live check.** It
   records what the operator had at copy time. Verify against the Hub when the
   distinction matters (calibration, cross-run comparison, release).
3. **Absence of an artifact is a measured fact.** `moeshawky/RockAndEgg-313M-v2`
   contains no `trainer_state.json` (measured 2026-11-15). That is not an
   acquisition failure and not evidence that training did not happen; it is
   evidence that this revision publishes no training state. The report says
   "training state: not published at this revision," and stops.
4. **Cache identity includes the revision** (see `acquisition-architecture.md`
   § caching): (revision, file sha256, sampling params, software version).

## 6. R-02 — the anti-manufacturing rule

R-02 (growth factor) fits a quadratic over isoFLOP points: (compute_flops,
model_params, measured_loss) triplets across **different** compute/size points.
One checkpoint is one point. A single checkpoint is therefore **structurally
incapable** of producing R-02 evidence, and no resampling, interpolation, or
FLOP-arithmetic on one checkpoint can manufacture the missing points — doing so
would fabricate a scaling law. The only feasible routes are:

- published multi-run evidence (several checkpoints of one recipe, each with
  logged loss at its own compute point) — class 3, with applicability restricted
  to that recipe's family;
- operator-run isoFLOP training (expensive; out of scope for the authority, which
  does not train).

Until then R-02 stays `skipped`/`insufficient_data` and the report says so.

## 7. Scientific calibration — a statistic is not a prediction

1. **A representation statistic is never a direct measurement of future learning
   gains.** Rank, RMS, entropy, and balance ratios describe the checkpoint as
   stored or as it behaved on sampled inputs. What a model *would* learn is a
   property of (model, intervention, data, budget) — none of which a
   representation statistic observes.
2. **The composite is a diagnostic ordering, not a probability.** It is a
   prior-weighted expectation over signal scores with declared coverage; the
   handoff for the Ornith run states the load-bearing line: "neither is a
   percentage of anything." `--analysis-mode decision_grade` gates on measured
   coverage (≥70%), not on the score's calibration.
3. **The only evidence that can validate a headroom prediction is observed
   intervention response** (the fifth ontology class):
   baseline measurement → intervention (LoRA, quantization recovery, weight edit)
   → post measurement, recorded in `InterventionRecord`
   (`src/model_headroom/analysis/intervention_record.py` — IMPLEMENTED at HEAD).
4. **Predictive-validation protocol (PROPOSED):** on held-out checkpoints, record
   pre-intervention signals and the realized outcome, then measure whether the
   signal ordering predicted the outcome. Report base rates honestly: if every
   intervened checkpoint improved, the ordering is untested, and the report says
   "calibration: untested," not "calibration: good."

## Cross-references

- `model-headroom/PROJECT-ROADMAP.md` — master roadmap; defects; milestones M-pub, M-cal
- `model-headroom/measurement-ontology.md` — the five classes and the never-conflate rule
- `model-headroom/acquisition-architecture.md` — published-evidence discovery mechanics, caching identity
- `concepts/provenance.md` — the frozen source_type / verification_status contract
- `architecture/contract.md` — invariant 3 (`null` ≠ healthy-zero)
- `docs/guides/support-matrix.md` — what is verified against real models
