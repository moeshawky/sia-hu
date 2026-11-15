# Handoff — ornith-1.5-9b headroom run (2026-11-12)

Production measurement run of the authority (`model_headroom`) against a real
9B multimodal checkpoint, on 2× T4. This document hands over what the run
found, what it could not measure, and the four code gaps it exposed.

**Checkpoint:** `/kaggle/input/models/moeshawky/ornith-1.5-9b/transformers/bf16/1`
`Qwen3_5ForConditionalGeneration`, bf16, 21 GB, 4 shards, 775 tensors.
Text: 32 layers, hidden 4096, intermediate 12288. Hybrid
`layer_types` = 3× `linear_attention` + 1× `full_attention`, repeating.
Plus a `model.visual.*` tower (27 blocks) and `mtp.layers.0.*` (1 layer).

**Artifacts:** `gdrive:ornith-1.5-9b-headroom-report-2026-11-12.zip`
(38,975 bytes, 21 files, sha256 `fdd8e8e13ac1fb373023e722ca6c7f98a07e377b4597259bf1c1b850fdb79c3d`)
— reports, per-signal tables, the dflash audit, the harness scripts, run logs.
Instance-local copies: `/kaggle/working/headroom-runs/ornith-fast/`.

**Repo state at handoff:** `781a79b`, pushed to the private Gogs `origin`.
No authority source was modified by the run.

---

## 1. What the run measured

| signal | status | source | note |
|---|---|---|---|
| R-03 effective_rank | computed | measured / verified | 20 real tensors, seed 42, 256 rows, parent shape declared |
| R-05 dead_neurons | computed | measured / verified | real forward pass, 32 text-stack MLPs, 24-token prompt |
| R-01 loss_saturation | skipped | unavailable | a weights-only checkpoint holds no training trajectory |
| R-02 growth_factor | skipped | unavailable | no compute-scaling points in a checkpoint |
| R-04 grad_weight_ratio | skipped | unavailable | no gradient evidence; the carrier emits a null gradient, never zeros |
| R-06 load_balance | single_expert | unavailable | dense checkpoint, no MoE routing |
| R-07 specialization | single_expert | unavailable | dense checkpoint, no expert weights |

Composite `0.2373`; measured-only `0.7594` at 31% coverage, range
`[0.237 .. 0.925]`; `decision_grade = False`. Architecture detected `DENSE`,
subtype `hybrid_linear_attention [linear_attn, full_attn, vision, mtp]`.

GPU hygiene: 0 MiB / no compute processes on both T4s before and after.

`dflash/` was never touched. `dflash_proof.json` records 9 open events,
5 distinct paths, none containing `/dflash/`, audited with a
`sys.addaudithook` on `os.open`; the four shard paths were taken from the
top-level `model.safetensors.index.json` only, with no recursive glob and no
`os.walk`. Note the audit hook sees Python-level opens only — safetensors
reads bypass it, and are excluded by construction rather than by observation.

---

## 2. Four gaps this run exposed

### GAP-1 (high) — the shipped CLI cannot ingest safetensors weight samples

`src/model_headroom/cli.py:191` — `_gguf_weight_samples` returns `None` for any
path that is not a `.gguf` file, so `weight_samples` is `None` for every
HuggingFace checkpoint and R-03 is **always** `skipped` in a CLI run.

Measured consequence on this checkpoint: the CLI reported composite `0.0609`
at 12.5% coverage with R-03 skipped; calling `run_analysis()` with samples
built from `safetensors.safe_open` produced composite `0.2373` at 31.25%
coverage with R-03 computed. Same checkpoint, different evidence — and the
CLI reports less without saying why.

Fix shape: add a safetensors sibling of `_gguf_weight_samples` that seeds row
selection, keeps the full column dimension, caps rows, and returns
`WeightSample` objects carrying the FULL `parent_shape` plus `truncated=True`.
The GGUF sampler's contract in `src/model_headroom/models/ternary_sampling.py`
is the template to match (seed 42 default, one tensor at a time, never the
whole file in RAM). Reuse `safe_open` so a 4.9 GB shard is never materialised
to read one matrix.

Until that lands, the only way to get R-03 out of a safetensors checkpoint is
to call `run_analysis(weight_samples=...)` directly. That is what the run did.

**Resolved (this change):** `src/model_headroom/models/safetensors_sampling.py`
adds the safetensors sibling, and `src/model_headroom/cli.py` gains
`_safetensors_weight_samples`, which the CLI now falls back to when the GGUF
sampler returns nothing. The contract matches the GGUF template: seeded row
selection (seed 42), the full column dimension kept, the row count capped, and
the FULL parent shape reported with `truncated=True`. One tensor is read at a
time through `safe_open`, so a multi-shard checkpoint is never materialised.

Measured on this checkpoint (2026-11-13, `--device cpu`): the CLI now computes
R-03 — layers 0/16/31 `mlp.gate_proj`, `parent_shape [12288, 4096]`,
`truncated=true`, coverage 2.08% per tensor, `comparable_to_full_matrix: false`,
composite 0.1815 at 18.75% coverage with `--weight-pairs-source-type measured`.
Before the change the same command reported R-03 `skipped`, composite 0.0609,
and no statement about what was missing.

### GAP-2 (high) — the dead-neuron criterion is wrong for signed activations

`src/model_headroom/signals/dead_neurons.py:31` — `dead_mask = neuron_means < threshold`.

The MLP intermediate captured at `mlp.down_proj` is **signed**. A neuron with
a strongly negative mean is counted as dead while being actively used. Measured
over all 32 layers of this checkpoint:

- flagged by the rule: **196,924**
- of those, negative-mean: **196,912 (99.994%)**
- never-active (max |value| == 0 across all 24 tokens): **0**, in every layer

So the headline `51.3% dead (6305/12288)` is a definition artifact, not 6305
unexcitable neurons. The tool's own recommendation ("consider adjusting
initialization or activation functions") is wrong on this evidence and must
not be quoted as a finding.

Fix shape — decide, do not silently change: either use `mean(|x|)` as the
activity statistic, or keep the signed mean and report the negative-mean count
alongside it, or add an explicit `never_active` count to
`DeadNeuronReport`. The signed breakdown script that produced these numbers
is in the zip as `dead_neuron_table.py`; it is the audit tool for whatever
you choose. Whichever way it goes, the report must keep the two populations
distinguishable — a neuron that is dead and a neuron that is negative are
different facts and the current contract collapses them.

**Resolved (this change):** `src/model_headroom/signals/dead_neurons.py` now
uses an activity-based dead criterion — a neuron is dead only when its maximum
absolute activation across all batch and token positions is exactly 0 (never
fires). The signed mean is no longer the dead criterion. `DeadNeuronReport`
carries separate `negative_mean_count`/`negative_mean_fraction` and
`never_active_count`/`never_active_fraction` statistics alongside
`dead_count`/`dead_fraction`.

> The Ornith regression figures (0 dead / 196,912 negative-mean / 0
> never-active across 32 layers) were measured once on a single 24-token
> prompt against this checkpoint. They are single-sample telemetry — not
> re-verified here, not generalizable across prompts, and not pinned in any
> test. Per AGENTS.md, single-sample inference telemetry is not quoted as a
> model-level claim; treat them as a one-off measurement, not a regression
> baseline.

### GAP-3 (medium) — `get_module_family` does not recognise `model.visual.*`

`src/model_headroom/models/loader.py:92` matches the substring `"vision"` (plus
`patch_embed`, `merger`). This checkpoint's vision prefix is `model.visual.*` —
no `"vision"` substring — so vision tensors classify as `attention`/`ffn` and
silently inflate those families' coverage.

Mitigation used by the run: `model.visual.*` and `mtp.layers.0.*` were
excluded from the sampled set, and the authority source was not patched. Any
future sampling that includes the vision tower will hit this until the
substring set is widened (e.g. also `"visual"`, `"visual.blocks"`, or the
`vision_config` presence in `config.json`).

**Resolved (this change):** `get_module_family` in
`src/model_headroom/models/loader.py` now matches `"visual"` alongside
`"vision"`, `"patch_embed"`, and `"merger"`. Before this,
`model.visual.blocks.0.attn.qkv.weight` classified as `attention` and
`model.visual.blocks.0.mlp.gate_proj.weight` as `ffn`, inflating those
families' coverage — while `model.visual.patch_embed.*` and
`model.visual.merger.*` did classify as `vision`, so the tower was only
partially recognised. The widening matches what `detect_architecture_details`
in the same module already did for the vision tower flag, which had been
recognising `"visual"` all along.

### GAP-4 (process) — derive layers from `layer_types`, never by hand

The first worker died with `KeyError: ...layers.15.linear_attn.in_proj_qkv.weight`.
`config.json` `text_config.layer_types[15] == "full_attention"`; the linear
attention tensors for layer 15 do not exist. Layer numbers must be read from
`layer_types`, not inferred from a "3 of every 4" pattern written by hand.
The corrected run substituted layer 2 for layer 15 in the linear-attention
group and kept layer 15 in the MLP group, where its tensors do exist.

**Not a repo-code gap (this change):** the trap lived in the run's harness, not
in the authority — `detect_architecture_details` reads `layer_types` from
`config.json` (`models/loader.py:200-205`) and derives nothing by hand. The
safetensors sampler added for GAP-1 samples MLP projections only, and those
exist for every layer index of this checkpoint (0–31, none missing), so the
`linear_attn` versus `self_attn` split cannot bite it: layer 15 has no
`linear_attn.in_proj_qkv.weight`, but it does have `mlp.gate_proj.weight`.

---

## 3. Reading the effective-rank numbers

The report's own `percentage_of_full_rank` (94–99.7%) is
`rank / min(tile_rows, tile_cols)` = `rank / 256` — of the **sampled tile**,
not of the 4096–12288-row parent. Parent-relative values are **5.88–6.23%**.
The human report already labels this `PARTIAL TILE` and states that inflation
against the full matrix is not measured by this tool, and that it is
comparable to full-matrix scores only above ~20% coverage. Per-tensor
coverage here was 2.08–6.25%.

Worst layer overall: `layers.19.self_attn.q_proj.weight`, rank 240.90 / 256.
Family worsts: `linear_attention` 95.0% (`layers.0.linear_attn.in_proj_qkv`),
`attention` 94.1% (`layers.19.self_attn.q_proj`), `ffn` 97.1%
(`layers.31.mlp.gate_proj`).

---

## 4. Reproducing the run

The harness is in the zip (`step1_effective_rank.py`, `step2_forward.py`,
`step3_report.py`, `dflash_audit.py`). It needs the repo's venv and
`PYTHONPATH=<repo>/src`, and it writes only outside the repo.

```bash
# forward pass: multimodal loader, weights split across both T4s
transformers.AutoModelForImageTextToText.from_pretrained(
    path, dtype=torch.bfloat16, device_map="auto",
    max_memory={0: "14GiB", 1: "14GiB", "cpu": "2GiB"})
```

Model load 120.4 s, forward pass 1.9 s. The prompt was:

> The Eiffel Tower was completed in 1889 and stands on the Champ de Mars in Paris, France.

24 tokens, hooked with `forward_pre_hook` on all 32
`model.language_model.layers.N.mlp.down_proj`, every tensor verified
`[1, 24, 12288]` before writing.

`fla`, `causal_conv1d` and `mamba_ssm` are not installed, but transformers
ships pure-torch `torch_chunk_gated_delta_rule` / `torch_recurrent_gated_delta_rule`
fallbacks, so the Gated DeltaNet layers run without them.

---

## 5. Left open on purpose

- **R-05 rests on one prompt, 24 tokens.** Per-neuron means over 24 tokens are
  a low-sample activity estimate. A neuron dormant for this prompt may be
  active for another. Averaging over several prompts is the obvious next step.
- **No gradient evidence anywhere.** R-08/R-09 gradient diagnostics were not
  run: no backward pass was performed and no weight/gradient pair artifact
  exists. The carrier's null-gradient contract means nothing was fabricated
  to fill that slot.
- **No untruncated full-matrix SVD**, so the R-03 inflation against the full
  matrix is still unmeasured.
- **The composite was not converted into a probability.** 0.2373 is a
  composite over mixed measured and neutral signals at 31% coverage; the
  measured-only 0.7594 with its range is the more honest of the two, and
  neither is a percentage of anything.
