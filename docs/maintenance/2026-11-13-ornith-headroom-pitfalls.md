# Ticket: Ornith-1.5-9B headroom analysis — blocked on hardware and missing inputs

**Date:** 2026-11-13
**Model:** Ornith-1.5-9B (`Qwen3_5ForConditionalGeneration`, dense, bf16, 9.65B params, 4 safetensor shards, ~17.98 GiB)
**Path:** `/kaggle/input/models/moeshawky/ornith-1.5-9b/transformers/bf16/1`
**Status:** BLOCKED — move to another machine with native bf16 GPU

---

## Summary

Attempted to run a full model-headroom analysis on Ornith-1.5-9B. Only 2 of 7 signals could be computed (effective_rank from a 2.1% tile sample, dead_neurons from actual forward pass). Five signals require missing input data, and two gradient diagnostics (R-08/R-09) require weight pairs that are infeasible to generate at full-tensor scale. All GPU-based forward+backward and full-matrix SVD attempts OOM'd on the available Tesla T4 (sm_75) hardware.

---

## Pitfall 1 — T4 (sm_75) does not natively support bf16 computation

**Severity:** Critical — blocks all GPU-based computation

**Details:**
- Model precision: bfloat16 (per config.json `dtype: "bfloat16"`)
- GPU: 2× Tesla T4, compute capability **sm_75** (Turing)
- Native bf16 tensor core support requires **sm_80+** (Ampere/A100 or later)
- PyTorch emulates bf16 on sm_75, but emulation consumes more VRAM and is slower

**Evidence:**
```
nvidia-smi: Tesla T4, CUDA Version: 13.0, Compute M.: Default
torch version: 2.11.0+cu128
model config: "dtype": "bfloat16"
```

**Consequence:** Full-matrix SVD (effective_rank) and forward+backward (grad_weight_ratio, gradient diagnostics) OOM'd because the model shard (~9GB per GPU) plus emulation workspace plus activation/gradient buffers exceeded 15GB VRAM.

---

## Pitfall 2 — Forward+backward OOM on T4

**What was attempted:** GPU forward+backward to generate gradient/weight pairs for grad_weight_ratio signal.

**Result:**
```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 20.00 MiB.
GPU 1 has a total capacity of 14.56 GiB of which 8.81 MiB is free.
Of the allocated memory 14.42 GiB is allocated by PyTorch.
```

**Root cause:** The 9.65B parameter bf16 model split across 2 T4s (~9GB/GPU) leaves no room for activation buffers and gradient buffers needed by autograd. Even `seq_len=4` was too much.

**Attempted workarounds:**
- `device_map="cpu"` — model loads but forward+backward is extremely slow on CPU
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` — not applied (would help fragmentation but not capacity)

---

## Pitfall 3 — Full-matrix SVD on all weight matrices is prohibitively slow

**What was attempted:** `/tmp/full_rank_optimized.py` computes SVD effective rank on every 2-D weight matrix across 4 safetensor shards (~700 matrices, many [12288, 4096]).

**Status:** Running at 86% CPU, no completion after several minutes. SVD on a single [12288, 4096] matrix is O(min·max²) = expensive; hundreds of such matrices = hours of single-GPU compute.

**Consequence:** Cannot get full-matrix effective_rank in reasonable time on this hardware. The previous diagnostic run used a 2.1% tile sample (not comparable to full-matrix scores per the tool's own warning).

---

## Pitfall 4 — Missing input data for 5 of 7 signals

The model-headroom CLI requires external data files for most signals. None exist for this model on this machine:

| Signal | Required input | CLI flag | Status |
|--------|---------------|----------|--------|
| Loss Saturation | Loss curve JSON | `--loss-curve` | ❌ Not available — needs training run |
| Growth Factor | IsoFLOP points JSON | `--isoflop` | ❌ Not available — needs training run |
| Grad Weight Ratio | Gradient/weight pairs JSON | `--weight-pairs-file` | ❌ Full-tensor JSON would be ~70GB — infeasible |
| Dead Neurons | MLP activations JSON | `--mlp-activations-file` | ✅ Generated (see Pitfall 5) |
| Load Balance | MoE routing | N/A | Excluded — dense model |
| Specialization | Expert weights | N/A | Excluded — dense model |

**Root cause:** Headroom analysis is designed to ingest *training telemetry* (loss curves, gradients, activations), not just weights. Ornith-1.5-9B is a released checkpoint with no accompanying training data.

---

## Pitfall 5 — Partial success: dead_neurons computed

**What worked:** Generated MLP activations via GPU forward pass (seq_len=16) and saved to `/tmp/ornith_input_data/mlp_activations.json` (80MB, 32 gate_proj layers).

**Results from `/tmp/ornith_with_dead_neurons.json`:**
- dead_count: 0 / 12288 neurons
- dead_fraction: 0.0
- never_active_count: 0 / 12288
- never_active_fraction: 0.0
- negative_mean_count: 9447 (76.9%)
- activation_rms: 0.147
- mean_abs_activation: 0.110
- source_type: measured, verification_status: provenance_incomplete

**Effective rank (tile sample):** rank=247.8/256 (96.8%), coverage=2.1%, fixture/unverified — **not comparable to full-matrix score**.

---

## Pitfall 6 — Gradient diagnostics (R-08/R-09) blocked

**What was attempted:** `--gradient-diagnostics` flag.

**Result:** Not triggered — requires `--weight-pairs-file` to be present. Since full-tensor weight pairs are infeasible (~70GB JSON), R-08 (gradient coherence) and R-09 (gradient effective rank) cannot be computed.

**Provenance gap:** `model_revision=null`, `dataset_hash=null` — gradient provenance incomplete per decision-grade requirements.

---

## Pitfall 7 — Transformers/huggingface-hub version mismatch (environment issue)

**What happened:** Pre-existing container had `transformers 5.12.1` with `huggingface-hub 2.4.0` — incompatible (transformers requires hf-hub <2.0).

**Fix applied:** Upgraded `transformers` to 5.23.0 and `huggingface-hub` to 1.33.0. This resolved imports but was a detour caused by environment drift.

**Note:** This is unrelated to the model analysis but consumed time during setup.

---

## What a complete valid report requires

1. **GPU with native bf16 support** (sm_80+, e.g. A100, H100, RTX 30/40 series) — not T4
2. **Sufficient VRAM** for 18GB model + activation buffers + gradient buffers simultaneously (24GB+ per GPU recommended with tensor parallelism)
3. **Training data / dataloader** for Ornith-1.5-9B to generate:
   - Loss curve points (multiple checkpoints at different token counts)
   - IsoFLOP points
   - MLP activations from forward pass on real data
   - Gradient/weight pairs from forward+backward on real data
4. **Time** for full-matrix SVD on ~700 large weight matrices

---

## Files created during investigation (not committed)

| Path | Purpose |
|------|---------|
| `/tmp/ornith_input_data/mlp_activations.json` | 80MB — MLP activations from forward pass (dead_neurons input) |
| `/tmp/ornith_with_dead_neurons.json` | Model-headroom report with dead_neurons computed |
| `/tmp/ornith_headroom.json` | Baseline diagnostic report (2.1% tile effective_rank only) |
| `/tmp/full_rank_optimized.py` | Full-matrix SVD script (running in background, PID 63505) |
| `/tmp/compute_grad_ratio_gpu.py` | GPU gradient ratio script (OOM'd) |
| `/tmp/compute_grad_ratio_cpu.py` | CPU gradient ratio script (killed, too slow) |
| `/tmp/generate_input_data.py` | Original input generation script (OOM'd on backward) |

---

## Recommended next steps

1. Move to a GPU instance with native bf16 (sm_80+) and ≥24GB VRAM per GPU
2. Acquire training data or a dataloader for Ornith-1.5-9B
3. Generate loss curve, IsoFLOP, MLP activations, and weight pairs
4. Re-run model-headroom with `--decision-grade` and `--gradient-diagnostics`
5. Target ≥70% measurement coverage for decision-grade scoring
