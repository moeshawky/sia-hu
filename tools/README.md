# tools/

Workspace-level shims. Nothing here belongs to either product repo.

## Why this directory exists

The authority (`model_headroom/`) scores and never proxies. The carrier
(`capture-proxy/`) emits and never scores, and keeps **zero** torch imports on its
serving path so headroom-off mode stays pure (R-01). A layer-splitting shim for
third-party modeling code is neither: it is not scoring and it is not capture.

Putting it in either repo would bend a contract to house a workaround. So it lives
here, in the root repo, importing neither package.

## `emo_runner/`

Loads a sharded MoE checkpoint across multiple GPUs and emits real expert-routing
counts in the schema `load_activations_from_json` accepts.

| File | Purpose |
|---|---|
| `moe_shard.py` | Placement policy, the no-straddle invariant, per-layer device hooks, routing collection, artifact writer |
| `run_probe.py` | Runs the load and emits three corpus lengths so liveness can be measured against corpus length |
| `test_moe_shard.py` | Proves the invariant and the expert-name parser without touching a GPU |

**Environment.** The main environment cannot import `transformers`
(`huggingface_hub==2.0.0` against a `<2.0` pin) and is left that way on purpose.
Use the isolated env, which inherits the working torch and downloads only what is
broken:

```bash
uv venv --system-site-packages /root/.venvs/emo
uv pip install --python /root/.venvs/emo/bin/python --no-deps \
    transformers==4.57.1 "huggingface-hub>=0.34.0,<1.0" "tokenizers>=0.22.0,<=0.23.0"

PYTHONPATH=tools /root/.venvs/emo/bin/python tools/emo_runner/run_probe.py
/root/.venvs/emo/bin/python -m pytest tools/emo_runner/test_moe_shard.py -q
```

`transformers==4.57.1` is required, not preferred: EMO's `modeling_emo.py` imports
`OutputRecorder` from `transformers.utils.generic`, which 5.17.0 removed. A
zero-download probe showed `modeling_layers`, `masking_utils` and
`modeling_rope_utils` all still present in 5.17.0 — **module file present is not
symbol present.**

## Placement policy

Layer-granular, never `device_map="auto"`.

Experts are **97.9%** of every EMO layer, so `auto` fills devices in parameter
order and cuts *mid-layer* — often mid-expert. With top-8-of-128 routing nearly
every token then selects a straddling expert, and this box has a **PHB** link
between the two T4s, so no NVLink to absorb it.

Measured: 16 layers split 8/8 is exactly even. Layers 0–7 plus embed, `lm_head`
and `norm` on cuda:0 at 13.02 GiB; layers 8–15 on cuda:1 at 12.25 GiB; headroom
1.43 / 2.19 GiB; 2048 experts tracked, 0 straddling.

## Why the forward hooks exist

Two failures were hit on real hardware, and both come from tensors **derived once
per forward pass**, not from weights:

1. `modeling_emo.py:156` — `(q * cos) + (rotate_half(q) * sin)`. RoPE `cos`/`sin`
   are computed once at model level on the first device, then every decoder layer
   consumes them. Layers on the second device fail with *"Expected all tensors to
   be on the same device."*
2. `modeling_emo.py:498` — `residual + hidden_states`. Accelerate's
   `AlignDevicesHook` moves each **submodule** input, so attention itself ran fine
   on the second device, but the decoder layer's own Python-level `residual` still
   referenced the tensor handed in from the previous device.

Neither is a weight, so relocating per layer is correct and costs only a small
copy of the RoPE tables.

This is the concrete form of *"MoE models move layers and weights differently than
dense models."* It is not about expert tensors at all.

## Reading the result

`run_probe.py` deliberately emits three corpus lengths. If the never-fired expert
set shrinks as the corpus grows, then a balance verdict from a short corpus
measures corpus length, not the router. On EMO, 99 experts that read as dead at
256 tokens are alive at 512.

`LoadBalanceReport` has no coverage field, and `balance_ratio` cannot separate
"unvisited" from "collapsed" on its own. Use `coverage()` for that distinction.
