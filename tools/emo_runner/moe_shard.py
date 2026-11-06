"""Load a sharded MoE checkpoint and emit real expert-routing counts for R-06.

This is a third-party compatibility shim, not a scoring or capture component. It
imports neither `model_headroom` nor `capture_proxy`, so it cannot violate either
repo's contract: the authority never proxies, the carrier never scores, and the
carrier keeps zero torch imports on its serving path.

Placement policy
----------------
Experts dominate a MoE layer (measured on EMO: 97.9% of every layer), so
`device_map="auto"` fills devices in parameter order and cuts *mid-layer* — often
mid-expert. With top-8-of-128 routing, nearly every token then selects an expert
that straddles two devices, and every such hop crosses PCIe. On a PHB link
(no NVLink) that is the difference between a measurement and a stall.

So placement here is *layer-granular*: whole decoder layers to one device, and an
asserted invariant that no expert's parameters span two devices. The assertion is
not decorative — a 3-D fused or mis-indexed expert is exactly defect D1.

Why the forward hooks exist
---------------------------
Two measured failures on real hardware, both from tensors that are *derived once
per forward pass* rather than being weights:

1. `modeling_emo.py:156` — `(q * cos) + (rotate_half(q) * sin)`. RoPE
   `cos`/`sin` are computed once at model level on the first device, then every
   decoder layer consumes them. Layers on the second device die with
   "Expected all tensors to be on the same device".

2. `modeling_emo.py:498` — `residual + hidden_states`. Accelerate's
   `AlignDevicesHook` moves each *submodule* input, so attention itself ran fine
   on the second device, but the decoder layer's own Python-level `residual`
   still referenced the tensor handed in from the previous device.

Neither is a weight, so relocating them per layer is correct and costs only a
small copy of the RoPE tables.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch  # noqa: E402

__all__ = [
    "PlacementReport",
    "plan_layer_placement",
    "build_device_map",
    "install_device_hooks",
    "collect_routing_counts",
    "write_routing_artifact",
    "load_sharded_model",
    "coverage",
]


@dataclass(frozen=True)
class PlacementReport:
    """Measured outcome of a placement plan, per device."""

    device_map: dict[str, int]
    planned_bytes: dict[int, int]
    experts_tracked: int
    experts_expected: int
    straddling: int

    @property
    def experts_whole(self) -> bool:
        return self.straddling == 0


def plan_layer_placement(
    num_layers: int,
    num_experts: int,
    num_gpus: int = 2,
    layers_first_gpu: int | None = None,
) -> tuple[dict[str, int], int]:
    """Assign whole decoder layers to devices; return (map, layers_on_gpu0).

    Layer-granular on purpose. `layers_first_gpu` defaults to a balanced split
    (`num_layers // num_gpus`), which for EMO's 16 layers is exactly 8/8.
    """
    if num_gpus < 1:
        msg = f"num_gpus must be >= 1, got {num_gpus}"
        raise ValueError(msg)
    if num_gpus > 2:
        msg = f"two-device placement only, got num_gpus={num_gpus}"
        raise ValueError(msg)
    if num_layers < 1:
        msg = f"num_layers must be >= 1, got {num_layers}"
        raise ValueError(msg)
    split = num_layers // num_gpus if layers_first_gpu is None else layers_first_gpu
    split = max(0, min(split, num_layers))
    return ({str(i): (0 if i < split else 1) for i in range(num_layers)}, split)


def _expert_key(param_name: str) -> tuple[int, int] | None:
    """Extract (layer, expert) from `...mlp.experts.<e>.<proj>.weight`.

    Returns None for anything that is not a per-expert parameter. The index
    positions are asserted by tests, because an off-by-one here silently yields
    an empty expert set and makes a "0 straddling" assertion vacuous.
    """
    parts = param_name.split(".")
    if len(parts) > 5 and parts[3] == "mlp" and parts[4] == "experts":
        return int(parts[2]), int(parts[5])
    return None


def build_device_map(model: Any, num_experts: int) -> PlacementReport:
    """Derive an explicit per-module device map and verify the no-straddle invariant.

    Raises if any expert's parameters land on more than one device. Requires the
    model to be inspectable without materialising weights (meta device is fine).
    """
    num_layers = len(model.model.layers)
    layer_dev, _split = plan_layer_placement(num_layers, num_experts)

    device_map: dict[str, int] = {}
    for name, module in model.named_modules():
        if len(list(module.parameters(recurse=False))) > 0:
            if name.startswith("model.layers."):
                device_map[name] = layer_dev[name.split(".")[2]]
            else:
                device_map[name] = 0
    device_map.pop("", None)

    planned: dict[int, int] = {}
    home: dict[tuple[int, int], int] = {}
    for name, param in model.named_parameters():
        parts = name.split(".")
        top = name.rsplit(".", 1)[0] if "." in name else name
        candidates = [k for k in device_map if name == k or name.startswith(k + ".")]
        if not candidates:
            msg = f"no device assigned for parameter {name!r}"
            raise ValueError(msg)
        dev = device_map[max(candidates, key=len)]
        planned[dev] = planned.get(dev, 0) + param.numel() * 2
        key = _expert_key(name)
        if key is not None:
            if home.setdefault(key, dev) != dev:
                msg = f"expert {key} would straddle devices {home[key]} and {dev}"
                raise ValueError(msg)

    return PlacementReport(
        device_map=device_map,
        planned_bytes=planned,
        experts_tracked=len(home),
        experts_expected=num_layers * num_experts,
        straddling=0,
    )


def install_device_hooks(model: Any) -> int:
    """Relocate per-forward derived tensors to each decoder layer's device.

    Returns the number of layers hooked. Idempotent per layer: re-registering
    replaces nothing, so callers should invoke once after loading.
    """
    hooked = 0
    for layer in model.model.layers:
        dev = next(layer.parameters()).device

        def _relocate(_module, args, kwargs, _dev=dev):  # noqa: ANN001
            if args and torch.is_tensor(args[0]) and args[0].device != _dev:
                args = (args[0].to(_dev), *args[1:])
            for key in ("hidden_states", "attention_mask"):
                value = kwargs.get(key)
                if torch.is_tensor(value) and value.device != _dev:
                    kwargs[key] = value.to(_dev)
            pos = kwargs.get("position_embeddings")
            if pos is not None:
                kwargs["position_embeddings"] = tuple(
                    t.to(_dev) if torch.is_tensor(t) else t for t in pos
                )
            return args, kwargs

        layer.register_forward_pre_hook(_relocate, with_kwargs=True)
        hooked += 1
    return hooked


def load_sharded_model(
    model_path: str | Path,
    num_experts: int,
    dtype: torch.dtype = torch.float16,
) -> tuple[Any, PlacementReport]:
    """Load a MoE checkpoint across all visible GPUs with layer-granular placement.

    Meta-device build first: derives and verifies the placement without spending
    VRAM, so a bad plan fails before a 50 GiB read rather than after it.
    """
    from accelerate import init_empty_weights
    from transformers import AutoConfig, AutoModelForCausalLM

    model_path = str(model_path)
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    with init_empty_weights():
        meta = AutoModelForCausalLM.from_config(config, trust_remote_code=True)

    report = build_device_map(meta, num_experts)
    if report.experts_tracked != report.experts_expected:
        msg = (
            f"tracked {report.experts_tracked} experts, expected "
            f"{report.experts_expected}; the expert-name parser is wrong"
        )
        raise ValueError(msg)

    del meta
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        dtype=dtype,
        device_map=report.device_map,
        low_cpu_mem_usage=True,
    )
    install_device_hooks(model)
    return model, report


def collect_routing_counts(
    model: Any,
    input_ids: torch.Tensor,
    num_experts: int,
    top_k: int,
    chunk: int = 256,
) -> list[list[int]]:
    """Per-layer, per-expert routing counts over the given token ids.

    Routed-expert decisions only. Experts that never fired are omitted from the
    returned lists so a downstream producer can distinguish "unvisited" from
    "present but idle" — the distinction the fed-vs-starved guard depends on.
    """
    num_layers = len(model.model.layers)
    totals = [torch.zeros(num_experts, dtype=torch.long) for _ in range(num_layers)]
    length = input_ids.shape[1]
    for start in range(0, length, chunk):
        window = input_ids[:, start : start + chunk]
        with torch.no_grad():
            out = model(input_ids=window, output_router_logits=True, use_cache=False)
        for index, router_logits in enumerate(out.router_logits):
            flat = router_logits.reshape(-1, num_experts).float()
            chosen = torch.topk(flat, top_k, dim=-1).indices
            totals[index] += torch.bincount(
                chosen.reshape(-1), minlength=num_experts
            ).cpu()
        del out
    return [counts.tolist() for counts in totals]


def write_routing_artifact(
    path: str | Path,
    counts: list[list[int]],
    num_experts: int,
    num_experts_architectural: int,
    top_k: int,
    num_tokens: int,
    source: str,
) -> Path:
    """Write counts in the schema `load_activations_from_json` accepts.

    The authority reads `metadata.num_experts` (fed) and
    `metadata.num_experts_architectural` (declared). Sending the architectural
    value as the denominator is deliberate: a short corpus leaves experts
    unvisited, and dividing by the fed count would silently hide that.

    `num_experts` is the FED count — experts that actually fired at least once.
    Callers that only know the declared model width must derive it from
    ``counts``; passing the declared value here mislabels starved routers as
    fully exercised. ``num_experts_architectural`` is the declared width.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    layers = {
        str(layer_index): {
            str(expert): int(count)
            for expert, count in enumerate(row)
            if count > 0
        }
        for layer_index, row in enumerate(counts)
    }
    artifact = {
        "metadata": {
            "num_experts": num_experts,
            "num_experts_architectural": num_experts_architectural,
            "topk": top_k,
            "tokens": num_tokens,
            "source": source,
        },
        "layers": layers,
    }
    path.write_text(json.dumps(artifact, indent=2))
    return path


def coverage(counts: list[list[int]], num_experts: int) -> float:
    """Fraction of (layer, expert) slots that received any routing decision.

    Not part of `LoadBalanceReport`. Entropy-based `balance_ratio` moves ~0.09%
    across a 4x corpus while this moves ~9%, so coverage is the field that
    actually distinguishes a short corpus from a starved router.
    """
    slots = [count for row in counts for count in row]
    if not slots:
        return 0.0
    return sum(1 for count in slots if count > 0) / len(slots)
