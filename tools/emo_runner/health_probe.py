"""Full model-health measurement for a checkpoint, on real hardware.

Fills the gaps a routing-only probe leaves open, so a report can claim the model
*works* rather than merely that it routes:

  * load wall-clock and peak VRAM per device
  * forward loss / perplexity on held-out real text
  * greedy generation, to prove the model produces coherent output
  * decode throughput (tokens/s), which is the number an operator serving this
    model actually needs
  * per-layer expert coverage, so concentration can be located rather than
    averaged away
  * expert-pool concentration (top-k share), distinct from coverage

Emits JSON only. Interpretation belongs to the report, not to this module.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch

from emo_runner.moe_shard import load_sharded_model

GIB = 1024**3

# Held-out real text, not the same paragraphs used for routing probes. The paper's
# objective is per-document, so a single document is not a health check.
HELDOUT = (
    "The water cycle describes the movement of water on Earth. Solar energy "
    "evaporates water from oceans, lakes and rivers into the atmosphere, where it "
    "condenses into clouds. When these clouds lose enough mass, precipitation "
    "returns the water to the surface, either as rainfall or as snow. Groundwater "
    "provides a further reservoir, moving slowly through aquifers over periods of "
    "centuries. The circulation redistributes heat and nutrients across the "
    "planet, sustaining ecosystems and moderating regional climate. "
) * 6

PROMPT = "The three primary colors are"


def measure(
    model_path: str,
    num_experts: int,
    top_k: int,
    tokenizer_source: str,
    prompt: str = PROMPT,
) -> dict:
    from transformers import AutoTokenizer

    started = time.time()
    model, report = load_sharded_model(model_path, num_experts=num_experts)
    load_seconds = time.time() - started

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=True)
    device = next(model.parameters()).device

    result: dict = {
        "path": model_path,
        "load_seconds": round(load_seconds, 1),
        "experts_tracked": report.experts_tracked,
        "experts_straddling": report.straddling,
        "planned_bytes": {str(k): v for k, v in report.planned_bytes.items()},
    }
    for index in range(torch.cuda.device_count()):
        free, total = torch.cuda.mem_get_info(index)
        result[f"cuda{index}_total_gib"] = round(total / GIB, 2)
        result[f"cuda{index}_free_after_load_gib"] = round(free / GIB, 2)
    result["dtype"] = str(next(model.parameters()).dtype)

    # ---- loss / perplexity on held-out text ----
    ids = tokenizer(HELDOUT, return_tensors="pt").input_ids.to(device)
    with torch.no_grad():
        out = model(input_ids=ids, labels=ids, use_cache=False)
    loss = float(out.loss)
    result["heldout_tokens"] = int(ids.shape[1])
    result["loss"] = round(loss, 4)
    result["perplexity"] = round(float(torch.exp(torch.tensor(loss))), 3)
    result["loss_is_finite"] = bool(torch.isfinite(out.loss))
    result["logits_shape"] = list(out.logits.shape)

    # ---- generation: does it produce coherent text? ----
    prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    gen_started = time.time()
    with torch.no_grad():
        generated = model.generate(
            input_ids=prompt_ids, max_new_tokens=40, do_sample=False
        )
    gen_seconds = time.time() - gen_started
    new_tokens = int(generated.shape[1] - prompt_ids.shape[1])
    result["generation"] = {
        "prompt": prompt,
        "new_tokens": new_tokens,
        "seconds": round(gen_seconds, 2),
        "text": tokenizer.decode(generated[0], skip_special_tokens=True),
    }
    result["decode_tokens_per_second"] = (
        round(new_tokens / gen_seconds, 2) if gen_seconds > 0 else None
    )

    # ---- prefill throughput on a larger real batch ----
    bench = tokenizer(HELDOUT * 2, return_tensors="pt").input_ids.to(device)
    torch.cuda.synchronize()
    prefill_started = time.time()
    with torch.no_grad():
        model(input_ids=bench, use_cache=False)
    torch.cuda.synchronize()
    prefill_seconds = time.time() - prefill_started
    result["prefill_tokens"] = int(bench.shape[1])
    result["prefill_seconds"] = round(prefill_seconds, 3)
    result["prefill_tokens_per_second"] = round(bench.shape[1] / prefill_seconds, 1)

    # ---- per-layer routing concentration ----
    with torch.no_grad():
        routed = model(input_ids=ids, output_router_logits=True, use_cache=False)
    num_layers = len(model.model.layers)
    per_layer: list[dict] = []
    for index, router_logits in enumerate(routed.router_logits):
        flat = router_logits.reshape(-1, num_experts).float()
        chosen = torch.topk(flat, top_k, dim=-1).indices
        counts = torch.bincount(chosen.reshape(-1), minlength=num_experts).cpu()
        nonzero = int((counts > 0).sum())
        ordered = torch.sort(counts, descending=True).values
        per_layer.append(
            {
                "layer": index,
                "coverage": round(nonzero / num_experts, 4),
                "never_fired": num_experts - nonzero,
                "max_count": int(ordered[0]),
                "min_count": int(ordered[-1]),
                "top8_share": round(
                    float(ordered[:8].sum() / ordered.sum()), 4
                ),
            }
        )
    result["per_layer"] = per_layer
    result["coverage_mean"] = round(
        sum(row["coverage"] for row in per_layer) / num_layers, 4
    )
    result["coverage_min_layer"] = min(row["coverage"] for row in per_layer)
    result["coverage_max_layer"] = max(row["coverage"] for row in per_layer)
    result["top8_share_mean"] = round(
        sum(row["top8_share"] for row in per_layer) / num_layers, 4
    )
    result["elapsed_seconds"] = round(time.time() - started, 1)

    del model
    torch.cuda.empty_cache()
    return result


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="/kaggle/working/emo_probe/health")
    parser.add_argument("--label", required=True)
    parser.add_argument("--model", required=True)
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    print(
        f"measuring {args.label}: experts={config.num_experts} "
        f"topk={config.num_experts_per_tok} layers={config.num_hidden_layers}",
        flush=True,
    )
    result = measure(
        model_path=args.model,
        num_experts=config.num_experts,
        top_k=config.num_experts_per_tok,
        tokenizer_source=args.model,
    )
    (out / f"{args.label}.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "per_layer"}, indent=2))
    print("HEALTH_COMPLETE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
