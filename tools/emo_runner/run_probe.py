"""Run the sharded EMO load and emit real R-06 routing for three corpus lengths.

The three lengths exist to measure the fed-vs-starved property, not to produce
three numbers: if the never-fired expert set shrinks as the corpus grows, then a
balance verdict from a short corpus is an artifact of corpus length, not a
property of the router.

Usage:
    /root/.venvs/emo/bin/python -m tools.emo_runner.run_probe [--model PATH] [--out DIR]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from emo_runner.moe_shard import (
    collect_routing_counts,
    coverage,
    load_sharded_model,
    write_routing_artifact,
)

PROBE_TEXT = (
    "Unified headroom measurement keeps the scoring authority separate from the "
    "capture carrier. A mixture of experts routes each token to a small subset of "
    "experts, so the balance of expert usage is a measurable property of the model "
    "rather than an artefact of the runtime. The router selects eight of one hundred "
    "and twenty eight experts per token, and the distribution of those selections is "
    "what the load balance signal reports. Routing must be measured over a corpus "
    "long enough that rare experts are not merely unvisited. "
) * 24

CORPUS_LENGTHS = (256, 512, 1024)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/root/Emo_1b14b_1T_EmoAnnealed")
    parser.add_argument("--out", default="/kaggle/working/emo_probe/refactor")
    args = parser.parse_args()

    from transformers import AutoConfig, AutoTokenizer

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    started = time.time()

    config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    num_experts = config.num_experts
    top_k = config.num_experts_per_tok
    print(f"model: {args.model}", flush=True)
    print(
        f"config: layers={config.num_hidden_layers} experts={num_experts} "
        f"topk={top_k} hidden={config.hidden_size}",
        flush=True,
    )

    model, report = load_sharded_model(args.model, num_experts=num_experts)
    gib = 1024**3
    print(
        f"[{time.time() - started:.1f}s] loaded; experts_tracked="
        f"{report.experts_tracked} straddling={report.straddling}",
        flush=True,
    )
    for device, planned in sorted(report.planned_bytes.items()):
        print(f"  planned cuda:{device} = {planned / gib:.2f} GiB", flush=True)
    assert report.experts_whole, "expert placement invariant violated"

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    ids = tokenizer(PROBE_TEXT, return_tensors="pt").input_ids.to("cuda:0")
    print(f"corpus tokens available: {ids.shape[1]}", flush=True)

    results: dict[str, list[list[int]]] = {}
    never_fired: dict[str, set[tuple[int, int]]] = {}
    for length in CORPUS_LENGTHS:
        if length > ids.shape[1]:
            print(f"skip {length}: corpus only {ids.shape[1]} tokens", flush=True)
            continue
        counts = collect_routing_counts(
            model, ids[:, :length], num_experts=num_experts, top_k=top_k
        )
        results[str(length)] = counts
        never_fired[str(length)] = {
            (layer, expert)
            for layer, row in enumerate(counts)
            for expert, count in enumerate(row)
            if count == 0
        }
        fed_experts = len({e for row in counts for e, c in enumerate(row) if c > 0})
        write_routing_artifact(
            out / f"{length}.activations.json",
            counts=counts,
            num_experts=fed_experts,
            num_experts_architectural=num_experts,
            top_k=top_k,
            num_tokens=length,
            source=f"emo_real_text_{length}tok",
        )
        print(
            f"{length} tok: coverage={coverage(counts, num_experts):.4f} "
            f"never_fired={len(never_fired[str(length)])} "
            f"[{time.time() - started:.1f}s]",
            flush=True,
        )

    print("\nFED-VS-STARVED: never-fired set as the corpus grows", flush=True)
    previous: set[tuple[int, int]] | None = None
    for length in CORPUS_LENGTHS:
        key = str(length)
        if key not in never_fired:
            continue
        current = never_fired[key]
        if previous is None:
            print(f"  {length:5d}: never_fired={len(current)}", flush=True)
        else:
            newly_fed = len(previous - current)
            newly_dead = len(current - previous)
            verdict = "monotone" if newly_fed and not newly_dead else "CHECK"
            print(
                f"  {length:5d}: never_fired={len(current)} newly_fed={newly_fed} "
                f"newly_dead={newly_dead} ({verdict})",
                flush=True,
            )
        previous = current

    violations = 0
    keys = sorted(results, key=int)
    for smaller, larger in zip(keys, keys[1:]):
        for layer in range(len(results[smaller])):
            for expert in range(num_experts):
                if results[smaller][layer][expert] > 0 and results[larger][layer][expert] == 0:
                    violations += 1
    print(f"\nMONOTONE VIOLATIONS (fired at N but starved at 2N) = {violations}", flush=True)

    summary = {
        "model": args.model,
        "num_experts": num_experts,
        "top_k": top_k,
        "num_layers": config.num_hidden_layers,
        "corpus_lengths": [int(k) for k in keys],
        "coverage": {k: coverage(v, num_experts) for k, v in results.items()},
        "never_fired": {k: len(v) for k, v in never_fired.items()},
        "monotone_violations": violations,
        "experts_tracked": report.experts_tracked,
        "experts_straddling": report.straddling,
        "planned_bytes_per_device": {str(k): v for k, v in report.planned_bytes.items()},
        "elapsed_seconds": round(time.time() - started, 1),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    (out / "routing_counts.json").write_text(json.dumps(results))
    print(f"\nwrote {out}/summary.json and routing_counts.json", flush=True)
    print(f"TOTAL_ELAPSED {time.time() - started:.1f}s", flush=True)
    print("PROBE_COMPLETE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
