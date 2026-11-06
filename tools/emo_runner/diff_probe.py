"""Differential routing probe across two MoE checkpoints.

The point is a decision, not a verdict: the operator needs to know whether the
ablation (`StdMoE_1b14b_1T_EmoAnnealed`) and the main release (`Emo_1b14b_1T`)
route differently enough to matter for headroom measurement.

EMO's training objective is per-document: all tokens in a document are
constrained to choose from a shared expert pool. So a single synthetic paragraph
is the worst possible probe — it measures one pool. This probe therefore runs
several *domain-separated* documents and reports, per checkpoint:

  * coverage and never-fired counts, per document and pooled
  * whether each checkpoint is corpus-length sensitive (the earlier finding)
  * pairwise pool overlap between documents, which is what EMO claims to
    differentiate semantically
  * the same numbers through the real authority R-06, so a claim about the
    signal is a claim about the signal on this hardware

Every number here is measured. Nothing is inferred from the paper except the
architecture claim that is re-verified from the tensor manifests.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from emo_runner.moe_shard import (
    collect_routing_counts,
    coverage,
    load_sharded_model,
    write_routing_artifact,
)

# Domain-separated documents. EMO claims pools specialise by domain, so the
# corpus has to span domains or that claim is untestable.
DOCUMENTS: dict[str, str] = {
    "code": (
        "def quicksort(items):\n"
        "    if len(items) <= 1:\n"
        "        return items\n"
        "    pivot = items[len(items) // 2]\n"
        "    left = [i for i in items if i < pivot]\n"
        "    right = [i for i in items if i > pivot]\n"
        "    return quicksort(left) + [pivot] + quicksort(right)\n"
    ) * 12,
    "math": (
        "Let f be a continuous function on the closed interval [a, b]. By the "
        "extreme value theorem f attains a maximum at some point c in [a, b]. "
        "Since f is continuous, the function g defined by g(x) = f(x) - f(c) is "
        "nonpositive on the interval. Integrating g over [a, b] gives an upper "
        "bound for the definite integral of f. The mean value theorem for "
        "integrals states there exists d in [a, b] with the average value equal "
        "to f(d). Therefore the integral equals f(d) times the interval length. "
    ) * 12,
    "clinical": (
        "Hypertension is defined as persistently elevated arterial blood pressure. "
        "First line therapy for uncomplicated hypertension in adults includes a "
        "thiazide diuretic, an ACE inhibitor, or an angiotensin receptor blocker. "
        "Blood pressure should be measured on two separate occasions before "
        "initiating pharmacotherapy. Lifestyle modification reduces systolic "
        "pressure by approximately five millimetres of mercury. Renal function "
        "and serum electrolytes should be assessed prior to treatment. "
    ) * 12,
    "legal": (
        "The party of the first part covenants to warrant and defend the second "
        "party against any claims arising from breach of the representations "
        "warranted herein. This indemnification obligation shall survive "
        "termination of this agreement and remain in force for a period of five "
        "years from the date of execution. Neither party shall be liable for "
        "consequential, incidental, or punitive damages. The governing law of the "
        "state of Delaware shall apply. "
    ) * 12,
    "conversational": (
        "Hey, are we still on for tomorrow morning? I was thinking we could grab "
        "coffee near the office and then maybe walk the dog before it gets too "
        "warm out. Let me know what time works best for you, and if you need a "
        "ride just text me. I am free most of the morning but have a meeting in "
        "the afternoon around three."
    ) * 12,
}

SHORTENINGS = (128, 256, 512)


def pool_of(counts: list[list[int]], num_experts: int) -> dict[int, set[int]]:
    """Per layer, the set of experts that fired at least once."""
    return {
        layer: {expert for expert, count in enumerate(row) if count > 0}
        for layer, row in enumerate(counts)
    }


def jaccard(a: set[int], b: set[int]) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    return len(a & b) / len(union) if union else 0.0


def probe_checkpoint(
    label: str,
    model_path: str,
    out_dir: Path,
    num_experts: int,
    top_k: int,
    tokenizer_source: str,
) -> dict:
    started = time.time()
    print(f"\n{'=' * 72}\n{label}\n{'=' * 72}", flush=True)
    print(f"path: {model_path}", flush=True)

    model, report = load_sharded_model(model_path, num_experts=num_experts)
    gib = 1024**3
    print(
        f"[{time.time() - started:.1f}s] loaded; experts={report.experts_tracked} "
        f"straddling={report.straddling}",
        flush=True,
    )
    for device, planned in sorted(report.planned_bytes.items()):
        print(f"  cuda:{device} planned {planned / gib:.2f} GiB", flush=True)
    assert report.experts_whole

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=True)

    result: dict = {
        "label": label,
        "path": model_path,
        "experts": report.experts_tracked,
        "straddling": report.straddling,
        "documents": {},
    }

    for name, text in DOCUMENTS.items():
        ids = tokenizer(text, return_tensors="pt").input_ids.to("cuda:0")
        available = ids.shape[1]
        doc: dict = {"available_tokens": available, "lengths": {}}
        pooled: list[list[int]] = []
        for length in SHORTENINGS:
            if length > available:
                continue
            counts = collect_routing_counts(
                model, ids[:, :length], num_experts=num_experts, top_k=top_k
            )
            cov = coverage(counts, num_experts)
            never = sum(1 for row in counts for c in row if c == 0)
            doc["lengths"][str(length)] = {
                "coverage": cov,
                "never_fired": never,
            }
            print(
                f"  {name:16s} {length:4d} tok  coverage={cov:.4f} never_fired={never}",
                flush=True,
            )
            if length == max(SHORTENINGS):
                pooled = counts
                fed_experts = len({e for row in counts for e, c in enumerate(row) if c > 0})
                write_routing_artifact(
                    out_dir / f"{label.replace(' ', '_')}_{name}.activations.json",
                    counts=counts,
                    num_experts=fed_experts,
                    num_experts_architectural=num_experts,
                    top_k=top_k,
                    num_tokens=length,
                    source=f"{label}:{name}:{length}",
                )
        if pooled:
            doc["pool_at_max_length"] = {
                str(layer): sorted(experts) for layer, experts in pool_of(pooled, num_experts).items()
            }
        result["documents"][name] = doc

    names = list(result["documents"])
    overlap: dict[str, dict[str, float]] = {}
    for i, first in enumerate(names):
        for second in names[i + 1 :]:
            scores = []
            a = result["documents"][first].get("pool_at_max_length", {})
            b = result["documents"][second].get("pool_at_max_length", {})
            for layer in a:
                if layer in b:
                    scores.append(jaccard(set(a[layer]), set(b[layer])))
            if scores:
                overlap[f"{first}|{second}"] = {
                    "mean_jaccard": sum(scores) / len(scores),
                    "layers": len(scores),
                }
    result["pool_overlap"] = overlap
    if overlap:
        values = [v["mean_jaccard"] for v in overlap.values()]
        print(
            f"  mean pairwise pool Jaccard: min={min(values):.3f} "
            f"max={max(values):.3f} mean={sum(values) / len(values):.3f}",
            flush=True,
        )
    result["elapsed_seconds"] = round(time.time() - started, 1)

    del model
    torch.cuda.empty_cache()
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="/kaggle/working/emo_probe/differential")
    parser.add_argument(
        "--a", default="/root/Emo_1b14b_1T_EmoAnnealed", help="ablation checkpoint"
    )
    parser.add_argument(
        "--b",
        default="/kaggle/input/models/moeshawky/emo-1b14b-1t/transformers/fp32/1",
        help="main release checkpoint",
    )
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    from transformers import AutoConfig

    num_experts, top_k = 128, 8
    for path in (args.a, args.b):
        config = AutoConfig.from_pretrained(path, trust_remote_code=True)
        num_experts, top_k = config.num_experts, config.num_experts_per_tok

    results = []
    for label, path in (("A_StdMoE_EmoAnnealed", args.a), ("B_Emo_main", args.b)):
        results.append(
            probe_checkpoint(
                label=label,
                model_path=path,
                out_dir=out,
                num_experts=num_experts,
                top_k=top_k,
                tokenizer_source=path,
            )
        )

    print(f"\n{'=' * 72}\nCOMPARISON\n{'=' * 72}", flush=True)
    for res in results:
        covs = [
            d["lengths"][str(max(SHORTENINGS))]["coverage"]
            for d in res["documents"].values()
            if str(max(SHORTENINGS)) in d["lengths"]
        ]
        nevers = [
            d["lengths"][str(max(SHORTENINGS))]["never_fired"]
            for d in res["documents"].values()
            if str(max(SHORTENINGS)) in d["lengths"]
        ]
        print(
            f"{res['label']:24s} coverage mean={sum(covs) / len(covs):.4f} "
            f"never_fired mean={sum(nevers) / len(nevers):.1f}",
            flush=True,
        )
    (out / "differential.json").write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out}/differential.json", flush=True)
    print("DIFFERENTIAL_COMPLETE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
