#!/usr/bin/env python3
"""Fetch contrastive prompt sets for the refusal-direction pipeline.

Harmful prompts come from the llm-attacks AdvBench dataset (MIT-licensed,
Andy Zou 2023) — the ``goal`` column of ``data/advbench/harmful_behaviors.csv``.
Harmless prompts come from the ``mlabonne/harmless_alpaca`` Hugging Face
dataset (the same source heretic uses internally).

Both sets are written as one-prompt-per-line ``.txt`` files into a cache
directory so the refusal CLI can consume them via ``--harmful-prompts`` /
``--harmless-prompts``. No attack code is imported or executed; only the
prompt strings are downloaded.

Usage:
    python scripts/fetch_refusal_prompts.py [--cache-dir DIR] [--n 100]
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import urllib.parse
import urllib.request
from pathlib import Path

HARMFUL_URL = (
    "https://raw.githubusercontent.com/llm-attacks/llm-attacks/main/"
    "data/advbench/harmful_behaviors.csv"
)
HARMLESS_DATASET = "mlabonne/harmless_alpaca"
HARMLESS_SPLIT = "train"


def fetch_harmful(n: int) -> list[str]:
    with urllib.request.urlopen(HARMFUL_URL) as resp:
        text = resp.read().decode("utf-8")
    reader = csv.DictReader(io.StringIO(text))
    prompts: list[str] = []
    for row in reader:
        goal = " ".join((row.get("goal") or "").split())
        if goal:
            prompts.append(goal)
        if len(prompts) >= n:
            break
    return prompts


def fetch_harmless(n: int) -> list[str]:
    prompts: list[str] = []
    offset = 0
    page_size = 100
    dataset = urllib.parse.quote(HARMLESS_DATASET, safe="")
    while len(prompts) < n:
        url = (
            "https://datasets-server.huggingface.co/rows"
            f"?dataset={dataset}&config=default&split={HARMLESS_SPLIT}"
            f"&offset={offset}&length={page_size}"
        )
        with urllib.request.urlopen(url) as resp:  # noqa: S310
            payload = json.loads(resp.read().decode("utf-8"))
        rows = payload.get("rows", []) or []
        if not rows:
            break
        for row in rows:
            text = " ".join((row.get("row", {}).get("text") or "").split())
            if text:
                prompts.append(text)
                if len(prompts) >= n:
                    break
        offset += page_size
    return prompts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-dir",
        default=os.path.expanduser("~/.cache/sia-hu/prompts"),
        help="directory to write prompt .txt files into",
    )
    parser.add_argument("--n", type=int, default=100, help="prompts per set")
    args = parser.parse_args()

    cache = Path(args.cache_dir)
    cache.mkdir(parents=True, exist_ok=True)

    harmful = fetch_harmful(args.n)
    harmless = fetch_harmless(args.n)

    harmful_path = cache / "harmful.txt"
    harmless_path = cache / "harmless.txt"
    harmful_path.write_text("\n".join(harmful) + "\n", encoding="utf-8")
    harmless_path.write_text("\n".join(harmless) + "\n", encoding="utf-8")

    print(f"Wrote {len(harmful)} harmful prompts -> {harmful_path}")
    print(f"Wrote {len(harmless)} harmless prompts -> {harmless_path}")


if __name__ == "__main__":
    main()
