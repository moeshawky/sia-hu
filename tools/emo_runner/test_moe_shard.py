"""Prove the no-straddle invariant and the expert-name parser before spending VRAM.

The parser is the load-bearing part: if `_expert_key` returns None for every
parameter, `build_device_map` reports 0 straddling over an *empty* expert set and
the assertion passes without testing anything. These tests fail on that exact
regression.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from emo_runner.moe_shard import (  # noqa: E402
    _expert_key,
    build_device_map,
    coverage,
    plan_layer_placement,
    write_routing_artifact,
)


def test_expert_key_parses_real_emo_naming() -> None:
    name = "model.layers.7.mlp.experts.63.gate_proj.weight"
    assert _expert_key(name) == (7, 63)


def test_expert_key_ignores_non_expert_parameters() -> None:
    for name in (
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.mlp.gate.weight",
        "model.embed_tokens.weight",
        "lm_head.weight",
        "model.norm.weight",
    ):
        assert _expert_key(name) is None, name


def test_parser_cannot_silently_return_nothing_for_experts() -> None:
    """Regression guard: the off-by-one that made the straddle check vacuous."""
    names = [
        f"model.layers.{layer}.mlp.experts.{expert}.{proj}.weight"
        for layer in range(2)
        for expert in range(3)
        for proj in ("gate_proj", "up_proj", "down_proj")
    ]
    parsed = [name for name in names if _expert_key(name) is not None]
    assert len(parsed) == len(names) == 18


class _Expert(torch.nn.Module):
    def __init__(self, hidden: int = 8, inter: int = 4) -> None:
        super().__init__()
        self.gate_proj = torch.nn.Linear(hidden, inter, bias=False, dtype=torch.float16)
        self.up_proj = torch.nn.Linear(hidden, inter, bias=False, dtype=torch.float16)
        self.down_proj = torch.nn.Linear(inter, hidden, bias=False, dtype=torch.float16)


class _Mlp(torch.nn.Module):
    def __init__(self, experts: int) -> None:
        super().__init__()
        self.gate = torch.nn.Linear(8, experts, bias=False, dtype=torch.float16)
        self.experts = torch.nn.ModuleList([_Expert() for _ in range(experts)])


class _Layer(torch.nn.Module):
    def __init__(self, experts: int) -> None:
        super().__init__()
        self.self_attn = torch.nn.Linear(8, 8, bias=False, dtype=torch.float16)
        self.mlp = _Mlp(experts)


class _Inner(torch.nn.Module):
    def __init__(self, layers: int, experts: int) -> None:
        super().__init__()
        self.embed_tokens = torch.nn.Embedding(32, 8, dtype=torch.float16)
        self.layers = torch.nn.ModuleList([_Layer(experts) for _ in range(layers)])
        self.norm = torch.nn.LayerNorm(8, dtype=torch.float16)


class _Fake(torch.nn.Module):
    def __init__(self, layers: int, experts: int) -> None:
        super().__init__()
        self.model = _Inner(layers, experts)
        self.lm_head = torch.nn.Linear(8, 32, bias=False, dtype=torch.float16)


def test_placement_tracks_and_places_every_expert() -> None:
    model = _Fake(layers=4, experts=6)
    report = build_device_map(model, num_experts=6)
    assert report.experts_tracked == report.experts_expected == 24
    assert report.straddling == 0
    assert report.experts_whole
    assert set(report.device_map.values()) == {0, 1}


def test_placement_is_layer_granular() -> None:
    model = _Fake(layers=4, experts=2)
    report = build_device_map(model, num_experts=2)
    for layer_index in range(4):
        devs = {
            report.device_map[name]
            for name in report.device_map
            if name.startswith(f"model.layers.{layer_index}.")
        }
        assert len(devs) == 1, f"layer {layer_index} split across {devs}"


def test_plan_splits_evenly_for_sixteen_layers() -> None:
    mapping, split = plan_layer_placement(16, 128, num_gpus=2)
    assert split == 8
    assert [mapping[str(i)] for i in range(16)] == [0] * 8 + [1] * 8


@pytest.mark.parametrize("bad", [0, -1])
def test_plan_rejects_nonpositive_gpu_count(bad: int) -> None:
    with pytest.raises(ValueError, match="num_gpus"):
        plan_layer_placement(16, 128, num_gpus=bad)


def test_plan_rejects_nonpositive_layer_count() -> None:
    with pytest.raises(ValueError, match="num_layers"):
        plan_layer_placement(0, 128)


def test_coverage_separates_short_corpus_from_starved_router() -> None:
    starved = [[10, 0, 0, 0], [10, 0, 0, 0]]
    broad = [[10, 10, 10, 10], [10, 10, 10, 10]]
    assert coverage(starved, 4) == 0.25
    assert coverage(broad, 4) == 1.0


def test_artifact_denominator_is_architectural() -> None:
    path = write_routing_artifact(
        Path("/tmp/moe_shard_test.json"),
        counts=[[3, 0, 1, 0], [2, 0, 0, 0]],
        num_experts=2,
        num_experts_architectural=4,
        top_k=2,
        num_tokens=8,
        source="unit-test",
    )
    import json

    data = json.loads(path.read_text())
    assert data["metadata"]["num_experts"] == 2
    assert data["metadata"]["num_experts_architectural"] == 4
    assert data["layers"]["0"] == {"0": 3, "2": 1}
    assert "1" not in data["layers"]["0"], "unvisited experts must be omitted, not zeroed"
