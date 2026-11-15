"""Tests for model_headroom.acquire.activations."""

from __future__ import annotations

import dataclasses

import pytest
import torch

from model_headroom.acquire.activations import (
    FFNChannelStats,
    capture_ffn_channel_stats,
    dead_neuron_report_from_channel_stats,
    tokenize_prompt,
)
from model_headroom.models.types import MLPActivations
from model_headroom.signals.dead_neurons import analyze_dead_neurons

HIDDEN = 8
INTER = 16


class _FFN(torch.nn.Module):
    """A SwiGLU FFN: container output is the down-projection result."""

    def __init__(self, hidden: int, inter: int) -> None:
        super().__init__()
        self.gate_proj = torch.nn.Linear(hidden, inter, bias=False)
        self.up_proj = torch.nn.Linear(hidden, inter, bias=False)
        self.down_proj = torch.nn.Linear(inter, hidden, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(
            torch.nn.functional.silu(self.gate_proj(x)) * self.up_proj(x)
        )


class _FakeModel(torch.nn.Module):
    """A two-layer decoder whose only FFN containers are the _FFN modules."""

    def __init__(self, layers: int = 2) -> None:
        super().__init__()
        self.layers = torch.nn.ModuleList([_FFN(HIDDEN, INTER) for _ in range(layers)])

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        gen = torch.Generator().manual_seed(1234)
        x = torch.randn(input_ids.shape[0], input_ids.shape[1], HIDDEN, generator=gen)
        for ffn in self.layers:
            x = x + ffn(x)
        return x


def _reference_channel_stats(x: torch.Tensor) -> tuple[float, ...]:
    """Per-channel mean of |x| over all token positions (reference check)."""
    flat = x.reshape(-1, x.shape[-1])
    return tuple(float(v) for v in flat.abs().mean(dim=0).tolist())


def test_capture_stats_branches_and_channels() -> None:
    torch.manual_seed(0)
    model = _FakeModel()
    inputs = {"input_ids": torch.randint(0, 100, (2, 5))}
    stats = capture_ffn_channel_stats(model, inputs)

    assert set(stats) == {
        "layers.0:gate",
        "layers.0:up",
        "layers.0:down",
        "layers.1:gate",
        "layers.1:up",
        "layers.1:down",
    }
    gate = stats["layers.0:gate"]
    assert gate.branch == "gate"
    assert gate.channels == INTER
    assert gate.tokens_observed == 10
    assert len(gate.rms) == INTER
    assert len(gate.mean) == INTER
    down = stats["layers.0:down"]
    assert down.branch == "down"
    assert down.channels == HIDDEN
    assert down.tokens_observed == 10


def test_capture_stats_match_reference() -> None:
    torch.manual_seed(1)
    model = _FakeModel()
    inputs = {"input_ids": torch.randint(0, 100, (3, 7))}

    captured: dict[str, list[torch.Tensor]] = {"gate": [], "up": [], "down": []}

    def _collect(key: str):
        def hook(_m: object, _i: object, out: object) -> None:
            if torch.is_tensor(out):
                captured[key].append(out.detach().clone())

        return hook

    handles = [
        model.layers[0].gate_proj.register_forward_hook(_collect("gate")),
        model.layers[0].up_proj.register_forward_hook(_collect("up")),
        model.layers[0].register_forward_hook(_collect("down")),
    ]
    try:
        with torch.no_grad():
            model(**inputs)
    finally:
        for h in handles:
            h.remove()

    stats = capture_ffn_channel_stats(model, inputs)

    ref_gate = torch.cat(captured["gate"], dim=0).reshape(-1, INTER)
    gate = stats["layers.0:gate"]
    torch.testing.assert_close(
        torch.tensor(gate.abs_mean), ref_gate.abs().mean(dim=0), rtol=1e-5, atol=1e-6
    )
    torch.testing.assert_close(
        torch.tensor(gate.mean), ref_gate.mean(dim=0), rtol=1e-5, atol=1e-6
    )
    ref_rms = (ref_gate.float() ** 2).mean(dim=0).sqrt()
    torch.testing.assert_close(torch.tensor(gate.rms), ref_rms, rtol=1e-5, atol=1e-6)
    ref_nz = (ref_gate != 0).float().mean(dim=0)
    torch.testing.assert_close(
        torch.tensor(gate.nonzero_fraction), ref_nz, rtol=1e-5, atol=1e-6
    )
    torch.testing.assert_close(
        torch.tensor(gate.min), ref_gate.min(dim=0).values, rtol=0, atol=0
    )
    torch.testing.assert_close(
        torch.tensor(gate.max), ref_gate.max(dim=0).values, rtol=0, atol=0
    )

    ref_down = torch.cat(captured["down"], dim=0).reshape(-1, HIDDEN)
    down = stats["layers.0:down"]
    torch.testing.assert_close(
        torch.tensor(down.mean), ref_down.mean(dim=0), rtol=1e-5, atol=1e-6
    )


def test_container_output_is_hidden_not_intermediate() -> None:
    torch.manual_seed(2)
    model = _FakeModel()
    inputs = {"input_ids": torch.randint(0, 100, (2, 4))}
    stats = capture_ffn_channel_stats(model, inputs)
    assert stats["layers.0:down"].channels == HIDDEN
    assert stats["layers.0:gate"].channels == INTER
    assert INTER != HIDDEN


def test_empty_input_raises() -> None:
    torch.manual_seed(3)
    model = _FakeModel()
    inputs = {"input_ids": torch.randint(0, 100, (2, 0))}
    try:
        capture_ffn_channel_stats(model, inputs)
        raise AssertionError("expected ValueError for a forward pass with no tokens")
    except ValueError as e:
        assert "no tokens" in str(e)


def test_no_ffn_container_raises() -> None:
    class _Plain(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = torch.nn.Linear(HIDDEN, HIDDEN)

        def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
            x = torch.randn(input_ids.shape[0], input_ids.shape[1], HIDDEN)
            return self.linear(x)

    model = _Plain()
    inputs = {"input_ids": torch.randint(0, 100, (2, 4))}
    try:
        capture_ffn_channel_stats(model, inputs)
        raise AssertionError("expected ValueError when no FFN container exists")
    except ValueError as e:
        assert "no FFN container" in str(e)


def _stats_from_tensor(x: torch.Tensor, layer: str, branch: str) -> FFNChannelStats:
    """Build FFNChannelStats exactly as the streaming accumulator would."""
    flat = x.reshape(-1, x.shape[-1]).float()
    tokens = flat.shape[0]
    mean = flat.mean(dim=0)
    rms = (flat.pow(2).mean(dim=0)).sqrt()
    abs_mean = flat.abs().mean(dim=0)
    nz = (flat != 0).float().mean(dim=0)

    def _t(v: torch.Tensor) -> tuple[float, ...]:
        return tuple(float(i) for i in v.tolist())

    return FFNChannelStats(
        layer_name=layer,
        branch=branch,
        channels=x.shape[-1],
        tokens_observed=tokens,
        rms=_t(rms),
        mean=_t(mean),
        abs_mean=_t(abs_mean),
        nonzero_fraction=_t(nz),
        min=_t(flat.min(dim=0).values),
        max=_t(flat.max(dim=0).values),
    )


def test_dead_neuron_report_matches_signal() -> None:
    torch.manual_seed(4)
    x = torch.randn(4, 6, 10)
    x[:, :, 0] = 0.0
    x[:, :, 1] = 0.0
    x[:, :, 2] = -0.5

    signal = analyze_dead_neurons(
        MLPActivations(activations=x, layer_name="x"), return_indices=True
    )
    stats = _stats_from_tensor(x, "x", "gate")
    report = dead_neuron_report_from_channel_stats(stats)

    assert report.signal_status == "computed"
    assert report.total_neurons == signal.total_neurons == 10
    assert report.dead_count == signal.dead_count == 2
    assert report.dead_fraction == signal.dead_fraction
    assert report.negative_mean_count == signal.negative_mean_count
    assert report.negative_mean_fraction == signal.negative_mean_fraction
    assert report.never_active_count == signal.never_active_count
    assert report.never_active_fraction == signal.never_active_fraction
    assert report.activation_rms == pytest.approx(signal.activation_rms, rel=1e-5)
    assert report.mean_abs_activation == pytest.approx(
        signal.mean_abs_activation, rel=1e-5
    )
    assert report.variance == pytest.approx(signal.variance, rel=1e-3, abs=1e-6)
    assert report.cross_token_variance == pytest.approx(
        signal.cross_token_variance, rel=1e-3, abs=1e-6
    )
    assert report.near_zero_occupancy is None
    assert signal.near_zero_occupancy is not None


def test_tokenize_prompt_duck_typed() -> None:
    class _Tok:
        def __call__(self, text: str, **kwargs: object) -> dict[str, torch.Tensor]:
            return {"input_ids": torch.tensor([[1, 2, 3]])}

    inputs = tokenize_prompt(_Tok(), "hello")
    assert "input_ids" in inputs
    assert inputs["input_ids"].tolist() == [[1, 2, 3]]


def test_stats_frozen() -> None:
    s = _stats_from_tensor(torch.randn(2, 3, 4), "l", "up")
    try:
        s.branch = "gate"
        raise AssertionError("mutation of a frozen dataclass must raise")
    except dataclasses.FrozenInstanceError:
        pass
