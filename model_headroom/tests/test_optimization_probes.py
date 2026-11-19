import torch

from model_headroom.acquire.optimization import (
    GradientMeasurement,
    OptimizationProbeResult,
    run_optimization_probe,
)
from model_headroom.acquire.probes import ProbeBudget, ProbeSpec


def _linear_model():
    return torch.nn.Linear(4, 2)


def _loss_fn(model, inputs):
    x, y = inputs
    return torch.nn.functional.mse_loss(model(x), y)


def _spec(**overrides):
    budget = ProbeBudget(max_seconds=10.0, max_model_loads=1, max_backward_passes=1)
    defaults = {
        "probe_id": "opt1",
        "question": "are gradients present?",
        "hypothesis": "backward produces nonzero grads",
        "target_model": "synthetic_linear",
        "target_revision": None,
        "measurement": "grad_norm",
        "acquisition_mechanism": "pytorch_backward",
        "input_data": None,
        "input_provenance": None,
        "experimental_control": None,
        "intervention": None,
        "seed": 7,
        "budget": budget,
        "stopping_condition": "never",
        "expected_observations": None,
    }
    defaults.update(overrides)
    return ProbeSpec(**defaults)


def test_run_optimization_probe_measures_gradients():
    inputs = (torch.randn(8, 4), torch.randn(8, 2))
    result = run_optimization_probe(
        _spec(),
        model_factory=_linear_model,
        loss_fn=_loss_fn,
        inputs=inputs,
        steps=1,
        lr=1e-3,
    )
    assert isinstance(result, OptimizationProbeResult)
    assert result.probe_result.status == "completed"
    assert len(result.gradients) == 2  # weight + bias
    assert result.gradient_presence_proven is True
    for g in result.gradients:
        assert isinstance(g, GradientMeasurement)
        assert g.grad_source == "backward"
        assert g.finite is True
        assert g.grad_norm is not None
        assert g.grad_norm > 0.0


def test_run_optimization_probe_records_parameter_updates():
    inputs = (torch.randn(8, 4), torch.randn(8, 2))
    result = run_optimization_probe(
        _spec(),
        model_factory=_linear_model,
        loss_fn=_loss_fn,
        inputs=inputs,
        steps=1,
        lr=1e-2,
    )
    assert len(result.parameter_updates) == 2
    for name, before_norm, after_norm in result.parameter_updates:
        assert isinstance(name, str)
        assert before_norm >= 0.0
        assert after_norm >= 0.0


def test_run_optimization_probe_held_out_loss_before_after():
    inputs = (torch.randn(8, 4), torch.randn(8, 2))
    held_out = (torch.randn(4, 4), torch.randn(4, 2))
    result = run_optimization_probe(
        _spec(),
        model_factory=_linear_model,
        loss_fn=_loss_fn,
        inputs=inputs,
        held_out_inputs=held_out,
        steps=1,
        lr=1e-3,
    )
    assert result.held_out_loss_before is not None
    assert result.held_out_loss_after is not None
    assert isinstance(result.held_out_loss_before, float)
    assert isinstance(result.held_out_loss_after, float)


def test_run_optimization_probe_component_attribution():
    inputs = (torch.randn(8, 4), torch.randn(8, 2))
    result = run_optimization_probe(
        _spec(),
        model_factory=_linear_model,
        loss_fn=_loss_fn,
        inputs=inputs,
        components=("weight", "bias"),
        steps=1,
        lr=1e-3,
    )
    assert result.component_attribution is not None
    assert "weight" in result.component_attribution
    assert "bias" in result.component_attribution
    for v in result.component_attribution.values():
        assert v >= 0.0


def test_run_optimization_probe_budget_blocks_steps():
    inputs = (torch.randn(8, 4), torch.randn(8, 2))
    budget = ProbeBudget(max_seconds=10.0, max_model_loads=1, max_backward_passes=0)
    result = run_optimization_probe(
        _spec(budget=budget),
        model_factory=_linear_model,
        loss_fn=_loss_fn,
        inputs=inputs,
        steps=5,
        lr=1e-3,
    )
    # max_backward_passes=0 means no backward passes allowed -> max_steps clamped to 1 then budget blocks
    assert result.probe_result.status in ("completed", "inconclusive")
