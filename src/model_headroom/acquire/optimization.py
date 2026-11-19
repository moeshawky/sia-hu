"""Optimization probes: bounded PyTorch experiments measuring gradients and
parameter updates on disposable experimental copies.

All mutation happens on copies constructed in memory; the source checkpoint is
never loaded or modified. Gradient presence alone does not prove that every
intended computational path contributed — this module reports what is actually
measured (grad_norm, grad_weight_ratio per parameter, component attribution)
and marks unattributed paths as unavailable.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import monotonic
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from collections.abc import Callable

    import torch

from model_headroom.acquire.probes import ProbeResult, ProbeSpec

GradientSource = Literal["backward", "autograd", "finite_difference", "none"]


@dataclass(frozen=True)
class GradientMeasurement:
    """One parameter's gradient measurement from a backward pass."""

    parameter_name: str
    grad_norm: float | None
    weight_norm: float | None
    grad_weight_ratio: float | None
    grad_source: GradientSource
    finite: bool


@dataclass(frozen=True)
class OptimizationProbeResult:
    """Outcome of one optimization probe."""

    probe_result: ProbeResult
    gradients: tuple[GradientMeasurement, ...]
    parameter_updates: tuple[
        tuple[str, float, float], ...
    ]  # (name, before_norm, after_norm)
    held_out_loss_before: float | None
    held_out_loss_after: float | None
    component_attribution: dict[str, float] | None
    gradient_presence_proven: bool


def _named_parameters(model: torch.nn.Module) -> dict[str, torch.nn.Parameter]:
    return dict(model.named_parameters())


def _copy_model(model: torch.nn.Module) -> torch.nn.Module:
    import copy

    return copy.deepcopy(model)


def _measure_gradients(model: torch.nn.Module) -> tuple[GradientMeasurement, ...]:
    import torch

    measurements: list[GradientMeasurement] = []
    for name, param in _named_parameters(model).items():
        if param.grad is None:
            measurements.append(
                GradientMeasurement(
                    parameter_name=name,
                    grad_norm=None,
                    weight_norm=float(param.detach().norm().item())
                    if param.detach().numel()
                    else 0.0,
                    grad_weight_ratio=None,
                    grad_source="none",
                    finite=bool(torch.isfinite(param.detach()).all().item()),
                )
            )
            continue
        grad = param.grad.detach()
        grad_norm = float(grad.norm().item())
        weight = param.detach()
        weight_norm = float(weight.norm().item())
        ratio = grad_norm / weight_norm if weight_norm > 0 else None
        measurements.append(
            GradientMeasurement(
                parameter_name=name,
                grad_norm=grad_norm,
                weight_norm=weight_norm,
                grad_weight_ratio=ratio,
                grad_source="backward",
                finite=bool(torch.isfinite(grad).all().item()),
            )
        )
    return tuple(measurements)


def _parameter_update_norms(
    before: torch.nn.Module, after: torch.nn.Module
) -> tuple[tuple[str, float, float], ...]:
    updates: list[tuple[str, float, float]] = []
    before_params = _named_parameters(before)
    after_params = _named_parameters(after)
    for name in before_params:
        if name not in after_params:
            continue
        b = before_params[name].detach()
        a = after_params[name].detach()
        updates.append((name, float(b.norm().item()), float(a.norm().item())))
    return tuple(updates)


def run_optimization_probe(
    spec: ProbeSpec,
    model_factory: Callable[[], torch.nn.Module],
    loss_fn: Callable[[torch.nn.Module, Any], torch.Tensor],
    inputs: Any,
    held_out_inputs: Any | None = None,
    components: tuple[str, ...] = (),
    steps: int = 1,
    lr: float = 1e-3,
) -> OptimizationProbeResult:
    """Run a bounded optimization probe on a disposable model copy.

    Parameters:
        spec: The probe contract (budget, seed, target revision, etc.).
        model_factory: Callable returning a fresh torch.nn.Module (the experimental copy).
        loss_fn: Callable(model, inputs) -> scalar loss tensor.
        inputs: Training inputs for the backward pass.
        held_out_inputs: Optional held-out inputs for before/after loss comparison.
        components: Parameter name prefixes for component attribution.
        steps: Number of optimization steps (bounded by spec.budget).
        lr: Learning rate for the update step.

    The source model is never loaded. model_factory constructs the experimental
    copy in memory. Gradient measurements are taken after backward; parameter
    update norms are compared before/after the step.
    """
    import torch

    max_steps = steps
    if spec.budget.max_backward_passes is not None:
        max_steps = min(max_steps, spec.budget.max_backward_passes)
    if spec.budget.max_forward_passes is not None:
        max_steps = min(max_steps, spec.budget.max_forward_passes)
    if max_steps < 1:
        max_steps = 1

    probe_start = monotonic()
    model = model_factory()
    before = _copy_model(model)

    held_out_loss_before = None
    held_out_loss_after = None
    if held_out_inputs is not None:
        with torch.no_grad():
            held_out_loss_before = float(loss_fn(before, held_out_inputs).item())

    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    gradients: tuple[GradientMeasurement, ...] = ()
    for _ in range(max_steps):
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(model, inputs)
        loss.backward()  # type: ignore[no-untyped-call]  # torch.Tensor.backward untyped in installed stubs; required for gradient measurement
        gradients = _measure_gradients(model)
        optimizer.step()

    after = _copy_model(model)
    if held_out_inputs is not None:
        with torch.no_grad():
            held_out_loss_after = float(loss_fn(after, held_out_inputs).item())

    updates = _parameter_update_norms(before, after)

    component_attribution: dict[str, float] | None = None
    if components and gradients:
        attribution: dict[str, float] = {}
        for comp in components:
            comp_grads = [g for g in gradients if g.parameter_name.startswith(comp)]
            if comp_grads:
                norms = [g.grad_norm for g in comp_grads if g.grad_norm is not None]
                attribution[comp] = sum(norms) / len(norms) if norms else 0.0
        component_attribution = attribution

    gradient_presence_proven = any(
        g.grad_norm is not None and g.grad_norm > 0 for g in gradients
    )

    probe_result = ProbeResult(
        probe_id=spec.probe_id,
        status="completed",
        observations={
            "steps": max_steps,
            "lr": lr,
            "parameter_count": len(_named_parameters(model)),
            "gradient_count": len(gradients),
            "gradients_with_presence": sum(
                1 for g in gradients if g.grad_norm is not None
            ),
        },
        provenance=None,
        duration_seconds=monotonic() - probe_start,
    )

    return OptimizationProbeResult(
        probe_result=probe_result,
        gradients=gradients,
        parameter_updates=updates,
        held_out_loss_before=held_out_loss_before,
        held_out_loss_after=held_out_loss_after,
        component_attribution=component_attribution,
        gradient_presence_proven=gradient_presence_proven,
    )
