"""Streaming per-channel FFN activation acquisition for evidence acquisition.

Acquires per-channel statistics of the FFN interior (the gate and up
pre-activation branches, and the post-activation contribution) by running a
real forward pass of a ``transformers`` model with torch forward hooks.

Why the existing flattened representation cannot be reused:
``model_headroom.paired.instrument._sample_flat`` does ``reshape(-1)`` on the
captured tensor, which destroys channel identity. Per-channel RMS,
near-zero occupancy, and cross-channel covariance CANNOT be computed from a
flattened sample — the channel dimension is gone. This module instead
accumulates STREAMING per-channel statistics and never stores raw activation
tensors: memory is O(channels), not O(tokens x channels).

CRITICAL CORRECTNESS POINT (measured on Qwen3): the FFN container's output is
the DOWN-PROJECTION result, shape ``[B, S, hidden_size]`` — NOT the
intermediate. The gate/up branches are ``[B, S, intermediate_size]``. This
module therefore targets the ``gate_proj`` and ``up_proj`` Linear OUTPUTS for
per-channel intermediate statistics, and the container output only for the
post-activation contribution statistic. Getting this backwards would silently
measure the residual stream instead of the FFN interior.

``torch`` is imported lazily inside functions so ``import
model_headroom.acquire`` works without the ``[headroom]`` extra.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, cast

from model_headroom.models.types import DeadNeuronReport, SourceType

if TYPE_CHECKING:
    import torch

BranchName = Literal["gate", "up", "down"]


@dataclass(frozen=True)
class FFNChannelStats:
    """Streaming per-channel statistics of one FFN branch of one layer.

    All per-channel vectors are indexed by channel position along the last
    dimension of the branch output. Statistics are accumulated over every
    token position observed in the forward pass(es) and are exact for the
    quantities recorded here.

    Attributes:
        layer_name: The FFN container module path (e.g. ``layers.0.mlp``).
        branch: Which branch these statistics describe. ``"gate"`` and
            ``"up"`` are the pre-activation branches (intermediate-size
            channels); ``"down"`` is the post-activation contribution
            (hidden-size channels, the container output on Qwen3/SwiGLU).
        channels: Number of channels (length of every per-channel vector).
        tokens_observed: Total token positions summed over all batches.
        rms: Per-channel root-mean-square, ``sqrt(sum(x^2)/count)``.
        mean: Per-channel arithmetic mean.
        abs_mean: Per-channel mean of the absolute value.
        nonzero_fraction: Per-channel fraction of observed tokens that were
            exactly non-zero.
        min: Per-channel minimum observed value.
        max: Per-channel maximum observed value.
    """

    layer_name: str
    branch: BranchName
    channels: int
    tokens_observed: int
    rms: tuple[float, ...]
    mean: tuple[float, ...]
    abs_mean: tuple[float, ...]
    nonzero_fraction: tuple[float, ...]
    min: tuple[float, ...]
    max: tuple[float, ...]


def _accumulate(
    acc: dict[str, Any],
    x: torch.Tensor,
    count: list[int],
) -> None:
    """Fold one batch of activations into the streaming accumulators.

    ``acc`` holds per-channel float32 vectors on the compute device;
    ``count`` holds the running token total as a single-element list so the
    closure can update it in place.
    """
    import torch

    tokens = int(x.shape[0])
    count[0] += tokens
    acc["sum"] += x.sum(dim=0)
    acc["sumsq"] += (x * x).sum(dim=0)
    acc["abs_sum"] += x.abs().sum(dim=0)
    acc["nonzero"] += (x != 0).sum(dim=0)
    acc["min"] = torch.minimum(acc["min"], x.min(dim=0).values)
    acc["max"] = torch.maximum(acc["max"], x.max(dim=0).values)


def _finalize(
    layer_name: str,
    branch: BranchName,
    acc: dict[str, Any],
    count: list[int],
) -> FFNChannelStats:
    """Collapse the streaming accumulators into a frozen FFNChannelStats."""
    import torch

    tokens = count[0]
    if tokens <= 0:
        raise ValueError(
            f"no tokens observed for {layer_name}:{branch}; the forward pass "
            "produced no activations, so no statistics can be computed"
        )
    channels = int(acc["sum"].shape[0])
    total = torch.tensor(float(tokens), device=acc["sum"].device)
    mean = acc["sum"] / total
    rms = (acc["sumsq"] / total).clamp(min=0.0).sqrt()
    abs_mean = acc["abs_sum"] / total
    nonzero_fraction = acc["nonzero"] / total

    def _vec(v: torch.Tensor) -> tuple[float, ...]:
        return tuple(float(item) for item in v.cpu().tolist())

    return FFNChannelStats(
        layer_name=layer_name,
        branch=branch,
        channels=channels,
        tokens_observed=tokens,
        rms=_vec(rms),
        mean=_vec(mean),
        abs_mean=_vec(abs_mean),
        nonzero_fraction=_vec(nonzero_fraction),
        min=_vec(acc["min"]),
        max=_vec(acc["max"]),
    )


def _find_ffn_containers(
    model: torch.nn.Module,
) -> dict[str, dict[str, str]]:
    """Locate FFN containers that expose gate_proj and up_proj Linear children.

    A container is any module whose direct children include both ``gate_proj``
    and ``up_proj`` with 2-D weights (i.e. Linear projections). On Qwen3/SwiGLU
    dense models this is the ``mlp`` module of each decoder layer; on MoE
    models the experts live deeper, so this first slice targets dense FFN
    containers only.
    """
    import torch

    containers: dict[str, dict[str, str]] = {}
    for path, module in model.named_modules():
        children = dict(module.named_children())
        gate = children.get("gate_proj")
        up = children.get("up_proj")
        if gate is None or up is None:
            continue
        if not isinstance(gate, torch.nn.Linear) or not isinstance(up, torch.nn.Linear):
            continue
        containers[path] = {"gate": f"{path}.gate_proj", "up": f"{path}.up_proj"}
    return containers


def capture_ffn_channel_stats(
    model: torch.nn.Module,
    inputs: dict[str, Any] | torch.Tensor,
    *,
    device: str | None = None,
) -> dict[str, FFNChannelStats]:
    """Capture per-channel FFN statistics from one real forward pass.

    Registers torch forward hooks on the gate_proj and up_proj Linear outputs
    (pre-activation branches) and on the FFN container output (the
    post-activation contribution), runs ``model(**inputs)`` once, and folds
    every captured activation into streaming per-channel accumulators. Raw
    activation tensors are never retained: memory is O(channels).

    Input:
        model: A transformers model (e.g. a Qwen3 causal-LM or base model).
        inputs: Forward-pass keyword arguments (typically
            ``{"input_ids": torch.Tensor}``, see :func:`tokenize_prompt`), or
            a bare ``input_ids`` tensor, which is wrapped for the caller.
        device: Device for the forward pass; defaults to the device the
            model's parameters already live on.
    Output:
        Mapping of ``"{container_path}:{branch}"`` (branch one of ``gate``,
        ``up``, ``down``) to that branch's ``FFNChannelStats``.
    Raises:
        ValueError: When the model exposes no gate/up FFN container, or when
            the forward pass observed no tokens.
    Invariants:
        - Hooks are always removed, even when the forward pass raises.
        - The ``down`` branch is the container OUTPUT (hidden-size channels on
          Qwen3/SwiGLU), not the intermediate.
    """
    import torch

    if torch.is_tensor(inputs):
        inputs = {"input_ids": inputs}

    if device is not None:
        compute_device = torch.device(device)
    else:
        try:
            compute_device = next(model.parameters()).device
        except StopIteration as e:
            raise ValueError("model has no parameters; no device to resolve") from e

    containers = _find_ffn_containers(model)
    if not containers:
        raise ValueError(
            "no FFN container with gate_proj/up_proj Linear children found in "
            "the model; this slice targets dense SwiGLU-style FFNs"
        )

    accumulators: dict[str, tuple[dict[str, Any], list[int]]] = {}
    for path in containers:
        for branch in ("gate", "up", "down"):
            accumulators[f"{path}:{branch}"] = ({}, [0])

    hooks: list[Any] = []

    def _make_hook(key: str) -> Any:
        def hook(_module: Any, _inputs: Any, output: Any) -> None:
            if not torch.is_tensor(output):
                return
            x = output.detach()
            if x.ndim not in (2, 3):
                return
            x = x.reshape(-1, x.shape[-1]).float().to(compute_device)
            if x.shape[0] == 0:
                return
            acc, count = accumulators[key]
            channels = int(x.shape[-1])
            if "sum" not in acc:
                zero = torch.zeros(channels, device=compute_device)
                acc["sum"] = zero.clone()
                acc["sumsq"] = zero.clone()
                acc["abs_sum"] = zero.clone()
                acc["nonzero"] = torch.zeros(channels, device=compute_device)
                acc["min"] = torch.full(
                    (channels,), float("inf"), device=compute_device
                )
                acc["max"] = torch.full(
                    (channels,), float("-inf"), device=compute_device
                )
            _accumulate(acc, x, count)

        return hook

    try:
        for path, branches in containers.items():
            hooks.append(
                model.get_submodule(branches["gate"]).register_forward_hook(
                    _make_hook(f"{path}:gate")
                )
            )
            hooks.append(
                model.get_submodule(branches["up"]).register_forward_hook(
                    _make_hook(f"{path}:up")
                )
            )
            hooks.append(
                model.get_submodule(path).register_forward_hook(
                    _make_hook(f"{path}:down")
                )
            )

        kwargs = {
            key: (value.to(compute_device) if torch.is_tensor(value) else value)
            for key, value in inputs.items()
        }
        with torch.no_grad():
            model(**kwargs)
    finally:
        for handle in hooks:
            handle.remove()

    stats: dict[str, FFNChannelStats] = {}
    for key, (acc, count) in accumulators.items():
        if count[0] <= 0:
            continue
        path, branch = key.rsplit(":", 1)
        stats[key] = _finalize(
            path, cast("Literal['gate', 'up', 'down']", branch), acc, count
        )
    if not stats:
        raise ValueError("no tokens observed; the forward pass produced no activations")
    return stats


def tokenize_prompt(tokenizer: Any, text: str) -> dict[str, Any]:
    """Tokenize a prompt into the forward-pass inputs this package consumes.

    Input:
        tokenizer: A transformers tokenizer (any object whose call returns a
            mapping containing ``"input_ids"`` when passed
            ``return_tensors="pt"``). Typed ``Any`` because transformers
            tokenizers are not a type dependency of this module.
        text: The prompt text.
    Output:
        ``{"input_ids": torch.Tensor}`` ready for
        :func:`capture_ffn_channel_stats`.
    """
    import torch

    encoded = tokenizer(text, return_tensors="pt")
    input_ids = encoded["input_ids"]
    if not torch.is_tensor(input_ids):
        input_ids = torch.as_tensor(input_ids)
    return {"input_ids": input_ids}


def dead_neuron_report_from_channel_stats(
    stats: FFNChannelStats,
    *,
    source_type: SourceType = "measured",
) -> DeadNeuronReport:
    """Compute the dead-neuron signal's statistics from channel stats.

    Computes, from streaming channel statistics, exactly the quantities that
    :func:`model_headroom.signals.dead_neurons.analyze_dead_neurons` computes
    from a 3-D activation tensor — WITHOUT reconstructing (fabricating) an
    activation tensor:

    - ``dead``: a channel is dead when its max-abs is exactly 0, which from
      the recorded extremes is ``max == 0 and min == 0``.
    - ``negative_mean_count``: channels whose mean is negative.
    - ``activation_rms``: sqrt of the mean over all elements of x^2, which is
      the mean over channels of per-channel rms^2.
    - ``mean_abs_activation``: the mean over channels of per-channel abs_mean.
    - ``variance``: the population variance over all elements,
      ``E[x^2] - E[x]^2`` (clamped at 0 to absorb float round-off).
    - ``cross_token_variance``: the population variance of the per-channel
      means.
    - ``never_active``: the signal's documented alias of ``dead``.

    ``near_zero_occupancy`` (the fraction of elements with ``|x| < 1e-5``)
    is NOT derivable from streaming statistics — it needs the element-level
    distribution, not the recorded moments — and is therefore reported as
    ``None`` (honestly unavailable), never fabricated.

    Input:
        stats: The channel statistics for one FFN branch of one layer.
        source_type: Provenance label carried on the report.
    Output:
        A ``DeadNeuronReport`` with every field the signal computes, except
        ``near_zero_occupancy`` which is ``None``.
    """
    import torch

    mean = torch.tensor(stats.mean, dtype=torch.float64)
    rms = torch.tensor(stats.rms, dtype=torch.float64)
    abs_mean = torch.tensor(stats.abs_mean, dtype=torch.float64)
    mins = torch.tensor(stats.min, dtype=torch.float64)
    maxs = torch.tensor(stats.max, dtype=torch.float64)

    total_neurons = stats.channels
    max_abs = torch.maximum(maxs, mins.abs())
    dead_mask = max_abs == 0
    dead_count = int(dead_mask.sum().item())
    dead_fraction = dead_count / total_neurons if total_neurons > 0 else 0.0

    negative_mean_count = int((mean < 0).sum().item())
    negative_mean_fraction = (
        negative_mean_count / total_neurons if total_neurons > 0 else 0.0
    )

    e_x2 = float((rms**2).mean().item())
    e_x = float(mean.mean().item())
    variance = max(0.0, e_x2 - e_x * e_x)

    cross_token_variance = (
        float(mean.var(unbiased=False).item()) if total_neurons > 1 else 0.0
    )

    return DeadNeuronReport(
        total_neurons=total_neurons,
        dead_count=dead_count,
        dead_fraction=dead_fraction,
        dead_indices=None,
        signal_status="computed",
        activation_rms=float(rms.pow(2).mean().sqrt().item()),
        mean_abs_activation=float(abs_mean.mean().item()),
        variance=variance,
        near_zero_occupancy=None,
        cross_token_variance=cross_token_variance,
        negative_mean_count=negative_mean_count,
        negative_mean_fraction=negative_mean_fraction,
        never_active_count=dead_count,
        never_active_fraction=dead_fraction,
        source_type=source_type,
        # Degenerate criterion: exact-zero amax essentially never fires on
        # real gated-FFN activations, so dead_count == 0 with an unverifiable
        # near_zero_occupancy means the criterion cannot discriminate —
        # withhold the confident computed score (neutral prior applies).
        degenerate_criterion=dead_count == 0,
    )
