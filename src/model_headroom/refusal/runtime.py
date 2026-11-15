from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import torch
from torch import nn

if TYPE_CHECKING:
    from collections.abc import Sequence

from model_headroom.refusal.extract import project_out

LAMBDA_ENV_VAR = "SIA_REFUSAL_LAMBDA"
ARTIFACT_ENV_VAR = "SIA_REFUSAL_ARTIFACT"
TENSOR_REFUSAL_DIRECTION = "refusal_direction"
TENSOR_DOMAIN_DIRECTION = "domain_direction"
DEFAULT_TARGET_SUFFIXES: tuple[str, ...] = (
    "o_proj",
    "out_proj",
    "down_proj",
    "shared_expert",
)
EXTENDED_TARGET_SUFFIXES: tuple[str, ...] = (
    *DEFAULT_TARGET_SUFFIXES,
    "experts",
    "linear_attn",
)
DEFAULT_STRENGTH = 0.0
UNIT_NORM_TOLERANCE = 1e-3


@dataclass
class RefusalArtifact:
    direction: torch.Tensor
    domain_direction: torch.Tensor | None = None
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass
class AblationHandle:
    strength: float
    strength_tensor: torch.Tensor | None
    handles: list[Any]
    targets: list[str]

    def set_strength(self, value: float) -> None:
        tensor = self.strength_tensor
        if tensor is None:
            if value != 0.0:
                raise RuntimeError(
                    "cannot set a non-zero strength on a no-op handle: install the "
                    "arithmetic hook (strength != 0.0) before capturing a CUDA graph"
                )
            self.strength = value
            return
        tensor.fill_(float(value))
        self.strength = value

    def remove(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def load_artifact(path: str | Path) -> RefusalArtifact:
    from safetensors import safe_open
    from safetensors.torch import load_file

    path = Path(path)
    tensors = load_file(str(path))
    if TENSOR_REFUSAL_DIRECTION not in tensors:
        raise ValueError(
            f"artifact {path} is missing the {TENSOR_REFUSAL_DIRECTION!r} tensor"
        )
    direction = tensors[TENSOR_REFUSAL_DIRECTION].to(torch.float32)
    if direction.ndim != 1:
        raise ValueError(
            f"{TENSOR_REFUSAL_DIRECTION} in {path} must be 1-D, got ndim={direction.ndim}"
        )
    metadata: dict[str, str] = {}
    with safe_open(str(path), framework="pt") as handle:
        loaded_metadata = handle.metadata()
        if loaded_metadata:
            metadata = dict(loaded_metadata)
    hidden_size = metadata.get("hidden_size")
    if hidden_size is not None and int(direction.shape[0]) != int(hidden_size):
        raise ValueError(
            f"{path}: direction length {int(direction.shape[0])} != hidden_size {hidden_size}"
        )
    domain_direction: torch.Tensor | None = None
    if TENSOR_DOMAIN_DIRECTION in tensors:
        domain_direction = tensors[TENSOR_DOMAIN_DIRECTION].to(torch.float32)
        if domain_direction.ndim != 1:
            raise ValueError(
                f"{TENSOR_DOMAIN_DIRECTION} in {path} must be 1-D, "
                f"got ndim={domain_direction.ndim}"
            )
        if hidden_size is not None and int(domain_direction.shape[0]) != int(
            hidden_size
        ):
            raise ValueError(
                f"{path}: domain_direction length {int(domain_direction.shape[0])} "
                f"!= hidden_size {hidden_size}"
            )
    norm = float(direction.norm())
    if abs(norm - 1.0) > UNIT_NORM_TOLERANCE:
        raise ValueError(
            f"{path}: {TENSOR_REFUSAL_DIRECTION} is not unit-norm (norm={norm})"
        )
    return RefusalArtifact(
        direction=direction, domain_direction=domain_direction, metadata=metadata
    )


def resolve_lambda(explicit: float | None = None) -> float:
    if explicit is not None:
        return float(explicit)
    raw = os.environ.get(LAMBDA_ENV_VAR)
    if raw is None:
        return DEFAULT_STRENGTH
    try:
        value = float(raw)
    except ValueError:
        raise ValueError(
            f"{LAMBDA_ENV_VAR}={raw!r} is not a valid finite float "
            f"(expected 0.0, 0.5, 1.0, etc.)"
        ) from None
    if not math.isfinite(value):
        raise ValueError(
            f"{LAMBDA_ENV_VAR}={raw!r} is not finite (expected 0.0, 0.5, 1.0, etc.)"
        )
    return value


def find_residual_writers(
    model: nn.Module, suffixes: Sequence[str] = DEFAULT_TARGET_SUFFIXES
) -> list[tuple[str, nn.Module]]:
    suffix_set = set(suffixes)
    candidates = [
        (name, module)
        for name, module in model.named_modules()
        if name and name.rsplit(".", 1)[-1] in suffix_set
    ]
    selected: list[tuple[str, nn.Module]] = []
    for name, module in candidates:
        if any(
            other != name and name.startswith(other + ".") for other, _ in candidates
        ):
            continue
        selected.append((name, module))
    selected.sort(key=lambda item: item[0])
    return selected


def _make_arithmetic_hook(
    direction: torch.Tensor, strength_tensor: torch.Tensor
) -> Any:
    def hook(module: Any, inputs: Any, output: Any) -> Any:
        if isinstance(output, torch.Tensor):
            original_dtype = output.dtype
            projected = project_out(
                output.to(torch.float32), direction, strength_tensor
            )
            return projected.to(original_dtype)
        if (
            isinstance(output, (tuple, list))
            and len(output) > 0
            and isinstance(output[0], torch.Tensor)
        ):
            original_dtype = output[0].dtype
            projected = project_out(
                output[0].to(torch.float32), direction, strength_tensor
            )
            return (projected.to(original_dtype), *output[1:])
        return None

    return hook


def install_refusal_ablation(
    model: nn.Module,
    artifact: RefusalArtifact,
    *,
    strength: float | None = None,
    targets: Sequence[str] | None = None,
) -> AblationHandle:
    resolved = resolve_lambda(strength)
    writers = find_residual_writers(model, targets or DEFAULT_TARGET_SUFFIXES)
    if not writers:
        raise ValueError(
            f"no residual writers matched suffixes "
            f"{tuple(targets or DEFAULT_TARGET_SUFFIXES)}"
        )
    direction = artifact.direction.to(torch.float32)
    first = writers[0][1]
    params = list(first.parameters())
    device = params[0].device if params else torch.device("cpu")
    direction = direction.to(device)
    strength_tensor: torch.Tensor | None = None
    if resolved != 0.0:
        strength_tensor = torch.tensor(resolved, dtype=torch.float32, device=device)
        hook = _make_arithmetic_hook(direction, strength_tensor)
    else:

        def hook(module: Any, inputs: Any, output: Any) -> Any:
            return None

    hook_handles = [module.register_forward_hook(hook) for _, module in writers]
    return AblationHandle(
        strength=resolved,
        strength_tensor=strength_tensor,
        handles=hook_handles,
        targets=[name for name, _ in writers],
    )


def verify_lambda_zero_identity(
    model: nn.Module,
    artifact: RefusalArtifact,
    *,
    inputs: Any,
    targets: Sequence[str] | None = None,
) -> bool:
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            output_before = _call_model(model, inputs)
            tensor_before = _output_tensor(output_before)
        handle = install_refusal_ablation(
            model, artifact, strength=0.0, targets=targets
        )
        try:
            with torch.no_grad():
                output_after = _call_model(model, inputs)
                tensor_after = _output_tensor(output_after)
        finally:
            handle.remove()
    finally:
        model.train(was_training)
    return torch.equal(tensor_before, tensor_after)


def _call_model(model: nn.Module, inputs: Any) -> Any:
    if isinstance(inputs, tuple):
        return model(*inputs)
    return model(inputs)


def _output_tensor(output: Any) -> torch.Tensor:
    if hasattr(output, "logits"):
        return cast("torch.Tensor", output.logits)
    return cast("torch.Tensor", output)
