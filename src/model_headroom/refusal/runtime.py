from __future__ import annotations

import fnmatch
import hashlib
import math
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

import torch
from torch import nn

if TYPE_CHECKING:
    from collections.abc import Sequence

from model_headroom.refusal.extract import project_out

LAMBDA_ENV_VAR = "SIA_REFUSAL_LAMBDA"
ARTIFACT_ENV_VAR = "SIA_REFUSAL_ARTIFACT"
TENSOR_REFUSAL_DIRECTION = "refusal_direction"
TENSOR_DOMAIN_DIRECTION = "domain_direction"
CAPTURE_SPACE_RESIDUAL = "residual"
CAPTURE_SPACE_WIDENED = "widened_stream"
CAPTURE_SPACES = (CAPTURE_SPACE_RESIDUAL, CAPTURE_SPACE_WIDENED)
FOLD_MODES = ("mean", "first", "least_squares")
GEOMETRY_FIELDS = (
    "capture_space",
    "capture_width",
    "source_hidden_size",
    "stream_count",
    "fold_mode",
    "geometry_id",
)
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
ARCHITECTURE_TARGET_MAPS: dict[str, tuple[str, ...]] = {
    "qwen2": DEFAULT_TARGET_SUFFIXES,
    "qwen3_moe": DEFAULT_TARGET_SUFFIXES,
    "qwen4_exp": EXTENDED_TARGET_SUFFIXES,
}
# HF model-class name suffixes stripped before matching the architecture map.
# Real serving classes are like Qwen2ForCausalLM / Qwen4ExpForCausalLM; the
# underscored config keys are qwen2 / qwen4_exp. Stripping the suffix and
# normalizing separators lets "auto" resolve a real class to its map entry.
_HF_CLASS_SUFFIXES: tuple[str, ...] = (
    "ForCausalLM",
    "ForConditionalGeneration",
    "ForSequenceClassification",
    "ForTokenClassification",
    "ForQuestionAnswering",
    "Model",
    "PreTrainedModel",
)
DEFAULT_STRENGTH = 0.0
UNIT_NORM_TOLERANCE = 1e-3


@dataclass
class RefusalArtifact:
    direction: torch.Tensor
    domain_direction: torch.Tensor | None = None
    metadata: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.metadata.get("geometry_id") is None:
            self.metadata["geometry_id"] = _geometry_id(self.direction)
        _validate_artifact_geometry(self)


def _geometry_id(direction: torch.Tensor) -> str:
    payload = direction.detach().cpu().to(torch.float64).numpy().tobytes()
    return f"sha256:{hashlib.sha256(payload).hexdigest()[:16]}"


def _parse_int_metadata(metadata: dict[str, str], key: str) -> int | None:
    raw = metadata.get(key)
    if raw is None:
        return None
    try:
        return int(raw)
    except ValueError:
        raise ValueError(f"{key}={raw!r} is not an integer") from None


def _validate_artifact_geometry(artifact: RefusalArtifact) -> None:
    metadata = artifact.metadata
    direction_width = int(artifact.direction.shape[0])
    capture_space = metadata.get("capture_space")
    if capture_space is not None and capture_space not in CAPTURE_SPACES:
        raise ValueError(
            f"capture_space={capture_space!r} is not one of {CAPTURE_SPACES}"
        )
    capture_width = _parse_int_metadata(metadata, "capture_width")
    if capture_width is not None and capture_width != direction_width:
        raise ValueError(
            f"capture_width={capture_width} != direction length {direction_width}"
        )
    stream_count = _parse_int_metadata(metadata, "stream_count")
    if stream_count is not None and stream_count > 1:
        source_hidden_size = _parse_int_metadata(metadata, "source_hidden_size")
        if source_hidden_size is None:
            raise ValueError(f"stream_count={stream_count} requires source_hidden_size")
        if source_hidden_size * stream_count != direction_width:
            raise ValueError(
                f"source_hidden_size*stream_count "
                f"({source_hidden_size}*{stream_count}={source_hidden_size * stream_count})"
                f" != direction length {direction_width}"
            )
    fold_mode = metadata.get("fold_mode")
    if fold_mode is not None and fold_mode != "none" and fold_mode not in FOLD_MODES:
        raise ValueError(f"fold_mode={fold_mode!r} is not one of {FOLD_MODES}")


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
    artifact = RefusalArtifact(
        direction=direction, domain_direction=domain_direction, metadata=metadata
    )
    capture_space = metadata.get("capture_space")
    if capture_space is None:
        stream_count = _parse_int_metadata(metadata, "stream_count")
        source_hidden_size = _parse_int_metadata(metadata, "source_hidden_size")
        if (
            stream_count is not None
            and source_hidden_size is not None
            and stream_count > 1
            and source_hidden_size * stream_count == int(direction.shape[0])
        ):
            artifact.metadata["capture_space"] = CAPTURE_SPACE_WIDENED
    _validate_artifact_geometry(artifact)
    return artifact


def project_direction_to_residual(
    artifact: RefusalArtifact,
    *,
    hidden_size: int,
    mode: Literal["mean", "first", "least_squares"] = "least_squares",
) -> RefusalArtifact:
    if mode not in FOLD_MODES:
        raise ValueError(f"mode={mode!r} is not one of {FOLD_MODES}")
    if hidden_size <= 0:
        raise ValueError(f"hidden_size={hidden_size} must be positive")
    source_stream_count = _parse_int_metadata(artifact.metadata, "stream_count")
    source_hidden_size = _parse_int_metadata(artifact.metadata, "source_hidden_size")
    if (
        source_stream_count is not None
        and source_stream_count > 1
        and source_hidden_size is not None
        and source_hidden_size != hidden_size
    ):
        raise ValueError(
            f"artifact stream_count={source_stream_count} encodes "
            f"source_hidden_size={source_hidden_size}; project_direction_to_residual "
            f"requires hidden_size={source_hidden_size}, got {hidden_size}"
        )
    direction_width = int(artifact.direction.shape[0])
    if direction_width % hidden_size != 0:
        raise ValueError(
            f"direction length {direction_width} is not a multiple of "
            f"hidden_size {hidden_size}"
        )
    stream_count = direction_width // hidden_size
    if (
        stream_count > 1
        and source_stream_count is not None
        and source_stream_count != stream_count
    ):
        raise ValueError(
            f"stream_count={source_stream_count} != direction_width // "
            f"hidden_size = {stream_count}"
        )
    if source_stream_count is not None and source_stream_count > 1 and mode == "first":
        raise ValueError(
            "first is rejected for declared multi-stream artifacts: the stream "
            "index is positional and not semantically privileged — use mean or "
            "least_squares"
        )
    chunks = artifact.direction.view(stream_count, hidden_size).to(torch.float32)
    if mode == "first":
        folded = chunks[0]
    elif mode == "least_squares":
        design = torch.eye(
            hidden_size, dtype=torch.float32, device=chunks.device
        ).repeat(stream_count, 1)
        target = chunks.reshape(-1)
        solution = torch.linalg.lstsq(design, target.unsqueeze(1)).solution
        folded = solution.squeeze(1)
    else:
        folded = chunks.mean(dim=0)
    norm = float(folded.norm())
    if norm < 1e-12:
        raise ValueError("projected direction has zero norm; cannot normalize")
    folded = folded / norm
    metadata = dict(artifact.metadata)
    metadata["capture_space"] = CAPTURE_SPACE_RESIDUAL
    metadata["capture_width"] = str(hidden_size)
    metadata["source_hidden_size"] = str(hidden_size)
    metadata["stream_count"] = "1"
    metadata["fold_mode"] = mode
    metadata["folded_from_capture_width"] = str(direction_width)
    metadata.pop("geometry_id", None)
    if artifact.domain_direction is not None:
        domain_width = int(artifact.domain_direction.shape[0])
        if domain_width % hidden_size != 0:
            raise ValueError(
                f"domain_direction length {domain_width} is not a multiple of "
                f"hidden_size {hidden_size}"
            )
        domain_chunks = artifact.domain_direction.view(
            domain_width // hidden_size, hidden_size
        ).to(torch.float32)
        domain_folded = domain_chunks.mean(dim=0)
        domain_norm = float(domain_folded.norm())
        if domain_norm < 1e-12:
            raise ValueError("projected domain_direction has zero norm")
        domain_direction = (domain_folded / domain_norm).to(
            artifact.domain_direction.dtype
        )
    else:
        domain_direction = None
    projected = RefusalArtifact(
        direction=folded.to(torch.float32),
        domain_direction=domain_direction,
        metadata=metadata,
    )
    _validate_artifact_geometry(projected)
    return projected


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


def _architecture_key(model: nn.Module) -> str:
    """Derive the architecture map key from a model instance.

    Prefers the config-declared architecture/model_type when available (real
    HF checkpoints expose ``model.config.architectures`` / ``model.config.
    model_type``), then falls back to the class name with HF suffixes stripped
    and separators normalized to underscores. Returns a lowercase, underscored
    key that can be looked up in ``ARCHITECTURE_TARGET_MAPS``.
    """
    config = getattr(model, "config", None)
    if config is not None:
        architectures = getattr(config, "architectures", None)
        if isinstance(architectures, (list, tuple)) and architectures:
            return _normalize_arch_name(str(architectures[0]))
        model_type = getattr(config, "model_type", None)
        if isinstance(model_type, str) and model_type:
            return _normalize_arch_name(model_type)
    return _normalize_arch_name(type(model).__name__)


def _normalize_arch_name(name: str) -> str:
    """Normalize an architecture name to an underscored map key.

    Strips known HF class suffixes (``ForCausalLM`` etc.), inserts underscores
    at camelCase and digit/letter boundaries, lowercases, and collapses runs of
    underscores. ``Qwen4ExpForCausalLM`` -> ``qwen4_exp``; ``Qwen2ForCausalLM``
    -> ``qwen2``; ``Qwen3MoeForCausalLM`` -> ``qwen3_moe``.
    """
    for suffix in _HF_CLASS_SUFFIXES:
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    name = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", name)
    name = re.sub(r"[^0-9A-Za-z]+", "_", name).strip("_").lower()
    return re.sub(r"_+", "_", name)


def _resolve_targets(
    named: list[tuple[str, nn.Module]],
    targets: Sequence[str] | None,
    model: nn.Module,
) -> list[tuple[str, nn.Module]]:
    if targets is None or tuple(targets) == ("auto",):
        if targets is None:
            resolved: Sequence[str] = DEFAULT_TARGET_SUFFIXES
        else:
            arch = _architecture_key(model)
            mapped = ARCHITECTURE_TARGET_MAPS.get(arch)
            if mapped is None:
                raise ValueError(
                    f"targets='auto' requires an architecture in "
                    f"{tuple(ARCHITECTURE_TARGET_MAPS)}; model class {type(model).__name__} "
                    f"(key {arch!r}) has no entry — pass an explicit target list"
                )
            resolved = mapped
    else:
        resolved = targets
    resolved_set = set(resolved)
    has_glob = any("*" in target or "?" in target for target in resolved)
    has_qualified = any("." in target for target in resolved)
    matched: list[tuple[str, nn.Module]] = []
    for name, module in named:
        if not name:
            continue
        if has_glob and any(fnmatch.fnmatch(name, target) for target in resolved_set):
            matched.append((name, module))
            continue
        if name in resolved_set:
            matched.append((name, module))
            continue
        if has_qualified and any(name.endswith(target) for target in resolved_set):
            matched.append((name, module))
            continue
        if name.rsplit(".", 1)[-1] in resolved_set:
            matched.append((name, module))
    selected: list[tuple[str, nn.Module]] = []
    for name, module in matched:
        if any(other != name and name.startswith(other + ".") for other, _ in matched):
            continue
        selected.append((name, module))
    selected.sort(key=lambda item: item[0])
    return selected


def find_residual_writers(
    model: nn.Module,
    targets: Sequence[str] | None = None,
) -> list[tuple[str, nn.Module]]:
    named = list(model.named_modules())
    resolved = _resolve_targets(named, targets, model)
    if resolved:
        return resolved
    raise ValueError(
        "no residual writers matched targets "
        f"{tuple(targets or DEFAULT_TARGET_SUFFIXES)}; "
        "accepted modes: bare suffixes (e.g. 'o_proj'), fully-qualified names "
        "(e.g. 'model.layers.0.mlp.down_proj'), glob patterns (e.g. "
        "'model.layers.*.mlp.down_proj'), or 'auto' for per-architecture maps "
        f"{tuple(ARCHITECTURE_TARGET_MAPS)}"
    )


def writer_output_width(module: nn.Module) -> int | None:
    if isinstance(module, nn.Linear):
        return int(module.out_features)
    for param in module.parameters(recurse=True):
        if param.ndim >= 2 and int(param.shape[0]) > 0:
            return int(param.shape[0])
    return None


def validate_install(
    model: nn.Module,
    artifact: RefusalArtifact,
    targets: Sequence[str] | None = None,
) -> list[tuple[str, nn.Module]]:
    _validate_artifact_geometry(artifact)
    if artifact.metadata.get("capture_space") == CAPTURE_SPACE_WIDENED:
        raise ValueError(
            f"artifact is in capture_space={CAPTURE_SPACE_WIDENED!r} "
            f"(width {int(artifact.direction.shape[0])}); call "
            "project_direction_to_residual() before installation — "
            "a widened vector is not residual-space geometry"
        )
    writers = find_residual_writers(model, targets)
    direction_width = int(artifact.direction.shape[0])
    for name, module in writers:
        width = writer_output_width(module)
        if width is None:
            continue
        if width != direction_width:
            raise ValueError(
                f"direction width {direction_width} != writer {name} output width "
                f"{width}; fold via project_direction_to_residual(hidden_size={width})"
            )
    return writers


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
    writers = validate_install(model, artifact, targets)
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
