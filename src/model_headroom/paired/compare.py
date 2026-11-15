"""Paired-model measurement comparison.

Pure functions consume two MeasurementDocuments and emit a typed
ComparisonReport. Layers are paired by layer_name, never by position. An
unmeasured quantity is None, never 0.0. An undefined cosine (zero candidate
norm) is reported as cosine=None, never as zero. A packed buffer never yields a
weight-spectrum number. A shape mismatch raises ComparisonError. Coverage
reports its own denominator, with None meaning unknown.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from model_headroom.paired.schema import (
        MeasurementDocument,
        TensorKind,
    )


class ComparisonError(ValueError):
    """Raised for a contract violation: shape mismatch, wrong tensor kind, zero reference norm."""


@dataclass(frozen=True)
class LayerComparison:
    """Frozen dataclass holding one layer's paired output comparison."""

    layer_name: str
    family: str
    cosine: float | None
    normalized_reconstruction_error: float | None
    norm_ratio: float | None
    reference_norm: float | None
    candidate_norm: float | None
    status: str
    reason: str | None


@dataclass(frozen=True)
class ModuleComparison:
    """Frozen dataclass holding one module's effective-weight reconstruction comparison."""

    module_path: str
    family: str
    effective_weight_reconstruction_error: float | None
    relative_error: float | None
    weight_kind: str
    status: str
    reason: str | None


@dataclass(frozen=True)
class GradientComparison:
    """Frozen dataclass holding the paired gradient comparison."""

    status: str
    reference_grad_norm: float | None
    candidate_grad_norm: float | None
    grad_norm_ratio: float | None
    coherence_delta: float | None
    effective_rank_ratio: float | None
    reason: str | None


@dataclass(frozen=True)
class ActivationOutlierReport:
    """Frozen dataclass holding activation outlier and near-inactive channel report."""

    layer_name: str
    outlier_channels: tuple[int, ...]
    near_inactive_channels: tuple[int, ...]
    outlier_fraction: float | None
    near_inactive_fraction: float | None
    status: str
    reason: str | None


@dataclass(frozen=True)
class ComparisonReport:
    """Frozen dataclass holding a complete paired-model comparison."""

    reference_model_id: str
    candidate_model_id: str
    input_fingerprint_match: bool | None
    layers: tuple[LayerComparison, ...]
    modules: tuple[ModuleComparison, ...]
    gradients: GradientComparison | None
    outliers: tuple[ActivationOutlierReport, ...]
    coverage: float | None
    measured_layer_count: int
    comparable_layer_count: int
    warnings: tuple[str, ...]


def compare_documents(
    reference: MeasurementDocument,
    candidate: MeasurementDocument,
    *,
    outlier_sigma: float = 3.0,
    near_inactive_threshold: float = 1e-3,
) -> ComparisonReport:
    ref_fp = reference.input_fingerprint
    cand_fp = candidate.input_fingerprint
    warnings: list[str] = []
    if ref_fp is not None and cand_fp is not None:
        if ref_fp != cand_fp:
            msg = f"input fingerprint mismatch: reference={ref_fp}, candidate={cand_fp}"
            raise ComparisonError(msg)
        fingerprint_match: bool | None = True
    else:
        fingerprint_match = None
        missing: list[str] = []
        if ref_fp is None:
            missing.append("reference")
        if cand_fp is None:
            missing.append("candidate")
        warnings.append(f"input fingerprint unknown on {', '.join(missing)}")

    ref_by_name = {layer.layer_name: layer for layer in reference.layers}
    cand_by_name = {layer.layer_name: layer for layer in candidate.layers}
    names = sorted(set(ref_by_name) | set(cand_by_name))

    layer_comparisons: list[LayerComparison] = []
    comparable_layer_count = 0
    measured_layer_count = 0
    for name in names:
        ref_layer = ref_by_name.get(name)
        cand_layer = cand_by_name.get(name)
        if ref_layer is None or cand_layer is None:
            absent_side = "reference" if ref_layer is None else "candidate"
            present_layer = ref_layer if ref_layer is not None else cand_layer
            if present_layer is None:
                continue
            family = present_layer.family
            layer_comparisons.append(
                LayerComparison(
                    layer_name=name,
                    family=family,
                    cosine=None,
                    normalized_reconstruction_error=None,
                    norm_ratio=None,
                    reference_norm=None,
                    candidate_norm=None,
                    status="not_measured",
                    reason=f"layer absent on {absent_side}",
                )
            )
            warnings.append(f"layer {name!r} absent on {absent_side}")
            continue
        comparable_layer_count += 1
        ref_status = ref_layer.status
        cand_status = cand_layer.status
        if ref_status == "measured" and cand_status == "measured":
            measured_layer_count += 1
            status = "measured"
            reason = None
            if ref_layer.output_cosine is None:
                reason = "output cosine undefined (zero candidate norm)"
        else:
            status = "not_measured"
            parts: list[str] = []
            if ref_status != "measured":
                parts.append(f"reference={ref_status}")
            if cand_status != "measured":
                parts.append(f"candidate={cand_status}")
            reason = "layer not measured on " + ", ".join(parts)
        layer_comparisons.append(
            LayerComparison(
                layer_name=name,
                family=ref_layer.family,
                cosine=ref_layer.output_cosine,
                normalized_reconstruction_error=ref_layer.normalized_reconstruction_error,
                norm_ratio=ref_layer.output_norm_ratio,
                reference_norm=None,
                candidate_norm=None,
                status=status,
                reason=reason,
            )
        )

    coverage: float | None = None
    if comparable_layer_count > 0:
        coverage = measured_layer_count / comparable_layer_count

    module_comparisons: list[ModuleComparison] = []
    for name in names:
        ref_layer = ref_by_name.get(name)
        cand_layer = cand_by_name.get(name)
        if ref_layer is None or cand_layer is None:
            continue
        ref_modules = {module.module_path: module for module in ref_layer.modules}
        cand_modules = {module.module_path: module for module in cand_layer.modules}
        module_names = sorted(set(ref_modules) | set(cand_modules))
        for module_name in module_names:
            ref_module = ref_modules.get(module_name)
            cand_module = cand_modules.get(module_name)
            if ref_module is None or cand_module is None:
                absent_side = "reference" if ref_module is None else "candidate"
                present_module = ref_module if ref_module is not None else cand_module
                if present_module is None:
                    continue
                module_comparisons.append(
                    ModuleComparison(
                        module_path=module_name,
                        family=present_module.family,
                        effective_weight_reconstruction_error=None,
                        relative_error=None,
                        weight_kind=present_module.weight_kind,
                        status="not_measured",
                        reason=f"module absent on {absent_side}",
                    )
                )
                warnings.append(
                    f"module {module_name!r} in layer {name!r} absent on {absent_side}"
                )
                continue
            weight_kind = ref_module.weight_kind
            if weight_kind == "packed_buffer":
                module_comparisons.append(
                    ModuleComparison(
                        module_path=module_name,
                        family=ref_module.family,
                        effective_weight_reconstruction_error=None,
                        relative_error=None,
                        weight_kind=weight_kind,
                        status="not_applicable",
                        reason=(
                            "packed buffer is not an effective weight matrix: "
                            f"kind={weight_kind}"
                        ),
                    )
                )
                warnings.append(
                    f"module {module_name!r} in layer {name!r} is a packed buffer, "
                    "no effective-weight statistic computed"
                )
                continue
            ref_value: float | None = None
            cand_value: float | None = None
            for tensor in ref_module.tensors:
                if tensor.kind == "effective_weight":
                    ref_value = tensor.value
            for tensor in cand_module.tensors:
                if tensor.kind == "effective_weight":
                    cand_value = tensor.value
            if ref_value is None or cand_value is None:
                missing_side = []
                if ref_value is None:
                    missing_side.append("reference")
                if cand_value is None:
                    missing_side.append("candidate")
                module_comparisons.append(
                    ModuleComparison(
                        module_path=module_name,
                        family=ref_module.family,
                        effective_weight_reconstruction_error=None,
                        relative_error=None,
                        weight_kind=weight_kind,
                        status="not_measured",
                        reason=(
                            "effective weight value missing on "
                            + ", ".join(missing_side)
                        ),
                    )
                )
                warnings.append(
                    f"module {module_name!r} in layer {name!r} missing effective "
                    f"weight value on {', '.join(missing_side)}"
                )
                continue
            absolute = abs(ref_value - cand_value)
            reference_abs = abs(ref_value)
            relative: float | None = None
            if reference_abs > 0.0:
                relative = absolute / reference_abs
            module_comparisons.append(
                ModuleComparison(
                    module_path=module_name,
                    family=ref_module.family,
                    effective_weight_reconstruction_error=absolute,
                    relative_error=relative,
                    weight_kind=weight_kind,
                    status="measured",
                    reason=None,
                )
            )

    gradients: GradientComparison | None = None
    ref_grad = reference.gradients
    cand_grad = candidate.gradients
    if ref_grad is None or cand_grad is None:
        missing_side = []
        if ref_grad is None:
            missing_side.append("reference")
        if cand_grad is None:
            missing_side.append("candidate")
        gradients = GradientComparison(
            status="not_measured",
            reference_grad_norm=None,
            candidate_grad_norm=None,
            grad_norm_ratio=None,
            coherence_delta=None,
            effective_rank_ratio=None,
            reason="gradients absent on " + ", ".join(missing_side),
        )
        warnings.append("gradient measurement absent on " + ", ".join(missing_side))
    elif ref_grad.status != "measured" or cand_grad.status != "measured":
        gradient_parts: list[str] = []
        if ref_grad.status != "measured":
            gradient_parts.append(
                f"reference status={ref_grad.status} "
                f"(source_type={ref_grad.source_type})"
            )
        if cand_grad.status != "measured":
            gradient_parts.append(
                f"candidate status={cand_grad.status} "
                f"(source_type={cand_grad.source_type})"
            )
        gradients = GradientComparison(
            status="not_measured",
            reference_grad_norm=None,
            candidate_grad_norm=None,
            grad_norm_ratio=None,
            coherence_delta=None,
            effective_rank_ratio=None,
            reason="gradient not measured on " + ", ".join(gradient_parts),
        )
        warnings.append("gradient not measured on " + ", ".join(gradient_parts))
    else:
        reference_grad_norm = ref_grad.grad_norm
        candidate_grad_norm = cand_grad.grad_norm
        grad_norm_ratio: float | None = None
        if (
            reference_grad_norm is not None
            and candidate_grad_norm is not None
            and reference_grad_norm != 0.0
        ):
            grad_norm_ratio = candidate_grad_norm / reference_grad_norm
        coherence_delta: float | None = None
        if ref_grad.coherence is not None and cand_grad.coherence is not None:
            coherence_delta = cand_grad.coherence - ref_grad.coherence
        effective_rank_ratio: float | None = None
        if (
            ref_grad.effective_rank is not None
            and cand_grad.effective_rank is not None
            and ref_grad.effective_rank != 0.0
        ):
            effective_rank_ratio = cand_grad.effective_rank / ref_grad.effective_rank
        gradients = GradientComparison(
            status="measured",
            reference_grad_norm=reference_grad_norm,
            candidate_grad_norm=candidate_grad_norm,
            grad_norm_ratio=grad_norm_ratio,
            coherence_delta=coherence_delta,
            effective_rank_ratio=effective_rank_ratio,
            reason=None,
        )

    outliers: tuple[ActivationOutlierReport, ...] = ()

    return ComparisonReport(
        reference_model_id=reference.model_id,
        candidate_model_id=candidate.model_id,
        input_fingerprint_match=fingerprint_match,
        layers=tuple(layer_comparisons),
        modules=tuple(module_comparisons),
        gradients=gradients,
        outliers=outliers,
        coverage=coverage,
        measured_layer_count=measured_layer_count,
        comparable_layer_count=comparable_layer_count,
        warnings=tuple(warnings),
    )


def layer_output_similarity(
    reference: torch.Tensor, candidate: torch.Tensor
) -> tuple[float | None, float, float, float]:
    if reference.ndim != 2 or candidate.ndim != 2:
        msg = (
            "tensor must be 2-D: "
            f"reference shape={tuple(reference.shape)}, "
            f"candidate shape={tuple(candidate.shape)}"
        )
        raise ComparisonError(msg)
    if reference.shape != candidate.shape:
        msg = (
            "shape mismatch: "
            f"reference shape={tuple(reference.shape)}, "
            f"candidate shape={tuple(candidate.shape)}"
        )
        raise ComparisonError(msg)
    reference_norm = float(reference.norm())
    candidate_norm = float(candidate.norm())
    if reference_norm == 0.0:
        msg = (
            "zero reference norm: "
            f"reference shape={tuple(reference.shape)}, "
            f"candidate shape={tuple(candidate.shape)}"
        )
        raise ComparisonError(msg)
    cosine: float | None
    if candidate_norm > 0.0:
        cosine = float(
            (reference * candidate).sum() / (reference_norm * candidate_norm)
        )
    else:
        cosine = None
    normalized_reconstruction_error = float(
        (candidate - reference).norm() / reference_norm
    )
    return (
        cosine,
        normalized_reconstruction_error,
        reference_norm,
        candidate_norm,
    )


def effective_weight_error(
    reference: torch.Tensor, candidate: torch.Tensor, *, kind: TensorKind
) -> tuple[float, float]:
    if kind == "packed_buffer":
        msg = (
            "effective weight error refused on packed buffer: "
            f"reference shape={tuple(reference.shape)}, "
            f"candidate shape={tuple(candidate.shape)}"
        )
        raise ComparisonError(msg)
    if reference.ndim != 2 or candidate.ndim != 2:
        msg = (
            "tensor must be 2-D: "
            f"reference shape={tuple(reference.shape)}, "
            f"candidate shape={tuple(candidate.shape)}"
        )
        raise ComparisonError(msg)
    if reference.shape != candidate.shape:
        msg = (
            "shape mismatch: "
            f"reference shape={tuple(reference.shape)}, "
            f"candidate shape={tuple(candidate.shape)}"
        )
        raise ComparisonError(msg)
    reference_norm = float(reference.norm())
    if reference_norm == 0.0:
        msg = (
            "zero reference norm: "
            f"reference shape={tuple(reference.shape)}, "
            f"candidate shape={tuple(candidate.shape)}"
        )
        raise ComparisonError(msg)
    absolute = float((candidate - reference).norm())
    relative = absolute / reference_norm
    return (absolute, relative)


def activation_outliers(
    activations: torch.Tensor, *, sigma: float, near_inactive_threshold: float
) -> ActivationOutlierReport:
    if activations.ndim != 2:
        return ActivationOutlierReport(
            layer_name="",
            outlier_channels=(),
            near_inactive_channels=(),
            outlier_fraction=None,
            near_inactive_fraction=None,
            status="not_measured",
            reason="activations must be 2-D",
        )
    num_channels = activations.shape[1]
    column_means = activations.abs().mean(dim=0)
    column_maxs = activations.abs().max(dim=0).values
    mean_of_means = float(column_means.mean())
    std_of_means = float(column_means.std(unbiased=True)) if num_channels > 1 else 0.0
    outlier_threshold = mean_of_means + sigma * std_of_means
    outlier_mask = column_means > outlier_threshold
    near_inactive_mask = column_maxs < near_inactive_threshold
    outlier_channels = tuple(
        int(index) for index in torch.nonzero(outlier_mask).flatten().tolist()
    )
    near_inactive_channels = tuple(
        int(index) for index in torch.nonzero(near_inactive_mask).flatten().tolist()
    )
    outlier_fraction: float | None = None
    near_inactive_fraction: float | None = None
    if num_channels > 0:
        outlier_fraction = len(outlier_channels) / num_channels
        near_inactive_fraction = len(near_inactive_channels) / num_channels
    return ActivationOutlierReport(
        layer_name="",
        outlier_channels=outlier_channels,
        near_inactive_channels=near_inactive_channels,
        outlier_fraction=outlier_fraction,
        near_inactive_fraction=near_inactive_fraction,
        status="measured",
        reason=None,
    )


def token_level_kl(
    teacher_logits: torch.Tensor,
    student_logits: torch.Tensor,
    *,
    temperature: float = 1.0,
) -> float:
    if teacher_logits.shape[-1] != student_logits.shape[-1]:
        msg = (
            "vocabulary size mismatch: "
            f"teacher vocab={teacher_logits.shape[-1]}, "
            f"student vocab={student_logits.shape[-1]}"
        )
        raise ComparisonError(msg)
    if temperature <= 0.0:
        msg = f"temperature must be positive, got {temperature}"
        raise ComparisonError(msg)
    log_p_t = torch.log_softmax(teacher_logits / temperature, dim=-1)
    log_p_s = torch.log_softmax(student_logits / temperature, dim=-1)
    p_t = log_p_t.exp()
    kl = (p_t * (log_p_t - log_p_s)).sum(dim=-1).mean()
    return float(kl)


def cross_entropy_delta(teacher_loss: float, student_loss: float) -> float:
    return student_loss - teacher_loss
