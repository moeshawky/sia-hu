"""Refusal-direction extraction math. Pure tensor functions: no model, no filesystem.

The direction is the normalised difference of means between a harmful and a
harmless prompt set, computed on residual-stream activations at one layer.
Everything here is model-agnostic: layers are ordered by their position in the
supplied activation sets, never by a vendor-specific name.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from model_headroom.models.types import (
    RefusalDirectionReport,
    RefusalDirectionSpec,
    RefusalLayerStat,
    ResidualBatch,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

_EPS = 1e-12


def normalize(vector: torch.Tensor) -> torch.Tensor:
    """Return the unit vector along ``vector``. Raises ValueError on zero magnitude."""
    norm = float(vector.norm())
    if norm < _EPS:
        raise ValueError("cannot normalize a zero-magnitude vector")
    return vector / norm


def row_norms(batch: torch.Tensor) -> torch.Tensor:
    """Return the L2 norm of every row of a 2-D activation batch."""
    norms: torch.Tensor = batch.norm(dim=1)
    return norms


def massive_activation_mask(batch: torch.Tensor, sigma: float) -> torch.Tensor:
    """Boolean mask of rows whose L2 norm exceeds mean + sigma * std.

    A batch with fewer than two rows has no dispersion to threshold against, so
    nothing is masked rather than everything.
    """
    norms = row_norms(batch)
    if norms.numel() < 2:
        return torch.zeros(norms.numel(), dtype=torch.bool)
    threshold = float(norms.mean()) + sigma * float(norms.std(unbiased=True))
    return norms > threshold


def winsorize_residuals(batch: torch.Tensor, quantile: float) -> torch.Tensor:
    """Symmetrically clamp every component of each row to its own quantile.

    For each prompt independently, compute the ``quantile`` of that row's
    absolute components and clamp all components to +/- it. This tames the
    "massive activations" a few coordinates exhibit without discarding the
    prompt the way masking whole rows does. A quantile of 1.0 is a no-op.
    """
    if not 0.0 <= quantile < 1.0:
        return batch
    thresholds = torch.quantile(batch.abs(), quantile, dim=1, keepdim=True)
    return torch.clamp(batch, -thresholds, thresholds)


def difference_of_means(harmful: torch.Tensor, harmless: torch.Tensor) -> torch.Tensor:
    """Mean difference ``mu_harmful - mu_harmless``, accumulated in float64.

    The residuals originate in bfloat16 on the serving path; summing thousands
    of them in float32 loses precision in the mean, which is the only quantity
    the direction is made of.
    """
    return harmful.double().mean(dim=0) - harmless.double().mean(dim=0)


def row_means(
    harmful: torch.Tensor, harmless: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-row mean vectors as float32, for effect-size reporting."""
    return (
        harmful.double().mean(dim=0).to(torch.float32),
        harmless.double().mean(dim=0).to(torch.float32),
    )


def cohens_d_along(
    harmful: torch.Tensor, harmless: torch.Tensor, direction: torch.Tensor
) -> float:
    """One-dimensional Cohen's d of the two sets projected onto ``direction``.

    Returns NaN when either set has fewer than two rows or the pooled standard
    deviation is zero: those cases have no defined effect size, and reporting 0.0
    would read as "measured, no separation" rather than "not measurable".
    """
    proj_h = harmful @ direction
    proj_s = harmless @ direction
    n_h = proj_h.numel()
    n_s = proj_s.numel()
    if n_h < 2 or n_s < 2:
        return math.nan
    var_h = proj_h.var(unbiased=True)
    var_s = proj_s.var(unbiased=True)
    pooled = ((n_h - 1) * var_h + (n_s - 1) * var_s) / (n_h + n_s - 2)
    denom = float(pooled.clamp(min=0.0).sqrt())
    if denom < _EPS:
        return math.nan
    return float(proj_h.mean() - proj_s.mean()) / denom


def orthogonalize_against(
    direction: torch.Tensor, reference: torch.Tensor
) -> torch.Tensor:
    """Remove the ``reference`` component from ``direction``.

    Projecting a residual stream along a direction that still points at the
    harmless mean removes helpfulness along with refusal. A zero-magnitude
    reference carries no direction to remove, so the input is returned unchanged.
    """
    ref_norm = float(reference.norm())
    if ref_norm < _EPS:
        return direction
    unit_ref = reference / ref_norm
    return direction - float(direction @ unit_ref) * unit_ref


def project_out(
    residual: torch.Tensor, direction: torch.Tensor, strength: float = 1.0
) -> torch.Tensor:
    """Subtract ``strength`` times the ``direction`` component from ``residual``.

    Reduces over the last dimension, so each leading index keeps its own
    component: it applies to a single vector, to [batch, hidden], or to a full
    [batch, seq, hidden] residual stream, removing the direction from every row
    independently rather than removing one shared row's component.
    """
    component = (residual * direction).sum(dim=-1, keepdim=True)
    return residual - strength * component * direction


def _ordered_index(
    batches: Sequence[ResidualBatch],
) -> dict[str, ResidualBatch]:
    """Index batches by layer name, keeping first-appearance order."""
    indexed: dict[str, ResidualBatch] = {}
    for batch in batches:
        indexed.setdefault(batch.layer_name, batch)
    return indexed


def _prepare(
    batch: ResidualBatch, spec: RefusalDirectionSpec
) -> tuple[torch.Tensor, int]:
    """Apply massive-activation masking. Returns the kept rows and the masked count."""
    activations = batch.activations
    if activations.dtype != torch.float32:
        activations = activations.to(torch.float32)
    if spec.winsorize_quantile is not None:
        activations = winsorize_residuals(activations, spec.winsorize_quantile)
    if spec.massive_activation_sigma is None:
        return activations, 0
    mask = massive_activation_mask(activations, spec.massive_activation_sigma)
    return activations[~mask], int(mask.sum().item())


def _width(batch: ResidualBatch) -> int | None:
    """Hidden width of a batch, or None when the tensor is not 2-D."""
    if batch.activations.ndim != 2:
        return None
    return int(batch.activations.shape[1])


def _layer_stat(
    index: int,
    name: str,
    harmful: ResidualBatch,
    harmless: ResidualBatch,
    spec: RefusalDirectionSpec,
) -> tuple[RefusalLayerStat, torch.Tensor | None, torch.Tensor | None]:
    """Measure one layer. Returns the stat plus the kept harmful/harmless rows.

    The direction is None when the layer cannot be measured; the stat still
    reports counts so the caller can see why.
    """
    h_width = _width(harmful)
    s_width = _width(harmless)
    hidden = h_width if h_width is not None else s_width

    kept_h, masked_h = _prepare(harmful, spec)
    kept_s, masked_s = _prepare(harmless, spec)

    masked = masked_h + masked_s
    n_h = int(kept_h.shape[0])
    n_s = int(kept_s.shape[0])

    def _empty(d: float) -> RefusalLayerStat:
        return RefusalLayerStat(
            layer_name=name,
            layer_index=index,
            harmful_count=n_h,
            harmless_count=n_s,
            cohen_d=d,
            mean_norm=0.0,
            masked_count=masked,
            direction_norm=0.0,
            orthogonality_with_harmless_mean=0.0,
        )

    if hidden is None or h_width is None or s_width is None or h_width != s_width:
        return (_empty(math.nan), None, None)
    if n_h == 0 or n_s == 0:
        return (_empty(math.nan), None, None)
    if not torch.isfinite(kept_h).all() or not torch.isfinite(kept_s).all():
        return (_empty(math.nan), None, None)

    raw = difference_of_means(kept_h, kept_s).to(torch.float32)
    raw_norm = float(raw.norm())
    if raw_norm < _EPS:
        return (_empty(0.0), None, None)

    unit = raw / raw_norm
    _, harmless_mean = row_means(kept_h, kept_s)
    orthogonality = abs(float(unit @ harmless_mean) / float(harmless_mean.norm()))

    stat = RefusalLayerStat(
        layer_name=name,
        layer_index=index,
        harmful_count=n_h,
        harmless_count=n_s,
        cohen_d=cohens_d_along(kept_h, kept_s, unit),
        mean_norm=float(
            torch.stack([row_norms(kept_h).mean(), row_norms(kept_s).mean()]).mean()
        ),
        masked_count=masked,
        direction_norm=raw_norm,
        orthogonality_with_harmless_mean=orthogonality,
    )
    return (stat, kept_h, kept_s)


def _select_index(shared: Sequence[str], spec: RefusalDirectionSpec) -> tuple[int, str]:
    """Resolve the requested layer to an index into the shared layer ordering."""
    last = len(shared) - 1
    if last < 0:
        raise ValueError("no layers to select from")
    if spec.layer_index is not None:
        if not 0 <= spec.layer_index <= last:
            raise ValueError(
                f"layer_index {spec.layer_index} outside the shared layer range "
                f"0..{last}"
            )
        return (spec.layer_index, "layer_index")
    if spec.layer_name is not None:
        if spec.layer_name not in shared:
            raise ValueError(
                f"layer {spec.layer_name!r} is not present in both activation sets"
            )
        return (list(shared).index(spec.layer_name), "layer_name")
    return (round(spec.depth_fraction * last), "depth_fraction")


def compute_refusal_direction(
    harmful: Sequence[ResidualBatch],
    harmless: Sequence[ResidualBatch],
    spec: RefusalDirectionSpec,
) -> RefusalDirectionReport:
    """Compute the refusal direction from two contrastive residual-activation sets.

    Every layer present in both sets is measured and returned in
    ``layer_stats`` regardless of which one is selected, so the measured sweep is
    available even when an explicit layer is requested. Returns status
    ``insufficient_data`` and a None direction when the selected layer has fewer
    than ``spec.min_pairs`` prompts on either side.
    """
    h_by = _ordered_index(harmful)
    s_by = _ordered_index(harmless)
    shared = [name for name in h_by if name in s_by]

    if not shared:
        return RefusalDirectionReport(
            axis="mean_difference",
            hidden_size=0,
            harmful_count=0,
            harmless_count=0,
            layer_stats=[],
            selection="none",
            direction_status="insufficient_data",
            error_message=(
                "no layer name is present in both the harmful and harmless "
                f"activation sets (harmful={list(h_by)[:5]}, "
                f"harmless={list(s_by)[:5]})"
            ),
        )

    stats: list[RefusalLayerStat] = []
    rows: list[tuple[torch.Tensor | None, torch.Tensor | None]] = []
    for index, name in enumerate(shared):
        stat, kept_h, kept_s = _layer_stat(index, name, h_by[name], s_by[name], spec)
        stats.append(stat)
        rows.append((kept_h, kept_s))

    try:
        chosen, how = _select_index(shared, spec)
    except ValueError as exc:
        return RefusalDirectionReport(
            axis="mean_difference",
            hidden_size=0,
            harmful_count=sum(s.harmful_count for s in stats),
            harmless_count=sum(s.harmless_count for s in stats),
            layer_stats=stats,
            selection="none",
            direction_status="error",
            error_message=str(exc),
        )

    stat = stats[chosen]
    kept_h, kept_s = rows[chosen]
    total_masked = sum(s.masked_count for s in stats)

    def _incomplete(message: str) -> RefusalDirectionReport:
        return RefusalDirectionReport(
            axis="mean_difference",
            hidden_size=kept_h.shape[1] if kept_h is not None else 0,
            harmful_count=stat.harmful_count,
            harmless_count=stat.harmless_count,
            layer_stats=stats,
            layer_name=stat.layer_name,
            layer_index=stat.layer_index,
            cohen_d=stat.cohen_d,
            massive_activation_sigma=spec.massive_activation_sigma,
            winsorize_quantile=spec.winsorize_quantile,
            masked_count=total_masked,
            selection=how,
            direction_status="insufficient_data",
            error_message=message,
        )

    if kept_h is None or kept_s is None:
        return _incomplete(
            f"layer {stat.layer_name!r} has no measurable direction "
            f"(harmful={stat.harmful_count}, harmless={stat.harmless_count}, "
            f"masked={stat.masked_count})"
        )

    if stat.harmful_count < spec.min_pairs or stat.harmless_count < spec.min_pairs:
        return _incomplete(
            f"layer {stat.layer_name!r} has {stat.harmful_count} harmful and "
            f"{stat.harmless_count} harmless prompts, below min_pairs="
            f"{spec.min_pairs}"
        )

    means_h, means_s = row_means(kept_h, kept_s)
    direction = normalize(means_h - means_s)
    if spec.orthogonalize:
        direction = normalize(orthogonalize_against(direction, means_s))

    return RefusalDirectionReport(
        axis="mean_difference",
        direction=direction,
        layer_name=stat.layer_name,
        layer_index=stat.layer_index,
        hidden_size=int(direction.shape[0]),
        harmful_count=stat.harmful_count,
        harmless_count=stat.harmless_count,
        cohen_d=cohens_d_along(kept_h, kept_s, direction),
        orthogonalized=spec.orthogonalize,
        massive_activation_sigma=spec.massive_activation_sigma,
        winsorize_quantile=spec.winsorize_quantile,
        masked_count=total_masked,
        layer_stats=stats,
        selection=how,
        direction_status="computed",
    )
