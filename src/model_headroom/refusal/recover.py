"""Recover a refusal direction from weight deltas of an already-abliterated model.

No activations and no forward pass are required. An abliteration following
Arditi et al. orthogonalises the refusal direction ``r`` (a vector in the
residual/output space of dimension ``out_features``) out of every matrix that
writes to the residual stream::

    W' = W - r (r^T W)   =>   delta = W' - W = - r (W^T r)^T

so ``delta`` is rank one with left factor ``r`` (length ``out_features``) and
right factor ``W^T r`` (length ``in_features``). ``r`` therefore lives in the
output/residual space, so the refusal direction is the top **left** singular
vector of the horizontally-stacked deltas, whose length is ``out_features``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from model_headroom.models.types import (
    RefusalDirectionReport,
    RefusalDirectionSpec,
    RefusalLayerStat,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

_EPS = 1e-12


def recover_direction_from_weight_deltas(
    deltas: Sequence[torch.Tensor],
    spec: RefusalDirectionSpec | None = None,
    device: str = "cpu",
) -> RefusalDirectionReport:
    """Recover the refusal direction as the top left singular vector of ``deltas``.

    Each delta must be 2-D ``[out_features, in_features]``. Returns status
    ``error`` when fewer than two matrices are supplied or a shape is wrong,
    ``degenerate`` when the stacked deltas carry no variance, and ``computed``
    otherwise. The leading singular value is reported as ``cohen_d``'s stand-in
    via ``direction_norm`` so callers can see how dominant the rank-1 component
    is before trusting it. The stacked deltas and SVD run on ``device``; the
    returned ``direction`` is moved back to CPU so the serialized report keeps a
    device-independent type that callers can convert with ``.numpy()``.
    """
    spec = spec or RefusalDirectionSpec()

    if len(deltas) < 2:
        return RefusalDirectionReport(
            axis="weight_delta_svd",
            hidden_size=0,
            harmful_count=0,
            harmless_count=0,
            layer_stats=[],
            selection="weight_delta_svd",
            direction_status="error",
            error_message=(
                f"need at least 2 weight deltas to recover a direction, got {len(deltas)}"
            ),
        )

    heights: set[int] = set()
    for delta in deltas:
        if delta.ndim != 2:
            return RefusalDirectionReport(
                axis="weight_delta_svd",
                hidden_size=0,
                harmful_count=0,
                harmless_count=0,
                layer_stats=[],
                selection="weight_delta_svd",
                direction_status="error",
                error_message=(
                    f"weight deltas must be 2-D [out, in]; got shape "
                    f"{list(delta.shape)}"
                ),
            )
        heights.add(int(delta.shape[0]))

    if len(heights) != 1:
        return RefusalDirectionReport(
            axis="weight_delta_svd",
            hidden_size=0,
            harmful_count=0,
            harmless_count=0,
            layer_stats=[],
            selection="weight_delta_svd",
            direction_status="error",
            error_message=(
                f"weight deltas disagree on out_features: got sorted heights {sorted(heights)}"
            ),
        )

    hidden = heights.pop()
    # Arditi abliteration lives in the output/residual space, so the refusal
    # direction is the top LEFT singular vector: stack deltas horizontally so
    # the left factor has length out_features (== hidden). Compute on the
    # requested device; the returned direction comes back to CPU.
    stacked = torch.cat([d.to(dtype=torch.float32) for d in deltas], dim=1).to(device)

    if not torch.isfinite(stacked).all():
        return RefusalDirectionReport(
            axis="weight_delta_svd",
            hidden_size=hidden,
            harmful_count=len(deltas),
            harmless_count=0,
            layer_stats=[],
            selection="weight_delta_svd",
            direction_status="error",
            error_message="weight deltas contain non-finite values",
        )

    try:
        u, singular, _vh = torch.linalg.svd(stacked, full_matrices=False)
    except RuntimeError as exc:
        return RefusalDirectionReport(
            axis="weight_delta_svd",
            hidden_size=hidden,
            harmful_count=len(deltas),
            harmless_count=0,
            layer_stats=[],
            selection="weight_delta_svd",
            direction_status="error",
            error_message=f"SVD failed on device {device}: {exc}; retry with --device cpu",
        )

    leading = float(singular[0]) if singular.numel() else 0.0
    if leading < _EPS:
        return RefusalDirectionReport(
            axis="weight_delta_svd",
            hidden_size=hidden,
            harmful_count=len(deltas),
            harmless_count=0,
            layer_stats=[],
            selection="weight_delta_svd",
            direction_status="degenerate",
            error_message="stacked weight deltas have a zero leading singular value",
        )

    direction = u[:, 0]
    # normalise to unit length (SVD vector is already unit, but be explicit)
    direction = direction / direction.norm().clamp(min=_EPS)
    # The report is serialized to JSON and callers may call .numpy(), so the
    # direction comes home to CPU: compute on the accelerator, return on host.
    direction = direction.to("cpu")
    energy = float((singular**2).sum())
    dominance = (leading**2) / energy if energy > _EPS else 0.0

    stat = RefusalLayerStat(
        layer_name=f"{len(deltas)} weight deltas",
        layer_index=-1,
        harmful_count=len(deltas),
        harmless_count=0,
        cohen_d=dominance,
        mean_norm=leading,
        masked_count=0,
        direction_norm=leading,
        orthogonality_with_harmless_mean=1.0,
    )

    return RefusalDirectionReport(
        axis="weight_delta_svd",
        direction=direction,
        layer_name=stat.layer_name,
        layer_index=-1,
        hidden_size=hidden,
        harmful_count=len(deltas),
        harmless_count=0,
        cohen_d=dominance,
        orthogonalized=False,
        massive_activation_sigma=spec.massive_activation_sigma,
        masked_count=0,
        layer_stats=[stat],
        selection="weight_delta_svd",
        direction_status="computed",
    )


def load_weight_deltas_from_json(json_path: str) -> list[torch.Tensor] | None:
    """Load weight-delta matrices from JSON.

    Accepts ``{"deltas": [...]}`` or a bare list of 2-D arrays. Returns None when
    the file is missing, malformed, or holds no usable 2-D matrix.
    """
    import json
    from pathlib import Path

    try:
        with Path(json_path).open() as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None

    entries = data.get("deltas", data) if isinstance(data, dict) else data
    if not isinstance(entries, list):
        return None

    deltas: list[torch.Tensor] = []
    for entry in entries:
        matrix = entry.get("delta") if isinstance(entry, dict) else entry
        if matrix is None:
            continue
        tensor = torch.tensor(matrix, dtype=torch.float32)
        if tensor.ndim == 2 and tensor.numel() > 0:
            deltas.append(tensor)
    return deltas if deltas else None
