from __future__ import annotations

import math
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from model_headroom.refusal.extract import normalize

if TYPE_CHECKING:
    from model_headroom.models.types import RefusalDirectionReport

ARTIFACT_SCHEMA_VERSION = "1"
TENSOR_REFUSAL_DIRECTION = "refusal_direction"
TENSOR_DOMAIN_DIRECTION = "domain_direction"
_UNIT_TOLERANCE = 1e-5


def write_refusal_artifact(
    report: RefusalDirectionReport,
    path: str | Path,
    *,
    model_id: str | None = None,
    domain_direction: torch.Tensor | None = None,
    domain_label: str | None = None,
    timestamp: str | None = None,
) -> Path:
    if report.direction is None:
        raise ValueError(
            "report.direction is None: no refusal direction computed; refusing to write artifact"
        )
    out_path = Path(path)
    direction = report.direction.detach().to(torch.float32).contiguous()
    pre_norm = float(direction.norm())
    if abs(pre_norm - 1.0) > _UNIT_TOLERANCE:
        direction = normalize(direction)
    domain: torch.Tensor | None = None
    if domain_direction is not None:
        domain = domain_direction.detach().to(torch.float32).contiguous()
        if int(domain.shape[0]) != int(report.hidden_size):
            raise ValueError(
                f"domain_direction length {int(domain.shape[0])} != "
                f"report.hidden_size {int(report.hidden_size)}"
            )
        if abs(float(domain.norm()) - 1.0) > _UNIT_TOLERANCE:
            domain = normalize(domain)
    metadata: dict[str, str] = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "model_id": model_id if model_id is not None else "",
        "layer_name": report.layer_name or "",
        "layer_index": (
            str(report.layer_index) if report.layer_index is not None else ""
        ),
        "hidden_size": str(report.hidden_size),
        "capture_point": (
            "weight_delta"
            if report.axis == "weight_delta_svd"
            else "last_token_residual"
        ),
        "source": (
            "weight_delta_recovery"
            if report.axis == "weight_delta_svd"
            else "base_model_activation_difference"
        ),
        "normalized": "true",
        "orthogonalized": "true" if report.orthogonalized else "false",
        "cohen_d": (
            "nan"
            if report.cohen_d is None or math.isnan(report.cohen_d)
            else repr(float(report.cohen_d))
        ),
        "cohen_d_kind": (
            "svd_dominance" if report.axis == "weight_delta_svd" else "effect_size"
        ),
        "harmful_count": str(report.harmful_count),
        "harmless_count": str(report.harmless_count),
        "selection": report.selection,
        "direction_status": report.direction_status,
        "axis": report.axis,
        "direction_norm": repr(float(pre_norm)),
        "has_domain_direction": "true" if domain is not None else "false",
        "domain_label": domain_label or "",
        "timestamp": (
            timestamp
            if timestamp is not None
            else datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
        ),
    }
    tensors: dict[str, torch.Tensor] = {TENSOR_REFUSAL_DIRECTION: direction}
    if domain is not None:
        tensors[TENSOR_DOMAIN_DIRECTION] = domain
    out_path.parent.mkdir(parents=True, exist_ok=True)
    from safetensors.torch import save_file

    save_file(tensors, str(out_path), metadata=metadata)
    return out_path
