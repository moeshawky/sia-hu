"""Rendering for refusal-direction reports. Human-readable text and JSON."""

from __future__ import annotations

import json
import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from model_headroom.models.types import RefusalDirectionReport

_WIDTH = 78

_LIMITATIONS = (
    "A refusal direction is a rank-1 approximation. Check cohen_d before use: a "
    "value near 0 means the two prompt sets were not separated at this layer, and "
    "no direction should be applied from it."
)


def _row(label: str, value: object) -> str:
    return f"  {label:<28} {value}"


def generate_refusal_report(report: RefusalDirectionReport) -> str:
    """Render a refusal-direction report as terminal text."""
    lines: list[str] = []
    lines.append("=" * _WIDTH)
    lines.append("REFUSAL DIRECTION".center(_WIDTH))
    lines.append("=" * _WIDTH)
    lines.append("")
    lines.append("Acquisition")
    lines.append(_row("axis", report.axis))
    lines.append(_row("status", report.direction_status))
    lines.append(_row("layer selection", report.selection))
    lines.append(_row("layer", report.layer_name or "-"))
    lines.append(
        _row("layer index", "-" if report.layer_index is None else report.layer_index)
    )
    lines.append("")

    lines.append("Data")
    lines.append(_row("hidden size", report.hidden_size))
    lines.append(_row("harmful prompts", report.harmful_count))
    lines.append(_row("harmless prompts", report.harmless_count))
    lines.append(_row("masked rows", report.masked_count))
    lines.append(
        _row("massive-activation sigma", report.massive_activation_sigma or "off")
    )
    lines.append(_row("winsorize quantile", report.winsorize_quantile or "off"))
    lines.append("")

    lines.append("Direction")
    if report.direction is None:
        lines.append(_row("direction", "not computed"))
    else:
        lines.append(_row("unit norm", f"{float(report.direction.norm()):.6f}"))
        lines.append(_row("orthogonalized", report.orthogonalized))
    if report.cohen_d is None:
        lines.append(_row("cohen d", "-"))
    else:
        lines.append(_row("cohen d", f"{report.cohen_d:.4f}"))
    lines.append("")

    if report.layer_stats:
        lines.append("Layer sweep (cohen d, higher = better separation)")
        lines.append(
            f"  {'layer':<34} {'idx':>4} {'n_h':>5} {'n_s':>5} {'cohen_d':>10} {'orth':>8}"
        )
        lines.append("  " + "-" * (_WIDTH - 4))
        for stat in sorted(
            report.layer_stats,
            key=lambda s: s.cohen_d if s.cohen_d == s.cohen_d else float("-inf"),
            reverse=True,
        ):
            lines.append(
                f"  {stat.layer_name[:33]:<34} {stat.layer_index:>4} "
                f"{stat.harmful_count:>5} {stat.harmless_count:>5} "
                f"{stat.cohen_d:>10.4f} {stat.orthogonality_with_harmless_mean:>8.4f}"
            )
        lines.append("")

    if report.error_message:
        lines.append("Note")
        lines.append(f"  {report.error_message}")
        lines.append("")

    lines.append("Limitations")
    for line in _wrap(_LIMITATIONS, _WIDTH - 4):
        lines.append(f"  {line}")
    lines.append("=" * _WIDTH)
    return "\n".join(lines)


def _wrap(text: str, width: int) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current: list[str] = []
    for word in words:
        if current and sum(len(w) + 1 for w in current) + len(word) > width:
            lines.append(" ".join(current))
            current = [word]
        else:
            current.append(word)
    if current:
        lines.append(" ".join(current))
    return lines


def generate_refusal_json(report: RefusalDirectionReport) -> str:
    """Render a refusal-direction report as JSON.

    The unit direction vector is emitted as a list of floats under
    ``refusal_direction`` so the artifact is compatible with the
    QWEN38_ABLITERATION runtime loader. ``capture_point``, ``source``, and
    ``normalized`` describe how the direction was computed.
    """
    payload: dict[str, object] = {
        "axis": report.axis,
        "direction_status": report.direction_status,
        "selection": report.selection,
        "layer_name": report.layer_name,
        "layer_index": report.layer_index,
        "hidden_size": report.hidden_size,
        "harmful_count": report.harmful_count,
        "harmless_count": report.harmless_count,
        "masked_count": report.masked_count,
        "massive_activation_sigma": report.massive_activation_sigma,
        "winsorize_quantile": report.winsorize_quantile,
        "orthogonalized": report.orthogonalized,
        "cohen_d": report.cohen_d,
        "error_message": report.error_message,
        "capture_point": "last_token_residual",
        "source": "base_model_activation_difference",
        "normalized": True,
        "layer_stats": [
            {
                "layer_name": s.layer_name,
                "layer_index": s.layer_index,
                "harmful_count": s.harmful_count,
                "harmless_count": s.harmless_count,
                "cohen_d": None if math.isnan(s.cohen_d) else s.cohen_d,
                "mean_norm": s.mean_norm,
                "masked_count": s.masked_count,
                "direction_norm": s.direction_norm,
                "orthogonality_with_harmless_mean": s.orthogonality_with_harmless_mean,
            }
            for s in report.layer_stats
        ],
    }
    if report.direction is not None:
        payload["refusal_direction"] = [float(v) for v in report.direction.tolist()]
    return json.dumps(payload, indent=2)
