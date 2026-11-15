"""Refusal comparison across reference, compressed, and recovered checkpoints.

Response-level refusal outcomes (refusal counts and rates on prompt sets) and
direction-level geometry (cosine to the reference direction, Cohen's d, norms,
orthogonality residuals) live in separate frozen dataclasses and are never
combined into a single score. A DirectionShift carries no verdict field, no
safety_preserved boolean, and no safety/quality scalar. An unmeasured rate is
None, never 0.0. An absent direction produces status="not_measured" rather than
a fabricated cosine. geometry_without_behaviour is True when at least one
DirectionShift is measured while the response-level rate for that side is
absent, unmeasured-status, or None — geometry without behaviour is never
interpreted as evidence of safety change. prompt_fingerprint_match is True when
both fingerprints are present and equal, False when both are present and differ,
and None when either fingerprint is unknown — absence is not reported as
agreement.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from collections.abc import Sequence

    import torch

    from model_headroom.models.types import RefusalDirectionReport
    from model_headroom.paired.schema import (
        MeasurementDocument,
        MeasurementStatus,
    )

from model_headroom.refusal import extract
from model_headroom.refusal.activations import load_prompts


@dataclass(frozen=True)
class PromptSetSpec:
    """Frozen dataclass holding fixed harmful/harmless prompt sets and their sha256 fingerprint."""

    harmful_prompts: tuple[str, ...]
    harmless_prompts: tuple[str, ...]
    fingerprint: str


@dataclass(frozen=True)
class RefusalOutcome:
    """Frozen dataclass holding one role's response-level refusal result on one prompt set."""

    role: Literal["reference", "compressed", "recovered"]
    set_kind: Literal["harmful", "harmless"]
    prompt_count: int
    refusal_count: int | None
    refusal_rate: float | None
    status: MeasurementStatus
    reason: str | None


@dataclass(frozen=True)
class DirectionShift:
    """Frozen dataclass holding one layer's geometry-level shift (reference vs candidate)."""

    layer_name: str
    cosine_to_reference: float | None
    cohen_d_reference: float | None
    cohen_d_candidate: float | None
    cohen_d_delta: float | None
    norm_ratio: float | None
    orthogonalization_residual: float | None
    status: MeasurementStatus
    reason: str | None


@dataclass(frozen=True)
class RefusalComparison:
    """Frozen dataclass holding a complete refusal comparison across prompt sets and layers."""

    prompt_fingerprint_match: bool | None
    outcomes: tuple[RefusalOutcome, ...]
    shifts: tuple[DirectionShift, ...]
    response_level_change: float | None
    overrefusal_change: float | None
    geometry_without_behaviour: bool
    warnings: tuple[str, ...]


def parse_prompt_set(harmful_path: str, harmless_path: str) -> PromptSetSpec:
    """Load two prompt sets and return a PromptSetSpec with a sha256 fingerprint of both."""
    harmful = load_prompts(harmful_path)
    harmless = load_prompts(harmless_path)
    harmful_tuple: tuple[str, ...] = tuple(harmful) if harmful is not None else ()
    harmless_tuple: tuple[str, ...] = tuple(harmless) if harmless is not None else ()
    joined = "\n".join(sorted(harmful_tuple + harmless_tuple))
    fingerprint = hashlib.sha256(joined.encode("utf-8")).hexdigest()
    return PromptSetSpec(
        harmful_prompts=harmful_tuple,
        harmless_prompts=harmless_tuple,
        fingerprint=fingerprint,
    )


def compare_refusal_directions(
    reference: RefusalDirectionReport, candidate: RefusalDirectionReport
) -> tuple[DirectionShift, ...]:
    """Pair layers by name between two reports and compute geometry-level shifts."""
    shifts: list[DirectionShift] = []
    ref_by_name = {stat.layer_name: stat for stat in reference.layer_stats}
    cand_by_name = {stat.layer_name: stat for stat in candidate.layer_stats}
    names = sorted(set(ref_by_name) | set(cand_by_name))
    ref_direction = reference.direction
    cand_direction = candidate.direction
    for name in names:
        ref_stat = ref_by_name.get(name)
        cand_stat = cand_by_name.get(name)
        if ref_direction is None or cand_direction is None:
            shifts.append(
                DirectionShift(
                    layer_name=name,
                    cosine_to_reference=None,
                    cohen_d_reference=ref_stat.cohen_d
                    if ref_stat is not None
                    else None,
                    cohen_d_candidate=cand_stat.cohen_d
                    if cand_stat is not None
                    else None,
                    cohen_d_delta=None,
                    norm_ratio=None,
                    orthogonalization_residual=None,
                    status="not_measured",
                    reason="direction absent on reference or candidate",
                )
            )
            continue
        cosine = _cosine(ref_direction, cand_direction)
        norm_ratio = _norm_ratio(ref_direction, cand_direction)
        orth_res = _orthogonalization_residual(ref_direction, cand_direction)
        cohen_d_delta: float | None = None
        if ref_stat is not None and cand_stat is not None:
            rd = ref_stat.cohen_d
            cd = cand_stat.cohen_d
            if rd == rd and cd == cd:
                cohen_d_delta = cd - rd
        shifts.append(
            DirectionShift(
                layer_name=name,
                cosine_to_reference=cosine,
                cohen_d_reference=ref_stat.cohen_d if ref_stat is not None else None,
                cohen_d_candidate=cand_stat.cohen_d if cand_stat is not None else None,
                cohen_d_delta=cohen_d_delta,
                norm_ratio=norm_ratio,
                orthogonalization_residual=orth_res,
                status="measured",
                reason=None,
            )
        )
    return tuple(shifts)


def compare_outcomes(
    reference: RefusalOutcome,
    candidate: RefusalOutcome,
    *,
    set_kind: Literal["harmful", "harmless"],
) -> float | None:
    """Return candidate.refusal_rate - reference.refusal_rate, or None when either is unmeasured."""
    if reference.set_kind != set_kind or candidate.set_kind != set_kind:
        return None
    if reference.refusal_rate is None or candidate.refusal_rate is None:
        return None
    return candidate.refusal_rate - reference.refusal_rate


def compare_refusal(
    reference: MeasurementDocument,
    candidate: MeasurementDocument,
    *,
    reference_outcomes: Sequence[RefusalOutcome],
    candidate_outcomes: Sequence[RefusalOutcome],
    reference_directions: Sequence[RefusalDirectionReport] | None = None,
    candidate_directions: Sequence[RefusalDirectionReport] | None = None,
) -> RefusalComparison:
    """Build a RefusalComparison from two measurement documents and their outcome/direction sequences."""
    warnings: list[str] = []
    ref_by_set = {o.set_kind: o for o in reference_outcomes}
    cand_by_set = {o.set_kind: o for o in candidate_outcomes}

    outcome_list: list[RefusalOutcome] = []
    for o in reference_outcomes:
        outcome_list.append(o)
    for o in candidate_outcomes:
        outcome_list.append(o)

    harmful_ref = ref_by_set.get("harmful")
    harmful_cand = cand_by_set.get("harmful")
    harmless_ref = ref_by_set.get("harmless")
    harmless_cand = cand_by_set.get("harmless")

    response_level_change = (
        compare_outcomes(harmful_ref, harmful_cand, set_kind="harmful")
        if harmful_ref is not None and harmful_cand is not None
        else None
    )
    overrefusal_change = (
        compare_outcomes(harmless_ref, harmless_cand, set_kind="harmless")
        if harmless_ref is not None and harmless_cand is not None
        else None
    )

    fingerprint_match: bool | None = None
    ref_fp = reference.input_fingerprint
    cand_fp = candidate.input_fingerprint
    if ref_fp is None or cand_fp is None:
        fingerprint_match = None
        missing = []
        if ref_fp is None:
            missing.append("reference")
        if cand_fp is None:
            missing.append("candidate")
        warnings.append(f"prompt fingerprint unknown on {', '.join(missing)}")
    elif ref_fp == cand_fp:
        fingerprint_match = True
    else:
        fingerprint_match = False
        warnings.append(
            f"prompt fingerprint mismatch: reference={ref_fp}, candidate={cand_fp}"
        )

    shifts: list[DirectionShift] = []
    if reference_directions is not None and candidate_directions is not None:
        for ref_report, cand_report in zip(
            reference_directions, candidate_directions, strict=False
        ):
            shifts.extend(compare_refusal_directions(ref_report, cand_report))

    geometry_without_behaviour = False
    for shift in shifts:
        if shift.status == "measured":
            outcome = harmful_cand
            if (
                outcome is None
                or outcome.status != "measured"
                or outcome.refusal_rate is None
            ):
                geometry_without_behaviour = True
                break

    return RefusalComparison(
        prompt_fingerprint_match=fingerprint_match,
        outcomes=tuple(outcome_list),
        shifts=tuple(shifts),
        response_level_change=response_level_change,
        overrefusal_change=overrefusal_change,
        geometry_without_behaviour=geometry_without_behaviour,
        warnings=tuple(warnings),
    )


def _cosine(reference: torch.Tensor, candidate: torch.Tensor) -> float | None:
    try:
        unit_ref = extract.normalize(reference)
        unit_cand = extract.normalize(candidate)
    except ValueError:
        return None
    return float(unit_ref @ unit_cand)


def _norm_ratio(reference: torch.Tensor, candidate: torch.Tensor) -> float | None:
    ref_norm = float(reference.norm())
    cand_norm = float(candidate.norm())
    if ref_norm < 1e-12:
        return None
    return cand_norm / ref_norm


def _orthogonalization_residual(
    reference: torch.Tensor, candidate: torch.Tensor
) -> float | None:
    try:
        unit_ref = extract.normalize(reference)
    except ValueError:
        return None
    projection = float(candidate @ unit_ref) * unit_ref
    residual = candidate - projection
    return float(residual.norm())
