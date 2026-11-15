from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Literal

import pytest
import torch

from model_headroom.paired.refusal_compare import (
    DirectionShift,
    RefusalComparison,
    RefusalOutcome,
    compare_outcomes,
    compare_refusal,
    compare_refusal_directions,
)
from model_headroom.paired.schema import LayerMeasurement, MeasurementDocument

if TYPE_CHECKING:
    from model_headroom.models.types import RefusalDirectionReport
    from model_headroom.paired.schema import MeasurementStatus

ModelRole = Literal["reference", "compressed", "recovered"]
SetKind = Literal["harmful", "harmless"]


def _outcome(
    role: ModelRole,
    set_kind: SetKind,
    *,
    prompt_count: int = 10,
    refusal_rate: float | None = 0.5,
    status: MeasurementStatus = "measured",
    refusal_count: int | None = 5,
) -> RefusalOutcome:
    return RefusalOutcome(
        role=role,
        set_kind=set_kind,
        prompt_count=prompt_count,
        refusal_count=refusal_count,
        refusal_rate=refusal_rate,
        status=status,
        reason=None,
    )


def _direction(
    values: tuple[float, ...],
    *,
    cohen_d: float = 0.5,
    layer_name: str = "encoder.layer0",
    measured: bool = True,
) -> RefusalDirectionReport:
    from model_headroom.models.types import RefusalDirectionReport, RefusalLayerStat

    return RefusalDirectionReport(
        axis="mean_difference",
        hidden_size=len(values),
        harmful_count=2,
        harmless_count=2,
        layer_stats=[
            RefusalLayerStat(
                layer_name=layer_name,
                layer_index=0,
                harmful_count=2,
                harmless_count=2,
                cohen_d=cohen_d,
                mean_norm=1.0,
                masked_count=0,
                direction_norm=float(torch.tensor(values).norm()),
                orthogonality_with_harmless_mean=0.0,
            )
        ],
        direction=torch.tensor(values) if measured else None,
        layer_name=layer_name,
        layer_index=0,
        cohen_d=cohen_d,
    )


def _document(
    model_id: str,
    role: ModelRole,
    *,
    fingerprint: str | None,
    layers: tuple[LayerMeasurement, ...] = (),
) -> MeasurementDocument:
    return MeasurementDocument(
        model_id=model_id,
        role=role,
        input_fingerprint=fingerprint,
        layers=layers,
    )


def _compare(
    *,
    reference_outcomes: tuple[RefusalOutcome, ...] = (),
    candidate_outcomes: tuple[RefusalOutcome, ...] = (),
    reference_directions: tuple[RefusalDirectionReport, ...] | None = None,
    candidate_directions: tuple[RefusalDirectionReport, ...] | None = None,
    reference_fingerprint: str | None = "fp-1",
    candidate_fingerprint: str | None = "fp-1",
) -> RefusalComparison:
    return compare_refusal(
        _document("ref", "reference", fingerprint=reference_fingerprint),
        _document("cand", "compressed", fingerprint=candidate_fingerprint),
        reference_outcomes=list(reference_outcomes),
        candidate_outcomes=list(candidate_outcomes),
        reference_directions=(
            list(reference_directions) if reference_directions is not None else None
        ),
        candidate_directions=(
            list(candidate_directions) if candidate_directions is not None else None
        ),
    )


class TestRefusalOutcomeRateGuard:
    def test_zero_prompt_set_carries_no_rate_and_produces_no_delta(self) -> None:
        reference = _outcome(
            "reference",
            "harmful",
            prompt_count=0,
            refusal_rate=None,
            status="not_measured",
            refusal_count=None,
        )
        candidate = _outcome(
            "compressed",
            "harmful",
            prompt_count=0,
            refusal_rate=None,
            status="not_measured",
            refusal_count=None,
        )
        assert reference.prompt_count == 0
        assert reference.refusal_rate is None
        assert compare_outcomes(reference, candidate, set_kind="harmful") is None

    def test_measured_zero_rate_on_both_sides_is_a_real_zero_delta(self) -> None:
        reference = _outcome("reference", "harmful", refusal_rate=0.0)
        candidate = _outcome("compressed", "harmful", refusal_rate=0.0)
        delta = compare_outcomes(reference, candidate, set_kind="harmful")
        assert delta is not None
        assert delta == 0.0

    def test_rate_delta_is_candidate_minus_reference(self) -> None:
        reference = _outcome("reference", "harmless", refusal_rate=0.25)
        candidate = _outcome("compressed", "harmless", refusal_rate=0.75)
        assert compare_outcomes(reference, candidate, set_kind="harmless") == 0.5

    def test_set_kind_mismatch_yields_no_delta(self) -> None:
        reference = _outcome("reference", "harmful")
        candidate = _outcome("compressed", "harmful")
        assert compare_outcomes(reference, candidate, set_kind="harmless") is None

    def test_unmeasured_status_alone_yields_no_delta(self) -> None:
        reference = _outcome(
            "reference", "harmful", refusal_rate=None, status="not_measured"
        )
        candidate = _outcome("compressed", "harmful")
        assert compare_outcomes(reference, candidate, set_kind="harmful") is None

    def test_missing_outcome_on_either_side_yields_no_change(self) -> None:
        reference_only = _compare(
            reference_outcomes=(_outcome("reference", "harmful", refusal_rate=0.5),),
            candidate_outcomes=(),
        )
        assert reference_only.response_level_change is None
        assert reference_only.overrefusal_change is None
        candidate_only = _compare(
            reference_outcomes=(),
            candidate_outcomes=(_outcome("compressed", "harmful", refusal_rate=0.5),),
        )
        assert candidate_only.response_level_change is None


class TestDirectionShiftGuard:
    def test_absent_direction_is_not_measured_not_a_fabricated_cosine(self) -> None:
        reference = _direction((1.0, 0.0, 0.0, 0.0))
        candidate = _direction((1.0, 0.0, 0.0, 0.0), measured=False)
        shifts = compare_refusal_directions(reference, candidate)
        assert len(shifts) == 1
        shift = shifts[0]
        assert shift.status == "not_measured"
        assert shift.cosine_to_reference is None
        assert shift.cohen_d_delta is None
        assert shift.norm_ratio is None
        assert shift.orthogonalization_residual is None
        assert shift.reason == "direction absent on reference or candidate"

    def test_measured_directions_report_cosine_ratio_and_cohen_d_delta(self) -> None:
        reference = _direction((1.0, 0.0, 0.0, 0.0), cohen_d=0.5)
        candidate = _direction((1.0, 0.0, 0.0, 0.0), cohen_d=0.9)
        shift = compare_refusal_directions(reference, candidate)[0]
        assert shift.status == "measured"
        assert shift.cosine_to_reference == pytest.approx(1.0)
        assert shift.norm_ratio == pytest.approx(1.0)
        assert shift.cohen_d_delta == pytest.approx(0.4)
        assert shift.orthogonalization_residual == pytest.approx(0.0)

    def test_zero_magnitude_candidate_leaves_the_cosine_undefined(self) -> None:
        reference = _direction((1.0, 0.0, 0.0, 0.0))
        candidate = _direction((0.0, 0.0, 0.0, 0.0))
        shift = compare_refusal_directions(reference, candidate)[0]
        assert shift.status == "measured"
        assert shift.cosine_to_reference is None
        assert shift.cosine_to_reference != 0.0
        assert shift.norm_ratio == 0.0
        assert shift.orthogonalization_residual == pytest.approx(0.0)

    def test_layer_present_on_one_side_still_yields_a_shift(self) -> None:
        reference = _direction((1.0, 0.0, 0.0, 0.0), cohen_d=0.5, layer_name="l0")
        candidate = _direction((0.0, 1.0, 0.0, 0.0), cohen_d=0.1, layer_name="l1")
        shifts = compare_refusal_directions(reference, candidate)
        assert [shift.layer_name for shift in shifts] == ["l0", "l1"]
        assert shifts[0].cohen_d_reference == pytest.approx(0.5)
        assert shifts[0].cohen_d_candidate is None
        assert shifts[0].status == "measured"
        assert shifts[1].cohen_d_candidate == pytest.approx(0.1)

    def test_cosine_tracks_the_candidate_direction(self) -> None:
        reference = _direction((1.0, 0.0, 0.0, 0.0))
        orthogonal = _direction((0.0, 1.0, 0.0, 0.0))
        opposite = _direction((-1.0, 0.0, 0.0, 0.0))
        assert compare_refusal_directions(reference, reference)[
            0
        ].cosine_to_reference == pytest.approx(1.0)
        assert compare_refusal_directions(reference, orthogonal)[
            0
        ].cosine_to_reference == pytest.approx(0.0)
        assert compare_refusal_directions(reference, opposite)[
            0
        ].cosine_to_reference == pytest.approx(-1.0)


class TestPromptFingerprintGuard:
    def test_equal_fingerprints_report_true(self) -> None:
        assert (
            _compare(
                reference_fingerprint="fp-1", candidate_fingerprint="fp-1"
            ).prompt_fingerprint_match
            is True
        )

    def test_differing_fingerprints_report_false(self) -> None:
        comparison = _compare(
            reference_fingerprint="fp-1", candidate_fingerprint="fp-2"
        )
        assert comparison.prompt_fingerprint_match is False
        assert any("mismatch" in warning for warning in comparison.warnings)

    @pytest.mark.parametrize(
        ("reference_fingerprint", "candidate_fingerprint", "missing_side"),
        [
            (None, "fp-1", "reference"),
            ("fp-1", None, "candidate"),
            (None, None, "reference, candidate"),
        ],
    )
    def test_unknown_fingerprint_on_either_side_is_none(
        self,
        reference_fingerprint: str | None,
        candidate_fingerprint: str | None,
        missing_side: str,
    ) -> None:
        comparison = _compare(
            reference_fingerprint=reference_fingerprint,
            candidate_fingerprint=candidate_fingerprint,
        )
        assert comparison.prompt_fingerprint_match is None
        assert comparison.prompt_fingerprint_match is not False
        assert any(
            f"unknown on {missing_side}" in warning for warning in comparison.warnings
        )


class TestGeometryWithoutBehaviour:
    def test_measured_shift_with_no_response_evidence_is_flagged(self) -> None:
        comparison = _compare(
            reference_directions=(_direction((1.0, 0.0, 0.0, 0.0)),),
            candidate_directions=(_direction((0.0, 1.0, 0.0, 0.0)),),
        )
        assert comparison.shifts
        assert comparison.geometry_without_behaviour is True

    def test_unmeasured_outcome_is_not_response_evidence(self) -> None:
        comparison = _compare(
            reference_outcomes=(
                _outcome(
                    "reference",
                    "harmful",
                    status="not_measured",
                    refusal_rate=None,
                ),
            ),
            candidate_outcomes=(
                _outcome(
                    "compressed",
                    "harmful",
                    status="not_measured",
                    refusal_rate=None,
                    prompt_count=0,
                    refusal_count=None,
                ),
            ),
            reference_directions=(_direction((1.0, 0.0, 0.0, 0.0)),),
            candidate_directions=(_direction((0.0, 1.0, 0.0, 0.0)),),
        )
        assert comparison.geometry_without_behaviour is True

    def test_measured_response_evidence_clears_the_flag(self) -> None:
        comparison = _compare(
            reference_outcomes=(_outcome("reference", "harmful", refusal_rate=0.5),),
            candidate_outcomes=(_outcome("compressed", "harmful", refusal_rate=0.5),),
            reference_directions=(_direction((1.0, 0.0, 0.0, 0.0)),),
            candidate_directions=(_direction((0.0, 1.0, 0.0, 0.0)),),
        )
        assert comparison.geometry_without_behaviour is False
        assert comparison.response_level_change == 0.0

    def test_no_measured_shift_means_no_geometry_without_behaviour(self) -> None:
        comparison = _compare(
            reference_directions=(_direction((1.0, 0.0, 0.0, 0.0), measured=False),),
            candidate_directions=(_direction((0.0, 1.0, 0.0, 0.0), measured=False),),
        )
        assert comparison.shifts
        assert all(shift.status == "not_measured" for shift in comparison.shifts)
        assert comparison.geometry_without_behaviour is False

    def test_no_direction_reports_at_all_means_no_geometry_without_behaviour(
        self,
    ) -> None:
        comparison = _compare(
            reference_outcomes=(_outcome("reference", "harmful", refusal_rate=0.5),),
            candidate_outcomes=(_outcome("compressed", "harmful", refusal_rate=0.25),),
        )
        assert comparison.shifts == ()
        assert comparison.geometry_without_behaviour is False


class TestNoCombinedScoreField:
    def test_no_refusal_dataclass_field_names_a_combined_score(self) -> None:
        forbidden = (
            "combined",
            "composite",
            "score",
            "verdict",
            "safety",
            "quality",
            "rating",
            "grade",
        )
        for dataclass_type in (RefusalOutcome, DirectionShift, RefusalComparison):
            for field in dataclasses.fields(dataclass_type):
                lowered = field.name.lower()
                assert not any(token in lowered for token in forbidden), field.name

    def test_direction_shift_carries_no_verdict_or_safety_flag(self) -> None:
        shift = compare_refusal_directions(
            _direction((1.0, 0.0, 0.0, 0.0)), _direction((1.0, 0.0, 0.0, 0.0))
        )[0]
        assert hasattr(shift, "verdict") is False
        assert hasattr(shift, "safety_preserved") is False
        assert hasattr(shift, "combined_score") is False


class TestRefusalOutcomeSetWiring:
    def test_response_level_change_reads_the_harmful_pair_only(self) -> None:
        comparison = _compare(
            reference_outcomes=(
                _outcome("reference", "harmful", refusal_rate=0.5),
                _outcome("reference", "harmless", refusal_rate=0.1),
            ),
            candidate_outcomes=(
                _outcome("compressed", "harmful", refusal_rate=0.25),
                _outcome("compressed", "harmless", refusal_rate=0.6),
            ),
        )
        assert comparison.response_level_change == pytest.approx(-0.25)
        assert comparison.overrefusal_change == pytest.approx(0.5)

    def test_outcomes_are_carried_through_untouched(self) -> None:
        reference = _outcome("reference", "harmful", refusal_rate=0.5)
        candidate = _outcome("compressed", "harmful", refusal_rate=0.25)
        comparison = _compare(
            reference_outcomes=(reference,), candidate_outcomes=(candidate,)
        )
        assert comparison.outcomes == (reference, candidate)
