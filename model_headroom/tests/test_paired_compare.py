from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import pytest
import torch

from model_headroom.paired.compare import (
    ComparisonError,
    activation_outliers,
    compare_documents,
    effective_weight_error,
    layer_output_similarity,
    token_level_kl,
)
from model_headroom.paired.schema import (
    GradientMeasurement,
    LayerMeasurement,
    MeasurementDocument,
    ModuleMeasurement,
    TensorMeasurement,
)

if TYPE_CHECKING:
    from model_headroom.models.types import ModuleFamily
    from model_headroom.paired.schema import MeasurementStatus, TensorKind

ModelRole = Literal["reference", "compressed", "recovered"]


def _measurement(
    name: str,
    *,
    kind: TensorKind = "effective_weight",
    status: MeasurementStatus = "measured",
    value: float | None = 0.5,
) -> TensorMeasurement:
    return TensorMeasurement(
        name=name,
        kind=kind,
        shape=(4, 4),
        dtype="float32",
        status=status,
        value=value,
    )


def _module(
    path: str,
    *,
    weight_kind: TensorKind = "effective_weight",
    value: float | None = 0.5,
) -> ModuleMeasurement:
    tensor_kind: TensorKind = (
        "packed_buffer" if weight_kind == "packed_buffer" else "effective_weight"
    )
    status: MeasurementStatus = "measured" if value is not None else "not_measured"
    return ModuleMeasurement(
        module_path=path,
        family="ffn",
        module_type="Linear",
        parameter_count=64,
        weight_kind=weight_kind,
        tensors=(_measurement(path, kind=tensor_kind, status=status, value=value),),
        status=status,
    )


def _layer(
    name: str,
    *,
    family: ModuleFamily = "attention",
    status: MeasurementStatus = "measured",
    cosine: float | None = 0.9,
    modules: tuple[ModuleMeasurement, ...] = (),
) -> LayerMeasurement:
    measured = status == "measured"
    return LayerMeasurement(
        layer_name=name,
        layer_index=0,
        family=family,
        output_cosine=cosine if measured else None,
        normalized_reconstruction_error=0.1 if measured else None,
        output_norm_ratio=1.0 if measured else None,
        status=status,
        modules=modules,
    )


def _document(
    model_id: str,
    role: ModelRole,
    layers: tuple[LayerMeasurement, ...],
    *,
    fingerprint: str | None = "fp-1",
    gradients: GradientMeasurement | None = None,
) -> MeasurementDocument:
    return MeasurementDocument(
        model_id=model_id,
        role=role,
        input_fingerprint=fingerprint,
        layers=layers,
        gradients=gradients,
    )


def _unmeasured_gradient(source_type: str) -> GradientMeasurement:
    return GradientMeasurement(
        status="not_measured",
        grad_norm=None,
        coherence=None,
        effective_rank=None,
        parameter_count=1024,
        source_type=source_type,  # type: ignore[arg-type]
        verification_status="unavailable",
    )


class TestLayerOutputSimilarityGuard:
    def test_shape_mismatch_raises_naming_both_shapes(self) -> None:
        reference = torch.ones(2, 3)
        candidate = torch.ones(3, 2)
        with pytest.raises(ComparisonError) as excinfo:
            layer_output_similarity(reference, candidate)
        message = str(excinfo.value)
        assert "shape mismatch" in message
        assert "(2, 3)" in message
        assert "(3, 2)" in message

    def test_non_2d_reference_raises(self) -> None:
        with pytest.raises(ComparisonError) as excinfo:
            layer_output_similarity(torch.ones(4), torch.ones(4))
        assert "must be 2-D" in str(excinfo.value)

    def test_non_2d_candidate_raises(self) -> None:
        with pytest.raises(ComparisonError) as excinfo:
            layer_output_similarity(torch.ones(2, 2), torch.ones(2, 2, 1))
        assert "must be 2-D" in str(excinfo.value)

    def test_zero_reference_norm_raises(self) -> None:
        with pytest.raises(ComparisonError) as excinfo:
            layer_output_similarity(torch.zeros(2, 2), torch.ones(2, 2))
        assert "zero reference norm" in str(excinfo.value)

    def test_zero_candidate_norm_is_undefined_cosine_and_unit_reconstruction_error(
        self,
    ) -> None:
        reference = torch.ones(2, 2)
        candidate = torch.zeros(2, 2)
        cosine, error, reference_norm, candidate_norm = layer_output_similarity(
            reference, candidate
        )
        assert cosine is None
        assert cosine != 0.0
        assert error == pytest.approx(1.0)
        assert reference_norm == pytest.approx(2.0)
        assert candidate_norm == 0.0

    def test_orthogonal_tensors_report_a_measured_zero_cosine(self) -> None:
        reference = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        candidate = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
        cosine, error, _, _ = layer_output_similarity(reference, candidate)
        assert cosine is not None
        assert cosine == 0.0
        assert error == pytest.approx(2.0 / (2.0**0.5))

    def test_identical_tensors_report_unit_cosine_and_zero_error(self) -> None:
        reference = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        cosine, error, _, _ = layer_output_similarity(reference, reference.clone())
        assert cosine == pytest.approx(1.0)
        assert error == pytest.approx(0.0)

    def test_reconstruction_error_is_scaled_by_the_reference_norm(self) -> None:
        reference = torch.tensor([[3.0, 0.0], [0.0, 4.0]])
        candidate = torch.tensor([[0.0, 0.0], [0.0, 1.0]])
        forward = layer_output_similarity(reference, candidate)
        reverse = layer_output_similarity(candidate, reference)
        assert forward[0] == pytest.approx(0.8)
        assert reverse[0] == pytest.approx(0.8)
        assert forward[1] == pytest.approx((18.0**0.5) / 5.0)
        assert reverse[1] == pytest.approx(18.0**0.5)
        assert forward[2] == pytest.approx(5.0)
        assert reverse[2] == pytest.approx(1.0)


class TestEffectiveWeightErrorGuard:
    def test_packed_buffer_is_refused_even_with_matching_shapes(self) -> None:
        reference = torch.ones(4, 4)
        candidate = torch.ones(4, 4)
        with pytest.raises(ComparisonError) as excinfo:
            effective_weight_error(reference, candidate, kind="packed_buffer")
        message = str(excinfo.value)
        assert "packed buffer" in message
        assert "(4, 4)" in message

    def test_packed_buffer_is_refused_before_the_2d_check(self) -> None:
        with pytest.raises(ComparisonError) as excinfo:
            effective_weight_error(torch.ones(4), torch.ones(4), kind="packed_buffer")
        assert "packed buffer" in str(excinfo.value)

    def test_effective_weight_kind_still_computes_both_errors(self) -> None:
        reference = torch.zeros(2, 2)
        reference[0, 0] = 3.0
        reference[1, 1] = 4.0
        candidate = reference.clone()
        candidate[0, 0] = 0.0
        absolute, relative = effective_weight_error(
            reference, candidate, kind="effective_weight"
        )
        assert absolute == pytest.approx(3.0)
        assert relative == pytest.approx(0.6)

    def test_effective_weight_kind_still_rejects_shape_mismatch(self) -> None:
        with pytest.raises(ComparisonError) as excinfo:
            effective_weight_error(
                torch.ones(2, 3), torch.ones(3, 3), kind="effective_weight"
            )
        assert "shape mismatch" in str(excinfo.value)

    def test_effective_weight_kind_still_rejects_zero_reference_norm(self) -> None:
        with pytest.raises(ComparisonError) as excinfo:
            effective_weight_error(
                torch.zeros(2, 2), torch.ones(2, 2), kind="effective_weight"
            )
        assert "zero reference norm" in str(excinfo.value)


class TestDocumentLayerPairing:
    def test_layer_present_on_one_side_is_not_measured_not_omitted(self) -> None:
        reference = _document(
            "ref",
            "reference",
            (_layer("encoder.layer0"), _layer("encoder.layer1")),
        )
        candidate = _document("cand", "compressed", (_layer("encoder.layer0"),))
        report = compare_documents(reference, candidate)
        assert [entry.layer_name for entry in report.layers] == [
            "encoder.layer0",
            "encoder.layer1",
        ]
        absent = report.layers[1]
        assert absent.status == "not_measured"
        assert absent.reason is not None
        assert "absent on" in absent.reason
        assert absent.cosine is None
        assert absent.normalized_reconstruction_error is None
        assert absent.norm_ratio is None
        assert any("encoder.layer1" in warning for warning in report.warnings)
        assert report.comparable_layer_count == 1
        assert report.measured_layer_count == 1
        assert report.coverage == pytest.approx(1.0)

    def test_module_present_on_one_side_is_not_measured_not_omitted(self) -> None:
        reference_module = _module("encoder.layer0.mlp.up_proj")
        report = compare_documents(
            _document(
                "ref",
                "reference",
                (_layer("encoder.layer0", modules=(reference_module,)),),
            ),
            _document("cand", "compressed", (_layer("encoder.layer0"),)),
        )
        assert [module.module_path for module in report.modules] == [
            "encoder.layer0.mlp.up_proj"
        ]
        entry = report.modules[0]
        assert entry.status == "not_measured"
        assert entry.effective_weight_reconstruction_error is None
        assert entry.relative_error is None

    def test_coverage_is_none_when_no_layer_is_comparable(self) -> None:
        reference = _document("ref", "reference", (_layer("encoder.layer0"),))
        candidate = _document("cand", "compressed", (_layer("decoder.layer0"),))
        report = compare_documents(reference, candidate)
        assert report.comparable_layer_count == 0
        assert report.measured_layer_count == 0
        assert report.coverage is None

    def test_coverage_is_none_when_both_documents_have_no_layers(self) -> None:
        reference = _document("ref", "reference", ())
        candidate = _document("cand", "compressed", ())
        report = compare_documents(reference, candidate)
        assert report.coverage is None
        assert report.layers == ()

    def test_unmeasured_reference_layer_is_not_measured_with_reason(self) -> None:
        reference = _document(
            "ref",
            "reference",
            (_layer("encoder.layer0", status="not_measured", cosine=None),),
        )
        candidate = _document("cand", "compressed", (_layer("encoder.layer0"),))
        report = compare_documents(reference, candidate)
        entry = report.layers[0]
        assert entry.status == "not_measured"
        assert entry.reason == "layer not measured on reference=not_measured"
        assert report.comparable_layer_count == 1
        assert report.measured_layer_count == 0
        assert report.coverage == 0.0

    def test_measured_zero_cosine_is_not_reported_as_unmeasured(self) -> None:
        reference = _document(
            "ref", "reference", (_layer("encoder.layer0", cosine=0.0),)
        )
        candidate = _document(
            "cand", "compressed", (_layer("encoder.layer0", cosine=0.0),)
        )
        report = compare_documents(reference, candidate)
        entry = report.layers[0]
        assert entry.status == "measured"
        assert entry.cosine is not None
        assert entry.cosine == 0.0
        assert entry.reason is None

    def test_layer_order_in_the_documents_does_not_change_the_report_order(
        self,
    ) -> None:
        first = _document(
            "ref",
            "reference",
            (_layer("encoder.layer1"), _layer("encoder.layer0")),
        )
        second = _document(
            "cand",
            "compressed",
            (_layer("encoder.layer0"), _layer("encoder.layer1")),
        )
        report = compare_documents(first, second)
        assert [entry.layer_name for entry in report.layers] == [
            "encoder.layer0",
            "encoder.layer1",
        ]
        reordered = compare_documents(second, first)
        assert reordered.layers == report.layers
        assert reordered.coverage == report.coverage

    def test_absent_layer_reason_names_the_side_that_lacks_the_layer(self) -> None:
        only_on_candidate = compare_documents(
            _document("ref", "reference", ()),
            _document("cand", "compressed", (_layer("decoder.layer9"),)),
        )
        assert only_on_candidate.layers[0].reason == "layer absent on reference"
        assert any(
            "decoder.layer9" in warning and "absent on reference" in warning
            for warning in only_on_candidate.warnings
        )
        only_on_reference = compare_documents(
            _document("ref", "reference", (_layer("encoder.layer0"),)),
            _document("cand", "compressed", ()),
        )
        assert only_on_reference.layers[0].reason == "layer absent on candidate"
        assert any(
            "encoder.layer0" in warning and "absent on candidate" in warning
            for warning in only_on_reference.warnings
        )

    def test_absent_module_reason_names_the_side_that_lacks_the_module(self) -> None:
        candidate_module = _module("encoder.layer0.mlp.down_proj")
        only_on_candidate = compare_documents(
            _document("ref", "reference", (_layer("encoder.layer0"),)),
            _document(
                "cand",
                "compressed",
                (_layer("encoder.layer0", modules=(candidate_module,)),),
            ),
        )
        assert only_on_candidate.modules[0].reason == "module absent on reference"
        assert any(
            "down_proj" in warning and "absent on reference" in warning
            for warning in only_on_candidate.warnings
        )
        reference_module = _module("encoder.layer0.mlp.up_proj")
        only_on_reference = compare_documents(
            _document(
                "ref",
                "reference",
                (_layer("encoder.layer0", modules=(reference_module,)),),
            ),
            _document("cand", "compressed", (_layer("encoder.layer0"),)),
        )
        assert only_on_reference.modules[0].reason == "module absent on candidate"


class TestDocumentFingerprintGuard:
    def test_fingerprint_mismatch_raises_naming_both_fingerprints(self) -> None:
        reference = _document("ref", "reference", (), fingerprint="fp-reference")
        candidate = _document("cand", "compressed", (), fingerprint="fp-candidate")
        with pytest.raises(ComparisonError) as excinfo:
            compare_documents(reference, candidate)
        message = str(excinfo.value)
        assert "input fingerprint mismatch" in message
        assert "fp-reference" in message
        assert "fp-candidate" in message

    def test_matching_fingerprints_report_true(self) -> None:
        reference = _document("ref", "reference", ())
        candidate = _document("cand", "compressed", ())
        assert compare_documents(reference, candidate).input_fingerprint_match is True

    def test_unknown_fingerprint_on_candidate_is_none_not_false(self) -> None:
        reference = _document("ref", "reference", (), fingerprint="fp-1")
        candidate = _document("cand", "compressed", (), fingerprint=None)
        report = compare_documents(reference, candidate)
        assert report.input_fingerprint_match is None
        assert report.input_fingerprint_match is not False
        assert any("unknown on candidate" in warning for warning in report.warnings)

    def test_unknown_fingerprint_on_reference_is_none_not_false(self) -> None:
        reference = _document("ref", "reference", (), fingerprint=None)
        candidate = _document("cand", "compressed", (), fingerprint="fp-1")
        report = compare_documents(reference, candidate)
        assert report.input_fingerprint_match is None
        assert any("unknown on reference" in warning for warning in report.warnings)


class TestDocumentModuleGuard:
    def test_packed_buffer_module_is_not_applicable_not_a_weight_error(self) -> None:
        reference = _module(
            "encoder.layer0.mlp.down_proj",
            weight_kind="packed_buffer",
            value=None,
        )
        candidate = _module(
            "encoder.layer0.mlp.down_proj",
            weight_kind="packed_buffer",
            value=None,
        )
        report = compare_documents(
            _document(
                "ref",
                "reference",
                (_layer("encoder.layer0", modules=(reference,)),),
            ),
            _document(
                "cand",
                "compressed",
                (_layer("encoder.layer0", modules=(candidate,)),),
            ),
        )
        module = report.modules[0]
        assert module.status == "not_applicable"
        assert module.effective_weight_reconstruction_error is None
        assert module.relative_error is None
        assert "packed buffer is not an effective weight matrix" in (
            module.reason or ""
        )
        assert any("packed buffer" in warning for warning in report.warnings)

    def test_effective_weight_present_on_both_sides_is_measured(self) -> None:
        report = compare_documents(
            _document(
                "ref",
                "reference",
                (_layer("encoder.layer0", modules=(_module("m.up_proj"),)),),
            ),
            _document(
                "cand",
                "compressed",
                (
                    _layer(
                        "encoder.layer0",
                        modules=(_module("m.up_proj", value=0.25),),
                    ),
                ),
            ),
        )
        module = report.modules[0]
        assert module.status == "measured"
        assert module.effective_weight_reconstruction_error == pytest.approx(0.25)
        assert module.relative_error == pytest.approx(0.5)

    def test_missing_effective_weight_value_is_not_measured(self) -> None:
        report = compare_documents(
            _document(
                "ref",
                "reference",
                (_layer("encoder.layer0", modules=(_module("m.up_proj"),)),),
            ),
            _document(
                "cand",
                "compressed",
                (
                    _layer(
                        "encoder.layer0",
                        modules=(_module("m.up_proj", value=None),),
                    ),
                ),
            ),
        )
        module = report.modules[0]
        assert module.status == "not_measured"
        assert module.reason is not None
        assert "missing on candidate" in module.reason


class TestDocumentGradientGuard:
    def test_absent_gradient_is_not_measured_with_every_numeric_none(self) -> None:
        report = compare_documents(
            _document("ref", "reference", ()),
            _document("cand", "compressed", ()),
        )
        gradients = report.gradients
        assert gradients is not None
        assert gradients.status == "not_measured"
        assert gradients.reference_grad_norm is None
        assert gradients.candidate_grad_norm is None
        assert gradients.grad_norm_ratio is None
        assert gradients.coherence_delta is None
        assert gradients.effective_rank_ratio is None

    def test_placeholder_gradient_is_not_measured_with_every_numeric_none(self) -> None:
        reference = _document(
            "ref",
            "reference",
            (),
            gradients=_unmeasured_gradient("unavailable"),
        )
        candidate = _document(
            "cand",
            "compressed",
            (),
            gradients=_unmeasured_gradient("unavailable"),
        )
        gradients = compare_documents(reference, candidate).gradients
        assert gradients is not None
        assert gradients.status == "not_measured"
        for field in (
            "reference_grad_norm",
            "candidate_grad_norm",
            "grad_norm_ratio",
            "coherence_delta",
            "effective_rank_ratio",
        ):
            value = getattr(gradients, field)
            assert value is None
            assert value != 0.0
        assert "source_type=unavailable" in (gradients.reason or "")

    def test_measured_gradient_ratio_is_a_ratio_not_a_placeholder(self) -> None:
        reference = _document(
            "ref",
            "reference",
            (),
            gradients=GradientMeasurement(
                status="measured",
                grad_norm=2.0,
                coherence=0.5,
                effective_rank=4.0,
                parameter_count=1024,
                source_type="measured",
                verification_status="verified",
            ),
        )
        candidate = _document(
            "cand",
            "compressed",
            (),
            gradients=GradientMeasurement(
                status="measured",
                grad_norm=1.0,
                coherence=0.25,
                effective_rank=2.0,
                parameter_count=1024,
                source_type="measured",
                verification_status="verified",
            ),
        )
        gradients = compare_documents(reference, candidate).gradients
        assert gradients is not None
        assert gradients.status == "measured"
        assert gradients.grad_norm_ratio == pytest.approx(0.5)
        assert gradients.coherence_delta == pytest.approx(-0.25)
        assert gradients.effective_rank_ratio == pytest.approx(0.5)


class TestActivationOutlierGuard:
    def test_non_2d_activations_report_not_measured(self) -> None:
        report = activation_outliers(
            torch.ones(4), sigma=3.0, near_inactive_threshold=1e-3
        )
        assert report.status == "not_measured"
        assert report.reason == "activations must be 2-D"
        assert report.outlier_fraction is None
        assert report.near_inactive_fraction is None

    def test_outlier_and_near_inactive_channels_are_exact(self) -> None:
        activations = torch.tensor([[0.0, 0.0, 0.0, 3.5, 10.0]] * 5)
        report = activation_outliers(
            activations, sigma=0.5, near_inactive_threshold=1e-3
        )
        assert report.status == "measured"
        assert report.outlier_channels == (4,)
        assert report.near_inactive_channels == (0, 1, 2)
        assert report.outlier_fraction == pytest.approx(0.2)
        assert report.near_inactive_fraction == pytest.approx(0.6)

    def test_channel_above_the_mean_below_the_sigma_threshold_is_not_an_outlier(
        self,
    ) -> None:
        activations = torch.tensor([[0.0, 0.0, 0.0, 3.5, 10.0]] * 5)
        loose = activation_outliers(
            activations, sigma=0.5, near_inactive_threshold=1e-3
        )
        assert loose.outlier_channels == (4,)
        assert 3 not in loose.outlier_channels
        tight = activation_outliers(
            activations, sigma=0.1, near_inactive_threshold=1e-3
        )
        assert tight.outlier_channels == (3, 4)

    def test_single_channel_reports_zero_fractions(self) -> None:
        report = activation_outliers(
            torch.zeros(3, 1), sigma=3.0, near_inactive_threshold=1e-3
        )
        assert report.status == "measured"
        assert report.outlier_channels == ()
        assert report.near_inactive_channels == (0,)
        assert report.outlier_fraction == 0.0
        assert report.near_inactive_fraction == 1.0


class TestTokenLevelKlGuard:
    def test_vocabulary_mismatch_raises_naming_both_vocabularies(self) -> None:
        teacher = torch.zeros(1, 4)
        student = torch.zeros(1, 3)
        with pytest.raises(ComparisonError) as excinfo:
            token_level_kl(teacher, student)
        message = str(excinfo.value)
        assert "vocabulary size mismatch" in message
        assert "vocab=4" in message
        assert "vocab=3" in message

    @pytest.mark.parametrize("temperature", [0.0, -1.0])
    def test_non_positive_temperature_raises(self, temperature: float) -> None:
        logits = torch.zeros(1, 4)
        with pytest.raises(ComparisonError) as excinfo:
            token_level_kl(logits, logits, temperature=temperature)
        assert "temperature must be positive" in str(excinfo.value)

    def test_token_level_kl_is_a_pure_function_of_its_two_logit_tensors(self) -> None:
        teacher = torch.tensor([[1.0, 2.0, 3.0]])
        student = torch.tensor([[1.5, 1.0, 0.5]])
        first = token_level_kl(teacher, student)
        second = token_level_kl(teacher, student)
        assert first == second
        assert first > 0.0
        assert token_level_kl(teacher, teacher.clone()) == pytest.approx(0.0)
        assert token_level_kl(teacher, student, temperature=2.0) != first


class TestCompareDocumentsDeterminism:
    def test_compare_documents_is_a_pure_function_of_its_two_documents(self) -> None:
        reference = _document("ref", "reference", (_layer("encoder.layer0"),))
        candidate = _document("cand", "compressed", (_layer("encoder.layer0"),))
        first = compare_documents(reference, candidate)
        second = compare_documents(reference, candidate)
        assert first == second
        shifted = compare_documents(
            reference,
            _document(
                "cand",
                "compressed",
                (_layer("encoder.layer0", status="not_measured", cosine=None),),
            ),
        )
        assert shifted != first
        assert shifted.layers[0].status == "not_measured"
        assert shifted.coverage == 0.0
