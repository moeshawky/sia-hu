from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, cast

import pytest

from model_headroom.paired.schema import (
    GradientMeasurement,
    LayerMeasurement,
    MeasurementDocument,
    ModuleMeasurement,
    TensorMeasurement,
)

if TYPE_CHECKING:
    from model_headroom.paired.schema import MeasurementStatus


def _tensor(
    *,
    status: MeasurementStatus = "measured",
    value: float | None = None,
    name: str = "encoder.layer0.q_proj.weight",
) -> TensorMeasurement:
    return TensorMeasurement(
        name=name,
        kind="effective_weight",
        shape=(4, 4),
        dtype="float32",
        status=status,
        value=value,
    )


def _gradient(
    *,
    status: MeasurementStatus = "measured",
    grad_norm: float | None = None,
    coherence: float | None = None,
    effective_rank: float | None = None,
) -> GradientMeasurement:
    return GradientMeasurement(
        status=status,
        grad_norm=grad_norm,
        coherence=coherence,
        effective_rank=effective_rank,
        parameter_count=1024,
        source_type="measured",
        verification_status="unverified",
    )


class TestTensorMeasurementInvariant:
    def test_unmeasured_with_zero_value_raises_naming_offender(self) -> None:
        with pytest.raises(ValueError) as excinfo:
            _tensor(status="not_measured", value=0.0)
        message = str(excinfo.value)
        assert "value" in message
        assert "not_measured" in message
        assert "0.0" in message

    def test_measured_without_value_raises(self) -> None:
        with pytest.raises(ValueError) as excinfo:
            _tensor(status="measured", value=None)
        message = str(excinfo.value)
        assert "value" in message
        assert "measured" in message

    def test_measured_zero_is_accepted(self) -> None:
        assert _tensor(status="measured", value=0.0).value == 0.0

    def test_measured_fraction_is_accepted(self) -> None:
        assert _tensor(status="measured", value=0.5).value == 0.5

    def test_not_applicable_defaults_to_none(self) -> None:
        measurement = _tensor(status="not_applicable")
        assert measurement.value is None
        assert measurement.status == "not_applicable"

    def test_not_applicable_with_value_raises(self) -> None:
        with pytest.raises(ValueError) as excinfo:
            _tensor(status="not_applicable", value=0.0)
        assert "not_applicable" in str(excinfo.value)

    def test_still_frozen(self) -> None:
        measurement = _tensor(status="measured", value=0.5)
        with pytest.raises(dataclasses.FrozenInstanceError):
            measurement.value = 1.0  # type: ignore[misc]


class TestGradientMeasurementInvariant:
    def test_unmeasured_grad_norm_zero_raises_naming_field(self) -> None:
        with pytest.raises(ValueError) as excinfo:
            _gradient(status="not_measured", grad_norm=0.0)
        message = str(excinfo.value)
        assert "grad_norm" in message
        assert "not_measured" in message
        assert "0.0" in message

    def test_unmeasured_coherence_zero_raises_naming_field(self) -> None:
        with pytest.raises(ValueError) as excinfo:
            _gradient(status="not_measured", coherence=0.0)
        assert "coherence" in str(excinfo.value)

    def test_unmeasured_effective_rank_zero_raises_naming_field(self) -> None:
        with pytest.raises(ValueError) as excinfo:
            _gradient(status="not_measured", effective_rank=0.0)
        assert "effective_rank" in str(excinfo.value)

    def test_unmeasured_names_every_offending_field(self) -> None:
        with pytest.raises(ValueError) as excinfo:
            _gradient(
                status="not_measured",
                grad_norm=0.0,
                coherence=0.0,
                effective_rank=0.0,
            )
        message = str(excinfo.value)
        assert "grad_norm" in message
        assert "coherence" in message
        assert "effective_rank" in message

    def test_measured_without_any_value_raises(self) -> None:
        with pytest.raises(ValueError) as excinfo:
            _gradient(status="measured")
        message = str(excinfo.value)
        assert "grad_norm" in message
        assert "measured" in message

    def test_measured_single_value_is_accepted(self) -> None:
        assert _gradient(status="measured", grad_norm=1.5).grad_norm == 1.5

    def test_measured_zero_grad_norm_is_accepted(self) -> None:
        assert _gradient(status="measured", grad_norm=0.0).grad_norm == 0.0

    def test_not_applicable_defaults_to_none(self) -> None:
        gradient = _gradient(status="not_applicable")
        assert gradient.grad_norm is None
        assert gradient.coherence is None
        assert gradient.effective_rank is None

    def test_still_frozen(self) -> None:
        gradient = _gradient(status="measured", grad_norm=1.5)
        with pytest.raises(dataclasses.FrozenInstanceError):
            gradient.grad_norm = 2.0  # type: ignore[misc]


class TestLayerMeasurementInvariant:
    def test_unmeasured_layer_with_zero_cosine_raises(self) -> None:
        with pytest.raises(ValueError) as excinfo:
            LayerMeasurement(
                layer_name="encoder.layer0",
                layer_index=0,
                family="attention",
                output_cosine=0.0,
                normalized_reconstruction_error=None,
                output_norm_ratio=None,
                status="not_measured",
            )
        message = str(excinfo.value)
        assert "output_cosine" in message
        assert "not_measured" in message

    def test_measured_layer_may_have_undefined_cosine(self) -> None:
        layer = LayerMeasurement(
            layer_name="encoder.layer0",
            layer_index=0,
            family="attention",
            output_cosine=None,
            normalized_reconstruction_error=None,
            output_norm_ratio=None,
            status="measured",
        )
        assert layer.output_cosine is None


class TestNoBypass:
    def test_from_dict_rejects_unmeasured_tensor_with_zero(self) -> None:
        payload = cast("dict[str, object]", _tensor(status="not_measured").to_dict())
        payload["value"] = 0.0
        with pytest.raises(ValueError) as excinfo:
            TensorMeasurement.from_dict(payload)
        assert "value" in str(excinfo.value)

    def test_from_dict_rejects_unmeasured_gradient_with_zero(self) -> None:
        payload = _gradient(status="not_measured").to_dict()
        payload["grad_norm"] = 0.0
        with pytest.raises(ValueError) as excinfo:
            GradientMeasurement.from_dict(payload)
        assert "grad_norm" in str(excinfo.value)

    def test_from_json_rejects_unmeasured_layer_with_zero(self) -> None:
        layer = LayerMeasurement(
            layer_name="encoder.layer0",
            layer_index=0,
            family="attention",
            output_cosine=None,
            normalized_reconstruction_error=None,
            output_norm_ratio=None,
            status="not_measured",
        )
        document_dict = MeasurementDocument(layers=(layer,)).to_dict()
        layers = document_dict["layers"]
        assert isinstance(layers, list)
        layers[0]["output_cosine"] = 0.0
        with pytest.raises(ValueError) as excinfo:
            MeasurementDocument.from_dict(document_dict)
        assert "output_cosine" in str(excinfo.value)

    def test_to_dict_round_trip_of_legitimate_tensor(self) -> None:
        measurement = _tensor(status="measured", value=0.5)
        restored = TensorMeasurement.from_dict(
            cast("dict[str, object]", measurement.to_dict())
        )
        assert restored == measurement


class TestLossyConverterRemoved:
    def test_to_measurements_not_exported(self) -> None:
        from model_headroom.paired import instrument

        assert hasattr(instrument, "to_measurements") is False
        assert "to_measurements" not in instrument.__all__


class TestDocumentRoundTrip:
    def test_unmeasured_gradient_and_packed_buffer_round_trip(self) -> None:
        packed = TensorMeasurement(
            name="encoder.layer0.mlp.down_proj",
            kind="packed_buffer",
            shape=(4, 4),
            dtype="int8",
            status="not_measured",
        )
        module = ModuleMeasurement(
            module_path="encoder.layer0.mlp.down_proj",
            family="ffn",
            module_type="Linear",
            parameter_count=64,
            weight_kind="packed_buffer",
            tensors=(packed,),
            status="not_measured",
        )
        layer = LayerMeasurement(
            layer_name="encoder.layer0",
            layer_index=0,
            family="attention",
            output_cosine=None,
            normalized_reconstruction_error=None,
            output_norm_ratio=None,
            status="not_measured",
            modules=(module,),
        )
        document = MeasurementDocument(
            model_id="m",
            role="reference",
            layers=(layer,),
            gradients=_gradient(status="not_measured"),
        )
        restored = MeasurementDocument.from_json(document.to_json())
        assert restored == document
        assert restored.gradients is not None
        assert restored.gradients.status == "not_measured"
        assert restored.gradients.grad_norm is None
        assert restored.layers[0].modules[0].weight_kind == "packed_buffer"
        assert restored.layers[0].modules[0].tensors[0].value is None
