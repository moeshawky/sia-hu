"""Versioned JSON measurement contract for paired-model diagnostics.

An unmeasured quantity is a named state in the type system, never a number.
The tri-state MeasurementStatus distinguishes "measured", "not_measured", and
"not_applicable". TensorKind distinguishes a packed ternary buffer
("packed_buffer") from the dequantized effective weight ("effective_weight")
so downstream consumers can refuse to compute weight-spectrum diagnostics on
packed bytes. All dataclasses are frozen; serialization round-trips through
plain JSON types (lists, not tuples). A document whose schema_version does not
match SCHEMA_VERSION is rejected with a ValueError naming both versions.
Construction rejects a status/value contradiction instead of repairing it: a
value under a status other than "measured" is always an error, and
TensorMeasurement and GradientMeasurement additionally require a value when the
status is "measured". Violations raise ValueError naming the field, the status,
and the offending value.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, cast

if TYPE_CHECKING:
    from model_headroom.models.types import ModuleFamily, SourceType, VerificationStatus

SCHEMA_VERSION: str = "1"

MeasurementStatus = Literal["measured", "not_measured", "not_applicable"]

TensorKind = Literal[
    "effective_weight",
    "packed_buffer",
    "activation",
    "gradient",
    "scale",
    "unknown",
]


@dataclass(frozen=True)
class TensorMeasurement:
    """Frozen dataclass holding one measured tensor or quantity."""

    name: str
    kind: TensorKind
    shape: tuple[int, ...]
    dtype: str
    status: MeasurementStatus
    value: float | None = None
    unit: str | None = None
    method: str | None = None
    source_type: SourceType = "measured"
    verification_status: VerificationStatus = "unverified"

    def __post_init__(self) -> None:
        if self.status == "measured":
            if self.value is None:
                msg = (
                    "value must not be None when status is 'measured' "
                    f"(name={self.name!r}, kind={self.kind!r})"
                )
                raise ValueError(msg)
            return
        if self.value is not None:
            msg = (
                f"value must be None when status is {self.status!r}, "
                f"got {self.value!r} (name={self.name!r})"
            )
            raise ValueError(msg)

    def to_dict(self) -> dict[str, str | int | float | list[int] | str | None]:
        return {
            "name": self.name,
            "kind": self.kind,
            "shape": list(self.shape),
            "dtype": self.dtype,
            "status": self.status,
            "value": self.value,
            "unit": self.unit,
            "method": self.method,
            "source_type": self.source_type,
            "verification_status": self.verification_status,
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> TensorMeasurement:
        return cls(
            name=str(data["name"]),
            kind=cast("TensorKind", data["kind"]),
            shape=tuple(int(v) for v in cast("list[int]", data["shape"])),
            dtype=str(data["dtype"]),
            status=cast("MeasurementStatus", data["status"]),
            value=cast("float | None", data["value"]),
            unit=cast("str | None", data["unit"]),
            method=cast("str | None", data["method"]),
            source_type=cast("SourceType", data["source_type"]),
            verification_status=cast("VerificationStatus", data["verification_status"]),
        )


@dataclass(frozen=True)
class ModuleMeasurement:
    """Frozen dataclass holding one module's measurement record."""

    module_path: str
    family: ModuleFamily
    module_type: str
    parameter_count: int | None
    weight_kind: TensorKind
    tensors: tuple[TensorMeasurement, ...]
    status: MeasurementStatus

    def to_dict(self) -> dict[str, object]:
        return {
            "module_path": self.module_path,
            "family": self.family,
            "module_type": self.module_type,
            "parameter_count": self.parameter_count,
            "weight_kind": self.weight_kind,
            "tensors": [t.to_dict() for t in self.tensors],
            "status": self.status,
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> ModuleMeasurement:
        return cls(
            module_path=str(data["module_path"]),
            family=cast("ModuleFamily", data["family"]),
            module_type=str(data["module_type"]),
            parameter_count=cast("int | None", data["parameter_count"]),
            weight_kind=cast("TensorKind", data["weight_kind"]),
            tensors=tuple(
                TensorMeasurement.from_dict(cast("dict[str, object]", t))
                for t in cast("list[object]", data["tensors"])
            ),
            status=cast("MeasurementStatus", data["status"]),
        )


@dataclass(frozen=True)
class LayerMeasurement:
    """Frozen dataclass holding one layer's paired observation slot."""

    layer_name: str
    layer_index: int | None
    family: ModuleFamily
    output_cosine: float | None
    normalized_reconstruction_error: float | None
    output_norm_ratio: float | None
    status: MeasurementStatus
    modules: tuple[ModuleMeasurement, ...] = ()

    def __post_init__(self) -> None:
        measured_fields = {
            "output_cosine": self.output_cosine,
            "normalized_reconstruction_error": self.normalized_reconstruction_error,
            "output_norm_ratio": self.output_norm_ratio,
        }
        if self.status == "measured":
            return
        offenders = {
            field: value
            for field, value in measured_fields.items()
            if value is not None
        }
        if offenders:
            rendered = ", ".join(
                f"{field}={value!r}" for field, value in offenders.items()
            )
            msg = (
                f"{', '.join(offenders)} must be None when status is "
                f"{self.status!r}, got {rendered} (layer_name={self.layer_name!r})"
            )
            raise ValueError(msg)

    def to_dict(self) -> dict[str, object]:
        return {
            "layer_name": self.layer_name,
            "layer_index": self.layer_index,
            "family": self.family,
            "output_cosine": self.output_cosine,
            "normalized_reconstruction_error": self.normalized_reconstruction_error,
            "output_norm_ratio": self.output_norm_ratio,
            "status": self.status,
            "modules": [m.to_dict() for m in self.modules],
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> LayerMeasurement:
        return cls(
            layer_name=str(data["layer_name"]),
            layer_index=cast("int | None", data["layer_index"]),
            family=cast("ModuleFamily", data["family"]),
            output_cosine=cast("float | None", data["output_cosine"]),
            normalized_reconstruction_error=cast(
                "float | None", data["normalized_reconstruction_error"]
            ),
            output_norm_ratio=cast("float | None", data["output_norm_ratio"]),
            status=cast("MeasurementStatus", data["status"]),
            modules=tuple(
                ModuleMeasurement.from_dict(cast("dict[str, object]", m))
                for m in cast("list[object]", data.get("modules", []))
            ),
        )


@dataclass(frozen=True)
class GradientMeasurement:
    """Frozen dataclass holding gradient evidence for a paired-model comparison."""

    status: MeasurementStatus
    grad_norm: float | None
    coherence: float | None
    effective_rank: float | None
    parameter_count: int | None
    source_type: SourceType
    verification_status: VerificationStatus

    def __post_init__(self) -> None:
        measured_fields = {
            "grad_norm": self.grad_norm,
            "coherence": self.coherence,
            "effective_rank": self.effective_rank,
        }
        if self.status == "measured":
            if all(value is None for value in measured_fields.values()):
                msg = (
                    "at least one of grad_norm, coherence, effective_rank must not "
                    "be None when status is 'measured' (got all None)"
                )
                raise ValueError(msg)
            return
        offenders = {
            field: value
            for field, value in measured_fields.items()
            if value is not None
        }
        if offenders:
            rendered = ", ".join(
                f"{field}={value!r}" for field, value in offenders.items()
            )
            msg = (
                f"{', '.join(offenders)} must be None when status is "
                f"{self.status!r}, got {rendered}"
            )
            raise ValueError(msg)

    def to_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "grad_norm": self.grad_norm,
            "coherence": self.coherence,
            "effective_rank": self.effective_rank,
            "parameter_count": self.parameter_count,
            "source_type": self.source_type,
            "verification_status": self.verification_status,
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> GradientMeasurement:
        return cls(
            status=cast("MeasurementStatus", data["status"]),
            grad_norm=cast("float | None", data["grad_norm"]),
            coherence=cast("float | None", data["coherence"]),
            effective_rank=cast("float | None", data["effective_rank"]),
            parameter_count=cast("int | None", data["parameter_count"]),
            source_type=cast("SourceType", data["source_type"]),
            verification_status=cast("VerificationStatus", data["verification_status"]),
        )


@dataclass(frozen=True)
class MeasurementDocument:
    """Frozen dataclass holding the complete versioned measurement document."""

    schema_version: str = SCHEMA_VERSION
    model_id: str = ""
    checkpoint_hash: str | None = None
    role: Literal["reference", "compressed", "recovered"] = "reference"
    captured_at: str = ""
    architecture: str | None = None
    torch_version: str | None = None
    device: str | None = None
    input_fingerprint: str | None = None
    layers: tuple[LayerMeasurement, ...] = ()
    gradients: GradientMeasurement | None = None
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "model_id": self.model_id,
            "checkpoint_hash": self.checkpoint_hash,
            "role": self.role,
            "captured_at": self.captured_at,
            "architecture": self.architecture,
            "torch_version": self.torch_version,
            "device": self.device,
            "input_fingerprint": self.input_fingerprint,
            "layers": [layer.to_dict() for layer in self.layers],
            "gradients": (
                self.gradients.to_dict() if self.gradients is not None else None
            ),
            "notes": list(self.notes),
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> MeasurementDocument:
        recorded = str(data["schema_version"])
        if recorded != SCHEMA_VERSION:
            msg = (
                f"schema_version mismatch: recorded '{recorded}', "
                f"expected '{SCHEMA_VERSION}'"
            )
            raise ValueError(msg)
        return cls(
            schema_version=recorded,
            model_id=str(data["model_id"]),
            checkpoint_hash=cast("str | None", data["checkpoint_hash"]),
            role=cast("Literal['reference', 'compressed', 'recovered']", data["role"]),
            captured_at=str(data["captured_at"]),
            architecture=cast("str | None", data["architecture"]),
            torch_version=cast("str | None", data["torch_version"]),
            device=cast("str | None", data["device"]),
            input_fingerprint=cast("str | None", data["input_fingerprint"]),
            layers=tuple(
                LayerMeasurement.from_dict(cast("dict[str, object]", layer))
                for layer in cast("list[object]", data.get("layers", []))
            ),
            gradients=(
                GradientMeasurement.from_dict(
                    cast("dict[str, object]", data["gradients"])
                )
                if data.get("gradients") is not None
                else None
            ),
            notes=tuple(
                str(note) for note in cast("list[object]", data.get("notes", []))
            ),
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_json(cls, text: str) -> MeasurementDocument:
        return cls.from_dict(json.loads(text))


__all__ = [
    "SCHEMA_VERSION",
    "GradientMeasurement",
    "LayerMeasurement",
    "MeasurementDocument",
    "MeasurementStatus",
    "ModuleMeasurement",
    "TensorKind",
    "TensorMeasurement",
]
