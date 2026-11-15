from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import torch

from model_headroom.models.types import RefusalDirectionReport
from model_headroom.refusal.artifact import (
    TENSOR_DOMAIN_DIRECTION,
    TENSOR_REFUSAL_DIRECTION,
    write_refusal_artifact,
)

if TYPE_CHECKING:
    from pathlib import Path


def _unit(dim: int = 8) -> torch.Tensor:
    vector = torch.arange(1, dim + 1, dtype=torch.float32)
    return vector / vector.norm()


def _report(
    *,
    axis: str = "mean_difference",
    direction: torch.Tensor | None = None,
    cohen_d: float | None = 1.0,
    hidden_size: int = 8,
    layer_index: int | None = 0,
    orthogonalized: bool = False,
    selection: str = "depth_fraction",
    status: str = "computed",
) -> RefusalDirectionReport:
    return RefusalDirectionReport(
        axis=axis,
        direction=direction,
        layer_name="l0" if direction is not None else None,
        layer_index=layer_index,
        hidden_size=hidden_size,
        harmful_count=10,
        harmless_count=10,
        layer_stats=[],
        cohen_d=cohen_d,
        orthogonalized=orthogonalized,
        selection=selection,
        direction_status=status,
    )


class TestRoundTrip:
    def test_loads_back_unit_float32_direction(self, tmp_path: Path) -> None:
        from safetensors.torch import load_file

        report = _report(direction=_unit(8))
        out = write_refusal_artifact(report, tmp_path / "artifact.safetensors")
        assert out == tmp_path / "artifact.safetensors"
        tensors = load_file(str(out))
        assert set(tensors) == {TENSOR_REFUSAL_DIRECTION}
        loaded = tensors[TENSOR_REFUSAL_DIRECTION]
        assert loaded.dtype == torch.float32
        assert loaded.shape == (8,)
        assert abs(float(loaded.norm()) - 1.0) <= 1e-5

    def test_every_metadata_value_is_str(self, tmp_path: Path) -> None:
        from safetensors import safe_open

        report = _report(direction=_unit(8))
        write_refusal_artifact(report, tmp_path / "a.safetensors")
        with safe_open(tmp_path / "a.safetensors", framework="pt") as handle:
            metadata = handle.metadata()
        assert metadata is not None
        assert all(isinstance(value, str) for value in metadata.values())

    def test_domain_tensor_present_only_when_supplied(self, tmp_path: Path) -> None:
        from safetensors.torch import load_file

        report = _report(direction=_unit(8))
        domain = _unit(8)
        out = write_refusal_artifact(
            report, tmp_path / "b.safetensors", domain_direction=domain
        )
        assert out == tmp_path / "b.safetensors"
        tensors = load_file(str(out))
        assert set(tensors) == {TENSOR_REFUSAL_DIRECTION, TENSOR_DOMAIN_DIRECTION}
        loaded_domain = tensors[TENSOR_DOMAIN_DIRECTION]
        assert loaded_domain.dtype == torch.float32
        assert loaded_domain.shape == (8,)
        assert abs(float(loaded_domain.norm()) - 1.0) <= 1e-5

    def test_no_domain_when_omitted(self, tmp_path: Path) -> None:
        from safetensors.torch import load_file

        report = _report(direction=_unit(8))
        write_refusal_artifact(report, tmp_path / "c.safetensors")
        assert TENSOR_DOMAIN_DIRECTION not in load_file(str(tmp_path / "c.safetensors"))


class TestMetadata:
    def test_model_id_absent_is_empty_string(self, tmp_path: Path) -> None:
        from safetensors import safe_open

        report = _report(direction=_unit(8))
        write_refusal_artifact(report, tmp_path / "d.safetensors")
        with safe_open(tmp_path / "d.safetensors", framework="pt") as handle:
            metadata = handle.metadata()
        assert metadata is not None
        assert metadata["model_id"] == ""

    def test_cohen_d_nan_case(self, tmp_path: Path) -> None:
        from safetensors import safe_open

        report = _report(
            axis="weight_delta_svd",
            direction=_unit(8),
            cohen_d=float("nan"),
            hidden_size=8,
            layer_index=-1,
            selection="weight_delta_svd",
        )
        write_refusal_artifact(report, tmp_path / "e.safetensors")
        with safe_open(tmp_path / "e.safetensors", framework="pt") as handle:
            metadata = handle.metadata()
        assert metadata is not None
        assert metadata["cohen_d"] == "nan"
        assert metadata["cohen_d_kind"] == "svd_dominance"
        assert metadata["capture_point"] == "weight_delta"
        assert metadata["source"] == "weight_delta_recovery"

    def test_cohen_d_kind_effect_size_for_mean_difference(self, tmp_path: Path) -> None:
        from safetensors import safe_open

        report = _report(direction=_unit(8), cohen_d=0.5)
        write_refusal_artifact(report, tmp_path / "f.safetensors")
        with safe_open(tmp_path / "f.safetensors", framework="pt") as handle:
            metadata = handle.metadata()
        assert metadata is not None
        assert metadata["cohen_d_kind"] == "effect_size"
        assert metadata["cohen_d"] == repr(0.5)
        assert metadata["capture_point"] == "last_token_residual"
        assert metadata["source"] == "base_model_activation_difference"

    def test_direction_norm_records_pre_normalization_norm(
        self, tmp_path: Path
    ) -> None:
        from safetensors import safe_open
        from safetensors.torch import load_file

        report = _report(direction=torch.tensor([3.0, 4.0]))
        write_refusal_artifact(report, tmp_path / "g.safetensors")
        with safe_open(tmp_path / "g.safetensors", framework="pt") as handle:
            metadata = handle.metadata()
        assert metadata is not None
        assert metadata["direction_norm"] == repr(5.0)
        loaded = load_file(str(tmp_path / "g.safetensors"))[TENSOR_REFUSAL_DIRECTION]
        assert abs(float(loaded.norm()) - 1.0) <= 1e-5


class TestErrors:
    def test_none_direction_raises_and_writes_no_file(self, tmp_path: Path) -> None:
        report = _report(direction=None, cohen_d=None, status="insufficient_data")
        path = tmp_path / "h.safetensors"
        with pytest.raises(ValueError, match="no refusal direction computed"):
            write_refusal_artifact(report, path)
        assert not path.exists()

    def test_domain_length_mismatch_raises(self, tmp_path: Path) -> None:
        report = _report(direction=_unit(8))
        domain = torch.zeros(4)
        with pytest.raises(
            ValueError, match=r"domain_direction length 4 != report.hidden_size 8"
        ):
            write_refusal_artifact(
                report, tmp_path / "i.safetensors", domain_direction=domain
            )
        assert not (tmp_path / "i.safetensors").exists()


class TestDefaults:
    def test_creates_missing_parent_directory(self, tmp_path: Path) -> None:
        from safetensors.torch import load_file

        report = _report(direction=_unit(8))
        out = write_refusal_artifact(
            report, tmp_path / "deep" / "nested" / "j.safetensors"
        )
        assert out.exists()
        loaded = load_file(str(out))[TENSOR_REFUSAL_DIRECTION]
        assert loaded.shape == (8,)

    def test_timestamp_defaults_to_iso_utc(self, tmp_path: Path) -> None:
        from safetensors import safe_open

        report = _report(direction=_unit(8))
        write_refusal_artifact(report, tmp_path / "k.safetensors")
        with safe_open(tmp_path / "k.safetensors", framework="pt") as handle:
            metadata = handle.metadata()
        assert metadata is not None
        assert metadata["timestamp"].endswith("Z")
        assert "T" in metadata["timestamp"]
