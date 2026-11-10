"""Adversarial tests for weight-delta direction recovery."""

from __future__ import annotations

import json
import math
from typing import TYPE_CHECKING

import pytest
import torch

if TYPE_CHECKING:
    from pathlib import Path

from model_headroom.refusal.recover import (
    load_weight_deltas_from_json,
    recover_direction_from_weight_deltas,
)


def _rank1_deltas(seed: int, n: int = 4, out: int = 6, hidden: int = 10) -> tuple:
    # Output-space (Arditi) abliteration: delta = - r (W^T r)^T, so the
    # refusal direction ``r`` has length ``out`` (out_features) and is the
    # top LEFT singular vector of the horizontally-stacked deltas.
    gen = torch.Generator().manual_seed(seed)
    direction = torch.randn(out, generator=gen)
    direction = direction / direction.norm()
    deltas = [
        torch.outer(direction, torch.randn(hidden, generator=gen)) for _ in range(n)
    ]
    return (deltas, direction)


class TestRecovery:
    def test_recovers_exact_rank1_direction(self) -> None:
        deltas, truth = _rank1_deltas(seed=1)
        report = recover_direction_from_weight_deltas(deltas)
        assert report.direction_status == "computed"
        assert report.direction is not None
        assert abs(abs(float(report.direction @ truth)) - 1.0) < 1e-4

    def test_direction_is_unit_norm(self) -> None:
        deltas, _ = _rank1_deltas(seed=2)
        report = recover_direction_from_weight_deltas(deltas)
        assert report.direction is not None
        assert float(report.direction.norm()) == pytest.approx(1.0, abs=1e-5)

    def test_dominance_is_near_one_for_rank1(self) -> None:
        deltas, _ = _rank1_deltas(seed=3)
        report = recover_direction_from_weight_deltas(deltas)
        assert report.cohen_d is not None
        assert report.cohen_d > 0.999

    def test_axis_is_declared(self) -> None:
        report = recover_direction_from_weight_deltas(_rank1_deltas(seed=4)[0])
        assert report.axis == "weight_delta_svd"

    def test_single_delta_is_an_error(self) -> None:
        report = recover_direction_from_weight_deltas([torch.ones(3, 4)])
        assert report.direction_status == "error"
        assert "at least 2" in (report.error_message or "")

    def test_empty_is_an_error(self) -> None:
        assert recover_direction_from_weight_deltas([]).direction_status == "error"

    def test_one_dimensional_delta_is_an_error(self) -> None:
        report = recover_direction_from_weight_deltas([torch.ones(4), torch.ones(4)])
        assert report.direction_status == "error"

    def test_mismatched_heights_are_an_error(self) -> None:
        report = recover_direction_from_weight_deltas(
            [torch.ones(3, 4), torch.ones(4, 4)]
        )
        assert report.direction_status == "error"
        assert "out_features" in (report.error_message or "")

    def test_zero_deltas_are_degenerate(self) -> None:
        report = recover_direction_from_weight_deltas(
            [torch.zeros(3, 4), torch.zeros(3, 4)]
        )
        assert report.direction_status == "degenerate"

    def test_non_finite_delta_is_an_error(self) -> None:
        poisoned = torch.ones(3, 4)
        poisoned[0, 0] = float("nan")
        report = recover_direction_from_weight_deltas([poisoned, torch.ones(3, 4)])
        assert report.direction_status == "error"

    def test_noisy_deltas_still_recover_approximately(self) -> None:
        deltas, truth = _rank1_deltas(seed=5)
        noisy = [d + 0.01 * torch.randn_like(d) for d in deltas]
        report = recover_direction_from_weight_deltas(noisy)
        assert report.direction is not None
        assert abs(abs(float(report.direction @ truth)) - 1.0) < 0.05

    def test_orthogonal_noise_defeats_recovery(self) -> None:
        gen = torch.Generator().manual_seed(6)
        report = recover_direction_from_weight_deltas(
            [torch.randn(5, 7, generator=gen) for _ in range(6)]
        )
        assert report.direction_status == "computed"
        assert report.cohen_d is not None
        assert report.cohen_d < 0.9


class TestRecoveryDevice:
    def test_cpu_device_returns_cpu_direction(self) -> None:
        deltas, truth = _rank1_deltas(seed=1)
        report = recover_direction_from_weight_deltas(deltas, device="cpu")
        assert report.direction_status == "computed"
        assert report.direction is not None
        assert report.direction.device.type == "cpu"
        assert abs(abs(float(report.direction @ truth)) - 1.0) < 1e-4

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
    def test_cuda_device_returns_home_to_cpu(self) -> None:
        deltas, truth = _rank1_deltas(seed=1)
        report = recover_direction_from_weight_deltas(deltas, device="cuda")
        assert report.direction_status == "computed"
        assert report.direction is not None
        # Compute ran on CUDA; the returned direction comes home to CPU so the
        # serialized report keeps a device-independent type.
        assert report.direction.device.type == "cpu"
        assert abs(abs(float(report.direction @ truth)) - 1.0) < 1e-4

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
    def test_svd_executes_on_requested_device(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        recorded: list[str] = []
        real_svd = torch.linalg.svd

        def recording_svd(*args, **kwargs):
            stacked = args[0]
            recorded.append(stacked.device.type)
            return real_svd(*args, **kwargs)

        monkeypatch.setattr(torch.linalg, "svd", recording_svd)
        deltas, _ = _rank1_deltas(seed=1)
        report = recover_direction_from_weight_deltas(deltas, device="cuda")
        assert report.direction_status == "computed"
        assert recorded == ["cuda"], f"SVD ran on {recorded}, expected cuda"


class TestDeltaLoading:
    def test_loads_wrapped_document(self, tmp_path: Path) -> None:
        path = tmp_path / "deltas.json"
        path.write_text(json.dumps({"deltas": [[[1.0, 2.0], [3.0, 4.0]]]}))
        deltas = load_weight_deltas_from_json(str(path))
        assert deltas is not None
        assert len(deltas) == 1
        assert deltas[0].shape == (2, 2)

    def test_loads_bare_list(self, tmp_path: Path) -> None:
        path = tmp_path / "deltas.json"
        path.write_text(json.dumps([[[1.0, 2.0], [3.0, 4.0]]]))
        deltas = load_weight_deltas_from_json(str(path))
        assert deltas is not None
        assert len(deltas) == 1

    def test_loads_object_entries(self, tmp_path: Path) -> None:
        path = tmp_path / "deltas.json"
        path.write_text(json.dumps({"deltas": [{"delta": [[1.0], [2.0]]}]}))
        deltas = load_weight_deltas_from_json(str(path))
        assert deltas is not None
        assert deltas[0].shape == (2, 1)

    def test_missing_file_returns_none(self, tmp_path: Path) -> None:
        assert load_weight_deltas_from_json(str(tmp_path / "nope.json")) is None

    def test_malformed_json_returns_none(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.json"
        path.write_text("{not json")
        assert load_weight_deltas_from_json(str(path)) is None

    def test_one_dimensional_entries_are_skipped(self, tmp_path: Path) -> None:
        path = tmp_path / "deltas.json"
        path.write_text(json.dumps({"deltas": [[1.0, 2.0], [[3.0, 4.0]]]}))
        deltas = load_weight_deltas_from_json(str(path))
        assert deltas is not None
        assert len(deltas) == 1

    def test_all_skipped_returns_none(self, tmp_path: Path) -> None:
        path = tmp_path / "deltas.json"
        path.write_text(json.dumps({"deltas": [[1.0, 2.0]]}))
        assert load_weight_deltas_from_json(str(path)) is None

    def test_empty_list_returns_none(self, tmp_path: Path) -> None:
        path = tmp_path / "deltas.json"
        path.write_text(json.dumps({"deltas": []}))
        assert load_weight_deltas_from_json(str(path)) is None


def test_roundtrip_file_to_recovery(tmp_path: Path) -> None:
    deltas, truth = _rank1_deltas(seed=7, out=3, hidden=5)
    path = tmp_path / "deltas.json"
    path.write_text(json.dumps({"deltas": [d.tolist() for d in deltas]}))
    loaded = load_weight_deltas_from_json(str(path))
    assert loaded is not None
    report = recover_direction_from_weight_deltas(loaded)
    assert report.direction is not None
    assert abs(abs(float(report.direction @ truth)) - 1.0) < 1e-4
    assert not math.isnan(report.cohen_d or 0.0)
