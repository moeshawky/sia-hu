"""Tests for the shared device resolver: accelerator-preferred with CPU fallback."""

from __future__ import annotations

from typing import Any

import pytest
import torch

from model_headroom.device import (
    available_accelerators,
    default_device,
    resolve_device,
)

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")


class TestResolveDevice:
    def test_cpu_is_cpu(self) -> None:
        assert resolve_device("cpu") == "cpu"

    def test_auto_prefers_available_accelerator(self) -> None:
        resolved = resolve_device("auto")
        if torch.cuda.is_available():
            assert resolved == "cuda"
        else:
            assert resolved == "cpu"

    def test_explicit_cuda_available(self) -> None:
        if torch.cuda.is_available():
            assert resolve_device("cuda") == "cuda"
        else:
            assert resolve_device("cuda") is None

    def test_explicit_mps_unavailable(self) -> None:
        if not torch.backends.mps.is_available():
            assert resolve_device("mps") is None

    def test_unknown_device_is_error(self) -> None:
        assert resolve_device("bogus") is None


class TestDefaultDevice:
    def test_accelerator_preferred(self) -> None:
        if torch.cuda.is_available():
            assert default_device() == "cuda"
        else:
            assert default_device() == "cpu"


class TestAvailableAccelerators:
    def test_matches_torch_cuda(self) -> None:
        if torch.cuda.is_available():
            assert "cuda" in available_accelerators()
        else:
            assert "cuda" not in available_accelerators()


class TestDeviceThreading:
    def test_run_analysis_records_device(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from model_headroom.analysis.orchestrator import run_analysis

        report = run_analysis(model_name="test", model_path="/tmp/test", device="cpu")
        assert report.metadata["device"] == "cpu"

    def test_run_analysis_default_resolves_to_accelerator(self) -> None:
        from model_headroom.analysis.orchestrator import run_analysis
        from model_headroom.device import default_device

        report = run_analysis(model_name="test", model_path="/tmp/test")
        assert report.metadata["device"] == default_device()


class TestSignalsHonorDevice:
    def test_compute_effective_rank_moves_to_device(self) -> None:
        from model_headroom.signals.effective_rank import compute_effective_rank

        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        weight = torch.randn(32, 64)
        result = compute_effective_rank(weight, device="cuda")
        assert result.signal_status == "computed"

    def test_compute_effective_rank_cpu(self) -> None:
        from model_headroom.signals.effective_rank import compute_effective_rank

        weight = torch.randn(32, 64)
        result = compute_effective_rank(weight, device="cpu")
        assert result.signal_status == "computed"

    def test_analyze_grad_weight_ratios_accepts_device(self) -> None:
        from model_headroom.models.types import GradientWeightPair
        from model_headroom.signals.grad_weight_ratio import (
            analyze_grad_weight_ratios,
        )

        pairs = [
            GradientWeightPair(
                gradient=torch.ones(8, 8) * 0.1,
                weight=torch.ones(8, 8),
                layer_name="layer_0",
            )
        ]
        result = analyze_grad_weight_ratios(pairs, device="cpu")
        assert len(result) == 1

    def test_analyze_dead_neurons_on_device(self) -> None:
        from model_headroom.models.types import MLPActivations
        from model_headroom.signals.dead_neurons import analyze_dead_neurons

        acts = MLPActivations(activations=torch.ones(2, 10, 64), layer_name="l0")
        result = analyze_dead_neurons(acts, device="cpu")
        assert result.signal_status == "computed"

    def test_analyze_specialization_on_device(self) -> None:
        from model_headroom.models.types import ExpertWeights
        from model_headroom.signals.specialization import analyze_specialization

        ew = ExpertWeights(
            expert_weights=[torch.randn(16, 32) for _ in range(3)],
            layer_name="l0",
        )
        result = analyze_specialization(ew, device="cpu")
        assert result.signal_status == "computed"


class TestCliDeviceFlag:
    def test_device_auto_reaches_run_analysis(
        self, tmp_path: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from model_headroom import cli
        from model_headroom.models.types import HeadroomReport

        seen: dict[str, Any] = {}

        def fake_run_analysis(**kwargs: Any) -> Any:
            seen.update(kwargs)
            return HeadroomReport(
                model_name="t",
                model_path=str(tmp_path),
                timestamp="2026-01-01T00:00:00+00:00",
                signals=[],
                overall_headroom_score=0.0,
                recommendations=[],
                metadata={},
            )

        monkeypatch.setattr(cli, "run_analysis", fake_run_analysis)
        rc = cli.main(["--model-name", "t", "--model-path", str(tmp_path)])
        assert rc == 0
        from model_headroom.device import default_device

        assert seen["device"] == default_device()

    def test_device_cpu_reaches_run_analysis(
        self, tmp_path: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from model_headroom import cli
        from model_headroom.models.types import HeadroomReport

        seen: dict[str, Any] = {}

        def fake_run_analysis(**kwargs: Any) -> Any:
            seen.update(kwargs)
            return HeadroomReport(
                model_name="t",
                model_path=str(tmp_path),
                timestamp="2026-01-01T00:00:00+00:00",
                signals=[],
                overall_headroom_score=0.0,
                recommendations=[],
                metadata={},
            )

        monkeypatch.setattr(cli, "run_analysis", fake_run_analysis)
        rc = cli.main(
            [
                "--model-name",
                "t",
                "--model-path",
                str(tmp_path),
                "--device",
                "cpu",
            ]
        )
        assert rc == 0
        assert seen["device"] == "cpu"

    def test_device_cuda_unavailable_exits_1(
        self, tmp_path: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        if torch.cuda.is_available():
            pytest.skip("CUDA available")
        from model_headroom import cli

        rc = cli.main(
            [
                "--model-name",
                "t",
                "--model-path",
                str(tmp_path),
                "--device",
                "cuda",
            ]
        )
        assert rc == 1

    def test_device_cuda_reaches_run_analysis(
        self, tmp_path: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        from model_headroom import cli
        from model_headroom.models.types import HeadroomReport

        seen: dict[str, Any] = {}

        def fake_run_analysis(**kwargs: Any) -> Any:
            seen.update(kwargs)
            return HeadroomReport(
                model_name="t",
                model_path=str(tmp_path),
                timestamp="2026-01-01T00:00:00+00:00",
                signals=[],
                overall_headroom_score=0.0,
                recommendations=[],
                metadata={},
            )

        monkeypatch.setattr(cli, "run_analysis", fake_run_analysis)
        rc = cli.main(
            [
                "--model-name",
                "t",
                "--model-path",
                str(tmp_path),
                "--device",
                "cuda",
            ]
        )
        assert rc == 0
        assert seen["device"] == "cuda"
