from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any, Literal

import pytest
import torch
from pytest import MonkeyPatch

from model_headroom.analysis.orchestrator import run_analysis
from model_headroom.analysis.scoring import compute_headroom_score
from model_headroom.models.loader import detect_architecture
from model_headroom.models.types import (
    DeadNeuronReport,
    EffectiveRankResult,
    ExpertActivationData,
    ExpertWeights,
    GradientWeightPair,
    GradWeightRatio,
    GrowthFactorReport,
    HeadroomReport,
    IsoFLOPPoint,
    LossDataPoint,
    MLPActivations,
    SaturationReport,
    SignalResult,
)


def _write_hf_config(tmp_path: Any, config: dict[str, Any]) -> str:
    (tmp_path / "config.json").write_text(json.dumps(config))
    return str(tmp_path)


class TestDetectArchitecture:
    def test_hf_moe_num_local_experts(self, tmp_path: Any) -> None:
        path = _write_hf_config(
            tmp_path, {"model_type": "qwen3_moe", "num_local_experts": 8}
        )
        assert detect_architecture(path) == "moe"

    def test_hf_moe_num_experts(self, tmp_path: Any) -> None:
        path = _write_hf_config(tmp_path, {"model_type": "mixtral", "num_experts": 2})
        assert detect_architecture(path) == "moe"

    def test_hf_moe_nested_text_config(self, tmp_path: Any) -> None:
        path = _write_hf_config(
            tmp_path,
            {"text_config": {"model_type": "qwen3_moe", "num_local_experts": 4}},
        )
        assert detect_architecture(path) == "moe"

    def test_hf_dense(self, tmp_path: Any) -> None:
        path = _write_hf_config(tmp_path, {"model_type": "llama", "hidden_size": 64})
        assert detect_architecture(path) == "dense"

    def test_missing_config_returns_unknown(self, tmp_path: Any) -> None:
        assert detect_architecture(str(tmp_path / "nope")) == "unknown"

    def test_malformed_config_returns_unknown(self, tmp_path: Any) -> None:
        (tmp_path / "config.json").write_text("{not json")
        assert detect_architecture(str(tmp_path)) == "unknown"

    def test_gguf_moe(self, tmp_path: Any, monkeypatch: MonkeyPatch) -> None:
        gguf_path = tmp_path / "model.gguf"
        gguf_path.write_bytes(b"GGUF")

        class FakeConfig:
            expert_count = 8
            architecture = "mixtral"
            block_count = 1
            intermediate_size = 16
            hidden_size = 8

        monkeypatch.setattr(
            "model_headroom.models.gguf_loader.load_gguf_config",
            lambda _p: FakeConfig(),
        )
        monkeypatch.setattr(
            "model_headroom.models.gguf_loader.open_gguf_reader",
            lambda _p: SimpleNamespace(tensors=[]),
        )
        assert detect_architecture(str(gguf_path)) == "moe"

    def test_gguf_dense(self, tmp_path: Any, monkeypatch: MonkeyPatch) -> None:
        gguf_path = tmp_path / "model.gguf"
        gguf_path.write_bytes(b"GGUF")

        class FakeConfig:
            expert_count = 1
            architecture = "llama"
            block_count = 1
            intermediate_size = 16
            hidden_size = 8

        monkeypatch.setattr(
            "model_headroom.models.gguf_loader.load_gguf_config",
            lambda _p: FakeConfig(),
        )
        monkeypatch.setattr(
            "model_headroom.models.gguf_loader.open_gguf_reader",
            lambda _p: SimpleNamespace(tensors=[]),
        )
        assert detect_architecture(str(gguf_path)) == "dense"

    def test_gguf_broken_returns_unknown(
        self, tmp_path: Any, monkeypatch: MonkeyPatch
    ) -> None:
        gguf_path = tmp_path / "model.gguf"
        gguf_path.write_bytes(b"GGUF")

        def _boom(_p: Any) -> Any:
            raise ValueError("broken")

        monkeypatch.setattr("model_headroom.models.gguf_loader.load_gguf_config", _boom)
        assert detect_architecture(str(gguf_path)) == "unknown"


def _full_data() -> dict[str, Any]:
    return {
        "loss_curve": [
            LossDataPoint(tokens_trained=100_000_000, validation_loss=2.0),
            LossDataPoint(tokens_trained=200_000_000, validation_loss=1.5),
            LossDataPoint(tokens_trained=300_000_000, validation_loss=1.0),
            LossDataPoint(tokens_trained=400_000_000, validation_loss=0.5),
            LossDataPoint(tokens_trained=500_000_000, validation_loss=0.3),
        ],
        "isoflop_points": [
            IsoFLOPPoint(
                compute_flops=1e18, model_params=10_000_000, measured_loss=2.0
            ),
            IsoFLOPPoint(
                compute_flops=1e19, model_params=100_000_000, measured_loss=1.5
            ),
        ],
        "weight_pairs": [
            GradientWeightPair(
                gradient=torch.ones(10, 10) * 0.1,
                weight=torch.ones(10, 10),
                layer_name="layer_0",
            ),
        ],
        "mlp_activations": [
            MLPActivations(
                activations=torch.ones(2, 10, 64),
                layer_name="layer_0",
            ),
        ],
        "expert_weight_sets": [
            ExpertWeights(
                expert_weights=[torch.randn(32, 64) for _ in range(3)],
                layer_name="layer_0",
            ),
        ],
        "activation_data": ExpertActivationData(
            metadata={"num_experts": 3, "num_layers": 1},
            layers={"0": {"0": 33, "1": 33, "2": 34}},
        ),
    }


class TestOrchestratorDense:
    def test_dense_marks_expert_signals_single_expert(self) -> None:
        report = run_analysis(
            model_name="dense",
            model_path="/tmp/dense",
            architecture="dense",
            **_full_data(),
        )
        assert len(report.signals) == 7
        by_name = {s.signal_name: s.status for s in report.signals}
        assert by_name["load_balance"] == "single_expert"
        assert by_name["specialization"] == "single_expert"
        assert by_name["loss_saturation"] == "computed"
        assert by_name["growth_factor"] == "insufficient_data"
        assert by_name["effective_rank"] == "computed"
        assert by_name["grad_weight_ratio"] == "computed"
        assert by_name["dead_neurons"] == "computed"
        assert report.metadata["architecture"] == "dense"

    def test_dense_expert_data_is_ignored(self) -> None:
        report = run_analysis(
            model_name="dense",
            model_path="/tmp/dense",
            architecture="dense",
            **_full_data(),
        )
        by_name = {s.signal_name: s for s in report.signals}
        assert by_name["load_balance"].data.signal_status == "single_expert"
        assert by_name["load_balance"].data.balance_ratio is None
        assert by_name["load_balance"].data.entropy_per_layer == []
        assert by_name["specialization"].data.signal_status == "single_expert"

    def test_auto_detect_dense_from_config(self, tmp_path: Any) -> None:
        path = _write_hf_config(tmp_path, {"model_type": "llama", "hidden_size": 64})
        report = run_analysis(model_name="dense", model_path=path)
        by_name = {s.signal_name: s.status for s in report.signals}
        assert by_name["load_balance"] == "single_expert"
        assert by_name["specialization"] == "single_expert"
        assert report.metadata["architecture"] == "dense"

    def test_auto_detect_moe_from_config(self, tmp_path: Any) -> None:
        path = _write_hf_config(tmp_path, {"model_type": "mixtral", "num_experts": 8})
        report = run_analysis(model_name="moe", model_path=path)
        assert report.metadata["architecture"] == "moe"

    def test_moe_without_expert_data_stays_skipped(self) -> None:
        report = run_analysis(
            model_name="moe",
            model_path="/tmp/moe",
            architecture="moe",
            loss_curve=_full_data()["loss_curve"],
        )
        by_name = {s.signal_name: s.status for s in report.signals}
        assert by_name["load_balance"] == "skipped"
        assert by_name["specialization"] == "skipped"
        assert by_name["loss_saturation"] == "computed"

    def test_explicit_architecture_overrides_auto_detect(self, tmp_path: Any) -> None:
        path = _write_hf_config(tmp_path, {"model_type": "mixtral", "num_experts": 8})
        report = run_analysis(
            model_name="forced", model_path=path, architecture="dense"
        )
        assert report.metadata["architecture"] == "dense"
        by_name = {s.signal_name: s.status for s in report.signals}
        assert by_name["load_balance"] == "single_expert"

    def test_unknown_architecture_falls_back_to_moe_behavior(self) -> None:
        report = run_analysis(model_name="x", model_path="/nonexistent/path")
        assert report.metadata["architecture"] == "unknown"
        by_name = {s.signal_name: s.status for s in report.signals}
        assert by_name["load_balance"] == "skipped"
        assert by_name["specialization"] == "skipped"


class TestScoringNotApplicable:
    @staticmethod
    def _full(signal_name: str) -> SignalResult:
        if signal_name == "loss_saturation":
            data: Any = SaturationReport(
                is_saturated=False,
                saturation_point_tokens=None,
                loss_delta_threshold=0.001,
                plateau_slope=0.0,
                data_points_used=5,
                signal_status="computed",
            )
        elif signal_name == "growth_factor":
            data = GrowthFactorReport(
                alpha=1.0,
                crossover_params=None,
                crossover_loss=None,
                fit_quality_r2=0.9,
                data_points_used=5,
                signal_status="computed",
            )
        elif signal_name == "effective_rank":
            data = EffectiveRankResult(
                rank=8.0,
                entropy=2.0,
                percentage_of_full_rank=1.0,
                singular_values_count=8,
                signal_status="computed",
            )
        elif signal_name == "grad_weight_ratio":
            data = [
                GradWeightRatio(
                    ratio=0.5,
                    is_high_pressure=False,
                    z_score=0.0,
                    layer_name="l0",
                    signal_status="computed",
                )
            ]
        else:
            data = DeadNeuronReport(
                total_neurons=10,
                dead_count=0,
                dead_fraction=0.0,
                dead_indices=None,
                signal_status="computed",
            )
        return SignalResult(
            signal_name=signal_name,
            status="computed",
            data=data,
            duration_seconds=0.0,
        )

    @staticmethod
    def _report(expert_status: Literal["not_applicable", "skipped"]) -> HeadroomReport:
        universal_names = [
            "loss_saturation",
            "growth_factor",
            "effective_rank",
            "grad_weight_ratio",
            "dead_neurons",
        ]
        signals: list[SignalResult] = [
            TestScoringNotApplicable._full(n) for n in universal_names
        ]
        signals.append(
            SignalResult(
                signal_name="load_balance",
                status=expert_status,
                data=None,
                duration_seconds=0.0,
            )
        )
        signals.append(
            SignalResult(
                signal_name="specialization",
                status=expert_status,
                data=None,
                duration_seconds=0.0,
            )
        )
        return HeadroomReport(
            model_name="model",
            model_path="/tmp/model",
            timestamp="2026-01-01T00:00:00+00:00",
            signals=signals,
            overall_headroom_score=0.0,
            recommendations=["a", "b", "c"],
            metadata={"architecture": expert_status},
        )

    def test_not_applicable_excluded_from_score(self) -> None:
        report = TestScoringNotApplicable._report("not_applicable")
        assert compute_headroom_score(report) == pytest.approx(1.0)

    def test_skipped_still_counts_neutral(self) -> None:
        score = compute_headroom_score(TestScoringNotApplicable._report("skipped"))
        assert score == pytest.approx(0.8)
        assert score < 1.0


class TestDenseAsOneExpert:
    @staticmethod
    def _measured_weight_report(arch: str) -> HeadroomReport:
        return run_analysis(
            model_name=f"{arch}-model",
            model_path="/tmp/model",
            architecture=arch,  # type: ignore[arg-type]
            weight_pairs=[
                GradientWeightPair(
                    gradient=torch.ones(8, 8) * 0.1,
                    weight=torch.ones(8, 8),
                    layer_name="model.layers.0.mlp.down_proj.weight",
                )
            ],
        )

    def test_dense_and_moe_coverage_are_equal(self) -> None:
        dense = TestDenseAsOneExpert._measured_weight_report("dense")
        moe = TestDenseAsOneExpert._measured_weight_report("moe")
        assert dense.coverage_metrics is not None
        assert moe.coverage_metrics is not None
        # defaults are unavailable: nothing is measured without explicit provenance
        assert dense.coverage_metrics.measured_weight == pytest.approx(0.0)
        assert moe.coverage_metrics.measured_weight == pytest.approx(0.0)
        # dense excludes its two single_expert signals from the denominator;
        # moe (no activation/expert data) keeps all seven applicable
        assert dense.coverage_metrics.total_applicable_weight == pytest.approx(0.80)
        assert moe.coverage_metrics.total_applicable_weight == pytest.approx(1.00)
        assert dense.coverage_metrics.coverage == pytest.approx(0.0)
        assert moe.coverage_metrics.coverage == pytest.approx(0.0)

    def test_dense_expert_signals_count_in_the_denominator(self) -> None:
        report = TestDenseAsOneExpert._measured_weight_report("dense")
        assert report.coverage_metrics is not None
        assert report.coverage_metrics.unmeasured_weight == pytest.approx(0.80)

    def test_dense_load_balance_is_never_zero(self) -> None:
        report = TestDenseAsOneExpert._measured_weight_report("dense")
        signal = next(s for s in report.signals if s.signal_name == "load_balance")
        assert signal.status == "single_expert"
        assert signal.data.balance_ratio is None

    def test_moe_with_one_expert_layer_is_single_expert(self) -> None:
        report = run_analysis(
            model_name="moe-one-expert",
            model_path="/tmp/moe",
            architecture="moe",
            expert_weight_sets=[
                ExpertWeights(expert_weights=[torch.ones(4, 4)], layer_name="layer_0")
            ],
        )
        signal = next(s for s in report.signals if s.signal_name == "specialization")
        assert signal.status == "single_expert"
        assert signal.data.signal_status == "single_expert"


class TestHumanReportDense:
    def test_not_applicable_line(
        self, headroom_report_computed: HeadroomReport
    ) -> None:
        report = HeadroomReport(
            model_name="dense",
            model_path="/tmp/dense",
            timestamp="2026-01-01T00:00:00+00:00",
            signals=[
                SignalResult(
                    signal_name="load_balance",
                    status="not_applicable",
                    data=None,
                    duration_seconds=0.0,
                )
            ],
            overall_headroom_score=0.5,
            recommendations=["a", "b", "c"],
            metadata={"architecture": "dense"},
        )
        from model_headroom.reporting.human_report import generate_human_report

        output = generate_human_report(report)
        assert "[N/A" in output
        assert "DENSE" in output

    def test_arch_line(self, headroom_report_computed: HeadroomReport) -> None:
        from model_headroom.reporting.human_report import generate_human_report

        output = generate_human_report(headroom_report_computed)
        assert "Arch:" in output


class TestCliArchitecture:
    def test_architecture_flag_default_auto(self) -> None:
        from model_headroom.cli import parse_args

        args = parse_args(["--model-name", "m", "--model-path", "/tmp"])
        assert args.architecture == "auto"

    def test_architecture_flag_explicit(self) -> None:
        from model_headroom.cli import parse_args

        args = parse_args(
            ["--model-name", "m", "--model-path", "/tmp", "--architecture", "dense"]
        )
        assert args.architecture == "dense"

    def test_architecture_flag_invalid(self) -> None:
        from model_headroom.cli import parse_args

        with pytest.raises(SystemExit) as exc_info:
            parse_args(
                ["--model-name", "m", "--model-path", "/tmp", "--architecture", "llm"]
            )
        assert exc_info.value.code == 2
