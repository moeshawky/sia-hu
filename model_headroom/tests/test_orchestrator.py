from __future__ import annotations

from model_headroom.analysis.orchestrator import run_analysis
from model_headroom.models.types import (
    ExpertActivationData,
    ExpertWeights,
    GradientWeightPair,
    IsoFLOPPoint,
    LossDataPoint,
    MLPActivations,
)


class TestOrchestratorFunctional:
    def test_all_signals_computed(self) -> None:
        report = run_analysis(
            model_name="test",
            model_path="/tmp/test",
            loss_curve=[
                LossDataPoint(tokens_trained=100_000_000, validation_loss=2.0),
                LossDataPoint(tokens_trained=200_000_000, validation_loss=1.5),
                LossDataPoint(tokens_trained=300_000_000, validation_loss=1.0),
                LossDataPoint(tokens_trained=400_000_000, validation_loss=0.5),
                LossDataPoint(tokens_trained=500_000_000, validation_loss=0.3),
            ],
            isoflop_points=[
                IsoFLOPPoint(
                    compute_flops=1e18, model_params=10_000_000, measured_loss=2.0
                ),
                IsoFLOPPoint(
                    compute_flops=1e19, model_params=100_000_000, measured_loss=1.5
                ),
                IsoFLOPPoint(
                    compute_flops=1e20, model_params=1_000_000_000, measured_loss=1.2
                ),
                IsoFLOPPoint(
                    compute_flops=1e21, model_params=10_000_000_000, measured_loss=1.0
                ),
            ],
            weight_pairs=[
                GradientWeightPair(
                    gradient=__import__("torch").ones(10, 10) * 0.1,
                    weight=__import__("torch").ones(10, 10),
                    layer_name="layer_0",
                ),
            ],
            mlp_activations=[
                MLPActivations(
                    activations=__import__("torch").ones(2, 10, 64),
                    layer_name="layer_0",
                ),
            ],
            expert_weight_sets=[
                ExpertWeights(
                    expert_weights=[
                        __import__("torch").randn(32, 64) for _ in range(3)
                    ],
                    layer_name="layer_0",
                ),
            ],
            activation_data=ExpertActivationData(
                metadata={"num_experts": 3, "num_layers": 1},
                layers={"0": {"0": 33, "1": 33, "2": 34}},
            ),
        )
        assert len(report.signals) == 7
        assert all(s.status == "computed" for s in report.signals)
        assert 0.0 <= report.overall_headroom_score <= 1.0
        assert len(report.recommendations) >= 3


class TestOrchestratorMixed:
    def test_mixed_computed_skipped(self) -> None:
        report = run_analysis(
            model_name="test",
            model_path="/tmp/test",
            loss_curve=[
                LossDataPoint(tokens_trained=100_000_000, validation_loss=2.0),
                LossDataPoint(tokens_trained=200_000_000, validation_loss=1.5),
                LossDataPoint(tokens_trained=300_000_000, validation_loss=1.0),
                LossDataPoint(tokens_trained=400_000_000, validation_loss=0.5),
                LossDataPoint(tokens_trained=500_000_000, validation_loss=0.3),
            ],
        )
        computed = sum(1 for s in report.signals if s.status == "computed")
        skipped = sum(1 for s in report.signals if s.status == "skipped")
        assert computed == 1
        assert skipped == 6

    def test_all_skipped(self) -> None:
        report = run_analysis(model_name="test", model_path="/tmp/test")
        assert len(report.signals) == 7
        assert all(s.status == "skipped" for s in report.signals)
        assert 0.0 <= report.overall_headroom_score <= 1.0
        assert len(report.recommendations) >= 3


class TestOrchestratorTiming:
    def test_duration_non_negative(self) -> None:
        report = run_analysis(model_name="test", model_path="/tmp/test")
        assert all(s.duration_seconds >= 0.0 for s in report.signals)
