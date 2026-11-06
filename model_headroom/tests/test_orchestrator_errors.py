from __future__ import annotations

from unittest.mock import patch

import torch

from model_headroom.analysis.orchestrator import run_analysis
from model_headroom.models.types import (
    ExpertActivationData,
    ExpertWeights,
    GradientWeightPair,
    MLPActivations,
    SaturationReport,
)


class TestOrchestratorAllSignalsError:
    def test_all_signals_raise_exceptions(self) -> None:
        def _boom(*args: object, **kwargs: object) -> None:
            raise RuntimeError("total failure")

        with (
            patch(
                "model_headroom.analysis.orchestrator.analyze_loss_saturation",
                side_effect=_boom,
            ),
            patch(
                "model_headroom.analysis.orchestrator.analyze_growth_factor",
                side_effect=_boom,
            ),
            patch(
                "model_headroom.analysis.orchestrator.compute_effective_rank",
                side_effect=_boom,
            ),
            patch(
                "model_headroom.analysis.orchestrator.analyze_grad_weight_ratios",
                side_effect=_boom,
            ),
            patch(
                "model_headroom.analysis.orchestrator.analyze_dead_neurons",
                side_effect=_boom,
            ),
            patch(
                "model_headroom.analysis.orchestrator.analyze_load_balance",
                side_effect=_boom,
            ),
            patch(
                "model_headroom.analysis.orchestrator.analyze_specialization",
                side_effect=_boom,
            ),
        ):
            report = run_analysis(
                model_name="test",
                model_path="/tmp/test",
                loss_curve=[
                    __import__(
                        "model_headroom.models.types", fromlist=["LossDataPoint"]
                    ).LossDataPoint(tokens_trained=100000000, validation_loss=2.0)
                ],
                isoflop_points=[
                    __import__(
                        "model_headroom.models.types", fromlist=["IsoFLOPPoint"]
                    ).IsoFLOPPoint(
                        compute_flops=1e18, model_params=10000000, measured_loss=2.0
                    )
                ],
                weight_pairs=[
                    GradientWeightPair(
                        gradient=torch.ones(4, 4),
                        weight=torch.ones(4, 4),
                        layer_name="l0",
                    )
                ],
                mlp_activations=[
                    MLPActivations(
                        activations=torch.ones(2, 4, 8),
                        layer_name="l0",
                    )
                ],
                expert_weight_sets=[
                    ExpertWeights(
                        expert_weights=[torch.randn(4, 8), torch.randn(4, 8)],
                        layer_name="l0",
                    )
                ],
                activation_data=ExpertActivationData(
                    metadata={"num_experts": 2, "num_layers": 1},
                    layers={"0": {"0": 50, "1": 50}},
                ),
            )
        assert len(report.signals) == 7
        assert all(s.status == "error" for s in report.signals)
        assert 0.0 <= report.overall_headroom_score <= 1.0


class TestOrchestratorSignalReturnsErrorReport:
    def test_loss_saturation_returns_error_report(self) -> None:
        error_report = SaturationReport(
            is_saturated=False,
            saturation_point_tokens=None,
            loss_delta_threshold=0.001,
            plateau_slope=0.0,
            data_points_used=0,
            signal_status="error",
            error_message="not enough data",
        )

        with patch(
            "model_headroom.analysis.orchestrator.analyze_loss_saturation",
            return_value=error_report,
        ):
            report = run_analysis(
                model_name="test",
                model_path="/tmp/test",
                loss_curve=[{"tokens_trained": 1, "validation_loss": 1.0}],
            )
        loss_signal = next(
            s for s in report.signals if s.signal_name == "loss_saturation"
        )
        assert loss_signal.status == "error"


class TestSpecializationZeroNormTensors:
    def test_zero_norm_expert_weights(self) -> None:
        from model_headroom.signals.specialization import analyze_specialization

        zero_weights = ExpertWeights(
            expert_weights=[
                torch.zeros(4, 8),
                torch.zeros(4, 8),
            ],
            layer_name="l0",
        )
        result = analyze_specialization(zero_weights)
        assert result.signal_status in ("computed", "error")


class TestOrchestratorWeightPairsWithDifferentSourceTypes:
    def test_weight_pairs_with_synthetic_source(self) -> None:
        report = run_analysis(
            model_name="test",
            model_path="/tmp/test",
            weight_pairs=[
                GradientWeightPair(
                    gradient=torch.ones(4, 4) * 0.1,
                    weight=torch.ones(4, 4),
                    layer_name="l0",
                )
            ],
            weight_pairs_source_type="synthetic",
            weight_pairs_verification_status="unverified",
        )
        rank_signal = next(
            s for s in report.signals if s.signal_name == "effective_rank"
        )
        assert rank_signal.source_type == "synthetic"
        grad_signal = next(
            s for s in report.signals if s.signal_name == "grad_weight_ratio"
        )
        assert grad_signal.source_type == "synthetic"
