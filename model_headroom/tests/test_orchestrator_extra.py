from __future__ import annotations

from unittest.mock import patch

import torch

from model_headroom.analysis.orchestrator import _run_signal, run_analysis
from model_headroom.models.types import (
    DeadNeuronReport,
    EffectiveRankResult,
    ExpertWeights,
    GradientWeightPair,
    MLPActivations,
    SaturationReport,
)


class TestRunSignalErrorPaths:
    def test_signal_returns_error_status(self) -> None:
        def failing_signal() -> SaturationReport:
            return SaturationReport(
                is_saturated=False,
                saturation_point_tokens=None,
                loss_delta_threshold=0.001,
                plateau_slope=0.0,
                data_points_used=0,
                signal_status="error",
                error_message="something broke",
            )

        result = _run_signal("test_signal", failing_signal)
        assert result.status == "error"
        assert result.signal_name == "test_signal"
        assert result.data is not None
        assert result.data.signal_status == "error"

    def test_signal_raises_exception(self) -> None:
        def exploding_signal() -> None:
            raise ValueError("kaboom")

        result = _run_signal("test_signal", exploding_signal)
        assert result.status == "error"
        assert result.error_message == "kaboom"
        assert result.data is None


class TestOrchestratorEffectiveRankAllFail:
    def test_all_rank_layers_fail(self) -> None:
        with patch(
            "model_headroom.analysis.orchestrator.compute_effective_rank"
        ) as mock:
            mock.return_value = EffectiveRankResult(
                rank=1.0,
                entropy=0.0,
                percentage_of_full_rank=0.0,
                singular_values_count=0,
                signal_status="error",
                error_message="fail",
            )
            report = run_analysis(
                model_name="test",
                model_path="/tmp/test",
                weight_pairs=[
                    GradientWeightPair(
                        gradient=torch.ones(4, 4),
                        weight=torch.ones(4, 4),
                        layer_name="l0",
                    ),
                ],
            )
        rank_signal = next(
            s for s in report.signals if s.signal_name == "effective_rank"
        )
        assert rank_signal.status == "error"


class TestOrchestratorDeadNeuronsAllFail:
    def test_all_dead_neuron_layers_fail(self) -> None:
        with patch("model_headroom.analysis.orchestrator.analyze_dead_neurons") as mock:
            mock.return_value = DeadNeuronReport(
                total_neurons=0,
                dead_count=0,
                dead_fraction=0.0,
                dead_indices=None,
                signal_status="error",
                error_message="fail",
            )
            report = run_analysis(
                model_name="test",
                model_path="/tmp/test",
                mlp_activations=[
                    MLPActivations(
                        activations=torch.ones(2, 10, 32),
                        layer_name="l0",
                    ),
                ],
            )
        dead_signal = next(s for s in report.signals if s.signal_name == "dead_neurons")
        assert dead_signal.status == "error"


class TestOrchestratorSpecializationAllFail:
    def test_all_specialization_layers_fail(self) -> None:

        def failing_spec(ew: ExpertWeights) -> None:
            raise RuntimeError("specialization failed")

        with patch(
            "model_headroom.analysis.orchestrator.analyze_specialization",
            side_effect=failing_spec,
        ):
            report = run_analysis(
                model_name="test",
                model_path="/tmp/test",
                expert_weight_sets=[
                    ExpertWeights(
                        expert_weights=[torch.randn(4, 8), torch.randn(4, 8)],
                        layer_name="l0",
                    ),
                ],
            )
        spec_signal = next(
            s for s in report.signals if s.signal_name == "specialization"
        )
        assert spec_signal.status == "error"
