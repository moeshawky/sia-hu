from __future__ import annotations

import json
from typing import Any

import pytest
import torch

from model_headroom.models.types import (
    DeadNeuronReport,
    EffectiveRankResult,
    ExpertActivationData,
    ExpertWeights,
    GradientWeightPair,
    GrowthFactorReport,
    HeadroomReport,
    IsoFLOPPoint,
    LoadBalanceReport,
    LossDataPoint,
    MLPActivations,
    SignalResult,
)


@pytest.fixture
def loss_curve_saturated() -> list[LossDataPoint]:
    return [
        LossDataPoint(tokens_trained=100_000_000, validation_loss=2.0),
        LossDataPoint(tokens_trained=200_000_000, validation_loss=1.5),
        LossDataPoint(tokens_trained=300_000_000, validation_loss=1.2),
        LossDataPoint(tokens_trained=400_000_000, validation_loss=1.199),
        LossDataPoint(tokens_trained=500_000_000, validation_loss=1.198),
        LossDataPoint(tokens_trained=600_000_000, validation_loss=1.197),
        LossDataPoint(tokens_trained=700_000_000, validation_loss=1.196),
    ]


@pytest.fixture
def loss_curve_unsaturated() -> list[LossDataPoint]:
    return [
        LossDataPoint(tokens_trained=100_000_000, validation_loss=2.0),
        LossDataPoint(tokens_trained=200_000_000, validation_loss=1.5),
        LossDataPoint(tokens_trained=300_000_000, validation_loss=1.0),
        LossDataPoint(tokens_trained=400_000_000, validation_loss=0.5),
        LossDataPoint(tokens_trained=500_000_000, validation_loss=0.3),
    ]


@pytest.fixture
def isoflop_points_good() -> list[IsoFLOPPoint]:
    return [
        IsoFLOPPoint(compute_flops=1e18, model_params=10_000_000, measured_loss=2.0),
        IsoFLOPPoint(compute_flops=1e19, model_params=100_000_000, measured_loss=1.5),
        IsoFLOPPoint(compute_flops=1e20, model_params=1_000_000_000, measured_loss=1.2),
        IsoFLOPPoint(
            compute_flops=1e21, model_params=10_000_000_000, measured_loss=1.0
        ),
        IsoFLOPPoint(
            compute_flops=1e22, model_params=100_000_000_000, measured_loss=0.85
        ),
    ]


@pytest.fixture
def weight_matrix_full_rank() -> torch.Tensor:
    return torch.randn(8, 8)


@pytest.fixture
def weight_matrix_low_rank() -> torch.Tensor:
    w = torch.zeros(8, 8)
    w[:2, :2] = torch.eye(2)
    return w


@pytest.fixture
def grad_weight_pairs() -> list[GradientWeightPair]:
    return [
        GradientWeightPair(
            gradient=torch.ones(10, 10) * 0.1,
            weight=torch.ones(10, 10),
            layer_name="layer_0",
        ),
        GradientWeightPair(
            gradient=torch.ones(10, 10) * 0.1,
            weight=torch.ones(10, 10),
            layer_name="layer_1",
        ),
        GradientWeightPair(
            gradient=torch.ones(10, 10) * 5.0,
            weight=torch.ones(10, 10),
            layer_name="layer_2",
        ),
    ]


@pytest.fixture
def mlp_activations_mixed() -> MLPActivations:
    alive = torch.ones(2, 10, 32) * 0.5
    dead = torch.zeros(2, 10, 32)
    return MLPActivations(
        activations=torch.cat([alive, dead], dim=2),
        layer_name="test_layer",
    )


@pytest.fixture
def activation_data_balanced() -> ExpertActivationData:
    return ExpertActivationData(
        metadata={"num_experts": 4, "num_layers": 2},
        layers={
            "0": {"0": 25, "1": 25, "2": 25, "3": 25},
            "1": {"0": 25, "1": 25, "2": 25, "3": 25},
        },
    )


@pytest.fixture
def activation_data_imbalanced() -> ExpertActivationData:
    return ExpertActivationData(
        metadata={"num_experts": 4, "num_layers": 2},
        layers={
            "0": {"0": 100, "1": 1, "2": 1, "3": 1},
            "1": {"0": 80, "1": 10, "2": 5, "3": 5},
        },
    )


@pytest.fixture
def expert_weights_collapsed() -> ExpertWeights:
    w = torch.ones(64, 128) * 0.5
    return ExpertWeights(
        expert_weights=[w.clone() for _ in range(3)],
        layer_name="collapsed_layer",
    )


@pytest.fixture
def expert_weights_specialized() -> ExpertWeights:
    return ExpertWeights(
        expert_weights=[torch.randn(64, 128) for _ in range(4)],
        layer_name="specialized_layer",
    )


@pytest.fixture
def headroom_report_minimal() -> HeadroomReport:
    signals = [
        SignalResult(
            signal_name=name,
            status="skipped",
            data=None,
            duration_seconds=0.0,
        )
        for name in [
            "loss_saturation",
            "growth_factor",
            "effective_rank",
            "grad_weight_ratio",
            "dead_neurons",
            "load_balance",
            "specialization",
        ]
    ]
    return HeadroomReport(
        model_name="test_model",
        model_path="/tmp/test",
        timestamp="2026-01-01T00:00:00+00:00",
        signals=signals,
        overall_headroom_score=0.5,
        recommendations=["rec1", "rec2", "rec3"],
        metadata={},
    )


@pytest.fixture
def headroom_report_computed() -> HeadroomReport:
    from model_headroom.models.types import (
        GradWeightRatio,
        SaturationReport,
        SpecializationReport,
    )

    signals = [
        SignalResult(
            signal_name="loss_saturation",
            status="computed",
            data=SaturationReport(
                is_saturated=False,
                saturation_point_tokens=None,
                loss_delta_threshold=0.001,
                plateau_slope=0.0,
                data_points_used=5,
                signal_status="computed",
            ),
            duration_seconds=0.01,
        ),
        SignalResult(
            signal_name="growth_factor",
            status="computed",
            data=GrowthFactorReport(
                alpha=0.5,
                crossover_params=None,
                crossover_loss=None,
                fit_quality_r2=0.8,
                data_points_used=5,
                signal_status="computed",
            ),
            duration_seconds=0.01,
        ),
        SignalResult(
            signal_name="effective_rank",
            status="computed",
            data=EffectiveRankResult(
                rank=6.0,
                entropy=1.5,
                percentage_of_full_rank=0.75,
                singular_values_count=8,
                signal_status="computed",
            ),
            duration_seconds=0.01,
        ),
        SignalResult(
            signal_name="grad_weight_ratio",
            status="computed",
            data=[
                GradWeightRatio(
                    ratio=0.5,
                    is_high_pressure=False,
                    z_score=0.0,
                    layer_name="l0",
                    signal_status="computed",
                ),
            ],
            duration_seconds=0.01,
        ),
        SignalResult(
            signal_name="dead_neurons",
            status="computed",
            data=DeadNeuronReport(
                total_neurons=100,
                dead_count=5,
                dead_fraction=0.05,
                dead_indices=None,
                signal_status="computed",
            ),
            duration_seconds=0.01,
        ),
        SignalResult(
            signal_name="load_balance",
            status="computed",
            data=LoadBalanceReport(
                entropy_per_layer=[1.3],
                min_entropy=1.3,
                max_entropy=1.3,
                ideal_entropy=1.386,
                balance_ratio=0.94,
                signal_status="computed",
            ),
            duration_seconds=0.01,
        ),
        SignalResult(
            signal_name="specialization",
            status="computed",
            data=SpecializationReport(
                mean_inter_expert_similarity=0.3,
                max_inter_expert_similarity=0.5,
                collapse_threshold=0.85,
                is_collapsed=False,
                signal_status="computed",
            ),
            duration_seconds=0.01,
        ),
    ]
    return HeadroomReport(
        model_name="test_model",
        model_path="/tmp/test",
        timestamp="2026-01-01T00:00:00+00:00",
        signals=signals,
        overall_headroom_score=0.75,
        recommendations=["rec1", "rec2", "rec3"],
        metadata={},
    )


@pytest.fixture
def tmp_json_file(tmp_path: Any) -> Any:
    def _write(data: Any, name: str = "test.json") -> str:
        p = tmp_path / name
        p.write_text(json.dumps(data))
        return str(p)

    return _write
