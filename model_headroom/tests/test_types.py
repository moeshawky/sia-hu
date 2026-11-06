from __future__ import annotations

import dataclasses

import pytest

from model_headroom.models.types import (
    DeadNeuronReport,
    EffectiveRankResult,
    GradWeightRatio,
    GrowthFactorReport,
    HeadroomReport,
    LoadBalanceReport,
    LossDataPoint,
    SaturationReport,
    SignalResult,
    SpecializationReport,
)


class TestFrozenDataclasses:
    def test_loss_data_point_frozen(self) -> None:
        p = LossDataPoint(tokens_trained=100, validation_loss=1.0)
        with pytest.raises(AttributeError):
            p.tokens_trained = 200  # type: ignore[misc]

    def test_saturation_report_frozen(self) -> None:
        r = SaturationReport(
            is_saturated=False,
            saturation_point_tokens=None,
            loss_delta_threshold=0.001,
            plateau_slope=0.0,
            data_points_used=5,
            signal_status="computed",
        )
        with pytest.raises(AttributeError):
            r.is_saturated = True  # type: ignore[misc]

    def test_growth_factor_report_frozen(self) -> None:
        r = GrowthFactorReport(
            alpha=0.5,
            crossover_params=None,
            crossover_loss=None,
            fit_quality_r2=0.8,
            data_points_used=5,
            signal_status="computed",
        )
        with pytest.raises(AttributeError):
            r.alpha = 1.0  # type: ignore[misc]

    def test_effective_rank_result_frozen(self) -> None:
        r = EffectiveRankResult(
            rank=5.0,
            entropy=1.2,
            percentage_of_full_rank=0.625,
            singular_values_count=8,
            signal_status="computed",
        )
        with pytest.raises(AttributeError):
            r.rank = 10.0  # type: ignore[misc]

    def test_grad_weight_ratio_frozen(self) -> None:
        r = GradWeightRatio(
            ratio=0.5,
            is_high_pressure=False,
            z_score=0.0,
            layer_name="l0",
            signal_status="computed",
        )
        with pytest.raises(AttributeError):
            r.ratio = 1.0  # type: ignore[misc]

    def test_dead_neuron_report_frozen(self) -> None:
        r = DeadNeuronReport(
            total_neurons=100,
            dead_count=5,
            dead_fraction=0.05,
            dead_indices=None,
            signal_status="computed",
        )
        with pytest.raises(AttributeError):
            r.dead_count = 10  # type: ignore[misc]

    def test_load_balance_report_frozen(self) -> None:
        r = LoadBalanceReport(
            entropy_per_layer=[1.0],
            min_entropy=1.0,
            max_entropy=1.0,
            ideal_entropy=1.386,
            balance_ratio=0.72,
            signal_status="computed",
        )
        with pytest.raises(AttributeError):
            r.balance_ratio = 1.0  # type: ignore[misc]

    def test_specialization_report_frozen(self) -> None:
        r = SpecializationReport(
            mean_inter_expert_similarity=0.3,
            max_inter_expert_similarity=0.5,
            collapse_threshold=0.85,
            is_collapsed=False,
            signal_status="computed",
        )
        with pytest.raises(AttributeError):
            r.is_collapsed = True  # type: ignore[misc]

    def test_signal_result_frozen(self) -> None:
        r = SignalResult(
            signal_name="test",
            status="computed",
            data=None,
            duration_seconds=0.0,
        )
        with pytest.raises(AttributeError):
            r.status = "error"  # type: ignore[misc]

    def test_headroom_report_frozen(self) -> None:
        r = HeadroomReport(
            model_name="m",
            model_path="/tmp",
            timestamp="2026-01-01T00:00:00+00:00",
            signals=[],
            overall_headroom_score=0.5,
            recommendations=[],
            metadata={},
        )
        with pytest.raises(AttributeError):
            r.overall_headroom_score = 1.0  # type: ignore[misc]

    def test_all_output_dataclasses_have_signal_status(self) -> None:
        for cls in [
            SaturationReport,
            GrowthFactorReport,
            EffectiveRankResult,
            DeadNeuronReport,
            LoadBalanceReport,
            SpecializationReport,
        ]:
            fields = {f.name for f in dataclasses.fields(cls)}
            assert "signal_status" in fields, f"{cls.__name__} missing signal_status"
