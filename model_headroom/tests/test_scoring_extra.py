from __future__ import annotations

import pytest

from model_headroom.analysis.scoring import (
    _score_signal,
    generate_recommendations,
)
from model_headroom.models.types import (
    DeadNeuronReport,
    EffectiveRankResult,
    GradWeightRatio,
    GrowthFactorReport,
    HeadroomReport,
    LoadBalanceReport,
    SaturationReport,
    SignalResult,
    SpecializationReport,
)


class TestScoreGrowthFactorPoorFit:
    def test_poor_fit_returns_03(self) -> None:
        data = GrowthFactorReport(
            alpha=0.8,
            crossover_params=None,
            crossover_loss=None,
            fit_quality_r2=0.3,
            data_points_used=5,
            signal_status="poor_fit",
        )
        signal = SignalResult(
            signal_name="growth_factor",
            status="computed",
            data=data,
            duration_seconds=0.0,
        )
        assert _score_signal(signal) == pytest.approx(0.3)


class TestScoreUnknownSignal:
    def test_unknown_signal_returns_05(self) -> None:
        data = SaturationReport(
            is_saturated=False,
            saturation_point_tokens=None,
            loss_delta_threshold=0.001,
            plateau_slope=0.0,
            data_points_used=5,
            signal_status="computed",
        )
        signal = SignalResult(
            signal_name="unknown_signal_name",
            status="computed",
            data=data,
            duration_seconds=0.0,
        )
        assert _score_signal(signal) == 0.5


class TestGradWeightRatioNonEmpty:
    def test_half_high_pressure(self) -> None:
        ratios = [
            GradWeightRatio(
                ratio=0.5,
                is_high_pressure=False,
                z_score=0.0,
                layer_name="l0",
                signal_status="computed",
            ),
            GradWeightRatio(
                ratio=5.0,
                is_high_pressure=True,
                z_score=2.5,
                layer_name="l1",
                signal_status="computed",
            ),
        ]
        signal = SignalResult(
            signal_name="grad_weight_ratio",
            status="computed",
            data=ratios,
            duration_seconds=0.0,
        )
        assert _score_signal(signal) == pytest.approx(0.5)


class TestRecommendations:
    def test_loss_saturation_saturated(self) -> None:
        report = HeadroomReport(
            model_name="m",
            model_path="/tmp",
            timestamp="2026-01-01T00:00:00+00:00",
            signals=[
                SignalResult(
                    signal_name="loss_saturation",
                    status="computed",
                    data=SaturationReport(
                        is_saturated=True,
                        saturation_point_tokens=100,
                        loss_delta_threshold=0.001,
                        plateau_slope=0.0005,
                        data_points_used=7,
                        signal_status="computed",
                    ),
                    duration_seconds=0.0,
                ),
            ],
            overall_headroom_score=0.5,
            recommendations=[],
            metadata={},
        )
        recs = generate_recommendations(report)
        assert any("Loss curve has saturated" in r for r in recs)

    def test_growth_factor_crossover(self) -> None:
        report = HeadroomReport(
            model_name="m",
            model_path="/tmp",
            timestamp="2026-01-01T00:00:00+00:00",
            signals=[
                SignalResult(
                    signal_name="growth_factor",
                    status="computed",
                    data=GrowthFactorReport(
                        alpha=0.5,
                        crossover_params=1_000_000,
                        crossover_loss=1.2,
                        fit_quality_r2=0.8,
                        data_points_used=5,
                        signal_status="computed",
                    ),
                    duration_seconds=0.0,
                ),
            ],
            overall_headroom_score=0.5,
            recommendations=[],
            metadata={},
        )
        recs = generate_recommendations(report)
        assert any("crossover" in r for r in recs)

    def test_low_effective_rank(self) -> None:
        report = HeadroomReport(
            model_name="m",
            model_path="/tmp",
            timestamp="2026-01-01T00:00:00+00:00",
            signals=[
                SignalResult(
                    signal_name="effective_rank",
                    status="computed",
                    data=EffectiveRankResult(
                        rank=2.0,
                        entropy=0.5,
                        percentage_of_full_rank=0.25,
                        singular_values_count=8,
                        signal_status="computed",
                    ),
                    duration_seconds=0.0,
                ),
            ],
            overall_headroom_score=0.5,
            recommendations=[],
            metadata={},
        )
        recs = generate_recommendations(report)
        assert any("Low effective rank" in r for r in recs)

    def test_high_grad_pressure(self) -> None:
        report = HeadroomReport(
            model_name="m",
            model_path="/tmp",
            timestamp="2026-01-01T00:00:00+00:00",
            signals=[
                SignalResult(
                    signal_name="grad_weight_ratio",
                    status="computed",
                    data=[
                        GradWeightRatio(
                            ratio=5.0,
                            is_high_pressure=True,
                            z_score=3.0,
                            layer_name="l0",
                            signal_status="computed",
                        ),
                    ],
                    duration_seconds=0.0,
                ),
            ],
            overall_headroom_score=0.5,
            recommendations=[],
            metadata={},
        )
        recs = generate_recommendations(report)
        assert any("high gradient-to-weight pressure" in r for r in recs)

    def test_dead_neurons_high(self) -> None:
        report = HeadroomReport(
            model_name="m",
            model_path="/tmp",
            timestamp="2026-01-01T00:00:00+00:00",
            signals=[
                SignalResult(
                    signal_name="dead_neurons",
                    status="computed",
                    data=DeadNeuronReport(
                        total_neurons=100,
                        dead_count=20,
                        dead_fraction=0.2,
                        dead_indices=None,
                        signal_status="computed",
                    ),
                    duration_seconds=0.0,
                ),
            ],
            overall_headroom_score=0.5,
            recommendations=[],
            metadata={},
        )
        recs = generate_recommendations(report)
        assert any("dead" in r for r in recs)

    def test_load_balance_unbalanced(self) -> None:
        report = HeadroomReport(
            model_name="m",
            model_path="/tmp",
            timestamp="2026-01-01T00:00:00+00:00",
            signals=[
                SignalResult(
                    signal_name="load_balance",
                    status="computed",
                    data=LoadBalanceReport(
                        entropy_per_layer=[0.5],
                        min_entropy=0.5,
                        max_entropy=0.5,
                        ideal_entropy=1.386,
                        balance_ratio=0.36,
                        signal_status="computed",
                    ),
                    duration_seconds=0.0,
                ),
            ],
            overall_headroom_score=0.5,
            recommendations=[],
            metadata={},
        )
        recs = generate_recommendations(report)
        assert any("unbalanced" in r for r in recs)

    def test_specialization_collapsed(self) -> None:
        report = HeadroomReport(
            model_name="m",
            model_path="/tmp",
            timestamp="2026-01-01T00:00:00+00:00",
            signals=[
                SignalResult(
                    signal_name="specialization",
                    status="computed",
                    data=SpecializationReport(
                        mean_inter_expert_similarity=0.9,
                        max_inter_expert_similarity=0.95,
                        collapse_threshold=0.85,
                        is_collapsed=True,
                        signal_status="computed",
                    ),
                    duration_seconds=0.0,
                ),
            ],
            overall_headroom_score=0.5,
            recommendations=[],
            metadata={},
        )
        recs = generate_recommendations(report)
        assert any("collapsed" in r for r in recs)

    def test_no_computed_signals_gets_default(self) -> None:
        report = HeadroomReport(
            model_name="m",
            model_path="/tmp",
            timestamp="2026-01-01T00:00:00+00:00",
            signals=[
                SignalResult(
                    signal_name="loss_saturation",
                    status="skipped",
                    data=None,
                    duration_seconds=0.0,
                ),
            ],
            overall_headroom_score=0.5,
            recommendations=[],
            metadata={},
        )
        recs = generate_recommendations(report)
        assert len(recs) >= 3
