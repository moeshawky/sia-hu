from __future__ import annotations

from model_headroom.models.types import IsoFLOPPoint
from model_headroom.signals.growth_factor import analyze_growth_factor


class TestGrowthFactorFunctional:
    def test_valid_quadratic_fit(self, isoflop_points_good: list[IsoFLOPPoint]) -> None:
        result = analyze_growth_factor(isoflop_points_good)
        assert result.signal_status == "computed"
        assert result.fit_quality_r2 >= 0.5
        assert result.data_points_used == 5

    def test_insufficient_data(self) -> None:
        points = [
            IsoFLOPPoint(
                compute_flops=1e18, model_params=10_000_000, measured_loss=2.0
            ),
            IsoFLOPPoint(
                compute_flops=1e19, model_params=100_000_000, measured_loss=1.5
            ),
        ]
        result = analyze_growth_factor(points)
        assert result.signal_status == "insufficient_data"
        assert result.data_points_used == 2

    def test_poor_fit(self) -> None:
        points = [
            IsoFLOPPoint(
                compute_flops=1e18, model_params=10_000_000, measured_loss=1.0
            ),
            IsoFLOPPoint(
                compute_flops=1e19, model_params=100_000_000, measured_loss=1.0
            ),
            IsoFLOPPoint(
                compute_flops=1e20, model_params=1_000_000_000, measured_loss=1.0
            ),
        ]
        result = analyze_growth_factor(points)
        assert result.signal_status == "poor_fit"
        assert result.fit_quality_r2 < 0.5


class TestGrowthFactorEdge:
    def test_empty_points(self) -> None:
        result = analyze_growth_factor([])
        assert result.signal_status == "insufficient_data"

    def test_two_points(self) -> None:
        points = [
            IsoFLOPPoint(
                compute_flops=1e18, model_params=10_000_000, measured_loss=2.0
            ),
            IsoFLOPPoint(
                compute_flops=1e19, model_params=100_000_000, measured_loss=1.5
            ),
        ]
        result = analyze_growth_factor(points)
        assert result.signal_status == "insufficient_data"

    def test_three_points_minimum(self) -> None:
        points = [
            IsoFLOPPoint(
                compute_flops=1e18, model_params=10_000_000, measured_loss=2.0
            ),
            IsoFLOPPoint(
                compute_flops=1e19, model_params=100_000_000, measured_loss=1.5
            ),
            IsoFLOPPoint(
                compute_flops=1e20, model_params=1_000_000_000, measured_loss=1.2
            ),
        ]
        result = analyze_growth_factor(points)
        assert result.signal_status in ("computed", "poor_fit")


class TestGrowthFactorError:
    def test_singular_matrix(self) -> None:
        points = [
            IsoFLOPPoint(
                compute_flops=1e18, model_params=10_000_000, measured_loss=2.0
            ),
            IsoFLOPPoint(
                compute_flops=1e18, model_params=10_000_000, measured_loss=2.0
            ),
            IsoFLOPPoint(
                compute_flops=1e18, model_params=10_000_000, measured_loss=2.0
            ),
        ]
        result = analyze_growth_factor(points)
        assert result.signal_status in ("poor_fit", "error")
        assert result.fit_quality_r2 == 0.0
