from __future__ import annotations

import math

from model_headroom.models.types import IsoFLOPPoint
from model_headroom.signals.growth_factor import _fit_quadratic, analyze_growth_factor


class TestFitQuadratic:
    def test_fewer_than_3_points(self) -> None:
        a, b, c, r2 = _fit_quadratic([1.0, 2.0], [3.0, 4.0])
        assert a == 0.0
        assert b == 0.0
        assert c == 0.0
        assert r2 == 0.0

    def test_singular_matrix(self) -> None:
        a, b, c, r2 = _fit_quadratic([1.0, 1.0, 1.0], [2.0, 2.0, 2.0])
        assert a == 0.0
        assert b == 0.0
        assert c == 0.0
        assert r2 == 0.0

    def test_perfect_fit(self) -> None:
        _a, _b, _c, r2 = _fit_quadratic([1.0, 2.0, 3.0], [1.0, 4.0, 9.0])
        assert r2 > 0.99

    def test_zero_ss_tot(self) -> None:
        _a, _b, _c, r2 = _fit_quadratic([1.0, 2.0, 3.0], [5.0, 5.0, 5.0])
        assert r2 == 0.0


class TestGrowthFactorException:
    def test_exception_handler_returns_error(self) -> None:
        points = [
            IsoFLOPPoint(
                compute_flops=1e18,
                model_params=1,
                measured_loss=1.0,
            ),
            IsoFLOPPoint(
                compute_flops=1e18,
                model_params=1,
                measured_loss=1.0,
            ),
            IsoFLOPPoint(
                compute_flops=1e18,
                model_params=1,
                measured_loss=1.0,
            ),
        ]
        result = analyze_growth_factor(points)
        assert result.signal_status in ("poor_fit", "error")


class TestGrowthFactorCrossover:
    def test_growth_factor_crossover_set(self) -> None:
        # Points exactly on y = 0.1*x^2 - 2*x + 40 (x = log params): the fit
        # recovers a=0.1 (> 1e-12), r^2=1.0 (>= 0.5), x_crossover = 15.0 (> 0),
        # so crossover_params/crossover_loss are SET.
        xs = [10.0, 12.0, 14.0, 16.0, 18.0]
        points = [
            IsoFLOPPoint(
                compute_flops=1e18,
                model_params=math.exp(x),
                measured_loss=math.exp(0.1 * x * x - 2.0 * x + 40.0),
            )
            for x in xs
        ]
        result = analyze_growth_factor(points)
        assert result.signal_status == "computed"
        assert result.crossover_params is not None
        assert result.crossover_loss is not None
        assert isinstance(result.crossover_params, int)
        assert result.crossover_params > 0

    def test_growth_factor_crossover_none_when_fit_flat(self) -> None:
        # Collinear log-log points (y = 2x + 5): quadratic coefficient a ~= 0,
        # r^2 = 1.0 -> status computed but the crossover branch stays None.
        xs = [10.0, 12.0, 14.0, 16.0]
        points = [
            IsoFLOPPoint(
                compute_flops=1e18,
                model_params=math.exp(x),
                measured_loss=math.exp(2.0 * x + 5.0),
            )
            for x in xs
        ]
        result = analyze_growth_factor(points)
        assert result.signal_status == "computed"
        assert result.crossover_params is None
        assert result.crossover_loss is None
