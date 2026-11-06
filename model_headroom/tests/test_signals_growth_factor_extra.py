from __future__ import annotations

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
