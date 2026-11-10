"""Growth factor upper bound isoFLOP sweep (R-02).

Fits quadratic log(params) vs log(loss) with R^2 >= 0.85.
Reports crossover point where g_upper < 1.0.
Returns None for crossover if R^2 < 0.5.

Dependencies: math, models.types (IsoFLOPPoint, GrowthFactorReport).
Provides: analyze_growth_factor().
"""

from __future__ import annotations

import math

from model_headroom.models.types import GrowthFactorReport, IsoFLOPPoint


def _fit_quadratic(x: list[float], y: list[float]) -> tuple[float, float, float, float]:
    """Fit y = a*x^2 + b*x + c using least squares.

    Returns (a, b, c, r_squared).
    """
    n = len(x)
    if n < 3:
        return (0.0, 0.0, 0.0, 0.0)

    # Build normal equations for quadratic fit
    # Sum terms
    sum_x = sum(x)
    sum_x2 = sum(xi**2 for xi in x)
    sum_x3 = sum(xi**3 for xi in x)
    sum_x4 = sum(xi**4 for xi in x)
    sum_y = sum(y)
    sum_xy = sum(xi * yi for xi, yi in zip(x, y, strict=False))
    sum_x2y = sum(xi**2 * yi for xi, yi in zip(x, y, strict=False))

    # Solve 3x3 system using Cramer's rule
    # Matrix: [[n, sum_x, sum_x2], [sum_x, sum_x2, sum_x3], [sum_x2, sum_x3, sum_x4]]
    # RHS: [sum_y, sum_xy, sum_x2y]
    det = (
        n * (sum_x2 * sum_x4 - sum_x3 * sum_x3)
        - sum_x * (sum_x * sum_x4 - sum_x3 * sum_x2)
        + sum_x2 * (sum_x * sum_x3 - sum_x2 * sum_x2)
    )

    if abs(det) < 1e-12:
        return (0.0, 0.0, 0.0, 0.0)

    det_a = (
        sum_y * (sum_x2 * sum_x4 - sum_x3 * sum_x3)
        - sum_x * (sum_xy * sum_x4 - sum_x3 * sum_x2y)
        + sum_x2 * (sum_xy * sum_x3 - sum_x2 * sum_x2y)
    )
    det_b = (
        n * (sum_xy * sum_x4 - sum_x3 * sum_x2y)
        - sum_y * (sum_x * sum_x4 - sum_x3 * sum_x2)
        + sum_x2 * (sum_x * sum_x2y - sum_xy * sum_x2)
    )
    det_c = (
        n * (sum_x2 * sum_x2y - sum_xy * sum_x3)
        - sum_x * (sum_x * sum_x2y - sum_xy * sum_x2)
        + sum_y * (sum_x * sum_x3 - sum_x2 * sum_x2)
    )

    # Variable vector is [c, b, a] (row 0 solves for c, row 2 for a).
    # det_a replaces column 0 (c's column) → det_a/det = c
    # det_c replaces column 2 (a's column) → det_c/det = a
    c = det_a / det
    b = det_b / det
    a = det_c / det

    # Compute R-squared
    y_mean = sum_y / n
    ss_tot = sum((yi - y_mean) ** 2 for yi in y)
    ss_res = sum(
        (yi - (a * xi**2 + b * xi + c)) ** 2 for xi, yi in zip(x, y, strict=False)
    )
    r_squared = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    r_squared = max(0.0, min(1.0, r_squared))

    return (a, b, c, r_squared)


def analyze_growth_factor(
    isoflop_points: list[IsoFLOPPoint],
) -> GrowthFactorReport:
    """Estimate the growth factor upper bound from isoFLOP data.

    Fits a quadratic to log(model_params) vs log(measured_loss).
    The alpha exponent indicates diminishing returns: alpha < 1.0 means
    larger models achieve proportionally lower loss.

    Input contract:
        isoflop_points: List of IsoFLOPPoint with at least 3 points for
            a valid quadratic fit.

    Output contract (success):
        Returns GrowthFactorReport with signal_status="computed".
        alpha > 0.0, fit_quality_r2 in [0.0, 1.0].

    Output contract (insufficient data):
        signal_status="insufficient_data" if < 3 points.

    Output contract (poor fit):
        signal_status="poor_fit" if R^2 < 0.5.

    Output contract (error):
        signal_status="error", error_message describes the failure.

    Invariants:
        - fit_quality_r2 in [0.0, 1.0]
        - If crossover_params is not None, crossover_loss is not None

    Dependencies: None.
    """
    if len(isoflop_points) < 3:
        return GrowthFactorReport(
            alpha=0.0,
            crossover_params=None,
            crossover_loss=None,
            fit_quality_r2=0.0,
            data_points_used=len(isoflop_points),
            signal_status="insufficient_data",
        )

    try:
        if any(p.model_params <= 0 or p.measured_loss <= 0 for p in isoflop_points):
            return GrowthFactorReport(
                alpha=0.0,
                crossover_params=None,
                crossover_loss=None,
                fit_quality_r2=0.0,
                data_points_used=len(isoflop_points),
                signal_status="error",
                error_message=(
                    "Invalid input: model_params and measured_loss must be > 0"
                ),
            )
        # Fit quadratic to log(params) vs log(loss)
        log_params = [math.log(p.model_params) for p in isoflop_points]
        log_loss = [math.log(p.measured_loss) for p in isoflop_points]

        a, b, c, r_squared = _fit_quadratic(log_params, log_loss)

        # Alpha is the exponent: loss ~ params^alpha
        # In log-log space with quadratic: log(loss) = a*log(params)^2 + b*log(params) + c
        # The effective alpha at a point is the derivative: 2*a*log(params) + b
        # Use the geometric mean of params for a representative alpha
        mean_log_params = sum(log_params) / len(log_params)
        alpha = 2 * a * mean_log_params + b

        # Find crossover where g_upper = 1.0
        # g_upper is the derivative of log(loss)/log(params)
        # We look for where alpha drops below 1.0
        crossover_params: int | None = None
        crossover_loss: float | None = None

        if r_squared >= 0.5 and abs(a) > 1e-12:
            x_crossover = (1.0 - b) / (2 * a)
            if x_crossover > 0:
                crossover_params = int(math.exp(x_crossover))
                crossover_loss = math.exp(a * x_crossover**2 + b * x_crossover + c)

        return GrowthFactorReport(
            alpha=alpha,
            crossover_params=crossover_params,
            crossover_loss=crossover_loss,
            fit_quality_r2=r_squared,
            data_points_used=len(isoflop_points),
            signal_status="computed" if r_squared >= 0.5 else "poor_fit",
        )
    except Exception as e:
        return GrowthFactorReport(
            alpha=0.0,
            crossover_params=None,
            crossover_loss=None,
            fit_quality_r2=0.0,
            data_points_used=len(isoflop_points),
            signal_status="error",
            error_message=f"Growth factor computation failed: {e}",
        )
