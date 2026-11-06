"""Loss curve saturation detection (R-01).

Detects plateau when slope < 0.001 per 100M tokens for 3+ consecutive intervals.
Returns is_saturated=False if fewer than 3 data points. No false positives on
monotonic decreasing loss.

Dependencies: models.types (LossDataPoint, SaturationReport).
Provides: analyze_loss_saturation().
"""

from __future__ import annotations

from model_headroom.models.types import LossDataPoint, SaturationReport


def analyze_loss_saturation(
    loss_curve: list[LossDataPoint],
    slope_threshold: float = 0.001,
    min_intervals: int = 3,
) -> SaturationReport:
    """Detect whether the loss curve has saturated (plateaued).

    Computes the slope of loss change per 100M tokens across consecutive
    intervals. If the last `min_intervals` slopes are all below
    `slope_threshold`, the curve is declared saturated.

    Input contract:
        loss_curve: Sorted list of LossDataPoint, ascending by tokens_trained.
            Must contain >= min_intervals + 1 points for a valid computation.
        slope_threshold: Maximum slope (loss decrease per 100M tokens) to
            consider a plateau. Default 0.001.
        min_intervals: Minimum consecutive intervals below threshold to
            declare saturation. Default 3.

    Output contract (success):
        Returns SaturationReport with signal_status="computed".
        is_saturated=True if plateau detected; saturation_point_tokens set.

    Output contract (insufficient data):
        signal_status="insufficient_data", is_saturated=False,
        saturation_point_tokens=None.

    Output contract (error):
        signal_status="error", error_message describes the failure.

    Invariants:
        - plateau_slope >= 0.0 always
        - data_points_used == len(loss_curve)
        - If is_saturated is True, saturation_point_tokens is not None

    Dependencies: None.
    """
    if len(loss_curve) < min_intervals + 1:
        return SaturationReport(
            is_saturated=False,
            saturation_point_tokens=None,
            loss_delta_threshold=slope_threshold,
            plateau_slope=0.0,
            data_points_used=len(loss_curve),
            signal_status="insufficient_data",
        )

    # Compute slopes per 100M tokens for each consecutive pair
    slopes: list[float] = []
    for i in range(1, len(loss_curve)):
        prev = loss_curve[i - 1]
        curr = loss_curve[i]
        token_delta = curr.tokens_trained - prev.tokens_trained
        if token_delta <= 0:
            continue
        loss_delta = abs(curr.validation_loss - prev.validation_loss)
        # Normalize to per 100M tokens
        slope = loss_delta / (token_delta / 1e8)
        slopes.append(slope)

    if not slopes:
        return SaturationReport(
            is_saturated=False,
            saturation_point_tokens=None,
            loss_delta_threshold=slope_threshold,
            plateau_slope=0.0,
            data_points_used=len(loss_curve),
            signal_status="computed",
        )

    # Check if the last `min_intervals` slopes are all below threshold
    is_saturated = False
    plateau_slope = 0.0
    saturation_point_tokens: int | None = None

    if len(slopes) >= min_intervals:
        last_slopes = slopes[-min_intervals:]
        if all(s < slope_threshold for s in last_slopes):
            is_saturated = True
            plateau_slope = sum(last_slopes) / len(last_slopes)
            # Saturation point is at the start of the plateau region
            saturation_idx = len(loss_curve) - min_intervals - 1
            if saturation_idx >= 0:
                saturation_point_tokens = loss_curve[saturation_idx].tokens_trained

    return SaturationReport(
        is_saturated=is_saturated,
        saturation_point_tokens=saturation_point_tokens,
        loss_delta_threshold=slope_threshold,
        plateau_slope=plateau_slope,
        data_points_used=len(loss_curve),
        signal_status="computed",
    )
