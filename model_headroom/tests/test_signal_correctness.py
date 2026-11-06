"""Signal correctness tests with analytically derivable expected values.

These tests construct inputs where the expected output is mathematically
deterministic, providing ground-truth verification of signal computation
that current tests (using only range/status assertions) cannot provide.
"""

from __future__ import annotations

import math

import pytest
import torch

from model_headroom.models.types import (
    GrowthFactorReport,
    IsoFLOPPoint,
    LossDataPoint,
    SaturationReport,
)
from model_headroom.signals.effective_rank import compute_effective_rank
from model_headroom.signals.growth_factor import analyze_growth_factor
from model_headroom.signals.loss_saturation import analyze_loss_saturation

# --- Loss saturation: analytically derivable ---


class TestLossSaturationCorrectness:
    """Loss saturation with inputs where expected output is exact."""

    def test_decreasing_loss_not_saturated(self) -> None:
        """Loss decreasing by 1.0 per 100M tokens: slope = 1.0 >> 0.001.

        Expected: is_saturated=False (slopes all exceed threshold).
        """
        points = [
            LossDataPoint(tokens_trained=i * 100_000_000, validation_loss=10.0 - i)
            for i in range(5)
        ]
        report = analyze_loss_saturation(points)
        assert isinstance(report, SaturationReport)
        assert report.signal_status == "computed"
        assert report.is_saturated is False
        assert report.saturation_point_tokens is None

    def test_plateau_detected_at_known_point(self) -> None:
        """Loss flat at 5.0 for last 3 intervals: slope = 0.0 < 0.001.

        First two intervals have slope 1.0, last three have slope 0.0.
        Expected: is_saturated=True, saturation_point at start of plateau.
        """
        points = [
            LossDataPoint(tokens_trained=0, validation_loss=10.0),
            LossDataPoint(tokens_trained=100_000_000, validation_loss=9.0),
            LossDataPoint(tokens_trained=200_000_000, validation_loss=8.0),
            LossDataPoint(tokens_trained=300_000_000, validation_loss=5.0),
            LossDataPoint(tokens_trained=400_000_000, validation_loss=5.0),
            LossDataPoint(tokens_trained=500_000_000, validation_loss=5.0),
            LossDataPoint(tokens_trained=600_000_000, validation_loss=5.0),
        ]
        report = analyze_loss_saturation(points)
        assert isinstance(report, SaturationReport)
        assert report.is_saturated is True
        # Plateau starts at interval index 3→4, saturation point is at
        # the start of the last 3 intervals: index len-4 = 3, tokens = 300M
        assert report.saturation_point_tokens == 300_000_000
        assert report.plateau_slope == pytest.approx(0.0, abs=1e-15)

    def test_insufficient_data_returns_insufficient(self) -> None:
        """Fewer than min_intervals+1=4 points → insufficient_data."""
        points = [
            LossDataPoint(tokens_trained=0, validation_loss=10.0),
            LossDataPoint(tokens_trained=100_000_000, validation_loss=9.0),
            LossDataPoint(tokens_trained=200_000_000, validation_loss=8.0),
        ]
        report = analyze_loss_saturation(points)
        assert isinstance(report, SaturationReport)
        assert report.signal_status == "insufficient_data"
        assert report.is_saturated is False

    def test_weak_plateau_above_threshold_not_saturated(self) -> None:
        """Slope exactly at threshold (0.001) should NOT trigger saturation.

        slope = 0.001 / (1e8/1e8) = 0.001, check is s < 0.001 → False.
        """
        points = [
            LossDataPoint(tokens_trained=0, validation_loss=10.0),
            LossDataPoint(tokens_trained=100_000_000, validation_loss=10.0 - 0.001),
            LossDataPoint(tokens_trained=200_000_000, validation_loss=10.0 - 0.002),
            LossDataPoint(tokens_trained=300_000_000, validation_loss=10.0 - 0.003),
            LossDataPoint(tokens_trained=400_000_000, validation_loss=10.0 - 0.004),
        ]
        report = analyze_loss_saturation(points)
        assert isinstance(report, SaturationReport)
        assert report.signal_status == "computed"
        assert report.is_saturated is False

    def test_custom_threshold_accepts_plateau(self) -> None:
        """With threshold=0.01, slope=0.005 should trigger saturation."""
        points = [
            LossDataPoint(tokens_trained=0, validation_loss=10.0),
            LossDataPoint(tokens_trained=100_000_000, validation_loss=9.995),
            LossDataPoint(tokens_trained=200_000_000, validation_loss=9.990),
            LossDataPoint(tokens_trained=300_000_000, validation_loss=9.985),
            LossDataPoint(tokens_trained=400_000_000, validation_loss=9.980),
        ]
        report = analyze_loss_saturation(points, slope_threshold=0.01)
        assert isinstance(report, SaturationReport)
        assert report.is_saturated is True


# --- Effective rank: analytically derivable ---


class TestEffectiveRankCorrectness:
    """Effective rank with inputs where the result is mathematically exact."""

    def test_identity_matrix_full_rank(self) -> None:
        """3x3 identity: SVs = [1,1,1], probs = [1/3,1/3,1/3].

        entropy = log(3), rank = exp(log(3)) = 3.0, percentage = 3/3 = 1.0.
        """
        w = torch.eye(3)
        result = compute_effective_rank(w, sampling_mode="full_matrix")
        assert isinstance(result, dict) or hasattr(result, "percentage_of_full_rank")
        assert result.percentage_of_full_rank == pytest.approx(1.0, abs=1e-6)
        assert result.rank == pytest.approx(3.0, abs=1e-6)
        assert result.entropy == pytest.approx(math.log(3), abs=1e-6)
        assert result.signal_status == "computed"
        assert result.confidence == pytest.approx(0.95, abs=1e-6)

    def test_rank_one_matrix(self) -> None:
        """2x2 matrix [[1,0],[0,0]]: SVs = [1,0].

        entropy ≈ 0 (one probability dominates), rank = exp(0) = 1,
        percentage = 1/2 = 0.5.
        """
        w = torch.tensor([[1.0, 0.0], [0.0, 0.0]])
        result = compute_effective_rank(w, sampling_mode="full_matrix")
        assert result.percentage_of_full_rank == pytest.approx(0.5, abs=1e-10)
        assert result.rank == pytest.approx(1.0, abs=1e-10)
        assert result.singular_values_count == 2

    def test_zero_tensor_is_degenerate(self) -> None:
        """All-zeros: abs().sum() == 0 → degenerate before SVD."""
        w = torch.zeros(4, 4)
        result = compute_effective_rank(w, sampling_mode="full_matrix")
        assert result.signal_status == "degenerate"
        assert result.percentage_of_full_rank == pytest.approx(1.0 / 4, abs=1e-10)

    def test_1d_tensor_degenerate(self) -> None:
        """1D tensor → degenerate (ndim < 2)."""
        w = torch.tensor([1.0, 2.0, 3.0])
        result = compute_effective_rank(w, sampling_mode="full_matrix")
        assert result.signal_status == "degenerate"

    def test_parent_shape_coverage_computed(self) -> None:
        """With parent_shape, coverage should be tile_elements/parent_elements."""
        w = torch.eye(3)  # 3x3 = 9 elements
        parent_shape = (6, 6)  # 36 elements
        result = compute_effective_rank(
            w, sampling_mode="single_tile", parent_shape=parent_shape, sample_count=1
        )
        assert result.coverage is not None
        assert result.coverage == pytest.approx(9.0 / 36.0, abs=1e-10)


# --- Growth factor: analytically derivable ---


class TestGrowthFactorCorrectness:
    """Growth factor with inputs where the fit is analytically known."""

    def test_linear_relationship_growth_two(self) -> None:
        """measured_loss = model_params^2 (linear in log space).

        log(loss) = 2*log(params), so y = 2*x. Quadratic fit: a=0, b=2, c=0.
        alpha = 2*0*mean + 2 = 2.0.
        """
        points = [
            IsoFLOPPoint(
                model_params=10,
                measured_loss=100,
                compute_flops=0.0,
            ),
            IsoFLOPPoint(
                model_params=100,
                measured_loss=10_000,
                compute_flops=0.0,
            ),
            IsoFLOPPoint(
                model_params=1_000,
                measured_loss=1_000_000,
                compute_flops=0.0,
            ),
        ]
        report = analyze_growth_factor(points)
        assert isinstance(report, GrowthFactorReport)
        assert report.signal_status == "computed"
        assert report.alpha == pytest.approx(2.0, abs=1e-6)

    def test_insufficient_data(self) -> None:
        """Fewer than 3 points → insufficient_data."""
        points = [
            IsoFLOPPoint(model_params=10, measured_loss=1.0, compute_flops=0.0),
            IsoFLOPPoint(model_params=100, measured_loss=0.5, compute_flops=0.0),
        ]
        report = analyze_growth_factor(points)
        assert isinstance(report, GrowthFactorReport)
        assert report.signal_status == "insufficient_data"
        assert report.data_points_used == 2

    def test_linear_relationship_growth_one(self) -> None:
        """measured_loss = model_params (linear in log space).

        log(loss) = log(params), so y = x. Quadratic fit: a=0, b=1, c=0.
        alpha = 2*0*mean + 1 = 1.0.
        """
        points = [
            IsoFLOPPoint(
                model_params=10,
                measured_loss=10,
                compute_flops=0.0,
            ),
            IsoFLOPPoint(
                model_params=100,
                measured_loss=100,
                compute_flops=0.0,
            ),
            IsoFLOPPoint(
                model_params=1_000,
                measured_loss=1_000,
                compute_flops=0.0,
            ),
        ]
        report = analyze_growth_factor(points)
        assert isinstance(report, GrowthFactorReport)
        assert report.signal_status == "computed"
        assert report.alpha == pytest.approx(1.0, abs=1e-6)
