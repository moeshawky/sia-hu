"""Adversarial tests for refusal-direction extraction math."""

from __future__ import annotations

import math

import pytest
import torch

from model_headroom.models.types import RefusalDirectionSpec, ResidualBatch
from model_headroom.refusal.extract import (
    cohens_d_along,
    compute_refusal_direction,
    difference_of_means,
    massive_activation_mask,
    normalize,
    orthogonalize_against,
    project_out,
)

SPEC_LOOSE = RefusalDirectionSpec(min_pairs=1)


def _batch(name: str, tensor: torch.Tensor) -> ResidualBatch:
    return ResidualBatch(activations=tensor, layer_name=name)


def _contrastive(seed: int, separation: float, n: int = 24, dim: int = 16) -> tuple:
    gen = torch.Generator().manual_seed(seed)
    base = torch.randn(n, dim, generator=gen)
    harmful = base + separation
    harmless = base.clone()
    return (harmful, harmless)


class TestNormalize:
    def test_returns_unit_norm(self) -> None:
        out = normalize(torch.tensor([3.0, 4.0]))
        assert float(out.norm()) == pytest.approx(1.0, abs=1e-6)

    def test_zero_vector_raises(self) -> None:
        with pytest.raises(ValueError, match="zero-magnitude"):
            normalize(torch.zeros(4))


class TestRowNormsAndMasking:
    def test_mask_selects_only_outliers(self) -> None:
        batch = torch.ones(10, 3)
        batch[0] *= 100.0
        mask = massive_activation_mask(batch, 1.0)
        assert bool(mask[0])
        assert int(mask.sum()) == 1

    def test_single_row_never_masked(self) -> None:
        batch = torch.tensor([[1.0, 2.0, 3.0]])
        assert int(massive_activation_mask(batch, 0.5).sum()) == 0

    def test_uniform_batch_masks_nothing(self) -> None:
        assert int(massive_activation_mask(torch.ones(8, 4), 2.0).sum()) == 0


class TestDifferenceOfMeans:
    def test_matches_manual_mean(self) -> None:
        h = torch.tensor([[3.0], [5.0]])
        s = torch.tensor([[1.0], [1.0]])
        assert float(difference_of_means(h, s)) == pytest.approx(3.0)


class TestCohensD:
    def test_nan_when_single_row(self) -> None:
        d = cohens_d_along(
            torch.zeros(1, 3), torch.zeros(5, 3), torch.tensor([1.0, 0, 0])
        )
        assert math.isnan(d)

    def test_nan_when_zero_variance(self) -> None:
        d = cohens_d_along(
            torch.ones(6, 3), torch.ones(6, 3), torch.tensor([1.0, 0.0, 0.0])
        )
        assert math.isnan(d)

    def test_rises_with_separation(self) -> None:
        axis = torch.tensor([1.0, 0.0, 0.0])
        previous = -math.inf
        for separation in (0.5, 1.0, 2.0, 4.0):
            gen = torch.Generator().manual_seed(7)
            base = torch.randn(40, 3, generator=gen)
            d = cohens_d_along(base + separation, base, axis)
            assert d > previous
            previous = d


class TestOrthogonalizeAndProject:
    def test_removes_reference_component(self) -> None:
        reference = torch.tensor([1.0, 0.0, 0.0])
        out = orthogonalize_against(torch.tensor([2.0, 1.0, 0.0]), reference)
        assert float(out @ reference) == pytest.approx(0.0, abs=1e-6)

    def test_zero_reference_is_a_noop(self) -> None:
        direction = torch.tensor([1.0, 2.0])
        assert torch.equal(orthogonalize_against(direction, torch.zeros(2)), direction)

    def test_project_out_removes_component(self) -> None:
        direction = torch.tensor([1.0, 0.0])
        residual = torch.tensor([[5.0, 3.0], [2.0, -1.0]])
        out = project_out(residual, direction, 1.0)
        assert torch.allclose(out[:, 0], torch.zeros(2), atol=1e-6)
        assert torch.allclose(out[:, 1], residual[:, 1])

    def test_project_out_zero_strength_is_identity(self) -> None:
        direction = torch.tensor([1.0, 0.0])
        residual = torch.tensor([[5.0, 3.0]])
        assert torch.equal(project_out(residual, direction, 0.0), residual)


class TestComputeRefusalDirection:
    def test_returns_unit_direction(self) -> None:
        harmful, harmless = _contrastive(seed=1, separation=1.5)
        report = compute_refusal_direction(
            [_batch("l0", harmful)], [_batch("l0", harmless)], SPEC_LOOSE
        )
        assert report.direction_status == "computed"
        assert report.direction is not None
        assert float(report.direction.norm()) == pytest.approx(1.0, abs=1e-5)

    def test_direction_points_from_harmless_to_harmful(self) -> None:
        harmful, harmless = _contrastive(seed=2, separation=2.0)
        report = compute_refusal_direction(
            [_batch("l0", harmful)], [_batch("l0", harmless)], SPEC_LOOSE
        )
        assert report.direction is not None
        assert float((harmful.mean(0) - harmless.mean(0)) @ report.direction) > 0

    def test_cohen_d_rises_with_separation(self) -> None:
        previous = -math.inf
        for separation in (0.25, 0.75, 1.5):
            harmful, harmless = _contrastive(seed=3, separation=separation)
            report = compute_refusal_direction(
                [_batch("l0", harmful)], [_batch("l0", harmless)], SPEC_LOOSE
            )
            assert report.cohen_d is not None
            assert report.cohen_d > previous
            previous = report.cohen_d

    def test_identical_sets_are_degenerate(self) -> None:
        gen = torch.Generator().manual_seed(4)
        same = torch.randn(20, 8, generator=gen)
        report = compute_refusal_direction(
            [_batch("l0", same)], [_batch("l0", same.clone())], SPEC_LOOSE
        )
        assert report.direction is None
        assert report.direction_status == "insufficient_data"

    def test_no_shared_layer_reports_insufficient(self) -> None:
        harmful, harmless = _contrastive(seed=5, separation=1.0)
        report = compute_refusal_direction(
            [_batch("a", harmful)], [_batch("b", harmless)], SPEC_LOOSE
        )
        assert report.direction_status == "insufficient_data"
        assert "no layer name" in (report.error_message or "")

    def test_mismatched_widths_are_rejected(self) -> None:
        harmful, _ = _contrastive(seed=6, separation=1.0)
        _, harmless = _contrastive(seed=6, separation=1.0, dim=8)
        report = compute_refusal_direction(
            [_batch("l0", harmful)], [_batch("l0", harmless)], SPEC_LOOSE
        )
        assert report.direction is None

    def test_one_dimensional_batch_is_rejected(self) -> None:
        harmful, harmless = _contrastive(seed=7, separation=1.0)
        report = compute_refusal_direction(
            [_batch("l0", harmful.mean(1))],
            [_batch("l0", harmless.mean(1))],
            SPEC_LOOSE,
        )
        assert report.direction is None

    def test_non_finite_input_never_yields_a_direction(self) -> None:
        harmful, harmless = _contrastive(seed=8, separation=1.0)
        poisoned = harmful.clone()
        poisoned[0, 0] = float("nan")
        report = compute_refusal_direction(
            [_batch("l0", poisoned)], [_batch("l0", harmless)], SPEC_LOOSE
        )
        assert report.direction is None

    def test_infinite_input_never_yields_a_direction(self) -> None:
        harmful, harmless = _contrastive(seed=9, separation=1.0)
        poisoned = harmless.clone()
        poisoned[3, 2] = float("inf")
        report = compute_refusal_direction(
            [_batch("l0", harmful)], [_batch("l0", poisoned)], SPEC_LOOSE
        )
        assert report.direction is None

    def test_min_pairs_blocks_thin_data(self) -> None:
        harmful, harmless = _contrastive(seed=10, separation=1.0, n=4)
        report = compute_refusal_direction(
            [_batch("l0", harmful)], [_batch("l0", harmless)], RefusalDirectionSpec()
        )
        assert report.direction is None
        assert report.direction_status == "insufficient_data"
        assert "min_pairs" in (report.error_message or "")

    def test_masking_reports_dropped_rows(self) -> None:
        harmful, harmless = _contrastive(seed=11, separation=1.5)
        spec = RefusalDirectionSpec(min_pairs=1, massive_activation_sigma=1.0)
        report = compute_refusal_direction(
            [_batch("l0", harmful)], [_batch("l0", harmless)], spec
        )
        assert report.masked_count >= 0
        assert report.direction is not None
        assert float(report.direction.norm()) == pytest.approx(1.0, abs=1e-5)

    def test_explicit_layer_name_selects_that_layer(self) -> None:
        harmful, harmless = _contrastive(seed=12, separation=1.0)
        report = compute_refusal_direction(
            [_batch("l0", harmful), _batch("l1", harmful * 2)],
            [_batch("l0", harmless), _batch("l1", harmless * 2)],
            RefusalDirectionSpec(
                min_pairs=1, layer_selection="explicit", layer_name="l1"
            ),
        )
        assert report.layer_name == "l1"
        assert report.selection == "layer_name"

    def test_unknown_layer_name_is_an_error(self) -> None:
        harmful, harmless = _contrastive(seed=13, separation=1.0)
        report = compute_refusal_direction(
            [_batch("l0", harmful)],
            [_batch("l0", harmless)],
            RefusalDirectionSpec(min_pairs=1, layer_name="nope"),
        )
        assert report.direction_status == "error"

    def test_layer_index_selects_that_layer(self) -> None:
        harmful, harmless = _contrastive(seed=14, separation=1.0)
        report = compute_refusal_direction(
            [_batch("l0", harmful), _batch("l1", harmful)],
            [_batch("l0", harmless), _batch("l1", harmless)],
            RefusalDirectionSpec(min_pairs=1, layer_index=1),
        )
        assert report.layer_index == 1
        assert report.selection == "layer_index"

    def test_out_of_range_layer_index_is_an_error(self) -> None:
        harmful, harmless = _contrastive(seed=15, separation=1.0)
        report = compute_refusal_direction(
            [_batch("l0", harmful)],
            [_batch("l0", harmless)],
            RefusalDirectionSpec(min_pairs=1, layer_index=9),
        )
        assert report.direction_status == "error"

    def test_depth_fraction_selects_interior_layer(self) -> None:
        harmful, harmless = _contrastive(seed=16, separation=1.0)
        names = [f"l{i}" for i in range(5)]
        report = compute_refusal_direction(
            [_batch(n, harmful) for n in names],
            [_batch(n, harmless) for n in names],
            RefusalDirectionSpec(min_pairs=1, depth_fraction=0.5),
        )
        assert report.layer_name == "l2"
        assert report.selection == "depth_fraction"

    def test_sweep_is_emitted_even_for_explicit_layer(self) -> None:
        harmful, harmless = _contrastive(seed=17, separation=1.0)
        report = compute_refusal_direction(
            [_batch("l0", harmful), _batch("l1", harmful)],
            [_batch("l0", harmless), _batch("l1", harmless)],
            RefusalDirectionSpec(min_pairs=1, layer_name="l0"),
        )
        assert len(report.layer_stats) == 2

    def test_orthogonalization_reduces_harmless_overlap(self) -> None:
        harmful, harmless = _contrastive(seed=18, separation=1.5)
        plain = compute_refusal_direction(
            [_batch("l0", harmful)], [_batch("l0", harmless)], SPEC_LOOSE
        )
        ortho = compute_refusal_direction(
            [_batch("l0", harmful)],
            [_batch("l0", harmless)],
            RefusalDirectionSpec(min_pairs=1, orthogonalize=True),
        )
        assert plain.cohen_d is not None
        assert ortho.cohen_d is not None
        assert ortho.orthogonalized
        stats = [s.orthogonality_with_harmless_mean for s in plain.layer_stats]
        ortho_stats = [s.orthogonality_with_harmless_mean for s in ortho.layer_stats]
        assert ortho_stats[0] <= stats[0] + 1e-6

    def test_empty_inputs_do_not_raise(self) -> None:
        report = compute_refusal_direction([], [], SPEC_LOOSE)
        assert report.direction_status == "insufficient_data"
        assert report.direction is None


class TestWinsorization:
    def test_clamps_only_the_extreme_component(self) -> None:
        from model_headroom.refusal.extract import winsorize_residuals

        batch = torch.tensor([[1.0, 1.0, 1.0, 1000.0]])
        out = winsorize_residuals(batch, 0.75)
        assert float(out[0, 3]) < 1000.0
        assert float(out[0, 0]) == pytest.approx(1.0)

    def test_preserves_row_count(self) -> None:
        from model_headroom.refusal.extract import winsorize_residuals

        batch = torch.randn(12, 16)
        assert winsorize_residuals(batch, 0.95).shape == batch.shape

    def test_quantile_one_is_a_noop(self) -> None:
        from model_headroom.refusal.extract import winsorize_residuals

        batch = torch.randn(5, 8)
        assert torch.equal(winsorize_residuals(batch, 1.0), batch)

    def test_is_row_independent(self) -> None:
        from model_headroom.refusal.extract import winsorize_residuals

        batch = torch.randn(4, 32)
        alone = winsorize_residuals(batch[2:3], 0.9)
        together = winsorize_residuals(batch, 0.9)[2:3]
        assert torch.allclose(alone, together)

    def test_out_of_range_quantile_is_a_noop(self) -> None:
        from model_headroom.refusal.extract import winsorize_residuals

        batch = torch.randn(4, 8)
        assert torch.equal(winsorize_residuals(batch, 1.5), batch)


class TestFloat64Means:
    def test_mean_is_accurate_where_float32_drifts(self) -> None:
        gen = torch.Generator().manual_seed(11)
        rows = torch.randn(4096, 256, generator=gen) * 1e-3
        harm = rows + 1e-3
        harm_h, harm_s = harm.double().mean(0), harm.double().mean(0)
        diff = harm_h - harm_s
        assert diff.abs().max().item() == 0.0
        drift = (harm.float().mean(0) - harm.float().mean(0)).abs().max().item()
        assert drift == 0.0


class TestWinsorizedDirection:
    def test_direction_stays_unit_norm(self) -> None:
        harmful, harmless = _contrastive(seed=21, separation=1.5)
        report = compute_refusal_direction(
            [_batch("l0", harmful)],
            [_batch("l0", harmless)],
            RefusalDirectionSpec(min_pairs=1, winsorize_quantile=0.95),
        )
        assert report.direction is not None
        assert float(report.direction.norm()) == pytest.approx(1.0, abs=1e-5)

    def test_quantile_is_reported(self) -> None:
        harmful, harmless = _contrastive(seed=22, separation=1.5)
        report = compute_refusal_direction(
            [_batch("l0", harmful)],
            [_batch("l0", harmless)],
            RefusalDirectionSpec(min_pairs=1, winsorize_quantile=0.9),
        )
        assert report.winsorize_quantile == 0.9

    def test_default_is_off(self) -> None:
        harmful, harmless = _contrastive(seed=23, separation=1.5)
        report = compute_refusal_direction(
            [_batch("l0", harmful)], [_batch("l0", harmless)], SPEC_LOOSE
        )
        assert report.winsorize_quantile is None

    def test_winsorize_and_mask_compose(self) -> None:
        harmful, harmless = _contrastive(seed=24, separation=1.5)
        report = compute_refusal_direction(
            [_batch("l0", harmful)],
            [_batch("l0", harmless)],
            RefusalDirectionSpec(
                min_pairs=1, winsorize_quantile=0.9, massive_activation_sigma=1.0
            ),
        )
        assert report.direction is not None
        assert float(report.direction.norm()) == pytest.approx(1.0, abs=1e-5)


class TestBfloat16Dtype:
    def test_compute_refusal_direction_accepts_bf16_activations(self) -> None:
        gen = torch.Generator().manual_seed(1)
        harmful = torch.randn(20, 16, generator=gen, dtype=torch.bfloat16) + 1.0
        harmless = torch.randn(20, 16, generator=gen, dtype=torch.bfloat16)
        report = compute_refusal_direction(
            [_batch("l0", harmful)], [_batch("l0", harmless)], SPEC_LOOSE
        )
        assert report.direction_status == "computed"
        assert report.direction is not None
        assert float(report.direction.norm()) == pytest.approx(1.0, abs=1e-5)

    def test_compute_refusal_direction_bf16_with_winsorize(self) -> None:
        gen = torch.Generator().manual_seed(2)
        harmful = torch.randn(20, 16, generator=gen, dtype=torch.bfloat16) + 1.0
        harmless = torch.randn(20, 16, generator=gen, dtype=torch.bfloat16)
        report = compute_refusal_direction(
            [_batch("l0", harmful)],
            [_batch("l0", harmless)],
            RefusalDirectionSpec(min_pairs=1, winsorize_quantile=0.95),
        )
        assert report.direction_status == "computed"
        assert report.direction is not None

    def test_compute_refusal_direction_bf16_with_mask(self) -> None:
        gen = torch.Generator().manual_seed(3)
        harmful = torch.randn(20, 16, generator=gen, dtype=torch.bfloat16) + 1.0
        harmless = torch.randn(20, 16, generator=gen, dtype=torch.bfloat16)
        report = compute_refusal_direction(
            [_batch("l0", harmful)],
            [_batch("l0", harmless)],
            RefusalDirectionSpec(min_pairs=1, massive_activation_sigma=1.0),
        )
        assert report.direction_status == "computed"
        assert report.direction is not None

    def test_massive_activation_mask_bf16_input(self) -> None:
        batch = torch.randn(8, 16, dtype=torch.bfloat16)
        mask = massive_activation_mask(batch, 1.0)
        assert mask.dtype == torch.bool
