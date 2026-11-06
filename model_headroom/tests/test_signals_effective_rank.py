from __future__ import annotations

import torch

from model_headroom.signals.effective_rank import compute_effective_rank


class TestEffectiveRankFunctional:
    def test_full_rank_matrix(self, weight_matrix_full_rank: torch.Tensor) -> None:
        result = compute_effective_rank(weight_matrix_full_rank)
        assert result.signal_status == "computed"
        assert result.percentage_of_full_rank > 0.5
        assert result.rank >= 1.0

    def test_low_rank_matrix(self, weight_matrix_low_rank: torch.Tensor) -> None:
        result = compute_effective_rank(weight_matrix_low_rank)
        assert result.signal_status == "computed"
        assert result.percentage_of_full_rank < 0.5

    def test_identity_matrix(self) -> None:
        result = compute_effective_rank(torch.eye(64))
        assert result.signal_status == "computed"
        assert result.percentage_of_full_rank > 0.9


class TestEffectiveRankEdge:
    def test_zero_tensor(self) -> None:
        result = compute_effective_rank(torch.zeros(64, 128))
        assert result.signal_status == "degenerate"
        assert result.rank == 1.0

    def test_1d_tensor(self) -> None:
        result = compute_effective_rank(torch.ones(64))
        assert result.signal_status == "degenerate"

    def test_rank_deficient_matrix(self) -> None:
        w = torch.ones(64, 128)
        result = compute_effective_rank(w)
        assert result.signal_status == "computed"
        assert result.rank <= 1.5
        assert result.percentage_of_full_rank < 0.05

    def test_large_tensor(self) -> None:
        result = compute_effective_rank(torch.randn(256, 256))
        assert result.signal_status == "computed"
        assert 1.0 <= result.rank <= 256.0


class TestEffectiveRankError:
    def test_nan_tensor(self) -> None:
        w = torch.full((8, 8), float("nan"))
        result = compute_effective_rank(w)
        assert result.signal_status in ("error", "degenerate")
