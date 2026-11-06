from __future__ import annotations

import torch

from model_headroom.signals.effective_rank import compute_effective_rank


class TestEffectiveRankZeroSingularValues:
    def test_total_zero_sv(self) -> None:
        w = torch.zeros(4, 8)
        result = compute_effective_rank(w)
        assert result.signal_status == "degenerate"
        assert result.singular_values_count == 0

    def test_near_zero_sv(self) -> None:
        w = torch.full((4, 8), 1e-15)
        result = compute_effective_rank(w)
        assert result.signal_status in ("degenerate", "computed")
        assert result.percentage_of_full_rank <= 1.0


class TestEffectiveRankSVD:
    def test_all_negative_values(self) -> None:
        w = -1.0 * torch.eye(8)
        result = compute_effective_rank(w)
        assert result.signal_status == "computed"
        assert result.rank >= 1.0
