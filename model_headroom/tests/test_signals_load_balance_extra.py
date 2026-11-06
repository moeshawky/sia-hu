from __future__ import annotations

from model_headroom.models.types import ExpertActivationData
from model_headroom.signals.load_balance import analyze_load_balance


class TestLoadBalanceEmptyEntropies:
    def test_layers_with_all_zero_totals(self) -> None:
        data = ExpertActivationData(
            metadata={"num_experts": 3},
            layers={"0": {"0": 0, "1": 0, "2": 0}},
        )
        result = analyze_load_balance(data)
        assert result.signal_status == "computed"
        assert result.entropy_per_layer[0] == 0.0

    def test_multiple_layers_all_zero(self) -> None:
        data = ExpertActivationData(
            metadata={"num_experts": 2},
            layers={
                "0": {"0": 0, "1": 0},
                "1": {"0": 0, "1": 0},
            },
        )
        result = analyze_load_balance(data)
        assert result.signal_status == "computed"
        assert result.balance_ratio == 0.0
