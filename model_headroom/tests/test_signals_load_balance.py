from __future__ import annotations

import math

import pytest

from model_headroom.models.types import ExpertActivationData
from model_headroom.signals.load_balance import analyze_load_balance


class TestLoadBalanceFunctional:
    def test_balanced_load(
        self, activation_data_balanced: ExpertActivationData
    ) -> None:
        result = analyze_load_balance(activation_data_balanced)
        assert result.balance_ratio is not None
        assert result.balance_ratio > 0.9
        assert result.signal_status == "computed"

    def test_unbalanced_load(
        self, activation_data_imbalanced: ExpertActivationData
    ) -> None:
        result = analyze_load_balance(activation_data_imbalanced)
        assert result.balance_ratio is not None
        assert result.balance_ratio < 0.5

    def test_single_expert(self) -> None:
        data = ExpertActivationData(
            metadata={
                "num_experts": 1,
                "num_experts_architectural": 1,
                "num_layers": 1,
            },
            layers={"0": {"0": 100}},
        )
        result = analyze_load_balance(data)
        assert result.signal_status == "single_expert"
        assert result.balance_ratio is None

    def test_single_expert_without_layers(self) -> None:
        data = ExpertActivationData(
            metadata={
                "num_experts": 1,
                "num_experts_architectural": 1,
                "num_layers": 0,
            },
            layers={},
        )
        result = analyze_load_balance(data)
        assert result.signal_status == "single_expert"
        assert result.balance_ratio is None

    def test_single_expert_never_reports_zero_balance(self) -> None:
        data = ExpertActivationData(
            metadata={"num_experts": 1, "num_experts_architectural": 1},
            layers={"0": {"0": 100}},
        )
        result = analyze_load_balance(data)
        assert result.balance_ratio is None
        assert result.balance_ratio != 0.0

    def test_collapsed_router_is_maximal_imbalance(self) -> None:
        data = ExpertActivationData(
            metadata={"num_experts": 1, "num_experts_architectural": 8},
            layers={"0": {"0": 100}},
        )
        result = analyze_load_balance(data)
        assert result.signal_status == "computed"
        assert result.balance_ratio == 0.0
        assert result.ideal_entropy == pytest.approx(math.log(8))

    def test_architectural_count_is_the_denominator(self) -> None:
        data = ExpertActivationData(
            metadata={"num_experts": 4, "num_experts_architectural": 8},
            layers={"0": {str(k): 25 for k in range(4)}},
        )
        result = analyze_load_balance(data)
        assert result.signal_status == "computed"
        assert result.ideal_entropy == pytest.approx(math.log(8))
        assert result.balance_ratio == pytest.approx(math.log(4) / math.log(8))

    def test_architectural_below_fed_falls_back_to_fed(self) -> None:
        data = ExpertActivationData(
            metadata={"num_experts": 8, "num_experts_architectural": 2},
            layers={"0": {str(k): 100 for k in range(8)}},
        )
        result = analyze_load_balance(data)
        assert result.signal_status == "computed"
        assert result.ideal_entropy == pytest.approx(math.log(8))

    def test_architectural_alone_is_sufficient(self) -> None:
        data = ExpertActivationData(
            metadata={"num_experts_architectural": 8},
            layers={"0": {str(k): 25 for k in range(8)}},
        )
        result = analyze_load_balance(data)
        assert result.signal_status == "computed"
        assert result.balance_ratio == pytest.approx(1.0)

    def test_fed_one_without_architectural_claims_neither(self) -> None:
        data = ExpertActivationData(
            metadata={"num_experts": 1, "num_layers": 1},
            layers={"0": {"0": 100}},
        )
        result = analyze_load_balance(data)
        assert result.signal_status == "expert_count_unknown"
        assert result.balance_ratio is None
        assert result.error_message is not None

    def test_architectural_zero_is_treated_as_absent(self) -> None:
        data = ExpertActivationData(
            metadata={"num_experts": 4, "num_experts_architectural": 0},
            layers={"0": {str(k): 25 for k in range(4)}},
        )
        result = analyze_load_balance(data)
        assert result.signal_status == "computed"
        assert result.ideal_entropy == pytest.approx(math.log(4))

    def test_two_expert_imbalance_stays_zero(self) -> None:
        data = ExpertActivationData(
            metadata={"num_experts": 2, "num_layers": 1},
            layers={"0": {"0": 100, "1": 0}},
        )
        result = analyze_load_balance(data)
        assert result.signal_status == "computed"
        assert result.balance_ratio == 0.0

    def test_two_expert_balance_stays_one(self) -> None:
        data = ExpertActivationData(
            metadata={"num_experts": 2, "num_layers": 1},
            layers={"0": {"0": 50, "1": 50}},
        )
        result = analyze_load_balance(data)
        assert result.signal_status == "computed"
        assert result.balance_ratio == 1.0


class TestLoadBalanceEdge:
    def test_empty_layers(self) -> None:
        data = ExpertActivationData(
            metadata={"num_experts": 4},
            layers={},
        )
        result = analyze_load_balance(data)
        assert result.signal_status == "error"

    def test_num_experts_zero(self) -> None:
        data = ExpertActivationData(
            metadata={"num_experts": 0},
            layers={"0": {"0": 10, "1": 10}},
        )
        result = analyze_load_balance(data)
        assert result.signal_status == "error"

    def test_all_zero_counts(self) -> None:
        data = ExpertActivationData(
            metadata={"num_experts": 2},
            layers={"0": {"0": 0, "1": 0}},
        )
        result = analyze_load_balance(data)
        assert result.signal_status == "computed"
        assert result.entropy_per_layer[0] == 0.0


class TestLoadBalanceError:
    def test_negative_counts(self) -> None:
        data = ExpertActivationData(
            metadata={"num_experts": 2},
            layers={"0": {"0": -5, "1": 10}},
        )
        result = analyze_load_balance(data)
        assert result.signal_status == "computed"
