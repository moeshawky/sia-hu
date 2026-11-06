"""Expert load balance entropy analysis (R-06).

Entropy = -sum(p * log(p)) where p = activation_count / total_tokens.
Ideal entropy = log(num_experts). balance_ratio = actual/ideal in [0, 1].

Dependencies: math (log), models.types (ExpertActivationData, LoadBalanceReport).
Provides: analyze_load_balance().
"""

from __future__ import annotations

import math

from model_headroom.models.types import ExpertActivationData, LoadBalanceReport


def analyze_load_balance(
    activation_data: ExpertActivationData,
) -> LoadBalanceReport:
    """Compute expert load balance entropy from activation counts.

    Entropy measures how evenly tokens are distributed across experts.
    Perfect balance has entropy = log(denominator). The balance_ratio
    normalizes to [0, 1] where 1.0 is perfectly balanced.

    Input contract:
        activation_data: ExpertActivationData with metadata containing
            num_experts (routing-observed experts fed by the corpus) and,
            optionally, num_experts_architectural (the experts the model
            declares). layers contains activation counts.

    Denominator rule:
        num_experts_architectural is the denominator when present and
        positive. It is the population the question asks about. A producer
        that sends only num_experts gets that value used, unchanged.

    Output contract (success):
        Returns LoadBalanceReport with signal_status="computed".
        balance_ratio in [0.0, 1.0].

    Output contract (single_expert):
        signal_status="single_expert" when num_experts_architectural == 1.
        Positive evidence of density, never inferred from a fed count.
        balance_ratio is None, never 0.0 — one expert has no load to
        measure, and 0.0 is the score for maximal imbalance.

    Output contract (expert_count_unknown):
        signal_status="expert_count_unknown" when num_experts == 1 and
        num_experts_architectural is absent. The file cannot say whether
        it describes a dense model or a collapsed router, so the authority
        claims neither. balance_ratio is None.

    Output contract (error):
        signal_status="error" if no layers or invalid data.

    Invariants:
        - balance_ratio in [0.0, 1.0] when computed, None otherwise
        - ideal_entropy > 0.0 for a denominator > 1
        - min_entropy <= max_entropy
        - len(entropy_per_layer) == number of layers processed

    Dependencies: Reads ExpertActivationData from moe_expert_profiler.py output.
    """
    fed = activation_data.metadata.get("num_experts", 0)
    architectural = activation_data.metadata.get("num_experts_architectural")

    if not isinstance(architectural, int) or isinstance(architectural, bool):
        architectural = None
    if architectural is not None and architectural < 1:
        architectural = None
    if not isinstance(fed, int) or isinstance(fed, bool):
        fed = 0

    if architectural is not None and fed > architectural:
        architectural = None

    if architectural == 1:
        return LoadBalanceReport(
            entropy_per_layer=[],
            min_entropy=0.0,
            max_entropy=0.0,
            ideal_entropy=0.0,
            balance_ratio=None,
            signal_status="single_expert",
        )

    if architectural is None and fed == 1:
        return LoadBalanceReport(
            entropy_per_layer=[],
            min_entropy=0.0,
            max_entropy=0.0,
            ideal_entropy=0.0,
            balance_ratio=None,
            signal_status="expert_count_unknown",
            error_message=(
                "num_experts=1 with no num_experts_architectural: cannot "
                "distinguish a dense model from a collapsed router"
            ),
        )

    if not activation_data.layers:
        return LoadBalanceReport(
            entropy_per_layer=[],
            min_entropy=0.0,
            max_entropy=0.0,
            ideal_entropy=0.0,
            balance_ratio=0.0,
            signal_status="error",
            error_message="No layer data provided",
        )

    num_experts = architectural if architectural is not None else fed

    if num_experts <= 0:
        return LoadBalanceReport(
            entropy_per_layer=[],
            min_entropy=0.0,
            max_entropy=0.0,
            ideal_entropy=0.0,
            balance_ratio=0.0,
            signal_status="error",
            error_message="Invalid num_experts in metadata",
        )

    ideal_entropy = math.log(num_experts)
    entropies: list[float] = []

    for _layer_idx, expert_counts in activation_data.layers.items():
        total = sum(expert_counts.values())
        if total == 0:
            entropies.append(0.0)
            continue

        # Compute entropy
        entropy = 0.0
        for count in expert_counts.values():
            if count > 0:
                p = count / total
                entropy -= p * math.log(p)
        entropies.append(entropy)

    if not entropies:
        return LoadBalanceReport(
            entropy_per_layer=[],
            min_entropy=0.0,
            max_entropy=0.0,
            ideal_entropy=ideal_entropy,
            balance_ratio=0.0,
            signal_status="error",
            error_message="No valid layers processed",
        )

    min_ent = min(entropies)
    max_ent = max(entropies)
    balance_ratio = min_ent / ideal_entropy if ideal_entropy > 0 else 0.0
    # Clamp to [0, 1]
    balance_ratio = max(0.0, min(1.0, balance_ratio))

    return LoadBalanceReport(
        entropy_per_layer=entropies,
        min_entropy=min_ent,
        max_entropy=max_ent,
        ideal_entropy=ideal_entropy,
        balance_ratio=balance_ratio,
        signal_status="computed",
    )
