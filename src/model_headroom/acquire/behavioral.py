"""Behavioral probes: controlled inference experiments against an inference
endpoint (real or mocked).

These observations expose input/output behavior only — never internal gradients
or activations. Exact input/output pairs are preserved and associated with
their experimental conditions, supporting behavioral diagnostics and
intervention comparisons without pretending to expose internals.

The endpoint client is injectable, so probes can be tested against a mocked
inference endpoint without installing SGLang or vLLM.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from model_headroom.acquire.probes import ProbeResult, ProbeSpec

BehavioralCondition = Literal[
    "baseline",
    "option_order_permuted",
    "state_ablated",
    "distractor_inserted",
    "repeated_request",
    "input_perturbed",
]


@dataclass(frozen=True)
class BehavioralObservation:
    """One input/output pair with its experimental condition."""

    condition: BehavioralCondition
    prompt: str
    output: str | None
    latency_ms: float | None
    error: str | None
    metadata: dict[str, Any]


@dataclass(frozen=True)
class BehavioralProbeResult:
    """Outcome of one behavioral probe."""

    probe_result: ProbeResult
    observations: tuple[BehavioralObservation, ...]
    consistency_score: float | None
    condition_deltas: dict[str, float] | None


class InferenceClient:
    """Injectable inference endpoint client.

    submit(prompt, **kwargs) -> str | None returns the model output for a
    prompt, or None on failure. Implementations may wrap SGLang, vLLM, or a
    mock for testing.
    """

    def submit(self, prompt: str, **kwargs: Any) -> str | None:
        raise NotImplementedError


class MockInferenceClient(InferenceClient):
    """Deterministic mock inference endpoint for testing behavioral probes."""

    def __init__(
        self, responses: dict[str, str] | None = None, latency_ms: float = 5.0
    ) -> None:
        self.responses = responses or {}
        self.latency_ms = latency_ms
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def submit(self, prompt: str, **kwargs: Any) -> str | None:
        self.calls.append((prompt, dict(kwargs)))
        if prompt in self.responses:
            return self.responses[prompt]
        return "mock_response_for_" + prompt[:20]


def run_behavioral_probe(
    spec: ProbeSpec,
    client: InferenceClient,
    conditions: tuple[BehavioralCondition, ...],
    prompts: tuple[str, ...],
    repeats: int = 1,
    option_permutation_seed: int | None = None,
) -> BehavioralProbeResult:
    """Run a bounded behavioral probe against an inference endpoint.

    Each (condition, prompt) pair is submitted to the client. The original
    checkpoint is never touched — only inference requests are made.

    Parameters:
        spec: The probe contract.
        client: The inference endpoint client (real or mocked).
        conditions: Experimental conditions to apply.
        prompts: Input prompts for each condition.
        repeats: Number of repeats per (condition, prompt) for consistency.
        option_permutation_seed: Seed for option-order permutation condition.

    Returns a BehavioralProbeResult with per-condition observations, a
    consistency score across repeats, and condition deltas.
    """
    observations: list[BehavioralObservation] = []
    call_count = 0

    for condition in conditions:
        for prompt in prompts:
            variant_prompt = prompt
            if (
                condition == "option_order_permuted"
                and option_permutation_seed is not None
            ):
                import random

                rng = random.Random(option_permutation_seed)  # noqa: S311 - shuffle only, not cryptographic
                words = prompt.split()
                if len(words) > 1:
                    shuffled = words[:]
                    rng.shuffle(shuffled)
                    variant_prompt = " ".join(shuffled)

            for repeat in range(repeats):
                call_count += 1
                start = monotonic_now()
                try:
                    output = client.submit(
                        variant_prompt, condition=condition, repeat=repeat
                    )
                    latency = monotonic_now() - start
                    observations.append(
                        BehavioralObservation(
                            condition=condition,
                            prompt=variant_prompt,
                            output=output,
                            latency_ms=latency * 1000.0,
                            error=None,
                            metadata={"repeat": repeat, "call": call_count},
                        )
                    )
                except Exception as exc:
                    latency = monotonic_now() - start
                    observations.append(
                        BehavioralObservation(
                            condition=condition,
                            prompt=variant_prompt,
                            output=None,
                            latency_ms=latency * 1000.0,
                            error=f"{type(exc).__name__}: {exc}",
                            metadata={"repeat": repeat, "call": call_count},
                        )
                    )

    # Consistency score: fraction of repeated (condition, prompt) pairs with
    # identical output across repeats.
    consistency_score = _consistency_score(observations, repeats)

    # Condition deltas: mean output-length delta per condition vs baseline.
    condition_deltas = _condition_deltas(observations, conditions)

    probe_result = ProbeResult(
        probe_id=spec.probe_id,
        status="completed",
        observations={
            "conditions": list(conditions),
            "prompts": list(prompts),
            "repeats": repeats,
            "total_calls": call_count,
            "endpoint_calls": call_count,
        },
        provenance=None,
        duration_seconds=0.0,
    )

    return BehavioralProbeResult(
        probe_result=probe_result,
        observations=tuple(observations),
        consistency_score=consistency_score,
        condition_deltas=condition_deltas,
    )


def monotonic_now() -> float:
    from time import monotonic

    return monotonic()


def _consistency_score(
    observations: list[BehavioralObservation], repeats: int
) -> float | None:
    if repeats < 2:
        return None
    groups: dict[tuple[BehavioralCondition, str], list[str | None]] = {}
    for obs in observations:
        groups.setdefault((obs.condition, obs.prompt), []).append(obs.output)
    if not groups:
        return None
    consistent = 0
    total = 0
    for _, outputs in groups.items():
        total += 1
        first = outputs[0]
        if all(o == first for o in outputs):
            consistent += 1
    return consistent / total


def _condition_deltas(
    observations: list[BehavioralObservation],
    conditions: tuple[BehavioralCondition, ...],
) -> dict[str, float] | None:
    if "baseline" not in conditions:
        return None
    baseline_lengths: list[float] = []
    other_lengths: dict[str, list[float]] = {
        c: [] for c in conditions if c != "baseline"
    }
    for obs in observations:
        if obs.output is None or obs.error is not None:
            continue
        length = float(len(obs.output))
        if obs.condition == "baseline":
            baseline_lengths.append(length)
        elif obs.condition in other_lengths:
            other_lengths[obs.condition].append(length)
    if not baseline_lengths:
        return None
    baseline_mean = sum(baseline_lengths) / len(baseline_lengths)
    deltas: dict[str, float] = {}
    for cond, lengths in other_lengths.items():
        if not lengths:
            continue
        cond_mean = sum(lengths) / len(lengths)
        deltas[cond] = cond_mean - baseline_mean
    return deltas or None
