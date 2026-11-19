from model_headroom.acquire.behavioral import (
    BehavioralProbeResult,
    MockInferenceClient,
    run_behavioral_probe,
)
from model_headroom.acquire.probes import ProbeBudget, ProbeSpec


def _spec(**overrides):
    budget = ProbeBudget(max_seconds=10.0, max_model_loads=1)
    defaults = {
        "probe_id": "beh1",
        "question": "does option order change output?",
        "hypothesis": "output is invariant to option order",
        "target_model": "mock_endpoint",
        "target_revision": None,
        "measurement": "consistency_score",
        "acquisition_mechanism": "behavioral_inference",
        "input_data": None,
        "input_provenance": None,
        "experimental_control": "baseline condition",
        "intervention": None,
        "seed": 3,
        "budget": budget,
        "stopping_condition": "never",
        "expected_observations": None,
    }
    defaults.update(overrides)
    return ProbeSpec(**defaults)


def test_run_behavioral_probe_against_mock_endpoint():
    client = MockInferenceClient(responses={"hello world": "answer_A"})
    result = run_behavioral_probe(
        _spec(),
        client=client,
        conditions=("baseline", "option_order_permuted"),
        prompts=("hello world", "goodbye world"),
        repeats=2,
        option_permutation_seed=5,
    )
    assert isinstance(result, BehavioralProbeResult)
    assert result.probe_result.status == "completed"
    assert (
        result.probe_result.observations["total_calls"] == 8
    )  # 2 conditions x 2 prompts x 2 repeats
    assert len(result.observations) == 8
    assert result.consistency_score is not None
    assert 0.0 <= result.consistency_score <= 1.0


def test_mock_inference_client_records_calls():
    client = MockInferenceClient()
    out1 = client.submit("prompt one")
    out2 = client.submit("prompt two")
    assert out1 is not None
    assert out2 is not None
    assert len(client.calls) == 2
    assert client.calls[0][0] == "prompt one"
    assert client.calls[1][0] == "prompt two"


def test_behavioral_observation_preserves_condition_and_output():
    client = MockInferenceClient(responses={"x": "out_x"})
    result = run_behavioral_probe(
        _spec(),
        client=client,
        conditions=("baseline",),
        prompts=("x",),
        repeats=1,
    )
    obs = result.observations[0]
    assert obs.condition == "baseline"
    assert obs.prompt == "x"
    assert obs.output == "out_x"
    assert obs.error is None
    assert obs.latency_ms is not None
    assert obs.latency_ms >= 0.0


def test_consistency_score_is_one_for_deterministic_mock():
    client = MockInferenceClient(responses={"p": "fixed_output"})
    result = run_behavioral_probe(
        _spec(),
        client=client,
        conditions=("baseline",),
        prompts=("p",),
        repeats=5,
    )
    assert result.consistency_score == 1.0


def test_condition_deltas_requires_baseline():
    client = MockInferenceClient(
        responses={"a": "short", "b": "much longer output here"}
    )
    result = run_behavioral_probe(
        _spec(),
        client=client,
        conditions=("baseline", "input_perturbed"),
        prompts=("a", "b"),
        repeats=1,
    )
    assert result.condition_deltas is not None
    assert "input_perturbed" in result.condition_deltas


def test_no_baseline_means_no_condition_deltas():
    client = MockInferenceClient(responses={"a": "x"})
    result = run_behavioral_probe(
        _spec(),
        client=client,
        conditions=("option_order_permuted",),
        prompts=("a",),
        repeats=1,
    )
    assert result.condition_deltas is None


def test_mock_endpoint_is_injectable_not_singleton():
    c1 = MockInferenceClient(responses={"p": "one"})
    c2 = MockInferenceClient(responses={"p": "two"})
    assert c1.submit("p") == "one"
    assert c2.submit("p") == "two"
