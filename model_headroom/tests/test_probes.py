from model_headroom.acquire.probes import (
    ProbeBudget,
    ProbeSpec,
    run_probe,
)


def _spec(**overrides):
    budget = ProbeBudget(max_seconds=10.0, max_model_loads=1, max_backward_passes=1)
    defaults = {
        "probe_id": "p1",
        "question": "does gradient exist?",
        "hypothesis": "gradient is nonzero",
        "target_model": "synthetic",
        "target_revision": None,
        "measurement": "grad_norm",
        "acquisition_mechanism": "pytorch_backward",
        "input_data": None,
        "input_provenance": None,
        "experimental_control": None,
        "intervention": None,
        "seed": 42,
        "budget": budget,
        "stopping_condition": "never",
        "expected_observations": None,
    }
    defaults.update(overrides)
    return ProbeSpec(**defaults)


def test_run_probe_completed_with_execute():
    result = run_probe(_spec(), execute=lambda: {"observed": 1.0})
    assert result.probe_id == "p1"
    assert result.status == "completed"
    assert result.observations == {"observed": 1.0}
    assert result.error_message is None
    assert result.budget_exhausted is False
    assert result.duration_seconds >= 0.0


def test_run_probe_failure_becomes_failed_not_negative_finding():
    def boom():
        raise RuntimeError("loss exploded")

    result = run_probe(_spec(), execute=boom)
    assert result.status == "failed"
    assert result.error_message is not None
    assert "RuntimeError" in result.error_message
    assert result.budget_exhausted is False


def test_run_probe_budget_exhausted_before_execution():
    budget = ProbeBudget(max_seconds=0.0, max_model_loads=1, max_backward_passes=1)
    result = run_probe(_spec(budget=budget), execute=lambda: {"x": 1})
    assert result.status == "inconclusive"
    assert result.budget_exhausted is True
    assert result.stopping_reason is not None
    assert result.observations == {}


def test_run_probe_evidence_bundle_wrap():
    result = run_probe(_spec(), execute=lambda: {"obs": 2})
    bundle = result.to_evidence_bundle("grad_weight_ratio")
    assert bundle.signal_name == "grad_weight_ratio"
    assert bundle.provenance.provenance_class == "locally_measured"
    assert bundle.provenance.acquisition_method.startswith("probe:")
    assert bundle.payload["probe_id"] == "p1"
    assert bundle.payload["status"] == "completed"


def test_probe_spec_requires_probe_id():
    budget = ProbeBudget()
    try:
        ProbeSpec(
            probe_id="",
            question="q",
            hypothesis="h",
            target_model="m",
            target_revision=None,
            measurement="m",
            acquisition_mechanism="pytorch_backward",
            input_data=None,
            input_provenance=None,
            experimental_control=None,
            intervention=None,
            seed=42,
            budget=budget,
            stopping_condition="never",
            expected_observations=None,
        )
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for empty probe_id")


def test_probe_result_distinguishes_inconclusive_from_completed():
    ok = run_probe(_spec(), execute=lambda: {"a": 1})
    assert ok.status == "completed"
    assert ok.budget_exhausted is False

    bad = run_probe(
        _spec(
            budget=ProbeBudget(
                max_seconds=0.0, max_model_loads=1, max_backward_passes=1
            )
        ),
        execute=lambda: {"a": 1},
    )
    assert bad.status == "inconclusive"
    assert bad.budget_exhausted is True
    assert ok is not bad
