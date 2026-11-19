import torch

from model_headroom.acquire.optimization import run_optimization_probe
from model_headroom.acquire.probe_integration import (
    optimization_result_to_evidence_bundles,
    probe_evidence_to_metadata,
    probe_to_evidence_bundle,
)
from model_headroom.acquire.probes import ProbeBudget, ProbeSpec, run_probe
from model_headroom.acquire.provenance import EvidenceBundle
from model_headroom.analysis.orchestrator import run_analysis
from model_headroom.models.types import GradientWeightPair


def _spec(**overrides):
    budget = ProbeBudget(max_seconds=10.0, max_model_loads=1, max_backward_passes=1)
    defaults = {
        "probe_id": "int1",
        "question": "q",
        "hypothesis": "h",
        "target_model": "m",
        "target_revision": None,
        "measurement": "m",
        "acquisition_mechanism": "pytorch_backward",
        "input_data": None,
        "input_provenance": None,
        "experimental_control": None,
        "intervention": None,
        "seed": 1,
        "budget": budget,
        "stopping_condition": "never",
        "expected_observations": None,
    }
    defaults.update(overrides)
    return ProbeSpec(**defaults)


def test_probe_to_evidence_bundle_shape():
    result = run_probe(_spec(), execute=lambda: {"obs": 1})
    bundle = probe_to_evidence_bundle(result, "grad_weight_ratio")
    assert isinstance(bundle, EvidenceBundle)
    assert bundle.signal_name == "grad_weight_ratio"
    assert bundle.provenance.provenance_class == "locally_measured"
    assert bundle.payload["probe_id"] == "int1"
    assert bundle.payload["status"] == "completed"


def test_optimization_result_to_evidence_bundles():
    inputs = (torch.randn(4, 4), torch.randn(4, 2))

    def factory():
        return torch.nn.Linear(4, 2)

    def loss_fn(model, inputs):
        x, y = inputs
        return torch.nn.functional.mse_loss(model(x), y)

    result = run_optimization_probe(
        _spec(),
        model_factory=factory,
        loss_fn=loss_fn,
        inputs=inputs,
        steps=1,
        lr=1e-3,
    )
    bundles = optimization_result_to_evidence_bundles(result)
    assert len(bundles) >= 1
    assert any(b.signal_name == "grad_weight_ratio" for b in bundles)


def test_probe_evidence_to_metadata_is_serializable():
    result = run_probe(_spec(), execute=lambda: {"obs": 1})
    bundle = probe_to_evidence_bundle(result, "behavioral_consistency")
    metadata = probe_evidence_to_metadata([bundle])
    assert "probe_evidence:behavioral_consistency" in metadata
    entry = metadata["probe_evidence:behavioral_consistency"]
    assert entry["signal_name"] == "behavioral_consistency"
    assert entry["provenance_class"] == "locally_measured"
    assert "payload" in entry


def test_run_analysis_accepts_probe_evidence_without_changing_signals():
    pair = GradientWeightPair(
        gradient=torch.tensor([1.0, 2.0]),
        weight=torch.tensor([3.0, 4.0]),
        layer_name="layer.0",
    )
    result = run_probe(_spec(), execute=lambda: {"obs": 1})
    bundle = probe_to_evidence_bundle(result, "grad_weight_ratio")

    report = run_analysis(
        model_name="synthetic",
        model_path="/tmp",
        weight_pairs=[pair],
        probe_evidence=[bundle],
    )
    assert "probe_evidence" in report.metadata
    assert "probe_evidence:grad_weight_ratio" in report.metadata["probe_evidence"]
    assert len(report.signals) >= 1
    assert report.overall_headroom_score >= 0.0


def test_run_analysis_without_probe_evidence_unchanged():
    pair = GradientWeightPair(
        gradient=torch.tensor([1.0, 2.0]),
        weight=torch.tensor([3.0, 4.0]),
        layer_name="layer.0",
    )
    report = run_analysis(
        model_name="synthetic",
        model_path="/tmp",
        weight_pairs=[pair],
    )
    assert "probe_evidence" not in report.metadata
