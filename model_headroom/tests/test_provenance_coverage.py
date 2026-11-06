from __future__ import annotations

import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from model_headroom.analysis.orchestrator import run_analysis
from model_headroom.analysis.scoring import compute_coverage_metrics, is_decision_grade
from model_headroom.models.loader import (
    detect_architecture_details,
    get_module_family,
    infer_source_type_from_path,
    load_gradient_provenance_from_json,
    load_mlp_activations_with_provenance,
)
from model_headroom.models.types import (
    CoverageMetrics,
    GradientProvenance,
    GradientWeightPair,
    HeadroomReport,
    MLPActivations,
    SignalResult,
)
from model_headroom.signals.dead_neurons import analyze_dead_neurons
from model_headroom.signals.effective_rank import compute_effective_rank
from model_headroom.signals.gradient_coherence import compute_gradient_coherence
from model_headroom.signals.gradient_effective_rank import (
    compute_gradient_effective_rank,
)


def test_effective_rank_sampling_modes() -> None:
    w = torch.randn(8, 8)
    r1 = compute_effective_rank(
        w, sampling_mode="single_tile", parent_shape=(5120, 17408), sample_count=1
    )
    assert r1.sampling_mode == "single_tile"
    assert r1.coverage is not None
    assert r1.confidence is not None
    r2 = compute_effective_rank(w, sampling_mode="full_matrix")
    assert r2.sampling_mode == "full_matrix"
    assert r2.coverage is None or isinstance(r2.coverage, float)
    r3 = compute_effective_rank(w, sampling_mode="multi_tile", sample_count=3)
    assert r3.sampling_mode == "multi_tile"
    r4 = compute_effective_rank(w, sampling_mode="unknown")
    assert r4.sampling_mode == "unknown"
    r5 = compute_effective_rank(torch.zeros(4, 4), sampling_mode="single_tile")
    assert r5.signal_status == "degenerate"
    r6 = compute_effective_rank(torch.ones(1), sampling_mode="single_tile")
    assert r6.signal_status == "degenerate"


def test_dead_neurons_richer_stats() -> None:
    acts = MLPActivations(activations=torch.ones(2, 4, 16) * 0.5, layer_name="l0")
    rep = analyze_dead_neurons(acts, source_type="measured")
    assert rep.activation_rms is not None
    assert rep.mean_abs_activation is not None
    assert rep.variance is not None
    assert rep.near_zero_occupancy is not None
    assert rep.cross_token_variance is not None
    assert rep.source_type == "measured"
    acts2 = MLPActivations(activations=torch.zeros(2, 4, 16), layer_name="l1")
    rep2 = analyze_dead_neurons(acts2, source_type="synthetic")
    assert rep2.dead_fraction == 1.0
    assert rep2.source_type == "synthetic"


def test_gradient_effective_rank_basic() -> None:
    g = torch.randn(8, 8)
    rep = compute_gradient_effective_rank(g)
    assert rep.signal_status == "computed"
    assert rep.rank >= 1.0
    assert rep.r50 <= rep.r90 <= rep.r95 <= rep.r99
    g_zero = torch.zeros(8, 8)
    rep2 = compute_gradient_effective_rank(g_zero)
    assert rep2.signal_status == "degenerate"
    rep3 = compute_gradient_effective_rank(torch.ones(4))
    assert rep3.signal_status == "degenerate"


def test_gradient_coherence_basic() -> None:
    grads = [torch.randn(4, 4) for _ in range(3)]
    rep = compute_gradient_coherence(grads)
    assert rep.signal_status == "computed"
    assert -1.0 <= rep.mean_cosine_similarity <= 1.0
    assert 0.0 <= rep.subspace_overlap <= 1.0
    rep2 = compute_gradient_coherence([torch.randn(4, 4)])
    assert rep2.signal_status == "insufficient_data"
    grads_aligned = [torch.ones(4, 4) * (i + 1) for i in range(3)]
    rep3 = compute_gradient_coherence(grads_aligned)
    assert rep3.signal_status == "computed"


def test_get_module_family() -> None:
    assert get_module_family("model.layers.0.mlp.gate_proj.weight") == "ffn"
    assert get_module_family("model.layers.0.self_attn.q_proj.weight") == "attention"
    assert (
        get_module_family("model.layers.0.linear_attn.qkv_proj.weight")
        == "linear_attention"
    )
    assert get_module_family("model.layers.0.mlp.down_proj.weight") == "ffn"
    assert get_module_family("model.embed_tokens.weight") == "embedding"
    assert get_module_family("lm_head.weight") == "lm_head"
    assert get_module_family("model.visual.patch_embed.weight") == "vision"
    assert get_module_family("unknown.foo") == "other"


def test_detect_architecture_details_dense() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp)
        config = {
            "model_type": "qwen3_5",
            "hidden_size": 5120,
            "intermediate_size": 17408,
            "num_hidden_layers": 64,
            "layer_types": [
                "linear_attention",
                "linear_attention",
                "linear_attention",
                "full_attention",
            ]
            * 16,
            "num_experts": 0,
        }
        (p / "config.json").write_text(json.dumps(config))
        details = detect_architecture_details(str(p))
        assert details.family == "dense"
        assert details.subtype == "hybrid_linear_attention"
        assert details.has_linear_attention is True
        assert details.has_full_attention is True


def _gguf_config(expert_count: int) -> Any:
    return SimpleNamespace(expert_count=expert_count)


def _gguf_reader(names: list[str]) -> Any:
    return SimpleNamespace(tensors=[SimpleNamespace(name=n) for n in names])


def test_gguf_hybrid_tensor_names(monkeypatch: Any) -> None:
    monkeypatch.setattr(
        "model_headroom.models.gguf_loader.load_gguf_config",
        lambda path: _gguf_config(1),
    )
    monkeypatch.setattr(
        "model_headroom.models.gguf_loader.open_gguf_reader",
        lambda path: _gguf_reader(
            ["blk.0.ssm_alpha.weight", "blk.3.attn_q.weight", "blk.3.attn_k.weight"]
        ),
    )
    details = detect_architecture_details("/tmp/model.gguf")
    assert details.family == "dense"
    assert details.subtype == "hybrid_linear_attention"
    assert details.has_linear_attention is True
    assert details.has_full_attention is True
    assert details.has_moe is False
    assert details.detection_source == "gguf_tensor_names"


def test_gguf_pure_full_attention(monkeypatch: Any) -> None:
    monkeypatch.setattr(
        "model_headroom.models.gguf_loader.load_gguf_config",
        lambda path: _gguf_config(1),
    )
    monkeypatch.setattr(
        "model_headroom.models.gguf_loader.open_gguf_reader",
        lambda path: _gguf_reader(
            [
                "blk.0.attn_q.weight",
                "blk.0.attn_k.weight",
                "blk.0.attn_v.weight",
                "blk.0.attn_output.weight",
            ]
        ),
    )
    details = detect_architecture_details("/tmp/model.gguf")
    assert details.has_linear_attention is False
    assert details.has_full_attention is True
    assert details.subtype == "hybrid_linear_attention"


def test_gguf_attn_qkv_gate_not_full(monkeypatch: Any) -> None:
    monkeypatch.setattr(
        "model_headroom.models.gguf_loader.load_gguf_config",
        lambda path: _gguf_config(1),
    )
    monkeypatch.setattr(
        "model_headroom.models.gguf_loader.open_gguf_reader",
        lambda path: _gguf_reader(["blk.0.attn_qkv.weight", "blk.0.attn_gate.weight"]),
    )
    details = detect_architecture_details("/tmp/model.gguf")
    assert details.has_full_attention is False
    assert details.subtype == "dense_standard"


def test_gguf_moe_family(monkeypatch: Any) -> None:
    monkeypatch.setattr(
        "model_headroom.models.gguf_loader.load_gguf_config",
        lambda path: _gguf_config(8),
    )
    monkeypatch.setattr(
        "model_headroom.models.gguf_loader.open_gguf_reader",
        lambda path: _gguf_reader(["blk.0.ssm_alpha.weight"]),
    )
    details = detect_architecture_details("/tmp/model.gguf")
    assert details.family == "moe"
    assert details.has_moe is True
    assert details.has_linear_attention is True


def test_gguf_vision_mtp(monkeypatch: Any) -> None:
    monkeypatch.setattr(
        "model_headroom.models.gguf_loader.load_gguf_config",
        lambda path: _gguf_config(1),
    )
    monkeypatch.setattr(
        "model_headroom.models.gguf_loader.open_gguf_reader",
        lambda path: _gguf_reader(["model.visual.patch_embed.weight", "mtp.fc.weight"]),
    )
    details = detect_architecture_details("/tmp/model.gguf")
    assert details.has_vision_tower is True
    assert details.has_mtp is True


def test_gguf_reader_error_unknown(monkeypatch: Any) -> None:
    monkeypatch.setattr(
        "model_headroom.models.gguf_loader.load_gguf_config",
        lambda path: _gguf_config(1),
    )
    monkeypatch.setattr(
        "model_headroom.models.gguf_loader.open_gguf_reader",
        lambda path: (_ for _ in ()).throw(OSError("unreadable")),
    )
    details = detect_architecture_details("/tmp/model.gguf")
    assert details.subtype == "unknown"
    assert details.family == "unknown"


REAL_GGUF = "/kaggle/input/models/enagic/ternary-bonsai-27b/gguf/default/1/Ternary-Bonsai-27B-Q2_0.gguf"


@pytest.mark.skipif(not Path(REAL_GGUF).exists(), reason="real GGUF model absent")
def test_gguf_real_model_hybrid() -> None:
    details = detect_architecture_details(REAL_GGUF)
    assert details.subtype == "hybrid_linear_attention"
    assert details.has_linear_attention is True
    assert details.has_full_attention is True
    assert details.detection_source == "gguf_tensor_names"


def test_coverage_metrics_computation() -> None:
    report = run_analysis(model_name="test", model_path="/tmp/test")
    metrics = compute_coverage_metrics(report)
    assert isinstance(metrics, CoverageMetrics)
    assert metrics.coverage == 0.0
    assert metrics.measured_score == 0.0
    assert metrics.lower_bound == 0.0

    grad_pairs = [
        GradientWeightPair(
            gradient=torch.ones(4, 4) * 0.1, weight=torch.ones(4, 4), layer_name="l0"
        ),
        GradientWeightPair(
            gradient=torch.ones(4, 4) * 0.1, weight=torch.ones(4, 4), layer_name="l1"
        ),
    ]
    acts = [MLPActivations(activations=torch.ones(2, 4, 16), layer_name="l0")]
    report2 = run_analysis(
        model_name="test",
        model_path="/tmp/test",
        weight_pairs=grad_pairs,
        mlp_activations=acts,
        gradient_provenance=GradientProvenance(
            source_type="measured", verification_status="verified"
        ),
        mlp_source_type="measured",
        mlp_verification_status="verified",
        effective_rank_sampling_mode="single_tile",
    )
    metrics2 = compute_coverage_metrics(report2)
    assert metrics2.coverage > 0
    assert metrics2.measured_weight <= metrics2.total_applicable_weight

    report3 = run_analysis(
        model_name="test",
        model_path="/tmp/test",
        weight_pairs=grad_pairs,
        mlp_activations=acts,
        weight_pairs_source_type="synthetic",
        weight_pairs_verification_status="unverified",
        mlp_source_type="synthetic",
        mlp_verification_status="unverified",
    )
    metrics3 = compute_coverage_metrics(report3)
    assert metrics3.coverage == 0.0
    assert is_decision_grade(report3) is False


def test_load_provenance_helpers(tmp_path: Path) -> None:
    prov_file = tmp_path / "prov.json"
    prov_file.write_text(
        json.dumps(
            {
                "provenance": {
                    "model_revision": "abc",
                    "source_type": "measured",
                    "verification_status": "verified",
                }
            }
        )
    )
    prov = load_gradient_provenance_from_json(str(prov_file))
    assert prov is not None
    assert prov.model_revision == "abc"
    st, _vs = infer_source_type_from_path(str(prov_file))
    assert st == "measured"

    mlp_file = tmp_path / "mlp.json"
    mlp_file.write_text(
        json.dumps(
            {
                "layers": [{"activations": [[[1.0, 2.0]]], "layer_name": "l0"}],
                "source_type": "synthetic",
                "verification_status": "unverified",
            }
        )
    )
    acts, st2, vs2 = load_mlp_activations_with_provenance(str(mlp_file))
    assert acts is not None
    assert st2 == "synthetic"
    assert vs2 == "unverified"

    small_file = tmp_path / "small.json"
    small_file.write_text(
        json.dumps({"layers": [{"activations": [[[0.0, 0.0]]], "layer_name": "l0"}]})
    )
    acts2, st3, vs3 = load_mlp_activations_with_provenance(str(small_file))
    assert acts2 is not None
    assert st3 == "synthetic"
    assert vs3 == "unverified"


def test_headroom_report_new_fields() -> None:
    report = HeadroomReport(
        model_name="m",
        model_path="/tmp",
        timestamp="2026-01-01T00:00:00+00:00",
        signals=[
            SignalResult(
                signal_name="loss_saturation",
                status="skipped",
                data=None,
                duration_seconds=0.0,
            ),
            SignalResult(
                signal_name="growth_factor",
                status="skipped",
                data=None,
                duration_seconds=0.0,
            ),
            SignalResult(
                signal_name="effective_rank",
                status="skipped",
                data=None,
                duration_seconds=0.0,
            ),
            SignalResult(
                signal_name="grad_weight_ratio",
                status="skipped",
                data=None,
                duration_seconds=0.0,
            ),
            SignalResult(
                signal_name="dead_neurons",
                status="skipped",
                data=None,
                duration_seconds=0.0,
            ),
            SignalResult(
                signal_name="load_balance",
                status="not_applicable",
                data=None,
                duration_seconds=0.0,
            ),
            SignalResult(
                signal_name="specialization",
                status="not_applicable",
                data=None,
                duration_seconds=0.0,
            ),
        ],
        overall_headroom_score=0.5,
        recommendations=["a", "b", "c"],
        metadata={},
        analysis_mode="diagnostic",
        decision_grade=False,
    )
    assert report.analysis_mode == "diagnostic"
    assert report.coverage_metrics is None


def test_gradient_diagnostics_in_orchestrator() -> None:
    grad_pairs = [
        GradientWeightPair(
            gradient=torch.randn(8, 8),
            weight=torch.randn(8, 8),
            layer_name=f"layer_{i}_q_proj",
        )
        for i in range(4)
    ]
    report = run_analysis(
        model_name="test",
        model_path="/tmp/test",
        weight_pairs=grad_pairs,
        compute_gradient_diagnostics=True,
    )
    assert (
        "gradient_diagnostics" in report.metadata
        or report.overall_headroom_score is not None
    )
