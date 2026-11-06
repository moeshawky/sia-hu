"""Adversarial regression tests for the cross-batch contract (R-08/R-09) and
provenance nesting semantics.

Each test proves a specific laundering path stays closed:
  1. cross-layer gradients are never cross-batch gradients;
  2. scalar norms can never satisfy matrix inputs;
  3. missing batches yield unavailable/insufficient_data, never zeros;
  4. malformed or wrongly-nested provenance is rejected, never reinterpreted.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
import torch

from model_headroom.analysis.orchestrator import run_analysis
from model_headroom.cli import main
from model_headroom.models.loader import load_gradient_provenance_from_json
from model_headroom.models.types import GradientWeightPair
from model_headroom.signals.gradient_coherence import (
    compute_coherence_for_layer,
    compute_gradient_coherence,
)
from model_headroom.signals.gradient_effective_rank import (
    analyze_gradient_effective_ranks,
    analyze_gradient_effective_ranks_for_pairs,
)


def _pair(layer: str, seed: int) -> GradientWeightPair:
    g = torch.randn(8, 8, generator=torch.Generator().manual_seed(seed))
    w = torch.randn(8, 8, generator=torch.Generator().manual_seed(seed + 999))
    return GradientWeightPair(gradient=g, weight=w, layer_name=layer)


class TestCrossLayerIsNotCrossBatch:
    def test_mixed_layer_names_rejected(self) -> None:
        grads = [torch.randn(8, 8) for _ in range(3)]
        rep = compute_gradient_coherence(
            grads, layer_names=["layer_0_q", "layer_1_q", "layer_2_q"]
        )
        assert rep.signal_status == "error"
        assert "cross-layer" in (rep.error_message or "")

    def test_identical_tensors_across_layers_still_rejected(self) -> None:
        g = torch.randn(8, 8)
        rep = compute_gradient_coherence(
            [g.clone(), g.clone()], layer_names=["layer_0_q", "layer_1_q"]
        )
        assert rep.signal_status == "error"

    def test_pairs_wrapper_rejects_mixed_layers(self) -> None:
        rep = compute_coherence_for_layer(
            [_pair("layer_0_q", 1), _pair("layer_1_q", 2)]
        )
        assert rep.signal_status == "error"

    def test_same_layer_batches_compute(self) -> None:
        grads = [torch.randn(8, 8) for _ in range(3)]
        rep = compute_gradient_coherence(grads, layer_names=["layer_0_q"] * 3)
        assert rep.signal_status == "computed"

    def test_layer_names_length_mismatch_rejected(self) -> None:
        grads = [torch.randn(8, 8) for _ in range(2)]
        rep = compute_gradient_coherence(grads, layer_names=["only_one"])
        assert rep.signal_status == "error"

    def test_shape_mismatch_rejected(self) -> None:
        grads = [torch.randn(8, 8), torch.randn(8, 16)]
        rep = compute_gradient_coherence(grads, layer_names=["layer_0_q", "layer_0_q"])
        assert rep.signal_status == "error"


class TestScalarNormsCannotSatisfyMatrixInputs:
    def test_zero_dim_tensors_rejected(self) -> None:
        grads = [torch.tensor(0.5), torch.tensor(0.7)]
        rep = compute_gradient_coherence(grads, layer_names=["layer_0_q", "layer_0_q"])
        assert rep.signal_status == "error"
        assert "scalar" in (rep.error_message or "")

    def test_single_element_tensors_rejected(self) -> None:
        grads = [torch.ones(1, 1) * 0.1, torch.ones(1, 1) * 0.2]
        rep = compute_gradient_coherence(grads, layer_names=["layer_0_q", "layer_0_q"])
        assert rep.signal_status == "error"

    def test_mixed_scalar_and_matrix_rejected(self) -> None:
        grads = [torch.randn(8, 8), torch.tensor(0.3)]
        rep = compute_gradient_coherence(grads, layer_names=["layer_0_q", "layer_0_q"])
        assert rep.signal_status == "error"


class TestMissingBatchesAreUnavailableNeverZero:
    def test_single_batch_insufficient(self) -> None:
        rep = compute_gradient_coherence([torch.randn(8, 8)])
        assert rep.signal_status == "insufficient_data"

    def test_single_pair_insufficient(self) -> None:
        rep = compute_coherence_for_layer([_pair("layer_0_q", 1)])
        assert rep.signal_status == "insufficient_data"

    def test_orchestrator_reports_unavailable_not_zero(self) -> None:
        pairs = [_pair(f"layer_{i}_q_proj", i) for i in range(4)]
        report = run_analysis(
            model_name="test",
            model_path="/tmp/test",
            weight_pairs=pairs,
            compute_gradient_diagnostics=True,
        )
        gd = report.metadata["gradient_diagnostics"]
        assert gd["coherence"] is None
        assert gd["coherence_by_layer"] == {}
        assert "unavailable" in (gd["coherence_note"] or "")

    def test_orchestrator_computes_same_layer_batches(self) -> None:
        pairs = [_pair("layer_0_q_proj", 10), _pair("layer_0_q_proj", 11)]
        report = run_analysis(
            model_name="test",
            model_path="/tmp/test",
            weight_pairs=pairs,
            compute_gradient_diagnostics=True,
        )
        gd = report.metadata["gradient_diagnostics"]
        assert "layer_0_q_proj" in gd["coherence_by_layer"]
        assert gd["coherence"] is not None
        assert gd["coherence"]["batches"] == 2


class TestProvenanceNestingNeverReinterpreted:
    def test_flat_provenance_rejected_by_loader(self, tmp_path: Any) -> None:
        p = tmp_path / "prov.json"
        p.write_text(
            json.dumps(
                {
                    "model_revision": "abc",
                    "source_type": "measured",
                    "verification_status": "verified",
                }
            )
        )
        assert load_gradient_provenance_from_json(str(p)) is None

    def test_flat_provenance_rejected_by_cli(self, tmp_path: Any) -> None:
        p = tmp_path / "prov.json"
        p.write_text(json.dumps({"model_revision": "abc"}))
        result = main(
            [
                "--model-name",
                "test",
                "--model-path",
                str(tmp_path),
                "--gradient-provenance-file",
                str(p),
            ]
        )
        assert result == 1

    def test_wrong_nesting_key_rejected(self, tmp_path: Any) -> None:
        p = tmp_path / "prov.json"
        p.write_text(json.dumps({"data": {"model_revision": "abc"}}))
        assert load_gradient_provenance_from_json(str(p)) is None

    def test_invalid_source_type_rejected(self, tmp_path: Any) -> None:
        p = tmp_path / "prov.json"
        p.write_text(json.dumps({"provenance": {"source_type": "telepathic"}}))
        assert load_gradient_provenance_from_json(str(p)) is None

    def test_invalid_verification_status_rejected(self, tmp_path: Any) -> None:
        p = tmp_path / "prov.json"
        p.write_text(json.dumps({"provenance": {"verification_status": "trust_me"}}))
        assert load_gradient_provenance_from_json(str(p)) is None

    def test_canonical_nested_provenance_accepted(self, tmp_path: Any) -> None:
        p = tmp_path / "prov.json"
        p.write_text(
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
        prov = load_gradient_provenance_from_json(str(p))
        assert prov is not None
        assert prov.model_revision == "abc"


class TestRankProvenanceSemantics:
    def test_pairs_attribution_order_and_names(self) -> None:
        pairs = [_pair("layer_3_v_proj", 5), _pair("layer_7_k_proj", 6)]
        out = analyze_gradient_effective_ranks_for_pairs(pairs)
        assert [name for name, _ in out] == ["layer_3_v_proj", "layer_7_k_proj"]
        assert all(r.signal_status == "computed" for _, r in out)

    def test_names_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="must match"):
            analyze_gradient_effective_ranks(
                [torch.randn(4, 4)], layer_names=["a", "b"]
            )

    def test_scalar_gradient_degenerate_not_rank(self) -> None:
        out = analyze_gradient_effective_ranks([torch.tensor(0.5)])
        assert out[0].signal_status == "degenerate"
