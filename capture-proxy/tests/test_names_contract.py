"""Conformance tests for the shared name contract (capture_proxy.names).

Guards the convergence: one signal vocabulary instead of two drifting ones, a
total CLI-to-emitted name map, and a tensor-name map that refuses rather than
guesses. Includes the differential against the authority's own detector, which
is the comparison target Phase 6 needs and must not be deleted out from under
it.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import pytest

from capture_proxy.headroom import _detect_architecture, _resolve_tensor_name

if TYPE_CHECKING:
    from pathlib import Path

from capture_proxy.names import (
    ARCHITECTURE_TENSOR_NAMES,
    HEADROOM_SIGNALS,
    MOE_MODEL_TYPES,
    SIGNAL_KEYS,
    SIGNAL_NAME_MAP,
)

# The authority's core-6 signal names, restated here as the contract the
# sidecar must satisfy. Kept literal on purpose: a test that imports the value
# it is asserting against proves nothing.
AUTH_CORE_6 = frozenset(
    {
        "effective_rank",
        "dead_neurons",
        "load_balance",
        "specialization",
        "loss_saturation",
        "growth_factor",
    }
)


def _model(tmp_path: Path, config: dict[str, Any], weight_names: list[str]) -> Path:
    d = tmp_path / "model"
    d.mkdir()
    (d / "config.json").write_text(json.dumps(config))
    (d / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": dict.fromkeys(weight_names, "s1.safetensors")})
    )
    return d


def _weight_map(d: Path) -> dict[str, str]:
    return json.loads((d / "model.safetensors.index.json").read_text())["weight_map"]


class TestSignalVocabulary:
    def test_every_cli_name_resolves_to_an_emitted_name(self) -> None:
        resolved = {SIGNAL_NAME_MAP[n] for n in HEADROOM_SIGNALS}
        assert resolved == set(SIGNAL_KEYS)

    def test_resolution_is_six_of_six_in_order(self) -> None:
        resolved = [SIGNAL_NAME_MAP[n] for n in HEADROOM_SIGNALS]
        assert resolved == list(SIGNAL_KEYS)
        assert sum(1 for a, b in zip(resolved, SIGNAL_KEYS, strict=True) if a == b) == 6

    def test_name_map_is_total_in_both_directions(self) -> None:
        assert set(SIGNAL_NAME_MAP) == set(HEADROOM_SIGNALS)
        assert set(SIGNAL_NAME_MAP.values()) == set(SIGNAL_KEYS)

    def test_emitted_keys_are_the_authority_core_six(self) -> None:
        assert set(SIGNAL_KEYS) == AUTH_CORE_6

    def test_resolved_cli_names_are_the_authority_core_six(self) -> None:
        assert {SIGNAL_NAME_MAP[n] for n in HEADROOM_SIGNALS} == AUTH_CORE_6

    def test_vocabularies_have_equal_length(self) -> None:
        assert len(HEADROOM_SIGNALS) == len(SIGNAL_KEYS) == 6

    def test_no_duplicate_names(self) -> None:
        assert len(set(HEADROOM_SIGNALS)) == len(HEADROOM_SIGNALS)
        assert len(set(SIGNAL_KEYS)) == len(SIGNAL_KEYS)


class TestTensorNameMapNegativeControls:
    def test_absent_name_refuses_rather_than_guesses(self) -> None:
        assert (
            _resolve_tensor_name({"totally.unrelated.weight": "x"}, "gate", 0) is None
        )

    def test_empty_index_refuses(self) -> None:
        assert _resolve_tensor_name({}, "gate", 0) is None

    def test_moe_expert_template_beats_dense_mlp(self) -> None:
        wm = {
            "model.layers.0.mlp.gate_proj.weight": "s1",
            "model.layers.0.mlp.experts.0.gate_proj.weight": "s1",
        }
        assert (
            _resolve_tensor_name(wm, "gate", 0)
            == "model.layers.0.mlp.experts.0.gate_proj.weight"
        )

    def test_dense_mlp_resolves_when_no_expert_stack(self) -> None:
        wm = {"model.layers.0.mlp.gate_proj.weight": "s1"}
        assert (
            _resolve_tensor_name(wm, "gate", 0) == "model.layers.0.mlp.gate_proj.weight"
        )

    def test_uncovered_architecture_refuses(self) -> None:
        wm = {"model.layers.0.self_attn.q_proj.weight": "s1"}
        assert _resolve_tensor_name(wm, "gate", 0) is None

    def test_every_projection_has_at_least_one_template(self) -> None:
        for proj, templates in ARCHITECTURE_TENSOR_NAMES.items():
            assert templates, proj
            assert all(isinstance(t, str) and t for t in templates), proj

    def test_layer_placeholder_is_substituted(self) -> None:
        wm = {"model.layers.7.mlp.gate_proj.weight": "s1"}
        assert (
            _resolve_tensor_name(wm, "gate", 7) == "model.layers.7.mlp.gate_proj.weight"
        )


class TestArchitectureVerdictsNegativeControls:
    @pytest.mark.parametrize(
        ("config", "want"),
        [
            ({"model_type": "mixtral", "num_local_experts": 8}, "moe"),
            ({"model_type": "qwen3_moe", "num_experts": 64}, "moe"),
            ({"model_type": "llama", "architectures": ["LlamaForCausalLM"]}, "dense"),
            ({"model_type": "bert", "architectures": ["BertModel"]}, "dense"),
        ],
    )
    def test_carrier_classifies_both_ways(
        self, tmp_path: Path, config: dict[str, Any], want: str
    ) -> None:
        d = _model(tmp_path, config, [])
        assert _detect_architecture(d) == want

    def test_unreadable_config_is_unknown_not_dense(self, tmp_path: Path) -> None:
        d = _model(tmp_path, {}, [])
        assert _detect_architecture(d) == "unknown"

    def test_missing_directory_is_unknown(self, tmp_path: Path) -> None:
        assert _detect_architecture(tmp_path / "nope") == "unknown"

    def test_expert_count_overrides_name_list(self, tmp_path: Path) -> None:
        d = _model(tmp_path, {"model_type": "llama", "num_experts": 8}, [])
        assert _detect_architecture(d) == "moe"

    def test_moe_name_list_is_not_empty(self) -> None:
        assert MOE_MODEL_TYPES


class TestDifferentialAgainstAuthority:
    """The Phase 6 comparison target. If this file is ever deleted, the harness
    Phase 6 step 6.2 depends on has lost its second detector."""

    @pytest.mark.parametrize(
        "config",
        [
            {"model_type": "mixtral", "num_local_experts": 8},
            {"model_type": "qwen3_moe", "num_experts": 64},
            {"model_type": "llama", "architectures": ["LlamaForCausalLM"]},
            {"model_type": "bert", "architectures": ["BertModel"]},
            {
                "model_type": "emo",
                "architectures": ["EmoForCausalLM"],
                "num_experts": 8,
            },
        ],
    )
    def test_same_config_same_verdict(
        self, tmp_path: Path, config: dict[str, Any]
    ) -> None:
        try:
            from model_headroom.models import loader as authority
        except ImportError as exc:  # genuine absence: nothing exists to compare
            pytest.skip(f"authority absent: {exc}")

        d = _model(tmp_path, config, [])
        ours = _detect_architecture(d)
        # No guard on the call: AttributeError (symbol missing),
        # NotImplementedError (placeholder), TypeError (wrong signature) and
        # ValueError (real rejection) all mean BROKEN authority and must fail
        # loudly. Only the import above may skip.
        theirs = authority.detect_architecture_details(str(d)).family
        assert ours == theirs, f"ours={ours} authority={theirs} config={config}"


class TestDenseFamilyResolution:
    """DeBERTa-v2 / RoBERTa resolve to real tensors, not None (DEFER-3).

    The dense entries below mirror the libraries' own module structure
    (primary evidence: modeling_roberta.py / modeling_deberta_v2.py —
    RobertaSelfAttention.query, DisentangledSelfAttention.query_proj,
    SelfOutput/Intermediate/Output .dense under roberta./deberta.
    prefixes). These tests pin the resolution mechanism on synthetic maps;
    the names themselves are evidenced, not recalled.
    """

    def test_roberta_attention_resolves(self) -> None:
        wm = {"roberta.encoder.layer.0.attention.self.query.weight": "s1"}
        assert (
            _resolve_tensor_name(wm, "attention", 0)
            == "roberta.encoder.layer.0.attention.self.query.weight"
        )

    def test_roberta_intermediate_and_output_resolve(self) -> None:
        wm = {
            "roberta.encoder.layer.3.intermediate.dense.weight": "s1",
            "roberta.encoder.layer.3.output.dense.weight": "s1",
        }
        assert (
            _resolve_tensor_name(wm, "intermediate", 3)
            == "roberta.encoder.layer.3.intermediate.dense.weight"
        )
        assert (
            _resolve_tensor_name(wm, "output", 3)
            == "roberta.encoder.layer.3.output.dense.weight"
        )

    def test_deberta_attention_resolves_query_proj(self) -> None:
        wm = {"deberta.encoder.layer.0.attention.self.query_proj.weight": "s1"}
        assert (
            _resolve_tensor_name(wm, "attention", 0)
            == "deberta.encoder.layer.0.attention.self.query_proj.weight"
        )

    def test_deberta_intermediate_and_output_resolve(self) -> None:
        wm = {
            "deberta.encoder.layer.2.intermediate.dense.weight": "s1",
            "deberta.encoder.layer.2.output.dense.weight": "s1",
        }
        assert (
            _resolve_tensor_name(wm, "intermediate", 2)
            == "deberta.encoder.layer.2.intermediate.dense.weight"
        )
        assert (
            _resolve_tensor_name(wm, "output", 2)
            == "deberta.encoder.layer.2.output.dense.weight"
        )

    def test_bert_precedence_locked_when_both_present(self) -> None:
        wm = {
            "bert.encoder.layer.0.attention.self.query.weight": "s1",
            "roberta.encoder.layer.0.attention.self.query.weight": "s1",
        }
        assert (
            _resolve_tensor_name(wm, "attention", 0)
            == "bert.encoder.layer.0.attention.self.query.weight"
        )


class TestFusedTemplatePrecedence:
    """Fused 3-D expert names resolve only where nothing 2-D matches
    (DEFER-2): conservative first-match order, behavior unchanged on
    existing per-expert indexes."""

    def test_fused_gate_up_resolves_when_only_match(self) -> None:
        wm = {"model.layers.0.mlp.experts.gate_up_proj.weight": "s1"}
        assert (
            _resolve_tensor_name(wm, "gate", 0)
            == "model.layers.0.mlp.experts.gate_up_proj.weight"
        )

    def test_fused_down_resolves_when_only_match(self) -> None:
        wm = {"model.layers.0.mlp.experts.down_proj.weight": "s1"}
        assert (
            _resolve_tensor_name(wm, "down", 0)
            == "model.layers.0.mlp.experts.down_proj.weight"
        )

    def test_indexed_expert_still_wins_when_both_present(self) -> None:
        wm = {
            "model.layers.0.mlp.experts.0.gate_proj.weight": "s1",
            "model.layers.0.mlp.experts.gate_up_proj.weight": "s1",
        }
        assert (
            _resolve_tensor_name(wm, "gate", 0)
            == "model.layers.0.mlp.experts.0.gate_proj.weight"
        )
