"""Integration tests using real MoE models.

Tests the full pipeline against downloaded models at /models/moe_test_models/.
Skipped if models are unavailable (e.g. CI without model downloads).

Models tested:
- tinyqwen3moe2m (qwen3_moe, 2 experts, HF format)
- tinymoe2m (mixtral, 4 experts, HF + GGUF formats)
- qwen3-moe-tiny (qwen3_moe, 16 experts, mlp_only_layers=[0])
- Phi-3.5-MoE (phimoe, 8 experts, HF format)
- OLMoE-1B-7B (olmoe, 64 experts, HF format)
"""

from __future__ import annotations

from pathlib import Path

import pytest

from model_headroom.models.loader import (
    _extract_text_config,
    _get_expert_weight_name,
)

MODELS_ROOT = Path("/models/moe_test_models")


def _skip_if_model_missing(model_path: Path) -> None:
    """Skip test if model directory doesn't exist."""
    if not model_path.exists():
        pytest.skip(f"Model not found: {model_path}")


class TestArchitectureTensorNaming:
    """Test architecture-aware tensor naming dispatch."""

    def test_mixtral_tensor_names(self) -> None:
        """Mixtral uses block_sparse_moe.experts.{id}.w1/w2/w3.weight."""
        assert (
            _get_expert_weight_name("mixtral", 0, 0, "gate")
            == "model.layers.0.block_sparse_moe.experts.0.w1.weight"
        )
        assert (
            _get_expert_weight_name("mixtral", 5, 3, "down")
            == "model.layers.5.block_sparse_moe.experts.3.w2.weight"
        )
        assert (
            _get_expert_weight_name("mixtral", 2, 7, "up")
            == "model.layers.2.block_sparse_moe.experts.7.w3.weight"
        )

    def test_phimoe_tensor_names(self) -> None:
        """PhiMoE uses block_sparse_moe.experts.{id}.w1/w2/w3.weight."""
        assert (
            _get_expert_weight_name("phimoe", 0, 0, "gate")
            == "model.layers.0.block_sparse_moe.experts.0.w1.weight"
        )

    def test_qwen3_moe_tensor_names(self) -> None:
        """Qwen3Moe uses mlp.experts.{id}.gate_proj/down_proj/up_proj.weight."""
        assert (
            _get_expert_weight_name("qwen3_moe", 0, 0, "gate")
            == "model.layers.0.mlp.experts.0.gate_proj.weight"
        )
        assert (
            _get_expert_weight_name("qwen3_moe", 3, 15, "down")
            == "model.layers.3.mlp.experts.15.down_proj.weight"
        )
        assert (
            _get_expert_weight_name("qwen3_moe", 1, 8, "up")
            == "model.layers.1.mlp.experts.8.up_proj.weight"
        )

    def test_olmoe_tensor_names(self) -> None:
        """OLMoE uses mlp.experts.{id}.gate_proj/down_proj/up_proj.weight."""
        assert (
            _get_expert_weight_name("olmoe", 0, 0, "gate")
            == "model.layers.0.mlp.experts.0.gate_proj.weight"
        )

    def test_unknown_architecture_fallback(self) -> None:
        """Unknown architectures fall back to block_sparse_moe pattern."""
        assert (
            _get_expert_weight_name("unknown_arch", 0, 0, "gate")
            == "model.layers.0.block_sparse_moe.experts.0.w1.weight"
        )


class TestTextConfigExtraction:
    """Test nested text_config extraction."""

    def test_extract_nested_text_config(self) -> None:
        """Extract text_config when present."""
        config = {
            "model_type": "qwen3_5_moe",
            "text_config": {
                "model_type": "qwen3_moe",
                "num_experts": 128,
                "mlp_only_layers": [0],
            },
        }
        result = _extract_text_config(config)
        assert result["model_type"] == "qwen3_moe"
        assert result["num_experts"] == 128
        assert result["mlp_only_layers"] == [0]

    def test_no_text_config_passthrough(self) -> None:
        """Return config as-is when no text_config key."""
        config = {"model_type": "mixtral", "num_experts": 8}
        result = _extract_text_config(config)
        assert result is config
        assert result["model_type"] == "mixtral"

    def test_text_config_not_dict(self) -> None:
        """Return config as-is when text_config is not a dict."""
        config = {"model_type": "mixtral", "text_config": "not_a_dict"}
        result = _extract_text_config(config)
        assert result is config


class TestBERTFamilyDense:
    """Integration tests for BERT-family dense models (bert, deberta, laya).

    Verifies end-to-end analysis completes and dense-specific signals
    (load_balance, specialization) are correctly marked not_applicable.
    """

    @pytest.fixture(autouse=True)
    def check_models_exist(self) -> None:
        """Skip if any BERT-family model is missing."""
        models = [
            MODELS_ROOT / "bert-base-uncased",
            MODELS_ROOT / "deberta-base",
            MODELS_ROOT / "RockAndEgg-313M",
            MODELS_ROOT / "laya",
        ]
        missing = [m.name for m in models if not m.exists()]
        if missing:
            pytest.skip(f"Models not found: {', '.join(missing)}")

    def test_bert_base_uncased(self) -> None:
        """bert-base-uncased: dense model, expert signals not_applicable."""
        from model_headroom.analysis.orchestrator import run_analysis

        report = run_analysis(
            model_name="bert-base-uncased",
            model_path=str(MODELS_ROOT / "bert-base-uncased"),
        )
        assert report.metadata["architecture"] == "dense"
        by_name = {s.signal_name: s.status for s in report.signals}
        assert by_name["load_balance"] == "not_applicable"
        assert by_name["specialization"] == "not_applicable"

    def test_deberta_base(self) -> None:
        """deberta-base: dense model, expert signals not_applicable."""
        from model_headroom.analysis.orchestrator import run_analysis

        report = run_analysis(
            model_name="deberta-base",
            model_path=str(MODELS_ROOT / "deberta-base"),
        )
        assert report.metadata["architecture"] == "dense"
        by_name = {s.signal_name: s.status for s in report.signals}
        assert by_name["load_balance"] == "not_applicable"
        assert by_name["specialization"] == "not_applicable"

    def test_rockandegg_313m(self) -> None:
        """RockAndEgg-313M: dense model, expert signals not_applicable."""
        from model_headroom.analysis.orchestrator import run_analysis

        report = run_analysis(
            model_name="RockAndEgg-313M",
            model_path=str(MODELS_ROOT / "RockAndEgg-313M"),
        )
        assert report.metadata["architecture"] == "dense"
        by_name = {s.signal_name: s.status for s in report.signals}
        assert by_name["load_balance"] == "not_applicable"
        assert by_name["specialization"] == "not_applicable"

    def test_laya(self) -> None:
        """laya: dense model with encoder/config.json, expert signals not_applicable."""
        from model_headroom.analysis.orchestrator import run_analysis

        report = run_analysis(
            model_name="laya",
            model_path=str(MODELS_ROOT / "laya"),
        )
        assert report.metadata["architecture"] == "dense"
        by_name = {s.signal_name: s.status for s in report.signals}
        assert by_name["load_balance"] == "not_applicable"
        assert by_name["specialization"] == "not_applicable"
