from __future__ import annotations

from model_headroom.models.loader import get_module_family


class TestGetModuleFamilyFFN:
    def test_gate_proj_is_ffn(self) -> None:
        assert get_module_family("model.layers.0.mlp.gate_proj.weight") == "ffn"

    def test_up_proj_is_ffn(self) -> None:
        assert get_module_family("model.layers.0.mlp.up_proj.weight") == "ffn"

    def test_down_proj_is_ffn(self) -> None:
        assert get_module_family("model.layers.0.mlp.down_proj.weight") == "ffn"

    def test_ffn_keyword(self) -> None:
        assert get_module_family("model.layers.0.ffn.weight") == "ffn"

    def test_mlp_keyword(self) -> None:
        assert get_module_family("model.layers.0.mlp.weight") == "ffn"


class TestGetModuleFamilyAttention:
    def test_q_proj(self) -> None:
        assert (
            get_module_family("model.layers.0.self_attn.q_proj.weight") == "attention"
        )

    def test_k_proj(self) -> None:
        assert (
            get_module_family("model.layers.0.self_attn.k_proj.weight") == "attention"
        )

    def test_v_proj(self) -> None:
        assert (
            get_module_family("model.layers.0.self_attn.v_proj.weight") == "attention"
        )

    def test_o_proj(self) -> None:
        assert (
            get_module_family("model.layers.0.self_attn.o_proj.weight") == "attention"
        )

    def test_qkv(self) -> None:
        assert get_module_family("model.layers.0.attn.qkv.weight") == "attention"

    def test_attn_keyword(self) -> None:
        assert get_module_family("model.layers.0.attn.weight") == "attention"


class TestGetModuleFamilyLinearAttention:
    def test_linear_attn_keyword(self) -> None:
        assert (
            get_module_family("model.layers.0.linear_attn.qkv_proj.weight")
            == "linear_attention"
        )

    def test_gated_deltanet(self) -> None:
        assert (
            get_module_family("model.layers.0.gated_deltanet.weight")
            == "linear_attention"
        )

    def test_mlp_with_linear_attn_not_misclassified(self) -> None:
        assert (
            get_module_family("model.layers.0.mlp.linear_attn_proj.weight")
            == "linear_attention"
        )

    def test_mlp_with_delta_not_misclassified(self) -> None:
        assert (
            get_module_family("model.layers.0.mlp.delta_proj.weight")
            == "linear_attention"
        )


class TestGetModuleFamilyEmbeddingAndHead:
    def test_embedding(self) -> None:
        assert get_module_family("model.embed_tokens.weight") == "embedding"

    def test_lm_head(self) -> None:
        assert get_module_family("lm_head.weight") == "lm_head"


class TestGetModuleFamilyExpertRouter:
    def test_expert(self) -> None:
        assert (
            get_module_family("model.layers.0.mlp.experts.0.gate_proj.weight") == "ffn"
        )
        assert get_module_family("model.layers.0.experts.0.weight") == "expert"

    def test_router(self) -> None:
        assert get_module_family("model.layers.0.mlp.gate.weight") == "ffn"
        assert get_module_family("model.layers.0.router.weight") == "router"


class TestGetModuleFamilyVisionAndMTP:
    def test_vision(self) -> None:
        assert get_module_family("model.visual.patch_embed.weight") == "vision"

    def test_vision_merger(self) -> None:
        assert get_module_family("model.visual.merger.weight") == "vision"

    def test_mtp(self) -> None:
        assert get_module_family("mtp.0.weight") == "mtp"

    def test_nextn(self) -> None:
        assert get_module_family("nextn_layer.weight") == "mtp"


class TestGetModuleFamilyStateAndOther:
    def test_state(self) -> None:
        assert get_module_family("model.layers.0.conv1d.weight") == "other"
        assert get_module_family("model.layers.0.state.weight") == "state"

    def test_z_proj_is_state(self) -> None:
        assert get_module_family("model.layers.0.z_proj.weight") == "state"

    def test_gate_in_state(self) -> None:
        assert get_module_family("model.layers.0.attn.gate.weight") == "attention"

    def test_other(self) -> None:
        assert get_module_family("unknown.foo.bar") == "other"

    def test_case_insensitive(self) -> None:
        assert get_module_family("MODEL.LAYERS.0.MLP.GATE_PROJ.WEIGHT") == "ffn"


class TestGetModuleFamilyClassifier:
    """BERT-family, ModernBERT, and laya weight name classification."""

    def test_bert_attention_query(self) -> None:
        """BERT attention query weight classifies as attention."""
        assert (
            get_module_family("bert.encoder.layer.0.attention.self.query.weight")
            == "attention"
        )

    def test_bert_attention_key(self) -> None:
        """BERT attention key weight classifies as attention."""
        assert (
            get_module_family("bert.encoder.layer.0.attention.self.key.weight")
            == "attention"
        )

    def test_bert_attention_value(self) -> None:
        """BERT attention value weight classifies as attention."""
        assert (
            get_module_family("bert.encoder.layer.0.attention.self.value.weight")
            == "attention"
        )

    def test_bert_attention_output_dense(self) -> None:
        """BERT attention output dense weight classifies as attention via nested exclusion."""
        assert (
            get_module_family("bert.encoder.layer.0.attention.output.dense.weight")
            == "attention"
        )

    def test_bert_ffn_intermediate_dense(self) -> None:
        """BERT FFN intermediate dense weight classifies as ffn."""
        assert (
            get_module_family("bert.encoder.layer.0.intermediate.dense.weight") == "ffn"
        )

    def test_bert_ffn_output_dense(self) -> None:
        """BERT FFN output dense weight classifies as ffn."""
        assert get_module_family("bert.encoder.layer.0.output.dense.weight") == "ffn"

    def test_bert_classifier_head(self) -> None:
        """BERT classifier head classifies as other."""
        assert get_module_family("classifier.weight") == "other"

    def test_bert_pooler_dense(self) -> None:
        """BERT pooler dense weight classifies as other."""
        assert get_module_family("pooler.dense.weight") == "other"

    def test_modernbert_attention(self) -> None:
        """ModernBERT attention query weight classifies as attention."""
        assert (
            get_module_family("model.encoder.layers.0.attention.self.query.weight")
            == "attention"
        )

    def test_modernbert_ffn(self) -> None:
        """ModernBERT FFN intermediate dense weight classifies as ffn."""
        assert (
            get_module_family("model.encoder.layers.0.intermediate.dense.weight")
            == "ffn"
        )

    def test_laya_head_attention(self) -> None:
        """laya head attention weight classifies as attention."""
        assert (
            get_module_family("head.layers.0.self_attn.in_proj_weight") == "attention"
        )

    def test_laya_head_linear(self) -> None:
        """laya head linear weight classifies as other."""
        assert get_module_family("head.layers.0.linear1.weight") == "other"

    def test_laya_scorer(self) -> None:
        """laya scorer weight classifies as other."""
        assert get_module_family("scorer.1.weight") == "other"

    def test_laya_act_head(self) -> None:
        """laya act_head weight classifies as other."""
        assert get_module_family("act_head.0.weight") == "other"

    def test_laya_type_emb(self) -> None:
        """laya type embedding classifies as other."""
        assert get_module_family("type_emb.weight") == "other"
