"""Buffer-backed module discovery.

Covers Defect B: discover_modules must see modules whose weights live in
buffers (QTEA packed ternary) rather than parameters, has_weight must be
True for a buffer named "weight", buffer accounting must be reported,
param_count must stay parameter-only, and the adapter byte estimate must
never silently be 0 for a buffer-backed module.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from model_headroom.analysis.module_discovery import (
    discover_modules,
    generate_adapter_candidates,
)
from model_headroom.models.loader import get_module_family


class BufferOnlyModule(nn.Module):
    """A leaf module whose weight lives in buffers, with no parameters."""

    def __init__(self, weight_elements: int, scale_elements: int) -> None:
        super().__init__()
        self.register_buffer("weight", torch.zeros(weight_elements))
        self.register_buffer("scale", torch.zeros(scale_elements))


def _make_model() -> nn.Module:
    model = nn.Module()
    model.encoder = nn.Module()
    model.encoder.gated_deltanet = BufferOnlyModule(16, 4)
    model.encoder.mlp = BufferOnlyModule(32, 8)
    model.encoder.q_proj = BufferOnlyModule(16, 4)
    return model


class TestDiscoverModulesIncludesBufferBacked:
    """Buffer-only modules must appear in the discovery inventory."""

    def test_buffer_only_module_is_included(self) -> None:
        model = _make_model()
        modules = discover_modules(model)
        paths = {m.path for m in modules}
        assert "encoder.gated_deltanet" in paths
        assert "encoder.mlp" in paths
        assert "encoder.q_proj" in paths

    def test_buffer_module_has_weight_true(self) -> None:
        model = _make_model()
        modules = discover_modules(model)
        by_path = {m.path: m for m in modules}
        assert by_path["encoder.gated_deltanet"].has_weight is True
        assert by_path["encoder.mlp"].has_weight is True
        assert by_path["encoder.q_proj"].has_weight is True

    def test_buffer_module_family_classified(self) -> None:
        model = _make_model()
        modules = discover_modules(model)
        by_path = {m.path: m for m in modules}
        assert by_path["encoder.gated_deltanet"].family == "linear_attention"
        assert by_path["encoder.mlp"].family == "ffn"
        assert by_path["encoder.q_proj"].family == "attention"

    def test_param_count_is_parameters_only(self) -> None:
        model = _make_model()
        modules = discover_modules(model)
        by_path = {m.path: m for m in modules}
        assert by_path["encoder.gated_deltanet"].param_count == 0
        assert by_path["encoder.mlp"].param_count == 0
        assert by_path["encoder.q_proj"].param_count == 0

    def test_buffer_accounting_fields(self) -> None:
        model = _make_model()
        modules = discover_modules(model)
        by_path = {m.path: m for m in modules}
        # gated_deltanet: weight(16) + scale(4) = 20 elements, 2 buffers
        m = by_path["encoder.gated_deltanet"]
        assert m.buffer_count == 2
        assert m.buffer_elements == 20
        assert m.has_buffer_weight is True
        # mlp: weight(32) + scale(8) = 40 elements
        m = by_path["encoder.mlp"]
        assert m.buffer_count == 2
        assert m.buffer_elements == 40
        assert m.has_buffer_weight is True


class TestAdapterCandidatesTargetBufferModules:
    """generate_adapter_candidates must not exclude buffer-backed modules."""

    def test_all_linear_includes_buffer_backed(self) -> None:
        model = _make_model()
        modules = discover_modules(model)
        candidates = generate_adapter_candidates(modules)
        all_linear = next(c for c in candidates if c.name == "all-linear")
        assert "encoder.gated_deltanet" in all_linear.target_modules
        assert "encoder.mlp" in all_linear.target_modules
        assert "encoder.q_proj" in all_linear.target_modules

    def test_attention_only_includes_q_proj(self) -> None:
        model = _make_model()
        modules = discover_modules(model)
        candidates = generate_adapter_candidates(modules)
        attn = next(c for c in candidates if c.name == "attention-only")
        assert "encoder.q_proj" in attn.target_modules

    def test_ffn_only_includes_mlp(self) -> None:
        model = _make_model()
        modules = discover_modules(model)
        candidates = generate_adapter_candidates(modules)
        ffn = next(c for c in candidates if c.name == "ffn-only")
        assert "encoder.mlp" in ffn.target_modules


class TestAdapterByteEstimateNotZero:
    """Byte estimate for buffer-backed modules must not be silently 0."""

    def test_buffer_backed_byte_estimate_is_positive(self) -> None:
        model = _make_model()
        modules = discover_modules(model)
        candidates = generate_adapter_candidates(modules)
        all_linear = next(c for c in candidates if c.name == "all-linear")
        assert all_linear.estimated_adapter_bytes > 0

    def test_buffer_backed_module_not_in_incompatibilities(self) -> None:
        model = _make_model()
        modules = discover_modules(model)
        candidates = generate_adapter_candidates(modules)
        all_linear = next(c for c in candidates if c.name == "all-linear")
        # No incompatibility claiming buffer-backed modules have 0 bytes
        assert not any("0 bytes" in inc for inc in all_linear.incompatibilities)

    def test_assumptions_note_buffer_backed_estimate(self) -> None:
        model = _make_model()
        modules = discover_modules(model)
        candidates = generate_adapter_candidates(modules)
        all_linear = next(c for c in candidates if c.name == "all-linear")
        assert any(
            "buffer-backed" in a and "buffer element count" in a
            for a in all_linear.assumptions
        )


class TestGetModuleFamilyDirect:
    """Family classification by name is unchanged for buffer-backed paths."""

    def test_gated_deltanet_is_linear_attention(self) -> None:
        assert (
            get_module_family("encoder.gated_deltanet.proj.weight")
            == "linear_attention"
        )

    def test_mlp_is_ffn(self) -> None:
        assert get_module_family("encoder.mlp.proj.weight") == "ffn"

    def test_q_proj_is_attention(self) -> None:
        assert get_module_family("encoder.q_proj.weight") == "attention"
