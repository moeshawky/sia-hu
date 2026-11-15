"""Tests for model_headroom.acquire.layout."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import torch
from safetensors.torch import save_file

from model_headroom.acquire.layout import (
    resolve_weight_layout,
    sample_layout_weights,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def _write_checkpoint(directory: Path, tensors: dict[str, torch.Tensor]) -> None:
    save_file(
        {name: t.contiguous() for name, t in tensors.items()},
        str(directory / "model.safetensors"),
    )


def _qwen3_style_names(
    layers: int = 3, hidden: int = 8, inter: int = 16
) -> dict[str, torch.Tensor]:
    tensors: dict[str, torch.Tensor] = {}
    for i in range(layers):
        tensors[f"layers.{i}.mlp.gate_proj.weight"] = torch.randn(inter, hidden)
        tensors[f"layers.{i}.mlp.up_proj.weight"] = torch.randn(inter, hidden)
        tensors[f"layers.{i}.mlp.down_proj.weight"] = torch.randn(hidden, inter)
        tensors[f"layers.{i}.self_attn.q_proj.weight"] = torch.randn(2 * hidden, hidden)
        tensors[f"layers.{i}.self_attn.k_proj.weight"] = torch.randn(hidden, hidden)
        tensors[f"layers.{i}.self_attn.v_proj.weight"] = torch.randn(hidden, hidden)
        tensors[f"layers.{i}.self_attn.o_proj.weight"] = torch.randn(hidden, 2 * hidden)
    tensors["embed_tokens.weight"] = torch.randn(64, hidden)
    tensors["norm.weight"] = torch.randn(hidden)
    return tensors


def test_prefix_free_qwen3_layout_resolved(tmp_path: Path) -> None:
    _write_checkpoint(tmp_path, _qwen3_style_names())
    layout = resolve_weight_layout(tmp_path)

    assert layout.prefix == ""
    assert layout.container == "mlp"
    assert layout.families == ("attention", "ffn_down", "ffn_gated")
    assert set(layout.layers) == {0, 1, 2}
    assert layout.layers[0] == {
        "gate": "layers.0.mlp.gate_proj.weight",
        "up": "layers.0.mlp.up_proj.weight",
        "down": "layers.0.mlp.down_proj.weight",
        "q": "layers.0.self_attn.q_proj.weight",
        "k": "layers.0.self_attn.k_proj.weight",
        "v": "layers.0.self_attn.v_proj.weight",
        "o": "layers.0.self_attn.o_proj.weight",
    }
    assert layout.shapes["layers.0.mlp.gate_proj.weight"] == (16, 8)
    assert layout.shapes["layers.0.mlp.down_proj.weight"] == (8, 16)
    assert "embed_tokens.weight" in layout.unmatched
    assert "norm.weight" in layout.unmatched
    assert len(layout.unmatched) == 2
    for name in layout.layers[0].values():
        assert name not in layout.unmatched


def test_model_prefixed_layout_resolved(tmp_path: Path) -> None:
    tensors: dict[str, torch.Tensor] = {
        "model.layers.0.mlp.gate_proj.weight": torch.randn(16, 8),
        "model.layers.0.mlp.up_proj.weight": torch.randn(16, 8),
        "model.layers.0.mlp.down_proj.weight": torch.randn(8, 16),
    }
    _write_checkpoint(tmp_path, tensors)
    layout = resolve_weight_layout(tmp_path)

    assert layout.prefix == "model"
    assert layout.container == "mlp"
    assert layout.families == ("ffn_down", "ffn_gated")
    assert layout.unmatched == ()


def test_w1w2w3_layout_resolved(tmp_path: Path) -> None:
    tensors: dict[str, torch.Tensor] = {
        "model.layers.0.ffn.w1.weight": torch.randn(16, 8),
        "model.layers.0.ffn.w2.weight": torch.randn(8, 16),
        "model.layers.0.ffn.w3.weight": torch.randn(16, 8),
    }
    _write_checkpoint(tmp_path, tensors)
    layout = resolve_weight_layout(tmp_path)

    assert layout.prefix == "model"
    assert layout.container == "ffn"
    assert layout.families == ("ffn_gated",)
    assert set(layout.layers[0]) == {"w1", "w2", "w3"}


def test_mixed_prefix_is_none(tmp_path: Path) -> None:
    tensors: dict[str, torch.Tensor] = {
        "model.layers.0.mlp.gate_proj.weight": torch.randn(16, 8),
        "model.layers.0.mlp.up_proj.weight": torch.randn(16, 8),
        "layers.1.mlp.gate_proj.weight": torch.randn(16, 8),
        "layers.1.mlp.up_proj.weight": torch.randn(16, 8),
    }
    _write_checkpoint(tmp_path, tensors)
    layout = resolve_weight_layout(tmp_path)

    assert layout.prefix is None
    assert set(layout.layers) == {0, 1}


def test_no_projections_yields_empty_layout_with_inventory(tmp_path: Path) -> None:
    _write_checkpoint(
        tmp_path,
        {"embed_tokens.weight": torch.randn(64, 8), "norm.weight": torch.randn(8)},
    )
    layout = resolve_weight_layout(tmp_path)

    assert layout.layers == {}
    assert layout.prefix is None
    assert layout.container is None
    assert layout.families == ()
    assert layout.unmatched == ("embed_tokens.weight", "norm.weight")


def test_missing_path_yields_empty_layout(tmp_path: Path) -> None:
    layout = resolve_weight_layout(tmp_path / "nope")
    assert layout.layers == {}
    assert layout.prefix is None
    assert layout.unmatched == ()


def test_non_checkpoint_dir_yields_empty_layout(tmp_path: Path) -> None:
    (tmp_path / "readme.txt").write_text("not a checkpoint")
    layout = resolve_weight_layout(tmp_path)
    assert layout.layers == {}
    assert layout.prefix is None
    assert layout.unmatched == ()


def test_sample_single_tile_truncates(tmp_path: Path) -> None:
    tensors: dict[str, torch.Tensor] = {
        "layers.0.mlp.gate_proj.weight": torch.randn(512, 64),
        "layers.0.mlp.up_proj.weight": torch.randn(512, 64),
        "layers.0.mlp.down_proj.weight": torch.randn(64, 512),
    }
    _write_checkpoint(tmp_path, tensors)
    layout = resolve_weight_layout(tmp_path)
    samples = sample_layout_weights(layout, tmp_path, max_rows=256)

    assert len(samples) == 3
    by_name = {s.layer_name: s for s in samples}
    for name in ("layers.0.mlp.gate_proj.weight", "layers.0.mlp.up_proj.weight"):
        s = by_name[name]
        assert s.weight.shape == (256, 64)
        assert s.truncated is True
        assert s.sampling_mode == "single_tile"
        assert s.parent_shape == (512, 64)
    down = by_name["layers.0.mlp.down_proj.weight"]
    assert down.weight.shape == (64, 512)
    assert down.truncated is False
    assert down.sampling_mode == "full_matrix"
    assert down.parent_shape == (64, 512)
    names = [s.layer_name for s in samples]
    assert names == sorted(names)


def test_sample_full_matrix_when_small(tmp_path: Path) -> None:
    tensors: dict[str, torch.Tensor] = {
        "layers.0.mlp.gate_proj.weight": torch.randn(128, 64),
        "layers.0.mlp.up_proj.weight": torch.randn(128, 64),
        "layers.0.mlp.down_proj.weight": torch.randn(64, 128),
    }
    _write_checkpoint(tmp_path, tensors)
    layout = resolve_weight_layout(tmp_path)
    samples = sample_layout_weights(layout, tmp_path, max_rows=256)

    assert len(samples) == 3
    for s in samples:
        assert s.truncated is False
        assert s.sampling_mode == "full_matrix"
        assert s.weight.shape[0] == s.parent_shape[0]


def test_sample_deterministic_per_seed(tmp_path: Path) -> None:
    tensors: dict[str, torch.Tensor] = {
        "layers.0.mlp.gate_proj.weight": torch.randn(512, 64),
        "layers.0.mlp.up_proj.weight": torch.randn(512, 64),
        "layers.0.mlp.down_proj.weight": torch.randn(64, 512),
    }
    _write_checkpoint(tmp_path, tensors)
    layout = resolve_weight_layout(tmp_path)
    a = sample_layout_weights(layout, tmp_path, max_rows=256, seed=7)
    b = sample_layout_weights(layout, tmp_path, max_rows=256, seed=7)
    for sa, sb in zip(a, b, strict=True):
        assert torch.equal(sa.weight, sb.weight)
        assert sa.layer_name == sb.layer_name


def test_sample_attention_excluded(tmp_path: Path) -> None:
    _write_checkpoint(tmp_path, _qwen3_style_names())
    layout = resolve_weight_layout(tmp_path)
    samples = sample_layout_weights(layout, tmp_path, max_rows=256)

    for s in samples:
        assert "self_attn" not in s.layer_name
    assert len(samples) == 9


def test_package_imports_without_torch() -> None:
    code = (
        "import sys\n"
        "import model_headroom.acquire\n"
        "assert 'torch' not in sys.modules, 'torch imported at package import time'\n"
        "print('OK')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env={"PYTHONPATH": str(REPO_ROOT / "src"), "PATH": "/usr/bin:/bin"},
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "OK" in result.stdout
