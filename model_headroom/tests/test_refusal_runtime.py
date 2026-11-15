from __future__ import annotations

import os
import sys
from typing import TYPE_CHECKING

import pytest
import torch
from torch import nn

if TYPE_CHECKING:
    from pathlib import Path

from model_headroom.refusal.runtime import (
    DEFAULT_STRENGTH,
    DEFAULT_TARGET_SUFFIXES,
    EXTENDED_TARGET_SUFFIXES,
    LAMBDA_ENV_VAR,
    TENSOR_REFUSAL_DIRECTION,
    UNIT_NORM_TOLERANCE,
    RefusalArtifact,
    _make_arithmetic_hook,
    find_residual_writers,
    install_refusal_ablation,
    load_artifact,
    resolve_lambda,
    verify_lambda_zero_identity,
)


class _Expert(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.down_proj = nn.Linear(dim, dim)


class _Model(nn.Module):
    def __init__(self, dim: int = 8) -> None:
        super().__init__()
        self.dim = dim
        self.self_attn = nn.Module()
        self.self_attn.o_proj = nn.Linear(dim, dim)
        self.mlp = nn.Module()
        self.mlp.down_proj = nn.Linear(dim, dim)
        self.mlp.shared_expert = _Expert(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.self_attn.o_proj(x) + self.mlp.down_proj(x)


class _TupleLeaf(nn.Module):
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return (x, torch.full_like(x, 2.0))


class _TupleModel(nn.Module):
    def __init__(self, dim: int = 8) -> None:
        super().__init__()
        self.tuplehead = _TupleLeaf()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.tuplehead(x)


class _Passthrough(nn.Module):
    def forward(self, x: object) -> object:
        return x


class _Wrapper(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.passthrough = _Passthrough()

    def forward(self, x: object) -> object:
        return self.passthrough(x)


_DIRECTION = torch.tensor([1.0] + [0.0] * 7)


def _write_artifact(
    path: Path, direction: torch.Tensor, metadata: dict[str, str] | None = None
) -> None:
    from safetensors import torch as safetensors_torch

    tensors = {TENSOR_REFUSAL_DIRECTION: direction}
    if metadata is None:
        metadata = {}
    safetensors_torch.save_file(tensors, path, metadata=metadata)


def test_constants() -> None:
    assert DEFAULT_TARGET_SUFFIXES == (
        "o_proj",
        "out_proj",
        "down_proj",
        "shared_expert",
    )
    assert (
        *DEFAULT_TARGET_SUFFIXES,
        "experts",
        "linear_attn",
    ) == EXTENDED_TARGET_SUFFIXES
    assert DEFAULT_STRENGTH == 0.0
    assert UNIT_NORM_TOLERANCE == 1e-3


def test_find_residual_writers_keeps_outmost_and_drops_descendant() -> None:
    model = _Model()
    writers = find_residual_writers(model)
    names = [name for name, _ in writers]
    assert "self_attn.o_proj" in names
    assert "mlp.down_proj" in names
    assert "mlp.shared_expert" in names
    assert "mlp.shared_expert.down_proj" not in names
    assert names == sorted(names)


def test_lambda_zero_bit_exact_no_op() -> None:
    model = _Model()
    artifact = RefusalArtifact(direction=_DIRECTION)
    inputs = torch.randn(2, 8)
    before = model(inputs)
    handle = install_refusal_ablation(model, artifact, strength=0.0)
    assert handle.strength_tensor is None
    after = model(inputs)
    handle.remove()
    assert torch.equal(before, after)


def test_lambda_zero_bit_exact_arithmetic_at_zero() -> None:
    model = _Model()
    artifact = RefusalArtifact(direction=_DIRECTION)
    inputs = torch.randn(2, 8)
    before = model(inputs)
    handle = install_refusal_ablation(model, artifact, strength=0.5)
    assert handle.strength_tensor is not None
    handle.strength_tensor.fill_(0.0)
    after = model(inputs)
    handle.remove()
    assert torch.equal(before, after)


def test_lambda_one_removes_direction_component() -> None:
    model = _Model()
    artifact = RefusalArtifact(direction=_DIRECTION)
    inputs = torch.randn(4, 8)
    handle = install_refusal_ablation(model, artifact, strength=1.0)
    output = model(inputs)
    handle.remove()
    assert float(output[..., 0].detach().abs().max()) < 1e-6


def test_tuple_output_projects_element_zero_keeps_rest() -> None:
    model = _TupleModel()
    artifact = RefusalArtifact(direction=_DIRECTION)
    inputs = torch.randn(2, 8)
    before = model(inputs)
    handle = install_refusal_ablation(
        model, artifact, strength=1.0, targets=("tuplehead",)
    )
    after = model(inputs)
    handle.remove()
    assert float(after[0][..., 0].abs().max()) < 1e-6
    assert torch.equal(after[1], before[1])


def test_unsupported_output_returns_none_unchanged() -> None:
    model = _Wrapper()
    artifact = RefusalArtifact(direction=_DIRECTION)
    obj = object()
    handle = install_refusal_ablation(
        model, artifact, strength=1.0, targets=("passthrough",)
    )
    out = model(obj)
    assert out is obj
    handle.remove()


def test_resolve_lambda_explicit_wins_over_env_and_default() -> None:
    os.environ[LAMBDA_ENV_VAR] = "0.3"
    try:
        assert resolve_lambda() == 0.3
        assert resolve_lambda(0.7) == 0.7
    finally:
        del os.environ[LAMBDA_ENV_VAR]
    assert resolve_lambda() == DEFAULT_STRENGTH
    assert resolve_lambda(0.9) == 0.9


def test_resolve_lambda_env_read_at_call_time() -> None:
    os.environ[LAMBDA_ENV_VAR] = "0.1"
    try:
        assert resolve_lambda() == 0.1
        os.environ[LAMBDA_ENV_VAR] = "0.4"
        assert resolve_lambda() == 0.4
    finally:
        os.environ.pop(LAMBDA_ENV_VAR, None)


@pytest.mark.parametrize("raw", ["nan", "inf", "-inf", "abc"])
def test_resolve_lambda_rejects_invalid(raw: str) -> None:
    os.environ[LAMBDA_ENV_VAR] = raw
    try:
        with pytest.raises(ValueError, match=LAMBDA_ENV_VAR):
            resolve_lambda()
    finally:
        os.environ.pop(LAMBDA_ENV_VAR, None)


def test_load_artifact_round_trip(tmp_path: Path) -> None:
    direction = torch.tensor([1.0, 0.0, 0.0])
    path = tmp_path / "artifact.safetensors"
    _write_artifact(path, direction, metadata={"hidden_size": "3", "source": "test"})
    artifact = load_artifact(str(path))
    assert torch.equal(artifact.direction, direction)
    assert artifact.metadata["hidden_size"] == "3"
    assert artifact.metadata["source"] == "test"


def test_load_artifact_missing_direction_raises(tmp_path: Path) -> None:
    from safetensors import torch as safetensors_torch

    path = tmp_path / "no_dir.safetensors"
    safetensors_torch.save_file({"other": torch.tensor([1.0])}, path)
    with pytest.raises(ValueError, match=TENSOR_REFUSAL_DIRECTION):
        load_artifact(str(path))


def test_load_artifact_non_unit_direction_raises(tmp_path: Path) -> None:
    direction = torch.tensor([2.0, 0.0])
    path = tmp_path / "nonunit.safetensors"
    _write_artifact(path, direction)
    with pytest.raises(ValueError, match="not unit-norm"):
        load_artifact(str(path))


def test_import_isolation_no_vllm_sglang() -> None:
    assert "vllm" not in sys.modules
    assert "sglang" not in sys.modules


def test_set_strength_noop_nonzero_raises() -> None:
    model = _Model()
    artifact = RefusalArtifact(direction=_DIRECTION)
    handle = install_refusal_ablation(model, artifact, strength=0.0)
    assert handle.strength_tensor is None
    with pytest.raises(RuntimeError):
        handle.set_strength(0.5)
    handle.remove()


def test_verify_lambda_zero_identity_returns_true() -> None:
    model = _Model()
    artifact = RefusalArtifact(direction=_DIRECTION)
    inputs = torch.randn(2, 8)
    assert verify_lambda_zero_identity(model, artifact, inputs=inputs) is True


def _make_hook_model(dtype: torch.dtype) -> tuple[_Model, RefusalArtifact]:
    torch.manual_seed(0)
    model = _Model()
    for module in model.modules():
        if isinstance(module, nn.Linear):
            module.weight.data = module.weight.data.to(dtype)
            if module.bias is not None:
                module.bias.data = module.bias.data.to(dtype)
    artifact = RefusalArtifact(direction=_DIRECTION.to(dtype))
    return model, artifact


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_hook_preserves_output_dtype(dtype: torch.dtype) -> None:
    model, artifact = _make_hook_model(dtype)
    inputs = torch.randn(2, 8, dtype=dtype)
    before = model(inputs)
    handle = install_refusal_ablation(model, artifact, strength=0.0)
    after = model(inputs)
    handle.remove()
    assert after.dtype == dtype
    assert torch.equal(after, before)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_hook_lambda_zero_identity_all_dtypes(dtype: torch.dtype) -> None:
    model, artifact = _make_hook_model(dtype)
    torch.manual_seed(1)
    inputs = torch.randn(2, 8, dtype=dtype)
    before = model(inputs)
    handle = install_refusal_ablation(model, artifact, strength=0.0)
    after = model(inputs)
    handle.remove()
    assert after.dtype == before.dtype
    assert torch.equal(after, before)


def test_hook_tuple_branch_preserves_dtype_and_tail() -> None:
    torch.manual_seed(2)
    model = _TupleModel()
    artifact = RefusalArtifact(direction=_DIRECTION)
    inputs = torch.randn(2, 8, dtype=torch.bfloat16)
    before = model(inputs)
    handle = install_refusal_ablation(
        model, artifact, strength=0.0, targets=("tuplehead",)
    )
    after = model(inputs)
    handle.remove()
    assert after[0].dtype == torch.bfloat16
    assert torch.equal(after[0], before[0])
    assert torch.equal(after[1], before[1])


def test_noop_mode_returns_none_and_allocates_no_f32() -> None:
    model = _Model()
    artifact = RefusalArtifact(direction=_DIRECTION)
    handle = install_refusal_ablation(model, artifact, strength=0.0)
    assert handle.strength_tensor is None
    handle.remove()


def test_hook_lambda_matches_float32_reference_on_bf16() -> None:
    torch.manual_seed(3)
    output = torch.randn(4, 8, dtype=torch.bfloat16)
    direction = _DIRECTION.to(torch.float32)
    strength_tensor = torch.tensor(0.3, dtype=torch.float32)

    hook = _make_arithmetic_hook(direction, strength_tensor)
    applied = hook(None, None, output)

    reference = output.to(torch.float32)
    proj = (reference * direction).sum(dim=-1, keepdim=True)
    expected = reference - 0.3 * proj * direction
    diff = (applied.float() - expected).abs().max().item()
    assert diff < 1e-2, f"effective lambda drift {diff} exceeds 1e-2 bound"
