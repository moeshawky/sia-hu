from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import torch
from torch import nn

if TYPE_CHECKING:
    from pathlib import Path

from model_headroom.refusal.runtime import (
    ARCHITECTURE_TARGET_MAPS,
    CAPTURE_SPACE_RESIDUAL,
    CAPTURE_SPACE_WIDENED,
    TENSOR_REFUSAL_DIRECTION,
    RefusalArtifact,
    find_residual_writers,
    install_refusal_ablation,
    load_artifact,
    project_direction_to_residual,
    validate_install,
    writer_output_width,
)

HIDDEN = 8
STREAMS = 4


def _residual_direction() -> torch.Tensor:
    direction = torch.zeros(HIDDEN)
    direction[0] = 1.0
    return direction


def _widened_direction(streams: int = STREAMS) -> torch.Tensor:
    stream_vectors = []
    for stream in range(streams):
        chunk = torch.zeros(HIDDEN)
        chunk[stream] = 1.0
        stream_vectors.append(chunk)
    return torch.cat(stream_vectors) / streams**0.5


def _folded_expectation(streams: int = STREAMS) -> torch.Tensor:
    folded = torch.zeros(HIDDEN)
    for stream in range(streams):
        folded[stream] = 1.0
    return folded / folded.norm()


def _widened_metadata(streams: int = STREAMS) -> dict[str, str]:
    return {
        "capture_space": CAPTURE_SPACE_WIDENED,
        "capture_width": str(HIDDEN * streams),
        "source_hidden_size": str(HIDDEN),
        "stream_count": str(streams),
    }


def test_geometry_id_is_deterministic_and_distinct() -> None:
    a = RefusalArtifact(direction=_residual_direction())
    b = RefusalArtifact(direction=_residual_direction())
    other = torch.zeros(HIDDEN)
    other[2] = 1.0
    c = RefusalArtifact(direction=other)
    assert a.metadata["geometry_id"] == b.metadata["geometry_id"]
    assert a.metadata["geometry_id"] != c.metadata["geometry_id"]


def test_validated_metadata_rejects_bad_geometry() -> None:
    with pytest.raises(ValueError, match="capture_width"):
        RefusalArtifact(
            direction=_residual_direction(),
            metadata={"capture_width": str(HIDDEN + 1)},
        )
    with pytest.raises(ValueError, match="source_hidden_size"):
        RefusalArtifact(
            direction=_widened_direction(),
            metadata={"stream_count": "4"},
        )
    with pytest.raises(ValueError, match="stream_count"):
        RefusalArtifact(
            direction=_widened_direction(),
            metadata={
                "source_hidden_size": str(HIDDEN),
                "stream_count": "3",
            },
        )
    with pytest.raises(ValueError, match="capture_space"):
        RefusalArtifact(
            direction=_residual_direction(),
            metadata={"capture_space": "wrong_space"},
        )


def test_load_artifact_infers_widened_and_refuses_install(tmp_path: Path) -> None:
    from safetensors import torch as safetensors_torch

    direction = _widened_direction()
    path = tmp_path / "widened.safetensors"
    safetensors_torch.save_file(
        {TENSOR_REFUSAL_DIRECTION: direction},
        path,
        metadata={"source_hidden_size": str(HIDDEN), "stream_count": str(STREAMS)},
    )
    artifact = load_artifact(str(path))
    assert artifact.metadata["capture_space"] == CAPTURE_SPACE_WIDENED
    with pytest.raises(ValueError, match="project_direction_to_residual"):
        install_refusal_ablation(nn.Linear(HIDDEN, HIDDEN), artifact, strength=0.0)


def test_project_direction_mean_provenance() -> None:
    artifact = RefusalArtifact(
        direction=_widened_direction(), metadata=_widened_metadata()
    )
    with pytest.raises(ValueError, match="capture_space"):
        validate_install(_TinyModel(), artifact)
    projected = project_direction_to_residual(artifact, hidden_size=HIDDEN, mode="mean")
    assert projected.metadata["capture_space"] == CAPTURE_SPACE_RESIDUAL
    assert projected.metadata["capture_width"] == str(HIDDEN)
    assert projected.metadata["source_hidden_size"] == str(HIDDEN)
    assert projected.metadata["stream_count"] == "1"
    assert projected.metadata["fold_mode"] == "mean"
    assert projected.metadata["folded_from_capture_width"] == str(HIDDEN * STREAMS)
    assert torch.allclose(projected.direction, _folded_expectation(), atol=1e-5)
    handle = install_refusal_ablation(_TinyModel(), projected, strength=1.0)
    assert handle.targets
    handle.remove()


def test_project_direction_first_and_lstsq_on_pure_stream() -> None:
    pure_stream = (
        torch.cat([_residual_direction() for _ in range(STREAMS)]) / STREAMS**0.5
    )
    artifact = RefusalArtifact(direction=pure_stream)
    first = project_direction_to_residual(artifact, hidden_size=HIDDEN, mode="first")
    assert torch.allclose(first.direction, _residual_direction(), atol=1e-5)
    lstsq = project_direction_to_residual(
        artifact, hidden_size=HIDDEN, mode="least_squares"
    )
    assert torch.allclose(lstsq.direction, _residual_direction(), atol=1e-3)
    projected = project_direction_to_residual(artifact, hidden_size=HIDDEN, mode="mean")
    assert torch.allclose(projected.direction, _residual_direction(), atol=1e-5)


def test_project_direction_rejects_bad_args() -> None:
    artifact = RefusalArtifact(direction=_widened_direction())
    with pytest.raises(ValueError, match="mode"):
        project_direction_to_residual(artifact, hidden_size=HIDDEN, mode="bogus")
    with pytest.raises(ValueError, match="hidden_size"):
        project_direction_to_residual(artifact, hidden_size=0)
    with pytest.raises(ValueError, match="multiple"):
        project_direction_to_residual(artifact, hidden_size=HIDDEN + 1)
    one_hot = torch.zeros(HIDDEN + 2)
    one_hot[0] = 1.0
    mismatched = RefusalArtifact(
        direction=torch.cat([one_hot for _ in range(STREAMS)]) / 2.0,
        metadata={"source_hidden_size": str(HIDDEN + 2), "stream_count": str(STREAMS)},
    )
    with pytest.raises(ValueError, match="hidden_size"):
        project_direction_to_residual(mismatched, hidden_size=HIDDEN)


def test_project_direction_rejects_first_for_declared_streams() -> None:
    artifact = RefusalArtifact(
        direction=_widened_direction(), metadata=_widened_metadata()
    )
    with pytest.raises(ValueError, match="first is rejected"):
        project_direction_to_residual(artifact, hidden_size=HIDDEN, mode="first")


def test_validate_install_fails_fast_on_width_mismatch() -> None:
    artifact = RefusalArtifact(direction=torch.zeros(HIDDEN * STREAMS))
    with pytest.raises(ValueError, match=f"{HIDDEN * STREAMS}"):
        validate_install(_TinyModel(), artifact)


def test_validate_install_refuses_widened_declared() -> None:
    artifact = RefusalArtifact(
        direction=_widened_direction(), metadata=_widened_metadata()
    )
    with pytest.raises(ValueError, match="capture_space"):
        validate_install(_TinyModel(), artifact)


class _TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.self_attn = nn.Module()
        self.self_attn.o_proj = nn.Linear(HIDDEN, HIDDEN)
        self.mlp = nn.Module()
        self.mlp.down_proj = nn.Linear(HIDDEN, HIDDEN)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.self_attn.o_proj(x) + self.mlp.down_proj(x)


class _Fused(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.gate_up = nn.Parameter(torch.empty(2 * width, width))


def test_writer_output_width_sources() -> None:
    assert writer_output_width(nn.Linear(HIDDEN, HIDDEN)) == HIDDEN
    fused = _Fused(HIDDEN)
    assert writer_output_width(fused) == 2 * HIDDEN
    assert writer_output_width(nn.Module()) is None


def test_find_residual_writers_accepts_qualified_and_glob() -> None:
    model = _TinyModel()
    qualified = find_residual_writers(model, ["self_attn.o_proj"])
    assert [name for name, _ in qualified] == ["self_attn.o_proj"]
    globbed = find_residual_writers(model, ["*.*_proj"])
    assert [name for name, _ in globbed] == ["mlp.down_proj", "self_attn.o_proj"]
    with pytest.raises(ValueError, match="fully-qualified"):
        find_residual_writers(model, ["mlp.*_proj_q"])


def test_find_residual_writers_auto_uses_architecture_map() -> None:
    qwen2_model = type("qwen2", (_TinyModel,), {})()
    writers = find_residual_writers(qwen2_model, ["auto"])
    assert [name for name, _ in writers] == ["mlp.down_proj", "self_attn.o_proj"]


def test_find_residual_writers_auto_requires_map() -> None:
    model = type("unknowndense", (_TinyModel,), {})()
    with pytest.raises(ValueError, match="targets='auto'"):
        find_residual_writers(model, ["auto"])


def test_find_residual_writers_auto_resolves_real_hf_class_name() -> None:
    # Real serving classes carry HF suffixes (Qwen4ExpForCausalLM) and the map
    # keys are underscored config names (qwen4_exp); auto must strip the suffix
    # and normalize so a real class resolves instead of raising.
    qwen4_exp_model = type("Qwen4ExpForCausalLM", (_TinyModel,), {})()
    writers = find_residual_writers(qwen4_exp_model, ["auto"])
    assert [name for name, _ in writers] == ["mlp.down_proj", "self_attn.o_proj"]


def test_find_residual_writers_auto_resolves_qwen2_causal_lm() -> None:
    qwen2_model = type("Qwen2ForCausalLM", (_TinyModel,), {})()
    writers = find_residual_writers(qwen2_model, ["auto"])
    assert [name for name, _ in writers] == ["mlp.down_proj", "self_attn.o_proj"]


def test_find_residual_writers_auto_unknown_arch_named_error() -> None:
    model = type("Qwen4ExpForCausalLM", (_TinyModel,), {})()
    # Temporarily remove qwen4_exp to confirm unknown arch still raises loudly.
    import model_headroom.refusal.runtime as rt

    original = dict(rt.ARCHITECTURE_TARGET_MAPS)
    rt.ARCHITECTURE_TARGET_MAPS = {
        k: v for k, v in original.items() if k != "qwen4_exp"
    }
    try:
        with pytest.raises(ValueError, match="targets='auto'"):
            find_residual_writers(model, ["auto"])
    finally:
        rt.ARCHITECTURE_TARGET_MAPS = original


@pytest.mark.parametrize(
    ("mode", "expected_fold"), [("mean", "mean"), ("least_squares", "least_squares")]
)
def test_projected_install_is_bit_exact_zero(mode: str, expected_fold: str) -> None:
    artifact = RefusalArtifact(
        direction=_widened_direction(), metadata=_widened_metadata()
    )
    projected = project_direction_to_residual(artifact, hidden_size=HIDDEN, mode=mode)
    assert projected.metadata["fold_mode"] == expected_fold
    model = _TinyModel()
    inputs = torch.randn(2, HIDDEN)
    before = model(inputs)
    handle = install_refusal_ablation(model, projected, strength=0.5)
    handle.strength_tensor.fill_(0.0)
    after = model(inputs)
    handle.remove()
    assert torch.equal(before, after)


def test_projected_geometry_id_differs_from_source() -> None:
    artifact = RefusalArtifact(
        direction=_widened_direction(), metadata=_widened_metadata()
    )
    projected = project_direction_to_residual(artifact, hidden_size=HIDDEN, mode="mean")
    assert projected.metadata["geometry_id"] != artifact.metadata["geometry_id"]


def test_lstsq_objective_matches_its_analytic_solution() -> None:
    torch.manual_seed(0)
    direction = torch.randn(HIDDEN * STREAMS)
    direction = direction / direction.norm()
    artifact = RefusalArtifact(direction=direction)
    mean = project_direction_to_residual(artifact, hidden_size=HIDDEN, mode="mean")
    lstsq = project_direction_to_residual(
        artifact, hidden_size=HIDDEN, mode="least_squares"
    )
    assert (mean.direction - lstsq.direction).norm() < 1e-5
    expected = direction.view(STREAMS, HIDDEN).mean(dim=0)
    expected = expected / expected.norm()
    assert (lstsq.direction - expected).norm() < 1e-5


def test_architecture_maps_cover_ticket_model_class() -> None:
    assert "qwen4_exp" in ARCHITECTURE_TARGET_MAPS
    assert "linear_attn" in ARCHITECTURE_TARGET_MAPS["qwen4_exp"]
