"""GGUF model weight extraction and metadata parsing.

Reads GGUF-format model files to extract architecture metadata and per-expert
weight tensors for model headroom analysis. All weights load to CPU RAM (NFR-01).

Dependencies: gguf (optional), torch, numpy, models.types (ExpertWeights).
Provides: load_gguf_config(), extract_expert_weight_from_gguf(),
    extract_expert_weights_for_layer_from_gguf(), ARCHITECTURE_TENSOR_NAMES,
    read_log_scales.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast

import numpy as np
import torch

from model_headroom.models.ternary_quants import (
    dequantize_ternary,
    prism_quant_size,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray


# Maps model type identifiers to GGUF tensor name templates for gate, up, and down
# expert weight projections. Each template uses {layer} and {expert} placeholders
# substituted at extraction time. Supported model types: mixtral (single expert naming:
# ffn_gate_exp), qwen2moe/qwen2_moe/phimoe/qwen3moe/qwen3_5_moe/qwen3_5_text (batch
# expert naming: ffn_gate_exps).
ARCHITECTURE_TENSOR_NAMES: dict[str, dict[str, str]] = {
    "mixtral": {
        "gate": "blk.{layer}.ffn_gate_exp.{expert}.weight",
        "up": "blk.{layer}.ffn_up_exp.{expert}.weight",
        "down": "blk.{layer}.ffn_down_exp.{expert}.weight",
    },
    "qwen2moe": {
        "gate": "blk.{layer}.ffn_gate_exps.{expert}.weight",
        "up": "blk.{layer}.ffn_up_exps.{expert}.weight",
        "down": "blk.{layer}.ffn_down_exps.{expert}.weight",
    },
    "qwen2_moe": {
        "gate": "blk.{layer}.ffn_gate_exps.{expert}.weight",
        "up": "blk.{layer}.ffn_up_exps.{expert}.weight",
        "down": "blk.{layer}.ffn_down_exps.{expert}.weight",
    },
    "phimoe": {
        "gate": "blk.{layer}.ffn_gate_exps.{expert}.weight",
        "up": "blk.{layer}.ffn_up_exps.{expert}.weight",
        "down": "blk.{layer}.ffn_down_exps.{expert}.weight",
    },
    "qwen3moe": {
        "gate": "blk.{layer}.ffn_gate_exps.{expert}.weight",
        "up": "blk.{layer}.ffn_up_exps.{expert}.weight",
        "down": "blk.{layer}.ffn_down_exps.{expert}.weight",
    },
    "qwen3_5_moe": {
        "gate": "blk.{layer}.ffn_gate_exps.{expert}.weight",
        "up": "blk.{layer}.ffn_up_exps.{expert}.weight",
        "down": "blk.{layer}.ffn_down_exps.{expert}.weight",
    },
    "qwen3_5_text": {
        "gate": "blk.{layer}.ffn_gate_exps.{expert}.weight",
        "up": "blk.{layer}.ffn_up_exps.{expert}.weight",
        "down": "blk.{layer}.ffn_down_exps.{expert}.weight",
    },
    "qwen35": {
        "gate": "blk.{layer}.ffn_gate_exps.{expert}.weight",
        "up": "blk.{layer}.ffn_up_exps.{expert}.weight",
        "down": "blk.{layer}.ffn_down_exps.{expert}.weight",
    },
}

# GGUF architectures known to be dense (single expert). When their expert_count
# metadata is absent, load_gguf_config defaults expert_count to 1 instead of
# raising missing_metadata.
_DENSE_ARCHITECTURES: frozenset[str] = frozenset({"qwen3_5_text", "qwen35"})


@dataclass(frozen=True)
class _GGUFModelConfig:
    """Module-private model configuration extracted from GGUF metadata.

    Attributes:
        architecture: Model architecture string (e.g. "mixtral", "qwen2moe",
            "qwen3moe", "qwen3_5_moe", "qwen3_5_text", "qwen2_moe").
            Must be a non-empty string matching a key in ARCHITECTURE_TENSOR_NAMES
            for full expert extraction support.
        expert_count: Number of experts per MoE layer. Must be > 0.
        block_count: Number of transformer layers. Must be > 0.
        intermediate_size: FFN intermediate dimension. Must be > 0.
        hidden_size: Hidden dimension size. Must be > 0.

    Invariants:
        - expert_count > 0
        - block_count > 0
        - intermediate_size > 0
        - hidden_size > 0

    Dependencies: None.
    """

    architecture: str
    expert_count: int
    block_count: int
    intermediate_size: int
    hidden_size: int


@dataclass(frozen=True)
class _GGUFLoadError(Exception):
    """Module-private error for GGUF loading failures.

    Attributes:
        error_type: Category of failure. One of "file_not_found",
            "invalid_gguf", "missing_metadata", "unsupported_architecture",
            "tensor_not_found".
        message: Human-readable description of the error.
        recoverable: True if retrying with a different file may succeed.

    Invariants:
        - error_type is one of the five allowed values.
        - message is non-empty.

    Dependencies: None.
    """

    error_type: Literal[
        "file_not_found",
        "invalid_gguf",
        "missing_metadata",
        "unsupported_architecture",
        "tensor_not_found",
    ]
    message: str
    recoverable: bool


def _read_metadata_string(field: object) -> str:
    """Extract a string value from a GGUF ReaderField.

    Input: field: A ReaderField from GGUFReader.fields. Must have contents()
        that returns a string.
    Output: The extracted string value.
    Errors: Raises _GGUFLoadError if field is None or contents() fails.

    Dependencies: gguf.gguf_reader.ReaderField.
    """
    if field is None:
        raise _GGUFLoadError(
            error_type="missing_metadata",
            message="Field is None",
            recoverable=False,
        )
    contents_fn = getattr(field, "contents", None)
    if contents_fn is None:
        raise _GGUFLoadError(
            error_type="missing_metadata",
            message="Field has no contents method",
            recoverable=False,
        )
    value = contents_fn()
    if value is None:
        raise _GGUFLoadError(
            error_type="missing_metadata",
            message="Field contents() returned None",
            recoverable=False,
        )
    return str(value)


def _read_metadata_int(field: object) -> int:
    """Extract an integer value from a GGUF ReaderField.

    Input: field: A ReaderField from GGUFReader.fields. Must have contents()
        that returns an int-compatible value.
    Output: The extracted integer value.
    Errors: Raises _GGUFLoadError if field is None, contents() fails,
        or the value is not an integer.

    Dependencies: gguf.gguf_reader.ReaderField.
    """
    if field is None:
        raise _GGUFLoadError(
            error_type="missing_metadata",
            message="Field is None",
            recoverable=False,
        )
    contents_fn = getattr(field, "contents", None)
    if contents_fn is None:
        raise _GGUFLoadError(
            error_type="missing_metadata",
            message="Field has no contents method",
            recoverable=False,
        )
    value = contents_fn()
    if value is None:
        raise _GGUFLoadError(
            error_type="missing_metadata",
            message="Field contents() returned None",
            recoverable=False,
        )
    return int(value)


def load_gguf_config(gguf_path: Path) -> _GGUFModelConfig:
    """Read GGUF metadata and return model configuration.

    Input:
        gguf_path: Path to a .gguf file. Must exist and be readable.
    Output (success):
        _GGUFModelConfig with architecture, expert_count, block_count,
        intermediate_size, hidden_size all populated from metadata.
    Output (error):
        Raises _GGUFLoadError with error_type in: "file_not_found",
        "invalid_gguf", "missing_metadata".
    Invariants:
        - expert_count > 0, block_count > 0, intermediate_size > 0,
          hidden_size > 0 on success.
    Dependencies: Requires gguf package. Reads file at gguf_path.
    """
    from gguf import GGUFReader

    if not gguf_path.exists():
        raise _GGUFLoadError(
            error_type="file_not_found",
            message=f"GGUF file not found: {gguf_path}",
            recoverable=True,
        )

    try:
        reader = GGUFReader(str(gguf_path))
    except Exception as e:
        raise _GGUFLoadError(
            error_type="invalid_gguf",
            message=f"Failed to read GGUF file: {e}",
            recoverable=False,
        ) from e

    arch = _read_metadata_string(reader.fields.get("general.architecture"))

    expert_count_field = reader.fields.get(f"{arch}.expert_count")
    if expert_count_field is None and arch in _DENSE_ARCHITECTURES:
        expert_count = 1
    else:
        expert_count = _read_metadata_int(expert_count_field)
    block_count = _read_metadata_int(reader.fields.get(f"{arch}.block_count"))
    intermediate_size = _read_metadata_int(
        reader.fields.get(f"{arch}.feed_forward_length")
    )
    hidden_size = _read_metadata_int(reader.fields.get(f"{arch}.embedding_length"))

    if (
        expert_count <= 0
        or block_count <= 0
        or intermediate_size <= 0
        or hidden_size <= 0
    ):
        raise _GGUFLoadError(
            error_type="missing_metadata",
            message=(
                f"Invalid metadata values: expert_count={expert_count}, "
                f"block_count={block_count}, intermediate_size={intermediate_size}, "
                f"hidden_size={hidden_size}"
            ),
            recoverable=False,
        )

    return _GGUFModelConfig(
        architecture=arch,
        expert_count=expert_count,
        block_count=block_count,
        intermediate_size=intermediate_size,
        hidden_size=hidden_size,
    )


def read_log_scales(gguf_path: Path) -> list[float] | None:
    """Read log_scales quantization group scales from GGUF metadata.

    log_scales holds per-group dequantization scale factors for quantized
    tensors. In GGUF this is stored as a companion tensor (not a KV metadata
    field), so this reads tensor data rather than metadata fields.

    Input:
        gguf_path: Path to a .gguf file. Must exist and be readable.
    Output:
        List of float scale values, or None if log_scales tensor is not
        present or cannot be read.
    Dependencies: Requires gguf and torch packages. Reads file at gguf_path.
    """
    from gguf import GGUFReader

    if not gguf_path.exists():
        return None

    try:
        reader = GGUFReader(str(gguf_path))
    except Exception:
        return None

    arch_field = reader.fields.get("general.architecture")
    arch = None
    if arch_field is not None:
        try:
            arch = _read_metadata_string(arch_field)
        except _GGUFLoadError:
            arch = None

    # Build candidate tensor names: per-architecture first, then global.
    candidate_names: list[str] = []
    if arch is not None:
        candidate_names.append(f"{arch}.log_scales")
    candidate_names.append("log_scales")

    for tensor in reader.tensors:
        if tensor.name not in candidate_names:
            continue
        qtype_name = _tensor_qtype_name(tensor)
        if qtype_name is not None and qtype_name != "F32":
            continue
        raw = getattr(tensor, "data", None)
        if raw is None:
            continue
        try:
            data = np.ascontiguousarray(raw).reshape(-1)
            floats = data.view(np.float32)
        except (ValueError, TypeError):
            continue
        return [float(v) for v in floats]

    return None


def extract_expert_weight_from_gguf(
    gguf_path: Path,
    tensor_name: str,
    device: str = "cpu",
) -> torch.Tensor | None:
    """Extract a single expert weight tensor from a GGUF file.

    Input:
        gguf_path: Path to a .gguf file. Must exist and be readable.
        tensor_name: Full tensor name (e.g. "blk.0.ffn_gate_exp.0.weight").
        device: Device for the output tensor. Default "cpu" for VRAM safety.
    Output (success):
        2D torch.Tensor, dtype=float32, on the requested device.
    Output (not found):
        Returns None if tensor_name does not exist in the GGUF file.
    Invariants:
        - Output tensor is 2D (ndim == 2).
        - Output tensor dtype is float32.
        - Output tensor is on the requested device.
    Dependencies: Requires gguf and torch packages. Reads file at gguf_path.
    """
    from gguf import GGUFReader

    if not gguf_path.exists():
        return None

    try:
        reader = GGUFReader(str(gguf_path))
    except Exception:
        return None

    for tensor in reader.tensors:
        if tensor.name == tensor_name:
            return _dequantize_gguf_tensor(tensor, device=device)

    return None


def _tensor_qtype_name(tensor: object) -> str | None:
    """Best-effort name of a tensor's quantization type.

    The installed gguf package may not know private Prism types (PQ2_0, PTQ1_0,
    Q2_0), so fall back to the raw enum / string / int representation.
    """
    qtype = getattr(tensor, "tensor_type", None)
    if qtype is None:
        return None
    name = getattr(qtype, "name", None)
    if name is not None:
        return str(name)
    return str(qtype)


def _validate_tensor_shape(shape: object) -> tuple[int, int]:
    """Validate a GGUF tensor shape is 2-D [rows, cols] and return it."""
    if shape is None:
        raise ValueError("GGUF tensor has no shape")
    shape_seq = cast("tuple[int, int] | list[int]", shape)
    try:
        rows = int(shape_seq[0])
        cols = int(shape_seq[1])
    except (TypeError, IndexError, ValueError) as e:
        raise ValueError(f"GGUF tensor shape is not 2-D [rows, cols]: {shape}") from e
    if rows <= 0 or cols <= 0:
        raise ValueError(f"GGUF tensor shape has non-positive dims: {shape}")
    return rows, cols


def _validate_quant_buffer(
    data: NDArray[np.uint8],
    qtype_name: str,
    block_size: int,
    type_size: int,
    rows: int,
    cols: int,
) -> None:
    """Validate block lengths and buffer contiguity before native code entry.

    Raises ValueError (clear error, never a segfault) on malformed input.
    """
    if data.ndim != 1:
        raise ValueError(
            f"Expected 1-D quantized byte buffer for {qtype_name}, got ndim={data.ndim}"
        )
    if not data.flags.c_contiguous:
        raise ValueError(
            f"{qtype_name} byte buffer is not contiguous before native dequantization"
        )
    if data.size % type_size != 0:
        raise ValueError(
            f"{qtype_name} buffer size {data.size} is not a multiple of type_size "
            f"{type_size}"
        )
    if cols % block_size != 0:
        raise ValueError(
            f"{qtype_name} tensor cols {cols} is not a multiple of block_size "
            f"{block_size}"
        )
    expected_blocks = rows * (cols // block_size)
    if data.size // type_size != expected_blocks:
        raise ValueError(
            f"{qtype_name} buffer implies {data.size // type_size} blocks but "
            f"shape [{rows}, {cols}] implies {expected_blocks} blocks"
        )


def _dequantize_gguf_tensor(
    tensor: object,
    device: str = "cpu",
) -> torch.Tensor | None:
    """Dequantize a single GGUF tensor into a 2-D float32 torch.Tensor.

    Reads one tensor at a time (never the full model). Dispatches PQ2_0/PTQ1_0 to the
    ternary adapters (Prism Q2_0/TQ1_0 codec logic at group-128), Q2_0/TQ1_0/TQ2_0 to
    the installed gguf dequantize, and F32 to a direct float32 view. Preserves GGUF
    [rows, cols] orientation (row-major, no transpose).
    """
    from gguf import GGMLQuantizationType

    try:
        rows, cols = _validate_tensor_shape(getattr(tensor, "shape", None))
    except ValueError:
        return None

    qtype_name = _tensor_qtype_name(tensor)
    if qtype_name is None:
        return None
    raw = getattr(tensor, "data", None)
    if raw is None:
        return None

    try:
        data: NDArray[np.uint8] = np.ascontiguousarray(raw).reshape(-1)
    except (ValueError, TypeError):
        return None

    if qtype_name == "F32":
        try:
            raw_data = data.view(np.float32)
        except (TypeError, ValueError):
            return None
        copied = raw_data.copy()
        result = torch.from_numpy(copied).float().to(device)
        if result.ndim == 1:
            result = result.view(rows, cols)
        return result

    prism_size = prism_quant_size(qtype_name)

    if prism_size is not None:
        block_size, type_size = prism_size
        try:
            _validate_quant_buffer(data, qtype_name, block_size, type_size, rows, cols)
        except ValueError:
            return None
        blocks = data.reshape((-1, type_size))
        try:
            raw_data = dequantize_ternary(blocks, qtype_name).reshape(-1)
        except ValueError:
            return None
        copied = raw_data.copy()
        result = torch.from_numpy(copied).float().to(device)
        if result.ndim == 1:
            result = result.view(rows, cols)
        return result

    # Non-ternary quantized types (e.g. Q4_0, Q8_0): installed gguf dequantize.
    if qtype_name in (
        "TQ1_0",
        "TQ2_0",
        "Q4_0",
        "Q4_1",
        "Q5_0",
        "Q5_1",
        "Q8_0",
        "Q2_K",
        "Q3_K",
        "Q4_K",
        "Q5_K",
        "Q6_K",
        "Q8_K",
        "F16",
        "BF16",
    ):
        from gguf import dequantize as gguf_dequantize

        try:
            gguf_qtype = getattr(GGMLQuantizationType, qtype_name)
        except AttributeError:
            return None
        try:
            raw_data = gguf_dequantize(data, gguf_qtype)
        except (NotImplementedError, ValueError, TypeError):
            return None
        copied = np.ascontiguousarray(raw_data).copy()
        result = torch.from_numpy(copied).float().to(device)
        if result.ndim == 1:
            result = result.view(rows, cols)
        return result

    return None


def extract_gguf_weight_sample(
    gguf_path: str | Path,
    tensor_name: str,
    max_rows: int = 256,
    seed: int = 42,
) -> torch.Tensor | None:
    """Bounded deterministic row sampling of a single GGUF weight tensor.

    Reads one tensor at a time, dequantizes it, samples deterministic rows, and returns
    a 2-D float32 torch.Tensor. Never loads the full model into CPU RAM.

    Input:
        gguf_path: Path to a .gguf file.
        tensor_name: Full tensor name (e.g. "blk.0.ffn_gate_exps.0.weight").
        max_rows: Upper bound on sampled rows (default 256).
        seed: Deterministic RNG seed (same seed → same rows).
    Output (success):
        2-D torch.Tensor, dtype=float32, shape [sampled_rows, cols].
    Output (not found / error): None.
    Invariants:
        - Output is 2-D float32 on CPU.
        - Rows are block-aligned for quantized tensors.
        - Same (gguf_path, tensor_name, max_rows, seed) → same rows.
    Dependencies: Requires gguf and torch. Reads file at gguf_path.
    """
    from gguf import GGUFReader

    if not Path(gguf_path).exists():
        return None

    try:
        reader = GGUFReader(str(gguf_path))
    except Exception:
        return None

    target = None
    for tensor in reader.tensors:
        if tensor.name == tensor_name:
            target = tensor
            break
    if target is None:
        return None

    from gguf import GGMLQuantizationType

    try:
        rows, cols = _validate_tensor_shape(getattr(target, "shape", None))
    except ValueError:
        return None

    qtype_name = _tensor_qtype_name(target)
    if qtype_name is None:
        return None
    raw = getattr(target, "data", None)
    if raw is None:
        return None

    try:
        data: NDArray[np.uint8] = np.ascontiguousarray(raw).reshape(-1)
    except (ValueError, TypeError):
        return None

    prism_size = prism_quant_size(qtype_name)

    block_size = 1
    type_size = None
    if prism_size is not None:
        block_size, type_size = prism_size
        try:
            _validate_quant_buffer(data, qtype_name, block_size, type_size, rows, cols)
        except ValueError:
            return None
        blocks = data.reshape((-1, type_size))
        try:
            flat = dequantize_ternary(blocks, qtype_name).reshape(-1)
        except ValueError:
            return None
    else:
        # Non-ternary quantized or F32: use installed gguf or direct view.
        if qtype_name == "F32":
            try:
                flat = data.view(np.float32)
            except (TypeError, ValueError):
                return None
        else:
            from gguf import dequantize as gguf_dequantize

            try:
                gguf_qtype = getattr(GGMLQuantizationType, qtype_name)
            except AttributeError:
                return None
            try:
                flat = gguf_dequantize(data, gguf_qtype)
            except (NotImplementedError, ValueError, TypeError):
                return None

    copied = np.ascontiguousarray(flat).copy()
    weight = torch.from_numpy(copied).float().view(rows, cols)

    n_rows = weight.shape[0]
    if n_rows == 0:
        return None

    rng = np.random.default_rng(seed)
    if block_size > 1:
        candidates = np.arange(0, n_rows, block_size, dtype=np.int64)
    else:
        candidates = np.arange(n_rows, dtype=np.int64)

    n_take = min(int(max_rows), int(candidates.shape[0]))
    if n_take <= 0:
        return None

    chosen = rng.choice(candidates, size=n_take, replace=False)
    chosen = np.sort(chosen)
    sample = weight[chosen]
    return sample.float()
