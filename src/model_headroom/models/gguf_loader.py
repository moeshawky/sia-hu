"""GGUF model weight extraction and metadata parsing.

Reads GGUF-format model files to extract architecture metadata and per-expert
weight tensors for model headroom analysis. All weights load to CPU RAM (NFR-01).

Dependencies: gguf (optional), torch, numpy, models.types (ExpertWeights).
Provides: load_gguf_config(), extract_expert_weight_from_gguf(),
    read_log_scales.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Protocol, cast

import numpy as np
import torch
from gguf.constants import (
    GGUF_DEFAULT_ALIGNMENT,
    GGUF_MAGIC,
    GGMLQuantizationType,
    GGUFValueType,
)

from model_headroom.models.ternary_quants import (
    _PRISM_QUANT_SIZES,
    dequantize_ternary,
    prism_quant_size,
)

ByteOrder = Literal["<", ">"]


class _TensorLike(Protocol):
    name: str
    tensor_type: object
    shape: object
    data: object


class _ReaderSurface(Protocol):
    fields: dict[str, object]
    tensors: list[_TensorLike]


if TYPE_CHECKING:
    from numpy.typing import NDArray


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

    def __str__(self) -> str:
        return self.message


class _PrismQType:
    def __init__(self, name: str) -> None:
        self.name = name


class _FallbackField:
    def __init__(self, name: str, value: object) -> None:
        self.name = name
        self._value = value

    def contents(self, index_or_slice: object = slice(None)) -> object:
        return self._value


_GGUF_SCALAR_TO_NP: dict[GGUFValueType, type[np.generic]] = {
    GGUFValueType.UINT8: np.uint8,
    GGUFValueType.INT8: np.int8,
    GGUFValueType.UINT16: np.uint16,
    GGUFValueType.INT16: np.int16,
    GGUFValueType.UINT32: np.uint32,
    GGUFValueType.INT32: np.int32,
    GGUFValueType.UINT64: np.uint64,
    GGUFValueType.INT64: np.int64,
    GGUFValueType.FLOAT32: np.float32,
    GGUFValueType.BOOL: np.bool_,
}


def _gguf_shape_from_dims(dims: np.ndarray) -> tuple[int, ...]:
    return tuple(int(d) for d in reversed(list(dims)))


class _GGUFFallbackReader:
    def __init__(self, path: str | Path) -> None:
        self.data = np.memmap(str(path), mode="r", dtype=np.uint8)
        self.byte_order = "I"
        self.alignment = GGUF_DEFAULT_ALIGNMENT
        offs = 0
        magic = int(self._get(offs, np.uint32, override_order="<")[0])
        if magic != GGUF_MAGIC:
            raise ValueError("GGUF magic invalid")
        offs += 4
        version_arr = self._get(offs, np.uint32)
        offs += 4
        if version_arr[0] & 65535 == 0:
            self.byte_order = "S"
            version_arr = version_arr.view(version_arr.dtype.newbyteorder("S"))
        version = int(version_arr[0])
        if version not in (2, 3):
            raise ValueError(f"Unsupported GGUF version {version}")
        order: ByteOrder = "<" if self.byte_order == "I" else ">"
        counts = self._get(offs, np.uint64, 2, override_order=order)
        offs += 16
        tensor_count = int(counts[0])
        kv_count = int(counts[1])
        self.fields: OrderedDict[str, _FallbackField] = OrderedDict()
        offs = self._build_fields(offs, kv_count, order)
        offs, tensor_fields = self._build_tensor_info(offs, tensor_count, order)
        align_field = self.fields.get("general.alignment")
        if align_field is not None:
            self.alignment = int(cast("int", align_field.contents()))
        padding = offs % self.alignment
        if padding != 0:
            offs += self.alignment - padding
        self.data_offset = offs
        self._build_tensors(offs, tensor_fields)
        self._resolve_types()

    def _get(
        self,
        offs: int,
        dtype: np.dtype[np.generic] | type[np.generic],
        count: int = 1,
        override_order: ByteOrder | None = None,
    ) -> np.ndarray:
        order: ByteOrder = (
            override_order
            if override_order is not None
            else ("<" if self.byte_order == "I" else ">")
        )
        dt = np.dtype(dtype).newbyteorder(order)
        itemsize = int(dt.itemsize)
        return np.frombuffer(self.data[offs : offs + itemsize * count], dtype=dt)[
            :count
        ]

    def _get_str(self, offs: int, order: ByteOrder) -> tuple[int, bytes, int]:
        slen = int(self._get(offs, np.uint64, override_order=order)[0])
        offs += 8
        raw = self.data[offs : offs + slen].tobytes()
        offs += slen
        return slen, raw, offs

    def _read_field_value(
        self, offs: int, gtype: GGUFValueType, order: ByteOrder
    ) -> tuple[object, int]:
        if gtype == GGUFValueType.STRING:
            _, raw, offs = self._get_str(offs, order)
            return str(raw, encoding="utf-8"), offs
        nptype = _GGUF_SCALAR_TO_NP.get(gtype)
        if nptype is not None:
            val = self._get(offs, nptype, override_order=order)
            offs += int(val.nbytes)
            item = val.tolist()[0]
            if gtype == GGUFValueType.BOOL:
                return bool(item), offs
            if gtype == GGUFValueType.FLOAT32:
                return float(item), offs
            return int(item), offs
        if gtype == GGUFValueType.ARRAY:
            raw_itype = self._get(offs, np.uint32, override_order=order)
            offs += 4
            alen = self._get(offs, np.uint64, override_order=order)
            offs += 8
            sub = GGUFValueType(int(raw_itype[0]))
            vals = []
            for _ in range(int(alen[0])):
                v, offs = self._read_field_value(offs, sub, order)
                vals.append(v)
            return vals, offs
        raise ValueError(f"Unhandled field type {gtype}")

    def _build_fields(self, offs: int, count: int, order: ByteOrder) -> int:
        for _ in range(count):
            _, kdata, offs = self._get_str(offs, order)
            name = str(kdata, encoding="utf-8")
            raw_type = self._get(offs, np.uint32, override_order=order)
            offs += 4
            gtype = GGUFValueType(int(raw_type[0]))
            value, offs = self._read_field_value(offs, gtype, order)
            self.fields[name] = _FallbackField(name, value)
        return offs

    def _get_tensor_info_field(
        self, offs: int, order: ByteOrder
    ) -> tuple[int, bytes, np.ndarray, np.ndarray, np.ndarray, int]:
        orig_offs = offs
        _, name_data, offs = self._get_str(offs, order)
        n_dims = int(self._get(offs, np.uint32, override_order=order)[0])
        offs += 4
        dims = self._get(offs, np.uint64, n_dims, override_order=order)
        offs += int(dims.nbytes)
        raw_dtype = self._get(offs, np.uint32, override_order=order)
        offs += 4
        offset_tensor = self._get(offs, np.uint64, override_order=order)
        offs += 8
        return (
            orig_offs,
            name_data,
            dims,
            raw_dtype,
            offset_tensor,
            offs,
        )

    def _build_tensor_info(
        self, offs: int, count: int, order: ByteOrder
    ) -> tuple[int, list[tuple[bytes, np.ndarray, np.ndarray, np.ndarray]]]:
        fields = []
        for _ in range(count):
            _, name_data, dims, raw_dtype, offset_tensor, offs = (
                self._get_tensor_info_field(offs, order)
            )
            fields.append((name_data, dims, raw_dtype, offset_tensor))
        return offs, fields

    def _build_tensors(
        self,
        start_offs: int,
        fields: list[tuple[bytes, np.ndarray, np.ndarray, np.ndarray]],
    ) -> None:
        self.tensors = []
        tensor_names = set()
        file_size = int(self.data.shape[0])
        sorted_fields = sorted(enumerate(fields), key=lambda t: int(t[1][3][0]))
        for i, (name_data, dims, raw_dtype, offset_tensor) in sorted_fields:
            name = str(name_data, encoding="utf-8")
            if name in tensor_names:
                raise ValueError(f"Duplicated tensor {name}")
            tensor_names.add(name)
            type_id = int(raw_dtype[0])
            shape = _gguf_shape_from_dims(dims)
            offset = int(offset_tensor[0])
            next_offset = (
                sorted_fields[i + 1][1][3][0]
                if i + 1 < len(sorted_fields)
                else file_size - start_offs
            )
            next_offset = int(next_offset)
            byte_length = next_offset - offset
            self.tensors.append(
                _FallbackTensor(
                    name=name,
                    shape=shape,
                    type_id=type_id,
                    offset=offset,
                    byte_length=byte_length,
                    reader=self,
                )
            )

    def _resolve_types(self) -> None:
        known_ids = {int(t) for t in GGMLQuantizationType}
        unknown = [t for t in self.tensors if t.type_id not in known_ids]
        if not unknown:
            for t in self.tensors:
                t.tensor_type = GGMLQuantizationType(t.type_id)
            return
        matches = []
        for qname, (block_size, type_size) in _PRISM_QUANT_SIZES.items():
            ok = True
            for t in unknown:
                rows, cols = int(t.shape[0]), int(t.shape[1])
                n_elems = rows * cols
                if n_elems * type_size // block_size != t.byte_length:
                    ok = False
                    break
            if ok:
                matches.append(qname)
        if len(matches) != 1:
            bits = None
            if unknown:
                t = unknown[0]
                rows, cols = int(t.shape[0]), int(t.shape[1])
                bits = t.byte_length * 8 / (rows * cols)
            raise _GGUFLoadError(
                error_type="invalid_gguf",
                message=(
                    f"Ambiguous Prism quant type for {len(unknown)} unknown-type "
                    f"tensors; matches={matches}; bits/weight={bits}"
                ),
                recoverable=False,
            )
        qname = matches[0]
        prism_type = _PrismQType(qname)
        for t in self.tensors:
            if t.type_id in known_ids:
                t.tensor_type = GGMLQuantizationType(t.type_id)
            else:
                t.tensor_type = prism_type


class _FallbackTensor:
    def __init__(
        self,
        name: str,
        shape: tuple[int, ...] | np.ndarray,
        type_id: int,
        offset: int,
        byte_length: int,
        reader: _GGUFFallbackReader,
    ) -> None:
        self.name = name
        self.shape = shape
        self.type_id = type_id
        self.offset = offset
        self.byte_length = byte_length
        self._reader = reader
        self.tensor_type: GGMLQuantizationType | _PrismQType | None = None
        self._data: np.ndarray | None = None

    @property
    def data(self) -> np.ndarray:
        if self._data is None:
            start = self._reader.data_offset + self.offset
            self._data = self._reader.data[start : start + self.byte_length]
        return self._data


def open_gguf_reader(path: str | Path) -> _ReaderSurface:
    from gguf import GGUFReader

    try:
        return cast("_ReaderSurface", GGUFReader(str(path)))
    except ValueError as e:
        if "is not a valid GGMLQuantizationType" in str(e):
            return cast("_ReaderSurface", _GGUFFallbackReader(str(path)))
        raise _GGUFLoadError(
            error_type="invalid_gguf",
            message=f"Failed to read GGUF file: {e}",
            recoverable=False,
        ) from e
    except Exception as e:
        raise _GGUFLoadError(
            error_type="invalid_gguf",
            message=f"Failed to read GGUF file: {e}",
            recoverable=False,
        ) from e


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
    if not gguf_path.exists():
        raise _GGUFLoadError(
            error_type="file_not_found",
            message=f"GGUF file not found: {gguf_path}",
            recoverable=True,
        )

    reader = open_gguf_reader(str(gguf_path))

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
    if not gguf_path.exists():
        return None

    try:
        reader = open_gguf_reader(str(gguf_path))
    except _GGUFLoadError:
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
        - Sampled rows are whole rows; block alignment is a property of the column axis.
        - Same (gguf_path, tensor_name, max_rows, seed) → same rows.
    Dependencies: Requires gguf and torch. Reads file at gguf_path.
    """
    if not Path(gguf_path).exists():
        return None

    try:
        reader = open_gguf_reader(str(gguf_path))
    except _GGUFLoadError:
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
    candidates = np.arange(n_rows, dtype=np.int64)

    n_take = min(int(max_rows), int(candidates.shape[0]))
    if n_take <= 0:
        return None

    chosen = rng.choice(candidates, size=n_take, replace=False)
    chosen = np.sort(chosen)
    sample = weight[chosen]
    return sample.float()
