"""
Headroom Capture Sidecar

Depends on capture_proxy.names for the shared name contract (signal vocabulary,
      tensor-name templates).
Owns: Headroom sidecar files under {base_path}/headroom/YYYY-MM-DD/ — one
       *.headroom.json per sampled capture plus one *weight_pairs.json when
       weight capture is enabled and one *expert_weights.json when the
       specialization signal is enabled; per-capture float32 byte cap shared
       across both tensor artifacts; 1/N sampling
Depends: std (json, pathlib, datetime, importlib, asyncio, contextlib,
      collections.abc); torch + safetensors are imported lazily inside the
      default tensor loader only (missing libraries degrade to None)
Provides: HeadroomWriter, HeadroomLogger, TensorBundle, TensorLoader,
       parse_byte_cap, build_activation_payload,
       RoutingCounts, RoutingCountSource

Invariants:
  - off mode: no files written, no torch/safetensors import (no writer built)
  - write_async never raises; returns None when nothing was written
  - device="cpu" only (NFR-01) — tensors never touch TPU/GPU memory
  - weight-pair files keep the model_headroom loader contract: every pair
    carries gradient, weight, and layer_name (zero-gradient arrays when no
    training hook exists), plus parent_shape, truncated, and sampling_mode on
    every pair whether or not the byte cap fired, so a capped tile is never
    indistinguishable downstream from a whole matrix
  - every pair also carries expert_id: the expert index parsed from the
    resolved tensor name (experts.N), the per-slice index for fused 3-D
    expert tensors, or None for dense/non-expert tensors — so a reader can
    distinguish 1-of-N sampling from 1-of-1 (DEFER-4). The authority's
    loader requires only the gradient/weight/layer_name keys and reads the
    rest via .get(), so the additive key is contract-safe.
  - the sidecar (*.headroom.json) is written after the weight-pair file and
    is the authoritative artifact; the CaptureRecord ref path is precomputed
  - task label is captured in the sidecar when CAPTURE_MODEL_TASKS maps the
    model to a task; None otherwise (classifier/image and chat/completion
    requests are all eligible)
"""

from __future__ import annotations

import asyncio
import contextlib
import importlib
import json
import re
import sys
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, NamedTuple

from capture_proxy.models import CapturedHeadroom

# Names, names-only: the shared name contract (signal vocabulary, CLI-to-emitted
# map, tensor-name templates, MoE model types) lives in capture_proxy.names,
# which imports nothing heavy. The constants this module previously declared a
# second time now have exactly one definition; the two vocabularies are linked
# by a total map instead of drifting apart.
from capture_proxy.names import (
    ARCHITECTURE_TENSOR_NAMES,
    MOE_MODEL_TYPES,
    SIGNAL_KEYS,
)

_BYTE_SUFFIXES = {"K": 1024, "M": 1024**2, "G": 1024**3}

# MVP captures layer 0 per enabled projection (design doc §1 R-04 math:
# one gate_proj of 5120x17408 is already ~356MB float32).
_CAPTURE_LAYER = 0

# Suffix of the expert-activation artifact written beside the sidecar. The
# payload is the model_headroom `load_activations_from_json` contract
# (models/loader.py:316-334): top-level "metadata" and "layers", with
# layers mapping "<layer_idx>" -> "<expert_id>" -> token count.
ACTIVATION_ARTIFACT_SUFFIX = ".activations.json"

ACTIVATION_FILENAME = "{capture_id}" + ACTIVATION_ARTIFACT_SUFFIX

# Suffix of the expert-weights artifact written beside the sidecar. The
# payload is the model_headroom `load_expert_weights_from_json` contract
# (models/loader.py): top-level "layers", each with "layer_name" and
# "expert_weights" (list of >= 2 same-shape 2-D weight arrays). The
# authority scores specialization; the carrier only emits.
EXPERT_WEIGHTS_SUFFIX = "-expert_weights.json"
EXPERT_WEIGHTS_FILENAME = "{capture_id}" + EXPERT_WEIGHTS_SUFFIX


def _capture_projs(
    architecture: str, weights: Literal["off", "gate", "all"]
) -> tuple[tuple[str, ...], int]:
    """Select projection keys and the fallback boundary for an architecture.

    Shared by the weight-pairs and expert-weights capture paths so both
    resolve the same tensors: "moe" takes gate (or gate+up+down);
    dense/unknown takes BERT-style projections first with MoE as fallback.

    Args:
        architecture: "moe", "dense", or "unknown" from _detect_architecture.
        weights: Which projections to snapshot — "off", "gate", or "all".

    Returns:
        (projs, fallback_start): Projection keys in resolution order and
            the index where fallback keys begin (names before it that miss
            are silent; fallback misses are reported).

    Raises: None.
    """
    if architecture == "moe":
        projs: tuple[str, ...] = (
            ("gate",) if weights == "gate" else ("gate", "up", "down")
        )
        return projs, len(projs)  # no fallback: all are primary
    # dense/unknown: BERT-style projections first, MoE as fallback
    primary = (
        ("attention",) if weights == "gate" else ("attention", "intermediate", "output")
    )
    fallback = ("gate",) if weights == "gate" else ("gate", "up", "down")
    return primary + fallback, len(primary)


class TensorBundle(NamedTuple):
    """A weight tensor and its (zero) gradient as float32 nested lists.

    Attributes:
        weight: 2-D float list (rows x cols), dequantized to float32.
        gradient: 2-D float list, same shape as weight; zeros when no
            training hook produced a real gradient.
        parent_shape: Shape of the tensor as read from the checkpoint, before
            any byte-cap row slice. None when the loader reports no shape.
        truncated: True when the byte cap actually cut rows from this tensor.
    """

    weight: list[list[float]]
    gradient: list[list[float]]
    parent_shape: tuple[int, int] | None = None
    truncated: bool = False


# (model_path, tensor_name) -> TensorBundle | None. Implementations must
# never raise; None means "tensor unavailable".
TensorLoader = Callable[[Path, str], TensorBundle | None]

# Observed router selections: layer index -> expert id -> token count.
RoutingCounts = Mapping[int, Mapping[str, int]]

# (model_path, num_experts) -> RoutingCounts | None. Same seam shape and
# never-raise contract as TensorLoader. The carrier captures weights, not
# router activations, so no source is installed by default and the
# load_balance artifact carries metadata only until one exists.
RoutingCountSource = Callable[[Path, int], RoutingCounts | None]


def _expert_id_key(raw: str) -> int | None:
    """Coerce a routing-count expert key to an int, or None when it is not one."""
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def build_activation_payload(
    *,
    num_experts_architectural: int | None,
    counts: RoutingCounts | None,
) -> dict[str, Any] | None:
    """Build the model_headroom expert-activation payload for one capture.

    ``metadata["num_experts"]`` is the FED expert count, never the
    architectural one, and ``metadata["num_experts_architectural"]`` is
    the count from config.json. The authority reads both and normalizes
    entropy by ``log(num_experts_architectural)`` when it is present
    (``signals/load_balance.py``), because the denominator of R-06 is
    the population the MODEL has. Emitting the fed count alone would
    leave that population unstated, and a file carrying only a fed count
    of 1 is ambiguous to the authority between a dense model and a
    collapsed router; the authority reports that as
    ``expert_count_unknown`` rather than guessing. A never-fed expert is
    therefore excluded from ``layers`` entirely and named in
    ``metadata["starved_experts"]``; it is never counted as dead. A layer
    whose experts were all unfed is excluded the same way and named in
    ``metadata["omitted_layers"]``, because the authority scores an
    all-zero layer as entropy 0.0 (``signals/load_balance.py``) and
    would read an unobserved layer as maximal imbalance.

    The distinction stays visible in the artifact rather than being folded
    into one integer: ``num_experts_architectural``, ``num_experts_fed``
    and ``starved_experts`` let a consumer recover either framing.

    Args:
        num_experts_architectural: Expert count from config.json, as read
            by ``_expert_count``. None when the model declares no expert
            stack, in which case there is nothing to describe.
        counts: Observed routing counts keyed by layer index and expert
            id. None when no routing source is installed, which yields a
            metadata-only payload with empty ``layers``.

    Returns:
        dict[str, Any] | None: The payload, or None when the
            architectural expert count is unknown or not positive.

    Raises: None.
    """
    arch = num_experts_architectural
    if arch is None or arch < 1:
        return None

    layers: dict[str, dict[str, int]] = {}
    fed: set[int] = set()
    inconsistent: set[str] = set()
    omitted: list[int] = []

    for layer_idx, expert_counts in sorted((counts or {}).items()):
        per_layer: dict[str, int] = {}
        for raw_id, raw_count in expert_counts.items():
            expert_id = _expert_id_key(raw_id)
            if expert_id is None or not 0 <= expert_id < arch:
                inconsistent.add(str(raw_id))
                continue
            try:
                count = int(raw_count)
            except (TypeError, ValueError):
                inconsistent.add(str(raw_id))
                continue
            if count <= 0:
                continue
            per_layer[str(expert_id)] = count
            fed.add(expert_id)
        if per_layer:
            layers[str(layer_idx)] = per_layer
        else:
            omitted.append(layer_idx)

    return {
        "metadata": {
            "num_experts": len(fed),
            "num_experts_architectural": arch,
            "num_experts_fed": len(fed),
            "starved_experts": sorted(set(range(arch)) - fed),
            "omitted_layers": sorted(omitted),
            "inconsistent_experts": sorted(inconsistent),
        },
        "layers": layers,
    }


def parse_byte_cap(value: str) -> int:
    """Parse a byte cap such as "0", "128K", "256M", or "1G" into bytes.

    Args:
        value: Digits optionally followed by a K/M/G suffix (case-insensitive).

    Returns:
        Non-negative byte count; 0 means unlimited.

    Raises:
        ValueError: On an empty string, a negative number, non-numeric input,
            or an unrecognized suffix.
    """
    text = value.strip()
    if not text:
        raise ValueError("byte cap must not be empty")
    suffix = text[-1].upper()
    digits = text[:-1] if suffix in _BYTE_SUFFIXES else text
    try:
        number = int(digits)
    except ValueError as exc:
        raise ValueError(f"invalid byte cap: {value!r}") from exc
    if number < 0:
        raise ValueError(f"byte cap must be >= 0: {value!r}")
    return number * _BYTE_SUFFIXES.get(suffix, 1)


class HeadroomLogger:
    """Writes headroom sidecar JSON files under the headroom/ tree.

    A CaptureLogger fork for the headroom/ directory: same per-call
    open/close style, directories created on first write, OSError raised
    (never swallowed) so the capture-path caller can suppress it.

    Invariants:
      - Files are per-capture (unique capture_id), written once each
      - Directory structure: {base_dir}/YYYY-MM-DD/
      - OSError propagates to the caller
    """

    def __init__(self, base_dir: Path) -> None:
        """Store the headroom root directory.

        Args:
            base_dir: Root of the headroom/ tree (typically {base_path}/headroom).

        Returns: None

        Raises: None

        Invariants:
          - Before: No directories created
          - After: Base dir stored; directories created on first write
        """
        self._base_dir = Path(base_dir)

    def day_dir(self, created_at: str) -> Path:
        """Return the {base_dir}/YYYY-MM-DD directory for a record timestamp.

        Args:
            created_at: ISO 8601 timestamp; unparseable values fall back to
                the current UTC date.

        Returns:
            Path: The day directory (not created by this call).

        Raises: None
        """
        try:
            dt = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
            date_str = dt.strftime("%Y-%m-%d")
        except ValueError:
            date_str = datetime.now(UTC).strftime("%Y-%m-%d")
        return self._base_dir / date_str

    def write_json(
        self, path: Path, payload: dict[str, Any], *, compact: bool = False
    ) -> Path:
        """Write one JSON file, creating parent directories as needed.

        Args:
            path: Destination file (under a headroom day directory).
            payload: JSON-serializable dict.
            compact: When True, serialize with separators=(",", ":") for
                large tensor payloads; otherwise indent=2 for readability.

        Returns:
            Path: The written file path.

        Raises:
            OSError: If directory creation or the file write fails.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        if compact:
            text = json.dumps(
                payload, ensure_ascii=False, separators=(",", ":"), default=str
            )
        else:
            text = json.dumps(payload, ensure_ascii=False, indent=2, default=str)
        path.write_text(text + "\n", encoding="utf-8")
        return path


def _shard_tensor_keys(shard_path: Path) -> set[str] | None:
    """List the tensor names in one safetensors shard (header only).

    Args:
        shard_path: Path to a .safetensors file.

    Returns:
        set[str] | None: The shard's tensor names; None when safetensors is
            unimportable or the header is unreadable.

    Raises: None
    """
    try:
        st_torch: Any = importlib.import_module("safetensors.torch")
    except ImportError:
        return None
    try:
        with st_torch.safe_open(str(shard_path), framework="pt", device="cpu") as shard:
            return set(shard.keys())
    except Exception:
        return None


_HEADROOM_DEP_WARNING = False


def _torch_stack_available() -> bool:
    """Report whether torch and safetensors import without error.

    Imports via importlib, so either source satisfies the check:
    the uv group, a system installation, or PYTHONPATH.

    Args: None

    Returns:
        bool: True when both modules are importable.

    Raises: None
    """
    try:
        importlib.import_module("torch")
        importlib.import_module("safetensors.torch")
        return True
    except ImportError:
        global _HEADROOM_DEP_WARNING
        if not _HEADROOM_DEP_WARNING:
            _HEADROOM_DEP_WARNING = True
            print(
                "warning: headroom enabled but torch/safetensors not "
                "importable; install via 'uv sync --group headroom', "
                "or make system torch/safetensors importable "
                "(PYTHONPATH / venv --system-site-packages)",
                file=sys.stderr,
            )
        return False


def _read_weight_map(model_path: Path) -> dict[str, str] | None:
    """Read the safetensors shard index for a model directory.

    Prefers model.safetensors.index.json; without an index, falls back to
    reading each shard's tensor header (lazy, header-only).

    Args:
        model_path: Model directory expected to hold model.safetensors.index.json
            or at least one *.safetensors file.

    Returns:
        dict[str, str]: Tensor name to shard file name. None when the
            directory is unreadable or no shard index/file exists.

    Raises: None
    """
    try:
        index_path = model_path / "model.safetensors.index.json"
        if index_path.is_file():
            data = json.loads(index_path.read_text(encoding="utf-8"))
            raw_map = data.get("weight_map", {}) if isinstance(data, dict) else {}
            if not isinstance(raw_map, dict):
                return None
            return {str(key): str(value) for key, value in raw_map.items()}
        shards = sorted(model_path.glob("*.safetensors"))
        if not shards:
            return None
        shard_map: dict[str, str] = {}
        for shard in shards:
            keys = _shard_tensor_keys(shard)
            if keys is None:
                continue
            for key in keys:
                shard_map.setdefault(key, shard.name)
        return shard_map
    except (OSError, json.JSONDecodeError):
        return None


def _resolve_tensor_name(
    weight_map: dict[str, str], proj: str, layer: int
) -> str | None:
    """Return the first candidate tensor name present in the index.

    Args:
        weight_map: Safetensors index weight map (tensor name to shard file).
        proj: Projection kind ("gate", "up", "down" for MoE;
            "attention", "intermediate", "output" for dense).
        layer: Layer index substituted into the candidate templates.

    Returns:
        str | None: The resolved tensor name, or None when no candidate
            template matches an index entry.

    Raises: None
    """
    for template in ARCHITECTURE_TENSOR_NAMES[proj]:
        name = template.format(layer=layer)
        if name in weight_map:
            return name
    return None


def _expert_count(config: dict[str, Any]) -> int | None:
    """Return the first usable expert count from a resolved config, or None.

    Reads ``num_local_experts`` then ``num_experts`` then
    ``num_experts_per_layer`` — deliberately the key order of
    model_headroom's loader (``hf_config.get("num_local_experts")
    or hf_config.get("num_experts", 0)``, models/loader.py:171-173),
    extended with ``num_experts_per_layer`` to cover checkpoint
    variants that emit only that key. The two tools classify the
    same config.json the same way where the authority is informed;
    where it is not (both authority keys absent/falsy), the sidecar
    covers the lag by consulting the additional key. The agreement
    is ORDER and FALLTHROUGH SEMANTICS: the sibling's ``or`` skips
    any falsy value (0, 0.0, None, "", [], {}), and this loop does
    the same — a numeric falsy value like 0 is skipped so the next
    key is consulted. Reads from the SAME resolved config object
    that ``_detect_architecture`` reads ``model_type`` from, so a
    nested-``text_config`` wrapper is classified by its own numbers.

    Args:
        config: Resolved config mapping (text_config when nested, else raw).

    Returns:
        int | None: The coerced expert count of the first key that
            yields a truthy value; None when no key produces a
            truthy int.

    Raises: None. bool is rejected explicitly (it is an int subclass,
        and True would otherwise read as a count of 1). Non-numeric
        values (None, "", [], {}, "many") are skipped, never raised
        on. The one documented exceedance: model_headroom CRASHES
        on a truthy non-numeric expert count (int("many") raises
        ValueError), while we keep the guard and fall through. This
        is a strict improvement on a malformed-config path, not a
        disagreement about a well-formed one.
    """
    for key in ("num_local_experts", "num_experts", "num_experts_per_layer"):
        raw = config.get(key)
        if raw is None or isinstance(raw, bool) or not raw:
            continue
        try:
            return int(raw)
        except (TypeError, ValueError):
            continue
    return None


def _resolved_config(model_path: Path) -> dict[str, Any] | None:
    """Read config.json and unwrap a nested ``text_config`` wrapper.

    Args:
        model_path: Model directory expected to hold config.json.

    Returns:
        dict[str, Any] | None: The resolved config mapping, or None when
            config.json is missing, unreadable, not JSON, or not a JSON
            object.

    Raises: None.
    """
    try:
        data = json.loads((model_path / "config.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    nested = data.get("text_config")
    return nested if isinstance(nested, dict) else data


def _detect_architecture(
    model_path: Path,
) -> Literal["dense", "moe", "unknown"]:
    """Detect the model class from config.json (dense/moe/unknown).

    Precedence, in order:
      1. a usable expert count >= 2 -> "moe" (structural gate; catches
         all families with a valid num_experts, including those
         absent from the name list)
      2. ``model_type`` in ``MOE_MODEL_TYPES`` -> "moe" (name-list
         fast path; only applies when the structural rule is
         inconclusive, i.e. no expert count is available. The
         structural rule is authoritative — model_headroom has no
         name list and decides purely on expert count, so the name
         list can never override a structural "dense" verdict)
      3. an ``architectures`` entry containing "moe" -> "moe"
      4. a non-empty ``model_type`` -> "dense"
      5. a non-empty ``architectures`` -> "dense"
      6. neither -> "unknown"

    Step 3 runs before step 4 so the architectures check is reachable for
    any named model_type; previously the step-4 early return made step 3
    dead code for every model with a model_type. A BERT-family config
    ("bert", "roberta", "deberta-v2") has no "moe" substring in its
    architectures entry and no expert count, so it still returns "dense".

    Args:
        model_path: Model directory expected to hold config.json.

    Returns:
        Literal: "moe" when the model is a structurally MoE model,
            "dense" for any other readable config (model_type or
            architectures present), "unknown" when config.json is
            missing, unreadable, or not a JSON object.

    Raises: None
    """
    config = _resolved_config(model_path)
    if config is None:
        return "unknown"
    expert_count = _expert_count(config)
    if expert_count is not None and expert_count >= 2:
        return "moe"
    model_type = str(config.get("model_type", ""))
    if model_type in MOE_MODEL_TYPES and expert_count is None:
        # Name-list fast path: only applies when the structural
        # rule is inconclusive (no expert count available).
        # The structural rule is authoritative — model_headroom
        # has no name list and decides purely on expert count.
        return "moe"
    architectures = config.get("architectures") or []
    if any("moe" in str(item).lower() for item in architectures):
        return "moe"
    if model_type:
        return "dense"
    return "dense" if architectures else "unknown"


def _expert_index_from_name(tensor_name: str) -> int | None:
    """Parse the expert index from an indexed expert template, if present.

    Args:
        tensor_name: Resolved tensor name, e.g.
            "model.layers.0.mlp.experts.12.gate_proj.weight".

    Returns:
        int | None: 12 for the example above; None for fused expert
            tensors without an index (mlp.experts.gate_up_proj — the index
            arrives per slice, not per name) and for dense/non-expert
            tensors. Never raises.

    Raises: None.
    """
    match = re.search(r"experts\.(\d+)", tensor_name)
    if match is None:
        return None
    return int(match.group(1))


def _tensor_to_bundle(
    torch_mod: Any,
    tensor_2d: Any,
    *,
    byte_budget: int | None,
) -> TensorBundle | None:
    """Convert one 2-D float tensor to a bundle, slicing rows pre-conversion.

    Rows are sliced to fit byte_budget BEFORE .numpy().tolist() so the
    Python-object allocation is bounded by the cap, not just the output
    file. parent_shape is the pre-slice 2-D shape. A budget that cannot
    fit even one row yields None (skip) rather than an over-cap emission —
    the same philosophy as the pair-level backstop in _capture_weight_pairs,
    which skips over-cap pairs instead of emitting them.

    Args:
        torch_mod: The torch module (for zeros_like).
        tensor_2d: 2-D float tensor on CPU.
        byte_budget: Optional float32 row budget in bytes; None or
            non-positive means unlimited.

    Returns:
        TensorBundle | None: The bundle, or None when the tensor is empty
            or the budget fits no row.

    Raises: None.
    """
    rows = int(tensor_2d.shape[0])
    cols = int(tensor_2d.shape[1])
    parent_shape = (rows, cols)
    truncated = False
    cap = byte_budget if byte_budget is not None and byte_budget > 0 else None
    if cap is not None:
        rows_fit = cap // (cols * 4) if cols else 0
        if 0 < rows_fit < rows:
            tensor_2d = tensor_2d[:rows_fit]
            truncated = True
        elif rows_fit < 1 and rows > 0:
            return None
    weight = tensor_2d.numpy().tolist()
    gradient = torch_mod.zeros_like(tensor_2d).numpy().tolist()
    if not weight or not weight[0]:
        return None
    return TensorBundle(
        weight=weight,
        gradient=gradient,
        parent_shape=parent_shape,
        truncated=truncated,
    )


def _default_tensor_loader_all(
    model_path: Path,
    tensor_name: str,
    *,
    byte_budget: int | None = None,
) -> tuple[list[tuple[int | None, TensorBundle]], bool]:
    """Load one tensor as 2-D expert bundles (fan-out for fused 3-D).

    A 2-D tensor yields one bundle with the expert index parsed from the
    name (or None). A 3-D [E, *, *] fused expert tensor (gate_up_proj
    [E,2I,H], down_proj [E,H,I]) yields one 2-D bundle per expert slice,
    each carrying its slice index; slices share byte_budget cumulatively
    and the stop flag reports budget exhaustion. parent_shape on a slice
    is the slice's 2-D shape; layer_name stays the fused source name so a
    reader can tell slicing from native layout. The fused tensor is NOT
    split into gate/up halves: which half is which would invent layout
    semantics this module does not own. Any other rank yields ([], False):
    skipped, never raised — a 1-D bias previously propagated TypeError out
    of the singular loader, violating the TensorLoader never-raise
    contract. All tensor work happens inside the shard context; nothing
    memory-mapped escapes it.

    Args:
        model_path: Model directory holding safetensors shards.
        tensor_name: Exact tensor name from the model's weight map.
        byte_budget: Optional total float32 budget in bytes for this
            tensor's slices; None or non-positive means unlimited.

    Returns:
        (bundles, stopped_early): bundles are (expert_id, bundle) pairs;
            stopped_early is True only when the budget cut the fan-out
            short (caller marks truncation). ([], False) on any failure.

    Raises: None (all errors yield ([], False)).
    """
    try:
        torch_mod: Any = importlib.import_module("torch")
        st_torch: Any = importlib.import_module("safetensors.torch")
    except ImportError:
        return [], False
    weight_map = _read_weight_map(model_path)
    if weight_map is None:
        return [], False
    shard_name = weight_map.get(tensor_name)
    if shard_name is None:
        return [], False
    cap = byte_budget if byte_budget is not None and byte_budget > 0 else None
    try:
        with st_torch.safe_open(
            str(model_path / shard_name), framework="pt", device="cpu"
        ) as shard:
            tensor = shard.get_tensor(tensor_name)
            scale = None
            # The safetensors shard object doesn't support the `in` operator,
            # so check for the scale tensor via try/except on get_tensor.
            try:
                scale = shard.get_tensor(tensor_name + ".weight_scale_inv")
            except Exception:
                scale = None
            if scale is not None:
                tensor = tensor.float() * scale.float()
            else:
                tensor = tensor.float()
            if tensor.dim() == 2:
                bundle = _tensor_to_bundle(torch_mod, tensor, byte_budget=cap)
                if bundle is None:
                    return [], False
                return [(_expert_index_from_name(tensor_name), bundle)], False
            if tensor.dim() == 3:
                bundles: list[tuple[int | None, TensorBundle]] = []
                spent = 0
                experts = int(tensor.shape[0])
                for i in range(experts):
                    allowance = cap - spent if cap is not None else None
                    if allowance is not None and allowance <= 0:
                        return bundles, True
                    bundle = _tensor_to_bundle(
                        torch_mod, tensor[i], byte_budget=allowance
                    )
                    if bundle is None:
                        return bundles, True
                    bundles.append((i, bundle))
                    cols = len(bundle.weight[0]) if bundle.weight else 0
                    spent += len(bundle.weight) * cols * 4
                return bundles, False
            return [], False
    except Exception:
        return [], False


def _default_tensor_loader(
    model_path: Path,
    tensor_name: str,
    *,
    byte_budget: int | None = None,
) -> TensorBundle | None:
    """Load one weight tensor (dequantized to float32) plus a zero gradient.

    Reads the shard holding tensor_name via safetensors.torch.safe_open with
    framework="pt" and device="cpu" (NFR-01). When a matching
    weight_scale_inv side tensor exists (fp8 checkpoints), the weight is
    dequantized by elementwise multiplication before the float32 cast.
    The gradient is a zero array of the weight's shape: the model_headroom
    loader requires the gradient key on every pair, and a zero gradient
    scores the grad/weight ratio as 0.0 without a training hook.

    For 3-D fused expert tensors this returns the expert-0 slice; use
    _default_tensor_loader_all for the full per-expert fan-out.
    Unsupported ranks return None (never raise).

    Args:
        model_path: Model directory holding safetensors shards.
        tensor_name: Exact tensor name from the model's weight map.
        byte_budget: Optional float32 row budget in bytes; when positive,
            tensor rows are sliced to fit BEFORE conversion so the
            Python-object allocation is bounded by the cap.

    Returns:
        TensorBundle | None: Weight and zero gradient as float32 nested
            lists; None when torch/safetensors are unimportable, the tensor
            is absent from the index or shard, or any read error occurs.

    Raises: None (all errors return None).
    """
    bundles, _ = _default_tensor_loader_all(
        model_path, tensor_name, byte_budget=byte_budget
    )
    if not bundles:
        return None
    return bundles[0][1]


class HeadroomWriter:
    """Writes per-capture headroom sidecar files after response finalization.

    Owns the {base_dir}/YYYY-MM-DD/ tree: one *.headroom.json sidecar per
    sampled capture and, when weight capture is enabled, one
    *weight_pairs.json beside it. The heavy tensor work runs in a worker
    thread (asyncio.to_thread) so the event loop is never blocked; the app
    spawns write_async as a fire-and-forget task (see keep()) and never
    awaits it on the response path.

    Dependencies: HeadroomLogger, a TensorLoader (default: safetensors +
    torch, imported lazily at write time).

    Invariants:
      - write_async never raises; returns None when nothing was written
      - device="cpu" only (NFR-01) — no TPU/GPU memory touched
      - the float32 payload is bounded by max_bytes (0 = unlimited);
        over-cap tensors are row-truncated and flagged truncated=True
      - interval N samples 1 of N eligible captures (0 or 1 = every call)
      - the sidecar JSON is written after the weight-pair file, so a
        present sidecar always reflects the weight file's final state
      - task label (from CAPTURE_MODEL_TASKS) is captured in the sidecar
        when available; None when the map is off or the id has no entry
    """

    def __init__(
        self,
        base_dir: Path,
        *,
        weights: Literal["off", "gate", "all"] = "off",
        max_bytes: int = 0,
        interval: int = 0,
        signals: frozenset[str] = frozenset(),
        tensor_loader: TensorLoader | None = None,
        routing_source: RoutingCountSource | None = None,
    ) -> None:
        """Configure a headroom sidecar writer.

        Args:
            base_dir: Root of the headroom/ tree ({base_path}/headroom by default).
            weights: Which projections to snapshot — "off", "gate", or
                "all". For MoE models (Qwen/Mixtral): gate, or gate+up+down.
                For dense models (BERT/DeBERTa/RoBERTa): attention, or
                attention+intermediate+output, with MoE projections as
                fallback when weights="all".
            max_bytes: Float32 payload byte cap per capture (rows*cols*4);
                0 = unlimited. Must be >= 0.
            interval: Sample 1 of every N eligible captures; 0 or 1 = every
                capture. Must be >= 0.
            signals: Enabled signal names (subset of HEADROOM_SIGNALS).
            tensor_loader: (model_path, tensor_name) -> TensorBundle | None.
                None selects _default_tensor_loader (safetensors + torch).
            routing_source: (model_path, num_experts) -> RoutingCounts | None.
                None means this deployment observes no router activations, so
                the load_balance artifact carries metadata only. Never
                imported from model_headroom; the payload is written in the
                loader's own JSON contract and the authority reads it back.

        Returns: None

        Raises:
            ValueError: When max_bytes or interval is negative.

        Invariants:
          - Before: No directories created, no tensor libraries imported
          - After: Writer configured; no work happens until write_async
        """
        if max_bytes < 0:
            raise ValueError("max_bytes must be >= 0")
        if interval < 0:
            raise ValueError("interval must be >= 0")
        self._base_dir = Path(base_dir)
        self._weights = weights
        self._max_bytes = max_bytes
        self._interval = interval
        self._signals = signals
        self._tensor_loader = tensor_loader
        self._routing_source = routing_source
        self._logger = HeadroomLogger(self._base_dir)
        self._counter = 0
        self._pending: set[asyncio.Task[Any]] = set()

    def should_sample(self) -> bool:
        """Advance the sampling counter and report whether this capture writes.

        Args: None

        Returns:
            bool: True when this call samples. interval <= 1 samples every
                call; otherwise exactly 1 of every N calls samples
                (counter % interval == 0).

        Raises: None

        Invariants:
          - The counter increments on every call, sampled or not
        """
        self._counter += 1
        if self._interval <= 1:
            return True
        return self._counter % self._interval == 0

    async def write_async(
        self,
        *,
        capture_id: str,
        parent_raw: Path,
        model: str,
        model_path: Path,
        created_at: str,
        task: str | None = None,
    ) -> Path | None:
        """Write the sidecar files for one capture in a worker thread.

        Args:
            capture_id: UUID capture identifier (validated upstream by
                HeadroomRef before this call is scheduled).
            parent_raw: Path of the raw JSONL line this sidecar pairs with.
            model: Served model name from the request.
            model_path: Model directory holding the weights; must exist for
                weight capture (missing dir returns None).
            created_at: ISO 8601 record timestamp; selects the day
                directory so it matches the precomputed ref paths.
            task: Task label for the served model, resolved from
                CAPTURE_MODEL_TASKS at request time. None when the
                map is off or the id has no entry.

        Returns:
            Path | None: The written sidecar path; None when nothing was
                written (missing model dir or a write error). A fatal
                weight step writes the sidecar with the reason in
                error_body instead of returning None.

        Raises: None (all failures return None).

        Invariants:
            - Runs _write_sync via asyncio.to_thread; the event loop is not
              blocked by tensor I/O
            - Never raises, including when the worker thread raises
        """
        try:
            return await asyncio.to_thread(
                self._write_sync, capture_id, parent_raw, model, model_path, created_at
            )
        except Exception:
            return None

    def keep(self, task: asyncio.Task[Any]) -> None:
        """Retain a reference to a running write task until it completes.

        Args:
            task: The asyncio task running write_async.

        Returns: None

        Raises: None

        Invariants:
          - The task is removed from the pending set when it finishes
          - Task exceptions are retrieved in the done callback (no
            'exception was never retrieved' warnings at GC)
        """
        self._pending.add(task)
        task.add_done_callback(self._on_task_done)

    def _on_task_done(self, task: asyncio.Task[Any]) -> None:
        """Drop a finished write task and retrieve its exception, if any."""
        self._pending.discard(task)
        with contextlib.suppress(Exception):
            task.exception()

    def planned_artifacts(
        self, capture_id: str, created_at: str
    ) -> tuple[Path, str | None, str | None, str | None]:
        """Return the precomputed sidecar paths for one capture.

        Args:
            capture_id: UUID capture identifier.
            created_at: ISO 8601 record timestamp (selects the day directory,
                the same one _write_sync uses).

        Returns:
            (Path, str | None, str | None, str | None): The planned sidecar
                path, the planned weight-pair path when weight capture will
                run (effective_rank enabled and weights not "off"), the
                planned expert-activation path when load_balance is enabled,
                and the planned expert-weights path when specialization is
                enabled and weights are not "off". None for any artifact
                that will not run.

        Raises: None

        Invariants:
          - Paths agree with what _write_sync materializes
        """
        day = self._logger.day_dir(created_at)
        weight_path: str | None = None
        if "effective_rank" in self._signals and self._weights != "off":
            weight_path = str(day / f"{capture_id}-weight_pairs.json")
        activation_path: str | None = None
        if "load_balance" in self._signals:
            activation_path = str(
                day / ACTIVATION_FILENAME.format(capture_id=capture_id)
            )
        expert_weights_path: str | None = None
        if "specialization" in self._signals and self._weights != "off":
            expert_weights_path = str(
                day / EXPERT_WEIGHTS_FILENAME.format(capture_id=capture_id)
            )
        return (
            day / f"{capture_id}.headroom.json",
            weight_path,
            activation_path,
            expert_weights_path,
        )

    def _capture_weight_pairs(
        self, model_path: Path, weight_path: Path
    ) -> tuple[list[dict[str, Any]] | None, int, bool, str | None]:
        """Load enabled projection tensors (layer 0) and write the pair file.

        Iterates the enabled projections selected by architecture:
          - "moe" (Qwen/Mixtral/OLMoE): gate, or gate+up+down for "all"
          - "dense"/"unknown" (BERT/DeBERTa/RoBERTa): attention, or
            attention+intermediate+output for "all", with MoE gate/up/down
            as fallback for backward compatibility with dense models that
            also carry mlp.gate_proj tensors.

        Resolves each tensor name against the safetensors index, loads it
        via the configured tensor loader, applies the float32 byte budget
        (row-truncation when over cap), and writes *weight_pairs.json in
        the model_headroom loader format (compact JSON). Every emitted pair
        carries parent_shape (the checkpoint shape before any cap slice),
        truncated (true only when rows were actually cut), and sampling_mode
        ("full_matrix" when nothing was cut, "single_tile" for a capped
        top-left tile), always present, so a reader of the artifact can tell a
        capped tile from a whole matrix of the same emitted size. Every pair
        also carries expert_id: the expert index parsed from the resolved
        name, the per-slice index for fused 3-D expert tensors (which fan
        out to one 2-D pair per expert, sharing the byte budget), or None
        for dense/non-expert tensors.

        Args:
            model_path: Model directory holding safetensors shards.
            weight_path: Destination for the weight-pair JSON file.

        Returns:
            (pairs, payload_bytes, truncated, note) where:
            - (None, 0, False, reason): fatal — no index or no shard under
              model_path; the caller writes the sidecar only, with the
              reason in error_body.
            - ([], 0, False, reason): no tensor resolved/readable; the caller
              writes the sidecar only, with the reason in error_body.
            - (pairs, n, t, note): the pair file was written; n is the
              float32 payload byte count, t flags row-truncation, note is
              the truncation/skip reason or None.

        Raises: None
        """
        if self._tensor_loader is None and not _torch_stack_available():
            return None, 0, False, "torch/safetensors not importable"
        weight_map = _read_weight_map(model_path)
        if weight_map is None:
            return None, 0, False, "no safetensors index under model_path"
        architecture = _detect_architecture(model_path)
        projs, fallback_start = _capture_projs(architecture, self._weights)
        pairs: list[dict[str, Any]] = []
        used = 0
        truncated = False
        not_found: str | None = None
        trunc_note: str | None = None
        for i, proj in enumerate(projs):
            name = _resolve_tensor_name(weight_map, proj, _CAPTURE_LAYER)
            if name is None:
                if i >= fallback_start:
                    not_found = f"no {proj}_proj weight found for layer 0"
                continue
            try:
                if self._tensor_loader is None:
                    remaining = self._max_bytes - used if self._max_bytes > 0 else 0
                    bundles, stopped = _default_tensor_loader_all(
                        model_path,
                        name,
                        byte_budget=remaining if remaining > 0 else None,
                    )
                else:
                    stub_bundle = self._tensor_loader(model_path, name)
                    bundles = (
                        [(_expert_index_from_name(name), stub_bundle)]
                        if stub_bundle is not None
                        else []
                    )
                    stopped = False
            except Exception:
                bundles, stopped = [], False
            if stopped:
                truncated = True
                trunc_note = trunc_note or (
                    f"{name} stopped early: byte cap exhausted mid-tensor"
                )
            if not bundles:
                if not_found is None:
                    not_found = f"tensor {name} unreadable"
                continue
            for expert_idx, bundle in bundles:
                weight = bundle.weight
                gradient = bundle.gradient
                cols = len(weight[0]) if weight else 0
                rows_before = len(weight)
                parent_shape = bundle.parent_shape
                if parent_shape is None:
                    parent_shape = (rows_before, cols) if cols else None
                pair_truncated = bundle.truncated
                if pair_truncated:
                    truncated = True
                    trunc_note = trunc_note or f"{name} row-truncated to fit byte cap"
                payload_bytes = len(weight) * cols * 4
                if self._max_bytes > 0:
                    remaining = self._max_bytes - used
                    if payload_bytes > remaining:
                        rows_fit = remaining // (cols * 4) if cols else 0
                        if rows_fit < 1:
                            truncated = True
                            trunc_note = (
                                trunc_note
                                or f"{name} exceeds remaining byte cap; skipped"
                            )
                            continue
                        weight = weight[:rows_fit]
                        gradient = gradient[:rows_fit]
                        payload_bytes = rows_fit * cols * 4
                        pair_truncated = pair_truncated or rows_fit < rows_before
                        truncated = True
                        trunc_note = (
                            trunc_note
                            or f"{name} truncated to {rows_fit} rows to fit byte cap"
                        )
                pairs.append(
                    {
                        "weight": weight,
                        "gradient": gradient,
                        "layer_name": name,
                        "expert_id": expert_idx,
                        "parent_shape": (
                            [parent_shape[0], parent_shape[1]]
                            if parent_shape is not None
                            else None
                        ),
                        "truncated": pair_truncated,
                        "sampling_mode": "single_tile"
                        if pair_truncated
                        else "full_matrix",
                    }
                )
                used += payload_bytes
        if pairs:
            self._logger.write_json(weight_path, {"pairs": pairs}, compact=True)
            return pairs, used, truncated, trunc_note
        return (
            [],
            0,
            truncated,
            trunc_note or not_found or f"no weight tensors resolved under {model_path}",
        )

    def _capture_expert_weights(
        self,
        model_path: Path,
        ew_path: Path,
        *,
        budget_remaining: int | None,
    ) -> tuple[list[dict[str, Any]] | None, int, bool, str | None]:
        """Write the expert-weights artifact for specialization scoring.

        Same tensor resolution as _capture_weight_pairs (shared
        _capture_projs), but grouped by resolved name: one layers entry per
        tensor with >= 2 same-shape 2-D expert tensors, in the authority's
        load_expert_weights_from_json contract (layer_name +
        expert_weights). Single-expert and mixed-shape groups are skipped
        with a note, never padded or reshaped; the fused tensor is not
        split into gate/up halves. The carrier emits weights only —
        similarity is scored authority-side.

        Args:
            model_path: Model directory holding safetensors shards.
            ew_path: Destination for the expert-weights JSON file.
            budget_remaining: Float32 byte allowance left by the
                weight-pairs path; None means unlimited. Shared
                cumulatively across entries.

        Returns:
            (entries, bytes, truncated, note) mirroring
                _capture_weight_pairs: None entries on fatal (no stack or
                no index), [] plus a note when nothing qualified.

        Raises: None.
        """
        if self._tensor_loader is None and not _torch_stack_available():
            return None, 0, False, "torch/safetensors not importable"
        weight_map = _read_weight_map(model_path)
        if weight_map is None:
            return None, 0, False, "no safetensors index under model_path"
        architecture = _detect_architecture(model_path)
        projs, fallback_start = _capture_projs(architecture, self._weights)
        entries: list[dict[str, Any]] = []
        used = 0
        truncated = False
        note: str | None = None
        for i, proj in enumerate(projs):
            name = _resolve_tensor_name(weight_map, proj, _CAPTURE_LAYER)
            if name is None:
                if i >= fallback_start and note is None:
                    note = f"no {proj}_proj weight found for layer 0"
                continue
            try:
                if self._tensor_loader is None:
                    if budget_remaining is not None and budget_remaining - used <= 0:
                        truncated = True
                        note = note or "byte cap exhausted before expert weights"
                        break
                    allowance = (
                        budget_remaining - used
                        if budget_remaining is not None
                        else None
                    )
                    bundles, stopped = _default_tensor_loader_all(
                        model_path,
                        name,
                        byte_budget=allowance,
                    )
                else:
                    stub_bundle = self._tensor_loader(model_path, name)
                    bundles = (
                        [(_expert_index_from_name(name), stub_bundle)]
                        if stub_bundle is not None
                        else []
                    )
                    stopped = False
            except Exception:
                bundles, stopped = [], False
            if stopped:
                truncated = True
                note = note or f"{name} stopped early: byte cap exhausted mid-tensor"
            if not bundles:
                if note is None:
                    note = f"tensor {name} unreadable"
                continue
            shapes = [
                (len(b.weight), len(b.weight[0]) if b.weight else 0) for _, b in bundles
            ]
            if len(bundles) < 2:
                if note is None:
                    note = f"{name} has fewer than 2 experts; skipped"
                continue
            if any(s != shapes[0] or s[1] == 0 for s in shapes):
                if note is None:
                    note = f"{name} experts differ in shape; skipped"
                continue
            entry_bytes = sum(r * c * 4 for r, c in shapes)
            if budget_remaining is not None and used + entry_bytes > budget_remaining:
                truncated = True
                note = note or f"{name} exceeds remaining byte cap; skipped"
                continue
            entries.append(
                {"layer_name": name, "expert_weights": [b.weight for _, b in bundles]}
            )
            used += entry_bytes
        if entries:
            self._logger.write_json(ew_path, {"layers": entries}, compact=True)
            return entries, used, truncated, note
        return (
            [],
            0,
            truncated,
            note or f"no expert weight groups resolved under {model_path}",
        )

    def _capture_expert_activations(
        self, model_path: Path, activation_path: Path
    ) -> tuple[Path | None, str | None]:
        """Write the expert-activation artifact for one capture.

        Reads the architectural expert count from config.json, asks the
        installed routing source for observed counts, and writes the
        model_headroom loader payload. Imports nothing beyond json and
        pathlib, so the metadata half of the contract stays available
        with neither torch nor safetensors installed.

        Args:
            model_path: Model directory holding config.json.
            activation_path: Destination for the activation JSON file.

        Returns:
            (Path | None, str | None): The written path and no reason, or
                (None, reason) when the model declares no expert stack.
                A configured-but-failing routing source is not a reason to
                withhold the artifact: the payload is written with empty
                layers, which the authority reports as an error rather
                than as a measurement.

        Raises: None
        """
        config = _resolved_config(model_path)
        num_experts = _expert_count(config) if config is not None else None
        counts: RoutingCounts | None = None
        if num_experts is not None and self._routing_source is not None:
            try:
                counts = self._routing_source(model_path, num_experts)
            except Exception:
                counts = None
        payload = build_activation_payload(
            num_experts_architectural=num_experts, counts=counts
        )
        if payload is None:
            return None, None
        self._logger.write_json(activation_path, payload, compact=True)
        return activation_path, None

    def _write_sync(
        self,
        capture_id: str,
        parent_raw: Path,
        model: str,
        model_path: Path,
        created_at: str,
        task: str | None = None,
    ) -> Path | None:
        """Write the sidecar files for one capture (worker-thread body).

        Written in this order: the weight-pair file (when weight capture
        produced pairs), then the sidecar JSON (authoritative, last). A
        fatal failure in the weight step writes the sidecar only, with the
        reason in error_body.

        Args:
            capture_id: UUID capture identifier.
            parent_raw: Path of the raw JSONL line this sidecar pairs with.
            model: Served model name from the request.
            model_path: Model directory holding the weights.
            created_at: ISO 8601 record timestamp; selects the day
                directory (matches the precomputed ref paths). collected_at
                records the actual write time.
            task: Task label for the served model; None when CAPTURE_MODEL_TASKS
                is off or the id has no entry.

        Returns:
            Path | None: The sidecar path when written; None when the model
                dir is missing or a write error occurred (a fatal weight
                step still writes the sidecar, with the reason in
                error_body).

        Raises: None

        Invariants:
            - sidecar JSON is the last file written (authoritative artifact)
            - the truncation flag is preserved even when the weight step
              produced zero pairs (byte cap dropped every tensor)
            - the task label is captured in the sidecar when available
        """
        try:
            model_path = Path(model_path)
            if not model_path.is_dir():
                return None
            architecture = _detect_architecture(model_path)
            signals: dict[str, str | None] = dict.fromkeys(SIGNAL_KEYS)
            bytes_written = 0
            truncated = False
            error_body: str | None = None

            day = self._logger.day_dir(created_at)
            sidecar_path = day / f"{capture_id}.headroom.json"
            weight_path = day / f"{capture_id}-weight_pairs.json"
            activation_path = day / ACTIVATION_FILENAME.format(capture_id=capture_id)
            expert_weights_path = day / EXPERT_WEIGHTS_FILENAME.format(
                capture_id=capture_id
            )

            if "effective_rank" in self._signals and self._weights != "off":
                pairs, payload_bytes, trunc, note = self._capture_weight_pairs(
                    model_path, weight_path
                )
                if pairs:
                    signals["effective_rank"] = str(weight_path)
                    bytes_written = payload_bytes
                    truncated = trunc
                    error_body = note
                else:
                    # Fatal (pairs is None: no index/shard under model_path)
                    # and zero-pairs cases both write the sidecar only, with
                    # the reason in error_body (truncation flag preserved).
                    truncated = trunc
                    error_body = note

            if "specialization" in self._signals and self._weights != "off":
                remaining = (
                    self._max_bytes - bytes_written if self._max_bytes > 0 else None
                )
                entries, ew_bytes, ew_trunc, ew_note = self._capture_expert_weights(
                    model_path, expert_weights_path, budget_remaining=remaining
                )
                if entries:
                    signals["specialization"] = str(expert_weights_path)
                    bytes_written += ew_bytes
                truncated = truncated or ew_trunc
                if ew_note:
                    error_body = f"{error_body}; {ew_note}" if error_body else ew_note

            if "load_balance" in self._signals:
                written, note = self._capture_expert_activations(
                    model_path, activation_path
                )
                if written is not None:
                    signals["load_balance"] = str(written)

            record = CapturedHeadroom(
                capture_id=capture_id,
                parent_raw=Path(parent_raw),
                model=model,
                model_path=str(model_path),
                architecture=architecture,
                task=task,
                collected_at=datetime.now(UTC).isoformat(),
                signals=signals,
                bytes_written=bytes_written,
                bytes_cap=self._max_bytes,
                truncated=truncated,
                error_body=error_body,
            )
            return self._logger.write_json(sidecar_path, record.model_dump(mode="json"))
        except (OSError, ValueError, TypeError):
            return None
