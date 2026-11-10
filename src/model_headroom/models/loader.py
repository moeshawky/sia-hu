from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import torch

from model_headroom.models.types import (
    ArchitectureDetails,
    ArchitectureType,
    ExpertActivationData,
    ExpertWeights,
    GradientProvenance,
    GradientWeightPair,
    MLPActivations,
    ModuleFamily,
    SamplingMode,
    SourceType,
    VerificationStatus,
)

_ALLOWED_SAMPLING_MODES = frozenset(
    {
        "full_matrix",
        "randomized_svd",
        "multi_tile",
        "single_tile",
        "external",
        "unknown",
    }
)


def _extract_text_config(config: dict[str, Any]) -> dict[str, Any]:
    if "text_config" in config and isinstance(config["text_config"], dict):
        return config["text_config"]
    return config


def _get_expert_weight_name(
    model_type: str,
    layer_idx: int,
    expert_id: int,
    weight_type: Literal["gate", "down", "up"],
) -> str:
    if model_type in ("mixtral", "phimoe"):
        name_map = {"gate": "w1", "down": "w2", "up": "w3"}
        weight_name = name_map[weight_type]
        return f"model.layers.{layer_idx}.block_sparse_moe.experts.{expert_id}.{weight_name}.weight"
    if model_type in ("qwen3_moe", "qwen3_5_moe", "qwen3_5_text", "qwen2_moe", "olmoe"):
        name_map = {"gate": "gate_proj", "down": "down_proj", "up": "up_proj"}
        weight_name = name_map[weight_type]
        return f"model.layers.{layer_idx}.mlp.experts.{expert_id}.{weight_name}.weight"
    return f"model.layers.{layer_idx}.block_sparse_moe.experts.{expert_id}.w1.weight"


def _detect_model_format(model_path: str) -> Literal["huggingface", "gguf"]:
    """Detect model format from path. Returns 'gguf' for .gguf files, 'huggingface'
    for directories containing config.json at root or encoder/config.json
    (laya format). Raises ValueError for unrecognized formats."""
    p = Path(model_path)
    if p.suffix == ".gguf":
        return "gguf"
    if p.is_dir():
        if (p / "config.json").exists():
            return "huggingface"
        if (p / "encoder" / "config.json").exists():
            return "huggingface"
    raise ValueError(
        f"Cannot determine model format for: {model_path}. "
        "Expected .gguf file or directory with config.json."
    )


def get_module_family(layer_name: str) -> ModuleFamily:
    """Classify a layer name into a module family. Checks for vision, MTP, linear
    attention, FFN/MLP, attention, state, embedding, lm_head, expert, and router
    keywords in the lowercased layer name. Returns 'other' for unrecognized names.
    Linear attention keywords checked first in FFN branch to avoid
    misclassification.

    Supports BERT-family naming (bert.encoder.layer.N.attention.self.query.weight →
    'attention'; bert.encoder.layer.N.intermediate.dense.weight → 'ffn';
    bert.encoder.layer.N.attention.output.dense.weight → 'attention' via nested
    exclusion; pooler.dense.weight → 'other' via pooler exclusion). ModernBERT uses
    identical naming. laya head weights without recognized keywords classify as
    'other'."""
    name = layer_name.lower()
    if "vision" in name or "patch_embed" in name or "merger" in name:
        return "vision"
    if "mtp" in name or "nextn" in name:
        return "mtp"
    if "linear_attn" in name or "gated_deltanet" in name:
        return "linear_attention"
    if any(
        k in name for k in ("gate_proj", "up_proj", "down_proj", "ffn", "mlp", "dense")
    ):
        if "linear_attn" in name or "delta" in name:
            return "linear_attention"
        if "attention" in name:
            return "attention"
        if "pooler" in name:
            return "other"
        return "ffn"
    if any(
        k in name for k in ("q_proj", "k_proj", "v_proj", "o_proj", "qkv", "attention")
    ):
        return "attention"
    if "attn" in name:
        return "attention"
    if any(k in name for k in ("state", "gate", "z_proj")):
        return "state"
    if "embed" in name:
        return "embedding"
    if "lm_head" in name:
        return "lm_head"
    if "expert" in name:
        return "expert"
    if "router" in name:
        return "router"
    return "other"


def detect_architecture_details(model_path: str) -> ArchitectureDetails:
    """Detect detailed model type properties from config files. Returns ArchitectureDetails
    with family (moe/dense/unknown), subtype, attention type flags, vision tower flag,
    MTP flag, and detection source. Checks num_local_experts/num_experts for MoE
    detection, layer_types for attention type, weight map for vision and MTP markers,
    and the GGUF tensor name table for GGUF files (header only, no dequantization).
    Catches all exceptions and returns unknown on failure."""
    try:
        fmt = _detect_model_format(model_path)
        if fmt == "gguf":
            from model_headroom.models.gguf_loader import (
                load_gguf_config,
                open_gguf_reader,
            )

            config = load_gguf_config(Path(model_path))
            has_moe = config.expert_count >= 2
            reader = open_gguf_reader(Path(model_path))
            names = [t.name for t in reader.tensors]
            lowered = [n.lower() for n in names]
            has_linear = any(("ssm_" in n) or ("linear_attn" in n) for n in lowered)
            has_full = any(
                ("attn_q." in n)
                or ("attn_k." in n)
                or ("attn_v." in n)
                or ("attn_output" in n)
                or ("self_attn" in n)
                for n in lowered
            )
            has_vision = any(
                ("visual" in n) or ("vision" in n) or ("patch_embed" in n)
                for n in lowered
            )
            has_mtp = any(("mtp" in n) or ("nextn" in n) for n in lowered)
            return ArchitectureDetails(
                family="moe" if has_moe else "dense",
                subtype=(
                    "hybrid_linear_attention"
                    if (has_linear or has_full)
                    else "dense_standard"
                ),
                has_moe=has_moe,
                has_full_attention=has_full,
                has_linear_attention=has_linear,
                has_vision_tower=has_vision,
                has_mtp=has_mtp,
                detection_source="gguf_tensor_names",
            )

        config_path = Path(model_path) / "config.json"
        with config_path.open() as f:
            config_raw: dict[str, Any] = json.load(f)

        raw_for_flags = config_raw
        hf_config: dict[str, Any] = _extract_text_config(config_raw)

        num_experts = hf_config.get("num_local_experts") or hf_config.get(
            "num_experts", 0
        )
        has_moe = int(num_experts) >= 2
        family: ArchitectureType = "moe" if has_moe else "dense"

        layer_types = hf_config.get("layer_types") or raw_for_flags.get("layer_types")
        has_linear = False
        has_full = False
        if isinstance(layer_types, list):
            has_linear = any(v == "linear_attention" for v in layer_types)
            has_full = any(v == "full_attention" for v in layer_types)

        if not has_linear and not has_full:
            try:
                index_path = Path(model_path) / "model.safetensors.index.json"
                if index_path.exists():
                    with index_path.open() as f:
                        wm = json.load(f).get("weight_map", {})
                    has_linear = any("linear_attn" in k for k in wm)
                    has_full = any(
                        "self_attn" in k or "attn." in k or "attention." in k
                        for k in wm
                    )
            except Exception:  # noqa: S110
                pass

        has_vision = False
        if "vision_config" in raw_for_flags or "vision_config" in hf_config:
            has_vision = True
        else:
            try:
                index_path = Path(model_path) / "model.safetensors.index.json"
                if index_path.exists():
                    with index_path.open() as f:
                        wm = json.load(f).get("weight_map", {})
                    has_vision = any(
                        k.startswith("model.visual")
                        or "vision" in k
                        or "patch_embed" in k
                        for k in wm
                    )
            except Exception:  # noqa: S110
                pass

        has_mtp = False
        if any("mtp" in k.lower() or "nextn" in k.lower() for k in hf_config):
            has_mtp = True
        if any("mtp" in k.lower() or "nextn" in k.lower() for k in raw_for_flags):
            has_mtp = True
        if not has_mtp:
            try:
                index_path = Path(model_path) / "model.safetensors.index.json"
                if index_path.exists():
                    with index_path.open() as f:
                        wm = json.load(f).get("weight_map", {})
                    has_mtp = any("mtp" in k.lower() for k in wm)
            except Exception:  # noqa: S110
                pass

        # Classifier detection: BERT/DeBERTa/ModernBERT/laya/RockAndEgg are
        # dense single-expert models but arrive via classifier architectures.
        architectures = hf_config.get("architectures") or raw_for_flags.get(
            "architectures", []
        )
        is_classifier = False
        if isinstance(architectures, list):
            is_classifier = any(
                isinstance(a, str)
                and any(
                    kw in a.lower()
                    for kw in ("bert", "deberta", "modernbert", "laya", "rockandegg")
                )
                for a in architectures
            )

        subtype: Literal["hybrid_linear_attention", "dense_standard", "unknown"] = (
            "hybrid_linear_attention" if (has_linear or has_full) else "dense_standard"
        )

        detection_source = "config.json+index"
        if is_classifier:
            detection_source = "config.json+index+classifier"

        return ArchitectureDetails(
            family=family,
            subtype=subtype,
            has_moe=has_moe,
            has_full_attention=has_full,
            has_linear_attention=has_linear,
            has_vision_tower=has_vision,
            has_mtp=has_mtp,
            detection_source=detection_source,
        )
    except Exception:
        return ArchitectureDetails(
            family="unknown",
            subtype="unknown",
            has_moe=False,
            has_full_attention=False,
            has_linear_attention=False,
            has_vision_tower=False,
            has_mtp=False,
            detection_source="fallback",
        )


def extract_layer_weights(
    model_path: str,
    layer_name: str,
    device: str = "cpu",
) -> torch.Tensor | None:
    """Extract a weight tensor by layer name from model files. Tries safetensors files
    first (loading each until layer found), falls back to .bin files (with
    weights_only=True, retrying weights_only=False on pickle error). Returns None
    if layer not found or on OSError/ValueError. All tensors load to specified device
    (default CPU)."""
    try:
        safetensor_files = list(Path(model_path).glob("*.safetensors"))
        if safetensor_files:
            from safetensors.torch import load_file

            for sf in safetensor_files:
                sd = load_file(str(sf), device=device)
                if layer_name in sd:
                    return sd[layer_name]

        bin_files = list(Path(model_path).glob("*.bin"))
        if bin_files:
            state_dict: dict[str, torch.Tensor] = {}
            for bf in bin_files:
                try:
                    state_dict.update(
                        torch.load(bf, map_location=device, weights_only=True)
                    )
                except (pickle.UnpicklingError, RuntimeError):
                    state_dict.update(
                        torch.load(bf, map_location=device, weights_only=False)
                    )
            return state_dict.get(layer_name)

        return None
    except (OSError, ValueError):
        return None


def detect_architecture(model_path: str) -> ArchitectureType:
    """Detect model type (moe or dense) from model path. Calls the detailed detection
    function and returns the family field. Single-line wrapper for callers needing
    only the binary MoE/dense classification without detailed properties."""
    return detect_architecture_details(model_path).family


def load_activations_from_json(
    json_path: str,
) -> ExpertActivationData | None:
    """Load expert activation data from a JSON file. Expects top-level 'metadata' and
    'layers' keys. Returns None if keys are missing, file not found, JSON is
    malformed, or types are wrong. Wraps ExpertActivationData constructor result."""
    try:
        with Path(json_path).open() as f:
            data = json.load(f)

        if "metadata" not in data or "layers" not in data:
            return None

        return ExpertActivationData(
            metadata=data["metadata"],
            layers=data["layers"],
        )
    except (FileNotFoundError, json.JSONDecodeError, KeyError, TypeError):
        return None


def _pair_parent_shape(raw: Any) -> tuple[int, int] | None:
    if not isinstance(raw, (list, tuple)) or len(raw) != 2:
        return None
    rows, cols = raw
    if isinstance(rows, bool) or isinstance(cols, bool):
        return None
    if not isinstance(rows, int) or not isinstance(cols, int):
        return None
    if rows < 0 or cols < 0:
        return None
    return (rows, cols)


def _pair_sampling_mode(raw: Any) -> SamplingMode | None:
    return cast("SamplingMode", raw) if raw in _ALLOWED_SAMPLING_MODES else None


def load_weight_pairs_from_json(
    json_path: str,
) -> list[GradientWeightPair] | None:
    """Load gradient/weight pairs from a JSON file. Accepts both wrapped format (dict
    with 'pairs' key) and unwrapped list format. Each pair must have 'gradient',
    'weight', and 'layer_name' keys. Converts to torch.float32 tensors. Returns None
    if file missing, malformed, or no valid pairs found. Raises ValueError naming the
    offending layer, field, and actual shape if any pair tensor is not 2-D; a 3-D fused
    expert stack is rejected rather than sliced, because slicing would invent a shape
    the checkpoint never had.

    Carries the optional per-pair provenance keys parent_shape, truncated, and
    sampling_mode when present. All three default to None, so a legacy three-key pair
    yields the same object as before and a missing key is distinguishable from an
    explicit false."""
    pairs: list[GradientWeightPair] = []
    try:
        with Path(json_path).open() as f:
            data = json.load(f)

        pairs_raw = data.get("pairs", data) if isinstance(data, dict) else data
        if not isinstance(pairs_raw, list):
            return None

        for p in pairs_raw:
            if not all(k in p for k in ("gradient", "weight", "layer_name")):
                continue
            pairs.append(
                GradientWeightPair(
                    gradient=torch.tensor(p["gradient"], dtype=torch.float32),
                    weight=torch.tensor(p["weight"], dtype=torch.float32),
                    layer_name=p["layer_name"],
                    parent_shape=_pair_parent_shape(p.get("parent_shape")),
                    truncated=(
                        p["truncated"] if isinstance(p.get("truncated"), bool) else None
                    ),
                    sampling_mode=_pair_sampling_mode(p.get("sampling_mode")),
                )
            )
    except (FileNotFoundError, json.JSONDecodeError, TypeError, ValueError):
        return None

    for pair in pairs:
        for field, tensor in (("gradient", pair.gradient), ("weight", pair.weight)):
            if tensor.ndim != 2:
                raise ValueError(
                    f"{json_path}: pair {pair.layer_name!r} field {field!r} has "
                    f"shape {list(tensor.shape)} (ndim {tensor.ndim}); weight "
                    f"pairs must be 2-D [rows, cols]. A 3-D fused expert stack is "
                    f"rejected, not sliced — slicing would invent a shape the "
                    f"checkpoint never had. Select one expert, or export the "
                    f"stack as expert weights."
                )
    return pairs if pairs else None


_ALLOWED_SOURCE_TYPES = frozenset(
    {"measured", "synthetic", "fixture", "estimated", "imported", "unknown"}
)
_ALLOWED_VERIFICATION_STATUSES = frozenset(
    {"verified", "provenance_incomplete", "unverified"}
)


def load_gradient_provenance_from_json(json_path: str) -> GradientProvenance | None:
    """Load gradient provenance metadata from a JSON file. Canonical format is a
    single nested document: ``{"provenance": {...fields...}}``. A flat document
    whose provenance fields sit at top level is NOT accepted here — silently
    unpacking it under the wrong meaning is how missing measurements get
    laundered into valid inputs, so it returns None instead. ``source_type``
    and ``verification_status`` values outside their allowed sets also return
    None. Defaults source_type to 'measured' and verification_status to
    'provenance_incomplete' when absent. Returns None on any error."""
    try:
        with Path(json_path).open() as f:
            data = json.load(f)
        prov_raw = data.get("provenance") if isinstance(data, dict) else None
        if not isinstance(prov_raw, dict):
            return None
        source_type = prov_raw.get("source_type", "measured")
        verification_status = prov_raw.get(
            "verification_status", "provenance_incomplete"
        )
        if not isinstance(source_type, str) or source_type not in _ALLOWED_SOURCE_TYPES:
            return None
        if (
            not isinstance(verification_status, str)
            or verification_status not in _ALLOWED_VERIFICATION_STATUSES
        ):
            return None
        return GradientProvenance(
            model_revision=prov_raw.get("model_revision"),
            dataset_hash=prov_raw.get("dataset_hash"),
            dataset_type=prov_raw.get("dataset_type"),
            token_count=prov_raw.get("token_count"),
            batch_count=prov_raw.get("batch_count"),
            loss_function=prov_raw.get("loss_function"),
            forward_dtype=prov_raw.get("forward_dtype"),
            backward_dtype=prov_raw.get("backward_dtype"),
            module_name=prov_raw.get("module_name"),
            parameter_name=prov_raw.get("parameter_name"),
            gradient_aggregation=prov_raw.get("gradient_aggregation"),
            seed=prov_raw.get("seed"),
            source_type=cast("SourceType", source_type),
            verification_status=cast("VerificationStatus", verification_status),
        )
    except (FileNotFoundError, json.JSONDecodeError, TypeError, ValueError):
        return None


def infer_source_type_from_path(
    json_path: str,
) -> tuple[SourceType, VerificationStatus]:
    """Infer source type and verification status from JSON file structure. Checks for
    nested 'provenance' dict first, then top-level 'source_type' key with validation
    against allowed values. Returns ('measured', 'provenance_incomplete') as default
    when no provenance info is found. Validates source_type is one of the six allowed
    values."""
    try:
        with Path(json_path).open() as f:
            data = json.load(f)
        if isinstance(data, dict) and "provenance" in data:
            prov = data["provenance"]
            if isinstance(prov, dict):
                st = prov.get("source_type", "measured")
                vs = prov.get("verification_status", "provenance_incomplete")
                return (st, vs)
        if isinstance(data, dict) and "source_type" in data:
            st = data["source_type"]
            vs = data.get("verification_status", "provenance_incomplete")
            if not isinstance(st, str) or st not in (
                "measured",
                "synthetic",
                "fixture",
                "estimated",
                "imported",
                "unknown",
            ):
                return ("unknown", "unverified")
            return (cast("SourceType", st), vs)
    except Exception:  # noqa: S110
        pass
    return ("measured", "provenance_incomplete")


def load_mlp_activations_from_json(
    json_path: str,
) -> list[MLPActivations] | None:
    """Load MLP activation tensors from a JSON file. Accepts wrapped format (dict with
    'layers' key) or unwrapped list. Each layer must have 'activations' (3D array)
    and 'layer_name' keys. Converts activations to torch.float32 tensors. Returns
    None if file missing, malformed, or no valid layers found."""
    try:
        with Path(json_path).open() as f:
            data = json.load(f)

        layers_raw = data.get("layers", data) if isinstance(data, dict) else data
        if not isinstance(layers_raw, list):
            return None

        layers: list[MLPActivations] = []
        for layer in layers_raw:
            if "activations" not in layer or "layer_name" not in layer:
                continue
            t = torch.tensor(layer["activations"], dtype=torch.float32)
            if t.ndim != 3:
                continue
            layers.append(MLPActivations(activations=t, layer_name=layer["layer_name"]))
        return layers if layers else None
    except (FileNotFoundError, json.JSONDecodeError, TypeError, ValueError):
        return None


def load_mlp_activations_with_provenance(
    json_path: str,
) -> tuple[list[MLPActivations] | None, SourceType, VerificationStatus]:
    """Load MLP activations from JSON along with source type and verification status.
    Delegates tensor loading to load_mlp_activations_from_json, then infers provenance
    from JSON structure (top-level source_type, nested provenance dict, or first
    layer's source_type). Falls back to synthetic/unverified if first activation
    tensor has <= 2048 elements. Returns tuple of (activations, source_type,
    verification_status)."""
    acts = load_mlp_activations_from_json(json_path)
    if acts is None:
        return (None, "unknown", "unverified")
    try:
        with Path(json_path).open() as f:
            data = json.load(f)
        if isinstance(data, dict):
            if "source_type" in data:
                st = data.get("source_type", "measured")
                vs = data.get("verification_status", "provenance_incomplete")
                return (acts, st, vs)
            if "provenance" in data and isinstance(data["provenance"], dict):
                prov = data["provenance"]
                return (
                    acts,
                    prov.get("source_type", "measured"),
                    prov.get("verification_status", "provenance_incomplete"),
                )
            layers_raw = data.get("layers", [])
            if isinstance(layers_raw, list) and layers_raw:
                first = layers_raw[0]
                if isinstance(first, dict) and "source_type" in first:
                    return (
                        acts,
                        first.get("source_type", "measured"),
                        first.get("verification_status", "provenance_incomplete"),
                    )
    except Exception:  # noqa: S110
        pass
    if acts and acts[0].activations.numel() <= 64 * 8 * 4:
        return (acts, "synthetic", "unverified")
    return (acts, "measured", "provenance_incomplete")


def load_expert_weights_from_json(
    json_path: str,
) -> list[ExpertWeights] | None:
    """Load expert weights from a JSON file. Accepts wrapped format (dict with 'layers'
    key) or unwrapped list. Each layer must have 'expert_weights' (list of >= 2 weight
    arrays) and 'layer_name' keys. Converts each weight to torch.float32 tensor.
    Returns None if file missing, malformed, or no valid layers found."""
    try:
        with Path(json_path).open() as f:
            data = json.load(f)

        layers_raw = data.get("layers", data) if isinstance(data, dict) else data
        if not isinstance(layers_raw, list):
            return None

        layers: list[ExpertWeights] = []
        for layer in layers_raw:
            if "expert_weights" not in layer or "layer_name" not in layer:
                continue
            weights_raw = layer["expert_weights"]
            if not isinstance(weights_raw, list) or len(weights_raw) < 2:
                continue
            weight_tensors = [torch.tensor(w, dtype=torch.float32) for w in weights_raw]
            layers.append(
                ExpertWeights(
                    expert_weights=weight_tensors,
                    layer_name=layer["layer_name"],
                )
            )
        return layers if layers else None
    except (FileNotFoundError, json.JSONDecodeError, TypeError, ValueError):
        return None


@dataclass(frozen=True)
class LoadOutcome:
    """Outcome of loading a JSON data file. data holds the loaded value on
    success; error holds a human-readable cause (with the offending path and
    the concrete fix) on failure. Exactly one of data/error is meaningful."""

    data: Any
    error: str | None


def _json_read(json_path: str) -> tuple[Any, str | None]:
    """Read and parse a JSON file. Returns (data, None) on success or
    (None, error) with a precise cause on failure."""
    path = Path(json_path)
    if not path.exists():
        return None, f"file does not exist: {json_path}"
    try:
        with path.open() as f:
            return json.load(f), None
    except json.JSONDecodeError as e:
        return None, f"JSON is malformed in {json_path}: {e.msg}"
    except OSError as e:
        return None, f"cannot read {json_path}: {e}"


def _weight_pairs_format_hint() -> str:
    return (
        'expected JSON object {"pairs": [{"gradient": [[...]], "weight": [[...]], '
        '"layer_name": "..."}]} or a list of such pairs'
    )


def load_weight_pairs_diagnostic(json_path: str) -> LoadOutcome:
    """Load gradient/weight pairs with a precise failure diagnostic. Same schema
    as load_weight_pairs_from_json but returns a LoadOutcome carrying the cause
    (missing file, malformed JSON, missing keys, wrong tensor rank, empty) instead
    of collapsing all failures into None."""
    data, err = _json_read(json_path)
    if err is not None:
        return LoadOutcome(data=None, error=err)
    if not isinstance(data, (dict, list)):
        return LoadOutcome(
            data=None,
            error=(
                f"expected JSON object or list in {json_path} but got "
                f"{type(data).__name__}; {_weight_pairs_format_hint()}"
            ),
        )
    pairs_raw = data.get("pairs", data) if isinstance(data, dict) else data
    if not isinstance(pairs_raw, list):
        return LoadOutcome(
            data=None,
            error=(
                f"expected 'pairs' to be a list in {json_path} but got "
                f"{type(pairs_raw).__name__}; {_weight_pairs_format_hint()}"
            ),
        )
    if not pairs_raw:
        return LoadOutcome(
            data=None,
            error=(
                f"no valid weight pairs found in {json_path}; the 'pairs' list is empty; "
                f"{_weight_pairs_format_hint()}"
            ),
        )
    pairs: list[GradientWeightPair] = []
    for i, p in enumerate(pairs_raw):
        if not isinstance(p, dict):
            return LoadOutcome(
                data=None,
                error=(
                    f"pair {i} in {json_path} is not a JSON object; "
                    f"expected keys 'gradient', 'weight', 'layer_name'; "
                    f"{_weight_pairs_format_hint()}"
                ),
            )
        missing = [k for k in ("gradient", "weight", "layer_name") if k not in p]
        if missing:
            return LoadOutcome(
                data=None,
                error=(
                    f"pair {i} in {json_path} is missing required keys {missing}; "
                    f"found keys {list(p.keys())}; expected 'gradient', 'weight', 'layer_name'; "
                    f"{_weight_pairs_format_hint()}"
                ),
            )
        try:
            gradient = torch.tensor(p["gradient"], dtype=torch.float32)
            weight = torch.tensor(p["weight"], dtype=torch.float32)
        except (TypeError, ValueError) as e:
            return LoadOutcome(
                data=None,
                error=f"pair {i} in {json_path}: cannot convert gradient/weight to tensors: {e}",
            )
        pairs.append(
            GradientWeightPair(
                gradient=gradient,
                weight=weight,
                layer_name=p["layer_name"],
                parent_shape=_pair_parent_shape(p.get("parent_shape")),
                truncated=(
                    p["truncated"] if isinstance(p.get("truncated"), bool) else None
                ),
                sampling_mode=_pair_sampling_mode(p.get("sampling_mode")),
            )
        )
    for pair in pairs:
        for field, tensor in (("gradient", pair.gradient), ("weight", pair.weight)):
            if tensor.ndim != 2:
                return LoadOutcome(
                    data=None,
                    error=(
                        f"{json_path}: pair {pair.layer_name!r} field {field!r} has "
                        f"shape {list(tensor.shape)} (ndim {tensor.ndim}); weight "
                        f"pairs must be 2-D [rows, cols]. A 3-D fused expert stack is "
                        f"rejected, not sliced — slicing would invent a shape the "
                        f"checkpoint never had. Select one expert, or export the "
                        f"stack as expert weights."
                    ),
                )
    return LoadOutcome(data=pairs, error=None)


def load_activations_diagnostic(json_path: str) -> LoadOutcome:
    """Load expert activation data with a precise failure diagnostic. Same schema
    as load_activations_from_json but returns a LoadOutcome carrying the cause."""
    data, err = _json_read(json_path)
    if err is not None:
        return LoadOutcome(data=None, error=err)
    if not isinstance(data, dict):
        return LoadOutcome(
            data=None,
            error=(
                f"expected JSON object in {json_path} but got "
                f"{type(data).__name__}; expert_activations JSON must be "
                f'{{"metadata": {{...}}, "layers": {{...}}}}; '
            ),
        )
    missing = [k for k in ("metadata", "layers") if k not in data]
    if missing:
        return LoadOutcome(
            data=None,
            error=(
                f"{json_path} is missing required keys {missing}; "
                f"found keys {list(data.keys())}; expected 'metadata' and 'layers'; "
            ),
        )
    try:
        return LoadOutcome(
            data=ExpertActivationData(metadata=data["metadata"], layers=data["layers"]),
            error=None,
        )
    except (TypeError, ValueError) as e:
        return LoadOutcome(
            data=None,
            error=f"{json_path}: cannot build activation data: {e}",
        )


def load_mlp_activations_diagnostic(json_path: str) -> LoadOutcome:
    """Load MLP activation tensors with a precise failure diagnostic. Same schema
    as load_mlp_activations_from_json but returns a LoadOutcome carrying the cause."""
    data, err = _json_read(json_path)
    if err is not None:
        return LoadOutcome(data=None, error=err)
    if not isinstance(data, (dict, list)):
        return LoadOutcome(
            data=None,
            error=(
                f"expected JSON object or list in {json_path} but got "
                f"{type(data).__name__}; mlp_activations JSON must be "
                f'{{"layers": [{{"activations": [[[...]]], "layer_name": "..."}}]}} '
                f"or a list of such layers; "
            ),
        )
    layers_raw = data.get("layers", data) if isinstance(data, dict) else data
    if not isinstance(layers_raw, list):
        return LoadOutcome(
            data=None,
            error=(
                f"expected 'layers' to be a list in {json_path} but got "
                f"{type(layers_raw).__name__}; "
            ),
        )
    if not layers_raw:
        return LoadOutcome(
            data=None,
            error=(
                f"no valid MLP activation layers found in {json_path}; the 'layers' list is empty; "
            ),
        )
    layers: list[MLPActivations] = []
    for i, layer in enumerate(layers_raw):
        if not isinstance(layer, dict):
            return LoadOutcome(
                data=None,
                error=(
                    f"layer {i} in {json_path} is not a JSON object; "
                    f"expected keys 'activations' and 'layer_name'"
                ),
            )
        missing = [k for k in ("activations", "layer_name") if k not in layer]
        if missing:
            return LoadOutcome(
                data=None,
                error=(
                    f"layer {i} in {json_path} is missing required keys {missing}; "
                    f"found keys {list(layer.keys())}; expected 'activations' and 'layer_name'"
                ),
            )
        try:
            t = torch.tensor(layer["activations"], dtype=torch.float32)
        except (TypeError, ValueError) as e:
            return LoadOutcome(
                data=None,
                error=f"layer {i} in {json_path}: cannot convert activations to tensor: {e}",
            )
        if t.ndim != 3:
            return LoadOutcome(
                data=None,
                error=(
                    f"{json_path}: layer {layer['layer_name']!r} activations has "
                    f"shape {list(t.shape)} (ndim {t.ndim}); MLP activations must be "
                    f"3-D [batch, seq, hidden]"
                ),
            )
        layers.append(MLPActivations(activations=t, layer_name=layer["layer_name"]))
    return LoadOutcome(data=layers, error=None)


def load_expert_weights_diagnostic(json_path: str) -> LoadOutcome:
    """Load expert weights with a precise failure diagnostic. Same schema as
    load_expert_weights_from_json but returns a LoadOutcome carrying the cause."""
    data, err = _json_read(json_path)
    if err is not None:
        return LoadOutcome(data=None, error=err)
    if not isinstance(data, (dict, list)):
        return LoadOutcome(
            data=None,
            error=(
                f"expected JSON object or list in {json_path} but got "
                f"{type(data).__name__}; expert_weights JSON must be "
                f'{{"layers": [{{"expert_weights": [[...]], "layer_name": "..."}}]}} '
                f"or a list of such layers; "
            ),
        )
    layers_raw = data.get("layers", data) if isinstance(data, dict) else data
    if not isinstance(layers_raw, list):
        return LoadOutcome(
            data=None,
            error=(
                f"expected 'layers' to be a list in {json_path} but got "
                f"{type(layers_raw).__name__}; "
            ),
        )
    if not layers_raw:
        return LoadOutcome(
            data=None,
            error=(
                f"no valid expert weight layers found in {json_path}; the 'layers' list is empty; "
            ),
        )
    layers: list[ExpertWeights] = []
    for i, layer in enumerate(layers_raw):
        if not isinstance(layer, dict):
            return LoadOutcome(
                data=None,
                error=(
                    f"layer {i} in {json_path} is not a JSON object; "
                    f"expected keys 'expert_weights' and 'layer_name'"
                ),
            )
        missing = [k for k in ("expert_weights", "layer_name") if k not in layer]
        if missing:
            return LoadOutcome(
                data=None,
                error=(
                    f"layer {i} in {json_path} is missing required keys {missing}; "
                    f"found keys {list(layer.keys())}; expected 'expert_weights' and 'layer_name'"
                ),
            )
        weights_raw = layer["expert_weights"]
        if not isinstance(weights_raw, list) or len(weights_raw) < 2:
            return LoadOutcome(
                data=None,
                error=(
                    f"layer {i} in {json_path}: 'expert_weights' must be a list of "
                    f"at least 2 weight arrays; got {type(weights_raw).__name__} "
                    f"with {len(weights_raw) if isinstance(weights_raw, list) else 'N/A'} items"
                ),
            )
        try:
            weight_tensors = [torch.tensor(w, dtype=torch.float32) for w in weights_raw]
        except (TypeError, ValueError) as e:
            return LoadOutcome(
                data=None,
                error=f"layer {i} in {json_path}: cannot convert expert weights to tensors: {e}",
            )
        layers.append(
            ExpertWeights(expert_weights=weight_tensors, layer_name=layer["layer_name"])
        )
    return LoadOutcome(data=layers, error=None)


def load_gradient_provenance_diagnostic(json_path: str) -> LoadOutcome:
    """Load gradient provenance metadata with a precise failure diagnostic. Same
    schema as load_gradient_provenance_from_json but returns a LoadOutcome
    carrying the cause."""
    data, err = _json_read(json_path)
    if err is not None:
        return LoadOutcome(data=None, error=err)
    if not isinstance(data, dict):
        return LoadOutcome(
            data=None,
            error=(
                f"expected JSON object in {json_path} but got "
                f"{type(data).__name__}; gradient provenance JSON must be "
                f'{{"provenance": {{"source_type": "...", "verification_status": "..."}}}}'
            ),
        )
    prov_raw = data.get("provenance")
    if not isinstance(prov_raw, dict):
        return LoadOutcome(
            data=None,
            error=(
                f"{json_path} is missing the nested 'provenance' object; "
                f"found top-level keys {list(data.keys())}; expected canonical "
                f'format {{"provenance": {{...}}}} with valid '
                f'"source_type" and "verification_status" values'
            ),
        )
    source_type = prov_raw.get("source_type", "measured")
    verification_status = prov_raw.get("verification_status", "provenance_incomplete")
    if not isinstance(source_type, str) or source_type not in _ALLOWED_SOURCE_TYPES:
        return LoadOutcome(
            data=None,
            error=(
                f"{json_path}: 'source_type' must be one of "
                f"{sorted(_ALLOWED_SOURCE_TYPES)}; got {source_type!r}"
            ),
        )
    if (
        not isinstance(verification_status, str)
        or verification_status not in _ALLOWED_VERIFICATION_STATUSES
    ):
        return LoadOutcome(
            data=None,
            error=(
                f"{json_path}: 'verification_status' must be one of "
                f"{sorted(_ALLOWED_VERIFICATION_STATUSES)}; got {verification_status!r}"
            ),
        )
    try:
        return LoadOutcome(
            data=GradientProvenance(
                model_revision=prov_raw.get("model_revision"),
                dataset_hash=prov_raw.get("dataset_hash"),
                dataset_type=prov_raw.get("dataset_type"),
                token_count=prov_raw.get("token_count"),
                batch_count=prov_raw.get("batch_count"),
                loss_function=prov_raw.get("loss_function"),
                forward_dtype=prov_raw.get("forward_dtype"),
                backward_dtype=prov_raw.get("backward_dtype"),
                module_name=prov_raw.get("module_name"),
                parameter_name=prov_raw.get("parameter_name"),
                gradient_aggregation=prov_raw.get("gradient_aggregation"),
                seed=prov_raw.get("seed"),
                source_type=cast("SourceType", source_type),
                verification_status=cast("VerificationStatus", verification_status),
            ),
            error=None,
        )
    except (TypeError, ValueError) as e:
        return LoadOutcome(
            data=None, error=f"{json_path}: cannot build provenance: {e}"
        )
