from __future__ import annotations

import json
import pickle
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
    detection, layer_types for attention type, weight map for vision and MTP markers.
    Catches all exceptions and returns unknown on failure."""
    try:
        fmt = _detect_model_format(model_path)
        if fmt == "gguf":
            from model_headroom.models.gguf_loader import load_gguf_config

            config = load_gguf_config(Path(model_path))
            has_moe = config.expert_count >= 2
            return ArchitectureDetails(
                family="moe" if has_moe else "dense",
                subtype="dense_standard",
                has_moe=has_moe,
                has_full_attention=False,
                has_linear_attention=False,
                has_vision_tower=False,
                has_mtp=False,
                detection_source="gguf_config",
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

        subtype: Literal["hybrid_linear_attention", "dense_standard", "unknown"] = (
            "hybrid_linear_attention" if (has_linear or has_full) else "dense_standard"
        )

        return ArchitectureDetails(
            family=family,
            subtype=subtype,
            has_moe=has_moe,
            has_full_attention=has_full,
            has_linear_attention=has_linear,
            has_vision_tower=has_vision,
            has_mtp=has_mtp,
            detection_source="config.json+index",
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
