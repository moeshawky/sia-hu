"""Dataset download and conversion for model headroom analysis.

Downloads dataset files from Hugging Face Hub (repo_type="dataset")
and converts them to the JSON schemas expected by model_headroom.models.loader.
"""

from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

from pathlib import Path

from huggingface_hub import hf_hub_download

DEFAULT_DATA_DIR = Path.home() / ".cache" / "headroom" / "datasets"

WEIGHT_PAIRS_FILE = "weight_pairs.json"
MLP_ACTIVATIONS_FILE = "mlp_activations.json"
EXPERT_ACTIVATIONS_FILE = "expert_activations.json"
EXPERT_WEIGHTS_FILE = "expert_weights.json"


def resolve_data_dir(data_dir: str | None = None) -> Path:
    """Resolve the data directory from flag, env var, or default."""
    if data_dir is not None:
        return Path(data_dir)
    env_dir = os.environ.get("HEADROOM_DATA_DIR")
    if env_dir:
        return Path(env_dir)
    return DEFAULT_DATA_DIR


def _dataset_repo_id(model_id: str) -> str:
    """Derive the HF dataset repo id from a model id."""
    return f"{model_id}-headroom"


def _try_download_json(
    repo_id: str,
    filename: str,
    directory: Path,
) -> Path | None:
    """Try to download a single JSON file from a dataset repo.

    Returns the local file path on success, None on any failure
    (missing repo, network error, authentication, etc.).
    """
    try:
        local = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type="dataset",
            local_dir=str(directory),
        )
        return Path(local)
    except Exception:
        return None


def _validate_weight_pairs(data: Any) -> bool:
    """Validate weight_pairs JSON schema: dict with 'pairs' list, each pair
    has 'gradient', 'weight', 'layer_name' keys."""
    if not isinstance(data, dict):
        return False
    pairs = data.get("pairs", data)
    if not isinstance(pairs, list) or not pairs:
        return False
    for p in pairs:
        if not isinstance(p, dict):
            return False
        if not all(k in p for k in ("gradient", "weight", "layer_name")):
            return False
    return True


def _validate_mlp_activations(data: Any) -> bool:
    """Validate mlp_activations JSON schema: dict with 'layers' list, each
    layer has 'activations' and 'layer_name' keys."""
    if not isinstance(data, dict):
        return False
    layers = data.get("layers", data)
    if not isinstance(layers, list) or not layers:
        return False
    for layer in layers:
        if not isinstance(layer, dict):
            return False
        if "activations" not in layer or "layer_name" not in layer:
            return False
    return True


def _validate_expert_activations(data: Any) -> bool:
    """Validate expert_activations JSON schema: dict with 'metadata' and
    'layers' keys."""
    if not isinstance(data, dict):
        return False
    return "metadata" in data and "layers" in data


def _validate_expert_weights(data: Any) -> bool:
    """Validate expert_weights JSON schema: dict with 'layers' list, each
    layer has 'expert_weights' (list >= 2) and 'layer_name' keys."""
    if not isinstance(data, dict):
        return False
    layers = data.get("layers", data)
    if not isinstance(layers, list) or not layers:
        return False
    for layer in layers:
        if not isinstance(layer, dict):
            return False
        if "expert_weights" not in layer or "layer_name" not in layer:
            return False
        if (
            not isinstance(layer["expert_weights"], list)
            or len(layer["expert_weights"]) < 2
        ):
            return False
    return True


def _load_json_file(path: Path) -> Any:
    """Load and parse a JSON file. Returns None on any error."""
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def _download_or_synthetic(
    repo_id: str,
    filename: str,
    directory: Path,
    model_id: str,
    validate_fn: Callable[[Any], bool],
    synthetic_fn: Callable[[Path, str], Path],
) -> Path:
    """Try to download a JSON file, validate its schema, and fall back to
    synthetic data if download fails or validation fails."""
    downloaded = _try_download_json(repo_id, filename, directory)
    if downloaded is not None:
        data = _load_json_file(downloaded)
        if data is not None and validate_fn(data):
            return downloaded
    return synthetic_fn(directory, model_id)


def _generate_synthetic_weight_pairs(directory: Path, model_id: str) -> Path:
    """Generate synthetic weight_pairs.json in the loader schema.

    Produces multiple pairs across attention and MLP layers with varied
    tensor shapes and optional schema fields (parent_shape, sampling_mode,
    truncated) so downstream signals exercise the full loader contract.
    """
    pairs = [
        {
            "gradient": [[0.1, -0.2], [0.3, 0.4]],
            "weight": [[1.0, 2.0], [3.0, 4.0]],
            "layer_name": "model.layers.0.self_attn.q_proj.weight",
            "parent_shape": [4096, 4096],
            "sampling_mode": "full_matrix",
            "truncated": False,
        },
        {
            "gradient": [[-0.1, 0.05], [0.2, -0.15]],
            "weight": [[0.5, -1.0], [2.0, 1.5]],
            "layer_name": "model.layers.0.self_attn.k_proj.weight",
            "parent_shape": [4096, 4096],
            "sampling_mode": "full_matrix",
            "truncated": False,
        },
        {
            "gradient": [[0.05, -0.1], [0.15, 0.2]],
            "weight": [[1.0, 0.5], [0.25, 0.75]],
            "layer_name": "model.layers.0.self_attn.v_proj.weight",
            "parent_shape": [4096, 4096],
            "sampling_mode": "randomized_svd",
            "truncated": True,
        },
        {
            "gradient": [[0.2, -0.3], [0.4, 0.1]],
            "weight": [[2.0, 1.0], [0.5, 1.5]],
            "layer_name": "model.layers.0.mlp.gate_proj.weight",
            "parent_shape": [11008, 4096],
            "sampling_mode": "single_tile",
            "truncated": False,
        },
    ]
    out = directory / WEIGHT_PAIRS_FILE
    out.write_text(json.dumps({"pairs": pairs}, indent=2))
    return out


def _generate_synthetic_mlp_activations(directory: Path, model_id: str) -> Path:
    """Generate synthetic mlp_activations.json in the loader schema.

    Produces multiple layers with varied 3-D activation tensor shapes and
    source_type provenance so downstream signals exercise the full loader
    contract.
    """
    layers = [
        {
            "activations": [[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]],
            "layer_name": "model.layers.0.mlp.gate_proj",
            "source_type": "synthetic",
        },
        {
            "activations": [[[0.5, 0.6, 0.7], [0.8, 0.9, 1.0], [1.1, 1.2, 1.3]]],
            "layer_name": "model.layers.0.mlp.up_proj",
            "source_type": "synthetic",
        },
        {
            "activations": [[[-0.1, 0.0, 0.1], [0.2, -0.2, 0.3]]],
            "layer_name": "model.layers.1.mlp.down_proj",
            "source_type": "synthetic",
        },
    ]
    out = directory / MLP_ACTIVATIONS_FILE
    out.write_text(json.dumps({"layers": layers}, indent=2))
    return out


def _generate_synthetic_expert_activations(directory: Path, model_id: str) -> Path:
    """Generate synthetic expert_activations.json in the loader schema.

    Produces multiple layers with expert activation counts and metadata so
    downstream signals exercise the full loader contract.
    """
    data = {
        "metadata": {
            "num_experts": 8,
            "num_experts_architectural": 8,
            "source_type": "synthetic",
        },
        "layers": {
            "model.layers.0.mlp.experts": {
                "0": 100,
                "1": 95,
                "2": 105,
                "3": 90,
                "4": 110,
                "5": 85,
                "6": 100,
                "7": 115,
            },
            "model.layers.1.mlp.experts": {
                "0": 98,
                "1": 102,
                "2": 94,
                "3": 106,
                "4": 88,
                "5": 112,
                "6": 90,
                "7": 110,
            },
        },
    }
    out = directory / EXPERT_ACTIVATIONS_FILE
    out.write_text(json.dumps(data, indent=2))
    return out


def _generate_synthetic_expert_weights(directory: Path, model_id: str) -> Path:
    """Generate synthetic expert_weights.json in the loader schema.

    Produces multiple layers with varied expert weight matrices and
    source_type provenance so downstream signals exercise the full loader
    contract.
    """
    layers = [
        {
            "expert_weights": [
                [[0.1, 0.2], [0.3, 0.4]],
                [[0.2, 0.1], [0.4, 0.3]],
                [[0.15, 0.25], [0.35, 0.45]],
                [[0.05, 0.15], [0.25, 0.35]],
                [[0.12, 0.22], [0.32, 0.42]],
                [[0.08, 0.18], [0.28, 0.38]],
                [[0.11, 0.21], [0.31, 0.41]],
                [[0.09, 0.19], [0.29, 0.39]],
            ],
            "layer_name": "model.layers.0.mlp.experts",
            "source_type": "synthetic",
        },
        {
            "expert_weights": [
                [[0.3, 0.1], [0.4, 0.2]],
                [[0.1, 0.3], [0.2, 0.4]],
                [[0.25, 0.35], [0.45, 0.55]],
                [[0.15, 0.25], [0.35, 0.45]],
                [[0.22, 0.32], [0.42, 0.52]],
                [[0.18, 0.28], [0.38, 0.48]],
                [[0.21, 0.31], [0.41, 0.51]],
                [[0.19, 0.29], [0.39, 0.49]],
            ],
            "layer_name": "model.layers.1.mlp.experts",
            "source_type": "synthetic",
        },
    ]
    out = directory / EXPERT_WEIGHTS_FILE
    out.write_text(json.dumps({"layers": layers}, indent=2))
    return out


def download_dataset_files(
    model_id: str,
    dataset_family: str,
    data_dir: str | None = None,
) -> dict[str, Path]:
    """Download dataset files for a model from Hugging Face Hub.

    Tries hf_hub_download from {model-id}-headroom dataset repo, falls back
    to generating synthetic data in the correct JSON format. Downloaded files
    are validated against loader schemas; invalid downloads fall back to
    synthetic data.
    Returns dict mapping data type to local JSON file path.
    """
    directory = resolve_data_dir(data_dir)
    directory.mkdir(parents=True, exist_ok=True)

    repo_id = _dataset_repo_id(model_id)
    result: dict[str, Path] = {}

    weight_pairs = _download_or_synthetic(
        repo_id,
        WEIGHT_PAIRS_FILE,
        directory,
        model_id,
        _validate_weight_pairs,
        _generate_synthetic_weight_pairs,
    )
    result["weight_pairs"] = weight_pairs

    mlp_acts = _download_or_synthetic(
        repo_id,
        MLP_ACTIVATIONS_FILE,
        directory,
        model_id,
        _validate_mlp_activations,
        _generate_synthetic_mlp_activations,
    )
    result["mlp_activations"] = mlp_acts

    if dataset_family in ("moe", "auto"):
        expert_acts = _download_or_synthetic(
            repo_id,
            EXPERT_ACTIVATIONS_FILE,
            directory,
            model_id,
            _validate_expert_activations,
            _generate_synthetic_expert_activations,
        )
        result["expert_activations"] = expert_acts

        expert_weights = _download_or_synthetic(
            repo_id,
            EXPERT_WEIGHTS_FILE,
            directory,
            model_id,
            _validate_expert_weights,
            _generate_synthetic_expert_weights,
        )
        result["expert_weights"] = expert_weights

    return result
