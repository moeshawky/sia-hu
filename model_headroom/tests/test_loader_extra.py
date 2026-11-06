from __future__ import annotations

import json
from typing import Any

from model_headroom.models.loader import (
    extract_layer_weights,
    load_expert_weights_from_json,
    load_mlp_activations_from_json,
    load_weight_pairs_from_json,
)


class TestExtractLayerWeights:
    def test_bin_file(self, tmp_path: Any) -> None:
        import torch

        state_dict = {"model.layers.0.weight": torch.randn(4, 8)}
        bin_path = tmp_path / "model.bin"
        torch.save(state_dict, bin_path)
        result = extract_layer_weights(str(tmp_path), "model.layers.0.weight")
        assert result is not None

    def test_safetensor_file(self, tmp_path: Any) -> None:
        import torch

        state_dict = {"model.layers.0.weight": torch.randn(4, 8)}
        sf_path = tmp_path / "model.safetensors"
        from safetensors.torch import save_file

        save_file(state_dict, str(sf_path))
        result = extract_layer_weights(str(tmp_path), "model.layers.0.weight")
        assert result is not None

    def test_layer_not_found(self, tmp_path: Any) -> None:
        import torch

        state_dict = {"model.layers.0.weight": torch.randn(4, 8)}
        bin_path = tmp_path / "model.bin"
        torch.save(state_dict, bin_path)
        result = extract_layer_weights(str(tmp_path), "model.layers.1.weight")
        assert result is None

    def test_no_weight_files(self, tmp_path: Any) -> None:
        result = extract_layer_weights(str(tmp_path), "any.weight")
        assert result is None


class TestLoadWeightPairsNonList:
    def test_non_list_data(self, tmp_path: Any) -> None:
        p = tmp_path / "wp.json"
        p.write_text(json.dumps({"pairs": "not a list"}))
        result = load_weight_pairs_from_json(str(p))
        assert result is None

    def test_all_pairs_missing_keys(self, tmp_path: Any) -> None:
        p = tmp_path / "wp.json"
        p.write_text(json.dumps({"pairs": [{"no_gradient": True}]}))
        result = load_weight_pairs_from_json(str(p))
        assert result is None


class TestLoadMlpActivationsMissingKeys:
    def test_layer_missing_activations(self, tmp_path: Any) -> None:
        p = tmp_path / "mlp.json"
        p.write_text(json.dumps({"layers": [{"layer_name": "l0"}]}))
        result = load_mlp_activations_from_json(str(p))
        assert result is None

    def test_layer_missing_layer_name(self, tmp_path: Any) -> None:
        p = tmp_path / "mlp.json"
        p.write_text(json.dumps({"layers": [{"activations": [[[1.0, 2.0]]]}]}))
        result = load_mlp_activations_from_json(str(p))
        assert result is None

    def test_wrong_ndim(self, tmp_path: Any) -> None:
        p = tmp_path / "mlp.json"
        p.write_text(
            json.dumps({"layers": [{"activations": [1.0, 2.0], "layer_name": "l0"}]})
        )
        result = load_mlp_activations_from_json(str(p))
        assert result is None

    def test_non_list_layers(self, tmp_path: Any) -> None:
        p = tmp_path / "mlp.json"
        p.write_text(json.dumps({"layers": "not a list"}))
        result = load_mlp_activations_from_json(str(p))
        assert result is None


class TestLoadExpertWeightsNonList:
    def test_non_list_data(self, tmp_path: Any) -> None:
        p = tmp_path / "ew.json"
        p.write_text(json.dumps({"layers": "not a list"}))
        result = load_expert_weights_from_json(str(p))
        assert result is None

    def test_layer_missing_expert_weights(self, tmp_path: Any) -> None:
        p = tmp_path / "ew.json"
        p.write_text(json.dumps({"layers": [{"layer_name": "l0"}]}))
        result = load_expert_weights_from_json(str(p))
        assert result is None

    def test_layer_missing_layer_name(self, tmp_path: Any) -> None:
        p = tmp_path / "ew.json"
        p.write_text(
            json.dumps(
                {
                    "layers": [
                        {
                            "expert_weights": [
                                [[1.0, 2.0], [3.0, 4.0]],
                                [[5.0, 6.0], [7.0, 8.0]],
                            ]
                        }
                    ]
                }
            )
        )
        result = load_expert_weights_from_json(str(p))
        assert result is None

    def test_single_expert(self, tmp_path: Any) -> None:
        p = tmp_path / "ew.json"
        p.write_text(
            json.dumps(
                {
                    "layers": [
                        {
                            "expert_weights": [[[1.0, 2.0], [3.0, 4.0]]],
                            "layer_name": "l0",
                        }
                    ]
                }
            )
        )
        result = load_expert_weights_from_json(str(p))
        assert result is None
