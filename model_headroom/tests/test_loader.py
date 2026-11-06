from __future__ import annotations

from typing import Any

from model_headroom.models.loader import (
    load_activations_from_json,
    load_expert_weights_from_json,
    load_mlp_activations_from_json,
    load_weight_pairs_from_json,
)


class TestLoadActivationsFromJson:
    def test_valid_json(self, tmp_json_file: Any) -> None:
        data = {
            "metadata": {"num_experts": 4, "num_layers": 2},
            "layers": {"0": {"0": 25, "1": 25, "2": 25, "3": 25}},
        }
        path = tmp_json_file(data)
        result = load_activations_from_json(path)
        assert result is not None
        assert result.metadata["num_experts"] == 4

    def test_missing_file(self) -> None:
        assert load_activations_from_json("/nonexistent/file.json") is None

    def test_malformed_json(self, tmp_path: Any) -> None:
        p = tmp_path / "bad.json"
        p.write_text("{invalid json")
        assert load_activations_from_json(str(p)) is None

    def test_missing_keys(self, tmp_json_file: Any) -> None:
        path = tmp_json_file({"wrong_key": "value"})
        assert load_activations_from_json(path) is None


class TestLoadWeightPairsFromJson:
    def test_valid_json(self, tmp_json_file: Any) -> None:
        data = {
            "pairs": [
                {
                    "gradient": [[1.0, 2.0], [3.0, 4.0]],
                    "weight": [[0.5, 0.5], [0.5, 0.5]],
                    "layer_name": "layer_0",
                }
            ]
        }
        path = tmp_json_file(data)
        result = load_weight_pairs_from_json(path)
        assert result is not None
        assert len(result) == 1
        assert result[0].layer_name == "layer_0"

    def test_missing_file(self) -> None:
        assert load_weight_pairs_from_json("/nonexistent/file.json") is None

    def test_malformed_json(self, tmp_path: Any) -> None:
        p = tmp_path / "bad.json"
        p.write_text("not json")
        assert load_weight_pairs_from_json(str(p)) is None

    def test_missing_keys(self, tmp_json_file: Any) -> None:
        path = tmp_json_file({"pairs": [{"no_gradient": True}]})
        result = load_weight_pairs_from_json(path)
        assert result is None


class TestLoadMlpActivationsFromJson:
    def test_valid_json(self, tmp_json_file: Any) -> None:
        data = {
            "layers": [
                {
                    "activations": [[[1.0, 2.0], [3.0, 4.0]]],
                    "layer_name": "layer_0",
                }
            ]
        }
        path = tmp_json_file(data)
        result = load_mlp_activations_from_json(path)
        assert result is not None
        assert len(result) == 1
        assert result[0].activations.ndim == 3

    def test_2d_activations_filtered(self, tmp_json_file: Any) -> None:
        data = {
            "layers": [
                {
                    "activations": [[1.0, 2.0, 3.0]],
                    "layer_name": "bad",
                }
            ]
        }
        path = tmp_json_file(data)
        result = load_mlp_activations_from_json(path)
        assert result is None

    def test_missing_file(self) -> None:
        assert load_mlp_activations_from_json("/nonexistent/file.json") is None


class TestLoadExpertWeightsFromJson:
    def test_valid_json(self, tmp_json_file: Any) -> None:
        data = {
            "layers": [
                {
                    "expert_weights": [
                        [[1.0, 2.0], [3.0, 4.0]],
                        [[5.0, 6.0], [7.0, 8.0]],
                    ],
                    "layer_name": "layer_0",
                }
            ]
        }
        path = tmp_json_file(data)
        result = load_expert_weights_from_json(path)
        assert result is not None
        assert len(result) == 1
        assert len(result[0].expert_weights) == 2

    def test_single_expert_filtered(self, tmp_json_file: Any) -> None:
        data = {
            "layers": [
                {
                    "expert_weights": [[[1.0, 2.0], [3.0, 4.0]]],
                    "layer_name": "single",
                }
            ]
        }
        path = tmp_json_file(data)
        result = load_expert_weights_from_json(path)
        assert result is None

    def test_missing_file(self) -> None:
        assert load_expert_weights_from_json("/nonexistent/file.json") is None
