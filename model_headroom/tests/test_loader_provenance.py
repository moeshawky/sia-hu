from __future__ import annotations

import json
from typing import Any

from model_headroom.models.loader import (
    detect_architecture,
    infer_source_type_from_path,
    load_gradient_provenance_from_json,
)


class TestInferSourceTypeFromPath:
    def test_provenance_dict_present(self, tmp_path: Any) -> None:
        p = tmp_path / "wp.json"
        p.write_text(
            json.dumps(
                {
                    "provenance": {
                        "source_type": "synthetic",
                        "verification_status": "unverified",
                    }
                }
            )
        )
        st, vs = infer_source_type_from_path(str(p))
        assert st == "synthetic"
        assert vs == "unverified"

    def test_source_type_present(self, tmp_path: Any) -> None:
        p = tmp_path / "wp.json"
        p.write_text(
            json.dumps({"source_type": "measured", "verification_status": "verified"})
        )
        st, vs = infer_source_type_from_path(str(p))
        assert st == "measured"
        assert vs == "verified"

    def test_invalid_source_type_returns_unknown(self, tmp_path: Any) -> None:
        p = tmp_path / "wp.json"
        p.write_text(json.dumps({"source_type": "bogus_value"}))
        st, vs = infer_source_type_from_path(str(p))
        assert st == "unknown"
        assert vs == "unverified"

    def test_file_not_found_returns_default(self) -> None:
        st, vs = infer_source_type_from_path("/nonexistent/file.json")
        assert st == "measured"
        assert vs == "provenance_incomplete"

    def test_malformed_json_returns_default(self, tmp_path: Any) -> None:
        p = tmp_path / "bad.json"
        p.write_text("not json at all")
        st, vs = infer_source_type_from_path(str(p))
        assert st == "measured"
        assert vs == "provenance_incomplete"

    def test_no_provenance_info_returns_default(self, tmp_path: Any) -> None:
        p = tmp_path / "wp.json"
        p.write_text(json.dumps({"pairs": []}))
        st, vs = infer_source_type_from_path(str(p))
        assert st == "measured"
        assert vs == "provenance_incomplete"

    def test_non_string_source_type(self, tmp_path: Any) -> None:
        p = tmp_path / "wp.json"
        p.write_text(json.dumps({"source_type": 42}))
        st, vs = infer_source_type_from_path(str(p))
        assert st == "unknown"
        assert vs == "unverified"

    def test_estimated_source_type(self, tmp_path: Any) -> None:
        p = tmp_path / "wp.json"
        p.write_text(json.dumps({"source_type": "estimated"}))
        st, vs = infer_source_type_from_path(str(p))
        assert st == "estimated"
        assert vs == "provenance_incomplete"


class TestLoadGradientProvenanceFromJson:
    def test_provenance_key_present(self, tmp_path: Any) -> None:
        p = tmp_path / "prov.json"
        p.write_text(
            json.dumps(
                {"provenance": {"model_revision": "rev1", "dataset_hash": "abc"}}
            )
        )
        result = load_gradient_provenance_from_json(str(p))
        assert result is not None
        assert result.model_revision == "rev1"
        assert result.dataset_hash == "abc"

    def test_defaults_applied(self, tmp_path: Any) -> None:
        p = tmp_path / "prov.json"
        p.write_text(json.dumps({"provenance": {"model_revision": "rev1"}}))
        result = load_gradient_provenance_from_json(str(p))
        assert result is not None
        assert result.source_type == "measured"
        assert result.verification_status == "provenance_incomplete"

    def test_top_level_source_type(self, tmp_path: Any) -> None:
        p = tmp_path / "prov.json"
        p.write_text(
            json.dumps(
                {
                    "provenance": {
                        "source_type": "synthetic",
                        "verification_status": "unverified",
                    }
                }
            )
        )
        result = load_gradient_provenance_from_json(str(p))
        assert result is not None
        assert result.source_type == "synthetic"
        assert result.verification_status == "unverified"

    def test_file_not_found_returns_none(self) -> None:
        assert load_gradient_provenance_from_json("/nonexistent/file.json") is None

    def test_malformed_json_returns_none(self, tmp_path: Any) -> None:
        p = tmp_path / "bad.json"
        p.write_text("not json")
        assert load_gradient_provenance_from_json(str(p)) is None

    def test_missing_provenance_key_returns_none(self, tmp_path: Any) -> None:
        p = tmp_path / "wp.json"
        p.write_text(json.dumps({"pairs": []}))
        assert load_gradient_provenance_from_json(str(p)) is None

    def test_provenance_not_dict_returns_none(self, tmp_path: Any) -> None:
        p = tmp_path / "prov.json"
        p.write_text(json.dumps({"provenance": "string"}))
        assert load_gradient_provenance_from_json(str(p)) is None

    def test_all_fields_populated(self, tmp_path: Any) -> None:
        p = tmp_path / "prov.json"
        data = {
            "provenance": {
                "model_revision": "abc123",
                "dataset_hash": "def456",
                "dataset_type": "web",
                "token_count": 1000000,
                "batch_count": 100,
                "loss_function": "cross_entropy",
                "forward_dtype": "fp32",
                "backward_dtype": "fp32",
                "module_name": "layers.0.mlp",
                "parameter_name": "weight",
                "gradient_aggregation": "mean",
                "seed": 42,
                "source_type": "measured",
                "verification_status": "verified",
            }
        }
        p.write_text(json.dumps(data))
        result = load_gradient_provenance_from_json(str(p))
        assert result is not None
        assert result.model_revision == "abc123"
        assert result.dataset_hash == "def456"
        assert result.token_count == 1000000
        assert result.seed == 42
        assert result.source_type == "measured"
        assert result.verification_status == "verified"


class TestDetectArchitectureEdges:
    def test_gguf_format_detection(self, tmp_path: Any) -> None:
        gguf_path = tmp_path / "model.gguf"
        gguf_path.write_bytes(b"GGUF" * 10)
        assert detect_architecture(str(gguf_path)) == "unknown"

    def test_directory_without_config_json(self, tmp_path: Any) -> None:
        assert detect_architecture(str(tmp_path)) == "unknown"

    def test_hf_dir_with_config_returns_type(self, tmp_path: Any) -> None:
        (tmp_path / "config.json").write_text(
            json.dumps({"model_type": "mixtral", "num_experts": 8})
        )
        assert detect_architecture(str(tmp_path)) == "moe"

    def test_hf_dir_dense(self, tmp_path: Any) -> None:
        (tmp_path / "config.json").write_text(json.dumps({"model_type": "llama"}))
        assert detect_architecture(str(tmp_path)) == "dense"
