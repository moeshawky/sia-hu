from __future__ import annotations

import json
from typing import Any

from model_headroom.cli import main


class TestCliRankParentShapes:
    def test_valid_parent_shapes(self, tmp_path: Any) -> None:
        shapes_file = tmp_path / "shapes.json"
        shapes_file.write_text(
            json.dumps({"layer_0": [512, 1024], "layer_1": [256, 512]})
        )
        result = main(
            [
                "--model-name",
                "test",
                "--model-path",
                str(tmp_path),
                "--rank-parent-shapes",
                str(shapes_file),
                "--output-format",
                "json",
            ]
        )
        assert result == 0

    def test_invalid_parent_shapes_json(self, tmp_path: Any) -> None:
        shapes_file = tmp_path / "shapes.json"
        shapes_file.write_text("not json")
        result = main(
            [
                "--model-name",
                "test",
                "--model-path",
                str(tmp_path),
                "--rank-parent-shapes",
                str(shapes_file),
            ]
        )
        assert result == 1

    def test_missing_parent_shapes_file(self, tmp_path: Any) -> None:
        result = main(
            [
                "--model-name",
                "test",
                "--model-path",
                str(tmp_path),
                "--rank-parent-shapes",
                "/nonexistent/shapes.json",
            ]
        )
        assert result == 1


class TestCliGradientProvenanceFile:
    def test_valid_provenance_file(self, tmp_path: Any) -> None:
        prov_file = tmp_path / "prov.json"
        prov_file.write_text(
            json.dumps(
                {
                    "provenance": {
                        "model_revision": "abc",
                        "dataset_hash": "def",
                        "source_type": "measured",
                        "verification_status": "verified",
                    }
                }
            )
        )
        wp_file = tmp_path / "wp.json"
        wp_file.write_text(
            json.dumps(
                {
                    "pairs": [
                        {"gradient": [[1.0]], "weight": [[1.0]], "layer_name": "l0"}
                    ]
                }
            )
        )
        result = main(
            [
                "--model-name",
                "test",
                "--model-path",
                str(tmp_path),
                "--gradient-provenance-file",
                str(prov_file),
                "--weight-pairs-file",
                str(wp_file),
                "--output-format",
                "json",
            ]
        )
        assert result == 0

    def test_invalid_provenance_file(self, tmp_path: Any) -> None:
        prov_file = tmp_path / "prov.json"
        prov_file.write_text("not json")
        result = main(
            [
                "--model-name",
                "test",
                "--model-path",
                str(tmp_path),
                "--gradient-provenance-file",
                str(prov_file),
            ]
        )
        assert result == 1

    def test_missing_provenance_file(self, tmp_path: Any) -> None:
        result = main(
            [
                "--model-name",
                "test",
                "--model-path",
                str(tmp_path),
                "--gradient-provenance-file",
                "/nonexistent/prov.json",
            ]
        )
        assert result == 1


class TestCliWeightPairsSourceTypeFallback:
    def test_source_type_fallback_creates_provenance(self, tmp_path: Any) -> None:
        wp_file = tmp_path / "wp.json"
        wp_file.write_text(
            json.dumps(
                {
                    "pairs": [
                        {"gradient": [[1.0]], "weight": [[1.0]], "layer_name": "l0"}
                    ]
                }
            )
        )
        result = main(
            [
                "--model-name",
                "test",
                "--model-path",
                str(tmp_path),
                "--weight-pairs-file",
                str(wp_file),
                "--weight-pairs-source-type",
                "synthetic",
                "--output-format",
                "json",
            ]
        )
        assert result == 0

    def test_measured_source_type_fallback(self, tmp_path: Any) -> None:
        wp_file = tmp_path / "wp.json"
        wp_file.write_text(
            json.dumps(
                {
                    "pairs": [
                        {"gradient": [[1.0]], "weight": [[1.0]], "layer_name": "l0"}
                    ]
                }
            )
        )
        result = main(
            [
                "--model-name",
                "test",
                "--model-path",
                str(tmp_path),
                "--weight-pairs-file",
                str(wp_file),
                "--weight-pairs-source-type",
                "measured",
                "--output-format",
                "json",
            ]
        )
        assert result == 0


class TestCliMlpSourceTypeInference:
    def test_explicit_mlp_source_type(self, tmp_path: Any) -> None:
        mlp_file = tmp_path / "mlp.json"
        mlp_file.write_text(
            json.dumps(
                {"layers": [{"activations": [[[1.0, 2.0]]], "layer_name": "l0"}]}
            )
        )
        result = main(
            [
                "--model-name",
                "test",
                "--model-path",
                str(tmp_path),
                "--mlp-activations-file",
                str(mlp_file),
                "--mlp-source-type",
                "synthetic",
                "--output-format",
                "json",
            ]
        )
        assert result == 0

    def test_mlp_source_type_inferred_from_file(self, tmp_path: Any) -> None:
        mlp_file = tmp_path / "mlp.json"
        mlp_file.write_text(
            json.dumps(
                {
                    "source_type": "estimated",
                    "layers": [{"activations": [[[1.0, 2.0]]], "layer_name": "l0"}],
                }
            )
        )
        result = main(
            [
                "--model-name",
                "test",
                "--model-path",
                str(tmp_path),
                "--mlp-activations-file",
                str(mlp_file),
                "--output-format",
                "json",
            ]
        )
        assert result == 0


class TestCliOutputFile:
    def test_json_output_to_file(self, tmp_path: Any) -> None:
        out_file = tmp_path / "report.json"
        result = main(
            [
                "--model-name",
                "test",
                "--model-path",
                str(tmp_path),
                "--output-format",
                "json",
                "--output-file",
                str(out_file),
            ]
        )
        assert result == 0
        assert out_file.exists()
        content = out_file.read_text()
        assert "model_name" in content

    def test_human_output_to_file(self, tmp_path: Any) -> None:
        out_file = tmp_path / "report.txt"
        result = main(
            [
                "--model-name",
                "test",
                "--model-path",
                str(tmp_path),
                "--output-format",
                "human",
                "--output-file",
                str(out_file),
            ]
        )
        assert result == 0
        assert out_file.exists()
        assert "MODEL HEADROOM ANALYSIS REPORT" in out_file.read_text()


class TestCliDecisionGradeNotMet:
    def test_decision_grade_not_met_returns_2(self, tmp_path: Any) -> None:
        result = main(
            [
                "--model-name",
                "test",
                "--model-path",
                str(tmp_path),
                "--analysis-mode",
                "decision_grade",
                "--output-format",
                "json",
            ]
        )
        assert result == 2
