from __future__ import annotations

import json
from typing import Any

import pytest

from model_headroom.cli import main, parse_args


class TestParseArgs:
    def test_valid_args(self) -> None:
        args = parse_args(["--model-name", "test", "--model-path", "/tmp/model"])
        assert args.model_name == "test"
        assert args.model_path == "/tmp/model"
        assert args.output_format == "human"

    def test_json_format(self) -> None:
        args = parse_args(
            [
                "--model-name",
                "test",
                "--model-path",
                "/tmp",
                "--output-format",
                "json",
            ]
        )
        assert args.output_format == "json"

    def test_missing_required(self) -> None:
        args = parse_args([])
        assert args.model_name is None
        assert args.model_path is None
        assert args.model_id is None

    def test_model_id_without_name_path(self) -> None:
        args = parse_args(["--model-id", "Qwen/Qwen3-4B"])
        assert args.model_id == "Qwen/Qwen3-4B"
        assert args.model_name is None
        assert args.model_path is None

    def test_invalid_output_format(self) -> None:
        with pytest.raises(SystemExit) as exc_info:
            parse_args(
                [
                    "--model-name",
                    "x",
                    "--model-path",
                    "/tmp",
                    "--output-format",
                    "xml",
                ]
            )
        assert exc_info.value.code == 2


class TestMain:
    def test_returns_0_on_success(self, tmp_path: Any) -> None:
        loss_file = tmp_path / "loss.json"
        loss_file.write_text(
            json.dumps(
                [
                    {"tokens_trained": 100000000, "validation_loss": 2.0},
                    {"tokens_trained": 200000000, "validation_loss": 1.5},
                    {"tokens_trained": 300000000, "validation_loss": 1.0},
                    {"tokens_trained": 400000000, "validation_loss": 0.5},
                    {"tokens_trained": 500000000, "validation_loss": 0.3},
                ]
            )
        )
        result = main(
            [
                "--model-name",
                "test",
                "--model-path",
                str(tmp_path),
                "--loss-curve",
                str(loss_file),
                "--output-format",
                "json",
            ]
        )
        assert result == 0

    def test_returns_1_on_bad_loss_file(self, tmp_path: Any) -> None:
        bad_file = tmp_path / "bad.json"
        bad_file.write_text("not json")
        result = main(
            [
                "--model-name",
                "test",
                "--model-path",
                str(tmp_path),
                "--loss-curve",
                str(bad_file),
            ]
        )
        assert result == 1

    def test_returns_1_on_missing_activations(self, tmp_path: Any) -> None:
        result = main(
            [
                "--model-name",
                "test",
                "--model-path",
                str(tmp_path),
                "--activations",
                "/nonexistent/file.json",
            ]
        )
        assert result == 1
