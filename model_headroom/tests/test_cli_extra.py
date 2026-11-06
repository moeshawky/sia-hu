from __future__ import annotations

import json
from typing import Any

from model_headroom.cli import main


class TestCliIsoflop:
    def test_isoflop_loading(self, tmp_path: Any) -> None:
        isoflop_file = tmp_path / "isoflop.json"
        isoflop_file.write_text(
            json.dumps(
                [
                    {
                        "compute_flops": 1e18,
                        "model_params": 10000000,
                        "measured_loss": 2.0,
                    },
                    {
                        "compute_flops": 1e19,
                        "model_params": 100000000,
                        "measured_loss": 1.5,
                    },
                    {
                        "compute_flops": 1e20,
                        "model_params": 1000000000,
                        "measured_loss": 1.2,
                    },
                ]
            )
        )
        result = main(
            [
                "--model-name",
                "test",
                "--model-path",
                str(tmp_path),
                "--isoflop",
                str(isoflop_file),
                "--output-format",
                "json",
            ]
        )
        assert result == 0

    def test_bad_isoflop_file(self, tmp_path: Any) -> None:
        bad = tmp_path / "bad.json"
        bad.write_text("not json")
        result = main(
            [
                "--model-name",
                "test",
                "--model-path",
                str(tmp_path),
                "--isoflop",
                str(bad),
            ]
        )
        assert result == 1


class TestCliWeightPairsFile:
    def test_weight_pairs_loading(self, tmp_path: Any) -> None:
        wp_file = tmp_path / "wp.json"
        wp_file.write_text(
            json.dumps(
                {
                    "pairs": [
                        {
                            "gradient": [[1.0, 2.0], [3.0, 4.0]],
                            "weight": [[0.5, 0.5], [0.5, 0.5]],
                            "layer_name": "layer_0",
                        }
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
                "--output-format",
                "json",
            ]
        )
        assert result == 0

    def test_bad_weight_pairs_file(self, tmp_path: Any) -> None:
        bad = tmp_path / "bad.json"
        bad.write_text("not json")
        result = main(
            [
                "--model-name",
                "test",
                "--model-path",
                str(tmp_path),
                "--weight-pairs-file",
                str(bad),
            ]
        )
        assert result == 1


class TestCliMlpActivationsFile:
    def test_mlp_activations_loading(self, tmp_path: Any) -> None:
        mlp_file = tmp_path / "mlp.json"
        mlp_file.write_text(
            json.dumps(
                {
                    "layers": [
                        {
                            "activations": [[[1.0, 2.0], [3.0, 4.0]]],
                            "layer_name": "layer_0",
                        }
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
                "--mlp-activations-file",
                str(mlp_file),
                "--output-format",
                "json",
            ]
        )
        assert result == 0

    def test_bad_mlp_activations_file(self, tmp_path: Any) -> None:
        bad = tmp_path / "bad.json"
        bad.write_text("not json")
        result = main(
            [
                "--model-name",
                "test",
                "--model-path",
                str(tmp_path),
                "--mlp-activations-file",
                str(bad),
            ]
        )
        assert result == 1


class TestCliExpertWeightsFile:
    def test_expert_weights_loading(self, tmp_path: Any) -> None:
        ew_file = tmp_path / "ew.json"
        ew_file.write_text(
            json.dumps(
                {
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
            )
        )
        result = main(
            [
                "--model-name",
                "test",
                "--model-path",
                str(tmp_path),
                "--expert-weights-file",
                str(ew_file),
                "--output-format",
                "json",
            ]
        )
        assert result == 0

    def test_bad_expert_weights_file(self, tmp_path: Any) -> None:
        bad = tmp_path / "bad.json"
        bad.write_text("not json")
        result = main(
            [
                "--model-name",
                "test",
                "--model-path",
                str(tmp_path),
                "--expert-weights-file",
                str(bad),
            ]
        )
        assert result == 1


class TestCliHumanOutput:
    def test_human_report_output(self, tmp_path: Any) -> None:
        result = main(
            [
                "--model-name",
                "test",
                "--model-path",
                str(tmp_path),
                "--output-format",
                "human",
            ]
        )
        assert result == 0


class TestCliOutputFile:
    def test_output_file_writing(self, tmp_path: Any) -> None:
        out_file = tmp_path / "report.txt"
        result = main(
            [
                "--model-name",
                "test",
                "--model-path",
                str(tmp_path),
                "--output-file",
                str(out_file),
            ]
        )
        assert result == 0
        assert out_file.exists()
        content = out_file.read_text()
        assert "MODEL HEADROOM ANALYSIS REPORT" in content


class TestCliAnalysisException:
    def test_analysis_failure_returns_1(self, tmp_path: Any) -> None:
        from unittest.mock import patch

        with patch(
            "model_headroom.cli.run_analysis",
            side_effect=RuntimeError("boom"),
        ):
            result = main(
                [
                    "--model-name",
                    "test",
                    "--model-path",
                    str(tmp_path),
                ]
            )
        assert result == 1


class TestCliGGUFWeightSamples:
    def test_directory_returns_none_sampler_not_called(
        self, tmp_path: Any, monkeypatch: Any
    ) -> None:
        from model_headroom.cli import _gguf_weight_samples

        called: list[Any] = []

        def fake_sampler(*args: Any, **kwargs: Any) -> list[Any]:
            called.append(True)
            return []

        monkeypatch.setattr(
            "model_headroom.models.ternary_sampling.sample_weight_for_effective_rank",
            fake_sampler,
        )
        result = _gguf_weight_samples(str(tmp_path), "single_tile", None)
        assert result is None
        assert called == []

    def test_gguf_file_returns_weight_samples(
        self, tmp_path: Any, monkeypatch: Any
    ) -> None:
        from model_headroom.cli import _gguf_weight_samples
        from model_headroom.models.types import WeightSample

        fake_file = tmp_path / "model.gguf"
        fake_file.write_bytes(b"fake")

        class FakeConfig:
            block_count = 4

        monkeypatch.setattr(
            "model_headroom.models.gguf_loader.load_gguf_config",
            lambda path: FakeConfig(),
        )

        def fake_sampler(
            gguf_path: str, tensor_names: list[str], max_rows: int = 256
        ) -> list[tuple[str, Any, tuple[int, int]]]:
            import torch

            return [("blk.0.ffn_gate.weight", torch.ones(2, 3), (5120, 17408))]

        monkeypatch.setattr(
            "model_headroom.models.ternary_sampling.sample_weight_for_effective_rank",
            fake_sampler,
        )
        result = _gguf_weight_samples(str(fake_file), "single_tile", None)
        assert result is not None
        assert len(result) == 1
        ws = result[0]
        assert isinstance(ws, WeightSample)
        assert ws.sampling_mode == "single_tile"
        assert ws.truncated is True
        assert ws.parent_shape == (5120, 17408)

    def test_sampler_raises_prints_warning_returns_none(
        self, tmp_path: Any, monkeypatch: Any, capsys: Any
    ) -> None:
        from model_headroom.cli import _gguf_weight_samples
        from model_headroom.models.gguf_loader import _GGUFLoadError

        fake_file = tmp_path / "model.gguf"
        fake_file.write_bytes(b"fake")

        class FakeConfig:
            block_count = 4

        monkeypatch.setattr(
            "model_headroom.models.gguf_loader.load_gguf_config",
            lambda path: FakeConfig(),
        )

        def fake_sampler(*args: Any, **kwargs: Any) -> list[Any]:
            raise _GGUFLoadError(
                error_type="invalid_gguf", message="boom", recoverable=False
            )

        monkeypatch.setattr(
            "model_headroom.models.ternary_sampling.sample_weight_for_effective_rank",
            fake_sampler,
        )
        result = _gguf_weight_samples(str(fake_file), "single_tile", None)
        assert result is None
        err = capsys.readouterr().err
        assert "warning: cannot sample GGUF weights" in err

    def test_weight_pairs_file_wins_weight_samples_none(
        self, tmp_path: Any, monkeypatch: Any
    ) -> None:
        from model_headroom.cli import main

        wp_file = tmp_path / "wp.json"
        wp_file.write_text(
            json.dumps(
                {
                    "pairs": [
                        {
                            "gradient": [[1.0, 2.0], [3.0, 4.0]],
                            "weight": [[0.5, 0.5], [0.5, 0.5]],
                            "layer_name": "layer_0",
                        }
                    ]
                }
            )
        )
        fake_file = tmp_path / "model.gguf"
        fake_file.write_bytes(b"fake")

        class FakeConfig:
            block_count = 4

        monkeypatch.setattr(
            "model_headroom.models.gguf_loader.load_gguf_config",
            lambda path: FakeConfig(),
        )

        sampler_calls: list[Any] = []

        def fake_sampler(*args: Any, **kwargs: Any) -> list[Any]:
            sampler_calls.append((args, kwargs))
            return []

        monkeypatch.setattr(
            "model_headroom.models.ternary_sampling.sample_weight_for_effective_rank",
            fake_sampler,
        )

        captured: dict[str, Any] = {}

        class FakeReport:
            decision_grade = True

        def fake_run_analysis(**kwargs: Any) -> FakeReport:
            captured.update(kwargs)
            return FakeReport()

        monkeypatch.setattr("model_headroom.cli.run_analysis", fake_run_analysis)
        monkeypatch.setattr(
            "model_headroom.cli.generate_human_report", lambda report: "report"
        )

        result = main(
            [
                "--model-name",
                "test",
                "--model-path",
                str(fake_file),
                "--weight-pairs-file",
                str(wp_file),
                "--output-format",
                "human",
            ]
        )
        assert result == 0
        assert captured.get("weight_samples") is None
        assert sampler_calls == []
