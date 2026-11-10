from __future__ import annotations

import io
import json
from contextlib import redirect_stderr
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
        result = _gguf_weight_samples(str(tmp_path), "single_tile", None, "cpu")
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
            gguf_path: str, tensor_names: list[str], max_rows: int = 256, **kwargs: Any
        ) -> list[tuple[str, Any, tuple[int, int]]]:
            import torch

            return [("blk.0.ffn_gate.weight", torch.ones(2, 3), (5120, 17408))]

        monkeypatch.setattr(
            "model_headroom.models.ternary_sampling.sample_weight_for_effective_rank",
            fake_sampler,
        )
        result = _gguf_weight_samples(str(fake_file), "single_tile", None, "cpu")
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
        result = _gguf_weight_samples(str(fake_file), "single_tile", None, "cpu")
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


class TestCliDataFileErrorMessages:
    """The CLI must print an instructive error (cause + path + flag + expected
    format + example location) for every data-file failure, never a bare path."""

    def _err(self, tmp_path: Any, *extra: str) -> str:
        args = [
            "--model-name",
            "test",
            "--model-path",
            str(tmp_path),
            "--output-format",
            "json",
        ]
        args.extend(extra)
        buf = io.StringIO()
        with redirect_stderr(buf):
            result = main(args)
        assert result == 1
        return buf.getvalue()

    # ---- weight pairs ----
    def test_weight_pairs_missing_file(self, tmp_path: Any) -> None:
        err = self._err(tmp_path, "--weight-pairs-file", "/nonexistent/wp.json")
        assert "file does not exist" in err
        assert "/nonexistent/wp.json" in err
        assert "--weight-pairs-file" in err
        assert "weight_pairs JSON" in err
        assert "download.py" in err

    def test_weight_pairs_malformed_json(self, tmp_path: Any) -> None:
        bad = tmp_path / "bad.json"
        bad.write_text("not json")
        err = self._err(tmp_path, "--weight-pairs-file", str(bad))
        assert "JSON is malformed" in err
        assert str(bad) in err
        assert "--weight-pairs-file" in err
        assert "weight_pairs JSON" in err

    def test_weight_pairs_wrong_top_level_type(self, tmp_path: Any) -> None:
        bad = tmp_path / "wp.json"
        bad.write_text('"just a string"')
        err = self._err(tmp_path, "--weight-pairs-file", str(bad))
        assert "expected JSON object or list" in err
        assert str(bad) in err
        assert "--weight-pairs-file" in err
        assert "pairs" in err

    def test_weight_pairs_missing_keys(self, tmp_path: Any) -> None:
        bad = tmp_path / "wp.json"
        bad.write_text(
            json.dumps({"pairs": [{"gradient": [[1.0]], "weight": [[1.0]]}]})
        )
        err = self._err(tmp_path, "--weight-pairs-file", str(bad))
        assert "missing required keys" in err
        assert "'layer_name'" in err
        assert str(bad) in err
        assert "--weight-pairs-file" in err

    def test_weight_pairs_empty(self, tmp_path: Any) -> None:
        bad = tmp_path / "wp.json"
        bad.write_text(json.dumps({"pairs": []}))
        err = self._err(tmp_path, "--weight-pairs-file", str(bad))
        assert "no valid weight pairs found" in err
        assert str(bad) in err
        assert "--weight-pairs-file" in err

    def test_weight_pairs_wrong_tensor_rank(self, tmp_path: Any) -> None:
        bad = tmp_path / "wp.json"
        bad.write_text(
            json.dumps(
                {
                    "pairs": [
                        {
                            "gradient": [[[1.0, 2.0], [3.0, 4.0]]],
                            "weight": [[1.0, 2.0], [3.0, 4.0]],
                            "layer_name": "l0",
                        }
                    ]
                }
            )
        )
        err = self._err(tmp_path, "--weight-pairs-file", str(bad))
        assert "2-D [rows, cols]" in err
        assert str(bad) in err
        assert "--weight-pairs-file" in err

    # ---- activations ----
    def test_activations_missing_file(self, tmp_path: Any) -> None:
        err = self._err(tmp_path, "--activations", "/nonexistent/acts.json")
        assert "file does not exist" in err
        assert "/nonexistent/acts.json" in err
        assert "--activations" in err

    def test_activations_malformed_json(self, tmp_path: Any) -> None:
        bad = tmp_path / "bad.json"
        bad.write_text("not json")
        err = self._err(tmp_path, "--activations", str(bad))
        assert "JSON is malformed" in err
        assert str(bad) in err
        assert "--activations" in err

    def test_activations_missing_keys(self, tmp_path: Any) -> None:
        bad = tmp_path / "acts.json"
        bad.write_text(json.dumps({"wrong": "shape"}))
        err = self._err(tmp_path, "--activations", str(bad))
        assert "missing required keys" in err
        assert "'metadata'" in err or "'layers'" in err
        assert str(bad) in err
        assert "--activations" in err

    # ---- mlp activations ----
    def test_mlp_activations_missing_file(self, tmp_path: Any) -> None:
        err = self._err(tmp_path, "--mlp-activations-file", "/nonexistent/mlp.json")
        assert "file does not exist" in err
        assert "/nonexistent/mlp.json" in err
        assert "--mlp-activations-file" in err

    def test_mlp_activations_wrong_ndim(self, tmp_path: Any) -> None:
        bad = tmp_path / "mlp.json"
        bad.write_text(
            json.dumps({"layers": [{"activations": [[1.0, 2.0]], "layer_name": "l0"}]})
        )
        err = self._err(tmp_path, "--mlp-activations-file", str(bad))
        assert "3-D [batch, seq, hidden]" in err
        assert str(bad) in err
        assert "--mlp-activations-file" in err

    def test_mlp_activations_empty(self, tmp_path: Any) -> None:
        bad = tmp_path / "mlp.json"
        bad.write_text(json.dumps({"layers": []}))
        err = self._err(tmp_path, "--mlp-activations-file", str(bad))
        assert "no valid MLP activation layers found" in err
        assert str(bad) in err
        assert "--mlp-activations-file" in err

    # ---- expert weights ----
    def test_expert_weights_missing_file(self, tmp_path: Any) -> None:
        err = self._err(tmp_path, "--expert-weights-file", "/nonexistent/ew.json")
        assert "file does not exist" in err
        assert "/nonexistent/ew.json" in err
        assert "--expert-weights-file" in err

    def test_expert_weights_single_expert(self, tmp_path: Any) -> None:
        bad = tmp_path / "ew.json"
        bad.write_text(
            json.dumps(
                {"layers": [{"expert_weights": [[[1.0, 2.0]]], "layer_name": "l0"}]}
            )
        )
        err = self._err(tmp_path, "--expert-weights-file", str(bad))
        assert "at least 2 weight arrays" in err
        assert str(bad) in err
        assert "--expert-weights-file" in err

    def test_expert_weights_empty(self, tmp_path: Any) -> None:
        bad = tmp_path / "ew.json"
        bad.write_text(json.dumps({"layers": []}))
        err = self._err(tmp_path, "--expert-weights-file", str(bad))
        assert "no valid expert weight layers found" in err
        assert str(bad) in err
        assert "--expert-weights-file" in err

    # ---- gradient provenance ----
    def test_gradient_provenance_missing_file(self, tmp_path: Any) -> None:
        err = self._err(
            tmp_path, "--gradient-provenance-file", "/nonexistent/prov.json"
        )
        assert "file does not exist" in err
        assert "/nonexistent/prov.json" in err
        assert "--gradient-provenance-file" in err

    def test_gradient_provenance_flat_wrong_nesting(self, tmp_path: Any) -> None:
        bad = tmp_path / "prov.json"
        bad.write_text(json.dumps({"source_type": "measured"}))
        err = self._err(tmp_path, "--gradient-provenance-file", str(bad))
        assert "missing the nested 'provenance' object" in err
        assert str(bad) in err
        assert "--gradient-provenance-file" in err

    def test_gradient_provenance_bad_source_type(self, tmp_path: Any) -> None:
        bad = tmp_path / "prov.json"
        bad.write_text(json.dumps({"provenance": {"source_type": "bogus"}}))
        err = self._err(tmp_path, "--gradient-provenance-file", str(bad))
        assert "'source_type' must be one of" in err
        assert str(bad) in err
        assert "--gradient-provenance-file" in err

    # ---- regression: pin exact informative text ----
    def test_weight_pairs_missing_file_pins_informative_text(
        self, tmp_path: Any
    ) -> None:
        err = self._err(tmp_path, "--weight-pairs-file", "/no/such/wp.json")
        assert (
            "Error loading weight pairs from /no/such/wp.json: "
            "file does not exist: /no/such/wp.json. "
            "Use --weight-pairs-file to provide a weight_pairs JSON file; "
            "see src/model_headroom/download.py for the expected JSON format."
        ) in err
