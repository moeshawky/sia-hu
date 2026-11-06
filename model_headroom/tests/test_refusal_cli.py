"""Tests for the model-headroom-refusal command-line interface."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from model_headroom.refusal.cli import main
from model_headroom.refusal.report import (
    generate_refusal_json,
    generate_refusal_report,
)


def _contrast(seed: int, separation: float, n: int = 20, dim: int = 8) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(n, dim, generator=gen) + separation


def _write_activations(path: Path, seed: int, separation: float) -> None:
    harmful = _contrast(seed, separation)
    harmless = _contrast(seed, 0.0)
    payload = {
        "layers": [
            {"layer_name": "l0", "activations": harmful.tolist()},
            {"layer_name": "l1", "activations": (harmful * 1.5).tolist()},
        ]
    }
    deltas = {"layers": [{"layer_name": "l0", "activations": harmless.tolist()}]}
    deltas["layers"].append(
        {"layer_name": "l1", "activations": (harmless * 1.5).tolist()}
    )
    path.write_text(json.dumps(payload if seed % 2 else deltas))


def _pairs(tmp_path: Path, n: int = 20) -> tuple[Path, Path]:
    harmful = tmp_path / "h.json"
    harmless = tmp_path / "s.json"
    _write_activations(harmful, 1, 2.0)
    _write_activations(harmless, 1, 0.0)
    return (harmful, harmless)


class TestOfflinePath:
    def test_computes_and_exits_zero(self, tmp_path: Path, capsys) -> None:
        harmful, harmless = _pairs(tmp_path)
        rc = main(
            [
                "--harmful-activations",
                str(harmful),
                "--harmless-activations",
                str(harmless),
                "--allow-small-samples",
            ]
        )
        assert rc == 0
        assert "REFUSAL DIRECTION" in capsys.readouterr().out

    def test_json_output_is_parseable(self, tmp_path: Path, capsys) -> None:
        harmful, harmless = _pairs(tmp_path)
        rc = main(
            [
                "--harmful-activations",
                str(harmful),
                "--harmless-activations",
                str(harmless),
                "--output-format",
                "json",
                "--allow-small-samples",
            ]
        )
        assert rc == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["axis"] == "mean_difference"
        assert payload["direction_status"] == "computed"
        assert len(payload["refusal_direction"]) == 8

    def test_output_file_is_written(self, tmp_path: Path) -> None:
        harmful, harmless = _pairs(tmp_path)
        out = tmp_path / "report.txt"
        rc = main(
            [
                "--harmful-activations",
                str(harmful),
                "--harmless-activations",
                str(harmless),
                "--output-file",
                str(out),
                "--allow-small-samples",
            ]
        )
        assert rc == 0
        assert "REFUSAL DIRECTION" in out.read_text()

    def test_min_pairs_blocks_and_exits_two(self, tmp_path: Path) -> None:
        harmful = tmp_path / "h.json"
        harmless = tmp_path / "s.json"
        harmful.write_text(
            json.dumps(
                {"layers": [{"layer_name": "l0", "activations": [[1.0], [2.0]]}]}
            )
        )
        harmless.write_text(
            json.dumps(
                {"layers": [{"layer_name": "l0", "activations": [[0.0], [0.5]]}]}
            )
        )
        rc = main(
            [
                "--harmful-activations",
                str(harmful),
                "--harmless-activations",
                str(harmless),
            ]
        )
        assert rc == 2

    def test_missing_activation_file_exits_one(self, tmp_path: Path) -> None:
        _, harmless = _pairs(tmp_path)
        rc = main(
            [
                "--harmful-activations",
                str(tmp_path / "nope.json"),
                "--harmless-activations",
                str(harmless),
            ]
        )
        assert rc == 1

    def test_no_acquisition_path_exits_one(self, capsys) -> None:
        assert main([]) == 1
        assert "acquisition path" in capsys.readouterr().err

    def test_orthogonalize_flag_is_accepted(self, tmp_path: Path, capsys) -> None:
        harmful, harmless = _pairs(tmp_path)
        rc = main(
            [
                "--harmful-activations",
                str(harmful),
                "--harmless-activations",
                str(harmless),
                "--orthogonalize",
                "--output-format",
                "json",
                "--allow-small-samples",
            ]
        )
        assert rc == 0
        assert json.loads(capsys.readouterr().out)["orthogonalized"] is True

    def test_layer_name_selection_is_honoured(self, tmp_path: Path, capsys) -> None:
        harmful, harmless = _pairs(tmp_path)
        rc = main(
            [
                "--harmful-activations",
                str(harmful),
                "--harmless-activations",
                str(harmless),
                "--layer-name",
                "l1",
                "--output-format",
                "json",
                "--allow-small-samples",
            ]
        )
        assert rc == 0
        assert json.loads(capsys.readouterr().out)["layer_name"] == "l1"

    def test_mask_sigma_flag_is_accepted(self, tmp_path: Path, capsys) -> None:
        harmful, harmless = _pairs(tmp_path)
        rc = main(
            [
                "--harmful-activations",
                str(harmful),
                "--harmless-activations",
                str(harmless),
                "--mask-sigma",
                "1.0",
                "--output-format",
                "json",
                "--allow-small-samples",
            ]
        )
        assert rc == 0
        assert json.loads(capsys.readouterr().out)["massive_activation_sigma"] == 1.0


class TestRecoveryPath:
    def test_weight_deltas_recover_a_direction(self, tmp_path: Path, capsys) -> None:
        gen = torch.Generator().manual_seed(3)
        truth = torch.randn(10, generator=gen)
        truth = truth / truth.norm()
        deltas = [
            torch.outer(truth, torch.randn(6, 10, generator=gen) @ truth)
            for _ in range(4)
        ]
        path = tmp_path / "deltas.json"
        path.write_text(json.dumps({"deltas": [d.tolist() for d in deltas]}))
        rc = main(["--weight-deltas", str(path), "--output-format", "json"])
        assert rc == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["axis"] == "weight_delta_svd"
        assert payload["direction_status"] == "computed"

    def test_too_few_deltas_exits_one(self, tmp_path: Path) -> None:
        path = tmp_path / "deltas.json"
        path.write_text(json.dumps({"deltas": [[[1.0, 2.0], [3.0, 4.0]]]}))
        assert main(["--weight-deltas", str(path)]) == 1

    def test_missing_delta_file_exits_one(self, tmp_path: Path) -> None:
        assert main(["--weight-deltas", str(tmp_path / "no.json")]) == 1


class TestReportRendering:
    def _report(self):
        harmful = _contrast(9, 2.0)
        harmless = _contrast(9, 0.0)
        from model_headroom.models.types import RefusalDirectionSpec, ResidualBatch
        from model_headroom.refusal.extract import compute_refusal_direction

        return compute_refusal_direction(
            [ResidualBatch(activations=harmful, layer_name="l0")],
            [ResidualBatch(activations=harmless, layer_name="l0")],
            RefusalDirectionSpec(min_pairs=1),
        )

    def test_text_report_contains_sweep(self) -> None:
        text = generate_refusal_report(self._report())
        assert "Layer sweep" in text
        assert "cohen d" in text
        assert "Limitations" in text

    def test_json_report_includes_direction(self) -> None:
        payload = json.loads(generate_refusal_json(self._report()))
        assert len(payload["refusal_direction"]) == 8

    def test_nan_cohen_d_serialises_as_null(self) -> None:
        from model_headroom.models.types import (
            RefusalDirectionReport,
            RefusalLayerStat,
        )

        report = RefusalDirectionReport(
            axis="mean_difference",
            hidden_size=2,
            harmful_count=0,
            harmless_count=0,
            layer_stats=[
                RefusalLayerStat(
                    layer_name="l0",
                    layer_index=0,
                    harmful_count=0,
                    harmless_count=0,
                    cohen_d=float("nan"),
                    mean_norm=0.0,
                    masked_count=0,
                    direction_norm=0.0,
                    orthogonality_with_harmless_mean=0.0,
                )
            ],
        )
        payload = json.loads(generate_refusal_json(report))
        assert payload["layer_stats"][0]["cohen_d"] is None

    def test_report_without_direction_renders(self) -> None:
        from model_headroom.models.types import RefusalDirectionReport

        report = RefusalDirectionReport(
            axis="mean_difference",
            hidden_size=0,
            harmful_count=0,
            harmless_count=0,
            layer_stats=[],
            error_message="nothing to see",
        )
        assert "not computed" in generate_refusal_report(report)


class TestIsolation:
    def test_refusal_package_is_not_a_headroom_signal(self) -> None:
        import model_headroom.analysis.orchestrator as orchestrator

        source = Path(orchestrator.__file__).read_text()
        assert "refusal" not in source

    def test_scoring_does_not_import_refusal(self) -> None:
        import model_headroom.analysis.scoring as scoring

        source = Path(scoring.__file__).read_text()
        assert "refusal" not in source

    def test_importing_refusal_does_not_pull_the_orchestrator(self) -> None:
        import subprocess
        import sys

        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import model_headroom.refusal.extract, sys; "
                "print('model_headroom.analysis.orchestrator' in sys.modules)",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        assert result.stdout.strip() == "False"


def test_help_exits_cleanly() -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["--help"])
    assert excinfo.value.code == 0


class TestSglangPath:
    def test_sglang_requires_prompts(self, capsys) -> None:
        assert main(["--sglang-url", "http://127.0.0.1:30000"]) == 1
        assert "--harmful-prompts" in capsys.readouterr().err

    def test_sglang_unreachable_exits_one(self, tmp_path: Path, capsys) -> None:
        prompts = tmp_path / "p.txt"
        prompts.write_text("a prompt\n")
        rc = main(
            [
                "--sglang-url",
                "http://127.0.0.1:1",
                "--harmful-prompts",
                str(prompts),
                "--harmless-prompts",
                str(prompts),
            ]
        )
        assert rc == 1
        assert "hidden states" in capsys.readouterr().err

    def test_sglang_missing_prompts_file_exits_one(self, tmp_path: Path) -> None:
        rc = main(
            [
                "--sglang-url",
                "http://127.0.0.1:30000",
                "--harmful-prompts",
                str(tmp_path / "no.txt"),
                "--harmless-prompts",
                str(tmp_path / "no.txt"),
            ]
        )
        assert rc == 1
