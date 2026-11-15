from __future__ import annotations

import json
import subprocess
import sys
from typing import TYPE_CHECKING, Any

import pytest

from model_headroom.analysis.intervention_record import InterventionRecord
from model_headroom.paired.cli import main
from model_headroom.paired.schema import (
    LayerMeasurement,
    MeasurementDocument,
)

if TYPE_CHECKING:
    from pathlib import Path


def _document(
    model_id: str,
    role: str,
    cosine: float,
    *,
    fingerprint: str | None = "fp-1",
) -> MeasurementDocument:
    layer = LayerMeasurement(
        layer_name="encoder.layer0",
        layer_index=0,
        family="attention",
        output_cosine=cosine,
        normalized_reconstruction_error=0.1,
        output_norm_ratio=1.0,
        status="measured",
    )
    return MeasurementDocument(
        model_id=model_id,
        role=role,  # type: ignore[arg-type]
        input_fingerprint=fingerprint,
        layers=(layer,),
    )


def _write(path: Path, doc: MeasurementDocument) -> Path:
    path.write_text(doc.to_json())
    return path


def _write_bad_version(path: Path, schema_version: str) -> Path:
    payload = _document("m", "reference", 1.0).to_dict()
    payload["schema_version"] = schema_version
    path.write_text(json.dumps(payload))
    return path


def _record(
    baseline: dict[str, float],
    post: dict[str, float],
    *,
    intervention_type: str = "continuous_parameters_frozen_codes",
) -> InterventionRecord:
    return InterventionRecord(
        intervention_id="irec-1",
        intervention_type=intervention_type,
        checkpoint_hash="c" * 16,
        architecture="bert-base",
        runtime_versions={"torch": "2.0"},
        affected_parameters=["encoder.layer0.q_proj.weight"],
        memory_budget_gb=0.5,
        baseline_measurements=baseline,
        post_measurements=post,
    )


def _write_record(path: Path, record: InterventionRecord) -> Path:
    path.write_text(record.to_json())
    return path


def _improving_record() -> InterventionRecord:
    return _record({"loss": 1.0, "kl": 0.5}, {"loss": 0.7, "kl": 0.4})


class TestSchemaVersion:
    def test_prints_version_and_exits_zero(self, capsys: Any) -> None:
        rc = main(["schema-version"])
        assert rc == 0
        assert capsys.readouterr().out.strip() == "1"


class TestCompare:
    def test_valid_documents_write_report(self, tmp_path: Path, capsys: Any) -> None:
        reference = _write(tmp_path / "r.json", _document("ref", "reference", 1.0))
        candidate = _write(tmp_path / "c.json", _document("cand", "compressed", 0.9))
        out = tmp_path / "cmp.json"
        rc = main(
            [
                "compare",
                "--reference",
                str(reference),
                "--candidate",
                str(candidate),
                "--out",
                str(out),
            ]
        )
        assert rc == 0
        capsys.readouterr()
        report = json.loads(out.read_text())
        assert report["coverage"] == 1.0
        assert report["input_fingerprint_match"] is True

    def test_missing_path_exits_one_with_cause(
        self, tmp_path: Path, capsys: Any
    ) -> None:
        candidate = _write(tmp_path / "c.json", _document("cand", "compressed", 0.9))
        rc = main(
            [
                "compare",
                "--reference",
                str(tmp_path / "absent.json"),
                "--candidate",
                str(candidate),
            ]
        )
        assert rc == 1
        err = capsys.readouterr().err
        assert "Cause:" in err
        assert "Exit:   1" in err

    def test_schema_version_mismatch_names_both_versions(
        self, tmp_path: Path, capsys: Any
    ) -> None:
        bad = _write_bad_version(tmp_path / "bad.json", "0")
        good = _write(tmp_path / "c.json", _document("cand", "compressed", 0.9))
        rc = main(["compare", "--reference", str(bad), "--candidate", str(good)])
        assert rc == 1
        err = capsys.readouterr().err
        assert "'0'" in err
        assert "'1'" in err


class TestDecide:
    def test_coverage_reaches_supported(self, tmp_path: Path, capsys: Any) -> None:
        record = _write_record(tmp_path / "irec.json", _improving_record())
        rc = main(
            [
                "decide",
                "--baseline",
                str(record),
                "--held-out-keys",
                "loss,kl",
                "--coverage",
                "0.9",
            ]
        )
        assert rc == 0
        decision = json.loads(capsys.readouterr().out)
        assert decision["strategies"][0]["verdict"] == "SUPPORTED"

    def test_coverage_outranks_comparison(self, tmp_path: Path, capsys: Any) -> None:
        reference = _write(tmp_path / "r.json", _document("ref", "reference", 1.0))
        candidate = _write(tmp_path / "c.json", _document("cand", "compressed", 0.9))
        comparison = tmp_path / "cmp.json"
        assert (
            main(
                [
                    "compare",
                    "--reference",
                    str(reference),
                    "--candidate",
                    str(candidate),
                    "--out",
                    str(comparison),
                ]
            )
            == 0
        )
        assert json.loads(comparison.read_text())["coverage"] == 1.0
        record = _write_record(tmp_path / "irec.json", _improving_record())
        rc = main(
            [
                "decide",
                "--baseline",
                str(record),
                "--held-out-keys",
                "loss",
                "--comparison",
                str(comparison),
                "--coverage",
                "0.5",
            ]
        )
        assert rc == 0
        evidence = json.loads(capsys.readouterr().out)["strategies"][0]
        assert evidence["coverage"] == 0.5
        assert evidence["verdict"] == "INCONCLUSIVE"

    def test_coverage_above_one_exits_one(self, tmp_path: Path, capsys: Any) -> None:
        record = _write_record(tmp_path / "irec.json", _improving_record())
        rc = main(
            [
                "decide",
                "--baseline",
                str(record),
                "--held-out-keys",
                "loss",
                "--coverage",
                "1.5",
            ]
        )
        assert rc == 1
        err = capsys.readouterr().err
        assert "1.5" in err
        assert "Exit:   1" in err


class TestVerify:
    def test_round_trip_verifies(self, tmp_path: Path, capsys: Any) -> None:
        record = _write_record(tmp_path / "irec.json", _improving_record())
        store = tmp_path / "ev"
        rc = main(
            [
                "decide",
                "--baseline",
                str(record),
                "--held-out-keys",
                "loss",
                "--coverage",
                "0.9",
                "--store",
                str(store),
                "--out",
                str(tmp_path / "dec.json"),
            ]
        )
        assert rc == 0
        err = capsys.readouterr().err
        artifact_id = err.split("artifact_id: ")[1].strip()
        rc = main(["verify", "--store", str(store), "--artifact-id", artifact_id])
        assert rc == 0
        assert capsys.readouterr().out.count('"verified": true') == 1

    def test_unknown_artifact_exits_one(self, tmp_path: Path, capsys: Any) -> None:
        rc = main(
            [
                "verify",
                "--store",
                str(tmp_path / "ev"),
                "--artifact-id",
                "missing_000",
            ]
        )
        assert rc == 1
        out = capsys.readouterr().out
        assert '"verified": false' in out


class TestHelp:
    def test_help_exits_zero(self, capsys: Any) -> None:
        with pytest.raises(SystemExit) as excinfo:
            main(["--help"])
        assert excinfo.value.code == 0
        assert "usage: model-headroom-paired" in capsys.readouterr().out

    def test_import_is_torch_free_in_a_fresh_process(self) -> None:
        code = "import sys, model_headroom.paired.cli; print('torch' in sys.modules)"
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            check=True,
        )
        assert result.stdout.strip() == "False"
