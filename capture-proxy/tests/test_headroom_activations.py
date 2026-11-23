"""Tests for the expert-activation producer that feeds model_headroom R-06.

The artifact is the model_headroom `load_activations_from_json` contract
(models/loader.py:316-334). Every assertion here goes through the REAL
authority loader and the REAL `analyze_load_balance`; a passing parse is
never accepted as a pass, the returned report's `signal_status` is what
is asserted.

Routing counts in this module are FIXTURES injected through the writer's
documented `routing_source` seam. The carrier captures weights, not
router activations, so the acquisition of measured counts is BLOCKED on
MODEL_SLOT-1; the producer logic under test is real, its input is not.
"""

import asyncio
import json
import math
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any

import pytest

from capture_proxy.headroom import (
    ACTIVATION_ARTIFACT_SUFFIX,
    HeadroomWriter,
    build_activation_payload,
)
from model_headroom.models.loader import load_activations_from_json
from model_headroom.signals.load_balance import analyze_load_balance

CREATED_AT = "2026-09-29T12:00:00+00:00"
ARCH = 8
FED = {0: 25, 1: 25, 2: 25, 3: 25}


def _moe_model_dir(tmp_path: Path, *, num_experts: int = ARCH) -> Path:
    d = tmp_path / "model"
    d.mkdir()
    (d / "config.json").write_text(
        json.dumps({"model_type": "qwen3_moe", "num_local_experts": num_experts})
    )
    return d


def _writer(
    tmp_path: Path,
    counts: dict[int, dict[str, int]] | None,
    *,
    signals: frozenset[str] = frozenset({"load_balance"}),
) -> HeadroomWriter:
    def _source(_path: Path, _num_experts: int) -> Any:
        return counts

    return HeadroomWriter(
        tmp_path / "h",
        signals=signals,
        routing_source=_source,
    )


def _run_capture(
    writer: HeadroomWriter, model_dir: Path, tmp_path: Path
) -> tuple[Path, Path]:
    capture_id = str(uuid.uuid4())
    sidecar = asyncio.run(
        writer.write_async(
            capture_id=capture_id,
            parent_raw=tmp_path / "raw.jsonl",
            model="m",
            model_path=model_dir,
            created_at=CREATED_AT,
        )
    )
    assert sidecar is not None
    day = sidecar.parent
    return sidecar, day / f"{capture_id}{ACTIVATION_ARTIFACT_SUFFIX}"


def _authority_status(activation_file: Path) -> str:
    data = load_activations_from_json(str(activation_file))
    if data is None:
        return "loader_rejected"
    return analyze_load_balance(data).signal_status


def _authority_report(activation_file: Path) -> Any:
    data = load_activations_from_json(str(activation_file))
    assert data is not None
    return analyze_load_balance(data)


# --- producer: fed versus starved ------------------------------------------


def test_starved_expert_is_excluded_and_named_not_counted_dead():
    payload = build_activation_payload(
        num_experts_architectural=ARCH, counts={0: {str(k): v for k, v in FED.items()}}
    )
    meta = payload["metadata"]
    assert meta["num_experts"] == 4
    assert meta["num_experts_architectural"] == 8
    assert meta["num_experts_fed"] == 4
    assert meta["starved_experts"] == [4, 5, 6, 7]
    assert sorted(payload["layers"]["0"]) == ["0", "1", "2", "3"]


def test_starved_expert_present_with_zero_count_is_still_excluded():
    with_zero = {"0": 25, "1": 25, "2": 25, "3": 25, "4": 0, "5": 0, "6": 0, "7": 0}
    payload = build_activation_payload(
        num_experts_architectural=ARCH, counts={0: with_zero}
    )
    assert sorted(payload["layers"]["0"]) == ["0", "1", "2", "3"]
    assert payload["metadata"]["starved_experts"] == [4, 5, 6, 7]


def test_layer_with_no_fed_expert_is_omitted_and_named():
    payload = build_activation_payload(
        num_experts_architectural=ARCH,
        counts={
            0: {str(k): v for k, v in FED.items()},
            1: {str(k): 0 for k in range(8)},
        },
    )
    assert sorted(payload["layers"]) == ["0"]
    assert payload["metadata"]["omitted_layers"] == [1]


def test_expert_id_outside_architecture_is_dropped_and_named():
    payload = build_activation_payload(
        num_experts_architectural=ARCH,
        counts={0: {"0": 10, "1": 10, "9": 5, "-1": 3, "x": 2}},
    )
    assert sorted(payload["layers"]["0"]) == ["0", "1"]
    assert payload["metadata"]["inconsistent_experts"] == ["-1", "9", "x"]
    assert payload["metadata"]["num_experts"] == 2


def test_payload_is_none_without_an_architectural_expert_count():
    assert (
        build_activation_payload(num_experts_architectural=None, counts={0: {"0": 1}})
        is None
    )
    assert build_activation_payload(num_experts_architectural=0, counts=None) is None


def test_duplicate_layer_indices_are_unrepresentable_in_the_written_file(tmp_path):
    duplicated: dict[int, dict[str, int]] = {}
    duplicated[0] = {"0": 5, "1": 5}
    duplicated[0] = {"0": 1}
    payload = build_activation_payload(num_experts_architectural=4, counts=duplicated)
    assert list(payload["layers"]) == ["0"]
    assert len(json.loads(json.dumps(payload))["layers"]) == 1
    assert json.loads(json.dumps(payload))["layers"] == {"0": {"0": 1}}


# --- writer: artifact materializes and is named in signals ------------------


def test_writer_materializes_artifact_and_names_it_in_signals(tmp_path):
    model_dir = _moe_model_dir(tmp_path)
    writer = _writer(tmp_path, {0: {str(k): v for k, v in FED.items()}})
    sidecar, activation = _run_capture(writer, model_dir, tmp_path)
    assert activation.is_file()
    record = json.loads(sidecar.read_text())
    assert record["signals"]["load_balance"] == str(activation)
    assert record["signals"]["effective_rank"] is None
    assert record["schema_version"] == 7
    assert record["error_body"] is None


def test_writer_without_routing_source_writes_metadata_only(tmp_path):
    model_dir = _moe_model_dir(tmp_path)
    writer = HeadroomWriter(tmp_path / "h", signals=frozenset({"load_balance"}))
    sidecar, activation = _run_capture(writer, model_dir, tmp_path)
    payload = json.loads(activation.read_text())
    assert payload["layers"] == {}
    assert payload["metadata"]["num_experts_architectural"] == ARCH
    assert payload["metadata"]["starved_experts"] == list(range(ARCH))
    assert json.loads(sidecar.read_text())["signals"]["load_balance"] == str(activation)


def test_writer_leaves_load_balance_null_when_signal_disabled(tmp_path):
    model_dir = _moe_model_dir(tmp_path)
    writer = _writer(
        tmp_path,
        {0: {str(k): v for k, v in FED.items()}},
        signals=frozenset({"effective_rank"}),
    )
    sidecar, activation = _run_capture(writer, model_dir, tmp_path)
    assert not activation.exists()
    assert json.loads(sidecar.read_text())["signals"]["load_balance"] is None


def test_writer_omits_artifact_and_keeps_error_body_clean_for_dense_model(tmp_path):
    d = tmp_path / "bert"
    d.mkdir()
    (d / "config.json").write_text(json.dumps({"model_type": "bert"}))
    writer = _writer(tmp_path, {0: {"0": 1}})
    sidecar, activation = _run_capture(writer, d, tmp_path)
    assert not activation.exists()
    record = json.loads(sidecar.read_text())
    assert record["signals"]["load_balance"] is None
    assert record["error_body"] is None


def test_failing_routing_source_still_writes_the_artifact(tmp_path):
    model_dir = _moe_model_dir(tmp_path)

    def _boom(_path: Path, _num_experts: int) -> Any:
        raise RuntimeError("router unavailable")

    writer = HeadroomWriter(
        tmp_path / "h",
        signals=frozenset({"load_balance"}),
        routing_source=_boom,
    )
    sidecar, activation = _run_capture(writer, model_dir, tmp_path)
    assert activation.is_file()
    assert json.loads(activation.read_text())["layers"] == {}
    assert json.loads(sidecar.read_text())["error_body"] is None


def test_planned_activations_path_agrees_with_what_is_written(tmp_path):
    model_dir = _moe_model_dir(tmp_path)
    writer = _writer(tmp_path, {0: {"0": 10, "1": 10}})
    capture_id = str(uuid.uuid4())
    sidecar, weight, activation, expert_weights = writer.planned_artifacts(
        capture_id, CREATED_AT
    )
    assert weight is None
    assert activation is not None
    assert expert_weights is None
    asyncio.run(
        writer.write_async(
            capture_id=capture_id,
            parent_raw=tmp_path / "raw.jsonl",
            model="m",
            model_path=model_dir,
            created_at=CREATED_AT,
        )
    )
    written = sidecar.parent / f"{capture_id}{ACTIVATION_ARTIFACT_SUFFIX}"
    assert Path(activation) == written
    assert written.is_file()


# --- PROBE 1: the round trip ------------------------------------------------


def test_probe1_round_trip_reaches_computed_through_real_authority(tmp_path):
    model_dir = _moe_model_dir(tmp_path)
    writer = _writer(tmp_path, {0: {str(k): v for k, v in FED.items()}})
    _, activation = _run_capture(writer, model_dir, tmp_path)
    report = _authority_report(activation)
    assert report.signal_status == "computed"
    assert report.balance_ratio == pytest.approx(math.log(4) / math.log(8))
    assert report.ideal_entropy == pytest.approx(math.log(8))


def test_probe1_malformed_artifacts_never_reach_computed(tmp_path):
    cases = {
        "missing_layers": {"metadata": {"num_experts": 4}},
        "missing_metadata": {"layers": {"0": {"0": 5, "1": 5}}},
        "num_experts_zero": {
            "metadata": {"num_experts": 0},
            "layers": {"0": {"0": 5, "1": 5}},
        },
        "layers_empty": {"metadata": {"num_experts": 4}, "layers": {}},
    }
    expected = {
        "missing_layers": "loader_rejected",
        "missing_metadata": "loader_rejected",
        "num_experts_zero": "error",
        "layers_empty": "error",
    }
    for name, payload in cases.items():
        f = tmp_path / f"{name}.json"
        f.write_text(json.dumps(payload))
        status = _authority_status(f)
        assert status != "computed", f"{name} must not reach computed"
        assert status == expected[name], f"{name} -> {status}"


# --- PROBE 2: dense path ----------------------------------------------------


def test_probe2_single_expert_returns_none_never_zero(tmp_path):
    model_dir = _moe_model_dir(tmp_path, num_experts=1)
    writer = _writer(tmp_path, {0: {"0": 100}})
    _, activation = _run_capture(writer, model_dir, tmp_path)
    report = _authority_report(activation)
    assert report.signal_status == "single_expert"
    assert report.balance_ratio is None
    assert report.balance_ratio != 0.0


def test_probe2_architecturally_single_expert_that_was_fed(tmp_path):
    model_dir = _moe_model_dir(tmp_path, num_experts=1)
    writer = _writer(tmp_path, {0: {"0": 7}})
    _, activation = _run_capture(writer, model_dir, tmp_path)
    assert json.loads(activation.read_text())["metadata"]["num_experts"] == 1
    assert _authority_report(activation).signal_status == "single_expert"


# --- PROBE 3: fed versus starved, corpus-length independence ---------------


def test_probe3_imbalance_is_identical_at_n_and_2n(tmp_path):
    at_n = build_activation_payload(
        num_experts_architectural=ARCH, counts={0: {str(k): v for k, v in FED.items()}}
    )
    doubled = {k: v * 2 for k, v in FED.items()}
    at_2n = build_activation_payload(
        num_experts_architectural=ARCH,
        counts={0: {str(k): v for k, v in doubled.items()}},
    )
    assert at_n["metadata"]["num_experts"] == at_2n["metadata"]["num_experts"]

    f_n = tmp_path / "n.json"
    f_2n = tmp_path / "2n.json"
    f_n.write_text(json.dumps(at_n))
    f_2n.write_text(json.dumps(at_2n))
    ratio_n = _authority_report(f_n).balance_ratio
    ratio_2n = _authority_report(f_2n).balance_ratio
    assert ratio_n == pytest.approx(ratio_2n)
    assert ratio_n == pytest.approx(math.log(4) / math.log(8))


def test_probe3_rewriting_the_fed_count_no_longer_moves_the_verdict(tmp_path):
    excluded = build_activation_payload(
        num_experts_architectural=ARCH, counts={0: {str(k): v for k, v in FED.items()}}
    )
    folded = json.loads(json.dumps(excluded))
    folded["metadata"]["num_experts"] = ARCH
    f_ok = tmp_path / "excluded.json"
    f_bad = tmp_path / "folded.json"
    f_ok.write_text(json.dumps(excluded))
    f_bad.write_text(json.dumps(folded))
    assert _authority_report(f_ok).balance_ratio == pytest.approx(
        math.log(4) / math.log(8)
    )
    assert _authority_report(f_bad).balance_ratio == pytest.approx(
        _authority_report(f_ok).balance_ratio
    )


def test_probe3_rare_expert_crossing_threshold_moves_the_estimate(tmp_path):
    at_n: dict[int, dict[str, int]] = {0: {str(k): 50 for k in range(4)}}
    at_2n: dict[int, dict[str, int]] = {
        0: {**{str(k): 10000 for k in range(4)}, "7": 8}
    }
    payload_n = build_activation_payload(num_experts_architectural=ARCH, counts=at_n)
    payload_2n = build_activation_payload(num_experts_architectural=ARCH, counts=at_2n)
    naive_dead_n = 4 / ARCH
    naive_dead_2n = 3 / ARCH
    assert naive_dead_n != naive_dead_2n
    assert payload_n["metadata"]["starved_experts"] == [4, 5, 6, 7]
    assert payload_2n["metadata"]["starved_experts"] == [4, 5, 6]
    assert payload_n["metadata"]["num_experts"] == 4
    assert payload_2n["metadata"]["num_experts"] == 5

    f_n = tmp_path / "rare_n.json"
    f_2n = tmp_path / "rare_2n.json"
    f_n.write_text(json.dumps(payload_n))
    f_2n.write_text(json.dumps(payload_2n))
    assert _authority_report(f_n).signal_status == "computed"
    assert _authority_report(f_2n).signal_status == "computed"
    assert _authority_report(f_n).balance_ratio < _authority_report(f_2n).balance_ratio


def test_probe3_collapsed_router_reports_maximal_imbalance(tmp_path):
    model_dir = _moe_model_dir(tmp_path)
    writer = _writer(tmp_path, {0: {"0": 100}})
    _, activation = _run_capture(writer, model_dir, tmp_path)
    meta = json.loads(activation.read_text())["metadata"]
    assert meta["num_experts_architectural"] == ARCH
    assert meta["num_experts_fed"] == 1
    assert meta["starved_experts"] == [1, 2, 3, 4, 5, 6, 7]
    report = _authority_report(activation)
    assert report.signal_status == "computed"
    assert report.balance_ratio == 0.0
    assert report.ideal_entropy == pytest.approx(math.log(ARCH))


def test_probe3_architectural_count_round_trips_into_the_authority(tmp_path):
    payload = build_activation_payload(
        num_experts_architectural=ARCH, counts={0: {str(k): v for k, v in FED.items()}}
    )
    f = tmp_path / "recover.json"
    f.write_text(json.dumps(payload))
    report = _authority_report(f)
    assert report.balance_ratio == pytest.approx(math.log(4) / math.log(8))
    data = load_activations_from_json(str(f))
    assert data is not None
    assert data.metadata["num_experts_architectural"] == ARCH
    assert data.metadata["starved_experts"] == [4, 5, 6, 7]


def test_probe3_ambiguous_legacy_file_is_never_guessed(tmp_path):
    payload = build_activation_payload(
        num_experts_architectural=ARCH, counts={0: {"0": 100}}
    )
    del payload["metadata"]["num_experts_architectural"]
    f = tmp_path / "ambiguous.json"
    f.write_text(json.dumps(payload))
    report = _authority_report(f)
    assert report.signal_status == "expert_count_unknown"
    assert report.balance_ratio is None


# --- PROBE 4 and 5: runtime purity and no response-path cost ---------------


def _subprocess_sys_modules(code: str) -> dict[str, bool]:
    out = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
        cwd=Path(__file__).resolve().parents[1],
    )
    return json.loads(out.stdout)


def test_probe4_import_pulls_neither_torch_nor_safetensors_nor_authority():
    result = _subprocess_sys_modules(
        "import json, sys\n"
        "import capture_proxy\n"
        "import capture_proxy.headroom\n"
        "print(json.dumps({m: m in sys.modules for m in "
        "['torch', 'safetensors', 'safetensors.torch', 'model_headroom']}))\n"
    )
    assert result == {
        "torch": False,
        "safetensors": False,
        "safetensors.torch": False,
        "model_headroom": False,
    }


def test_probe5_writer_absent_when_capture_disabled(tmp_path):
    result = _subprocess_sys_modules(
        "import json, os, sys\n"
        "os.environ['CAPTURE_HEADROOM'] = 'off'\n"
        "from capture_proxy.app import create_app\n"
        "app = create_app()\n"
        "print(json.dumps({m: m in sys.modules for m in "
        "['torch', 'safetensors', 'model_headroom']}))\n"
    )
    assert result == {
        "torch": False,
        "safetensors": False,
        "model_headroom": False,
    }


def test_probe5_building_the_writer_imports_nothing_heavy(tmp_path):
    result = _subprocess_sys_modules(
        "import json, sys\n"
        "from capture_proxy.headroom import HeadroomWriter\n"
        "from pathlib import Path\n"
        "HeadroomWriter(Path('/tmp/x'), signals=frozenset({'load_balance'}))\n"
        "print(json.dumps({m: m in sys.modules for m in "
        "['torch', 'safetensors', 'model_headroom']}))\n"
    )
    assert result == {
        "torch": False,
        "safetensors": False,
        "model_headroom": False,
    }
