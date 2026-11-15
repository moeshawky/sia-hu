"""Tests for model_headroom.acquire.published (network-isolated)."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import TYPE_CHECKING

from model_headroom.acquire.published import (
    HubFile,
    acquire_published_loss_evidence,
    extract_loss_trajectory,
    identify_training_history_artifacts,
)
from model_headroom.models.types import LossDataPoint

if TYPE_CHECKING:
    import pytest


class _FakeResponse:
    def __init__(self, body: bytes) -> None:
        self._body = body

    def read(self) -> bytes:
        return self._body

    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, *args: object) -> bool:
        return False


def _patch_urlopen(monkeypatch: pytest.MonkeyPatch, handler: object) -> None:
    monkeypatch.setattr(urllib.request, "urlopen", handler)


def test_extract_training_losses_bare_array() -> None:
    bundle = extract_loss_trajectory(
        {"losses": [2.0, 1.5, 1.2, 1.0]},
        repo_id="m/x",
        file_path="run_record.json",
    )
    assert bundle.signal_name == "loss_saturation"
    assert bundle.provenance.provenance_class == "published_context"
    payload = bundle.payload
    assert isinstance(payload, list)
    assert payload == [
        LossDataPoint(tokens_trained=0, validation_loss=2.0),
        LossDataPoint(tokens_trained=1, validation_loss=1.5),
        LossDataPoint(tokens_trained=2, validation_loss=1.2),
        LossDataPoint(tokens_trained=3, validation_loss=1.0),
    ]
    lim = " | ".join(bundle.provenance.limitations)
    assert "TRAINING losses" in lim
    assert "no validation-split loss was published" in lim
    assert "step values were not logged" in lim
    asm = " | ".join(bundle.provenance.assumptions)
    assert "step ordinal" in asm
    assert bundle.provenance.source_url == (
        "https://huggingface.co/models/m/x/resolve/main/run_record.json"
    )
    assert bundle.provenance.acquisition_method == "hub_metadata_tree"


def test_extract_with_tokens_per_step() -> None:
    bundle = extract_loss_trajectory(
        {"losses": [2.0, 1.5]},
        repo_id="m/x",
        file_path="run_record.json",
        tokens_per_step=1000,
    )
    payload = bundle.payload
    assert payload[0] == LossDataPoint(tokens_trained=0, validation_loss=2.0)
    assert payload[1] == LossDataPoint(tokens_trained=1000, validation_loss=1.5)
    asm = " | ".join(bundle.provenance.assumptions)
    assert "step * 1000" in asm


def test_extract_validation_preferred_over_training() -> None:
    artifact = {
        "losses": [2.0, 1.5],
        "eval": [
            {"step": 10, "eval_loss": 0.9},
            {"step": 20, "eval_loss": 0.8},
        ],
    }
    bundle = extract_loss_trajectory(
        artifact, repo_id="m/x", file_path="run_record.json"
    )
    payload = bundle.payload
    assert payload == [
        LossDataPoint(tokens_trained=10, validation_loss=0.9),
        LossDataPoint(tokens_trained=20, validation_loss=0.8),
    ]
    lim = " | ".join(bundle.provenance.limitations)
    assert "validation/eval" in lim
    assert "was skipped" in lim
    assert "no validation-split loss was published" not in lim


def test_extract_trainer_state_log_history() -> None:
    artifact = {
        "log_history": [
            {"step": 100, "loss": 2.0},
            {"step": 200, "loss": 1.5},
        ]
    }
    bundle = extract_loss_trajectory(
        artifact, repo_id="m/x", file_path="trainer_state.json"
    )
    payload = bundle.payload
    assert payload == [
        LossDataPoint(tokens_trained=100, validation_loss=2.0),
        LossDataPoint(tokens_trained=200, validation_loss=1.5),
    ]
    lim = " | ".join(bundle.provenance.limitations)
    assert "step values were not logged" not in lim
    assert "TRAINING losses" in lim


def test_extract_no_loss_series_unavailable() -> None:
    bundle = extract_loss_trajectory(
        {"composite": [0.5, 0.6], "accuracy": [0.9, 0.95]},
        repo_id="m/x",
        file_path="run_record.json",
    )
    assert bundle.provenance.provenance_class == "unavailable"
    assert bundle.payload is None
    lim = " | ".join(bundle.provenance.limitations)
    assert "no loss-like series" in lim


def test_extract_history_entries_refused() -> None:
    artifact = {
        "losses": [1.0, 0.9],
        "history": [
            {"step": 1, "composite": 0.5, "pres_mse": 0.3},
            {"step": 2, "composite": 0.6, "pres_mse": 0.25},
        ],
    }
    bundle = extract_loss_trajectory(
        artifact, repo_id="m/x", file_path="run_record.json"
    )
    assert bundle.provenance.provenance_class == "published_context"
    assert len(bundle.payload) == 2
    lim = " | ".join(bundle.provenance.limitations)
    assert "'history'" in lim
    assert "composite" in lim
    assert "pres_mse" in lim
    assert "not a loss" in lim


def test_lineage_attestation_gates_class() -> None:
    artifact = {"losses": [1.0, 0.9]}
    unattested = extract_loss_trajectory(
        artifact, repo_id="m/x", file_path="run_record.json"
    )
    attested = extract_loss_trajectory(
        artifact,
        repo_id="m/x",
        file_path="run_record.json",
        lineage_authenticated=True,
    )
    assert unattested.provenance.provenance_class == "published_context"
    assert attested.provenance.provenance_class == "authenticated_ancestor"
    assert "attested" in attested.provenance.applicability


def test_extract_top_level_list_artifact() -> None:
    bundle = extract_loss_trajectory(
        [1.0, 0.8, 0.6], repo_id="m/x", file_path="training_loss.json"
    )
    assert len(bundle.payload) == 3
    lim = " | ".join(bundle.provenance.limitations)
    assert "TRAINING losses" in lim

    refused = extract_loss_trajectory(
        [1.0, 0.8, 0.6], repo_id="m/x", file_path="scores.json"
    )
    assert refused.provenance.provenance_class == "unavailable"


def test_extract_nonfinite_skipped() -> None:
    bundle = extract_loss_trajectory(
        {"losses": [1.0, float("inf"), 0.5]},
        repo_id="m/x",
        file_path="run_record.json",
    )
    assert len(bundle.payload) == 2
    lim = " | ".join(bundle.provenance.limitations)
    assert "non-finite" in lim


def test_identify_candidates_priority() -> None:
    files = [
        HubFile(path="runs/events.out.tfevents.123", size=5, lfs=False),
        HubFile(path="a/b/my_loss_curve.json", size=5, lfs=False),
        HubFile(path="training_runs/control_off/run_record.json", size=5, lfs=False),
        HubFile(path="weights.bin", size=1, lfs=False),
        HubFile(path="trainer_state.json", size=5, lfs=False),
    ]
    ranked = identify_training_history_artifacts(files)
    assert [f.path for f in ranked] == [
        "training_runs/control_off/run_record.json",
        "trainer_state.json",
        "a/b/my_loss_curve.json",
        "runs/events.out.tfevents.123",
    ]


def test_acquire_network_failure_degrades(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_urlopen(url: str, timeout: float = 30.0) -> object:
        raise urllib.error.URLError("connection refused")

    _patch_urlopen(monkeypatch, fake_urlopen)
    bundle = acquire_published_loss_evidence("m/x", timeout=5.0)
    assert bundle.provenance.provenance_class == "unavailable"
    assert bundle.payload is None
    lim = " | ".join(bundle.provenance.limitations)
    assert "cannot list" in lim


def test_acquire_tfevents_only_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    tree = json.dumps(
        [{"type": "file", "path": "runs/events.out.tfevents.123", "size": 5}]
    ).encode()

    def fake_urlopen(url: str, timeout: float = 30.0) -> object:
        return _FakeResponse(tree)

    _patch_urlopen(monkeypatch, fake_urlopen)
    bundle = acquire_published_loss_evidence("m/x", timeout=5.0)
    assert bundle.provenance.provenance_class == "unavailable"
    lim = " | ".join(bundle.provenance.limitations)
    assert "protobuf" in lim


def test_acquire_no_candidate_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    tree = json.dumps([{"type": "file", "path": "weights.bin", "size": 1}]).encode()

    def fake_urlopen(url: str, timeout: float = 30.0) -> object:
        return _FakeResponse(tree)

    _patch_urlopen(monkeypatch, fake_urlopen)
    bundle = acquire_published_loss_evidence("m/x", timeout=5.0)
    assert bundle.provenance.provenance_class == "unavailable"
    lim = " | ".join(bundle.provenance.limitations)
    assert "no training-history artifact found" in lim


def test_acquire_download_not_json(monkeypatch: pytest.MonkeyPatch) -> None:
    tree = json.dumps(
        [{"type": "file", "path": "run_record.json", "size": 10}]
    ).encode()

    def fake_urlopen(url: str, timeout: float = 30.0) -> object:
        if "tree" in url:
            return _FakeResponse(tree)
        return _FakeResponse(b"this is not json")

    _patch_urlopen(monkeypatch, fake_urlopen)
    bundle = acquire_published_loss_evidence("m/x", timeout=5.0)
    assert bundle.provenance.provenance_class == "unavailable"
    lim = " | ".join(bundle.provenance.limitations)
    assert "not valid JSON" in lim


def test_acquire_end_to_end_json(monkeypatch: pytest.MonkeyPatch) -> None:
    tree = json.dumps(
        [
            {
                "type": "file",
                "path": "training_runs/control_off/run_record.json",
                "size": 50,
            }
        ]
    ).encode()
    artifact = json.dumps(
        {"losses": [1.0, 0.9, 0.85], "history": [{"step": 1, "composite": 0.5}]}
    ).encode()

    def fake_urlopen(url: str, timeout: float = 30.0) -> object:
        if "tree" in url:
            return _FakeResponse(tree)
        return _FakeResponse(artifact)

    _patch_urlopen(monkeypatch, fake_urlopen)
    bundle = acquire_published_loss_evidence("m/x", kind="datasets", timeout=5.0)
    assert bundle.provenance.provenance_class == "published_context"
    assert len(bundle.payload) == 3
    assert bundle.provenance.artifact_path == (
        "training_runs/control_off/run_record.json"
    )
    assert bundle.provenance.source_url.startswith(
        "https://huggingface.co/datasets/m/x/resolve/main/"
    )
