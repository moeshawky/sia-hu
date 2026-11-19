"""Tests for the headroom capture sidecar (opt-in, off by default).

Covers: off-mode zero overhead (no files, no torch/safetensors import),
loader-contract weight-pair files, byte-cap row truncation, 1/N sampling,
precomputed ref paths in the JSONL line, config construction/precedence,
and streaming losslessness with headroom enabled.
"""

import asyncio
import glob
import json
import os
import sys
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from capture_proxy.app import CaptureProxyApp
from capture_proxy.headroom import (
    HeadroomLogger,
    HeadroomWriter,
    TensorBundle,
    _detect_architecture,
    _resolve_tensor_name,
    parse_byte_cap,
)
from capture_proxy.models import (
    CapturedHeadroom,
    CaptureRecord,
    HeadroomRef,
)
from capture_proxy.names import (
    ARCHITECTURE_TENSOR_NAMES,
    HEADROOM_SIGNALS,
    SIGNAL_KEYS,
)
from capture_proxy.runtime import RuntimeSnapshot
from capture_proxy.upstream import StreamingUpstreamResponse


def _make_streaming_response(
    status_code: int,
    chunks: list[bytes],
    content_type: str = "text/event-stream",
) -> StreamingUpstreamResponse:
    """Build a StreamingUpstreamResponse from raw byte chunks."""

    async def _aiter():
        for chunk in chunks:
            yield chunk

    return StreamingUpstreamResponse(
        status_code=status_code,
        headers={"content-type": content_type},
        content_type=content_type,
        stream=_aiter(),
    )


def _make_model_dir(tmp_path: Path, prefix: str = "model") -> Path:
    """Create a fake model dir: config.json + safetensors index (1 tensor)."""
    d = tmp_path / "model"
    d.mkdir()
    (d / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen3_5",
                "architectures": ["Qwen3_5ForConditionalGeneration"],
            }
        )
    )
    name = f"{prefix}.layers.0.mlp.gate_proj.weight"
    (d / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {name: "model-00001-of-00002.safetensors"}})
    )
    return d


def _fake_loader(rows: int = 4, cols: int = 3, delay: float = 0.0):
    """Build a deterministic TensorLoader returning rows x cols weights."""

    def loader(model_path: Path, tensor_name: str) -> TensorBundle | None:
        if delay:
            time.sleep(delay)
        weight = [[float(i * cols + j) for j in range(cols)] for i in range(rows)]
        gradient = [[0.0] * cols for _ in range(rows)]
        return TensorBundle(weight=weight, gradient=gradient)

    return loader


def _eligible_record(model_path: str) -> CaptureRecord:
    """Build a finalized, headroom-eligible CaptureRecord."""
    return CaptureRecord(
        session_id="sess",
        turn_id="turn",
        capture_id=str(uuid.uuid4()),
        created_at=datetime.now(UTC).isoformat(),
        latency_ms=1.0,
        client_ip="127.0.0.1",
        client_user_agent="test",
        model="m",
        request={"model": "m", "messages": []},
        upstream_status=200,
        request_type="chat",
        runtime_settings={"model_path": model_path},
    )


def _summary_records(base: Path) -> list[dict]:
    """Read summary JSONL records from raw/ and errors/ (stream companions excluded).

    Args:
        base: Capture base path (the directory holding raw/ and errors/).

    Returns:
        One record dict per summary JSONL file (first line), sorted by path
        within each category.

    Raises: None
    """
    records = []
    for category in ("raw", "errors"):
        files = glob.glob(str(base / category / "**" / "*.jsonl"), recursive=True)
        for f in sorted(f for f in files if ".stream.jsonl" not in f):
            with open(f) as fh:
                records.append(json.loads(fh.readline()))
    return records


@pytest.fixture
def app_factory(tmp_path):
    """Build CaptureProxyApp under controlled env; yields (proxy, base_path)."""

    def make(**env):
        base = str(tmp_path / "cap")
        full = {"CAPTURE_BASE_PATH": base, **env}
        with (
            patch.dict(os.environ, full),
        ):
            proxy = CaptureProxyApp()
            return proxy, base

    return make


# ---------------------------------------------------------------------------
# parse_byte_cap
# ---------------------------------------------------------------------------


class TestParseByteCap:
    def test_plain_int(self):
        assert parse_byte_cap("0") == 0
        assert parse_byte_cap("1234") == 1234

    def test_suffixes(self):
        assert parse_byte_cap("256M") == 256 * 1024 * 1024
        assert parse_byte_cap("1G") == 1024**3
        assert parse_byte_cap("128k") == 128 * 1024

    def test_invalid(self):
        for bad in ("", "abc", "-5", "1X", "1.5M"):
            with pytest.raises(ValueError):
                parse_byte_cap(bad)


# ---------------------------------------------------------------------------
# HeadroomRef / CapturedHeadroom schema
# ---------------------------------------------------------------------------


class TestHeadroomModels:
    def test_ref_valid_uuid(self):
        ref = HeadroomRef(
            capture_id=str(uuid.uuid4()),
            parent_raw=Path("/raw/x.jsonl"),
            model="m",
            model_path="/m",
        )
        assert ref.schema_version == 6
        assert ref.architecture == "unknown"
        assert ref.task is None
        assert ref.signals == {}

    def test_ref_bad_uuid_raises(self):
        with pytest.raises(ValidationError):
            HeadroomRef(
                capture_id="not-a-uuid",
                parent_raw=Path("/raw/x.jsonl"),
                model="m",
                model_path="/m",
            )

    def test_captured_headroom_defaults(self):
        rec = CapturedHeadroom(
            capture_id=str(uuid.uuid4()),
            parent_raw=Path("/raw/x.jsonl"),
            model="m",
            model_path="/m",
            collected_at="2026-01-01T00:00:00Z",
        )
        assert rec.schema_version == 7
        assert rec.task is None
        assert rec.bytes_written == 0
        assert rec.truncated is False
        assert rec.error_body is None

    def test_captured_headroom_bad_uuid_raises(self):
        with pytest.raises(ValidationError):
            CapturedHeadroom(
                capture_id="x",
                parent_raw=Path("/raw/x.jsonl"),
                model="m",
                model_path="/m",
                collected_at="2026-01-01T00:00:00Z",
            )

    def test_capture_record_v6_additive(self):
        record = _eligible_record("/m")
        assert record.schema_version == 6
        assert record.headroom is None
        # A v4 line (no headroom key) still validates.
        line = record.model_dump()
        line["schema_version"] = 4
        line.pop("headroom", None)
        assert CaptureRecord(**line).schema_version == 4
        # A v5 line (no user_id key) still validates.
        line5 = record.model_dump()
        line5["schema_version"] = 5
        line5.pop("user_id", None)
        parsed5 = CaptureRecord(**line5)
        assert parsed5.schema_version == 5
        assert parsed5.user_id is None


def _make_bert_model_dir(
    tmp_path: Path, model_type: str = "bert", prefix: str = "bert"
) -> Path:
    """Create a fake BERT/DeBERTa-style model dir: config.json + safetensors index.

    Args:
        tmp_path: Parent directory.
        model_type: model_type value in config.json (bert, deberta-v2, etc.).
        prefix: Tensor name prefix (bert., model.bert., etc.).

    Returns:
        Path to the created model directory.
    """
    d = tmp_path / "model"
    d.mkdir()
    (d / "config.json").write_text(
        json.dumps(
            {
                "model_type": model_type,
                "architectures": [f"{prefix}ForSequenceClassification"],
            }
        )
    )
    name = f"{prefix}.encoder.layer.0.attention.self.query.weight"
    (d / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {name: "model-00001-of-00002.safetensors"}})
    )
    return d


class TestHeadroomDenseModels:
    """Tests for dense/BERT-style model capture (DeBERTa, BERT, RoBERTa)."""

    def test_detect_bert_dense(self, tmp_path):
        d = _make_bert_model_dir(tmp_path, model_type="bert")
        assert _detect_architecture(d) == "dense"

    def test_detect_deberta_dense(self, tmp_path):
        d = _make_bert_model_dir(tmp_path, model_type="deberta-v2")
        assert _detect_architecture(d) == "dense"

    def test_detect_roberta_dense(self, tmp_path):
        d = _make_bert_model_dir(tmp_path, model_type="roberta")
        assert _detect_architecture(d) == "dense"

    def test_resolve_bert_attention_tensor(self, tmp_path):
        d = _make_bert_model_dir(tmp_path)
        weight_map = json.loads((d / "model.safetensors.index.json").read_text())[
            "weight_map"
        ]
        name = _resolve_tensor_name(weight_map, "attention", 0)
        assert name is not None
        assert "attention" in name and "query.weight" in name

    def test_resolve_bert_intermediate_tensor(self, tmp_path):
        d = _make_bert_model_dir(tmp_path)
        weight_map = json.loads((d / "model.safetensors.index.json").read_text())[
            "weight_map"
        ]
        # Add intermediate tensor to the map for the test
        weight_map["bert.encoder.layer.0.intermediate.dense.weight"] = (
            "model-00001-of-00002.safetensors"
        )
        name = _resolve_tensor_name(weight_map, "intermediate", 0)
        assert name == "bert.encoder.layer.0.intermediate.dense.weight"

    def test_resolve_bert_output_tensor(self, tmp_path):
        d = _make_bert_model_dir(tmp_path)
        weight_map = json.loads((d / "model.safetensors.index.json").read_text())[
            "weight_map"
        ]
        weight_map["bert.encoder.layer.0.output.dense.weight"] = (
            "model-00001-of-00002.safetensors"
        )
        name = _resolve_tensor_name(weight_map, "output", 0)
        assert name == "bert.encoder.layer.0.output.dense.weight"

    def test_dense_architecture_templates_exist(self):
        """Dense projections must be registered in ARCHITECTURE_TENSOR_NAMES."""
        for proj in ("attention", "intermediate", "output"):
            assert proj in ARCHITECTURE_TENSOR_NAMES
            templates = ARCHITECTURE_TENSOR_NAMES[proj]
            assert len(templates) >= 1
            assert "{layer}" in templates[0]


class TestExpertIdRecording:
    """Pairs record which expert was sampled (DEFER-4): indexed expert
    templates carry their index, fused slices carry the slice index, and
    dense/non-expert tensors carry None — never an invented id."""

    def _expert_dir(self, tmp_path: Path) -> Path:
        d = tmp_path / "model"
        d.mkdir()
        (d / "config.json").write_text(
            json.dumps({"model_type": "qwen3_moe", "num_experts": 8})
        )
        (d / "model.safetensors.index.json").write_text(
            json.dumps(
                {
                    "weight_map": {
                        "model.layers.0.mlp.experts.0.gate_proj.weight": "s1",
                    }
                }
            )
        )
        return d

    def test_indexed_expert_pair_carries_index(self, tmp_path: Path) -> None:
        d = self._expert_dir(tmp_path)
        w = HeadroomWriter(
            tmp_path / "headroom", weights="gate", tensor_loader=_fake_loader()
        )
        pairs, _, _, _ = w._capture_weight_pairs(d, tmp_path / "pairs.json")
        assert pairs
        assert pairs[0]["layer_name"] == "model.layers.0.mlp.experts.0.gate_proj.weight"
        assert pairs[0]["expert_id"] == 0

    def test_dense_pair_carries_none_not_zero(self, tmp_path: Path) -> None:
        d = _make_bert_model_dir(tmp_path)
        weight_map = json.loads((d / "model.safetensors.index.json").read_text())[
            "weight_map"
        ]
        name = _resolve_tensor_name(weight_map, "attention", 0)
        assert name is not None
        w = HeadroomWriter(
            tmp_path / "headroom", weights="gate", tensor_loader=_fake_loader()
        )
        pairs, _, _, _ = w._capture_weight_pairs(d, tmp_path / "pairs.json")
        assert pairs
        assert pairs[0]["layer_name"] == name
        assert pairs[0]["expert_id"] is None


# ---------------------------------------------------------------------------
# HeadroomLogger
# ---------------------------------------------------------------------------


class TestHeadroomLogger:
    def test_write_json_indented(self, tmp_path):
        logger = HeadroomLogger(tmp_path / "h")
        out = logger.write_json(tmp_path / "h" / "d" / "a.json", {"a": 1})
        text = out.read_text()
        assert '"a": 1' in text
        assert out.exists()

    def test_write_json_compact(self, tmp_path):
        logger = HeadroomLogger(tmp_path / "h")
        out = logger.write_json(
            tmp_path / "h" / "d" / "a.json", {"a": [1, 2]}, compact=True
        )
        assert out.read_text() == '{"a":[1,2]}\n'

    def test_write_json_oserror_propagates(self, tmp_path):
        blocker = tmp_path / "file"
        blocker.write_text("x")
        logger = HeadroomLogger(blocker / "sub")
        with pytest.raises(OSError):
            logger.write_json(blocker / "sub" / "a.json", {"a": 1})

    def test_day_dir_from_timestamp(self, tmp_path):
        logger = HeadroomLogger(tmp_path / "h")
        d = logger.day_dir("2026-08-21T19:40:00+00:00")
        assert d == tmp_path / "h" / "2026-08-21"
        d2 = logger.day_dir("garbage")
        assert d2.name == datetime.now(UTC).strftime("%Y-%m-%d")


# ---------------------------------------------------------------------------
# HeadroomWriter unit behavior
# ---------------------------------------------------------------------------


class TestHeadroomWriterUnit:
    def test_should_sample_interval(self):
        w = HeadroomWriter(Path("/x"), interval=2)
        assert [w.should_sample() for _ in range(4)] == [False, True, False, True]

    def test_should_sample_every(self):
        for interval in (0, 1):
            w = HeadroomWriter(Path("/x"), interval=interval)
            assert all(w.should_sample() for _ in range(3))

    def test_negative_params_raise(self):
        with pytest.raises(ValueError):
            HeadroomWriter(Path("/x"), max_bytes=-1)
        with pytest.raises(ValueError):
            HeadroomWriter(Path("/x"), interval=-2)

    def test_write_missing_model_dir_returns_none(self, tmp_path):
        w = HeadroomWriter(
            tmp_path / "h",
            weights="gate",
            signals=frozenset({"effective_rank"}),
            tensor_loader=_fake_loader(),
        )
        out = asyncio.run(
            w.write_async(
                capture_id=str(uuid.uuid4()),
                parent_raw=tmp_path / "raw.jsonl",
                model="m",
                model_path=tmp_path / "nope",
                created_at="2026-08-21T19:40:00+00:00",
            )
        )
        assert out is None
        assert not (tmp_path / "h").exists()

    def test_write_fatal_torch_unimportable_writes_sidecar(self, tmp_path):
        """Fatal weight step (torch unimportable) still writes the sidecar,
        with the reason in error_body — no dangling refs (regression)."""
        model_dir = _make_model_dir(tmp_path)
        w = HeadroomWriter(
            tmp_path / "h",
            weights="gate",
            signals=frozenset({"effective_rank"}),
        )
        capture_id = str(uuid.uuid4())
        with patch("capture_proxy.headroom._torch_stack_available", return_value=False):
            out = asyncio.run(
                w.write_async(
                    capture_id=capture_id,
                    parent_raw=tmp_path / "raw.jsonl",
                    model="m",
                    model_path=model_dir,
                    created_at="2026-08-21T19:40:00+00:00",
                )
            )
        assert out is not None
        assert out.name == f"{capture_id}.headroom.json"
        sidecar = json.loads(out.read_text())
        assert sidecar["error_body"] == "torch/safetensors not importable"
        assert all(v is None for v in sidecar["signals"].values())

    def test_write_gate_full_flow(self, tmp_path):
        model_dir = _make_model_dir(tmp_path)
        w = HeadroomWriter(
            tmp_path / "h",
            weights="gate",
            signals=frozenset({"effective_rank"}),
            tensor_loader=_fake_loader(rows=2, cols=3),
        )
        capture_id = str(uuid.uuid4())
        out = asyncio.run(
            w.write_async(
                capture_id=capture_id,
                parent_raw=tmp_path / "raw.jsonl",
                model="m",
                model_path=model_dir,
                created_at="2026-08-21T19:40:00+00:00",
            )
        )
        assert out is not None
        assert out.name == f"{capture_id}.headroom.json"
        assert out.parent == tmp_path / "h" / "2026-08-21"

        sidecar = json.loads(out.read_text())
        assert sidecar["schema_version"] == 7
        assert sidecar["capture_id"] == capture_id
        assert sidecar["architecture"] == "dense"
        assert sidecar["task"] is None
        assert sidecar["parent_raw"] == str(tmp_path / "raw.jsonl")
        assert set(sidecar["signals"].keys()) == set(SIGNAL_KEYS)

        weight_path = sidecar["signals"]["effective_rank"]
        assert weight_path is not None
        weight_file = Path(weight_path)
        assert weight_file.name == f"{capture_id}-weight_pairs.json"
        pairs = json.loads(weight_file.read_text())["pairs"]
        assert len(pairs) == 1
        pair = pairs[0]
        # model_headroom loader contract: all three keys present, 2-D,
        # gradient same shape as weight, plus the v7 truncation provenance
        # keys and the DEFER-4 expert_id key on every pair.
        assert set(pair.keys()) == {
            "gradient",
            "weight",
            "layer_name",
            "expert_id",
            "parent_shape",
            "truncated",
            "sampling_mode",
        }
        assert pair["layer_name"] == "model.layers.0.mlp.gate_proj.weight"
        assert pair["expert_id"] is None
        assert len(pair["weight"]) == 2
        assert len(pair["weight"][0]) == 3
        assert len(pair["gradient"]) == 2
        assert len(pair["gradient"][0]) == 3
        assert pair["gradient"] == [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
        assert sidecar["bytes_written"] == 2 * 3 * 4
        assert sidecar["truncated"] is False
        assert sidecar["error_body"] is None

    def test_write_weights_off_sidecar_only(self, tmp_path):
        model_dir = _make_model_dir(tmp_path)
        w = HeadroomWriter(
            tmp_path / "h",
            weights="off",
            signals=frozenset({"effective_rank"}),
            tensor_loader=_fake_loader(),
        )
        capture_id = str(uuid.uuid4())
        out = asyncio.run(
            w.write_async(
                capture_id=capture_id,
                parent_raw=tmp_path / "raw.jsonl",
                model="m",
                model_path=model_dir,
                created_at="2026-08-21T19:40:00+00:00",
            )
        )
        assert out is not None
        sidecar = json.loads(out.read_text())
        assert all(v is None for v in sidecar["signals"].values())
        assert not list((tmp_path / "h").rglob("*weight_pairs.json"))

    def test_truncation_row_math(self, tmp_path):
        # 10x4 weight = 160 float32 bytes; cap 48 -> 3 rows (48 bytes).
        model_dir = _make_model_dir(tmp_path)
        w = HeadroomWriter(
            tmp_path / "h",
            weights="gate",
            max_bytes=48,
            signals=frozenset({"effective_rank"}),
            tensor_loader=_fake_loader(rows=10, cols=4),
        )
        capture_id = str(uuid.uuid4())
        out = asyncio.run(
            w.write_async(
                capture_id=capture_id,
                parent_raw=tmp_path / "raw.jsonl",
                model="m",
                model_path=model_dir,
                created_at="2026-08-21T19:40:00+00:00",
            )
        )
        assert out is not None
        sidecar = json.loads(out.read_text())
        assert sidecar["truncated"] is True
        assert "truncated to 3 rows" in (sidecar["error_body"] or "")
        assert sidecar["bytes_written"] == 48
        pairs = json.loads(Path(sidecar["signals"]["effective_rank"]).read_text())[
            "pairs"
        ]
        assert len(pairs[0]["weight"]) == 3
        assert len(pairs[0]["gradient"]) == 3

    def test_truncation_cap_below_one_row(self, tmp_path):
        # 10x4 weight = 160 B; cap 8 -> even one row (16 B) does not fit.
        model_dir = _make_model_dir(tmp_path)
        w = HeadroomWriter(
            tmp_path / "h",
            weights="gate",
            max_bytes=8,
            signals=frozenset({"effective_rank"}),
            tensor_loader=_fake_loader(rows=10, cols=4),
        )
        capture_id = str(uuid.uuid4())
        out = asyncio.run(
            w.write_async(
                capture_id=capture_id,
                parent_raw=tmp_path / "raw.jsonl",
                model="m",
                model_path=model_dir,
                created_at="2026-08-21T19:40:00+00:00",
            )
        )
        assert out is not None
        sidecar = json.loads(out.read_text())
        assert sidecar["truncated"] is True
        assert sidecar["signals"]["effective_rank"] is None
        assert "exceeds remaining byte cap" in (sidecar["error_body"] or "")
        assert not list((tmp_path / "h").rglob("*weight_pairs.json"))

    def test_unresolvable_tensor_sidecar_with_note(self, tmp_path):
        # Index lists a tensor name the loader cannot read -> sidecar + note.
        model_dir = _make_model_dir(tmp_path)

        def missing(model_path: Path, tensor_name: str) -> None:
            return None

        w = HeadroomWriter(
            tmp_path / "h",
            weights="gate",
            signals=frozenset({"effective_rank"}),
            tensor_loader=missing,
        )
        out = asyncio.run(
            w.write_async(
                capture_id=str(uuid.uuid4()),
                parent_raw=tmp_path / "raw.jsonl",
                model="m",
                model_path=model_dir,
                created_at="2026-08-21T19:40:00+00:00",
            )
        )
        assert out is not None
        sidecar = json.loads(out.read_text())
        assert sidecar["signals"]["effective_rank"] is None
        assert "unreadable" in (sidecar["error_body"] or "")

    def test_language_model_prefix_resolution(self, tmp_path):
        model_dir = _make_model_dir(tmp_path, prefix="model.language_model")
        w = HeadroomWriter(
            tmp_path / "h",
            weights="gate",
            signals=frozenset({"effective_rank"}),
            tensor_loader=_fake_loader(rows=1, cols=2),
        )
        out = asyncio.run(
            w.write_async(
                capture_id=str(uuid.uuid4()),
                parent_raw=tmp_path / "raw.jsonl",
                model="m",
                model_path=model_dir,
                created_at="2026-08-21T19:40:00+00:00",
            )
        )
        assert out is not None
        sidecar = json.loads(out.read_text())
        assert sidecar["signals"]["effective_rank"] is not None
        pairs = json.loads(Path(sidecar["signals"]["effective_rank"]).read_text())[
            "pairs"
        ]
        assert (
            pairs[0]["layer_name"]
            == "model.language_model.layers.0.mlp.gate_proj.weight"
        )

    def test_write_fatal_no_index_writes_sidecar(self, tmp_path):
        # No safetensors index or shards under model_path -> fatal; the
        # sidecar is still written, with the reason in error_body (no
        # dangling refs — pre-patch this case wrote no files at all).
        model_dir = _make_model_dir(tmp_path)
        (model_dir / "model.safetensors.index.json").unlink()
        w = HeadroomWriter(
            tmp_path / "h",
            weights="gate",
            signals=frozenset({"effective_rank"}),
        )
        capture_id = str(uuid.uuid4())
        with patch("capture_proxy.headroom._torch_stack_available", return_value=True):
            out = asyncio.run(
                w.write_async(
                    capture_id=capture_id,
                    parent_raw=tmp_path / "raw.jsonl",
                    model="m",
                    model_path=model_dir,
                    created_at="2026-08-21T19:40:00+00:00",
                )
            )
        assert out is not None
        assert out.name == f"{capture_id}.headroom.json"
        sidecar = json.loads(out.read_text())
        assert sidecar["error_body"] == "no safetensors index under model_path"
        assert all(v is None for v in sidecar["signals"].values())

    def test_detect_architecture(self, tmp_path):
        from capture_proxy.headroom import _detect_architecture

        d = _make_model_dir(tmp_path)
        assert _detect_architecture(d) == "dense"
        (d / "config.json").write_text(json.dumps({"model_type": "qwen3_moe"}))
        assert _detect_architecture(d) == "moe"
        (d / "config.json").unlink()
        assert _detect_architecture(d) == "unknown"

    def test_planned_artifacts_agree_with_write(self, tmp_path):
        model_dir = _make_model_dir(tmp_path)
        w = HeadroomWriter(
            tmp_path / "h",
            weights="gate",
            signals=frozenset({"effective_rank"}),
            tensor_loader=_fake_loader(rows=1, cols=1),
        )
        capture_id = str(uuid.uuid4())
        sidecar_path, weight_path, activation_path, expert_weights_path = (
            w.planned_artifacts(capture_id, "2026-08-21T19:40:00+00:00")
        )
        assert sidecar_path.name == f"{capture_id}.headroom.json"
        assert weight_path == str(
            sidecar_path.parent / f"{capture_id}-weight_pairs.json"
        )
        assert activation_path is None
        assert expert_weights_path is None
        out = asyncio.run(
            w.write_async(
                capture_id=capture_id,
                parent_raw=tmp_path / "raw.jsonl",
                model="m",
                model_path=model_dir,
                created_at="2026-08-21T19:40:00+00:00",
            )
        )
        assert out == sidecar_path


# ---------------------------------------------------------------------------
# App construction (config -> writer)
# ---------------------------------------------------------------------------


class TestHeadroomConfig:
    def test_off_default_no_writer(self, app_factory):
        proxy, _ = app_factory()
        assert proxy._headroom_writer is None

    def test_all_builds_writer(self, app_factory):
        proxy, base = app_factory(
            CAPTURE_HEADROOM="all", CAPTURE_HEADROOM_WEIGHTS="gate"
        )
        w = proxy._headroom_writer
        assert w is not None
        assert w._weights == "gate"
        assert w._signals == frozenset(HEADROOM_SIGNALS)
        assert w._base_dir == Path(base) / "headroom"

    def test_list_builds_writer(self, app_factory):
        proxy, _ = app_factory(CAPTURE_HEADROOM="effective_rank")
        assert proxy._headroom_writer is not None
        assert proxy._headroom_writer._signals == frozenset({"effective_rank"})

    def test_invalid_mode_falls_back_to_off(self, app_factory, capsys):
        proxy, _ = app_factory(CAPTURE_HEADROOM="bogus")
        assert proxy._headroom_writer is None
        assert "invalid CAPTURE_HEADROOM" in capsys.readouterr().err

    def test_invalid_weights_falls_back_to_off(self, app_factory, capsys):
        proxy, _ = app_factory(
            CAPTURE_HEADROOM="all", CAPTURE_HEADROOM_WEIGHTS="sideways"
        )
        assert proxy._headroom_writer is not None
        assert proxy._headroom_writer._weights == "off"
        assert "invalid CAPTURE_HEADROOM_WEIGHTS" in capsys.readouterr().err

    def test_max_bytes_and_interval_parsed(self, app_factory):
        proxy, _ = app_factory(
            CAPTURE_HEADROOM="all",
            CAPTURE_HEADROOM_MAX_BYTES="256M",
            CAPTURE_HEADROOM_INTERVAL="3",
        )
        w = proxy._headroom_writer
        assert w._max_bytes == 256 * 1024 * 1024
        assert w._interval == 3

    def test_headroom_dir_override(self, app_factory, tmp_path):
        proxy, _ = app_factory(
            CAPTURE_HEADROOM="all", CAPTURE_HEADROOM_DIR=str(tmp_path / "custom")
        )
        assert proxy._headroom_writer._base_dir == tmp_path / "custom"


class TestPrecedence:
    def test_cli_beats_env_beats_file(self, tmp_path, monkeypatch):
        from capture_proxy import __main__ as m

        monkeypatch.setenv("CAPTURE_HEADROOM", "all")
        cfg = {"headroom": "effective_rank"}
        # env beats file
        assert m._resolve_setting(None, "CAPTURE_HEADROOM", cfg, "off") == "all"
        # cli beats env
        assert (
            m._resolve_setting("effective_rank", "CAPTURE_HEADROOM", cfg, "off")
            == "effective_rank"
        )
        # file beats default when env unset
        monkeypatch.delenv("CAPTURE_HEADROOM")
        assert m._resolve_setting(None, "CAPTURE_HEADROOM", cfg, "off") == (
            "effective_rank"
        )
        # default when nothing set
        assert m._resolve_setting(None, "CAPTURE_HEADROOM", {}, "off") == "off"

    def test_parse_args_new_flags(self, tmp_path):
        from capture_proxy import __main__ as m

        dir_value = str(tmp_path / "h")
        parsed = m.parse_args(
            [
                "--headroom",
                "effective_rank",
                "--headroom-weights",
                "gate",
                "--headroom-max-bytes",
                "256M",
                "--headroom-interval",
                "2",
                "--headroom-dir",
                dir_value,
            ]
        )
        assert parsed.headroom == "effective_rank"
        assert parsed.headroom_weights == "gate"
        assert parsed.headroom_max_bytes == "256M"
        assert parsed.headroom_interval == 2
        assert parsed.headroom_dir == dir_value


# ---------------------------------------------------------------------------
# App integration (mocked upstream, no real HTTP)
# ---------------------------------------------------------------------------


@pytest.fixture
def headroom_proxy(tmp_path):
    """App with headroom enabled, mocked upstream, fake tensor loader.

    Returns (proxy, base_path, model_dir). The app-built writer is replaced
    with an identical writer carrying a fake tensor_loader (the test venv
    has no torch/safetensors).
    """
    base = str(tmp_path / "cap")
    model_dir = _make_model_dir(tmp_path)
    env = {
        "CAPTURE_BASE_PATH": base,
        "CAPTURE_HEADROOM": "all",
        "CAPTURE_HEADROOM_WEIGHTS": "gate",
    }
    fake = _fake_loader(rows=2, cols=3)
    with (
        patch.dict(os.environ, env),
        patch.object(RuntimeSnapshot, "get") as mock_snap,
    ):
        mock_snap.return_value = MagicMock(
            run_id="run_test",
            settings={"model_path": str(model_dir), "backend": "llamacpp"},
        )
        proxy = CaptureProxyApp()
        mock_upstream = MagicMock()
        mock_upstream.health_check = AsyncMock(return_value=True)
        mock_upstream.proxy = AsyncMock(
            return_value=MagicMock(
                status_code=200,
                body={"labels": ["ok"], "scores": [0.9], "model": "m"},
            )
        )
        proxy._upstream_client = mock_upstream
        proxy._headroom_writer = HeadroomWriter(
            Path(base) / "headroom",
            weights="gate",
            signals=frozenset(HEADROOM_SIGNALS),
            tensor_loader=fake,
        )
        yield proxy, base, model_dir, mock_upstream


def _chat_body() -> dict:
    return {"model": "m", "messages": [{"role": "user", "content": "hi"}]}


def _classification_body() -> dict:
    return {"model": "clf-model", "inputs": ["hello world"]}


def _image_body() -> dict:
    return {"model": "img-model", "prompt": "a sunset"}


def _eligible_record_for(model_path: str, request_type: str) -> CaptureRecord:
    """Build a finalized, headroom-eligible CaptureRecord for a given request type."""
    return CaptureRecord(
        session_id="sess",
        turn_id="turn",
        capture_id=str(uuid.uuid4()),
        created_at=datetime.now(UTC).isoformat(),
        latency_ms=1.0,
        client_ip="127.0.0.1",
        client_user_agent="test",
        model="m",
        request={"model": "m"},
        upstream_status=200,
        request_type=request_type,
        runtime_settings={"model_path": model_path},
    )


class TestHeadroomIntegration:
    def test_off_writes_no_files_and_no_imports(self, app_factory):
        proxy, base = app_factory()
        mock = MagicMock()
        mock.proxy = AsyncMock(
            return_value=MagicMock(
                status_code=200,
                body={"id": "c1", "choices": [{"message": {"content": "ok"}}]},
            )
        )
        proxy._upstream_client = mock
        with TestClient(proxy.app) as c:
            resp = c.post("/v1/chat/completions", json=_chat_body())
        assert resp.status_code == 200
        records = _summary_records(Path(base))
        assert len(records) == 1
        assert records[0]["headroom"] is None
        assert not (Path(base) / "headroom").exists()
        # off mode never imports the tensor stack
        assert "safetensors" not in sys.modules

    def test_non_streaming_writes_sidecar_and_ref(self, headroom_proxy):
        proxy, base, _model_dir, mock = headroom_proxy
        mock.proxy = AsyncMock(
            return_value=MagicMock(
                status_code=200,
                body={"id": "c1", "choices": [{"message": {"content": "ok"}}]},
            )
        )
        with TestClient(proxy.app) as c:
            resp = c.post("/v1/chat/completions", json=_chat_body())
            assert resp.status_code == 200
            # wait for the background write task (portal loop is alive)
            deadline = time.time() + 10
            sidecars = glob.glob(
                str(Path(base) / "headroom" / "**" / "*.headroom.json"), recursive=True
            )
            while not sidecars and time.time() < deadline:
                time.sleep(0.01)
                sidecars = glob.glob(
                    str(Path(base) / "headroom" / "**" / "*.headroom.json"),
                    recursive=True,
                )

        records = _summary_records(Path(base))
        assert len(records) == 1
        record = records[0]
        ref = record["headroom"]
        assert ref is not None
        assert ref["capture_id"] == record["capture_id"]
        assert ref["schema_version"] == 6
        assert ref["task"] is None
        assert Path(ref["parent_raw"]).exists()
        assert Path(ref["parent_raw"]).name.endswith(f"-{record['turn_id']}.jsonl")
        weight_ref = ref["signals"]["effective_rank"]
        assert weight_ref is not None
        assert Path(weight_ref).exists()

        assert len(sidecars) == 1
        sidecar = json.loads(Path(sidecars[0]).read_text())
        assert sidecar["capture_id"] == record["capture_id"]
        assert sidecar["model_path"] == str(_model_dir)
        assert sidecar["architecture"] == "dense"
        pairs = json.loads(Path(weight_ref).read_text())["pairs"]
        assert set(pairs[0].keys()) == {
            "gradient",
            "weight",
            "layer_name",
            "expert_id",
            "parent_shape",
            "truncated",
            "sampling_mode",
        }

    def test_streaming_lossless_with_headroom(self, headroom_proxy):
        proxy, base, _model_dir, mock = headroom_proxy
        chunks = [
            b'data: {"choices":[{"delta":{"content":"hel"}}]}\n\n',
            b'data: {"choices":[{"delta":{"content":"lo"}}]}\n\n',
            b"data: [DONE]\n\n",
        ]
        mock.proxy = AsyncMock(return_value=_make_streaming_response(200, chunks))
        with TestClient(proxy.app) as c:
            resp = c.post("/v1/chat/completions", json={**_chat_body(), "stream": True})
            assert resp.status_code == 200
            body = resp.content
        # INV-3: bytes forwarded verbatim
        assert body == b"".join(chunks)

        records = _summary_records(Path(base))
        assert len(records) == 1
        ref = records[0]["headroom"]
        assert ref is not None
        assert ref["capture_id"] == records[0]["capture_id"]

    def test_interval_samples_one_of_n(self, headroom_proxy):
        proxy, base, _model_dir, mock = headroom_proxy
        proxy._headroom_writer._interval = 2
        mock.proxy = AsyncMock(
            return_value=MagicMock(
                status_code=200,
                body={"id": "c1", "choices": [{"message": {"content": "ok"}}]},
            )
        )
        with TestClient(proxy.app) as c:
            for _ in range(4):
                assert (
                    c.post("/v1/chat/completions", json=_chat_body()).status_code == 200
                )
            deadline = time.time() + 10
            sidecars = []
            while time.time() < deadline:
                sidecars = glob.glob(
                    str(Path(base) / "headroom" / "**" / "*.headroom.json"),
                    recursive=True,
                )
                if len(sidecars) >= 2:
                    break
                time.sleep(0.01)
        # 4 requests, interval 2 -> requests 2 and 4 sample -> 2 sidecars
        assert len(sidecars) == 2

    def test_error_capture_gets_no_sidecar(self, headroom_proxy):
        proxy, base, _model_dir, mock = headroom_proxy
        mock.proxy = AsyncMock(
            return_value=MagicMock(
                status_code=500,
                body={"error": "boom"},
            )
        )
        with TestClient(proxy.app) as c:
            resp = c.post("/v1/chat/completions", json=_chat_body())
            assert resp.status_code == 500
        records = _summary_records(Path(base))
        assert records[0]["headroom"] is None
        assert not (Path(base) / "headroom").exists()

    def test_spawn_hook_under_5ms(self, headroom_proxy):
        proxy, _base, _model_dir, _mock = headroom_proxy
        results = []

        async def probe():
            for _ in range(200):
                t0 = time.perf_counter()
                proxy._spawn_headroom(_eligible_record(str(_model_dir)))
                results.append((time.perf_counter() - t0) * 1000)
            await asyncio.sleep(0.05)

        asyncio.run(probe())
        results.sort()
        p99 = results[int(len(results) * 0.99)]
        assert p99 < 5, f"spawn hook p99 {p99:.3f}ms exceeds 5ms"

    def test_classification_gets_sidecar(self, headroom_proxy):
        proxy, base, _model_dir, _mock = headroom_proxy
        with TestClient(proxy.app) as c:
            resp = c.post("/v1/classification", json=_classification_body())
            assert resp.status_code == 200
        records = _summary_records(Path(base))
        assert records[0]["headroom"] is not None
        assert records[0]["headroom"]["task"] is None

    def test_image_gets_sidecar(self, headroom_proxy):
        proxy, base, _model_dir, _mock = headroom_proxy
        with TestClient(proxy.app) as c:
            resp = c.post("/v1/images/generations", json=_image_body())
            assert resp.status_code == 200
        records = _summary_records(Path(base))
        assert records[0]["headroom"] is not None
        assert records[0]["headroom"]["task"] is None

    def test_task_label_propagated_to_sidecar(self, tmp_path):
        """Verify CAPTURE_MODEL_TASKS task label reaches the headroom sidecar."""
        import json as _json

        base = str(tmp_path / "cap")
        model_dir = _make_model_dir(tmp_path)
        env = {
            "CAPTURE_BASE_PATH": base,
            "CAPTURE_HEADROOM": "all",
            "CAPTURE_HEADROOM_WEIGHTS": "gate",
            "CAPTURE_MODEL_TASKS": _json.dumps({"clf-model": "classification"}),
        }
        fake = _fake_loader(rows=2, cols=3)
        with (
            patch.dict(os.environ, env),
            patch.object(RuntimeSnapshot, "get") as mock_snap,
        ):
            mock_snap.return_value = MagicMock(
                run_id="run_test",
                settings={"model_path": str(model_dir), "backend": "llamacpp"},
            )
            proxy = CaptureProxyApp()
            mock_upstream = MagicMock()
            mock_upstream.proxy = AsyncMock(
                return_value=MagicMock(
                    status_code=200,
                    body={"labels": ["ok"], "scores": [0.9], "model": "clf-model"},
                )
            )
            proxy._upstream_client = mock_upstream
            proxy._headroom_writer = HeadroomWriter(
                Path(base) / "headroom",
                weights="gate",
                signals=frozenset(HEADROOM_SIGNALS),
                tensor_loader=fake,
            )
            with TestClient(proxy.app) as c:
                resp = c.post(
                    "/v1/classification",
                    json={"model": "clf-model", "inputs": ["hello"]},
                )
                assert resp.status_code == 200
            records = _summary_records(Path(base))
            assert records[0]["headroom"] is not None
            assert records[0]["headroom"]["task"] == "classification"
