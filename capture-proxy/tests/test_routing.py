"""Tests for the routed-experts routing sidecar (opt-in, off by default).

Covers: base64-int32 decode contract parity with SGLang's
extract_routed_experts_from_meta_info, non-48x10 tensor shapes (width and
layer count are decoded, never hardcoded), extractor carrier precedence
(sglext first, meta_info fallback),
fire-and-forget writer artifact contents, day-directory layout, and the
default-off behavior: unset flag or absent routed_experts produce a raw
capture whose field set is identical to the untouched behavior (zero new
response-path work).
"""

import asyncio
import base64
import glob
import json
import struct
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from capture_proxy.app import CaptureProxyApp
from capture_proxy.routing import (
    RoutedExpertsWriter,
    decode_routed_experts,
    extract_routed_experts_b64,
    parse_routing_shape,
)


def _b64_int32(values):
    """Encode ints as the native-int32 base64 blob SGLang emits."""
    return base64.b64encode(b"".join(struct.pack("i", v) for v in values)).decode(
        "ascii"
    )


# ---- decode contract ----


def test_decode_derives_shape_from_tensor():
    """num_tokens is computed from the tensor; 3-layer x 2-topk is handled."""
    flat = list(range(5 * 3 * 2))  # 5 tokens x 3 layers x 2 topk
    decoded = decode_routed_experts(_b64_int32(flat), num_layers=3, topk=2)
    assert decoded.num_tokens == 5
    assert decoded.num_layers == 3
    assert decoded.topk == 2
    assert decoded.expert_ids == flat
    assert len(decoded.expert_ids) == len(flat)


def test_decode_rejects_bad_shapes():
    raw = _b64_int32([1, 2, 3])
    with pytest.raises(ValueError):
        decode_routed_experts(raw, num_layers=2, topk=2)  # 3 % 4 != 0
    with pytest.raises(ValueError):
        decode_routed_experts("!!!not-base64!!!", num_layers=1, topk=1)
    with pytest.raises(ValueError):
        decode_routed_experts(_b64_int32([1]), num_layers=0, topk=1)
    with pytest.raises(ValueError):
        decode_routed_experts(_b64_int32([1]), num_layers=1, topk=0)


def test_decode_accepts_empty_tensor():
    decoded = decode_routed_experts(
        base64.b64encode(b"").decode("ascii"), num_layers=48, topk=10
    )
    assert decoded.num_tokens == 0
    assert decoded.expert_ids == []


def test_parse_routing_shape():
    assert parse_routing_shape("48x10") == (48, 10)
    assert parse_routing_shape("48X10") == (48, 10)
    assert parse_routing_shape(" 3 x 2 ") == (3, 2)
    assert parse_routing_shape("48") is None
    assert parse_routing_shape("0x10") is None
    assert parse_routing_shape("") is None


# ---- extractor ----


def test_extractor_finds_top_level_and_per_choice_meta_info():
    raw = _b64_int32([7])
    top = extract_routed_experts_b64({"meta_info": {"routed_experts": raw}})
    assert top == raw
    per_choice = extract_routed_experts_b64(
        {"choices": [{"text": "x", "meta_info": {"routed_experts": raw}}]}
    )
    assert per_choice == raw


def test_extractor_absent_paths():
    assert extract_routed_experts_b64(None) is None
    assert extract_routed_experts_b64({}) is None
    assert extract_routed_experts_b64({"meta_info": {}}) is None
    assert extract_routed_experts_b64({"choices": []}) is None
    assert extract_routed_experts_b64({"meta_info": {"routed_experts": None}}) is None
    assert extract_routed_experts_b64({"meta_info": {"routed_experts": 12}}) is None


def test_extractor_finds_sglext_carrier():
    """sglext.routed_experts is the proven carrier."""
    raw = _b64_int32([7])
    assert extract_routed_experts_b64({"sglext": {"routed_experts": raw}}) == raw


def test_extractor_sglext_takes_precedence():
    """sglext wins when the same response carries blobs under several keys."""
    raw_sglext = _b64_int32([1])
    raw_meta = _b64_int32([2])
    body = {
        "sglext": {"routed_experts": raw_sglext},
        "meta_info": {"routed_experts": raw_meta},
        "choices": [{"meta_info": {"routed_experts": raw_meta}}],
    }
    assert extract_routed_experts_b64(body) == raw_sglext


def test_extractor_legacy_meta_info_still_works():
    """meta_info scan survives as the forward-compat fallback."""
    raw = _b64_int32([7])
    assert extract_routed_experts_b64({"meta_info": {"routed_experts": raw}}) == raw
    assert (
        extract_routed_experts_b64(
            {"choices": [{"text": "x", "meta_info": {"routed_experts": raw}}]}
        )
        == raw
    )
    # sglext present but empty/blobless — falls back to meta_info
    assert (
        extract_routed_experts_b64({"sglext": {}, "meta_info": {"routed_experts": raw}})
        == raw
    )


def test_extractor_neither_carrier_present():
    """sglext and meta_info both empty/blobless → None (absence signature)."""
    assert extract_routed_experts_b64({"sglext": {}}) is None
    assert extract_routed_experts_b64({"sglext": {"routed_experts": ""}}) is None
    assert extract_routed_experts_b64({"sglext": {"other": "x"}}) is None
    assert (
        extract_routed_experts_b64(
            {"sglext": {"routed_experts": ""}, "meta_info": {"routed_experts": None}}
        )
        is None
    )


def test_extractor_malformed_sglext_never_raises():
    """Wrong-typed sglext values are a silent miss or a defensive fallthrough."""
    raw = _b64_int32([7])
    assert extract_routed_experts_b64({"sglext": None}) is None
    assert extract_routed_experts_b64({"sglext": "not-a-dict"}) is None
    assert extract_routed_experts_b64({"sglext": {"routed_experts": None}}) is None
    assert extract_routed_experts_b64({"sglext": {"routed_experts": 12}}) is None
    assert extract_routed_experts_b64({"sglext": {"routed_experts": ["x"]}}) is None
    # malformed sglext with a valid fallback: fallback wins, no raise
    assert (
        extract_routed_experts_b64(
            {"sglext": {"routed_experts": 999}, "meta_info": {"routed_experts": raw}}
        )
        == raw
    )


def test_extractor_bad_base64_in_sglext_is_decode_error_not_absence():
    """Bad base64 is surfaced for decode rejection (probe exit 1), not absence.

    The extractor's contract is find-the-string; decode validates it. This
    keeps response-path work at zero and keeps the probe's exit-1 (decode
    mismatch) and exit-3 (absent) signatures distinguishable.
    """
    body = {"sglext": {"routed_experts": "!!!not-base64!!!"}}
    raw = extract_routed_experts_b64(body)
    assert raw == "!!!not-base64!!!"
    with pytest.raises(ValueError):
        decode_routed_experts(raw, num_layers=1, topk=1)


# ---- writer ----


def test_writer_writes_decoded_artifact(tmp_path):
    writer = RoutedExpertsWriter(tmp_path / "routing", num_layers=4, topk=3)
    raw = _b64_int32(range(2 * 4 * 3))  # 2 tokens x 4 layers x 3 topk
    capture_id = str(uuid.uuid4())
    out = writer._write_sync(
        capture_id=capture_id,
        session_id="s-1",
        turn_id="t-1",
        model="m-1",
        created_at="2026-10-13T00:00:00+00:00",
        raw_base64=raw,
    )
    assert out is not None
    assert out.name == f"{capture_id}.routing.json"
    assert out.parent.name == "2026-10-13"
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["capture_id"] == capture_id
    assert payload["num_layers"] == 4  # config-driven, not 48
    assert payload["topk"] == 3  # config-driven, not 10
    assert payload["num_tokens"] == 2
    assert payload["layout"].startswith("flat row-major int32")
    assert payload["expert_ids"] == list(range(2 * 4 * 3))


def test_writer_swallows_decode_errors(tmp_path):
    writer = RoutedExpertsWriter(tmp_path / "routing", num_layers=2, topk=2)
    out = writer._write_sync(
        capture_id=str(uuid.uuid4()),
        session_id="s",
        turn_id="t",
        model="m",
        created_at="2026-10-13T00:00:00+00:00",
        raw_base64="!!!not-base64!!!",
    )
    assert out is None
    assert list((tmp_path / "routing").rglob("*.json")) == []


def test_writer_never_raises_on_bad_dir(tmp_path):
    blocked = tmp_path / "blocked"
    blocked.write_text("I am a file, not a directory")
    writer = RoutedExpertsWriter(blocked / "routing", num_layers=1, topk=1)
    out = asyncio.run(
        writer.write_async(
            capture_id=str(uuid.uuid4()),
            session_id="s",
            turn_id="t",
            model="m",
            created_at="2026-10-13T00:00:00+00:00",
            raw_base64=_b64_int32([0]),
        )
    )
    assert out is None  # OSError suppressed, never propagates


# ---- app integration (mock upstream, no real HTTP) ----


def _completion_response(with_routing):
    body = {
        "id": "cmpl-x",
        "object": "text_completion",
        "model": "m",
        "choices": [{"index": 0, "text": "hi", "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6},
    }
    if with_routing:
        body["choices"][0]["meta_info"] = {
            "routed_experts": _b64_int32(list(range(2 * 4 * 3)))
        }
    return MagicMock(
        status_code=200,
        body=body,
        elapsed_ms=5.0,
        headers={"content-type": "application/json"},
    )


def _make_client(tmp_path, env):
    base_env = {
        "CAPTURE_BASE_PATH": str(tmp_path / "captures"),
        "CAPTURE_UPSTREAM_URL": "http://127.0.0.1:31000",
        "CAPTURE_BACKEND": "sglang",
        "CAPTURE_MODEL_TASKS": "",
        "CAPTURE_ROUTED_EXPERTS": "",
        "CAPTURE_ROUTED_EXPERTS_SHAPE": "",
        "CAPTURE_ROUTED_EXPERTS_DIR": "",
    }
    base_env.update(env)
    with patch.dict("os.environ", base_env):
        proxy = CaptureProxyApp()
    mock = MagicMock()
    mock.health_check = AsyncMock(return_value=True)
    mock.proxy = AsyncMock()
    proxy._upstream_client = mock
    return TestClient(proxy.app), mock, proxy


def _spy_writer(proxy):
    """Replace write_async/keep with spies; returns the writer."""
    writer = proxy._routed_experts_writer
    writer.write_async = AsyncMock(return_value=None)
    writer.keep = MagicMock()
    return writer


def test_flag_off_ignores_routed_experts(tmp_path):
    client, mock, proxy = _make_client(tmp_path, {})
    assert proxy._routed_experts_writer is None  # off: no writer built at all
    mock.proxy.return_value = _completion_response(with_routing=True)
    resp = client.post(
        "/v1/completions",
        json={"model": "m", "prompt": "hi", "max_tokens": 2},
    )
    assert resp.status_code == 200
    assert not (tmp_path / "captures" / "routing").exists()


def test_invalid_shape_warns_and_disables(tmp_path):
    client, mock, proxy = _make_client(
        tmp_path,
        {
            "CAPTURE_ROUTED_EXPERTS": "on",
            "CAPTURE_ROUTED_EXPERTS_SHAPE": "nonsense",
        },
    )
    assert proxy._routed_experts_writer is None
    mock.proxy.return_value = _completion_response(with_routing=True)
    resp = client.post(
        "/v1/completions",
        json={"model": "m", "prompt": "hi", "max_tokens": 2},
    )
    assert resp.status_code == 200
    assert not (tmp_path / "captures" / "routing").exists()


def test_flag_on_persists_routing(tmp_path):
    client, mock, proxy = _make_client(
        tmp_path,
        {
            "CAPTURE_ROUTED_EXPERTS": "on",
            "CAPTURE_ROUTED_EXPERTS_SHAPE": "4x3",
        },
    )
    assert proxy._routed_experts_writer is not None
    writer = _spy_writer(proxy)
    response = _completion_response(with_routing=True)
    mock.proxy.return_value = response
    resp = client.post(
        "/v1/completions",
        json={"model": "m", "prompt": "hi", "max_tokens": 2},
    )
    assert resp.status_code == 200
    # deterministic spawn check (content is validated in writer tests)
    writer.write_async.assert_awaited_once()
    kwargs = writer.write_async.call_args.kwargs
    assert (
        kwargs["raw_base64"]
        == response.body["choices"][0]["meta_info"]["routed_experts"]
    )
    assert kwargs["model"] == "m"
    (raw_line,) = _read_raw(tmp_path)
    assert kwargs["capture_id"] == raw_line["capture_id"]
    writer.keep.assert_called_once()


def test_flag_on_without_routing_writes_nothing(tmp_path):
    client, mock, proxy = _make_client(
        tmp_path,
        {
            "CAPTURE_ROUTED_EXPERTS": "on",
            "CAPTURE_ROUTED_EXPERTS_SHAPE": "4x3",
        },
    )
    writer = _spy_writer(proxy)
    mock.proxy.return_value = _completion_response(with_routing=False)
    resp = client.post(
        "/v1/completions",
        json={"model": "m", "prompt": "hi", "max_tokens": 2},
    )
    assert resp.status_code == 200
    writer.write_async.assert_not_awaited()
    assert not (tmp_path / "captures" / "routing").exists()


def test_raw_capture_field_set_unchanged(tmp_path):
    """The raw JSONL record gains no field whether routing capture is on."""
    client_off, mock_off, _ = _make_client(tmp_path / "off", {})
    mock_off.proxy.return_value = _completion_response(with_routing=True)
    client_off.post(
        "/v1/completions",
        json={"model": "m", "prompt": "hi", "max_tokens": 2},
    )
    client_on, mock_on, _ = _make_client(
        tmp_path / "on",
        {
            "CAPTURE_ROUTED_EXPERTS": "on",
            "CAPTURE_ROUTED_EXPERTS_SHAPE": "4x3",
        },
    )
    mock_on.proxy.return_value = _completion_response(with_routing=True)
    client_on.post(
        "/v1/completions",
        json={"model": "m", "prompt": "hi", "max_tokens": 2},
    )
    off_recs = _read_raw(tmp_path / "off")
    on_recs = _read_raw(tmp_path / "on")
    assert len(off_recs) == len(on_recs) == 1
    assert set(off_recs[0]) == set(on_recs[0])


def _read_raw(base):
    files = glob.glob(str(base / "captures" / "raw" / "*" / "*.jsonl"))
    files = [f for f in files if ".stream.jsonl" not in f]
    with open(files[0], encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]
