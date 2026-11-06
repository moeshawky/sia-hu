"""Tests for proxy-side return_routed_experts injection.

Covers: injection on for chat and completion when the routing sidecar is
armed, no injection when the writer is off, client-set flag (true or
false) never overridden, no injection for embeddings/rerank/classification,
streaming path forwards the flag and the client-visible response is
unchanged, and the option (b) decision rule: the raw capture keeps the
client-sent body while request_body_sha256 covers the forwarded (injected)
body, with the injected preimage stored so the handle resolves.
"""

from __future__ import annotations

import glob
import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.testclient import TestClient

from capture_proxy.app import CaptureProxyApp
from capture_proxy.blobs import content_digest
from capture_proxy.routing import ROUTED_EXPERTS_FLAG
from capture_proxy.upstream import StreamingUpstreamResponse

ARMED = {
    "CAPTURE_ROUTED_EXPERTS": "on",
    "CAPTURE_ROUTED_EXPERTS_SHAPE": "4x3",
}


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


def _read_raw(base) -> list[dict[str, Any]]:
    files = glob.glob(str(base / "captures" / "raw" / "*" / "*.jsonl"))
    files = [f for f in files if ".stream.jsonl" not in f]
    with open(files[0], encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _chat_response():
    body = {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "model": "m",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "hi"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6},
    }
    return MagicMock(
        status_code=200,
        body=body,
        elapsed_ms=5.0,
        headers={"content-type": "application/json"},
    )


def _completion_response():
    body = {
        "id": "cmpl-1",
        "object": "text_completion",
        "model": "m",
        "choices": [{"index": 0, "text": "hi", "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6},
    }
    return MagicMock(
        status_code=200,
        body=body,
        elapsed_ms=5.0,
        headers={"content-type": "application/json"},
    )


def _forwarded_body(mock) -> dict[str, Any]:
    return mock.proxy.call_args.kwargs["body"]


# ---- injection on when armed ----


def test_injection_on_chat_when_armed(tmp_path):
    client, mock, proxy = _make_client(tmp_path, ARMED)
    assert proxy._routed_experts_writer is not None
    mock.proxy.return_value = _chat_response()
    body = {"model": "m", "messages": [{"role": "user", "content": "hi"}]}
    resp = client.post("/v1/chat/completions", json=body)
    assert resp.status_code == 200
    assert _forwarded_body(mock)[ROUTED_EXPERTS_FLAG] is True
    # request-side flag only: the client sees the upstream response as-is
    assert ROUTED_EXPERTS_FLAG not in resp.json()


def test_injection_on_completion_when_armed(tmp_path):
    client, mock, proxy = _make_client(tmp_path, ARMED)
    assert proxy._routed_experts_writer is not None
    mock.proxy.return_value = _completion_response()
    resp = client.post(
        "/v1/completions",
        json={"model": "m", "prompt": "hi", "max_tokens": 2},
    )
    assert resp.status_code == 200
    assert _forwarded_body(mock)[ROUTED_EXPERTS_FLAG] is True


# ---- no injection when writer is None ----


def test_no_injection_when_writer_off(tmp_path):
    client, mock, proxy = _make_client(tmp_path, {})
    assert proxy._routed_experts_writer is None
    mock.proxy.return_value = _chat_response()
    client.post(
        "/v1/chat/completions",
        json={"model": "m", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert ROUTED_EXPERTS_FLAG not in _forwarded_body(mock)
    mock.proxy.return_value = _completion_response()
    client.post("/v1/completions", json={"model": "m", "prompt": "hi"})
    assert ROUTED_EXPERTS_FLAG not in _forwarded_body(mock)


# ---- client-sent flag wins ----


def test_client_flag_true_preserved(tmp_path):
    client, mock, _ = _make_client(tmp_path, ARMED)
    mock.proxy.return_value = _chat_response()
    body = {
        "model": "m",
        "messages": [{"role": "user", "content": "hi"}],
        ROUTED_EXPERTS_FLAG: True,
    }
    resp = client.post("/v1/chat/completions", json=body)
    assert resp.status_code == 200
    assert _forwarded_body(mock)[ROUTED_EXPERTS_FLAG] is True
    (rec,) = _read_raw(tmp_path)
    # client-set flag is part of the client body, so the raw keeps it
    assert rec["request"][ROUTED_EXPERTS_FLAG] is True
    assert rec["request_body_sha256"] == content_digest(body)


def test_client_flag_false_not_overridden_when_armed(tmp_path):
    client, mock, _ = _make_client(tmp_path, ARMED)
    mock.proxy.return_value = _chat_response()
    body = {
        "model": "m",
        "messages": [{"role": "user", "content": "hi"}],
        ROUTED_EXPERTS_FLAG: False,
    }
    resp = client.post("/v1/chat/completions", json=body)
    assert resp.status_code == 200
    # armed sidecar must not override an explicit client false
    assert _forwarded_body(mock)[ROUTED_EXPERTS_FLAG] is False
    (rec,) = _read_raw(tmp_path)
    assert rec["request"][ROUTED_EXPERTS_FLAG] is False
    assert rec["request_body_sha256"] == content_digest(body)


# ---- no injection for non-chat/completion routes ----


def test_no_injection_for_embeddings(tmp_path):
    client, mock, proxy = _make_client(tmp_path, ARMED)
    assert proxy._routed_experts_writer is not None
    mock.proxy.return_value = MagicMock(
        status_code=200,
        body={
            "object": "list",
            "data": [{"object": "embedding", "index": 0, "embedding": [0.1]}],
            "model": "m",
            "usage": {"prompt_tokens": 5, "total_tokens": 5},
        },
        elapsed_ms=5.0,
    )
    resp = client.post(
        "/v1/embeddings",
        json={"input": "hello", "model": "m"},
    )
    assert resp.status_code == 200
    assert ROUTED_EXPERTS_FLAG not in _forwarded_body(mock)


def test_no_injection_for_rerank(tmp_path):
    client, mock, proxy = _make_client(tmp_path, ARMED)
    assert proxy._routed_experts_writer is not None
    mock.proxy.return_value = MagicMock(
        status_code=200,
        body={"results": [{"index": 0, "relevance_score": 0.9}]},
        elapsed_ms=5.0,
    )
    resp = client.post(
        "/v1/rerank",
        json={"model": "m", "query": "q", "documents": ["d"]},
    )
    assert resp.status_code == 200
    assert ROUTED_EXPERTS_FLAG not in _forwarded_body(mock)


def test_no_injection_for_classification(tmp_path):
    client, mock, proxy = _make_client(tmp_path, ARMED)
    assert proxy._routed_experts_writer is not None
    mock.proxy.return_value = MagicMock(
        status_code=200,
        body={"labels": ["a"], "scores": [1.0], "model": "m"},
        elapsed_ms=5.0,
    )
    resp = client.post(
        "/v1/classification",
        json={"model": "m", "inputs": "x"},
    )
    assert resp.status_code == 200
    assert ROUTED_EXPERTS_FLAG not in _forwarded_body(mock)


# ---- streaming path ----


def test_streaming_body_carries_flag_response_unchanged(tmp_path):
    client, mock, proxy = _make_client(tmp_path, ARMED)
    assert proxy._routed_experts_writer is not None

    async def _aiter():
        for chunk in _SSE_CHUNKS:
            yield chunk

    mock.proxy.return_value = StreamingUpstreamResponse(
        status_code=200,
        headers={"content-type": "text/event-stream"},
        content_type="text/event-stream",
        stream=_aiter(),
    )
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        },
    )
    assert resp.status_code == 200
    forwarded = _forwarded_body(mock)
    assert forwarded[ROUTED_EXPERTS_FLAG] is True
    assert forwarded["stream"] is True
    # client-visible SSE bytes are verbatim: flag is request-side only
    assert resp.text == "".join(c.decode("utf-8") for c in _SSE_CHUNKS)


_SSE_CHUNKS = [
    b'data: {"id": "chatcmpl-9", "model": "m", "choices": '
    b'[{"index": 0, "delta": {"role": "assistant", "content": "hello"}, '
    b'"finish_reason": null}]}\n\n',
    b'data: {"id": "chatcmpl-9", "model": "m", "choices": '
    b'[{"index": 0, "delta": {"content": " world"}, "finish_reason": '
    b'"stop"}]}\n\n',
    b"data: [DONE]\n\n",
]


# ---- option (b): raw keeps client body, hash covers forwarded body ----


def test_option_b_raw_keeps_client_body_hash_covers_forwarded(tmp_path):
    client, mock, proxy = _make_client(tmp_path, ARMED)
    assert proxy._routed_experts_writer is not None
    mock.proxy.return_value = _chat_response()
    client_body = {"model": "m", "messages": [{"role": "user", "content": "hi"}]}
    resp = client.post("/v1/chat/completions", json=client_body)
    assert resp.status_code == 200
    injected = {**client_body, ROUTED_EXPERTS_FLAG: True}
    (rec,) = _read_raw(tmp_path)
    # INV-3: the raw capture records the body as received
    assert rec["request"] == client_body
    assert ROUTED_EXPERTS_FLAG not in rec["request"]
    # option (b): the hash covers what upstream saw — the injected body,
    # and nothing else (the one canonical diff)
    assert rec["request_body_sha256"] == content_digest(injected)
    assert rec["request_body_sha256"] != content_digest(client_body)
    # the handle resolves: the injected preimage is stored
    stored = proxy._blob_store.get(rec["request_body_sha256"])
    assert stored is not None
    assert json.loads(stored) == injected


def test_off_path_hash_and_body_unchanged(tmp_path):
    client, mock, proxy = _make_client(tmp_path, {})
    assert proxy._routed_experts_writer is None
    mock.proxy.return_value = _chat_response()
    client_body = {"model": "m", "messages": [{"role": "user", "content": "hi"}]}
    client.post("/v1/chat/completions", json=client_body)
    (rec,) = _read_raw(tmp_path)
    # no injection: forwarded body is the client body itself
    assert _forwarded_body(mock) == client_body
    assert rec["request"] == client_body
    assert rec["request_body_sha256"] == content_digest(client_body)
