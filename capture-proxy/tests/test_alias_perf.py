"""Performance smoke for alias helpers (local only, no network)."""

from __future__ import annotations

import json
import time

from capture_proxy.alias import (
    AliasMap,
    rewrite_request_model,
    sanitize_models_list,
    sanitize_response_model,
    sanitize_sse_event,
)

PUBLIC = "qwen3.8-27b-fp8"
UPSTREAM = "davidau-qwen3.8-27b-twin-turbo-fable"


def _pair() -> AliasMap:
    return AliasMap({PUBLIC: UPSTREAM})


class TestAliasPerf:
    def test_1000_iter_helpers_fast(self) -> None:
        m = _pair()
        body = {"model": PUBLIC, "messages": [{"role": "user", "content": "hi"}]}
        payload = {"model": UPSTREAM, "choices": [], "usage": {}}
        raw = json.dumps({"model": UPSTREAM, "choices": []})
        models = {"data": [{"id": UPSTREAM, "root": UPSTREAM} for _ in range(10)]}
        start = time.perf_counter()
        for _ in range(1000):
            rewrite_request_model(body, m)
            sanitize_response_model(payload, m)
            sanitize_sse_event(raw, m)
            sanitize_models_list(models, m)
        elapsed = time.perf_counter() - start
        assert elapsed < 2.0

    def test_1000_iter_off_identity_fast(self) -> None:
        body = {"model": PUBLIC}
        payload = {"model": UPSTREAM}
        start = time.perf_counter()
        for _ in range(1000):
            out, _, _ = rewrite_request_model(body, None)
            assert out is body
            assert sanitize_response_model(payload, None) is payload
            assert sanitize_models_list(payload, None) is payload
        elapsed = time.perf_counter() - start
        assert elapsed < 2.0
