"""Tests for the gpu-arm native JSON passthrough routes.

Seven routes (/v1/systemone, /v1/classify, /v1/batch-classify, /v1/ner,
/v1/batch-ner, /v1/timeseries, /v1/batch-timeseries) forward the request
body verbatim to the upstream, return the upstream response unchanged, and
capture both sides to JSONL. There is no declared request schema and no
required-field validation — the proxy never invents a shape the checkpoint
never had. Multipart /v1/transcribe is deliberately NOT registered.
"""

import contextlib
import glob
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from capture_proxy.app import CaptureProxyApp, _compute_sha256

# (route path, request_type label stored on the capture)
NATIVE_ROUTES = [
    ("/v1/systemone", "systemone"),
    ("/v1/classify", "classify"),
    ("/v1/batch-classify", "batch_classify"),
    ("/v1/ner", "ner"),
    ("/v1/batch-ner", "batch_ner"),
    ("/v1/timeseries", "timeseries"),
    ("/v1/batch-timeseries", "batch_timeseries"),
]


@pytest.fixture
def make_app(tmp_path):
    """Yield a factory returning (TestClient, mock_upstream, proxy).

    The env patch is held open on an ExitStack for the whole test rather
    than only during construction, because task labels are resolved from
    ``CAPTURE_MODEL_TASKS`` at request time — a patch that ends when the
    factory returns would leave the request path reading the ambient
    environment.

    Args:
        tmp_path: Per-test directory used as the capture base path.

    Yields:
        Callable: ``factory(env=None, upstream_url=...)`` -> (TestClient,
            MagicMock upstream, CaptureProxyApp). ``env`` keys are merged
            over the defaults (``CAPTURE_MODEL_TASKS`` pinned to "" = off).
    """
    stack = contextlib.ExitStack()

    def _factory(env=None, upstream_url="http://127.0.0.1:8001"):
        base_env = {
            "CAPTURE_BASE_PATH": str(tmp_path / "captures"),
            "CAPTURE_UPSTREAM_URL": upstream_url,
            "CAPTURE_MODEL_TASKS": "",
        }
        if env:
            base_env.update(env)
        stack.enter_context(patch.dict("os.environ", base_env))
        proxy = CaptureProxyApp()
        mock_upstream = MagicMock()
        mock_upstream.health_check = AsyncMock(return_value=True)
        mock_upstream.proxy = AsyncMock()
        proxy._upstream_client = mock_upstream
        return TestClient(proxy.app), mock_upstream, proxy

    yield _factory
    stack.close()


@pytest.fixture
def client(make_app):
    """Default app: llama.cpp-style upstream, task labels off."""
    return make_app()


def _get_summary_records(tmp_path):
    """Read all summary JSONL records from captures/raw."""
    jsonl_files = glob.glob(
        str(tmp_path / "captures" / "raw" / "**" / "*.jsonl"),
        recursive=True,
    )
    summary_files = [f for f in jsonl_files if ".stream.jsonl" not in f]
    records = []
    for f in summary_files:
        with open(f) as fh:
            records.append(json.loads(fh.readline()))
    return records


def _get_error_records(tmp_path):
    """Read all summary JSONL records from captures/errors."""
    jsonl_files = glob.glob(
        str(tmp_path / "captures" / "errors" / "**" / "*.jsonl"),
        recursive=True,
    )
    summary_files = [f for f in jsonl_files if ".stream.jsonl" not in f]
    records = []
    for f in summary_files:
        with open(f) as fh:
            records.append(json.loads(fh.readline()))
    return records


def _ok_body():
    """Return a gpu-arm classify-style upstream body (label + score)."""
    return {"label": "positive", "score": 0.98, "model": "finbert"}


def _ok_upstream(mock, body=None, status=200):
    """Program the mocked upstream with a non-streaming response."""
    mock.proxy.return_value = MagicMock(
        status_code=status,
        body=body if body is not None else _ok_body(),
        elapsed_ms=5.0,
    )
    return mock


NATIVE_ALIAS_ENV = {
    "CAPTURE_MODEL_ALIASES": '{"finbert":"finbert-upstream"}',
    "CAPTURE_MODEL_TASKS": '{"finbert":"classification"}',
}


class TestNativeRouteRegistration:
    """Each native route is registered and forwards to its own upstream path."""

    @pytest.mark.parametrize("route,expected_type", NATIVE_ROUTES)
    def test_route_is_registered_and_forwards(self, client, route, expected_type):
        c, mock, _ = client
        _ok_upstream(mock)
        resp = c.post(route, json={"model": "finbert", "text": "Revenue beat."})
        assert resp.status_code == 200
        call = mock.proxy.call_args
        assert call[1]["path"] == route
        assert call[1]["method"] == "POST"
        assert call[1]["stream"] is False
        # request_type label stored on the capture matches the route
        # (asserted via the record in TestNativeCaptureRecord; here the
        # forwarding path is what registration guarantees).
        del expected_type

    def test_all_seven_routes_present(self, client):
        _, _, proxy = client
        paths = {getattr(r, "path", None) for r in proxy.app.routes}
        for route, _ in NATIVE_ROUTES:
            assert route in paths

    def test_transcribe_is_not_registered(self, client):
        """Multipart /v1/transcribe is out of scope."""
        _, _, proxy = client
        paths = {getattr(r, "path", None) for r in proxy.app.routes}
        assert "/v1/transcribe" not in paths


class TestNativePassthrough:
    """Body forwards verbatim; the upstream response returns unchanged."""

    def test_body_forwarded_verbatim_with_extra_fields(self, client):
        c, mock, _ = client
        _ok_upstream(mock)
        body = {
            "model": "finbert",
            "text": "Revenue beat.",
            "temperature": 0.2,
            "top_p": 0.9,
            "extra_flag": True,
        }
        c.post("/v1/classify", json=body)
        # No schema filter: every field the client sent is forwarded.
        assert mock.proxy.call_args[1]["body"] == body

    def test_wider_upstream_shape_still_forwarded(self, client):
        """A response carrying extra keys is returned, never filtered."""
        c, mock, _ = client
        upstream_body = {**_ok_body(), "meta": {"ms": 3}, "top_k": [0.98, 0.01]}
        _ok_upstream(mock, upstream_body)
        resp = c.post("/v1/classify", json={"model": "finbert", "text": "x"})
        assert resp.status_code == 200
        assert resp.json() == upstream_body

    @pytest.mark.parametrize("route,_type", NATIVE_ROUTES)
    def test_ollama_forwards_instead_of_rejecting(self, make_app, route, _type):
        """Native routes have no ollama branch — every backend forwards."""
        c, mock, _ = make_app(upstream_url="http://127.0.0.1:11434")
        _ok_upstream(mock)
        resp = c.post(route, json={"model": "m", "text": "x"})
        assert resp.status_code == 200
        assert mock.proxy.call_args[1]["path"] == route


class TestNativeCaptureRecord:
    """CaptureRecord fields populated by a native passthrough request."""

    def test_request_holds_model_and_text(self, client, tmp_path):
        c, mock, _ = client
        _ok_upstream(mock)
        c.post("/v1/classify", json={"model": "finbert", "text": "Revenue beat."})
        record = _get_summary_records(tmp_path)[-1]
        assert record["request"]["model"] == "finbert"
        assert record["request"]["text"] == "Revenue beat."

    def test_response_holds_label_and_score(self, client, tmp_path):
        c, mock, _ = client
        _ok_upstream(mock)
        c.post("/v1/classify", json={"model": "finbert", "text": "x"})
        record = _get_summary_records(tmp_path)[-1]
        assert record["response"]["label"] == "positive"
        assert record["response"]["score"] == 0.98
        assert record["upstream_status"] == 200

    @pytest.mark.parametrize("route,expected_type", NATIVE_ROUTES)
    def test_request_type_is_stored(self, client, tmp_path, route, expected_type):
        c, mock, _ = client
        _ok_upstream(mock)
        c.post(route, json={"model": "m", "text": "x"})
        record = _get_summary_records(tmp_path)[-1]
        assert record["request_type"] == expected_type

    def test_request_sha_hashes_forwarded_body(self, client, tmp_path):
        c, mock, _ = client
        _ok_upstream(mock)
        body = {"model": "finbert", "text": "x"}
        c.post("/v1/classify", json=body)
        record = _get_summary_records(tmp_path)[-1]
        assert record["request_body_sha256"] == _compute_sha256(body)
        assert record["messages_sha256"] == ""

    def test_record_has_identity_fields(self, client, tmp_path):
        c, mock, _ = client
        _ok_upstream(mock)
        c.post(
            "/v1/classify",
            json={"model": "finbert", "text": "x"},
            headers={"x-session-id": "s1", "x-user-id": "alice"},
        )
        record = _get_summary_records(tmp_path)[-1]
        assert record["session_id"] == "s1"
        assert record["user_id"] == "alice"


class TestNativeTaskLabels:
    """model_tasks must be able to label the native routes."""

    def test_task_is_populated_on_exact_hit(self, make_app, tmp_path):
        c, mock, _ = make_app(
            env={"CAPTURE_MODEL_TASKS": '{"finbert":"classification"}'}
        )
        _ok_upstream(mock)
        c.post("/v1/classify", json={"model": "finbert", "text": "Revenue beat."})
        record = _get_summary_records(tmp_path)[-1]
        assert record["task"] == "classification"
        assert record["task_rule"] == "operator_model_task_map"

    def test_clef_flash_is_labeled_decision(self, make_app, tmp_path):
        c, mock, _ = make_app(env={"CAPTURE_MODEL_TASKS": '{"clef_flash":"decision"}'})
        _ok_upstream(mock)
        c.post(
            "/v1/systemone",
            json={"model": "clef_flash", "state": "active"},
        )
        record = _get_summary_records(tmp_path)[-1]
        assert record["task"] == "decision"
        assert record["task_rule"] == "operator_model_task_map"

    def test_map_miss_falls_back_rather_than_nulling(self, client, tmp_path):
        """With the map off the structural classifier still labels the row."""
        c, mock, _ = client
        _ok_upstream(mock)
        c.post("/v1/classify", json={"model": "finbert", "text": "x"})
        record = _get_summary_records(tmp_path)[-1]
        assert record["task"] is not None
        assert record["task_rule"] != "operator_model_task_map"

    def test_task_lookup_is_case_sensitive(self, make_app, tmp_path):
        c, mock, _ = make_app(env={"CAPTURE_MODEL_TASKS": '{"FinBERT":"x"}'})
        _ok_upstream(mock)
        c.post("/v1/classify", json={"model": "finbert", "text": "x"})
        record = _get_summary_records(tmp_path)[-1]
        assert record["task"] != "x"
        assert record["task_rule"] != "operator_model_task_map"


class TestNativeAlias:
    """Alias rewrite-before-hash and sanitize-before-serialize still apply."""

    def test_model_rewrite_and_sanitize(self, make_app, tmp_path):
        c, mock, _ = make_app(env=NATIVE_ALIAS_ENV)
        _ok_upstream(
            mock, {"label": "positive", "score": 0.9, "model": "finbert-upstream"}
        )
        resp = c.post("/v1/classify", json={"model": "finbert", "text": "x"})
        # Client sees the public id, not the upstream id.
        assert resp.json()["model"] == "finbert"
        # Upstream received the forwarded id.
        assert mock.proxy.call_args[1]["body"]["model"] == "finbert-upstream"
        record = _get_summary_records(tmp_path)[-1]
        assert record["model"] == "finbert"
        assert record["request_model_original"] == "finbert"
        assert record["request_model_forwarded"] == "finbert-upstream"


class TestNativeValidation:
    """Native routes reject non-JSON and non-object bodies, nothing else."""

    def test_invalid_json_returns_422(self, client):
        c, _, _ = client
        resp = c.post(
            "/v1/classify",
            content=b"not json",
            headers={"Content-Type": "application/json"},
        )
        assert resp.status_code == 422

    def test_non_object_json_returns_422(self, client):
        c, _, _ = client
        resp = c.post(
            "/v1/classify",
            content=b'["a","b"]',
            headers={"Content-Type": "application/json"},
        )
        assert resp.status_code == 422

    def test_validation_failure_is_not_captured(self, client, tmp_path):
        c, mock, _ = client
        c.post(
            "/v1/classify",
            content=b"not json",
            headers={"Content-Type": "application/json"},
        )
        mock.proxy.assert_not_called()
        assert _get_summary_records(tmp_path) == []
        assert _get_error_records(tmp_path) == []

    def test_empty_body_forwards_verbatim(self, client):
        """No required-field validation: an empty body is forwarded as {}."""
        c, mock, _ = client
        _ok_upstream(mock)
        resp = c.post("/v1/classify", headers={"Content-Type": "application/json"})
        assert resp.status_code == 200
        assert mock.proxy.call_args[1]["body"] == {}


class TestNativeErrors:
    """Upstream failures route to errors/ and 502 on connection failure."""

    def test_upstream_error_routed_to_errors(self, client, tmp_path):
        c, mock, _ = client
        _ok_upstream(mock, {"error": "internal"}, status=500)
        resp = c.post("/v1/classify", json={"model": "finbert", "text": "x"})
        assert resp.status_code == 500
        record = _get_error_records(tmp_path)[-1]
        assert record["upstream_status"] == 500
        assert record["request_type"] == "classify"

    def test_upstream_exception_returns_502(self, client, tmp_path):
        c, mock, _ = client
        mock.proxy.side_effect = RuntimeError("conn refused")
        resp = c.post("/v1/classify", json={"model": "finbert", "text": "x"})
        assert resp.status_code == 502
        record = _get_error_records(tmp_path)[-1]
        assert record["upstream_status"] == 502
        assert "conn refused" in record["error_body"]
