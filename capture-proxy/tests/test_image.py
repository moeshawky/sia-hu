"""Tests for POST /v1/images/generations endpoint.

Covers: OpenAI-passthrough forwarding, generation options forwarded
verbatim, validation (including n >= 1), CaptureRecord fields, task-label
resolution from CAPTURE_MODEL_TASKS, alias rewrite/sanitize, error
handling, and OSError resilience during logging.
"""

import contextlib
import glob
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from capture_proxy.app import CaptureProxyApp, _compute_sha256
from capture_proxy.models import ImageResponse


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
    """Return a conforming upstream image body."""
    return {
        "data": [{"url": "https://example.invalid/img0.png"}],
        "model": "m",
        "object": "list",
    }


ALIAS_ENV = {
    "CAPTURE_MODEL_ALIASES": json.dumps({"pub-img": "upstream-img"}),
    "CAPTURE_MODEL_TASKS": json.dumps({"pub-img": "image"}),
}
ALIAS_UPSTREAM_BODY = {
    "data": [{"url": "u"}],
    "model": "upstream-img",
    "object": "list",
}


def _ok_upstream(mock, body=None, status=200):
    """Program the mocked upstream with a non-streaming response."""
    mock.proxy.return_value = MagicMock(
        status_code=status,
        body=body if body is not None else _ok_body(),
        elapsed_ms=5.0,
    )
    return mock


class TestImagePassthrough:
    """Tests for image generation requests — OpenAI-compatible passthrough."""

    def test_minimal_request_returns_200(self, client):
        c, mock, _ = client
        upstream_body = _ok_body()
        _ok_upstream(mock, upstream_body)
        resp = c.post(
            "/v1/images/generations",
            json={"model": "sdxl", "prompt": "a red bicycle"},
        )
        assert resp.status_code == 200
        assert resp.json() == upstream_body

    def test_proxies_to_v1_images_generations(self, client):
        c, mock, _ = client
        _ok_upstream(mock)
        c.post("/v1/images/generations", json={"model": "m", "prompt": "p"})
        call_args = mock.proxy.call_args
        assert call_args[1]["path"] == "/v1/images/generations"
        assert call_args[1]["method"] == "POST"
        assert call_args[1]["stream"] is False

    def test_body_forwarded_verbatim(self, client):
        c, mock, _ = client
        _ok_upstream(mock)
        body = {
            "model": "m",
            "prompt": "p",
            "n": 3,
            "size": "512x512",
            "response_format": "b64_json",
        }
        c.post("/v1/images/generations", json=body)
        assert mock.proxy.call_args[1]["body"] == body

    def test_defaults_are_not_injected(self, client):
        """The client body is forwarded as sent — no field is filled in."""
        c, mock, _ = client
        _ok_upstream(mock)
        c.post("/v1/images/generations", json={"model": "m", "prompt": "p"})
        assert mock.proxy.call_args[1]["body"] == {"model": "m", "prompt": "p"}

    def test_defaults_match_declared_schema(self, client):
        """Schema defaults are the documented ones, read off the schema itself."""
        from capture_proxy.models import ImageRequest

        request = ImageRequest(model="m", prompt="p")
        assert request.n == 1
        assert request.size == "1024x1024"
        assert request.response_format == "url"
        assert ImageResponse(data=[{"url": "u"}], model="m").object == "list"

    def test_response_matches_declared_contract(self, client):
        c, mock, _ = client
        _ok_upstream(mock)
        resp = c.post("/v1/images/generations", json={"model": "m", "prompt": "p"})
        parsed = ImageResponse(**resp.json())
        assert parsed.object == "list"
        assert parsed.model == "m"
        assert parsed.data[0]["url"] == "https://example.invalid/img0.png"

    def test_b64_json_response_forwarded(self, client):
        c, mock, _ = client
        upstream_body = {
            "data": [{"b64_json": "aGVsbG8="}],
            "model": "m",
            "object": "list",
        }
        _ok_upstream(mock, upstream_body)
        resp = c.post(
            "/v1/images/generations",
            json={"model": "m", "prompt": "p", "response_format": "b64_json"},
        )
        assert resp.status_code == 200
        assert resp.json()["data"][0]["b64_json"] == "aGVsbG8="

    def test_wider_upstream_shape_still_forwarded(self, client):
        c, mock, _ = client
        upstream_body = {**_ok_body(), "usage": {"images": 1}, "seed": 7}
        _ok_upstream(mock, upstream_body)
        resp = c.post("/v1/images/generations", json={"model": "m", "prompt": "p"})
        assert resp.status_code == 200
        assert resp.json() == upstream_body


class TestImageBackends:
    """Tests that every backend forwards /v1/images/generations unchanged.

    Image generation has no ollama translation branch and no ollama
    rejection, so ollama reaches the same upstream path.
    """

    def test_llamacpp_forwards(self, client):
        c, mock, _ = client
        _ok_upstream(mock)
        resp = c.post("/v1/images/generations", json={"model": "m", "prompt": "p"})
        assert resp.status_code == 200
        assert mock.proxy.call_args[1]["path"] == "/v1/images/generations"

    def test_ollama_forwards_instead_of_rejecting(self, make_app):
        c, mock, _ = make_app(upstream_url="http://127.0.0.1:11434")
        _ok_upstream(mock)
        resp = c.post("/v1/images/generations", json={"model": "m", "prompt": "p"})
        assert resp.status_code == 200
        assert mock.proxy.call_args[1]["path"] == "/v1/images/generations"

    def test_vllm_forwards(self, make_app):
        c, mock, _ = make_app(
            env={"CAPTURE_BACKEND": "vllm"}, upstream_url="http://127.0.0.1:8000"
        )
        _ok_upstream(mock)
        resp = c.post("/v1/images/generations", json={"model": "m", "prompt": "p"})
        assert resp.status_code == 200
        assert mock.proxy.call_args[1]["path"] == "/v1/images/generations"


class TestImageValidation:
    """Tests for input validation on image generation requests."""

    def test_empty_model_returns_422(self, client):
        c, _, _ = client
        resp = c.post("/v1/images/generations", json={"model": "", "prompt": "p"})
        assert resp.status_code == 422
        assert "model" in resp.json()["error"]

    def test_whitespace_model_returns_422(self, client):
        c, _, _ = client
        resp = c.post("/v1/images/generations", json={"model": " ", "prompt": "p"})
        assert resp.status_code == 422

    def test_missing_model_returns_422(self, client):
        c, _, _ = client
        resp = c.post("/v1/images/generations", json={"prompt": "p"})
        assert resp.status_code == 422

    def test_empty_prompt_returns_422(self, client):
        c, _, _ = client
        resp = c.post("/v1/images/generations", json={"model": "m", "prompt": ""})
        assert resp.status_code == 422
        assert "prompt" in resp.json()["error"]

    def test_whitespace_prompt_returns_422(self, client):
        c, _, _ = client
        resp = c.post("/v1/images/generations", json={"model": "m", "prompt": "  \n "})
        assert resp.status_code == 422

    def test_missing_prompt_returns_422(self, client):
        c, _, _ = client
        resp = c.post("/v1/images/generations", json={"model": "m"})
        assert resp.status_code == 422

    def test_n_zero_returns_422(self, client):
        c, _, _ = client
        resp = c.post(
            "/v1/images/generations", json={"model": "m", "prompt": "p", "n": 0}
        )
        assert resp.status_code == 422
        assert "n must be >= 1" in resp.json()["error"]

    def test_n_negative_returns_422(self, client):
        c, _, _ = client
        resp = c.post(
            "/v1/images/generations", json={"model": "m", "prompt": "p", "n": -2}
        )
        assert resp.status_code == 422

    def test_non_integer_n_returns_422(self, client):
        c, _, _ = client
        resp = c.post(
            "/v1/images/generations", json={"model": "m", "prompt": "p", "n": "many"}
        )
        assert resp.status_code == 422

    def test_invalid_response_format_returns_422(self, client):
        c, _, _ = client
        resp = c.post(
            "/v1/images/generations",
            json={"model": "m", "prompt": "p", "response_format": "png"},
        )
        assert resp.status_code == 422

    def test_invalid_json_returns_422(self, client):
        c, _, _ = client
        resp = c.post(
            "/v1/images/generations",
            content=b"not json",
            headers={"Content-Type": "application/json"},
        )
        assert resp.status_code == 422

    def test_validation_failure_is_not_captured_as_a_request(self, client, tmp_path):
        """422 validation exits before a CaptureRecord is built."""
        c, mock, _ = client
        c.post("/v1/images/generations", json={"model": "m"})
        mock.proxy.assert_not_called()
        assert _get_summary_records(tmp_path) == []


class TestImageCaptureRecord:
    """Tests for CaptureRecord fields populated by image requests."""

    def test_request_type_is_image(self, client, tmp_path):
        c, mock, _ = client
        _ok_upstream(mock)
        c.post("/v1/images/generations", json={"model": "m", "prompt": "p"})
        assert _get_summary_records(tmp_path)[-1]["request_type"] == "image"

    def test_record_has_request_body(self, client, tmp_path):
        c, mock, _ = client
        _ok_upstream(mock)
        c.post(
            "/v1/images/generations",
            json={"model": "m", "prompt": "a cat", "n": 2, "size": "256x256"},
        )
        record = _get_summary_records(tmp_path)[-1]
        assert record["request"]["model"] == "m"
        assert record["request"]["prompt"] == "a cat"
        assert record["request"]["n"] == 2
        assert record["request"]["size"] == "256x256"

    def test_record_has_upstream_response(self, client, tmp_path):
        c, mock, _ = client
        _ok_upstream(mock)
        c.post("/v1/images/generations", json={"model": "m", "prompt": "p"})
        record = _get_summary_records(tmp_path)[-1]
        assert (
            record["response"]["data"][0]["url"] == "https://example.invalid/img0.png"
        )
        assert record["upstream_status"] == 200

    def test_record_has_identity_fields(self, client, tmp_path):
        c, mock, _ = client
        _ok_upstream(mock)
        c.post(
            "/v1/images/generations",
            json={"model": "m", "prompt": "p"},
            headers={"x-session-id": "s9", "x-user-id": "bob"},
        )
        record = _get_summary_records(tmp_path)[-1]
        assert record["session_id"] == "s9"
        assert record["user_id"] == "bob"

    def test_request_sha_hashes_forwarded_body(self, client, tmp_path):
        c, mock, _ = client
        _ok_upstream(mock)
        body = {"model": "m", "prompt": "p"}
        c.post("/v1/images/generations", json=body)
        record = _get_summary_records(tmp_path)[-1]
        assert record["request_body_sha256"] == _compute_sha256(body)
        assert record["messages_sha256"] == ""

    def test_upstream_error_routed_to_errors(self, client, tmp_path):
        c, mock, _ = client
        _ok_upstream(mock, {"error": "internal"}, status=500)
        c.post("/v1/images/generations", json={"model": "m", "prompt": "p"})
        record = _get_error_records(tmp_path)[-1]
        assert record["upstream_status"] == 500
        assert record["request_type"] == "image"


class TestImageTaskLabels:
    """Task resolution: operator map first, structural fallback second.

    A map miss no longer leaves the capture unlabelled — null-on-miss is
    the defect these tests previously encoded, and the map is off by
    default, so it nulled every capture in the corpus. Each test keeps its
    intent about the MAP and additionally pins the fallback.
    """

    def test_task_falls_back_to_structure_when_map_is_off(self, client, tmp_path):
        c, mock, _ = client
        _ok_upstream(mock)
        c.post("/v1/images/generations", json={"model": "m", "prompt": "p"})
        record = _get_summary_records(tmp_path)[-1]
        assert record["task"] == "image.generate"
        assert record["task_rule"] == "request_type.image"

    def test_task_is_populated_on_exact_hit(self, make_app, tmp_path):
        c, mock, _ = make_app(env={"CAPTURE_MODEL_TASKS": '{"sdxl":"image"}'})
        _ok_upstream(mock)
        c.post("/v1/images/generations", json={"model": "sdxl", "prompt": "p"})
        record = _get_summary_records(tmp_path)[-1]
        assert record["task"] == "image"
        assert record["task_rule"] == "operator_model_task_map"

    def test_map_miss_falls_back_rather_than_nulling(self, make_app, tmp_path):
        c, mock, _ = make_app(env={"CAPTURE_MODEL_TASKS": '{"sdxl":"image"}'})
        _ok_upstream(mock)
        c.post("/v1/images/generations", json={"model": "other", "prompt": "p"})
        record = _get_summary_records(tmp_path)[-1]
        assert record["task"] != "image"
        assert record["task_rule"] != "operator_model_task_map"

    def test_malformed_map_fails_open(self, make_app, tmp_path):
        c, mock, _ = make_app(env={"CAPTURE_MODEL_TASKS": "{not json"})
        _ok_upstream(mock)
        resp = c.post("/v1/images/generations", json={"model": "m", "prompt": "p"})
        assert resp.status_code == 200
        record = _get_summary_records(tmp_path)[-1]
        assert record["task_rule"] != "operator_model_task_map"

    def test_one_map_labels_both_new_routes(self, make_app, tmp_path):
        """A single configured map labels classification and image records.

        Compared order-independently: the logger writes one file per capture
        ({timestamp}-{turn_id}.jsonl), so directory listing order across two
        captures in the same second is not stable.
        """
        tasks = json.dumps({"clf": "classification", "img": "image"})
        c, mock, _ = make_app(env={"CAPTURE_MODEL_TASKS": tasks})
        _ok_upstream(mock, {"labels": ["a"], "scores": [1.0], "model": "clf"})
        c.post("/v1/classification", json={"model": "clf", "inputs": "x"})
        _ok_upstream(mock)
        c.post("/v1/images/generations", json={"model": "img", "prompt": "p"})
        records = _get_summary_records(tmp_path)
        assert len(records) == 2
        assert sorted(r["task"] for r in records) == ["classification", "image"]
        assert {r["request_type"] for r in records} == {"classification", "image"}


class TestImageAlias:
    """Tests for alias rewrite/sanitize on the image route."""

    def test_rewrite_before_hash(self, make_app, tmp_path):
        c, mock, _ = make_app(env=ALIAS_ENV, upstream_url="http://127.0.0.1:8080")
        _ok_upstream(mock, ALIAS_UPSTREAM_BODY)
        c.post("/v1/images/generations", json={"model": "pub-img", "prompt": "p"})
        forwarded = mock.proxy.call_args[1]["body"]
        assert forwarded["model"] == "upstream-img"
        record = _get_summary_records(tmp_path)[-1]
        assert record["model"] == "pub-img"
        assert record["request_model_original"] == "pub-img"
        assert record["request_model_forwarded"] == "upstream-img"
        # INV-4: the hash covers the forwarded body, not the client body.
        assert record["request_body_sha256"] == _compute_sha256(forwarded)
        assert record["request"] == forwarded

    def test_sanitize_before_log_and_serialize(self, make_app, tmp_path):
        c, mock, _ = make_app(env=ALIAS_ENV, upstream_url="http://127.0.0.1:8080")
        _ok_upstream(mock, ALIAS_UPSTREAM_BODY)
        resp = c.post(
            "/v1/images/generations", json={"model": "pub-img", "prompt": "p"}
        )
        assert resp.json()["model"] == "pub-img"
        assert _get_summary_records(tmp_path)[-1]["response"]["model"] == "pub-img"


class TestImageErrors:
    """Tests for error handling — upstream failures and OSError resilience."""

    def test_upstream_down_returns_502(self, client):
        c, mock, _ = client
        mock.proxy.side_effect = Exception("connection refused")
        resp = c.post("/v1/images/generations", json={"model": "m", "prompt": "p"})
        assert resp.status_code == 502
        assert resp.json()["error"] == "upstream unavailable"

    def test_upstream_404_passthrough(self, client):
        c, mock, _ = client
        _ok_upstream(mock, {"error": "model not found"}, status=404)
        resp = c.post("/v1/images/generations", json={"model": "m", "prompt": "p"})
        assert resp.status_code == 404
        assert resp.json()["error"] == "model not found"

    def test_upstream_429_passthrough(self, client):
        c, mock, _ = client
        _ok_upstream(mock, {"error": "rate limited"}, status=429)
        resp = c.post("/v1/images/generations", json={"model": "m", "prompt": "p"})
        assert resp.status_code == 429

    def test_502_record_is_captured(self, client, tmp_path):
        c, mock, _ = client
        mock.proxy.side_effect = Exception("connection refused")
        c.post("/v1/images/generations", json={"model": "m", "prompt": "p"})
        record = _get_error_records(tmp_path)[-1]
        assert record["upstream_status"] == 502
        assert record["request_type"] == "image"
        assert record["error_body"] == "connection refused"

    def test_os_error_during_logging_does_not_break_response(self, client):
        c, mock, proxy = client
        _ok_upstream(mock)
        with patch.object(
            proxy._capture_logger, "log", side_effect=OSError("disk full")
        ):
            resp = c.post("/v1/images/generations", json={"model": "m", "prompt": "p"})
        assert resp.status_code == 200
        assert resp.json()["object"] == "list"

    def test_os_error_during_error_logging_does_not_break_502(self, client):
        c, mock, proxy = client
        mock.proxy.side_effect = Exception("connection refused")
        with patch.object(
            proxy._capture_logger, "log_error", side_effect=OSError("disk full")
        ):
            resp = c.post("/v1/images/generations", json={"model": "m", "prompt": "p"})
        assert resp.status_code == 502
        assert resp.json()["error"] == "upstream unavailable"

    def test_os_error_during_upstream_error_log_does_not_break_404(self, client):
        c, mock, proxy = client
        _ok_upstream(mock, {"error": "not found"}, status=404)
        with patch.object(
            proxy._capture_logger, "log", side_effect=OSError("disk full")
        ):
            resp = c.post("/v1/images/generations", json={"model": "m", "prompt": "p"})
        assert resp.status_code == 404

    def test_os_error_during_validation_log_does_not_break_422(self, client):
        """A disk fault while logging a validation failure still returns 422.

        Failure class: capture I/O escaping into the response path. The
        validation-error log is not routed through _log_capture, so it needs
        its own suppression.
        """
        c, mock, proxy = client
        with patch.object(
            proxy._capture_logger, "log_error", side_effect=OSError("disk full")
        ):
            resp = c.post("/v1/images/generations", json={"model": "m"})
        assert resp.status_code == 422
        assert "image" in resp.json()["error"]
        mock.proxy.assert_not_called()
