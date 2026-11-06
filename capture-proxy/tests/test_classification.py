"""Tests for POST /v1/classification endpoint.

Covers: OpenAI-passthrough forwarding for every backend, validation,
CaptureRecord fields, task-label resolution from CAPTURE_MODEL_TASKS,
alias rewrite/sanitize, error handling, and OSError resilience during
logging.
"""

import contextlib
import glob
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from capture_proxy.app import CaptureProxyApp, _compute_sha256
from capture_proxy.models import ClassificationResponse


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
    """Return a conforming upstream classification body."""
    return {"labels": ["positive", "negative"], "scores": [0.98, 0.02], "model": "m"}


ALIAS_ENV = {
    "CAPTURE_MODEL_ALIASES": json.dumps({"pub-clf": "upstream-clf"}),
    "CAPTURE_MODEL_TASKS": json.dumps({"pub-clf": "classification"}),
}
ALIAS_UPSTREAM_BODY = {
    "labels": ["a"],
    "scores": [1.0],
    "model": "upstream-clf",
}


def _ok_upstream(mock, body=None, status=200):
    """Program the mocked upstream with a non-streaming response."""
    mock.proxy.return_value = MagicMock(
        status_code=status,
        body=body if body is not None else _ok_body(),
        elapsed_ms=5.0,
    )
    return mock


class TestClassificationPassthrough:
    """Tests for classification requests — OpenAI-compatible passthrough."""

    def test_single_input_returns_200(self, client):
        c, mock, _ = client
        upstream_body = _ok_body()
        _ok_upstream(mock, upstream_body)
        resp = c.post(
            "/v1/classification",
            json={"model": "classifier", "inputs": "a great movie"},
        )
        assert resp.status_code == 200
        assert resp.json() == upstream_body

    def test_list_input_returns_200(self, client):
        c, mock, _ = client
        _ok_upstream(mock)
        resp = c.post(
            "/v1/classification",
            json={"model": "classifier", "inputs": ["one", "two"]},
        )
        assert resp.status_code == 200
        assert resp.json()["labels"] == ["positive", "negative"]

    def test_proxies_to_v1_classification(self, client):
        c, mock, _ = client
        _ok_upstream(mock)
        c.post("/v1/classification", json={"model": "m", "inputs": "x"})
        call_args = mock.proxy.call_args
        assert call_args[1]["path"] == "/v1/classification"
        assert call_args[1]["method"] == "POST"
        assert call_args[1]["stream"] is False

    def test_body_forwarded_verbatim(self, client):
        c, mock, _ = client
        _ok_upstream(mock)
        body = {
            "model": "m",
            "inputs": ["a", "b"],
            "parameters": {"truncate": True},
        }
        c.post("/v1/classification", json=body)
        assert mock.proxy.call_args[1]["body"] == body

    def test_response_matches_declared_contract(self, client):
        c, mock, _ = client
        _ok_upstream(mock)
        resp = c.post("/v1/classification", json={"model": "m", "inputs": "x"})
        parsed = ClassificationResponse(**resp.json())
        assert parsed.labels == ["positive", "negative"]
        assert parsed.scores == [0.98, 0.02]
        assert parsed.model == "m"

    def test_wider_upstream_shape_still_forwarded(self, client):
        """A response carrying extra keys is forwarded, never filtered."""
        c, mock, _ = client
        upstream_body = {**_ok_body(), "meta": {"ms": 3}, "top_k": 2}
        _ok_upstream(mock, upstream_body)
        resp = c.post("/v1/classification", json={"model": "m", "inputs": "x"})
        assert resp.status_code == 200
        assert resp.json() == upstream_body


class TestClassificationBackends:
    """Tests that every backend forwards /v1/classification unchanged.

    Classification has no ollama translation branch and no ollama rejection
    (unlike rerank), so ollama reaches the same upstream path.
    """

    def test_llamacpp_forwards(self, client):
        c, mock, _ = client
        _ok_upstream(mock)
        resp = c.post("/v1/classification", json={"model": "m", "inputs": "x"})
        assert resp.status_code == 200
        assert mock.proxy.call_args[1]["path"] == "/v1/classification"

    def test_ollama_forwards_instead_of_rejecting(self, make_app):
        c, mock, _ = make_app(upstream_url="http://127.0.0.1:11434")
        _ok_upstream(mock)
        resp = c.post("/v1/classification", json={"model": "m", "inputs": "x"})
        assert resp.status_code == 200
        assert mock.proxy.call_args[1]["path"] == "/v1/classification"

    def test_vllm_forwards(self, make_app):
        c, mock, _ = make_app(
            env={"CAPTURE_BACKEND": "vllm"}, upstream_url="http://127.0.0.1:8000"
        )
        _ok_upstream(mock)
        resp = c.post("/v1/classification", json={"model": "m", "inputs": "x"})
        assert resp.status_code == 200
        assert mock.proxy.call_args[1]["path"] == "/v1/classification"


class TestClassificationValidation:
    """Tests for input validation on classification requests."""

    def test_empty_model_returns_422(self, client):
        c, _, _ = client
        resp = c.post("/v1/classification", json={"model": "", "inputs": "x"})
        assert resp.status_code == 422
        assert "model" in resp.json()["error"]

    def test_whitespace_model_returns_422(self, client):
        c, _, _ = client
        resp = c.post("/v1/classification", json={"model": "   ", "inputs": "x"})
        assert resp.status_code == 422

    def test_missing_model_returns_422(self, client):
        c, _, _ = client
        resp = c.post("/v1/classification", json={"inputs": "x"})
        assert resp.status_code == 422

    def test_empty_inputs_string_returns_422(self, client):
        c, _, _ = client
        resp = c.post("/v1/classification", json={"model": "m", "inputs": ""})
        assert resp.status_code == 422
        assert "inputs" in resp.json()["error"]

    def test_whitespace_inputs_string_returns_422(self, client):
        c, _, _ = client
        resp = c.post("/v1/classification", json={"model": "m", "inputs": "  \t "})
        assert resp.status_code == 422

    def test_empty_inputs_list_returns_422(self, client):
        c, _, _ = client
        resp = c.post("/v1/classification", json={"model": "m", "inputs": []})
        assert resp.status_code == 422
        assert "inputs" in resp.json()["error"]

    def test_missing_inputs_returns_422(self, client):
        c, _, _ = client
        resp = c.post("/v1/classification", json={"model": "m"})
        assert resp.status_code == 422

    def test_non_string_inputs_returns_422(self, client):
        c, _, _ = client
        resp = c.post("/v1/classification", json={"model": "m", "inputs": 7})
        assert resp.status_code == 422

    def test_non_dict_parameters_returns_422(self, client):
        c, _, _ = client
        resp = c.post(
            "/v1/classification",
            json={"model": "m", "inputs": "x", "parameters": ["a"]},
        )
        assert resp.status_code == 422

    def test_invalid_json_returns_422(self, client):
        c, _, _ = client
        resp = c.post(
            "/v1/classification",
            content=b"not json",
            headers={"Content-Type": "application/json"},
        )
        assert resp.status_code == 422

    def test_validation_failure_is_not_captured_as_a_request(self, client, tmp_path):
        """422 validation exits before a CaptureRecord is built."""
        c, mock, _ = client
        c.post("/v1/classification", json={"model": "m"})
        mock.proxy.assert_not_called()
        assert _get_summary_records(tmp_path) == []


class TestClassificationCaptureRecord:
    """Tests for CaptureRecord fields populated by classification requests."""

    def test_request_type_is_classification(self, client, tmp_path):
        c, mock, _ = client
        _ok_upstream(mock)
        c.post("/v1/classification", json={"model": "m", "inputs": "x"})
        records = _get_summary_records(tmp_path)
        assert records[-1]["request_type"] == "classification"

    def test_record_has_request_body(self, client, tmp_path):
        c, mock, _ = client
        _ok_upstream(mock)
        c.post(
            "/v1/classification",
            json={"model": "m", "inputs": ["a", "b"], "parameters": {"truncate": True}},
        )
        record = _get_summary_records(tmp_path)[-1]
        assert record["request"]["model"] == "m"
        assert record["request"]["inputs"] == ["a", "b"]
        assert record["request"]["parameters"] == {"truncate": True}

    def test_record_has_upstream_response(self, client, tmp_path):
        c, mock, _ = client
        _ok_upstream(mock)
        c.post("/v1/classification", json={"model": "m", "inputs": "x"})
        record = _get_summary_records(tmp_path)[-1]
        assert record["response"]["labels"] == ["positive", "negative"]
        assert record["upstream_status"] == 200

    def test_record_has_identity_fields(self, client, tmp_path):
        c, mock, _ = client
        _ok_upstream(mock)
        c.post(
            "/v1/classification",
            json={"model": "m", "inputs": "x"},
            headers={"x-session-id": "s1", "x-user-id": "alice"},
        )
        record = _get_summary_records(tmp_path)[-1]
        assert record["session_id"] == "s1"
        assert record["user_id"] == "alice"

    def test_request_sha_hashes_forwarded_body(self, client, tmp_path):
        c, mock, _ = client
        _ok_upstream(mock)
        body = {"model": "m", "inputs": "x"}
        c.post("/v1/classification", json=body)
        record = _get_summary_records(tmp_path)[-1]
        assert record["request_body_sha256"] == _compute_sha256(body)
        assert record["messages_sha256"] == ""

    def test_upstream_error_routed_to_errors(self, client, tmp_path):
        c, mock, _ = client
        _ok_upstream(mock, {"error": "internal"}, status=500)
        c.post("/v1/classification", json={"model": "m", "inputs": "x"})
        record = _get_error_records(tmp_path)[-1]
        assert record["upstream_status"] == 500
        assert record["request_type"] == "classification"


class TestClassificationTaskLabels:
    """Task resolution: operator map first, structural fallback second.

    The map still wins on an exact key hit, and a map miss no longer
    leaves the capture unlabelled — that null-on-miss behavior is the
    defect these tests previously encoded (task was null in 100% of the
    corpus because the map is off by default). Each test below keeps its
    original intent about the MAP, and additionally pins the fallback.
    """

    def test_task_falls_back_to_structure_when_map_is_off(self, client, tmp_path):
        c, mock, _ = client
        _ok_upstream(mock)
        c.post("/v1/classification", json={"model": "m", "inputs": "x"})
        record = _get_summary_records(tmp_path)[-1]
        assert record["task"] == "classification.single"
        assert record["task_rule"] == "request_type.classification"

    def test_task_is_populated_on_exact_hit(self, make_app, tmp_path):
        c, mock, _ = make_app(env={"CAPTURE_MODEL_TASKS": '{"m":"classification"}'})
        _ok_upstream(mock)
        c.post("/v1/classification", json={"model": "m", "inputs": "x"})
        record = _get_summary_records(tmp_path)[-1]
        assert record["task"] == "classification"
        assert record["task_rule"] == "operator_model_task_map"

    def test_map_miss_falls_back_rather_than_nulling(self, make_app, tmp_path):
        c, mock, _ = make_app(env={"CAPTURE_MODEL_TASKS": '{"other":"classification"}'})
        _ok_upstream(mock)
        c.post("/v1/classification", json={"model": "m", "inputs": "x"})
        record = _get_summary_records(tmp_path)[-1]
        assert record["task"] != "classification"
        assert record["task_rule"] != "operator_model_task_map"

    def test_task_lookup_is_case_sensitive(self, make_app, tmp_path):
        """A case-mismatched key must NOT match the operator map."""
        c, mock, _ = make_app(env={"CAPTURE_MODEL_TASKS": '{"M":"case-test"}'})
        _ok_upstream(mock)
        c.post("/v1/classification", json={"model": "m", "inputs": "x"})
        record = _get_summary_records(tmp_path)[-1]
        assert record["task"] != "case-test"
        assert record["task_rule"] != "operator_model_task_map"

    def test_malformed_map_fails_open(self, make_app, tmp_path):
        c, mock, _ = make_app(env={"CAPTURE_MODEL_TASKS": "{not json"})
        _ok_upstream(mock)
        resp = c.post("/v1/classification", json={"model": "m", "inputs": "x"})
        assert resp.status_code == 200
        record = _get_summary_records(tmp_path)[-1]
        assert record["task_rule"] != "operator_model_task_map"

    def test_non_object_map_fails_open(self, make_app, tmp_path):
        c, mock, _ = make_app(
            env={"CAPTURE_MODEL_TASKS": json.dumps(["m", "classification"])}
        )
        _ok_upstream(mock)
        resp = c.post("/v1/classification", json={"model": "m", "inputs": "x"})
        assert resp.status_code == 200
        record = _get_summary_records(tmp_path)[-1]
        assert record["task_rule"] != "operator_model_task_map"

    def test_non_string_value_falls_back(self, make_app, tmp_path):
        c, mock, _ = make_app(env={"CAPTURE_MODEL_TASKS": json.dumps({"m": 3})})
        _ok_upstream(mock)
        c.post("/v1/classification", json={"model": "m", "inputs": "x"})
        record = _get_summary_records(tmp_path)[-1]
        assert record["task_rule"] != "operator_model_task_map"

    def test_whitespace_value_falls_back(self, make_app, tmp_path):
        c, mock, _ = make_app(env={"CAPTURE_MODEL_TASKS": '{"m":"   "}'})
        _ok_upstream(mock)
        c.post("/v1/classification", json={"model": "m", "inputs": "x"})
        record = _get_summary_records(tmp_path)[-1]
        assert record["task_rule"] != "operator_model_task_map"

    def test_off_sentinel_falls_back(self, make_app, tmp_path):
        c, mock, _ = make_app(env={"CAPTURE_MODEL_TASKS": "off"})
        _ok_upstream(mock)
        c.post("/v1/classification", json={"model": "m", "inputs": "x"})
        record = _get_summary_records(tmp_path)[-1]
        assert record["task_rule"] != "operator_model_task_map"

    def test_task_read_at_request_time(self, make_app, tmp_path):
        """An env edit after construction takes effect without a restart."""
        c, mock, proxy = make_app()
        _ok_upstream(mock)
        with patch.dict(
            "os.environ", {"CAPTURE_MODEL_TASKS": '{"m":"classification"}'}
        ):
            c.post("/v1/classification", json={"model": "m", "inputs": "x"})
            assert proxy._resolve_task("m") == "classification"
        assert _get_summary_records(tmp_path)[-1]["task"] == "classification"
        # The map is gone again once the patch unwinds — no cached state.
        assert proxy._resolve_task("m") is None

    def test_malformed_map_warns_once_per_process(self, capsys):
        """The one-time stderr warning fires once, not once per request."""
        import capture_proxy.app as app_mod

        with patch.object(app_mod, "_task_map_warned", False):
            app_mod._task_map_warned_once("{not json")
            app_mod._task_map_warned_once("{still not json")
        err = capsys.readouterr().err
        assert err.count("CAPTURE_MODEL_TASKS") == 1


class TestClassificationAlias:
    """Tests for alias rewrite/sanitize on the classification route."""

    def test_rewrite_before_hash(self, make_app, tmp_path):
        c, mock, _ = make_app(env=ALIAS_ENV, upstream_url="http://127.0.0.1:8080")
        _ok_upstream(mock, ALIAS_UPSTREAM_BODY)
        c.post("/v1/classification", json={"model": "pub-clf", "inputs": "x"})
        forwarded = mock.proxy.call_args[1]["body"]
        assert forwarded["model"] == "upstream-clf"
        record = _get_summary_records(tmp_path)[-1]
        assert record["model"] == "pub-clf"
        assert record["request_model_original"] == "pub-clf"
        assert record["request_model_forwarded"] == "upstream-clf"
        # INV-4: the hash covers the forwarded body, not the client body.
        assert record["request_body_sha256"] == _compute_sha256(forwarded)
        assert record["request"] == forwarded

    def test_sanitize_before_log_and_serialize(self, make_app, tmp_path):
        c, mock, _ = make_app(env=ALIAS_ENV, upstream_url="http://127.0.0.1:8080")
        _ok_upstream(mock, ALIAS_UPSTREAM_BODY)
        resp = c.post("/v1/classification", json={"model": "pub-clf", "inputs": "x"})
        assert resp.json()["model"] == "pub-clf"
        assert _get_summary_records(tmp_path)[-1]["response"]["model"] == "pub-clf"

    def test_task_keyed_on_public_id(self, make_app, tmp_path):
        """The map is keyed on what the client sends, not the upstream id."""
        c, mock, _ = make_app(env=ALIAS_ENV, upstream_url="http://127.0.0.1:8080")
        _ok_upstream(mock, ALIAS_UPSTREAM_BODY)
        c.post("/v1/classification", json={"model": "pub-clf", "inputs": "x"})
        assert _get_summary_records(tmp_path)[-1]["task"] == "classification"


class TestClassificationErrors:
    """Tests for error handling — upstream failures and OSError resilience."""

    def test_upstream_down_returns_502(self, client):
        c, mock, _ = client
        mock.proxy.side_effect = Exception("connection refused")
        resp = c.post("/v1/classification", json={"model": "m", "inputs": "x"})
        assert resp.status_code == 502
        assert resp.json()["error"] == "upstream unavailable"

    def test_upstream_404_passthrough(self, client):
        c, mock, _ = client
        _ok_upstream(mock, {"error": "model not found"}, status=404)
        resp = c.post("/v1/classification", json={"model": "m", "inputs": "x"})
        assert resp.status_code == 404
        assert resp.json()["error"] == "model not found"

    def test_upstream_400_passthrough(self, client):
        c, mock, _ = client
        _ok_upstream(mock, {"error": "bad request"}, status=400)
        resp = c.post("/v1/classification", json={"model": "m", "inputs": "x"})
        assert resp.status_code == 400

    def test_502_record_is_captured(self, client, tmp_path):
        c, mock, _ = client
        mock.proxy.side_effect = Exception("connection refused")
        c.post("/v1/classification", json={"model": "m", "inputs": "x"})
        record = _get_error_records(tmp_path)[-1]
        assert record["upstream_status"] == 502
        assert record["request_type"] == "classification"
        assert record["error_body"] == "connection refused"

    def test_os_error_during_logging_does_not_break_response(self, client):
        c, mock, proxy = client
        _ok_upstream(mock)
        with patch.object(
            proxy._capture_logger, "log", side_effect=OSError("disk full")
        ):
            resp = c.post("/v1/classification", json={"model": "m", "inputs": "x"})
        assert resp.status_code == 200
        assert resp.json()["labels"] == ["positive", "negative"]

    def test_os_error_during_error_logging_does_not_break_502(self, client):
        c, mock, proxy = client
        mock.proxy.side_effect = Exception("connection refused")
        with patch.object(
            proxy._capture_logger, "log_error", side_effect=OSError("disk full")
        ):
            resp = c.post("/v1/classification", json={"model": "m", "inputs": "x"})
        assert resp.status_code == 502
        assert resp.json()["error"] == "upstream unavailable"

    def test_os_error_during_upstream_error_log_does_not_break_404(self, client):
        c, mock, proxy = client
        _ok_upstream(mock, {"error": "not found"}, status=404)
        with patch.object(
            proxy._capture_logger, "log", side_effect=OSError("disk full")
        ):
            resp = c.post("/v1/classification", json={"model": "m", "inputs": "x"})
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
            resp = c.post("/v1/classification", json={"model": "m"})
        assert resp.status_code == 422
        assert "classification" in resp.json()["error"]
        mock.proxy.assert_not_called()
