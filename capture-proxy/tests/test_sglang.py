"""Tests for the SGLang backend.

Covers: explicit CAPTURE_BACKEND=sglang resolution (SGLang's default port
30000 is not auto-detected, same as vLLM's 8000), OpenAI-passthrough
forwarding for chat/models, the backend label on capture records, and
RuntimeSnapshot parsing of `sglang serve` command lines.
"""

import contextlib
import glob
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from capture_proxy.app import CaptureProxyApp
from capture_proxy.runtime import RuntimeSnapshot


@pytest.fixture
def make_app(tmp_path):
    """Yield a factory returning (TestClient, mock_upstream, proxy)."""
    stack = contextlib.ExitStack()

    def _factory(env=None, upstream_url="http://127.0.0.1:31000"):
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


def _ok_chat(mock):
    """Program the mock upstream with a minimal SGLang chat body."""
    mock.proxy.return_value = MagicMock(
        status_code=200,
        body={
            "id": "chatcmpl-sglang",
            "choices": [
                {
                    "message": {"role": "assistant", "content": "OK"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 2,
                "total_tokens": 12,
            },
        },
        elapsed_ms=5.0,
    )


class TestSglangBackendDetection:
    """Backend resolution: sglang requires explicit CAPTURE_BACKEND."""

    def test_explicit_sglang_backend(self, make_app):
        _c, _, proxy = make_app(env={"CAPTURE_BACKEND": "sglang"})
        assert proxy._backend == "sglang"

    def test_sglang_default_port_not_auto_detected(self, make_app):
        """SGLang's default port 30000 is not auto-detectable — the fallback
        stays llamacpp, so sglang requires explicit CAPTURE_BACKEND (same as
        vllm on 8000 and node-llama-cpp)."""
        _c, _, proxy = make_app(
            env={"CAPTURE_BACKEND": ""},
            upstream_url="http://127.0.0.1:30000",
        )
        assert proxy._backend == "llamacpp"


class TestSglangPassthrough:
    """SGLang is OpenAI-compatible: every route proxies verbatim."""

    def test_chat_forwards_verbatim(self, make_app):
        c, mock, _ = make_app(env={"CAPTURE_BACKEND": "sglang"})
        _ok_chat(mock)
        body = {
            "model": "CYBER-FROST-3.8-FP8",
            "messages": [{"role": "user", "content": "hi"}],
        }
        resp = c.post("/v1/chat/completions", json=body)
        assert resp.status_code == 200
        assert mock.proxy.call_args[1]["path"] == "/v1/chat/completions"
        assert mock.proxy.call_args[1]["body"]["model"] == "CYBER-FROST-3.8-FP8"

    def test_chat_record_carries_sglang_backend(self, make_app, tmp_path):
        c, mock, _ = make_app(env={"CAPTURE_BACKEND": "sglang"})
        _ok_chat(mock)
        resp = c.post(
            "/v1/chat/completions",
            json={
                "model": "CYBER-FROST-3.8-FP8",
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
        assert resp.status_code == 200
        records = _get_summary_records(tmp_path)
        assert len(records) == 1
        assert records[0]["model"] == "CYBER-FROST-3.8-FP8"
        # CaptureRecord carries no top-level backend key — the backend label
        # lives in runtime_settings (RuntimeSnapshot manifest).
        assert records[0]["runtime_settings"]["backend"] == "sglang"

    def test_models_passthrough(self, make_app):
        c, mock, _ = make_app(env={"CAPTURE_BACKEND": "sglang"})
        mock.proxy.return_value = MagicMock(
            status_code=200,
            body={"object": "list", "data": [{"id": "CYBER-FROST-3.8-FP8"}]},
            elapsed_ms=1.0,
        )
        resp = c.get("/v1/models")
        assert resp.status_code == 200
        assert mock.proxy.call_args[1]["path"] == "/v1/models"


class TestRunningSglangArgs:
    """Adversarial: _running_sglang_args parsing and fault injection."""

    @patch.object(RuntimeSnapshot, "_command_output")
    def test_valid_serve_command_parses_options(self, mock_output: object) -> None:
        """A real-shape sglang serve line parses into structured options."""
        mock_output.return_value = (
            "999 /opt/venv/bin/sglang serve --model-path /root/blackfrost/fp8 "
            "--served-model-name CYBER-FROST-3.8-FP8 --host 0.0.0.0 --port 31000 "
            "--tp-size 1 --max-running-requests 10 --reasoning-parser qwen3 "
            "--tool-call-parser qwen3_coder"
        )
        result = RuntimeSnapshot._running_sglang_args()

        assert result["model_path"] == "/root/blackfrost/fp8"
        assert result["served_model_name"] == "CYBER-FROST-3.8-FP8"
        assert result["tp_size"] == "1"
        assert result["max_running_requests"] == "10"
        assert result["reasoning_parser"] == "qwen3"
        assert result["tool_call_parser"] == "qwen3_coder"
        assert result["host"] == "0.0.0.0"
        assert result["port"] == "31000"

    @patch.object(RuntimeSnapshot, "_command_output")
    def test_module_invocation_detected(self, mock_output: object) -> None:
        """python -m sglang.launch_server invocation is server-shaped."""
        mock_output.return_value = (
            "999 /opt/venv/bin/python -m sglang.launch_server "
            "--model-path /models/test --port 30000"
        )
        result = RuntimeSnapshot._running_sglang_args()

        assert result["model_path"] == "/models/test"
        assert result["port"] == "30000"

    @patch.object(RuntimeSnapshot, "_command_output")
    def test_non_server_line_before_server_line_selects_server(
        self, mock_output: object
    ) -> None:
        """pgrep matches non-server PIDs first; selection prefers the
        sglang server-shaped line."""
        mock_output.return_value = (
            "100 /usr/bin/python3 /root/blackfrost/bin/frpc -c sglang-notes.txt\n"
            "200 /opt/venv/bin/sglang serve --model-path /models/test --port 31000"
        )
        result = RuntimeSnapshot._running_sglang_args()

        assert result["model_path"] == "/models/test"
        assert result["port"] == "31000"

    @patch.object(RuntimeSnapshot, "_command_output")
    def test_no_process_returns_empty(self, mock_output: object) -> None:
        """No sglang process degrades to {} (advisory, never fatal)."""
        mock_output.return_value = ""
        assert RuntimeSnapshot._running_sglang_args() == {}

    def test_server_line_predicate(self) -> None:
        """_is_sglang_server_line accepts console-script and module shapes."""
        assert RuntimeSnapshot._is_sglang_server_line(
            "999 /opt/venv/bin/sglang serve --model-path /m"
        )
        assert RuntimeSnapshot._is_sglang_server_line(
            "999 python -m sglang.launch_server --model-path /m"
        )
        assert not RuntimeSnapshot._is_sglang_server_line(
            "100 python -c from multiprocessing.resource_tracker import main"
        )

    @patch.object(RuntimeSnapshot, "_command_output")
    def test_sglang_settings_shape(self, mock_output: object) -> None:
        """_build_sglang_settings carries SGLang-flavored provenance keys."""
        mock_output.side_effect = [
            "999 /opt/venv/bin/sglang serve --model-path /root/blackfrost/fp8 "
            "--served-model-name CYBER-FROST-3.8-FP8 --tp-size 1 "
            "--max-running-requests 10",
            "",  # nvidia-smi absent (advisory probe degrades to "")
        ]
        snapshot = RuntimeSnapshot.__new__(RuntimeSnapshot)
        snapshot._backend = "sglang"
        settings = snapshot._build_sglang_settings()

        assert settings["backend"] == "sglang"
        assert settings["served_model"] == "CYBER-FROST-3.8-FP8"
        assert settings["model_path"] == "/root/blackfrost/fp8"
        assert settings["max_running_requests"] == 10
        assert settings["tp_size"] == 1
        assert "sglang_version" not in settings

    def test_sglang_run_id_shape(self) -> None:
        """SGLang run_ids hash on the SGLang identity shape (stable + distinct)."""
        settings = {
            "backend": "sglang",
            "model_path": "/m",
            "tp_size": "1",
            "max_running_requests": "10",
            "reasoning_parser": "qwen3",
            "tool_call_parser": "qwen3_coder",
        }
        run_id = RuntimeSnapshot._build_run_id(settings)
        assert run_id.startswith("run_")
        other = RuntimeSnapshot._build_run_id({**settings, "tp_size": "2"})
        assert other != run_id
