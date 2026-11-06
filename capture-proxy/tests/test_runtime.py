"""Adversarial tests for RuntimeSnapshot — fault injection and boundary testing."""

import logging
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from capture_proxy.runtime import RuntimeSnapshot


class TestIntOrNone:
    """Adversarial: _int_or_none boundary and fault injection tests."""

    def test_valid_numeric_string_returns_int(self) -> None:
        assert RuntimeSnapshot._int_or_none("42") == 42

    def test_zero_returns_zero(self) -> None:
        assert RuntimeSnapshot._int_or_none("0") == 0

    def test_negative_returns_negative(self) -> None:
        assert RuntimeSnapshot._int_or_none("-1") == -1

    def test_large_number_returns_int(self) -> None:
        assert RuntimeSnapshot._int_or_none("999999999") == 999999999

    def test_non_numeric_returns_none(self) -> None:
        assert RuntimeSnapshot._int_or_none("abc") is None

    def test_empty_string_returns_none(self) -> None:
        assert RuntimeSnapshot._int_or_none("") is None

    def test_none_returns_none(self) -> None:
        assert RuntimeSnapshot._int_or_none(None) is None

    def test_float_string_returns_none(self) -> None:
        assert RuntimeSnapshot._int_or_none("3.14") is None

    def test_whitespace_returns_none(self) -> None:
        assert RuntimeSnapshot._int_or_none("  ") is None


class TestShlexSplitGuard:
    """Adversarial: shlex.split ValueError guard in _running_llama_args."""

    @patch.object(RuntimeSnapshot, "_command_output")
    def test_malformed_command_returns_raw_line(self, mock_output: object) -> None:
        """Unmatched quote in process command triggers shlex.split ValueError fallback."""
        mock_output.return_value = '12345 llama-server -m "/path/to/model" --port "8080'
        result = RuntimeSnapshot._running_llama_args()

        assert "process_command" in result
        assert result["process_command"] == (
            'llama-server -m "/path/to/model" --port "8080'
        )
        assert "model_path" not in result

    @patch.object(RuntimeSnapshot, "_command_output")
    def test_valid_command_parses_options(self, mock_output: object) -> None:
        """Valid command line is parsed into structured options."""
        mock_output.return_value = (
            "12345 llama-server -m /models/test.gguf --port 8080 --ctx-size 4096"
        )
        result = RuntimeSnapshot._running_llama_args()

        assert result["model_path"] == "/models/test.gguf"
        assert result["port"] == "8080"
        assert result["ctx_size"] == "4096"
        assert result["process_command"] == (
            "llama-server -m /models/test.gguf --port 8080 --ctx-size 4096"
        )

    @patch.object(RuntimeSnapshot, "_command_output")
    def test_empty_pgrep_output_returns_empty(self, mock_output: object) -> None:
        """No llama-server process running returns empty dict."""
        mock_output.return_value = ""
        result = RuntimeSnapshot._running_llama_args()
        assert result == {}

    @patch.object(RuntimeSnapshot, "_command_output")
    def test_command_without_options_returns_raw(self, mock_output: object) -> None:
        """Process line with command but no recognized options returns raw command."""
        mock_output.return_value = "12345 llama-server"
        result = RuntimeSnapshot._running_llama_args()
        assert result == {"process_command": "llama-server"}


class TestRunningVllmArgs:
    """Adversarial: _running_vllm_args parsing and fault injection."""

    @patch.object(RuntimeSnapshot, "_command_output")
    def test_valid_command_parses_options(self, mock_output: object) -> None:
        """Valid vllm serve command line is parsed into structured options."""
        mock_output.return_value = (
            "999 vllm serve /models/test --tensor-parallel-size 2 "
            "--max-model-len 8192 --kv-cache-dtype fp8 --quantization awq"
        )
        result = RuntimeSnapshot._running_vllm_args()

        assert result["model_path"] == "/models/test"
        assert result["tensor_parallel_size"] == "2"
        assert result["max_model_len"] == "8192"
        assert result["kv_cache_dtype"] == "fp8"
        assert result["quantization"] == "awq"
        assert result["process_command"] == (
            "vllm serve /models/test --tensor-parallel-size 2 "
            "--max-model-len 8192 --kv-cache-dtype fp8 --quantization awq"
        )

    @patch.object(RuntimeSnapshot, "_command_output")
    def test_python_module_invocation_finds_positional_model(
        self, mock_output: object
    ) -> None:
        """python -m vllm.entrypoints... invocation still yields the model path."""
        mock_output.return_value = (
            "999 python -m vllm.entrypoints.openai.api_server /models/test "
            "--max-model-len 4096"
        )
        result = RuntimeSnapshot._running_vllm_args()

        assert result["model_path"] == "/models/test"
        assert result["max_model_len"] == "4096"

    @patch.object(RuntimeSnapshot, "_command_output")
    def test_absolute_path_interpreter_finds_positional_model(
        self, mock_output: object
    ) -> None:
        """Full-path interpreter whose venv path lacks 'vllm' is still skipped."""
        mock_output.return_value = (
            "999 /opt/venv/bin/python -m vllm.entrypoints.openai.api_server "
            "/models/test --max-model-len 4096"
        )
        result = RuntimeSnapshot._running_vllm_args()

        assert result["model_path"] == "/models/test"
        assert result["max_model_len"] == "4096"

    @patch.object(RuntimeSnapshot, "_command_output")
    def test_non_server_line_before_server_line_selects_server(
        self, mock_output: object
    ) -> None:
        """pgrep matches non-server PIDs first; selection prefers the
        vllm server-shaped line."""
        mock_output.return_value = (
            "100 /opt/venv-tpu/bin/python -c "
            "from multiprocessing.resource_tracker import main\n"
            "200 /opt/venvs/vllm/bin/vllm serve /models/test --max-model-len 4096"
        )
        result = RuntimeSnapshot._running_vllm_args()

        assert result["process_command"] == (
            "/opt/venvs/vllm/bin/vllm serve /models/test --max-model-len 4096"
        )
        assert result["model_path"] == "/models/test"

    @patch.object(RuntimeSnapshot, "_command_output")
    def test_flag_values_not_treated_as_model(self, mock_output: object) -> None:
        """Flag values (e.g. --quantization awq) are not mistaken for the model."""
        mock_output.return_value = (
            "999 vllm serve --quantization awq /models/test --max-model-length 2048"
        )
        result = RuntimeSnapshot._running_vllm_args()

        assert result["model_path"] == "/models/test"
        assert result["max_model_len"] == "2048"
        assert result["quantization"] == "awq"

    @patch.object(RuntimeSnapshot, "_command_output")
    def test_model_after_flag_value(self, mock_output: object) -> None:
        """Model placed after flag pairs is still the first positional candidate."""
        mock_output.return_value = (
            "999 vllm serve --max-model-len 4096 /models/test --dtype bfloat16"
        )
        result = RuntimeSnapshot._running_vllm_args()

        assert result["model_path"] == "/models/test"
        assert result["dtype"] == "bfloat16"

    @patch.object(RuntimeSnapshot, "_command_output")
    def test_empty_pgrep_output_returns_empty(self, mock_output: object) -> None:
        """No vLLM process running returns empty dict."""
        mock_output.return_value = ""
        result = RuntimeSnapshot._running_vllm_args()
        assert result == {}

    @patch.object(RuntimeSnapshot, "_command_output")
    def test_malformed_command_returns_raw_line(self, mock_output: object) -> None:
        """Unmatched quote in process command triggers shlex.split ValueError fallback."""
        mock_output.return_value = '999 vllm serve "/model x'
        result = RuntimeSnapshot._running_vllm_args()

        assert result == {"process_command": 'vllm serve "/model x'}
        assert "model_path" not in result


class TestVllmSettings:
    """Settings construction and run_id identity for backend='vllm'."""

    @patch.object(RuntimeSnapshot, "_command_output")
    def test_vllm_settings_populate_from_process(
        self, mock_output: object, tmp_path
    ) -> None:
        def fake_output(command: list[str], timeout: float = 5.0) -> str:
            if command[0] == "pgrep":
                return (
                    "999 vllm serve /models/test --tensor-parallel-size 2 "
                    "--max-model-len 4096 --kv-cache-dtype fp8"
                )
            if command[0] == "vllm":
                return "0.8.5"
            return ""

        mock_output.side_effect = fake_output
        snapshot = RuntimeSnapshot(str(tmp_path / "captures"), "vllm")

        assert snapshot.settings["backend"] == "vllm"
        assert snapshot.settings["model_path"] == "/models/test"
        assert snapshot.settings["tensor_parallel_size"] == 2
        assert snapshot.settings["max_model_len"] == 4096
        assert snapshot.settings["kv_cache_dtype"] == "fp8"
        assert snapshot.settings["vllm_version"] == "0.8.5"
        assert snapshot.settings["process_command"] == (
            "vllm serve /models/test --tensor-parallel-size 2 "
            "--max-model-len 4096 --kv-cache-dtype fp8"
        )

    @patch.object(RuntimeSnapshot, "_command_output")
    def test_served_model_from_flag(self, mock_output: object, tmp_path) -> None:
        """First --served-model-name alias wins; absent flag falls back."""
        mock_output.side_effect = lambda command, timeout=5.0: (
            "999 vllm serve /models/test "
            "--served-model-name qwen3.8-27b-fp8 Extra-Alias --port 8000"
            if command[0] == "pgrep"
            else ""
        )
        snapshot = RuntimeSnapshot(str(tmp_path / "captures"), "vllm")
        assert snapshot.settings["served_model"] == "qwen3.8-27b-fp8"

        mock_output.side_effect = lambda command, timeout=5.0: (
            "999 vllm serve /models/test" if command[0] == "pgrep" else ""
        )
        snapshot = RuntimeSnapshot(str(tmp_path / "captures2"), "vllm")
        assert snapshot.settings["served_model"] == "private-model"

    @patch.object(RuntimeSnapshot, "_command_output")
    @patch("capture_proxy.runtime.os.path.isfile", return_value=True)
    def test_vllm_version_venv_fallback(
        self, _mock_isfile: object, mock_output: object, tmp_path
    ) -> None:
        """Bare `vllm` not on PATH — venv-local entrypoint via interpreter dir."""
        mock_output.side_effect = lambda command, timeout=5.0: (
            "999 /venvs/vllm/bin/python /venvs/vllm/bin/vllm serve /models/test"
            if command[0] == "pgrep"
            else (
                "INFO 08-16 16:50:02 [__init__.py:76] TPU info: node_name=worker\n"
                "WARNING 08-16 [tpu_platform.py:402] Pin memory is not supported.\n"
                "0.26.0"
            )
            if command == ["/venvs/vllm/bin/vllm", "--version"]
            else ""
        )
        snapshot = RuntimeSnapshot(str(tmp_path / "captures"), "vllm")
        assert snapshot.settings["vllm_version"] == "0.26.0"

    @patch.object(RuntimeSnapshot, "_command_output")
    def test_vllm_run_id_uses_vllm_identity(
        self, mock_output: object, tmp_path
    ) -> None:
        """vLLM snapshot hashes its own identity shape (distinct from llama.cpp)."""
        mock_output.return_value = ""
        vllm_snapshot = RuntimeSnapshot(str(tmp_path / "captures"), "vllm")
        llama_snapshot = RuntimeSnapshot(str(tmp_path / "captures2"), "llamacpp")

        assert vllm_snapshot.run_id != llama_snapshot.run_id
        assert vllm_snapshot.run_id.startswith("run_")


class TestWriteManifestSuppression:
    """Fault injection: _write_manifest failure must not escape (get: Raises None)."""

    @patch.object(RuntimeSnapshot, "_command_output", return_value="")
    def test_manifest_oserror_does_not_raise(
        self, _mock_output: object, tmp_path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A capture-tree fault during manifest write degrades with a named
        warning (file + errno), not an exception from get()/construction."""
        with (
            patch.object(
                Path, "mkdir", side_effect=OSError(30, "Read-only file system")
            ),
            caplog.at_level(logging.WARNING, logger="capture_proxy.runtime"),
        ):
            snapshot = RuntimeSnapshot(str(tmp_path / "captures"), "llamacpp")

        assert snapshot.run_id.startswith("run_")
        assert any("manifest" in r.message for r in caplog.records)


class TestCommandOutputFaultTolerance:
    """Fault injection: _command_output must degrade to "" for execution
    faults — the probe is advisory (version strings for run manifests)."""

    def test_permission_error_returns_empty(self) -> None:
        """An unexecutable binary (EACCES) degrades to "" — same contract
        as FileNotFoundError/SubprocessError."""
        with patch.object(
            subprocess, "check_output", side_effect=PermissionError("denied")
        ):
            out = RuntimeSnapshot._command_output(["vllm", "--version"])
        assert out == ""

    def test_file_not_found_still_returns_empty(self) -> None:
        with patch.object(
            subprocess,
            "check_output",
            side_effect=FileNotFoundError("no such file"),
        ):
            out = RuntimeSnapshot._command_output(["vllm", "--version"])
        assert out == ""
