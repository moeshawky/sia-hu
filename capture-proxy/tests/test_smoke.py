"""
Smoke Tests

Pre-startup verification that the capture proxy is functional.
"""

from __future__ import annotations

import glob
import json
import signal
import subprocess
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from capture_proxy.app import CaptureProxyApp
from capture_proxy.upstream import StreamingUpstreamResponse


def _make_streaming_response(chunks: list[bytes]) -> StreamingUpstreamResponse:
    async def _aiter():
        for chunk in chunks:
            yield chunk

    return StreamingUpstreamResponse(
        status_code=200,
        headers={"content-type": "text/event-stream"},
        content_type="text/event-stream",
        stream=_aiter(),
    )


@pytest.fixture
def smoke_app(tmp_path):
    """Create app with mocked upstream for smoke testing."""
    with patch.dict("os.environ", {"CAPTURE_BASE_PATH": str(tmp_path / "captures")}):
        proxy = CaptureProxyApp()

        mock_upstream = MagicMock()
        mock_upstream.health_check = AsyncMock(return_value=True)
        mock_upstream.proxy = AsyncMock()
        proxy._upstream_client = mock_upstream

        yield proxy.app, mock_upstream, tmp_path


@pytest.fixture
def smoke_client(smoke_app):
    """Provide TestClient for smoke tests."""
    app, mock, tmp = smoke_app
    return TestClient(app), mock, tmp


class TestSmoke:
    """5 smoke tests required by R-21."""


class TestPortParsingBoundary:
    """Adversarial: __main__.py port parsing with invalid input."""

    def test_invalid_port_env_var_raises_value_error(self) -> None:
        """Non-numeric CAPTURE_PORT raises ValueError at startup."""
        import os
        from unittest.mock import patch

        with (
            patch.dict(os.environ, {"CAPTURE_PORT": "not-a-number"}),
            patch("capture_proxy.__main__.uvicorn") as mock_uvicorn,
        ):
            from capture_proxy.__main__ import main

            with pytest.raises(ValueError):
                main(args=[])
            mock_uvicorn.run.assert_not_called()

    def test_zero_port_is_valid(self) -> None:
        """CAPTURE_PORT=0 is parsed to int 0 without error."""
        import os
        from unittest.mock import patch

        with (
            patch.dict(os.environ, {"CAPTURE_PORT": "0"}),
            patch("capture_proxy.__main__.uvicorn") as mock_uvicorn,
        ):
            from capture_proxy.__main__ import main

            main(args=[])
            mock_uvicorn.run.assert_called_once()
            call_kwargs = mock_uvicorn.run.call_args
            assert call_kwargs[1]["port"] == 0

    def test_negative_port_is_valid_int(self) -> None:
        """CAPTURE_PORT=-1 is parsed to int -1 (int() allows negatives)."""
        import os
        from unittest.mock import patch

        with (
            patch.dict(os.environ, {"CAPTURE_PORT": "-1"}),
            patch("capture_proxy.__main__.uvicorn") as mock_uvicorn,
        ):
            from capture_proxy.__main__ import main

            main(args=[])
            mock_uvicorn.run.assert_called_once()
            call_kwargs = mock_uvicorn.run.call_args
            assert call_kwargs[1]["port"] == -1

    def test_large_port_is_valid_int(self) -> None:
        """CAPTURE_PORT=99999 is parsed to int 99999."""
        import os
        from unittest.mock import patch

        with (
            patch.dict(os.environ, {"CAPTURE_PORT": "99999"}),
            patch("capture_proxy.__main__.uvicorn") as mock_uvicorn,
        ):
            from capture_proxy.__main__ import main

            main(args=[])
            mock_uvicorn.run.assert_called_once()
            call_kwargs = mock_uvicorn.run.call_args
            assert call_kwargs[1]["port"] == 99999


class TestHelpFlag:
    """Verify --help and --version work correctly."""

    def test_help_flag_exits_cleanly(self) -> None:
        """--help prints usage and exits with code 0."""
        from capture_proxy.__main__ import parse_args

        with pytest.raises(SystemExit) as exc_info:
            parse_args(["--help"])
        assert exc_info.value.code == 0

    def test_help_flag_output_contains_usage(self, capsys) -> None:
        """--help output contains 'usage:' header."""
        from capture_proxy.__main__ import parse_args

        with pytest.raises(SystemExit):
            parse_args(["--help"])
        captured = capsys.readouterr()
        assert "usage:" in captured.out.lower()

    def test_help_flag_mentions_all_options(self, capsys) -> None:
        """--help output mentions --port, --host, --upstream-url, --base-path."""
        from capture_proxy.__main__ import parse_args

        with pytest.raises(SystemExit):
            parse_args(["--help"])
        captured = capsys.readouterr()
        assert "--port" in captured.out
        assert "--host" in captured.out
        assert "--upstream-url" in captured.out
        assert "--base-path" in captured.out

    def test_cli_port_overrides_env(self) -> None:
        """--port 9999 overrides CAPTURE_PORT env var."""
        import os
        from unittest.mock import patch

        with (
            patch.dict(os.environ, {"CAPTURE_PORT": "1234"}),
            patch("capture_proxy.__main__.uvicorn") as mock_uvicorn,
        ):
            from capture_proxy.__main__ import main

            main(args=["--port", "9999"])
            mock_uvicorn.run.assert_called_once()
            assert mock_uvicorn.run.call_args[1]["port"] == 9999

    def test_cli_host_overrides_env(self) -> None:
        """--host 127.0.0.1 overrides CAPTURE_HOST env var."""
        import os
        from unittest.mock import patch

        with (
            patch.dict(os.environ, {"CAPTURE_HOST": "0.0.0.0"}),
            patch("capture_proxy.__main__.uvicorn") as mock_uvicorn,
        ):
            from capture_proxy.__main__ import main

            main(args=["--host", "127.0.0.1"])
            mock_uvicorn.run.assert_called_once()
            assert mock_uvicorn.run.call_args[1]["host"] == "127.0.0.1"

    def test_smoke_models_passthrough(self, smoke_client):
        """Smoke 1: /v1/models returns upstream response verbatim."""
        c, mock, _ = smoke_client
        upstream_body = {"data": [{"id": "private-model", "object": "model"}]}
        mock.proxy.return_value = MagicMock(
            status_code=200, body=upstream_body, headers={}
        )
        resp = c.get("/v1/models")
        assert resp.status_code == 200
        assert resp.json() == upstream_body

    def test_smoke_non_streaming_logs_record(self, smoke_client):
        """Smoke 2: Non-streaming request logs exactly 1 JSONL record."""
        c, mock, tmp = smoke_client
        mock.proxy.return_value = MagicMock(
            status_code=200,
            body={
                "id": "chatcmpl-1",
                "choices": [{"message": {"content": "ok"}}],
            },
            headers={},
            elapsed_ms=5.0,
        )
        resp = c.post(
            "/v1/chat/completions",
            json={
                "model": "test",
                "messages": [{"role": "user", "content": "hello"}],
            },
        )
        assert resp.status_code == 200

        jsonl_files = glob.glob(
            str(tmp / "captures" / "raw" / "**" / "*.jsonl"),
            recursive=True,
        )
        summary_files = [f for f in jsonl_files if ".stream.jsonl" not in f]
        assert len(summary_files) >= 1
        with open(summary_files[0]) as f:
            lines = f.read().strip().split("\n")
        assert len(lines) == 1

    def test_smoke_streaming_logs_reconstructed_response(self, smoke_client):
        """Smoke 3: Streaming request logs reconstructed response from chunks."""
        c, mock, tmp = smoke_client

        sse_chunks = [
            b'data: {"id":"chatcmpl-2","model":"test","choices":[{"index":0,"delta":{"role":"assistant","content":"Hello"},"finish_reason":null}]}\n\n',
            b'data: {"id":"chatcmpl-2","model":"test","choices":[{"index":0,"delta":{"content":" world"},"finish_reason":"stop"}],"usage":{"prompt_tokens":5,"completion_tokens":2,"total_tokens":7}}\n\n',
            b"data: [DONE]\n\n",
        ]
        mock.proxy.return_value = _make_streaming_response(sse_chunks)

        resp = c.post(
            "/v1/chat/completions",
            json={
                "model": "test",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": True,
            },
        )
        assert resp.status_code == 200

        jsonl_files = glob.glob(
            str(tmp / "captures" / "raw" / "**" / "*.jsonl"),
            recursive=True,
        )
        summary_files = [f for f in jsonl_files if ".stream.jsonl" not in f]
        assert len(summary_files) >= 1
        with open(summary_files[0]) as f:
            record = json.loads(f.readline())
        assert record["response"] is not None
        assert record["response"]["id"] == "chatcmpl-2"
        assert record["response"]["choices"][0]["message"]["content"] == "Hello world"
        assert record["response"]["usage"]["total_tokens"] == 7

    def test_smoke_upstream_failure_logs_to_errors(self, smoke_client):
        """Smoke 5: Upstream failure logs with upstream_status=500."""
        c, mock, tmp = smoke_client
        mock.proxy.return_value = MagicMock(
            status_code=500,
            body={"error": "internal server error"},
            headers={},
            elapsed_ms=10.0,
        )
        resp = c.post(
            "/v1/chat/completions",
            json={
                "model": "test",
                "messages": [{"role": "user", "content": "fail"}],
            },
        )
        assert resp.status_code == 500

        raw_files = glob.glob(
            str(tmp / "captures" / "errors" / "**" / "*.jsonl"),
            recursive=True,
        )
        summary_files = [f for f in raw_files if ".stream.jsonl" not in f]
        assert len(summary_files) >= 1
        with open(summary_files[0]) as f:
            record = json.loads(f.readline())
        assert record["upstream_status"] == 500


class TestProcessManagement:
    """CLI process management: config file, daemon, stop, status."""

    def _write_config(self, tmp_path, content: str) -> None:
        """Write config.toml under the hermetic HOME."""
        config_dir = tmp_path / ".config" / "capture-proxy"
        config_dir.mkdir(parents=True, exist_ok=True)
        (config_dir / "config.toml").write_text(content)

    def _home_env(self, tmp_path) -> dict[str, str]:
        """Env overrides: pin HOME to tmp_path, blank out CAPTURE_* vars."""
        return {
            "HOME": str(tmp_path),
            "CAPTURE_PORT": "",
            "CAPTURE_UPSTREAM_URL": "",
            # NOT "": an empty value is treated as unset by config resolution,
            # which falls back to the real default /root/captures and lets the
            # test write into a live capture corpus. Pin it under tmp_path.
            "CAPTURE_BASE_PATH": str(tmp_path / "captures"),
            "CAPTURE_BACKEND": "",
        }

    def test_config_file_sets_port(self, tmp_path) -> None:
        """Config file port is used when no CLI/env override exists."""
        import os
        from unittest.mock import patch

        from capture_proxy.__main__ import main

        self._write_config(tmp_path, "port = 9999\n")
        with (
            patch.dict(os.environ, self._home_env(tmp_path)),
            patch("capture_proxy.__main__.uvicorn") as mock_uvicorn,
        ):
            main(args=[])
        mock_uvicorn.run.assert_called_once()
        assert mock_uvicorn.run.call_args[1]["port"] == 9999

    def test_cli_overrides_config_file(self, tmp_path) -> None:
        """--port flag beats both CAPTURE_PORT env and config file."""
        import os
        from unittest.mock import patch

        from capture_proxy.__main__ import main

        self._write_config(tmp_path, "port = 1111\n")
        with (
            patch.dict(
                os.environ, {**self._home_env(tmp_path), "CAPTURE_PORT": "2222"}
            ),
            patch("capture_proxy.__main__.uvicorn") as mock_uvicorn,
        ):
            main(args=["--port", "3333"])
        mock_uvicorn.run.assert_called_once()
        assert mock_uvicorn.run.call_args[1]["port"] == 3333

    def test_env_overrides_config_file(self, tmp_path) -> None:
        """CAPTURE_PORT env beats the config file port."""
        import os
        from unittest.mock import patch

        from capture_proxy.__main__ import main

        self._write_config(tmp_path, "port = 1111\n")
        with (
            patch.dict(
                os.environ, {**self._home_env(tmp_path), "CAPTURE_PORT": "2222"}
            ),
            patch("capture_proxy.__main__.uvicorn") as mock_uvicorn,
        ):
            main(args=[])
        mock_uvicorn.run.assert_called_once()
        assert mock_uvicorn.run.call_args[1]["port"] == 2222

    def test_daemon_writes_pid_file(self, tmp_path) -> None:
        """--daemon spawns a detached child and writes PID + info files."""
        import os
        from unittest.mock import patch

        from capture_proxy.__main__ import main

        mock_popen = MagicMock()
        mock_popen.pid = 424242
        with (
            patch.dict(os.environ, self._home_env(tmp_path)),
            patch(
                "capture_proxy.__main__.subprocess.Popen", return_value=mock_popen
            ) as popen_mock,
        ):
            rc = main(args=["--daemon", "--port", "8000"])
        assert rc == 0
        popen_mock.assert_called_once()
        call_kwargs = popen_mock.call_args.kwargs
        assert call_kwargs["start_new_session"] is True
        assert call_kwargs["stdin"] == subprocess.DEVNULL
        assert call_kwargs["stderr"] == subprocess.STDOUT
        assert popen_mock.call_args.args[0] == [
            sys.executable,
            "-m",
            "capture_proxy",
            "--port",
            "8000",
        ]
        pid_file = tmp_path / ".cache" / "capture-proxy" / "proxy.pid"
        assert pid_file.read_text().strip() == "424242"
        info_file = tmp_path / ".cache" / "capture-proxy" / "proxy.info.json"
        assert info_file.exists()
        info = json.loads(info_file.read_text())
        assert info["pid"] == "424242"
        assert info["port"] == "8000"

    def test_stop_terminates_and_cleans(self, tmp_path) -> None:
        """--stop sends SIGTERM and removes the PID and info files."""
        import os
        from unittest.mock import patch

        from capture_proxy.__main__ import main

        pid_file = tmp_path / ".cache" / "capture-proxy" / "proxy.pid"
        pid_file.parent.mkdir(parents=True)
        pid_file.write_text("12345\n")
        info_file = pid_file.with_name("proxy.info.json")
        info_file.write_text(json.dumps({"pid": "12345", "port": "7777"}))
        with (
            patch.dict(os.environ, self._home_env(tmp_path)),
            patch("capture_proxy.__main__.os.kill") as mock_kill,
        ):
            rc = main(args=["--stop"])
        assert rc == 0
        mock_kill.assert_called_once_with(12345, signal.SIGTERM)
        assert not pid_file.exists()
        assert not info_file.exists()

    def test_status_not_running(self, tmp_path, capsys) -> None:
        """--status with no PID file reports not running and exits 1."""
        import os
        from unittest.mock import patch

        from capture_proxy.__main__ import main

        with patch.dict(os.environ, self._home_env(tmp_path)):
            rc = main(args=["--status"])
        assert rc == 1
        assert "not running" in capsys.readouterr().out

    def test_status_running(self, tmp_path, capsys) -> None:
        """--status reports the daemon's port from the info sidecar."""
        import os
        from unittest.mock import patch

        from capture_proxy.__main__ import main

        pid_file = tmp_path / ".cache" / "capture-proxy" / "proxy.pid"
        pid_file.parent.mkdir(parents=True)
        pid_file.write_text(f"{os.getpid()}\n")
        info_file = pid_file.with_name("proxy.info.json")
        info_file.write_text(json.dumps({"pid": str(os.getpid()), "port": "7777"}))
        with patch.dict(os.environ, self._home_env(tmp_path)):
            rc = main(args=["--status"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "running" in out
        assert "7777" in out

    def test_status_running_falls_back_without_info(self, tmp_path, capsys) -> None:
        """--status without a sidecar falls back to the resolution chain."""
        import os
        from unittest.mock import patch

        from capture_proxy.__main__ import main

        pid_file = tmp_path / ".cache" / "capture-proxy" / "proxy.pid"
        pid_file.parent.mkdir(parents=True)
        pid_file.write_text(f"{os.getpid()}\n")
        with patch.dict(os.environ, self._home_env(tmp_path)):
            rc = main(args=["--status"])
        assert rc == 0
        assert "port: 8000" in capsys.readouterr().out

    def test_daemon_pid_file_oserror_does_not_crash(self, tmp_path, capsys) -> None:
        """A disk fault writing the PID file warns on stderr and reports
        failure instead of crashing the parent — the daemon still spawns."""
        from unittest.mock import patch

        from capture_proxy.__main__ import _write_pid_file

        target = tmp_path / "proxy.pid"
        with patch("os.makedirs", side_effect=OSError("disk full")):
            result = _write_pid_file(target, 4242)
        assert result is False
        assert "PID file" in capsys.readouterr().err

    def test_daemon_info_file_oserror_does_not_crash(self, tmp_path, capsys) -> None:
        """A disk fault writing the info sidecar warns on stderr and
        reports failure instead of crashing."""
        from unittest.mock import patch

        from capture_proxy.__main__ import _write_info_file

        target = tmp_path / "proxy.info.json"
        with patch("os.makedirs", side_effect=OSError("disk full")):
            result = _write_info_file(target, {"pid": "1"})
        assert result is False
        assert "info file" in capsys.readouterr().err
