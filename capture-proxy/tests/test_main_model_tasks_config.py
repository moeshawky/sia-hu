"""Tests for model-tasks config surface (CLI/env/TOML/default).

Covers the wiring added for CAPTURE_MODEL_TASKS: the --model-tasks flag,
the TOML ``model_tasks`` key, the CLI > env > config > default precedence
chain, the JSON serialization injected into the environment, and the
reachability of the value in CaptureProxyApp._resolve_task (the
request-time consumer of the injected env var).
"""

from __future__ import annotations

import json
import os
from unittest.mock import patch

import pytest

CLF = "nomic-embed-text"
MAP_JSON = json.dumps({CLF: "embedding"})


def _home_env(tmp_path) -> dict[str, str]:
    return {
        "HOME": str(tmp_path),
        "CAPTURE_PORT": "",
        "CAPTURE_UPSTREAM_URL": "",
        # NOT "": an empty value is treated as unset by config resolution,
        # which falls back to the real default /root/captures and lets the
        # test write into a live capture corpus. Pin it under tmp_path.
        "CAPTURE_BASE_PATH": str(tmp_path / "captures"),
        "CAPTURE_BACKEND": "",
        "CAPTURE_MODEL_TASKS": "",
    }


def _write_config(tmp_path, content: str) -> None:
    config_dir = tmp_path / ".config" / "capture-proxy"
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "config.toml").write_text(content)


def _run_main(args: list[str]) -> None:
    from capture_proxy.__main__ import main

    with patch("capture_proxy.__main__.uvicorn") as mock_uvicorn:
        main(args=args)
        mock_uvicorn.run.assert_called_once()


class TestHelpDocumentsShape:
    def test_help_mentions_flag_and_example(self, capsys) -> None:
        from capture_proxy.__main__ import parse_args

        with pytest.raises(SystemExit) as exc:
            parse_args(["--help"])
        assert exc.value.code == 0
        out = capsys.readouterr().out
        assert "--model-tasks" in out
        # Argparse wraps the long example across lines, so assert fragments
        # plus the unbreakable model id and env/TOML surface names.
        compact = out.replace("\n", "").replace(" ", "")
        assert CLF in compact
        assert "embedding" in compact
        assert "CAPTURE_MODEL_TASKS" in out
        assert "model_tasks" in out

    def test_cli_parses_json_string(self) -> None:
        from capture_proxy.__main__ import parse_args

        parsed = parse_args(["--model-tasks", MAP_JSON])
        assert parsed.model_tasks == MAP_JSON


class TestPrecedence:
    def test_default_is_off(self, tmp_path) -> None:
        with patch.dict(os.environ, _home_env(tmp_path), clear=False):
            os.environ.pop("CAPTURE_MODEL_TASKS", None)
            _run_main([])
            assert os.environ.get("CAPTURE_MODEL_TASKS") in (None, "", "off")

    def test_toml_table_accepted(self, tmp_path) -> None:
        _write_config(tmp_path, f'[model_tasks]\n"{CLF}" = "embedding"\n')
        env = _home_env(tmp_path)
        with patch.dict(os.environ, env, clear=False):
            os.environ.pop("CAPTURE_MODEL_TASKS", None)
            _run_main([])
            injected = os.environ.get("CAPTURE_MODEL_TASKS", "")
            assert json.loads(injected) == {CLF: "embedding"}

    def test_env_beats_toml(self, tmp_path) -> None:
        _write_config(tmp_path, '[model_tasks]\n"toml-model" = "chat"\n')
        env = _home_env(tmp_path)
        env["CAPTURE_MODEL_TASKS"] = MAP_JSON
        with patch.dict(os.environ, env, clear=False):
            _run_main([])
            assert json.loads(os.environ["CAPTURE_MODEL_TASKS"]) == {CLF: "embedding"}

    def test_cli_beats_env_and_toml(self, tmp_path) -> None:
        _write_config(tmp_path, '[model_tasks]\n"toml-model" = "chat"\n')
        env = _home_env(tmp_path)
        env["CAPTURE_MODEL_TASKS"] = json.dumps({"env-model": "chat"})
        cli_json = json.dumps({"cli-model": "rerank"})
        with patch.dict(os.environ, env, clear=False):
            _run_main(["--model-tasks", cli_json])
            assert json.loads(os.environ["CAPTURE_MODEL_TASKS"]) == {
                "cli-model": "rerank"
            }

    def test_unknown_toml_key_types_fall_back_to_off(self, tmp_path) -> None:
        """A non-table, non-string model_tasks value resolves to off."""
        _write_config(tmp_path, "model_tasks = 7\n")
        env = _home_env(tmp_path)
        with patch.dict(os.environ, env, clear=False):
            os.environ.pop("CAPTURE_MODEL_TASKS", None)
            _run_main([])
            assert os.environ.get("CAPTURE_MODEL_TASKS") in (None, "", "off")

    def test_config_ignores_unknown_keys(self, tmp_path) -> None:
        """model_tasks is a known key; an unknown sibling is filtered out."""
        from capture_proxy.__main__ import _load_config

        _write_config(tmp_path, '[model_tasks]\n"{CLF}" = "embedding"\nnot_a_key = 1\n')
        config = _load_config(tmp_path / ".config" / "capture-proxy" / "config.toml")
        assert "model_tasks" in config
        assert "not_a_key" not in config


class TestReachability:
    """The injected value must reach _resolve_task from the default entry point."""

    def test_resolve_task_reads_injected_env(self, tmp_path) -> None:
        from capture_proxy.app import CaptureProxyApp

        env = {"CAPTURE_BASE_PATH": str(tmp_path / "captures")}
        with patch.dict(os.environ, env, clear=False):
            os.environ["CAPTURE_MODEL_TASKS"] = MAP_JSON
            proxy = CaptureProxyApp()
            assert proxy._resolve_task(CLF) == "embedding"

    def test_resolve_task_off_by_default(self, tmp_path) -> None:
        from capture_proxy.app import CaptureProxyApp

        env = {"CAPTURE_BASE_PATH": str(tmp_path / "captures")}
        with patch.dict(os.environ, env, clear=False):
            os.environ.pop("CAPTURE_MODEL_TASKS", None)
            proxy = CaptureProxyApp()
            assert proxy._resolve_task(CLF) is None
            assert proxy._resolve_task("") is None

    def test_resolver_preserves_toml_table_type(self) -> None:
        from capture_proxy.__main__ import _resolve_model_tasks_raw

        table = {CLF: "embedding"}
        assert _resolve_model_tasks_raw(None, "CAPTURE_MODEL_TASKS", {}, "off") == "off"
        assert (
            _resolve_model_tasks_raw(
                None, "CAPTURE_MODEL_TASKS", {"model_tasks": table}, "off"
            )
            == table
        )
        assert (
            _resolve_model_tasks_raw("cli", "CAPTURE_MODEL_TASKS", {}, "off") == "cli"
        )

    def test_resolver_ignores_unsupported_config_type(self) -> None:
        from capture_proxy.__main__ import _resolve_model_tasks_raw

        resolved = _resolve_model_tasks_raw(
            None, "CAPTURE_MODEL_TASKS", {"model_tasks": 3}, "off"
        )
        assert resolved == "off"
