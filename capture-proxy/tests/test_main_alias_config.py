"""Tests for model-aliases config surface (CLI/env/TOML/default)."""

from __future__ import annotations

import json
import os
from unittest.mock import patch

import pytest

PUBLIC = "qwen3.8-27b-fp8"
UPSTREAM = "davidau-qwen3.8-27b-twin-turbo-fable"
PAIR_JSON = json.dumps({PUBLIC: UPSTREAM})


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
        "CAPTURE_MODEL_ALIASES": "",
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
        assert "--model-aliases" in out
        # Argparse wraps the long example across lines, so assert fragments
        # plus the unbreakable public id and env/TOML surface names.
        compact = out.replace("\n", "").replace(" ", "")
        assert PUBLIC in compact
        assert "davidau" in compact
        assert "CAPTURE_MODEL_ALIASES" in out
        assert "model_aliases" in out

    def test_cli_parses_json_string(self) -> None:
        from capture_proxy.__main__ import parse_args

        parsed = parse_args(["--model-aliases", PAIR_JSON])
        assert parsed.model_aliases == PAIR_JSON


class TestPrecedence:
    def test_default_is_off(self, tmp_path) -> None:
        with patch.dict(os.environ, _home_env(tmp_path), clear=False):
            os.environ.pop("CAPTURE_MODEL_ALIASES", None)
            _run_main([])
            assert os.environ.get("CAPTURE_MODEL_ALIASES") in (None, "", "off")

    def test_toml_table_accepted(self, tmp_path) -> None:
        _write_config(tmp_path, f'[model_aliases]\n"{PUBLIC}" = "{UPSTREAM}"\n')
        env = _home_env(tmp_path)
        env.pop("CAPTURE_MODEL_ALIASES", None)
        with patch.dict(os.environ, env, clear=False):
            os.environ.pop("CAPTURE_MODEL_ALIASES", None)
            _run_main([])
            injected = os.environ.get("CAPTURE_MODEL_ALIASES", "")
            assert json.loads(injected) == {PUBLIC: UPSTREAM}

    def test_env_beats_toml(self, tmp_path) -> None:
        _write_config(tmp_path, '[model_aliases]\n"toml-pub" = "toml-up"\n')
        env = _home_env(tmp_path)
        env["CAPTURE_MODEL_ALIASES"] = PAIR_JSON
        with patch.dict(os.environ, env, clear=False):
            _run_main([])
            assert json.loads(os.environ["CAPTURE_MODEL_ALIASES"]) == {PUBLIC: UPSTREAM}

    def test_cli_beats_env_and_toml(self, tmp_path) -> None:
        _write_config(tmp_path, '[model_aliases]\n"toml-pub" = "toml-up"\n')
        env = _home_env(tmp_path)
        env["CAPTURE_MODEL_ALIASES"] = json.dumps({"env-pub": "env-up"})
        cli_json = json.dumps({"cli-pub": "cli-up"})
        with patch.dict(os.environ, env, clear=False):
            _run_main(["--model-aliases", cli_json])
            assert json.loads(os.environ["CAPTURE_MODEL_ALIASES"]) == {
                "cli-pub": "cli-up"
            }


class TestInvalidAndFatal:
    def test_invalid_json_warns_and_off(self, tmp_path, capsys) -> None:
        from capture_proxy.app import CaptureProxyApp

        env = {
            "CAPTURE_BASE_PATH": str(tmp_path / "captures"),
            "CAPTURE_MODEL_ALIASES": "{not-json",
        }
        with patch.dict(os.environ, env, clear=False):
            proxy = CaptureProxyApp()
            assert proxy._aliases is None
        assert "warning" in capsys.readouterr().err.lower()

    def test_app_off_by_default(self, tmp_path) -> None:
        from capture_proxy.app import CaptureProxyApp

        env = {"CAPTURE_BASE_PATH": str(tmp_path / "captures")}
        with patch.dict(os.environ, env, clear=False):
            os.environ.pop("CAPTURE_MODEL_ALIASES", None)
            proxy = CaptureProxyApp()
            assert proxy._aliases is None

    def test_app_builds_map_from_env(self, tmp_path) -> None:
        from capture_proxy.app import CaptureProxyApp

        env = {
            "CAPTURE_BASE_PATH": str(tmp_path / "captures"),
            "CAPTURE_MODEL_ALIASES": PAIR_JSON,
        }
        with patch.dict(os.environ, env, clear=False):
            proxy = CaptureProxyApp()
            assert proxy._aliases is not None
            assert proxy._aliases.forward == {PUBLIC: UPSTREAM}

    def test_duplicate_upstream_fatal_exit_2(self, tmp_path, capsys) -> None:
        from capture_proxy.app import CaptureProxyApp

        dup_json = json.dumps({"a": "x", "b": "x"})
        env = {
            "CAPTURE_BASE_PATH": str(tmp_path / "captures"),
            "CAPTURE_MODEL_ALIASES": dup_json,
        }
        with patch.dict(os.environ, env, clear=False):
            with pytest.raises(SystemExit) as exc:
                CaptureProxyApp()
            assert exc.value.code == 2
        err = capsys.readouterr().err
        assert "x" in err
