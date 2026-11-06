"""Tests for --ensure-config (create-if-absent) and --print-effective-config.

The deployment property under test: an instance that boots with no config file
still ends up on the deployment's upstream/port/base_path instead of the
built-in :11434 ollama default, and a boot script can read back what was
actually loaded.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from capture_proxy.__main__ import _DEFAULT_CONFIG_TOML, _ensure_config, main

REPO_CONFIG = Path(__file__).resolve().parent.parent / "config" / "capture-proxy.toml"


def _config_path(tmp_path: Path) -> Path:
    return tmp_path / ".config" / "capture-proxy" / "config.toml"


class TestEnsureConfigUnit:
    def test_creates_file_and_parents_when_absent(self, tmp_path: Path) -> None:
        target = _config_path(tmp_path)
        assert not target.parent.exists()

        assert _ensure_config(target) is True

        assert target.is_file()
        assert target.read_text(encoding="utf-8") == _DEFAULT_CONFIG_TOML

    def test_existing_file_is_left_byte_identical(self, tmp_path: Path) -> None:
        target = _config_path(tmp_path)
        target.parent.mkdir(parents=True)
        original = 'upstream_url = "http://127.0.0.1:9999"\n# operator edit\n'
        target.write_text(original, encoding="utf-8")

        assert _ensure_config(target) is False

        assert target.read_text(encoding="utf-8") == original

    def test_tracked_template_matches_the_code_constant(self) -> None:
        """One source of truth: the repo copy and the written bytes agree.

        Guards the doc file drifting from what --ensure-config actually writes.
        """
        assert REPO_CONFIG.is_file(), f"missing tracked template: {REPO_CONFIG}"
        assert REPO_CONFIG.read_text(encoding="utf-8") == _DEFAULT_CONFIG_TOML


class TestEnsureConfigCli:
    def test_creates_config_and_exits_without_serving(self, tmp_path, capsys) -> None:
        target = _config_path(tmp_path)

        rc = main(args=["--ensure-config", "--config", str(target)])

        assert rc == 0
        assert target.is_file()
        out = capsys.readouterr().out
        assert "created" in out
        # It must NOT have fallen through into the foreground server path.
        assert "Started server" not in out

    def test_second_run_reports_existing_and_keeps_bytes(
        self, tmp_path, capsys
    ) -> None:
        target = _config_path(tmp_path)
        main(args=["--ensure-config", "--config", str(target)])
        first = target.read_text(encoding="utf-8")

        capsys.readouterr()
        rc = main(args=["--ensure-config", "--config", str(target)])

        assert rc == 0
        assert target.read_text(encoding="utf-8") == first
        assert "kept existing" in capsys.readouterr().out

    def test_operator_edit_survives_repeated_ensure(self, tmp_path) -> None:
        """The whole point of never-clobber: local edits outlive every boot."""
        target = _config_path(tmp_path)
        main(args=["--ensure-config", "--config", str(target)])
        target.write_text('upstream_url = "http://127.0.0.1:8082"\n', encoding="utf-8")

        main(args=["--ensure-config", "--config", str(target)])

        assert target.read_text(encoding="utf-8") == (
            'upstream_url = "http://127.0.0.1:8082"\n'
        )


class TestPrintEffectiveConfig:
    def test_prints_resolved_settings_from_created_file(
        self, tmp_path, capsys, monkeypatch
    ) -> None:
        # This asserts the DEFAULT base_path, so it must not inherit one
        # from the environment — conftest isolates CAPTURE_BASE_PATH for
        # every test so a run can never write into a live capture corpus.
        monkeypatch.delenv("CAPTURE_BASE_PATH", raising=False)
        target = _config_path(tmp_path)
        main(args=["--ensure-config", "--config", str(target)])

        rc = main(
            args=["--print-effective-config", "--config", str(target)],
        )

        assert rc == 0
        out = capsys.readouterr().out
        resolved = dict(
            line.split("=", 1) for line in out.strip().splitlines() if "=" in line
        )
        assert resolved["upstream_url"] == "http://127.0.0.1:8080"
        assert resolved["port"] == "8000"
        assert resolved["host"] == "127.0.0.1"
        assert resolved["base_path"] == "/root/captures"
        assert resolved["backend"] == "llamacpp"
        assert resolved["config_present"] == "True"

    def test_env_var_still_wins_over_file(self, tmp_path, capsys, monkeypatch) -> None:
        """Precedence is unchanged by the new flags: CLI > env > file > default."""
        target = _config_path(tmp_path)
        main(args=["--ensure-config", "--config", str(target)])
        monkeypatch.setenv("CAPTURE_UPSTREAM_URL", "http://127.0.0.1:8001")

        main(args=["--print-effective-config", "--config", str(target)])

        out = capsys.readouterr().out
        assert "upstream_url=http://127.0.0.1:8001" in out

    def test_missing_config_reports_absent_and_built_in_default(
        self, tmp_path, capsys
    ) -> None:
        """Read-back exposes a missing config instead of implying one loaded."""
        target = _config_path(tmp_path)

        main(args=["--print-effective-config", "--config", str(target)])

        out = capsys.readouterr().out
        assert "config_present=False" in out
        # Built-in default, stated plainly: :11434 is ollama, not this box.
        assert "upstream_url=http://127.0.0.1:11434" in out


@pytest.mark.parametrize("flag", ["--ensure-config", "--print-effective-config"])
def test_help_documents_new_flags(flag: str, capsys) -> None:
    from capture_proxy.__main__ import parse_args

    with pytest.raises(SystemExit) as exc:
        parse_args(["--help"])
    assert exc.value.code == 0
    assert flag in capsys.readouterr().out
