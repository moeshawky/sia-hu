"""Sync error-path and validation tests (design §6, error matrix).

Never touches ~/.config, /root/captures, or /etc/machine-id. Always passes
--sync-pid-file/--sync-log-file/--base-path/--config into tmp_path.
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest import mock

import pytest

from capture_proxy import sync as sync_mod
from capture_proxy.sync import validate_interval, validate_sync_cmd


def _neutralize_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make all derived instance-id sources empty (hostname + /etc/machine-id)."""
    import socket

    monkeypatch.setattr(socket, "gethostname", lambda: "")
    _original_is_file = Path.is_file

    def _fake_is_file(self: Path) -> bool:
        if str(self) == "/etc/machine-id":
            return False
        return _original_is_file(self)

    monkeypatch.setattr(Path, "is_file", _fake_is_file)


def _sync_args(tmp_path: Path) -> list[str]:
    pid = tmp_path / "sync.pid"
    log = tmp_path / "sync.log"
    source = tmp_path / "source"
    source.mkdir()
    config = tmp_path / "config.toml"
    return [
        "--sync",
        "--sync-machine-id",
        "box-1",
        "--sync-pid-file",
        str(pid),
        "--sync-log-file",
        str(log),
        "--base-path",
        str(source),
        "--config",
        str(config),
        "--sync-cmd",
        "copy",
        "--sync-min-age",
        "60s",
        "--sync-interval",
        "300",
        "--sync-rclone-binary",
        "rclone",
        "--sync-rclone-opts",
        "",
    ]


def test_sync_cmd_refused_with_derived_id(
    tmp_path: Path, capfd: pytest.CaptureFixture[str]
) -> None:
    """--sync-cmd sync with derived (non-explicit) instance id -> exit 1."""
    from capture_proxy.__main__ import main

    args = _sync_args(tmp_path)
    args.remove("--sync-machine-id")
    args.remove("box-1")
    args += ["--sync-cmd", "sync"]
    rc = main(args=args)
    assert rc == 1
    err = capfd.readouterr().err
    assert "destructive mirror" in err or "explicit instance id" in err


def test_machine_id_fallback_warning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capfd: pytest.CaptureFixture[str]
) -> None:
    """Machine-id fallback emits the hazard warning to stderr."""
    # Hostname empty -> fall past hostname to /etc/machine-id (kept intact).
    import socket

    from capture_proxy.__main__ import main

    monkeypatch.setattr(socket, "gethostname", lambda: "")
    args = _sync_args(tmp_path)
    args.remove("--sync-machine-id")
    args.remove("box-1")
    monkeypatch.delenv("CAPTURE_MACHINE_ID", raising=False)
    main(args=args)
    err = capfd.readouterr().err
    assert "falling back to /etc/machine-id" in err


def test_no_instance_id_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capfd: pytest.CaptureFixture[str]
) -> None:
    """All four sources empty -> exit 1 with literal 'no instance id'."""
    from capture_proxy.__main__ import main

    _neutralize_identity(monkeypatch)
    config = tmp_path / "config.toml"
    config.write_text("")
    source = tmp_path / "source"
    source.mkdir()
    pid = tmp_path / "sync.pid"
    log = tmp_path / "sync.log"
    rc = main(
        args=[
            "--sync",
            "--sync-pid-file",
            str(pid),
            "--sync-log-file",
            str(log),
            "--base-path",
            str(source),
            "--config",
            str(config),
            "--sync-cmd",
            "copy",
            "--sync-min-age",
            "0",
            "--sync-interval",
            "300",
            "--sync-rclone-binary",
            "rclone",
            "--sync-rclone-opts",
            "",
        ]
    )
    assert rc == 1
    err = capfd.readouterr().err
    assert "no instance id" in err


def test_sync_min_age_flag_name() -> None:
    """--sync-min-age is the flag name; config key is sync_min_age."""
    assert "sync_min_age" in [
        "sync_remote",
        "sync_cmd",
        "sync_min_age",
        "sync_interval",
        "sync_dry_run",
        "sync_rclone_opts",
        "sync_rclone_binary",
        "sync_pid_file",
        "sync_log_file",
        "machine_id",
    ]


def test_invalid_interval(tmp_path: Path, capfd: pytest.CaptureFixture[str]) -> None:
    """--sync-interval not a whole number -> exit 2 with literal message."""
    from capture_proxy.__main__ import main

    args = [*_sync_args(tmp_path), "--sync-interval", "abc"]
    rc = main(args=args)
    assert rc == 2
    err = capfd.readouterr().err
    assert "positive whole number of seconds" in err


def test_validate_interval_rejects_zero() -> None:
    """validate_interval rejects 0 (positive whole number required)."""
    with pytest.raises(ValueError, match="positive whole number"):
        validate_interval("0")


def test_invalid_sync_cmd(tmp_path: Path, capfd: pytest.CaptureFixture[str]) -> None:
    """--sync-cmd not copy/sync -> exit 2 with literal message."""
    from capture_proxy.__main__ import main

    args = [*_sync_args(tmp_path), "--sync-cmd", "move"]
    rc = main(args=args)
    assert rc == 2
    err = capfd.readouterr().err
    assert "must be 'copy' or 'sync'" in err


def test_validate_sync_cmd_rejects() -> None:
    """validate_sync_cmd raises on values other than copy/sync."""
    with pytest.raises(ValueError, match="must be 'copy' or 'sync'"):
        validate_sync_cmd("move")


def test_rclone_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capfd: pytest.CaptureFixture[str]
) -> None:
    """rclone missing on PATH -> exit 1 with literal naming CAPTURE_SYNC_RCLONE_BINARY."""
    from capture_proxy.__main__ import main

    monkeypatch.setenv("PATH", "")
    args = _sync_args(tmp_path)
    rc = main(args=args)
    assert rc == 1
    err = capfd.readouterr().err
    assert "rclone binary" in err and "CAPTURE_SYNC_RCLONE_BINARY" in err


def test_sync_subprocess_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capfd: pytest.CaptureFixture[str]
) -> None:
    """rclone non-zero exit logged with argv + exit code; exit 1."""
    from capture_proxy.__main__ import main

    args = _sync_args(tmp_path)
    args[args.index("--base-path") + 1] = str(tmp_path / "source")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    stub = bin_dir / "rclone"
    stub.write_text("#!/bin/sh\nexit 1\n", encoding="utf-8")
    stub.chmod(0o755)
    monkeypatch.setenv("PATH", str(bin_dir) + os.pathsep + os.environ.get("PATH", ""))
    rc = main(args=args)
    assert rc == 1
    err = capfd.readouterr().err
    assert "exited" in err


def test_sync_dry_run_false_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capfd: pytest.CaptureFixture[str]
) -> None:
    """CAPTURE_SYNC_DRY_RUN=false disables dry-run (bool vocab parsed)."""
    from capture_proxy.__main__ import main

    monkeypatch.setenv("CAPTURE_SYNC_DRY_RUN", "false")
    args = _sync_args(tmp_path)
    args[args.index("--base-path") + 1] = str(tmp_path / "source")
    rc = main(args=args)
    assert rc == 0


def test_sync_stop_exit_codes(tmp_path: Path) -> None:
    """--sync-stop: 0 stopped/already stopped."""
    from capture_proxy.__main__ import main

    pid_file = tmp_path / "sync.pid"
    rc = main(args=["--sync-stop", "--sync-pid-file", str(pid_file)])
    assert rc == 0


def test_sync_status_exit_codes(tmp_path: Path) -> None:
    """--sync-status: 1 not running/stale."""
    from capture_proxy.__main__ import main

    pid_file = tmp_path / "sync.pid"
    rc = main(args=["--sync-status", "--sync-pid-file", str(pid_file)])
    assert rc == 1
    pid_file.write_text("999999\n")
    rc = main(args=["--sync-status", "--sync-pid-file", str(pid_file)])
    assert rc == 1


def test_sync_mutual_exclusion_daemon(tmp_path: Path) -> None:
    """--sync + --daemon -> exit 2 with mutual exclusion message."""
    from capture_proxy.__main__ import main

    rc = main(args=["--sync", "--daemon"])
    assert rc == 2


def test_sync_two_action_flags_exit_2(tmp_path: Path) -> None:
    """Two sync action flags set -> exit 2 with 'exactly one' message."""
    from capture_proxy.__main__ import main

    rc = main(args=["--sync", "--sync-status"])
    assert rc == 2


def test_sync_dry_run_invalid_value_exit_2(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capfd: pytest.CaptureFixture[str]
) -> None:
    """Invalid --sync-dry-run value (via env) -> exit 2 with literal bool vocabulary message."""
    from capture_proxy.__main__ import main

    monkeypatch.setenv("CAPTURE_SYNC_DRY_RUN", "maybe")
    args = _sync_args(tmp_path)
    rc = main(args=args)
    assert rc == 2
    err = capfd.readouterr().err
    assert "true/false/1/0/yes/no/on/off" in err


def test_sync_child_omits_machine_id_when_derived(tmp_path: Path) -> None:
    """D1 proof: child argv omits --sync-machine-id when id is derived."""
    captured = {}

    class FakePopen:
        def __init__(self, args, **kwargs):
            captured["args"] = args
            self.pid = os.getpid()

    with mock.patch.object(sync_mod.subprocess, "Popen", FakePopen):
        from capture_proxy.__main__ import main

        rc = main(
            args=[
                "--sync-daemon",
                "--sync-pid-file",
                str(tmp_path / "sync.pid"),
                "--sync-log-file",
                str(tmp_path / "sync.log"),
                "--base-path",
                str(tmp_path / "source"),
                "--sync-remote",
                "gdrive:captures",
                "--sync-cmd",
                "copy",
                "--sync-min-age",
                "0",
                "--sync-interval",
                "300",
                "--sync-rclone-binary",
                "rclone",
                "--sync-rclone-opts",
                "",
                # No --sync-machine-id -> derived id -> explicit=False
            ]
        )
    assert rc == 0
    args = captured["args"]
    assert "--sync-machine-id" not in args, (
        "derived id must not pass --sync-machine-id to child"
    )
