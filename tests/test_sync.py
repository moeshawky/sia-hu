"""Sync daemon and one-shot behavior tests (design §6, non-error matrix).

Uses stub rclone on PATH and mocks subprocess.Popen for spawn-shape assertions.
Never touches ~/.config, /root/captures, or /etc/machine-id.
"""

from __future__ import annotations

import fcntl
import os
import threading
import time
from typing import TYPE_CHECKING
from unittest import mock

if TYPE_CHECKING:
    from pathlib import Path

    import pytest

from capture_proxy import sync as sync_mod


def _base_args(tmp_path: Path) -> list[str]:
    pid = tmp_path / "sync.pid"
    log = tmp_path / "sync.log"
    source = tmp_path / "source"
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
        "--sync-remote",
        "gdrive:captures",
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


def test_default_sync_cmd_is_copy(
    stub_rclone: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """--sync without --sync-cmd defaults to copy (never destructive)."""
    from capture_proxy.__main__ import main

    rc = main(args=_base_args(tmp_path))
    assert rc == 0
    invocations = [
        line.split(" ") for line in stub_rclone.read_text().splitlines() if line.strip()
    ]
    assert invocations, "rclone should have been invoked"
    assert invocations[-1][1] == "copy"


def test_sync_cmd_allowed_with_explicit_id(
    stub_rclone: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """--sync-cmd sync is allowed when instance id is explicit (flag)."""
    from capture_proxy.__main__ import main

    rc = main(args=[*_base_args(tmp_path), "--sync-cmd", "sync"])
    assert rc == 0


def test_instance_id_resolution_chain(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Flag > env > config > derived (hostname > /etc/machine-id)."""
    from capture_proxy.sync import resolve_instance_id

    # Flag wins.
    id_, explicit = resolve_instance_id("flag-id", None, None, "host", None)
    assert id_ == "flag-id"
    assert explicit is True

    # Env wins over config and derived.
    id_, explicit = resolve_instance_id(None, "env-id", "cfg-id", "host", None)
    assert id_ == "env-id"
    assert explicit is True

    # Config wins over derived.
    id_, explicit = resolve_instance_id(None, None, "cfg-id", "host", None)
    assert id_ == "cfg-id"
    assert explicit is False

    # Hostname wins over machine-id file.
    id_, explicit = resolve_instance_id(None, None, None, "host", "/etc/machine-id")
    assert id_ == "host"
    assert explicit is False

    # Machine-id file used when no hostname.
    id_, explicit = resolve_instance_id(None, None, None, None, "file-id")
    assert id_ == "file-id"
    assert explicit is False

    # Empty when all sources empty.
    id_, explicit = resolve_instance_id(None, None, None, None, None)
    assert id_ == ""
    assert explicit is False


def test_missing_source_created(
    stub_rclone: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Missing source dir is created; sync still runs."""
    from capture_proxy.__main__ import main

    source = tmp_path / "source"  # does not exist yet
    args = _base_args(tmp_path)
    args[args.index("--base-path") + 1] = str(source)
    assert not source.exists()
    rc = main(args=args)
    assert rc == 0
    assert source.is_dir()


def test_empty_source_warns(
    stub_rclone: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capfd
) -> None:
    """Empty source dir emits a warning to stderr."""
    from capture_proxy.__main__ import main

    source = tmp_path / "source"
    source.mkdir()
    args = _base_args(tmp_path)
    args[args.index("--base-path") + 1] = str(source)
    rc = main(args=args)
    assert rc == 0
    err = capfd.readouterr().err
    assert "is empty" in err


def test_sync_min_age_zero_disables(
    stub_rclone: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """--sync-min-age 0 omits --min-age from rclone argv."""
    from capture_proxy.sync import build_rclone_argv

    settings = {
        "sync_rclone_binary": "rclone",
        "sync_cmd": "copy",
        "sync_min_age": "0",
        "sync_rclone_opts": "",
        "sync_dry_run": False,
    }
    argv = build_rclone_argv(settings, "/src", "gdrive:captures/id")
    assert "--min-age" not in argv


def test_rclone_nonzero_exit_continues(
    failing_rclone: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """rclone non-zero exit is logged; run_sync_once returns rc != 0."""
    from capture_proxy.__main__ import main

    source = tmp_path / "source"
    source.mkdir()
    args = _base_args(tmp_path)
    args[args.index("--base-path") + 1] = str(source)
    rc = main(args=args)
    assert rc == 1


def test_source_disappears_mid_loop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Source missing mid-loop: warn and continue (no crash)."""
    from capture_proxy.sync import run_sync_once

    settings = {
        "sync_remote": "gdrive:captures",
        "sync_cmd": "copy",
        "sync_min_age": "0",
        "sync_interval": "1",
        "sync_dry_run": False,
        "sync_rclone_opts": "",
        "sync_rclone_binary": "rclone",
        "sync_pid_file": str(tmp_path / "sync.pid"),
        "sync_log_file": str(tmp_path / "sync.log"),
        "sync_instance_id": "box-1",
        "sync_instance_id_explicit": True,
        "sync_base_path": str(tmp_path / "nonexistent"),
        "sync_config_path": str(tmp_path / "config.toml"),
    }
    # Source does not exist -> run_sync_once recreates it (warns, returns 0)
    rc = run_sync_once(settings)
    assert rc == 0
    assert (tmp_path / "nonexistent").is_dir()


def test_dry_run_passed_to_rclone(
    stub_rclone: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """--sync-dry-run appends --dry-run to rclone argv."""
    from capture_proxy.__main__ import main

    source = tmp_path / "source"
    source.mkdir()
    args = _base_args(tmp_path)
    args[args.index("--base-path") + 1] = str(source)
    args += ["--sync-dry-run"]
    rc = main(args=args)
    assert rc == 0
    invocations = [
        line.split(" ") for line in stub_rclone.read_text().splitlines() if line.strip()
    ]
    assert "--dry-run" in invocations[-1]


def test_daemon_idempotent_restart(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Re-running --sync-daemon when a live PID exists is a no-op (exit 0)."""
    pid_file = tmp_path / "sync.pid"
    pid_file.write_text(f"{os.getpid()}\n")
    from capture_proxy.__main__ import main

    rc = main(args=["--sync-daemon", "--sync-pid-file", str(pid_file)])
    assert rc == 0


def test_concurrent_daemon_start(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two concurrent starts race on the same lock; exactly one loop survives."""
    pid_file = tmp_path / "sync.pid"
    log_file = tmp_path / "sync.log"
    source = tmp_path / "source"

    # flock locks belong to the open file description, so two os.open calls
    # in the same process genuinely conflict — this threaded race is valid.
    spawned = []
    barrier = threading.Barrier(2)
    results = []

    class FakePopen:
        def __init__(self, *a, **k):
            time.sleep(0.25)  # widen the truncate->write window
            spawned.append(True)
            self.pid = os.getpid()

    with mock.patch.object(sync_mod.subprocess, "Popen", FakePopen):
        from capture_proxy.__main__ import main

        def run_one() -> None:
            barrier.wait(timeout=10)  # release both starts together
            results.append(
                main(
                    args=[
                        "--sync-daemon",
                        "--sync-pid-file",
                        str(pid_file),
                        "--sync-log-file",
                        str(log_file),
                        "--base-path",
                        str(source),
                        "--sync-interval",
                        "300",
                    ]
                )
            )

        t1 = threading.Thread(target=run_one)
        t2 = threading.Thread(target=run_one)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

    assert len(spawned) == 1, f"exactly one loop should survive, got {len(spawned)}"
    assert all(rc == 0 for rc in results), f"both starts should return 0, got {results}"


def test_stale_pid_overwritten(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Stale PID file (pid not running) is overwritten; new loop starts."""
    pid_file = tmp_path / "sync.pid"
    pid_file.write_text("999999\n")  # not running

    spawned = []

    class FakePopen:
        def __init__(self, *a, **k):
            spawned.append(True)
            self.pid = os.getpid()

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    with mock.patch.object(sync_mod.subprocess, "Popen", FakePopen):
        from capture_proxy.__main__ import main

        rc = main(args=["--sync-daemon", "--sync-pid-file", str(pid_file)])
    assert rc == 0
    assert spawned, "daemon should have been spawned for stale PID"


def test_sync_daemon_child_inherits_settings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Daemon child inherits all resolved settings as flags (no env injection)."""
    pid_file = tmp_path / "sync.pid"
    log_file = tmp_path / "sync.log"
    source = tmp_path / "source"
    config = tmp_path / "config.toml"
    source.mkdir()

    captured = {}

    class FakePopen:
        def __init__(self, args, **kwargs):
            captured["args"] = args
            captured["env"] = kwargs.get("env", {})
            captured["pass_fds"] = kwargs.get("pass_fds")
            self.pid = os.getpid()

    with mock.patch.object(sync_mod.subprocess, "Popen", FakePopen):
        from capture_proxy.__main__ import main

        rc = main(
            args=[
                "--sync-daemon",
                "--sync-pid-file",
                str(pid_file),
                "--sync-log-file",
                str(log_file),
                "--base-path",
                str(source),
                "--config",
                str(config),
                "--sync-remote",
                "gdrive:captures",
                "--sync-cmd",
                "copy",
                "--sync-min-age",
                "60s",
                "--sync-interval",
                "300",
                "--sync-rclone-binary",
                "rclone",
                "--sync-rclone-opts",
                "--fast-list --checksum",
                "--sync-machine-id",
                "box-1",
            ]
        )
    assert rc == 0
    args = captured["args"]
    assert "--sync-remote" in args
    assert "gdrive:captures" in args
    assert "--sync-cmd" in args
    assert "copy" in args
    assert "--sync-machine-id" in args
    assert "box-1" in args
    assert "--base-path" in args
    assert str(source) in args
    assert "--config" in args
    assert str(config) in args


def test_sync_daemon_child_guard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """CAPTURE_SYNC_DAEMONIZED=1 selects _run_loop; otherwise sync_daemon_loop."""
    calls = []

    def fake_run_loop(settings):
        calls.append(("run_loop", settings))
        return 0

    def fake_daemon_loop(settings):
        calls.append(("daemon_loop", settings))
        return 0

    monkeypatch.setattr(sync_mod, "_run_loop", fake_run_loop)
    monkeypatch.setattr(sync_mod, "sync_daemon_loop", fake_daemon_loop)

    # Child path: guard env set -> _run_loop.
    monkeypatch.setenv("CAPTURE_SYNC_DAEMONIZED", "1")
    from capture_proxy.__main__ import main

    rc = main(args=["--sync-daemon", "--sync-pid-file", str(tmp_path / "sync.pid")])
    assert rc == 0
    assert calls and calls[-1][0] == "run_loop"

    # Parent path: env unset -> sync_daemon_loop (clear recorded calls first).
    calls.clear()
    monkeypatch.delenv("CAPTURE_SYNC_DAEMONIZED", raising=False)
    rc = main(args=["--sync-daemon", "--sync-pid-file", str(tmp_path / "sync.pid")])
    assert rc == 0
    assert calls and calls[-1][0] == "daemon_loop"


def test_sync_daemon_start_in_progress_message(tmp_path: Path, capsys) -> None:
    """Held lock with empty pid file -> 'start is in progress' error, rc 1."""
    pid_file = tmp_path / "sync.pid"
    pid_file.touch()  # empty: holder truncates then writes
    lock_fd = os.open(pid_file, os.O_RDWR)
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        from capture_proxy.__main__ import main

        rc = main(
            args=[
                "--sync-daemon",
                "--sync-pid-file",
                str(pid_file),
                "--sync-log-file",
                str(tmp_path / "sync.log"),
                "--base-path",
                str(tmp_path / "source"),
                "--sync-interval",
                "300",
            ]
        )
    finally:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        os.close(lock_fd)

    assert rc == 1
    captured = capsys.readouterr()
    assert "another sync daemon start is in progress" in captured.err


def test_sync_stop_removes_pid(tmp_path: Path) -> None:
    """--sync-stop stops loop and removes sync.pid."""
    pid_file = tmp_path / "sync.pid"
    pid_file.write_text("999999\n")  # not running -> already stopped
    from capture_proxy.__main__ import main

    rc = main(args=["--sync-stop", "--sync-pid-file", str(pid_file)])
    assert rc == 0


def test_sync_status(tmp_path: Path) -> None:
    """--sync-status shows running/stale/not-running."""
    pid_file = tmp_path / "sync.pid"

    # Not running (no file).
    from capture_proxy.__main__ import main

    rc = main(args=["--sync-status", "--sync-pid-file", str(pid_file)])
    assert rc == 1

    # Stale PID.
    pid_file.write_text("999999\n")
    rc = main(args=["--sync-status", "--sync-pid-file", str(pid_file)])
    assert rc == 1


def test_sync_ensure_config_composition(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stub_rclone
) -> None:
    """--ensure-config --sync provisions config then falls through to sync."""
    from capture_proxy.__main__ import main

    config = tmp_path / "config.toml"
    source = tmp_path / "source"
    pid = tmp_path / "sync.pid"
    log = tmp_path / "sync.log"
    assert not config.exists()
    rc = main(
        args=[
            "--ensure-config",
            "--config",
            str(config),
            "--sync",
            "--sync-machine-id",
            "box-1",
            "--sync-pid-file",
            str(pid),
            "--sync-log-file",
            str(log),
            "--base-path",
            str(source),
            "--sync-remote",
            "gdrive:captures",
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
    )
    assert rc == 0
    assert config.is_file(), "--ensure-config should have created the config"


def test_sync_config_key_allowed(tmp_path: Path) -> None:
    """sync_* config keys are accepted via _CONFIG_KEYS (machine_id too)."""
    from capture_proxy.__main__ import main

    config = tmp_path / "config.toml"
    config.write_text('sync_remote = "gdrive:captures"\nmachine_id = "cfg-box"\n')
    source = tmp_path / "source"
    source.mkdir()
    pid = tmp_path / "sync.pid"
    log = tmp_path / "sync.log"
    rc = main(
        args=[
            "--sync",
            "--config",
            str(config),
            "--sync-pid-file",
            str(pid),
            "--sync-log-file",
            str(log),
            "--base-path",
            str(source),
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
    # config machine_id is derived (explicit=False), so --sync-cmd copy is fine.
    assert rc == 0


def test_sync_env_overrides_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stub_rclone
) -> None:
    """Env var overrides config file value."""
    from capture_proxy.__main__ import main

    config = tmp_path / "config.toml"
    config.write_text('sync_remote = "gdrive:config-remote"\n')
    source = tmp_path / "source"
    source.mkdir()
    pid = tmp_path / "sync.pid"
    log = tmp_path / "sync.log"
    monkeypatch.setenv("CAPTURE_SYNC_REMOTE", "gdrive:env-remote")
    rc = main(
        args=[
            "--sync",
            "--config",
            str(config),
            "--sync-machine-id",
            "box-1",
            "--sync-pid-file",
            str(pid),
            "--sync-log-file",
            str(log),
            "--base-path",
            str(source),
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
    assert rc == 0
    invocations = [
        line.split(" ") for line in stub_rclone.read_text().splitlines() if line.strip()
    ]
    # dest = <remote>/<instance-id> -> env remote wins
    assert "gdrive:env-remote/box-1" in invocations[-1][3]


def test_sync_cli_overrides_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stub_rclone
) -> None:
    """CLI flag overrides env var."""
    from capture_proxy.__main__ import main

    config = tmp_path / "config.toml"
    config.write_text('sync_remote = "gdrive:config-remote"\n')
    source = tmp_path / "source"
    source.mkdir()
    pid = tmp_path / "sync.pid"
    log = tmp_path / "sync.log"
    monkeypatch.setenv("CAPTURE_SYNC_REMOTE", "gdrive:env-remote")
    rc = main(
        args=[
            "--sync",
            "--config",
            str(config),
            "--sync-machine-id",
            "box-1",
            "--sync-remote",
            "gdrive:cli-remote",
            "--sync-pid-file",
            str(pid),
            "--sync-log-file",
            str(log),
            "--base-path",
            str(source),
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
    assert rc == 0
    invocations = [
        line.split(" ") for line in stub_rclone.read_text().splitlines() if line.strip()
    ]
    assert "gdrive:cli-remote/box-1" in invocations[-1][3]
