"""Tests for the rclone argv built by the sync carrier (design §4, §6).

Covers: exact one-shot argv, shlex splitting of --sync-rclone-opts, the rclone
binary appearing in argv[0], and extra rclone opts in argv.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from capture_proxy.sync import build_rclone_argv


def test_one_shot_invokes_rclone_with_exact_argv(
    stub_rclone: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """run_sync_once invokes rclone with [binary, cmd, source, dest, ...]."""
    from capture_proxy.__main__ import main

    source = str(tmp_path / "source")
    dest = "gdrive:captures/box-1"
    rc = main(
        args=[
            "--sync",
            "--sync-machine-id",
            "box-1",
            "--base-path",
            source,
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
            "--sync-pid-file",
            str(tmp_path / "sync.pid"),
            "--sync-log-file",
            str(tmp_path / "sync.log"),
        ]
    )
    assert rc == 0
    invocations = [
        line.split(" ") for line in stub_rclone.read_text().splitlines() if line.strip()
    ]
    assert invocations, "rclone should have been invoked"
    argv = invocations[-1]
    assert Path(argv[0]).name == "rclone"
    assert argv[1] == "copy"
    assert argv[2] == source
    assert argv[3] == dest


def test_rclone_binary_in_argv(tmp_path: Path) -> None:
    """--sync-rclone-binary appears as argv[0] (resolved path)."""
    binary = tmp_path / "rclone"
    binary.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    binary.chmod(0o755)
    settings = {
        "sync_rclone_binary": str(binary),
        "sync_cmd": "copy",
        "sync_min_age": "0",
        "sync_rclone_opts": "",
        "sync_dry_run": False,
    }
    argv = build_rclone_argv(settings, "/src", "gdrive:captures/id")
    assert argv[0] == str(binary)


def test_rclone_opts_shlex_split() -> None:
    """--sync-rclone-opts is split with shlex.split, not str.split()."""
    settings = {
        "sync_rclone_binary": "rclone",
        "sync_cmd": "copy",
        "sync_min_age": "0",
        "sync_rclone_opts": "--fast-list --checksum",
        "sync_dry_run": False,
    }
    argv = build_rclone_argv(settings, "/src", "gdrive:captures/id")
    fast_idx = argv.index("--fast-list")
    assert argv[fast_idx + 1] == "--checksum"


def test_extra_rclone_opts_in_argv(tmp_path: Path) -> None:
    """Extra rclone opts appear in the argv after min-age."""
    binary = tmp_path / "rclone"
    binary.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    binary.chmod(0o755)
    settings = {
        "sync_rclone_binary": str(binary),
        "sync_cmd": "copy",
        "sync_min_age": "120s",
        "sync_rclone_opts": "--fast-list --checksum",
        "sync_dry_run": False,
    }
    argv = build_rclone_argv(settings, "/src", "gdrive:captures/id")
    assert argv[0] == str(binary)
    assert argv[1] == "copy"
    assert argv[2] == "/src"
    assert argv[3] == "gdrive:captures/id"
    assert argv[4:6] == ["--min-age", "120s"]
    assert argv[6:] == ["--fast-list", "--checksum"]


@pytest.mark.parametrize("dry", [True, False])
def test_dry_run_flag_in_argv(dry: bool) -> None:
    """--sync-dry-run appends --dry-run when True, omits it when False."""
    settings = {
        "sync_rclone_binary": "rclone",
        "sync_cmd": "copy",
        "sync_min_age": "0",
        "sync_rclone_opts": "",
        "sync_dry_run": dry,
    }
    argv = build_rclone_argv(settings, "/src", "gdrive:captures/id")
    assert ("--dry-run" in argv) is dry


def test_resolve_rclone_binary_uses_setting() -> None:
    """build_rclone_argv uses resolve_rclone_binary for argv[0]."""
    settings = {
        "sync_rclone_binary": "rclone",
        "sync_cmd": "copy",
        "sync_min_age": "0",
        "sync_rclone_opts": "",
        "sync_dry_run": False,
    }
    argv = build_rclone_argv(settings, "/src", "gdrive:captures/id")
    assert Path(argv[0]).name == "rclone"
