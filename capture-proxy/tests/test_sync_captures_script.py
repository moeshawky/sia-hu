"""Behaviour tests for rclone-sync-captures.sh.

These run the real script with a stub `rclone` on PATH and read back the exact
argv it would have executed. Asserting on the script's own text would only
prove the comment says the right thing; this proves the COMMAND does.

The property under test is the one that silently destroys data: which instance
id becomes the destination prefix, and whether the default can delete.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "rclone-sync-captures.sh"


def _stub_bin(tmp_path: Path, args_file: Path) -> Path:
    """A PATH shim that records the rclone argv instead of transferring."""
    bin_dir = tmp_path / "stub-bin"
    bin_dir.mkdir()
    stub = bin_dir / "rclone"
    stub.write_text(
        f'#!/usr/bin/env bash\nprintf "%s\\n" "$*" >> "{args_file}"\nexit 0\n',
        encoding="utf-8",
    )
    stub.chmod(0o755)
    return bin_dir


def _run(
    tmp_path: Path,
    *,
    source: Path,
    env_overrides: dict[str, str] | None = None,
) -> str:
    args_file = tmp_path / "rclone-args.txt"
    bin_dir = _stub_bin(tmp_path, args_file)
    source.mkdir(parents=True, exist_ok=True)

    env = dict(os.environ)
    env.update(
        {
            "PATH": f"{bin_dir}:{env['PATH']}",
            "HOME": str(tmp_path),
            "CAPTURE_BASE_PATH": str(source),
            "DRY_RUN": "true",
            "LOG_DIR": str(tmp_path / "logs"),
        }
    )
    for key in ("CAPTURE_MACHINE_ID", "SYNC_CMD", "REMOTE"):
        env.pop(key, None)
    env.update(env_overrides or {})

    proc = subprocess.run(  # fixed argv: the script under test, no shell
        ["bash", str(SCRIPT)],  # noqa: S607 - bash resolved from PATH by design
        capture_output=True,
        text=True,
        env=env,
        check=False,
        timeout=60,
    )
    assert proc.returncode == 0, f"stdout={proc.stdout!r} stderr={proc.stderr!r}"
    assert args_file.is_file(), f"rclone was never invoked: {proc.stdout!r}"
    return args_file.read_text(encoding="utf-8")


class TestDestinationKeying:
    def test_destination_is_hostname_not_machine_id(self, tmp_path: Path) -> None:
        """Hostname wins: /etc/machine-id is image-baked and shared by every
        instance of this image (measured 2026-10-13), which would put two live
        instances under one prefix."""
        recorded = _run(tmp_path, source=tmp_path / "captures")

        host = subprocess.run(
            ["hostname"],  # noqa: S607 - hostname resolved from PATH by design
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        machine_id = Path("/etc/machine-id").read_text(encoding="utf-8").strip()

        assert f"gdrive:captures/{host}" in recorded
        if machine_id and machine_id != host:
            assert f"gdrive:captures/{machine_id}" not in recorded

    def test_explicit_override_wins(self, tmp_path: Path) -> None:
        recorded = _run(
            tmp_path,
            source=tmp_path / "captures",
            env_overrides={"CAPTURE_MACHINE_ID": "stable-dev-box"},
        )
        assert "gdrive:captures/stable-dev-box" in recorded


class TestSyncCommandDefault:
    def test_default_is_copy_never_a_destructive_mirror(self, tmp_path: Path) -> None:
        recorded = _run(tmp_path, source=tmp_path / "captures")

        assert recorded.startswith("copy "), recorded
        assert " sync " not in f" {recorded} "

    def test_sync_still_available_when_explicitly_requested(
        self, tmp_path: Path
    ) -> None:
        recorded = _run(
            tmp_path,
            source=tmp_path / "captures",
            env_overrides={"SYNC_CMD": "sync"},
        )
        assert recorded.startswith("sync "), recorded


class TestBootRobustness:
    def test_missing_source_dir_is_created_not_fatal(self, tmp_path: Path) -> None:
        """A fresh instance has no /root/captures until the first capture."""
        source = tmp_path / "not-yet"
        assert not source.exists()

        _run(tmp_path, source=source)

        assert source.is_dir()

    def test_rejects_unknown_sync_command(self, tmp_path: Path) -> None:
        env = dict(os.environ)
        env.update(
            {
                "HOME": str(tmp_path),
                "CAPTURE_BASE_PATH": str(tmp_path / "captures"),
                "SYNC_CMD": "delete-everything",
            }
        )
        proc = subprocess.run(  # fixed argv: the script under test, no shell
            ["bash", str(SCRIPT)],  # noqa: S607 - bash resolved from PATH by design
            capture_output=True,
            text=True,
            env=env,
            check=False,
            timeout=60,
        )
        assert proc.returncode == 1
        assert "SYNC_CMD" in proc.stderr


@pytest.mark.parametrize("flag", ["--help"])
def test_script_is_valid_bash_and_documents_keys(flag: str) -> None:
    syntax = subprocess.run(
        ["bash", "-n", str(SCRIPT)],  # noqa: S607 - bash from PATH by design
        capture_output=True,
        text=True,
        check=False,
    )
    assert syntax.returncode == 0, syntax.stderr

    help_out = subprocess.run(
        ["bash", str(SCRIPT), flag],  # noqa: S607 - bash from PATH by design
        capture_output=True,
        text=True,
        check=False,
    ).stdout
    assert "CAPTURE_MACHINE_ID" in help_out
    assert "hostname" in help_out
