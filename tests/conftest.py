"""Shared fixtures for the root ``tests/`` suite (capture-proxy sync).

Provides a stub ``rclone`` on PATH that records its argv to a file and exits 0
(or a configurable exit code), so sync tests never touch the real network or
rclone binary.
"""

from __future__ import annotations

import os
import stat
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

import pytest


@pytest.fixture
def stub_rclone(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Create a stub ``rclone`` on PATH that records argv and exits 0.

    Returns the path to the recorded-argv file. Each invocation appends one
    line (the argv as a space-joined string) to that file.
    """
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    record_file = bin_dir / "rclone.argv"
    stub = bin_dir / "rclone"
    stub.write_text(
        '#!/bin/sh\necho "$0 $*" >> "' + str(record_file) + '"\nexit 0\n',
        encoding="utf-8",
    )
    stub.chmod(stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR)
    monkeypatch.setenv("PATH", str(bin_dir) + os.pathsep + os.environ.get("PATH", ""))
    return record_file


@pytest.fixture
def failing_rclone(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Stub ``rclone`` that records argv and exits non-zero (1)."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    record_file = bin_dir / "rclone.argv"
    stub = bin_dir / "rclone"
    stub.write_text(
        '#!/bin/sh\necho "$0 $*" >> "' + str(record_file) + '"\nexit 1\n',
        encoding="utf-8",
    )
    stub.chmod(stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR)
    monkeypatch.setenv("PATH", str(bin_dir) + os.pathsep + os.environ.get("PATH", ""))
    return record_file


def read_argv(record_file: Path) -> list[list[str]]:
    """Return the list of argv invocations captured by the stub rclone."""
    if not record_file.is_file():
        return []
    return [
        line.split(" ")
        for line in record_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def run_sync(args: list[str], monkeypatch: pytest.MonkeyPatch) -> int:
    """Invoke ``capture_proxy.__main__.main`` with the given args and return
    its exit code, isolating env so tests don't leak into each other."""
    from capture_proxy.__main__ import main

    return main(args=args)
