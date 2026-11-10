"""Tests for ``sia doctor`` — read-only workspace preflight.

All checks are driven by monkeypatching ``shutil.which``, ``subprocess.run``
(for git), and ``sia.doctor._CONFIG_PATH``. Nothing touches the network, writes
a file, or runs a real verify-chain tool.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

import pytest

import sia.doctor


class _Git:
    """Fake ``subprocess.run`` for git read-only commands."""

    def __init__(self) -> None:
        self.commands: list[list[str]] = []
        self.returns: dict[tuple[str, ...], tuple[int, str, str]] = {}

    def set(self, *git_args: str, rc: int = 0, out: str = "", err: str = "") -> None:
        self.returns[tuple(git_args)] = (rc, out, err)

    def __call__(
        self, cmd: list[str], **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        self.commands.append(list(cmd))
        sub = tuple(cmd[1:])  # drop the leading "git"
        rc, out, err = self.returns.get(sub, (128, "", "faked git failure"))
        return subprocess.CompletedProcess(cmd, rc, out, err)


@pytest.fixture
def fake_git(monkeypatch: pytest.MonkeyPatch) -> _Git:
    g = _Git()
    monkeypatch.setattr(sia.doctor.subprocess, "run", g)
    return g


def _tools_present(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(shutil, "which", lambda tool: f"/usr/bin/{tool}")


def _tools_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(shutil, "which", lambda tool: None)


def _config_valid(tmp_path: Path) -> Path:
    cfg = tmp_path / "config.toml"
    cfg.write_text(
        "port = 8000\nupstream_url = 'http://localhost:11434'\n", encoding="utf-8"
    )
    sia.doctor._CONFIG_PATH = cfg
    return cfg


def _git_ok(fake_git: _Git) -> None:
    fake_git.set("--version", rc=0, out="git version 2.43.0")
    fake_git.set("rev-parse", "--show-toplevel", rc=0, out="/repo")
    fake_git.set("symbolic-ref", "--short", "HEAD", rc=0, out="main")
    fake_git.set(
        "remote", "get-url", "origin", rc=0, out="http://moe:gtok@host/moe/sia-hu.git"
    )


def test_doctor_all_pass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_git: _Git
) -> None:
    """Every check passes → exit 0."""
    _tools_present(monkeypatch)
    _git_ok(fake_git)
    _config_valid(tmp_path)

    rc = sia.doctor.main([])
    assert rc == 0


def test_doctor_git_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_git: _Git
) -> None:
    """git not on PATH → error names fix command, exit 1."""
    _tools_absent(monkeypatch)
    _config_valid(tmp_path)

    rc = sia.doctor.main([])
    assert rc == 1


def test_doctor_inside_package_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_git: _Git
) -> None:
    """cwd inside a package dir → refusal, exit 1."""
    _tools_present(monkeypatch)
    _git_ok(fake_git)
    _config_valid(tmp_path)
    monkeypatch.setattr(os, "getcwd", lambda: "/repo/capture-proxy")

    rc = sia.doctor.main([])
    assert rc == 1


def test_doctor_no_origin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_git: _Git
) -> None:
    """no origin remote → error names fix command, exit 1."""
    _tools_present(monkeypatch)
    fake_git.set("--version", rc=0, out="git version 2.43.0")
    fake_git.set("rev-parse", "--show-toplevel", rc=0, out="/repo")
    fake_git.set("symbolic-ref", "--short", "HEAD", rc=0, out="main")
    fake_git.set("remote", "get-url", "origin", rc=1, err="no such remote")
    _config_valid(tmp_path)

    rc = sia.doctor.main([])
    assert rc == 1


def test_doctor_root_pyproject(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_git: _Git
) -> None:
    """root pyproject + 3 inits + wheel packages list → ok lines, exit 0."""
    _tools_present(monkeypatch)
    _git_ok(fake_git)
    _config_valid(tmp_path)

    rc = sia.doctor.main([])
    assert rc == 0


def test_doctor_invalid_toml_fix_text(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_git: _Git
) -> None:
    """invalid TOML → error names fix command, never overwrites, exit 1."""
    _tools_present(monkeypatch)
    _git_ok(fake_git)
    cfg = tmp_path / "config.toml"
    cfg.write_text("port = \n", encoding="utf-8")  # invalid TOML
    sia.doctor._CONFIG_PATH = cfg

    rc = sia.doctor.main([])
    assert rc == 1
    # config file untouched (never overwritten)
    assert cfg.read_text(encoding="utf-8") == "port = \n"


def test_doctor_verify_tool_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_git: _Git
) -> None:
    """ruff missing → error names fix command, exit 1."""

    def which(tool: str) -> str | None:
        if tool == "ruff":
            return None
        return f"/usr/bin/{tool}"

    monkeypatch.setattr(shutil, "which", which)
    _git_ok(fake_git)
    _config_valid(tmp_path)

    rc = sia.doctor.main([])
    assert rc == 1


def test_doctor_read_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_git: _Git
) -> None:
    """doctor performs no file writes and no mutating git calls."""
    _tools_present(monkeypatch)
    _git_ok(fake_git)
    _config_valid(tmp_path)

    writes: list[str] = []
    real_open = open

    def tracking_open(file, mode="r", *a, **k):
        if "w" in mode or "a" in mode or "+" in mode:
            writes.append(str(file))
        return real_open(file, mode, *a, **k)

    monkeypatch.setattr("builtins.open", tracking_open)
    sia.doctor.main([])
    assert writes == []

    git_cmds = [c[1:] for c in fake_git.commands]
    mutating = {"add", "commit", "push", "reset", "checkout", "merge", "rebase"}
    for cmd in git_cmds:
        assert cmd[0] not in mutating, f"mutating git call: {cmd}"
