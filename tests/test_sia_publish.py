"""Tests for ``sia publish`` — push current branch and verify remote ref.

Driven by monkeypatching ``subprocess.run`` (for git) and capturing stdout.
Never creates a commit; never force-pushes.
"""

from __future__ import annotations

import subprocess
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

import pytest

import sia.publish


class _Git:
    """Fake ``subprocess.run`` for git commands used by publish."""

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
    monkeypatch.setattr(sia.publish.subprocess, "run", g)
    return g


def _clean_repo(fake_git: _Git, branch: str = "main") -> None:
    fake_git.set("status", "--porcelain", rc=0, out="")
    fake_git.set("symbolic-ref", "--short", "HEAD", rc=0, out=branch)
    fake_git.set("remote", "get-url", "origin", rc=0, out="http://host/moe/sia-hu.git")
    fake_git.set("push", "-u", "origin", branch, rc=0, out="")
    fake_git.set("rev-parse", "HEAD", rc=0, out="abc123def456")
    fake_git.set(
        "ls-remote",
        "origin",
        f"refs/heads/{branch}",
        rc=0,
        out="abc123def456\trefs/heads/main\n",
    )


def test_publish_verified_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_git: _Git, capsys
) -> None:
    """push success + SHA match → exit 0."""
    _clean_repo(fake_git)

    rc = sia.publish.main([])

    assert rc == 0
    out = capsys.readouterr().out
    assert (
        "ok: push verified — remote refs/heads/main == local HEAD abc123def456" in out
    )
    # push was issued
    assert any(c[1:] == ["push", "-u", "origin", "main"] for c in fake_git.commands)


def test_publish_dirty_tree_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_git: _Git, capsys
) -> None:
    """dirty tree refused → exit 1, no push issued."""
    fake_git.set("status", "--porcelain", rc=0, out=" M src/sia/cli.py\n")

    rc = sia.publish.main([])

    assert rc == 1
    err = capsys.readouterr().err
    assert "working tree is dirty" in err
    assert "Commit or stash first, or use --allow-dirty to override" in err
    assert not any(c[1:3] == ["push"] for c in fake_git.commands)


def test_publish_allow_dirty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_git: _Git, capsys
) -> None:
    """--allow-dirty overrides dirty tree, proceeds, exit 0."""
    fake_git.set("status", "--porcelain", rc=0, out=" M src/sia/cli.py\n")
    _clean_repo(fake_git)

    rc = sia.publish.main(["--allow-dirty"])

    assert rc == 0
    assert any(c[1:] == ["push", "-u", "origin", "main"] for c in fake_git.commands)


def test_publish_detached_head_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_git: _Git, capsys
) -> None:
    """detached HEAD → exit 1."""
    fake_git.set("status", "--porcelain", rc=0, out="")
    fake_git.set("symbolic-ref", "--short", "HEAD", rc=1, err="HEAD detached")

    rc = sia.publish.main([])

    assert rc == 1
    err = capsys.readouterr().err
    assert "detached HEAD" in err
    assert "git checkout -b" in err


def test_publish_no_origin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_git: _Git, capsys
) -> None:
    """no origin → exit 1."""
    fake_git.set("status", "--porcelain", rc=0, out="")
    fake_git.set("symbolic-ref", "--short", "HEAD", rc=0, out="main")
    fake_git.set("remote", "get-url", "origin", rc=1, err="no such remote")

    rc = sia.publish.main([])

    assert rc == 1
    err = capsys.readouterr().err
    assert "no origin remote" in err


def test_publish_dry_run_skips_verification(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_git: _Git, capsys
) -> None:
    """--dry-run: prints command, skips push and verification, exit 0."""
    fake_git.set("status", "--porcelain", rc=0, out="")
    fake_git.set("symbolic-ref", "--short", "HEAD", rc=0, out="main")
    fake_git.set("remote", "get-url", "origin", rc=0, out="http://host/moe/sia-hu.git")

    rc = sia.publish.main(["--dry-run"])

    assert rc == 0
    out = capsys.readouterr().out
    assert "git push -u origin main" in out
    assert "dry-run: not pushed, not verified" in out
    assert not any(c[1:3] == ["push"] for c in fake_git.commands)
    assert not any(c[1:3] == ["rev-parse"] for c in fake_git.commands)
    assert not any(c[1:3] == ["ls-remote"] for c in fake_git.commands)


def test_publish_remote_sha_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_git: _Git, capsys
) -> None:
    """remote SHA mismatch → hard failure, both SHAs, exit 1."""
    fake_git.set("status", "--porcelain", rc=0, out="")
    fake_git.set("symbolic-ref", "--short", "HEAD", rc=0, out="main")
    fake_git.set("remote", "get-url", "origin", rc=0, out="http://host/moe/sia-hu.git")
    fake_git.set("push", "-u", "origin", "main", rc=0, out="")
    fake_git.set("rev-parse", "HEAD", rc=0, out="abc123def456")
    fake_git.set(
        "ls-remote",
        "origin",
        "refs/heads/main",
        rc=0,
        out="deadbeef\trefs/heads/main\n",
    )

    rc = sia.publish.main([])

    assert rc == 1
    err = capsys.readouterr().err
    assert "push verification failed" in err
    assert "abc123def456" in err
    assert "deadbeef" in err


def test_publish_no_auto_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_git: _Git
) -> None:
    """publish never issues git add/commit."""
    _clean_repo(fake_git)
    sia.publish.main([])
    git_cmds = [c[1:] for c in fake_git.commands]
    for cmd in git_cmds:
        assert cmd[0] not in ("add", "commit")


def test_publish_redacts_credentials(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_git: _Git, capsys
) -> None:
    """credential-bearing origin URL is redacted in output."""
    _clean_repo(fake_git)
    fake_git.set(
        "remote",
        "get-url",
        "origin",
        rc=0,
        out="http://moe:gtok-secret@host/moe/sia-hu.git",
    )

    sia.publish.main([])
    out = capsys.readouterr().out
    err = capsys.readouterr().err
    combined = out + err
    assert "://***@" in combined
    assert "gtok-secret" not in combined


def test_publish_push_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_git: _Git, capsys
) -> None:
    """push fails → exit 1, no verification."""
    fake_git.set("status", "--porcelain", rc=0, out="")
    fake_git.set("symbolic-ref", "--short", "HEAD", rc=0, out="main")
    fake_git.set("remote", "get-url", "origin", rc=0, out="http://host/moe/sia-hu.git")
    fake_git.set("push", "-u", "origin", "main", rc=1, err="remote rejected")

    rc = sia.publish.main([])

    assert rc == 1
    err = capsys.readouterr().err
    assert "push failed" in err
    assert not any(c[1:3] == ["rev-parse"] for c in fake_git.commands)
