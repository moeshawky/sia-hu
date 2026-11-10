"""sia publish — push current branch and verify the remote ref.

Pure stdlib. Flags: ``--dry-run``, ``--allow-dirty``. Never auto-commits —
publish only pushes existing commits. A dirty tree is refused unless
``--allow-dirty`` overrides the refusal (it never commits, never force-pushes).

Exit codes: 0 on verified success, 1 on any refusal/failure/mismatch, 2 on flag
errors.
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys

_REDACT = re.compile(r"://[^/@]+@")

#: git binary resolved to an absolute path (avoids bandit S607 partial-path).
_GIT_BIN = shutil.which("git") or "git"


def _redact(value: str) -> str:
    """Redact credentials embedded in a remote URL."""
    return _REDACT.sub("://***@", value)


def _git(args: list[str]) -> subprocess.CompletedProcess[str]:
    """Run a git command, capturing output without raising on non-zero rc."""
    return subprocess.run(  # noqa: S603 - explicit argv, no shell; _GIT_BIN is absolute
        [_GIT_BIN, *args], capture_output=True, text=True, check=False
    )


def main(args: list[str] | None = None) -> int:
    """Publish the current branch and verify the remote ref matches local HEAD."""
    if args is None:
        args = sys.argv[1:]
    parser = argparse.ArgumentParser(
        prog="sia publish",
        description=(
            "Push the current branch and verify the remote ref matches local "
            "HEAD. Never auto-commits."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="show the commands, execute none (skip push and verification)",
    )
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help=("override the dirty-tree refusal (does not commit, never force-push)"),
    )
    parsed = parser.parse_args(args)

    # 1. Dirty-tree check (unless --allow-dirty)
    proc = _git(["status", "--porcelain"])
    dirty = bool(proc.stdout.strip())
    if dirty and not parsed.allow_dirty:
        print(
            "error: working tree is dirty; commit or stash first, or use "
            "--allow-dirty to override — run: git status --short",
            file=sys.stderr,
        )
        return 1
    if dirty:
        print(
            "warn: working tree is dirty; --allow-dirty overrides refusal "
            "(no commit, no force-push)"
        )

    # 2. Detached HEAD check
    proc = _git(["symbolic-ref", "--short", "HEAD"])
    if proc.returncode != 0:
        print(
            "error: detached HEAD; checkout a branch first — run: git checkout -b <branch>",
            file=sys.stderr,
        )
        return 1
    branch = proc.stdout.strip()

    # 4. Origin check
    proc = _git(["remote", "get-url", "origin"])
    if proc.returncode != 0:
        print(
            "error: no origin remote; add one — run: git remote add origin <url>",
            file=sys.stderr,
        )
        return 1
    origin_url = _redact(proc.stdout.strip())
    print(f"remote origin {origin_url}")

    # 5. Push (executed unless --dry-run)
    push_cmd = ["push", "-u", "origin", branch]
    print("git " + " ".join(_redact(c) for c in push_cmd))
    if not parsed.dry_run:
        proc = _git(push_cmd)
        if proc.returncode != 0:
            message = proc.stderr.strip() or proc.stdout.strip()
            print(f"error: push failed — {message}", file=sys.stderr)
            return 1

    if parsed.dry_run:
        print("dry-run: not pushed, not verified")
        return 0

    # 6. Verify remote ref matches local HEAD
    local_sha = _git(["rev-parse", "HEAD"]).stdout.strip()
    remote_proc = _git(["ls-remote", "origin", f"refs/heads/{branch}"])
    remote_lines = remote_proc.stdout.strip().splitlines()
    remote_sha = remote_lines[0].split()[0] if remote_lines else ""
    if not remote_sha or remote_sha != local_sha:
        print(
            "error: push verification failed — local HEAD "
            f"{local_sha} != remote refs/heads/{branch} "
            f"{remote_sha or '<absent>'}; re-run sia publish",
            file=sys.stderr,
        )
        return 1

    print(f"ok: push verified — remote refs/heads/{branch} == local HEAD {local_sha}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
