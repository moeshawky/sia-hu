"""sia doctor — read-only workspace preflight.

Pure stdlib: no torch/uvicorn/fastapi import. Changes nothing — no file writes,
no ``git add``/``commit``/``push``, no mutating subprocess. Only ``command -v``
style availability probes, read-only git commands, ``tomllib.load`` (read-only),
and ``importlib.metadata.version``.

Each check prints one line prefixed ``ok:``/``warn:``/``error:``; every
``error:`` ends with `` — `` plus the exact fix command.

Exit codes: 0 if zero errors (warnings allowed), 1 if any error, 2 on flag
errors.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tomllib
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _dist_version
from pathlib import Path

#: git binary resolved to an absolute path (avoids bandit S607 partial-path);
#: step 1 errors out before any subprocess runs if git is missing.
_GIT_BIN = shutil.which("git") or "git"

#: capture-proxy default config path (sia doctor has no --config flag).
_CONFIG_PATH = Path.home() / ".config" / "capture-proxy" / "config.toml"

#: wheel packages the root pyproject.toml must list for all three packages.
_WHEEL_PACKAGES = ("src/capture_proxy", "src/model_headroom", "src/sia")

#: package dirs that must exist under src/.
_PACKAGE_DIRS = ("capture_proxy", "model_headroom", "sia")

#: package dirs whose presence is an error (not a warn) if missing.
_REQUIRED_INITS = ("capture_proxy", "model_headroom", "sia")


def _git(args: list[str]) -> tuple[int, str, str]:
    """Run a read-only git command; return ``(rc, stdout, stderr)``."""
    try:
        proc = subprocess.run(  # noqa: S603 - explicit argv, no shell; _GIT_BIN is absolute
            [_GIT_BIN, *args], capture_output=True, text=True, check=False
        )
    except FileNotFoundError:
        return 127, "", "git not found"
    return proc.returncode, proc.stdout.strip(), proc.stderr.strip()


def _tool_version(tool: str) -> str | None:
    """Version string for an installed distribution, or ``None``."""
    try:
        return _dist_version(tool)
    except PackageNotFoundError:
        return None


def _redact(url: str) -> str:
    """Redact credentials embedded in a remote URL."""
    import re

    return re.sub(r"://[^/@]+@", "://***@", url)


def main(args: list[str] | None = None) -> int:
    """Run the preflight checks and return an exit code."""
    if args is None:
        args = sys.argv[1:]
    parser = argparse.ArgumentParser(
        prog="sia doctor",
        description="Read-only workspace preflight. Exits 0 if clean, 1 on any error.",
    )
    parser.parse_args(args)

    errors = 0
    warns = 0

    def ok(msg: str) -> None:
        print(f"ok: {msg}")

    def warn(msg: str) -> None:
        nonlocal warns
        warns += 1
        print(f"warn: {msg}")

    def error(msg: str) -> None:
        nonlocal errors
        errors += 1
        print(f"error: {msg}")

    # 1. git on PATH
    if shutil.which("git") is None:
        error("git not found on PATH — install git — run: apt-get install git")
    else:
        rc, ver, _ = _git(["--version"])
        ok(f"git {ver if ver else 'available'}")

    # 2. cwd is not a package dir
    cwd_name = Path.cwd().name
    if cwd_name in ("capture-proxy", "model_headroom"):
        error(
            f"refusing: already inside a package dir ({cwd_name}); cd to the parent — run: cd .."
        )
    else:
        ok(f"cwd {Path.cwd()} is repo root")

    # 3. repo root and branch
    rc, root, _ = _git(["rev-parse", "--show-toplevel"])
    if rc != 0:
        error("not a git repository — run: git init")
    else:
        rc2, branch, _ = _git(["symbolic-ref", "--short", "HEAD"])
        if rc2 != 0:
            error("not on a branch (detached HEAD) — run: git checkout -b <branch>")
        else:
            ok(f"repo root {root}, branch {branch}")

    # 4. remote origin resolvable
    rc, url, _ = _git(["remote", "get-url", "origin"])
    if rc != 0:
        error("no origin remote; add one — run: git remote add origin <url>")
    else:
        ok(f"remote origin {_redact(url)}")

    # 5. tracked package presence: root pyproject.toml + 3 __init__ + wheel list
    repo_root = Path.cwd()
    pyproject = repo_root / "pyproject.toml"
    if not pyproject.is_file():
        error("root pyproject.toml missing — run from the repo root")
    else:
        missing_inits = [
            pkg
            for pkg in _REQUIRED_INITS
            if not (repo_root / "src" / pkg / "__init__.py").is_file()
        ]
        if missing_inits:
            error(
                f"{', '.join(missing_inits)} missing or untracked — a release checkout "
                "must carry all three packages — run: git checkout -- capture-proxy model_headroom src"
            )
        else:
            ok("capture-proxy present — tracked package")

        try:
            data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
        except tomllib.TOMLDecodeError as exc:
            error(f"invalid TOML in root pyproject.toml: {exc} — fix the TOML syntax")
            data = {}
        packages = (
            data.get("tool", {})
            .get("hatch", {})
            .get("build", {})
            .get("targets", {})
            .get("wheel", {})
            .get("packages", [])
        )
        missing_packages = [p for p in _WHEEL_PACKAGES if p not in packages]
        if missing_packages:
            error(
                f"{', '.join(missing_packages)} missing from wheel packages list in "
                "root pyproject.toml — run: edit pyproject.toml"
            )

    # 6. capture-proxy config path and parse validity (read-only, never overwrites)
    if not _CONFIG_PATH.is_file():
        warn(f"no config at {_CONFIG_PATH} — run: capture-proxy --ensure-config")
    else:
        try:
            tomllib.loads(_CONFIG_PATH.read_text(encoding="utf-8"))
            ok(f"config {_CONFIG_PATH} valid TOML")
        except tomllib.TOMLDecodeError as exc:
            error(
                f"invalid TOML in config {_CONFIG_PATH}: {exc} — fix the TOML "
                "syntax, or move the file aside and run capture-proxy --ensure-config "
                "(it never overwrites an existing file)"
            )

    # 7. verify-chain tools available (probe only — never execute)
    for tool in ("ruff", "mypy", "pytest"):
        if shutil.which(tool) is None:
            error(f"{tool} not found on PATH — run: uv sync --extra dev")
        else:
            version = _tool_version(tool)
            ok(f"{tool} {version if version else 'available'}")

    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
