"""sia — dispatch layer for capture-proxy, model-headroom, and sia-owned verbs.

Pure dispatch: imports the existing mains lazily (so ``sia`` itself pulls in no
heavy dependencies) and forwards all remaining args to the matching sub-main,
propagating its exit code. Existing carrier/authority CLIs are never modified.

``sia`` also owns two read-only/write-verified verbs implemented here:
``doctor`` (workspace preflight) and ``publish`` (push + verify remote ref,
never auto-commits).
"""

from __future__ import annotations

import argparse
import importlib
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

#: (subcommand name, importable module, attribute name of main, help text)
_SUBCOMMANDS = (
    ("capture-proxy", "capture_proxy.__main__", "main", "Run the capture-proxy CLI."),
    ("model-headroom", "model_headroom.cli", "main", "Run the model-headroom CLI."),
    (
        "model-headroom-refusal",
        "model_headroom.refusal.cli",
        "main",
        "Run the model-headroom-refusal CLI.",
    ),
    ("doctor", "sia.doctor", "main", "Read-only workspace preflight."),
    (
        "publish",
        "sia.publish",
        "main",
        "Push current branch; verify remote ref never auto-commits.",
    ),
)

_VALID_COMMANDS = {name for name, _, _, _ in _SUBCOMMANDS}


def _top_parser() -> argparse.ArgumentParser:
    """Top-level parser used only to render ``sia`` help (lists subcommands)."""
    parser = argparse.ArgumentParser(
        prog="sia",
        add_help=False,
        description=(
            "sia — unified CLI aliasing capture-proxy, model-headroom, and "
            "model-headroom-refusal. All arguments after the subcommand are "
            "forwarded unchanged to that CLI."
        ),
    )
    parser.add_argument(
        "command",
        nargs="?",
        choices=sorted(_VALID_COMMANDS),
        help="Subcommand to run.",
    )
    return parser


def main(args: list[str] | None = None) -> int:
    """Parse args, dispatch to the matching sub-main, return its exit code.

    Args:
        args: Argument list. ``None`` means ``sys.argv[1:]``.

    Returns:
        The exit code returned by the underlying sub-main.
    """
    if args is None:
        args = sys.argv[1:]

    parser = _top_parser()

    # No subcommand, or explicit help: render the top-level help (which lists
    # all five subcommands) and exit successfully. ``--help``/-h is handled
    # here so it is never swallowed by the top parser and can instead be
    # forwarded to a sub-main (e.g. ``sia capture-proxy --help``).
    if not args:
        parser.print_help(sys.stderr)
        return 2
    if args[0] in ("--help", "-h"):
        parser.print_help()
        return 0

    command = args[0]
    rest = args[1:]

    if command not in _VALID_COMMANDS:
        print(f"sia: unknown subcommand {command!r}", file=sys.stderr)
        parser.print_help(sys.stderr)
        return 2

    for name, module, func, _ in _SUBCOMMANDS:
        if command == name:
            mod = importlib.import_module(module)
            sub_main: Callable[[list[str] | None], int] = getattr(mod, func)
            return sub_main(rest)

    # Unreachable thanks to the membership check above, but keep a safe fallback.
    parser.print_help(sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
