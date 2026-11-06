#!/usr/bin/env python3
"""Set the model for opencode agent files (.opencode/agents/*.md).

Reads every markdown agent definition in a directory, extracts the current
model from the YAML frontmatter ``model:`` field, and rewrites EVERY
occurrence of that model identifier — the full ``provider/model`` form and
the bare suffix form used in prose markers like ``[ROLE on model]`` and
``Model: provider/model.`` — with a new model id supplied by the operator.

Two modes:

  Flag mode       set-agent-model.py --model <id> [--dir DIR] [--write]
  Interactive     set-agent-model.py [--dir DIR]
  Query only      set-agent-model.py --list [--dir DIR]

Writes are opt-in: without ``--write`` the script prints a dry-run preview.
Interactive mode additionally asks for an explicit y/N confirmation before
writing.

# Errors

Exits with status code:
  0  success (including "0 files needed changes")
  1  runtime failure — directory missing, no *.md files, or no file carries
     a frontmatter ``model:`` field
  2  usage error — invalid arguments (argparse)

Individual files without a ``model:`` field are reported as skipped and do
not change the exit status, unless NO file has one (then 1).
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path

DEFAULT_DIR = ".opencode/agents"
# Characters that may legally appear inside a model id segment; a match
# adjacent to any of these is part of a longer token, not a standalone id.
_ID_NEIGHBOR = r"[\w.\-/]"
_FRONTMATTER_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*\n?", re.DOTALL)
_MODEL_LINE_RE = re.compile(r"^model:\s*(\S+)\s*$", re.MULTILINE)


@dataclass(frozen=True)
class AgentFile:
    path: Path
    model: str  # current full model id, e.g. "prankh/qwen3.8-27b-fp8"


def scan_agents(directory: Path) -> tuple[list[AgentFile], list[tuple[str, str]]]:
    """Read all ``*.md`` files and extract each frontmatter ``model:`` value.

    Returns ``(found, skipped)`` where *skipped* holds ``(filename, reason)``
    pairs for files with no detectable model field. Unreadable files abort
    the scan with SystemExit(1).
    """
    found: list[AgentFile] = []
    skipped: list[tuple[str, str]] = []
    md_files = sorted(directory.glob("*.md"))
    if not md_files:
        _fail(f"no .md files found in {directory}")
    for path in md_files:
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:
            _fail(f"cannot read {path}: {exc}")
        match = _MODEL_LINE_RE.search(_frontmatter_of(text))
        if match:
            found.append(AgentFile(path=path, model=match.group(1)))
        else:
            skipped.append((path.name, "no frontmatter 'model:' field"))
    return found, skipped


def _frontmatter_of(text: str) -> str:
    m = _FRONTMATTER_RE.match(text)
    return m.group(1) if m else ""


def compute_replacement(text: str, old_full: str, new_full: str) -> tuple[str, int]:
    """Return (new_text, occurrences_replaced).

    Strategy, in order:
      1. Literal replace of the full id (``provider/model``) — covers the
         functional frontmatter field and fully-qualified prose references.
      2. Boundary-guarded replace of the remaining bare suffix — covers
         short-form markers (``on qwen-x``) without touching tokens that
         merely contain the suffix (``other-qwen-x``, ``qwen-x-v2``).

    Invariant: the returned text contains zero boundary-delimited
    occurrences of *old_full* or its bare suffix.
    """
    count = text.count(old_full)
    out = text.replace(old_full, new_full)
    old_bare = old_full.rsplit("/", 1)[-1]
    new_bare = new_full.rsplit("/", 1)[-1]
    if old_bare != old_full and old_bare != new_bare:
        pattern = re.compile(
            rf"(?<!{_ID_NEIGHBOR})" + re.escape(old_bare) + rf"(?!{_ID_NEIGHBOR})"
        )
        out, n = pattern.subn(new_bare, out)
        count += n
    return out, count


def apply(files: list[AgentFile], new_model: str, *, write: bool) -> int:
    """Rewrite every occurrence of each file's current model. Returns number
    of files changed (or that WOULD change when ``write`` is False).

    Raises SystemExit(1) if a post-write verification finds stale
    references — never claims success silently.
    """
    changed = 0
    for af in files:
        if af.model == new_model:
            print(f"  = {af.path.name}: already {new_model}")
            continue
        try:
            text = af.path.read_text(encoding="utf-8")
        except OSError as exc:
            _fail(f"cannot read {af.path}: {exc}")
        new_text, n = compute_replacement(text, af.model, new_model)
        if n == 0:
            print(
                f"  ! {af.path.name}: frontmatter says '{af.model}' but no occurrence "
                "found in body — file may be malformed; skipped"
            )
            continue
        label = "UPDATED" if write else "would update"
        print(
            f"  {label} {af.path.name} ({n} occurrence(s)): {af.model} -> {new_model}"
        )
        if write:
            try:
                af.path.write_text(new_text, encoding="utf-8")
            except OSError as exc:
                _fail(f"cannot write {af.path}: {exc}")
            verify(af.path, af.model)
        changed += 1
    return changed


def verify(path: Path, old_full: str) -> None:
    """Assert no stale references remain after a write."""
    text = path.read_text(encoding="utf-8")
    old_bare = old_full.rsplit("/", 1)[-1]
    stale_full = old_full in text
    stale_bare = (
        old_bare != old_full
        and re.search(
            rf"(?<!{_ID_NEIGHBOR}){re.escape(old_bare)}(?!{_ID_NEIGHBOR})", text
        )
        is not None
    )
    if stale_full or stale_bare:
        _fail(f"verification failed: {path.name} still references '{old_full}'")


def print_table(files: list[AgentFile], skipped: list[tuple[str, str]]) -> None:
    width = max((len(a.path.name) for a in files), default=0)
    for af in files:
        print(f"  {af.path.name:<{width}}  {af.model}")
    for name, reason in skipped:
        print(f"  {name:<{width}}  [skip: {reason}]")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="set-agent-model.py",
        description="Swap the model id in opencode agent files (frontmatter + inline references).",
        epilog=(
            "examples:\n"
            "  %(prog)s --list\n"
            "  %(prog)s --model opencode/x-preview-f-free            # dry-run\n"
            "  %(prog)s --model opencode/x-preview-f-free --write    # apply\n"
            "  %(prog)s                                              # interactive\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dir", default=DEFAULT_DIR, help=f"agents directory (default: {DEFAULT_DIR})"
    )
    model_group = parser.add_mutually_exclusive_group()
    model_group.add_argument(
        "--model", help="new model id, e.g. provider/name — enables flag mode"
    )
    model_group.add_argument(
        "--list", action="store_true", help="print file -> model table and exit"
    )
    parser.add_argument(
        "--write", action="store_true", help="apply changes (default: dry-run preview)"
    )
    parser.add_argument(
        "--quiet", action="store_true", help="suppress skip notices and hints"
    )
    args = parser.parse_args(argv)
    if args.write and not args.model:
        parser.error("--write only makes sense together with --model")
    return args


def interactive_prompt(files: list[AgentFile]) -> tuple[str, bool]:
    """Ask which model to apply. Returns (new_model, confirmed)."""
    print("\nEnter new model id (e.g. opencode/x-preview-f-free): ", end="", flush=True)
    new_model = input().strip()
    while not new_model:
        print("A model id is required. Enter new model id: ", end="", flush=True)
        new_model = input().strip()
    same = [a.path.name for a in files if a.model == new_model]
    targets = [a for a in files if a.model != new_model]
    if not targets:
        print(f"\nAll files already use {new_model}. Nothing to do.")
        return new_model, True
    print(f"\nWill update {len(targets)} file(s):")
    for name in (a.path.name for a in targets):
        print(f"  - {name}")
    for name in same:
        print(f"  = {name} (already set)")
    answer = input("\nApply changes? [y/N] ").strip().lower()
    return new_model, answer in {"y", "yes"}


def _fail(message: str) -> None:
    print(f"error: {message}", file=sys.stderr)
    raise SystemExit(1)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv if argv is not None else sys.argv[1:])
    directory = Path(args.dir)
    if not directory.is_dir():
        print(f"error: not a directory: {directory}", file=sys.stderr)
        return 1

    files, skipped = scan_agents(directory)
    if not files:
        print(
            f"error: no agent file in {directory} has a frontmatter 'model:' field",
            file=sys.stderr,
        )
        return 1

    if args.list:
        print_table(files, skipped)
        return 0

    print(f"Scanned {directory} — {len(files)} agent file(s), {len(skipped)} skipped:")
    print_table(files, skipped)
    if skipped and not args.quiet:
        print("(skipped files have no frontmatter 'model:' field)")

    if args.model is None:
        new_model, confirmed = interactive_prompt(files)
        if not confirmed:
            print("Aborted — nothing written.")
            return 0
        write = True
    else:
        new_model = args.model
        write = args.write

    print(f"\n{'Applying' if write else 'DRY RUN'} — target model: {new_model}")
    changed = apply(files, new_model, write=write)
    verb = "changed" if write else "would change"
    print(
        f"\nDone: {verb} {changed}/{len(files)} file(s)."
        + ("" if write else " Re-run with --write to apply.")
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
