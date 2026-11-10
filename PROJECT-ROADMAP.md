# PROJECT-ROADMAP.md — sia-hu unified CLI

## Intent Statement

Provide one operator surface, `sia`, over the three package CLIs — `capture-proxy`, `model-headroom`, and `model-headroom-refusal` — and over the workspace operations that were previously carried by shell scripts.

The aliasing contract is unchanged: `sia <subcommand> <args>` forwards to the matching existing CLI's `main()` and propagates its exit code. Two deliberate expansions are authorized by the operator (2026-11-10), because the shell scripts that carried these capabilities were removed on purpose and must not be restored:

- **`capture-proxy --sync`** — native off-box capture sync, replacing `rclone-sync-captures.sh` and the sync step of `bootstrap-capture.sh`. Every other capability of `bootstrap-capture.sh` already exists as a native flag (`--ensure-config`, `--daemon`, `--status`, `--stop`, `--print-effective-config`).
- **`sia doctor` and verified publish** — native workspace preflight and push verification, replacing `bootstrap.sh`. The load-bearing capability there is refusing to trust `git push` exit 0: the remote ref is compared to local HEAD.

`capture-proxy` is therefore no longer byte-for-byte unchanged. That is a recorded decision, not drift: the removed scripts are the specification, and the native flags are their replacement.

## Current State (Verified from codebase survey)

- **Single root project**: `sia-hu` 0.3.0, one `pyproject.toml`, one `uv.lock`, one ruff/mypy/pytest config at the repo root
- **Three console scripts** already defined in `[project.scripts]`:
  - `capture-proxy = "capture_proxy.__main__:main"`
  - `model-headroom = "model_headroom.cli:main"`
  - `model-headroom-refusal = "model_headroom.refusal.cli:main"`
- **Each `main()`** returns an `int` and only calls `sys.exit()` inside its `if __name__ == "__main__"` guard — safe to import and call directly from a new dispatcher
- **Packages live under `src/`**: `src/capture_proxy/`, `src/model_headroom/`, `src/sia/`
- **The `sia` dispatcher exists** (Phase 1) and currently adds no verbs of its own
- **Verify chain**: `ruff check . && ruff format --check . && mypy src/ && pytest` (run from repo root)

## Gap

Measured 2026-11-10 against `master-archive` (the branch that still holds the scripts) and against the current `src/` tree:

- `bootstrap-capture.sh` (115 lines): its config, daemon, status, and effective-config steps are native; its sync step is not.
- `rclone-sync-captures.sh` (191 lines): **no native equivalent anywhere in `src/`**. `rg rclone` hits only `AGENTS.md`.
- `bootstrap.sh` (330 lines): **no native equivalent**. Its unique load-bearing piece is push verification.
- `sia` has **no tests**: `testpaths` covers only `capture-proxy/tests` and `model_headroom/tests`, and no test file references `sia`.

## Planned Phases

### Phase 1: Add `sia` CLI package (COMPLETE)
- **Goal**: Create `src/sia/` package with `cli.py` that uses argparse subparsers to dispatch `capture-proxy`, `model-headroom`, `model-headroom-refusal` to the existing `main()` functions
- **Success criterion**: `sia capture-proxy --help`, `sia model-headroom --help`, `sia model-headroom-refusal --help` each show the corresponding subcommand's help; `sia --help` lists all three subcommands; exit codes propagate
- **Scope**: New files only — `src/sia/__init__.py`, `src/sia/cli.py`. Edit `pyproject.toml` to add `sia` console script and `src/sia` to wheel packages. Zero changes to `src/capture_proxy/` or `src/model_headroom/`.

### Phase 2: Verify (COMPLETE)
- **Goal**: Run the full verify chain and confirm both package test suites still pass
- **Success criterion**: `ruff check . && ruff format --check . && mypy src/ && pytest` all exit 0; existing `capture-proxy`, `model-headroom`, `model-headroom-refusal` console scripts still work unchanged

### Phase 3: Native `capture-proxy --sync` (AUTHORIZED — not started)
- **Goal**: Replace `rclone-sync-captures.sh` with a native sync mode, so off-box capture sync is a property of the installed binary rather than of a shell file that no longer exists
- **Success criterion**: one-shot sync and daemon loop both work against an rclone remote; re-running the daemon start is idempotent; a missing rclone produces an instructive error naming the flag, the expected value, and a valid example
- **Constraints**: `copy` is the default; `sync` (destructive mirror) is refused unless the instance id is explicit; instance keying preserves `CAPTURE_MACHINE_ID` > hostname > `/etc/machine-id`; `--min-age` skips files still being written; sync never touches the response path; no new Python dependency
- **Scope**: new `src/capture_proxy/sync.py`, `__main__.py` wiring, `_CONFIG_KEYS` additions, tests under `capture-proxy/tests/`

### Phase 4: `sia doctor` + verified publish (AUTHORIZED — not started)
- **Goal**: Replace `bootstrap.sh` with native workspace verbs: a read-only preflight, and a publish that refuses to trust `git push` exit 0
- **Success criterion**: `sia doctor` changes nothing and reports every failed check with the command that fixes it; `sia publish` pushes the current branch and fails loudly when the remote ref does not match local HEAD
- **Constraints**: doctor is read-only; publish never auto-commits; a dirty tree is refused unless explicitly overridden; credential-bearing URLs are redacted in every message; `sia` keeps lazy imports so `sia --help` pulls no heavy dependency
- **Scope**: `src/sia/` native verbs, tests (new `tests/` path or an explicit `testpaths` addition), docs

## Scope-Creep Defenses

- IN scope (authorized 2026-11-10): `capture-proxy --sync`; `sia doctor`; `sia` verified publish; the tests and docs that cover them
- OUT of scope: restoring any removed shell script; changing the existing behavior of the three aliased CLIs; adding scoring to the carrier; any sync backend other than rclone; auto-commit in publish
- The earlier "no new flags, no new subcommands" rule is **superseded only for the two items above**. Any further expansion needs a new explicit operator request recorded in this file first

## Controlled-Evolution Path

- `sia` remains an aliasing layer for the three package CLIs, plus the two native workspace verbs recorded in Phase 4
- `capture-proxy` gains `--sync` only; no other behavior changes
- A new subcommand or flag requires an explicit operator request recorded in this file before implementation

## Fringe Modules (working, not tested in core verify chain)

- `model-headroom-refusal` (refusal-direction extraction) is implemented and smoke-tested
  (Qwen3-0.6b: direction_status=computed, cohen_d=4.36, layer model.layers.16).
  It is a fringe module: not part of the core verify chain, not client-perspective tested,
  and parked while higher-priority work (release hygiene, carrier/authority contract) proceeds.
  Revisit only on explicit operator request.

## Drift Checkpoints

- After implementation: re-read intent — confirm no removed shell script was restored, and no scoring entered the carrier
- Before delivery: confirm all three existing console scripts still work unchanged for their existing flags
- Verify chain must pass in order: ruff → format → mypy → pytest

---
*Living document — update at each phase transition.*