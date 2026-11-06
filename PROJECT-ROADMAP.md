# PROJECT-ROADMAP.md — sia-hu unified CLI

## Intent Statement

Create a unified `sia` CLI tool that aliases the existing `capture-proxy`, `model-headroom`, and `model-headroom-refusal` CLIs without modifying or removing them. The `sia` command is a thin dispatch layer: `sia <subcommand> <args>` forwards to the corresponding existing CLI's `main()` function. Existing CLIs remain byte-for-byte unchanged.

## Current State (Verified from codebase survey)

- **Single root project**: `sia-hu` 0.2.0, one `pyproject.toml`, one `uv.lock`, one ruff/mypy/pytest config at the repo root
- **Three console scripts** already defined in `[project.scripts]`:
  - `capture-proxy = "capture_proxy.__main__:main"`
  - `model-headroom = "model_headroom.cli:main"`
  - `model-headroom-refusal = "model_headroom.refusal.cli:main"`
- **Each `main()`** returns an `int` and only calls `sys.exit()` inside its `if __name__ == "__main__"` guard — safe to import and call directly from a new dispatcher
- **Packages live under `src/`**: `src/capture_proxy/`, `src/model_headroom/`
- **No unified `sia` entry point exists** — this is the gap
- **Verify chain**: `ruff check . && ruff format --check . && mypy src/ && pytest` (run from repo root)

## Gap

No `sia` command exists. Operators must remember three separate CLI names and their distinct flag sets. A unified `sia` entry point provides a single command surface that aliases all three.

## Planned Phases

### Phase 1: Add `sia` CLI package (CURRENT)
- **Goal**: Create `src/sia/` package with `cli.py` that uses argparse subparsers to dispatch `capture-proxy`, `model-headroom`, `model-headroom-refusal` to the existing `main()` functions
- **Success criterion**: `sia capture-proxy --help`, `sia model-headroom --help`, `sia model-headroom-refusal --help` each show the corresponding subcommand's help; `sia --help` lists all three subcommands; exit codes propagate
- **Scope**: New files only — `src/sia/__init__.py`, `src/sia/cli.py`. Edit `pyproject.toml` to add `sia` console script and `src/sia` to wheel packages. Zero changes to `src/capture_proxy/` or `src/model_headroom/`.

### Phase 2: Verify
- **Goal**: Run the full verify chain and confirm both package test suites still pass
- **Success criterion**: `ruff check . && ruff format --check . && mypy src/ && pytest` all exit 0; existing `capture-proxy`, `model-headroom`, `model-headroom-refusal` console scripts still work unchanged

## Scope-Creep Defenses

- IN scope: New `sia` package at root level; pyproject.toml entry point addition; dispatch to the three existing CLIs via direct `main()` calls
- OUT of scope: Modifying any existing CLI code; adding new flags or behavior to existing CLIs; removing or deprecating existing console scripts; adding subcommands beyond the three existing CLIs; changing the verify chain
- Adding a new subcommand requires explicit operator request (the three existing CLIs are the complete authorized set)

## Controlled-Evolution Path

- The `sia` CLI is a pure aliasing layer — it gains new subcommands only when the operator adds a new CLI to the project and explicitly requests it be surfaced through `sia`
- No behavioral changes to existing CLIs are ever made through this package

## Fringe Modules (working, not tested in core verify chain)

- `model-headroom-refusal` (refusal-direction extraction) is implemented and smoke-tested
  (Qwen3-0.6b: direction_status=computed, cohen_d=4.36, layer model.layers.16).
  It is a fringe module: not part of the core verify chain, not client-perspective tested,
  and parked while higher-priority work (release hygiene, carrier/authority contract) proceeds.
  Revisit only on explicit operator request.

## Drift Checkpoints

- After implementation: re-read intent — confirm `sia` only aliases, never modifies
- Before delivery: confirm all three existing console scripts are intact and unchanged
- Verify chain must pass in order: ruff → format → mypy → pytest

---
*Living document — update at each phase transition.*