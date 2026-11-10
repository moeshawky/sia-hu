# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.4.0] - 2026-11-10

### Added
- `capture-proxy --sync` (one-shot), `--sync-daemon`, `--sync-stop`, `--sync-status` — native off-box capture sync via rclone subprocess; no new runtime Python dependency (pure stdlib), no restored shell scripts. New optional `env` extra (`wrapt>=1.16`) for Kaggle self-healing; not in `[project].dependencies`. Settings flags: `--sync-remote`, `--sync-cmd` (default `copy`), `--sync-min-age` (default `60s`), `--sync-interval` (default `300`), `--sync-dry-run`, `--sync-rclone-opts`, `--sync-rclone-binary`, `--sync-machine-id`, `--sync-pid-file`, `--sync-log-file`.
- `sia doctor` — read-only workspace preflight; every failure names its fix command; exit 0/1/2 (2 on flag errors).
- `sia publish` — push current branch and verify the remote ref matches local HEAD; never auto-commits; `--dry-run` / `--allow-dirty`; credential URLs redacted in output.

### Changed
- Version bumped `0.3.0` → `0.4.0` (`pyproject.toml`).
- `pytest testpaths` extended to include root `tests/`; `sia` added to coverage `source`.

### Removed
- `rclone-sync-captures.sh`, `bootstrap-capture.sh`, `bootstrap.sh` shell scripts — replaced by native `capture-proxy --sync*` flags and `sia doctor`/`sia publish` verbs.

## [0.3.0] - 2026-11-10

### Added
- `--device {auto,cuda,cpu}` on `model-headroom` (default `auto`): accelerator-preferred device selection (cuda → xpu → mps → cpu), resolving through new `src/model_headroom/device.py`. `auto` picks the first available accelerator else CPU; `--device cuda` on a box with no CUDA exits 1 with an informative error, never silently downgrades. The resolved device is recorded in `report.metadata["device"]`.
- New public module `src/model_headroom/device.py` (`available_accelerators()`, `default_device()`, `resolve_device(requested) -> str | None`) — the single source of truth for device selection, threaded into `run_analysis(..., device=...)` and into the GGUF weight-sampling path.
- New `tools/engine_probe.py` capability probe: workspace shim importing neither package and no torch; `probe_engines()` returns a JSON manifest of installed/running serving engines with capability set (`chat_completions`, `embeddings`, `rerank`, `routed_experts`, `hidden_states`), measured hardware, and hard limits. Never raises, never starts/stops a server, never binds a port; identifies an engine by its own `/v1/models` metadata.
- New `env` optional-dependency extra (`wrapt>=1.16`): self-healing on Kaggle-class hosts whose system `sitecustomize.py` imports `wrapt`. Deliberately NOT in `[project].dependencies` — sia-hu never imports wrapt, and a published wheel must not ship a bogus runtime dependency.
- New "Engine Selection Policy" section separating HARD constraints (routed_experts and hidden-state capture require SGLang; measurement signals cannot be served and need in-process forward passes) from PREFERENCES (llama.cpp for GGUF/low-VRAM/FastMTP, vLLM for dense throughput and `/v1/rerank`).

### Changed
- NFR-01 rewritten: previously mandated CPU-only weight loading; now mandates accelerator-preferred with CPU fallback. The 21 frozen dataclasses were not touched.
- `model-headroom-refusal --device` default changed from `cpu` to `auto`: an unflagged refusal run now selects the first available accelerator (cuda → xpu → mps → cpu) instead of silently capturing on CPU; `--device cuda` on a box with no CUDA exits 1, `--device cpu` stays on CPU. `src/model_headroom/refusal/cli.py:158-163`.
- `recover_direction_from_weight_deltas(deltas, spec, device="cpu")` now runs `torch.linalg.svd` on the resolved device and returns the direction on CPU so the serialized report stays device-independent. `src/model_headroom/refusal/recover.py:34-38,103-119,149`; wired at `src/model_headroom/refusal/cli.py:417`.
- The SVD failure message now names the device and suggests `--device cpu`: `SVD failed on device {device}: {exc}; retry with --device cpu`. `src/model_headroom/refusal/recover.py:128`.

### Fixed
- Loader error output now reports the specific cause (file does not exist / JSON malformed / missing required keys with the found keys / tensor is 3-D but must be 2-D / empty-but-valid), the offending path, the flag to use, the expected JSON shape, and where a valid example lives — instead of discarding the cause. New `LoadOutcome` + `*_diagnostic` loader functions in `src/model_headroom/models/loader.py`; existing loaders keep their documented `None`-returning contracts.

## [0.2.0] - 2026-11-06

### Added
- Unified `sia` CLI dispatcher aliasing `capture-proxy`, `model-headroom`, and `model-headroom-refusal`.
- Unified single project layout (`src/`, one `pyproject.toml`, one `uv.lock`).
- MIT license.
- CI workflows for testing and PyPI publishing.

### Changed
- Raised `requires-python` to `>=3.12` (numpy 2.5.4 requirement).
- Bumped all dependencies to latest PyPI versions.
- Ruff target version `py310` → `py312`.

### Fixed
- Removed engineering-process artifacts from published tree (AGENTS.md, agent definitions, tickets, session artifacts, dev tooling configs).