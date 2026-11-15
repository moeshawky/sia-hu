# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.5.0] - 2026-11-15

### Added
- `model-headroom-paired` CLI (`model_headroom.paired.cli:main`) — paired-model recovery diagnostics: `compare` (two measurement documents → per-layer differences), `decide` (intervention evidence → per-strategy verdict: SUPPORTED / REFUTED / INCONCLUSIVE), `verify` (re-hash a persisted artifact against its evidence-store index), and `schema-version`. Four subcommands, each leaving a durable artifact; importing the module leaves torch and numpy out of sys.modules. Wired into the `sia` dispatcher as `sia model-headroom-paired`.
- New `src/model_headroom/paired/` package: `schema.py` (tri-state MeasurementDocument with `measured`/`not_measured`/`not_applicable`, TensorKind distinguishing `packed_buffer` from `effective_weight`), `compare.py` (layer-pairing by `layer_name`, never position), `decision.py` (coverage ≥ 0.8 required for SUPPORTED; zero delta and in-sample deltas yield INCONCLUSIVE), `instrument.py` (PairedActivationRecorder with byte budget, no GPU memory allocated on import), `refusal_compare.py` (response-level refusal outcomes and direction-level geometry kept separate), `evidence_store` integration (SHA-256 artifact verification). New console script `model-headroom-paired` in `[project.scripts]`.
- Refusal runtime-ablation artifact and hook (`model-headroom-refusal`): `refusal/artifact.py` writes a safetensors artifact carrying the unit `refusal_direction` plus optional `domain_direction` with 20 string metadata keys; `refusal/runtime.py` loads the artifact and installs a per-forward projection hook on the residual writers (`DEFAULT_TARGET_SUFFIXES` / `EXTENDED_TARGET_SUFFIXES`), reusing `extract.project_out` as the single owner of the projection math. Lambda lives in a persistent 0-dim device tensor so a captured CUDA graph reads it at replay; `set_strength` on a no-op (lambda == 0) handle raises by design. Verified bit-exact at lambda 0 (`torch.equal`) and not part of the headroom score.
- New safetensors sampling path (`src/model_headroom/models/safetensors_sampling.py`): bounded deterministic sampling of safetensors weight tensors for the R-03 effective-rank analyzer — one tensor read at a time, full column dimension preserved, row count capped, multi-shard checkpoints never materialised in RAM. Reads `model.safetensors.index.json` when present, falls back to loose shards. Weight evidence only — no gradient is manufactured.
- `sia doctor` — read-only workspace preflight (git on PATH, cwd not inside a package dir, repo root/branch, origin remote, root `pyproject.toml` + the three `src/<pkg>/__init__.py` files + wheel packages list, capture-proxy config TOML validity, verify-chain tool availability probed but never executed). Every error names its fix command; exit 0/1/2 (2 on flag errors). Never mutates.
- `sia publish` — push current branch and verify the remote ref matches local HEAD (dirty-tree refusal overridable with `--allow-dirty`, which never commits or force-pushes; detached-HEAD refusal; credential-bearing origin URLs redacted; `--dry-run` skips push and verification). Never auto-commits.
- `capture-proxy --sync` native surface (replacing the removed `rclone-sync-captures.sh`): `--sync` (one-shot), `--sync-daemon`, `--sync-stop`, `--sync-status`, plus `--sync-remote`, `--sync-cmd {copy,sync}`, `--sync-min-age`, `--sync-interval`, `--sync-dry-run`, `--sync-rclone-opts`, `--sync-rclone-binary`, `--sync-machine-id`, `--sync-pid-file`, `--sync-log-file`. `copy` is the default; destructive `sync` requires an explicit instance id. No new runtime Python dependency.

### Changed
- Version bumped `0.4.0` → `0.5.0` (`pyproject.toml`).
- Dead-neuron criterion (`R-05`) now separates inactive from negative-mean channels: a neuron is dead/inactive only when its maximum absolute activation is exactly 0 (never fires), so signed-but-active channels are no longer misclassified as dead. Dead/inactive, negative-mean, and never-active channels tracked as separate statistics.
- Honesty fixes: the carrier no longer fabricates a zero gradient it never observed — weight-only pairs emit `gradient: null` with `gradient_status: "not_captured"`, and `grad_weight_ratio` reports `no_gradient` with a null ratio. Missing gradients stay missing; absence is not scored as evidence of spare training capacity. Zero-gradient claims in headroom contract docs corrected (superseded ADR-002 behaviour).
- Accelerator-preferred device selection (NFR-01 overturned): accelerator preferred (cuda → xpu → mps → cpu), CPU as fallback; resolved device recorded in report metadata. Loader failures now name the cause, the flag, the expected shape, and where a valid example lives. New `tools/engine_probe.py` capability probe.

### Removed
- `rclone-sync-captures.sh`, `bootstrap-capture.sh`, `bootstrap.sh` shell scripts — replaced by native `capture-proxy --sync*` flags and `sia doctor`/`sia publish` verbs.


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