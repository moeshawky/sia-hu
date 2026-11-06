# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Package metadata: PEP 639 `license = "MIT"` (emitted as `License-Expression: MIT`
  in Metadata 2.4), authors, keywords, classifiers, and `project.urls`;
  MIT `LICENSE` file at the repo root.
- uv cache moved from the in-repo `.uv-cache/` to the sibling `../.uv-cache/`
  so the cache stays on the same filesystem as `.venv` (hardlinks) while
  sitting outside the build source.
- `CHANGELOG.md` (this file).

## [0.2.0] - 2026-10-17

### Added

- OpenAI-compatible capture proxy for llama.cpp, ollama, node-llama-cpp, vLLM,
  and SGLang (`CAPTURE_BACKEND`; SGLang explicit-only, not auto-detected) that
  logs raw JSONL of all requests/responses before forwarding.
- Persistent configuration in `~/.config/capture-proxy/config.toml` with
  CLI flag > env var > config file > default resolution, plus the
  `--ensure-config` and `--print-effective-config` provisioning flags.
- Daemon operation via `--daemon` / `--status` / `--stop`, plus
  `--ensure-config` and `--print-effective-config` provisioning flags.
- Opt-in headroom sidecar (default off): post-response, never-awaited
  emission of weight pairs, expert activation metadata, expert weights, and a
  routed-experts sidecar in the `model_headroom` loader format, with interval
  sampling and byte caps.
- Headroom capture for dense/BERT-style models (single-expert reporting,
  no fabricated expert counts).
- Lossless streaming capture with the stream-body invariant pinned so a
  streamed body stays verifiable end to end.
- Self-describing captures: per-run schema manifest (version, artifact
  layout, task vocabulary, index key) with tri-state field classes.
