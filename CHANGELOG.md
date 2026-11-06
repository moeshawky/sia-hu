# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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