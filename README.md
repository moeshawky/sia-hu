# sia-hu — Sia + Hu

**Sia** (sỉꜣ): divine perception, deep insight, interior knowledge — perceiving what
is hidden inside a closed form BEFORE it is spoken (the pre-verbal activation state,
latent internal representation). **Hu** (ḥw): the uttered word, spoken output, token
generation.

A logit lens that reads Sia before it becomes Hu is a 1-to-1 conceptual match:

| Egyptian concept | sia-hu role | Repository | What it does |
|------------------|-------------|------------|--------------|
| **Sia** — hidden perception | AUTHORITY — scores | `model_headroom/` | Reads model weights (the hidden form) and computes a headroom score in [0.0, 1.0]. Nine signals inspect latent structure: weight spectra, activation balance, expert routing, gradient ratios. |
| **Hu** — uttered word | CARRIER — emits | `capture-proxy/` | Runs an OpenAI-compatible proxy that logs every request/response before forwarding. Emitted JSONL records, sidecar `headroom.json`, and `weight_pairs.json` are the uttered output — token generation made observable. |
| The bridge | CLI | `sia` | `sia <subcommand>` forwards to the corresponding existing CLI. Neither scores nor emits itself — it bridges Sia and Hu. |

**Contract (never inverted):**
- Authority scores (`model_headroom`); carrier never scores (`capture-proxy`).
- Proxy never scores; authority never proxies.
- Proxy → authority: emits `{capture_id}.headroom.json` +
  `{capture_id}-weight_pairs.json` in the loader format the authority consumes.
- Authority → proxy: defines the 2-D pair contract and signal semantics; the proxy
  never invents a signal meaning.
- Agreement between the authority and carrier implementations is the cross-check — one
  voice is an echo.

See `docs/architecture/sia-hu-identity.md` for the full cosmology, and
`docs/index.md` for the unified documentation tree.

---

## One project, two packages

| Package | Role | Rule |
|---------|------|------|
| `src/model_headroom/` | **Authority** — scores training headroom (reads Sia) | Never proxies |
| `src/capture_proxy/` | **Carrier** — emits sidecar capture (emits Hu) | Never scores |

Both are **tracked packages inside this single repo** — not submodules, not gitignored, not separate remotes. One clone carries the whole workspace: single `pyproject.toml`, single `uv.lock`, single ruff/mypy/pytest config, installed editable from the repo root, exposing `capture-proxy`, `model-headroom`, `model-headroom-refusal`, and `sia`.

`model_headroom` is the sole scoring authority; `capture-proxy` emits artifacts in the format the authority consumes. Agreement between the two is the cross-check.

---

## The unified `sia` CLI

`sia <subcommand> <args>` forwards to the corresponding existing CLI's `main()`:

| `sia` subcommand | Forwards to | Role |
|------------------|-------------|------|
| `sia capture-proxy` | `capture-proxy` | Carrier — emits Hu |
| `sia model-headroom` | `model-headroom` | Authority — scores Sia |
| `sia model-headroom-refusal` | `model-headroom-refusal` | Refusal analysis |

Each `main()` returns an `int` and only calls `sys.exit()` inside its
`if __name__ == "__main__"` guard. Exit codes propagate. `sia` adds no flags,
no behavior, no new subcommands beyond the three it aliases.

See `docs/guides/sia-cli.md` and `docs/reference/cli-flags.md`.

---

## Quick start

```bash
pip install -e .                # carrier package only
pip install -e ".[headroom]"     # + torch / numpy / safetensors (scoring)
pip install -e ".[dev]"         # + pytest / mypy / ruff
```

Verify: `ruff check . && ruff format --check . && mypy src/ && pytest`

Full install, bootstrap, and verification steps: `docs/guides/quickstart.md`.
Verify chain details: `docs/operations/verify-chain.md`.

---

## Repository layout

```
sia-hu/
  README.md            → this file
  AGENTS.md            → agent/operator gate (gitignored, on disk)
  PROJECT-ROADMAP.md   → unified `sia` CLI plan, phases 1-2
  pyproject.toml       → single project (sia-hu); one lockfile
  bootstrap.sh         → clone + verify + push gate
  src/                 → source packages (capture_proxy, model_headroom, sia)
  capture-proxy/       → carrier tests, config, scripts (docs moved to docs/)
  model_headroom/      → authority tests (docs moved to docs/)
  docs/                → unified documentation tree
  tools/               → repo tools
```

## Documentation

Unified docs live in `docs/` at the repo root. Start with
[`docs/index.md`](docs/index.md).

| Section | Purpose |
|---------|---------|
| `docs/architecture/` | Sia/Hu identity, authority/carrier division, bidirectional contract |
| `docs/concepts/` | Headroom signals (R-01–R-09), provenance, frozen contracts |
| `docs/guides/` | Quickstart, `sia` CLI, carrier guide, authority guide |
| `docs/operations/` | Verify chain, dev commands, environment |
| `docs/reference/` | CLI flags, env vars, file layout |
| `docs/capture-proxy/` | Carrier README, roadmap, design notes (moved from `capture-proxy/`) |
| `docs/model-headroom/` | Authority README, roadmap, test protocol (moved from `model_headroom/`) |

Authority and carrier roadmaps, test protocol, and package READMEs now live
under `docs/` (see `docs/reference/file-layout.md`).

Key docs:
- `docs/architecture/sia-hu-identity.md` — cosmology in detail
- `docs/architecture/authority-carrier.md` — who scores, who emits
- `docs/architecture/contract.md` — bidirectional contract, 2-D pair contract
- `docs/concepts/headroom-signals.md` — R-01 through R-09 signal reference
- `docs/guides/quickstart.md` — install, bootstrap, verify
- `docs/guides/capture-proxy.md` — carrier usage
- `docs/guides/model-headroom.md` — authority usage
- `docs/operations/verify-chain.md` — ruff → format → mypy → pytest
- `docs/reference/cli-flags.md` — all CLI flags
- `docs/reference/env-vars.md` — all env vars

---

## License

MIT — see [`LICENSE`](LICENSE).