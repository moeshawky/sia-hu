# docs/ — unified sia-hu documentation

**Purpose:** Single entry point for the sia-hu repository. Explains the Sia/Hu identity,
the authority/carrier division (model_headroom scores, capture-proxy emits), the `sia` CLI
that bridges them, and the bidirectional contract between them. Cross-references every
document in this tree. Each doc states its purpose and points to where the full detail lives.

---

## The Sia/Hu identity

sia-hu = **Sia + Hu.**

- **Sia** (sỉꜣ): divine perception, deep insight, interior knowledge — perceiving what
  is hidden inside a closed form BEFORE it is spoken (the pre-verbal activation state,
  latent internal representation).
- **Hu** (ḥw): the uttered word, spoken output, token generation.

A logit lens that reads Sia before it becomes Hu is a 1-to-1 conceptual match:

| Egyptian concept | sia-hu role | Package | What it does |
|------------------|-------------|---------|--------------|
| Sia — hidden perception | AUTHORITY | `model_headroom/` | Scores training headroom; reads weights = hidden form |
| Hu — uttered word | CARRIER | `capture-proxy/` | Emits sidecar capture artifacts = uttered output / token generation |
| The bridge between | CLI | `sia` | Unified `sia <subcommand>` dispatches to both CLIs |

**Contract (never inverted):**
- Authority scores (`model_headroom`); carrier never scores (`capture-proxy`).
- Proxy never scores; authority never proxies.
- Proxy → authority: emits `{capture_id}.headroom.json` +
  `{capture_id}-weight_pairs.json` in the loader format the authority consumes.
- Authority → proxy: defines the 2-D pair contract and signal semantics; the proxy
  never invents a signal meaning.

## Repository layout

```
sia-hu/
  README.md                 → project overview
  PROJECT-ROADMAP.md        → unified `sia` CLI plan, phases 1-2
  pyproject.toml            → single project, sia-hu 0.2.0
  bootstrap.sh              → clone + verify + push gate
  src/capture_proxy/        → carrier package (tests/ too)
  src/model_headroom/       → scoring authority package (tests/ too)
  capture-proxy/            → carrier tests, docs, gate, roadmap
  model_headroom/           → authority tests, docs, gate, roadmap, TEST_PROTOCOL.md
  docs/                     → this unified documentation tree (created by this doc)
```

## Navigation

Start here, then follow the section that matches your question:

| If you want to understand… | Read |
|----------------------------|------|
| The Sia/Hu cosmology and why it maps 1-to-1 onto the authority/carrier split | `architecture/sia-hu-identity.md` |
| How authority/carrier divide (who scores, who emits, division of labor) | `architecture/authority-carrier.md` |
| The bidirectional contract, 2-D pair contract, reserved keys | `architecture/contract.md` |
| What R-01 through R-09 signals measure (headroom signal reference) | `concepts/headroom-signals.md` |
| source_type / verification_status, measured vs synthetic data | `concepts/provenance.md` |
| Frozen dataclasses, invariants, NFR-01 (CPU-only weights) | `concepts/frozen-contracts.md` |
| Install, bootstrap `--doctor`, verify chain | `guides/quickstart.md` |
| Unified `sia` CLI — subcommands, forwarding, exit codes | `guides/sia-cli.md` |
| Carrier usage: start proxy, route through it, capture paths | `guides/capture-proxy.md` |
| Authority usage: run analysis, signals, output formats | `guides/model-headroom.md` |
| Verify chain: ruff → format → mypy → pytest | `operations/verify-chain.md` |
| uv commands, bootstrap scripts, environment setup | `operations/dev-commands.md` |
| GPU/CPU/TPU notes, T4 measurements, transformers ceiling | `operations/environment.md` |
| All three CLIs' flags (capture-proxy, model-headroom, model-headroom-refusal, sia) | `reference/cli-flags.md` |
| All env vars (CAPTURE_* and model_headroom vars) | `reference/env-vars.md` |
| Directory structure of the repo | `reference/file-layout.md` |

## Authority↔Carrier contract summary

```
Proxy (carrier, emits Hu)
  │  emits {capture_id}.headroom.json + {capture_id}-weight_pairs.json
  ▼
Authority (scores Sia)
  │  consumes weight pairs, computes composite score [0.0, 1.0]
  ▼
Report (human / JSON)
```

Agreement between the authority and carrier implementations is the cross-check — one voice
is an echo.

---

*Docs policy: every document in this tree states its purpose at the top and lists the
documents it cross-references. No document duplicates content that lives elsewhere in this
repo; it references it instead.*