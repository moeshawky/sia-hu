# File layout — directory structure of the repo

**Purpose:** Document where everything lives in the sia-hu repository. Cross-references
`guides/quickstart.md` (install), `operations/verify-chain.md` (verify chain), and
`architecture/authority-carrier.md` (which package lives where). No Python code
changes are proposed here.

---

## Repository root

```
sia-hu/
  README.md                  → project overview
  LICENSE                    → MIT
  PROJECT-ROADMAP.md         → unified `sia` CLI plan, phases 1-2
  pyproject.toml             → single project (sia-hu); hatchling build
  uv.lock                    → single lockfile at root
  bootstrap.sh               → clone + verify + push gate
  docs/                      → unified documentation tree (this tree)
  src/                       → source packages
  capture-proxy/             → carrier tests, roadmap (docs moved to docs/capture-proxy/)
  model_headroom/            → authority tests, roadmap, TEST_PROTOCOL.md (moved to docs/model-headroom/)
  tools/                     → repo tools
  .gitignore                 → unified ignore rules
  .python-version            → Python 3.13
```

---

## Source packages (under src/)

```
src/
  capture_proxy/             → carrier sidecar package
    __main__.py    → CLI entry point (config, daemon/stop/status, uvicorn)
    app.py         → FastAPI routes, handlers, streaming
    alias.py       → Model alias map
    taskclass.py   → Structural task classification
    blobs.py       → Content-addressed store
    schema.py      → schema.json manifest
    upstream.py    → httpx async proxy to upstream
    logger.py      → JSONL file writer (raw/ and errors/)
    headroom.py    → Opt-in headroom sidecar writer
    routing.py     → Opt-in routed-experts sidecar writer
    session.py     → session_id resolution
    runtime.py     → Per-backend process metadata
    stream.py      → SSE parser and streaming log writer
    models.py      → Pydantic schemas (CaptureRecord, HeadroomRef, …)
    names.py       → Shared name contract (stdlib only, R-01 safe)
  model_headroom/            → scoring authority package
    cli.py           → CLI entry point (19 flags)
    __main__.py      → Enables python -m model_headroom
    analysis/
      orchestrator.py  → Runs signals with timing, assembles HeadroomReport
      scoring.py     → Weighted composite score → [0.0, 1.0]
    models/
      types.py       → 21 frozen dataclasses
      loader.py      → HF weight extraction
      gguf_loader.py → GGUF weight extraction (optional extra)
    reporting/
      json_report.py → JSON output
      human_report.py → Terminal 78-col output (≤200 lines)
    signals/
      loss_saturation.py       → R-01
      growth_factor.py         → R-02
      effective_rank.py        → R-03
      grad_weight_ratio.py     → R-04
      dead_neurons.py          → R-05
      load_balance.py          → R-06
      specialization.py        → R-07
      gradient_coherence.py    → R-08 (diagnostic)
      gradient_effective_rank.py → R-09 (diagnostic)
  sia/                       → unified `sia` CLI dispatcher
```

---

## Test directories

```
capture-proxy/tests/         → carrier pytest tests (all upstream mocked)
model_headroom/tests/        → authority pytest tests
```

---

## Subrepo directories

```
capture-proxy/              → carrier subtree
  tests/                    → carrier pytest tests
  config/capture-proxy.toml → canonical default config
  scripts/                  → capture_query.py, verify_routed_experts.py
  (docs/README/roadmap moved to docs/capture-proxy/)

model_headroom/             → authority subtree
  tests/                    → authority pytest tests
  (docs/README/roadmap/TEST_PROTOCOL moved to docs/model-headroom/)
```

---

## Capture output layout (runtime)

```
{CAPTURE_BASE_PATH}/                         → default /root/captures
  raw/YYYY-MM-DD/*.jsonl       → successful captures
  errors/YYYY-MM-DD/*.jsonl    → failed captures
  blobs/sha256/{dd}/{digest}   → content-addressed request/message bodies
  headroom/YYYY-MM-DD/{capture_id}.headroom.json   → headroom sidecar
  headroom/YYYY-MM-DD/{capture_id}-weight_pairs.json → weight pairs
  routing/YYYY-MM-DD/{capture_id}.routing.json     → routed-experts sidecar
  index/sessions/*.jsonl       → per-session capture index
  index/users/*.jsonl          → per-user capture index
  schema.json                  → schema manifest (written at startup)
```

---

## Cross-references

- `guides/quickstart.md` — install and bootstrap
- `operations/verify-chain.md` — verify chain (ruff → format → mypy → pytest)
- `architecture/authority-carrier.md` — which package does what