# Authority vs carrier — the division of labor

**Purpose:** Define exactly which repository does what, which signals belong to
which side, and why the division must never invert. Cross-references
`architecture/sia-hu-identity.md` (the Sia/Hu cosmology) and
`architecture/contract.md` (the bidirectional contract). No Python code changes are
proposed here.

---

## The rule

> **AUTHORITY DEFERS TO model_headroom ON ANY SCORE DISPUTE**
> **CARRIER NEVER SCORES — capture-proxy EMITS, model_headroom JUDGES**
> **VERIFY BEFORE CLAIMING DONE — ruff check . && ruff format --check . && mypy src/ && pytest**

This is the operational Iron Law of sia-hu. It is not a preference; it is the
invariant that keeps the two packages from producing conflicting
measurements.

---

## model_headroom = AUTHORITY = scores Sia

**Role:** Reads the hidden internal form (model weights) and produces a headroom
score.

**What it owns:**
- 21 frozen dataclasses in `models/types.py` — all input/output/error type contracts
- 9 signal modules (7 core + 2 diagnostic) in `signals/`
- `analysis/orchestrator.py` — runs all signals with timing, assembles `HeadroomReport`
- `analysis/scoring.py` — weighted composite score → [0.0, 1.0]
- `reporting/json_report.py` and `reporting/human_report.py` — output formats
- `models/loader.py` and `models/gguf_loader.py` — weight extraction (device resolved via `device.py`: accelerator preferred, CPU fallback)
- CLI entry `cli.py` — 20 flags, `--analysis-mode decision_grade` (exit 2 if <70% measured coverage)

**Signal ownership (authority side):**

| Signal | Name | Scope | Measures |
|--------|------|-------|----------|
| R-01 | Loss Saturation | Universal | Training loss plateau detection |
| R-02 | Growth Factor | Universal | Quadratic loss fit, crossover detection |
| R-03 | Effective Rank | Universal | SVD-based spectral entropy of weight matrices |
| R-04 | Gradient/Weight Ratio | Universal | Norm ratio, z-score computation |
| R-05 | Dead Neurons | Universal | Near-zero activation channel detection |
| R-06 | Load Balance | MoE only | Router entropy / balance ratio |
| R-07 | Specialization | MoE only | Pairwise expert cosine similarity |
| R-08 | Gradient Coherence | Diagnostic (opt-in) | Cross-batch gradient cosine similarity |
| R-09 | Gradient Effective Rank | Diagnostic (opt-in) | SVD rank analysis of gradient matrices |

On dense models, R-06 and R-07 are marked `not_applicable` and excluded from the
composite score. Skipped signals score 0.5 (neutral). Errors score 0.0. R-08/R-09
never affect the headroom score.

**What it never does:**
- Never proxies, never forwards HTTP, never runs a server
- Never imports torch on the carrier path (R-01 purity)
- Never emits capture artifacts — that is the carrier's job

---

## capture-proxy = CARRIER = emits Hu

**Role:** Runs an OpenAI-compatible proxy that logs every request/response pair
before forwarding to the upstream backend. Emits capture artifacts in the format the
authority consumes.

**What it owns:**
- `src/capture_proxy/app.py` — FastAPI routes: `/health`, `/v1/models`,
  `/v1/chat/completions`, `/v1/completions`, `/v1/embeddings`, `/v1/rerank`,
  `/v1/classification`, `/v1/images/generations`, `/v1/sessions*`, `/v1/users*`
- `src/capture_proxy/logger.py` — JSONL writer for `raw/` and `errors/` directories
- `src/capture_proxy/headroom.py` — opt-in headroom sidecar writer (per-capture JSON +
  weight-pair files), post-response, async, never awaited on the response path
- `src/capture_proxy/taskclass.py` — structural task classification (shape, never
  vocabulary)
- `src/capture_proxy/names.py` — shared name contract: signal vocabulary, CLI-to-emitted
  name map, tensor-name templates, MoE model-type set (stdlib only, R-01 safe)
- `src/capture_proxy/models.py` — Pydantic schemas: `CaptureRecord`, `HeadroomRef`,
  `CapturedHeadroom`, request/response models
- `src/capture_proxy/upstream.py` — httpx async proxy to upstream backend
- `src/capture_proxy/__main__.py` — CLI entry point: daemon, stop, status, config

**Eligible sidecar request types (headroom capture):** `chat`, `completion`,
`classification`, `image`. **Excluded:** `embeddings`, `rerank` — neither handler
calls `_spawn_headroom` and the request-type filter rejects them.

**What it never does:**
- Never scores — no composite, no signal computation, no headroom score
- Never imports torch/safetensors unless `--headroom` is enabled (off by default)
- Never alters the proxied response (returns upstream status and body unchanged)

---

## Signal flow across the boundary

```
capture-proxy (carrier, emits Hu)
  │
  │ emits: {capture_id}.headroom.json  +  {capture_id}-weight_pairs.json
  │          (loader format the authority consumes)
  ▼
model_headroom (authority, scores Sia)
  │
  │ consumes weight pairs, computes per-signal scores,
  │   weighted composite → [0.0, 1.0]
  ▼
Report: human (78-col, ≤200 lines) or JSON (full HeadroomReport)
```

---

## Cross-references

- `architecture/sia-hu-identity.md` — Sia/Hu cosmology, why the mapping is 1-to-1
- `architecture/contract.md` — bidirectional contract, 2-D pair contract, reserved keys
- `concepts/headroom-signals.md` — R-01 through R-09 signal reference
- `concepts/frozen-contracts.md` — frozen dataclasses, invariants, NFR-01
- `guides/capture-proxy.md` — carrier usage guide
- `guides/model-headroom.md` — authority usage guide
- `reference/cli-flags.md` — all CLI flags across both packages plus `sia`
- `reference/env-vars.md` — all env vars
