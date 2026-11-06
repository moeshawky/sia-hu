# Carrier usage guide — capture-proxy

**Purpose:** How to use capture-proxy (the carrier, emits Hu): start the proxy,
route traffic through it, understand capture paths and sidecar emission.
Cross-references `architecture/authority-carrier.md` (carrier role),
`architecture/contract.md` (what the carrier emits), `concepts/headroom-signals.md`
(signal keys in sidecars), and `reference/env-vars.md` (all `CAPTURE_*` vars).
No Python code changes are proposed here.

---

## What capture-proxy is

Capture proxy for llama.cpp, ollama, node-llama-cpp, vLLM, and SGLang — logs raw
JSONL of all requests/responses before forwarding. Exposes an OpenAI-compatible API
so any agent using OpenAI client libraries can route through it transparently.

**Version:** 0.2.0 — the carrier half of sia-hu. Never scores; always emits.

---

## Start the proxy

```bash
# Default upstream (ollama on :11434)
CAPTURE_UPSTREAM_URL=http://127.0.0.1:11434 uv run capture-proxy --port 8000

# Explicit backend
CAPTURE_UPSTREAM_URL=http://localhost:3000 CAPTURE_BACKEND=node-llama-cpp \
  uv run capture-proxy --port 8000

# vLLM (port 8000 not auto-detected — pick a different proxy port)
CAPTURE_UPSTREAM_URL=http://localhost:8000 CAPTURE_BACKEND=vllm \
  uv run capture-proxy --port 8001

# SGLang (port 30000 not auto-detected — explicit backend required)
CAPTURE_UPSTREAM_URL=http://127.0.0.1:31000 CAPTURE_BACKEND=sglang \
  uv run capture-proxy --port 8001
```

Settings resolve in order: **CLI flag > env var > config file > default.** Persist in
`~/.config/capture-proxy/config.toml`.

### Background service

```bash
capture-proxy --daemon      # start in background (PID file + log redirect)
capture-proxy --status      # running? PID? actual settings?
capture-proxy --stop        # SIGTERM + cleanup
```

PID/log paths default to `~/.cache/capture-proxy/`.

---

## Route traffic through the proxy

Configure your OpenAI client to use the proxy as the base URL:

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed",  # proxy doesn't auth
)
```

**Endpoints:**

| Method | Path | Purpose | Streaming? |
|--------|------|---------|------------|
| GET | `/health` | Health check | — |
| GET | `/v1/models` | List models | — |
| POST | `/v1/chat/completions` | Chat completions | Yes |
| POST | `/v1/completions` | Text completions | Yes |
| POST | `/v1/embeddings` | Embeddings | No |
| POST | `/v1/rerank` | Rerank documents | No |
| POST | `/v1/classification` | Classifier labels + scores | No |
| POST | `/v1/images/generations` | Image generation | No |
| GET | `/v1/sessions` | List session ids | — |
| GET | `/v1/sessions/{id}/captures` | Captures for one session | — |
| GET | `/v1/users` | List user ids | — |
| GET | `/v1/users/{id}/captures` | Captures for one user | — |

---

## Session & user tracking

Every capture carries `session_id` (WHICH conversation) and `user_id` (WHO sent it).
Send `x-session-id` / `x-user-id` headers (case-insensitive). Without them, `user_id`
falls back to a hash of the Authorization header (raw key never stored), then
`"anonymous"`.

Each capture appends one line to `{CAPTURE_BASE_PATH}/index/sessions/` and
`.../index/users/` — lookups read one small index file, never a raw JSONL scan.

```bash
curl -s http://localhost:8000/v1/sessions/my-session-123/captures
curl -s http://localhost:8000/v1/users/alice/captures
curl -s http://localhost:8000/v1/users
```

Dedupe on `capture_id` before counting turns — a capture with both session and user
is written to both indexes (two views of one turn).

---

## What gets captured

Every request/response pair is saved as a JSONL record containing:

- Full request body with SHA-256 hashes
- Full response body (reconstructed from SSE chunks for streaming)
- Latency, session ID, client IP, model, upstream status
- Token usage (normalized)
- Streaming diagnostics (first-byte timing, chunk counts, errors)
- Runtime metadata (run ID, per-backend serve settings)

**Path:** `{CAPTURE_BASE_PATH}/raw/` (success), `{CAPTURE_BASE_PATH}/errors/`
(failures).

**Hashes resolve:** `request_body_sha256` and `messages_sha256` are backed by a
content-addressed store at `{CAPTURE_BASE_PATH}/blobs/sha256/{digest[:2]}/{digest}`.
Set `CAPTURE_BLOB_STORE=off` to stop writing (hashes still compute, just dangling).

**Streamed captures are self-describing:** the `request_start` line of a
`*.stream.jsonl` carries the full `request_body` and the `task` label.

**Schema declares itself:** `{CAPTURE_BASE_PATH}/schema.json` is written at startup
— version, layout, task vocabulary, tri-state field classification (absent vs null).

---

## Task labels

`CAPTURE_MODEL_TASKS` maps a **public** model id to a task-type label. Keys match
exactly, case-sensitively, against the public id (never the forwarded upstream id).

```bash
CAPTURE_MODEL_TASKS='{"nomic-embed-text":"embedding","roberta-base-mnli":"classification"}'
```

Off by default. When off, labels are derived structurally from request shape:

| Rule | Task |
|------|------|
| endpoint family | `embedding.batch` / `rerank` / `classification.*` / `image.generate` / `completion.text` |
| non-empty `tools` array | `chat.agentic` |
| `input` is list of strings | `embedding.batch` |
| `query` + `documents` | `rerank` |
| code payload + summarization instruction | `code_summarize` |
| 1–4 messages, no tools, < 4 KiB | `chat.single_shot` |
| otherwise | `chat.agentic` |

Classification never reads topic vocabulary — structural only (document-scaffold
gate: skill bodies are Markdown, summarization payloads are raw source).

---

## Headroom sidecar (opt-in, off by default)

When `CAPTURE_HEADROOM` is enabled, sampled eligible captures additionally write:

- `{CAPTURE_HEADROOM_DIR}/YYYY-MM-DD/{capture_id}.headroom.json` — sidecar
- `{CAPTURE_HEADROOM_DIR}/YYYY-MM-DD/{capture_id}-weight_pairs.json` — weight pairs
  in the model_headroom loader format

**Eligible request types:** `chat`, `completion`, `classification`, `image`.
**Excluded:** `embeddings`, `rerank` — neither handler calls `_spawn_headroom`.

The sidecar is written post-response, async, never awaited on the response path
(R-06: `_spawn_headroom` wall time < 5 ms p99). If `CAPTURE_HEADROOM=off`, no writer
is constructed — no files, no torch/safetensors import, nothing on the response
path (R-01 purity).

**Signal keys:** `effective_rank` is emitted by the MVP; `load_balance`,
`dead_neurons`, `specialization`, `loss_saturation`, `growth_factor` remain reserved
null until producers land.

**Key env vars:**

| Var | Default | Purpose |
|-----|---------|---------|
| `CAPTURE_HEADROOM` | `off` | `off` \| `all` \| comma list of signals |
| `CAPTURE_HEADROOM_WEIGHTS` | `off` | `off` \| `gate` \| `all` |
| `CAPTURE_HEADROOM_MAX_BYTES` | `0` | Float32 byte cap; 0 = unlimited |
| `CAPTURE_HEADROOM_INTERVAL` | `0` | 1 of every N captures; 0 = every |
| `CAPTURE_HEADROOM_DIR` | `{base_path}/headroom` | Sidecar root |

---

## Aggregates (jq over JSONL, no endpoint)

No `/v1/stats` route exists. Aggregates are answered by `jq`/`rg` over JSONL:

```bash
BASE=${CAPTURE_BASE_PATH:-/root/captures}
# per-model counts (success + errors)
jq -r '.model' "$BASE"/raw/*/*.jsonl "$BASE"/errors/*/*.jsonl 2>/dev/null \
  | sort | uniq -c | sort -nr
```

---

## Operational boundary (router vs proxy)

Router lifecycle — demand-load model loads, `models_max` LRU evictions, VRAM
pressure, slots/queue depth — stays with the router log plus `GET /v1/models`
status plus `nvidia-smi`. The proxy exposes only its registered routes and provides
no alerting and no live-tail endpoint. Proxy-visible failures are JSONL rows in
`raw/` vs `errors/` plus the proxied 502 status.

---

## Cross-references

- `architecture/authority-carrier.md` — carrier role, what it never does
- `architecture/contract.md` — what the carrier emits to the authority
- `concepts/headroom-signals.md` — signal keys in sidecars
- `operations/environment.md` — GPU/CPU/TPU notes
- `reference/cli-flags.md` — capture-proxy CLI flags
- `reference/env-vars.md` — all `CAPTURE_*` env vars
- `reference/file-layout.md` — capture directory layout
