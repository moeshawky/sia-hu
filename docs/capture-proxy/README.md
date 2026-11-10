# capture-proxy

Capture proxy for llama.cpp, ollama, node-llama-cpp, vLLM, and SGLang — logs raw JSONL of all requests/responses before forwarding. Exposes an **OpenAI-compatible API** so any agent using OpenAI client libraries can route through it transparently.

**Version:** 0.3.0 — **Last Verified:** 2026-09-26 UTC

## Quick start

```bash
# Point at your upstream (llama.cpp/ollama/node-llama-cpp)
CAPTURE_UPSTREAM_URL=http://host.docker.internal:8080 uv run capture-proxy --port 8000

# Or explicitly set the backend type
CAPTURE_UPSTREAM_URL=http://localhost:3000 CAPTURE_BACKEND=node-llama-cpp uv run capture-proxy --port 8000

# vLLM (defaults to :8000 — pick a different proxy port to avoid the clash)
CAPTURE_UPSTREAM_URL=http://localhost:8000 CAPTURE_BACKEND=vllm uv run capture-proxy --port 8001

# SGLang (defaults to :30000 — not auto-detected, explicit backend required)
CAPTURE_UPSTREAM_URL=http://127.0.0.1:31000 CAPTURE_BACKEND=sglang uv run capture-proxy --port 8001

# Expected: "INFO:     Started server on 0.0.0.0:8000"
```

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed",
)
```

## Configuration

Settings resolve in order: **CLI flag > env var > config file > default**.

Persist settings in `~/.config/capture-proxy/config.toml`:

```toml
upstream_url = "http://127.0.0.1:8080"
port = 8000
host = "0.0.0.0"
base_path = "/root/captures"
backend = "llamacpp"   # auto | llamacpp | ollama | node-llama-cpp | vllm | sglang
# model_aliases = { "qwen3.8-27b-fp8" = "davidau-qwen3.8-27b-twin-turbo-fable" }  # off by default; proxy-restart-only
# model_tasks = { "nomic-embed-text" = "embedding" }  # off by default; re-read from the environment per request
# headroom = "off"       # off | all | comma list: effective_rank,dead_neurons,load_balance,specialization,loss,isoflop
# headroom_weights = "off"   # off | gate | all
# headroom_max_bytes = "0"   # e.g. "256M"; 0 = unlimited
# headroom_interval = 0      # 1 of every N eligible captures; 0 = every
# headroom_dir = "/root/captures/headroom"
```

A missing or invalid config file is ignored (warning to stderr), never fatal. Omitted keys fall back to env var, then default. Override the path with `--config PATH`.

The canonical default lives in [`config/capture-proxy.toml`](config/capture-proxy.toml) and is what `--ensure-config` writes on a box that has no config yet — it binds `127.0.0.1` (loopback-only by policy: one port out, one port in), so a fresh box never listens on every interface by accident.

### Provisioning a restarted instance

```bash
capture-proxy --ensure-config            # create the config if absent; never overwrite an existing one
capture-proxy --print-effective-config  # print resolved settings (CLI > env > file > default), then exit
```

`--ensure-config` is idempotent and never clobbers an operator edit: the second run reports `kept existing` and leaves the bytes alone. That is what makes it safe to run on every boot. `--print-effective-config` exists so a boot script can *read back* what was loaded instead of assuming the file it wrote is the file that got read.



## Running as a background service

```bash
capture-proxy --daemon      # start in background
capture-proxy --status      # running? PID? actual settings?
capture-proxy --stop        # SIGTERM + cleanup
```

- PID file: `~/.cache/capture-proxy/proxy.pid`
- Settings sidecar (what the daemon actually started with): `~/.cache/capture-proxy/proxy.info.json`
- Logs: `~/.cache/capture-proxy/proxy.log`
- Override paths with `--pid-file PATH` / `--log-file PATH`

## CLI reference

| Flag | Purpose |
|------|---------|
| `--port`, `--host`, `--upstream-url`, `--base-path`, `--backend` | Override settings (env var fallback) |
| `--model-aliases '{"qwen3.8-27b-fp8":"davidau-qwen3.8-27b-twin-turbo-fable"}'` | Public→upstream model map (env `CAPTURE_MODEL_ALIASES`, TOML `model_aliases` table; off by default; proxy-restart-only) |
| `--model-tasks '{"nomic-embed-text":"embedding"}'` | Public-model-id → task-type label map (env `CAPTURE_MODEL_TASKS`, TOML `model_tasks` table; off by default; re-read per request) |
| `--headroom`, `--headroom-weights`, `--headroom-max-bytes`, `--headroom-interval`, `--headroom-dir` | Headroom sidecar capture (off by default) |
| `--config PATH` | Config file path (default `~/.config/capture-proxy/config.toml`) |
| `--ensure-config` | Create the default config file if absent; never overwrite an existing one |
| `--print-effective-config` | Print resolved settings (`upstream_url`, `port`, `host`, `base_path`, `backend`, headroom) and exit |
| `--daemon` | Run in background (PID file + log redirect) |
| `--stop` | Stop the background instance |
| `--status` | Show background instance status |
| `--pid-file PATH`, `--log-file PATH` | Override daemon paths |


## Endpoints

| Method | Path | Purpose | Streaming? |
|--------|------|---------|------------|
| `GET` | `/health` | Health check with upstream status | — |
| `GET` | `/v1/models` | List available models from upstream | — |
| `POST` | `/v1/chat/completions` | Chat completions | Yes |
| `POST` | `/v1/completions` | Text completions | Yes |
| `POST` | `/v1/embeddings` | Embeddings | No |
| `POST` | `/v1/rerank` | Rerank documents by relevance to query | No |
| `POST` | `/v1/classification` | Classifier labels + scores (BERT/DeBERTa) | No |
| `POST` | `/v1/images/generations` | Image generation | No |
| `GET` | `/v1/sessions` | List known session ids (most recent first) | — |
| `GET` | `/v1/sessions/{id}/captures` | All captures for one session id | — |
| `GET` | `/v1/users` | List known user ids (most recent first) | — |
| `GET` | `/v1/users/{id}/captures` | All captures for one user id | — |

**Source:** `src/capture_proxy/app.py:385-502` — all routes registered in `_register_routes()`

### Session & user tracking

Every capture carries a `session_id` (WHICH conversation) and a `user_id`
(WHO sent it). The user id resolves from, in priority order:

1. `x-user-id` request header (case-insensitive; blank values ignored)
2. `user-` + SHA-256 prefix of the `Authorization` header (raw key never stored)
3. `"anonymous"` (headerless traffic groups into one queryable bucket)

Each capture appends one line to `{CAPTURE_BASE_PATH}/index/sessions/` and
`.../index/users/`, so lookups read a single small index file instead of
scanning `raw/` JSONL:

```bash
# every capture for one session
curl -s http://localhost:8000/v1/sessions/my-session-123/captures

# everything a user ever sent
curl -s http://localhost:8000/v1/users/alice/captures

# who has been connecting lately?
curl -s http://localhost:8000/v1/users
```

Index writes never block or fail the proxied response; a broken index tree
degrades to a one-time warning while captures continue.

### Rerank

Cohere/Jina-compatible rerank endpoint. Passthrough to llama.cpp, vLLM, and SGLang (vLLM exposes `/v1/rerank` since v0.7, for cross-encoder/score models); returns 501 for ollama.

```python
response = client.post(
    "http://localhost:8000/v1/rerank",
    json={
        "model": "llama3.2",
        "query": "what is python",
        "documents": ["doc A", "doc B"],
    },
)
print(response.json()["results"])
```

**Errors:**
- `stream=True` → 400
- Empty `model`, `query`, or `documents` → 422
- Ollama backend → 501
- Upstream unreachable → 502

**Source:** `src/capture_proxy/app.py:985-1150` (`_handle_rerank_request`), `src/capture_proxy/models.py:383-405` (`RerankRequest` model)

### Embeddings

OpenAI-compatible embeddings. For llama.cpp, node-llama-cpp, vLLM, and SGLang: verbatim passthrough to `/v1/embeddings`. For ollama: translated to `/api/embed` and converted back.

```python
response = client.embeddings.create(
    model="nomic-embed-text",
    input="Hello world",
)
print(response.data[0].embedding)
```

### Classification

Classifier labels + scores. Body is forwarded to upstream `/v1/classification`
verbatim and the upstream response is returned unchanged. No backend branch —
llama.cpp, ollama, node-llama-cpp, vLLM, and SGLang all forward; unlike `/v1/rerank`,
ollama is **not** rejected.

```python
response = client.post(
    "http://localhost:8000/v1/classification",
    json={"model": "roberta-base-mnli", "inputs": "I love this"},
)
print(response.json()["labels"], response.json()["scores"])
```

Request: `model` (non-empty), `inputs` (non-empty string or list of strings),
`parameters` (optional passthrough). Response: `labels` (one per input, in input
order), `scores` (parallel to `labels`), `model`.

**Errors:**
- Invalid JSON, failed schema validation, or empty `model` / `inputs` → 422
- Upstream 4xx/5xx → propagated verbatim with its body
- Upstream unreachable → 502

Schema validation failures are written to `{base_path}/errors/`; the
early-validation 422s (empty/blank fields) exit before a capture exists and are
not logged as captures.

**Source:** `src/capture_proxy/app.py:1153-1312` (`_handle_classification_request`), `src/capture_proxy/models.py:408-449` (`ClassificationRequest`, `ClassificationResponse`)

### Image generation

Image generation. Body is forwarded to upstream `/v1/images/generations` verbatim
and the upstream response is returned unchanged. No backend branch — every
backend forwards, including ollama.

```python
response = client.post(
    "http://localhost:8000/v1/images/generations",
    json={"model": "sdxl-turbo", "prompt": "a red cube", "n": 1},
)
print(response.json()["data"])
```

Request: `model` (non-empty), `prompt` (non-empty), `n` (>= 1), `size`
(`"WIDTHxHEIGHT"`), `response_format` (`"url"` or `"b64_json"`). Response:
`data` (one entry per image), `model`, `object` (always `"list"`).

Defaults exist for schema validation only — the proxy never injects `n`, `size`,
or `response_format` into the forwarded body, and never resizes or re-encodes a
generated image. `b64_json` payloads are forwarded as received.

**Errors:**
- Invalid JSON, failed schema validation, empty `model` / `prompt`, or `n < 1` → 422
- Upstream 4xx/5xx → propagated verbatim with its body
- Upstream unreachable → 502

**Source:** `src/capture_proxy/app.py:1315-1470` (`_handle_image_request`), `src/capture_proxy/models.py:452-498` (`ImageRequest`, `ImageResponse`)

### Task labels

`CAPTURE_MODEL_TASKS` maps a **public** model id (the value stored in
`CaptureRecord.model`, never the forwarded upstream id) to a task-type label.
Keys match exactly, case-sensitively.

```bash
CAPTURE_MODEL_TASKS='{"roberta-base-mnli":"classification","nomic-embed-text":"embedding"}'
```

The resolved label lands in `CaptureRecord.task` and, when a headroom sidecar is
written, in `HeadroomRef.task` / `CapturedHeadroom.task`. Off by default; a
malformed map warns once on stderr and degrades to off (never fails a request).
The proxy re-reads the environment per request, so editing the env var takes
effect without a restart — but a TOML `model_tasks` table or `--model-tasks`
flag is resolved once at startup and still needs a restart.

**Source:** `src/capture_proxy/app.py` (`_resolve_task`, `_derive_task`), `src/capture_proxy/__main__.py` (`_resolve_model_tasks_raw`)

#### Structural labels (when the map is off)

The map is off by default, so on a default deployment it labelled nothing —
`task` was null in every captured request, which left a corpus with no
supported way to answer "show me the reranking captures". Every capture is
now labelled from request **shape** instead:

| Rule | Task |
|------|------|
| endpoint declares the family (`embedding`, `rerank`, `classification`, `image`, `completion`) | `embedding.batch` / `embedding.single` / `rerank` / `classification.*` / `image.generate` / `completion.text` |
| non-empty `tools` array | `chat.agentic` |
| `input` is a list of strings | `embedding.batch` |
| payload carries `query` + `documents` | `rerank` |
| raw code payload + summarization instruction, not a document | `code_summarize` |
| 1–4 messages, no tools, body < 4 KiB | `chat.single_shot` |
| otherwise | `chat.agentic` |

Each record also carries `task_rule` (which rule fired) and `task_evidence`
(the measurements it read), so a label can be audited without re-running the
classifier.

**Classification never reads topic vocabulary.** Agent sessions carry the
`cannon-gate` and `symbiose` skill bodies in their system prompts, and those
documents discuss reranking and code summarization throughout — a
topic-matching classifier flags every session in the corpus. The guard is a
*document-scaffold* measurement: a skill body is Markdown (fences, headings,
tables, bullets), a summarization payload is raw source. Measured over 251
deployed skill documents and 69 source files, this admits **0/251** skill
documents and 64/69 source files. `tests/test_taskclass_adversarial.py` sweeps
the deployed corpus on every run and fails on any false positive.

#### Querying a corpus

```bash
uv run python scripts/capture_query.py --base /root/captures tasks
uv run python scripts/capture_query.py --base /root/captures captures --task rerank
uv run python scripts/capture_query.py --base /root/captures captures --task code_summarize
uv run python scripts/capture_query.py --base /root/captures captures --stream
uv run python scripts/capture_query.py --base /root/captures stats
```

Works on new and legacy stores: a null `task` is backfilled by running the
same classifier over the stored request body. Turns are deduped on
`capture_id` — see the index note below.

## Upstream compatibility

| Backend | Detection | Chat | Completions | Embeddings | Rerank |
|---------|-----------|------|-------------|------------|--------|
| llama.cpp | `upstream_url` port ≠ 11434 | Native | Native | Native | Native |
| ollama | `upstream_url` port = 11434 | Native | Native | Translated (`/api/embed`) | 501 |
| node-llama-cpp | `CAPTURE_BACKEND=node-llama-cpp` | Native | Native | Native | Passthrough |
| vllm | `CAPTURE_BACKEND=vllm` (port 8000 not auto-detected) | Native | Native | Native | Passthrough (vLLM ≥ 0.7) |

`/v1/classification` and `/v1/images/generations` are backend-agnostic: every
backend forwards them verbatim, and none returns 501.

## Captured data

Every request/response pair is saved as a JSONL record containing:
- Full request body with SHA-256 hashes
- Full response body (reconstructed from SSE chunks for streaming)
- Latency, session ID, client IP, model, upstream status
- Token usage (normalized for diagnostics)
- Streaming diagnostics (first-byte timing, chunk counts, errors)
- Runtime metadata (run ID, per-backend serve settings — llama-server or vLLM)
- Task label (`task`), `task_rule`, and `task_evidence` — from
  `CAPTURE_MODEL_TASKS` when it maps the public model id, otherwise derived
  structurally from request shape (see [Task labels](#task-labels))

**Path:** `{CAPTURE_BASE_PATH}/raw/` (success), `{CAPTURE_BASE_PATH}/errors/` (failures)

**Hashes resolve.** `request_body_sha256` and `messages_sha256` are backed by a
content-addressed store at `{CAPTURE_BASE_PATH}/blobs/sha256/{digest[:2]}/{digest}`
— deduplicated, written once per distinct body. Set `CAPTURE_BLOB_STORE=off` to
stop writing, which leaves those hashes dangling.

**Streamed captures are self-describing.** The `request_start` line of a
`*.stream.jsonl` file carries the full `request_body` and the `task` label, not
just a hash and a message count, so a stream capture is analyzable standalone.

**The schema declares itself.** `{CAPTURE_BASE_PATH}/schema.json` is written at
startup: schema version, artifact layout, task vocabulary, index key, and a
tri-state field classification that keeps *absent* (not recorded) distinct from
*null* (not applicable to this request type).

**Index entries carry `task`, `task_rule`, and `stream_file`.** A capture with
both a session and a user id is written to **both** `index/sessions/` and
`index/users/` — two views of one turn. Dedupe on `index_key` (the declared
key, equal to `capture_id`) before counting turns.

**Headroom sidecar (opt-in, off by default):** when `CAPTURE_HEADROOM` is enabled, sampled eligible captures additionally write `{CAPTURE_HEADROOM_DIR}/YYYY-MM-DD/{capture_id}.headroom.json` plus a `*weight_pairs.json` (model_headroom loader format) — post-response, async, never awaited on the response path. End-to-end example: `example-headroom.md`.

Eligibility covers four of the six request types: `chat`, `completion`,
`classification`, and `image`. **Embeddings and rerank never write a sidecar** —
their handlers do not call `_spawn_headroom`, and the request-type filter
rejects them. The sidecar carries the `task` label when `CAPTURE_MODEL_TASKS`
maps the model. Full gate list: `example-headroom.md` §2.

### Operational boundary (router vs proxy)

Operational boundary: router lifecycle — demand-load model loads, `models_max` LRU evictions, VRAM pressure, and slots/queue depth — stays with the router log plus `GET /v1/models` status plus `nvidia-smi`; the proxy exposes only the routes registered in `src/capture_proxy/app.py:381-499` (`/health`, `/v1/models`, `/v1/chat/completions`, `/v1/completions`, `/v1/embeddings`, `/v1/rerank`, `/v1/classification`, `/v1/images/generations`, `/v1/sessions*`, `/v1/users*`) and provides no alerting and no live-tail endpoint; proxy-visible failures are JSONL rows in `{base}/raw/` vs `{base}/errors/` plus the proxied 502 status; additional router observability or alerting is a follow-up ticket, not this proxy.

### Aggregates (jq over JSONL, no endpoint)

Aggregates are answered by `jq`/`rg` over JSONL, not by a `/v1/stats` endpoint (no such route exists in `src/capture_proxy/app.py:381-499`); success rows live in `{base}/raw/YYYY-MM-DD/*.jsonl` and failure rows in `{base}/errors/YYYY-MM-DD/*.jsonl` (`src/capture_proxy/logger.py:102-141`); fields are `CaptureRecord.model`, `.latency_ms`, `.created_at`, `.upstream_status`, `.usage_normalized.{prompt_total,completion,context_total}` (`src/capture_proxy/models.py:41-51,262-285`); verify with `rg -n "_register_routes|/v1/stats" src/capture_proxy/app.py` and `rg -n "class CaptureRecord|latency_ms|usage_normalized|upstream_status" src/capture_proxy/models.py`; last verified 2026-10-02 UTC.

```bash
BASE=${CAPTURE_BASE_PATH:-/root/captures}
# per-model counts (success + errors)
jq -r '.model' "$BASE"/raw/*/*.jsonl "$BASE"/errors/*/*.jsonl 2>/dev/null | sort | uniq -c | sort -nr
# error rate by model (errors / total)
jq -r '.model' "$BASE"/raw/*/*.jsonl 2>/dev/null | sort | uniq -c | sort > /tmp/ok.txt; jq -r '.model' "$BASE"/errors/*/*.jsonl 2>/dev/null | sort | uniq -c | sort > /tmp/err.txt; join -a1 -a2 -o 0,1.1,2.1 -e 0 /tmp/ok.txt /tmp/err.txt
# p50/p99 latency_ms per model (raw only)
jq -r '[.model, .latency_ms] | @tsv' "$BASE"/raw/*/*.jsonl 2>/dev/null | sort -k1,1 -k2,2n | awk '{m=$1; n[m]++; v[m,n[m]]=$2} END {for (m in n) {c=n[m]; p50=v[m,int((c+1)/2)]; p99=v[m,int(c*0.99+0.5)]; print m, "n="c, "p50="p50, "p99="p99}}' | sort
# tokens/min from usage_normalized.context_total bucketed by created_at minute
jq -r '[.created_at[0:16], (.usage_normalized.context_total // 0)] | @tsv' "$BASE"/raw/*/*.jsonl 2>/dev/null | awk '{t[$1]+=$2; c[$1]++} END {for (k in t) print k, "tokens="t[k], "n="c[k]}' | sort
```

## Environment

| Variable | Default | Purpose |
|----------|---------|---------|
| `CAPTURE_PORT` | `8000` | Proxy listen port |
| `CAPTURE_HOST` | `0.0.0.0` | Bind address |
| `CAPTURE_UPSTREAM_URL` | `http://127.0.0.1:11434` | Target backend |
| `CAPTURE_BACKEND` | *(auto)* | Backend type: `auto`, `llamacpp`, `ollama`, `node-llama-cpp`, `vllm` |
| `CAPTURE_MODEL_ALIASES` | `off` | Public→upstream model map as JSON, e.g. `{"qwen3.8-27b-fp8":"davidau-qwen3.8-27b-twin-turbo-fable"}` (TOML `model_aliases` table; proxy-restart-only) |
| `CAPTURE_MODEL_TASKS` | `off` | Public-model-id → task-type map as JSON, e.g. `{"nomic-embed-text":"embedding"}` (TOML `model_tasks` table; keys match the public id exactly; read per request) |
| `CAPTURE_BASE_PATH` | `/root/captures` | JSONL storage root |
| `CAPTURE_HEADROOM` | `off` | Headroom sidecar capture: `off` \| `all` \| comma list of signals (default `effective_rank`-only when `all`). Eligible request types: `chat`, `completion`, `classification`, `image` |
| `CAPTURE_HEADROOM_WEIGHTS` | `off` | Which projections to snapshot: `off` \| `gate` \| `all`. For MoE models: `gate`, or `gate`+`up`+`down`. For dense models (BERT/DeBERTa/RoBERTa): `attention`, or `attention`+`intermediate`+`output`, with MoE projections as fallback |
| `CAPTURE_HEADROOM_MAX_BYTES` | `0` | Float32 byte cap per dump (e.g. `256M`); 0 = unlimited |
| `CAPTURE_HEADROOM_INTERVAL` | `0` | Sample 1 of every N eligible captures; 0 = every capture |
| `CAPTURE_HEADROOM_DIR` | `{base_path}/headroom` | Headroom sidecar root |
| `CAPTURE_BLOB_STORE` | `on` | Write request/message bodies to `{base_path}/blobs/sha256/` so `request_body_sha256` / `messages_sha256` resolve. `off` keeps hashing but stops writing, leaving those handles dangling |

## Code layout

```
src/capture_proxy/
  taskclass.py   → structural task classification (request shape, never vocabulary)
  blobs.py       → content-addressed store making request/messages hashes resolvable
  schema.py      → schema.json manifest (version, layout, tri-state field classes)
  __main__.py    → CLI entry point (config file, daemon/stop/status, uvicorn runner)
  app.py         → FastAPI app: routes, streaming/non-streaming handlers, embeddings, rerank
  alias.py       → Model alias map (exact public→upstream rewrite/sanitize, off by default)
  upstream.py    → httpx async proxy to upstream backend
  logger.py      → JSONL file writer (raw/ and errors/ dirs)
  headroom.py    → opt-in headroom sidecar writer (per-capture JSON + weight-pair files)
  session.py     → session_id resolution from headers
  runtime.py     → captures per-backend process metadata (llama-server/vllm) for run manifests
  stream.py      → SSE parser and streaming log writer
  models.py      → Pydantic schemas (CaptureRecord, RerankRequest, ChatCompletion*, Embedding*, Classification*, Image*)
tests/           → 710 tests, all upstream mocked
```

## Verification

### Test suite

```bash
uv run pytest                          # 475 tests, all pass
uv run pytest tests/test_rerank.py     # 29 rerank-specific tests
```

### Lint and types

```bash
uv run ruff check src/                 # 0 findings
uv run ruff format . --check           # formatting clean
```

## Conventions

- Python 3.14 runtime, ruff/mypy target 3.12
- Line length 88, ruff with `E,W,F,I,UP,B,SIM,RUF,S,C4,RET,TC,FA` rules
- mypy strict mode (requires manual install)
- pytest `asyncio_mode = "auto"` — no manual markers
- All upstream calls mocked in tests — no real HTTP
