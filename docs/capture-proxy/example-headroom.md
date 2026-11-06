# Example: Headroom Sidecar Capture (end-to-end)

Companion to `docs/headroom-capture-design.md` (ADR-001). Full loop: start the
proxy with the sidecar on, send one eligible request, inspect the emitted
files, score them with `model_headroom`.

## 1. Start the proxy with headroom capture

```bash
# vLLM serving Qwen3.8-27B (match the model name to what upstream serves)
uv run capture-proxy \
  --upstream-url http://localhost:8000 --backend vllm \
  --port 8001 \
  --headroom all \
  --headroom-weights gate \
  --headroom-max-bytes 256M
```

- `--headroom all` turns on all six signal keys; the MVP emits only
  `effective_rank` — the other five are parsed, recorded, and stay `null`
  in the sidecar (`src/capture_proxy/app.py:337-352` mode parsing;
  `src/capture_proxy/headroom.py:813` has the only signal branch).
- `--headroom-weights gate` snapshots layer-0 `gate_proj` for MoE models or
  layer-0 `attention` for dense models (BERT/DeBERTa/RoBERTa); `all` captures
  gate+up+down or attention+intermediate+output respectively
  (`src/capture_proxy/headroom.py:105` `_CAPTURE_LAYER = 0`,
  `src/capture_proxy/headroom.py:683-696` projection list).
- `--headroom-max-bytes 256M` caps the float32 payload; over-cap tensors
  are row-truncated and flagged `truncated: true` with an `error_body`
  note (`src/capture_proxy/headroom.py:440-442` row truncation,
  `src/capture_proxy/headroom.py:733-745` `trunc_note`).
- Env-var equivalent: `CAPTURE_HEADROOM=all CAPTURE_HEADROOM_WEIGHTS=gate
  CAPTURE_HEADROOM_MAX_BYTES=256M uv run capture-proxy --port 8001`.
  Precedence is CLI > env > config file > default
  (`src/capture_proxy/__main__.py:484-495`).

## 2. Send one eligible request

```bash
curl -s http://localhost:8001/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model": "qwen3.8-27b-fp8",
       "messages": [{"role": "user", "content": "hello"}],
       "stream": false}'
```

Classification and image requests are eligible too — same flags, no extra
setup:

```bash
curl -s http://localhost:8001/v1/classification \
  -H 'Content-Type: application/json' \
  -d '{"model": "roberta-base-mnli", "inputs": "hello"}'

curl -s http://localhost:8001/v1/images/generations \
  -H 'Content-Type: application/json' \
  -d '{"model": "sdxl-turbo", "prompt": "a red cube", "n": 1}'
```

Set `CAPTURE_MODEL_TASKS` to label the sidecar; the task string is copied into
`HeadroomRef.task` and `CapturedHeadroom.task` when the **public** model id
matches exactly:

```bash
CAPTURE_MODEL_TASKS='{"roberta-base-mnli":"classification"}'
```

Eligibility (`src/capture_proxy/app.py:1846-1922` `_spawn_headroom`):
`request_type` chat, completion, classification, or image;
`upstream_status < 400` and no
`error_body`, `runtime_settings.model_path` pointing at an existing
directory, `capture_id` parsing as a UUID, and the interval sampling this
capture. Embeddings and rerank never write a sidecar — neither handler calls
`_spawn_headroom` and the request-type filter rejects them.

## 3. Inspect what got written

Under the day directory (default `{base_path}/headroom/`, typically
`/root/captures/headroom/`; override with `--headroom-dir`):

- `{capture_id}.headroom.json` — the sidecar (authoritative artifact,
  written last: `src/capture_proxy/headroom.py:843`)
- `{capture_id}-weight_pairs.json` — compact JSON in the `model_headroom`
  loader format (`src/capture_proxy/headroom.py:643` `_capture_weight_pairs`)

```bash
cat /root/captures/headroom/*/*.headroom.json | python -m json.tool
```

The glob matches exactly one file until more captures land; once there are
more, pick one file — `python -m json.tool` reads stdin as a single JSON
document.

The paired raw JSONL line carries a precomputed `HeadroomRef` in its
`headroom` key (`src/capture_proxy/models.py:88-128`). If the sidecar file
is missing, the background write did not complete — the ref path is
precomputed and the sidecar is authoritative
(`src/capture_proxy/models.py:20-24`).

Schema versions are additive: `HeadroomRef` and `CapturedHeadroom` are
`schema_version: 6`, so a v5 sidecar without the `task` key still validates
(`task` defaults to `None`).

## 4. Score with model_headroom

`model_headroom` runs in its own venv (torch/safetensors); the proxy venv
has neither by design (`docs/headroom-capture-design.md` §6).

```bash
python -m model_headroom \
  --model-path /path/to/model \
  --weight-pairs-file /root/captures/headroom/YYYY-MM-DD/{capture_id}-weight_pairs.json \
  --output-format json | python -m json.tool
```

Weight-only pairs carry a real zero-gradient array of the weight's shape
(ADR-002; `src/capture_proxy/headroom.py:393`): the effective-rank signal
scores from the weight, and the grad/weight ratio comes back 0.0 without
a training hook.

## 5. Off by default

With `CAPTURE_HEADROOM` unset or `off`, no writer is constructed — no
files, no torch/safetensors import, nothing on the response path
(`src/capture_proxy/app.py:337-339`,
`src/capture_proxy/headroom.py:13-14`).
