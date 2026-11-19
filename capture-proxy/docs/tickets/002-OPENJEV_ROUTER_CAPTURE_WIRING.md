# 002: Wire the systemone decision stack through the proxy (shim → proxy → router) and close the monitoring gaps

**Status:** SUPERSEDED by ticket 007 (`007-SYSTEMONE_ROUTER_CAPTURE_WIRING.md`) — openjev serves systemone format; every openjev decision call is a `:3000/v1/systemone` call, so ticket 007 covers openjev fully. No separate openjev ticket is needed.
**Severity:** Medium (observability — every decision call is currently uncaptured; no traffic loss, no wrong answers)
**Written by:** CEO agent, during cross-repo investigation of `/root/openjev-serve` → this proxy
**Project of record:** `capture-proxy`
**Direction:** this repo owns the proxy and its config surface; `/root/openjev-serve`
owns the shim and the router — the wiring change there is proposed, not made

---

## 1. Summary

The systemone decision stack (served by openjev) is live and direct-wired: shim (`:3000`) → llama.cpp router (`:8012`), with `VLLM=http://127.0.0.1:8012/v1` baked into `start-all.sh:31`. No capture proxy sits in the path, so zero decision traffic is recorded — every `/v1/systemone` prompt, every `top_logprobs` readout, every latency sample is emitted and lost. Port `:8000` is free and `/root/captures` has no decision traffic.

The fix is two env changes, not code: the shim already builds its entire OpenAI client from `$VLLM` (`shim.py:17`), and every LLM call in the shim (`shim.py:110,151,157` — all `model="qwen"`, non-streaming, logprobs on) flows through that one base URL. Point the shim at the proxy, point the proxy at the router:

```bash
# proxy: upstream is the router, explicit llamacpp backend, :8000
CAPTURE_UPSTREAM_URL=http://127.0.0.1:8012 \
CAPTURE_BACKEND=llamacpp \
CAPTURE_PORT=8000 \
CAPTURE_BASE_PATH=/root/captures \
CAPTURE_MODEL_TASKS='{"qwen":"decision"}' \
capture-proxy --daemon

# shim: VLLM moves off the router onto the proxy (only change)
VLLM=http://127.0.0.1:8000/v1
```

Deliberate non-settings: backend is explicit `llamacpp` (auto would infer it by
the port rule, but explicit is the documented contract and survives port
moves); aliases stay OFF (the router resolves `model="qwen"` itself via
`router/models.ini` — a proxy-side rewrite would break the load-bearing
alias); the tasks map needs exactly one key because the shim sends exactly one
public id. Bring-up order is router → proxy daemon → shim restart, all under
the existing PID-file discipline.

## 2. Evidence

Reproduce from the two workspaces (live system, 2026-10-01):

```bash
ss -ltn | grep -E ":(8000|8012|3000) "
# -> :8012 router UP, :3000 shim UP, :8000 FREE

grep -n "VLLM" /root/openjev-serve/start-all.sh
# -> 31: VLLM="http://127.0.0.1:$LLAMA_PORT/v1"  (direct-wired, no proxy)

grep -n "x-session-id\|x-user-id\|extra_headers" /root/openjev-serve/helper/shim.py
# -> 0 matches: the shim sends no attribution headers (see §4 item 2)

ls /root/captures 2>/dev/null || echo "no captures yet"
```

Proxy-side facts used above (all in `AGENTS.md`, verified by reading, not
assumed): endpoints include `/v1/chat/completions` non-streaming;
`model_tasks` keys match the **public** id per request; aliases are exact,
case-sensitive, off by default, restart-only; `top_logprobs` bodies are
stored verbatim (calibration data survives); chat captures are headroom
sidecar-eligible.

## 3. Why it matters

The shim's readout is logprob-based (`READOUT_T`, `READOUT_NOUL_T`,
`top_logprobs=max(20, len(options))`) — the probabilities are the product, not
a byproduct. Without captures there is no calibration audit, no prompt
regression set, no latency history, and no record of what the decision model
actually saw. The wiring is cheap (two env vars, zero code) and the data it
unlocks is the precondition for every measurement conversation about this
stack. The gaps in §4 are listed here so the wiring is not mistaken for full
monitoring.

## 4. What is being asked

1. **Land the wiring** as the official path: proxy daemon `:8000` upstream
   `:8012` per the config above; shim `VLLM` → `:8000/v1`; integrate both
   into `start-all.sh` / `stop-all.sh` (proxy up before shim, down after;
   PID files, never pkill). Verify: proxy `/health` shows upstream reachable
   → one `:3000/v1/systemone` request → `/root/captures/raw/` gains a line
   with `model:"qwen"`, the full `top_logprobs` body, and `task:"decision"`.
2. **Attribution.** Either send `x-session-id` / `x-user-id` from the shim
   (`extra_headers` per call — small shim change) so `/v1/sessions` and
   `/v1/users` become useful, or record accepted-anonymous here and stop
   expecting those endpoints to work on decision traffic. Today every capture
   collapses to one pseudo-user with ungrouped sessions.
3. **Aggregates.** Decide the official answer for "per-model counts, error
   rates, p50/p99, tokens/min": a new stats endpoint, or a documented
   `jq`/`rg`-over-JSONL recipe. The proxy exposes neither today — only raw
   records plus session/user lookups.
4. **Record the out-of-scope boundary** (in AGENTS.md or README, one
   paragraph): router lifecycle — demand-load loads, `models_max` LRU
   evictions, VRAM, slots/queue — stays with the router log + `/v1/models`
   `status` + `nvidia-smi`; no alerting and no live-tail endpoint exist
   (failures are JSONL rows and 502s). If any of these is actually wanted,
   that is a follow-up ticket, not this one.
5. **Headroom env (reference, not duplicate).** Chat captures are sidecar
   eligible, but weight telemetry stays dark until the torch/safetensors venv
   work lands (tracked in `next-session.md` TODO 1). No action here beyond not
   claiming coverage.

## 5. Scope note

Filed from a cross-repo investigation; no file in either repository was
modified in the course of filing this. Items 1 (shim half), 2-shim-side, and
the router-lifecycle monitoring in item 4 belong to `/root/openjev-serve`.
Proxy-side work (daemon config, item 3, item 4 docs) belongs here.
Acceptance for the wiring half: §4 item 1's verification paragraph plus the
full gate green (`ruff check`, `mypy strict`, `pytest`) and the lifecycle
scripts updated.

## 6. Status log (append-only; the sections above are history)

- 2026-10-02 — items 3, 4 LANDED proxy-side in `6048c4a` ("land 002
  boundary/aggregates sections"): aggregates answered by the §7b
  jq-over-JSONL recipe (no stats endpoint), boundary recorded in §7a.
  No code change needed for either; decision recorded, not deferred.
- 2026-10-12 — items 1, 2 BLOCKED on this machine: `/root/openjev-serve`
  absent, none of :3000/:8012/:8000 listening. The wiring half (proxy
  daemon config, shim `VLLM` move, lifecycle scripts, attribution headers)
  can only land where the stack lives. Item 5 remains reference-only.
  Ticket stays OPEN pending a machine with the stack.
- 2026-11-16 — SUPERSEDED by ticket 007 (`007-SYSTEMONE_ROUTER_CAPTURE_WIRING.md`).
  Openjev serves systemone format; every openjev decision call is a
  `:3000/v1/systemone` call, so ticket 007 covers openjev fully. No separate
  openjev ticket is needed.
