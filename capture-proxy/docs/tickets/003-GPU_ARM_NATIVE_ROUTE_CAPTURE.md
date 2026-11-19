# 003: Capture gpu-arm's native routes — the proxy speaks OpenAI, gpu-arm does not

**Status:** OPEN
**Severity:** High (observability — 100% of gpu-arm traffic is uncaptured, including the decision traffic the calibration work depends on; no traffic loss, no wrong answers)
**Written by:** BUILD agent, this session, while arming capture on a restarted Kaggle GPU box
**Project of record:** `capture-proxy`
**Direction:** this repo owns the proxy and its route surface. gpu-arm
(`/kaggle/working/gpu-arm`) owns the routes being captured — nothing there
changes except that clients would point at this proxy instead of the runner
directly.

---

## 1. Summary

capture-proxy is now armed on every boot (`b53cf77`): config ensured, daemon
started, captures synced per instance. It then immediately demonstrated the
remaining gap: **it cannot see a single gpu-arm request**, because every route
it registers is OpenAI-shaped while every route that matters here is native.

| Route gpu-arm actually serves | Reachable on gpu-arm | Through capture-proxy |
|---|---|---|
| `POST /v1/systemone` (clef decision, ADR-021) | 400 — route live, body wrong | **404 Not Found** |
| `POST /v1/classify` | live | **404 Not Found** |
| `POST /v1/batch-classify` | live | **404 Not Found** |
| `POST /v1/chat/completions` | 400 tombstone (ADR-022) | forwards upstream |
| `POST /v1/completions` | n/a | forwards upstream — real 500 from clef |
| `POST /v1/classification` | n/a | forwards upstream — real 502 when no upstream |
| `POST /v1/embeddings` | 400 → Lemonade (embed retired) | forwards upstream — real 400 from clef |

The OpenAI routes do reach llama-server and do produce real captures (§2), but
`clef_flash` is a **decision** model: it refuses both OpenAI text routes
(`the current context does not support logits computation`) and embeddings
(`Pooling type 'none' is not OAI compatible`). So no amount of upstream
re-pointing captures the traffic worth capturing — the fix has to be a
**passthrough for gpu-arm's native routes**, not a backend change.

## 2. Evidence

Live on this box, 2026-10-13, after `bash bootstrap-capture.sh`:

```bash
capture-proxy --status
# -> running (pid …) port: 8000  upstream: http://127.0.0.1:8080  backend: llamacpp

curl -s -o /dev/null -w '%{http_code}\n' -X POST http://127.0.0.1:8000/v1/classify \
  -H 'Content-Type: application/json' -d '{"model":"finbert","text":"Revenue beat."}'
# -> 404   (route not registered)

curl -s -o /dev/null -w '%{http_code}\n' -X POST http://127.0.0.1:8000/v1/batch-classify \
  -H 'Content-Type: application/json' -d '{"model":"finbert","texts":["a"]}'
# -> 404

curl -s -o /dev/null -w '%{http_code}\n' -X POST http://127.0.0.1:8000/v1/systemone \
  -H 'Content-Type: application/json' -d "$BODY"
# -> 404   … while the SAME body direct to :8080 answers 400
#    ("state" must be provided) — i.e. the upstream route is alive and the
#    proxy is what refuses it.
```

Real traffic did reach the proxy and was captured, including its failures
(proof the arming half works end-to-end, not just that the process started):

```bash
curl -s -X POST http://127.0.0.1:8000/v1/chat/completions \
  -H 'x-session-id: spec-003-real-capture' -H 'x-user-id: gpu-arm-operator' \
  -d '{"model":"clef_flash","messages":[{"role":"user","content":"State the interest cover …"}],"max_tokens":120}'
# -> 500 {"error":{"code":500,"message":"the current context does not support logits computation. skipping"}}

cat /root/captures/errors/2026-10-13/*.jsonl | head -1
# -> model: clef_flash  upstream_status: 500  session: spec-003-real-capture  user: gpu-arm-operator
```

Upstream used for the probes, started with the entrypoint this repo already
publishes in gpu-arm's `CLEF_SERVE` (loopback `:8080`, never frpc-tunneled):

```bash
llama-server -m /kaggle/input/models/manarmilad/clef_flash/gguf/q8/1/Clef-Flash-Q8_0.gguf \
  --host 127.0.0.1 --port 8080 --tensor-split 15,15 -ngl 999 --ctx-size 8192 \
  --parallel 2 -kvu --cache-type-k q8_0 --cache-type-v q8_0 --flash-attn on
# -> "model loaded" after 4m27s (mmap from the Kaggle mount; RSS climbed to ~9.8 GB)
```

Loader facts used above, all read from the tree rather than assumed: registered
routes (`app.py:_register_routes`), forwards-once semantics per route, error
captures land in `errors/` with `upstream_status` and attribution
(`logger.py`, `session.py`), `model_tasks` keys match the **public** id
(`AGENTS.md`).

## 3. Why it matters

1. **The redundancy decision is blocked on this.** Several gpu-arm slugs are
   functional siblings (three financial-news sentiment heads, `finbert` vs
   `finbert-tone`, two CamelBERT sentiment models). Which to keep cannot be
   decided from model cards, and it is not decided by agreement alone either —
   it needs a captured slice of real traffic through each sibling.
2. **Decision traffic is the calibration surface.** `clef_flash` answers on
   `/v1/systemone`, and that route is precisely the one the proxy drops. Every
   criteria object, every verdict, every latency sample is emitted and lost.
3. **The gap is silent.** A 404 from the proxy and a 404 from gpu-arm look
   identical to a client. Nothing in the current wiring says "capture was
   configured and is working" except a human running `--status`.

## 4. What is being asked

1. **Passthrough for native JSON routes.** Register
   `POST /v1/systemone`, `/v1/classify`, `/v1/batch-classify`, `/v1/ner`,
   `/v1/batch-ner`, `/v1/timeseries`, `/v1/batch-timeseries` as body-verbatim
   forwards (the pattern `/v1/classification` already uses: forward, return the
   upstream response unchanged, capture both sides). `model_tasks` must be able
   to label them (`"finbert" → "classification"`, `"clef_flash" → "decision"`).
   **Multipart `/v1/transcribe` is explicitly out of scope for this ticket** —
   audio bodies are 32 MB and the capture record shape for them is a separate
   decision, not a freebie.
2. **Upstream must be gpu-arm, not llama-server.** The documented start shape
   becomes `CAPTURE_UPSTREAM_URL=http://127.0.0.1:8001` with the proxy on a
   different port (`:8000` free). One port out, one port in is preserved: this
   is loopback-only, and the proxy is never frpc-tunneled.
3. **Verification, stated as a postcondition:** a real `POST /v1/classify`
   through the proxy returns gpu-arm's own JSON, and
   `/root/captures/raw/<date>/` gains a line whose `request` holds the model +
   text and whose `response` holds `label`/`score`, with `task` populated from
   `model_tasks`. Process-started is not acceptance; the record on disk is.
4. **Record the boundary (one paragraph, README or AGENTS):** llama-server's own
   routes (`/completion`, `/systemone`, `/slots`) stay the backend's business;
   this proxy captures what clients send, and a client that bypasses the proxy
   is invisible. State that plainly so nobody assumes full coverage.
5. **Attribution.** gpu-arm clients must send `x-session-id` / `x-user-id`, or
   captures collapse to one pseudo-user. Same open question as 002 §4 item 2;
   do not resolve it twice — cross-reference 002.

## 5. Scope note

Filed during boot-arming work; the arming half (`b53cf77`, `2a42bef`) is
already landed and verified, and no gpu-arm source file was modified to file
this. Item 1 is proxy-side only. Item 5 is the same attribution question 002
already carries. The decision-model constraint documented in §1 is a property
of `clef_flash`, not a proxy defect — it is why re-pointing the upstream is
not sufficient.

## 6. Status log (append-only; the sections above are history)

- 2026-10-13 — filed after `b53cf77` armed capture on this box. §1's route table
  and §2's probes are live measurements from this machine, not reproduction
  from documentation. Real captures exist for the OpenAI-shaped routes
  (`raw/` and `errors/` lines with session/user attribution); the native routes
  remain at 404. Not started: item 1.