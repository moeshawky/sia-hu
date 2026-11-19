# 005: Proxy injects `return_routed_experts` upstream when the routing sidecar is on

| Field | Value |
|---|---|
| Status | **FIXED** — injection code committed in `31eb860`; tests added in `test_routing.py` (6 tests, all passing) |
| Severity | HIGH (004 §4 item 4 cannot be measured by design without it) |
| Component | `src/capture_proxy/app.py` (request-forward path) + `src/capture_proxy/routing.py` (flag helper if extracted) |
| Supersedes / relates | 004-PERTOKEN_ROUTING_CAPTURER.md §3-4 (live-capture protocol assumes passive capture catches normal traffic) |

## 1. Summary

The routing sidecar is passive by design: it captures only responses that
carry routed experts, and SGLang attaches the blob only when the request sets
`return_routed_experts: true` (`protocol.py:896`, default `False`). Proxy
traffic from ordinary clients never sets the flag, so the sidecar produces
**zero routing files for normal traffic**. Ticket 004 §4 item 4 (measured
overhead / sustain-under-traffic) is unmeasurable under passive capture —
there is no routing data to measure.

Fix: when the routing sidecar is armed
(`_routed_experts_writer is not None`) and the route is `chat`/`completion`,
inject `"return_routed_experts": true` into the **upstream request body**
unless the client already set it. Client identity is preserved for logging
(raw/ capture keeps recording the body as received — see §4 INV-3); only the
forwarded payload gains the flag.

## 2. Evidence

- `protocol.py:896`: `return_routed_experts: bool = False` (ChatCompletionRequest).
- Live A/B, 2026-10-13, proxy on `:8001`, sidecar `on`, shape `48x10`:
  - request **without** the flag → completion `stop`, routing file count unchanged;
  - request **with** the flag → routing file appeared
    (`routing/2026-10-13/*.routing.json`).
- `grep return_routed_experts src/capture_proxy/*.py` → zero hits: the proxy
  neither injects nor forwards the flag itself; it is client-passthrough only.
- Server side already proven: capturer survives graph capture, returns
  per-token routing (004 §6). The missing piece is purely the request flag.

## 3. Why it matters

- 004 §4: "N tunnel requests → N routing files" — unreachable passively.
- Operator decision 2026-10-13: capture belongs at the unified server-facing
  proxy; clients must not be asked to know about telemetry (they neither
  should nor will remember to set a telemetry flag).

## 4. What is being asked

1. In the completion request handler, when `_routed_experts_writer is not
   None`, set `return_routed_experts = true` on the forwarded upstream body
   for `chat` and `completion` routes, non-streaming and streaming.
2. Client-sent flag wins when explicitly present (do not override `false`…
   but do force `true` when the sidecar is armed — see decision rule below).
3. Raw captures record the body as received; the injected flag is visible in
   the *upstream* request only (INV-3 deviation handling defined in §5).
4. Tests: injection on when armed; no injection when writer is None; flag
   not injected for embeddings/rerank/classification; streaming path
   unchanged byte-wise in the client-visible response.

## 5. Decision rule (INV-3 conflict)

`request_body_sha256` is computed on the forwarded body (app.py INV-4).
Injected flag changes that hash. Two options — pick one explicitly:
  (a) hash the **client-sent** body (raw identity wins, hash no longer
      proves byte-identity of the forwarded body); or
  (b) hash the forwarded body and note the sidecar flag as the only
      canonical diff (raw/ stays byte-identical to the response, request
      hash covers the injected payload).
Recommendation: (b) — raw/ already stores upstream response bytes; the hash
should describe what upstream saw.

## 6. Rollback

Delete the injection (one call site); sidecar returns to passive capture.

## 7. Status log (append-only)

- 2026-10-13 — Filed. Live A/B evidence in §2. Operator directive: proxy-side
  injection, clients untouched.
- 2026-11-16 — FIXED. Injection code already committed in `31eb860`
  (`app.py:905-920`): when `_routed_experts_writer is not None` and client did
  not set `ROUTED_EXPERTS_FLAG`, the forwarded body gains
  `return_routed_experts: true`; client-set flag never overridden; raw capture
  keeps client-sent body; hash covers forwarded body (option (b) per §5); both
  streaming and non-streaming share `_handle_completion_request`. Added 6 tests
  in `test_routing.py` covering: injection on when armed, no injection when
  writer None, no injection for embeddings/rerank/classification, streaming
  response byte-identical. All 26 test_routing.py tests pass; ruff check and
  format clean.
