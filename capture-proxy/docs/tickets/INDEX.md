# Ticket Index — capture-proxy

One row per ticket, oldest first. The `Why it was needed` column is the short claim;
the ticket file carries the evidence. Slot `001` is held by `EMO-001`; numbering
continues `002, 003, …` from there.

| Added | Ticket | Why it was needed | Status |
|---|---|---|---|
| 2026-09-27 | [EMO-001-headroom-capture.md](EMO-001-headroom-capture.md) | Headroom sidecar wrote no weight pairs for `allenai/Emo_1b14b_1T` (MoE gate miss + a repo-wide loader bug that made weight capture dead for every model) | FIXED — F1+F2 landed + loader fix, verified 2026-09-27 by 3-pass chain |
| 2026-10-01 | [002-OPENJEV_ROUTER_CAPTURE_WIRING.md](002-OPENJEV_ROUTER_CAPTURE_WIRING.md) | The systemone decision stack (served by openjev) is live but direct-wired (shim → router), so zero decision traffic is captured; wire it through the proxy and settle the monitoring gaps | SUPERSEDED by 007 — openjev serves systemone format; every openjev decision call is `:3000/v1/systemone`, covered by ticket 007; items 3, 4 landed (`6048c4a`, §§7a/7b); items 1, 2 blocked (no `/root/openjev-serve` on this machine); see ticket §6 |
| 2026-10-13 | [003-GPU_ARM_NATIVE_ROUTE_CAPTURE.md](003-GPU_ARM_NATIVE_ROUTE_CAPTURE.md) | The proxy registers only OpenAI-shaped routes, so 100% of gpu-arm traffic 404s through it (`/v1/systemone`, `/v1/classify`, `/v1/batch-classify` measured); clef refuses the OpenAI routes, so re-pointing the upstream cannot capture it | OPEN — filed 2026-10-13 with live evidence; arming half landed in `b53cf77`; item 1 not started |
| 2026-10-13 | [004-PERTOKEN_ROUTING_CAPTURER.md](004-PERTOKEN_ROUTING_CAPTURER.md) | Native per-token MoE recorder is verified dead on the frozen SGLang image for the top-10 artifact (width-8 hardcode kills graph capture); stat aggregates work — persist per-token routing via the capturer fallback | OPEN — §6 item 1 PROVEN 2026-10-13 (0 graph failures, 33×48×10) + proxy correction LANDED (sglext-first, 7/7 tests); §4 item 4 (overhead/sustain numbers) not done; renumbered 003→004 2026-10-13 (slot collision with origin 003-GPU_ARM) |
| 2026-10-13 | [005-PROXY_SIDE_ROUTING_INJECTION.md](005-PROXY_SIDE_ROUTING_INJECTION.md) | Routing sidecar is passive — ordinary clients never send `return_routed_experts`, so normal traffic yields zero routing files and 004 §4 item 4 is unmeasurable; proxy must inject the flag upstream when the sidecar is on | FIXED — injection code committed in `31eb860`; 6 tests added in `test_routing.py`, all passing |

| 2026-10-15 | [006-FINETUNE_USABLE_CAPTURE.md](006-FINETUNE_USABLE_CAPTURE.md) | `task` is resolved from public model id (`_resolve_task(model)`), so it labels the model not the request — 0.0% populated across 582 captures; and `.stream.jsonl` (46% of raw) stores no request body, so the corpus cannot be mined for finetuning | FIXED (code) — `33d4edd`, `74bceab`, `10707b5`; fresh-mirror re-run still pending. See §9 for two corrected figures |
| 2026-11-16 | [007-SYSTEMONE_ROUTER_CAPTURE_WIRING.md](007-SYSTEMONE_ROUTER_CAPTURE_WIRING.md) | Corrects ticket 002's protocol naming: the traffic being captured is `:3000/v1/systemone` (System One / Jev decision-model protocol), not "openjev" (the stack name). Wiring shim → proxy → router for systemone decision traffic; two env changes, zero code. | OPEN — supersedes 002 (`002-OPENJEV_ROUTER_CAPTURE_WIRING.md`); items 1, 2 blocked (no `/root/openjev-serve` on this machine); items 3, 4 landed proxy-side in `6048c4a`. See ticket §6 for full status log. |

**Read discipline:** this folder is checked when the operator names a ticket or asks
for a triage. No proactive sweeps.

**Write discipline:** one ticket per distinct need. Dedup against this index before
filing. Never rewrite a row — supersede with a new ticket and note it here.
