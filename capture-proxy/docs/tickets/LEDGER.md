# Ticket Attribution Ledger — capture-proxy

Append-only. Attribution of record: which mind actually **wrote** each ticket.
A receiving agent must be able to tell that a foreign agent filed the claim.

| Ticket | Written | Project | Agent |
|---|---|---|---|
| [EMO-001-headroom-capture.md](EMO-001-headroom-capture.md) | 2026-09-27 | unknown — predates this ledger, not recorded | unknown |
| [002-OPENJEV_ROUTER_CAPTURE_WIRING.md](002-OPENJEV_ROUTER_CAPTURE_WIRING.md) | 2026-10-01 | capture-proxy (filed from cross-repo investigation of `/root/openjev-serve`) | CEO agent, this session |
| [003-GPU_ARM_NATIVE_ROUTE_CAPTURE.md](003-GPU_ARM_NATIVE_ROUTE_CAPTURE.md) | 2026-10-13 | capture-proxy (filed while arming capture on a restarted Kaggle GPU box) | BUILD agent, this session |
| [003-GPU_ARM_NATIVE_ROUTE_CAPTURE.md](003-GPU_ARM_NATIVE_ROUTE_CAPTURE.md) | 2026-10-13 | capture-proxy (live /v1/systemone capture audit) | gpu-arm serving session |
| [004-PERTOKEN_ROUTING_CAPTURER.md](004-PERTOKEN_ROUTING_CAPTURER.md) | 2026-10-13 | blackfrost serving session (filed from verified per_token RCA + stat-mode success on CYBER-FROST-3.8-FP8) | BUILD agent, this session |
| [005-PROXY_SIDE_ROUTING_INJECTION.md](005-PROXY_SIDE_ROUTING_INJECTION.md) | 2026-10-13 | blackfrost serving session (live A/B: flagless traffic → 0 routing files, flagged request → sidecar fires) | BUILD agent, this session |

| [006-FINETUNE_USABLE_CAPTURE.md](006-FINETUNE_USABLE_CAPTURE.md) | 2026-10-15 | capture-proxy (filed while mining a 6.7 GiB mirrored corpus for a code-summarization finetune) | BUILD agent, this session |
| [007-SYSTEMONE_ROUTER_CAPTURE_WIRING.md](007-SYSTEMONE_ROUTER_CAPTURE_WIRING.md) | 2026-11-16 | capture-proxy (supersedes 002; corrected protocol name from "openjev" to "systemone", matching the actual endpoint `:3000/v1/systemone`) | BUILD agent, this session |

## Rules

- **Attribute the agent that wrote the file.** Name the subagent, not just the project.
- **Never rewrite a row.** Supersede with a new ticket and note the supersession in
  `INDEX.md`; the ledger is a history, not a status board.
- **One ticket per distinct need.** Dedup against `INDEX.md` before writing.
- The `EMO-001` attribution row is honest about what is unknown: backfilling a
  guess would be worse than a gap.
