# 004: Per-token MoE routing telemetry via the routed-experts capturer fallback

| Field | Value |
|---|---|
| Status | **OPEN** — filed 2026-10-13 |
| Severity | HIGH (mission telemetry gap: aggregate counts work, per-token routing does not) |
| Component | `src/capture_proxy/` (proxy side) + SGLang launch profile (server side, `sglang-tlmstat` successor) |
| Target model | `Blackfrost-AI/CYBER-FROST-3.8-FP8` (`qwen4_exp`, 512 experts, top-10, 48 layers), served from `/root/blackfrost` on MI300X via `lmsysorg/sglang-rocm:v0.5.21-rocm724-mi30x-20261008` |
| Blocks | Continuous per-token MoE routing telemetry (Phase B of the serving mission) |

## 1. Summary

The native per-token recorder (`--expert-distribution-recorder-mode per_token`)
is **verified dead** on the frozen serving image for this artifact (RCA below —
not a suspicion, a traceback). The `stat` recorder works and currently yields
aggregate per-expert counts (`logical_count [1000, 48, 512]`, 24,248 nonzero,
verified 2026-10-13), but aggregates cannot answer per-token routing questions.

This ticket asks for the fallback path: enable SGLang's `RoutedExpertsCapturer`
(`--enable-return-routed-experts`), which returns the top-k expert ids per token
in the response `meta_info`, and extend the proxy to persist those per-token
routing vectors alongside the captures it already writes. The capturer is
width-safe by construction for this artifact (`topk_size` comes from the model
config's `num_experts_per_tok=10`, not a hardcoded 8) — but whether its hook
survives decode graph capture is **unproven** and must be established by a
proving launch before any proxy work depends on it.

## 2. Evidence

- `Blackfrost-AI/CYBER-FROST-3.8-FP8`, 48 layers, 512 experts, `num_experts_per_tok=10`
  (artifact `config.json`, read 2026-10-13).
- Telemetry launch with `per_token` died 2026-10-13 08:46:43 during decode graph
  capture. First exception in the chained traceback
  (`/root/blackfrost/logs/tlm-20261013T084247Z.log:195-253`):
  `eplb/expert_distribution.py:499, in on_select_experts` —
  `RuntimeError: expanded size (8) vs existing (10), target [6,8], tensor [6,10]`.
  The `hipErrorStreamCaptureUnjoined` / "Capture cuda graph failed" error is the
  downstream casualty (exception poisoned the capture stream), not the cause.
- Installed source (frozen image, read 2026-10-13):
  `_DetailSinglePassGatherer._TOP_K_NUM = 8` ("DeepSeek V3 has this value;
  should generalize later", `expert_distribution.py:461-462`); the hook runs
  during stream capture regardless of recording state (`_on_hook`, the
  `or is_current_stream_capturing()` clause), so the kill is deterministic —
  retrying `per_token` on this image cannot pass.
- Control: frozen baseline (`recorder_mode None`) passes graph capture on the
  same host, twice (07:15, 09:00). Only delta was the recorder flag.
- `stat` mode proving launch 2026-10-13 (`sglang-tlmstat`, profile
  `/root/blackfrost/scripts/launch.CKPT-full-262k-telemetry-stat.sh`): 0 graph
  failures, START → generation → STOP → DUMP lifecycle works,
  `expert_distribution_recorder_*.pt` (98 MB) contains `logical_count
  [1000,48,512]` int32 with 24,248 nonzero. Aggregate path closed; per-token
  path still open — this ticket.
- Capturer width-safety (installed source, read 2026-10-13):
  `RoutedExpertsCapturer` takes `topk_size =
  model_config.hf_text_config.num_experts_per_tok` (=10 here);
  `BaseDeviceCache.capture` writes `buffer[:batch, layer_id, :] = topk_indices`
  with no width hardcode. Capture-during-graph behavior: no guard either way
  in source — UNVERIFIED, hence item 1 below comes first.

## 3. Why it matters

Stat aggregates answer "which experts are hot." They cannot answer "which
experts served this token" — the routing question the telemetry mission exists
for (load-balance validation, specialization analysis, per-request attribution).
The proxy already captures every request/response body; routing vectors landing
in responses are a small, well-typed addition with large downstream value
(`ExpertActivationData`-shaped, per the headroom sidecar's reserved slots —
cf. EMO-001 §"Why this proxy is still worth fixing").

## 4. What is being asked

1. **Proving launch (server side, needs downtime coordination with the
   operator):** launch a profile identical to frozen baseline plus only
   `--enable-return-routed-experts`; confirm the server reaches ready (graph
   capture survives the capturer hook) and that a completion response carries
   `meta_info.routed_experts`. If capture fails the same way, STOP — two
   same-signature failures is a redesign signal, not a retry signal — and
   record the traceback here.
2. **Proxy persistence:** decode `routed_experts` from `meta_info` (SGLang
   ships `extract_routed_experts_from_meta_info`, base64 int32) and persist
   per-token per-layer expert ids keyed by the existing capture correlation id.
   Verify 48 layers × top-10 coverage on a probe request.
3. **No regression:** frozen serving profile (`launch.CKPT-full-262k.sh`) and
   inference parity untouched; telemetry profiles live in their own files.
4. **Report back:** dump shape, layer coverage, per-request overhead, and
   whether the capturer path sustains the tunnel's normal traffic — as measured
   numbers, not adjectives.

## 5. Scope note

No SGLang source edits (a local engine patch is banned by the serving
contract; the real per_token fix — generalizing `_TOP_K_NUM` — belongs
upstream or in a newer image, not here). No serving-flag changes beyond item
1's single flag. Prompt PII discipline carries over: routing ids are safe to
store; prompt text handling follows existing proxy policy.

## 6. Status log (append-only; the sections above are history)

- 2026-10-13 — Filed (serving session). Native per_token RCA VERIFIED;
  stat aggregate path VERIFIED working; capturer fallback width-safe but
  capture-interaction UNVERIFIED. Awaiting proving launch (item 1).
- 2026-10-13 — Item 1 PROVEN (serving session, profile
  `launch.CKPT-full-262k-sharper-capturer.sh` = operator template + the one
  flag): server ready 12:32, **0 graph failures**; `HostCache (258816,48,10)` +
  `DeviceCache (16384,48,10)` allocated. Probe with
  `"return_routed_experts":true` returned per-token routing; decoded payload
  = **33 tokens × 48 layers × top-10**, ids in [0,511], all 512 experts
  touched. CORRECTION to item 2: the server delivers the blob under
  top-level **`sglext.routed_experts`**, NOT `meta_info.routed_experts`
  (both chat and completions; `meta_info` carries nothing). The proxy's
  `extract_routed_experts_b64` currently scans only `meta_info` (+ per-choice
  `meta_info`) and will capture NOTHING until it also checks `sglext`.
  Required proxy fix: check `response_body["sglext"]["routed_experts"]` first.
- 2026-10-13 — Proxy correction LANDED (per §6 item-2 correction):
  `extract_routed_experts_b64` checks `response_body["sglext"]["routed_experts"]`
  FIRST — `src/capture_proxy/routing.py:155` — meta_info and per-choice
  meta_info scans kept as forward-compat fallbacks; malformed sglext falls
  through, extraction never raises on the response path. Probe
  (`scripts/verify_routed_experts.py:112`) reads the same extractor; exit-3
  wording records joint absence from sglext AND meta_info as the
  proving-launch failure signature. Tests: 6 new extractor tests
  (sglext-present, precedence, legacy meta_info fallback, dual-absence →
  None, malformed sglext → None, bad-base64 → decode-error-not-absence);
  rollback-verified (pre-patch snapshot → 3 sglext tests fail; re-apply →
  7/7 pass). Verify chain in capture-proxy/: `ruff check .` All checks
  passed; `mypy src/` no issues (13 files); `pytest -q` 632 passed,
  17 skipped. Flagged (not edited, out of this correction's scope):
  `app.py:454` `_spawn_routing_capture` docstring and the AGENTS.md
  `CAPTURE_ROUTED_EXPERTS` env table still say `meta_info.routed_experts`
  only. TICKET REMAINS OPEN — §4 item 4 (measured per-request overhead /
  sustain-under-traffic numbers) is not done.
- 2026-10-13 — Renumbered 003→004: slot 003 collides with origin's 003-GPU_ARM_NATIVE_ROUTE_CAPTURE.md (fetched 2026-10-13); no history above rewritten (§§1-5 frozen, §6 append-only).
