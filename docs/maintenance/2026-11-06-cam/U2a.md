# CAM UNIT U2a — REVIEWER REPORT (READ-ONLY)

Repo: /kaggle/working/sia-hu (sia-hu, unified gate per AGENTS.md)
Scope: src/capture_proxy/headroom.py (1432), src/capture_proxy/routing.py (432), src/capture_proxy/schema.py (192)
Cross-checked: src/capture_proxy/app.py, models.py, names.py; src/model_headroom/models/loader.py
Docs read: AGENTS.md ("Bidirectional Contract", §7d), docs/architecture/contract.md, docs/capture-proxy/headroom-capture-design.md, docs/capture-proxy/PROJECT-ROADMAP.md (DEFER-2/3/4/6 + silent-failure history)
No repo file modified. No edits, no patches, no mutating commands.

## Inventory

- headroom.py — HeadroomWriter/HeadroomLogger/TensorBundle/TensorLoader; default safetensors+torch loader (lazy, CPU); 2-D weight-pair dump; byte cap; interval sampling; `_write_sync` writes weight file then sidecar last; all writes via asyncio.to_thread fire-and-forget; `write_async` never raises.
- routing.py — decode_routed_experts / extract_routed_experts_b64 / parse_routing_shape / RoutedExpertsWriter; LxK shape from env only; base64→native int32 row-major; malformed→None (silent skip), original b64 stays in raw/.
- schema.py — SCHEMA_VERSION=6; tri-state REQUIRED (13), NULLABLE_NOT_APPLICABLE (20), OPTIONAL_UNRECORDED (2); build/write/read manifest; OSError suppressed at write.

## Contract conformance table

| Contract bullet | Verdict | Evidence (file:line) |
|---|---|---|
| Emits {capture_id}.headroom.json + {capture_id}-weight_pairs.json in loader format | CONFORMS | headroom.py:1372-1373, 1430-1431; loader contract loader.py:353-406 |
| 2-D pair: gradient/weight/layer_name, gradient same shape as weight | CONFORMS | headroom.py:1139-1155 (pairs); loader.py:378-391 (requires 3 keys, 2-D) |
| Loader rejects non-2-D with named ValueError (file/layer/field/shape/ndim), no auto-slice | CONFORMS | loader.py:395-405 (raises ValueError naming all five); writer fans 3-D out to per-expert 2-D slices, never emits 3-D — headroom.py:738-754 |
| dtype/endianness/order match loader | CONFORMS | writer .numpy().tolist() → float32 nested lists row-major; loader torch.tensor(..., dtype=float32) — loader.py:382-383 |
| Five reserved keys null except effective_rank | CONFORMS | headroom.py:1366 dict.fromkeys(SIGNAL_KEYS) all None; only effective_rank/load_balance/specialization set (1384, 1403, 1414); dead_neurons/loss_saturation/growth_factor never set | names.py:118-125 SIGNAL_KEYS; SIGNAL_NAME_MAP loss→loss_saturation, isoflop→growth_factor (names.py:147-148) |
| Eligible: chat/completion/classification/image; embeddings+rerank excluded at BOTH call sites AND request_type filter | CONFORMS | filter app.py:2406; call sites app.py:1023 (completion), 1651 (classification), 1816 (image), 1966 (native-passthrough—guard returns early, docstring app.py:1884-1885), 2357 (stream); embedding ~1069 and rerank ~1338 handlers call neither _spawn_headroom nor pass eligible request_type |
| Post-response, async, never on response path | CONFORMS | app.py:2457-2467 asyncio.create_task(write_async); headroom.py:943-947 asyncio.to_thread(_write_sync); spawn before _log_capture/return (1023-1025); no await on response path |
| Off-mode purity: no torch/safetensors/model_headroom import, zero bytes | CONFORMS | app.py:494-496 returns None before any import; names.py imports std only (names.py:6-8) |
| CAPTURE_HEADROOM_WEIGHTS off|gate|all | CONFORMS | app.py:455-473 _parse_headroom_weights; headroom.py:827-831 |
| CAPTURE_HEADROOM_MAX_BYTES float32 cap; never mid-tensor (clean skip/refuse) | CONFORMS | headroom.py:646-667 (_tensor_to_bundle: rows_fit<1 → None skip); 1119-1129 (pair-level backstop: skip when rows_fit<1, truncated=True); truncates whole rows only |
| CAPTURE_HEADROOM_INTERVAL: 0 = every capture; counter process-global | CONFORMS | headroom.py:884-902 (interval<=1 → every call; else counter%interval==0; self._counter instance attr on HeadroomWriter singleton) |
| load_balance / specialization artifacts match authority loader contract | CONFORMS | activations: headroom.py:234-244 (metadata+layers) matches loader.py:315-333 (load_activations_from_json); expert_weights: headroom.py:1265-1267 (layers[layer_name, expert_weights>=2]) matches loader.py:569-598 (load_expert_weights_from_json) |
| Routing requires CAPTURE_ROUTED_EXPERTS_SHAPE LxK; invalid → warns, stays off; shape never hardcoded | CONFORMS | routing.py:63 _SHAPE_RE ^(\d+)x(\d+)$; 188-207 parse_routing_shape returns None for malformed/degenerate; app.py:570-578 fail-open to off with warning |
| Routing decode: malformed base64 / wrong length / token mismatch → None (silent skip), not garbage | CONFORMS | routing.py:112-131 ValueError on invalid b64 / not int32-aligned / length not multiple of num_layers*topk; 385-391 _write_sync catches ValueError → None; original b64 stays in raw/ |
| schema.json tri-state vs CaptureRecord fields | VIOLATES | 36 CaptureRecord fields (models.py:283-320); classified 13+20+2=35; `attempt_number` (models.py:300) in NONE of the three lists → 1 unclassified field (schema.py:31-80) |

## Deferral re-adjudication (DEFER-2/3/4/6 + silent-failure history)

All four DEFER items are FIXED per the roadmap (2026-10-12) AND verified still present in source — none are stale, none re-deferred by inertia.

- DEFER-2 (fused 3-D → per-expert 2-D pairs): FIXED. headroom.py:_default_tensor_loader_all fans 3-D [E,*,*] into one 2-D bundle per expert slice sharing byte budget cumulatively (headroom.py:738-754); fused tensor not split into gate/up halves; 1-D/unsupported ranks return [], never raise (closes latent 1-D TypeError). Still fixed.
- DEFER-3 (DeBERTa-v2 / RoBERTa resolve): FIXED. names.py:96-110 dense templates for deberta./roberta. prefixes read off library module structure; BERT precedence locked by test. Still fixed.
- DEFER-4 (every pair carries expert_id): FIXED. headroom.py:1144 expert_id on every pair (parsed from indexed name, per-slice index for fused 3-D, None for dense); additive key, loader reads via .get() so 2-D contract untouched. Still fixed.
- DEFER-6 (real-I/O test): FIXED. capture-proxy/tests/test_headroom_real_io.py exists (14 tests, real .safetensors fixture in tmp_path); skips cleanly without headroom group; bare chain stays green. Still fixed.

Silent-failure history: the recorded silent failures (weight-capture dead repo-wide, D2 resolution miss, torch/safetensors-unavailable) are FIXED. However, write-failure graceful degradation remains by design (R-09): _write_sync catches (OSError, ValueError, TypeError) → None (headroom.py:1431-1432); write_async catches all Exception → None (headroom.py:946-947); _on_task_done suppresses exceptions (headroom.py:967-971). A disk/permission write failure yields a sidecar-less capture, a ref pointing at a missing file, no warning, and no error row in raw/ or errors/. This is the documented contract, not a regressed defect — but it IS still a silent failure mode.

## Findings

### U2a-01 — Silent write failure produces sidecar-less capture with no warning and no error row
file:line: headroom.py:946-947, 1431-1432, 967-971
severity: LOW (by design per R-09; "a missing warning is LOW")
code: G-ERR
evidence:
```
        except Exception:
            return None          # write_async, headroom.py:946-947
        except (OSError, ValueError, TypeError):
            return None          # _write_sync, headroom.py:1431-1432
        with contextlib.suppress(Exception):
            task.exception()     # _on_task_done, headroom.py:967-971
```
mechanism: A write failure (disk full, permissions) makes write_async return None; _on_task_done retrieves and suppresses the exception; no error row is written to raw/ or errors/; the HeadroomRef in the JSONL line points at a file that will never exist.
why-not-intended: R-09 deliberately trades capture reliability for response-path safety ("never raises, proxy response untouched"); the silent-failure history recorded in the roadmap referred to specific DEFER defects (now fixed), not this graceful-degradation contract.
test_protocol_gap: No test asserts that a forced write failure surfaces a warning or error row; off-mode purity is tested but write-failure observability is not.
suggested fix: Either document silent write failure as intended (R-09) in AGENTS.md/headroom.py, or emit an errors/ row + stderr warning when a background write fails.

### U2a-02 — `attempt_number` is an unclassified CaptureRecord field (not in any schema.json tri-state list)
file:line: models.py:300 (field); schema.py:31-80 (lists)
severity: LOW
code: CONTRACT
evidence: 36 CaptureRecord fields vs 13 REQUIRED + 20 NULLABLE + 2 OPTIONAL = 35 classified; `attempt_number: int = 1` appears in none. It is present on every record (default 1) but undeclared, so schema.json cannot distinguish "recorded" from "absent" for it — the very distinction §7d makes load-bearing.
mechanism: build_schema_manifest emits only the three lists (schema.py:120-122); read_schema_manifest returns them as the contract; attempt_number is invisible to the manifest.
why-not-intended: §7d requires every field to be in exactly one of the three classes so "null" and "absent" are distinct claims; attempt_number breaks that invariant.
test_protocol_gap: No test cross-checks CaptureRecord field set against schema.py tri-state lists.
suggested fix: Add `attempt_number` to REQUIRED_FIELDS (it always has a default of 1) in schema.py, or to OPTIONAL_UNRECORDED if absence is meaningful.

### U2a-03 — Routing decode malformed/mismatched input silently skips (None), not garbage artifact
file:line: routing.py:112-131 (decode raises ValueError), 385-391 (_write_sync catches → None)
severity: N/A — conforms (fail-open by design)
code: —
evidence: Invalid base64 / non-int32-aligned length / length not multiple of num_layers*topk all raise ValueError; _write_sync returns None; original base64 remains in raw/ for audit.
mechanism: decode failures never propagate; no artifact written; no warning emitted beyond the shape-validation warning at app.py:573-576.
why-not-intended: Intentional fail-open — a bad tensor must not corrupt the capture or fail the request.
test_protocol_gap: None (conforms).
suggested fix: None needed; confirmed conformance only.

## Rejected candidates

- "Sidecar is off by default is a gap" — REJECTED. Design law (AGENTS.md BANNED §a, R-01): default-off is the design; zero bytes on hot path.
- "Missing score in the carrier" — REJECTED. Carrier never scores (Iron Law AGENTS.md:19-20); authority owns scoring.
- "streaming path awaits headroom before returning" — REJECTED. _spawn_headroom at app.py:2357 runs in stream_generator finally via asyncio.create_task fire-and-forget; no await (app.py:2457-2467).
- "ELIGIBILITY leak: rerank/embedding call _spawn_headroom" — REJECTED. Verified: embedding (~1069) and rerank (~1338) handlers contain no _spawn_headroom call; line 1966 is _handle_native_passthrough whose request_type guard (app.py:2406) returns early for ineligible native types (docstring app.py:1884-1885).
- "CAPTURE_HEADROOM=loss writes non-null into reserved key" — REJECTED. CLI "loss" maps to emitted "loss_saturation" (SIGNAL_NAME_MAP names.py:147), but _write_sync never writes loss_saturation/growth_factor/dead_neurons (only effective_rank/load_balance/specialization are set at 1384/1403/1414); reserved keys stay None.
- "Byte cap truncates mid-tensor / corrupts artifact" — REJECTED. Cap slices whole rows only (_tensor_to_bundle headroom.py:652-657); rows_fit<1 → None skip (headroom.py:656-657); pair-level backstop skips when rows_fit<1 (headroom.py:1123-1129); truncated=True flag preserved.
- "INTERVAL=0 means never / counter is per-request" — REJECTED. interval<=1 → every call; self._counter is HeadroomWriter instance attr (process-global singleton) (headroom.py:884-902).
- "3-D fused expert tensor auto-sliced" — REJECTED BLOCKER-pattern. Writer fans 3-D into per-expert 2-D slices (never emits 3-D); loader rejects any non-2-D with named ValueError (loader.py:395-405). No silent slice.
- "routing shape hardcoded / malformed shape writes garbage" — REJECTED. Shape comes only from CAPTURE_ROUTED_EXPERTS_SHAPE (routing.py:63, app.py:570-578); malformed → parse_routing_shape returns None → writer off with warning; no hardcoded default.
- "loss/isoflop reserved keys violated" — REJECTED. CLI names map via SIGNAL_NAME_MAP; emitted SIGNAL_KEYS uses loss_saturation/growth_factor which are never written (stay None).

## Coverage statement

Read in full: headroom.py (all 1432 lines, two reads), routing.py (all 432), schema.py (all 192), AGENTS.md (all 550), app.py boundary sections (_spawn_headroom 2372-2467, _build_headroom_writer 475-540, _build_routed_experts_writer 543-588, _spawn_routing_capture 590-643, call sites 1023/1651/1816/1966/2357, embedding ~1069 & rerank ~1338 confirmed no call), models.py CaptureRecord/HeadroomRef/CapturedHeadroom (85-324), names.py (all 149), loader.py load_weight_pairs_from_json/load_activations_from_json/load_expert_weights_from_json (315-600), test_headroom_real_io.py (DEFER-6 evidence). Contract table complete; all four DEFER items re-adjudicated from source; rejected candidates non-empty.

## Verdict: APPROVE with REQUIRED

No BLOCKERs. U2a-02 (unclassified `attempt_number`) is REQUIRED — one-line fix: add `attempt_number` to REQUIRED_FIELDS in schema.py. U2a-01 is LOW/by-design (document or add errors/ row). Contract table complete; deferrals all FIXED and re-verified; exclusion and async invariants CONFORM. Re-run root verify chain (`ruff check . && ruff format --check . && mypy src/ && pytest`) if any change is made.
