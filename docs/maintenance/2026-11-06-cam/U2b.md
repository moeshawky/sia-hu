# CAM U2b — Carrier Persistence, Task Classification, Runtime, Entrypoint (read-only review)

## Inventory

All six in-scope files read in full; out-of-scope calls followed only for mechanism evidence.

| File | Lines | Role |
|------|-------|------|
| `src/capture_proxy/logger.py` | 428 | JSONL writer, index files, id sanitization |
| `src/capture_proxy/blobs.py` | 185 | Content-addressed store, canonical serializer |
| `src/capture_proxy/taskclass.py` | 559 | Structural task classification |
| `src/capture_proxy/runtime.py` | 650 | Per-backend run manifests |
| `src/capture_proxy/names.py` | 149 | Signal/tensor name contract |
| `src/capture_proxy/__main__.py` | 720 | Config, daemon/stop/status, uvicorn runner |

Cross-file verification performed: `app.py:_compute_sha256` delegates to `blobs.content_digest`→`canonical_bytes` (app.py:92-108); `app._persist_capture` wraps `log()`/`log_error()` in try/except OSError → None with one-time `_capture_write_failed` warning (app.py:2565-2570, 140-157); `CaptureProxyApp._resolve_task` reads `CAPTURE_MODEL_TASKS` per request (app.py:312-341, call sites 881/948/1203/1468/1630/1797/1945); `read_index` used only at app.py:2494.

## Findings

### U2b-01 — blobs.py dedupe race: non-atomic check-then-write can tear a blob
**file:line** `src/capture_proxy/blobs.py:161-165`
**severity** REQUIRED
**code** INV-BLOB-IDEMPOTENT (docstring claims idempotent writes at blobs.py:14-15, 76)

**evidence**
```
161:             if target.exists() and target.stat().st_size > 0:
162:                 return digest
163:             target.parent.mkdir(parents=True, exist_ok=True)
164:             target.write_bytes(payload)
```
Two concurrent identical requests both pass the `exists()` check, then both `write_bytes` (open "wb", truncate) to the same path. One can truncate mid-write of the other → torn/partial blob stored under a valid digest. `BlobStore.get` returns torn bytes; `has()` reports non-empty size. Capture line itself is unaffected (blob is a redundancy, blobs.py:18-19).

**mechanism** check-then-act with no temp-file + `os.replace`, no lock.
**why-not-intended** breaks the documented idempotency invariant under concurrency; contained to the redundant blob store (not capture truth).
**test_protocol_gap** no concurrent put() test; dedupe exercised only serially.
**suggested fix** write payload to a temp file in the same directory then `os.replace(temp, target)` for atomic, idempotent publication.

### U2b-02 — __main__.py --stop kills pid-file PID with no ownership guard (PID-reuse hazard)
**file:line** `src/capture_proxy/__main__.py:463-482` (kill at 469-470; unlink at 479-480); contrast start guard at 538-541
**severity** REQUIRED
**code** G-EDGE (stop hazard)

**evidence**
```
469:     try:
470:         os.kill(pid, signal.SIGTERM)
...
479:     pid_file.unlink(missing_ok=True)
```
`_stop` reads the PID from the file and SIGTERMs it. `_pid_alive` returns True for a foreign same-user process (PermissionError → True, `__main__.py:457-459`). A stale pid file whose PID has been reused by an unrelated live process gets SIGTERM'd. `_daemonize` guards start (538-541: refuses if `existing` alive) but `_stop` has no equivalent ownership/cmdline verification.

**mechanism** stop trusts pid-file contents without confirming the process is this proxy.
**why-not-intended** "--stop must not kill an unrelated PID" is an explicit anchor requirement; no guard exists on the stop path.
**test_protocol_gap** no test for stop against a stale/foreign pid.
**suggested fix** before SIGTERM, verify `/proc/{pid}/cmdline` (or `ps`) names this proxy; else refuse and report stale file.

### U2b-03 — logger.py index_filename docstring overclaims "no .. sequence"
**file:line** `src/capture_proxy/logger.py:230-254` (docstring lines 244-247; charset line 252)
**severity** SUGGESTION
**code** DOC-MISMATCH

**evidence**
```
244:           - Output contains no "/" or ".." sequence — safe to join under
...
252:         prefix = _INDEX_SAFE_RE.sub("_", identifier)[:_INDEX_PREFIX_LEN] or "empty"
```
`_INDEX_SAFE_RE = re.compile(r"[^A-Za-z0-9._-]")` (line 42) keeps `.`, so identifier `".."` yields prefix `".."` → filename `..-<hash>.jsonl`, which literally contains `..`. Not traversal (no `../`; full path is `index/sessions/..-hash.jsonl`, a literal filename), so contained, but the stated invariant is false for that input.

**mechanism** "." is in the safe charset, contradicting the docstring.
**why-not-intended** invariant as documented doesn't hold for identifier `".."`.
**test_protocol_gap** no index_filename test for `".."` / dots-only identifiers.
**suggested fix** align the docstring (dots permitted, traversal blocked via `/`→`_` and NUL→`_`) or exclude `.` from the safe charset.

## Rejected candidates

- **"logger.py needs a file lock / concurrent appends corrupt JSONL lines"** — REJECTED. Hot path is one `f.write(jsonl_line)` of a single escaped line (logger.py:221-223): `json.dumps` escapes embedded newlines as `\n`, `ensure_ascii=False` keeps UTF-8 but still one line, `O_APPEND` single `write()` under PIPE_BUF is atomic for regular files. No interleaving can split a line. Ruled out per failure-prevention rule (a): no naming of a corrupting interleaving is possible.
- **"taskclass.py vocabulary is a keyword discriminator → HIGH"** — REJECTED. `TASK_VOCABULARY` (taskclass.py:91-103) is never consulted by `classify_request`; it is only the closed output vocabulary written to schema.json. The sole lexical component `SUMMARY_INSTRUCTION_TOKENS` (taskclass.py:73-87) is reachable only from `_classify_code_summarize` (line 341), gated behind code-density + document-scaffold measurement (lines 328-338). No keyword list influences any label. Rule "vocabulary is never the discriminator" holds.
- **"runtime.py malformed command line yields wrong tp_size (silent misattribution)"** — REJECTED. `_command_output` (runtime.py:590-610) returns `""` on any subprocess error; `_running_vllm_args`/`_running_sglang_args` return `{}` or partial maps; `_int_or_none` (runtime.py:536-542) returns `None` for non-int values. Malformed CLI → omitted/`None` field, never a wrong parsed `tp_size`. "Best-effort" means absent, not incorrect.
- **"__main__.py config-precedence inversion"** — REJECTED. `_resolve_setting` (lines 296-309) is strictly CLI > env > config > default, verified for `upstream_url` (600-602), `port` (609), `backend` (606-608). No inversion.
- **"capture-disk write failure silently fails every request"** — REJECTED. `_write_record` raises OSError (logger.py:186-227); `app._persist_capture` catches it once and returns None (app.py:2565-2570); `_capture_write_failed` (app.py:140-157) emits a one-time named warning via `_capture_write_warned`. Request/forward still succeed; failure recorded once per process, not per request.
- **"names.py generated-name collision"** — REJECTED. `names.py` contains only static maps (`SIGNAL_KEYS`, `HEADROOM_SIGNALS`, `SIGNAL_NAME_MAP`, `MOE_MODEL_TYPES`, `ARCHITECTURE_TENSOR_NAMES`). It generates no names and exposes no collision surface; the map is total both directions (names.py:142-149).

## Invariant check (PASS/FAIL)

| Invariant | File:Lines | Result | Evidence |
|-----------|------------|--------|----------|
| Single JSONL line per record; records single-line | logger.py:221-223 | PASS | `json.dumps(...)+"\n"`, one `f.write`; newlines escaped |
| Capture-write failure does not fail request; recorded once | logger.py:186-227; app.py:140-157,2565-2570 | PASS | OSError → None; one-time `_capture_write_failed` warning |
| Index write failure never propagates | logger.py:280-322 | PASS | `except OSError → _index_write_failed` (one-time) |
| Index traversal-safe (`/`, NUL → `_`; no `../`) | logger.py:42,252-254 | PASS | charset excludes `/`; prefix never `..`/`../` |
| `read_index` exact-id filter, torn lines skipped | logger.py:345-357 | PASS | `entry.get(id_field) == identifier`; `JSONDecodeError → continue` |
| One canonical serializer for preimage & stored bytes | blobs.py:36-63,156-165; app.py:92-108 | PASS | `content_digest` and `BlobStore.put` both call `canonical_bytes`; `sort_keys=True` |
| Task vocabulary never a discriminator | taskclass.py:91-103,303-356 | PASS | `TASK_VOCABULARY` unconsulted; lexical token gated |
| Classification never raises on pathological body | taskclass.py:418-423,234-237,273-300 | PASS | non-dict → fallback; `_serialized_size` catches TypeError/ValueError |
| Best-effort manifest: malformed → omitted, not wrong | runtime.py:590-610,536-542 | PASS | failures → `""`/`None` |
| Config precedence CLI>env>file>default | __main__.py:296-309,600-609 | PASS | `_resolve_setting` chain |
| `--ensure-config` never overwrites | __main__.py:289-293 | PASS | `if path.exists(): return False` |
| `--daemon` doesn't reuse live foreign PID | __main__.py:538-541 | PASS | refuses if `_pid_alive(existing)` |
| `--stop` doesn't kill unrelated PID | __main__.py:463-482 | FAIL | U2b-02 — no ownership check |
| index_filename no `..` sequence | logger.py:244-247,252 | FAIL (doc only) | U2b-03 — `.` kept in charset |

## Contract check

Emitter/consumer impact: `N/A — single-repo diff` (no diff under review; this is a read-only audit of six carrier files). No cross-package boundary touched. `names.py` SIGNAL_KEYS/SIGNAL_NAME_MAP remain aligned with the authority's core-6 (SIGNAL_KEYS order: effective_rank, dead_neurons, load_balance, specialization, loss_saturation, growth_factor); CLI shorthand `loss`→`loss_saturation`, `isoflop`→`growth_factor` (names.py:142-149), consistent with AGENTS.md §7c reserved-null position (loss/isoflop/dead_neurons reserved null at proxy layer — not fabricated here).

## Gate results

Not run — read-only review per unit instructions. No ruff/format/mypy/pytest executed. No repo file modified. Absence probe N/A (this is the carrier tree, already cloned; `capture-proxy/AGENTS.md` corresponds to root AGENTS.md carrier sections already anchored).

## Verdict

APPROVE with REQUIRED. Two REQUIRED findings (U2b-01 blob dedupe race, U2b-02 stop PID-ownership guard) must be fixed before merge; one SUGGESTION (U2b-03 docstring). No BLOCKER. Rejected candidates: 6 (listed above). Coverage: all six in-scope files inventoried; every finding verified at file:line.
