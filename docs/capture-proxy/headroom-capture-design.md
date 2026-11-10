# Design: Headroom Capture Sidecar (ADR-001)

**Companion to** `docs/headroom-capture-proposal.md` (measured intent) — this file is the
contract-first design that implements it. Status: approved for implementation 2026-08-21.

## 1. Requirements (Sekel — numbers, no adjectives)

| ID | Requirement | Acceptance (measurable) |
|----|-------------|-------------------------|
| R-01 | Opt-in sidecar, off by default | `CAPTURE_HEADROOM=off`: 0 files under `headroom/`, 0 `torch`/`safetensors` imports, 0 bytes on hot path (sys.modules spy + dir listing test) |
| R-02 | Emitted files consumable by `model_headroom` | `*weight_pairs.json` passes `loader.py:398` `load_weight_pairs_from_json` contract: every pair has `gradient`, `weight`, `layer_name` keys, 2-D float arrays, `gradient` same shape as `weight` |
| R-03 | `CaptureRecord` additive bump 4→5 | `schema_version=5` default; `headroom: HeadroomRef \| None = None`; v4 records (no key) still validate; `model_dump()` of v5 line contains `headroom` |
| R-04 | Size control never OOMs | `CAPTURE_HEADROOM_MAX_BYTES` bounds float32 payload bytes (`rows*cols*4`); 256M cap on 5120×17408 gate → 3855-row truncation (3855×17408×4 = 268,431,360 B ≤ 268,435,456 B); `truncated=true` + `error_body` note; `max_bytes=0` unlimited |
| R-05 | Sampling | `CAPTURE_HEADROOM_INTERVAL=N` → exactly 1 write per N eligible captures (counter test: 5 requests, N=2 → 2 or 3 writes bounded by floor/ceil 2±1; deterministic) |
| R-06 | Post-response async, no critical-path await | `_spawn_headroom` wall time < 5 ms (p99 test on this box); `asyncio.create_task` fire-and-forget; streaming tee bytes unchanged (existing INV-3 lossless tests stay green) |
| R-07 | Config precedence CLI > env > file > default | `--headroom` beats `CAPTURE_HEADROOM` beats `config.toml headroom` beats `off` (precedence test) |
| R-08 | Configurable location, no hardcoded paths | `CAPTURE_HEADROOM_DIR` default `{CAPTURE_BASE_PATH}/headroom`; never literal `/root/captures` in code |
| R-09 | Failure containment | Missing model dir / missing safetensors / OSError → `write_async` returns `None`, never raises, proxy response untouched; OSError suppressed at call site (existing `contextlib.suppress(OSError)` pattern) |
| R-10 | Tensor read path | Tensors are read from safetensors via `safe_open(..., framework="pt")` + `.float()` to CPU for JSON serialization (carrier side); no TPU/XLA import; fp8 dequant via `weight_scale_inv` side tensor when present. The authority's NFR-01 (accelerator-preferred, CPU fallback) governs where the authority computes on these tensors — see `src/model_headroom/device.py`. |

**NFR-01** (carried from proposal, now rewritten): accelerator-preferred weight loading, CPU as fallback — see `src/model_headroom/device.py` (`available_accelerators()`, `default_device()`, `resolve_device()`). The carrier reads tensors to CPU for JSON serialization (R-10 above); the authority computes on the resolved device. vLLM HBM is untouched by the carrier.

## 2. ADR-001: fire-and-forget task + precomputed deterministic sidecar path

- **Context:** `record.headroom` must appear in the JSONL line, but the line is written
  synchronously (`_log_capture`) while tensor extraction takes seconds and must not be
  awaited on the response path (R-06). JSONL is append-only (INV-2) — the line cannot be
  rewritten after the write completes.
- **Decision:** precompute both sidecar paths deterministically from `capture_id`
  (`{dir}/{YYYY-MM-DD}/{capture_id}.headroom.json`, `{capture_id}-weight_pairs.json`),
  put the `HeadroomRef` in the record before `_log_capture`, then
  `asyncio.create_task(writer.write_async(...))` which materializes the files
  (weight file first, sidecar last — sidecar is the authoritative artifact).
- **Consequences:** on a failed background write the ref points at a missing file.
  Documented invariant on `HeadroomRef`: *path is precomputed; absence of the file means
  the background write did not complete; the sidecar is authoritative.* Alternative
  rejected: synchronous await in the streaming `finally` delays stream-end for the client
  by the extraction duration (violates R-06); separate daemon (proposal §9) loses
  per-request correlation.
- **ADR-002 (loader contract):** `model_headroom` `loader.py:398` skips any pair missing
  `gradient`/`weight`/`layer_name`, and `torch.tensor(None)` raises `TypeError` caught at
  file level → whole file `None`. Therefore weight-only capture emits a **real zero
  gradient array** of the weight's shape (R-03 effective rank uses weight only; R-04
  ratio is 0.0 for zero gradient — clean semantics). The proposal's "gradient null"
  wording is superseded by this measurement.
- **ADR-003 (tensor names):** measured `weight_map` of the target model uses
  `model.language_model.layers.{i}.mlp.{proj}_proj.weight` (Qwen3_5 conditional
  generation wrapper), not the proposal example's `model.layers.{i}...`. Resolution tries
  candidate templates in order (`ARCHITECTURE_TENSOR_NAMES`), first match in the
  safetensors index wins.
- **ADR-004 (lazy imports):** proxy venv (3.14.6) has no `torch`/`safetensors`. They are
  imported only inside the default tensor loader at write time; absence → `None` (R-09).
  No `pyproject.toml` dependency change. Tests inject a fake via the `tensor_loader`
  constructor seam.

## 3. Typed contracts

### `HeadroomRef` (models.py — written into `CaptureRecord.headroom`)
```
schema_version: int = 5
capture_id: str                      # validator: must parse as UUID, else ValidationError
parent_raw: Path                     # planned raw JSONL path (plan_path(), single-sourced)
model: str                           # served model name from the request
model_path: str                      # model dir from runtime_settings (must exist at spawn)
architecture: Literal["dense","moe","unknown"] = "unknown"
signals: dict[str, str | None]       # signal name -> artifact path or None (6 keys)
```
Sidecar file schema = `CapturedHeadroom(HeadroomRef fields)` +
`collected_at: str`, `bytes_written: int = 0`, `bytes_cap: int = 0`,
`truncated: bool = False`, `error_body: str | None = None`.

> **Superseded (2026-09-26):** `schema_version` is now `6` and both `HeadroomRef`
> and `CapturedHeadroom` carry an additive `task: str | None = None`. The bump was
> additive — v5 lines without the `task` key still validate.

### `HeadroomWriter` (headroom.py — new module)
```
__init__(base_dir: Path, weights: "off"|"gate"|"all", max_bytes: int,
         interval: int, signals: frozenset[str],
         tensor_loader: TensorLoader | None = None)
    # weights: MoE -> gate or gate+up+down; dense -> attention or
    #   attention+intermediate+output, MoE projs as fallback
    # constraints: max_bytes >= 0 (0 = unlimited), interval >= 0 (0 = every)
should_sample() -> bool              # 1/N counter; interval <= 1 -> always
write_async(*, capture_id: str, parent_raw: Path, model: str,
            model_path: Path) -> Path | None
    # returns: sidecar Path when written; None when model_path missing,
    #   safetensors/torch unimportable (default loader), or write failed
    # never raises; runs _write_sync via asyncio.to_thread
```
`TensorLoader = Callable[[Path, str], TensorBundle | None]`,
`TensorBundle = NamedTuple(weight: list[list[float]], gradient: list[list[float]])`.
Default loader: `model.safetensors.index.json` (or single-shard fallback) →
`safe_open(shard, framework="pt")` → `get_tensor(name)` → `weight_scale_inv` dequant →
`.float()` → `.numpy().tolist()` + zero gradient. Any exception → `None`.

### `HeadroomLogger` (headroom.py — CaptureLogger fork)
```
__init__(base_dir: Path)
write_json(path: Path, payload: dict[str, Any], *, compact: bool = False) -> Path
    # mkdir -p on first write; JSON line/file; raises OSError (callers suppress)
```

### app.py seam
`_spawn_headroom(record) -> HeadroomRef | None` — eligibility: writer enabled,
`request_type in {chat, completion}`, valid-UUID `capture_id`, `upstream_status < 400`,
`error_body is None`, `runtime_settings.model_path` is an existing dir, `should_sample()`.
Sets `record.headroom`, spawns task. Called from `_handle_non_streaming` (before return)
and `stream_generator` `finally` (after `_log_capture`). `_log_capture` now returns
`Path | None` (raw/errors JSONL path; `None` on OSError). `CaptureLogger.plan_path()`
extracts the existing `{timestamp}-{turn_id}.jsonl` naming so ref and writer agree.

> **Superseded (2026-09-26):** eligibility is now
> `request_type in {chat, completion, classification, image}`; embeddings and
> rerank remain ineligible. `_handle_classification_request` and
> `_handle_image_request` also call `_spawn_headroom` before `_log_capture`.

### Config (precedence CLI > env > file > default)
| Env | CLI | file key | default |
|-----|-----|----------|---------|
| `CAPTURE_HEADROOM` | `--headroom` | `headroom` | `off` |
| `CAPTURE_HEADROOM_WEIGHTS` | `--headroom-weights` | `headroom_weights` | `off` |
| `CAPTURE_HEADROOM_MAX_BYTES` | `--headroom-max-bytes` | `headroom_max_bytes` | `0` |
| `CAPTURE_HEADROOM_INTERVAL` | `--headroom-interval` | `headroom_interval` | `0` |
| `CAPTURE_HEADROOM_DIR` | `--headroom-dir` | `headroom_dir` | `{base_path}/headroom` |

`CAPTURE_HEADROOM` value: `off` \| `all` \| comma list of
`effective_rank,dead_neurons,load_balance,specialization,loss,isoflop`. MVP implements
`effective_rank`; other keys parse, record, and stay `null` in the sidecar.
`--headroom-max-bytes` accepts `256M`-style suffixes (`parse_byte_cap`).

## 4. DNA contract map (every public item touched)

| File | Item | DNA |
|------|------|-----|
| models.py | module docstring (Provides), `CaptureRecord` (version, headroom attr, invariants), `HeadroomRef` (fields, validator, invariants), `CapturedHeadroom` (sidecar schema) | new/updated |
| headroom.py | module, `TensorBundle`, `TensorLoader`, `ARCHITECTURE_TENSOR_NAMES` (inline comment), `parse_byte_cap`, `HeadroomLogger` (+`write_json`), `HeadroomWriter` (+`__init__`, `should_sample`, `write_async`, `_write_sync`, `_detect_architecture`, `_resolve_tensor`, `_default_tensor_loader`) | new |
| logger.py | `plan_path` (new), `_write_record` (updated — uses plan_path), module Invariants | updated |
| app.py | `CaptureProxyApp.__init__` (headroom env block), `_log_capture` (returns Path|None), `_spawn_headroom` (new), `_handle_non_streaming` / `stream_generator.finally` call sites, module Invariants (headroom lines) | updated |
| `__main__.py` | `parse_args` (5 flags), `_CONFIG_KEYS`, `main` (resolve + env inject), `_daemonize` resolved dict | updated |

## 5. Implementation plan (ordered, each verifiable)

1. `models.py`: `HeadroomRef` + `CapturedHeadroom` + `CaptureRecord` 4→5 — verify: `uv run mypy src/` + models import.
2. `headroom.py`: `parse_byte_cap`, `HeadroomLogger`, `HeadroomWriter` + default loader — verify: unit tests for truncation math, sampling, loader-contract shape.
3. `logger.py`: `plan_path()` extraction — verify: existing `tests/test_logger.py` green.
4. `__main__.py`: flags + precedence — verify: new precedence test.
5. `app.py`: writer construction + `_spawn_headroom` + call sites — verify: `tests/test_headroom.py` (off/all/interval/cap/uuid/streaming) + full suite.
6. Live verification: restart proxy on :8001 with `CAPTURE_HEADROOM=all CAPTURE_HEADROOM_WEIGHTS=gate CAPTURE_HEADROOM_MAX_BYTES=256M`, curl, inspect sidecar, run `model_headroom` scorer in its own venv.

## 6. Not built (negative space)

- No `torch`/`safetensors` in `pyproject.toml` (900 MB+ for a feature that is off by
  default; lazy import + documented env for verification).
- No `dead_neurons`/`load_balance`/`specialization`/`loss`/`isoflop` file emission
  (needs vLLM forward hooks or a training pipe — proposal §4.4 marks these future).
- No pending-task drain at shutdown (no drain machinery exists in the codebase;
  capture-tool process-exit semantics acceptable).
- No GGUF support: `capture-proxy` is safetensors-only and never imports a GGUF
  reader. A `.gguf` `runtime_settings["model_path"]` is detected by suffix before
  the directory gate, yields no sidecar (`record.headroom` stays None), and logs
  one deduplicated warning naming the path — GGUF is an explicit non-goal.
