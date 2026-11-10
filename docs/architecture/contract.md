# Bidirectional contract — authority ↔ carrier

**Purpose:** Define the contract between model_headroom (authority) and
capture-proxy (carrier): what flows in each direction, the 2-D pair contract, the
reserved keys, schema versions, and the invariants that must never be violated.
Cross-references `architecture/sia-hu-identity.md` and
`architecture/authority-carrier.md`. No Python code changes are proposed here.

---

## The contract, in one paragraph

The carrier emits capture artifacts (`{capture_id}.headroom.json` plus
`{capture_id}-weight_pairs.json`) in the loader format the authority consumes.
The authority defines the 2-D pair contract and signal semantics; the carrier never
invents a signal meaning. The proxy never scores. The authority never proxies.
Agreement between the authority and carrier implementations is the cross-check — one voice
is an echo.

---

## Direction 1: Proxy → authority (carrier emits, authority consumes)

**Artifacts emitted per eligible capture:**

| File | Path pattern | Format | Authority consumer |
|------|-------------|--------|-------------------|
| Sidecar | `{headroom_dir}/YYYY-MM-DD/{capture_id}.headroom.json` | JSON (authoritative artifact, written last) | `model_headroom` loader contract |
| Weight pairs | `{headroom_dir}/YYYY-MM-DD/{capture_id}-weight_pairs.json` | Compact JSON | `loader.py:398` `load_weight_pairs_from_json` |

**Sidecar contents (`HeadroomRef`, written into `CaptureRecord.headroom`):**

| Field | Type | Meaning |
|-------|------|---------|
| `schema_version` | int | Currently 6 (additive bump from 5; v5 lines without `task` still validate) |
| `capture_id` | str | UUID, validated |
| `parent_raw` | Path | Planned raw JSONL path |
| `model` | str | Served model name from the request |
| `model_path` | str | Model dir from runtime_settings (must exist at spawn) |
| `architecture` | Literal["dense","moe","unknown"] | Detected architecture |
| `signals` | dict[str, str \| None] | Signal name → artifact path or None (6 keys) |
| `task` | str \| None | Task label from `CAPTURE_MODEL_TASKS` (additive, schema v6) |

**Weight pairs contract (consumed by `load_weight_pairs_from_json`):**
- Every pair has `gradient`, `weight`, `layer_name` keys
- `gradient` and `weight` are 2-D float arrays
- `gradient` has the same shape as `weight`
- Missing `gradient`/`weight`/`layer_name` → pair skipped
- Weight-only capture emits a real zero-gradient array of the weight's shape (ADR-002)

**Eligibility for sidecar emission:**
- Writer enabled (`CAPTURE_HEADROOM` not `off`)
- `request_type` ∈ {chat, completion, classification, image}
- Valid UUID `capture_id`
- `upstream_status` < 400 and no `error_body`
- `runtime_settings.model_path` is an existing directory
- `should_sample()` passes (interval counter)

**Embeddings and rerank never write a sidecar** — neither handler calls
`_spawn_headroom` and the request-type filter rejects them.

---

## Direction 2: Authority → proxy (authority defines, carrier conforms)

**What the authority defines that the carrier must conform to:**

| Item | Authority source | Carrier conforms by |
|------|-----------------|---------------------|
| 2-D pair contract | `loader.py:398` `load_weight_pairs_from_json` | Emitting pairs with `gradient`/`weight`/`layer_name`, 2-D arrays, matching shapes |
| Signal names | `names.py` `SIGNAL_KEYS` (authority's core-6 verbatim) | Using `SIGNAL_KEYS` as sidecar signal keys |
| Signal semantics | `models/types.py` frozen dataclasses | Never inventing a signal meaning; `null` = unproduced, never healthy-zero |
| Frozen dataclass invariants | `models/types.py` (21 frozen dataclasses) | Never mutating; constructing new instances only |
| NFR-01 (accelerator-preferred weights, CPU fallback) | Architecture decision | When an accelerator exists, use it; CPU when none exists or `--device cpu`; requesting an unavailable accelerator is an error, never a silent downgrade |

**Reserved keys (5 of 6 sidecar signal keys remain null except effective_rank):**

| Signal | Sidecar key | MVP status |
|--------|-------------|------------|
| effective_rank | `effective_rank` | Implemented (weight tensors from checkpoint) |
| load_balance | `load_balance` | Reserved null — needs inference-time routing histogram |
| dead_neurons | `dead_neurons` | Reserved null — needs forward hooks |
| specialization | `specialization` | Reserved null — weak offline from weight vectors |
| loss_saturation | `loss_saturation` | Reserved null — training telemetry only |
| growth_factor | `growth_factor` | Reserved null — training telemetry only |

R-01 (loss saturation) and R-02 (growth factor) are hard-blocked on training telemetry
(`tokens_trained` + `validation_loss`, `compute_flops` + `model_params` +
`measured_loss`); inference traffic carries neither.

---

## Schema versions

| Version | Status | Change |
|---------|--------|--------|
| 4 | Legacy | Pre-headroom records; still validate |
| 5 | Superseded by 6 | `headroom: HeadroomRef \| None = None` added |
| 6 | Current | Additive `task: str \| None = None` on both `HeadroomRef` and `CapturedHeadroom` |

Version bumps are additive — older records without new keys still validate.

---

## Invariants that must never be violated

1. **Proxy never scores.** No composite, no signal computation, no headroom score
   anywhere in `capture-proxy`.
2. **Authority never proxies.** No HTTP forwarding, no server, no capture emission
   anywhere in `model_headroom`.
3. **`null` ≠ healthy-zero.** A `null` signal means unproduced; it must never be read
   as a healthy measurement (the `60e0e4c` lesson).
4. **R-01 off-mode purity.** With `CAPTURE_HEADROOM=off`, the carrier imports neither
   torch, nor safetensors, nor model_headroom. Zero bytes on the hot path.
5. **Weight-only pairs carry real zero gradients.** Not null — a real zero-gradient
   array of the weight's shape, so effective_rank scores from weight and
   grad/weight ratio returns 0.0 cleanly.
6. **The sidecar is authoritative.** If the sidecar file is missing, the background
   write did not complete; the ref path is precomputed and the sidecar is the source
   of truth.

---

## Cross-references

- `architecture/sia-hu-identity.md` — Sia/Hu cosmology
- `architecture/authority-carrier.md` — division of labor
- `concepts/headroom-signals.md` — R-01 through R-09 signal reference
- `concepts/frozen-contracts.md` — frozen dataclasses, NFR-01
- `docs/capture-proxy/example-headroom.md` — end-to-end capture loop
- `docs/capture-proxy/headroom-capture-design.md` — ADR-001 sidecar design
- `reference/env-vars.md` — `CAPTURE_HEADROOM*` and related env vars
