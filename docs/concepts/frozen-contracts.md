# Frozen contracts — dataclasses, invariants, NFR-01

**Purpose:** Document the frozen dataclasses in model_headroom, the invariants they
carry, and NFR-01 (CPU-only weight loading). Cross-references
`architecture/authority-carrier.md` (authority owns the types) and
`concepts/headroom-signals.md` (signal result types). No Python code changes are
proposed here.

---

## Frozen dataclasses

model_headroom defines **21 frozen dataclasses** (all in `models/types.py`), all
immutable (`frozen=True`). They are the authority's type contracts — every
input/output/error type flows through them.

**Invariant:** Never mutate a frozen dataclass. Create new instances, never modify
existing ones. `SignalError` is defined but never used — error handling uses string
messages (documented known issue).

**The `py.typed` marker is present** — the package declares itself typed for
downstream consumers.

---

## Key dataclasses

| Dataclass | Purpose | Key fields |
|-----------|---------|------------|
| `HeadroomReport` | Top-level analysis result | composite score, per-signal results, recommendations |
| `SignalResult` | Per-signal output | score, status, data, timing |
| `GradientWeightPair` | Weight + gradient pair | weight, gradient, layer_name, parent_shape, truncated, sampling_mode |
| `EffectiveRankResult` | R-03 output | rank, pct_of_full, coverage |
| `LossDataPoint` | R-01 input | tokens_trained, validation_loss |
| `IsoFLOPPoint` | R-02 input | compute_flops, model_params, measured_loss |
| `ExpertActivationData` | R-05/R-06 input | layers: dict[str, dict[str, int]], metadata |
| `HeadroomRef` | Carrier→authority ref | capture_id, parent_raw, model, model_path, architecture, signals, task |
| `CapturedHeadroom` | Sidecar file schema | HeadroomRef fields + collected_at, bytes_written, truncated, error_body |

---

## Signal scoring invariants

| Invariant | Meaning |
|-----------|---------|
| Skipped signals score 0.5 | Neutral prior, not a measurement |
| `not_applicable` excluded | Removed from numerator and denominator (dense models: R-06, R-07) |
| Errors score 0.0 | Computation failure |
| Composite range | [0.0, 1.0] |
| Decision-grade threshold | ≥70% measured coverage (`--analysis-mode decision_grade`, exit 2) |
| R-08/R-09 never affect score | Diagnostic only, opt-in |

---

## NFR-01 — CPU-only weight loading

**Requirement:** Tensors load to CPU RAM only. vLLM HBM (or any GPU memory) is
untouched. This is an intentional VRAM constraint — the authority is an offline
analysis tool, not a serving component.

**Consequences:**
- Weights are extracted to CPU RAM by design (`models/loader.py`, `models/gguf_loader.py`)
- No TPU/XLA import anywhere in the authority
- FP8 dequant via `weight_scale_inv` side tensor when present
- `device="cpu"` via `safe_open(..., framework="pt")` + `.float()`

**R-01 purity (carrier side):** With `CAPTURE_HEADROOM=off`, the carrier imports
neither torch, nor safetensors, nor model_headroom. Zero heavy imports on the hot
path. This is why `names.py` is stdlib-only — it lives in the carrier, not the
authority, precisely so it can be imported in off mode.

---

## Cross-references

- `architecture/authority-carrier.md` — authority owns types and scoring
- `architecture/contract.md` — frozen dataclass invariants the carrier must respect
- `concepts/headroom-signals.md` — signal result types and status values
- `concepts/provenance.md` — source_type / verification_status
- `guides/model-headroom.md` — running analysis, output formats
- `reference/env-vars.md` — `CAPTURE_HEADROOM*` (carrier-side R-01 purity)
