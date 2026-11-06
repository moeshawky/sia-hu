# Work Order: Signal-Artifact Recipe Convention — define it, don't build it

**Status:** OPEN — definition only. No runner, no registry module, no TOML grammar engine, no endpoint.
**Thread:** ticket `002` §4 items 3–4 (aggregates: recipe over endpoint) + `EMO-001` follow-on (R-06 producer gap).
**Rule of this order:** name the mechanism that already exists. Any proposal requiring new machinery is out of scope until a third producer proves the definition insufficient.

## 1. Mission (one sentence)

Any current or future signal producer (weight rank, routing balance, classifier scores, image telemetry, audio — whatever comes third) feeds its signal through one documented door, with no new system built per signal.

## 2. The convention (defined)

> A signal is fed by writing one file in the exact shape its consumer's loader demands, at the deterministic per-capture path, recorded under `signals[<SIGNAL_KEYS name>]`; `null` means unproduced, never healthy-zero. A new signal adds a key + a path-suffix constant + a writer branch — nothing else.

Four rules, inherited from the two shipped instances — stated once, binding on every member:

1. **Declaration, never execution.** The recipe declares (key, shape, path, eligibility); the writer executes. No logic lives in the declaration.
2. **Off by default, loud on malformed.** Unknown keys rejected; unproduced signals stay `null`. A `null` must never read as a healthy measurement (the `60e0e4c` lesson: `load_balance`/`specialization` serialized `null` indistinguishably from healthy, so the gap read as handled for months).
3. **Shape is the consumer's, not the producer's.** The file must satisfy the authority's loader (`load_weight_pairs_from_json`, `load_activations_from_json`, …). Producer-convenient shapes are rejected.
4. **Weight-derived scalars are keys, never averages.** One snapshot per capture is not a sample — group/filter by them, never `avg()` them (same artifact class as the naive R-05 dead-neuron number, which measured prompt length).

## 3. Evidence (measured, not asserted)

| Instance | Key | Artifact path (deterministic per capture) | Shape contract | Writer branch |
|---|---|---|---|---|
| `effective_rank` | `signals["effective_rank"]` | `{day}/{capture_id}-weight_pairs.json` (`headroom.py:1077-1085`) | authority `load_weight_pairs_from_json` shape | `if "effective_rank" in self._signals and self._weights != "off"` (`headroom.py:1080`) |
| `load_balance` | `signals["load_balance"]` | `{day}/{capture_id}.activations.json` (`headroom.py:1078,1096-1101`, suffix `headroom.py:67`) | authority `load_activations_from_json` shape (`ExpertActivationData.layers: dict[str, dict[str, int]]` + fed/arch/starved metadata, `headroom.py:112-153`) | `if "load_balance" in self._signals` (`headroom.py:1096`) |

Handles (live, no reconstruction): `signals: dict = dict.fromkeys(SIGNAL_KEYS)` (`headroom.py:1070`); precomputed `signals = {key: planned.get(key) for key in SIGNAL_KEYS}` (`app.py:1920`); `CapturedHeadroom.signals` (`models.py`); vocabulary `SIGNAL_KEYS` = authority's core-6 verbatim (`names.py:92-99`).

Historical probe (OBSERVED from git): commit `60e0e4c` fed `load_balance` via exactly key + suffix + writer branch — no registry, no grammar, no runner. Negative control: `dead_neurons/specialization/loss_saturation/growth_factor` sit `null` today, proving the map tolerates unproduced signals without breaking readers.

## 4. Per-signal recipe schema (the client drops this, then it's set)

Four lines per signal. Filled example is measured; blank row is the door.

```toml
# load_balance (shipped — measured against headroom.py:1096-1101)
[signal.load_balance]
artifact_shape = "ExpertActivationData (model_headroom loader: load_activations_from_json)"
path_suffix = ".activations.json"          # ACTIVATION_ARTIFACT_SUFFIX, headroom.py:67
eligible_request_types = ["chat"]          # inference traffic only; weight reads can't produce it
```

```toml
# <next signal> (blank — e.g. classifier label/score telemetry)
[signal.<name>]
artifact_shape = "<consumer loader contract, exact function/file>"
path_suffix = ".<name>.json"
eligible_request_types = ["classification"]  # or ["image"], ["chat"], … — one shape per recipe
```

Note on the classifier door: `ClassificationResponse{labels: list[str], scores: list[float]}` are parallel per-input arrays the proxy neither clamps nor reorders (`models.py:438-456`). A classifier-signal recipe therefore declares per-input `{label, score}` pairs plus the score-floor predicate — same four lines, no new machinery.

## 5. Explicitly out of scope

- A recipe runner, registry module, or TOML grammar engine (the writers + dict already execute; nothing to register into).
- `/v1/stats` or any new serving route (002 item 3 settled: `jq`/`rg` over JSONL, see `README.md` Aggregates section).
- New aggregation semantics over weight scalars (rule 4 forbids).
- Renaming `SIGNAL_KEYS` or CLI names (fixed by the authority / breaking change respectively — `names.py:17-24`).

## 6. Acceptance

1. Both shipped producers cite key + suffix + branch above — MET (table §3).
2. A third producer can be specified using only the four-line schema — MET when one is filed against this order with no accompanying machinery change.
3. Full gate green on the commit carrying this file (`ruff check`, `mypy strict`, `pytest`; docs-only change, code untouched).

## 7. Verification

```bash
rg -n "dict.fromkeys\(SIGNAL_KEYS\)|ACTIVATION_ARTIFACT_SUFFIX|signals\[" src/capture_proxy/headroom.py
rg -n "planned.get\(key\) for key in SIGNAL_KEYS" src/capture_proxy/app.py
rg -n "SIGNAL_KEYS" src/capture_proxy/names.py | head -5
```

Last verified: 2026-10-02 UTC.
