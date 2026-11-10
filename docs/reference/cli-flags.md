# CLI flags — all three CLIs plus `sia`

**Purpose:** Complete flag reference for `capture-proxy`, `model-headroom`,
`model-headroom-refusal`, and the unified `sia` CLI. Cross-references
`guides/sia-cli.md` (dispatch behavior), `guides/capture-proxy.md` (carrier usage),
`guides/model-headroom.md` (authority usage), and `reference/env-vars.md` (env var
equivalents). No Python code changes are proposed here.

---

## capture-proxy (carrier)

| Flag | Purpose | Env equivalent |
|------|---------|----------------|
| `--port` | Proxy listen port | `CAPTURE_PORT` |
| `--host` | Bind address | `CAPTURE_HOST` |
| `--upstream-url` | Target upstream backend | `CAPTURE_UPSTREAM_URL` |
| `--base-path` | JSONL storage root | `CAPTURE_BASE_PATH` |
| `--backend` | Backend type (auto/llamacpp/ollama/node-llama-cpp/vllm/sglang) | `CAPTURE_BACKEND` |
| `--model-aliases '{"pub":"upstream"}'` | Public→upstream model map (restart-only) | `CAPTURE_MODEL_ALIASES` |
| `--model-tasks '{"pub":"task"}'` | Public-model-id → task-type map (per-request) | `CAPTURE_MODEL_TASKS` |
| `--headroom` | Enable headroom sidecar capture | `CAPTURE_HEADROOM` |
| `--headroom-weights` | Which projections to snapshot (off/gate/all) | `CAPTURE_HEADROOM_WEIGHTS` |
| `--headroom-max-bytes` | Float32 byte cap per dump (e.g. 256M) | `CAPTURE_HEADROOM_MAX_BYTES` |
| `--headroom-interval` | Sample 1 of every N eligible captures | `CAPTURE_HEADROOM_INTERVAL` |
| `--headroom-dir` | Headroom sidecar root | `CAPTURE_HEADROOM_DIR` |
| `--config PATH` | Config file path | — |
| `--ensure-config` | Create default config if absent; never overwrite | — |
| `--print-effective-config` | Print resolved settings and exit | — |
| `--daemon` | Run in background (PID file + log redirect) | — |
| `--stop` | Stop background instance | — |
| `--status` | Show background instance status | — |
| `--pid-file PATH` | Override daemon PID path | — |
| `--log-file PATH` | Override daemon log path | — |

### capture-proxy sync flags (native `--sync`, replacing removed `rclone-sync-captures.sh`)

| Flag | Purpose | Env equivalent |
|------|---------|----------------|
| `--sync` | One-shot sync: run rclone once, then exit | — |
| `--sync-daemon` | Start sync daemon loop (rclone every `--sync-interval` seconds) | — |
| `--sync-stop` | Stop sync daemon loop (SIGTERM to its PID, remove sync.pid) | — |
| `--sync-status` | Show sync daemon status (PID, pid file path) | — |
| `--sync-remote REMOTE` | Sync destination remote/prefix | `CAPTURE_SYNC_REMOTE` |
| `--sync-cmd {copy,sync}` | `copy` (default, never deletes) or `sync` (mirror, DELETES) | `CAPTURE_SYNC_CMD` |
| `--sync-min-age` | Minimum file age to sync; default `60s`, empty disables age filtering | `CAPTURE_SYNC_MIN_AGE` |
| `--sync-interval SECONDS` | Seconds between sync loop cycles (default `300`) | `CAPTURE_SYNC_INTERVAL` |
| `--sync-dry-run` / `--no-sync-dry-run` | Dry-run only | `CAPTURE_SYNC_DRY_RUN` |
| `--sync-rclone-opts OPTS` | Extra rclone options | `CAPTURE_SYNC_RCLONE_OPTS` |
| `--sync-rclone-binary PATH` | Path to rclone binary (default `rclone`) | `CAPTURE_SYNC_RCLONE_BINARY` |
| `--sync-machine-id ID` | Sync instance id override | `CAPTURE_SYNC_MACHINE_ID` |
| `--sync-pid-file PATH` | Sync daemon PID file | `CAPTURE_SYNC_PID_FILE` |
| `--sync-log-file PATH` | Sync daemon log file | `CAPTURE_SYNC_LOG_FILE` |

Settings resolve in order: **CLI flag > env var > config file > default.**

---

## model-headroom (authority)

All 20 flags, declared in `src/model_headroom/cli.py:parse_args`:

| Flag | Purpose |
|------|---------|
| `--model-name` | Model name for the report |
| `--model-path` | Path to model directory (HF) or .gguf file |
| `--architecture` | Override auto-detection (auto/moe/dense) |
| `--loss-curve` | Path to loss.json (R-01 training dynamics) |
| `--isoflop` | Path to isoflop.json (R-02 training dynamics) |
| `--weight-pairs-file` | Path to weight pairs JSON (R-03/R-04) |
| `--mlp-activations-file` | Path to MLP activations JSON (R-05) |
| `--expert-weights-file` | Path to expert weights JSON (R-07) |
| `--activations` | Path to expert_activations.json (R-05/R-06) |
| `--rank-parent-shapes` | Path to parent shapes for R-03 coverage |
| `--rank-sample-count` | Sample count for R-03 SVD |
| `--sampling-mode` | Sampling mode for R-03 |
| `--gradient-provenance-file` | Path to gradient provenance (R-08/R-09) |
| `--weight-pairs-source-type` | Source type override (measured/synthetic/fixture/estimated/imported) |
| `--mlp-source-type` | MLP source type override |
| `--analysis-mode` | Analysis mode (decision_grade requires ≥70% coverage) |
| `--gradient-diagnostics` | Enable R-08/R-09 diagnostic signals |
| `--device` | Device for tensor computation: auto (first available accelerator, cuda preferred, else cpu), cuda, or cpu (default: auto) |
| `--output-format` | Output format (human/json) |
| `--output-file` | Write output to file instead of stdout |

**Exit codes:** 0 success, 1 error, 2 decision-grade not met.

---

## model-headroom-refusal (authority)

Refusal analysis CLI. Entry point: `model-headroom-refusal = "model_headroom.refusal.cli:main"`. Same installation and invocation pattern as `model-headroom`. See `model_headroom/refusal/cli.py` for its flag set. `--device {auto,cuda,cpu}` (default: `auto`) picks the first available accelerator (cuda → xpu → mps → cpu) else CPU — an unflagged refusal run prefers an accelerator instead of silently capturing on CPU; `--device cuda` on a box without CUDA exits 1.

---

## sia (unified dispatcher)

`sia` has no flags of its own. It dispatches via subcommands:

```bash
sia --help                    # lists all five subcommands
sia <subcommand> --help       # shows the underlying CLI's help
sia capture-proxy <args>      # forwards to capture_proxy.__main__:main
sia model-headroom <args>     # forwards to model_headroom.cli:main
sia model-headroom-refusal <args>   # forwards to model_headroom.refusal.cli:main
sia doctor <args>             # forwards to sia.doctor:main (read-only preflight)
sia publish <args>            # forwards to sia.publish:main (push + verify, never auto-commits)
```

Exit codes propagate from the underlying CLI. `sia` adds no flags, no behavior,
no new subcommands beyond the five it dispatches. `doctor` and `publish` are
`sia`-owned verbs; they do not modify existing CLIs.

| `sia` subcommand | Module | Role |
|---|---|---|
| `sia capture-proxy` | `capture_proxy.__main__:main` | Carrier — emits Hu (capture artifacts) |
| `sia model-headroom` | `model_headroom.cli:main` | Authority — scores Sia (headroom analysis) |
| `sia model-headroom-refusal` | `model_headroom.refusal.cli:main` | Refusal analysis |
| `sia doctor` | `sia.doctor:main` | Read-only workspace preflight; exit 0/1/2 |
| `sia publish` | `sia.publish:main` | Push current branch; verify remote ref matches local HEAD; never auto-commits; `--dry-run`/`--allow-dirty`; credential URLs redacted |

`sia doctor` takes no flags. `sia publish` takes `--dry-run` and `--allow-dirty`.

---

## Cross-references

- `guides/sia-cli.md` — unified `sia` CLI dispatch behavior
- `guides/capture-proxy.md` — carrier usage guide
- `guides/model-headroom.md` — authority usage guide
- `reference/env-vars.md` — all env vars (CAPTURE_* and model_headroom vars)
