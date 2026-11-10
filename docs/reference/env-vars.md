# Environment variables — all CAPTURE_* and model_headroom vars

**Purpose:** Complete environment variable reference. Cross-references
`reference/cli-flags.md` (flag equivalents), `guides/capture-proxy.md` (carrier
usage), `guides/model-headroom.md` (authority usage), and
`architecture/contract.md` (headroom sidecar contract). No Python code changes are
proposed here.

---

## capture-proxy env vars

| Variable | Default | Purpose |
|----------|---------|---------|
| `CAPTURE_PORT` | `8000` | Proxy listen port |
| `CAPTURE_HOST` | `0.0.0.0` | Bind address |
| `CAPTURE_UPSTREAM_URL` | `http://127.0.0.1:11434` | Target upstream backend |
| `CAPTURE_BACKEND` | *(auto)* | Backend type: `auto`, `llamacpp`, `ollama`, `node-llama-cpp`, `vllm`, `sglang` |
| `CAPTURE_BASE_PATH` | `/root/captures` | JSONL storage root |
| `CAPTURE_MODEL_ALIASES` | `off` | Public→upstream model map as JSON. TOML `model_aliases` table; `--model-aliases` flag; proxy-restart-only |
| `CAPTURE_MODEL_TASKS` | `off` | Public-model-id → task-type map as JSON. Keys match the public id exactly; read per request (env) or once at startup (TOML/flag) |
| `CAPTURE_HEADROOM` | `off` | Headroom sidecar capture: `off` \| `all` \| comma list of signals (`effective_rank`, `dead_neurons`, `load_balance`, `specialization`, `loss`, `isoflop`). MVP implements `effective_rank`; others parse and stay `null` |
| `CAPTURE_HEADROOM_WEIGHTS` | `off` | Which projections: `off` \| `gate` \| `all`. MoE: `gate` or `gate`+`up`+`down`. Dense: `attention` or `attention`+`intermediate`+`output` |
| `CAPTURE_HEADROOM_MAX_BYTES` | `0` | Float32 byte cap per dump (e.g. `256M`); 0 = unlimited |
| `CAPTURE_HEADROOM_INTERVAL` | `0` | Sample 1 of every N eligible captures; 0 = every capture |
| `CAPTURE_HEADROOM_DIR` | `{base_path}/headroom` | Headroom sidecar root |
| `CAPTURE_ROUTED_EXPERTS` | `off` | Routed-experts routing sidecar: `on` captures routing blob |
| `CAPTURE_ROUTED_EXPERTS_SHAPE` | *(unset)* | `LxK` tensor shape for decoding; `on` without valid shape warns and stays off |
| `CAPTURE_ROUTED_EXPERTS_DIR` | `{base_path}/routing` | Routing sidecar root |
| `CAPTURE_BLOB_STORE` | `on` | Write request/message bodies to `{base_path}/blobs/sha256/` so hashes resolve. `off` keeps hashing but stops writing |
| `CAPTURE_SYNC_REMOTE` | `gdrive:captures` | Sync destination remote/prefix. TOML `sync_remote` table; `--sync-remote` flag |
| `CAPTURE_SYNC_CMD` | `copy` | Sync command: `copy` (default, never deletes) or `sync` (mirror, DELETES remote files absent locally). TOML `sync_cmd` table; `--sync-cmd` flag |
| `CAPTURE_SYNC_MIN_AGE` | `60s` | Minimum age of files to sync; empty disables age filtering. TOML `sync_min_age` table; `--sync-min-age` flag |
| `CAPTURE_SYNC_INTERVAL` | `300` | Seconds between sync loop cycles. TOML `sync_interval` table; `--sync-interval` flag |
| `CAPTURE_SYNC_DRY_RUN` | `off` | Dry-run: pass `--dry-run` to rclone without transferring. TOML `sync_dry_run` table; `--sync-dry-run`/`--no-sync-dry-run` flag |
| `CAPTURE_SYNC_RCLONE_OPTS` | *(unset)* | Extra rclone options as a shell-quoted string. TOML `sync_rclone_opts` table; `--sync-rclone-opts` flag |
| `CAPTURE_SYNC_RCLONE_BINARY` | `rclone` | Path to rclone binary. TOML `sync_rclone_binary` table; `--sync-rclone-binary` flag |
| `CAPTURE_SYNC_MACHINE_ID` | *(hostname)* | Sync instance id; may be overridden by `--sync-machine-id`. TOML `machine_id` table; `--sync-machine-id` flag |
| `CAPTURE_SYNC_PID_FILE` | `~/.cache/capture-proxy/sync.pid` | Sync daemon PID file. TOML `sync_pid_file` table; `--sync-pid-file` flag |
| `CAPTURE_SYNC_LOG_FILE` | `~/.cache/capture-proxy/sync.log` | Sync daemon log file. TOML `sync_log_file` table; `--sync-log-file` flag |

**Headroom eligibility:** covers `chat`, `completion`, `classification`, `image`.
**Excluded:** `embeddings`, `rerank`.

**Config precedence:** CLI flag > env var > config file > default.

---

## model_headroom env vars

model_headroom does not declare runtime env vars in its public interface — all
configuration is via CLI flags. See `reference/cli-flags.md` for the 20 flags.
**There is no device-related env var** — `--device {auto,cuda,cpu}` is CLI-only
(`src/model_headroom/cli.py:162-167`).

---

## Cross-references

- `reference/cli-flags.md` — all CLI flags with env var equivalents
- `architecture/contract.md` — headroom sidecar contract, reserved keys
- `guides/capture-proxy.md` — carrier usage guide
- `guides/model-headroom.md` — authority usage guide
