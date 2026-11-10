# Dev commands — uv, bootstrap, environment setup

**Purpose:** Document the uv commands, bootstrap scripts, and environment setup for
both packages. Cross-references `operations/verify-chain.md` (verify chain),
`guides/quickstart.md` (install), and `reference/env-vars.md` (env vars). No Python
code changes are proposed here.

---

## uv commands (carrier)

The carrier uses uv with a sibling cache (`[tool.uv] cache-dir = "../.uv-cache"` in
pyproject.toml, git-ignored) so it stays on the same filesystem as `.venv` and uv
hardlinks instead of slow-copying.

```bash
uv run --locked pytest                        # all tests
uv run --locked pytest tests/test_smoke.py    # smoke tests only
uv run --locked ruff check .                  # lint
uv run --locked ruff format .                 # format
uv run --locked mypy src/                     # typecheck
uv run --locked ruff check . && uv run --locked mypy src/ && uv run --locked pytest  # full check
```

**Always pass `--locked`:** a bare `uv run` re-locks as a side effect.

---

## Install from repo root

```bash
pip install -e .                # carrier package only
pip install -e ".[headroom]"    # + torch / numpy / safetensors
pip install -e ".[dev]"         # + pytest / mypy / ruff / pre-commit
pip install -e ".[dev,headroom,gguf]"   # everything
```

---

## Preflight and publish (sia)

```bash
sia doctor              # read-only workspace preflight; changes nothing; exit 0/1
sia publish             # push current branch, verify remote ref, never auto-commit
sia publish --dry-run   # show commands, execute none (skip push + verification)
sia publish --allow-dirty   # override dirty-tree refusal (no commit, no force-push)
```

---

## Carrier-specific dev commands

```bash
capture-proxy --ensure-config            # create config if absent; never overwrite
capture-proxy --print-effective-config   # print resolved settings and exit
capture-proxy --daemon                   # start background instance
capture-proxy --status                   # show status
capture-proxy --stop                     # SIGTERM + cleanup
capture-proxy --sync                     # one-shot sync: run rclone once, then exit
capture-proxy --sync-daemon              # start sync loop (rclone every --sync-interval seconds)
capture-proxy --sync-stop                # stop sync daemon loop (SIGTERM + remove sync.pid)
capture-proxy --sync-status              # show sync daemon status (PID, liveness, log path)

```

**Sync flags:**

| Flag | Effect |
|------|--------|
| `--sync-remote` | Sync destination remote/prefix (default `gdrive:captures`) |
| `--sync-cmd` | `copy` (default, never deletes) or `sync` (mirror, DELETES) |
| `--sync-min-age` | Minimum file age to sync; empty disables age filtering |
| `--sync-interval` | Seconds between loop cycles (default 300) |
| `--sync-dry-run` / `--no-sync-dry-run` | Dry-run only |
| `--sync-rclone-opts` | Extra rclone options |
| `--sync-rclone-binary` | Path to rclone binary (default `rclone`) |
| `--sync-machine-id` | Sync instance id override |
| `--sync-pid-file` | Sync daemon PID file |
| `--sync-log-file` | Sync daemon log file |

---

## Environment variables used in dev

| Var | Purpose |
|-----|---------|
| `CAPTURE_PORT` | Proxy listen port (default 8000) |
| `CAPTURE_HOST` | Bind address (default 0.0.0.0) |
| `CAPTURE_UPSTREAM_URL` | Target upstream backend |
| `CAPTURE_BACKEND` | Backend type (auto/llamacpp/ollama/node-llama-cpp/vllm/sglang) |
| `CAPTURE_BASE_PATH` | JSONL storage root (default /root/captures) |
| `CAPTURE_HEADROOM` | Headroom sidecar capture (off/all/comma list) |
| `CAPTURE_HEADROOM_WEIGHTS` | Which projections to snapshot |
| `CAPTURE_HEADROOM_MAX_BYTES` | Float32 byte cap per dump |
| `CAPTURE_HEADROOM_INTERVAL` | Sample 1 of every N captures |
| `CAPTURE_HEADROOM_DIR` | Headroom sidecar root |
| `CAPTURE_MODEL_ALIASES` | Public→upstream model map as JSON |
| `CAPTURE_MODEL_TASKS` | Public-model-id → task-type map as JSON |
| `CAPTURE_BLOB_STORE` | Write bodies to content-addressed store |

See `reference/env-vars.md` for the complete list.

---

## Cross-references

- `operations/verify-chain.md` — verify chain
- `guides/quickstart.md` — install and bootstrap
- `reference/env-vars.md` — all env vars
- `reference/file-layout.md` — repo layout
