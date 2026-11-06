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

## Bootstrap (bootstrap.sh)

```bash
./bootstrap.sh --doctor     # preflight only; changes nothing; exit 0
./bootstrap.sh              # verify both packages, commit, push, verify
./bootstrap.sh --dry-run    # prints every git command, executes none
./bootstrap.sh --no-push    # everything except publish
./bootstrap.sh --yes        # skip pre-push confirmation
```

**Flags:**

| Flag | Effect |
|------|--------|
| *(none)* | full run |
| `--doctor` | preflight only; changes nothing; exit 0 |
| `--dry-run` | prints every git command, executes none |
| `--no-push` | everything except publish |
| `--yes` | skip the pre-push confirmation prompt |
| `-h`, `--help` | usage |

**Bootstrap steps (idempotent):**
1. Outer repo — keeps `.git`; `git init` only if absent
2. Verify the 2 packages — counts tracked files; never clones into them
3. Write `.gitignore` — appends only missing entries
4. Stage and report — prints tracked vs ignored with a count
5. Commit — no commit if nothing staged
6. Push, then verify — compares remote ref to local `HEAD`

---

## Carrier-specific dev commands

```bash
capture-proxy --ensure-config            # create config if absent; never overwrite
capture-proxy --print-effective-config   # print resolved settings and exit
capture-proxy --daemon                   # start background instance
capture-proxy --status                   # show status
capture-proxy --stop                     # SIGTERM + cleanup

```

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
| `GIT_USER` / `GIT_PASS` | Git credential overrides for bootstrap |
| `OUTER_NAME` / `REMOTE_BASE` | Remote name overrides for bootstrap |

See `reference/env-vars.md` for the complete list.

---

## Cross-references

- `operations/verify-chain.md` — verify chain
- `guides/quickstart.md` — install and bootstrap
- `reference/env-vars.md` — all env vars
- `reference/file-layout.md` — repo layout
