# sia-hu — Sia + Hu

**Sia** (sỉꜣ): divine perception, deep insight, interior knowledge — perceiving what
is hidden inside a closed form BEFORE it is spoken (the pre-verbal activation state,
latent internal representation). **Hu** (ḥw): the uttered word, spoken output, token
generation.

A logit lens that reads Sia before it becomes Hu is a 1-to-1 conceptual match:

| Egyptian concept | sia-hu role | Repository | What it does |
|------------------|-------------|------------|--------------|
| **Sia** — hidden perception | AUTHORITY — scores | `model_headroom/` | Reads model weights (the hidden form) and computes a headroom score in [0.0, 1.0]. Nine signals inspect latent structure: weight spectra, activation balance, expert routing, gradient ratios. |
| **Hu** — uttered word | CARRIER — emits | `capture-proxy/` | Runs an OpenAI-compatible proxy that logs every request/response before forwarding. Emitted JSONL records, sidecar `headroom.json`, and `weight_pairs.json` are the uttered output — token generation made observable. |
| The bridge | CLI | `sia` | `sia <subcommand>` forwards to the corresponding existing CLI. Neither scores nor emits itself — it bridges Sia and Hu. |

**Contract (never inverted):**
- Authority scores (`model_headroom`); carrier never scores (`capture-proxy`).
- Proxy never scores; authority never proxies.
- Proxy → authority: emits `{capture_id}.headroom.json` +
  `{capture_id}-weight_pairs.json` in the loader format the authority consumes.
- Authority → proxy: defines the 2-D pair contract and signal semantics; the proxy
  never invents a signal meaning.
- Agreement between the authority and carrier implementations is the cross-check — one
  voice is an echo.

See `docs/architecture/sia-hu-identity.md` for the full cosmology, and
`docs/index.md` for the unified documentation tree.

---

## One project, two packages

| Package | Role | Rule |
|---------|------|------|
| `src/model_headroom/` | **Authority** — scores training headroom (reads Sia) | Never proxies |
| `src/capture_proxy/` | **Carrier** — emits sidecar capture (emits Hu) | Never scores |

Both are **tracked packages inside this single repo** — not submodules, not gitignored, not separate remotes. One clone carries the whole workspace.

The two packages are the **single source of one project**: `sia-hu` 0.2.0, built by one `pyproject.toml`, one `uv.lock`, one ruff/mypy/pytest config, installed editable from the repo root and exposing both entry points (`capture-proxy`, `model-headroom`, `model-headroom-refusal`). The packages keep separate code, separate tests, and separate `src/` trees — but they no longer carry their own `pyproject.toml`, their own lockfile, or their own install step.

`model_headroom` is the sole scoring authority. `capture-proxy` emits artifacts in the format the authority consumes. Agreement between the authority and carrier implementations is the cross-check — one voice is an echo.

---

## The unified `sia` CLI

`sia <subcommand> <args>` forwards to the corresponding existing CLI's `main()`
function:

| `sia` subcommand | Forwards to | Role |
|------------------|-------------|------|
| `sia capture-proxy` | `capture-proxy` | Carrier — emits Hu |
| `sia model-headroom` | `model-headroom` | Authority — scores Sia |
| `sia model-headroom-refusal` | `model-headroom-refusal` | Refusal analysis |

Each `main()` returns an `int` and only calls `sys.exit()` inside its
`if __name__ == "__main__"` guard — safe to import and call directly. Exit codes
propagate. `sia` adds no flags, no behavior, no new subcommands beyond the three
it aliases. Existing CLIs remain byte-for-byte unchanged.

See `docs/guides/sia-cli.md` for the full `sia` CLI reference, and
`docs/reference/cli-flags.md` for all flags across all CLIs.

---

## Quick start

```bash
git clone <outer-url> sia-hu
cd sia-hu
./bootstrap.sh --doctor     # measure the environment, change nothing
./bootstrap.sh              # verify both packages, commit, push, verify
```

**A clone is already a working workspace.** The script does not fetch the two
packages — they arrive with the clone. Step 2 verifies each one is present and
installable rather than cloning over a populated directory — which had failed in
practice before the flatten.

**Last verified:** 2026-10-19 — `bash -n`, plus `--doctor` and `--dry-run`
executed end to end against a fresh clone and a tunnel clone. `pip install -e`
measured from a fresh clone. The full commit-and-push path is
still unexercised end to end. Treat a first real push as unproven until it
completes once.

**Install from the repo root.** Base installs the carrier package (no heavy deps);
add the scoring deps for `model-headroom`; add `gguf` for GGUF loading.

```bash
pip install -e .                # sia-hu 0.2.0; carrier package only
pip install -e ".[headroom]"    # + torch / numpy / safetensors (model-headroom scoring)
pip install -e ".[dev]"         # + pytest / mypy / ruff / pre-commit
pip install -e ".[dev,headroom,gguf]"   # everything
```

`model-headroom`'s heavy deps were moved to the `headroom` extra so a carrier-only
install pulls none of them. The carrier sidecar deliberately imports no torch
(`R-01` purity) — `pip install -e .` is enough to run the proxy.

**Verify it yourself:**
```bash
./bootstrap.sh --doctor    # expected: all preflight lines prefixed ✓, exit 0
```

---

## What `bootstrap.sh` does

Five steps plus a verification gate. Each is idempotent.

| # | Step | Behaviour if already done |
|---|------|---------------------------|
| 1 | Outer repo | keeps existing `.git`; `git init` only if absent |
| 2 | Verify the 2 packages | counts tracked files; **never clones into them** |
| 3 | Write `.gitignore` | appends only missing entries, never rewrites |
| 4 | Stage and report | prints tracked vs ignored with a count |
| 5 | Commit | no commit if nothing staged |
| — | Push, then **verify** | sets `origin` only if unset |

Ignored: `.ix/`, `.codegraph/`, `__pycache__/`, `.venv/`.
Tracked: the gates, the plan, the runbook, this script, **and both packages**
(441 files).

### The verification gate

After `git push` exits 0, the script compares the remote ref to local `HEAD`:

```bash
git ls-remote origin refs/heads/master | cut -f1   # must equal git rev-parse HEAD
```

A non-zero exit from `git push` is **not** treated as proof of delivery. This
system has a documented history of silent-failure paths, so the script fails
loudly on a mismatch instead of reporting success. This is the whole reason the
gate exists.

---

## Credentials

**No credential is stored in this file or in the script.** They resolve in order:

1. userinfo already embedded in an existing clone's `origin` URL
2. `$GIT_USER` / `$GIT_PASS`
3. whatever git's credential helper supplies

Every URL the script prints passes through `redact()`, which strips userinfo
before output. To see a real URL yourself, use `git remote -v` and read the
credential deliberately.

Overrides:

```bash
OUTER_NAME=my_repo \
REMOTE_BASE='https://git.example.com/ns' \
GIT_USER=me GIT_PASS=… \
./bootstrap.sh
```

---

## Flags

| Flag | Effect |
|------|--------|
| *(none)* | full run |
| `--doctor` | preflight only; changes nothing; exit 0 |
| `--dry-run` | prints every git command, executes none |
| `--no-push` | everything except publish |
| `--yes` | skip the pre-push confirmation prompt |
| `-h`, `--help` | usage |

---

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `cannot determine the remote host/namespace` | no clone present and `REMOTE_BASE` unset | set `REMOTE_BASE` + `GIT_USER`/`GIT_PASS` |
| push fails, project does not exist | server has no auto-create on push | create `<base>/<OUTER_NAME>.git` in the web UI, then `./bootstrap.sh --no-push && git push -u origin master` |
| `refusing: already inside a package dir` | CWD is `model_headroom/` or `capture-proxy/` | `cd ..` first |
| `VERIFICATION FAILED` | push reported success but the remote ref differs | do not trust the repo; compare `git rev-parse HEAD` against `git ls-remote` |
| `<pkg> missing — the outer clone did not carry it` | incomplete checkout | re-clone; the packages are tracked here, nothing fetches them |

## Rollback

The local commit is never destroyed by a failed push. To undo everything this
script created in the parent:

```bash
rm -f .gitignore                     # restore: git checkout -- .gitignore
git reset --soft HEAD~1              # undo the commit, keep files staged
```

The two packages are **tracked here**. They are not fetched by this script and
have no separate remote to fall back on — do not un-track them.

---

## Measured state of this workspace

Recorded 2026-10-19, from `git` directly.

| Fact | Value |
|------|-------|
| Packages | 2 — tracked subtrees, one `pyproject.toml` |
| Authority tracked files | 147 |
| Carrier tracked files | 294 |
| Both build from a fresh clone | yes — single `uv build` of the root project exits 0 |
| `import capture_proxy` pulls torch / model_headroom / safetensors | no — R-01 purity holds |
| `bootstrap.sh` | `bash -n` clean; `--doctor` and `--dry-run` exit 0 |

---

## Running the verify chain

One chain from the repo root, in order. It lints and types both `src/` packages
and runs **both** test suites (root pytest reads
`capture-proxy/tests` and `model_headroom/tests`):

```bash
ruff check . && ruff format --check . && mypy src/ && pytest
```

- `ruff check .` — root config: `select = E,W,F,I,UP,B,SIM,RUF,S,C4,RET,TC,FA`,
  `E501` ignored, `.pi/` and `tools/` excluded.
- `ruff format --check .` — same root config.
- `mypy src/` — root config is `strict = true` (mypy 2.4.0). `mypy .` is wrong:
  it sweeps `tests/`, which is not in scope and yields pre-existing errors.
- `pytest` — root `addopts` is `-v --tb=short`, `asyncio_mode = "auto"`,
  `testpaths` is both suites. `mypy src/` takes 1m20s–3m03s; a timeout means
  *unmeasured*, not green.

**One lockfile.** There is a single `uv.lock` at the root; the per-package locks
and `uv run --locked` were dropped with the per-package `pyproject.toml`. `uv
run --locked pytest` still works (the root lock is read-only-checked), but the
chain above uses a bare `pytest`.

Two traps, both measured:

- The authority's suite needs the package installed editable
  (`pip install -e .`) or collection fails on
  `ModuleNotFoundError: model_headroom`.
- The carrier's chain is `mypy src/`, **not** `mypy .`, which sweeps `tests/`
  and yields pre-existing errors. The carrier's `tests/conftest.py` aborts the
  run when `pytest-asyncio` is missing — run the suite with `pytest`, not a bare
  system `pytest`.

`pytest` output goes to the root `.pytest_cache/`; coverage to the root
`htmlcov/` with `source = ["capture_proxy", "model_headroom"]`.

---

## Documentation

Unified docs live in `docs/` at the repo root:

| Section | Purpose |
|---------|---------|
| `docs/index.md` | Unified entry point — Sia/Hu identity, packages, contract, navigation |
| `docs/architecture/` | Sia/Hu identity, authority/carrier division, bidirectional contract |
| `docs/concepts/` | Headroom signals (R-01–R-09), provenance, frozen contracts |
| `docs/guides/` | Quickstart, `sia` CLI, carrier guide, authority guide |
| `docs/operations/` | Verify chain, dev commands, environment |
| `docs/reference/` | CLI flags, env vars, file layout |

Start with [`docs/index.md`](docs/index.md). Every document states its purpose
and cross-references related docs. No information from existing package docs is
duplicated — each references its source instead.

See also:
- `docs/architecture/sia-hu-identity.md` — the Sia/Hu cosmology in detail
- `docs/architecture/authority-carrier.md` — who scores, who emits, division of labor
- `docs/architecture/contract.md` — bidirectional contract, 2-D pair contract, reserved keys
- `docs/concepts/headroom-signals.md` — R-01 through R-09 signal reference
- `docs/guides/quickstart.md` — install, bootstrap, verify
- `docs/guides/sia-cli.md` — unified `sia` CLI
- `docs/guides/capture-proxy.md` — carrier usage guide
- `docs/guides/model-headroom.md` — authority usage guide
- `docs/operations/verify-chain.md` — ruff → format → mypy → pytest
- `docs/reference/cli-flags.md` — all CLI flags
- `docs/reference/env-vars.md` — all env vars
- `docs/reference/file-layout.md` — directory structure

## License

MIT — see [`LICENSE`](LICENSE).
