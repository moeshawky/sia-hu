# Verify chain — ruff → format → mypy → pytest

**Purpose:** Document the single verify chain run from the repo root, what each step
checks, the order that is load-bearing, and the per-package variations. Cross-references
`operations/dev-commands.md` (uv/bootstrap commands), `guides/quickstart.md` (install),
and `reference/file-layout.md` (repo layout). No Python code changes are proposed
here.

---

## The chain

Run from the repo root, in order:

```bash
ruff check . && ruff format --check . && mypy src/ && pytest
```

**Order is load-bearing.** Each step catches different errors. Running pytest before
mypy hides type bugs. Never skip steps.

---

## Step 1: ruff check .

Lints both `src/` packages (`capture_proxy` and `model_headroom`).

- Root config: `select = E,W,F,I,UP,B,SIM,RUF,S,C4,RET,TC,FA`
- `E501` ignored (line length handled by the formatter)
- `.pi/` and `tools/` excluded
- Excludes `.pi/` and `tools/`

---

## Step 2: ruff format --check .

Checks formatting of both packages against the root config. Same exclusions as step 1.

---

## Step 3: mypy src/

Type-checks `src/` only. **Never `mypy .`** — that sweeps `tests/` and yields
pre-existing errors.

- Root config: `strict = true` (mypy 2.4.0)
- `warn_return_any`, `warn_unused_ignores`, `strict` all enabled
- `mypy src/` takes 1m20s–3m03s; a timeout means *unmeasured*, not green
- Authority chain: `mypy .` has 20 pre-existing errors in 4 `tests/` files —
  `mypy src/` is clean across source files

---

## Step 4: pytest

Runs both test suites. Root `addopts`: `-v --tb=short`, `asyncio_mode = "auto"`,
`testpaths` = `capture-proxy/tests` + `model_headroom/tests`.

**Two traps, both measured:**

1. The authority's suite needs the package installed editable
   (`pip install -e .`) or collection fails on
   `ModuleNotFoundError: model_headroom`.
2. The carrier's chain is `mypy src/`, not `mypy .`. The carrier's
   `tests/conftest.py` aborts the run when `pytest-asyncio` is missing — run the
   suite with `pytest`, not a bare system `pytest`.

**Coverage:** `pytest` output goes to the root `.pytest_cache/`; coverage to the
root `htmlcov/` with `source = ["capture_proxy", "model_headroom"]`.

---

## Per-repo variations

### Authority (model_headroom)

| Aspect | Value |
|--------|-------|
| Verify chain | `ruff check . && ruff format --check . && mypy src/ && pytest` |
| mypy target | `mypy src/` (never `mypy .`) |
| pytest from | root (testpaths covers both suites) |
| Coverage target | ≥80% (pyproject.toml enforced) |

### Carrier (capture-proxy)

| Aspect | Value |
|--------|-------|
| Verify chain | `uv run ruff check . && uv run mypy src/ && uv run pytest` |
| mypy target | `mypy src/` (never `mypy .`) |
| pytest | `uv run --locked pytest` (always pass `--locked`) |
| ruff format | Not in the carrier's documented chain — GAP-04 |

**Always pass `--locked`** with the carrier: a bare `uv run` re-locks as a side
effect, so even `ruff format` can rewrite `uv.lock` mid-command. `uv lock --check`
is the read-only probe.

---

## Cross-references

- `operations/dev-commands.md` — uv commands, bootstrap scripts
- `guides/quickstart.md` — install and bootstrap
- `reference/file-layout.md` — repo layout, where `src/` lives
