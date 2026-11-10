# Unified `sia` CLI

**Purpose:** Document the `sia` CLI — a dispatch layer for the three existing
CLIs (`capture-proxy`, `model-headroom`, `model-headroom-refusal`) plus two
`sia`-owned verbs (`doctor`, `publish`). Cross-references
`architecture/sia-hu-identity.md` (the bridge role) and
`reference/cli-flags.md` (all flags). Existing CLIs remain byte-for-byte
unchanged; `sia` adds new subcommands without modifying them.

---

## What `sia` is

The `sia` CLI is a thin dispatch layer: `sia <subcommand> <args>` forwards to
the corresponding existing CLI's `main()` function. Existing carrier/authority
CLIs remain unchanged. `sia` also owns two read-only/write-verified verbs
implemented in `src/sia/`:
- `sia doctor` → `sia.doctor:main` → read-only workspace preflight
- `sia publish` → `sia.publish:main` → push current branch, verify remote ref,
  never auto-commits

Each `main()` returns an `int` and only calls `sys.exit()` inside its
`if __name__ == "__main__"` guard — safe to import and call directly from the
dispatcher. Exit codes propagate.

---

## Subcommands

| `sia` subcommand | Forwards to | Role |
|------------------|-------------|------|
| `sia capture-proxy` | `capture-proxy` | Carrier — emits Hu (capture artifacts) |
| `sia model-headroom` | `model-headroom` | Authority — scores Sia (headroom analysis) |
| `sia model-headroom-refusal` | `model-headroom-refusal` | Refusal analysis |
| `sia doctor` | `sia.doctor:main` | Read-only workspace preflight |
| `sia publish` | `sia.publish:main` | Push current branch; verify remote ref never auto-commits |

Each subcommand's flag set is the underlying CLI's own flags — see
`reference/cli-flags.md` for the complete list. `doctor` and `publish` carry
their own flags (`--dry-run`, `--allow-dirty` on publish; none on doctor).

**Scope note:** `doctor` and `publish` are new `sia`-owned verbs; they do not
modify existing CLIs, add flags to existing CLIs, or remove any console
scripts. `sia doctor`/`sia publish` are `sia`-owned only.

---

## Cross-references

- `architecture/sia-hu-identity.md` — `sia` as the bridge between Sia and Hu
- `architecture/authority-carrier.md` — division of labor
- `guides/quickstart.md` — install and preflight (sia doctor / sia publish)
- `guides/capture-proxy.md` — carrier usage
- `guides/model-headroom.md` — authority usage
- `reference/cli-flags.md` — all flags for all CLIs including `sia`


---

## Usage

```bash
# List subcommands (now five)
sia --help

# Run a subcommand with its own flags
sia capture-proxy --port 8001 --headroom all --headroom-weights gate
sia model-headroom --model-name Emo_1b14b_1T --model-path /path/to/model
sia model-headroom-refusal --model-name NAME --model-path PATH

# sia-owned verbs
sia doctor
sia publish --dry-run
sia publish --allow-dirty

# Help for any subcommand (shows the underlying CLI's help)
sia capture-proxy --help
sia model-headroom --help
sia doctor --help
sia publish --help
```
