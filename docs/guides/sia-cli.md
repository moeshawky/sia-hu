# Unified `sia` CLI

**Purpose:** Document the `sia` CLI — a thin dispatch layer that aliases the three
existing CLIs (`capture-proxy`, `model-headroom`, `model-headroom-refusal`) without
modifying them. Cross-references `architecture/sia-hu-identity.md` (the bridge role)
and `reference/cli-flags.md` (all flags). No Python code changes are proposed here.

---

## What `sia` is

The `sia` CLI is a thin dispatch layer: `sia <subcommand> <args>` forwards to the
corresponding existing CLI's `main()` function. Existing CLIs remain byte-for-byte
unchanged. `sia` adds no flags, no behavior, no new subcommands beyond the three it
aliases.

**Subcommands:**

| `sia` subcommand | Forwards to | Role |
|------------------|-------------|------|
| `sia capture-proxy` | `capture-proxy` | Carrier — emits Hu (capture artifacts) |
| `sia model-headroom` | `model-headroom` | Authority — scores Sia (headroom analysis) |
| `sia model-headroom-refusal` | `model-headroom-refusal` | Refusal analysis |

Each `main()` returns an `int` and only calls `sys.exit()` inside its
`if __name__ == "__main__"` guard — safe to import and call directly from the
dispatcher. Exit codes propagate.

---

## Usage

```bash
# List subcommands
sia --help

# Run a subcommand with its own flags
sia capture-proxy --port 8001 --headroom all --headroom-weights gate
sia model-headroom --model-name Emo_1b14b_1T --model-path /path/to/model
sia model-headroom-refusal --model-name NAME --model-path PATH

# Help for any subcommand (shows the underlying CLI's help)
sia capture-proxy --help
sia model-headroom --help
```

Each subcommand's flag set is the underlying CLI's own flags — see
`reference/cli-flags.md` for the complete list.

---

## Why `sia` exists

The operator previously had to remember three separate CLI names and their distinct
flag sets. `sia` provides a single command surface. It is the bridge in the
Sia/Hu identity — neither scoring nor emitting itself, but connecting the authority
and the carrier.

---

## What `sia` does NOT do (scope boundaries)

- Does not modify any existing CLI code
- Does not add new flags or behavior to existing CLIs
- Does not remove or deprecate existing console scripts
- Does not add subcommands beyond the three existing CLIs
- Does not change the verify chain

Adding a new subcommand requires explicit operator request — the three existing CLIs
are the complete authorized set.

---

## Cross-references

- `architecture/sia-hu-identity.md` — `sia` as the bridge between Sia and Hu
- `architecture/authority-carrier.md` — division of labor
- `guides/quickstart.md` — install and bootstrap
- `guides/capture-proxy.md` — carrier usage
- `guides/model-headroom.md` — authority usage
- `reference/cli-flags.md` — all flags for all CLIs including `sia`
