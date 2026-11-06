# Sia/Hu cosmology — the sia-hu identity

**Purpose:** Explain the Sia/Hu metaphor in detail, connect each concept to its
concrete repository and signal, and show why the mapping is 1-to-1. Cross-references
`architecture/authority-carrier.md` (division of labor) and `architecture/contract.md`
(the bidirectional contract). No Python code is described here — only the identity.

---

## The two Egyptian concepts

**Sia** (sỉꜣ): divine perception, deep insight, interior knowledge. In Egyptian
cosmology Sia perceives what is hidden inside a closed form *before* it is spoken —
the pre-verbal activation state, the latent internal representation that exists prior
to utterance.

**Hu** (ḥw): the uttered word, spoken output, token generation. Hu is what emerges
when the hidden form is spoken aloud.

**sia-hu = Sia + Hu.** A logit lens that reads Sia before it becomes Hu is a 1-to-1
conceptual match: the hidden internal representation (Sia) is observed before it is
tokenized and emitted as speech (Hu).

---

## Mapping onto the two packages

| Egyptian concept | English meaning | sia-hu role | Package | What it concretely does |
|------------------|-----------------|-------------|---------|-------------------------|
| Sia | hidden perception, interior knowledge | AUTHORITY | `model_headroom/` | Scores training headroom. Reads model weights — the hidden internal form — and computes a composite headroom score in [0.0, 1.0]. Nine signals (7 core + 2 diagnostic) inspect latent structure: weight spectra, activation balance, expert routing, gradient ratios. |
| Hu | uttered word, spoken output | CARRIER | `capture-proxy/` | Emits capture artifacts. Runs an OpenAI-compatible proxy that logs every request/response before forwarding. The emitted JSONL records, sidecar `headroom.json` files, and `weight_pairs.json` are the uttered output — token generation made observable. |

**Why this is 1-to-1:** Sia is the hidden form *before* speech; model_headroom reads
weights — the hidden internal representation *before* tokens are generated. Hu is
the uttered word; capture-proxy emits the request/response pairs — the tokens as they
are spoken. The proxy captures Hu at the moment of utterance; the authority scores
Sia from the form that precedes it.

---

## The `sia` CLI as the bridge

The unified `sia` CLI is the third term in the identity: it neither scores nor emits
itself. It is a thin dispatch layer — `sia <subcommand> <args>` forwards to the
corresponding existing CLI's `main()` function:

- `sia capture-proxy ...` → `capture-proxy` (the carrier, emits Hu)
- `sia model-headroom ...` → `model-headroom` (the authority, scores Sia)
- `sia model-headroom-refusal ...` → `model-headroom-refusal` (refusal analysis)

The existing CLIs remain byte-for-byte unchanged. `sia` adds no flags, no behavior,
no new subcommands beyond the three it aliases. It is the bridge that lets an
operator address both sides of the identity with one command.

---

## The division in one sentence

**model_headroom scores Sia (the hidden form). capture-proxy emits Hu (the uttered
word). `sia` bridges them. The proxy never scores; the authority never proxies.**

---

## Cross-references

- `architecture/authority-carrier.md` — how authority and carrier divide labor
- `architecture/contract.md` — the bidirectional contract between them
- `guides/sia-cli.md` — the unified `sia` CLI usage
- `guides/capture-proxy.md` — carrier (Hu) usage guide
- `guides/model-headroom.md` — authority (Sia) usage guide
- `reference/file-layout.md` — where each package lives on disk
