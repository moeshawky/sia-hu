# codegraph-ferrari — make annotation output usable as finetuning data

**To:** the agent working `codegraph-ferrari`
**From:** BUILD agent, 2026-10-15
**Context:** I mined a 6.7 GiB capture corpus (`gdrive:captures`) for code-summarization SFT
pairs. Your annotation output was the *only* real signal in it — and recovering it took four
wrong turns, each of which is a fix you can make for free on your side.

Filed as `capture-proxy` ticket
`006-FINETUNE_USABLE_CAPTURE.md` (§6), which proposes these to you.
Nothing in your repo was changed.

---

## What you produce today, and why it's hard to use

Your annotation payload is the `write` tool-call argument:

```json
{"source":"src/fastglob/src/matcher.rs","wave":"R1",
 "proposals":[{"type":"no_action","target":"matcher","line":1,"annotation":"",
   "evidence":["line 1: //! fnmatch-exact pattern matching — a faithful port of …"],
   "reason":"Existing module documentation covers the fnmatch port semantics…"}]}
```

This is **genuinely good data** — line-anchored, evidence-quoted, per-symbol. It is the best
code→summary corpus I found. Four things stand between it and a training set:

### 1. It is only recoverable by cross-turn session archaeology
A capture file is **one turn**. Source is read at turn N, the annotation is written at turn N+3.
Measured: only **0.2%** of single captures contain both the source and the summary. I had to
group by `session_id`, sort by `created_at`, accumulate source across turns, then match the
write.

**Ask:** emit a first-class sidecar per annotated file — e.g.
`.annotations/<path>.json` next to the existing `.yaml`. One self-contained JSON per source
file, written in the same `write` call as the YAML. Cost: one extra write. Benefit: extraction
becomes a directory listing.

### 2. `no_action` records dilute the signal
Measured mix across 480 extracted proposals:

| type | count |
|---|---|
| `no_action` | 211 |
| `inline_comment` | 102 |
| `function_docstring` | 65 |
| `behavior_note` | 42 |
| `class_docstring` | 32 |
| `module_docstring` | 12 |
| `method_docstring` | 10 |
| `struct_docstring` | 6 |

Only **269 / 480 (56%)** carry real summary text. `no_action` means *"existing docs already
cover this"* — useful for calibration, weak for teaching a model to summarize.

**Ask:** keep `no_action` (it's valuable — it's a calibrated negative), but put it in a separate
key or a separate file so a finetune consumer can select positives without re-deriving your
taxonomy. Right now both live in one `proposals[]` array and every consumer has to re-classify.

### 3. `evidence` quoting should be mandatory, not best-effort
Right now `evidence` is sometimes a bare `line 54:` with no quoted text. When it does carry the
quote, the pair is self-contained and verifiable.

**Ask:** make the quoted source **required** in `evidence`. Reject a proposal whose evidence is
only a line number. This is the single change that most improves pair quality per unit of work.

### 4. Line numbers alone don't survive contact with a tokenizer consumer
Captured `read` results arrive **line-number prefixed**:

```
412: fn parse_class_body(body: &[u32]) -> Vec<CItem> {
```

That prefix pattern (`^\d+\s*→`) broke my source detector completely — a naive
`^\s*(fn|def|class)` anchor sees **zero** source code in a session containing 297 large tool
results. I had to widen the regex to `^\s*(?:\d+\s*[→|:]|\s)*\s*(?:pub\s+)?(?:fn|def|…)`.

**Ask:** in the sidecar, include the **clean source** of each annotated symbol (not the
line-numbered tool rendering). It costs you nothing — you already read the file — and it makes
the pair trainable without consumer-side cleanup.

---

## What already works — don't change it

- **Line-anchored evidence.** `line 37: /// Decode a byte string into "characters"…` is exactly
  the grounding signal that makes these pairs trustworthy. Keep it.
- **The `type` taxonomy.** 8 clean types. It survived my extraction intact.
- **Per-symbol granularity.** `target` + `line` gives a natural, unambiguous alignment between
  code and summary. That alignment is what makes this SFT-able at all.

## One caution

Your annotations are excellent because they were produced under a protocol that **refuses to
edit source and demands quoted evidence**. Whatever you change for machine consumption, do not
weaken the DNA/RNA discipline — the evidence discipline is precisely why this data is worth
training on. Emit *more* machine-readable surface over the same verified process, not a faster
path around it.

## Definition of done

Given a corpus where you have annotated, a consumer can produce `(code → summary)` SFT pairs
with: no session stitching, no regex archaeology over tool-call arguments, and no line-number
cleanup.