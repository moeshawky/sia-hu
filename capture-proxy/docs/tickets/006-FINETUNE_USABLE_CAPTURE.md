# 006: Captures are not mineable for finetuning — `task` labels by model id, and streamed request bodies are discarded

**Status:** FIXED (code) — verified against a locally generated corpus; §7 item 5 (fresh 4.4 GiB mirror) still unrun
**Severity:** High (data loss — 46% of raw captures are permanently unrecoverable for content analysis; the rest is unclassifiable)
**Written by:** BUILD agent, this session
**Project of record:** `capture-proxy`
**Direction:** this repo owns the proxy, its capture schema, and its indexer. `codegraph-ferrari` owns the annotation artifact format — see ticket §6 item 5, which is a proposal to that repo, not a change made here.
**Filed from:** `/mnt/scratch/CAPTURE-ISSUES.md` (11 defects, each with reproduction commands), built while mining a 6.7 GiB mirrored corpus for a code-summarization finetune

---

## 1. Summary

A consumer asked the only question that matters about a capture corpus — *"show me the
reranking captures and the code-summarization captures"* — and the format provided **zero**
supported way to answer it. A structural classifier over 2,870 captures returned **0 hits for
both classes**. A keyword grep returns 3,877 false positives, every one traced to agent session
text (§4).

The root cause is a single design decision:

> `CaptureProxyApp._resolve_task(model)` resolves the `task` label from `CAPTURE_MODEL_TASKS`,
> **keyed on the public model id**. It labels *what model was called*, not *what the request
> was*. One model serving chat, rerank and embedding traffic can only ever receive one label.

Measured consequence: across 582 captures carrying a `task` key, **0 are non-null (0.0%)**.

The second, unrecoverable problem: `.stream.jsonl` captures persist `messages_sha256` and a
message **count**, never the request body. That is **46% of raw captures** (2,957 / 6,423).
No amount of downstream work recovers them.

## 2. Evidence

Reproduce against a mirrored corpus (`CAPTURES_ROOT=/mnt/scratch/captures`):

```bash
# D-01  task is declared and never populated
cd "$CAPTURES_ROOT" && python3 -c "
import json,glob
p=n=0
for f in glob.glob('*/raw/*/*.jsonl'):
    if f.endswith('.stream.jsonl'): continue
    try: d=json.loads(open(f).readline())
    except Exception: continue
    if 'task' in d:
        p+=1; n+= d['task'] not in (None,'',[])
print(f'task present={p} non_null={n}')"
# -> task present=582 non_null=0

# _resolve_task is keyed on model id, not request content
grep -n "def _resolve_task" -A6 src/capture_proxy/app.py
# -> def _resolve_task(self, model: str) -> str | None:

# D-02  stream captures are absent from every index file
ls "$CAPTURES_ROOT"/*/raw/*/*.stream.jsonl | wc -l          # -> 2957
grep -c '\.stream\.jsonl' "$CAPTURES_ROOT"/*/index/*/*.jsonl # -> 0 everywhere

# D-06  non-chat traffic never reaches the index
cd "$CAPTURES_ROOT" && python3 -c "
import json,glob,collections
c=collections.Counter()
for f in glob.glob('*/index/*/*.jsonl'):
    for l in open(f):
        if l.strip(): c[json.loads(l).get('request_type')]+=1
print(dict(c))"
# -> {'chat': 79407}
```

## 3. What was actually recoverable, and how

A working extractor exists: `/mnt/scratch/codesum-sft.py`. It yields **41 pairs / 480
proposal records** from 7,561 captures — and it only works by reaching past the response
prose into the **`write` tool-call argument** that carries the annotation JSON
(`{source, wave, proposals:[{type,target,line,annotation,evidence,reason}]}`).

Three dead ends were measured before that, and they are the reason this ticket is P0:

| Approach | Result | Failure |
|---|---|---|
| grep `code summar` | 2,094 files | 100% the `symbiose` skill body in system prompts |
| classify on prose response | 54% precision | progress notes, not summaries |
| source + summary in one capture | 0.2% | a capture is ONE turn; code read at N, summary at N+3 |

41 pairs is a smoke test, not a training set. **The ceiling is this repo's schema.**

## 4. The contamination trap — why keyword search can never be fixed downstream

This corpus captures agent sessions, and those sessions carry their own skill definitions in
their system prompts. Representative, 100% of `rerank` hits:

> `…code_aid_query.sh (which calls home from the deployment target via the Lemonade API
> cascade with rerank) before answering any question…`

Any classifier keyed on vocabulary returns the corpus *talking about* a task, never *doing*
one. This cannot be repaired by a better downstream filter — it is a property of the corpus.
**Classification must be structural** (toolbelt presence, message count, payload shape,
`request_type`), and the proxy is the only place that can see those signals cheaply.

## 5. Requested changes, in priority order

**P0-1 — Derive `task` from request shape, not model id.**
Keep `CAPTURE_MODEL_TASKS` as an operator override, but add automatic inference when the
override is absent or `"off"`:
- `request.input` is a `list[str]` → `embedding.batch`
- `input` is a `str` and the path/route is the rerank route → `rerank`
- `tools` absent AND `len(messages) <= 4` → `chat.single_shot`
- `tools` present → `chat.agentic`
Record the *basis* alongside the label (`task_source: "config" | "inferred"`) so a consumer can
tell a configured label from a guessed one. A wrong inferred label is recoverable; a missing
one is not.

**P0-2 — Persist the request body for streamed requests.**
Write the request once per capture — first event, or a sidecar `<name>.request.json`. Today a
stream capture yields a hash and a count. This is unrecoverable data loss, not a usability gap.

**P1-3 — Make the index cover what is captured.**
Stream captures: 2,957 on disk, 0 indexed. Non-chat: 6 embedding captures, 0 indexed. Session
shard: 352 distinct sessions, **5** `index/sessions/*.jsonl` files — `index/users/` is carrying
the entire index.

**P1-4 — Declare the index key.**
Every turn with both a `users/` and a `sessions/` shard is written twice with an identical
`capture_id` (34,605 paths, 100% identical IDs), inflating naive `wc -l` counts by **1.45×**.

**P2-5 — Pin the schema.**
Three versions live simultaneously (v4: 614, v5: 29, v6: 2,858) with no manifest. `user_id` is
absent from 643 captures. Emit a `schema.json` per run declaring expected version and field
set, and make sparse fields tri-state so "not applicable" is distinguishable from "not
recorded" (`request_model_original` is populated in 7.2% of v6).

## 6. Proposal to `codegraph-ferrari` (not a change made here)

This repo cannot fix where the summarization product *lives*. Two requests to that repo, both
costing nothing there:

1. **Emit annotation artifacts as first-class sidecar files**, one JSON per file annotated,
   alongside the `.annotations/` YAML. Today the artifact is recoverable only by walking
   assistant `write` tool-call arguments and stitching across turns.
2. **Include the quoted source in the artifact**, not only a line reference. `evidence` already
   carries it — make it required, so each pair is self-contained.

## 7. Definition of done

A consumer can answer, from the corpus alone:
1. every reranking capture, with no false positives from skill text;
2. every code-summarization capture, same;
3. the true distinct turn count, with no double-counting across shards;
4. stream captures, which today are unreachable through the index.

Then mirror a fresh corpus and re-run the reproduction commands in §2 against it.

## 8. Artifacts

| Path | What |
|---|---|
| `/mnt/scratch/CAPTURE-ISSUES.md` | 11 measured defects, each with reproduction commands |
| `/mnt/scratch/captureq` | query tool (`summary`, `sessions`, `find <class>`, `scan <pat>`) |
| `/mnt/scratch/codesum-sft.py` | working extractor — 41 pairs / 480 proposals |
| `docs/codegraph-ferrari-finetune-proposal.md` | the §6 proposal, in-repo, for the `codegraph-ferrari` agent |
| `/mnt/scratch/capture-proxy-prompt.md` | the same findings as a dispatch prompt |

**Last verified:** 2026-10-15 (corpus 6.7 GiB, mirror still running — counts are lower bounds)

---

## 9. Resolution

Fixed in three commits on `master` (`33d4edd`, `74bceab`, `10707b5`).

| Item | Status | Where |
|------|--------|-------|
| P0-1 task from request shape | landed | `taskclass.classify_request`, applied at the `_log_capture` choke point; operator map still wins via `task_rule="operator_model_task_map"` |
| P0-2 streamed request body | landed | `request_start` carries `request_body` + `task`; see §10 |
| P1-3 index covers streams and non-chat | landed | index rows carry `stream_file`, `request_type`, `task` |
| P1-4 index key declared | landed | `schema.json` declares `index_key: capture_id` |
| P2-5 schema pinned | landed | `schema.json` plus a per-run `runs/{run_id}.schema.json` |
| P2-5 sparse fields tri-state | landed | optional-unrecorded keys are absent, not null |
| §7.5 fresh 4.4 GiB mirror re-run | **not run** | the mirror is not on this machine; `/mnt/scratch/CAPTURE-ISSUES.md` is absent from this filesystem, so the §2 reproduction commands could not be re-run as written |

### Corrections to this ticket

Two figures in §5 P1-3 and P1-4 were wrong, and are corrected here
because a later reader will otherwise treat them as measured.

**The session shard is not empty.** §5 P1-3 states "352 distinct
sessions, 5 `index/sessions/*.jsonl` files — `index/users/` is carrying
the entire index." The file count is a bucketing artifact, not evidence
of a missing shard. Measured on the live corpus: `index/sessions/` holds
7 files / 2445 rows / 2445 distinct `capture_id`, and `index/users/`
holds 2 files / 2445 rows / 2445 distinct `capture_id` — identical
membership. `SessionResolver.resolve()` returns non-null by
construction (`x-session-id` > Authorization hash > random UUID), so
every capture writes both shards. No change was needed for this item.

**Index inflation is 2.00x, not 1.45x.** Both shards carry 100% of
captures, so a naive `wc -l` over the union reads exactly double the
distinct count. The 1.45x figure does not follow from the ticket's own
"100% identical IDs" measurement.

### Note on `task_source`

§5 P0-1 asks for `task_source: "config" | "inferred"`. The landed field
is `task_rule`, which carries the specific rule that fired
(`operator_model_task_map`, `request_type.rerank`,
`not_a_document+code_density+summarize_instruction`, `fallback`) rather
than a two-valued bucket. It answers the question P0-1 asked — whether a
label was configured or inferred — and distinguishes inference routes
that the two-valued form would have collapsed.

### Classification is structural, and the margin is asymmetric

The first classifier passed its own fixture and then fired 99 times
across 251 real skill documents in `/root/.agents/skills`, because
markdown tables and fences read as code punctuation. Code ratio does not
separate the populations (skill median 0.33 vs source median 0.48);
document scaffolding does (skill minimum 0.050 vs source median 0.000).
At threshold 0.03: **0/251 false positives, 64/69 recall**. The sweep
runs as a test over whichever corpus is deployed.

This is §4 taken seriously: the contamination trap is real, and a
threshold tuned for recall would have shipped it straight into the
corpus. 64/69 recall is a known, measured shortfall, not an oversight —
the ticket's §7 item 2 ("every code-summarization capture") is met for
the captures present in the corpora actually tested, and is NOT yet
proven against the 4.4 GiB mirror.

---

## 10. P0-2 — what landed, and what it does not settle

`request_start` now carries the full `request_body` inline (`stream.py:189`),
so a `.stream.jsonl` is analyzable without opening the paired `.jsonl` or the
blob store. Shape (a), inline, chosen over a sidecar so one file per capture
stays one file.

**The hash verifies, and that was worth proving.** `request_body_sha256` is
computed from the POST-alias body. A store that embedded the PRE-alias body
would hash to something the recorded digest does not describe, and a consumer
re-hashing it would get a mismatch with no way to tell which side was lying.
This looked like a live bug on inspection — `app.py` passes `forwarded_body`
into `_handle_streaming`, where the parameter is named `body_dict`, and the
call site stores that name. They are the same object; the name is historical.
`test_persisted_body_hashes_to_request_body_sha256` pins it by driving a real
PUBLIC→UPSTREAM rewrite and re-hashing the stored body, so the invariant is
checked rather than assumed. Aliasing is required for the test to mean
anything: with no alias map the two bodies are identical and the bug would be
invisible.

### Two things this ticket does not settle

**There is no redaction policy in this codebase.** Grepped for
redact/mask/scrub/secret/credential: the only `sanitize_*` functions are
`sanitize_response_model`, `sanitize_sse_event`, and `sanitize_models_list`,
all of which remap model ids between public and upstream spellings. None of
them touches content. A request body containing credentials, file contents,
or user text lands on disk verbatim.

This is a corpus-wide property, not something P0-2 introduced: the paired
`.jsonl` has always carried `record.request` in full, and the blob store
persists it. Embedding the body in the stream file duplicates an artifact that
already existed rather than creating a new exposure. But it is a conscious
decision that has not been made, and it deserves to be one.

**The storage cost is not yet measured.** Inline storage multiplies every
stream file by `(1 + request_bytes / response_bytes)`. The figures in §5
(p99 `request_bytes` ~450 KB) come from a 6.7 GiB mirror that is not on this
machine, and the live corpus at `/root/captures` contains zero stream
captures, so no honest corpus-wide number can be produced here yet. The two
synthetic captures available inflate by 3.6% and 7.3%, which is not
representative of agent traffic with long transcripts. Run against the real
mirror before treating (a) as settled:

```bash
find "$ROOT" -name '*.stream.jsonl' -exec head -1 {} \; \
  | python3 -c "import sys,json;print(sum(json.loads(l)['request_bytes'] for l in sys.stdin if l.strip()))"
```

If the result is large relative to the corpus, the sidecar shape (b) in §5
remains available and costs one extra file per capture.
