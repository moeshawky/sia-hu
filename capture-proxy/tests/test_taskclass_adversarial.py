"""
Adversarial regression: task classification must ignore topic vocabulary.

The measured failure this file exists to prevent: grepping a capture
corpus for ``rerank`` returns thousands of files and for ``code summar``
returns thousands more, and EVERY hit is agent session text — because
sessions carry the ``cannon-gate`` and ``symbiose`` skill bodies in their
system prompts. A vocabulary classifier built on those greps answers a
question about what the corpus TALKS ABOUT, not what it DID, and returns
false positives on every session that mentions the task.

The fixture below is VERBATIM skill text, not a paraphrase, because the
paraphrase is where such a test would quietly stop testing the real case.

Invariants exercised:
  - a session carrying the skill bodies is never labelled rerank or
    code_summarize, with or without a toolbelt
  - structural positives still fire (a real rerank body, a real
    code-bearing summarization body)
  - the gate that saves us is the toolbelt + payload-key gate, and the
    tests below pin WHICH gate, so removing one does not silently shift
    the failure onto the other
"""

from __future__ import annotations

import pytest

from capture_proxy.taskclass import classify_request

# Verbatim excerpts from the deployed skill bodies that seed agent session
# system prompts. Truncated at natural paragraph boundaries; wording is
# unmodified because the trigger words are the point.
CANNON_GATE_TEXT = """---
name: cannon-gate
description: |
  The gate between the model and the corpus. Forces the model to
  code_aid_query.sh (which calls home from the deployment target via
  the Lemonade API cascade with rerank) before answering any question
  whose answer should come from the corpus, and to abstain —
  explicitly "not enough evidence" — when the corpus does not actually
  contain the answer.
---

# Cannon Gate

*Find the passage, then read the chapter it sits in. A reranked hit is a
pointer; only the book text is evidence.*

## Iron Law

```
THE OPERATOR PRACTICES A RELIGIOUS TRADITION THEY DO NOT SHARE WITH YOU.
YOU DO NOT NEED TO UNDERSTAND IT. YOU DO NOT NEED TO AGREE WITH IT.
YOU NEED TO OBEY IT.
```

- NO ANSWER WITHOUT RETRIEVAL FIRST.
  Any question whose answer may live in the corpus must go through
  code_aid_query.sh BEFORE the model forms an answer.

## The Gate — Three Checks

Run `code_aid_query.sh` (rerank on, by default), then `code_aid_read.sh`
on the hit that matters, and evaluate in order. Stop at the first
failure.

### Check 1 — Did the tool actually run?

- `code_aid_query.sh` was invoked, not skipped.
- Rerank was on (no `--fast`). Unreranked output is raw vector order.
- Exit code was 0, or a documented failure code.

| Using `--fast` for a factual question | Disables rerank | Rerank is the default |
"""

SYMBIOSE_TEXT = """---
name: symbiose
description: |
  An industrial data ecology skill based on the Kalundborg Symbiosis model.
  Transforms raw interaction logs, repetitive code summaries, and failed
  reasoning traces into high-signal SFT, CoT, and Recursive Training
  resources.
---

# Symbiose

*You are a scavenger in a data ecosystem where entropy is the primary fuel.*

## The Iron Law

```
NO GOLD WITHOUT EXECUTION GROUNDING (CODE MUST COMPILE, CLAIMS MUST BE VERIFIED)
```

Symbiose routes repeated code summaries and dead reasoning into SFT and
Recursive Training resources. Every by-product is ore for a specialized
model flavor.
"""

# A real source file a summarization call would carry.
CODE_SAMPLE = """def classify_request(request, request_type=None):
    tools = request.get("tools")
    if isinstance(tools, list) and tools:
        return TaskClassification(task="chat.agentic", rule="toolbelt_present")
    messages = request.get("messages")
    if not isinstance(messages, list):
        return TaskClassification(task="chat.task_api", rule="no_tools")
    count = len(messages)
    if 1 <= count <= 4:
        return TaskClassification(task="chat.single_shot", rule="short_no_tools")
    return TaskClassification(task="chat.agentic", rule="long_conversation")
"""

AGENT_TOOLS = [
    {
        "type": "function",
        "function": {"name": "searx", "description": "search the web"},
    },
    {
        "type": "function",
        "function": {"name": "think", "description": "sequential thinking"},
    },
]


def _session_request(skill_texts: list[str]) -> dict:
    """Build a realistic agent-session body carrying the skill texts."""
    return {
        "model": "MiMo-V2.6-Distill-Qwen-9B-INT8-AutoRound",
        "messages": [
            {"role": "system", "content": skill_texts[0]},
            {"role": "system", "content": skill_texts[1]},
            {"role": "user", "content": "Find the passage about rerank ranking."},
            {"role": "assistant", "content": "Running code_aid_query.sh now."},
            {"role": "user", "content": "Now summarize that code for me."},
        ],
        "tools": AGENT_TOOLS,
        "stream": False,
    }


class TestSkillTextIsNotATask:
    """The corpus-wide false-positive generator must not classify."""

    def test_session_with_skill_text_is_agentic(self) -> None:
        result = classify_request(_session_request([CANNON_GATE_TEXT, SYMBIOSE_TEXT]))
        assert result.task == "chat.agentic"
        assert result.rule == "toolbelt_present"

    def test_skill_text_never_yields_rerank(self) -> None:
        result = classify_request(_session_request([CANNON_GATE_TEXT, SYMBIOSE_TEXT]))
        assert result.task != "rerank"

    def test_skill_text_never_yields_code_summarize(self) -> None:
        result = classify_request(_session_request([CANNON_GATE_TEXT, SYMBIOSE_TEXT]))
        assert result.task != "code_summarize"

    def test_toolbelt_removed_does_not_open_the_door(self) -> None:
        """Stripping the toolbelt must not let skill text through.

        This is the load-bearing case: the toolbelt gate is the first line
        of defense, but the second pass must ALSO reject a long prose body
        on its own merits. If only the toolbelt gate existed, a client
        that sends no tools would get skill text labelled as a task.
        """
        request = _session_request([CANNON_GATE_TEXT, SYMBIOSE_TEXT])
        del request["tools"]
        result = classify_request(request)
        assert result.task not in {"rerank", "code_summarize"}

    def test_rerank_words_as_bare_top_level_keys_still_need_documents(self) -> None:
        """A chat body with a stray `query` key is not a rerank.

        The rerank rule requires `documents` to be an actual array. Prose
        that uses the word "query" never produces that key.
        """
        request = {
            "model": "m",
            "messages": [
                {"role": "user", "content": "Give me the query for this corpus."}
            ],
            "query": "what is the capital of France?",
        }
        result = classify_request(request)
        assert result.task != "rerank"

    def test_summarize_word_without_code_is_not_code_summarize(self) -> None:
        """The instruction alone is insufficient; code density is required."""
        request = {
            "model": "m",
            "messages": [
                {
                    "role": "user",
                    "content": "Please summarize the history of the Roman empire.",
                }
            ],
        }
        result = classify_request(request)
        assert result.task != "code_summarize"


class TestStructuralPositivesStillFire:
    """The adversarial cases must not be bought with lost recall."""

    def test_real_rerank_body(self) -> None:
        result = classify_request(
            {
                "model": "bge-reranker",
                "query": "what is the capital of France?",
                "documents": [
                    "Paris is the capital of France.",
                    "London is the capital of the UK.",
                ],
                "top_n": 2,
            }
        )
        assert result.task == "rerank"
        assert result.rule == "payload_keys.query+documents"

    def test_rerank_via_request_type(self) -> None:
        result = classify_request(
            {"model": "bge-reranker", "query": "q", "documents": ["d"]},
            request_type="rerank",
        )
        assert result.task == "rerank"

    def test_real_code_summarization_body(self) -> None:
        request = {
            "model": "m",
            "messages": [
                {
                    "role": "user",
                    "content": f"Summarize what this module does:\n\n{CODE_SAMPLE}",
                }
            ],
        }
        result = classify_request(request)
        assert result.task == "code_summarize"

    def test_code_without_instruction_stays_task_api(self) -> None:
        """Code with no summarization instruction is NOT code_summarize.

        It falls through the ladder to the short-chat label, which is the
        correct end state: a bare code dump is not a summarization task.
        """
        request = {
            "model": "m",
            "messages": [{"role": "user", "content": CODE_SAMPLE}],
        }
        result = classify_request(request)
        assert result.task != "code_summarize"
        assert result.task == "chat.single_shot"

    def test_long_code_without_instruction_is_agentic(self) -> None:
        """The same body, padded past 4 KiB, ends as agentic."""
        request = {
            "model": "m",
            "messages": [{"role": "user", "content": CODE_SAMPLE * 20}],
        }
        result = classify_request(request)
        assert result.task == "chat.agentic"

    def test_toolbelt_beats_a_rerank_shaped_payload(self) -> None:
        """An agent session that happens to carry query/documents keys."""
        request = {
            "model": "m",
            "messages": [{"role": "user", "content": "rank these"}],
            "query": "q",
            "documents": ["a", "b"],
            "tools": AGENT_TOOLS,
        }
        result = classify_request(request)
        assert result.task == "chat.agentic"


class TestShapeRules:
    """The first-pass shape ladder."""

    def test_short_no_tools_chat_is_single_shot(self) -> None:
        result = classify_request(
            {"model": "m", "messages": [{"role": "user", "content": "Hello!"}]}
        )
        assert result.task == "chat.single_shot"

    def test_long_no_tools_chat_falls_through(self) -> None:
        result = classify_request(
            {
                "model": "m",
                "messages": [{"role": "user", "content": "x" * 9000}],
            }
        )
        assert result.task == "chat.agentic"

    def test_batch_embedding_input(self) -> None:
        result = classify_request(
            {"model": "e", "input": ["a", "b", "c"]}, request_type="embedding"
        )
        assert result.task == "embedding.batch"

    def test_empty_tool_list_is_not_a_toolbelt(self) -> None:
        result = classify_request(
            {
                "model": "m",
                "messages": [{"role": "user", "content": "hi"}],
                "tools": [],
            }
        )
        assert result.task == "chat.single_shot"

    def test_non_dict_body_degrades_to_agentic(self) -> None:
        result = classify_request(["not", "a", "dict"])  # type: ignore[arg-type]
        assert result.task == "chat.agentic"
        assert result.rule == "fallback"


class TestVocabularyIsNotTheDiscriminator:
    """Guards the invariant itself, not just the fixtures."""

    def test_evidence_never_contains_a_topic_word(self) -> None:
        """Every evidence value is a measurement, not a matched keyword."""
        for request in (
            _session_request([CANNON_GATE_TEXT, SYMBIOSE_TEXT]),
            {"model": "m", "messages": [{"role": "user", "content": "hi"}]},
            {
                "model": "m",
                "messages": [{"role": "user", "content": f"summarize:\n{CODE_SAMPLE}"}],
            },
        ):
            result = classify_request(request)
            rendered = repr(result.evidence).lower()
            assert "rerank" not in rendered
            assert "cannon" not in rendered
            assert "symbiose" not in rendered

    def test_unknown_vocabulary_word_does_not_change_the_label(self) -> None:
        """Perturbing the prompt's topic words leaves the label identical."""
        base = {"model": "m", "messages": [{"role": "user", "content": "Hello!"}]}
        noisy = {
            "model": "m",
            "messages": [
                {
                    "role": "user",
                    "content": "Hello! This is about rerank and code summarization.",
                }
            ],
        }
        assert classify_request(base).task == classify_request(noisy).task


@pytest.mark.parametrize(
    "text",
    [CANNON_GATE_TEXT, SYMBIOSE_TEXT],
)
def test_each_skill_body_alone_is_inert(text: str) -> None:
    """Each skill body on its own must not classify as a task."""
    result = classify_request(
        {
            "model": "m",
            "messages": [{"role": "system", "content": text}],
        }
    )
    assert result.task not in {"rerank", "code_summarize"}


class TestDeployedSkillCorpus:
    """Sweep the REAL deployed skill corpus, not a hand-picked excerpt.

    The excerpt fixtures above passed while the first implementation of
    the code detector still fired on 99 of 251 real skill documents once
    the toolbelt was stripped — Markdown tables and fenced blocks read as
    source punctuation. A hand-written fixture cannot catch that; it was
    written by the same author who got it wrong. This sweep runs whatever
    documents are actually deployed.

    Skipped when no skill corpus is present, so the suite stays green on a
    machine without ~/.agents/skills.
    """

    @staticmethod
    def _skill_documents() -> list[str]:
        import glob

        return sorted(glob.glob("/root/.agents/skills/**/*.md", recursive=True))

    def test_no_skill_document_classifies_as_a_task(self) -> None:
        documents = self._skill_documents()
        if not documents:
            pytest.skip("no deployed skill corpus at /root/.agents/skills")
        offenders: list[str] = []
        for path in documents:
            try:
                with open(path, encoding="utf-8") as handle:
                    text = handle.read()
            except OSError:
                continue
            bodies = (
                # As a real session: skill text in the system prompt, tools present.
                {
                    "model": "m",
                    "messages": [
                        {"role": "system", "content": text},
                        {"role": "user", "content": "do the thing"},
                    ],
                    "tools": AGENT_TOOLS,
                },
                # Harder: toolbelt stripped, so only the document gate stands.
                {
                    "model": "m",
                    "messages": [
                        {"role": "system", "content": text},
                        {"role": "user", "content": "do the thing"},
                    ],
                },
            )
            for body in bodies:
                if classify_request(body).task in {"rerank", "code_summarize"}:
                    offenders.append(path)
        assert not offenders, f"classifier fired on skill documents: {offenders[:5]}"

    def test_real_source_files_still_classify(self) -> None:
        """The document gate must not have been bought with lost recall.

        Guards against 'fixing' the false positive by rejecting all code:
        a genuine source payload with a summarization instruction still
        has to be labelled.
        """
        source = CODE_SAMPLE
        result = classify_request(
            {
                "model": "m",
                "messages": [
                    {"role": "user", "content": f"Summarize this:\n\n{source}"}
                ],
            }
        )
        assert result.task == "code_summarize"

    def test_python_file_body_is_code_not_document(self) -> None:
        """A Python file's own text carries near-zero document scaffold."""
        import capture_proxy.taskclass as tc

        text = CODE_SAMPLE
        lines = [ln for ln in text.splitlines() if ln.strip()]
        scaffold = tc._scaffold_ratio(lines)
        assert scaffold <= tc.DOC_SCAFFOLD_MAX_RATIO
