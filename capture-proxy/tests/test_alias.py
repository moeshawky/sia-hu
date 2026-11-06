"""Tests for model alias map and per-request helpers."""

from __future__ import annotations

import json

import pytest

from capture_proxy.alias import (
    AliasConfigError,
    AliasMap,
    resolve_aliases,
    rewrite_request_model,
    sanitize_models_list,
    sanitize_response_model,
    sanitize_sse_event,
)

PUBLIC = "qwen3.8-27b-fp8"
UPSTREAM = "davidau-qwen3.8-27b-twin-turbo-fable"


def _pair() -> AliasMap:
    return AliasMap({PUBLIC: UPSTREAM})


class TestAliasMap:
    def test_forward_and_reverse(self) -> None:
        m = _pair()
        assert m.forward == {PUBLIC: UPSTREAM}
        assert m.reverse == {UPSTREAM: PUBLIC}

    def test_frozen(self) -> None:
        m = _pair()
        with pytest.raises(AttributeError):
            m.forward = {}  # type: ignore[misc]

    def test_duplicate_upstream_raises(self) -> None:
        with pytest.raises(AliasConfigError, match="duplicate upstream"):
            AliasMap({"a": "x", "b": "x"})

    def test_self_map_raises(self) -> None:
        with pytest.raises(AliasConfigError, match="both public and upstream"):
            AliasMap({"a": "a"})

    def test_chain_a_to_b_b_to_c_fails_visibly(self) -> None:
        with pytest.raises(AliasConfigError, match="both public and upstream"):
            AliasMap({"A": "B", "B": "C"})

    def test_empty_raises(self) -> None:
        with pytest.raises(AliasConfigError):
            AliasMap({})

    def test_oversized_raises(self) -> None:
        big = {f"p{i}": f"u{i}" for i in range(65)}
        with pytest.raises(AliasConfigError, match=r"1\.\.64"):
            AliasMap(big)

    def test_blank_entry_raises(self) -> None:
        with pytest.raises(AliasConfigError):
            AliasMap({"  ": "x"})
        with pytest.raises(AliasConfigError):
            AliasMap({"a": "   "})

    def test_stripped_entries_stored_clean(self) -> None:
        m = AliasMap({"  a  ": "  x  "})
        assert m.forward == {"a": "x"}

    def test_non_string_entry_raises(self) -> None:
        with pytest.raises(AliasConfigError):
            AliasMap({"a": 1})  # type: ignore[dict-item]

    def test_defensive_copy(self) -> None:
        src = {"a": "x"}
        m = AliasMap(src)
        src["b"] = "y"
        assert m.forward == {"a": "x"}


class TestResolveAliases:
    def test_none_is_off(self) -> None:
        assert resolve_aliases(None) is None

    def test_empty_string_is_off(self) -> None:
        assert resolve_aliases("") is None
        assert resolve_aliases("   ") is None

    def test_off_string_is_off(self) -> None:
        assert resolve_aliases("off") is None
        assert resolve_aliases("OFF") is None

    def test_ellipsis_is_off(self) -> None:
        assert resolve_aliases("...") is None

    def test_json_string_parses(self) -> None:
        m = resolve_aliases(json.dumps({PUBLIC: UPSTREAM}))
        assert m is not None
        assert m.forward == {PUBLIC: UPSTREAM}

    def test_dict_table_parses(self) -> None:
        m = resolve_aliases({PUBLIC: UPSTREAM})
        assert m is not None
        assert m.reverse == {UPSTREAM: PUBLIC}

    def test_malformed_json_raises(self) -> None:
        with pytest.raises(AliasConfigError):
            resolve_aliases("{not json")

    def test_json_non_object_raises(self) -> None:
        with pytest.raises(AliasConfigError):
            resolve_aliases("[1,2]")

    def test_json_non_string_value_raises(self) -> None:
        with pytest.raises(AliasConfigError):
            resolve_aliases('{"a": 1}')

    def test_duplicate_upstream_via_json_raises(self) -> None:
        with pytest.raises(AliasConfigError, match="duplicate upstream"):
            resolve_aliases('{"a": "x", "b": "x"}')

    def test_chain_via_resolve_fails(self) -> None:
        with pytest.raises(AliasConfigError):
            resolve_aliases({"A": "B", "B": "C"})


class TestRewriteRequest:
    def test_off_returns_same_object(self) -> None:
        body = {"model": PUBLIC, "messages": []}
        out, orig, fwd = rewrite_request_model(body, None)
        assert out is body
        assert orig is None and fwd is None

    def test_empty_off_identity(self) -> None:
        body: dict[str, object] = {}
        out, orig, fwd = rewrite_request_model(body, None)
        assert out is body
        assert (orig, fwd) == (None, None)

    def test_hit_copies_and_maps(self) -> None:
        m = _pair()
        body = {"model": PUBLIC, "messages": [{"role": "user"}]}
        out, orig, fwd = rewrite_request_model(body, m)
        assert out is not body
        assert out["model"] == UPSTREAM
        assert (orig, fwd) == (PUBLIC, UPSTREAM)
        assert body["model"] == PUBLIC

    def test_unknown_returns_same_object(self) -> None:
        m = _pair()
        body = {"model": "other", "x": 1}
        out, orig, fwd = rewrite_request_model(body, m)
        assert out is body
        assert (orig, fwd) == (None, None)

    def test_missing_model_noop(self) -> None:
        m = _pair()
        body: dict[str, object] = {"messages": []}
        out, _, _ = rewrite_request_model(body, m)
        assert out is body

    def test_non_string_model_noop(self) -> None:
        m = _pair()
        body: dict[str, object] = {"model": 123}
        out, orig, fwd = rewrite_request_model(body, m)
        assert out is body
        assert (orig, fwd) == (None, None)

    def test_mutation_probe_input_never_mutated(self) -> None:
        m = _pair()
        body = {"model": PUBLIC, "nested": {"a": 1}}
        snapshot = dict(body)
        rewrite_request_model(body, m)
        assert body == snapshot

    def test_case_sensitive_no_hit(self) -> None:
        m = AliasMap({"Abc": "X"})
        body = {"model": "abc"}
        out, _, _ = rewrite_request_model(body, m)
        assert out is body


class TestSanitizeResponse:
    def test_none_passthrough(self) -> None:
        assert sanitize_response_model(None, _pair()) is None

    def test_off_same_object(self) -> None:
        payload = {"model": UPSTREAM, "choices": []}
        assert sanitize_response_model(payload, None) is payload

    def test_hit_copies_only_model(self) -> None:
        m = _pair()
        payload = {
            "model": UPSTREAM,
            "choices": [{"message": {"content": UPSTREAM}}],
            "usage": {"prompt_tokens": 1},
        }
        out = sanitize_response_model(payload, m)
        assert out is not payload
        assert out is not None and out["model"] == PUBLIC
        assert payload["model"] == UPSTREAM
        assert out["choices"] == payload["choices"]
        assert out["usage"] == payload["usage"]

    def test_content_with_model_name_untouched(self) -> None:
        m = _pair()
        content = f"the model {UPSTREAM} is great"
        payload = {
            "model": UPSTREAM,
            "choices": [{"message": {"content": content}}],
        }
        out = sanitize_response_model(payload, m)
        assert out is not None
        assert out["choices"][0]["message"]["content"] == content

    def test_unknown_same_object(self) -> None:
        m = _pair()
        payload = {"model": "other"}
        assert sanitize_response_model(payload, m) is payload

    def test_non_string_model_same_object(self) -> None:
        m = _pair()
        payload: dict[str, object] = {"model": 5}
        assert sanitize_response_model(payload, m) is payload

    def test_missing_model_same_object(self) -> None:
        m = _pair()
        payload: dict[str, object] = {"id": "x"}
        assert sanitize_response_model(payload, m) is payload


class TestSanitizeSSE:
    def test_done_verbatim(self) -> None:
        assert sanitize_sse_event("[DONE]", _pair()) == "[DONE]"

    def test_off_verbatim(self) -> None:
        raw = '{"model": "x", "choices": []}'
        assert sanitize_sse_event(raw, None) == raw

    def test_non_json_verbatim(self) -> None:
        raw = "not-json{{{"
        assert sanitize_sse_event(raw, _pair()) == raw

    def test_non_object_verbatim(self) -> None:
        assert sanitize_sse_event("[1,2]", _pair()) == "[1,2]"

    def test_hit_remaps_model_only(self) -> None:
        m = _pair()
        raw = json.dumps(
            {
                "id": "c1",
                "model": UPSTREAM,
                "choices": [{"delta": {"content": f"mention {UPSTREAM} inside"}}],
            },
            ensure_ascii=False,
        )
        out = sanitize_sse_event(raw, m)
        parsed = json.loads(out)
        assert parsed["model"] == PUBLIC
        assert parsed["choices"][0]["delta"]["content"] == f"mention {UPSTREAM} inside"

    def test_delta_content_with_model_name_untouched(self) -> None:
        m = _pair()
        body_text = f"say {UPSTREAM} aloud"
        raw = json.dumps(
            {"model": UPSTREAM, "choices": [{"delta": {"content": body_text}}]}
        )
        out = json.loads(sanitize_sse_event(raw, m))
        assert out["choices"][0]["delta"]["content"] == body_text

    def test_miss_returns_input_unchanged_no_reserialize(self) -> None:
        m = _pair()
        raw = '{"model": "other",  "choices": []}'
        assert sanitize_sse_event(raw, m) == raw

    def test_missing_model_verbatim(self) -> None:
        m = _pair()
        raw = '{"choices": []}'
        assert sanitize_sse_event(raw, m) == raw


class TestSanitizeModelsList:
    def test_none_passthrough(self) -> None:
        assert sanitize_models_list(None, _pair()) is None

    def test_off_unchanged_identity(self) -> None:
        payload = {"data": [{"id": UPSTREAM}]}
        assert sanitize_models_list(payload, None) is payload

    def test_empty_data_off_identity(self) -> None:
        payload: dict[str, object] = {"data": []}
        assert sanitize_models_list(payload, None) is payload

    def test_hit_remaps_and_drops_root(self) -> None:
        m = _pair()
        payload = {
            "object": "list",
            "data": [
                {"id": UPSTREAM, "object": "model", "root": UPSTREAM},
                {"id": "other", "object": "model", "root": "other"},
            ],
        }
        out = sanitize_models_list(payload, m)
        assert out is not payload
        assert out is not None
        assert len(out["data"]) == 2
        assert out["data"][0]["id"] == PUBLIC
        assert "root" not in out["data"][0]
        assert out["data"][1]["id"] == "other"
        assert "root" not in out["data"][1]
        assert out["object"] == "list"
        assert "root" in payload["data"][0]

    def test_order_len_preserved(self) -> None:
        m = _pair()
        payload = {"data": [{"id": "b"}, {"id": UPSTREAM}, {"id": "a"}]}
        out = sanitize_models_list(payload, m)
        assert out is not None
        assert [e["id"] for e in out["data"]] == ["b", PUBLIC, "a"]

    def test_missing_data_verbatim(self) -> None:
        m = _pair()
        payload: dict[str, object] = {"object": "list"}
        assert sanitize_models_list(payload, m) is payload

    def test_non_list_data_passthrough(self) -> None:
        m = _pair()
        payload: dict[str, object] = {"data": "nope"}
        assert sanitize_models_list(payload, m) is payload

    def test_non_dict_entries_preserved(self) -> None:
        m = _pair()
        payload: dict[str, object] = {"data": ["x", {"id": UPSTREAM}]}
        out = sanitize_models_list(payload, m)
        assert out is not None
        assert out["data"][0] == "x"
        assert out["data"][1]["id"] == PUBLIC
