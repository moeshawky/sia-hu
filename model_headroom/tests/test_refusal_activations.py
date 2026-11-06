"""Tests for residual-activation acquisition: loaders and the live hook."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import pytest
import torch

if TYPE_CHECKING:
    from pathlib import Path

from model_headroom.refusal.activations import (
    capture_last_token_residuals,
    discover_residual_layers,
    load_prompts,
    load_residual_batches_from_json,
)


class TestBatchLoading:
    def test_loads_wrapped_document(self, tmp_path: Path) -> None:
        path = tmp_path / "acts.json"
        path.write_text(
            json.dumps({"layers": [{"layer_name": "l0", "activations": [[1.0, 2.0]]}]})
        )
        batches = load_residual_batches_from_json(str(path))
        assert batches is not None
        assert batches[0].layer_name == "l0"
        assert batches[0].activations.shape == (1, 2)

    def test_loads_bare_list(self, tmp_path: Path) -> None:
        path = tmp_path / "acts.json"
        path.write_text(json.dumps([{"layer_name": "l0", "activations": [[1.0]]}]))
        batches = load_residual_batches_from_json(str(path))
        assert batches is not None
        assert len(batches) == 1

    def test_optional_fields_are_carried(self, tmp_path: Path) -> None:
        path = tmp_path / "acts.json"
        path.write_text(
            json.dumps(
                {
                    "layers": [
                        {
                            "layer_name": "l0",
                            "activations": [[1.0, 2.0]],
                            "position": "mean",
                            "source_type": "measured",
                            "prompt_count": 7,
                        }
                    ]
                }
            )
        )
        batches = load_residual_batches_from_json(str(path))
        assert batches is not None
        assert batches[0].position == "mean"
        assert batches[0].prompt_count == 7

    def test_non_2d_activations_are_skipped(self, tmp_path: Path) -> None:
        path = tmp_path / "acts.json"
        path.write_text(
            json.dumps({"layers": [{"layer_name": "l0", "activations": [1.0, 2.0]}]})
        )
        assert load_residual_batches_from_json(str(path)) is None

    def test_empty_activations_are_skipped(self, tmp_path: Path) -> None:
        path = tmp_path / "acts.json"
        path.write_text(
            json.dumps({"layers": [{"layer_name": "l0", "activations": []}]})
        )
        assert load_residual_batches_from_json(str(path)) is None

    def test_bad_position_is_skipped(self, tmp_path: Path) -> None:
        path = tmp_path / "acts.json"
        path.write_text(
            json.dumps(
                {
                    "layers": [
                        {
                            "layer_name": "l0",
                            "activations": [[1.0]],
                            "position": "middle",
                        }
                    ]
                }
            )
        )
        assert load_residual_batches_from_json(str(path)) is None

    def test_missing_file_returns_none(self, tmp_path: Path) -> None:
        assert load_residual_batches_from_json(str(tmp_path / "no.json")) is None

    def test_malformed_json_returns_none(self, tmp_path: Path) -> None:
        path = tmp_path / "acts.json"
        path.write_text("{oops")
        assert load_residual_batches_from_json(str(path)) is None

    def test_non_dict_entries_are_skipped(self, tmp_path: Path) -> None:
        path = tmp_path / "acts.json"
        path.write_text(json.dumps({"layers": ["junk", 5]}))
        assert load_residual_batches_from_json(str(path)) is None


class TestPromptLoading:
    def test_reads_txt_one_per_line(self, tmp_path: Path) -> None:
        path = tmp_path / "p.txt"
        path.write_text("first prompt\nsecond prompt\n")
        prompts = load_prompts(str(path))
        assert prompts == ["first prompt", "second prompt"]

    def test_skips_comments_and_blanks(self, tmp_path: Path) -> None:
        path = tmp_path / "p.txt"
        path.write_text("# header\n\nreal prompt\n")
        prompts = load_prompts(str(path))
        assert prompts == ["real prompt"]

    def test_reads_json_list(self, tmp_path: Path) -> None:
        path = tmp_path / "p.json"
        path.write_text(json.dumps(["a", "b"]))
        assert load_prompts(str(path)) == ["a", "b"]

    def test_reads_json_objects(self, tmp_path: Path) -> None:
        path = tmp_path / "p.json"
        path.write_text(json.dumps({"prompts": [{"prompt": "a"}, {"prompt": "b"}]}))
        assert load_prompts(str(path)) == ["a", "b"]

    def test_reads_jsonl(self, tmp_path: Path) -> None:
        path = tmp_path / "p.jsonl"
        path.write_text('{"prompt": "a"}\n{"prompt": "b"}\n')
        assert load_prompts(str(path)) == ["a", "b"]

    def test_bad_jsonl_lines_are_skipped(self, tmp_path: Path) -> None:
        path = tmp_path / "p.jsonl"
        path.write_text('{"prompt": "a"}\nnot json\n')
        assert load_prompts(str(path)) == ["a"]

    def test_missing_file_returns_none(self, tmp_path: Path) -> None:
        assert load_prompts(str(tmp_path / "no.txt")) is None

    def test_empty_file_returns_none(self, tmp_path: Path) -> None:
        path = tmp_path / "p.txt"
        path.write_text("\n\n")
        assert load_prompts(str(path)) is None

    def test_blank_json_strings_are_dropped(self, tmp_path: Path) -> None:
        path = tmp_path / "p.json"
        path.write_text(json.dumps(["a", "   ", "b"]))
        assert load_prompts(str(path)) == ["a", "b"]


class _StubTokenizer:
    """Minimal tokenizer stand-in: whitespace ids, real padding behaviour."""

    def __init__(self, vocab: int = 32) -> None:
        self.vocab = vocab
        self.padding_side = "right"
        self.pad_token = "<pad>"  # noqa: S105 - tokenizer special token, not a secret
        self.eos_token = "<eos>"  # noqa: S105 - tokenizer special token, not a secret
        self.padding_side_during_call: list[str] = []

    def __call__(self, text: Any, **kwargs: Any) -> dict[str, torch.Tensor]:
        self.padding_side_during_call.append(self.padding_side)
        if isinstance(text, str):
            text = [text]
        sequences = [
            [min(abs(hash(w)) % self.vocab, self.vocab - 1) for w in s.split()][:16]
            or [1]
            for s in text
        ]
        width = max(len(s) for s in sequences)
        pad = self.vocab - 1
        if self.padding_side == "left":
            padded = [[pad] * (width - len(s)) + s for s in sequences]
            mask = [[0] * (width - len(s)) + [1] * len(s) for s in sequences]
        else:
            padded = [s + [pad] * (width - len(s)) for s in sequences]
            mask = [[1] * len(s) + [0] * (width - len(s)) for s in sequences]
        return {
            "input_ids": torch.tensor(padded, dtype=torch.long),
            "attention_mask": torch.tensor(mask, dtype=torch.long),
        }


def _tiny_model() -> Any:
    transformers = pytest.importorskip("transformers")
    config = transformers.LlamaConfig(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=4,
        num_attention_heads=2,
        num_key_value_heads=2,
        vocab_size=32,
    )
    torch.manual_seed(0)
    return transformers.LlamaModel(config)


class TestLayerDiscovery:
    def test_discovers_layers_in_depth_order(self) -> None:
        names = discover_residual_layers(_tiny_model())
        assert len(names) == 4
        assert names[0].endswith("layers.0")
        assert names[-1].endswith("layers.3")

    def test_non_model_returns_empty(self) -> None:
        class Empty:
            def named_modules(self) -> list:
                return []

        assert discover_residual_layers(Empty()) == []


class TestLiveCapture:
    def test_captures_every_prompt(self) -> None:
        model, tokenizer = _tiny_model(), _StubTokenizer()
        batches = capture_last_token_residuals(
            model, tokenizer, ["a b c", "d", "e f"], ["layers.1"], batch_size=2
        )
        assert batches is not None
        assert batches["layers.1"].activations.shape == (3, 16)

    def test_prompt_count_is_recorded(self) -> None:
        model, tokenizer = _tiny_model(), _StubTokenizer()
        batches = capture_last_token_residuals(
            model, tokenizer, ["a b", "c"], ["layers.0"]
        )
        assert batches is not None
        assert batches["layers.0"].prompt_count == 2

    def test_padding_side_is_left_during_capture(self) -> None:
        model, tokenizer = _tiny_model(), _StubTokenizer()
        capture_last_token_residuals(
            model, tokenizer, ["a b", "c"], ["layers.0"], batch_size=2
        )
        assert set(tokenizer.padding_side_during_call) == {"left"}

    def test_padding_side_is_restored(self) -> None:
        model, tokenizer = _tiny_model(), _StubTokenizer()
        tokenizer.padding_side = "right"
        capture_last_token_residuals(model, tokenizer, ["a b"], ["layers.0"])
        assert tokenizer.padding_side == "right"

    def test_mean_position_does_not_force_left_padding(self) -> None:
        model, tokenizer = _tiny_model(), _StubTokenizer()
        tokenizer.padding_side = "right"
        capture_last_token_residuals(
            model, tokenizer, ["a b", "c"], ["layers.0"], position="mean"
        )
        assert set(tokenizer.padding_side_during_call) == {"right"}

    def test_hooks_are_removed_afterwards(self) -> None:
        model, tokenizer = _tiny_model(), _StubTokenizer()
        before = len(model._forward_hooks)
        capture_last_token_residuals(model, tokenizer, ["a b"], ["layers.0"])
        assert len(model._forward_hooks) == before

    def test_hooks_are_removed_even_on_failure(self) -> None:
        model = _tiny_model()
        before = len(model._forward_hooks)

        class Boom(_StubTokenizer):
            def __call__(self, text: Any, **kwargs: Any) -> Any:
                raise RuntimeError("boom")

        assert capture_last_token_residuals(model, Boom(), ["a"], ["layers.0"]) is None
        assert len(model._forward_hooks) == before

    def test_unknown_layer_returns_none(self) -> None:
        model, tokenizer = _tiny_model(), _StubTokenizer()
        assert (
            capture_last_token_residuals(model, tokenizer, ["a"], ["no.such.layer"])
            is None
        )

    def test_hook_reads_the_expected_tensor_and_index(self) -> None:
        model, tokenizer = _tiny_model(), _StubTokenizer()
        prompts = ["alpha beta", "gamma"]
        batches = capture_last_token_residuals(
            model, tokenizer, prompts, ["layers.0"], batch_size=2
        )
        assert batches is not None
        captured = batches["layers.0"].activations

        tokenizer.padding_side = "left"
        encoded = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )
        with torch.no_grad():
            reference = model(
                input_ids=encoded["input_ids"],
                attention_mask=encoded["attention_mask"],
                output_hidden_states=True,
            )
        expected = reference.hidden_states[1][:, -1, :]
        assert torch.allclose(captured, expected, atol=1e-5)

    def test_left_padding_means_last_index_is_a_real_token(self) -> None:
        tokenizer = _StubTokenizer()
        prompts = ["alpha beta gamma", "delta"]
        tokenizer.padding_side = "left"
        encoded = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )
        assert int(encoded["attention_mask"][1, -1]) == 1

    def test_captures_multiple_layers(self) -> None:
        model, tokenizer = _tiny_model(), _StubTokenizer()
        targets = ["layers.0", "layers.2"]
        batches = capture_last_token_residuals(model, tokenizer, ["a b"], targets)
        assert batches is not None
        assert set(batches) == set(targets)

    def test_capture_returns_float32_for_bf16_model(self) -> None:
        transformers = pytest.importorskip("transformers")
        config = transformers.LlamaConfig(
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=2,
            vocab_size=32,
        )
        torch.manual_seed(0)
        model = transformers.LlamaModel(config).to(torch.bfloat16)
        tokenizer = _StubTokenizer()
        batches = capture_last_token_residuals(
            model, tokenizer, ["a b", "c"], ["layers.0"]
        )
        assert batches is not None
        assert batches["layers.0"].activations.dtype == torch.float32


class TestSglangCapture:
    """Exercise the sglang HTTP path against a real local SSE server."""

    def _serve(self, handler):
        import threading
        from http.server import BaseHTTPRequestHandler, HTTPServer

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a: object) -> None:
                pass

            def do_POST(self) -> None:
                handler(self)

        server = HTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        return server

    def _sse(self, payload: object) -> bytes:
        import json as _json

        body = _json.dumps(
            {"choices": [{"index": 0, "delta": {"hidden_states": payload}}]}
        )
        return b"data: " + body.encode() + b"\n\n"

    def test_captures_last_token_rows(self) -> None:
        from model_headroom.refusal.activations import capture_residuals_sglang

        def handler(h: object) -> None:
            h.send_response(200)  # type: ignore[attr-defined]
            h.send_header("Content-Type", "text/event-stream")  # type: ignore[attr-defined]
            h.end_headers()  # type: ignore[attr-defined]
            h.wfile.write(self._sse([[1.0, 2.0, 3.0]]))  # type: ignore[attr-defined]
            h.wfile.write(b"data: [DONE]\n\n")  # type: ignore[attr-defined]

        server = self._serve(handler)
        try:
            batches = capture_residuals_sglang(
                f"http://127.0.0.1:{server.server_port}", "m", ["a", "b"]
            )
        finally:
            server.shutdown()
        assert batches is not None
        assert batches["final"].activations.shape == (2, 3)

    def test_bare_vector_is_promoted_to_a_row(self) -> None:
        from model_headroom.refusal.activations import capture_residuals_sglang

        def handler(h: object) -> None:
            h.send_response(200)  # type: ignore[attr-defined]
            h.send_header("Content-Type", "text/event-stream")  # type: ignore[attr-defined]
            h.end_headers()  # type: ignore[attr-defined]
            h.wfile.write(self._sse([1.0, 2.0]))  # type: ignore[attr-defined]

        server = self._serve(handler)
        try:
            batches = capture_residuals_sglang(
                f"http://127.0.0.1:{server.server_port}", "m", ["a"]
            )
        finally:
            server.shutdown()
        assert batches is not None
        assert batches["final"].activations.shape == (1, 2)

    def test_no_hidden_states_returns_none(self) -> None:
        from model_headroom.refusal.activations import capture_residuals_sglang

        def handler(h: object) -> None:
            h.send_response(200)  # type: ignore[attr-defined]
            h.end_headers()  # type: ignore[attr-defined]
            h.wfile.write(b'data: {"choices":[{"delta":{"content":"hi"}}]}\n\n')  # type: ignore[attr-defined]

        server = self._serve(handler)
        try:
            assert (
                capture_residuals_sglang(
                    f"http://127.0.0.1:{server.server_port}", "m", ["a"]
                )
                is None
            )
        finally:
            server.shutdown()

    def test_unreachable_server_returns_none(self) -> None:
        from model_headroom.refusal.activations import capture_residuals_sglang

        assert capture_residuals_sglang("http://127.0.0.1:1", "m", ["a"]) is None

    def test_non_http_scheme_is_rejected(self) -> None:
        from model_headroom.refusal.activations import capture_residuals_sglang

        with pytest.raises(ValueError, match="scheme"):
            capture_residuals_sglang("file:///etc/passwd", "m", ["a"])

    def test_one_request_per_prompt(self) -> None:
        from model_headroom.refusal.activations import capture_residuals_sglang

        seen: list[int] = []

        def handler(h: object) -> None:
            length = int(h.headers["Content-Length"])  # type: ignore[attr-defined]
            h.rfile.read(length)  # type: ignore[attr-defined]
            seen.append(1)
            h.send_response(200)  # type: ignore[attr-defined]
            h.end_headers()  # type: ignore[attr-defined]
            h.wfile.write(self._sse([[1.0, 0.0]]))  # type: ignore[attr-defined]

        server = self._serve(handler)
        try:
            capture_residuals_sglang(
                f"http://127.0.0.1:{server.server_port}", "m", ["a", "b", "c"]
            )
        finally:
            server.shutdown()
        assert len(seen) == 3
