"""Residual-activation acquisition for refusal-direction extraction.

Three model-agnostic sources, none of which hardcode a vendor layer name:

* :func:`load_residual_batches_from_json` — pre-captured tensors on disk.
* :func:`load_prompts` — contrastive prompt sets from a text or JSON file.
* :func:`discover_residual_layers` / :func:`capture_last_token_residuals` — a live
  forward hook over a transformers model.

``transformers`` is not a declared dependency of this package, so the live path
imports it lazily and fails with a clear message rather than at import time.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

from model_headroom.models.types import ResidualBatch, ResidualPosition

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

_LAYER_INDEX_RE = re.compile(r"^(?P<prefix>.*(?:^|\.)(?:layers|h|blocks))\.(\d+)\.")

_ALLOWED_POSITIONS = ("last", "mean")


def load_residual_batches_from_json(json_path: str) -> list[ResidualBatch] | None:
    """Load residual activations from JSON.

    Accepts a wrapped ``{"layers": [...]}`` document or a bare list. Each entry
    needs ``layer_name`` and a 2-D ``activations`` array; optional ``position``,
    ``source_type`` and ``prompt_count`` are carried through. Returns None when
    the file is missing, malformed, or yields no usable layer, so callers can
    distinguish "no data" from "an empty set".
    """
    try:
        with Path(json_path).open() as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None

    entries = data.get("layers", data) if isinstance(data, dict) else data
    if not isinstance(entries, list):
        return None

    batches: list[ResidualBatch] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        if "activations" not in entry or "layer_name" not in entry:
            continue
        tensor = torch.tensor(entry["activations"], dtype=torch.float32)
        if tensor.ndim != 2 or tensor.shape[0] == 0:
            continue
        position = entry.get("position", "last")
        if position not in _ALLOWED_POSITIONS:
            continue
        prompt_count = entry.get("prompt_count")
        batches.append(
            ResidualBatch(
                activations=tensor,
                layer_name=str(entry["layer_name"]),
                position=position,
                source_type=entry.get("source_type", "measured"),
                prompt_count=(
                    int(prompt_count) if isinstance(prompt_count, int) else None
                ),
            )
        )
    return batches if batches else None


def load_prompts(path: str) -> list[str] | None:
    """Load a prompt set from a newline-delimited text file, JSON, or JSONL.

    Returns None when the file is missing or contains no non-empty prompt, so a
    caller never silently computes a direction from an invented contrast.
    """
    try:
        raw = Path(path).read_text()
    except OSError:
        return None

    suffix = Path(path).suffix.lower()
    prompts: list[str] = []

    if suffix == ".json":
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return None
        items = data.get("prompts", data) if isinstance(data, dict) else data
        if isinstance(items, list):
            for item in items:
                text = item.get("prompt") if isinstance(item, dict) else item
                if isinstance(text, str) and text.strip():
                    prompts.append(text)
    else:
        for line in raw.splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if suffix == ".jsonl":
                try:
                    entry = json.loads(stripped)
                except json.JSONDecodeError:
                    continue
                text = entry.get("prompt") if isinstance(entry, dict) else entry
                if isinstance(text, str) and text.strip():
                    prompts.append(text)
            else:
                prompts.append(stripped)

    return prompts if prompts else None


def discover_residual_layers(
    model: Any,
) -> list[str]:
    """Return the model's decoder layer module names in canonical depth order.

    Discovers the layer container by pattern rather than by a fixed attribute
    chain, so ``model.layers``, ``transformer.h``, ``decoder.layers`` and
    ``blocks`` all work. Returns an empty list when no indexable layer sequence
    is found.
    """
    prefix_to_indices: dict[str, set[int]] = {}
    for name, _ in model.named_modules():
        match = _LAYER_INDEX_RE.match(name + ".")
        if match is None:
            continue
        prefix = match.group("prefix").lstrip(".")
        prefix_to_indices.setdefault(prefix, set()).add(int(match.group(2)))

    if not prefix_to_indices:
        return []

    prefix = max(prefix_to_indices, key=lambda key: len(prefix_to_indices[key]))
    modules = dict(model.named_modules())
    names: list[tuple[int, str]] = []
    for index in sorted(prefix_to_indices[prefix]):
        candidate = f"{prefix}.{index}"
        if candidate in modules:
            names.append((index, candidate))
    return [name for _, name in names]


def _capture_hook(
    store: list[torch.Tensor], position: ResidualPosition
) -> Callable[..., None]:
    """Build a forward hook that appends one reduced residual per forward pass.

    ``h[:, -1, :]`` is only the last real token when the batch is left-padded;
    :func:`capture_last_token_residuals` enforces that. ``mean`` reduces over the
    sequence instead and is padding-insensitive for uniform weighting.
    """

    def hook(module: Any, inputs: Any, output: Any) -> None:
        hidden = output[0] if isinstance(output, tuple) else output
        if not isinstance(hidden, torch.Tensor) or hidden.ndim != 3:
            return
        reduced = hidden.mean(dim=1) if position == "mean" else hidden[:, -1, :]
        store.append(reduced.detach())

    return hook


def capture_last_token_residuals(
    model: Any,
    tokenizer: Any,
    prompts: Sequence[str],
    layer_names: Sequence[str],
    device: str = "cpu",
    batch_size: int = 8,
    max_length: int = 512,
    position: ResidualPosition = "last",
) -> dict[str, ResidualBatch] | None:
    """Run ``prompts`` through ``model`` and capture residual activations.

    Padding is forced to the left for the duration of the call and restored
    afterwards, because ``position="last"`` reads index ``-1``: under right
    padding that index is a pad token for every sequence shorter than the batch
    maximum, which silently yields activations of padding rather than of the
    prompt. Returns None when transformers is unavailable or no requested layer
    resolves to a module.
    """
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: F401
    except ImportError:
        return None

    modules = dict(model.named_modules())
    selected: dict[str, Any] = {}
    for name in layer_names:
        if name in modules:
            selected[name] = modules[name]
    if not selected:
        return None

    store: dict[str, list[torch.Tensor]] = {name: [] for name in selected}
    handles: list[Any] = []
    for name, module in selected.items():
        handles.append(
            module.register_forward_hook(_capture_hook(store[name], position))
        )

    previous_padding_side = getattr(tokenizer, "padding_side", "right")
    if position == "last" and hasattr(tokenizer, "padding_side"):
        tokenizer.padding_side = "left"
    if getattr(tokenizer, "pad_token", None) is None and hasattr(
        tokenizer, "eos_token"
    ):
        tokenizer.pad_token = tokenizer.eos_token

    try:
        with torch.no_grad():
            for start in range(0, len(prompts), batch_size):
                chunk = list(prompts[start : start + batch_size])
                encoded = tokenizer(
                    chunk,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                )
                encoded = {k: v.to(device) for k, v in encoded.items()}
                model(**encoded)
    except (RuntimeError, ValueError, TypeError, AttributeError, ImportError, OSError):
        return None
    finally:
        for handle in handles:
            handle.remove()
        if position == "last" and hasattr(tokenizer, "padding_side"):
            tokenizer.padding_side = previous_padding_side

    batches: dict[str, ResidualBatch] = {}
    for name, chunks in store.items():
        if not chunks:
            continue
        batches[name] = ResidualBatch(
            activations=torch.cat(chunks, dim=0).to(torch.float32),
            layer_name=name,
            position=position,
            source_type="measured",
            prompt_count=len(prompts),
        )
    return batches if batches else None


def _coerce_hidden_batch(raw: object, name: str) -> torch.Tensor | None:
    """Normalise a server-returned hidden-state payload into a 2-D tensor."""
    tensor = torch.tensor(raw, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 2 or tensor.shape[0] == 0 or tensor.shape[1] == 0:
        return None
    return tensor[-1:].contiguous()


def capture_residuals_sglang(
    base_url: str,
    model: str,
    prompts: Sequence[str],
    layer_name: str = "final",
    timeout: float = 300.0,
) -> dict[str, ResidualBatch] | None:
    """Capture final-layer last-token residuals from a running sglang server.

    Requires the server to be started with ``--enable-return-hidden-states``
    (or ``--return-hidden-states-mode full``); without it the server silently
    omits the field and this returns None rather than an empty batch.

    sglang returns hidden states only on the streaming path, as an SSE chunk
    carrying ``choices[].delta.hidden_states``. One request is issued per
    prompt, because ``last`` mode reports only the final token of each
    sequence — batching would report one vector per request, not per prompt.

    Only the final pre-lm_head layer is available through this route: sglang
    captures in ``LogitsProcessor`` at the tensor feeding ``lm_head``, and
    ``qwen4_exp`` does not populate the multi-layer ``aux_hidden_states``
    collector. Returns None if no prompt produced a usable vector.
    """
    import json
    import urllib.error
    import urllib.request
    from urllib.parse import urlparse

    parsed = urlparse(base_url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError(f"unsupported sglang URL scheme: {parsed.scheme!r}")

    rows: list[torch.Tensor] = []
    endpoint = base_url.rstrip("/") + "/v1/chat/completions"

    for prompt in prompts:
        payload = json.dumps(
            {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 1,
                "temperature": 0,
                "stream": True,
                "stream_options": {"include_usage": False},
                "return_hidden_states": "last",
            }
        ).encode()

        request = urllib.request.Request(  # noqa: S310 - scheme validated above
            endpoint,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
                for raw_line in response:
                    line = raw_line.decode("utf-8", errors="replace").strip()
                    if not line.startswith("data:"):
                        continue
                    body = line[5:].strip()
                    if not body or body == "[DONE]":
                        continue
                    try:
                        chunk = json.loads(body)
                    except json.JSONDecodeError:
                        continue
                    for choice in chunk.get("choices", []):
                        hidden = (choice.get("delta") or {}).get("hidden_states")
                        if hidden is None:
                            continue
                        batch = _coerce_hidden_batch(hidden, layer_name)
                        if batch is not None:
                            rows.append(batch)
        except (urllib.error.URLError, OSError, TimeoutError, ValueError):
            continue

    if not rows:
        return None

    return {
        layer_name: ResidualBatch(
            activations=torch.cat(rows, dim=0),
            layer_name=layer_name,
            position="last",
            source_type="measured",
            prompt_count=len(rows),
        )
    }
