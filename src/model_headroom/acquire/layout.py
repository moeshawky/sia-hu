"""Checkpoint weight-layout resolver for evidence acquisition.

Fixes a confirmed layout-matching bug: the CLI's safetensors sampler
(``model_headroom.cli._SAFETENSORS_TENSOR_TEMPLATES``) hardcodes tensor-name
templates that all carry a ``model.`` prefix (e.g.
``model.layers.{layer}.mlp.{proj}_proj.weight``). Real checkpoints do not
agree on a prefix — the RockAndEgg-313M-v2 Qwen3 checkpoint stores
``layers.0.mlp.gate_proj.weight`` with NO prefix — so the templated sampler
finds nothing and R-03 effective rank cannot be acquired at all.

This module resolves the REAL layout by reading the safetensors header only
(tensor names and shapes, never tensor data) and discovering the prefix,
per-layer FFN projections, and attention projections from the actual tensor
names. It never assumes a fixed prefix.

Module-kind semantics (do not treat as interchangeable): in a gated FFN,
``gate`` and ``up`` are the two PRE-activation branches (they produce the
gate and up vectors that are multiplied and activated together); ``down`` is
the POST-activation contribution (the projection back to the hidden size).
They perform different mathematical operations, so each is recorded under its
own kind key.

Tensor reading is lazy: ``safetensors`` and the torch-backed sampling helper
are imported inside functions, so ``import model_headroom.acquire`` works
without the ``[headroom]`` extra.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from model_headroom.models.types import SamplingMode, WeightSample

_LAYER_ANCHOR = re.compile(r"(?:^|\.)layers\.(\d+)\.")
_PROJECTION_KINDS = (
    "gate_proj",
    "up_proj",
    "down_proj",
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "w1",
    "w2",
    "w3",
)
_PROJECTION_SUFFIX = re.compile(
    r"\.(?P<kind>" + "|".join(_PROJECTION_KINDS) + r")\.weight$"
)

_FFN_GATE_UP = ("gate", "up")
_FFN_W1W2W3 = ("w1", "w2", "w3")
_ATTENTION = ("q", "k", "v", "o")
_FFN_KINDS = ("gate", "up", "down", "w1", "w2", "w3")

_KIND_TO_LABEL = {
    "gate_proj": "gate",
    "up_proj": "up",
    "down_proj": "down",
    "q_proj": "q",
    "k_proj": "k",
    "v_proj": "v",
    "o_proj": "o",
    "w1": "w1",
    "w2": "w2",
    "w3": "w3",
}


@dataclass(frozen=True)
class WeightLayout:
    """Resolved weight layout of a safetensors checkpoint.

    Attributes:
        prefix: The resolved name prefix before ``layers.{i}``, without the
            trailing dot (e.g. ``"model"``). An empty string ``""`` means the
            checkpoint stores tensors at the top level with no wrapper
            prefix (the resolved, honest answer for prefix-free checkpoints
            like Qwen3 base). ``None`` means nothing was resolved, or the
            matched projections disagreed on a prefix.
        layers: Mapping of layer index -> projection kind -> tensor name.
            Kinds carry module-kind semantics: ``gate``/``up`` are the
            pre-activation branches, ``down`` is the post-activation
            contribution, ``q``/``k``/``v``/``o`` are attention projections,
            and ``w1``/``w2``/``w3`` are the alternate gated-FFN triple.
        shapes: Mapping of tensor name -> 2-D shape ``(rows, cols)`` for every
            classified projection tensor.
        container: The dominant FFN container segment between ``layers.{i}``
            and the projection (e.g. ``"mlp"``), or ``None`` when no FFN
            projection was classified. Attention projections do not vote for
            the container, so an attention-only checkpoint yields ``None``.
        families: The sorted set of projection families present, drawn from
            ``"ffn_gated"`` (gate+up both present, or the w1/w2/w3 triple),
            ``"ffn_down"`` (down present), and ``"attention"`` (q/k/v/o).
        unmatched: Every tensor name in the header that was NOT classified as
            a projection (embeddings, norms, attention biases, extra MoE
            experts, ...). Recorded so the caller can see exactly what the
            resolver did not match — nothing is silently dropped.
    """

    prefix: str | None
    layers: dict[int, dict[str, str]]
    shapes: dict[str, tuple[int, int]]
    container: str | None
    families: tuple[str, ...]
    unmatched: tuple[str, ...]


def _classify_name(name: str) -> tuple[int, str, str, str] | None:
    """Classify one tensor name into (layer, kind, prefix, container).

    Returns ``None`` when the name matches no projection suffix or has no
    ``layers.{i}`` anchor.
    """
    suffix = _PROJECTION_SUFFIX.search(name)
    if suffix is None:
        return None
    anchor = _LAYER_ANCHOR.search(name)
    if anchor is None:
        return None
    kind = _KIND_TO_LABEL[suffix.group("kind")]
    prefix = name[: anchor.start()]
    segment = name[anchor.end() : suffix.start()].strip(".")
    container = segment.split(".")[0] if segment else ""
    return int(anchor.group(1)), kind, prefix, container


def _families_for_layer(kinds: dict[str, str]) -> set[str]:
    """Derive the family labels present for one layer's classified kinds."""
    keys = set(kinds)
    families: set[str] = set()
    if all(k in keys for k in _FFN_GATE_UP):
        families.add("ffn_gated")
    if all(k in keys for k in _FFN_W1W2W3):
        families.add("ffn_gated")
    if "down" in keys:
        families.add("ffn_down")
    if any(k in keys for k in _ATTENTION):
        families.add("attention")
    return families


def _shard_header_names(shard_path: Path) -> list[tuple[str, tuple[int, ...]]]:
    """Read one shard's tensor names and shapes, or [] when unreadable."""
    try:
        from safetensors import safe_open
    except ImportError:
        return []
    try:
        with safe_open(str(shard_path), framework="pt") as handle:
            return [
                (name, tuple(handle.get_slice(name).get_shape()))
                for name in handle.keys()  # noqa: SIM118 - safe_open handle is not iterable, only .keys() works
            ]
    except Exception:
        return []


def _header_tensors(
    model_path: Path,
) -> list[tuple[str, tuple[int, ...]]]:
    """Read tensor names and shapes from every safetensors shard header.

    Reads the header only (names + shapes), never tensor data. Returns an
    empty list when no readable safetensors checkpoint is found under
    ``model_path`` or the shards cannot be read — the caller degrades to an
    empty layout rather than failing the run.
    """
    try:
        from model_headroom.models.safetensors_sampling import find_checkpoint_dir
    except ImportError:
        return []

    directory = find_checkpoint_dir(model_path)
    if directory is None:
        return []

    tensors: list[tuple[str, tuple[int, ...]]] = []
    for shard in sorted(directory.glob("*.safetensors")):
        tensors.extend(_shard_header_names(shard))
    return tensors


def _empty_layout(unmatched: tuple[str, ...] = ()) -> WeightLayout:
    """The honest empty layout: nothing was resolved, nothing is claimed."""
    return WeightLayout(
        prefix=None,
        layers={},
        shapes={},
        container=None,
        families=(),
        unmatched=unmatched,
    )


def resolve_weight_layout(model_path: str | Path) -> WeightLayout:
    """Resolve the real weight layout of a safetensors checkpoint.

    Discovers the prefix, per-layer FFN and attention projections, and their
    shapes by reading the safetensors header only. The prefix is discovered
    from the actual tensor names, never assumed.

    Input:
        model_path: Path to a checkpoint directory, or to a parent of one.
    Output:
        A ``WeightLayout`` describing every classified projection tensor and
        the full list of unmatched header names. When nothing can be
        resolved (missing path, no safetensors, unreadable shards, or a
        tensor-name scheme no projection pattern recognises), the result is
        an EMPTY layout: ``layers == {}``, ``prefix is None``, and — when the
        header was readable — ``unmatched`` carries the full header inventory
        so the caller can see exactly what was there.
    Failure modes:
        - Never raises: an unresolvable layout degrades to the empty layout,
          the same pattern as the repo's existing samplers. Callers that find
          ``layers == {}`` must report "no layout resolved" and must not
          claim weight coverage.
    Invariants:
        - Never reads tensor data, only names and shapes.
        - ``prefix is None`` means "nothing resolved or projections
          disagreed"; ``prefix == ""`` means "resolved to a top-level
          checkpoint with no wrapper prefix" (e.g. Qwen3 base).
    """
    path = Path(model_path)
    if not path.exists():
        return _empty_layout()

    tensors = _header_tensors(path)
    if not tensors:
        return _empty_layout()

    layers: dict[int, dict[str, str]] = {}
    shapes: dict[str, tuple[int, int]] = {}
    prefix_votes: dict[str, int] = {}
    container_votes: dict[str, int] = {}
    matched: set[str] = set()

    for name, shape in tensors:
        classified = _classify_name(name)
        if classified is None:
            continue
        layer, kind, name_prefix, name_container = classified
        if len(shape) != 2:
            continue
        layer_kinds = layers.setdefault(layer, {})
        if kind in layer_kinds:
            continue
        layer_kinds[kind] = name
        matched.add(name)
        shapes[name] = (int(shape[0]), int(shape[1]))
        prefix_votes[name_prefix] = prefix_votes.get(name_prefix, 0) + 1
        if name_container and kind in _FFN_KINDS:
            container_votes[name_container] = container_votes.get(name_container, 0) + 1

    if not matched:
        return _empty_layout(unmatched=tuple(sorted(name for name, _ in tensors)))

    prefix: str | None
    prefix = next(iter(prefix_votes)) if len(prefix_votes) == 1 else None

    container: str | None
    if container_votes:
        container = max(container_votes, key=lambda k: (container_votes[k], k))
    else:
        container = None

    families = set()
    for kinds in layers.values():
        families |= _families_for_layer(kinds)

    unmatched = tuple(sorted(name for name, _ in tensors if name not in matched))

    return WeightLayout(
        prefix=prefix,
        layers=layers,
        shapes=shapes,
        container=container,
        families=tuple(sorted(families)),
        unmatched=unmatched,
    )


def sample_layout_weights(
    layout: WeightLayout,
    model_path: str | Path,
    max_rows: int = 256,
    seed: int = 42,
    device: str = "cpu",
) -> list[WeightSample]:
    """Sample weight tiles from the classified FFN projections of a layout.

    Feeds the DISCOVERED tensor names (not hardcoded templates) into the
    existing deterministic row sampler, one tensor at a time, so a
    multi-shard checkpoint is never materialised in RAM.

    Input:
        layout: A layout previously resolved by :func:`resolve_weight_layout`.
        model_path: Path to the checkpoint the layout was resolved from.
        max_rows: Upper bound on sampled rows per tensor (default 256).
        seed: Deterministic RNG seed (same seed -> same rows).
        device: Device the sampled rows are returned on.
    Output:
        One ``WeightSample`` per classified FFN projection tensor (kinds
        ``gate``/``up``/``down``/``w1``/``w2``/``w3`` across all layers),
        ordered by layer then kind. Attention projections are not sampled
        here — R-03 targets FFN matrices.
    Invariants:
        - ``sampling_mode`` is ``"single_tile"`` when rows were sampled and
          ``"full_matrix"`` when the whole matrix was used.
        - ``truncated`` is True exactly when fewer rows than the parent matrix
          were kept.
    """
    tensor_names: list[str] = []
    for layer in sorted(layout.layers):
        kinds = layout.layers[layer]
        for kind in sorted(kinds):
            if kind in _FFN_KINDS:
                tensor_names.append(kinds[kind])
    if not tensor_names:
        return []

    try:
        from model_headroom.models.safetensors_sampling import (
            sample_safetensors_weight_for_effective_rank,
        )
    except ImportError:
        return []

    selected = sample_safetensors_weight_for_effective_rank(
        model_path, tensor_names, max_rows=max_rows, seed=seed, device=device
    )

    samples: list[WeightSample] = []
    for layer_name, tensor, parent_shape in selected:
        truncated = tensor.shape[0] < parent_shape[0]
        sampling_mode: SamplingMode = "single_tile" if truncated else "full_matrix"
        samples.append(
            WeightSample(
                weight=tensor,
                layer_name=layer_name,
                parent_shape=parent_shape,
                truncated=truncated,
                sampling_mode=sampling_mode,
            )
        )
    return samples
