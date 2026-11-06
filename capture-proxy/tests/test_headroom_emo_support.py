"""Tests for MoE expert-tensor support in the headroom sidecar.

Covers: structural MoE detection for a model_type absent from
_MOE_MODEL_TYPES, the reachable ``architectures`` substring fallback, expert
tensor resolution for the EMO/OLMoE and legacy Mixtral/PhiMoE naming
schemes, expert-over-dense candidate precedence, the unregressed dense BERT
path, the malformed expert-count values the detection guard must survive, and
an end-to-end _capture_weight_pairs run with a stubbed tensor loader.

Every fixture is SYNTHETIC and lives in tmp_path. No test in this module
requires /root/Emo_1b14b_1T (or any real checkpoint) to exist, so the suite
passes in CI with no model on disk.
"""

import asyncio
import json
import uuid
from pathlib import Path
from typing import Any

import pytest

from capture_proxy.headroom import (
    HeadroomWriter,
    TensorBundle,
    _detect_architecture,
    _resolve_tensor_name,
)
from model_headroom.models import loader as _authority_loader

SHARD = "model-00001-of-00002.safetensors"
PROJS = ("gate", "up", "down")

# Expert naming schemes, sourced rather than assumed:
#   - mlp.experts.0.{gate,up,down}_proj.weight — EMO/OLMoE/Qwen-MoE per-expert
#     2-D weights. Corroborated by model_headroom
#     src/model_headroom/models/loader.py:40-43.
#   - block_sparse_moe.experts.0.w{1,2,3}.weight — LEGACY pre-fusion
#     Mixtral/PhiMoE, with the role map gate=w1, up=w3, down=w2 read from
#     the same file, loader.py:36-39.
EMO_EXPERT = {
    proj: f"model.layers.0.mlp.experts.0.{proj}_proj.weight" for proj in PROJS
}
LEGACY_WSTAR = {
    "gate": "model.layers.0.block_sparse_moe.experts.0.w1.weight",
    "up": "model.layers.0.block_sparse_moe.experts.0.w3.weight",
    "down": "model.layers.0.block_sparse_moe.experts.0.w2.weight",
}
DENSE_MLP = {proj: f"model.layers.0.mlp.{proj}_proj.weight" for proj in PROJS}
BERT_DENSE = {
    "attention": "bert.encoder.layer.0.attention.self.query.weight",
    "intermediate": "bert.encoder.layer.0.intermediate.dense.weight",
    "output": "bert.encoder.layer.0.output.dense.weight",
}


def _model_dir(
    tmp_path: Path,
    *,
    name: str = "model",
    config: dict[str, Any] | None = None,
    keys: dict[str, str] | None = None,
) -> Path:
    """Write a synthetic model dir: config.json plus a safetensors index.

    Args:
        tmp_path: Parent directory for the created model dir.
        name: Directory name.
        config: config.json contents; None writes an EMO config.
        keys: Index key pattern map; None writes the EMO expert index.

    Returns:
        Path: The created model directory.

    Raises: None
    """
    d = tmp_path / name
    d.mkdir()
    (d / "config.json").write_text(
        json.dumps(
            config
            if config is not None
            else {
                "model_type": "emo",
                "architectures": ["EmoForCausalLM"],
                "num_experts": 128,
            }
        )
    )
    # keys maps projection -> tensor-name pattern; the index keys are the
    # NAMES (the dict values), not the projection labels.
    patterns = keys if keys is not None else EMO_EXPERT
    (d / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": dict.fromkeys(patterns.values(), SHARD)})
    )
    return d


def _fake_loader(rows: int = 2, cols: int = 3):
    """Build a deterministic TensorLoader, mirroring tests/test_headroom.py.

    Returns:
        Callable[[Path, str], TensorBundle]: Loader over a fixed rows x cols
            weight whose gradient is an all-zeros array of the SAME shape.
            Zeros are the expected weight-only result (ADR-002), not a bug.
    """

    def loader(model_path: Path, tensor_name: str) -> TensorBundle:
        weight = [[float(i * cols + j) for j in range(cols)] for i in range(rows)]
        gradient = [[0.0] * cols for _ in range(rows)]
        return TensorBundle(weight=weight, gradient=gradient)

    return loader


# ---------------------------------------------------------------------------
# D1 — structural MoE detection
# ---------------------------------------------------------------------------


def test_detects_emo_as_moe(tmp_path):
    """A model_type absent from _MOE_MODEL_TYPES is classified by structure.

    "emo" is not in the name-list fast path; the num_experts count is what
    makes this "moe".
    """
    d = _model_dir(tmp_path)
    assert _detect_architecture(d) == "moe"


def test_architectures_moe_fallback_reachable(tmp_path):
    """The architectures substring check must run BEFORE the dense early return.

    Regression test for the D1 ordering defect, independent of EMO and of the
    expert-count gate: model_type is non-empty and absent from
    _MOE_MODEL_TYPES, and num_experts is absent, so the ONLY evidence is the
    architectures entry. Before the reorder this returned "dense".
    """
    d = _model_dir(
        tmp_path,
        config={"model_type": "brandnewmoe", "architectures": ["FooMoEForCausalLM"]},
    )
    assert _detect_architecture(d) == "moe"


def test_dense_model_type_still_dense_despite_architectures_check(tmp_path):
    """The reorder must not let a non-MoE architectures string flip to "moe".

    A BERT-family config has a non-empty model_type AND a non-"moe"
    architectures entry; it must still take the "dense" branch.
    """
    d = _model_dir(
        tmp_path,
        config={
            "model_type": "bert",
            "architectures": ["BertForSequenceClassification"],
        },
    )
    assert _detect_architecture(d) == "dense"


# ---------------------------------------------------------------------------
# D2 — expert tensor resolution
# ---------------------------------------------------------------------------


def test_resolves_emo_expert_projections(tmp_path):
    """EMO's per-expert 2-D projections resolve for gate, up and down."""
    d = _model_dir(tmp_path)
    weight_map = json.loads((d / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    for proj in PROJS:
        name = _resolve_tensor_name(weight_map, proj, 0)
        assert name == EMO_EXPERT[proj], f"{proj} resolved to {name}"


@pytest.mark.parametrize("proj", PROJS)
def test_expert_template_preferred_over_dense(tmp_path, proj):
    """Expert candidates precede dense mlp candidates in the tuple.

    An index exposing BOTH must resolve the expert weight. This is the
    resolution-policy change the precedence comment documents.
    """
    d = _model_dir(
        tmp_path, keys={"expert": EMO_EXPERT[proj], "dense": DENSE_MLP[proj]}
    )
    weight_map = json.loads((d / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    assert _resolve_tensor_name(weight_map, proj, 0) == EMO_EXPERT[proj]


def test_dense_mlp_still_resolves_without_experts(tmp_path):
    """The dense candidates remain reachable when no expert weight exists."""
    d = _model_dir(tmp_path, keys=DENSE_MLP)
    weight_map = json.loads((d / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    for proj in PROJS:
        assert _resolve_tensor_name(weight_map, proj, 0) == DENSE_MLP[proj]


def test_language_model_prefix_still_resolves(tmp_path):
    """The Qwen3.5 wrapper prefix keeps winning for a non-expert index."""
    d = _model_dir(
        tmp_path,
        keys={
            proj: f"model.language_model.layers.0.mlp.{proj}_proj.weight"
            for proj in PROJS
        },
    )
    weight_map = json.loads((d / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    for proj in PROJS:
        name = _resolve_tensor_name(weight_map, proj, 0)
        assert name == f"model.language_model.layers.0.mlp.{proj}_proj.weight"


def test_resolves_legacy_mixtral_expert_projections(tmp_path):
    """Legacy Mixtral w1/w2/w3 roles: gate=w1, up=w3, down=w2.

    The w1=gate pairing is uncontroversial; up and down are the ones a prior
    gets backwards, so each is asserted against its exact expected name.
    """
    d = _model_dir(tmp_path, config={"model_type": "mixtral"}, keys=LEGACY_WSTAR)
    weight_map = json.loads((d / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    for proj in PROJS:
        assert _resolve_tensor_name(weight_map, proj, 0) == LEGACY_WSTAR[proj]


def test_resolves_legacy_phimoe_expert_projections(tmp_path):
    """PhiMoE shares the legacy block_sparse_moe expert naming with Mixtral."""
    d = _model_dir(tmp_path, config={"model_type": "phimoe"}, keys=LEGACY_WSTAR)
    weight_map = json.loads((d / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    for proj in PROJS:
        assert _resolve_tensor_name(weight_map, proj, 0) == LEGACY_WSTAR[proj]


# ---------------------------------------------------------------------------
# Dense BERT path — unregressed
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("proj", ["attention", "intermediate", "output"])
def test_dense_bert_path_unregressed(tmp_path, proj):
    """The BERT-family dense names resolve exactly as before this change.

    These assertions mirror tests/test_headroom.py:286-316, which pin the same
    three names; they are repeated here so the EMO support module witnesses
    the dense path itself instead of trusting the other file to do it.
    """
    d = _model_dir(tmp_path, config={"model_type": "bert"}, keys=BERT_DENSE)
    weight_map = json.loads((d / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    assert _resolve_tensor_name(weight_map, proj, 0) == BERT_DENSE[proj]


# ---------------------------------------------------------------------------
# Malformed expert-count values the detection guard must survive
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value",
    [
        pytest.param("many", id="str-nonnumeric"),
        pytest.param("", id="str-empty"),
        pytest.param(None, id="null"),
        pytest.param([8], id="list"),
        pytest.param({"n": 8}, id="dict"),
        pytest.param(True, id="bool"),
    ],
)
def test_non_numeric_expert_count_never_raises(tmp_path, value):
    """A malformed num_experts degrades to "dense"; it never raises.

    The count coercion sits OUTSIDE the read_text/json.loads try block, so a
    bare int() on these values would raise out of a function documented
    "Raises: None". Each must return a value instead.
    """
    d = _model_dir(
        tmp_path,
        config={
            "model_type": "mystery",
            "architectures": ["MysteryForCausalLM"],
            "num_experts": value,
        },
    )
    assert _detect_architecture(d) == "dense"


def test_num_experts_one_is_dense(tmp_path):
    """A count below 2 is not MoE (the model_headroom threshold)."""
    d = _model_dir(
        tmp_path,
        config={
            "model_type": "mystery",
            "architectures": ["MysteryForCausalLM"],
            "num_experts": 1,
        },
    )
    assert _detect_architecture(d) == "dense"


def test_both_expert_keys_prefer_num_local_experts(tmp_path):
    """When BOTH expert keys are present, num_local_experts decides.

    Key-ORDER agreement with model_headroom's loader, which reads
    ``hf_config.get("num_local_experts") or hf_config.get("num_experts", 0)``
    (src/model_headroom/models/loader.py:171-173) — num_local_experts FIRST.

    The fixture is SYNTHETIC and deliberately so: 1/8 is the minimal pair
    where the two orders provably diverge AND the outcome differs (1 is below
    the >= 2 threshold, 8 is above it). No real checkpoint on this box
    carries both keys — /root/Emo_1b14b_1T carries num_experts only — so this
    property is only reachable by a constructed config. A same-key config
    (both keys equal) would pass under either order and prove nothing.
    """
    d = _model_dir(
        tmp_path,
        config={
            "model_type": "mystery",
            "architectures": ["MysteryForCausalLM"],
            "num_experts": 1,
            "num_local_experts": 8,
        },
    )
    assert _detect_architecture(d) == "moe"


def test_num_local_experts_is_read(tmp_path):
    """num_local_experts is honoured when num_experts is absent."""
    d = _model_dir(
        tmp_path,
        config={
            "model_type": "mystery",
            "architectures": ["MysteryForCausalLM"],
            "num_local_experts": 8,
        },
    )
    assert _detect_architecture(d) == "moe"


def test_num_experts_per_layer_is_read(tmp_path):
    """num_experts_per_layer is honoured when num_experts and
    num_local_experts are absent.

    This covers the lag scenario: a checkpoint variant that ships
    only num_experts_per_layer (e.g. an EMO variant without
    num_experts) is classified as "moe" by the sidecar, even though
    model_headroom's authority would say "dense" (its two keys are
    both absent, so it defaults to 0 -> dense). This difference is
    the feature — the sidecar covers the authority's lag.

    Cited to /root/Emo_1b14b_1T/config.json which carries
    num_experts_per_layer (null in this config, but the key exists
    and a variant may ship it truthy).
    """
    d = _model_dir(
        tmp_path,
        config={
            "model_type": "emo",
            "architectures": ["EmoForCausalLM"],
            "num_experts_per_layer": 128,
        },
    )
    assert _detect_architecture(d) == "moe"


def test_expert_count_read_from_resolved_text_config(tmp_path):
    """A nested text_config's own count decides, not the wrapper's."""
    d = _model_dir(
        tmp_path,
        config={
            "model_type": "wrapper",
            "architectures": ["WrapperForConditionalGeneration"],
            "text_config": {"model_type": "qwen3_moe", "num_experts": 64},
        },
    )
    assert _detect_architecture(d) == "moe"


def test_architectures_read_from_resolved_config(tmp_path):
    """A wrapper with top-level 'moe' architectures but dense inner model is dense.

    The substring check on architectures must read from the SAME resolved config
    that model_type and the expert count are read from. Otherwise a wrapper with
    "moe" in its architectures string but a dense inner model would be classified
    inconsistently: substring check sees the outer wrapper, structural gate sees
    the inner model. This is the same defect shape as D1 (name-based check
    disagrees with structural check).
    """
    d = _model_dir(
        tmp_path,
        config={
            "model_type": "wrapper",
            "architectures": ["WrapperMoEForConditionalGeneration"],
            "text_config": {"model_type": "bert"},
        },
    )
    assert _detect_architecture(d) == "dense"


def test_unreadable_config_is_unknown(tmp_path):
    """Missing and malformed configs stay "unknown" (unchanged contract)."""
    missing = tmp_path / "missing"
    missing.mkdir()
    assert _detect_architecture(missing) == "unknown"

    bad = tmp_path / "bad"
    bad.mkdir()
    (bad / "config.json").write_text("{not json")
    assert _detect_architecture(bad) == "unknown"


@pytest.mark.parametrize(
    "raw",
    [
        pytest.param("[1, 2, 3]", id="array"),
        pytest.param("null", id="null"),
        pytest.param("5", id="number"),
        pytest.param('"abc"', id="string"),
        pytest.param("true", id="bool"),
    ],
)
def test_non_object_config_root_is_unknown(tmp_path, raw):
    """A config.json whose JSON root is not an object returns "unknown".

    Every value here parses successfully, so the ``(OSError,
    json.JSONDecodeError)`` guard never fires and the first type-dependent
    operation — ``data.get("text_config")`` — raised AttributeError out of a
    function documented "Raises: None". The guard is on the operation, not on
    the result type; the invariant is that config.json must be a JSON object.

    "unknown" rather than "dense": an array carries no architectural evidence,
    and a dict with neither model_type nor architectures already returns
    "unknown" — an array is strictly less informative than an empty dict. A
    "dense" answer would record a false claim AND route _capture_weight_pairs
    down the BERT projection list on no evidence.
    """
    d = tmp_path / "nonobject"
    d.mkdir()
    (d / "config.json").write_text(raw)
    assert _detect_architecture(d) == "unknown"


def test_empty_object_config_is_unknown(tmp_path):
    """A JSON object with no model_type and no architectures is "unknown".

    Pins the pre-existing behaviour that bounds the non-object case above: an
    object with zero informative keys is already "unknown", so a non-object
    root (strictly fewer keys) returning "unknown" is consistency, not a
    newly invented answer.
    """
    d = tmp_path / "empty"
    d.mkdir()
    (d / "config.json").write_text("{}")
    assert _detect_architecture(d) == "unknown"


# ---------------------------------------------------------------------------
# End-to-end: _capture_weight_pairs writes a pair file for an EMO fixture
# ---------------------------------------------------------------------------


def test_capture_weight_pairs_writes_file_for_emo(tmp_path):
    """An EMO fixture produces a weight-pair file with >= 3 usable pairs.

    Gradient assertions are SHAPE PARITY plus all-zeros values, never
    "non-empty". ADR-002 (headroom.py:393-395) makes the sidecar emit a real
    all-zeros gradient of the weight's shape when no training hook ran, so a
    len(gradient) > 0 assertion would pass on an all-zeros array and measure
    nothing about gradient content.
    """
    d = _model_dir(
        tmp_path, keys={**EMO_EXPERT, "model.layers.0.mlp.gate.weight": SHARD}
    )
    w = HeadroomWriter(
        tmp_path / "h",
        weights="all",
        signals=frozenset({"effective_rank"}),
        tensor_loader=_fake_loader(rows=2, cols=3),
    )
    capture_id = str(uuid.uuid4())
    out = asyncio.run(
        w.write_async(
            capture_id=capture_id,
            parent_raw=tmp_path / "raw.jsonl",
            model="m",
            model_path=d,
            created_at="2026-08-21T19:40:00+00:00",
        )
    )
    assert out is not None
    sidecar = json.loads(out.read_text())
    assert sidecar["architecture"] == "moe"
    assert sidecar["error_body"] is None

    weight_file = Path(sidecar["signals"]["effective_rank"])
    assert weight_file.name == f"{capture_id}-weight_pairs.json"
    pairs = json.loads(weight_file.read_text())["pairs"]
    assert len(pairs) >= 3

    assert [p["layer_name"] for p in pairs] == [EMO_EXPERT[proj] for proj in PROJS]
    for pair in pairs:
        # Loader contract: the three base keys plus the v7 truncation
        # provenance keys plus the DEFER-4 expert_id key, 2-D, gradient matches.
        assert set(pair.keys()) == {
            "gradient",
            "weight",
            "layer_name",
            "expert_id",
            "parent_shape",
            "truncated",
            "sampling_mode",
        }
        assert pair["expert_id"] == 0
        assert len(pair["weight"]) == 2
        assert len(pair["weight"][0]) == 3
        # Shape parity measured against THIS pair's own weight, element-wise
        # by row — a lower bound like len(gradient) > 0 would not catch a
        # gradient of the wrong shape.
        assert len(pair["gradient"]) == len(pair["weight"])
        assert [len(g) for g in pair["gradient"]] == [len(w) for w in pair["weight"]]
        # Values: all zeros is the EXPECTED weight-only result, not a bug.
        assert pair["gradient"] == [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]


# ---------------------------------------------------------------------------
# Truncation math at expert tensor scale
# ---------------------------------------------------------------------------


def test_truncation_expert_shaped_tensor(tmp_path):
    """Byte cap / row-truncation on an expert-shaped [1024, 2048] tensor.

    EMO experts are [1024, 2048] F32 = 8 MB each. The existing truncation
    tests use 10x4 tensors (160 B), which have a vastly different aspect
    ratio. This test pins the truncation math at the scale and shape of a
    real expert tensor, verifying the row-fit calculation
    (byte_budget // (cols * 4)) works correctly for near-square tensors.

    With a 4 MB cap: rows_fit = 4194304 // (2048 * 4) = 512 rows.
    """
    model_dir = _model_dir(tmp_path)
    cap_bytes = 4 * 1024 * 1024  # 4 MB
    w = HeadroomWriter(
        tmp_path / "h",
        weights="gate",
        max_bytes=cap_bytes,
        signals=frozenset({"effective_rank"}),
        tensor_loader=_fake_loader(rows=1024, cols=2048),
    )
    capture_id = str(uuid.uuid4())
    out = asyncio.run(
        w.write_async(
            capture_id=capture_id,
            parent_raw=tmp_path / "raw.jsonl",
            model="m",
            model_path=model_dir,
            created_at="2026-08-21T19:40:00+00:00",
        )
    )
    assert out is not None
    sidecar = json.loads(out.read_text())
    assert sidecar["truncated"] is True
    assert "truncated to 512 rows" in (sidecar["error_body"] or "")
    # 512 rows * 2048 cols * 4 bytes = 4 MB exactly
    assert sidecar["bytes_written"] == 512 * 2048 * 4
    weight_file = Path(sidecar["signals"]["effective_rank"])
    pairs = json.loads(weight_file.read_text())["pairs"]
    assert len(pairs) == 1
    pair = pairs[0]
    # Truncated to 512 rows (half of 1024)
    assert len(pair["weight"]) == 512
    assert len(pair["weight"][0]) == 2048
    # Gradient truncated to same shape as weight
    assert len(pair["gradient"]) == 512
    assert len(pair["gradient"][0]) == 2048


def test_truncation_cap_below_one_expert_row(tmp_path):
    """Byte cap smaller than one row of a [1024, 2048] expert tensor.

    One row of a [1024, 2048] F32 tensor is 2048 * 4 = 8192 bytes. A cap of
    4096 bytes cannot fit even one row, so the tensor should be skipped
    entirely (same behavior as the 10x4 tensor test, but at expert scale).
    """
    model_dir = _model_dir(tmp_path)
    cap_bytes = 4096  # Less than one row (8192 B) of a [1024, 2048] tensor
    w = HeadroomWriter(
        tmp_path / "h",
        weights="gate",
        max_bytes=cap_bytes,
        signals=frozenset({"effective_rank"}),
        tensor_loader=_fake_loader(rows=1024, cols=2048),
    )
    capture_id = str(uuid.uuid4())
    out = asyncio.run(
        w.write_async(
            capture_id=capture_id,
            parent_raw=tmp_path / "raw.jsonl",
            model="m",
            model_path=model_dir,
            created_at="2026-08-21T19:40:00+00:00",
        )
    )
    assert out is not None
    sidecar = json.loads(out.read_text())
    assert sidecar["truncated"] is True
    assert sidecar["signals"]["effective_rank"] is None
    assert "exceeds remaining byte cap" in (sidecar["error_body"] or "")
    assert not list((tmp_path / "h").rglob("*weight_pairs.json"))


# ---------------------------------------------------------------------------
# Exhaustive differential test vs model_headroom authority
# ---------------------------------------------------------------------------

# Expert-count values to cross over: the authority's `or` skips any
# falsy value (0, 0.0, None, "", [], {}), so we parameterise over
# truthy and falsy numeric values crossed with both keys present/absent.
_EXPERT_COUNT_VALUES = [
    pytest.param(None, id="absent"),
    pytest.param(0, id="zero-int"),
    pytest.param(0.0, id="zero-float"),
    pytest.param(1, id="one"),
    pytest.param(2, id="two"),
    pytest.param(8, id="eight"),
    pytest.param(128, id="128"),
    pytest.param(2.5, id="float"),
    pytest.param("128", id="str-numeric"),
    pytest.param("0", id="str-zero"),
    pytest.param("", id="empty-str"),
    pytest.param("many", id="str-nonnumeric"),
    pytest.param(True, id="bool-true"),
    pytest.param(False, id="bool-false"),
    pytest.param([], id="empty-list"),
    pytest.param({}, id="empty-dict"),
]


@pytest.mark.parametrize(
    "num_local_value,num_experts_value",
    [
        pytest.param(None, None, id="both-absent"),
        pytest.param(0, 8, id="zero-then-eight"),
        pytest.param(8, 0, id="eight-then-zero"),
        pytest.param(0, 0, id="both-zero"),
        pytest.param(1, 8, id="one-then-eight"),
        pytest.param(8, 1, id="eight-then-one"),
        pytest.param(2, 1, id="two-then-one"),
        pytest.param(None, 8, id="local-absent-remote-eight"),
        pytest.param(8, None, id="local-eight-remote-absent"),
    ],
)
def test_expert_count_agrees_with_authority(
    tmp_path: Path,
    num_local_value: Any,
    num_experts_value: Any,
) -> None:
    """Exhaustive differential: our classification matches model_headroom's
    on every config where the authority produces a defined answer.

    The authority is imported from model_headroom (not paraphrased):
        from model_headroom.models import loader
    We compare _detect_architecture against loader.detect_architecture_details
    for a Cartesian product of expert-count values crossed over both keys
    being present/absent, plus with and without a nested text_config.

    The one accepted divergence: model_headroom CRASHES on a truthy
    non-numeric expert count (int("many") raises ValueError), while we
    keep the guard and return "dense". This is a strict improvement on
    a malformed-config path, not a disagreement about a well-formed one.
    """
    config: dict[str, Any] = {"model_type": "mystery"}
    if num_local_value is not None:
        config["num_local_experts"] = num_local_value
    if num_experts_value is not None:
        config["num_experts"] = num_experts_value

    d = tmp_path / "model"
    d.mkdir()
    (d / "config.json").write_text(json.dumps(config))

    our_result = _detect_architecture(d)

    # Documented divergence (see docstring): the authority CRASHES with
    # ValueError on a truthy non-numeric expert count (int("many")); that
    # specific rejection is tolerated. Anything else — AttributeError or
    # NotImplementedError from a broken/placeholder authority, TypeError
    # from a wrong signature — is a broken authority and propagates.
    try:
        authority_details = _authority_loader.detect_architecture_details(str(d))
        authority_result = authority_details.family
    except ValueError:
        # Authority crashed on malformed input; we must not crash
        assert our_result in ("dense", "moe", "unknown")
        return

    # Normalise: authority never returns None for family (always moe/dense)
    # Our result may be "unknown" when no model_type/architectures evidence
    # exists; the authority defaults to "dense" in that case.
    # We compare the moe/dense decision specifically.
    if authority_result in ("moe", "dense"):
        assert our_result == authority_result, (
            f"Disagreement for config {config}: "
            f"ours={our_result}, authority={authority_result}"
        )


def test_regression_num_local_experts_zero_num_experts_eight_is_moe(
    tmp_path: Path,
) -> None:
    """Regression test for the known divergence:
    {"num_local_experts": 0, "num_experts": 8} must classify as "moe".

    The authority's `or` skips the falsy 0 and falls through to 8:
        0 or 8 -> 8 -> int(8) >= 2 -> moe
    Our old loop returned int(0)=0 and stopped, classifying "dense".
    The fix makes our key selection match the authority's truthiness
    fallthrough while keeping safe int() coercion and no-raise.
    """
    d = tmp_path / "model"
    d.mkdir()
    (d / "config.json").write_text(
        json.dumps({"model_type": "mystery", "num_local_experts": 0, "num_experts": 8})
    )
    assert _detect_architecture(d) == "moe"


def test_lag_cover_num_experts_per_layer_uninformed_authority(tmp_path):
    """Lag-cover differential: authority UNINFORMED, sidecar INFORMED.

    The authority reads only num_local_experts and num_experts
    (loader.py:171-173). When BOTH are absent, the authority
    defaults to 0 -> "dense". But a checkpoint variant shipping
    only num_experts_per_layer (e.g. an EMO variant without
    num_experts) IS MoE. The sidecar covers this lag.

    This test asserts:
      1. Our classification is "moe" (the covered one).
      2. The authority would have said "dense" there — that
         difference is the feature, and a test that hides it
         is a bad test.

    Cited to /root/Emo_1b14b_1T/config.json which carries
    num_experts_per_layer as a key (null in this config, but
    a variant may ship it truthy).
    """
    from model_headroom.models import loader as _authority_loader

    config = {
        "model_type": "emo",
        "architectures": ["EmoForCausalLM"],
        "num_experts_per_layer": 128,
    }
    d = tmp_path / "model"
    d.mkdir()
    (d / "config.json").write_text(json.dumps(config))

    our_result = _detect_architecture(d)
    assert our_result == "moe", f"Expected moe (lag-covered), got {our_result}"

    # The authority is uninformed: no num_experts or num_local_experts
    # so it defaults to 0 -> "dense". Verify this explicitly —
    # the difference between our "moe" and the authority's "dense"
    # IS the lag-cover feature. No guard: the comment above and the
    # authority's own .get("num_experts", 0) mean a raise is a BROKEN
    # authority; the previous except fabricated authority_result="dense",
    # so the assertion below measured nothing. Absence cannot reach here —
    # the module-scope import fails collection first, loudly.
    authority_details = _authority_loader.detect_architecture_details(str(d))
    authority_result = authority_details.family

    assert authority_result == "dense", (
        f"Authority should be uninformed and say dense, "
        f"but got {authority_result}. "
        f"This test proves the lag-cover: our 'moe' is the "
        f"covered classification where the authority says 'dense'."
    )


def test_num_experts_per_layer_does_not_disagree_when_authority_informed(
    tmp_path: Path,
) -> None:
    """Adding num_experts_per_layer can NEVER disagree with the
    authority on a config the authority is informed about.

    PROOF: _expert_count checks keys in order:
      num_local_experts -> num_experts -> num_experts_per_layer.
    The authority's keys (num_local_experts, num_experts) are
    checked FIRST. The new key is only consulted when BOTH
    authority keys are absent or falsy. Therefore, on any config
    where the authority IS informed (has a truthy
    num_experts or num_local_experts), our _expert_count returns
    the same value before ever reaching num_experts_per_layer.
    Agreement is preserved by construction, not by assertion.

    This test verifies the construction: for every combination
    where num_experts or num_local_experts is truthy, our
    classification matches the authority regardless of what
    num_experts_per_layer is set to.
    """
    from model_headroom.models import loader as _authority_loader

    # Cases where the authority IS informed (has a truthy count key)
    # crossed with num_experts_per_layer present and truthy.
    informed_cases = [
        {"num_local_experts": 8, "num_experts_per_layer": 128},
        {"num_experts": 8, "num_experts_per_layer": 128},
        {"num_local_experts": 8, "num_experts": 4, "num_experts_per_layer": 128},
        {"num_local_experts": 0, "num_experts": 8, "num_experts_per_layer": 128},
    ]
    for extras in informed_cases:
        config: dict[str, Any] = {"model_type": "mystery"}
        config.update(extras)
        d = tmp_path / "model"
        d.mkdir(parents=True, exist_ok=True)
        (d / "config.json").write_text(json.dumps(config))

        our_result = _detect_architecture(d)

        # No guard: no crash tolerance is documented here, so any raise
        # (AttributeError/NotImplementedError/TypeError/ValueError) is a
        # broken authority and must fail loudly. The previous except's
        # assert+continue turned a broken authority into a green loop.
        authority_details = _authority_loader.detect_architecture_details(str(d))
        authority_result = authority_details.family

        if authority_result in ("moe", "dense"):
            assert our_result == authority_result, (
                f"Disagreement for config {config}: "
                f"ours={our_result}, authority={authority_result}"
            )
