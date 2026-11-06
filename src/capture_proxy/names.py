"""
Shared name contract for headroom capture — names only, zero heavy dependencies.

Owns: the single signal vocabulary (CLI and emitted forms), the CLI-to-emitted
      name map, the checkpoint tensor-name templates, and the MoE model-type set
Depends: std only (typing). NO torch, NO safetensors, NO numpy, NO
      model_headroom — this module is importable in off mode, which is what keeps
      the R-01 purity invariant intact while sharing names across repos.

Why this module exists: the carrier used to declare TWO hand-written signal
vocabularies (HEADROOM_SIGNALS for the CLI/env knob, SIGNAL_KEYS for the emitted
sidecar keys). They drifted: the CLI accepted `loss` and `isoflop` while the
sidecar emitted `loss_saturation` and `growth_factor`, so 2 of 6 CLI names did
not name the authority's signals. A name that resolves differently on the way in
and on the way out is exactly the shape that made this failure mode hard to diagnose.

Invariants:
  - every name in HEADROOM_SIGNALS resolves to exactly one name in SIGNAL_KEYS
    through SIGNAL_NAME_MAP; the map is total in both directions
  - SIGNAL_KEYS is the authority's core-6 signal names verbatim, in stable order
  - CLI names are operator-facing and are NOT renamed here; renaming a
    CAPTURE_HEADROOM env value is a breaking change, so the divergence is made
    explicit and total instead of silently collapsed
  - importing this module pulls no heavy dependency (R-01)
"""

from __future__ import annotations

from typing import Final

# MoE model types recognized from config.json model_type. Deliberately a NAME
# list only: it is a fast path for a config that carries no expert count, never
# an override of a structural verdict. An expert count >= 2 always wins.
MOE_MODEL_TYPES: Final[frozenset[str]] = frozenset(
    {"qwen3_moe", "qwen3_5_moe", "qwen2_moe", "mixtral", "olmoe", "phimoe"}
)

# Candidate tensor-name templates per projection, tried in order. The
# Qwen3_5 conditional-generation wrapper prefixes language_model.; plain
# dense checkpoints do not. The first name present in the safetensors
# index wins. {layer} is substituted with the captured layer index.
#
# Dense entries below are read off the libraries' own module structure,
# not recalled: RoBERTa keeps BERT's query/key/value + dense layout under
# a roberta. prefix (RobertaSelfAttention.query, RobertaSelfOutput.dense,
# RobertaIntermediate.dense, RobertaOutput.dense); DeBERTa-v2 uses
# query_proj/key_proj/value_proj under a deberta. prefix
# (DisentangledSelfAttention.query_proj, DebertaV2SelfOutput.dense,
# DebertaV2Intermediate.dense, DebertaV2Output.dense). A name that never
# matches is a dead entry, never a wrong read: resolution refuses rather
# than guesses, so an imperfect entry costs nothing and a missing entry
# is a silent skip (DEFER-3).
#
# Fused MoE entries (gate_up_proj [E,2I,H], down_proj [E,H,I]) sit with
# the per-expert templates, AFTER the indexed ones: on an index carrying
# both, the 2-D native weight keeps winning and behavior is unchanged;
# on a fused-only index the fused name is the only match. A 3-D match
# fans out to one 2-D bundle per expert in the loader (DEFER-2); the
# template layer only names, never reshapes.
#
# PRECEDENCE POLICY: MoE expert templates are tried BEFORE the dense mlp
# templates, because a checkpoint carrying both an expert stack and a dense
# mlp is a MoE checkpoint and the expert weight is the one that carries
# headroom signal. This is a resolution-policy change, not a cosmetic
# reorder: on such an index the dense name no longer wins. The two
# language_model.*/dense mlp names keep their existing relative order
# (wrapper prefix first) and remain last.
ARCHITECTURE_TENSOR_NAMES: Final[dict[str, tuple[str, ...]]] = {
    # MoE projections (Qwen/OLMoE/EMO families, then legacy Mixtral/PhiMoE).
    # EMO/OLMoE/Qwen-MoE keep per-expert 2-D weights under mlp.experts.{E}.
    "gate": (
        "model.layers.{layer}.mlp.experts.0.gate_proj.weight",
        "model.layers.{layer}.block_sparse_moe.experts.0.w1.weight",
        "model.layers.{layer}.mlp.experts.gate_up_proj.weight",
        "model.language_model.layers.{layer}.mlp.gate_proj.weight",
        "model.layers.{layer}.mlp.gate_proj.weight",
    ),
    # Mixtral/PhiMoE legacy pre-fusion roles: up=w3, down=w2 (NOT w2/w3).
    "up": (
        "model.layers.{layer}.mlp.experts.0.up_proj.weight",
        "model.layers.{layer}.block_sparse_moe.experts.0.w3.weight",
        "model.language_model.layers.{layer}.mlp.up_proj.weight",
        "model.layers.{layer}.mlp.up_proj.weight",
    ),
    "down": (
        "model.layers.{layer}.mlp.experts.0.down_proj.weight",
        "model.layers.{layer}.block_sparse_moe.experts.0.w2.weight",
        "model.layers.{layer}.mlp.experts.down_proj.weight",
        "model.language_model.layers.{layer}.mlp.down_proj.weight",
        "model.layers.{layer}.mlp.down_proj.weight",
    ),
    # Dense BERT-style projections (BERT/DeBERTa/RoBERTa).
    "attention": (
        "model.bert.encoder.layer.{layer}.attention.self.query.weight",
        "bert.encoder.layer.{layer}.attention.self.query.weight",
        "deberta.encoder.layer.{layer}.attention.self.query_proj.weight",
        "roberta.encoder.layer.{layer}.attention.self.query.weight",
    ),
    "intermediate": (
        "model.bert.encoder.layer.{layer}.intermediate.dense.weight",
        "bert.encoder.layer.{layer}.intermediate.dense.weight",
        "deberta.encoder.layer.{layer}.intermediate.dense.weight",
        "roberta.encoder.layer.{layer}.intermediate.dense.weight",
    ),
    "output": (
        "model.bert.encoder.layer.{layer}.output.dense.weight",
        "bert.encoder.layer.{layer}.output.dense.weight",
        "deberta.encoder.layer.{layer}.output.dense.weight",
        "roberta.encoder.layer.{layer}.output.dense.weight",
    ),
}

# The emitted sidecar signal keys, in stable order. These are the authority's
# core-6 signal names verbatim (model_headroom names: loss_saturation,
# growth_factor, effective_rank, dead_neurons, load_balance, specialization).
# The sidecar is a contract the authority consumes, so these names are fixed by
# the authority and are not the carrier's to rename.
SIGNAL_KEYS: Final[tuple[str, ...]] = (
    "effective_rank",
    "dead_neurons",
    "load_balance",
    "specialization",
    "loss_saturation",
    "growth_factor",
)

# The CLI/env vocabulary (CAPTURE_HEADROOM accepts a comma list of these).
# Operator-facing names; kept as-is so an existing env value keeps working.
HEADROOM_SIGNALS: Final[tuple[str, ...]] = (
    "effective_rank",
    "dead_neurons",
    "load_balance",
    "specialization",
    "loss",
    "isoflop",
)

# Total CLI -> emitted name map. Four names are identical on both sides; two are
# operator shorthand for an authority signal. The map is total in both
# directions so a name can never resolve to nothing and a dead name can never
# linger unlisted.
SIGNAL_NAME_MAP: Final[dict[str, str]] = {
    "effective_rank": "effective_rank",
    "dead_neurons": "dead_neurons",
    "load_balance": "load_balance",
    "specialization": "specialization",
    "loss": "loss_saturation",
    "isoflop": "growth_factor",
}
