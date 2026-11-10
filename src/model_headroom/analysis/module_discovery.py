"""Architecture-aware module discovery for intervention targeting.

Generalized module discovery that works from the model's actual computational
structure (named_modules / named_parameters), not assumptions inherited from
transformer tutorials. Classifies modules into families and generates candidate
adapter configurations with explicit inclusion/exclusion tracking.

Platform-agnostic: no hardcoded architecture names. Uses introspection only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ModuleInfo:
    """Metadata for a single discoverable module."""

    path: str
    family: str  # ModuleFamily value
    module_type: str  # type(module).__name__
    param_count: int
    trainable: bool
    has_weight: bool
    has_bias: bool
    children: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class AdapterCandidate:
    """A candidate adapter configuration with explicit targeting."""

    name: str
    target_modules: list[str]
    excluded_modules: list[str]
    matched_count: int
    trainable_params: int
    frozen_params: int
    estimated_adapter_bytes: int
    family_coverage: dict[str, float]  # family -> fraction targeted
    assumptions: list[str]
    incompatibilities: list[str]


def discover_modules(model: Any) -> list[ModuleInfo]:
    """Discover all leaf modules with parameters from a model.

    Uses named_modules() introspection. Classifies each module by family
    using get_module_family(). Returns modules sorted by path.
    """
    from model_headroom.models.loader import get_module_family

    modules: list[ModuleInfo] = []
    all_paths = {name for name, _ in model.named_modules()}

    for name, mod in model.named_modules():
        if name == "":
            continue
        # Only leaf modules (no children) with parameters
        children = [c for c in all_paths if c.startswith(name + ".") and c != name]
        is_leaf = len(children) == 0
        params = list(mod.parameters(recurse=False))
        if not params and not is_leaf:
            continue
        if not params:
            continue

        param_count = sum(p.numel() for p in params)
        trainable = any(p.requires_grad for p in params)
        has_weight = any("weight" in n for n, _ in mod.named_parameters(recurse=False))
        has_bias = any("bias" in n for n, _ in mod.named_parameters(recurse=False))

        modules.append(
            ModuleInfo(
                path=name,
                family=get_module_family(name),
                module_type=type(mod).__name__,
                param_count=param_count,
                trainable=trainable,
                has_weight=has_weight,
                has_bias=has_bias,
                children=children,
            )
        )

    modules.sort(key=lambda m: m.path)
    return modules


def generate_adapter_candidates(
    modules: list[ModuleInfo],
    target_families: list[str] | None = None,
    exclude_families: list[str] | None = None,
    rank: int = 8,
    alpha: int = 16,
) -> list[AdapterCandidate]:
    """Generate candidate adapter configurations from a module inventory.

    Produces candidates for common targeting strategies:
    - attention-only (q/k/v/o projections)
    - ffn-only (gate/up/down)
    - attention+ffn (standard LoRA)
    - all-linear (broadest)
    - custom (user-specified families)

    Each candidate tracks explicit inclusion/exclusion and family coverage.
    """
    if target_families is None:
        target_families = []
    if exclude_families is None:
        exclude_families = ["vision", "embedding", "lm_head", "mtp"]

    candidates: list[AdapterCandidate] = []

    # Strategy 1: attention-only
    attn_modules = [m for m in modules if m.family == "attention" and m.has_weight]
    candidates.append(
        _build_candidate(
            "attention-only",
            attn_modules,
            modules,
            exclude_families,
            rank,
            alpha,
            assumptions=[
                "Targets query/key/value/output projections only.",
                "Excludes FFN, embeddings, lm_head, vision, MTP.",
            ],
        )
    )

    # Strategy 2: ffn-only
    ffn_modules = [m for m in modules if m.family in ("ffn", "expert") and m.has_weight]
    candidates.append(
        _build_candidate(
            "ffn-only",
            ffn_modules,
            modules,
            exclude_families,
            rank,
            alpha,
            assumptions=[
                "Targets feed-forward projections only.",
                "May miss attention-mediated adaptation.",
            ],
        )
    )

    # Strategy 3: attention+ffn (standard)
    std_modules = [
        m
        for m in modules
        if m.family in ("attention", "ffn", "expert") and m.has_weight
    ]
    candidates.append(
        _build_candidate(
            "attention+ffn",
            std_modules,
            modules,
            exclude_families,
            rank,
            alpha,
            assumptions=[
                "Standard LoRA targeting: attention + FFN projections.",
                "Excludes embeddings, lm_head, vision, MTP.",
            ],
        )
    )

    # Strategy 4: all-linear (broadest)
    all_linear = [
        m for m in modules if m.has_weight and m.family not in exclude_families
    ]
    candidates.append(
        _build_candidate(
            "all-linear",
            all_linear,
            modules,
            exclude_families,
            rank,
            alpha,
            assumptions=[
                "Broadest targeting: all modules with weights except excluded families.",
                "Higher parameter count, more memory.",
            ],
            incompatibilities=[
                "May include modules not supported by all LoRA implementations.",
            ],
        )
    )

    # Strategy 5: custom families (if specified)
    if target_families:
        custom = [m for m in modules if m.family in target_families and m.has_weight]
        candidates.append(
            _build_candidate(
                "custom",
                custom,
                modules,
                exclude_families,
                rank,
                alpha,
                assumptions=[f"User-specified families: {target_families}."],
            )
        )

    return candidates


def _build_candidate(
    name: str,
    target_modules: list[ModuleInfo],
    all_modules: list[ModuleInfo],
    exclude_families: list[str],
    rank: int,
    alpha: int,
    assumptions: list[str] | None = None,
    incompatibilities: list[str] | None = None,
) -> AdapterCandidate:
    if assumptions is None:
        assumptions = []
    if incompatibilities is None:
        incompatibilities = []

    target_paths = [m.path for m in target_modules]
    excluded = [m.path for m in all_modules if m.family in exclude_families]
    # Also exclude non-target modules that are not in excluded families
    all_paths = {m.path for m in all_modules}
    target_set = set(target_paths)
    excluded_set = set(excluded)
    other_excluded = [
        p for p in all_paths if p not in target_set and p not in excluded_set
    ]
    all_excluded = sorted(excluded_set | set(other_excluded))

    trainable = sum(m.param_count for m in target_modules)
    frozen = sum(m.param_count for m in all_modules) - trainable
    # LoRA adapter: A (d x r) + B (r x d) per target = 2 * r * d per module
    adapter_bytes = (
        sum(
            2 * rank * m.param_count // max(m.param_count, 1) * 4
            for m in target_modules
        )
        * 0
    )  # placeholder
    # Better estimate: r * (in_features + out_features) * 4 bytes per module
    adapter_bytes = 0
    for m in target_modules:
        # Estimate from param_count: assume square-ish, adapter = 2 * rank * sqrt(params) * 4
        import math

        side = int(math.sqrt(max(m.param_count, 1)))
        adapter_bytes += 2 * rank * side * 4

    # Family coverage
    family_counts: dict[str, int] = {}
    family_totals: dict[str, int] = {}
    for m in all_modules:
        family_totals[m.family] = family_totals.get(m.family, 0) + 1
    for m in target_modules:
        family_counts[m.family] = family_counts.get(m.family, 0) + 1
    family_coverage = {}
    for fam, total in family_totals.items():
        targeted = family_counts.get(fam, 0)
        family_coverage[fam] = targeted / total if total > 0 else 0.0

    return AdapterCandidate(
        name=name,
        target_modules=target_paths,
        excluded_modules=all_excluded,
        matched_count=len(target_paths),
        trainable_params=trainable,
        frozen_params=frozen,
        estimated_adapter_bytes=adapter_bytes,
        family_coverage=family_coverage,
        assumptions=assumptions,
        incompatibilities=incompatibilities,
    )
