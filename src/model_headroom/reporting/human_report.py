from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from model_headroom.models.types import HeadroomReport, SignalResult


def _format_signal_status(signal: Any) -> str:
    name = signal.signal_name.replace("_", " ").title()

    if signal.status == "not_applicable":
        return f"  {name:<30} [N/A — not applicable to this architecture]"
    if signal.status == "single_expert":
        return f"  {name:<30} [SINGLE EXPERT — no routing to measure]"
    if signal.status == "expert_count_unknown":
        return (
            f"  {name:<30} [EXPERT COUNT UNKNOWN — 1 expert fed, architectural"
            " count absent; cannot tell a dense model from a collapsed"
            " router]"
        )
    if signal.status == "skipped":
        return f"  {name:<30} [SKIPPED — no data provided]"
    if signal.status == "error":
        return f"  {name:<30} [ERROR — {getattr(signal.data, 'error_message', 'unknown') if signal.data else 'no data'}]"

    data = signal.data
    if data is None:
        return f"  {name:<30} [NO DATA]"

    prov = ""
    st = getattr(signal, "source_type", "measured")
    vs = getattr(signal, "verification_status", "verified")
    if st in ("synthetic", "fixture", "estimated", "unknown") or vs in (
        "unverified",
        "provenance_incomplete",
    ):
        prov = f" [{st}/{vs}]"
    sampling = getattr(signal, "sampling_mode", None)
    samp_str = f" {sampling}" if sampling and sampling != "unknown" else ""

    if signal.signal_name == "loss_saturation":
        status = "SATURATED" if data.is_saturated else "NOT SATURATED"
        return f"  {name:<30} {status} (slope={data.plateau_slope:.4f}){prov}"

    if signal.signal_name == "growth_factor":
        return f"  {name:<30} alpha={data.alpha:.3f} R\u00b2={data.fit_quality_r2:.3f}{prov}"

    if signal.signal_name == "effective_rank":
        cov = getattr(data, "coverage", None)
        cov_str = f" cov={cov:.1%}" if isinstance(cov, float) else ""
        mode = getattr(data, "sampling_mode", "unknown")
        mode_str = f" {mode}" if mode != "unknown" else samp_str
        return f"  {name:<30} rank={data.rank:.1f} ({data.percentage_of_full_rank:.1%}){mode_str}{cov_str}{prov}"

    if signal.signal_name == "grad_weight_ratio":
        if not data:
            return f"  {name:<30} [NO DATA]{prov}"
        high = sum(1 for r in data if r.is_high_pressure)
        return f"  {name:<30} {high}/{len(data)} layers high pressure{prov}"

    if signal.signal_name == "dead_neurons":
        src = getattr(data, "source_type", "measured")
        src_str = f" {src}" if src != "measured" else ""
        return f"  {name:<30} {data.dead_fraction:.1%} dead ({data.dead_count}/{data.total_neurons}){src_str}{prov}"

    if signal.signal_name == "load_balance":
        return f"  {name:<30} balance={data.balance_ratio:.2f} (ideal={data.ideal_entropy:.2f}){prov}"

    if signal.signal_name == "specialization":
        status = "COLLAPSED" if data.is_collapsed else "SPECIALIZED"
        return (
            f"  {name:<30} {status} (sim={data.mean_inter_expert_similarity:.3f}){prov}"
        )

    return f"  {name:<30} [UNKNOWN SIGNAL]{prov}"


_MISSING_SIGNAL_GUIDANCE: dict[str, tuple[str, str]] = {
    "loss_saturation": ("--loss-curve", "loss_curve JSON"),
    "growth_factor": ("--isoflop", "isoflop_points JSON"),
    "effective_rank": ("--weight-pairs-file", "weight_pairs JSON"),
    "grad_weight_ratio": ("--weight-pairs-file", "weight_pairs JSON"),
    "dead_neurons": ("--mlp-activations-file", "mlp_activations JSON"),
    "load_balance": ("--activations", "expert_activations JSON"),
    "specialization": ("--expert-weights-file", "expert_weights JSON"),
}


def _missing_signal_guidance(signals: list[SignalResult]) -> list[str]:
    missing = [s for s in signals if s.status in ("skipped", "not_applicable", "error")]
    if not missing:
        return []
    lines = ["  TO COMPUTE MISSING SIGNALS:"]
    lines.append("  " + "-" * 68)
    for signal in missing:
        flag, fmt = _MISSING_SIGNAL_GUIDANCE.get(signal.signal_name, ("—", "—"))
        lines.append(f"  {signal.signal_name:<22} {flag:<24} {fmt}")
    return lines


def _format_r03_bound(comparable: dict[str, Any]) -> list[str]:
    state = comparable.get("state")
    coverage_min = comparable.get("coverage_min")
    coverage_reported = comparable.get("coverage_reported")
    layers = comparable.get("layers_considered")
    reference = comparable.get("coverage_reference")

    coverage_str = (
        f"{coverage_min:.1%}" if isinstance(coverage_min, (int, float)) else "unknown"
    )
    reported_str = (
        f"{coverage_reported:.1%}"
        if isinstance(coverage_reported, (int, float))
        else "unknown"
    )
    reference_str = (
        f"~{reference:.0%}" if isinstance(reference, (int, float)) else "unknown"
    )

    if state == "absent":
        return [
            "  R-03 BOUND: [NO MEASUREMENT] effective_rank scored no rows of parent",
            f"     coverage {coverage_str} across {layers} layer(s);"
            f" scored layer covers {reported_str}.",
            "     The score line above describes an ABSENT measurement.",
            "     Do not use it for comparison, ranking, or a go/no-go decision.",
        ]

    if state == "full":
        return []

    lines = [
        "  R-03 BOUND: [PARTIAL TILE] effective_rank is not a full-matrix score",
        f"     tile covers {coverage_str} of parent (worst of {layers} layer(s)).",
        f"     The layer that produced the score covers {reported_str}.",
        "     Inflation against the full matrix is NOT measured by this tool.",
        f"     Comparable to full-matrix scores only above {reference_str} coverage.",
        f"     {reference_str} is an empirical reference from synthetic"
        " spectra, not a law.",
    ]
    conflict = comparable.get("layers_untruncated_but_partial")
    if conflict:
        lines.append(
            f"     {conflict} layer(s) declare truncated=false while the tile is"
            " a strict"
        )
        lines.append("     subset of the parent. Geometry was trusted, not the flag.")
    return lines


def generate_human_report(report: HeadroomReport) -> str:
    """Format a HeadroomReport into an 80-column terminal report string. Builds a header with model metadata, a composite score bar (40 chars), optional measured-score bar and score range from coverage_metrics, a per-signal status section via _format_signal_status, wrapped numbered recommendations (78-col width), gradient diagnostics and module family summary when present in metadata, and a footer. Truncates to 200 lines when output is longer, appending an ellipsis line."""
    lines: list[str] = []

    lines.append("=" * 72)
    lines.append("  MODEL HEADROOM ANALYSIS REPORT")
    lines.append("=" * 72)
    lines.append(f"  Model: {report.model_name}")
    lines.append(f"  Path:  {report.model_path}")
    lines.append(f"  Time:  {report.timestamp}")
    architecture = report.metadata.get("architecture", "unknown")
    lines.append(f"  Arch:  {architecture.upper()}")
    arch_details = report.metadata.get("architecture_details")
    if isinstance(arch_details, dict):
        subtype = arch_details.get("subtype", "unknown")
        flags = []
        if arch_details.get("has_linear_attention"):
            flags.append("linear_attn")
        if arch_details.get("has_full_attention"):
            flags.append("full_attn")
        if arch_details.get("has_vision_tower"):
            flags.append("vision")
        if arch_details.get("has_mtp"):
            flags.append("mtp")
        flag_str = ",".join(flags) if flags else "none"
        lines.append(f"  Subtype: {subtype} [{flag_str}]")
    lines.append(
        f"  Mode: {report.analysis_mode}  Decision-grade: {report.decision_grade}"
    )
    lines.append("")

    score = report.overall_headroom_score
    bar_len = 40
    filled = int(score * bar_len)
    bar = "█" * filled + "░" * (bar_len - filled)
    lines.append(f"  HEADROOM SCORE: [{bar}] {score:.2f} (composite)")
    comparable = report.metadata.get("effective_rank_comparability")
    if isinstance(comparable, dict):
        lines.extend(_format_r03_bound(comparable))
    if report.coverage_metrics is not None:
        cm = report.coverage_metrics
        measured_bar = "█" * int(cm.measured_score * bar_len) + "░" * (
            bar_len - int(cm.measured_score * bar_len)
        )
        lines.append(
            f"  MEASURED SCORE: [{measured_bar}] {cm.measured_score:.3f} ({cm.coverage:.0%} coverage)"
        )
        lines.append(
            f"  SCORE RANGE: [{cm.lower_bound:.3f} .. {cm.upper_bound:.3f}] ({cm.measured_weight:.2f}/{cm.total_applicable_weight:.2f} measured)"
        )
        if not cm.decision_grade:
            lines.append(
                "  ⚠  NOT decision-grade: collect missing signals / verify provenance"
            )
            gaps = report.metadata.get("decision_grade_gaps")
            if isinstance(gaps, list) and gaps:
                for gap in gaps:
                    lines.append(f"     - {gap}")
    lines.append("")

    lines.append("  SIGNALS:")
    lines.append("  " + "-" * 68)
    for signal in report.signals:
        lines.append(_format_signal_status(signal))
    lines.extend(_missing_signal_guidance(report.signals))
    lines.append("")

    lines.append("  RECOMMENDATIONS:")
    lines.append("  " + "-" * 68)
    for i, rec in enumerate(report.recommendations, 1):
        words = rec.split()
        current_line = f"  {i}. "
        for word in words:
            if len(current_line) + len(word) + 1 > 78:
                lines.append(current_line)
                current_line = "     " + word + " "
            else:
                current_line += word + " "
        lines.append(current_line.rstrip())
    lines.append("")

    if report.metadata.get("gradient_diagnostics"):
        lines.append("  GRADIENT DIAGNOSTICS:")
        lines.append("  " + "-" * 68)
        gd = report.metadata["gradient_diagnostics"]
        fams = gd.get("module_families", [])
        if fams:
            lines.append(f"  Module families: {', '.join(str(f) for f in fams)}")
        coh = gd.get("coherence")
        if coh:
            lines.append(
                f"  Coherence: cos={coh.get('mean_cosine', 0):.3f} overlap={coh.get('subspace_overlap', 0):.3f}"
            )
        lines.append("")
    fam_summary = report.metadata.get("module_family_summary")
    if isinstance(fam_summary, dict):
        lines.append("  MODULE FAMILY SUMMARY:")
        lines.append("  " + "-" * 68)
        pressure = fam_summary.get("grad_pressure")
        if isinstance(pressure, dict):
            for fam, counts in pressure.items():
                lines.append(
                    f"  {fam:<20} grad pressure "
                    f"{counts.get('high_pressure', 0)}/{counts.get('layers', 0)} layers"
                )
        rank_fams = fam_summary.get("effective_rank")
        if isinstance(rank_fams, dict):
            for fam, entry in rank_fams.items():
                lines.append(
                    f"  {fam:<20} worst rank "
                    f"{entry.get('min_percentage_of_full_rank', 0):.1%} "
                    f"({entry.get('worst_layer', 'n/a')})"
                )
        lines.append("")

    lines.append("=" * 72)
    lines.append("  Generated by model_headroom v0.2.0")
    lines.append("=" * 72)

    report_text = "\n".join(lines)

    all_lines = report_text.split("\n")
    if len(all_lines) > 200:
        all_lines = all_lines[:198]
        all_lines.append("... (truncated)")
    return "\n".join(all_lines)
