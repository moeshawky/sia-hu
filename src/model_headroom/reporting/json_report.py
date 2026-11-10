from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from model_headroom.models.types import HeadroomReport


def _signal_to_dict(signal: Any) -> dict[str, Any]:
    result: dict[str, Any] = {
        "signal_name": signal.signal_name,
        "status": signal.status,
        "duration_seconds": signal.duration_seconds,
        "source_type": getattr(signal, "source_type", "unavailable"),
        "verification_status": getattr(signal, "verification_status", "unavailable"),
    }
    sampling_mode = getattr(signal, "sampling_mode", None)
    if sampling_mode is not None:
        result["sampling_mode"] = sampling_mode

    if signal.data is not None:
        if isinstance(signal.data, list) and all(
            hasattr(item, "__dict__") for item in signal.data
        ):
            result["data"] = [
                {
                    key: (list(value) if isinstance(value, tuple) else value)
                    for key, value in item.__dict__.items()
                }
                for item in signal.data
            ]
        elif hasattr(signal.data, "__dict__"):
            result["data"] = {}
            for key, value in signal.data.__dict__.items():
                result["data"][key] = value
                if isinstance(value, tuple):
                    result["data"][key] = list(value)
        else:
            result["data"] = str(signal.data)

    if getattr(signal, "error_message", None) is not None:
        result["error_message"] = signal.error_message

    return result


def generate_json_report(report: HeadroomReport) -> str:
    """Serialize a HeadroomReport to a JSON string with 2-space indentation. Builds a top-level dict containing model_name, model_path, timestamp, overall_headroom_score, recommendations, metadata, analysis_mode, decision_grade, and a signals list produced by _signal_to_dict. When coverage_metrics is present, adds the full metrics dict plus flattened top-level fields (measurement_coverage, headroom_score_measured, score_lower_bound, score_upper_bound). Uses default=str for non-serializable values."""
    report_dict: dict[str, Any] = {
        "model_name": report.model_name,
        "model_path": report.model_path,
        "timestamp": report.timestamp,
        "overall_headroom_score": report.overall_headroom_score,
        "recommendations": report.recommendations,
        "metadata": report.metadata,
        "signals": [_signal_to_dict(s) for s in report.signals],
        "analysis_mode": report.analysis_mode,
        "decision_grade": report.decision_grade,
    }

    if report.coverage_metrics is not None:
        report_dict["coverage_metrics"] = {
            "measured_score": report.coverage_metrics.measured_score,
            "coverage": report.coverage_metrics.coverage,
            "lower_bound": report.coverage_metrics.lower_bound,
            "upper_bound": report.coverage_metrics.upper_bound,
            "measured_weight": report.coverage_metrics.measured_weight,
            "total_applicable_weight": report.coverage_metrics.total_applicable_weight,
            "unmeasured_weight": report.coverage_metrics.unmeasured_weight,
            "decision_grade": report.coverage_metrics.decision_grade,
        }
        report_dict["measurement_coverage"] = report.coverage_metrics.coverage
        report_dict["headroom_score_measured"] = report.coverage_metrics.measured_score
        report_dict["score_lower_bound"] = report.coverage_metrics.lower_bound
        report_dict["score_upper_bound"] = report.coverage_metrics.upper_bound

    return json.dumps(report_dict, indent=2, default=str)
