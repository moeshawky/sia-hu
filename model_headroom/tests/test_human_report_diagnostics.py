from __future__ import annotations

from model_headroom.models.types import HeadroomReport, SignalResult
from model_headroom.reporting.human_report import generate_human_report


def _make_report(metadata: dict[str, object]) -> HeadroomReport:
    signals = [
        SignalResult(
            signal_name=name,
            status="skipped",
            data=None,
            duration_seconds=0.0,
        )
        for name in [
            "loss_saturation",
            "growth_factor",
            "effective_rank",
            "grad_weight_ratio",
            "dead_neurons",
            "load_balance",
            "specialization",
        ]
    ]
    return HeadroomReport(
        model_name="test_model",
        model_path="/tmp/test",
        timestamp="2026-01-01T00:00:00+00:00",
        signals=signals,
        overall_headroom_score=0.5,
        recommendations=["rec1"],
        metadata=metadata,
    )


class TestHumanReportGradientDiagnostics:
    def test_gradient_diagnostics_shown(self) -> None:
        report = _make_report(
            {
                "architecture": "dense",
                "gradient_diagnostics": {
                    "module_families": ["ffn", "attention"],
                    "coherence": {"mean_cosine": 0.75, "subspace_overlap": 0.6},
                },
            }
        )
        output = generate_human_report(report)
        assert "GRADIENT DIAGNOSTICS" in output
        assert "Module families" in output
        assert "ffn" in output
        assert "attention" in output
        assert "Coherence" in output
        assert "cos=0.750" in output
        assert "overlap=0.600" in output

    def test_gradient_diagnostics_no_families(self) -> None:
        report = _make_report(
            {
                "architecture": "dense",
                "gradient_diagnostics": {
                    "module_families": [],
                    "coherence": None,
                },
            }
        )
        output = generate_human_report(report)
        assert "GRADIENT DIAGNOSTICS" in output

    def test_gradient_diagnostics_no_coherence(self) -> None:
        report = _make_report(
            {
                "architecture": "dense",
                "gradient_diagnostics": {
                    "module_families": ["ffn"],
                    "coherence": None,
                },
            }
        )
        output = generate_human_report(report)
        assert "GRADIENT DIAGNOSTICS" in output
        assert "Coherence" not in output


class TestHumanReportModuleFamilySummary:
    def test_module_family_summary_shown(self) -> None:
        report = _make_report(
            {
                "architecture": "dense",
                "module_family_summary": {
                    "grad_pressure": {
                        "ffn": {"high_pressure": 2, "layers": 10},
                        "attention": {"high_pressure": 0, "layers": 5},
                    },
                    "effective_rank": {
                        "ffn": {
                            "min_percentage_of_full_rank": 0.45,
                            "worst_layer": "layer_3",
                        },
                    },
                },
            }
        )
        output = generate_human_report(report)
        assert "MODULE FAMILY SUMMARY" in output
        assert "ffn" in output
        assert "grad pressure" in output
        assert "2/10" in output
        assert "attention" in output
        assert "0/5" in output
        assert "worst tile rank" in output
        assert "45.0%" in output
        assert "layer_3" in output

    def test_module_family_summary_empty_pressure(self) -> None:
        report = _make_report(
            {
                "architecture": "dense",
                "module_family_summary": {
                    "grad_pressure": {},
                    "effective_rank": {},
                },
            }
        )
        output = generate_human_report(report)
        assert "MODULE FAMILY SUMMARY" in output


class TestHumanReportBothSections:
    def test_both_gradient_diagnostics_and_family_summary(self) -> None:
        report = _make_report(
            {
                "architecture": "dense",
                "gradient_diagnostics": {
                    "module_families": ["ffn"],
                    "coherence": {"mean_cosine": 0.5, "subspace_overlap": 0.3},
                },
                "module_family_summary": {
                    "grad_pressure": {"ffn": {"high_pressure": 1, "layers": 5}},
                    "effective_rank": {
                        "ffn": {"min_percentage_of_full_rank": 0.5, "worst_layer": "l0"}
                    },
                },
            }
        )
        output = generate_human_report(report)
        assert "GRADIENT DIAGNOSTICS" in output
        assert "MODULE FAMILY SUMMARY" in output
        lines = output.split("\n")
        grad_idx = lines.index(
            next(line for line in lines if "GRADIENT DIAGNOSTICS" in line)
        )
        fam_idx = lines.index(
            next(line for line in lines if "MODULE FAMILY SUMMARY" in line)
        )
        assert grad_idx < fam_idx
