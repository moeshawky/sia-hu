from __future__ import annotations

from typing import TYPE_CHECKING

from model_headroom.reporting.human_report import generate_human_report

if TYPE_CHECKING:
    from model_headroom.models.types import HeadroomReport


class TestHumanReport:
    def test_line_count(self, headroom_report_computed: HeadroomReport) -> None:
        output = generate_human_report(headroom_report_computed)
        assert len(output.split("\n")) <= 200

    def test_fits_80_columns(self, headroom_report_computed: HeadroomReport) -> None:
        output = generate_human_report(headroom_report_computed)
        for line in output.split("\n"):
            assert len(line) <= 80

    def test_minimal_report(self, headroom_report_minimal: HeadroomReport) -> None:
        output = generate_human_report(headroom_report_minimal)
        assert len(output.split("\n")) <= 200
        assert "MODEL HEADROOM ANALYSIS REPORT" in output

    def test_contains_all_signals(
        self, headroom_report_computed: HeadroomReport
    ) -> None:
        output = generate_human_report(headroom_report_computed)
        for name in [
            "Loss Saturation",
            "Growth Factor",
            "Effective Rank",
            "Grad Weight Ratio",
            "Dead Neurons",
            "Load Balance",
            "Specialization",
        ]:
            assert name in output
