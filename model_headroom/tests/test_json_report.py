from __future__ import annotations

import json
from typing import TYPE_CHECKING

from model_headroom.reporting.json_report import generate_json_report

if TYPE_CHECKING:
    from model_headroom.models.types import HeadroomReport


class TestJsonReport:
    def test_valid_json(self, headroom_report_computed: HeadroomReport) -> None:
        output = generate_json_report(headroom_report_computed)
        parsed = json.loads(output)
        assert isinstance(parsed, dict)

    def test_contains_required_keys(
        self, headroom_report_computed: HeadroomReport
    ) -> None:
        output = generate_json_report(headroom_report_computed)
        parsed = json.loads(output)
        assert "overall_headroom_score" in parsed
        assert "recommendations" in parsed
        assert "signals" in parsed
        assert "model_name" in parsed

    def test_score_in_range(self, headroom_report_computed: HeadroomReport) -> None:
        output = generate_json_report(headroom_report_computed)
        parsed = json.loads(output)
        assert 0.0 <= parsed["overall_headroom_score"] <= 1.0

    def test_recommendations_count(
        self, headroom_report_computed: HeadroomReport
    ) -> None:
        output = generate_json_report(headroom_report_computed)
        parsed = json.loads(output)
        assert len(parsed["recommendations"]) >= 3

    def test_minimal_report(self, headroom_report_minimal: HeadroomReport) -> None:
        output = generate_json_report(headroom_report_minimal)
        parsed = json.loads(output)
        assert len(parsed["signals"]) == 7
