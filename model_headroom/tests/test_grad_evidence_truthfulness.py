"""Truthfulness of gradient evidence end to end.

Covers Defect A: the carrier stops fabricating zero gradients, the authority
loader tolerates an explicit absence, the signal reports None instead of 0.0,
the score is computed over measured pairs only, and the report renders the
measured/total distinction.
"""

from __future__ import annotations

import torch

from model_headroom.analysis.scoring import _score_signal, compute_headroom_score
from model_headroom.models.loader import load_weight_pairs_from_json
from model_headroom.models.types import (
    GradientWeightPair,
    GradWeightRatio,
    HeadroomReport,
    SignalResult,
)
from model_headroom.reporting.human_report import generate_human_report
from model_headroom.signals.grad_weight_ratio import analyze_grad_weight_ratios


def _pair(
    gradient: torch.Tensor | None,
    weight: torch.Tensor | None = None,
    layer_name: str = "l",
) -> GradientWeightPair:
    if weight is None:
        weight = torch.ones(4, 4)
    return GradientWeightPair(
        gradient=gradient,
        weight=weight,
        layer_name=layer_name,
    )


class TestSignalNullGradient:
    """Null or missing gradient must yield None, not 0.0."""

    def test_null_gradient_yields_none(self) -> None:
        data = analyze_grad_weight_ratios([_pair(None)])
        assert len(data) == 1
        r = data[0]
        assert r.signal_status == "no_gradient"
        assert r.ratio is None
        assert r.z_score is None
        assert r.is_high_pressure is None

    def test_all_zero_gradient_yields_none(self) -> None:
        data = analyze_grad_weight_ratios([_pair(torch.zeros(4, 4))])
        assert len(data) == 1
        r = data[0]
        assert r.signal_status == "no_gradient"
        assert r.ratio is None
        assert r.z_score is None
        assert r.is_high_pressure is None

    def test_real_gradient_is_unchanged(self) -> None:
        gradient = torch.ones(4, 4) * 0.5
        weight = torch.ones(4, 4)
        data = analyze_grad_weight_ratios([_pair(gradient, weight, "l0")])
        assert len(data) == 1
        r = data[0]
        assert r.signal_status == "computed"
        assert r.ratio == 0.5
        assert r.z_score == 0.0
        assert r.is_high_pressure is False


class TestScoreMeasuredOnly:
    """Composite score must exclude unmeasured pairs from the denominator."""

    def test_all_real_gradients_identical_to_pre_change(self) -> None:
        pairs = [
            _pair(torch.ones(4, 4) * 0.1, torch.ones(4, 4), "l1"),
            _pair(torch.ones(4, 4) * 0.2, torch.ones(4, 4), "l2"),
        ]
        data = analyze_grad_weight_ratios(pairs)
        sr = SignalResult(
            signal_name="grad_weight_ratio",
            status="computed",
            data=data,
            duration_seconds=0.0,
        )
        score = _score_signal(sr)
        # Both computed, neither high pressure: 1.0 - 0/2 = 1.0
        assert score == 1.0

    def test_composite_all_placeholders_is_zero_not_one(self) -> None:
        pairs = [
            _pair(None, torch.ones(4, 4), "l1"),
            _pair(None, torch.ones(4, 4), "l2"),
        ]
        data = analyze_grad_weight_ratios(pairs)
        signals = [
            SignalResult(
                signal_name="grad_weight_ratio",
                status="computed",
                data=data,
                duration_seconds=0.0,
            )
        ]
        report = HeadroomReport(
            model_name="m",
            model_path="/tmp",
            timestamp="t",
            signals=signals,
            overall_headroom_score=0.0,
            recommendations=[],
            metadata={},
        )
        score = compute_headroom_score(report)
        # Composite is a float: None signal score → 0.0 numerator, weight retained.
        # Score is 0.0, never 1.0 (the inflation is gone).
        assert score == 0.0

    def test_mix_uses_measured_denominator(self) -> None:
        data = [
            GradWeightRatio(
                ratio=0.1,
                is_high_pressure=False,
                z_score=0.0,
                layer_name="l1",
                signal_status="computed",
            ),
            GradWeightRatio(
                ratio=10.0,
                is_high_pressure=True,
                z_score=1.0,
                layer_name="l2",
                signal_status="computed",
            ),
            GradWeightRatio(
                ratio=None,
                is_high_pressure=None,
                z_score=None,
                layer_name="l3",
                signal_status="no_gradient",
            ),
        ]
        sr = SignalResult(
            signal_name="grad_weight_ratio",
            status="computed",
            data=data,
            duration_seconds=0.0,
        )
        score = _score_signal(sr)
        # Measured only: 2 pairs, 1 high pressure: 1.0 - 1/2 = 0.5
        assert score == 0.5


class TestLoaderToleratesNullGradient:
    """The authority loader must accept null/absent gradient without dropping."""

    def test_null_gradient_loads_pair_not_dropped(self, tmp_json_file) -> None:
        payload = {
            "pairs": [
                {
                    "gradient": None,
                    "weight": [[1.0, 2.0], [3.0, 4.0]],
                    "layer_name": "l1",
                }
            ]
        }
        path = tmp_json_file(payload)
        pairs = load_weight_pairs_from_json(path)
        assert pairs is not None
        assert len(pairs) == 1
        assert pairs[0].gradient is None
        assert pairs[0].layer_name == "l1"

    def test_absent_gradient_key_loads_pair(self, tmp_json_file) -> None:
        payload = {
            "pairs": [
                {
                    "weight": [[1.0, 2.0], [3.0, 4.0]],
                    "layer_name": "l1",
                }
            ]
        }
        path = tmp_json_file(payload)
        pairs = load_weight_pairs_from_json(path)
        assert pairs is not None
        assert len(pairs) == 1
        assert pairs[0].gradient is None

    def test_legacy_all_zero_gradient_still_loads(self, tmp_json_file) -> None:
        payload = {
            "pairs": [
                {
                    "gradient": [[0.0, 0.0], [0.0, 0.0]],
                    "weight": [[1.0, 2.0], [3.0, 4.0]],
                    "layer_name": "l1",
                }
            ]
        }
        path = tmp_json_file(payload)
        pairs = load_weight_pairs_from_json(path)
        assert pairs is not None
        assert len(pairs) == 1
        assert pairs[0].gradient is not None
        assert torch.all(pairs[0].gradient == 0.0)


class TestCompositeScoreIntegration:
    """End-to-end composite score behaviour."""

    def test_composite_real_gradients_identical(self) -> None:
        pairs = [
            _pair(torch.ones(4, 4) * 0.1, torch.ones(4, 4), "l1"),
            _pair(torch.ones(4, 4) * 0.2, torch.ones(4, 4), "l2"),
        ]
        data = analyze_grad_weight_ratios(pairs)
        signals = [
            SignalResult(
                signal_name="grad_weight_ratio",
                status="computed",
                data=data,
                duration_seconds=0.0,
            )
        ]
        report = HeadroomReport(
            model_name="m",
            model_path="/tmp",
            timestamp="t",
            signals=signals,
            overall_headroom_score=0.0,
            recommendations=[],
            metadata={},
        )
        score = compute_headroom_score(report)
        assert score == 1.0

    def test_composite_all_placeholders_is_zero_not_one(self) -> None:
        pairs = [
            _pair(None, torch.ones(4, 4), "l1"),
            _pair(None, torch.ones(4, 4), "l2"),
        ]
        data = analyze_grad_weight_ratios(pairs)
        signals = [
            SignalResult(
                signal_name="grad_weight_ratio",
                status="computed",
                data=data,
                duration_seconds=0.0,
            )
        ]
        report = HeadroomReport(
            model_name="m",
            model_path="/tmp",
            timestamp="t",
            signals=signals,
            overall_headroom_score=0.0,
            recommendations=[],
            metadata={},
        )
        score = compute_headroom_score(report)
        # Composite is a float: None signal score → 0.0 numerator, weight retained.
        # Score is 0.0, never 1.0 (the inflation is gone).
        assert score == 0.0

    def test_composite_mix_measured_denominator(self) -> None:
        data = [
            GradWeightRatio(
                ratio=0.1,
                is_high_pressure=False,
                z_score=0.0,
                layer_name="l1",
                signal_status="computed",
            ),
            GradWeightRatio(
                ratio=10.0,
                is_high_pressure=True,
                z_score=1.0,
                layer_name="l2",
                signal_status="computed",
            ),
            GradWeightRatio(
                ratio=None,
                is_high_pressure=None,
                z_score=None,
                layer_name="l3",
                signal_status="no_gradient",
            ),
        ]
        signals = [
            SignalResult(
                signal_name="grad_weight_ratio",
                status="computed",
                data=data,
                duration_seconds=0.0,
            )
        ]
        report = HeadroomReport(
            model_name="m",
            model_path="/tmp",
            timestamp="t",
            signals=signals,
            overall_headroom_score=0.0,
            recommendations=[],
            metadata={},
        )
        score = compute_headroom_score(report)
        # Measured only: 2 pairs, 1 high pressure: 1.0 - 1/2 = 0.5
        assert score == 0.5


class TestReportMeasuredTotal:
    """Human report must render measured/total, not imply all layers measured."""

    def test_report_renders_measured_total(self) -> None:
        data = [
            GradWeightRatio(
                ratio=0.1,
                is_high_pressure=False,
                z_score=0.0,
                layer_name="l1",
                signal_status="computed",
            ),
            GradWeightRatio(
                ratio=None,
                is_high_pressure=None,
                z_score=None,
                layer_name="l2",
                signal_status="no_gradient",
            ),
        ]
        signals = [
            SignalResult(
                signal_name="grad_weight_ratio",
                status="computed",
                data=data,
                duration_seconds=0.0,
            )
        ]
        report = HeadroomReport(
            model_name="m",
            model_path="/tmp",
            timestamp="t",
            signals=signals,
            overall_headroom_score=0.0,
            recommendations=[],
            metadata={},
        )
        text = generate_human_report(report)
        assert "1/2 measured" in text
        assert "0/1 layers high pressure" in text
