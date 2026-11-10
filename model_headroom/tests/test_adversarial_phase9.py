"""Adversarial tests for Phase 9 verification criteria."""

from __future__ import annotations

import torch

from model_headroom.analysis.scoring import _score_signal, compute_headroom_score
from model_headroom.analysis.training_feasibility import estimate_training_feasibility
from model_headroom.models.types import (
    GradientWeightPair,
    GradWeightRatio,
    HeadroomReport,
    MLPActivations,
    SignalResult,
)
from model_headroom.signals.dead_neurons import analyze_dead_neurons
from model_headroom.signals.effective_rank import compute_effective_rank
from model_headroom.signals.grad_weight_ratio import analyze_grad_weight_ratios


class TestSyntheticGradients:
    """Synthetic (all-zero) gradients must not pass as measured evidence."""

    def test_all_zero_gradient_is_not_computed(self) -> None:
        pairs = [
            GradientWeightPair(
                gradient=torch.zeros(4, 4),
                weight=torch.randn(4, 4),
                layer_name="layer.0",
            )
        ]
        data = analyze_grad_weight_ratios(pairs)
        assert data[0].signal_status == "no_gradient"
        sr = SignalResult(
            signal_name="grad_weight_ratio",
            status=data[0].signal_status,
            data=data,
            duration_seconds=0.0,
        )
        assert _score_signal(sr) is None

    def test_nan_gradient_is_invalid(self) -> None:
        pairs = [
            GradientWeightPair(
                gradient=torch.full((4, 4), float("nan")),
                weight=torch.randn(4, 4),
                layer_name="layer.0",
            )
        ]
        data = analyze_grad_weight_ratios(pairs)
        assert data[0].signal_status == "invalid_evidence"


class TestValidZeroGradients:
    """Genuine zero gradients (from backward) are distinguishable from missing."""

    def test_zero_gradient_has_no_gradient_status(self) -> None:
        # A real backward pass producing all zeros → no_gradient (not skipped/error)
        pairs = [
            GradientWeightPair(
                gradient=torch.zeros(4, 4),
                weight=torch.randn(4, 4),
                layer_name="layer.0",
            )
        ]
        data = analyze_grad_weight_ratios(pairs)
        assert data[0].signal_status == "no_gradient"
        assert data[0].signal_status != "skipped"


class TestMissingSignalsNoReward:
    """Missing/invalid signals must not raise composite confidence."""

    def test_all_skipped_composite_zero(self) -> None:
        signals = [
            SignalResult(
                signal_name="loss_saturation",
                status="skipped",
                data=None,
                duration_seconds=0.0,
            ),
            SignalResult(
                signal_name="growth_factor",
                status="skipped",
                data=None,
                duration_seconds=0.0,
            ),
            SignalResult(
                signal_name="effective_rank",
                status="skipped",
                data=None,
                duration_seconds=0.0,
            ),
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
        assert score == 0.0

    def test_invalid_signal_does_not_improve_composite(self) -> None:
        # no_gradient (cat 3) → 0.0 numerator, weight retained
        data = [
            GradWeightRatio(
                ratio=0.0,
                is_high_pressure=False,
                z_score=0.0,
                layer_name="l",
                signal_status="no_gradient",
            )
        ]
        sr = SignalResult(
            signal_name="grad_weight_ratio",
            status="no_gradient",
            data=data,
            duration_seconds=0.0,
        )
        assert _score_signal(sr) is None


class TestSparseCoverageInconclusive:
    """Sparse signal coverage must produce inconclusive result."""

    def test_single_signal_low_coverage(self) -> None:
        from model_headroom.analysis.scoring import compute_coverage_metrics
        from model_headroom.models.types import SaturationReport

        sat = SaturationReport(
            is_saturated=False,
            saturation_point_tokens=None,
            loss_delta_threshold=0.01,
            plateau_slope=0.0,
            data_points_used=5,
            signal_status="computed",
        )
        signals = [
            SignalResult(
                signal_name="loss_saturation",
                status="computed",
                data=sat,
                duration_seconds=0.0,
                source_type="measured",
            ),
            SignalResult(
                signal_name="growth_factor",
                status="skipped",
                data=None,
                duration_seconds=0.0,
            ),
            SignalResult(
                signal_name="effective_rank",
                status="skipped",
                data=None,
                duration_seconds=0.0,
            ),
            SignalResult(
                signal_name="grad_weight_ratio",
                status="skipped",
                data=None,
                duration_seconds=0.0,
            ),
            SignalResult(
                signal_name="dead_neurons",
                status="skipped",
                data=None,
                duration_seconds=0.0,
            ),
            SignalResult(
                signal_name="load_balance",
                status="skipped",
                data=None,
                duration_seconds=0.0,
            ),
            SignalResult(
                signal_name="specialization",
                status="skipped",
                data=None,
                duration_seconds=0.0,
            ),
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
        metrics = compute_coverage_metrics(report)
        assert metrics.coverage < 0.7  # below decision-grade


class TestDenseMoESemantics:
    """Dense and MoE applicability must be explicit."""

    def test_effective_rank_nan_returns_error(self) -> None:
        w = torch.full((4, 4), float("nan"))
        result = compute_effective_rank(w)
        assert result.signal_status == "error"

    def test_dead_neurons_nan_returns_error(self) -> None:
        acts = MLPActivations(
            activations=torch.full((2, 4, 4), float("nan")), layer_name="l"
        )
        result = analyze_dead_neurons(acts)
        assert result.signal_status == "error"


class TestRankSamplingLimitations:
    """Rank analysis must record sampling limitations."""

    def test_sampled_state_recorded(self) -> None:
        w = torch.randn(4, 8)
        result = compute_effective_rank(
            w, sampling_mode="single_tile", parent_shape=(64, 8), sample_count=1
        )
        assert result.state == "sampled"
        assert result.coverage is not None
        assert result.coverage < 1.0
        assert result.confidence == 0.3


class TestAdapterCounts:
    """Adapter counts must match instantiated parameters (structural check)."""

    def test_module_discovery_counts(self) -> None:
        import torch.nn as nn

        from model_headroom.analysis.module_discovery import discover_modules

        m = nn.Sequential(nn.Linear(4, 8), nn.Linear(8, 4))
        mods = discover_modules(m)
        total = sum(m.param_count for m in mods)
        actual = sum(p.numel() for p in m.parameters())
        assert total == actual


class TestMemoryEstimates:
    """Memory estimates must declare assumptions."""

    def test_estimate_has_assumptions(self) -> None:
        e = estimate_training_feasibility(model_params=1_000_000)
        assert len(e.assumptions) > 0
        assert e.lower_bound_gb < e.total_training_gb
        assert e.total_training_gb < e.upper_bound_gb

    def test_quantized_warning(self) -> None:
        e = estimate_training_feasibility(
            model_params=1_000_000, weight_dtype="int4", optimizer="adamw"
        )
        assert len(e.warnings) > 0


class TestVerificationLevels:
    """Verification must distinguish cached from reproduced."""

    def test_evidence_store_verify(self) -> None:
        import tempfile

        from model_headroom.analysis.evidence_store import EvidenceStore

        with tempfile.TemporaryDirectory() as d:
            store = EvidenceStore(d)
            art = store.store("test", {"value": 42})
            result = store.verify(art.artifact_id)
            assert result["verified"] is True

    def test_evidence_store_missing(self) -> None:
        import tempfile

        from model_headroom.analysis.evidence_store import EvidenceStore

        with tempfile.TemporaryDirectory() as d:
            store = EvidenceStore(d)
            result = store.verify("nonexistent")
            assert result["verified"] is False
