from __future__ import annotations

from typing import Any, ClassVar

from model_headroom.analysis.orchestrator import run_analysis
from model_headroom.models.types import (
    ExpertActivationData,
    ExpertWeights,
    GradientWeightPair,
    IsoFLOPPoint,
    LossDataPoint,
    MLPActivations,
    WeightSample,
)


class TestOrchestratorFunctional:
    def test_all_signals_computed(self) -> None:
        report = run_analysis(
            model_name="test",
            model_path="/tmp/test",
            loss_curve=[
                LossDataPoint(tokens_trained=100_000_000, validation_loss=2.0),
                LossDataPoint(tokens_trained=200_000_000, validation_loss=1.5),
                LossDataPoint(tokens_trained=300_000_000, validation_loss=1.0),
                LossDataPoint(tokens_trained=400_000_000, validation_loss=0.5),
                LossDataPoint(tokens_trained=500_000_000, validation_loss=0.3),
            ],
            isoflop_points=[
                IsoFLOPPoint(
                    compute_flops=1e18, model_params=10_000_000, measured_loss=2.0
                ),
                IsoFLOPPoint(
                    compute_flops=1e19, model_params=100_000_000, measured_loss=1.5
                ),
                IsoFLOPPoint(
                    compute_flops=1e20, model_params=1_000_000_000, measured_loss=1.2
                ),
                IsoFLOPPoint(
                    compute_flops=1e21, model_params=10_000_000_000, measured_loss=1.0
                ),
            ],
            weight_pairs=[
                GradientWeightPair(
                    gradient=__import__("torch").ones(10, 10) * 0.1,
                    weight=__import__("torch").ones(10, 10),
                    layer_name="layer_0",
                ),
                GradientWeightPair(
                    gradient=__import__("torch").ones(10, 10) * 0.2,
                    weight=__import__("torch").ones(10, 10),
                    layer_name="layer_1",
                ),
                GradientWeightPair(
                    gradient=__import__("torch").ones(10, 10) * 0.3,
                    weight=__import__("torch").ones(10, 10),
                    layer_name="layer_2",
                ),
            ],
            mlp_activations=[
                MLPActivations(
                    activations=__import__("torch").ones(2, 10, 64),
                    layer_name="layer_0",
                ),
            ],
            expert_weight_sets=[
                ExpertWeights(
                    expert_weights=[
                        __import__("torch").randn(32, 64) for _ in range(3)
                    ],
                    layer_name="layer_0",
                ),
            ],
            activation_data=ExpertActivationData(
                metadata={"num_experts": 3, "num_layers": 1},
                layers={"0": {"0": 33, "1": 33, "2": 34}},
            ),
        )
        assert len(report.signals) == 7
        assert all(s.status == "computed" for s in report.signals)
        assert 0.0 <= report.overall_headroom_score <= 1.0
        assert len(report.recommendations) >= 3


class TestOrchestratorMixed:
    def test_mixed_computed_skipped(self) -> None:
        report = run_analysis(
            model_name="test",
            model_path="/tmp/test",
            loss_curve=[
                LossDataPoint(tokens_trained=100_000_000, validation_loss=2.0),
                LossDataPoint(tokens_trained=200_000_000, validation_loss=1.5),
                LossDataPoint(tokens_trained=300_000_000, validation_loss=1.0),
                LossDataPoint(tokens_trained=400_000_000, validation_loss=0.5),
                LossDataPoint(tokens_trained=500_000_000, validation_loss=0.3),
            ],
        )
        computed = sum(1 for s in report.signals if s.status == "computed")
        skipped = sum(1 for s in report.signals if s.status == "skipped")
        assert computed == 1
        assert skipped == 6

    def test_all_skipped(self) -> None:
        report = run_analysis(model_name="test", model_path="/tmp/test")
        assert len(report.signals) == 7
        assert all(s.status == "skipped" for s in report.signals)
        assert 0.0 <= report.overall_headroom_score <= 1.0
        assert len(report.recommendations) >= 3


class TestOrchestratorTiming:
    def test_duration_non_negative(self) -> None:
        report = run_analysis(model_name="test", model_path="/tmp/test")
        assert all(s.duration_seconds >= 0.0 for s in report.signals)


class TestOrchestratorWeightSamples:
    def _samples(self) -> list[Any]:
        torch = __import__("torch")
        return [
            WeightSample(
                weight=torch.ones(8, 8),
                layer_name="layer_0",
                parent_shape=(16, 16),
                truncated=False,
                sampling_mode="full_matrix",
            ),
        ]

    def _signal(self, report: Any, name: str) -> Any:
        return next(s for s in report.signals if s.signal_name == name)

    def test_weight_samples_alone_effective_rank_computed_grad_skipped(self) -> None:
        report = run_analysis(
            model_name="test",
            model_path="/tmp/test",
            architecture="dense",
            weight_samples=self._samples(),
        )
        assert self._signal(report, "effective_rank").status == "computed"
        assert self._signal(report, "grad_weight_ratio").status == "skipped"

    def test_weight_samples_provenance_key(self) -> None:
        report = run_analysis(
            model_name="test",
            model_path="/tmp/test",
            architecture="dense",
            weight_samples=self._samples(),
        )
        assert "weight_sample_provenance" in report.metadata
        assert "weight_pair_provenance" not in report.metadata

    def test_weight_pairs_win_over_samples(self) -> None:
        torch = __import__("torch")
        report = run_analysis(
            model_name="test",
            model_path="/tmp/test",
            architecture="dense",
            weight_pairs=[
                GradientWeightPair(
                    gradient=torch.ones(8, 8) * 0.1,
                    weight=torch.ones(8, 8),
                    layer_name="layer_0",
                    truncated=True,
                ),
            ],
            weight_samples=self._samples(),
        )
        assert "weight_pair_provenance" in report.metadata
        assert "weight_sample_provenance" not in report.metadata

    def test_empty_both_skipped(self) -> None:
        report = run_analysis(
            model_name="test",
            model_path="/tmp/test",
            architecture="dense",
            weight_pairs=[],
            weight_samples=[],
        )
        assert self._signal(report, "effective_rank").status == "skipped"

    def test_partial_view_comparability_not_absent(self) -> None:
        torch = __import__("torch")
        report = run_analysis(
            model_name="test",
            model_path="/tmp/test",
            architecture="dense",
            weight_samples=[
                WeightSample(
                    weight=torch.ones(32, 128),
                    layer_name="layer_0",
                    parent_shape=(64, 128),
                    truncated=False,
                ),
            ],
        )
        comp = report.metadata.get("effective_rank_comparability")
        assert comp is not None
        assert comp["state"] != "absent"


class TestOrchestratorTrainingProvenance:
    """Training telemetry (R-01 loss_saturation, R-02 growth_factor) provenance
    stamping per ticket 2026-11-16-headroom-training-signal-provenance-defect."""

    _loss_curve: ClassVar[list[Any]] = [
        LossDataPoint(tokens_trained=100_000_000, validation_loss=2.0),
        LossDataPoint(tokens_trained=200_000_000, validation_loss=1.5),
        LossDataPoint(tokens_trained=300_000_000, validation_loss=1.0),
        LossDataPoint(tokens_trained=400_000_000, validation_loss=0.5),
        LossDataPoint(tokens_trained=500_000_000, validation_loss=0.3),
    ]

    _isoflop_points: ClassVar[list[Any]] = [
        IsoFLOPPoint(compute_flops=1e18, model_params=10_000_000, measured_loss=2.0),
        IsoFLOPPoint(compute_flops=1e19, model_params=100_000_000, measured_loss=1.5),
        IsoFLOPPoint(compute_flops=1e20, model_params=1_000_000_000, measured_loss=1.2),
        IsoFLOPPoint(
            compute_flops=1e21, model_params=10_000_000_000, measured_loss=1.0
        ),
    ]

    def _signal(self, report: Any, name: str) -> Any:
        return next(s for s in report.signals if s.signal_name == name)

    def test_no_provenance_defaults_to_unavailable(self) -> None:
        """Without training provenance args, R-01/R-02 stay unavailable (no
        silent reclassification of historical callers)."""
        report = run_analysis(
            model_name="test",
            model_path="/tmp/test",
            loss_curve=self._loss_curve,
            isoflop_points=self._isoflop_points,
        )
        r01 = self._signal(report, "loss_saturation")
        r02 = self._signal(report, "growth_factor")
        assert r01.status == "computed"
        assert r02.status == "computed"
        assert r01.source_type == "unavailable"
        assert r01.verification_status == "unavailable"
        assert r02.source_type == "unavailable"
        assert r02.verification_status == "unavailable"

    def test_training_source_type_stamps_both_signals(self) -> None:
        """--training-source-type measured stamps R-01 and R-02."""
        report = run_analysis(
            model_name="test",
            model_path="/tmp/test",
            loss_curve=self._loss_curve,
            isoflop_points=self._isoflop_points,
            training_source_type="measured",
            training_verification_status="verified",
        )
        r01 = self._signal(report, "loss_saturation")
        r02 = self._signal(report, "growth_factor")
        assert r01.source_type == "measured"
        assert r01.verification_status == "verified"
        assert r02.source_type == "measured"
        assert r02.verification_status == "verified"

    def test_training_provenance_flags_independent_of_signal_status(self) -> None:
        """Provenance is stamped even when growth_factor is poor_fit (status
        and provenance are orthogonal per AC2)."""
        poor_points = [
            IsoFLOPPoint(
                compute_flops=1e18, model_params=10_000_000, measured_loss=1.0
            ),
            IsoFLOPPoint(
                compute_flops=1e19, model_params=100_000_000, measured_loss=1.0
            ),
            IsoFLOPPoint(
                compute_flops=1e20, model_params=1_000_000_000, measured_loss=1.0
            ),
        ]
        report = run_analysis(
            model_name="test",
            model_path="/tmp/test",
            loss_curve=self._loss_curve,
            isoflop_points=poor_points,
            training_source_type="measured",
            training_verification_status="verified",
        )
        r01 = self._signal(report, "loss_saturation")
        r02 = self._signal(report, "growth_factor")
        assert r01.source_type == "measured"
        assert r01.verification_status == "verified"
        assert r02.status == "poor_fit"
        assert r02.source_type == "measured"
        assert r02.verification_status == "verified"

    def test_five_measured_signals_reaches_full_coverage(self) -> None:
        """Dense model with all five applicable signals measured reaches
        coverage >= 0.70 (AC3)."""
        torch = __import__("torch")
        report = run_analysis(
            model_name="test",
            model_path="/tmp/test",
            architecture="dense",
            loss_curve=self._loss_curve,
            isoflop_points=self._isoflop_points,
            training_source_type="measured",
            training_verification_status="verified",
            weight_pairs=[
                GradientWeightPair(
                    gradient=torch.ones(10, 10) * (0.1 + i * 0.1),
                    weight=torch.ones(10, 10),
                    layer_name=f"layer_{i}",
                )
                for i in range(3)
            ],
            weight_pairs_source_type="measured",
            weight_pairs_verification_status="verified",
            mlp_activations=[
                MLPActivations(
                    activations=torch.ones(2, 10, 64),
                    layer_name="layer_0",
                ),
            ],
            mlp_source_type="measured",
            mlp_verification_status="verified",
        )
        coverage = report.coverage_metrics
        assert coverage is not None
        # All five applicable signals measured -> coverage = 1.0
        assert coverage.coverage >= 0.70

    def test_three_measured_signals_reports_50_percent(self) -> None:
        """Dense model with only three measured signals (R-03, R-04, R-05)
        reports ~50% coverage and is not decision-grade (AC3). R-01/R-02
        remain unmeasured because no training provenance is supplied."""
        torch = __import__("torch")
        report = run_analysis(
            model_name="test",
            model_path="/tmp/test",
            architecture="dense",
            loss_curve=self._loss_curve,
            isoflop_points=self._isoflop_points,
            # No training_source_type -> R-01/R-02 stay "unavailable"
            weight_pairs=[
                GradientWeightPair(
                    gradient=torch.ones(10, 10) * 0.1,
                    weight=torch.ones(10, 10),
                    layer_name="layer_0",
                ),
            ],
            weight_pairs_source_type="measured",
            weight_pairs_verification_status="verified",
            mlp_activations=[
                MLPActivations(
                    activations=torch.ones(2, 10, 64),
                    layer_name="layer_0",
                ),
            ],
            mlp_source_type="measured",
            mlp_verification_status="verified",
        )
        coverage = report.coverage_metrics
        assert coverage is not None
        # R-01/R-02 are NOT measured (no training provenance) so coverage
        # cannot reach decision-grade threshold even with R-03/R-04/R-05
        assert coverage.coverage < 0.70
        assert coverage.decision_grade is False
