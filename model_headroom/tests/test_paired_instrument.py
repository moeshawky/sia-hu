from __future__ import annotations

import inspect
import subprocess
import sys

import pytest
import torch

from model_headroom.paired import instrument
from model_headroom.paired.instrument import (
    InstrumentationBudget,
    PairedActivationRecorder,
    discover_buffer_backed_modules,
)


class _Block(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mlp = torch.nn.Linear(4, 4)
        self.input_layernorm = torch.nn.LayerNorm(4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.input_layernorm(self.mlp(x))


class _Model(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.block = _Block()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class _Wide(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mlp = torch.nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class _Sixty(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = torch.nn.ModuleList([torch.nn.Linear(2, 2) for _ in range(60)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


def _sixty_paths(model: _Sixty) -> tuple[str, ...]:
    return tuple(
        name for name, _ in model.named_modules() if name.startswith("layers.")
    )


def _fixed_linear(linear: torch.nn.Linear) -> None:
    with torch.no_grad():
        linear.weight.copy_(torch.arange(16, dtype=torch.float32).reshape(4, 4))
        linear.bias.zero_()


class _StickyHooks(dict[int, object]):
    def pop(self, key: int, default: object = None) -> object:
        return default


class TestBudgetDefaults:
    def test_default_budget_bounds_modules_bytes_and_tokens(self) -> None:
        budget = InstrumentationBudget()
        assert budget.max_modules == 8
        assert budget.max_bytes_per_module == 1 << 20
        assert budget.max_tokens_per_module == 64
        assert budget.sample_every_n_tokens == 1

    def test_default_budget_captures_eight_of_sixty_and_records_no_skips(self) -> None:
        model = _Sixty()
        paths = _sixty_paths(model)
        recorder = PairedActivationRecorder(model, module_paths=paths)
        with recorder.capture_forward():
            model(torch.ones(1, 2))
        manifest = recorder.last_manifest
        assert manifest is not None
        assert len(manifest.module_paths) == 60
        assert len(manifest.captured) == 8
        assert [entry.module_path for entry in manifest.captured] == list(paths[:8])
        assert manifest.skipped == ()

    def test_default_module_selection_is_not_exhaustive(self) -> None:
        model = _Model()
        recorder = PairedActivationRecorder(model)
        with recorder.capture_forward():
            model(torch.ones(1, 4))
        manifest = recorder.last_manifest
        assert manifest is not None
        assert "block.mlp" in manifest.module_paths
        assert "block.input_layernorm" in manifest.module_paths
        assert "block" not in manifest.module_paths
        assert "" not in manifest.module_paths

    def test_available_module_paths_lists_every_module(self) -> None:
        recorder = PairedActivationRecorder(_Model())
        paths = recorder.available_module_paths()
        assert "" in paths
        assert "block" in paths
        assert "block.mlp" in paths
        assert recorder.available_module_paths() is paths


class TestRelease:
    def test_hooks_are_removed_when_the_body_raises(self) -> None:
        model = _Model()
        recorder = PairedActivationRecorder(model, module_paths=("block.mlp",))
        with pytest.raises(RuntimeError, match="boom"), recorder.capture_forward():
            raise RuntimeError("boom")
        assert dict(model.block.mlp._forward_hooks) == {}
        manifest = recorder.last_manifest
        assert manifest is not None
        assert manifest.release_verified is True

    def test_backward_hooks_are_removed_when_the_body_raises(self) -> None:
        model = _Model()
        recorder = PairedActivationRecorder(model, module_paths=("block.mlp",))
        with pytest.raises(RuntimeError, match="boom"), recorder.capture_backward():
            raise RuntimeError("boom")
        assert dict(model.block.mlp._backward_hooks) == {}
        manifest = recorder.last_manifest
        assert manifest is not None
        assert manifest.release_verified is True

    def test_release_is_idempotent_and_leaves_no_handles(self) -> None:
        model = _Model()
        recorder = PairedActivationRecorder(model, module_paths=("block.mlp",))
        with recorder.capture_forward():
            model.block.mlp(torch.ones(1, 4))
        assert recorder.release() is True
        assert recorder.release() is True
        assert dict(model.block.mlp._forward_hooks) == {}

    def test_release_removes_hooks_registered_outside_the_context_manager(self) -> None:
        model = _Model()
        recorder = PairedActivationRecorder(model, module_paths=("block.mlp",))
        context = recorder.capture_forward()
        context.__enter__()
        model(torch.ones(1, 4))
        assert dict(model.block.mlp._forward_hooks) != {}
        assert recorder.release() is True
        assert dict(model.block.mlp._forward_hooks) == {}

    def test_second_release_after_an_out_of_band_capture_is_true(self) -> None:
        model = _Model()
        recorder = PairedActivationRecorder(model, module_paths=("block.mlp",))
        context = recorder.capture_forward()
        context.__enter__()
        model(torch.ones(1, 4))
        assert recorder.release() is True
        assert recorder.release() is True
        assert dict(model.block.mlp._forward_hooks) == {}

    def test_backward_release_removes_hooks_registered_outside_the_context_manager(
        self,
    ) -> None:
        model = _Model()
        recorder = PairedActivationRecorder(model, module_paths=("block.mlp",))
        context = recorder.capture_backward()
        context.__enter__()
        model(torch.ones(1, 4, requires_grad=True)).sum().backward()
        assert dict(model.block.mlp._backward_hooks) != {}
        assert recorder.release() is True
        assert dict(model.block.mlp._backward_hooks) == {}

    def test_release_reports_failure_when_a_handle_survives_removal(self) -> None:
        model = _Model()
        recorder = PairedActivationRecorder(model, module_paths=("block.mlp",))
        context = recorder.capture_forward()
        context.__enter__()
        model(torch.ones(1, 4))
        model.block.mlp._forward_hooks = _StickyHooks(model.block.mlp._forward_hooks)
        assert len(model.block.mlp._forward_hooks) == 1
        assert recorder.release() is False
        assert len(model.block.mlp._forward_hooks) == 1

    def test_unknown_module_path_raises_naming_the_path(self) -> None:
        recorder = PairedActivationRecorder(_Model(), module_paths=("block.absent",))
        with (
            pytest.raises(KeyError, match=r"module path not found: block\.absent"),
            recorder.capture_forward(),
        ):
            pass


class TestBudgetExclusion:
    def test_row_exceeding_the_byte_budget_is_not_measured_not_dropped(self) -> None:
        model = _Wide()
        budget = InstrumentationBudget(max_bytes_per_module=8)
        recorder = PairedActivationRecorder(model, module_paths=("mlp",), budget=budget)
        with recorder.capture_forward():
            model(torch.ones(4, 4))
        manifest = recorder.last_manifest
        assert manifest is not None
        assert len(manifest.captured) == 1
        entry = manifest.captured[0]
        assert entry.module_path == "mlp"
        assert entry.status == "not_measured"
        assert entry.sampled_rows == 0
        assert entry.values == ()
        assert entry.byte_size == 0
        assert entry.sampling_strategy == "not_sampled:row_exceeds_byte_budget"
        assert manifest.skipped == ()
        assert manifest.dropped_bytes == 64

    def test_row_slice_within_the_byte_budget_is_measured(self) -> None:
        model = _Wide()
        budget = InstrumentationBudget(max_bytes_per_module=16)
        recorder = PairedActivationRecorder(model, module_paths=("mlp",), budget=budget)
        with recorder.capture_forward():
            model(torch.ones(4, 4))
        manifest = recorder.last_manifest
        assert manifest is not None
        entry = manifest.captured[0]
        assert entry.status == "measured"
        assert entry.sampling_strategy == "row_slice:1rows_of_4,stride_1"
        assert entry.sampled_rows == 4
        assert entry.byte_size == 16
        assert len(entry.values) == 4

    def test_token_cap_limits_the_number_of_sampled_values(self) -> None:
        model = _Model()
        budget = InstrumentationBudget(max_tokens_per_module=3)
        recorder = PairedActivationRecorder(
            model, module_paths=("block.mlp",), budget=budget
        )
        _fixed_linear(model.block.mlp)
        with recorder.capture_forward():
            model(torch.ones(2, 4))
        manifest = recorder.last_manifest
        assert manifest is not None
        entry = manifest.captured[0]
        assert entry.status == "measured"
        assert entry.sampled_rows == 3
        assert entry.values == (6.0, 22.0, 38.0)

    def test_token_stride_pins_which_values_are_sampled(self) -> None:
        model = _Model()
        budget = InstrumentationBudget(max_tokens_per_module=3, sample_every_n_tokens=2)
        recorder = PairedActivationRecorder(
            model, module_paths=("block.mlp",), budget=budget
        )
        _fixed_linear(model.block.mlp)
        with recorder.capture_forward():
            model(torch.ones(2, 4))
        manifest = recorder.last_manifest
        assert manifest is not None
        entry = manifest.captured[0]
        assert entry.sampled_rows == 3
        assert entry.values == (6.0, 38.0, 6.0)


class TestBackwardCapture:
    def test_backward_capture_records_a_measured_gradient(self) -> None:
        model = _Model()
        recorder = PairedActivationRecorder(model, module_paths=("block.mlp",))
        with recorder.capture_backward(kind="gradient"):
            model(torch.ones(1, 4, requires_grad=True)).sum().backward()
        manifest = recorder.last_manifest
        assert manifest is not None
        assert len(manifest.captured) == 1
        entry = manifest.captured[0]
        assert entry.kind == "gradient"
        assert entry.status == "measured"
        assert entry.values != ()
        assert dict(model.block.mlp._backward_hooks) == {}


class TestBufferBackedDiscovery:
    def test_keyword_buffers_are_reported(self) -> None:
        model = torch.nn.Module()
        model.scale = torch.nn.Module()
        model.scale.register_buffer("scale", torch.ones(2))
        model.codes = torch.nn.Module()
        model.codes.register_buffer("codes", torch.ones(2))
        model.codebook = torch.nn.Module()
        model.codebook.register_buffer("codebook", torch.ones(2))
        model.residual = torch.nn.Module()
        model.residual.register_buffer("residual", torch.ones(2))
        assert discover_buffer_backed_modules(model) == (
            "scale",
            "codes",
            "codebook",
            "residual",
        )

    def test_plain_linear_is_not_reported(self) -> None:
        model = _Model()
        assert discover_buffer_backed_modules(model) == ()

    def test_unmatched_buffer_name_is_not_reported(self) -> None:
        model = torch.nn.Module()
        model.norm = torch.nn.Module()
        model.norm.register_buffer("position", torch.ones(2))
        assert discover_buffer_backed_modules(model) == ()


class TestImportIsolation:
    def test_import_is_torch_free_in_a_fresh_process(self) -> None:
        code = (
            "import sys, model_headroom.paired.instrument; "
            "print('torch' in sys.modules)"
        )
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            check=True,
        )
        assert result.stdout.strip() == "False"

    def test_module_scope_imports_no_torch(self) -> None:
        source = inspect.getsource(instrument)
        module_scope = source.split("class _Handle")[0]
        assert "import torch" not in module_scope
