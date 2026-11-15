"""Hook-based paired-activation and gradient capture for externally built models.

Budgets are pre-registered before any tensor is touched: max_bytes_per_module,
max_modules, max_tokens_per_module, sample_every_n_tokens. Defaults make
capturing every layer impossible by construction (max_bytes_per_module=1<<20,
max_modules=8, max_tokens_per_module=64, sample_every_n_tokens=1). The default
module selection is non-exhaustive: only paths ending in ".mlp" or
".input_layernorm" are selected; the full available module count is recorded in
the manifest so the caller sees the denominator. Buffer-backed module discovery
matches the small keyword set {"scale", "codes", "codebook", "residual"} against
buffer names discovered through named_modules() + named_buffers(). Torch is
imported lazily; the module imports cleanly when torch is absent. Every handle
is removed unconditionally in finally blocks; release() is idempotent and
returns False if a registered handle is still present after the removal attempt.

Budget exclusion: when a module's tensor cannot be sampled within
max_bytes_per_module (even the minimum non-empty row exceeds the cap), the
module still appears in manifest.captured with status="not_measured",
sampled_rows=0, values=(), byte_size=0, and a sampling_strategy naming the
exclusion. dropped_bytes records the bytes of budget-excluded modules as a
scalar side-channel, never as a substitute for the per-module record.
Module cap: max_modules is enforced inside each hook, so at most max_modules
tensors are recorded and the remaining paths are absent from captured with no
per-path record. skipped only collects paths the registration loop declined,
which needs the cap already reached before registration; captured grows during
the body, so with a positive max_modules skipped stays empty and module_paths
still carries the full denominator.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from time import perf_counter
from typing import TYPE_CHECKING, Literal, Protocol, cast

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Sequence


class _Handle(Protocol):
    id: int


class _HostModule(Protocol):
    def named_modules(self) -> object: ...

    def named_buffers(self) -> object: ...

    def register_forward_hook(self, hook: object) -> object: ...

    def register_full_backward_hook(self, hook: object) -> object: ...

    _forward_hooks: dict[int, object]
    _backward_hooks: dict[int, object]


_BUFFER_KEYWORDS = ("scale", "codes", "codebook", "residual")
_DEFAULT_MODULE_SUFFIXES = (".mlp", ".input_layernorm")


@dataclass(frozen=True)
class InstrumentationBudget:
    """Frozen capture budget bounding bytes, modules, tokens, and sampling stride."""

    max_bytes_per_module: int = 1 << 20
    max_modules: int = 8
    max_tokens_per_module: int = 64
    sample_every_n_tokens: int = 1


@dataclass(frozen=True)
class CapturedTensor:
    """Frozen record of one sampled activation or gradient tensor."""

    module_path: str
    kind: str
    shape: tuple[int, ...]
    dtype: str
    sampled_rows: int
    sampling_strategy: str
    status: str
    values: tuple[float, ...] = ()
    byte_size: int = 0


@dataclass(frozen=True)
class CaptureManifest:
    """Frozen record of one capture run: modules, budget, captures, release state."""

    model_id: str | None
    checkpoint_hash: str | None
    module_paths: tuple[str, ...]
    budget: InstrumentationBudget
    captured: tuple[CapturedTensor, ...]
    skipped: tuple[str, ...]
    dropped_bytes: int
    duration_ms: float
    release_verified: bool


class PairedActivationRecorder:
    """Forward/backward hook observer that always releases its handles."""

    def __init__(
        self,
        model: object,
        *,
        module_paths: Sequence[str] | None = None,
        budget: InstrumentationBudget | None = None,
    ) -> None:
        self._model: _HostModule = cast("_HostModule", model)
        self._budget = budget or InstrumentationBudget()
        self._handles: list[tuple[_HostModule, int]] = []
        self._available: tuple[str, ...] | None = None
        self._last_manifest: CaptureManifest | None = None
        if module_paths is None:
            self._module_paths: tuple[str, ...] = self._default_module_paths()
        else:
            self._module_paths = tuple(module_paths)

    @property
    def last_manifest(self) -> CaptureManifest | None:
        return self._last_manifest

    def _named_modules(self) -> dict[str, object]:
        result: dict[str, object] = {}
        for name, mod in cast(
            "Iterator[tuple[str, object]]", self._model.named_modules()
        ):
            result[name] = mod
        return result

    def _default_module_paths(self) -> tuple[str, ...]:
        paths: list[str] = []
        for name, _module in self._named_modules().items():
            if name.endswith(_DEFAULT_MODULE_SUFFIXES):
                paths.append(name)
        return tuple(paths)

    def available_module_paths(self) -> tuple[str, ...]:
        if self._available is not None:
            return self._available
        paths = tuple(name for name, _ in self._named_modules().items())
        self._available = paths
        return paths

    def _resolve_module(self, path: str) -> _HostModule:
        modules = self._named_modules()
        if path not in modules:
            msg = f"module path not found: {path}"
            raise KeyError(msg)
        return cast("_HostModule", modules[path])

    @staticmethod
    def _tensor_bytes(tensor: object) -> int:
        import torch

        if not torch.is_tensor(tensor):
            return 0
        return int(tensor.element_size()) * int(tensor.numel())

    @staticmethod
    def _shape_dtype(tensor: object) -> tuple[tuple[int, ...], str]:
        import torch

        if torch.is_tensor(tensor):
            return tuple(int(d) for d in tensor.shape), str(tensor.dtype)
        return (), "unknown"

    def _sample_flat(
        self, tensor: object, max_tokens: int, stride: int
    ) -> tuple[float, ...]:
        import torch

        if not torch.is_tensor(tensor) or tensor.numel() == 0:
            return ()
        step = max(1, stride)
        flat = tensor.detach().reshape(-1)
        indices = list(range(0, flat.numel(), step))[:max_tokens]
        if not indices:
            return ()
        sampled = flat.index_select(0, torch.tensor(indices, dtype=torch.long))
        return tuple(float(v) for v in sampled.cpu().tolist())

    def _sample_rows(
        self, tensor: object, max_bytes: int, max_tokens: int, stride: int
    ) -> tuple[tuple[float, ...], int, str]:
        import torch

        if not torch.is_tensor(tensor) or tensor.numel() == 0:
            return (), 0, "not_a_tensor"
        element_size = int(tensor.element_size())
        numel = int(tensor.numel())
        dim0 = int(tensor.shape[0]) if tensor.ndim > 0 else 1
        bytes_per_row = element_size * (numel // dim0) if dim0 else element_size * numel
        max_rows = max(1, max_bytes // bytes_per_row) if bytes_per_row else 1
        max_rows = min(max_rows, dim0)
        rows = list(range(0, dim0, stride))[:max_rows]
        if not rows:
            return (), 0, "stride_skipped_all"
        index = torch.tensor(rows, dtype=torch.long)
        slice_tensor = tensor.index_select(0, index).detach().cpu()
        values = tuple(float(v) for v in slice_tensor.reshape(-1).tolist())
        byte_size = element_size * len(rows) * (numel // dim0 if dim0 else numel)
        strategy = f"row_slice:{len(rows)}rows_of_{dim0},stride_{stride}"
        return values, byte_size, strategy

    def _capture_entry(
        self, module_path: str, kind: str, tensor: object
    ) -> CapturedTensor:
        import torch

        shape, dtype = self._shape_dtype(tensor)
        if not torch.is_tensor(tensor):
            return CapturedTensor(
                module_path=module_path,
                kind=kind,
                shape=shape,
                dtype=dtype,
                sampled_rows=0,
                sampling_strategy="not_a_tensor",
                status="not_measured",
            )
        total_bytes = self._tensor_bytes(tensor)
        if total_bytes <= self._budget.max_bytes_per_module:
            values = self._sample_flat(
                tensor,
                self._budget.max_tokens_per_module,
                self._budget.sample_every_n_tokens,
            )
            return CapturedTensor(
                module_path=module_path,
                kind=kind,
                shape=shape,
                dtype=dtype,
                sampled_rows=len(values),
                sampling_strategy="full",
                status="measured",
                values=values,
                byte_size=total_bytes,
            )
        values, byte_size, strategy = self._sample_rows(
            tensor,
            self._budget.max_bytes_per_module,
            self._budget.max_tokens_per_module,
            self._budget.sample_every_n_tokens,
        )
        if byte_size > self._budget.max_bytes_per_module:
            return CapturedTensor(
                module_path=module_path,
                kind=kind,
                shape=shape,
                dtype=dtype,
                sampled_rows=0,
                sampling_strategy="not_sampled:row_exceeds_byte_budget",
                status="not_measured",
                values=(),
                byte_size=0,
            )
        return CapturedTensor(
            module_path=module_path,
            kind=kind,
            shape=shape,
            dtype=dtype,
            sampled_rows=len(values),
            sampling_strategy=strategy,
            status="measured",
            values=values,
            byte_size=byte_size,
        )

    def _build_manifest(
        self,
        captured: tuple[CapturedTensor, ...],
        skipped: tuple[str, ...],
        dropped_bytes: int,
        duration_ms: float,
        release_verified: bool,
    ) -> CaptureManifest:
        model_id: str | None = None
        if hasattr(self._model, "name"):
            model_id = str(self._model.name)
        elif hasattr(self._model, "__class__"):
            model_id = self._model.__class__.__name__
        return CaptureManifest(
            model_id=model_id,
            checkpoint_hash=None,
            module_paths=self._module_paths,
            budget=self._budget,
            captured=captured,
            skipped=skipped,
            dropped_bytes=dropped_bytes,
            duration_ms=duration_ms,
            release_verified=release_verified,
        )

    @contextmanager
    def capture_forward(
        self, *, kind: str = "activation"
    ) -> Iterator[PairedActivationRecorder]:
        handles: list[tuple[_HostModule, int]] = []
        captured: list[CapturedTensor] = []
        skipped: list[str] = []
        dropped_bytes = 0
        start = perf_counter()
        try:
            for path in self._module_paths:
                if len(captured) >= self._budget.max_modules:
                    skipped.append(path)
                    continue
                module = self._resolve_module(path)

                def make_hook(p: str) -> Callable[[object, object, object], None]:
                    def hook(_mod: object, _inputs: object, output: object) -> None:
                        if len(captured) >= self._budget.max_modules:
                            return
                        entry = self._capture_entry(p, kind, output)
                        captured.append(entry)
                        if entry.status == "not_measured":
                            nonlocal dropped_bytes
                            dropped_bytes += self._tensor_bytes(output)

                    return hook

                handle = cast("_Handle", module.register_forward_hook(make_hook(path)))
                entry = (module, handle.id)
                handles.append(entry)
                self._handles.append(entry)
            yield self
        finally:
            for module, handle_id in handles:
                module._forward_hooks.pop(handle_id, None)
            self._handles = [h for h in self._handles if h not in handles]
            duration_ms = (perf_counter() - start) * 1000.0
            release_verified = all(
                handle_id not in module._forward_hooks for module, handle_id in handles
            )
            self._last_manifest = self._build_manifest(
                tuple(captured),
                tuple(skipped),
                dropped_bytes,
                duration_ms,
                release_verified,
            )

    @contextmanager
    def capture_backward(
        self, *, kind: str = "gradient"
    ) -> Iterator[PairedActivationRecorder]:
        handles: list[tuple[_HostModule, int]] = []
        captured: list[CapturedTensor] = []
        skipped: list[str] = []
        dropped_bytes = 0
        start = perf_counter()
        try:
            for path in self._module_paths:
                if len(captured) >= self._budget.max_modules:
                    skipped.append(path)
                    continue
                module = self._resolve_module(path)

                def make_hook(p: str) -> Callable[[object, object, object], None]:
                    def hook(
                        _mod: object, grad_input: object, grad_output: object
                    ) -> None:
                        if len(captured) >= self._budget.max_modules:
                            return
                        go = cast("tuple[object, ...]", grad_output)
                        target = go[0] if go else None
                        if target is None:
                            captured.append(
                                CapturedTensor(
                                    module_path=p,
                                    kind=kind,
                                    shape=(),
                                    dtype="unknown",
                                    sampled_rows=0,
                                    sampling_strategy="no_gradient",
                                    status="not_measured",
                                )
                            )
                            return
                        entry = self._capture_entry(p, kind, target)
                        captured.append(entry)
                        if entry.status == "not_measured":
                            nonlocal dropped_bytes
                            dropped_bytes += self._tensor_bytes(target)

                    return hook

                handle = cast(
                    "_Handle", module.register_full_backward_hook(make_hook(path))
                )
                entry = (module, handle.id)
                handles.append(entry)
                self._handles.append(entry)
            yield self
        finally:
            for module, handle_id in handles:
                module._backward_hooks.pop(handle_id, None)
            self._handles = [h for h in self._handles if h not in handles]
            duration_ms = (perf_counter() - start) * 1000.0
            release_verified = all(
                handle_id not in module._backward_hooks for module, handle_id in handles
            )
            self._last_manifest = self._build_manifest(
                tuple(captured),
                tuple(skipped),
                dropped_bytes,
                duration_ms,
                release_verified,
            )

    def release(self) -> bool:
        leaked = False
        for module, handle_id in self._handles:
            module._forward_hooks.pop(handle_id, None)
            module._backward_hooks.pop(handle_id, None)
            if (
                handle_id in module._forward_hooks
                or handle_id in module._backward_hooks
            ):
                leaked = True
        self._handles = []
        return not leaked

    def __enter__(self) -> PairedActivationRecorder:
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> Literal[False]:
        self.release()
        return False


def discover_buffer_backed_modules(model: object) -> tuple[str, ...]:
    host = cast("_HostModule", model)
    named_buffers: dict[str, object] = {}
    for name, tensor in cast("Iterator[tuple[str, object]]", host.named_buffers()):
        named_buffers[name] = tensor
    buffer_owner: dict[str, str] = {}
    for buf_name, _tensor in named_buffers.items():
        parts = buf_name.split(".")
        owner = ".".join(parts[:-1]) if len(parts) > 1 else ""
        buffer_owner[owner] = buf_name
    result: list[str] = []
    named_modules: dict[str, object] = {}
    for name, mod in cast("Iterator[tuple[str, object]]", host.named_modules()):
        named_modules[name] = mod
    for name in named_modules:
        if not name:
            continue
        if name in buffer_owner:
            buf_name = buffer_owner[name]
            base = buf_name.split(".")[-1]
            if any(kw in base.lower() for kw in _BUFFER_KEYWORDS):
                result.append(name)
    return tuple(result)


__all__ = [
    "CaptureManifest",
    "CapturedTensor",
    "InstrumentationBudget",
    "PairedActivationRecorder",
    "discover_buffer_backed_modules",
]
