"""Shared device resolution for the authority. Accelerator-preferred with CPU
fallback: auto resolves to the first available accelerator (cuda, then any
other accelerator torch exposes, such as mps or xpu), else cpu. An explicit
accelerator that is unavailable is an error, never a silent downgrade."""

from __future__ import annotations

import torch

_ACCELERATOR_CANDIDATES: tuple[tuple[str, bool], ...] = (
    ("cuda", torch.cuda.is_available()),
    ("xpu", getattr(torch, "xpu", None) is not None and torch.xpu.is_available()),
    ("mps", torch.backends.mps.is_available()),
)

_ALL_DEVICES = ("cpu", *(name for name, _ in _ACCELERATOR_CANDIDATES))


def available_accelerators() -> list[str]:
    """List accelerator names torch currently exposes as available, in
    preference order."""
    return [name for name, ok in _ACCELERATOR_CANDIDATES if ok]


def default_device() -> str:
    """Resolve the accelerator-preferred default device: the first available
    accelerator, else cpu."""
    for name, ok in _ACCELERATOR_CANDIDATES:
        if ok:
            return name
    return "cpu"


def resolve_device(requested: str) -> str | None:
    """Resolve a requested device to a usable torch device string.

    "auto" -> the first available accelerator, else cpu
    "cpu" -> cpu
    an explicit accelerator -> that accelerator when available, else None
    Returns None only when a requested accelerator is unavailable.
    """
    if requested == "auto":
        return default_device()
    if requested == "cpu":
        return "cpu"
    for name, ok in _ACCELERATOR_CANDIDATES:
        if requested == name:
            return name if ok else None
    return None
