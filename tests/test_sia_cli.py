"""Tests for the sia dispatch layer (src/sia/cli.py).

Pins the observable contract of cli.main: exit codes for no args / --help /
unknown subcommand, help listing all six subcommands, RC propagation to the
lazily-imported sub-main, and lazy imports (no torch/safetensors pulled in by
--help).

These tests exercise source behaviour only; they do not import or execute any
sub-main. Source pinned: src/sia/cli.py.
"""

from __future__ import annotations

import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import patch

import sia.cli as cli_mod

_ALL_SUBCOMMANDS = (
    "capture-proxy",
    "model-headroom",
    "model-headroom-refusal",
    "model-headroom-paired",
    "doctor",
    "publish",
)


def test_sia_cli_no_args_exits_2():
    """main([]) renders help to stderr and returns 2 (src/sia/cli.py:88-90)."""
    assert cli_mod.main([]) == 2


def test_sia_cli_help_exits_0(capsys):
    """main(["--help"]) prints help to stdout and returns 0 (src/sia/cli.py:91-93)."""
    rc = cli_mod.main(["--help"])
    assert rc == 0
    assert "sia —" in capsys.readouterr().out


def test_sia_cli_unknown_subcommand_exits_2():
    """main(["bogus"]) reports unknown subcommand and returns 2 (src/sia/cli.py:98-101)."""
    assert cli_mod.main(["bogus"]) == 2


def test_sia_cli_help_lists_all_six_subcommands(capsys):
    """--help output lists every registered subcommand name (src/sia/cli.py:50-67)."""
    cli_mod.main(["--help"])
    out = capsys.readouterr().out
    for name in _ALL_SUBCOMMANDS:
        assert name in out, f"{name!r} missing from --help output"


def test_sia_cli_forwards_and_propagates_rc():
    """RC propagates from the lazily imported sub-main (src/sia/cli.py:103-107)."""
    sentinel = SimpleNamespace(main=lambda args=None: 7)
    with patch.object(cli_mod.importlib, "import_module", return_value=sentinel):
        assert cli_mod.main(["doctor"]) == 7


def test_sia_cli_lazy_import_keeps_help_light():
    """Running --help via sia.cli introduces no torch/safetensors imports
    (src/sia/cli.py:12-17; lazy-import contract).

    Uses a subprocess so torch/safetensors are NOT preloaded by other tests.
    """
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import sia.cli; sia.cli.main(['--help']); "
            "assert 'torch' not in sys.modules; "
            "assert 'safetensors' not in sys.modules",
        ],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
