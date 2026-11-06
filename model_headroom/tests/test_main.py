from __future__ import annotations

import subprocess
import sys


class TestMainModule:
    def test_help_exits_zero(self) -> None:
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys; sys.argv = ['model-headroom', '--help']; "
                "from model_headroom.cli import main; main()",
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert (
            "model-headroom" in result.stdout.lower()
            or "usage" in result.stdout.lower()
        )
