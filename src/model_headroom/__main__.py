"""Allow running model_headroom as a module: python -m model_headroom."""

from __future__ import annotations

import sys

from model_headroom.cli import main

sys.exit(main())
