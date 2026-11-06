"""
Regression guard: a test run must never touch the live capture corpus.

CaptureProxyApp writes schema.json and the blob store under
CAPTURE_BASE_PATH at construction time, and that variable defaults to the
real /root/captures. Several tests previously blanked CAPTURE_BASE_PATH to
let a config file win; an empty value is treated as unset by config
resolution, so those tests resolved back to the operator's live corpus and
wrote into it. The conftest autouse fixture isolates the variable, and this
file asserts the isolation still holds.

Kept separate from the capture-contract suite so the hygiene fix can be
reviewed and reverted independently of the feature it protects.
"""

from __future__ import annotations

import glob
import os
from unittest.mock import patch

from capture_proxy.app import CaptureProxyApp


class TestTestIsolation:
    """Guards the guarantee that a test run cannot touch a live corpus."""

    def test_capture_base_is_isolated_from_the_real_default(self) -> None:
        base = os.environ.get("CAPTURE_BASE_PATH")
        assert base, "CAPTURE_BASE_PATH must be set during tests"
        assert base != "/root/captures", (
            "tests would write into the live capture corpus"
        )

    def test_an_app_built_here_writes_under_the_isolated_root(self) -> None:
        before = set(glob.glob(f"{os.environ['CAPTURE_BASE_PATH']}/**", recursive=True))
        with patch.dict(
            "os.environ",
            {
                "CAPTURE_UPSTREAM_URL": "http://127.0.0.1:8001",
                "CAPTURE_BACKEND": "llamacpp",
            },
        ):
            CaptureProxyApp()
        after = set(glob.glob(f"{os.environ['CAPTURE_BASE_PATH']}/**", recursive=True))
        assert after - before, "app construction wrote nothing at all"
