from __future__ import annotations

import importlib.util

import pytest


@pytest.fixture(autouse=True)
def _isolate_capture_base(tmp_path_factory: pytest.TempPathFactory) -> None:
    """Point every test's capture root at a throwaway directory.

    ``CaptureProxyApp`` writes ``schema.json`` and the blob store under
    ``CAPTURE_BASE_PATH`` at construction time, defaulting to
    ``/root/captures``. On a machine where the proxy is actually running,
    a test that constructs the app without setting that variable writes
    into the operator's LIVE capture corpus. That is how a stray
    ``schema.json`` appeared in a running corpus during this work.

    Tests that care set their own ``CAPTURE_BASE_PATH`` via
    ``patch.dict``/``monkeypatch``, which overrides this. A test asserting
    the *default* must ``delenv`` it explicitly — relying on ambient env
    being unset was fragile regardless.
    """
    import os

    isolated = tmp_path_factory.mktemp("capture-base")
    previous = os.environ.get("CAPTURE_BASE_PATH")
    os.environ["CAPTURE_BASE_PATH"] = str(isolated)
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("CAPTURE_BASE_PATH", None)
        else:
            os.environ["CAPTURE_BASE_PATH"] = previous


def pytest_collection_modifyitems(config: pytest.Config) -> None:
    """Fail loudly when pytest-asyncio is absent.

    tests/test_upstream.py is the only file exercising the async UpstreamClient
    surface. Without pytest-asyncio every async test errors at collection and
    the rest of the suite reports green while upstream.py executes zero lines.
    Skip silently: run the suite with `uv run pytest`.
    """
    if importlib.util.find_spec("pytest_asyncio") is None:
        pytest.exit(
            "pytest-asyncio is not installed; tests/test_upstream.py would be "
            "collected as errors and upstream.py coverage would silently drop "
            "to zero. Use `uv run pytest`.",
            returncode=1,
        )
