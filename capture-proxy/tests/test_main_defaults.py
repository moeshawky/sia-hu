from __future__ import annotations

from unittest.mock import patch

from capture_proxy.__main__ import main


class TestMainBindDefaults:
    """W5: bind default — operator decision is "make it work with both".

    Without a config file, the built-in default binds all interfaces
    (0.0.0.0). With a generated config, the loopback-only default
    (127.0.0.1) from the config file wins.

    Every main() call is wrapped in ``patch.dict("os.environ", {})`` because
    main() writes resolved settings directly into os.environ (see
    src/capture_proxy/__main__.py:875-913). Without the wrapper, leaked keys
    survive into downstream tests via patch.dict without clear=True.

    Pins: src/capture_proxy/__main__.py:28 (_DEFAULT_HOST = "0.0.0.0"),
    :72-80 (_DEFAULT_CONFIG_TOML with host = "127.0.0.1"),
    :404-417 (_resolve_setting: CLI > env > config > default),
    :834 (host resolution), :1077 (uvicorn.run call site).
    """

    def test_effective_bind_default_without_config_is_all_interfaces(
        self, monkeypatch, tmp_path
    ) -> None:
        captured = {}

        def fake_run(app, host=None, port=None, log_level=None):
            captured["host"] = host

        monkeypatch.setattr("capture_proxy.__main__.uvicorn.run", fake_run)
        monkeypatch.setattr(
            "capture_proxy.__main__._default_config_path",
            lambda: tmp_path / "no-such-config.toml",
        )
        monkeypatch.delenv("CAPTURE_HOST", raising=False)

        with patch.dict("os.environ", {}):
            main([])
        assert captured["host"] == "0.0.0.0"

    def test_effective_bind_uses_loopback_from_generated_config(
        self, monkeypatch, tmp_path
    ) -> None:
        captured = {}

        def fake_run(app, host=None, port=None, log_level=None):
            captured["host"] = host

        monkeypatch.setattr("capture_proxy.__main__.uvicorn.run", fake_run)

        config_path = tmp_path / "config.toml"

        with patch.dict("os.environ", {}):
            main(["--ensure-config", "--config", str(config_path)])
            main(["--config", str(config_path)])
        assert captured["host"] == "127.0.0.1"
