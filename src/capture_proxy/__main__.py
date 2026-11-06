"""Entry point for capture-proxy — CLI, config file, foreground/daemon modes.

Runs uvicorn in the foreground, or manages a background instance via
--daemon/--stop/--status. Settings resolve: CLI flag > env var > config file
(~/.config/capture-proxy/config.toml) > default.

--ensure-config writes the default config when it is ABSENT (never overwrites),
so a fresh or restarted instance is never left on the :11434/ollama built-in
default. --print-effective-config prints the resolved settings and exits, so a
boot script can verify what was loaded instead of assuming it.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import tomllib
from pathlib import Path

import uvicorn

from capture_proxy.app import create_app

_DEFAULT_HOST = "0.0.0.0"  # noqa: S104
_DEFAULT_PORT = "8000"
_DEFAULT_UPSTREAM_URL = "http://127.0.0.1:11434"
_DEFAULT_BASE_PATH = "/root/captures"
_DEFAULT_BACKEND = "auto"

_CONFIG_KEYS = (
    "upstream_url",
    "port",
    "host",
    "base_path",
    "backend",
    "model_aliases",  # TOML table: {public_id = upstream_id} (proxy-restart-only)
    "model_tasks",  # TOML table: {public_id = task_type} (read per request)
    "headroom",
    "headroom_weights",
    "headroom_max_bytes",
    "headroom_interval",
    "headroom_dir",
)

#: Canonical default config written by ``--ensure-config``. Lives in CODE, not
#: as a packaged data file, so a wheel install can never ship without it (the
#: editable install used on Kaggle resolves the same constant either way).
#: ``config/capture-proxy.toml`` in the repo is the human-readable copy and a
#: test asserts the two are byte-identical — one source of truth, two readers.
#:
#: ``host = "127.0.0.1"`` is deliberate: this proxy is loopback-only by policy
#: (one port out, one port in — an extra externally-bound port is a banned
#: exposure). A box-local caller reads it on localhost; a remote caller runs
#: its OWN capture-proxy upstream at the public URL instead of widening this
#: listener. An operator who needs 0.0.0.0 edits the config after creation —
#: ``--ensure-config`` never clobbers an existing file, so that edit survives
#: every subsequent boot.
_DEFAULT_CONFIG_TOML = """\
# capture-proxy configuration
# Created by `capture-proxy --ensure-config`. Edit freely: the flag only
# writes this file when it is ABSENT, so local edits are never overwritten.
# Precedence: CLI flag > env var (CAPTURE_*) > this file > built-in default.

upstream_url = "http://127.0.0.1:8080"
port = 8000
host = "127.0.0.1"
base_path = "/root/captures"
backend = "llamacpp"   # auto | llamacpp | ollama | node-llama-cpp | vllm | sglang
"""


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments.

    Args:
        args: Argument list. ``None`` means ``sys.argv[1:]``.

    Returns:
        Parsed :class:`argparse.Namespace`.
    """
    parser = argparse.ArgumentParser(
        prog="capture-proxy",
        description="Capture proxy for llama.cpp/ollama/node-llama-cpp/vllm/sglang — logs raw JSONL of all requests/responses.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Proxy listen port (env: CAPTURE_PORT, default: 8000)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default=None,
        help="Bind address (env: CAPTURE_HOST, default: 0.0.0.0)",
    )
    parser.add_argument(
        "--upstream-url",
        type=str,
        default=None,
        help="Upstream URL (env: CAPTURE_UPSTREAM_URL, default: http://127.0.0.1:11434)",
    )
    parser.add_argument(
        "--base-path",
        type=str,
        default=None,
        help="JSONL storage root (env: CAPTURE_BASE_PATH, default: /root/captures)",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default=None,
        choices=["auto", "llamacpp", "ollama", "node-llama-cpp", "vllm", "sglang"],
        help="Backend type override (env: CAPTURE_BACKEND, default: auto-detect from port)",
    )
    parser.add_argument(
        "--model-aliases",
        type=str,
        default=None,
        help="Public-to-upstream model map as JSON, e.g. "
        '{"qwen3.8-27b-fp8":"davidau-qwen3.8-27b-twin-turbo-fable"} '
        "(env: CAPTURE_MODEL_ALIASES, TOML: model_aliases table, "
        "default: off; proxy-restart-only)",
    )
    parser.add_argument(
        "--model-tasks",
        type=str,
        default=None,
        help="Public-model-id to task-type map as JSON, e.g. "
        '{"nomic-embed-text":"embedding"} '
        "(env: CAPTURE_MODEL_TASKS, TOML: model_tasks table, "
        "default: off; read per request)",
    )
    parser.add_argument(
        "--headroom",
        type=str,
        default=None,
        help="Headroom capture: off | all | comma list of effective_rank,"
        "dead_neurons,load_balance,specialization,loss,isoflop "
        "(env: CAPTURE_HEADROOM, default: off)",
    )
    parser.add_argument(
        "--headroom-weights",
        type=str,
        default=None,
        choices=["off", "gate", "all"],
        help="Which MLP projections to snapshot "
        "(env: CAPTURE_HEADROOM_WEIGHTS, default: off)",
    )
    parser.add_argument(
        "--headroom-max-bytes",
        type=str,
        default=None,
        help="Float32 byte cap per dump, e.g. 256M "
        "(env: CAPTURE_HEADROOM_MAX_BYTES, default: 0 = unlimited)",
    )
    parser.add_argument(
        "--headroom-interval",
        type=int,
        default=None,
        help="Sample 1 of every N eligible captures "
        "(env: CAPTURE_HEADROOM_INTERVAL, default: 0 = every capture)",
    )
    parser.add_argument(
        "--headroom-dir",
        type=str,
        default=None,
        help="Headroom sidecar root (env: CAPTURE_HEADROOM_DIR, "
        "default: {base_path}/headroom)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Config file path (default: ~/.config/capture-proxy/config.toml)",
    )
    parser.add_argument(
        "--ensure-config",
        action="store_true",
        help=(
            "Create the default config file if it is absent (never overwrite an "
            "existing one); run before --daemon/--status to make a boot idempotent"
        ),
    )
    parser.add_argument(
        "--print-effective-config",
        action="store_true",
        help=(
            "Print the resolved settings (upstream_url, port, host, base_path, "
            "backend, headroom) after CLI > env > file > default resolution, then exit"
        ),
    )
    parser.add_argument(
        "--daemon",
        action="store_true",
        help="Run in background; write PID file and redirect logs",
    )
    parser.add_argument(
        "--stop",
        action="store_true",
        help="Stop the background instance",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Show background instance status",
    )
    parser.add_argument(
        "--pid-file",
        type=str,
        default=None,
        help="PID file path (default: ~/.cache/capture-proxy/proxy.pid)",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Log file path when daemonized (default: ~/.cache/capture-proxy/proxy.log)",
    )
    return parser.parse_args(args)


def _default_config_path() -> Path:
    """Return the default TOML config file path."""
    return Path.home() / ".config" / "capture-proxy" / "config.toml"


def _default_pid_path() -> Path:
    """Return the default PID file path for daemon instances."""
    return Path.home() / ".cache" / "capture-proxy" / "proxy.pid"


def _default_log_path() -> Path:
    """Return the default log file path for daemon instances."""
    return Path.home() / ".cache" / "capture-proxy" / "proxy.log"


def _load_config(path: Path) -> dict[str, object]:
    """Load TOML config, keeping only known keys.

    Missing file: silently ignored. Invalid TOML: warning to stderr, ignored.
    Never raises.
    """
    if not path.is_file():
        return {}
    try:
        with path.open("rb") as f:
            data = tomllib.load(f)
    except (tomllib.TOMLDecodeError, OSError) as err:
        print(f"warning: ignoring invalid config file {path}: {err}", file=sys.stderr)
        return {}
    return {key: value for key, value in data.items() if key in _CONFIG_KEYS}


def _ensure_config(path: Path) -> bool:
    """Write the default config file when it is absent; never overwrite.

    Purpose: make "a config exists on every boot" a property of the binary
      rather than of whatever provisioning happened to run. An ephemeral
      instance that boots without a config silently falls back to
      ``_DEFAULT_UPSTREAM_URL`` (:11434, ollama) and ``auto`` backend — wrong
      for this deployment, and invisible until traffic mysteriously 502s.

    Args:
        path: Destination config path (``~/.config/capture-proxy/config.toml``
          unless ``--config`` overrode it).

    Returns:
        bool: True when this call CREATED the file; False when it already
        existed and was left byte-for-byte untouched.

    Raises:
        OSError: Propagated when the file cannot be created (read-only home,
          permissions, full disk). Boot scripts want this to be loud: a config
          that cannot be written is a provisioning failure, not something to
          paper over with a default that points at the wrong upstream.

    Invariants:
        - Existing file is never read, rewritten, or truncated.
        - Parent directories are created as needed.
        - Written content is exactly ``_DEFAULT_CONFIG_TOML``.

    Post: ``path.is_file()`` holds, or it already did.
    """
    if path.exists():
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_DEFAULT_CONFIG_TOML, encoding="utf-8")
    return True


def _resolve_setting(
    cli: object | None, env_name: str, config: dict[str, object], default: str
) -> str:
    """Resolve a setting: CLI flag > env var > config file > default."""
    if cli is not None:
        return str(cli)
    env_value = os.environ.get(env_name)
    if env_value:
        return env_value
    config_key = env_name[len("CAPTURE_") :].lower()
    config_value = config.get(config_key)
    if config_value is not None:
        return str(config_value)
    return default


def _resolve_aliases_raw(
    cli: object | None, env_name: str, config: dict[str, object], default: str
) -> str | dict[str, object] | None:
    """Resolve the model-aliases setting preserving TOML table type.

    Precedence is CLI flag > env var > config file > default. Unlike
    :func:`_resolve_setting`, a TOML table (dict) is returned as-is
    instead of being stringified, so the caller can serialize it.

    Args:
        cli: CLI value (JSON string) or None when the flag was absent.
        env_name: Environment variable name (``CAPTURE_MODEL_ALIASES``).
        config: Filtered TOML config mapping.
        default: Fallback sentinel (``"off"``).

    Returns:
        JSON string, TOML table dict, or the default sentinel.

    Raises: None.
    """
    if cli is not None:
        return str(cli)
    env_value = os.environ.get(env_name)
    if env_value:
        return env_value
    config_key = env_name[len("CAPTURE_") :].lower()
    config_value = config.get(config_key)
    if config_value is not None:
        if isinstance(config_value, (dict, str)):
            return config_value
        return default
    return default


def _resolve_model_tasks_raw(
    cli: object | None, env_name: str, config: dict[str, object], default: str
) -> str | dict[str, object] | None:
    """Resolve the model-tasks setting preserving TOML table type.

    Shares :func:`_resolve_aliases_raw`'s precedence chain (CLI flag > env
    var > config file > default) and TOML-table handling, but is kept as a
    separate function so the two settings can diverge without coupling:
    aliases are built once at app startup, while model_tasks is re-read from
    the environment on every captured request by
    ``CaptureProxyApp._resolve_task``.

    Args:
        cli: CLI value (JSON string) or None when the flag was absent.
        env_name: Environment variable name (``CAPTURE_MODEL_TASKS``).
        config: Filtered TOML config mapping (already limited to
            ``_CONFIG_KEYS``, which includes ``model_tasks``).
        default: Fallback sentinel (``"off"``).

    Returns:
        JSON string, TOML table dict, or the default sentinel. A config value
            that is neither a dict nor a string yields the default (off)
            rather than being stringified.

    Raises: None.

    Invariants:
        - Pre: ``config`` contains only known keys.
        - Post: a returned dict is the TOML table verbatim; main() serializes
            it to JSON before injecting CAPTURE_MODEL_TASKS.
        - Side effects: none (reads the env and config mapping only).
    """
    if cli is not None:
        return str(cli)
    env_value = os.environ.get(env_name)
    if env_value:
        return env_value
    config_key = env_name[len("CAPTURE_") :].lower()
    config_value = config.get(config_key)
    if config_value is not None:
        if isinstance(config_value, (dict, str)):
            return config_value
        return default
    return default


def _write_pid_file(path: Path, pid: int) -> bool:
    """Write the PID file, creating parent directories as needed.

    Args:
        path: Destination path for the PID file.
        pid: Process id to record.

    Returns:
        bool: True when the file was written; False when an OSError
        (full disk, read-only tree, permissions) prevented it — the daemon
        still runs, the fault is reported on stderr.
    """
    try:
        os.makedirs(path.parent, exist_ok=True)
        path.write_text(f"{pid}\n")
    except OSError as e:
        print(f"warning: could not write PID file {path}: {e}", file=sys.stderr)
        return False
    return True


def _write_info_file(path: Path, info: dict[str, str]) -> bool:
    """Write the daemon info sidecar (resolved settings) next to the PID file.

    Args:
        path: Destination path for the info sidecar.
        info: Resolved settings to persist for ``--status``.

    Returns:
        bool: True when the file was written; False when an OSError
        (full disk, read-only tree, permissions) prevented it — the daemon
        still runs, the fault is reported on stderr.
    """
    try:
        os.makedirs(path.parent, exist_ok=True)
        path.write_text(json.dumps(info))
    except OSError as e:
        print(f"warning: could not write daemon info file {path}: {e}", file=sys.stderr)
        return False
    return True


def _read_info_file(path: Path) -> dict[str, object] | None:
    """Read the daemon info sidecar; None if missing or invalid JSON."""
    try:
        data = json.loads(path.read_text())
    except (FileNotFoundError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _read_pid(path: Path) -> int | None:
    """Read a PID from the file; ``None`` if missing or unparsable."""
    try:
        return int(path.read_text().strip())
    except (FileNotFoundError, ValueError):
        return None


def _pid_alive(pid: int) -> bool:
    """Return ``True`` if a process with ``pid`` exists."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # Process exists but is owned by another user — report it as alive
        return True
    return True


def _stop(pid_file: Path) -> int:
    """Send SIGTERM to the daemon PID and remove the PID file."""
    pid = _read_pid(pid_file)
    if pid is None:
        print("capture-proxy: not running")
        return 0
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        pass  # already gone — treat as stopped
    except PermissionError:
        print(
            f"capture-proxy: cannot stop pid {pid} (permission denied)",
            file=sys.stderr,
        )
        return 1
    pid_file.unlink(missing_ok=True)
    pid_file.with_name("proxy.info.json").unlink(missing_ok=True)
    print(f"capture-proxy: stopped (pid {pid})")
    return 0


def _status(pid_file: Path, config: dict[str, object]) -> int:
    """Print daemon status and return a process exit code.

    Prefers the daemon sidecar (the resolved settings the daemon was started
    with); falls back to this invocation's resolution chain when the sidecar
    is missing or its pid does not match.
    """
    pid = _read_pid(pid_file)
    if pid is None:
        print("capture-proxy: not running")
        return 1
    if not _pid_alive(pid):
        print(f"capture-proxy: stale pid file (pid {pid} not running)")
        return 1
    info = _read_info_file(pid_file.with_name("proxy.info.json"))
    if info is not None and str(info.get("pid")) == str(pid):
        port = str(info.get("port", ""))
        upstream = str(info.get("upstream_url", ""))
        backend = str(info.get("backend", ""))
        host = str(info.get("host", ""))
        base_path = str(info.get("base_path", ""))
        print(f"capture-proxy: running (pid {pid})")
        print(f"  port: {port}  upstream: {upstream}  backend: {backend}")
        print(f"  host: {host}  base_path: {base_path}")
        headroom = str(info.get("headroom", ""))
        if headroom:
            weights = str(info.get("headroom_weights", ""))
            headroom_dir = str(info.get("headroom_dir", ""))
            print(f"  headroom: {headroom}  weights: {weights}  dir: {headroom_dir}")
        return 0
    port = _resolve_setting(None, "CAPTURE_PORT", config, _DEFAULT_PORT)
    upstream = _resolve_setting(
        None, "CAPTURE_UPSTREAM_URL", config, _DEFAULT_UPSTREAM_URL
    )
    backend = _resolve_setting(None, "CAPTURE_BACKEND", config, _DEFAULT_BACKEND)
    print(f"capture-proxy: running (pid {pid})")
    print(f"  port: {port}  upstream: {upstream}  backend: {backend}")
    return 0


def _daemonize(
    args_raw: list[str] | None,
    pid_file: Path,
    log_file: Path,
    resolved: dict[str, str],
) -> int:
    """Spawn a detached background child and record its PID + settings.

    The child re-resolves config and env itself from the CLI args. ``resolved``
    holds the effective settings (port, host, upstream_url, base_path, backend)
    which are persisted to the info sidecar so ``--status`` can report what the
    daemon was actually started with. Returns the parent-side exit code.
    """
    existing = _read_pid(pid_file)
    if existing is not None and _pid_alive(existing):
        print(f"capture-proxy: already running (pid {existing})")
        return 1
    child_args = list(args_raw if args_raw is not None else sys.argv[1:])
    if "--daemon" in child_args:
        child_args.remove("--daemon")
    os.makedirs(log_file.parent, exist_ok=True)
    with open(log_file, "ab") as fh:
        child = subprocess.Popen(  # noqa: S603 - re-invokes this package, explicit args, no shell
            [sys.executable, "-m", "capture_proxy", *child_args],
            stdout=fh,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
            cwd=str(Path.cwd()),
        )
    _write_pid_file(pid_file, child.pid)
    _write_info_file(
        pid_file.with_name("proxy.info.json"),
        {"pid": str(child.pid), **resolved},
    )
    print(f"capture-proxy: started (pid {child.pid}, log: {log_file})")
    return 0


def main(args: list[str] | None = None) -> int:
    """Start the capture-proxy server, or manage the background instance.

    Args:
        args: CLI arguments. ``None`` means ``sys.argv[1:]``.

    Returns:
        Process exit code (0 = success). The foreground path blocks inside
        uvicorn and returns 0 on shutdown; ``--stop``/``--status`` return
        their own codes.
    """
    parsed = parse_args(args)
    pid_file = Path(parsed.pid_file) if parsed.pid_file else _default_pid_path()

    if parsed.stop:
        return _stop(pid_file)

    config_path = Path(parsed.config) if parsed.config else _default_config_path()

    # --ensure-config runs BEFORE the config is read, so it also works on a box
    # that has no config at all. Alone it provisions and exits; combined with an
    # action flag it provisions and falls through, so a boot is one command.
    if parsed.ensure_config:
        created = _ensure_config(config_path)
        print(
            f"capture-proxy: "
            f"{'created' if created else 'kept existing'} config {config_path}"
        )
        if not (parsed.status or parsed.daemon or parsed.print_effective_config):
            return 0

    config = _load_config(config_path)

    if parsed.status:
        return _status(pid_file, config)

    upstream_url = _resolve_setting(
        parsed.upstream_url, "CAPTURE_UPSTREAM_URL", config, _DEFAULT_UPSTREAM_URL
    )
    base_path = _resolve_setting(
        parsed.base_path, "CAPTURE_BASE_PATH", config, _DEFAULT_BASE_PATH
    )
    backend = _resolve_setting(
        parsed.backend, "CAPTURE_BACKEND", config, _DEFAULT_BACKEND
    )
    port = int(_resolve_setting(parsed.port, "CAPTURE_PORT", config, _DEFAULT_PORT))
    if not (1 <= port <= 65535):
        print(f"error: port {port} is out of range (1-65535)", file=sys.stderr)
        return 2
    host = _resolve_setting(parsed.host, "CAPTURE_HOST", config, _DEFAULT_HOST)
    headroom = _resolve_setting(parsed.headroom, "CAPTURE_HEADROOM", config, "off")
    headroom_weights = _resolve_setting(
        parsed.headroom_weights, "CAPTURE_HEADROOM_WEIGHTS", config, "off"
    )
    headroom_max_bytes = _resolve_setting(
        parsed.headroom_max_bytes, "CAPTURE_HEADROOM_MAX_BYTES", config, "0"
    )
    headroom_interval = _resolve_setting(
        parsed.headroom_interval, "CAPTURE_HEADROOM_INTERVAL", config, "0"
    )
    headroom_dir = _resolve_setting(
        parsed.headroom_dir, "CAPTURE_HEADROOM_DIR", config, ""
    )
    model_aliases_raw = _resolve_aliases_raw(
        parsed.model_aliases, "CAPTURE_MODEL_ALIASES", config, "off"
    )
    model_tasks_raw = _resolve_model_tasks_raw(
        parsed.model_tasks, "CAPTURE_MODEL_TASKS", config, "off"
    )

    if parsed.print_effective_config:
        # Read-back, not an inference: these are the values this process would
        # actually serve with. Boot verification greps this instead of
        # assuming the file it wrote is the file that got loaded.
        for key, value in (
            ("config_path", str(config_path)),
            ("config_present", str(config_path.is_file())),
            ("upstream_url", upstream_url),
            ("port", str(port)),
            ("host", host),
            ("base_path", base_path),
            ("backend", backend),
            ("headroom", headroom),
            ("headroom_weights", headroom_weights),
            ("headroom_dir", headroom_dir or f"{base_path}/headroom"),
        ):
            print(f"{key}={value}")
        return 0

    # Inject resolved values into env so create_app picks them up.
    os.environ["CAPTURE_UPSTREAM_URL"] = upstream_url
    os.environ["CAPTURE_BASE_PATH"] = base_path
    if backend != _DEFAULT_BACKEND:
        os.environ["CAPTURE_BACKEND"] = backend
    else:
        # "auto" means auto-detect; app.py treats any truthy CAPTURE_BACKEND
        # as explicit, so the sentinel must not be injected.
        os.environ.pop("CAPTURE_BACKEND", None)
    os.environ["CAPTURE_HEADROOM"] = headroom
    os.environ["CAPTURE_HEADROOM_WEIGHTS"] = headroom_weights
    os.environ["CAPTURE_HEADROOM_MAX_BYTES"] = headroom_max_bytes
    os.environ["CAPTURE_HEADROOM_INTERVAL"] = headroom_interval
    if headroom_dir:
        os.environ["CAPTURE_HEADROOM_DIR"] = headroom_dir
    else:
        # Empty means "derive from base_path"; the app applies that default.
        os.environ.pop("CAPTURE_HEADROOM_DIR", None)
    # Aliases are proxy-restart-only: serialize TOML tables to JSON so the
    # app (which reads the env string) builds one AliasMap at startup.
    if isinstance(model_aliases_raw, dict):
        os.environ["CAPTURE_MODEL_ALIASES"] = json.dumps(model_aliases_raw)
    elif isinstance(model_aliases_raw, str):
        if model_aliases_raw.strip() == "":
            os.environ.pop("CAPTURE_MODEL_ALIASES", None)
        else:
            os.environ["CAPTURE_MODEL_ALIASES"] = model_aliases_raw
    else:
        os.environ.pop("CAPTURE_MODEL_ALIASES", None)
    # Model tasks are re-read per request by CaptureProxyApp._resolve_task, so
    # the injected value is the JSON text the app parses each time. A TOML
    # table is serialized here; blank/None unsets the key so an inherited env
    # value from a parent shell does not leak into this run.
    if isinstance(model_tasks_raw, dict):
        os.environ["CAPTURE_MODEL_TASKS"] = json.dumps(model_tasks_raw)
    elif isinstance(model_tasks_raw, str):
        if model_tasks_raw.strip() == "":
            os.environ.pop("CAPTURE_MODEL_TASKS", None)
        else:
            os.environ["CAPTURE_MODEL_TASKS"] = model_tasks_raw
    else:
        os.environ.pop("CAPTURE_MODEL_TASKS", None)

    if parsed.daemon:
        log_file = Path(parsed.log_file) if parsed.log_file else _default_log_path()
        resolved = {
            "port": str(port),
            "host": host,
            "upstream_url": upstream_url,
            "base_path": base_path,
            "backend": backend,
            "headroom": headroom,
            "headroom_weights": headroom_weights,
            "headroom_max_bytes": headroom_max_bytes,
            "headroom_interval": str(headroom_interval),
            "headroom_dir": headroom_dir,
        }
        return _daemonize(args, pid_file, log_file, resolved)

    uvicorn.run(create_app(), host=host, port=port, log_level="info")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SystemExit as exc:
        sys.exit(exc.code)
