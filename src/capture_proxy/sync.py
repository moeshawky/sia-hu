"""Native off-box capture sync via rclone subprocess.

Pure-stdlib module: no uvicorn/fastapi/httpx/torch imports. All rclone/subprocess
logic lives here. ``sync.py`` does not import ``__main__.py`` or ``app.py`` — no
circular dependency, and the proxy-server path never loads this module.

The sync daemon spawn mechanism (design §4): the parent acquires
``fcntl.flock(LOCK_EX | LOCK_NB)`` on the PID file before check-and-spawn, holds
it across the whole operation, then ``subprocess.Popen([sys.executable, '-m',
'capture_proxy', '--sync-daemon', <each resolved setting as its flag>],
start_new_session=True, stdout=log, stderr=log)`` and writes the child PID to
``sync.pid``. The child sees ``CAPTURE_SYNC_DAEMONIZED=1``, skips the spawn,
and runs the loop every ``--sync-interval`` seconds forever, installing a
SIGTERM handler that removes ``sync.pid`` and exits 0.
"""

from __future__ import annotations

import contextlib
import fcntl
import os
import shlex
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

_RCLONE_MISSING_TEMPLATE = (
    "error: rclone binary {binary} not found on PATH; install rclone "
    "(https://rclone.org/install/) or set CAPTURE_SYNC_RCLONE_BINARY to a valid "
    "path — valid example: CAPTURE_SYNC_RCLONE_BINARY=/usr/local/bin/rclone "
    "capture-proxy --sync"
)

_SYNC_CMD_INVALID_TEMPLATE = (
    "error: --sync-cmd must be 'copy' or 'sync' (got: {value}) — valid "
    "example: --sync-cmd copy"
)

_INTERVAL_INVALID_TEMPLATE = (
    "error: --sync-interval must be a positive whole number of seconds "
    "(got: {value}) — valid example: --sync-interval 300"
)

_NO_INSTANCE_ID_TEMPLATE = (
    "error: no instance id — --sync-machine-id, CAPTURE_MACHINE_ID, config "
    "machine_id, hostname, and /etc/machine-id are all empty; set one — valid "
    "example: CAPTURE_MACHINE_ID=prod-box-1 capture-proxy --sync"
)

_SYNC_CMD_SYNC_DERIVED_TEMPLATE = (
    "error: --sync-cmd sync requires an explicit instance id you own alone; "
    "derived id {id} from hostname or /etc/machine-id is not safe for a "
    "destructive mirror — set CAPTURE_MACHINE_ID or pass --sync-machine-id — "
    "valid example: CAPTURE_MACHINE_ID=my-dev-box capture-proxy --sync-cmd sync"
)

_SOURCE_CREATE_FAIL_TEMPLATE = (
    "error: cannot create source directory {path}: {os_error}; check "
    "permissions or set CAPTURE_BASE_PATH to a writable dir — valid example: "
    "CAPTURE_BASE_PATH=/writable/captures capture-proxy --sync"
)

_RCLONE_FAILURE_TEMPLATE = (
    "error: rclone {cmd} {source} {dest} exited {rc}; full argv: {argv} — "
    "check upstream_status in errors/ and network to {remote}"
)

_ALREADY_RUNNING_TEMPLATE = (
    "capture-proxy: sync daemon already running (pid {pid}) — stop with: "
    "capture-proxy --sync-stop"
)

_STALE_PID_TEMPLATE = (
    "capture-proxy: stale sync PID file (pid {pid} not running) — starting a new loop"
)

_MACHINE_ID_FALLBACK_TEMPLATE = (
    "WARNING: falling back to /etc/machine-id ({id}) — image-baked "
    "machine-ids are shared across instances; set CAPTURE_MACHINE_ID or "
    "--sync-machine-id for a unique key; never use --sync-cmd sync with a "
    "derived id"
)

_DAEMON_STARTED_TEMPLATE = "capture-proxy: sync daemon started (pid {pid})"

_DAEMON_STOPPED_TEMPLATE = "capture-proxy: sync daemon stopped (pid {pid})"

_DAEMON_NOT_RUNNING_TEMPLATE = "capture-proxy: sync daemon not running"

_DAEMON_STOPPED_ALREADY_TEMPLATE = "capture-proxy: sync daemon already stopped"

_SYNC_RESULT_TEMPLATE = (
    "capture-proxy: sync {cmd} {source} -> {dest} exited {rc} ({elapsed:.1f}s)"
)


def _is_explicit_true(value: object) -> bool:
    """Return True when *value* is a truthy boolean or a truthy boolean string."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "on"}
    return False


def _read_pid(path: Path) -> int | None:
    """Read a PID from the file; None if missing or unparsable."""
    try:
        return int(path.read_text().strip())
    except (FileNotFoundError, ValueError, OSError):
        return None


def _pid_alive(pid: int) -> bool:
    """Return True if a process with *pid* exists."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def resolve_instance_id(
    machine_id_cli: str | None,
    machine_id_env: str | None,
    machine_id_config: str | None,
    hostname: str | None,
    machine_id_file: str | None,
) -> tuple[str, bool]:
    """Return (instance_id, explicit).

    Chain: flag > CAPTURE_MACHINE_ID env > config machine_id > derived
    (hostname > /etc/machine-id). ``explicit=True`` only when flag or
    CAPTURE_MACHINE_ID env was set.
    """
    if machine_id_cli:
        return machine_id_cli, True
    if machine_id_env:
        return machine_id_env, True
    if machine_id_config:
        return machine_id_config, False
    if hostname:
        return hostname, False
    if machine_id_file:
        return machine_id_file, False
    return "", False


def validate_sync_cmd(value: str) -> None:
    """Raise ValueError with a literal string if value is not 'copy' or 'sync'."""
    if value not in ("copy", "sync"):
        raise ValueError(_SYNC_CMD_INVALID_TEMPLATE.format(value=value))


def validate_interval(value: str) -> int:
    """Parse whole-number seconds; raise ValueError with literal string if invalid."""
    try:
        parsed = int(value)
    except ValueError as err:
        raise ValueError(_INTERVAL_INVALID_TEMPLATE.format(value=value)) from err
    if parsed < 1:
        raise ValueError(_INTERVAL_INVALID_TEMPLATE.format(value=value))
    return parsed


def resolve_rclone_binary(value: str) -> str:
    """Resolve the rclone binary.

    If *value`` contains '/', require ``Path(value).is_file()``; otherwise use
    ``shutil.which(value)``. Raise ValueError with a literal string naming the
    resolved binary and ``--sync-rclone-binary`` when not found.
    """
    if "/" in value:
        if Path(value).is_file():
            return value
    else:
        found = shutil.which(value)
        if found:
            return found
    raise ValueError(_RCLONE_MISSING_TEMPLATE.format(binary=value))


def build_rclone_argv(settings: dict[str, object], source: str, dest: str) -> list[str]:
    """Build the exact rclone argv: [binary, cmd, source, dest, ...age_opts,
    ...shlex.split(opts), ...dry_flag].

    The binary is resolved here via :func:`resolve_rclone_binary` so argv[0]
    is always the resolved path (never a bare ``rclone`` that PATH may resolve
    differently in the child).
    """
    binary = resolve_rclone_binary(str(settings.get("sync_rclone_binary", "rclone")))
    cmd = str(settings.get("sync_cmd", "copy"))
    argv: list[str] = [binary, cmd, source, dest]

    min_age = str(settings.get("sync_min_age", "60s"))
    if min_age != "0":
        argv.extend(["--min-age", min_age])

    opts = str(settings.get("sync_rclone_opts", ""))
    if opts:
        argv.extend(shlex.split(opts))

    if _is_explicit_true(settings.get("sync_dry_run", False)):
        argv.append("--dry-run")

    return argv


def _source_dir(source: str) -> None:
    """Create the source directory if missing; warn if empty."""
    path = Path(source)
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError as err:
        raise OSError(
            _SOURCE_CREATE_FAIL_TEMPLATE.format(path=source, os_error=err)
        ) from err
    if path.is_dir() and not any(path.iterdir()):
        print(f"warning: source directory {source} is empty", file=sys.stderr)


def run_sync_once(settings: dict[str, object]) -> int:
    """Create source dir if missing (warn if empty), run rclone once, log
    result, return exit code."""
    source = str(settings.get("sync_base_path", "/root/captures"))
    remote = str(settings.get("sync_remote", "gdrive:captures"))
    instance_id = str(settings.get("sync_instance_id", ""))
    explicit = bool(settings.get("sync_instance_id_explicit", False))
    cmd = str(settings.get("sync_cmd", "copy"))

    if cmd == "sync" and not explicit:
        print(
            _SYNC_CMD_SYNC_DERIVED_TEMPLATE.format(id=instance_id),
            file=sys.stderr,
        )
        return 1

    dest = f"{remote}/{instance_id}"
    try:
        _source_dir(source)
        argv = build_rclone_argv(settings, source, dest)
    except (ValueError, OSError) as err:
        print(str(err), file=sys.stderr)
        return 1

    start = time.monotonic()
    try:
        proc = subprocess.run(  # noqa: S603 - explicit argv, no shell
            argv,
            capture_output=True,
            text=True,
        )
        rc = proc.returncode
    except OSError as err:
        rc = 1
        elapsed = time.monotonic() - start
        print(
            _RCLONE_FAILURE_TEMPLATE.format(
                cmd=cmd,
                source=source,
                dest=dest,
                rc=rc,
                argv=" ".join(argv),
                remote=remote,
            ),
            file=sys.stderr,
        )
        print(f"rclone subprocess OSError: {err}", file=sys.stderr)
        elapsed = time.monotonic() - start
        print(
            _SYNC_RESULT_TEMPLATE.format(
                cmd=cmd, source=source, dest=dest, rc=rc, elapsed=elapsed
            ),
            file=sys.stderr,
        )
        return 1

    elapsed = time.monotonic() - start
    if rc != 0:
        print(
            _RCLONE_FAILURE_TEMPLATE.format(
                cmd=cmd,
                source=source,
                dest=dest,
                rc=rc,
                argv=" ".join(argv),
                remote=remote,
            ),
            file=sys.stderr,
        )
        if proc.stderr:
            print(proc.stderr, file=sys.stderr)
    print(
        _SYNC_RESULT_TEMPLATE.format(
            cmd=cmd, source=source, dest=dest, rc=rc, elapsed=elapsed
        )
    )
    return rc


def sync_daemon_loop(settings: dict[str, object]) -> int:
    """Idempotent start: acquire exclusive lock ``fcntl.flock(LOCK_EX | LOCK_NB)``
    on the PID file and hand the held lock to the child via ``pass_fds`` so the
    lock stays held for the child's lifetime (no spawn/write window). Returns
    the parent-side exit code."""
    pid_file = Path(str(settings.get("sync_pid_file", "")))
    log_file = Path(str(settings.get("sync_log_file", "")))

    existing = _read_pid(pid_file)
    if existing is not None and _pid_alive(existing):
        print(_ALREADY_RUNNING_TEMPLATE.format(pid=existing))
        return 0

    lock_fd: int | None = None
    try:
        pid_file.parent.mkdir(parents=True, exist_ok=True)
        lock_fd = os.open(pid_file, os.O_RDWR | os.O_CREAT, 0o644)
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            # Another parent holds the lock — a loop is being spawned right now.
            # Poll for a live pid for up to 2.0 s at 50 ms intervals; the holder
            # truncates then writes the pid file, so the pid may be absent at first.
            deadline = time.monotonic() + 2.0
            while time.monotonic() < deadline:
                existing_after = _read_pid(pid_file)
                if existing_after is not None and _pid_alive(existing_after):
                    print(_ALREADY_RUNNING_TEMPLATE.format(pid=existing_after))
                    return 0
                time.sleep(0.05)
            print(
                "error: another sync daemon start is in progress; "
                "retry in a moment — run: capture-proxy --sync-status",
                file=sys.stderr,
            )
            return 1

        # Re-check under the lock: stale PID -> overwrite; live -> no-op.
        existing = _read_pid(pid_file)
        if existing is not None and _pid_alive(existing):
            print(_ALREADY_RUNNING_TEMPLATE.format(pid=existing))
            return 0

        os.makedirs(log_file.parent, exist_ok=True)
        log_fh = open(log_file, "ab")  # noqa: SIM115 — passed to Popen; closed after spawn
        child_args = _build_child_args(settings)
        env = dict(os.environ)
        env["CAPTURE_SYNC_DAEMONIZED"] = "1"
        env["CAPTURE_SYNC_LOCK_FD"] = str(lock_fd)
        child = subprocess.Popen(  # noqa: S603 - re-invokes this package, explicit args, no shell
            child_args,
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
            cwd=str(Path.cwd()),
            env=env,
            pass_fds=(lock_fd,),
        )
        log_fh.close()
        os.ftruncate(lock_fd, 0)
        os.write(lock_fd, f"{child.pid}\n".encode())
        os.fsync(lock_fd)
        # Parent closes its fd; the child inherits the same open file
        # description so the lock stays held for the child's lifetime.
        with contextlib.suppress(OSError):
            os.close(lock_fd)
        print(_DAEMON_STARTED_TEMPLATE.format(pid=child.pid))
        return 0
    except OSError as err:
        print(f"error: cannot start sync daemon: {err}", file=sys.stderr)
        return 1
    finally:
        if lock_fd is not None:
            with contextlib.suppress(OSError):
                os.close(lock_fd)


def _build_child_args(settings: dict[str, object]) -> list[str]:
    """Build the child argv inheriting every resolved setting as its flag."""
    args: list[str] = [
        sys.executable,
        "-m",
        "capture_proxy",
        "--sync-daemon",
        "--sync-remote",
        str(settings.get("sync_remote", "gdrive:captures")),
        "--sync-cmd",
        str(settings.get("sync_cmd", "copy")),
        "--sync-min-age",
        str(settings.get("sync_min_age", "60s")),
        "--sync-interval",
        str(settings.get("sync_interval", "300")),
        "--sync-rclone-binary",
        str(settings.get("sync_rclone_binary", "rclone")),
        "--sync-rclone-opts",
        str(settings.get("sync_rclone_opts", "")),
        "--sync-pid-file",
        str(settings.get("sync_pid_file", "")),
        "--sync-log-file",
        str(settings.get("sync_log_file", "")),
        "--base-path",
        str(settings.get("sync_base_path", "/root/captures")),
    ]
    # D1: only pass --sync-machine-id when the id was explicitly provided
    # (flag or CAPTURE_MACHINE_ID env). When derived (hostname / /etc/machine-id)
    # the flag is omitted so the child re-derives the same value and keeps
    # explicit=False — a derived id must never become explicit in the child,
    # or destructive --sync-cmd sync becomes legal on a shared hostname id.
    if settings.get("sync_instance_id_explicit"):
        args.extend(["--sync-machine-id", str(settings.get("sync_instance_id", ""))])
    config = settings.get("sync_config_path")
    if config:
        args.extend(["--config", str(config)])
    return args


def _run_loop(settings: dict[str, object]) -> int:
    """Run the sync loop forever (child entry point). Installs a SIGTERM
    handler that removes sync.pid and exits 0."""
    pid_file = Path(str(settings.get("sync_pid_file", "")))
    interval = int(str(settings.get("sync_interval", 300)))

    def _handle_sigterm(signum: int, frame: object) -> None:
        with contextlib.suppress(OSError):
            pid_file.unlink(missing_ok=True)
        sys.exit(0)

    signal.signal(signal.SIGTERM, _handle_sigterm)

    # Hold the flock for the child's lifetime. When spawned by
    # sync_daemon_loop the parent already holds the lock on this same open
    # file description (passed via CAPTURE_SYNC_LOCK_FD / pass_fds); keep it
    # and do not re-flock. Otherwise (manual child invocation) acquire it.
    lock_fd_env = os.environ.get("CAPTURE_SYNC_LOCK_FD")
    lock_fd: int | None = None
    if lock_fd_env is not None:
        try:
            lock_fd = int(lock_fd_env)
            os.fstat(lock_fd)  # raises OSError if fd is not valid/open
        except (ValueError, OSError):
            lock_fd = None
    if lock_fd is None:
        try:
            pid_file.parent.mkdir(parents=True, exist_ok=True)
            lock_fd = os.open(pid_file, os.O_RDWR | os.O_CREAT, 0o644)
            fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            os.write(lock_fd, f"{os.getpid()}\n".encode())
            os.fsync(lock_fd)
        except BlockingIOError:
            print(
                "error: another sync daemon holds the lock; stop it with: "
                "capture-proxy --sync-stop",
                file=sys.stderr,
            )
            return 1
        except OSError as err:
            print(f"error: cannot acquire sync lock: {err}", file=sys.stderr)
            return 1

    while True:
        source = str(settings.get("sync_base_path", "/root/captures"))
        if Path(source).exists():
            run_sync_once(settings)
        else:
            print(
                f"warning: source directory {source} disappeared mid-loop; "
                "recreating on next cycle",
                file=sys.stderr,
            )
        time.sleep(interval)
    return 0  # pragma: no cover — unreachable, loop runs forever


def sync_status(pid_file: Path) -> int:
    """Print sync daemon status; return 0 running, 1 not running/stale."""
    pid = _read_pid(pid_file)
    if pid is None:
        print(_DAEMON_NOT_RUNNING_TEMPLATE)
        return 1
    if not _pid_alive(pid):
        print(f"capture-proxy: stale sync pid file (pid {pid} not running)")
        return 1
    print(f"capture-proxy: sync daemon running (pid {pid})")
    print(f"  pid file: {pid_file}")
    return 0


def sync_stop(pid_file: Path) -> int:
    """SIGTERM the sync daemon PID, remove sync.pid; return 0 stopped/already
    stopped, 1 on failure."""
    pid = _read_pid(pid_file)
    if pid is None:
        print(_DAEMON_STOPPED_ALREADY_TEMPLATE)
        return 0
    if not _pid_alive(pid):
        with contextlib.suppress(OSError):
            pid_file.unlink(missing_ok=True)
        print(_DAEMON_STOPPED_ALREADY_TEMPLATE)
        return 0
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        with contextlib.suppress(OSError):
            pass
    except PermissionError:
        print(
            f"capture-proxy: cannot stop sync pid {pid} (permission denied)",
            file=sys.stderr,
        )
        return 1
    pid_file.unlink(missing_ok=True)
    print(_DAEMON_STOPPED_TEMPLATE.format(pid=pid))
    return 0
