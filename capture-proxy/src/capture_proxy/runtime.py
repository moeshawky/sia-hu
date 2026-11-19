"""
Runtime metadata capture for benchmark provenance.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shlex
import subprocess
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, ClassVar

_logger = logging.getLogger(__name__)
_manifest_warned = False


def _manifest_write_failed(exc: OSError) -> None:
    """One-time named warning for a suppressed run-manifest write failure."""
    global _manifest_warned
    if _manifest_warned:
        return
    _manifest_warned = True
    _logger.warning(
        "run manifest write failed (file=%s, errno=%s); "
        "continuing without the manifest",
        getattr(exc, "filename", None) or "unknown",
        exc.errno,
    )


class RuntimeSnapshot:
    """
    Captures serving settings once per proxy process and writes a run manifest.

    Use ``RuntimeSnapshot.get(...)`` instead of direct construction: the
    snapshot is built once per (capture_base_path, backend) key and cached.
    Construction runs advisory subprocesses (pgrep, systemctl,
    llama-server/vllm --version, nvidia-smi); per-process caching keeps those
    out of the per-request path. ``force=True`` rebuilds the entry (used to
    re-capture metadata after an upstream restart).
    """

    _VLLM_BINARIES: ClassVar[frozenset[str]] = frozenset(
        {"python", "python3", "vllm", "serve", "api_server"}
    )

    _cache: ClassVar[dict[tuple[str, str], RuntimeSnapshot]] = {}
    _cache_lock: ClassVar[threading.Lock] = threading.Lock()

    @classmethod
    def get(
        cls,
        capture_base_path: str,
        backend: str = "llamacpp",
        *,
        force: bool = False,
    ) -> RuntimeSnapshot:
        """Return the cached per-process snapshot for (capture_base_path, backend).

        Builds and caches on first call; ``force=True`` rebuilds and replaces
        the cache entry. A lock serializes construction so concurrent requests
        never run the advisory subprocesses more than once per key.

        Args:
            capture_base_path: Root capture directory (cache key component).
            backend: Backend type string (cache key component).
            force: When True, rebuild the snapshot and update the cache.

        Returns:
            RuntimeSnapshot: cached (or freshly built) snapshot.

        Raises: None
        """
        key = (capture_base_path, backend)
        if not force:
            with cls._cache_lock:
                cached = cls._cache.get(key)
                if cached is not None:
                    return cached
        with cls._cache_lock:
            snapshot = cls(capture_base_path, backend)
            cls._cache[key] = snapshot
            return snapshot

    def __init__(self, capture_base_path: str, backend: str = "llamacpp") -> None:
        self._capture_base_path = Path(capture_base_path)
        self._backend = backend
        self.settings = self._build_settings()
        self.run_id = self._build_run_id(self.settings)
        self.settings["run_id"] = self.run_id
        self._write_manifest()

    def _build_settings(self) -> dict[str, Any]:
        if self._backend == "vllm":
            return self._build_vllm_settings()
        if self._backend == "sglang":
            return self._build_sglang_settings()
        service_env = self._private_model_environment()
        process_args = self._running_llama_args()
        model_path = process_args.get("model_path") or service_env.get("MODEL_PATH", "")
        cache_type_k = process_args.get("kv_cache_k") or service_env.get(
            "CACHE_TYPE_K", "q8_0"
        )
        cache_type_v = process_args.get("kv_cache_v") or service_env.get(
            "CACHE_TYPE_V", "q8_0"
        )

        return {
            "captured_at": datetime.now(UTC).isoformat(),
            "backend": self._backend,
            "served_model": "private-model",
            "model_path": model_path,
            "model_quant": self._parse_model_quant(model_path),
            "ctx_size": self._int_or_none(
                process_args.get("ctx_size") or service_env.get("CTX_SIZE")
            ),
            "parallel": self._int_or_none(
                process_args.get("parallel") or service_env.get("PARALLEL")
            ),
            "batch_size": self._int_or_none(
                process_args.get("batch_size") or service_env.get("BATCH_SIZE")
            ),
            "ubatch_size": self._int_or_none(
                process_args.get("ubatch_size") or service_env.get("UBATCH_SIZE")
            ),
            "gpu_layers": self._int_or_none(
                process_args.get("gpu_layers") or service_env.get("GPU_LAYERS")
            ),
            "kv_cache_k": cache_type_k,
            "kv_cache_v": cache_type_v,
            "cache_ram": self._int_or_none(
                process_args.get("cache_ram") or service_env.get("CACHE_RAM")
            ),
            "tools": process_args.get("tools") or service_env.get("TOOLS"),
            "host": process_args.get("host") or service_env.get("HOST"),
            "port": self._int_or_none(
                process_args.get("port") or service_env.get("PORT")
            ),
            "process_command": process_args.get("process_command", ""),
            "llama_version": self._command_output(["llama-server", "--version"]),
            "gpu": self._command_output(
                [
                    "nvidia-smi",
                    "--query-gpu=name,driver_version,memory.total",
                    "--format=csv,noheader",
                ]
            ),
        }

    def _build_vllm_settings(self) -> dict[str, Any]:
        """Build runtime settings for the vLLM backend.

        Advisory: probes the running vLLM process command line,
        ``vllm --version``, and nvidia-smi. All probes are best-effort —
        a missing process or command leaves fields empty without failing
        the snapshot.

        Args: None

        Returns:
            dict: Manifest fields. vLLM-specific provenance keys:
                max_model_len, tensor_parallel_size, pipeline_parallel_size,
                kv_cache_dtype, quantization, dtype, gpu_memory_utilization,
                vllm_version.

        Raises: None
        """
        process_args = self._running_vllm_args()
        model_path = process_args.get("model_path", "")
        return {
            "captured_at": datetime.now(UTC).isoformat(),
            "backend": self._backend,
            "served_model": process_args.get("served_model_name", "")
            or "private-model",
            "model_path": model_path,
            "model_quant": self._parse_model_quant(model_path),
            "max_model_len": self._int_or_none(process_args.get("max_model_len")),
            "tensor_parallel_size": self._int_or_none(
                process_args.get("tensor_parallel_size")
            ),
            "pipeline_parallel_size": self._int_or_none(
                process_args.get("pipeline_parallel_size")
            ),
            "kv_cache_dtype": process_args.get("kv_cache_dtype", ""),
            "quantization": process_args.get("quantization"),
            "dtype": process_args.get("dtype"),
            "gpu_memory_utilization": process_args.get("gpu_memory_utilization"),
            "process_command": process_args.get("process_command", ""),
            "vllm_version": self._vllm_version(process_args.get("process_command", "")),
            "gpu": self._command_output(
                [
                    "nvidia-smi",
                    "--query-gpu=name,driver_version,memory.total",
                    "--format=csv,noheader",
                ]
            ),
        }

    def _build_sglang_settings(self) -> dict[str, Any]:
        """Build runtime settings for the SGLang backend.

        Advisory: probes the running ``sglang serve`` process command line
        and nvidia-smi. All probes are best-effort — a missing process or
        command leaves fields empty without failing the snapshot.
        No version probe: the ``sglang --version`` flag is unverified
        against installed builds, and a probe that prints usage text
        would pollute the manifest, so the field is omitted by design.

        Args: None

        Returns:
            dict: Manifest fields. SGLang-specific provenance keys:
                served_model, tp_size, max_running_requests,
                reasoning_parser, tool_call_parser.

        Raises: None
        """
        process_args = self._running_sglang_args()
        model_path = process_args.get("model_path", "")
        served_model = (
            process_args.get("served_model_name", "")
            or Path(model_path).name
            or "private-model"
        )
        return {
            "captured_at": datetime.now(UTC).isoformat(),
            "backend": self._backend,
            "served_model": served_model,
            "model_path": model_path,
            "model_quant": self._parse_model_quant(model_path),
            "tp_size": self._int_or_none(process_args.get("tp_size")),
            "max_running_requests": self._int_or_none(
                process_args.get("max_running_requests")
            ),
            "reasoning_parser": process_args.get("reasoning_parser", ""),
            "tool_call_parser": process_args.get("tool_call_parser", ""),
            "host": process_args.get("host", ""),
            "port": self._int_or_none(process_args.get("port")),
            "process_command": process_args.get("process_command", ""),
            "gpu": self._command_output(
                [
                    "nvidia-smi",
                    "--query-gpu=name,driver_version,memory.total",
                    "--format=csv,noheader",
                ]
            ),
        }

    @classmethod
    def _running_sglang_args(cls) -> dict[str, str]:
        """Parse the running SGLang process command line.

        SGLang runs as a console script (``sglang serve --model-path ...``)
        or a module invocation (``python -m sglang.launch_server``), so the
        full command line is matched (``pgrep -af sglang``) — name-only
        matching misses both shapes.

        Args: None

        Returns:
            dict: ``process_command`` (always when a process matched) plus
                parsed option values (model_path, served_model_name,
                tp_size, max_running_requests, reasoning_parser,
                tool_call_parser, host, port). Empty dict when no SGLang
                process is running.

        Raises: None
        """
        output = RuntimeSnapshot._command_output(["pgrep", "-af", "sglang"])
        if not output:
            return {}
        lines = output.splitlines()
        # pgrep -af matches any command line mentioning "sglang" (smallest
        # PID first) — filter to server-shaped lines so line selection
        # matches the SGLang server, not an unrelated process (fallback:
        # first line).
        server_lines = [line for line in lines if cls._is_sglang_server_line(line)]
        first_line = (server_lines or lines)[0]
        _, _, command = first_line.partition(" ")
        if not command:
            return {}

        try:
            argv = shlex.split(command)
        except ValueError:
            return {"process_command": command}

        option_map = {
            "--model-path": "model_path",
            "--served-model-name": "served_model_name",
            "--tp-size": "tp_size",
            "--max-running-requests": "max_running_requests",
            "--reasoning-parser": "reasoning_parser",
            "--tool-call-parser": "tool_call_parser",
            "--host": "host",
            "--port": "port",
        }
        parsed: dict[str, str] = {"process_command": command}
        for index, arg in enumerate(argv):
            key = option_map.get(arg)
            if key and index + 1 < len(argv):
                parsed[key] = argv[index + 1]
        return parsed

    @staticmethod
    def _is_sglang_server_line(line: str) -> bool:
        """Report whether a pgrep -af line is SGLang server-shaped.

        Server-shaped: a path component exactly "sglang" (the console
        script, e.g. /opt/venv/bin/sglang) or an "sglang.launch_server" /
        "sglang.serve" module invocation.

        Args:
            line: One "PID command ..." line from pgrep -af sglang.

        Returns:
            bool: True when the command line looks like the SGLang server.

        Raises: None
        """
        for token in line.split():
            if "sglang.launch_server" in token or "sglang.serve" in token:
                return True
            if any(part == "sglang" for part in token.split("/")):
                return True
        return False

    @staticmethod
    def _running_llama_args() -> dict[str, str]:
        output = RuntimeSnapshot._command_output(["pgrep", "-a", "llama-server"])
        if not output:
            # Advisory metadata — empty triggers env-var fallback in _build_settings
            return {}
        first_line = output.splitlines()[0]
        _, _, command = first_line.partition(" ")
        if not command:
            return {}

        try:
            argv = shlex.split(command)
        except ValueError:
            return {"process_command": command}

        option_map = {
            "-m": "model_path",
            "--model": "model_path",
            "--ctx-size": "ctx_size",
            "--parallel": "parallel",
            "--batch-size": "batch_size",
            "--ubatch-size": "ubatch_size",
            "-ngl": "gpu_layers",
            "--n-gpu-layers": "gpu_layers",
            "--cache-type-k": "kv_cache_k",
            "--cache-type-v": "kv_cache_v",
            "--cache-ram": "cache_ram",
            "--tools": "tools",
            "--host": "host",
            "--port": "port",
        }
        parsed = {"process_command": command}
        for index, arg in enumerate(argv):
            key = option_map.get(arg)
            if key and index + 1 < len(argv):
                parsed[key] = argv[index + 1]
        return parsed

    @classmethod
    def _running_vllm_args(cls) -> dict[str, str]:
        """Parse the running vLLM process command line.

        vLLM runs as a Python process, so the full command line is matched
        (``pgrep -af vllm``) — name-only matching misses
        ``python -m vllm.entrypoints...`` invocations.

        Args: None

        Returns:
            dict: ``process_command`` (always when a process matched) plus
                parsed option values (max_model_len, tensor_parallel_size,
                pipeline_parallel_size, kv_cache_dtype, quantization, dtype,
                gpu_memory_utilization) and ``model_path`` from the
                positional model argument. Empty dict when no vLLM process
                is running.

        Raises: None
        """
        output = RuntimeSnapshot._command_output(["pgrep", "-af", "vllm"])
        if not output:
            return {}
        lines = output.splitlines()
        # pgrep -af matches any command line mentioning "vllm" (smallest PID
        # first) — filter to server-shaped lines so line selection matches
        # the vLLM server, not an unrelated process (fallback: first line).
        server_lines = [line for line in lines if cls._is_vllm_server_line(line)]
        first_line = (server_lines or lines)[0]
        _, _, command = first_line.partition(" ")
        if not command:
            return {}

        try:
            argv = shlex.split(command)
        except ValueError:
            return {"process_command": command}

        option_map = {
            "--max-model-len": "max_model_len",
            "--max-model-length": "max_model_len",
            "--served-model-name": "served_model_name",
            "--tensor-parallel-size": "tensor_parallel_size",
            "--pipeline-parallel-size": "pipeline_parallel_size",
            "--kv-cache-dtype": "kv_cache_dtype",
            "--quantization": "quantization",
            "--dtype": "dtype",
            "--gpu-memory-utilization": "gpu_memory_utilization",
        }
        parsed: dict[str, str] = {"process_command": command}
        for index, arg in enumerate(argv):
            key = option_map.get(arg)
            if key and index + 1 < len(argv):
                parsed[key] = argv[index + 1]
        parsed["model_path"] = cls._vllm_model_positional(argv)
        return parsed

    @staticmethod
    def _is_vllm_server_line(line: str) -> bool:
        """Report whether a pgrep -af line is vLLM server-shaped.

        Server-shaped: a path component exactly "vllm" (the executable,
        e.g. /venv/bin/vllm) or a "-m vllm.entrypoints" module invocation.

        Args:
            line: One "PID command ..." line from pgrep -af vllm.

        Returns:
            bool: True when the command line looks like the vLLM server.

        Raises: None
        """
        for token in line.split():
            if "vllm.entrypoints" in token:
                return True
            if any(part == "vllm" for part in token.split("/")):
                return True
        return False

    @classmethod
    def _vllm_model_positional(cls, argv: list[str]) -> str:
        """Extract the positional model argument from a vLLM command line.

        vLLM takes the model positionally (``vllm serve /path --flag v``).
        Skips the interpreter, the vllm entrypoint, ``serve``, and values of
        preceding flags; the first remaining token is the model path.

        Args:
            argv: Tokenized vLLM process command line.

        Returns:
            str: Candidate model path, or "" when no candidate exists.

        Raises: None
        """
        for index, arg in enumerate(argv):
            if arg.startswith("-"):
                continue
            if index > 0 and argv[index - 1].startswith("-"):
                continue  # value of the preceding flag
            if (
                arg in cls._VLLM_BINARIES
                or "vllm" in arg
                or arg.startswith("python")
                or os.path.basename(arg).startswith("python")
            ):
                continue
            return arg
        return ""

    def _write_manifest(self) -> None:
        try:
            runs_dir = self._capture_base_path / "runs"
            runs_dir.mkdir(parents=True, exist_ok=True)
            manifest_path = runs_dir / f"{self.run_id}.json"
            manifest_path.write_text(
                json.dumps(self.settings, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        except OSError as e:
            # Advisory metadata write — a capture-tree fault must not fail
            # every request (get() contract: Raises: None)
            _manifest_write_failed(e)

    @staticmethod
    def _private_model_environment() -> dict[str, str]:
        output = RuntimeSnapshot._command_output(
            [
                "systemctl",
                "show",
                "private-model.service",
                "-p",
                "Environment",
                "--value",
            ]
        )
        if not output:
            # Advisory metadata — empty triggers env-var fallback in _build_settings
            return {}
        env: dict[str, str] = {}
        for item in shlex.split(output):
            if "=" in item:
                key, value = item.split("=", 1)
                env[key] = value
        return env

    @staticmethod
    def _parse_model_quant(model_path: str) -> str | None:
        name = Path(model_path).name
        patterns = [
            r"(Q\d+_[A-Z0-9]+)",
            r"(Q\d+_K_[A-Z])",
            r"(Q\d+_K)",
            r"(Q\d+)",
            r"(F\d+)",
        ]
        for pattern in patterns:
            match = re.search(pattern, name, re.IGNORECASE)
            if match:
                return match.group(1)
        return None

    @staticmethod
    def _int_or_none(value: str | None) -> int | None:
        if value is None or value == "":
            return None
        try:
            return int(value)
        except ValueError:
            return None

    @classmethod
    def _vllm_version(cls, process_command: str) -> str:
        """Resolve the vLLM version, PATH first, venv-local second.

        ``vllm --version`` fails when the serve process runs from a venv
        that is not on PATH (the Kaggle/common case). A venv's bin/ dir
        holds the interpreter and the vllm script side by side, so the
        interpreter path from the process command line resolves it.

        Args:
            process_command: Full command line of the vLLM process, or "".

        Returns:
            str: Version string, or "" when unresolvable.

        Raises: None
        """
        # Slow on TPU stacks (jax/TPU init runs even for --version, >10s) —
        # the generic 5s timeout would silently fail the probe. TPU stacks
        # also prefix --version output with init logs; the version itself is
        # the last non-empty line (plain installs print only that line).
        version = cls._command_output(["vllm", "--version"], timeout=30)
        if version:
            return cls._last_nonempty_line(version)
        try:
            argv = shlex.split(process_command)
        except ValueError:
            return ""
        if not argv or argv[0].startswith("-"):
            return ""
        candidate = os.path.join(os.path.dirname(argv[0]), "vllm")
        if os.path.isfile(candidate):
            return cls._last_nonempty_line(
                cls._command_output([candidate, "--version"], timeout=30)
            )
        return ""

    @staticmethod
    def _last_nonempty_line(output: str) -> str:
        """Return the last non-empty line of subprocess output, stripped."""
        for line in reversed(output.splitlines()):
            if line.strip():
                return line.strip()
        return ""

    @staticmethod
    def _command_output(command: list[str], timeout: float = 5.0) -> str:
        """Run an advisory command and return its stripped stdout.

        Returns the command's stripped stdout, or an empty string when the
        command cannot be executed or fails: the subprocess is advisory
        (version probes for run manifests), never critical. Subprocess errors,
        a missing executable (FileNotFoundError), and an unexecutable one
        (PermissionError) all degrade to "".
        """
        try:
            # Advisory subprocess on a fixed internal command list — no shell,
            # no user input; output is discarded on error
            return subprocess.check_output(  # noqa: S603
                command,
                text=True,
                stderr=subprocess.STDOUT,
                timeout=timeout,
            ).strip()
        except (subprocess.SubprocessError, FileNotFoundError, PermissionError):
            # Error output discarded — subprocess is advisory, not critical
            return ""

    @staticmethod
    def _build_run_id(settings: dict[str, Any]) -> str:
        if settings.get("backend") == "vllm":
            # Distinct identity shape so vLLM configs hash independently and
            # existing llama.cpp run_ids stay stable.
            identity = {
                "backend": settings.get("backend"),
                "model_path": settings.get("model_path"),
                "max_model_len": settings.get("max_model_len"),
                "tensor_parallel_size": settings.get("tensor_parallel_size"),
                "kv_cache_dtype": settings.get("kv_cache_dtype"),
                "quantization": settings.get("quantization"),
            }
        elif settings.get("backend") == "sglang":
            # Distinct identity shape so SGLang configs hash independently
            # and existing vLLM/llama.cpp run_ids stay stable.
            identity = {
                "backend": settings.get("backend"),
                "model_path": settings.get("model_path"),
                "tp_size": settings.get("tp_size"),
                "max_running_requests": settings.get("max_running_requests"),
                "reasoning_parser": settings.get("reasoning_parser"),
                "tool_call_parser": settings.get("tool_call_parser"),
            }
        else:
            identity = {
                "backend": settings.get("backend"),
                "model_path": settings.get("model_path"),
                "ctx_size": settings.get("ctx_size"),
                "parallel": settings.get("parallel"),
                "kv_cache_k": settings.get("kv_cache_k"),
                "kv_cache_v": settings.get("kv_cache_v"),
                "cache_ram": settings.get("cache_ram"),
                "tools": settings.get("tools"),
            }
        digest = hashlib.sha256(
            json.dumps(identity, sort_keys=True).encode()
        ).hexdigest()
        return f"run_{digest[:16]}"
