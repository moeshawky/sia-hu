"""
Engine probe shim — workspace-level, importing neither product package.

Best-effort, observation-only discovery of installed and running inference
engines plus hardware limits. Never raises, never starts or stops a server,
never guesses a backend from a port number alone.

Usage:
    python tools/engine_probe.py            # JSON manifest to stdout, exit 0
    python tools/engine_probe.py --compact  # human-readable summary
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from typing import Any

# Short timeout so each localhost probe is bounded; total stay well under a few
# seconds even when nothing is listening.
_HEALTH_TIMEOUT = 2.0

# Separate, larger bound for the heavy vllm framework import (measured ~15s on
# this box). Kept distinct from _HEALTH_TIMEOUT so the localhost endpoint probes
# stay short and bounded; never changed together with _HEALTH_TIMEOUT.
_VLLM_VERSION_TIMEOUT = 30.0

# Candidate localhost endpoints to probe for running servers. Port alone never
# identifies an engine; the server's own reported metadata does.
_CANDIDATE_ENDPOINTS: tuple[str, ...] = (
    "http://127.0.0.1:11434",  # ollama default
    "http://127.0.0.1:8000",  # vLLM default — ambiguous (also proxy default)
    "http://127.0.0.1:8001",  # vLLM alternate — ambiguous
    "http://127.0.0.1:8080",  # llama.cpp default
    "http://127.0.0.1:30000",  # sglang default
)

_ENGINE_CAPABILITIES: dict[str, dict[str, bool]] = {
    "llamacpp": {
        "chat_completions": True,
        "embeddings": True,
        "rerank": True,
        "routed_experts": False,
        "hidden_states": False,
    },
    "vllm": {
        "chat_completions": True,
        "embeddings": True,
        "rerank": True,  # vLLM >= 0.7 exposes /v1/rerank
        "routed_experts": False,
        "hidden_states": False,
    },
    "ollama": {
        "chat_completions": True,
        "embeddings": True,
        "rerank": False,  # ollama returns 501 on /v1/rerank
        "routed_experts": False,
        "hidden_states": False,
    },
    "sglang": {
        "chat_completions": True,
        "embeddings": True,
        "rerank": True,
        "routed_experts": True,
        "hidden_states": True,  # requires --enable-return-hidden-states
    },
}


def _iso_now() -> str:
    """Current UTC time as ISO8601."""
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _run(command: list[str], timeout: float = _HEALTH_TIMEOUT) -> tuple[int, str]:
    """Run a command, returning (exit_code, stripped_output).

    Never raises: missing binaries, timeouts, non-zero exits, and decode
    errors all degrade to reported facts. stderr is merged into stdout because
    some tools (e.g. llama-server --version) emit their version on stderr.
    """
    try:
        # Advisory subprocess on a fixed internal command list — no shell, no
        # user input; a missing binary or timeout degrades to a reported fact
        result = subprocess.run(  # noqa: S603
            command,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return result.returncode, (result.stdout + result.stderr).strip()
    except (subprocess.SubprocessError, FileNotFoundError, PermissionError):
        return -1, ""


def _last_nonempty_line(output: str) -> str:
    """Return the last non-empty line of subprocess output, stripped."""
    for line in reversed(output.splitlines()):
        if line.strip():
            return line.strip()
    return ""


def _version_line(output: str) -> str:
    """Return the most version-like line of tool output.

    llama-server prints "version: ..." on stderr and a build line last; the
    first line mentioning "version" or "commit" carries the identity.
    Falls back to the last non-empty line.
    """
    for line in output.splitlines():
        low = line.lower()
        if "version" in low or "commit" in low or "__version__" in low:
            return line.strip()
    return _last_nonempty_line(output)


def _sm_form(compute_capability: str | None) -> str | None:
    """Normalize nvidia-smi "7.5" -> "sm_75", or None when absent.

    Purely derived from the measured capability string; never guesses from a
    device name.
    """
    if compute_capability is None:
        return None
    try:
        major, minor = str(compute_capability).split(".")
        return f"sm_{major}{minor}"
    except (ValueError, AttributeError):
        return None


def _http_get_json(url: str) -> Any:
    """GET a URL and return parsed JSON, or None on any failure.

    Socket-based (no httpx dependency), short timeout, observation only.
    """
    try:
        import urllib.request

        with urllib.request.urlopen(  # noqa: S310
            url, timeout=_HEALTH_TIMEOUT
        ) as response:
            body = response.read().decode("utf-8", errors="replace")
        return json.loads(body)
    except Exception:
        return None


def _which(binary: str) -> str | None:
    """Return the resolved path of a binary on PATH, or None."""
    return shutil.which(binary)


def _probe_version(binary: str, args: list[str]) -> str | None:
    """Best-effort version probe: run [binary, *args], return version line."""
    code, output = _run([binary, *args])
    if code == 0 and output:
        return _version_line(output)
    return None


def _discover_llama() -> dict[str, Any] | None:
    """Discover llama-server: PATH binary, --version, --list-devices GPU lines."""
    binary = _which("llama-server")
    if binary is None:
        return None
    version = _probe_version(binary, ["--version"])
    devices_output = ""
    code, output = _run([binary, "--list-devices"], timeout=10.0)
    if code == 0:
        devices_output = output
    return {
        "engine": "llamacpp",
        "binary": binary,
        "version": version,
        "running": False,
        "endpoints": [],
        "capabilities": _ENGINE_CAPABILITIES["llamacpp"],
        "notes": [
            "llama-server on PATH; --list-devices GPU lines reported below",
        ]
        + (
            [f"list-devices: {devices_output.splitlines()[0]}"]
            if devices_output
            else []
        ),
        "_raw_devices": devices_output,
    }


def _dist_info_version(venv_python: str) -> str | None:
    """Read vllm's installed distribution version from dist-info METADATA.

    No framework import — just reads the venv's site-packages metadata, so it is
    fast and never triggers vllm's heavy import chain. Falls back to None when
    the metadata is not discoverable.
    """
    try:
        venv_root = os.path.dirname(os.path.dirname(venv_python))
        pattern = os.path.join(
            venv_root, "lib", "python*", "site-packages", "vllm-*.dist-info", "METADATA"
        )
        matches = glob.glob(pattern)
        if not matches:
            return None
        with open(matches[0], encoding="utf-8", errors="replace") as handle:
            for line in handle:
                if line.lower().startswith("version:"):
                    return line.split(":", 1)[1].strip()
    except Exception:
        return None
    return None


def _discover_vllm() -> dict[str, Any] | None:
    """Discover vLLM: probe the venv at /root/.venvs/vllm/bin/python.

    A missing venv is a reported fact, not an error.
    """
    venv_python = "/root/.venvs/vllm/bin/python"
    if not shutil.which("python3") and not _file_exists(venv_python):
        return None
    if not _file_exists(venv_python):
        return {
            "engine": "vllm",
            "binary": None,
            "version": None,
            "running": False,
            "endpoints": [],
            "capabilities": _ENGINE_CAPABILITIES["vllm"],
            "notes": ["vllm venv at /root/.venvs/vllm/bin/python not present"],
        }
    # Lighter-weight first: read the installed distribution version from the
    # venv's dist-info metadata without importing vllm.
    version = _dist_info_version(venv_python)
    source = "dist-info metadata"
    if version is None:
        # Fallback: heavy framework import, bounded by its own timeout so the
        # short _HEALTH_TIMEOUT (endpoint probes) is never affected.
        code, output = _run(
            [venv_python, "-c", "import vllm; print(vllm.__version__)"],
            timeout=_VLLM_VERSION_TIMEOUT,
        )
        if code == 0 and output:
            version = output.strip()
            source = "vllm import"
    notes = [f"vllm probed via venv interpreter {venv_python}"]
    if version:
        notes.append(f"version from {source}")
    else:
        notes.append("vllm version unavailable (dist-info and import both failed)")
    return {
        "engine": "vllm",
        "binary": venv_python,
        "version": version,
        "running": False,
        "endpoints": [],
        "capabilities": _ENGINE_CAPABILITIES["vllm"],
        "notes": notes,
    }


def _file_exists(path: str) -> bool:
    try:
        return __import__("os").path.exists(path)
    except Exception:
        return False


def _discover_ollama() -> dict[str, Any] | None:
    """Discover ollama: PATH binary + version."""
    binary = _which("ollama")
    if binary is None:
        return None
    version = _probe_version(binary, ["--version"])
    return {
        "engine": "ollama",
        "binary": binary,
        "version": version,
        "running": False,
        "endpoints": [],
        "capabilities": _ENGINE_CAPABILITIES["ollama"],
        "notes": ["ollama on PATH"],
    }


def _discover_sglang() -> dict[str, Any] | None:
    """Discover sglang: PATH binary + version."""
    binary = _which("sglang")
    if binary is None:
        return None
    version = _probe_version(binary, ["--version"])
    return {
        "engine": "sglang",
        "binary": binary,
        "version": version,
        "running": False,
        "endpoints": [],
        "capabilities": _ENGINE_CAPABILITIES["sglang"],
        "notes": ["sglang on PATH"],
    }


def _classify_engine(models_payload: Any, fingerprint: str | None) -> str | None:
    """Identify an engine from its own /v1/models metadata, never the port.

    llama.cpp: model path ends in .gguf OR system_fingerprint contains a build
    id (e.g. b11395-7f2dd88b0).
    vLLM: owned_by == "vllm".
    ollama: owned_by == "ollama" (or generic /api/tags shape).
    sglang: owned_by == "sglang" or model path ends in .safetensors / .bin.
    Returns None when ambiguous.
    """
    if not isinstance(models_payload, dict):
        return None
    data = models_payload.get("data") or models_payload.get("models")
    if not isinstance(data, list):
        return None
    for entry in data:
        if not isinstance(entry, dict):
            continue
        model_id = str(entry.get("id") or entry.get("model") or "")
        owned_by = str(entry.get("owned_by") or "")
        details = entry.get("details") or {}
        fmt = str(details.get("format") or "")
        family = str(details.get("family") or "")
        meta = entry.get("meta") or {}
        if isinstance(meta, dict):
            fmt = fmt or str(meta.get("format") or "")
        if model_id.endswith(".gguf") or fmt == "gguf":
            return "llamacpp"
        if fingerprint and ("b11395" in fingerprint or "7f2dd88b0" in fingerprint):
            return "llamacpp"
        if owned_by == "vllm" or "vllm" in str(entry.get("object") or "").lower():
            return "vllm"
        if owned_by == "ollama":
            return "ollama"
        if owned_by == "sglang" or family == "sglang":
            return "sglang"
        if model_id.endswith((".safetensors", ".bin", ".pt")):
            return "sglang"
    return None


def _probe_endpoint(base_url: str) -> dict[str, Any] | None:
    """Probe one candidate endpoint for /health and /v1/models.

    Returns an endpoint record with engine identified from server metadata,
    or None when nothing responds.
    """
    health = _http_get_json(base_url + "/health")
    if health is None:
        return None
    models = _http_get_json(base_url + "/v1/models")
    fingerprint = None
    if isinstance(models, dict):
        fingerprint = models.get("system_fingerprint")
    engine = _classify_engine(models, fingerprint)
    healthy = True
    models_list: list[str] = []
    if isinstance(models, dict):
        data = models.get("data") or models.get("models")
        if isinstance(data, list):
            for entry in data:
                if isinstance(entry, dict):
                    mid = entry.get("id") or entry.get("model")
                    if mid:
                        models_list.append(str(mid))
                elif isinstance(entry, str):
                    models_list.append(entry)
    notes: list[str] = []
    if engine is None:
        notes.append("ambiguous: engine identified by port alone is unreliable")
    return {
        "url": base_url,
        "healthy": healthy,
        "models": models_list,
        "engine": engine,
        "fingerprint": fingerprint,
        "_notes": notes,
    }


def _hardware() -> dict[str, Any]:
    """Detect hardware: GPU devices via llama-server --list-devices / nvidia-smi.

    Best-effort; empty lists when no GPU tooling is available.
    """
    devices: list[dict[str, Any]] = []
    notes: list[str] = []

    llama = _discover_llama()
    if llama is not None and llama.get("_raw_devices"):
        for line in llama["_raw_devices"].splitlines():
            line = line.strip()
            if not line or line.startswith("Available devices:"):
                continue
            # "CUDA0: Tesla T4 (14911 MiB, 11296 MiB free)"
            name = None
            total_mib = None
            if ":" in line:
                _, rest = line.split(":", 1)
                paren = rest.find("(")
                if paren != -1:
                    inside = rest[paren + 1 : rest.find(")", paren)]
                    for token in inside.split(","):
                        token = token.strip()
                        if token.endswith(" MiB") and total_mib is None:
                            try:
                                total_mib = int(token[:-4].strip())
                            except ValueError:
                                total_mib = None
                    name = rest[:paren].strip()
            if name:
                devices.append(
                    {
                        "name": name,
                        "total_memory_mib": total_mib,
                        "compute_capability": None,
                        "compute_capability_sm": None,
                    }
                )

    # nvidia-smi is the measured source for compute_capability; run it whenever a
    # GPU tool is available so the capability is never left hardcoded None. Parse
    # all nvidia-smi devices, then merge into devices already named by
    # llama-server by index order (both list GPUs in the same order), adding any
    # nvidia-smi-only devices. On an older driver without compute_cap the field
    # degrades to measured-absent (None) — never an exception, never a guess.
    nvidia_devices: list[dict[str, Any]] = []
    code, output = _run(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,compute_cap",
            "--format=csv,noheader",
        ],
        timeout=5.0,
    )
    if code == 0 and output:
        for line in output.splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 2 and parts[0]:
                total_mib = None
                if parts[1].endswith(" MiB"):
                    try:
                        total_mib = int(parts[1][:-4].strip())
                    except ValueError:
                        total_mib = None
                compute_capability = None
                if len(parts) >= 3 and parts[2]:
                    compute_capability = parts[2]
                nvidia_devices.append(
                    {
                        "name": parts[0],
                        "total_memory_mib": total_mib,
                        "compute_capability": compute_capability,
                        "compute_capability_sm": _sm_form(compute_capability),
                    }
                )

    for i, nd in enumerate(nvidia_devices):
        if i < len(devices):
            d = devices[i]
            d["compute_capability"] = nd["compute_capability"]
            d["compute_capability_sm"] = nd["compute_capability_sm"]
            if d["total_memory_mib"] is None:
                d["total_memory_mib"] = nd["total_memory_mib"]
        else:
            devices.append(nd)

    # sm_75 present per the MEASURED compute capability (not the device name):
    # fp8 requires sm_89+, so fp8 checkpoints dequantize to bf16.
    sm75_present = any(d.get("compute_capability_sm") == "sm_75" for d in devices)
    if sm75_present:
        notes.append(
            "T4 present (sm_75): fp8 requires sm_89+, so fp8 checkpoints "
            "dequantize to bf16 and never run fp8 kernels"
        )
    for d in devices:
        if d["total_memory_mib"] is not None:
            notes.append(f"{d['name']}: {d['total_memory_mib']} MiB VRAM")

    return {
        "devices": devices,
        "notes": notes,
    }


def _limits(devices: list[dict[str, Any]]) -> list[str]:
    """Hardware/engine constraint strings the operator must know."""
    limits: list[str] = []
    t4_present = any("T4" in d["name"] for d in devices)
    vram_values = [d["total_memory_mib"] for d in devices if d["total_memory_mib"]]
    if t4_present:
        limits.append(
            "T4 is sm_75: fp8 requires sm_89+, fp8 checkpoints dequantize "
            "to bf16 and never execute fp8 kernels"
        )
    if vram_values:
        min_vram = min(vram_values)
        limits.append(
            f"min device VRAM {min_vram} MiB: the 18GB mimo-v2.6-distill-qwen-9b "
            "checkpoint does NOT fit in 15GB VRAM — CPU-RAM loading plus GPU "
            "compute is the correct architecture"
        )
    if not devices:
        limits.append("no GPU detected: measurement falls back to CPU")
    return limits


def probe_engines() -> dict[str, Any]:
    """Probe installed and running engines; return a JSON-serializable manifest.

    Never raises. Every probe is best-effort; an unexpected error yields a valid
    manifest with an ``errors`` list.
    """
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "generated_at": _iso_now(),
        "hardware": {"devices": [], "notes": []},
        "engines": {"installed": []},
        "limits": [],
    }
    errors: list[str] = []

    try:
        hardware = _hardware()
        manifest["hardware"] = hardware
    except Exception as exc:  # pragma: no cover — defensive outer wrapper
        errors.append(f"hardware probe failed: {exc!r}")
        hardware = {"devices": [], "notes": []}

    discovered: list[dict[str, Any]] = []

    for discover in (
        _discover_llama,
        _discover_vllm,
        _discover_ollama,
        _discover_sglang,
    ):
        try:
            entry = discover()
            if entry is not None:
                discovered.append(entry)
        except Exception as exc:  # pragma: no cover — defensive outer wrapper
            errors.append(f"{discover.__name__} failed: {exc!r}")

    # Probe running servers: identify by server metadata, never port alone.
    running_by_engine: dict[str, list[dict[str, Any]]] = {}
    for base_url in _CANDIDATE_ENDPOINTS:
        try:
            endpoint = _probe_endpoint(base_url)
        except Exception as exc:  # pragma: no cover — defensive outer wrapper
            errors.append(f"endpoint probe {base_url} failed: {exc!r}")
            continue
        if endpoint is None:
            continue
        engine = endpoint.get("engine")
        endpoint_record = {
            "url": endpoint["url"],
            "healthy": endpoint["healthy"],
            "models": endpoint["models"],
        }
        if engine is None:
            endpoint_record["notes"] = endpoint.get("_notes", [])
        if engine:
            running_by_engine.setdefault(engine, []).append(endpoint_record)

    # Merge running info into installed entries.
    for entry in discovered:
        engine = entry["engine"]
        endpoints = running_by_engine.get(engine, [])
        entry["running"] = bool(endpoints)
        entry["endpoints"] = endpoints
        if endpoints and engine is None:
            entry["notes"] = [*list(entry.get("notes", [])), "ambiguous engine"]
        # Drop the internal raw devices field from the public manifest.
        entry.pop("_raw_devices", None)
        entry.pop("_notes", None)
        manifest["engines"]["installed"].append(entry)

    # Merge any endpoints whose engine could not be identified (ambiguous).
    for engine, endpoints in running_by_engine.items():
        if engine is None:
            for endpoint in endpoints:
                manifest["engines"]["installed"].append(
                    {
                        "engine": None,
                        "binary": None,
                        "version": None,
                        "running": True,
                        "endpoints": [endpoint],
                        "capabilities": {},
                        "notes": ["ambiguous: identified by port only, engine unknown"],
                    }
                )

    manifest["limits"] = _limits(hardware.get("devices", []))
    if errors:
        manifest["errors"] = errors
    return manifest


def compact_summary(manifest: dict[str, Any]) -> str:
    """Human-readable summary of the manifest."""
    lines: list[str] = []
    lines.append(
        f"engine_probe schema v{manifest.get('schema_version')} "
        f"generated {manifest.get('generated_at')}"
    )
    hardware = manifest.get("hardware", {})
    devices = hardware.get("devices", [])
    if devices:
        lines.append("hardware:")
        for d in devices:
            cc = d.get("compute_capability")
            lines.append(
                f"  - {d.get('name')} {d.get('total_memory_mib')} MiB "
                f"(compute_capability={cc})"
            )
    else:
        lines.append("hardware: no devices detected")
    for note in hardware.get("notes", []):
        lines.append(f"  note: {note}")
    installed = manifest.get("engines", {}).get("installed", [])
    if installed:
        lines.append("engines:")
        for entry in installed:
            engine = entry.get("engine") or "unknown"
            running = entry.get("running", False)
            version = entry.get("version")
            lines.append(
                f"  - {engine} version={version} running={running} "
                f"binary={entry.get('binary')}"
            )
            for ep in entry.get("endpoints", []):
                lines.append(
                    f"      endpoint {ep.get('url')} healthy={ep.get('healthy')} "
                    f"models={len(ep.get('models', []))}"
                )
            for note in entry.get("notes", []):
                lines.append(f"      note: {note}")
    else:
        lines.append("engines: none installed")
    limits = manifest.get("limits", [])
    if limits:
        lines.append("limits:")
        for limit in limits:
            lines.append(f"  - {limit}")
    if manifest.get("errors"):
        lines.append("errors:")
        for err in manifest["errors"]:
            lines.append(f"  - {err}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: print manifest JSON (or --compact summary), exit 0."""
    parser = argparse.ArgumentParser(
        description="Probe installed and running inference engines."
    )
    parser.add_argument(
        "--compact",
        action="store_true",
        help="print a human-readable summary instead of JSON",
    )
    args = parser.parse_args(argv)
    manifest = probe_engines()
    if args.compact:
        print(compact_summary(manifest))
    else:
        print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
