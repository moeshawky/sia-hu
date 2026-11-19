"""Serving-engine adapter clients for behavioral probes against SGLang and vLLM.

These are injectable inference clients (like :class:`MockInferenceClient`) that
submit prompts to a running SGLang or vLLM OpenAI-compatible endpoint and return
the output. They preserve exact input/output pairs associated with experimental
conditions. They expose behavior only — never internal gradients or activations.

No server is launched here. The adapter assumes an operator has already started
a server (per ``docs/INFERENCE-OPERATIONS.md``) and supplies its base URL.

Both engines expose OpenAI-compatible ``/v1/chat/completions`` and
``/v1/completions``. SGLang additionally exposes ``/generate``. The adapter
normalizes all three into a single ``submit(prompt, **kwargs) -> str | None``
contract.

T4 constraints (from INFERENCE-OPERATIONS.md): T4 (SM75) has no native
bfloat16; servers must use ``--dtype float16``. Tensor-parallel-size 2 spans
both T4s and would conflict with GPU-0 usage by the operator; adapters do not
launch servers, so they carry no parallelism defaults.
"""

from __future__ import annotations

from typing import Any

import httpx

from model_headroom.acquire.behavioral import InferenceClient


class ServingEngineClient(InferenceClient):
    """HTTP client for a running SGLang or vLLM inference server.

    Submits prompts to an OpenAI-compatible endpoint and returns the generated
    text. On failure returns ``None`` with the error recorded in metadata —
    never fabricates an output.

    Attributes:
        base_url: Server base URL, e.g. ``"http://127.0.0.1:8000/v1"``.
        engine: ``"sglang"`` or ``"vllm"`` — informs endpoint selection and
            error interpretation but does not change the submit contract.
        timeout: Per-request timeout in seconds.
        model: Optional model name to send in the request body. When omitted,
            the server's default served model is used.
    """

    def __init__(
        self,
        base_url: str,
        *,
        engine: str = "vllm",
        timeout: float = 30.0,
        model: str | None = None,
    ) -> None:
        if engine not in ("sglang", "vllm"):
            raise ValueError(f"engine must be 'sglang' or 'vllm', got {engine!r}")
        self.base_url = base_url.rstrip("/")
        self.engine = engine
        self.timeout = timeout
        self.model = model
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def submit(self, prompt: str, **kwargs: Any) -> str | None:
        """Submit one prompt to the serving engine.

        Returns the generated text, or ``None`` on failure (HTTP error,
        connection refused, malformed response). The failure is recorded in the
        call metadata via the ``error`` kwarg convention used by
        :func:`run_behavioral_probe`.
        """
        self.calls.append((prompt, dict(kwargs)))
        try:
            return self._submit_once(prompt, **kwargs)
        except Exception as exc:  # adapter boundary captures all failures
            kwargs.setdefault("error", f"{type(exc).__name__}: {exc}")
            return None

    def _submit_once(self, prompt: str, **kwargs: Any) -> str | None:
        if self.engine == "sglang":
            return self._submit_sglang(prompt, **kwargs)
        return self._submit_vllm(prompt, **kwargs)

    def _submit_vllm(self, prompt: str, **kwargs: Any) -> str | None:
        url = f"{self.base_url}/completions"
        payload: dict[str, Any] = {
            "model": self.model or "",
            "prompt": prompt,
            "temperature": 0.0,
            "max_tokens": 1,
        }
        payload.update(
            {
                k: v
                for k, v in kwargs.items()
                if k not in ("error", "condition", "repeat")
            }
        )
        with httpx.Client(timeout=self.timeout) as client:
            response = client.post(url, json=payload)
            response.raise_for_status()
            data = response.json()
        choices = data.get("choices", [])
        if not choices:
            return None
        text = choices[0].get("text")
        if text is None:
            message = choices[0].get("message", {})
            text = message.get("content")
        return text if isinstance(text, str) else None

    def _submit_sglang(self, prompt: str, **kwargs: Any) -> str | None:
        """Prefer SGLang's native /generate endpoint; fall back to /v1/completions."""
        try:
            return self._submit_sglang_generate(prompt, **kwargs)
        except Exception:
            return self._submit_vllm(prompt, **kwargs)

    def _submit_sglang_generate(self, prompt: str, **kwargs: Any) -> str | None:
        url = f"{self.base_url.rsplit('/v1', 1)[0]}/generate"
        payload: dict[str, Any] = {
            "text": prompt,
            "sampling_params": {
                "temperature": 0.0,
                "max_new_tokens": 1,
            },
        }
        payload.update(
            {
                k: v
                for k, v in kwargs.items()
                if k not in ("error", "condition", "repeat")
            }
        )
        with httpx.Client(timeout=self.timeout) as client:
            response = client.post(url, json=payload)
            response.raise_for_status()
            data = response.json()
        text = data.get("text")
        if isinstance(text, str):
            return text
        return None


def health_check(base_url: str, *, engine: str = "vllm", timeout: float = 5.0) -> bool:
    """Return True if the serving engine's /v1/models endpoint responds HTTP 200.

    Does not launch or modify anything — read-only probe of server readiness.
    """
    try:
        with httpx.Client(timeout=timeout) as client:
            response = client.get(f"{base_url.rstrip('/')}/models")
            return response.status_code == 200
    except Exception:  # health check degrades to False
        return False
