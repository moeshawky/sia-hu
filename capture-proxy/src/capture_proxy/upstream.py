"""
Upstream Client

Owns: Async HTTP proxying to llama.cpp
Depends: httpx (async HTTP client)
Provides: UpstreamClient.proxy(), UpstreamClient.health_check()

Invariants:
  - Non-streaming: semantically identical JSON (re-serialized; non-ASCII re-escaped, whitespace re-flowed)
  - Streaming: raw bytes tee'd — identical bytes forwarded to client and recorded for diagnostics
  - aiter_raw() used for streaming — network chunks do not align with SSE events
"""

from __future__ import annotations

import json
import time
from typing import TYPE_CHECKING, Any, Literal, overload

import httpx

if TYPE_CHECKING:
    from collections.abc import AsyncIterator


class UpstreamResponse:
    """Wraps a non-streaming upstream HTTP response."""

    def __init__(
        self,
        status_code: int,
        headers: dict[str, str],
        body: dict[str, Any] | None = None,
        elapsed_ms: float = 0.0,
    ) -> None:
        self.status_code = status_code
        self.headers = headers
        self.body = body
        self.elapsed_ms = elapsed_ms


class StreamingUpstreamResponse:
    """Wraps a streaming upstream HTTP response with raw byte access.

    Attributes:
        status_code: HTTP status code from upstream
        headers: Response headers from upstream
        content_type: Content-Type header value
        stream: Async iterator yielding raw network bytes
    """

    def __init__(
        self,
        status_code: int,
        headers: dict[str, str],
        content_type: str,
        stream: AsyncIterator[bytes],
    ) -> None:
        self.status_code = status_code
        self.headers = headers
        self.content_type = content_type
        self.stream = stream


class UpstreamClient:
    """Async HTTP client proxying requests to llama.cpp.

    Uses aiter_raw() for streaming to preserve network chunk boundaries.
    """

    def __init__(self, upstream_url: str = "http://127.0.0.1:11434") -> None:
        self._upstream_url = upstream_url.rstrip("/")
        self._client = httpx.AsyncClient(
            base_url=self._upstream_url,
            timeout=httpx.Timeout(300.0, connect=10.0),
        )

    async def health_check(self, path: str = "/health") -> bool:
        """Check if upstream server is reachable.

        Args:
            path: Upstream health endpoint path (default "/health").

        Returns:
            bool: True if the upstream responds 200 on the health path.

        Raises: None
        """
        try:
            resp = await self._client.get(path)
            return resp.status_code == 200
        except httpx.RequestError:
            return False

    @overload
    async def proxy(
        self,
        method: str,
        path: str,
        headers: dict[str, str],
        body: dict[str, Any] | None = None,
        stream: Literal[False] = False,
    ) -> UpstreamResponse: ...

    @overload
    async def proxy(
        self,
        method: str,
        path: str,
        headers: dict[str, str],
        body: dict[str, Any] | None = None,
        stream: Literal[True] = True,
    ) -> StreamingUpstreamResponse: ...

    async def proxy(
        self,
        method: str,
        path: str,
        headers: dict[str, str],
        body: dict[str, Any] | None = None,
        stream: bool = False,
    ) -> UpstreamResponse | StreamingUpstreamResponse:
        """Forward request to upstream.

        Returns UpstreamResponse for non-streaming, StreamingUpstreamResponse for streaming.
        """
        forwarded = {k: v for k, v in headers.items() if k.lower() != "content-length"}
        if stream:
            return await self._proxy_stream(method, path, forwarded, body)
        return await self._proxy_sync(method, path, forwarded, body)

    async def _proxy_sync(
        self,
        method: str,
        path: str,
        headers: dict[str, str],
        body: dict[str, Any] | None,
    ) -> UpstreamResponse:
        """Non-streaming proxy: forward request, return complete response."""
        forwarded = {k: v for k, v in headers.items() if k.lower() != "content-length"}
        start = time.monotonic()
        resp = await self._client.request(
            method=method,
            url=path,
            headers=forwarded,
            content=json.dumps(body).encode() if body else None,
        )
        elapsed_ms = (time.monotonic() - start) * 1000

        response_body: dict[str, Any] | None = None
        if resp.content:
            try:
                response_body = resp.json()
            except Exception:
                # Graceful degradation — upstream returned non-JSON; raw body preserved for logging
                response_body = {"raw": resp.text}

        return UpstreamResponse(
            status_code=resp.status_code,
            headers=dict(resp.headers),
            body=response_body,
            elapsed_ms=elapsed_ms,
        )

    async def _proxy_stream(
        self,
        method: str,
        path: str,
        headers: dict[str, str],
        body: dict[str, Any] | None,
    ) -> StreamingUpstreamResponse:
        """Streaming proxy: open connection, return raw byte stream.

        Uses aiter_raw() — network chunks do not align with SSE events.
        The caller is responsible for:
          1. Forwarding raw bytes to the client (transparent tee)
          2. Parsing a copy for diagnostics
        """
        forwarded = {k: v for k, v in headers.items() if k.lower() != "content-length"}
        request_body = dict(body) if body else {}
        request_body["stream"] = True

        response = self._client.build_request(
            method=method,
            url=path,
            headers=forwarded,
            content=json.dumps(request_body).encode(),
        )
        send_response = await self._client.send(response, stream=True)

        content_type = send_response.headers.get("content-type", "")

        return StreamingUpstreamResponse(
            status_code=send_response.status_code,
            headers=dict(send_response.headers),
            content_type=content_type,
            stream=send_response.aiter_raw(),
        )
