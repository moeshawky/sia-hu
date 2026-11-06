"""Tests for UpstreamClient — async HTTP proxy to llama.cpp."""

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from capture_proxy.upstream import (
    StreamingUpstreamResponse,
    UpstreamClient,
    UpstreamResponse,
)


class TestHealthCheck:
    """Tests for UpstreamClient.health_check()."""

    @pytest.mark.asyncio
    async def test_health_check_connected(self) -> None:
        client = UpstreamClient()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        with patch.object(
            client._client, "get", new_callable=AsyncMock, return_value=mock_resp
        ):
            result = await client.health_check()
        assert result is True

    @pytest.mark.asyncio
    async def test_health_check_disconnected(self) -> None:
        client = UpstreamClient()
        with patch.object(
            client._client,
            "get",
            new_callable=AsyncMock,
            side_effect=httpx.ConnectError("refused"),
        ):
            result = await client.health_check()
        assert result is False

    @pytest.mark.asyncio
    async def test_health_check_timeout(self) -> None:
        client = UpstreamClient()
        with patch.object(
            client._client,
            "get",
            new_callable=AsyncMock,
            side_effect=httpx.TimeoutException("timeout"),
        ):
            result = await client.health_check()
        assert result is False


class TestProxySync:
    """Tests for non-streaming proxy."""

    @pytest.mark.asyncio
    async def test_proxy_returns_upstream_response(self) -> None:
        client = UpstreamClient()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.headers = {"content-type": "application/json"}
        mock_resp.content = b'{"id":"chatcmpl-123","choices":[]}'
        mock_resp.json.return_value = {"id": "chatcmpl-123", "choices": []}

        with patch.object(
            client._client, "request", new_callable=AsyncMock, return_value=mock_resp
        ):
            result = await client.proxy(
                method="POST",
                path="/v1/chat/completions",
                headers={"Content-Type": "application/json"},
                body={"model": "test", "messages": []},
                stream=False,
            )

        assert isinstance(result, UpstreamResponse)
        assert result.status_code == 200
        assert result.body == {"id": "chatcmpl-123", "choices": []}
        assert isinstance(result.elapsed_ms, float)

    @pytest.mark.asyncio
    async def test_proxy_500_returns_status(self) -> None:
        client = UpstreamClient()
        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_resp.headers = {"content-type": "application/json"}
        mock_resp.content = b'{"error":"internal"}'
        mock_resp.json.return_value = {"error": "internal"}

        with patch.object(
            client._client, "request", new_callable=AsyncMock, return_value=mock_resp
        ):
            result = await client.proxy(
                method="POST",
                path="/v1/chat/completions",
                headers={},
                body={"model": "test", "messages": []},
                stream=False,
            )

        assert result.status_code == 500
        assert result.body == {"error": "internal"}


class TestNonJsonUpstreamResponse:
    """Adversarial: non-JSON upstream response wrapped as {"raw": resp.text}."""

    @pytest.mark.asyncio
    async def test_non_json_response_wraps_raw_text(self) -> None:
        """When upstream returns non-JSON body, response is wrapped preserving raw text."""
        client = UpstreamClient()
        raw_text = "upstream returned HTML error page"
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.headers = {"content-type": "text/html"}
        mock_resp.content = raw_text.encode()
        mock_resp.json.side_effect = ValueError("No JSON object could be decoded")
        mock_resp.text = raw_text

        with patch.object(
            client._client, "request", new_callable=AsyncMock, return_value=mock_resp
        ):
            result = await client.proxy(
                method="POST",
                path="/v1/chat/completions",
                headers={},
                body={"model": "test", "messages": []},
                stream=False,
            )

        assert isinstance(result, UpstreamResponse)
        assert result.status_code == 200
        assert result.body is not None
        assert result.body == {"raw": raw_text}
        assert result.body["raw"] == raw_text

    @pytest.mark.asyncio
    async def test_non_json_empty_body_returns_none(self) -> None:
        """When upstream returns empty body, response body is None."""
        client = UpstreamClient()
        mock_resp = MagicMock()
        mock_resp.status_code = 204
        mock_resp.headers = {}
        mock_resp.content = b""
        mock_resp.json.side_effect = ValueError("No JSON")
        mock_resp.text = ""

        with patch.object(
            client._client, "request", new_callable=AsyncMock, return_value=mock_resp
        ):
            result = await client.proxy(
                method="POST",
                path="/v1/chat/completions",
                headers={},
                body=None,
                stream=False,
            )

        assert result.body is None


class TestProxyStream:
    """Tests for streaming proxy — returns StreamingUpstreamResponse with raw bytes."""

    @pytest.mark.asyncio
    async def test_stream_returns_streaming_response(self) -> None:
        """Streaming proxy returns StreamingUpstreamResponse."""
        client = UpstreamClient()

        mock_send_response = MagicMock()
        mock_send_response.status_code = 200
        mock_send_response.headers = {"content-type": "text/event-stream"}

        async def mock_aiter_raw():
            yield b'data: {"id":"c1","choices":[{"delta":{"content":"Hi"}}]}\n\n'
            yield b"data: [DONE]\n\n"

        mock_send_response.aiter_raw = mock_aiter_raw

        mock_request = MagicMock()
        with (
            patch.object(client._client, "build_request", return_value=mock_request),
            patch.object(
                client._client,
                "send",
                new_callable=AsyncMock,
                return_value=mock_send_response,
            ),
        ):
            result = await client.proxy(
                method="POST",
                path="/v1/chat/completions",
                headers={},
                body={"model": "test", "messages": [], "stream": True},
                stream=True,
            )

        assert isinstance(result, StreamingUpstreamResponse)
        assert result.status_code == 200
        assert result.content_type == "text/event-stream"

        chunks = []
        async for chunk in result.stream:
            chunks.append(chunk)

        assert len(chunks) == 2
        assert b"Hi" in chunks[0]
        assert b"[DONE]" in chunks[1]

    @pytest.mark.asyncio
    async def test_stream_yields_raw_bytes(self) -> None:
        """Streaming proxy yields raw bytes, not formatted strings."""
        client = UpstreamClient()

        mock_send_response = MagicMock()
        mock_send_response.status_code = 200
        mock_send_response.headers = {"content-type": "text/event-stream"}

        raw1 = b'data: {"choices":[{"delta":{"content":"A"}}]}\n\n'
        raw2 = b'data: {"choices":[{"delta":{"content":"B"}}]}\n\n'
        raw3 = b"data: [DONE]\n\n"

        async def mock_aiter_raw():
            yield raw1
            yield raw2
            yield raw3

        mock_send_response.aiter_raw = mock_aiter_raw

        mock_request = MagicMock()
        with (
            patch.object(client._client, "build_request", return_value=mock_request),
            patch.object(
                client._client,
                "send",
                new_callable=AsyncMock,
                return_value=mock_send_response,
            ),
        ):
            result = await client.proxy(
                method="POST",
                path="/v1/chat/completions",
                headers={},
                body={"model": "test", "messages": [], "stream": True},
                stream=True,
            )
            chunks = []
            async for chunk in result.stream:
                chunks.append(chunk)

        assert chunks == [raw1, raw2, raw3]
