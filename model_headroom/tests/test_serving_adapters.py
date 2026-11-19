"""Mock tests for SGLang/vLLM serving adapter clients.

No real inference server is launched. httpx.Client is patched to return
scripted responses, so the adapter logic is tested without network or GPU.
"""

from unittest.mock import MagicMock, patch

import httpx

from model_headroom.acquire.serving import ServingEngineClient, health_check


def _mock_client(
    status_code: int = 200, json_payload: object | None = None, text: str | None = None
):
    mock_response = MagicMock(spec=httpx.Response)
    mock_response.status_code = status_code
    mock_response.raise_for_status = MagicMock()
    if json_payload is not None:
        mock_response.json = MagicMock(return_value=json_payload)
    else:
        mock_response.json = MagicMock(side_effect=ValueError("no json"))
    mock_response.text = text or ""
    mock_client = MagicMock(spec=httpx.Client)
    mock_client.post = MagicMock(return_value=mock_response)
    mock_client.get = MagicMock(return_value=mock_response)
    mock_client.__enter__ = MagicMock(return_value=mock_client)
    mock_client.__exit__ = MagicMock(return_value=False)
    return mock_client


def test_vllm_submit_returns_completions_text():
    with patch(
        "model_headroom.acquire.serving.httpx.Client",
        return_value=_mock_client(json_payload={"choices": [{"text": "the answer"}]}),
    ):
        client = ServingEngineClient("http://127.0.0.1:8001/v1", engine="vllm")
        out = client.submit("What is 2+2?")
        assert out == "the answer"
        assert len(client.calls) == 1
        assert client.calls[0][0] == "What is 2+2?"


def test_vllm_submit_uses_message_content_fallback():
    with patch(
        "model_headroom.acquire.serving.httpx.Client",
        return_value=_mock_client(
            json_payload={"choices": [{"message": {"content": "chat answer"}}]}
        ),
    ):
        client = ServingEngineClient("http://127.0.0.1:8001/v1", engine="vllm")
        out = client.submit("hello")
        assert out == "chat answer"


def test_vllm_submit_returns_none_on_http_error():
    mock_resp = MagicMock(spec=httpx.Response)
    mock_resp.status_code = 500
    mock_resp.raise_for_status = MagicMock(
        side_effect=httpx.HTTPStatusError(
            "boom", request=MagicMock(), response=mock_resp
        )
    )
    mock_client = MagicMock(spec=httpx.Client)
    mock_client.post = MagicMock(return_value=mock_resp)
    mock_client.__enter__ = MagicMock(return_value=mock_client)
    mock_client.__exit__ = MagicMock(return_value=False)
    with patch("model_headroom.acquire.serving.httpx.Client", return_value=mock_client):
        client = ServingEngineClient("http://127.0.0.1:8001/v1", engine="vllm")
        out = client.submit("hello")
        assert out is None


def test_sglang_submit_uses_generate_endpoint():
    with patch(
        "model_headroom.acquire.serving.httpx.Client",
        return_value=_mock_client(json_payload={"text": "sglang output"}),
    ) as mock_client_cls:
        client = ServingEngineClient("http://127.0.0.1:8000/v1", engine="sglang")
        out = client.submit("prompt")
        assert out == "sglang output"
        # Verify /generate URL was constructed (base without /v1 + /generate)
        posted_url = mock_client_cls.return_value.post.call_args[0][0]
        assert posted_url == "http://127.0.0.1:8000/generate"


def test_sglang_falls_back_to_completions_when_generate_fails():
    def side_effect(*args, **kwargs):
        if "/generate" in args[0]:
            raise httpx.ConnectError("refused")
        mock_resp = MagicMock(spec=httpx.Response)
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json = MagicMock(return_value={"choices": [{"text": "fallback"}]})
        mock_resp.text = ""
        return mock_resp

    with patch(
        "model_headroom.acquire.serving.httpx.Client",
        return_value=_mock_client(),
    ) as mock_client_cls:
        mock_client_cls.return_value.post = MagicMock(side_effect=side_effect)
        client = ServingEngineClient("http://127.0.0.1:8000/v1", engine="sglang")
        out = client.submit("prompt")
        assert out == "fallback"


def test_invalid_engine_raises():
    try:
        ServingEngineClient("http://127.0.0.1:8000/v1", engine="ollama")  # type: ignore[arg-type]
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for invalid engine")


def test_health_check_true_on_200():
    with patch(
        "model_headroom.acquire.serving.httpx.Client",
        return_value=_mock_client(json_payload={"data": [{"id": "qwen3"}]}),
    ):
        assert health_check("http://127.0.0.1:8001/v1", engine="vllm") is True


def test_health_check_false_on_connection_error():
    with patch(
        "model_headroom.acquire.serving.httpx.Client",
        side_effect=httpx.ConnectError("refused"),
    ):
        assert health_check("http://127.0.0.1:8001/v1", engine="vllm") is False


def test_client_is_injectable_not_singleton():
    c1 = ServingEngineClient("http://127.0.0.1:8000/v1", engine="sglang")
    c2 = ServingEngineClient("http://127.0.0.1:8001/v1", engine="vllm")
    assert c1.engine == "sglang"
    assert c2.engine == "vllm"
    assert c1.base_url != c2.base_url
