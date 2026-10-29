"""Any chat endpoint that speaks Chat Completions format"""
from __future__ import annotations
import asyncio
import json
import httpx
from typing import Any
from agentic_suite.clients.base import ModelResponse, ToolCall, parse_tool_argument, openai_assistant_turn, openai_tool_result
from agentic_suite.clients.errors import ProviderConfigError, ProviderHTTPError
from agentic_suite.clients.schema_adapters import adapt_tools
from agentic_suite.config import get_settings

DEFAULT_MAX_TOKENS = 4096
DEFAULT_TIMEOUT_SECONDS = 120.0

class OpenAICompatClient:
    """Async client for OpenAI-compatible chat endpoints."""
    provider = 'openai_compat'

    def __init__(
        self,
        model: str,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        adapt_tool_schemas: bool | None = None,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        transport: httpx.AsyncBaseTransport | None = None
    ) -> None:
        settings = get_settings()
        url = base_url or settings.openai_compat_base_url

        if not url:
            raise ProviderConfigError(f'Model {model!r} uses an OpenAI-compatible endpoint, but OPENAI_COMPAT_BASE_URL is not set.')

        self.model = model
        self._api_key = api_key or settings.openai_compat_api_key or None
        self._adapt = settings.openai_compat_adapt_schema if adapt_tool_schemas is None else adapt_tool_schemas
        self._http = httpx.AsyncClient(base_url=url.rstrip('/') + '/', timeout=timeout, transport=transport)

    async def complete(self, messages: list[Any], tools: list[dict[str, Any]] | None = None, *, max_tokens: int = DEFAULT_MAX_TOKENS) -> ModelResponse:
        payload: dict[str, Any] = {
            'model': self.model,
            'messages': messages,
            'temperature': 0,
            'max_tokens': max_tokens
        }

        if tools:
            payload['tools'] = adapt_tools(tools) if self._adapt else tools

        headers = {'Authorization': f'Bearer {self._api_key}'} if self._api_key else {}
        try:
            response = await self._http.post('chat/completions', json=payload, headers=headers)
        except httpx.TimeoutException as exc:
            raise asyncio.TimeoutError(str(exc)) from exc
        except httpx.TransportError as exc:
            raise ConnectionError(str(exc)) from exc

        if response.status_code >= 400:
            raise ProviderHTTPError(response.status_code, dict(response.headers), response.text)

        data = response.json()
        choice = data['choices'][0]
        message = choice.get('message') or {}
        content = message.get('content')
        text = content if isinstance(content, str) and content else None
        tool_calls = []

        for call in message.get('tool_calls') or []:
            function = call.get('function') or {}
            arguments = function.get('arguments') or '{}'

            if not isinstance(arguments, str):
                arguments = json.dumps(arguments)
            
            tool_calls.append(parse_tool_argument(function.get('name', ''), arguments, call.get('id') or ''))

        usage = data.get('usage') or {}

        return ModelResponse(
            text = text,
            tool_calls = tool_calls,
            finish_reason = choice.get('finish_reason'),
            model = self.model,
            prompt_tokens = int(usage.get('prompt_tokens') or 0),
            completion_tokens = int(usage.get('completion_tokens') or 0),
            raw_message = message
        )

    def format_assistant_turn(self, response: ModelResponse) -> dict[str, Any]:
        return openai_assistant_turn(response)

    def format_tool_result(self, call: ToolCall, content: str) -> dict[str, Any]:
        return openai_tool_result(call, content)