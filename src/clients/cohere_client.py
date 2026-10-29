"""Cohere Command models via the v2 Chat API."""
from __future__ import annotations
import cohere
from typing import Any
from agentic_suite.clients.base import ModelResponse, ToolCall, parse_tool_argument
from agentic_suite.clients.errors import ProviderConfigError
from agentic_suite.clients.schema_adapters import adapt_tools
from agentic_suite.config import get_settings

DEFAULT_MAX_TOKENS = 4096
_THINKING_MODES = {'enabled', 'disabled'}

class CohereClient:
    """Async Cohere v2 chat client parameterised by model."""
    provider = 'cohere'

    def __init__(self, model: str, *, api_key: str | None = None, thinking: str | None = None) -> None:
        key = api_key or get_settings().cohere_api_key

        if not key:
            raise ProviderConfigError(f'Model {model!r} is a Cohere model, but COHERE_API_KEY is not set.')
        if thinking and thinking not in _THINKING_MODES:
            raise ValueError(f'thinking must be one of {sorted(_THINKING_MODES)}, got {thinking!r}.')

        self.model = model
        self._thinking = thinking
        self._client = cohere.AsyncClientV2(api_key=key)

    async def complete(self, messages: list[Any], tools: list[dict[str, Any]] | None = None, *, max_tokens: int = DEFAULT_MAX_TOKENS) -> ModelResponse:
        kwargs: dict[str, Any] = {
            'model': self.model,
            'messages': messages,
            'temperature': 0,
            'max_tokens': max_tokens
        }

        if tools:
            kwargs['tools'] = adapt_tools(tools)
        if self._thinking:
            kwargs['thinking'] = {'type': self._thinking}

        response = await self._client.chat(**kwargs)
        message = response.message
        text = ''.join(item.text for item in (message.content or []) if getattr(item, 'type', None) == 'text')
        tool_calls = [parse_tool_argument(call.function.name, call.function.arguments, call.id or '' ) for call in (message.tool_calls or [])]
        tokens = getattr(getattr(response, 'usage', None), 'tokens', None)

        return ModelResponse(
            text = text or None,
            tool_calls = tool_calls,
            finish_reason = str(response.finish_reason) if response.finish_reason else None,
            model = self.model,
            prompt_tokens = int(getattr(tokens, 'input_tokens', 0) or 0),
            completion_tokens = int(getattr(tokens, 'output_tokens', 0) or 0),
            raw_message = message
        )

    def format_assistant_turn(self, response: ModelResponse) -> Any:
        if response.raw_message is not None:
            return response.raw_message

        return {
            'role': 'assistant',
            'tool_calls': [
                {
                    'id': call.id,
                    'type': 'function',
                    'function': {'name': call.name, 'arguments': call.raw_arguments or '{}'}
                }
                for call in response.tool_calls
            ]
        }

    def format_tool_result(self, call: ToolCall, content: str) -> dict[str, Any]:
        return {
            'role': 'tool',
            'tool_call_id': call.id,
            'content': [{'type': 'document', 'document': {'data': content}}]
        }