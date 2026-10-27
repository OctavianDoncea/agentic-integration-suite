from __future__ import annotations
import logging
from typing import Any
from groq import AsyncGroq
from agentic_suite.config import get_settings
from agentic_suite.clients.base import ModelResponse, ToolCall, openai_assistant_turn, openai_tool_result, parse_tool_argument

__all__ = ['GroqClient', 'ModelResponse', 'ToolCall']

logger = logging.getLogger('clients.groq')

class GroqClient:
    """Async Groq client parameterised by model."""

    provider = 'groq'

    def __init__(self, model: str | None = None, *, api_key: str | None = None) -> None:
        settings = get_settings()
        self.model = model or settings.groq_model_full
        self._client = AsyncGroq(api_key=api_key or settings.groq_api_key)

    async def complete(self, messages: list[Any], tools: list[dict[str, Any]] | None = None, *, tool_choice: str = 'auto', max_tokens: int = 1024) -> ModelResponse:
        kwargs: dict[str, Any] = {
            'model': self.model,
            'messages': messages,
            'temperature': 0,
            'max_tokens': max_tokens
        }

        if tools:
            kwargs['tools'] = tools
            kwargs['tool_choice'] = tool_choice

        completion = await self._client.chat.completions.create(**kwargs)
        choice = completion.choices[0]
        message = choice.message

        tool_calls = [
            parse_tool_argument(raw.function.name, raw.function.arguments, raw.id)
            for raw in message.tool_calls or []
        ]

        usage = completion.usage

        return ModelResponse(
            text = message.content,
            tool_calls = tool_calls,
            finish_reason = choice.finish_reason,
            model = completion.model,
            prompt_tokens = getattr(usage, 'prompt_tokens', 0) or 0,
            completion_tokens = getattr(usage, 'completion_tokens', 0) or 0
        )

    def format_assistant_turn(self, response: ModelResponse) -> dict[str, Any]:
        return openai_assistant_turn(response)

    def format_tool_result(self, call: ToolCall, content: str) -> dict[str, Any]:
        return openai_tool_result(call, content)