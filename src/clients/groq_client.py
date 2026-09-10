from __future__ import annotations
import logging
from dataclasses import dataclass, field
from typing import Any
from groq import AsyncGroq
from agentic_suite.config import get_settings

logger = logging.getLogger('clients.groq')

@dataclass
class ToolCall:
    name: str
    arguments: dict[str, Any]
    raw_arguments: str | None = None
    parse_error: str | None = None

    @property
    def parsed_ok(self) -> bool:
        return self.parse_error is None


@dataclass
class ModelResponse:
    text: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str | None = None
    model: str | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def called_a_tool(self) -> bool:
        return bool(self.tool_calls)

    @property
    def first_tool_call(self) -> ToolCall | None:
        return self.tool_calls[0] if self.tool_calls else None


class GroqClient:
    def __init__(self, model: str | None = None, *, api_key: str | None = None) -> None:
        settings = get_settings()
        self.model = model or settings.groq_model_full
        self._client = AsyncGroq(api_key=api_key or settings.groq_api_key)

    async def complete(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None, *, tool_choice: str = 'auto', max_tokens: int = 1024) -> ModelResponse:
        import json

        kwargs: dict[str, Any] = {
            'model': self.model,
            'messages': messages,
            'temperature': 0,
            'max_tokens': max_tokens,
        }
        if tools:
            kwargs['tools'] = tools
            kwargs['tool_choice'] = tool_choice

        completion = await self._client.chat.completions.create(**kwargs)
        choice = completion.choices[0]
        message = choice.message

        tool_calls: list[ToolCall] = []
        for raw in message.tool_calls or []:
            arguments_text = raw.function.arguments
            try:
                arguments = json.loads(arguments_text)
            except json.JSONDecodeError as e:
                tool_calls.append(ToolCall(
                    name=raw.function.name,
                    arguments={},
                    raw_arguments=arguments_text,
                    parse_error=str(e),
                ))
                continue

            if not isinstance(arguments, dict):
                tool_calls.append(ToolCall(
                    name=raw.function.name,
                    arguments={},
                    raw_arguments=arguments_text,
                    parse_error=f'arguments must be an object, got {type(arguments).__name__}',
                ))
                continue

            tool_calls.append(ToolCall(name=raw.function.name, arguments=arguments))

        usage = completion.usage
        return ModelResponse(
            text=message.content,
            tool_calls=tool_calls,
            finish_reason=choice.finish_reason,
            model=completion.model,
            prompt_tokens=getattr(usage, 'prompt_tokens', 0) or 0,
            completion_tokens=getattr(usage, 'completion_tokens', 0) or 0
        )