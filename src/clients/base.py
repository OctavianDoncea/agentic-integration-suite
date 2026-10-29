"""Provider-neutral types for chat models with tool calling."""
from __future__ import annotations
import json
from dataclasses import dataclass, field
from typing import Any, Protocol

@dataclass
class ToolCall:
    name: str
    arguments: dict[str, Any]
    raw_arguments: str | None = None
    parse_error: str | None = None
    id: str = ''

    @property
    def parsed_ok(self) -> bool:
        return self.parse_error is None


@dataclass
class ModelResponse:
    """A normalized view of one completion"""
    
    text: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str | None = None
    model: str | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    raw_message: Any = None

    @property
    def called_a_tool(self) -> bool:
        return bool(self.tool_calls)

    @property
    def first_tool_call(self) -> ToolCall | None:
        return self.tool_calls[0] if self.tool_calls else None


def parse_tool_argument(name: str, arguments_text: str | None, call_id: str = '') -> ToolCall:
    text = arguments_text or '{}'
    try:
        arguments = json.loads(text)
    except json.JSONDecodeError as exc:
        return ToolCall(name, {}, text, str(exc), call_id)

    if not isinstance(arguments, dict):
        return ToolCall(name, {}, text, f'arguments must be an object, got {type(arguments).__name__}', call_id)

    return ToolCall(name, arguments, text, None, call_id)

def openai_assistant_turn(response: ModelResponse) -> dict[str, Any]:
    message: dict[str, Any] = {'role': 'assistant', 'content': response.text or ''}

    if response.tool_calls:
        message['tool_calls'] = [
            {
                'id': call.id,
                'type': 'function',
                'function': {
                    'name': call.name,
                    'arguments': call.raw_arguments if call.raw_arguments is not None else json.dumps(call.arguments)
                }
            }
            for call in response.tool_calls
        ]

    return message

def openai_tool_result(call: ToolCall, content: str) -> dict[str, Any]:
    return {'role': 'tool', 'tool_call_id': call.id, 'content': content}

class ChatClient(Protocol):
    model: str

    async def complete(self, messages: list[Any], tools:list[dict[str, Any]] | None = None, *, max_tokens: int = 1024) -> ModelResponse: ...

    def format_assistant_turn(self, response: ModelResponse) -> dict[str, Any]: ...

    def format_tool_result(self, call: ToolCall, content: str) -> dict[str, Any]: ...