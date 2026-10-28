"""The agent loop: model -> tools -> model, until the model answers"""

from __future__ import annotations
import json
from dataclasses import dataclass, field
from typing import Any, Literal
from agentic_suite.agents.executor import ToolExecutor, tool_error_payload
from agentic_suite.agent.prompt import load_system_prompt
from agentic_suite.clients.base import ChatClient
from agentic_suite.middleware.retry import with_retry

DEFAULT_MAX_STEPS = 5

@dataclass
class AgentStep:
    tool: str
    status: Literal['ok', 'error']
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {'tool': self.tool, 'status': self.status, 'error': self.error}

    
@dataclass
class AgentResult:
    model: str
    final_text: str | None
    stop_reason: Literal['completed', 'max_steps']
    steps: list[AgentStep] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            'model': self.model,
            'stop_reason': self.stop_reason,
            'final_text': self.final_text,
            'steps': [step.to_dict() for step in self.steps]
        }


async def run_agent(
    prompt: str,
    *,
    client: ChatClient,
    executor: ToolExecutor,
    runtime_context: dict[str, Any] | None = None,
    system_prompt: str | None = None,
    max_steps: int = DEFAULT_MAX_STEPS
) -> AgentResult:
    """run one user request to completion, or until `max_steps` is reached."""
    
    runtime_context = runtime_context or {}
    messages: list[Any] = [
        {'role': 'system', 'content': system_prompt or load_system_prompt()},
        {'role': 'user', 'content': prompt}
    ]
    tools = executor.registry.get_schema_for_all()
    steps: list[AgentStep] = []

    complete = with_retry(max_attempts=3, max_wait=30.0)(client.complete)

    for _ in range(max_steps):
        response = await complete(messages, tools)

        if not response.called_a_tool:
            return AgentResult(client.model, response.text, 'completed', steps)

        messages.append(client.format_assistant_turn(response))

        for call in response.tool_calls:
            if not call.parsed_ok:
                payload: dict[str, Any] = {'error': 'invalid_json', 'detail': call.parse_error}
                steps.append(AgentStep(call.name, 'error', 'invalid_json'))
            else:
                try:
                    result = await executor.execute(call.name, call.arguments, **runtime_context)
                except Exception as exc:
                    payload = tool_error_payload(exc)
                    steps.append(AgentStep(call.name, 'error', payload['error']))
                else:
                    payload = {'result': result}
                    steps.append(AgentStep(call.name, 'ok'))

            messages.append(client.format_tool_result(call, json.dumps(payload, default=str)))

    return AgentResult(client.model, None, 'max_steps', steps)