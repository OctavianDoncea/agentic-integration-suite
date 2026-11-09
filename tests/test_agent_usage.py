from __future__ import annotations
import json
import pytest
from typing import Any
from agentic_suite.agents.executor import ToolExecutor
from agentic_suite.agents.loop import RunTrace, run_agent
from agentic_suite.clients.base import ModelResponse, ToolCall
from agentic_suite.sdk.base import BaseTool
from agentic_suite.sdk.registry import ToolRegistry
from tests.fakes import FakeChatClient, no_sleep

class EchoTool(BaseTool):
    """Return what it was given."""

    tool_name = 'echo_tool'
    read_only = True
    text: str

    async def execute(self, **kwargs: Any) -> dict[str, Any]:
        return {'echo': self.text}

@pytest.fixture
def executor() -> ToolExecutor:
    registry = ToolRegistry()
    registry.register(EchoTool)
    return ToolExecutor(registry, sleep=no_sleep)

def tool_turn(text: str = 'hi') -> ModelResponse:
    args = {'text': text}
    return ModelResponse(
        tool_calls=[ToolCall(name='echo_tool', arguments=args, raw_arguments=json.dumps(args), id='c1')],
        prompt_tokens=100,
        completion_tokens=10,
    )

def answer(text: str = 'Done.') -> ModelResponse:
    return ModelResponse(text=text, prompt_tokens=150, completion_tokens=20)

async def test_tokens_and_calls_are_summed_across_turns(executor):
    client = FakeChatClient([tool_turn(), answer()])
    result = await run_agent('go', client=client, executor=executor)

    assert result.usage.model_calls == 2
    assert result.usage.prompt_tokens == 250
    assert result.usage.completion_tokens == 30

async def test_timings_are_recorded_and_nested(executor):
    client = FakeChatClient([tool_turn(), answer()])
    result = await run_agent('go', client=client, executor=executor)

    assert result.usage.duration_ms >= result.usage.model_ms >= 0

async def test_every_executed_step_has_a_latency(executor):
    client = FakeChatClient([tool_turn(), answer()])
    result = await run_agent('go', client=client, executor=executor)

    assert result.steps[0].latency_ms is not None
    assert result.steps[0].latency_ms >= 0

async def test_the_response_shape_gains_usage_but_steps_do_not_gain_latency(executor):
    client = FakeChatClient([tool_turn(), answer()])
    body = (await run_agent('go', client=client, executor=executor)).to_dict()

    assert set(body['usage']) == {'model_calls', 'prompt_tokens', 'completion_tokens', 'model_ms', 'duration_ms'}
    assert body['steps'] == [{'tool': 'echo_tool', 'status': 'ok', 'error': None}]

async def test_a_trace_survives_a_model_failure(executor):
    client = FakeChatClient([tool_turn(), RuntimeError('provider down')])
    trace = RunTrace()

    with pytest.raises(RuntimeError):
        await run_agent('go', client=client, executor=executor, trace=trace)

    assert [step.tool for step in trace.steps] == ['echo_tool']
    assert trace.usage.model_calls == 2
    assert trace.usage.prompt_tokens == 100
    assert trace.usage.duration_ms >= 0

async def test_without_a_trace_the_loop_behaves_as_before(executor):
    client = FakeChatClient([answer('No tools needed.')])
    result = await run_agent('go', client=client, executor=executor)

    assert result.stop_reason == 'completed'
    assert result.steps == []