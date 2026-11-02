from __future__ import annotations
import json
import logging
from typing import Any, ClassVar
import pytest
from agentic_suite.agents.executor import ToolExecutor
from agentic_suite.agents.loop import TOOL_OUTPUT_NOTICE, run_agent
from agentic_suite.agents.policy import WritePolicy
from agentic_suite.sdk.base import BaseTool
from agentic_suite.sdk.registry import ToolRegistry
from tests.fakes import FakeChatClient, no_sleep, text_response, tool_call_response

TELEMETRY = 'telemetry.tool_invocation'
PROMPT = 'Tell #eng hi'
GOOD = {'channel': '#eng', 'text': 'hi'}
BAD_TARGET = {'channel': '#random', 'text': 'hi'}

class NoteTool(BaseTool):
    """Post a note to a channel."""

    tool_name = 'note_tool'
    target_fields = ('channel',)
    sent: ClassVar[list[tuple[str, str]]] = []

    channel: str
    text: str

    async def execute(self, **kwargs: Any) -> dict[str, Any]:
        NoteTool.sent.append((self.channel, self.text))
        return {'ok': True}


class LookupTool(BaseTool):
    """Read something."""

    tool_name = 'lookup_tool'
    read_only = True

    query: str

    async def execute(self, **kwargs: Any) -> dict[str, Any]:
        return {'found': self.query}


@pytest.fixture(autouse=True)
def reset_sent():
    NoteTool.sent = []

@pytest.fixture
def executor() -> ToolExecutor:
    registry = ToolRegistry()
    registry.register(NoteTool)
    registry.register(LookupTool)
    return ToolExecutor(registry, sleep=no_sleep)

def tool_turn(client: FakeChatClient, request: int = 1) -> dict:
    return json.loads(client.requests[request][-1]['content'])

async def test_writes_are_previewed_when_disabled(executor):
    client = FakeChatClient(
        [tool_call_response('note_tool', GOOD), text_response('I would post it.')]
    )
    result = await run_agent(
        PROMPT, client=client, executor=executor, policy=WritePolicy(allow_writes=False)
    )

    assert NoteTool.sent == []
    assert result.steps[0].status == 'preview'
    assert result.steps[0].to_dict()['arguments'] == GOOD
    assert tool_turn(client)['executed'] is False

async def test_a_target_the_user_didnt_name_is_blocked(executor):
    client = FakeChatClient([tool_call_response('note_tool', BAD_TARGET), text_response("I couldn't send that.")])
    result = await run_agent(PROMPT, client=client, executor=executor, policy=WritePolicy(allow_writes=True))

    assert NoteTool.sent == []
    assert (result.steps[0].status, result.steps[0].error) == ('blocked', 'target_not_in_request')
    payload = tool_turn(client)
    assert payload['error'] == 'blocked_by_policy'
    assert payload['rule'] == 'target_not_in_request'
    assert "Don't retry" in payload['hint']

async def test_the_model_can_recover_after_a_block(executor):
    client = FakeChatClient(
        [
            tool_call_response('note_tool', BAD_TARGET, call_id='c1'),
            tool_call_response('note_tool', GOOD, call_id='c2'),
            text_response('Posted to #eng.'),
        ]
    )
    result = await run_agent(PROMPT, client=client, executor=executor, policy=WritePolicy(allow_writes=True))

    assert [s.status for s in result.steps] == ['blocked', 'ok']
    assert NoteTool.sent == [('#eng', 'hi')]

async def test_an_allowed_write_runs(executor):
    client = FakeChatClient([tool_call_response('note_tool', GOOD), text_response('Done.')])
    result = await run_agent(PROMPT, client=client, executor=executor, policy=WritePolicy(allow_writes=True))

    assert NoteTool.sent == [('#eng', 'hi')]
    assert result.steps[0].to_dict() == {'tool': 'note_tool', 'status': 'ok', 'error': None}

async def test_without_a_policy_the_loop_behaves_as_before(executor):
    client = FakeChatClient([tool_call_response('note_tool', BAD_TARGET), text_response('Done.')])
    await run_agent(PROMPT, client=client, executor=executor)

    assert NoteTool.sent == [('#random', 'hi')]

async def test_the_allowlist_blocks_a_tool_that_is_not_listed(executor):
    client = FakeChatClient([tool_call_response('note_tool', GOOD), text_response('No.')])
    policy = WritePolicy(allow_writes=True, allowed_tools=frozenset())
    result = await run_agent(PROMPT, client=client, executor=executor, policy=policy)

    assert (result.steps[0].status, result.steps[0].error) == ('blocked', 'tool_not_allowed')
    assert NoteTool.sent == []

async def test_read_only_tools_run_even_with_writes_disabled(executor):
    client = FakeChatClient([tool_call_response('lookup_tool', {'query': 'q'}), text_response('Found.')])
    result = await run_agent(PROMPT, client=client, executor=executor, policy=WritePolicy(allow_writes=False))

    assert result.steps[0].status == 'ok'
    assert tool_turn(client)['result'] == {'found': 'q'}

async def test_tool_results_are_framed_as_data_by_default(executor):
    client = FakeChatClient([tool_call_response('lookup_tool', {'query': 'q'}), text_response('Found.')])
    await run_agent(PROMPT, client=client, executor=executor)

    assert tool_turn(client)['notice'] == TOOL_OUTPUT_NOTICE


async def test_framing_can_be_turned_off(executor):
    client = FakeChatClient([tool_call_response('lookup_tool', {'query': 'q'}), text_response('Found.')])
    await run_agent(PROMPT, client=client, executor=executor, frame_tool_results=False)

    assert 'notice' not in tool_turn(client)


async def test_arguments_are_recorded_on_every_step_but_exposed_only_when_stopped(executor):
    client = FakeChatClient([tool_call_response('lookup_tool', {'query': 'q'}), text_response('Found.')])
    result = await run_agent(PROMPT, client=client, executor=executor)

    assert result.steps[0].arguments == {'query': 'q'}
    assert 'arguments' not in result.steps[0].to_dict()


async def test_policy_decisions_are_logged(executor, caplog):
    client = FakeChatClient([tool_call_response('note_tool', BAD_TARGET), text_response('No.')])
    with caplog.at_level(logging.INFO, logger=TELEMETRY):
        await run_agent(PROMPT, client=client, executor=executor, policy=WritePolicy(allow_writes=True))

    lines = [r.message for r in caplog.records if r.name == TELEMETRY]
    assert any('policy_block' in line for line in lines)