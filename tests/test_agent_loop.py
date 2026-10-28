from __future__ import annotations
import json
import pytest
from agentic_suite.agents.executor import ToolExecutor
from agentic_suite.agents.loop import run_agent
from agentic_suite.clients.base import ModelResponse, ToolCall
from agentic_suite.sdk.registry import ToolRegistry
from agentic_suite.tools.mock.jira_tool import JiraIssueTool
from tests.fakes import FakeChatClient, StatusError, no_sleep, text_response, tool_call_response

JIRA_ARGS = {"title": "Fix login", "priority": "High", "project_key": "ENG"}

@pytest.fixture
def executor() -> ToolExecutor:
    registry = ToolRegistry()
    registry.register(JiraIssueTool)
    return ToolExecutor(registry, sleep=no_sleep)

async def test_answers_directly_when_no_tool_is_needed(executor):
    client = FakeChatClient([text_response('We avoid Friday deploys.')])
    result = await run_agent('Friday deploys?', client=client, executor=executor)

    assert result.stop_reason == 'completed'
    assert result.final_text == 'We avoid Friday deploys.'
    assert result.steps == []

async def test_executes_a_tool_and_feeds_the_result_back(executor):
    client = FakeChatClient(
        [tool_call_response('jira_issue_tool', JIRA_ARGS), text_response('Filed ENG-123.')]
    )
    result = await run_agent('File a ticket', client=client, executor=executor)

    assert result.final_text == 'Filed ENG-123.'
    assert [s.to_dict() for s in result.steps] == [
        {'tool': 'jira_issue_tool', 'status': 'ok', 'error': None}
    ]

    second_request = client.requests[1]
    assistant_turn, tool_turn = second_request[-2], second_request[-1]
    assert assistant_turn['tool_calls'][0]['function']['name'] == 'jira_issue_tool'
    assert tool_turn['tool_call_id'] == 'call_1'
    assert json.loads(tool_turn['content'])['result']['status'] == 'created'

async def test_model_recovers_from_a_validation_error(executor):
    client = FakeChatClient(
        [
            tool_call_response('jira_issue_tool', {'title': 'Fix login'}, call_id='c1'),
            tool_call_response('jira_issue_tool', JIRA_ARGS, call_id='c2'),
            text_response('Filed it.'),
        ]
    )
    result = await run_agent('File a ticket', client=client, executor=executor)

    assert [(s.status, s.error) for s in result.steps] == [
        ('error', 'invalid_arguments'),
        ('ok', None),
    ]
    error_turn = json.loads(client.requests[1][-1]['content'])
    assert error_turn['error'] == 'invalid_arguments'

async def test_unparseable_arguments_are_reported_not_executed(executor):
    bad_call = ModelResponse(
        tool_calls=[
            ToolCall(
                name='jira_issue_tool',
                arguments={},
                raw_arguments='{title: unquoted',
                parse_error='Expecting property name',
                id='c1',
            )
        ]
    )
    client = FakeChatClient([bad_call, text_response('Sorry.')])
    result = await run_agent('File a ticket', client=client, executor=executor)

    assert [(s.status, s.error) for s in result.steps] == [('error', 'invalid_json')]

async def test_stops_after_max_steps(executor):
    client = FakeChatClient([tool_call_response('jira_issue_tool', JIRA_ARGS)] * 3)
    result = await run_agent('Loop forever', client=client, executor=executor, max_steps=3)

    assert result.stop_reason == 'max_steps'
    assert result.final_text is None
    assert len(result.steps) == 3

async def test_model_rate_limits_are_retried(executor):
    client = FakeChatClient([StatusError(429, retry_after='0'), text_response('Done.')])
    result = await run_agent('Hello', client=client, executor=executor)

    assert result.final_text == 'Done.'
    assert len(client.requests) == 2

async def test_system_prompt_is_the_packaged_one(executor):
    from agentic_suite.agents.prompt import load_system_prompt

    client = FakeChatClient([text_response('Hi.')])
    await run_agent('Hello', client=client, executor=executor)

    assert client.requests[0][0] == {'role': 'system', 'content': load_system_prompt()}