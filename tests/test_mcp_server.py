from __future__ import annotations
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any
import pytest
from mcp import Client
from agentic_suite.agents.executor import ToolExecutor
from agentic_suite.agents.registry import build_live_registry
from agentic_suite.integrations.slack.models import SlackInstallation
from agentic_suite.integrations.slack.tools import SlackPostMessageTool
from agentic_suite.mcp_server import SIMULATED_NOTE, build_server, make_database_context, tool_to_mcp
from agentic_suite.tools.mock.github_tool import GitHubPRTool
from agentic_suite.tools.mock.jira_tool import JiraIssueTool
from tests.fakes import TEST_TEAM_ID, FakeSlackClient, FakeSlackFactory, no_sleep

JIRA_ARGS = {'title': 'Fix login', 'priority': 'High', 'project_key': 'ENG'}
SLACK_ARGS = {'channel': '#general', 'message': 'Shipped!'}

def make_executor(slack_factory: FakeSlackFactory) -> ToolExecutor:
    return ToolExecutor(
        build_live_registry(),
        base_context={'slack_client_factory': slack_factory},
        sleep=no_sleep,
    )

def session_context(session) -> Any:
    """Context factory that reuses one test session and never commits."""

    @asynccontextmanager
    async def context() -> AsyncIterator[dict[str, Any]]:
        yield {'session': session, 'team_id': TEST_TEAM_ID}

    return context

# Pure mapping
def test_schema_is_the_registry_schema_verbatim():
    tool = tool_to_mcp(JiraIssueTool)
    assert tool.input_schema == JiraIssueTool.get_schema()['function']['parameters']
    assert 'inject_500_error' not in tool.input_schema['properties']

def test_read_only_tools_carry_no_write_hints():
    annotations = tool_to_mcp(GitHubPRTool).annotations
    assert annotations.read_only_hint is True
    assert annotations.destructive_hint is None
    assert annotations.idempotent_hint is None

def test_slack_post_is_marked_non_idempotent():
    annotations = tool_to_mcp(SlackPostMessageTool).annotations
    assert annotations.read_only_hint is False
    assert annotations.destructive_hint is False
    assert annotations.idempotent_hint is False
    assert annotations.open_world_hint is True

def test_only_simulated_tools_say_so():
    assert tool_to_mcp(JiraIssueTool).description.endswith(SIMULATED_NOTE)
    assert SIMULATED_NOTE not in tool_to_mcp(SlackPostMessageTool).description

# Protocol round trip
async def test_lists_every_live_tool(slack_session):
    server = build_server(make_executor(FakeSlackFactory()), session_context(slack_session))

    async with Client(server, raise_exceptions=True) as client:
        result = await client.list_tools()

    names = {tool.name for tool in result.tools}
    assert names == {cls.get_name() for cls in build_live_registry().tools()}

async def test_successful_call_returns_structured_content(slack_session):
    server = build_server(make_executor(FakeSlackFactory()), session_context(slack_session))

    async with Client(server, raise_exceptions=True) as client:
        result = await client.call_tool('jira_issue_tool', JIRA_ARGS)

    assert not result.is_error
    assert result.structured_content['status'] == 'created'
    assert json.loads(result.content[0].text)['status'] == 'created'

async def test_invalid_arguments_come_back_as_a_readable_tool_error(slack_session):
    server = build_server(make_executor(FakeSlackFactory()), session_context(slack_session))

    async with Client(server, raise_exceptions=True) as client:
        result = await client.call_tool('jira_issue_tool', {'title': 'only this'})

    assert result.is_error
    assert result.structured_content['error'] == 'invalid_arguments'
    fields = {detail['field'] for detail in result.structured_content['details']}
    assert fields == {'priority', 'project_key'}

async def test_unknown_tool_is_a_tool_error_not_a_protocol_error(slack_session):
    server = build_server(make_executor(FakeSlackFactory()), session_context(slack_session))

    async with Client(server, raise_exceptions=True) as client:
        result = await client.call_tool('hallucinated_tool', {})

    assert result.is_error
    assert result.structured_content['error'] == 'unknown_tool'

async def test_real_slack_tool_posts_through_mcp(slack_session):
    factory = FakeSlackFactory()
    server = build_server(make_executor(factory), session_context(slack_session))

    async with Client(server, raise_exceptions=True) as client:
        result = await client.call_tool('slack_message_tool', SLACK_ARGS)

    assert not result.is_error
    assert factory.client.posts == [('#general', 'Shipped!')]

async def test_revocation_flag_survives_a_failed_call(slack_session_factory):
    """The tool fails, but the database context must still commit its flag."""
    factory = FakeSlackFactory(FakeSlackClient(error='token_revoked'))
    server = build_server(make_executor(factory), make_database_context(slack_session_factory))

    async with Client(server, raise_exceptions=True) as client:
        result = await client.call_tool('slack_message_tool', SLACK_ARGS)

    assert result.is_error
    assert result.structured_content['error'] == 'reauthorization_required'

    with slack_session_factory() as fresh:  # a new session sees only committed data
        row = fresh.query(SlackInstallation).filter_by(team_id=TEST_TEAM_ID).one()
        assert row.needs_reauth is True

async def test_a_broken_context_is_reported_not_raised():
    @asynccontextmanager
    async def broken() -> AsyncIterator[dict[str, Any]]:
        raise ConnectionError('database unreachable')
        yield {}  # pragma: no cover

    server = build_server(make_executor(FakeSlackFactory()), broken)

    async with Client(server, raise_exceptions=True) as client:
        result = await client.call_tool('jira_issue_tool', JIRA_ARGS)

    assert result.is_error
    assert result.structured_content == {'error': 'server_error', 'type': 'ConnectionError'}