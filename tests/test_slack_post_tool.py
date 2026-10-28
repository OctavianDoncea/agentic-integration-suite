from __future__ import annotations
import pytest
from pydantic import ValidationError
from agentic_suite.agents.executor import ToolExecutor
from agentic_suite.integrations.slack.models import SlackInstallation
from agentic_suite.integrations.slack.oauth import InstallationNotFoundError
from agentic_suite.integrations.slack.tools import MissingRuntimeContextError, SlackDeliveryError, SlackPostMessageTool, SlackRateLimitedError
from agentic_suite.middleware.retry import get_retry_after, is_retryable
from agentic_suite.sdk.registry import ToolRegistry
from agentic_suite.tools.mock.slack_tool import SlackMessageTool
from agentic_suite.integrations.slack.oauth import ReauthRequiredError
from tests.conftest import slack_session_factory
from tests.fakes import TEST_BOT_TOKEN, TEST_TEAM_ID, FakeSlackClient, FakeSlackFactory, no_sleep

def make_tool() -> SlackPostMessageTool:
    return SlackPostMessageTool(channel='#general', message='Deploy finished')

async def post(session, factory, team_id=TEST_TEAM_ID):
    return await make_tool().execute(session=session, team_id=team_id, slack_client_factory=factory)

# Contract
def test_shares_the_mock_contract():
    assert SlackPostMessageTool.get_schema() == SlackMessageTool.get_schema()

def test_channel_validation_matches_the_mock():
    with pytest.raises(ValidationError):
        SlackPostMessageTool(channel='general', message='hi')

# Happy path
async def test_posts_with_the_stored_tokens(slack_session):
    factory = FakeSlackFactory()
    result = await post(slack_session, factory)

    assert result == {'ok': True, 'channel': 'C0123456789', 'ts': '1700000000.000100'}
    assert factory.tokens == [TEST_BOT_TOKEN]
    assert factory.client.posts == [('#general', 'Deploy finished')]

#Failure classification
async def test_rate_limit_is_retryable_and_carries_retry_after(slack_session):
    factory = FakeSlackFactory(FakeSlackClient(error='ratelimited', status_code=429, headers={'Retry-After': '3'}))

    with pytest.raises(SlackRateLimitedError) as exc_info:
        await post(slack_session, factory)

    assert is_retryable(exc_info.value)
    assert get_retry_after(exc_info.value) == 3.0

async def test_revoked_token_flags_the_installation(slack_session):
    factory = FakeSlackFactory(FakeSlackClient(error='token_revoked'))

    with pytest.raises(ReauthRequiredError) as exc_info:
        await post(slack_session, factory)

    assert not is_retryable(exc_info.value)
    row = slack_session.query(SlackInstallation).filter_by(team_id=TEST_TEAM_ID).one()
    assert row.needs_reauth is True

async def test_channel_errors_are_terminal_with_a_readable_code(slack_session):
    factory = FakeSlackFactory(FakeSlackClient(error='channel_not_found'))

    with pytest.raises(SlackDeliveryError) as exc_info:
        await post(slack_session, factory)

    assert exc_info.value.slack_error == 'channel_not_found'
    assert not is_retryable(exc_info.value)

async def test_server_errors_are_not_retried(slack_session):
    factory = FakeSlackFactory(FakeSlackClient(error='internal_error', status_code=500))

    with pytest.raises(SlackDeliveryError) as exc_info:
        await post(slack_session, factory)

    assert exc_info.value.http_status == 500
    assert not is_retryable(exc_info.value)

async def test_timeouts_are_not_retried(slack_session):
    factory = FakeSlackFactory(FakeSlackClient(timeout=True))

    with pytest.raises(SlackDeliveryError) as exc_info:
        await post(slack_session, factory)

    assert not is_retryable(exc_info.value)

# Missing context
async def test_missing_session_is_a_clear_error():
    with pytest.raises(MissingRuntimeContextError):
        await make_tool().execute(team_id=TEST_TEAM_ID)

async def test_missing_workspace_is_reported(slack_session):
    with pytest.raises(InstallationNotFoundError):
        await post(slack_session, FakeSlackFactory(), team_id=None)

# Through the executor
async def test_rate_limit_is_retried_through_the_executor(slack_session):
    client = FakeSlackClient(error='ratelimited', status_code=429, headers={'Retry-After': '0'}, fail_times=1)
    registry = ToolRegistry()
    registry.register(SlackPostMessageTool)
    executor = ToolExecutor(registry, base_context={'slack_client_factory': FakeSlackFactory(client)}, sleep=no_sleep)
    result = await executor.execute('slack_message_tool', {'channel': '#general', 'message': 'hi'}, session=slack_session, team_id=TEST_TEAM_ID)

    assert result['ok'] is True
    assert len(client.posts) == 2

async def test_server_error_is_attempted_exactly_once_through_the_executor(slack_session):
    client = FakeSlackClient(error='internal_error', status_code=500)
    registry = ToolRegistry()
    registry.register(SlackPostMessageTool)
    executor = ToolExecutor(registry, base_context={'slack_client_factory': FakeSlackFactory(client)}, sleep=no_sleep)

    with pytest.raises(SlackDeliveryError):
        await executor.execute('slack_message_tool', {'channel': '#general', 'message': 'hi'}, session=slack_session, team_id=TEST_TEAM_ID)

    assert len(client.posts) == 1