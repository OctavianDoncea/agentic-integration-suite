from __future__ import annotations
from collections.abc import Iterator
import pytest
from fastapi.testclient import TestClient
from agentic_suite.agents.executor import ToolExecutor
from agentic_suite.agents.registry import build_live_registry
from agentic_suite.config import get_settings
from agentic_suite.db import get_session
from agentic_suite.main import app, get_chat_client, get_executor
from tests.fakes import TEST_BOT_TOKEN, FakeChatClient, FakeSlackFactory, no_sleep, text_response, tool_call_response

AGENT_KEY = 'test-agent-key'

@pytest.fixture
def agent_key(monkeypatch):
    monkeypatch.setenv('AGENT_API_KEY', AGENT_KEY)
    get_settings.cache_clear()

@pytest.fixture
def slack_factory() -> FakeSlackFactory:
    return FakeSlackFactory()

@pytest.fixture
def wire(slack_session, slack_factory) -> Iterator[dict]:
    """Override the model, the database, and the Slack client."""
    holder: dict = {}
    executor = ToolExecutor(
        build_live_registry(),
        base_context={'slack_client_factory': slack_factory},
        sleep=no_sleep,
    )
    app.dependency_overrides[get_session] = lambda: slack_session
    app.dependency_overrides[get_executor] = lambda: executor
    app.dependency_overrides[get_chat_client] = lambda: holder['client']
    yield holder
    app.dependency_overrides.clear()

def post(body: dict, key: str | None = AGENT_KEY):
    headers = {'X-Agent-Key': key} if key else {}
    with TestClient(app) as client:
        return client.post('/agent/run', json=body, headers=headers)

def test_disabled_without_a_configured_key(wire):
    wire['client'] = FakeChatClient([])
    assert post({'prompt': 'hi'}).status_code == 503

def test_rejects_a_wrong_key(agent_key, wire):
    wire['client'] = FakeChatClient([])
    assert post({'prompt': 'hi'}, key='wrong').status_code == 401
    assert post({'prompt': 'hi'}, key=None).status_code == 401

def test_posts_to_slack_end_to_end(agent_key, wire, slack_factory):
    wire['client'] = FakeChatClient(
        [
            tool_call_response('slack_message_tool', {'channel': '#general', 'message': 'Shipped!'}),
            text_response('Posted to #general.'),
        ]
    )
    response = post({'prompt': 'Tell #general we shipped'})

    assert response.status_code == 200
    body = response.json()
    assert body['stop_reason'] == 'completed'
    assert body['steps'] == [{'tool': 'slack_message_tool', 'status': 'ok', 'error': None}]
    assert slack_factory.tokens == [TEST_BOT_TOKEN]
    assert slack_factory.client.posts == [('#general', 'Shipped!')]

def test_missing_workspace_becomes_a_tool_error_not_a_crash(agent_key, wire, slack_session):
    from agentic_suite.integrations.slack.models import SlackInstallation

    slack_session.query(SlackInstallation).delete()
    slack_session.commit()

    wire['client'] = FakeChatClient(
        [
            tool_call_response('slack_message_tool', {'channel': '#general', 'message': 'hi'}),
            text_response('Slack isn\'t connected.'),
        ]
    )
    response = post({'prompt': 'Say hi in #general'})

    assert response.status_code == 200
    assert response.json()["steps"][0] == {
        'tool': 'slack_message_tool',
        'status': 'error',
        'error': 'not_installed',
    }

def test_model_failure_is_a_502(agent_key, wire):
    wire['client'] = FakeChatClient([RuntimeError('provider down')])
    assert post({'prompt': 'hi'}).status_code == 502

def test_prompt_length_is_bounded(agent_key, wire):
    wire['client'] = FakeChatClient([])
    assert post({'prompt': 'x' * 2001}).status_code == 422