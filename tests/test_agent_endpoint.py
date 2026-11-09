from __future__ import annotations
import pytest
import re
from collections.abc import Iterator
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

def get(path: str, key: str | None = AGENT_KEY):
    headers = {'X-Agent-Key': key} if key else {}
    with TestClient(app) as client:
        return client.get(path, headers=headers)

@pytest.fixture
def content_off(monkeypatch):
    monkeypatch.setenv('AGENT_RUN_STORE_CONTENT', 'false')
    get_settings.cache_clear()
    yield
    monkeypatch.undo()
    get_settings.cache_clear()

def scripted_post(wire):
    """One run that posts to #general, so there's something to read back."""
    wire['client'] = FakeChatClient(
        [
            tool_call_response('slack_message_tool', {'channel': '#general', 'message': 'Shipped!'}),
            text_response('Posted to #general.'),
        ]
    )
    return post({'prompt': 'Tell #general we shipped', 'allow_writes': True})

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
    response = post({'prompt': 'Tell #general we shipped', 'allow_writes': True})

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
    response = post({'prompt': 'Say hi in #general', 'allow_writes': True})

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

def test_writes_are_previews_by_default(agent_key, wire, slack_factory):
    args = {'channel': '#general', 'message': 'Shipped!'}
    wire['client'] = FakeChatClient([tool_call_response('slack_message_tool', args), text_response('I would post it.')])
    response = post({'prompt': 'Tell #general we shipped'})

    assert response.status_code == 200
    assert response.json()['steps'] == [{'tool': 'slack_message_tool', 'status': 'preview', 'error': None, 'arguments': args}]
    assert slack_factory.client.posts == []

def test_a_target_the_user_didnt_name_is_blocked(agent_key, wire, slack_factory):
    args = {'channel': '#random', 'message': 'Shipped!'}
    wire['client'] = FakeChatClient([tool_call_response('slack_message_tool', args), text_response("I couldn't post that.")])
    response = post({'prompt': 'Tell #general we shipped', 'allow_writes': True})

    step = response.json()['steps'][0]
    assert (step['status'], step['error']) == ('blocked', 'target_not_in_request')
    assert slack_factory.client.posts == []

def test_write_tools_restricts_which_tools_may_run(agent_key, wire, slack_factory):
    args = {'channel': '#general', 'message': 'Shipped!'}
    wire['client'] = FakeChatClient([tool_call_response('slack_message_tool', args), text_response('Not allowed.')])
    response = post({'prompt': 'Tell #general we shipped', 'allow_writes': True, 'write_tools': ['jira_issue_tool']})

    step = response.json()['steps'][0]
    assert (step['status'], step['error']) == ('blocked', 'tool_not_allowed')
    assert slack_factory.client.posts == []

def test_write_tools_must_be_a_list(agent_key, wire):
    wire['client'] = FakeChatClient([])
    assert post({'prompt': 'hi', 'write_tools': 'slack_message_tool'}).status_code == 422

def test_a_run_is_recorded_and_can_be_fetched(agent_key, wire):
    response = scripted_post(wire)
    body = response.json()

    assert response.status_code == 200
    assert re.fullmatch(r'[0-9a-f]{32}', body['run_id'])
    assert body['usage']['model_calls'] == 2

    record = get(f"/agent/runs/{body['run_id']}").json()
    assert record['status'] == 'completed'
    assert record['request']['prompt'] == 'Tell #general we shipped'
    assert record['request']['allow_writes'] is True
    assert record['steps'][0]['tool'] == 'slack_message_tool'
    assert record['steps'][0]['arguments'] == {'channel': '#general', 'message': 'Shipped!'}
    assert record['usage']['model_calls'] == 2

def test_run_endpoints_need_the_agent_key(agent_key, wire):
    wire['client'] = FakeChatClient([])
    some_id = '0' * 32

    for path in (f'/agent/runs/{some_id}', '/agent/runs', '/agent/stats'):
        assert get(path, key=None).status_code == 401
        assert get(path, key='wrong').status_code == 401

def test_an_unknown_run_is_a_404_and_a_malformed_id_is_rejected(agent_key, wire):
    wire['client'] = FakeChatClient([])

    assert get(f'/agent/runs/{'0' * 32}').status_code == 404
    assert get('/agent/runs/not-a-run-id').status_code == 422

def test_runs_list_newest_first(agent_key, wire):
    first = scripted_post(wire).json()['run_id']
    second = scripted_post(wire).json()['run_id']

    listing = get('/agent/runs').json()

    assert [run['id'] for run in listing['runs']] == [second, first]
    assert listing['runs'][0]['prompt_preview'] == 'Tell #general we shipped'
    assert listing['next_before'] is None

def test_a_failed_run_is_recorded_before_the_502(agent_key, wire):
    wire['client'] = FakeChatClient([RuntimeError('provider down')])
    assert post({'prompt': 'hi'}).status_code == 502

    failed = get('/agent/runs?status=failed').json()['runs']

    assert len(failed) == 1
    run = get(f'/agent/runs/{failed[0]['id']}').json()
    assert run['status'] == 'failed'
    assert run['error_type'] == 'RuntimeError'

def test_content_can_be_left_out_of_records(agent_key, wire, content_off):
    run_id = scripted_post(wire).json()['run_id']
    record = get(f'/agent/runs/{run_id}').json()

    assert record['request']['prompt'] is None
    assert record['final_text'] is None
    assert record['steps'][0]['argument_keys'] == ['channel', 'message']
    assert 'arguments' not in record['steps'][0]

def test_a_recording_failure_does_not_fail_the_request(agent_key, wire, monkeypatch):
    from agentic_suite.agents import run_store

    def broken(*args, **kwargs):
        raise RuntimeError('db down')

    monkeypatch.setattr(run_store, 'record_run', broken)
    response = scripted_post(wire)

    assert response.status_code == 200
    assert response.json()['run_id'] is None
    assert response.json()['final_text'] == 'Posted to #general.'


def test_stats_cover_recent_runs(agent_key, wire):
    scripted_post(wire)
    stats = get('/agent/stats').json()

    assert stats['runs'] == 1
    assert stats['by_status'] == {'completed': 1}
    assert stats['duration_ms_p50'] is not None