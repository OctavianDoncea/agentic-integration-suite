from __future__ import annotations
from collections.abc import Iterator
import httpx
import pytest
import respx
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from agentic_suite.db import Base, get_session
from agentic_suite.integrations.slack import oauth
from agentic_suite.integrations.slack.models import SlackInstallation
from agentic_suite.integrations.slack.oauth import SLACK_ACCESS_URL
from agentic_suite.main import app

TOKEN_PAYLOAD = {
    'ok': True,
    'access_token': 'xoxb-real-looking-token',
    'scope': 'chat:write,channels:read',
    'bot_user_id': 'UOKRQLJ9H',
    'team': {'id': 'T012AB3C4', 'name': 'Test Team'}
}

@pytest.fixture
def db_session() -> Iterator[Session]:
    engine = create_engine('sqlite+pysqlite:///:memory:')
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine, expire_on_commit=False)()
    try:
        yield session
    finally:
        session.close()

@pytest.fixture
def client(db_session: Session) -> Iterator[TestClient]:
    app.dependency_overrides[get_session] = lambda: db_session
    oauth.state_store.clear()
    with TestClient(app, follow_redirects=False) as test_client:
        yield test_client
    app.dependency_overrides.clear()
    oauth.state_store.clear()

def _issued_state(client: TestClient) -> str:
    response = client.get('/slack/install')
    assert response.status_code == 302
    return dict(httpx.URL(response.headers['location']).params)['state']

def test_install_redirects_to_slack_with_expected_params(client: TestClient):
    response = client.get('/slack/install')
    assert response.status_code == 302

    url = httpx.URL(response.headers['location'])
    params = dict(url.params)

    assert str(url).startswith('https://slack.com/oauth/v2/authorize')
    assert params['client_id'] == '1234567890.0987654321'
    assert params['scope'] == 'chat:write,channels:read'
    assert params['redirect_uri'] == 'http://testserver/slack/oauth/callback'
    assert params['state']

def test_each_install_issues_a_distinct_state(client: TestClient):
    assert _issued_state(client) != _issued_state(client)

@respx.mock
def test_full_install_to_callback_creates_db_row(client: TestClient, db_session: Session):
    respx.post(SLACK_ACCESS_URL).mock(return_value=httpx.Response(200, json=TOKEN_PAYLOAD))

    state = _issued_state(client)
    response = client.get(f'/slack/oauth/callback?code=fake-code&state={state}')

    assert response.status_code == 200

    row = db_session.query(SlackInstallation).filter_by(team_id='T012AB3C4').one()
    assert row.access_token == 'xoxb-real-looking-token'
    assert row.team_name == 'Test Team'
    assert row.scope == 'chat:write,channels:read'
    assert row.needs_reauth is False

@respx.mock
def test_reinstall_updates_the_existing_row(client: TestClient, db_session: Session):
    route = respx.post(SLACK_ACCESS_URL).mock(return_value=httpx.Response(200, json=TOKEN_PAYLOAD))
    client.get(f'/slack/oauth/callback?code=c1&state={_issued_state(client)}')

    updated = {**TOKEN_PAYLOAD, 'access_token': 'xoxb-rotated', 'scope': 'chat:write'}
    route.mock(return_value=httpx.Response(200, json=updated))
    client.get(f'/slack/oauth/callback?code=c2&state={_issued_state(client)}')

    rows = db_session.query(SlackInstallation).filter_by(team_id='T012AB3C4').all()
    assert len(rows) == 1
    assert rows[0].access_token == 'xoxb-rotated'
    assert rows[0].scope == 'chat:write'

@respx.mock
def test_mismatched_state_is_rejected_without_calling_slack(client: TestClient):
    route = respx.post(SLACK_ACCESS_URL).mock(return_value=httpx.Response(200, json=TOKEN_PAYLOAD))
    response = client.get('/slack/oauth/callback?code=fake-code&state=forged-state')

    assert response.status_code == 400
    assert response.json()['error'] == 'invalid_state'
    assert not route.called

def test_missing_state_is_rejected(client: TestClient):
    assert client.get('/slack/oauth/callback?code=fake-code').status_code == 400

@respx.mock
def test_state_cannot_be_replayed(client: TestClient):
    respx.post(SLACK_ACCESS_URL).mock(return_value=httpx.Response(200, json=TOKEN_PAYLOAD))

    state = _issued_state(client)
    assert client.get(f'/slack/oauth/callback?code=c1&state={state}').status_code == 200

    second = client.get(f'/slack/oauth/callback?code=c1&state={state}')
    assert second.status_code == 400
    assert 'already been used' in second.json()['detail']

def test_expired_state_is_rejected(client: TestClient, monkeypatch):
    monkeypatch.setenv('OAUTH_STATE_TTL_SECONDS', '0')
    from agentic_suite.config import get_settings

    get_settings.cache_clear()
    state = _issued_state(client)

    import time as time_module

    monkeypatch.setattr(oauth.time, 'time', lambda: time_module.time() + 60)
    response = client.get(f'/slack/oauth/callback?code=c&state={state}')

    assert response.status_code == 400
    assert 'expired' in response.json()['detail']
    get_settings.cache_clear()

def test_user_denial_returns_a_friendly_page(client: TestClient, db_session: Session):
    response = client.get('/slack/oauth/callback?error=access_denied')

    assert response.status_code == 200
    assert 'cancelled' in response.text.lower()
    assert db_session.query(SlackInstallation).count() == 0

@respx.mock
def test_slack_rejecting_the_code_is_reported_not_crashed(client: TestClient, db_session: Session):
    respx.post(SLACK_ACCESS_URL).mock(return_value=httpx.Response(200, json={'ok': False, 'error': 'invalid_code'}))

    response = client.get(f'/slack/oauth/callback?code=stale&state={_issued_state(client)}')

    assert response.status_code == 400
    assert response.json()['detail'] == 'invalid_code'
    assert db_session.query(SlackInstallation).count() == 0