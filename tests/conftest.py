from __future__ import annotations
import pytest
from cryptography.fernet import Fernet
from collections.abc import Iterator
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool
from agentic_suite.config import get_settings
from tests.fakes import TEST_BOT_TOKEN, TEST_TEAM_ID

FAKE_ENV = {
    "GROQ_API_KEY": "test-groq-key",
    "SLACK_CLIENT_ID": "1234567890.0987654321",
    "SLACK_CLIENT_SECRET": "test-client-secret",
    "SLACK_SIGNING_SECRET": "test-signing-secret",
    "SLACK_TOKEN_ENCRYPTION_KEY": Fernet.generate_key().decode(),
    "DATABASE_URL": "postgresql+psycopg2://user:pass@localhost:5432/testdb",
    "APP_BASE_URL": "http://testserver",
    "ENVIRONMENT": "ci",
    "AGENT_API_KEY": "",
    "AGENT_MODEL": "",
    'DEFAULT_SLACK_TEAM_ID': '',
    'COHERE_API_KEY': '',
    'OPENAI_COMPAT_BASE_URL': '',
    'OPENAI_COMPAT_API_KEY': '',
    'OPENAI_COMPAT_ADAPT_SCHEMA': 'false',
}

@pytest.fixture(autouse=True)
def fake_environment(monkeypatch: pytest.MonkeyPatch):
    for key, value in FAKE_ENV.items():
        monkeypatch.setenv(key, value)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()

@pytest.fixture
def slack_engine():
    from agentic_suite.db import Base
    from agentic_suite.integrations.slack import models
    from agentic_suite.integrations.slack.crypto import get_fernet

    get_fernet.cache_clear()
    engine = create_engine('sqlite+pysqlite://', poolclass=StaticPool, connect_args={'check_same_thread': False})
    Base.metadata.create_all(engine)
    yield engine
    Base.metadata.drop_all(engine)
    get_fernet.cache_clear()

@pytest.fixture
def slack_session_factory(slack_engine):
    from agentic_suite.integrations.slack.models import SlackInstallation

    factory = sessionmaker(bind=slack_engine, expire_on_commit=False)

    with factory() as session:
        session.add(SlackInstallation(team_id=TEST_TEAM_ID, team_name='Test workspace', access_token=TEST_BOT_TOKEN, scope='chat:write'))
        session.commit()

    return factory

@pytest.fixture
def slack_session(slack_session_factory) -> Iterator[Session]:
    session = slack_session_factory()
    yield session
    session.close()