from __future__ import annotations
import pytest
from collections.abc import Iterator
from cryptography.fernet import Fernet
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.exc import IntegrityError
from agentic_suite.db import Base
from agentic_suite.integrations.slack.crypto import get_fernet
from agentic_suite.integrations.slack.models import SlackInstallation

@pytest.fixture
def db_session() -> Iterator[Session]:
    get_fernet.cache_clear()
    engine = create_engine('sqlite+pysqlite:///:memory:')
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    session = factory()

    try:
        yield session
    finally:
        session.close()
        Base.metadata.drop_all(engine)
        get_fernet.cache_clear()

def test_installation_round_trips(db_session: Session):
    db_session.add(SlackInstallation(
        team_id = 'T012ABC34',
        team_name = 'Test Workspace',
        access_token = 'xoxb-fake-token',
        scope = 'chat:write,channels:read',
        bot_user_id = 'UOKRQLJ9H'
    ))
    db_session.commit()

    row = db_session.query(SlackInstallation).filter_by(team_id='T012ABC34').one()
    assert row.access_token == 'xoxb-fake-token'
    assert row.refresh_token is None
    assert row.needs_reauth is False
    assert row.installed_at is not None

def test_token_encrypted_in_the_database(db_session: Session):
    db_session.add(SlackInstallation(team_id='T1', access_token='xoxb-secret-value', scope='chat:write,channels:read'))
    db_session.commit()

    raw = db_session.execute(text("SELECT access_token FROM slack_installations WHERE team_id = 'T1'")).scalar_one()

    assert raw != 'xoxb-secret-value'
    assert 'xoxb' not in raw
    assert get_fernet().decrypt(raw.encode()).decode() == 'xoxb-secret-value'

def test_team_id_is_unique(db_session: Session):
    db_session.add(SlackInstallation(team_id='T1', access_token='a', scope=''))
    db_session.commit()
    db_session.add(SlackInstallation(team_id='T1', access_token='b', scope=''))

    with pytest.raises(IntegrityError):
        db_session.commit()

def test_wrong_key_raises_rather_than_returning_garbage(db_session: Session, monkeypatch):
    from agentic_suite.integrations.slack.crypto import TokenDecryptionError

    db_session.add(SlackInstallation(team_id='T1', access_token='xoxb-x', scope=''))
    db_session.commit()
    db_session.expunge_all()

    monkeypatch.setenv('SLACK_TOKEN_ENCRYPTION_KEY', Fernet.generate_key().decode())
    from agentic_suite.config import get_settings

    get_settings.cache_clear()
    get_fernet.cache_clear()

    with pytest.raises(TokenDecryptionError):
        db_session.query(SlackInstallation).filter_by(team_id='T1').one()