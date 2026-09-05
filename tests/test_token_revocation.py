from __future__ import annotations
import pytest
from collections.abc import Iterator
from datetime import datetime, timedelta, timezone
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from agentic_suite.db import Base
from agentic_suite.integrations.slack.models import SlackInstallation
from agentic_suite.integrations.slack.oauth import InstallationNotFoundError, ReauthRequiredError, SlackOAuthError, get_valid_token, raise_for_slack_error

@pytest.fixture
def db_session() -> Iterator[Session]:
    engine = create_engine('sqlite+pysqlite:///:memory:')
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine, expire_on_commit=False)()
    session.add(SlackInstallation(team_id='T012AB3C4', access_token='xoxb-token', scope='chat:write'))
    session.commit()

    try:
        yield session
    finally:
        session.close()

def test_get_valid_token_returns_the_decrypted_token(db_session: Session):
    assert get_valid_token(db_session, 'T012AB3C4') == 'xoxb-token'

def test_unknown_team_raises_a_specific_error(db_session: Session):
    with pytest.raises(InstallationNotFoundError):
        get_valid_token(db_session, 'T-nonexistent')

@pytest.mark.parametrize('slack_error', ['token_revoked', 'invalid_auth', 'account_inactive', 'not_authed'])
def test_auth_errors_flag_the_row_and_raise(db_session: Session, slack_error: str):
    with pytest.raises(ReauthRequiredError) as exc_info:
        raise_for_slack_error(db_session, 'T012AB3C4', {'ok': False, 'error': slack_error})

    assert exc_info.value.slack_error == slack_error
    row = db_session.query(SlackInstallation).filter_by(team_id='T012AB3C4').one()
    assert row.needs_reauth is True

def test_non_auth_errors_do_not_flag_the_row(db_session: Session):
    with pytest.raises(SlackOAuthError) as exc_info:
        raise_for_slack_error(db_session, 'T012AB3C4', {'ok': False, 'error': 'ratelimited'})

    assert not isinstance(exc_info.value, ReauthRequiredError)
    row = db_session.query(SlackInstallation).filter_by(team_id='T012AB3C4').one()
    assert row.needs_reauth is False

def test_successful_response_is_a_no_op(db_session: Session):
    raise_for_slack_error(db_session, 'T012AB3C4', {'ok': True, 'ts': 1.0})
    row = db_session.query(SlackInstallation).filter_by(team_id='T012AB3C4').one()
    assert row.needs_reauth is False

def test_flagged_installation_refuses_to_hand_out_its_token(db_session: Session):
    row = db_session.query(SlackInstallation).filter_by(team_id='T012AB3C4').one()
    row.needs_reauth = True
    db_session.commit()

    with pytest.raises(ReauthRequiredError):
        get_valid_token(db_session, 'T012AB3C4')

def test_expired_rotating_token_flags_and_raises(db_session: Session):
    row = db_session.query(SlackInstallation).filter_by(team_id='T012AB3C4').one()
    row.token_expires_at = datetime.now(timezone.utc) - timedelta(minutes=5)
    db_session.commit()

    with pytest.raises(ReauthRequiredError) as exc_info:
        get_valid_token(db_session, 'T012AB3C4')

    assert exc_info.value.slack_error == 'token_expired'
    db_session.refresh(row)
    assert row.needs_reauth is True