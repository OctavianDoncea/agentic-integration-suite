import pytest
from __future__ import annotations
from pydantic_settings import ValidationError
from agentic_suite.config import Settings, get_settings

REQUIRED_VARS = ['GROQ_API_KEY', 'SLACK_CLIENT_ID', 'SLACK_CLIENT_SECRET', 'SLACK_SIGNING_SECRET', 'DATABASE_URL']

def test_settings_load_from_env():
    settings = Settings(_env_file=None)
    assert settings.groq_api_key == 'test-groq-key'
    assert settings.environment == 'ci'
    assert settings.sql_echo is False

@pytest.mark.parametrize('missing', REQUIRED_VARS)
def test_missing_required_var_raises_clear_error(monkeypatch, missing):
    monkeypatch.delenv(missing, raising=False)

    with pytest.raises(ValidationError) as exc_info:
        Settings(_env_file=None)

    message = str(exc_info.value)
    assert missing.lower() in message
    assert 'Field required' in message

def test_empty_string_is_rejected(monkeypatch):
    monkeypatch.setenv("SLACK_SIGNING_SECRET", "")
    with pytest.raises(ValidationError):
        Settings(_env_file=None)

def test_redirect_uri_is_derived_and_slash_safe(monkeypatch):
    monkeypatch.setenv("APP_BASE_URL", "https://example.onrender.com/")
    settings = Settings(_env_file=None)
    assert settings.slack_redirects_uri == 'https://example.onrender.com/slack/oauth/callback'

def test_scope_list_parsing(monkeypatch):
    monkeypatch.setenv("SLACK_BOT_SCOPES", "chat:write, channels:read")
    assert Settings(_env_file=None).scope_list == ['chat:write', 'channels:read']

def test_get_settings_is_cached():
    assert get_settings() is get_settings()