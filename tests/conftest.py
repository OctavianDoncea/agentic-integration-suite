import pytest
from __future__ import annotations
from agentic_suite.config import get_settings

FAKE_ENV = {
    'GROQ_API_KEY': 'test-groq-key',
    'SLACK_CLIENT_ID': '1234567890.0987654321',
    'SLACK_CLIENT_SECRET': 'test-client-secret',
    'SLACK_SIGNING_SECRET': 'test-signing-secret',
    'DATABASE_URL': 'postgresql+psycopg2://user:pass@localhost:5432/testdb',
    'APP_BASE_URL': 'http://testserver',
    'ENVIRONMENT': 'ci'
}

@pytest.fixture(autouse=True)
def fake_environment(monkeypatch: pytest.MonkeyPatch):
    for key, value in FAKE_ENV.items():
        monkeypatch.setenv(key, value)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()