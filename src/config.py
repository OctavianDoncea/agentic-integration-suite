from __future__ import annotations
from functools import lru_cache
from typing import Literal
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file='.env', env_file_encoding='utf-8', case_sensitive=False, extra='ignore')

    groq_api_key: str = Field(min_length=1)
    slack_client_id: str = Field(min_length=1)
    slack_client_secret: str = Field(min_length=1)
    slack_signing_secret: str = Field(min_length=1)
    database_url: str = Field(min_length=1)

    groq_model_smoke: str = 'llama-3.1-8b-instant'
    groq_model_full: str = 'openai/gpt-oss-120b'
    app_base_url: str = 'http://localhost:8000'
    slack_bot_scopes: str = 'chat:write,channels:read'
    environment: Literal['development', 'ci', 'production'] = 'development'
    log_level: str = 'INFO'
    sql_echo: bool = False

    @field_validator('app_base_url')
    def _strip_trailing_slash(cls, value: str) -> str:
        return value.rstrip('/')

    @property
    def slack_redirects_uri(self) -> str:
        return f'{self.app_base_url}/slack/oauth/callback'

    @property
    def scope_list(self) -> list[str]:
        return [s.strip() for s in self.slack_bot_scopes.split(',') if s.strip()]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()