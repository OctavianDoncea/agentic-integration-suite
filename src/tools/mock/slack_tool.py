"""Thin mock Slack tool, offline unit tests only."""
from __future__ import annotations
from typing import Any
from pydantic import Field, field_validator
from agentic_suite.sdk.base import BaseTool

class SlackMessageTool(BaseTool):
    channel: str = Field(description="Target channel, e.g. '#general'.")
    message: str = Field(min_length=1, max_length=4000, description='Message text.')

    @field_validator('channel')
    @classmethod
    def _channel_must_start_with_hash(cls, value: str) -> str:
        if not value.startswith('#'):
            raise ValueError('Channel must start with a hash (#)')
        return value

    async def exceute(self, **kwargs: Any) -> dict[str, Any]:
        return {'channel': self.channel, 'ok': True, 'ts': '1234567890.000100'}