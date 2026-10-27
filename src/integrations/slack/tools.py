from __future__ import annotations
import asyncio
from collections.abc import Callable, Mapping
from typing import Any
from pydantic import Field, field_validator
from slack_sdk.errors import SlackApiError
from slack_sdk.web.async_client import AsyncWebClient
from agentic_suite.integrations.slack.oauth import REAUTH_REQUIRED_ERRORS, InstallationNotFoundError, ReauthRequiredError, get_valid_token, mark_needs_reauth
from agentic_suite.sdk.base import BaseTool

SlackClientFactory = Callable[[str], Any]

def default_slack_client(token: str) -> AsyncWebClient:
    return AsyncWebClient(token=token, retry_handlers=[])

class MissingRuntimeContextError(RuntimeError):
    """The caller didn't supply the context this tool needs (a bug, not a user's error)."""


class SlackRateLimitedError(RuntimeError):
    """Slack answered 429."""

    def __init__(self, retry_after: str | None) -> None:
        self.status = 429
        self.headers = {'Retry-After': retry_after} if retry_after else {}
        super().__init__(f'Slack rate limited the request (Retry-After: {retry_after})')


class SlackDeliveryError(RuntimeError):
    """A terminal posting failure."""

    def __init__(self, slack_error: str, *, http_status: int | None = None) -> None:
        self.slack_error = slack_error
        self.http_status = http_status
        super().__init__(f'Slack post failed: {slack_error}')


def _header(headers: Mapping[str, Any] | None, name: str) -> str | None:
    for key, value in (headers or {}).items():
        if key.lower() == name.lower():
            return str(value)

    return None

class SlackPostMessageTool(BaseTool):
    """Post a message to a Slack channel."""
    
    tool_name = 'slack_message_tool'
    display_title = 'Post to Slack'
    destructive = False
    idempotent = False

    channel: str = Field(description="Target channel, e.g. '#general'.")
    message: str = Field(min_length=1, max_length=4000, description='Message text.')

    @field_validator('channel')
    @classmethod
    def _channel_must_start_with_hash(cls, value: str) -> str:
        if not value.startswith('#'):
            raise ValueError('Channel must start with a #.')
        
        return value

    async def execute(self, **kwargs: Any) -> dict[str, Any]:
        session = kwargs.get('session')

        if session is None:
            raise MissingRuntimeContextError('SlackPostMessageTool needs a database session.')

        team_id = kwargs.get('team_id')
        if not team_id:
            raise InstallationNotFoundError('(none)')

        factory: SlackClientFactory = kwargs.get('slack_client_factory') or default_slack_client
        token = get_valid_token(session, team_id)
        client = factory(token)

        try:
            response = await client.chat_postMessage(channel=self.channel, text=self.message)
        except SlackApiError as exc:
            status = getattr(exc.response, 'status_code', None)
            error = exc.response.get('error') if exc.response is not None else None

            if status == 429 or error == 'ratelimited':
                retry_after = _header(getattr(exc.response, 'headers', None), 'Retry-After')
                raise SlackRateLimitedError(retry_after) from exc

            if error in REAUTH_REQUIRED_ERRORS:
                mark_needs_reauth(session, team_id)
                raise ReauthRequiredError(team_id, error) from exc

            raise SlackDeliveryError(error or 'unknown_error', http_status=status) from exc
        except asyncio.TimeoutError as exc:
            raise SlackDeliveryError('delivery_unknown_after_timeout') from exc

        return {'ok': True, 'channel': response.get('channel'), 'ts': response.get('ts')}