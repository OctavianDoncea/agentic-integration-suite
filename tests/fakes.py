"""Test doubles shared by the agent, executor, endpoint, and MCP tests."""

from __future__ import annotations
import asyncio
import copy
import json
from typing import Any
from slack_sdk.errors import SlackApiError
from slack_sdk.web.slack_response import SlackResponse
from agentic_suite.clients.base import ModelResponse, ToolCall, openai_assistant_turn, openai_tool_result

TEST_TEAM_ID = 'T012AB3C4'
TEST_BOT_TOKEN = 'xoxb-test-token'

def tool_call_response(name: str, arguments: dict[str, Any], call_id: str = 'call_1') -> ModelResponse:
    return ModelResponse(tool_calls=[ToolCall(name=name, arguments=arguments, raw_arguments=json.dumps(arguments), id=call_id)])

def text_response(text: str) -> ModelResponse:
    return ModelResponse(text=text)

class FakeChatClient:
    """Replays a script of responses (or exceptions) and records every request."""

    provider = 'fake'

    def __init__(self, script: list[ModelResponse | BaseException], model: str = 'fake-model') -> None:
        self.model = model
        self._script = list(script)
        self.requests: list[list[Any]] = []

    async def complete(self, messages, tools=None, *, max_tokens: int = 1024) -> ModelResponse:
        self.requests.append(copy.deepcopy(messages))

        if not self._script:
            raise AssertionError('FakeChatClient ran out of scripted responses')

        item = self._script.pop(0)
        if isinstance(item, BaseException):
            raise item

        return item

    def format_assistant_turn(self, response: ModelResponse) -> dict[str, Any]:
        return openai_assistant_turn(response)

    def format_tool_result(self, call: ToolCall, content: str) -> dict[str, Any]:
        return openai_tool_result(call, content)


class FakeSlackClient:
    """Stands in for slack_sdk's AsyncWebClient."""

    def __init__(self, *, error: str | None = None, status_code: int = 200, headers: dict[str, str] | None = None, timeout: bool = False, fail_times: int | None = None) -> None:
        self._error = error
        self._status_code = status_code
        self._headers = headers or {}
        self._timeout = timeout
        self._fail_times = fail_times
        self.posts: list[tuple[str, str]] = []

    async def chat_postMessage(self, *, channel: str, text: str) -> dict[str, Any]:
        self.posts.append((channel, text))
        failing = self._fail_times is None or len(self.posts) <= self._fail_times

        if self._timeout and failing:
            raise asyncio.TimeoutError()

        if self._error and failing:
            response = SlackResponse(
                client = None,
                http_verb = 'POST',
                api_url = 'https://slack.com/api/chat.postMessage',
                req_args = {},
                data = {'ok': False, 'error': self._error},
                headers = self._headers,
                status_code = self._status_code
            )
            raise SlackApiError(f'Slack error: {self._error}', response)

        return {'ok': True, 'channel': 'C0123456789', 'ts': '1700000000.000100'}


class FakeSlackFactory:
    """Records which token each client was built with."""

    def __init__(self, client: FakeSlackClient | None = None) -> None:
        self.client = client or FakeSlackClient()
        self.tokens: list[str] = []

    def __call__(self, token: str) -> FakeSlackClient:
        self.tokens.append(token)

        return self.client

    
class StatusError(Exception):
    """A generic HTTP-shaped error for retry and breaker tests."""

    def __init__(self, status: int, retry_after: str | None = None) -> None:
        self.status = status
        self.headers = {'Retry-After': retry_after} if retry_after else {}
        super().__init__(f'HTTP {status}')


async def no_sleep(_: float) -> None:
    return None