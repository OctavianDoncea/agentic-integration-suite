from __future__ import annotations
import json
from types import SimpleNamespace as NS
import httpx
import pytest
from agentic_suite.clients.errors import ProviderConfigError, ProviderHTTPError
from agentic_suite.clients.openai_compat_client import OpenAICompatClient
from agentic_suite.middleware.retry import get_retry_after, is_retryable
from agentic_suite.tools.mock.jira_tool import JiraIssueTool

def completion(*, content=None, tool_calls=None, finish_reason='stop') -> dict:
    return {
        'choices': [
            {
                'message': {'role': 'assistant', 'content': content, 'tool_calls': tool_calls},
                'finish_reason': finish_reason,
            }
        ],
        'usage': {'prompt_tokens': 120, 'completion_tokens': 15},
    }

def wire_tool_call(name: str, arguments: str, call_id: str = 'call_1') -> dict:
    return {'id': call_id, 'type': 'function', 'function': {'name': name, 'arguments': arguments}}

def make_client(body=None, *, status=200, headers=None, **kwargs):
    """A client whose transport returns `body` and records the requests it saw."""
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(status, json=body or completion(content="ok"), headers=headers)

    client = OpenAICompatClient(
        'some-model',
        base_url='https://example.test/v1',
        api_key=kwargs.pop('api_key', 'test-key'),
        transport=httpx.MockTransport(handler),
        **kwargs,
    )
    return client, seen


# Complete
async def test_sends_a_deterministic_request_to_the_chat_endpoint():
    client, seen = make_client()
    await client.complete([{'role': 'user', 'content': 'hi'}], [JiraIssueTool.get_schema()])

    request = seen[0]
    sent = json.loads(request.content)
    assert request.url.path == '/v1/chat/completions'
    assert request.headers['authorization'] == 'Bearer test-key'
    assert sent['model'] == 'some-model'
    assert sent['temperature'] == 0
    # Full schema by default: constraints stay as schema keywords.
    title = sent['tools'][0]['function']['parameters']['properties']['title']
    assert 'maxLength' in title

async def test_schema_is_adapted_only_when_enabled():
    client, seen = make_client(adapt_tool_schemas=True)
    await client.complete([{'role': 'user', 'content': 'hi'}], [JiraIssueTool.get_schema()])

    title = json.loads(seen[0].content)['tools'][0]['function']['parameters']['properties']['title']
    assert 'maxLength' not in title
    assert 'max length 255' in title['description']

async def test_no_authorization_header_without_a_key():
    client, seen = make_client(api_key=None)
    await client.complete([{'role': 'user', 'content': 'hi'}])
    assert 'authorization' not in seen[0].headers

async def test_parses_text_and_usage():
    client, _ = make_client(completion(content='Which project?'))
    result = await client.complete([{'role': 'user', 'content': 'file a ticket'}])

    assert result.text == 'Which project?'
    assert result.tool_calls == []
    assert result.finish_reason == 'stop'
    assert (result.prompt_tokens, result.completion_tokens) == (120, 15)

async def test_parses_tool_calls():
    body = completion(
        tool_calls=[wire_tool_call('jira_issue_tool', '{"title": "x", "priority": "High"}')],
        finish_reason='tool_calls',
    )
    client, _ = make_client(body)
    result = await client.complete([{'role': 'user', 'content': 'file it'}])

    call = result.first_tool_call
    assert call.name == 'jira_issue_tool'
    assert call.arguments == {'title': 'x', 'priority': 'High'}
    assert call.id == 'call_1'
    assert result.text is None

async def test_malformed_arguments_are_classified_not_raised():
    body = completion(tool_calls=[wire_tool_call('jira_issue_tool', '{bad json')])
    client, _ = make_client(body)
    result = await client.complete([{'role': 'user', 'content': 'x'}])
    assert not result.first_tool_call.parsed_ok


# Errors
async def test_http_errors_carry_status_and_retry_after():
    client, _ = make_client({'error': 'slow down'}, status=429, headers={'Retry-After': '2'})

    with pytest.raises(ProviderHTTPError) as caught:
        await client.complete([{'role': 'user', 'content': 'x'}])

    assert caught.value.status_code == 429
    assert is_retryable(caught.value) is True
    assert get_retry_after(caught.value) == 2.0

async def test_client_errors_are_not_retryable():
    client, _ = make_client({'error': 'bad request'}, status=400)

    with pytest.raises(ProviderHTTPError) as caught:
        await client.complete([{'role': 'user', 'content': 'x'}])

    assert is_retryable(caught.value) is False

async def test_connection_failures_become_connection_errors():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError('refused')

    client = OpenAICompatClient(
        'some-model', base_url='https://example.test/v1', transport=httpx.MockTransport(handler)
    )
    with pytest.raises(ConnectionError):
        await client.complete([{'role': 'user', 'content': 'x'}])

def test_missing_base_url_fails_with_a_clear_error():
    with pytest.raises(ProviderConfigError, match='OPENAI_COMPAT_BASE_URL'):
        OpenAICompatClient('some-model')


# Conversation turns
async def test_assistant_turn_uses_the_shared_openai_format():
    body = completion(tool_calls=[wire_tool_call('jira_issue_tool', '{"title": "x"}')])
    client, _ = make_client(body)
    result = await client.complete([{'role': 'user', 'content': 'x'}])

    assert client.format_assistant_turn(result) == {
        'role': 'assistant',
        'content': '',
        'tool_calls': [
            {
                'id': 'call_1',
                'type': 'function',
                'function': {'name': 'jira_issue_tool', 'arguments': '{"title": "x"}'},
            }
        ],
    }

def test_tool_results_are_plain_tool_messages():
    client, _ = make_client()
    assert client.format_tool_result(NS(id='call_1'), '{"result": 1}') == {
        'role': 'tool',
        'tool_call_id': 'call_1',
        'content': '{"result": 1}',
    }