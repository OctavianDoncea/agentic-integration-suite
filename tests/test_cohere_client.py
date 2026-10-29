from __future__ import annotations
from types import SimpleNamespace as NS
import pytest
from agentic_suite.clients.cohere_client import CohereClient
from agentic_suite.tools.mock.jira_tool import JiraIssueTool

class FakeCohereChat:
    """Replaces cohere.AsyncClientV2; records the kwargs of every chat call."""

    def __init__(self, response) -> None:
        self.response = response
        self.calls: list[dict] = []

    async def chat(self, **kwargs):
        self.calls.append(kwargs)
        return self.response

def cohere_response(*, text=None, tool_calls=None, thinking=None):
    content = []
    if thinking:
        content.append(NS(type='thinking', thinking=thinking))
    if text:
        content.append(NS(type='text', text=text))
    message = NS(
        role='assistant',
        content=content or None,
        tool_calls=tool_calls,
        tool_plan='I will call the tool.' if tool_calls else None,
    )
    return NS(
        message=message,
        finish_reason='TOOL_CALL' if tool_calls else 'COMPLETE',
        usage=NS(tokens=NS(input_tokens=120, output_tokens=15)),
    )

def cohere_tool_call(name: str, arguments: str, call_id: str = 'tool_1'):
    return NS(id=call_id, type='function', function=NS(name=name, arguments=arguments))

def make_client(response, **kwargs) -> tuple[CohereClient, FakeCohereChat]:
    client = CohereClient('command-a-03-2025', api_key='test-key', **kwargs)
    fake = FakeCohereChat(response)
    client._client = fake
    return client, fake

# Complete
async def test_sends_deterministic_request_with_adapted_tools():
    client, fake = make_client(cohere_response(text='Hi'))
    await client.complete([{'role': 'user', 'content': 'hi'}], [JiraIssueTool.get_schema()])

    sent = fake.calls[0]
    assert sent['model'] == 'command-a-03-2025'
    assert sent['temperature'] == 0
    assert 'thinking' not in sent
    title = sent['tools'][0]['function']['parameters']['properties']['title']
    assert 'maxLength' not in title

async def test_parses_tool_calls():
    response = cohere_response(
        tool_calls=[cohere_tool_call('jira_issue_tool', '{"title": "x", "priority": "High"}')]
    )
    client, _ = make_client(response)
    result = await client.complete([{'role': 'user', 'content': 'file it'}])

    call = result.first_tool_call
    assert call.name == 'jira_issue_tool'
    assert call.arguments == {'title': 'x', 'priority': 'High'}
    assert call.id == 'tool_1'
    assert result.text is None
    assert (result.prompt_tokens, result.completion_tokens) == (120, 15)
    assert result.raw_message is response.message

async def test_thinking_blocks_are_not_response_text():
    client, _ = make_client(cohere_response(text='Which project?', thinking='The user...'))
    result = await client.complete([{'role': 'user', 'content': 'file a ticket'}])
    assert result.text == 'Which project?'

async def test_malformed_arguments_are_classified_not_raised():
    response = cohere_response(tool_calls=[cohere_tool_call('jira_issue_tool', '{bad json')])
    client, _ = make_client(response)
    result = await client.complete([{'role': 'user', 'content': 'x'}])
    assert not result.first_tool_call.parsed_ok

async def test_thinking_is_sent_only_when_configured():
    client, fake = make_client(cohere_response(text='ok'), thinking='disabled')
    await client.complete([{'role': 'user', 'content': 'x'}])
    assert fake.calls[0]['thinking'] == {'type': 'disabled'}

def test_invalid_thinking_mode_is_rejected():
    with pytest.raises(ValueError):
        CohereClient('command-a-plus-05-2026', api_key='k', thinking='maybe')

# Conversation turns
async def test_assistant_turn_echoes_the_raw_message():
    response = cohere_response(tool_calls=[cohere_tool_call('jira_issue_tool', '{}')])
    client, _ = make_client(response)
    result = await client.complete([{'role': 'user', 'content': 'x'}])
    assert client.format_assistant_turn(result) is response.message

def test_tool_results_are_document_blocks():
    client, _ = make_client(cohere_response(text='x'))
    call = NS(id='tool_1')
    assert client.format_tool_result(call, '{"result": 1}') == {
        'role': 'tool',
        'tool_call_id': 'tool_1',
        'content': [{'type': 'document', 'document': {'data': '{"result": 1}'}}],
    }