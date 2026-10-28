from __future__ import annotations
import json
import logging
import pytest
from typing import Any, ClassVar
from pydantic import ValidationError
from agentic_suite.agents.executor import ToolExecutor, tool_error_payload
from agentic_suite.middleware.circuit_breaker import CircuitOpenError
from agentic_suite.middleware.retry import RetryExhaustedError
from agentic_suite.sdk.base import BaseTool
from agentic_suite.sdk.registry import ToolNotRegisteredError, ToolRegistry
from agentic_suite.tools.mock.jira_tool import JiraIssueTool
from tests.fakes import StatusError, no_sleep

TELEMETRY = 'telemetry.tool_invocation'
JIRA_ARGS = {'title': 'Fix login', 'priority': 'High', 'project_key': 'ENG'}

class FlakyTool(BaseTool):
    """A tool that fails a set number of times, then succeeds."""

    failures_left: ClassVar[int] = 0
    value: str

    async def execute(self, **kwargs: Any) -> dict[str, Any]:
        if FlakyTool.failures_left > 0:
            FlakyTool.failures_left -= 1
            raise StatusError(503)

        return {'value': self.value}


class AlwaysFailsTool(BaseTool):
    """A tool whose dependency is down."""

    calls: ClassVar[int] = 0
    value: str

    async def execute(self, **kwargs: Any) -> dict[str, Any]:
        AlwaysFailsTool.calls += 1
        raise StatusError(503)


@pytest.fixture(autouse=True)
def reset_counters():
    FlakyTool.failures_left = 0
    AlwaysFailsTool.calls = 0

def make_executor(*tools: type[BaseTool], **kwargs) -> ToolExecutor:
    registry = ToolRegistry()

    for tool in tools:
        registry.register(tool)

    return ToolExecutor(registry, sleep=no_sleep, **kwargs)

def telemetry_records(caplog) -> list[dict]:
    return [json.loads(r.message) for r in caplog.records if r.name == TELEMETRY]

async def test_success_returns_the_result_and_logs_once(caplog):
    executor = make_executor(JiraIssueTool)

    with caplog.at_level(logging.INFO, logger=TELEMETRY):
        result = await executor.execute('jira_issue_tool', JIRA_ARGS)

    assert result['status'] == 'created'
    records = telemetry_records(caplog)
    assert [r['status'] for r in records] == ['success']
    assert records[0]['retry_count'] == 0

async def test_transient_failures_are_retried_and_counted(caplog):
    FlakyTool.failures_left = 2
    executor = make_executor(FlakyTool)

    with caplog.at_level(logging.INFO, logger=TELEMETRY):
        result = await executor.execute('flaky_tool', {'value': 'x'})

    assert result == {'value': 'x'}
    records = telemetry_records(caplog)
    assert [r['status'] for r in records] == ['tool_error', 'tool_error', 'success']
    assert [r['retry_count'] for r in records] == [0, 1, 2]

async def test_validation_errors_are_not_retried(caplog):
    executor = make_executor(JiraIssueTool)

    with caplog.at_level(logging.INFO, logger=TELEMETRY):
        with pytest.raises(ValidationError):
            await executor.execute('jira_issue_tool', {'title': 'only this'})

    assert [r['status'] for r in telemetry_records(caplog)] == ['validation_error']
    assert executor.breaker_for('jira_issue_tool').failure_count == 0

async def test_unknown_tools_raise_without_creating_a_breaker():
    executor = make_executor(JiraIssueTool)

    with pytest.raises(ToolNotRegisteredError):
        await executor.execute('hallucinated_tool', {})

    assert executor.breaker_names == []

async def test_open_circuit_sheds_calls_and_logs_them(caplog):
    executor = make_executor(AlwaysFailsTool, max_attempts=1, failure_threshold=3)

    for _ in range(3):
        with pytest.raises(RetryExhaustedError):
            await executor.execute('always_fails_tool', {'value': 'x'})

    assert AlwaysFailsTool.calls == 3

    with caplog.at_level(logging.INFO, logger=TELEMETRY):
        with pytest.raises(CircuitOpenError):
            await executor.execute('always_fails_tool', {'value': 'x'})

    assert AlwaysFailsTool.calls == 3
    assert telemetry_records(caplog)[-1]['status'] == 'circuit_open'

async def test_breakers_are_per_tool():
    executor = make_executor(AlwaysFailsTool, JiraIssueTool, max_attempts=1, failure_threshold=1)
    with pytest.raises(RetryExhaustedError):
        await executor.execute('always_fails_tool', {'value': 'x'})

    assert (await executor.execute('jira_issue_tool', JIRA_ARGS))['status'] == 'created'

async def test_base_context_reaches_the_tool():
    class ContextTool(BaseTool):
        """Echo a context value."""

        value: str

        async def execute(self, **kwargs: Any) -> dict[str, Any]:
            return {'seen': kwargs.get('marker')}

    executor = make_executor(ContextTool, base_context={"marker": "from-base"})
    assert await executor.execute('context_tool', {'value': 'x'}) == {'seen': 'from-base'}

# Error payloads

def test_validation_payload_names_the_fields():
    with pytest.raises(ValidationError) as exc_info:
        JiraIssueTool.validate_arguments({'title': 'x'})

    payload = tool_error_payload(exc_info.value)
    assert payload['error'] == 'invalid_arguments'
    assert {d['field'] for d in payload['details']} == {'priority', 'project_key'}

def test_circuit_open_payload():
    payload = tool_error_payload(CircuitOpenError('jira', 12.34))
    assert payload == {
        'error': 'service_unavailable',
        'retry_after_seconds': 12.3,
        'hint': "The service is failing. Tell the user; don't retry right now.",
    }

def test_unknown_exceptions_leak_no_message():
    payload = tool_error_payload(RuntimeError('password=hunter2'))
    assert payload == {'error': 'tool_failed', 'type': 'RuntimeError'}