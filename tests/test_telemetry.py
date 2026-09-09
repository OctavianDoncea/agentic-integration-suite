from __future__ import annotations
import asyncio
import json
import logging
import pytest
from collections.abc import Iterator
from agentic_suite.middleware.circuit_breaker import CircuitBreaker, CircuitOpenError
from agentic_suite.middleware.telemetry import log_invocation, redact_arguments, timed
from agentic_suite.sdk.registry import ToolNotRegisteredError, ToolRegistry
from agentic_suite.tools.mock.github_tool import GitHubPRTool
from agentic_suite.tools.mock.jira_tool import JiraIssueTool, JiraServerError

TELEMETRY_LOGGER = 'telemetry.tool_invocation'

@pytest.fixture
def registry() -> ToolRegistry:
    reg = ToolRegistry()
    reg.register(JiraIssueTool)
    reg.register(GitHubPRTool)
    return reg

VALID_ARGS = {'title': 'Fix login bug', 'priority': 'High', 'project_key': 'ENG'}

def test_success_record_shape(caplog):
    with caplog.at_level(logging.INFO, logger=TELEMETRY_LOGGER):
        log_invocation('jira_issue_tool', VALID_ARGS, 12.345, 'success', retry_count=2)

    record = json.loads(caplog.records[0].message)

    assert record['event'] == 'tool_invocation'
    assert record['tool_name'] == 'jira_issue_tool'
    assert record['status'] == 'success'
    assert record['retry_count'] == 2
    assert isinstance(record['timestamp'], float)
    assert isinstance(record['latency_ms'], float)
    assert isinstance(record['arguments'], dict)

def test_latency_is_rounded(caplog):
    with caplog.at_level(logging.INFO, logger=TELEMETRY_LOGGER):
        log_invocation('t', {}, 12.3456789, 'success')
    assert json.loads(caplog.records[0].message)['latency_ms'] == 12.35

def test_error_fields_appear_only_on_failure(caplog):
    with caplog.at_level(logging.INFO, logger=TELEMETRY_LOGGER):
        log_invocation('t', {}, 1.0, 'success')
        log_invocation('t', {}, 1.0, 'tool_error', error_type='JiraServerError', error_message='boom')

    success, failure = (json.loads(r.message) for r in caplog.records)

    assert 'error_type' not in success
    assert failure['error_type'] == 'JiraServerError'
    assert failure['error_message'] == 'boom'

def test_long_error_message_are_truncated(caplog):
    with caplog.at_level(logging.INFO, logger=TELEMETRY_LOGGER):
        log_invocation('t', {}, 1.0, 'tool_error', error_message='x' * 2000)
    assert len(json.loads(caplog.records[0].message)['error_message']) == 500

def test_failures_log_at_warning_successes_at_info(caplog):
    with caplog.at_level(logging.INFO, logger=TELEMETRY_LOGGER):
        log_invocation('t', {}, 1.0, 'success')
        log_invocation('t', {}, 1.0, 'tool_error')

    assert caplog.records[0].levelno == logging.INFO
    assert caplog.records[1].levelno == logging.WARNING

def test_output_is_exactly_one_line_of_valid_json(caplog):
    with caplog.at_level(logging.INFO, logger=TELEMETRY_LOGGER):
        log_invocation('t', VALID_ARGS, 1.0, 'success')

    message = caplog.records[0].message
    assert '\n' not in message
    json.loads(message)

def test_non_serializable_extras_do_not_crash_the_logger(caplog):
    class Opaque:
        pass


    with caplog.at_level(logging.INFO, logger=TELEMETRY_LOGGER):
        log_invocation('t', {}, 1.0, 'success', session=Opaque())

    assert 'Opaque' in json.loads(caplog.records[0].message)['session']

def test_short_scalars_are_kept_verbatim():
    summary = redact_arguments({'priority': 'High', 'pr_id': 42, 'flag': True})
    assert summary == {'priority': "'High'", 'pr_id': '42', 'flag': 'True'}

def test_long_strings_are_reduced_to_type_and_length():
    summary = redact_arguments({'message': 'x' * 500})
    assert summary['message'] == 'str(len=500)'
    assert 'xxx' not in json.dumps(summary)

def test_sensitive_keys_are_never_recorded():
    summary = redact_arguments({'access_token': 'xoxb-real', 'api-key': 'sk-abc'})
    assert summary == {'access_token': '<redacted>', 'api-key': '<redacted>'}

def test_collections_are_summarized_by_length():
    assert redact_arguments({'labels': ['a', 'b', 'c']})['labels'] == 'list(len=3)'

def test_none_and_nested_objects():
    summary = redact_arguments({'optional': None, 'nested': {'a': 1}})
    assert summary == {'optional': 'None', 'nested': 'dict'}

def test_empty_and_missing_arguments():
    assert redact_arguments(None) == {}
    assert redact_arguments({}) == {}

async def test_timed_reports_a_positive_elapsed_value():
    import asyncio

    with timed() as elapsed:
        await asyncio.sleep(0.01)

    assert elapsed['elapsed_ms'] >= 10.0

def test_timed_populates_on_the_exception_path():
    captured_dict = None
    with pytest.raises(ValueError):
        with timed() as elapsed:
            captured_dict = elapsed
            raise ValueError('boom')

    assert captured_dict['elapsed_ms'] > 0

async def test_every_successful_call_logs_exactly_one_line(registry, caplog):
    with caplog.at_level(logging.INFO, logger=TELEMETRY_LOGGER):
        await registry.execute('jira_issue_tool', VALID_ARGS)

    assert len(caplog.records) == 1
    record = json.loads(caplog.records[0].message)
    assert record['status'] == 'success'
    assert record['tool_name'] == 'jira_issue_tool'
    assert record['latency_ms'] > 0

async def test_tool_error_is_logged_and_still_raised(registry, caplog):
    with caplog.at_level(logging.INFO, logger=TELEMETRY_LOGGER):
        with pytest.raises(JiraServerError):
            await registry.execute('jira_issue_tool', {**VALID_ARGS, 'inject_500_error': True})

    record = json.loads(caplog.records[0].message)
    assert record['status'] == 'tool_error'
    assert record['error_type'] == 'JiraServerError'

async def test_validation_error_is_logged(registry, caplog):
    with caplog.at_level(logging.INFO, logger=TELEMETRY_LOGGER):
        with pytest.raises(Exception):
            await registry.execute('jira_issue_tool', {'title': 'only this'})

    record = json.loads(caplog.records[0].message)
    assert record['status'] == 'validation_error'
    assert 'validation_error' in record['error_message']

async def test_unregistered_tool_is_logged(registry, caplog):
    with caplog.at_level(logging.INFO, logger=TELEMETRY_LOGGER):
        with pytest.raises(ToolNotRegisteredError):
            await registry.execute('not_a_tool', {})

    record = json.loads(caplog.records[0].message)
    assert record['status'] == 'not_registered'
    assert record['tool_name'] == 'not_a_tool'

async def test_circuit_open_gets_its_own_status(registry, caplog):
    breaker = CircuitBreaker(name='jira', failure_threshold=1)

    async def guarded():
        return await breaker.call(registry.execute, 'jira_issue_tool', {**VALID_ARGS, 'inject_500_error': True})

    with pytest.raises(JiraServerError):
        await guarded()

    caplog.clear()
    with caplog.at_level(logging.INFO, logger=TELEMETRY_LOGGER):
        with pytest.raises(CircuitOpenError):
            await breaker.call(registry.execute, 'jira_issue_tool', VALID_ARGS)

    assert caplog.records == []

async def test_retry_count_is_recorded(registry, caplog):
    with caplog.at_level(logging.INFO, logger=TELEMETRY_LOGGER):
        await registry.execute('jira_issue_tool', VALID_ARGS, retry_count=2)

    assert json.loads(caplog.records[0].message)['retry_count'] == 2

async def test_runtime_context_is_not_logged_as_an_argument(registry, caplog):
    with caplog.at_level(logging.INFO, logger=TELEMETRY_LOGGER):
        await registry.execute('jira_issue_tool', VALID_ARGS, db_session='sensitive-session-handle')

    assert 'sensitive-session-handle' not in caplog.records[0].message

async def test_real_slack_message_text_is_not_logged(registry, caplog):
    from agentic_suite.tools.mock.slack_tool import SlackMessageTool

    registry.register(SlackMessageTool)
    secret_text = 'the acquisition closes on Tuesday and the price is 40 million'

    with caplog.at_level(logging.INFO, logger=TELEMETRY_LOGGER):
        await registry.execute('slack_message_tool', {'channel': '#general', 'message': secret_text})

    line = caplog.records[0].message
    assert 'acquisition' not in line
    assert 'str(len=' in line