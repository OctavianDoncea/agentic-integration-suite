from __future__ import annotations
from datetime import datetime, timedelta, timezone
from sqlalchemy import select
from agentic_suite.agents import run_store
from agentic_suite.agents.loop import AgentStep, RunTrace, Usage
from agentic_suite.agents.models import AgentRun
from agentic_suite.agents.run_store import get_run, list_runs, new_run_id, percentile, persist_run, prune_runs, run_summary,run_to_dict, stats

NOW = datetime(2026, 11, 1, 12, 0, tzinfo=timezone.utc)

def make_trace(*steps: AgentStep, duration_ms: float = 55.0) -> RunTrace:
    return RunTrace(
        steps=list(steps),
        usage=Usage(model_calls=2, prompt_tokens=250, completion_tokens=30, model_ms=40.0, duration_ms=duration_ms),
    )

SEND = AgentStep('slack_message_tool', 'ok', None, {'channel': '#eng', 'message': 'hi'}, 12.34)

def record(session, **overrides) -> str | None:
    fields = {
        'run_id': new_run_id(),
        'prompt': 'Tell #eng hi',
        'trace': make_trace(SEND),
        'status': 'completed',
        'model': 'fake-model',
        'provider': 'fake',
        'final_text': 'Done.',
        'error_type': None,
        'team_id': 'T1',
        'allow_writes': True,
        'write_tools': None,
        'store_content': True,
        'now': NOW,
    }
    fields.update(overrides)
    return persist_run(session, **fields)

# Writing
def test_a_run_round_trips(slack_session):
    run_id = record(slack_session)
    body = run_to_dict(get_run(slack_session, run_id))

    assert body['id'] == run_id and len(run_id) == 32
    assert body['status'] == 'completed'
    assert body['created_at'] == '2026-11-01T12:00:00+00:00'
    assert body['request']['prompt'] == 'Tell #eng hi'
    assert body['request']['prompt_chars'] == len('Tell #eng hi')
    assert len(body['request']['prompt_sha256']) == 64
    assert body['final_text'] == 'Done.'
    assert body['usage'] == {
        'model_calls': 2,
        'prompt_tokens': 250,
        'completion_tokens': 30,
        'model_ms': 40.0,
        'duration_ms': 55.0,
    }
    assert body['steps'] == [
        {
            'tool': 'slack_message_tool',
            'status': 'ok',
            'error': None,
            'latency_ms': 12.3,
            'arguments': {'channel': '#eng', 'message': 'hi'},
        }
    ]

def test_content_can_be_left_out_while_the_shape_is_kept(slack_session):
    run_id = record(slack_session, store_content=False)
    body = run_to_dict(get_run(slack_session, run_id))

    assert body['request']['prompt'] is None
    assert body['final_text'] is None
    assert body['request']['prompt_chars'] == len('Tell #eng hi')
    assert len(body['request']['prompt_sha256']) == 64
    step = body['steps'][0]
    assert 'arguments' not in step
    assert step['argument_keys'] == ['channel', 'message']
    assert step['tool'] == 'slack_message_tool'

def test_policy_stops_are_counted(slack_session):
    trace = make_trace(
        AgentStep('slack_message_tool', 'preview', None, {'channel': '#a'}, 0.0),
        AgentStep('slack_message_tool', 'blocked', 'target_not_in_request', {'channel': '#b'}, 0.0),
        SEND,
    )
    run_id = record(slack_session, trace=trace)
    run = get_run(slack_session, run_id)

    assert run.tool_calls == 3
    assert run.policy_stops == 2

def test_a_failed_run_keeps_its_partial_steps(slack_session):
    run_id = record(slack_session, status='failed', error_type='RuntimeError', final_text=None)
    run = get_run(slack_session, run_id)

    assert (run.status, run.error_type) == ('failed', 'RuntimeError')
    assert run.tool_calls == 1

def test_a_failed_write_returns_none_and_leaves_the_session_usable(slack_session, monkeypatch):
    with monkeypatch.context() as patch:
        patch.setattr(run_store, 'record_run', lambda *a, **k: (_ for _ in ()).throw(RuntimeError('db down')))
        assert record(slack_session) is None

    assert record(slack_session) is not None

def test_pending_session_changes_are_committed_before_the_run_is_recorded(slack_session, monkeypatch):
    from agentic_suite.integrations.slack.models import SlackInstallation

    row = slack_session.query(SlackInstallation).first()
    row.needs_reauth = True

    with monkeypatch.context() as patch:
        patch.setattr(run_store, 'record_run', lambda *a, **k: (_ for _ in ()).throw(RuntimeError('db down')))
        assert record(slack_session) is None

    slack_session.expire_all()
    assert slack_session.query(SlackInstallation).first().needs_reauth is True

# Reading
def test_runs_list_newest_first_with_filters_and_a_cursor(slack_session):
    ids = [
        record(slack_session, now=NOW + timedelta(minutes=minute), status=status)
        for minute, status in [(0, 'completed'), (1, 'failed'), (2, 'completed')]
    ]

    newest = list_runs(slack_session, limit=2)
    assert [run.id for run in newest] == [ids[2], ids[1]]

    only_failed = list_runs(slack_session, status='failed')
    assert [run.id for run in only_failed] == [ids[1]]

    older = list_runs(slack_session, before=NOW + timedelta(minutes=1))
    assert [run.id for run in older] == [ids[0]]

def test_a_naive_cursor_is_treated_as_utc(slack_session):
    ids = [record(slack_session, now=NOW + timedelta(minutes=m)) for m in range(2)]
    cursor = (NOW + timedelta(minutes=1)).replace(tzinfo=None)

    assert [run.id for run in list_runs(slack_session, before=cursor)] == [ids[0]]


def test_the_summary_previews_a_long_prompt(slack_session):
    run_id = record(slack_session, prompt='x' * 200)
    summary = run_summary(get_run(slack_session, run_id))

    assert summary['prompt_preview'] == 'x' * 80 + '...'
    assert summary['tokens'] == 280

def test_the_summary_has_no_preview_when_content_is_off(slack_session):
    run_id = record(slack_session, store_content=False)
    assert run_summary(get_run(slack_session, run_id))['prompt_preview'] is None

# Stats
def test_percentiles_are_nearest_rank():
    assert percentile([], 50) is None
    assert percentile([10, 20, 30, 40], 50) == 20
    assert percentile([10, 20, 30, 40], 95) == 40
    assert percentile([5], 95) == 5

def test_stats_summarise_the_window(slack_session):
    for index, duration in enumerate([10.0, 20.0, 30.0, 40.0]):
        record(
            slack_session,
            now=NOW - timedelta(hours=index),
            trace=make_trace(SEND, duration_ms=duration),
            model='model-a' if index < 3 else 'model-b',
            status='failed' if index == 3 else 'completed',
        )
    record(slack_session, now=NOW - timedelta(hours=48))

    summary = stats(slack_session, hours=24, now=NOW)

    assert summary['runs'] == 4
    assert summary['by_status'] == {'completed': 3, 'failed': 1}
    assert summary['prompt_tokens'] == 1000
    assert summary['duration_ms_p50'] == 20.0
    assert summary['duration_ms_p95'] == 40.0
    assert summary['by_model']['model-a']['runs'] == 3
    assert summary['truncated'] is False

def test_stats_of_an_empty_window(slack_session):
    summary = stats(slack_session, hours=1, now=NOW)

    assert summary['runs'] == 0
    assert summary['duration_ms_p50'] is None

# Maintenance
def test_prune_deletes_only_old_runs(slack_session):
    old = record(slack_session, now=NOW - timedelta(days=40))
    recent = record(slack_session, now=NOW - timedelta(days=5))

    deleted = prune_runs(slack_session, older_than_days=30, now=NOW)

    assert deleted == 1
    remaining = set(slack_session.scalars(select(AgentRun.id)))
    assert remaining == {recent}
    assert old not in remaining

def test_the_table_is_registered_with_the_metadata():
    from agentic_suite.db import Base

    assert 'agent_runs' in Base.metadata.tables