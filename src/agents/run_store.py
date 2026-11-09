"""Writing, reading, summarising, and pruning agent runs."""
from __future__ import annotations
import hashlib
import argparse
import logging
import math
import uuid
from collections import Counter
from datetime import datetime, timezone, timedelta
from typing import Any
from sqlalchemy import delete, select
from sqlalchemy.orm import Session
from agentic_suite.agents.loop import AgentStep, RunTrace
from agentic_suite.agents.models import AgentRun

logger = logging.getLogger(__name__)

PREVIEW_CHARS = 80
STATS_MAX_ROWS = 5000

def _utcnow() -> datetime:
    return datetime.now(timezone.utc)

def _aware(value: datetime) -> datetime:
    return value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)

def _iso(value: datetime | None) -> str | None:
    return _aware(value).isoformat() if value is not None else None

def new_run_id() -> str:
    return uuid.uuid4().hex

# Writing
def step_record(step: AgentStep, store_content: bool) -> dict[str, Any]:
    record: dict[str, Any] = {
        'tool': step.tool,
        'status': step.status,
        'error': step.error,
        'latency_ms': round(step.latency_ms, 1) if step.latency_ms is not None else None
    }

    if store_content:
        record['arguments'] = step.arguments
    else:
        record['arguments'] = sorted(step.arguments or [])

    return record

def record_run(session: Session, *, run_id: str, prompt: str, trace: RunTrace, status: str, model: str, provider: str | None, final_text: str | None,
    error_type: str | None, team_id: str | None, allow_writes: bool, write_tools: list[str] | None, store_content: bool, now: datetime | None = None
) -> AgentRun:
    """Add a run to the session. The caller commits."""
    usage = trace.usage
    steps = [step_record(step, store_content) for step in trace.steps]
    run = AgentRun(id=run_id, created_at=now or _utcnow(), status=status, model=model, provider=provider, team_id=team_id, allow_writes=allow_writes, 
        write_tools=sorted(write_tools) if write_tools is not None else None, prompt_sha256=hashlib.sha256(prompt.encode('utf-8')).hexdigest(),
        prompt_chars=len(prompt), prompt=prompt if store_content else None, final_text=final_text if store_content else None, error_type=error_type,
        steps=steps, tool_calls=len(steps), policy_stops=sum(1 for step in trace.steps if step.status in ('preview', 'blocked')),
        model_calls=usage.model_calls, prompt_tokens=usage.prompt_tokens, completion_tokens=usage.completion_tokens, model_ms=usage.model_ms, duration_ms=usage.duration_ms
    )
    session.add(run)
    return run

def persist_run(session: Session, **fields: Any) -> str | None:
    """Record a run in its own transaction. returns the id, or None on failure."""

    try:
        session.commit()
        record_run(session, **fields)
        session.commit()
        return fields['run_id']
    except Exception:
        logger.exception('Could not persist agent run')
        session.rollback()
        return None

# Reading
def get_run(session: Session, run_id: str) -> AgentRun | None:
    return session.get(AgentRun, run_id)

def list_runs(session: Session, *, limit: int = 20, status: str | None = None, before: datetime | None = None) -> list[AgentRun]:
    query = select(AgentRun).order_by(AgentRun.created_at.desc(), AgentRun.id.desc()).limit(limit)

    if status:
        query = query.where(AgentRun.status == status)
    if before:
        query = query.where(AgentRun.created_at < _aware(before))

    return list(session.scalars(query))

def run_to_dict(run: AgentRun) -> dict[str, Any]:
    return {
        'id': run.id,
        'created_at': _iso(run.created_at),
        'status': run.status,
        'model': run.model,
        'provider': run.provider,
        'team_id': run.team_id,
        'request': {
            'allow_writes': run.allow_writes,
            'write_tools': run.write_tools,
            'prompt': run.prompt,
            'prompt_chars': run.prompt_chars,
            'prompt_sha256': run.prompt_sha256,
        },
        'final_text': run.final_text,
        'error_type': run.error_type,
        'steps': run.steps,
        'tool_calls': run.tool_calls,
        'policy_stops': run.policy_stops,
        'usage': {
            'model_calls': run.model_calls,
            'prompt_tokens': run.prompt_tokens,
            'completion_tokens': run.completion_tokens,
            'model_ms': round(run.model_ms, 1),
            'duration_ms': round(run.duration_ms, 1)
        }
    }

def run_summary(run: AgentRun) -> dict[str, Any]:
    preview = None

    if run.prompt is not None:
        preview = run.prompt[:PREVIEW_CHARS] + ('...' if len(run.prompt) > PREVIEW_CHARS else '')

    return {
        'id': run.id,
        'created_at': _iso(run.created_at),
        'status': run.status,
        'model': run.model,
        'prompt_preview': preview,
        'prompt_chars': run.prompt_chars,
        'tool_calls': run.tool_calls,
        'policy_stops': run.policy_stops,
        'tokens': run.prompt_tokens + run.completion_tokens,
        'duration_ms': round(run.duration_ms, 1)
    }

def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None

    ordered = sorted(values)
    rank = max(1, math.ceil(q / 100 * len(ordered)))

    return ordered[rank - 1]

def stats(session: Session, *, hours: int = 24, now: datetime | None = None) -> dict[str, Any]:
    since = (now or _utcnow()) - timedelta(hours=hours)
    rows = session.execute(select(
        AgentRun.status, AgentRun.model, AgentRun.prompt_tokens, AgentRun.completion_tokens, AgentRun.duration_ms, AgentRun.policy_stops
    ).where(AgentRun.created_at >= since).order_by(AgentRun.created_at.desc()).limit(STATS_MAX_ROWS+1)).all()
    truncated = len(rows) > STATS_MAX_ROWS
    rows = rows[:STATS_MAX_ROWS]
    by_model: dict[str, dict[str, int]] = {}

    for row in rows:
        entry = by_model.setdefault(row.model, {'runs': 0, 'prompt_tokens': 0, 'completion_tokens': 0})
        entry['runs'] += 1
        entry['prompt_tokens'] += row.prompt_tokens
        entry['completion_tokens'] += row.completion_tokens

    durations = [row.duration_ms for row in rows]

    return {
        'window_hours': hours,
        'truncated': truncated,
        'runs': len(rows),
        'by_status': dict(Counter(row.status for row in rows)),
        'prompt_tokens': sum(row.prompt_tokens for row in rows),
        'completion_tokens': sum(row.completion_tokens for row in rows),
        'policy_stops': sum(row.policy_stops for row in rows),
        'duration_ms_p50': percentile(durations, 50),
        'duration_ms_p95': percentile(durations, 95),
        'by_model': by_model
    }

# Maintenance
def prune_runs(session: Session, *, older_than_days: int, now: datetime | None = None) -> int:
    cutoff = (now or _utcnow()) - timedelta(days=older_than_days)
    result = session.execute(delete(AgentRun).where(AgentRun.created_at < cutoff))
    session.commit()

    return result.rowcount or 0

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description='Agent run maintenance')
    commands = parser.add_subparsers(dest='command', required=True)
    prune = commands.add_parser('prune', help='Delete runs older than N days.')
    prune.add_argument('--days', type=int, required=True)
    args = parser.parse_args(argv)

    if args.days < 1:
        parser.error('--days must be at least 1')

    from agentic_suite.db import get_session_factory

    with get_session_factory()() as session:
        deleted = prune_runs(session, older_than_days=args.days)

    print(f'Deleted {deleted} runs older than {args.days} days.')

if __name__ == '__main__':
    main()