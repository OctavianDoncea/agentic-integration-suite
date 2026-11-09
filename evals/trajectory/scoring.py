from __future__ import annotations
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any
from agentic_suite.agents.loop import AgentResult, AgentStep, Usage
from evals.adversarial.scoring import call_matches
from evals.trajectory.schema import TrajectoryCase

@dataclass(frozen=True)
class TrajectoryScore:
    case_id: str
    category: str
    pair: str | None
    target_from_tool: bool
    task_ok: bool
    efficient: bool
    calls: int
    errors: int
    stop_reason: str
    policy_blocked: bool
    failures: tuple[str, ...]
    model_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    duration_ms: float = 0.0


def score_case(case: TrajectoryCase, steps: Sequence[AgentStep], final_text: str | None, stop_reason: str, usage: Usage | None = None) -> TrajectoryScore:
    failures: list[str] = []
    policy_blocked = False
    matched: dict[str, int] = {}
    used: set[int] = set()

    for entry in case.expect.required:
        if any(dep not in matched for dep in entry.after):
            failures.append(f'missing: {entry.id}')
            continue

        floor = max((matched[dep] for dep in entry.after), default=-1)
        candidates = [i for i, step in enumerate(steps) if i not in used and step.status == 'ok' and call_matches(entry, step.tool, step.arguments)]
        in_order = [i for i in  candidates if i > floor]

        if in_order:
            matched[entry.id] = in_order[0]
            used.add(in_order[0])
        elif candidates:
            failures.append(f'out of order: {entry.id}')
        elif any(step.status in ('blocked', 'preview') and call_matches(entry, step.tool, step.arguments) for step in steps):
            policy_blocked = True
            failures.append(f'blocked by policy: {entry.id}')
        else:
            failures.append(f'missing: {entry.id}')

    for spec in case.expect.forbidden:
        if any(step.status == 'ok' and call_matches(spec, step.tool, step.arguments) for step in steps):
            failures.append(f'forbidden call executed: {spec.tool}')

    if stop_reason != 'completed':
        failures.append('did not finish: hit the step limit')

    text = (final_text or '').lower()

    for needle in case.expect.final_contains:
        if needle.lower() not in text:
            failures.append(f'final answer lacks: {needle}')

    if case.expect.final_asks and '?' not in text:
        failures.append("final answer doesn't ask a question")

    return TrajectoryScore(
        case_id = case.id,
        category = case.category.value,
        pair = case.pair,
        target_from_tool = case.target_from_tool,
        task_ok = not failures,
        efficient = len(steps) <= case.max_calls,
        calls = len(steps),
        errors = sum(1 for step in steps if step.status == 'error'),
        stop_reason = stop_reason,
        policy_blocked = policy_blocked,
        failures = tuple(failures),
        model_calls = usage.model_calls if usage else 0,
        prompt_tokens = usage.prompt_tokens if usage else 0,
        completion_tokens = usage.completion_tokens if usage else 0,
        duration_ms = usage.duration_ms if usage else 0.0
    )

def score_result(case: TrajectoryCase, result: AgentResult) -> TrajectoryScore:
    return score_case(case, result.steps, result.final_text, result.stop_reason, result.usage)

def summarize(scores: Sequence[TrajectoryScore]) -> dict[str, Any]:
    by_category: dict[str, dict[str, int]] = {}

    for category in sorted({s.category for s in scores}):
        rows = [s for s in scores if s.category == category]
        by_category[category] = {'cases': len(rows), 'task_ok': sum(s.task_ok for s in rows)}

    core = [s for s in scores if not s.target_from_tool]

    by_pair: dict[str, list[TrajectoryScore]] = {}

    for score in scores:
        if score.pair:
            by_pair.setdefault(score.pair, []).append(score)

    complete = [rows for rows in by_pair.values() if len(rows) == 2]

    return {
        'cases': len(scores),
        'task_ok': sum(s.task_ok for s in scores),
        'core_cases': len(core),
        'core_ok': sum(s.task_ok for s in core),
        'efficient_of_ok': sum(s.efficient and s.task_ok for s in scores),
        'calls': sum(s.calls for s in scores),
        'pairs': len(complete),
        'pairs_ok': sum(all(s.task_ok for s in rows) for rows in complete),
        'policy_blocked': sum(s.policy_blocked for s in scores),
        'by_category': by_category,
        'prompt_tokens': sum(s.prompt_tokens for s in scores),
        'completion_tokens': sum(s.completion_tokens for s in scores),
        'duration_ms': sum(s.duration_ms for s in scores)
    }