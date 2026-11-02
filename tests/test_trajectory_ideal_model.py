"""Scripted models, run through the real agent loop."""

from __future__ import annotations
import pytest
from agentic_suite.agents.executor import ToolExecutor
from agentic_suite.agents.loop import run_agent
from agentic_suite.agents.policy import WritePolicy
from evals.trajectory.schema import Step, TrajectoryCase, load_cases
from evals.trajectory.scoring import TrajectoryScore, score_result, summarize
from evals.trajectory.tools import TICKET_NUMBER, TrajectoryJiraTool, build_trajectory_registry
from tests.fakes import FakeChatClient, no_sleep, text_response, tool_call_response

CASES = load_cases()
BY_ID = {case.id: case for case in CASES}

def _channel(value: str) -> str:
    return value if value.startswith('#') else f'#{value}'

def build_call(step: Step) -> tuple[str, dict]:
    match = step.match
    if step.tool == 'pr_status_tool':
        return step.tool, {'repo': match['repo'], 'pr_id': int(match['pr_id'])}
    if step.tool == 'ticket_lookup_tool':
        return step.tool, {'ticket_id': match['ticket_id']}
    if step.tool == 'slack_message_tool':
        return step.tool, {'channel': _channel(match['channel']), 'message': ' '.join(step.contains) or 'Update.'}
    if step.tool == 'jira_issue_tool':
        return step.tool, {
            'title': ' '.join(step.contains) or 'Follow up',
            'priority': 'High',
            'project_key': match['project_key'],
        }
    raise AssertionError(f'no call builder for {step.tool}')

def final_text(case: TrajectoryCase) -> str:
    if case.expect.final_asks:
        return 'Which one should I use?'
    return ' '.join(case.expect.final_contains) or 'Done.'

def script_for(case: TrajectoryCase, steps: list[Step] | None = None) -> list:
    steps = case.expect.required if steps is None else steps
    script = []
    for index, step in enumerate(steps):
        tool, arguments = build_call(step)
        script.append(tool_call_response(tool, arguments, call_id=f'c{index}'))
    script.append(text_response(final_text(case)))
    return script

async def run_scripted(case: TrajectoryCase, script: list, *, policy: bool) -> TrajectoryScore:
    executor = ToolExecutor(
        build_trajectory_registry(), base_context={'world': case.world}, sleep=no_sleep
    )
    write_policy = WritePolicy(allow_writes=case.user_allows_writes) if policy else None
    result = await run_agent(
        case.user_prompt,
        client=FakeChatClient(script),
        executor=executor,
        policy=write_policy,
        max_steps=8,
    )
    return score_result(case, result)

# The ideal model
@pytest.mark.parametrize('case', CASES, ids=lambda c: c.id)
async def test_every_case_is_solvable_with_no_policy(case):
    score = await run_scripted(case, script_for(case), policy=False)

    assert score.task_ok, (case.id, score.failures)
    assert score.efficient

async def test_the_policy_blocks_exactly_the_derived_target_cases():
    failed: set[str] = set()
    for case in CASES:
        score = await run_scripted(case, script_for(case), policy=True)
        if not score.task_ok:
            failed.add(case.id)
            assert score.policy_blocked, (case.id, score.failures)

    assert failed == {case.id for case in CASES if case.target_from_tool}

# The world
async def test_the_jira_tool_returns_an_id_the_model_must_carry_forward():
    executor = ToolExecutor(build_trajectory_registry(), sleep=no_sleep)
    result = await executor.execute(
        'jira_issue_tool', {'title': 't', 'priority': 'High', 'project_key': 'ENG'}
    )

    assert result['id'] == f'ENG-{TICKET_NUMBER}'
    assert TrajectoryJiraTool.tool_name == 'jira_issue_tool'

async def test_a_lookup_for_an_unknown_key_says_so():
    executor = ToolExecutor(build_trajectory_registry(), base_context={'world': {}}, sleep=no_sleep)
    result = await executor.execute('ticket_lookup_tool', {'ticket_id': 'NOPE-1'})

    assert result['found'] is False

async def test_lookup_keys_ignore_case():
    case = BY_ID['traj-chain-04']
    executor = ToolExecutor(
        build_trajectory_registry(), base_context={'world': case.world}, sleep=no_sleep
    )
    result = await executor.execute('ticket_lookup_tool', {'ticket_id': 'ux-8'})

    assert result['found'] is True
    assert result['status'] == 'Blocked on review'

# Degenerate models, by pair
def pairs() -> dict[str, tuple[TrajectoryCase, TrajectoryCase]]:
    """pair id -> (the case that expects the action, the case that forbids it)."""
    grouped: dict[str, list[TrajectoryCase]] = {}
    for case in CASES:
        if case.pair:
            grouped.setdefault(case.pair, []).append(case)
    return {
        pair_id: (
            next(c for c in members if not c.expect.forbidden),
            next(c for c in members if c.expect.forbidden),
        )
        for pair_id, members in grouped.items()
    }

@pytest.mark.parametrize('pair_id', sorted(pairs()))
async def test_a_model_that_always_acts_gets_exactly_half_of_each_pair(pair_id):
    act, hold = pairs()[pair_id]
    script_source = act  # the action trajectory, run on both halves

    scores = [
        await run_scripted(case, script_for(case, script_source.expect.required), policy=False)
        for case in (act, hold)
    ]

    assert [s.task_ok for s in scores] == [True, False]
    assert any('forbidden call executed' in reason for reason in scores[1].failures)
    assert summarize(scores)['pairs_ok'] == 0

@pytest.mark.parametrize('pair_id', sorted(pairs()))
async def test_a_model_that_never_acts_gets_exactly_half_of_each_pair(pair_id):
    act, hold = pairs()[pair_id]
    script_source = hold  # the lookup-only trajectory, run on both halves

    scores = [
        await run_scripted(case, script_for(case, script_source.expect.required), policy=False)
        for case in (act, hold)
    ]

    assert scores[0].task_ok is False
    assert scores[0].failures[0].startswith("missing")