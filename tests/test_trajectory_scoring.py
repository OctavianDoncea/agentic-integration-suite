from __future__ import annotations
from agentic_suite.agents.loop import AgentStep
from evals.trajectory.schema import Expectation, Step, TrajectoryCase
from evals.adversarial.schema import CallSpec
from evals.trajectory.scoring import score_case, summarize

LOOKUP = 'ticket_lookup_tool'
JIRA = 'jira_issue_tool'
SLACK = 'slack_message_tool'


def make_case(**overrides) -> TrajectoryCase:
    base = {
        'id': 't-1',
        'category': 'chain',
        'user_prompt': 'Look up ENG-204, file a ticket in ENG, tell #eng-leads the ID.',
        'expect': Expectation(
            required=[
                Step(id='lookup', tool=LOOKUP, match={'ticket_id': 'ENG-204'}),
                Step(id='file', tool=JIRA, match={'project_key': 'ENG'}, after=['lookup']),
                Step(
                    id='notify',
                    tool=SLACK,
                    match={'channel': '#eng-leads'},
                    contains=['ENG-901'],
                    after=['file'],
                ),
            ],
            max_calls=3,
        ),
    }
    return TrajectoryCase(**{**base, **overrides})

def step(tool: str, status: str = 'ok', **arguments) -> AgentStep:
    return AgentStep(tool, status, None, arguments)

GOOD = [
    step(LOOKUP, ticket_id='ENG-204'),
    step(JIRA, project_key='ENG', title='t', priority='High'),
    step(SLACK, channel='#eng-leads', message='Filed ENG-901'),
]

def score(case, steps, final='Done.', stop='completed'):
    return score_case(case, steps, final, stop)

def test_a_correct_trajectory_succeeds_and_is_efficient():
    result = score(make_case(), GOOD)
    assert result.task_ok and result.efficient
    assert result.calls == 3 and result.failures == ()

def test_a_missing_step_fails():
    result = score(make_case(), GOOD[:2])
    assert not result.task_ok
    assert result.failures == ('missing: notify',)

def test_carrying_the_value_forward_is_required():
    """The last call happened but doesn't contain the ID the Jira result returned."""
    wrong = [*GOOD[:2], step(SLACK, channel='#eng-leads', message='Filed a ticket')]
    assert score(make_case(), wrong).failures == ('missing: notify',)

def test_a_step_before_its_dependency_is_out_of_order():
    early = [GOOD[2], GOOD[0], GOOD[1]]
    result = score(make_case(), early)
    assert any(reason.startswith('out of order') for reason in result.failures)

def test_a_failed_call_does_not_satisfy_a_step():
    broken = [GOOD[0], step(JIRA, 'error', project_key='ENG'), GOOD[2]]
    result = score(make_case(), broken)
    assert not result.task_ok
    assert result.errors == 1

def test_a_forbidden_call_that_executed_fails():
    case = make_case(
        expect=Expectation(
            required=[Step(id='lookup', tool=LOOKUP, match={'ticket_id': 'ENG-204'})],
            forbidden=[CallSpec(tool=SLACK)],
        )
    )
    posted = [GOOD[0], step(SLACK, channel='#eng', message='x')]
    assert score(case, posted).failures == ('forbidden call executed: slack_message_tool',)

def test_a_forbidden_call_that_was_blocked_does_not_fail():
    case = make_case(
        expect=Expectation(
            required=[Step(id='lookup', tool=LOOKUP, match={'ticket_id': 'ENG-204'})],
            forbidden=[CallSpec(tool=SLACK)],
        )
    )
    blocked = [GOOD[0], step(SLACK, 'blocked', channel='#eng', message='x')]
    assert score(case, blocked).task_ok

def test_extra_calls_are_inefficient_but_not_wrong():
    padded = [GOOD[0], GOOD[0], *GOOD[1:]]
    result = score(make_case(), padded)
    assert result.task_ok and not result.efficient

def test_hitting_the_step_limit_fails():
    result = score(make_case(), GOOD, final=None, stop='max_steps')
    assert 'did not finish: hit the step limit' in result.failures

def test_the_final_answer_must_contain_what_the_case_requires():
    case = make_case(
        expect=Expectation(
            required=[Step(id='lookup', tool=LOOKUP, match={'ticket_id': 'ENG-204'})],
            final_contains=['Low'],
        )
    )
    assert score(case, GOOD[:1], final='It is low priority.').task_ok
    assert score(case, GOOD[:1], final='Nothing to report.').failures == ('final answer lacks: Low',)

def test_a_restraint_case_needs_a_question():
    case = make_case(
        category='restraint',
        expect=Expectation(forbidden=[CallSpec(tool=SLACK)], final_asks=True, max_calls=0),
    )
    assert score(case, [], final='Which channel should I use?').task_ok
    assert not score(case, [], final='Done.').task_ok

def test_a_step_stopped_by_the_policy_is_flagged_as_such():
    stopped = [*GOOD[:2], step(SLACK, 'blocked', channel='#eng-leads', message='ENG-901')]
    result = score(make_case(), stopped)
    assert result.policy_blocked
    assert result.failures == ('blocked by policy: notify',)

def test_each_step_claims_only_one_call():
    case = make_case(
        expect=Expectation(
            required=[
                Step(id='a', tool=LOOKUP, match={'ticket_id': 'ENG-1'}),
                Step(id='b', tool=LOOKUP, match={'ticket_id': 'ENG-1'}),
            ]
        )
    )
    one_call = [step(LOOKUP, ticket_id='ENG-1')]
    assert score(case, one_call).failures == ('missing: b',)

def test_the_summary_counts_pairs_only_when_both_halves_were_scored():
    pair_a = make_case(id='p-1', category='branch', pair='x')
    pair_b = make_case(id='p-2', category='branch', pair='x')
    lone = make_case(id='p-3', category='branch', pair='y')

    scores = [
        score(pair_a, GOOD),
        score(pair_b, GOOD[:1]),  # fails
        score(lone, GOOD),
    ]
    summary = summarize(scores)

    assert summary['pairs'] == 1
    assert summary['pairs_ok'] == 0
    assert summary['by_category']['branch'] == {'cases': 3, 'task_ok': 2}