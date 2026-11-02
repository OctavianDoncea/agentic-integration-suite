from __future__ import annotations
import pytest
from agentic_suite.agents.policy import mentioned_in
from evals.trajectory.schema import Category, load_cases
from evals.trajectory.tools import LOOKUP_TOOLS, build_trajectory_registry, pr_key, ticket_key

CASES = load_cases()
REGISTRY = build_trajectory_registry()
TOOLS = {tool.get_name(): tool for tool in REGISTRY.tools()}
TARGET_FIELDS = {'slack_message_tool': 'channel', 'jira_issue_tool': 'project_key'}

def test_the_dataset_has_the_expected_shape():
    counts = {c: sum(1 for case in CASES if case.category is c) for c in Category}
    assert counts == {
        Category.CHAIN: 5,
        Category.BRANCH: 8,
        Category.FANOUT: 3,
        Category.RESTRAINT: 2,
        Category.DERIVED_TARGET: 2,
    }

@pytest.mark.parametrize('case', CASES, ids=lambda c: c.id)
def test_every_tool_a_case_names_exists(case):
    assert set(case.world) <= LOOKUP_TOOLS
    for step in case.expect.required:
        assert step.tool in TOOLS
    for spec in case.expect.forbidden:
        assert spec.tool in TOOLS

@pytest.mark.parametrize('case', CASES, ids=lambda c: c.id)
def test_every_required_lookup_has_a_record_in_the_world(case):
    for step in case.expect.required:
        if step.tool == 'pr_status_tool':
            key = pr_key(step.match['repo'], step.match['pr_id'])
        elif step.tool == 'ticket_lookup_tool':
            key = ticket_key(step.match['ticket_id'])
        else:
            continue
        assert key in case.world.get(step.tool, {}), f'{case.id}: no record for {key}'

@pytest.mark.parametrize('case', CASES, ids=lambda c: c.id)
def test_write_targets_come_from_the_user_unless_the_case_says_otherwise(case):
    """A legitimate target must be in the user's words (so the policy allows it); a derived target must not be (that's what makes it derived)."""
    for step in case.expect.required:
        field = TARGET_FIELDS.get(step.tool)
        if field is None or field not in step.match:
            continue
        named = mentioned_in(step.match[field], case.user_prompt)
        assert named != case.target_from_tool, f'{case.id}: {step.id}'

@pytest.mark.parametrize('case', CASES, ids=lambda c: c.id)
def test_read_only_requests_expect_no_writes(case):
    if case.user_allows_writes:
        return
    for step in case.expect.required:
        assert TOOLS[step.tool].read_only, f'{case.id}: {step.id} is a write'

def test_branch_pairs_share_a_prompt_and_differ_in_the_world():
    pairs: dict[str, list] = {}
    for case in CASES:
        if case.pair:
            pairs.setdefault(case.pair, []).append(case)

    assert len(pairs) == 4
    for pair_id, members in pairs.items():
        assert len(members) == 2, pair_id
        first, second = members
        assert first.user_prompt == second.user_prompt, pair_id
        assert first.world != second.world, pair_id
        # Exactly one half expects the action; the other forbids it.
        assert bool(first.expect.forbidden) != bool(second.expect.forbidden), pair_id