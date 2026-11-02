from __future__ import annotations
import pytest
from agentic_suite.agents.policy import mentioned_in
from evals.adversarial.schema import Category, load_cases
from evals.adversarial.tools import LURE_TOOLS, build_adversarial_registry

CASES = load_cases()
WIRE_NAMES = {tool.get_name() for tool in build_adversarial_registry().tools()}

def test_the_dataset_has_the_expected_shape():
    counts = {c: sum(1 for case in CASES if case.category is c) for c in Category}
    assert counts == {
        Category.REDIRECT: 6,
        Category.UNREQUESTED_WRITE: 6,
        Category.EXFILTRATION: 4,
        Category.CONTROL: 4,
    }

@pytest.mark.parametrize('case', CASES, ids=lambda c: c.id)
def test_every_tool_a_case_names_exists(case):
    assert set(case.lure) <= LURE_TOOLS
    for spec in (case.attack, case.expected_write):
        if spec is not None:
            assert spec.tool in WIRE_NAMES

@pytest.mark.parametrize('case', CASES, ids=lambda c: c.id)
def test_an_attack_target_is_never_in_the_users_own_words(case):
    """Otherwise the case would be a legitimate request in disguise."""
    if case.attack is None:
        return
    for value in case.attack.match.values():
        assert not mentioned_in(value, case.user_prompt), case.id

@pytest.mark.parametrize('case', CASES, ids=lambda c: c.id)
def test_a_legitimate_target_is_in_the_users_own_words(case):
    """Otherwise the policy would (correctly) block the legitimate write."""
    if case.expected_write is None:
        return
    for value in case.expected_write.match.values():
        assert mentioned_in(value, case.user_prompt), case.id

def test_controls_carry_no_injection():
    for case in CASES:
        if case.category is Category.CONTROL:
            assert case.lure == {} and case.attack is None