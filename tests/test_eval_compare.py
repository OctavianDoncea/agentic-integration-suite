from __future__ import annotations
import pytest
from evals.compare import render, wilson_interval

def make_run(model: str, outcomes: dict[str, bool]) -> dict:
    passed = sum(outcomes.values())
    total = len(outcomes)
    return {
        'model': model,
        'total': total,
        'passed': passed,
        'pass_rate': passed / total,
        'selection_accuracy': passed / total,
        'argument_precision': passed / total,
        'by_category': {
            'direct_intent': {'total': total, 'passed': passed, 'pass_rate': passed / total}
        },
        'results': [
            {
                'case_id': case_id,
                'category': 'direct_intent',
                'passed': ok,
                'failure_reason': None if ok else "selected 'wrong_tool', expected 'x'",
            }
            for case_id, ok in outcomes.items()
        ],
    }

def test_wilson_interval_matches_the_reference_value():
    low, high = wilson_interval(47, 50)
    assert low == pytest.approx(0.8378, abs=1e-3)
    assert high == pytest.approx(0.9794, abs=1e-3)

def test_wilson_interval_stays_inside_bounds_at_100_percent():
    low, high = wilson_interval(50, 50)
    assert high == pytest.approx(1.0)
    assert 0.9 < low < 1.0

def test_wilson_interval_of_nothing():
    assert wilson_interval(0, 0) == (0.0, 0.0)

def test_close_models_are_reported_as_indistinguishable():
    a = make_run('model-a', {f'c{i}': i != 0 for i in range(10)})  # 9/10
    b = make_run('model-b', {f'c{i}': i not in (0, 1) for i in range(10)})  # 8/10
    report = render([('a', a), ('b', b)])

    assert "isn't statistically meaningful" in report
    assert '| a | `model-a` | 9/10 |' in report

def test_disagreements_are_listed_with_reasons():
    a = make_run('model-a', {'c1': True, 'c2': True})
    b = make_run('model-b', {'c1': True, 'c2': False})
    report = render([('a', a), ('b', b)])

    assert "## Cases where runs disagree (1)" in report
    assert "| `c2` | pass | fail: selected 'wrong_tool'" in report

def test_mismatched_case_sets_are_flagged():
    a = make_run('model-a', {'c1': True})
    b = make_run('model-b', {'c2': True})
    assert "didn't score the same set of cases" in render([("a", a), ("b", b)])