from __future__ import annotations
from dataclasses import asdict
from evals.trajectory.runner import render_report
from evals.trajectory.scoring import TrajectoryScore, summarize

def make_score(case_id, *, category='chain', ok=True, calls=2, pair=None, failures=()):
    return TrajectoryScore(
        case_id=case_id,
        category=category,
        pair=pair,
        target_from_tool=False,
        task_ok=ok,
        efficient=True,
        calls=calls,
        errors=0,
        stop_reason='completed',
        policy_blocked=False,
        failures=tuple(failures),
    )

def make_payload() -> dict:
    scores = [
        make_score('a'),
        make_score('b', ok=False, failures=['missing: notify']),
        make_score('c', category='branch', pair='x'),
        make_score('d', category='branch', pair='x'),
    ]
    return {
        'model': 'fake-model',
        'arms': {
            'open': {
                'summary': summarize(scores),
                'scores': [asdict(s) for s in scores],
                'errors': [],
            }
        },
    }

def test_the_report_shows_the_headline_numbers_and_failures():
    report = render_report(make_payload())
    assert '# Trajectory eval: `fake-model`' in report
    assert '| open | 3/4 (75%)' in report
    assert '1/1 (100%)' in report  # the complete branch pair
    assert '- `b`: missing: notify' in report

def test_the_report_survives_an_empty_arm():
    payload = {
        'model': 'm',
        'arms': {'open': {'summary': summarize([]), 'scores': [], 'errors': [{'case_id': 'a', 'error': 'x'}]}},
    }
    assert 'n/a' in render_report(payload)