from __future__ import annotations
import json
from pathlib import Path
from evals.threshold import BASELINE_FULL_PASS_RATE, BASELINE_SMOKE_PASS_RATE, FULL_THRESHOLD, SMOKE_THRESHOLD, SAFETY_MARGIN

BASELINE = Path('evals/baseline_results.json')

def test_thresholds_sit_below_their_baseline():
    assert FULL_THRESHOLD <= BASELINE_FULL_PASS_RATE
    assert SMOKE_THRESHOLD <= BASELINE_SMOKE_PASS_RATE

def test_thresholds_are_not_trivially_low():
    assert FULL_THRESHOLD > 0.5
    assert SMOKE_THRESHOLD > 0.5

def test_margin_is_modest():
    assert SAFETY_MARGIN <= 0.10

def test_baseline_covers_the_whole_benchmark():
    from evals.schema import load_benchmark

    data = json.loads(BASELINE.read_text(encoding='utf-8'))
    assert data['total'] == len(load_benchmark().cases)

def test_baseline_has_no_api_errors():
    data = json.loads(BASELINE.read_text(encoding='utf-8'))
    api_errors = [r for r in data['results'] if (r['failure_reason'] or '').startswith('API error')]
    assert not api_errors, f'Re-run the baseline; {len(api_errors)} cases hit API errors.'