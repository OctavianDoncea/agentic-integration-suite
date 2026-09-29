"""CI regression thresholds, derived from recorded baseline runs."""

from __future__ import annotations
import json
from pathlib import Path

SAFETY_MARGIN = 0.05
_HERE = Path(__file__).parent

def _measured(filename: str) -> tuple[str, float]:
    data = json.loads((_HERE / filename).read_text(encoding='utf-8'))
    return data['model'], data['pass_rate']

BASELINE_FULL_MODEL, BASELINE_FULL_PASS_RATE = _measured('baseline_results.json')
BASELINE_SMOKE_MODEL, BASELINE_SMOKE_PASS_RATE = _measured('baseline_smoke.json')

FULL_THRESHOLD = round(max(0.0, BASELINE_FULL_PASS_RATE - SAFETY_MARGIN), 4)
SMOKE_THRESHOLD = round(max(0.0, BASELINE_SMOKE_PASS_RATE - SAFETY_MARGIN), 4)