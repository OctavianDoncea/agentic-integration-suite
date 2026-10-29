"""Compare eval runs across models, with confidence intervals."""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
from typing import Any

Z_95 = 1.959964 # Wilson score intervals
Run = tuple[str, dict[str, Any]]

def wilson_interval(passed: int, total: int, z: float = Z_95) -> tuple[float, float]:
    if total == 0:
        return (0.0, 0.0)

    p = passed / total
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    half_width = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator

    return (max(0.0, centre - half_width), min(1.0, centre + half_width))

def load_runs(paths: list[Path]) -> list[Run]:
    files: list[Path] = []

    for path in paths:
        files.extend(sorted(path.glob('*.json')) if path.is_dir() else [path])

    if not files:
        raise SystemExit('No result files found.')

    return [(f.stem, json.loads(f.read_text(encoding='utf-8'))) for f in files]

def _pct(value: float) -> str:
    return f'{value:.1%}'

def _short(reason: str | None, limit: int = 60) -> str:
    text = (reason or '').replace('|', '\\|')
    return text if len(text) <= limit else text[: limit - 1] + '...'

def _api_errors(data: dict[str, Any]) -> int:
    return sum(1 for r in data['results'] if (r.get('failure_reason') or '').startswith('API error:'))

def _separability_note(runs: list[Run], intervals: dict[str, tuple[float, float]]) -> str:
    if len(runs) < 2:
        return 'Only one run, so there is nothing to compare.'

    best_label, best = max(runs, key=lambda run: run[1]['pass_rate'])
    best_low = intervals[best_label][0]
    overlapping = [label for label, _ in runs if label != best_label and intervals[label][1] >= best_low]

    if not overlapping:
        return f"**{best_label}**'s 95% interval doesn't overlap any other run's."

    names = ', '.join(f'**{label}**' for label in overlapping)
    return (
        f"The 95% intervals of **{best_label}** and {names} overlap, so at {best['total']} cases their difference isn't statistically meaningful." 
        '(Overlap is a conservative check, not a formal significance test.)'
    )

def render(runs: list[Run]) -> str:
    lines = ['# Cross-model eval comparison', '']
    case_sets = {frozenset(r['case_id'] for r in data['results']) for _, data in runs}

    if len(case_sets) > 1:
        lines += ["> **Warning:** these runs didn't score the same set of cases, so their rates aren't directly comparable.", '']

    # Overall
    lines += ['## Overall', '', f'| Run | Model | Passed | Pass rate | 95% CI | Tool selection | Argument precision | API errors |', '|---|---|---|---|---|---|---|---|']
    intervals: dict[str, tuple[float, float]] = {}

    for label, data in runs:
        low, high = wilson_interval(data['passed'], data['total'])
        intervals[label] = (low, high)
        lines.append(
            f"| {label} | `{data['model']}` | {data['passed']}/{data['total']} "
            f"| {_pct(data['pass_rate'])} | {_pct(low)}-{_pct(high)} "
            f"| {_pct(data['selection_accuracy'])} | {_pct(data['argument_precision'])} "
            f"| {_api_errors(data)} |"
        )

    lines += ['', _separability_note(runs, intervals), '']

    # By category
    labels = [label for label, _ in runs]
    categories = sorted({c for _, data in runs for c in data['by_category']})
    lines += ['## By category', '', '| Category | ' + ' | '.join(labels) + ' |', '|---|' + '---|' * len(runs)]

    for category in categories:
        cells = []

        for _, data in runs:
            stats = data['by_category'].get(category)
            cells.append(f"{stats['passed']}/{stats['total']}" if stats else '-')
        
        lines.append(f'| {category} | ' + ' | '.join(cells) + ' |')

    # Disagreements
    outcomes = {label: {r['case_id']: r for r in data['results']} for label, data in runs}
    all_ids = sorted(set().union(*(set(o) for o in outcomes.values())))
    disagreements = [case_id for case_id in all_ids if len({o[case_id]['passed'] for o in outcomes.values() if case_id in o}) > 1]
    lines += ['', f'## Cases where runs disagree ({len(disagreements)})', '']

    if not disagreements:
        lines.append('None.')
    else:
        lines += ['| Case | ' + ' | '.join(labels) + ' |', '|---|' + '---|' * len(runs)]

        for case_id in disagreements:
            cells = []

            for label in labels:
                result = outcomes[label].get(case_id)

                if result is None:
                    cells.append('-')
                elif result['passed']:
                    cells.append('pass')
                else:
                    cells.append('fail: ' + _short(result.get('failure_reason')))
            
            lines.append(f'| `{case_id}` | ' + ' | '.join(cells) + ' |')

    return '\n'.join(lines) + '\n'

def main() -> None:
    parser = argparse.ArgumentParser(description='Compare eval result files.')
    parser.add_argument('paths', nargs='+', type=Path, help='result files or directories.')
    parser.add_argument('--out', type=Path, default=None, help='Also write the report here.')
    args = parser.parse_args()

    report = render(load_runs(args.paths))
    print(report)

    if args.out:
        args.out.write_text(report, encoding='utf-8')
        print(f'Wrote {args.out}')

if __name__ == '__main__':
    main()