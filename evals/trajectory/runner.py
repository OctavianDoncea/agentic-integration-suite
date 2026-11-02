"""Run the trajectory benchmark against a model, with and without the write policy."""
from __future__ import annotations
import argparse
import asyncio
import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from agentic_suite.agents.executor import ToolExecutor
from agentic_suite.agents.policy import WritePolicy
from agentic_suite.agents.loop import run_agent
from agentic_suite.clients.base import ChatClient
from agentic_suite.clients.factory import infer_provider, make_chat_client
from evals.compare import wilson_interval
from evals.throttle import MinIntervalLimiter, ThrottledClient, resolve_rpm
from evals.trajectory.schema import TrajectoryCase, load_cases
from evals.trajectory.scoring import TrajectoryScore, score_result, summarize
from evals.trajectory.tools import build_trajectory_registry

RESULTS_DIR = Path(__file__).parent / 'results'
DEFAULT_MAX_STEPS = 8
MAX_FAILURES_SHOWN = 12

@dataclass(frozen=True)
class Arm:
    name: str
    enforce_policy: bool


ARMS: dict[str, Arm] = {
    'open': Arm('open', enforce_policy=False),
    'policy': Arm('policy', enforce_policy=True)
}

async def _no_sleep(_: float) -> None:
    return None

async def run_arm(cases: list[TrajectoryCase], arm: Arm, client: ChatClient, *, max_steps: int = DEFAULT_MAX_STEPS) -> tuple[list[TrajectoryScore], list[dict[str, str]]]:
    scores: list[TrajectoryScore] = []
    errors: list[dict[str, str]] = []

    for case in cases:
        executor = ToolExecutor(build_trajectory_registry(), base_context={'world': case.world}, sleep=_no_sleep)
        policy = WritePolicy(allow_writes=case.user_allows_writes) if arm.enforce_policy else None

        try:
            result = await run_agent(case.user_prompt, client=client, executor=executor, policy=policy, max_steps=max_steps)
        except Exception as exc:
            errors.append({'case_id': case.id, 'error': f'{type(exc).__name__}: {exc}'})
            continue

        scores.append(score_result(case, result))

    return scores, errors

# Report
def _frac(n: int, d: int) -> str:
    return 'n/a' if d == 0 else f'{n}/{d} ({n / d:.0%})'

def _interval(n: int, d: int) -> str:
    if d == 0:
        return 'n/a'

    low, high = wilson_interval(n, d)
    return f'{low:.0%}-{high:.0%}'

def render_report(payload: dict[str, Any]) -> str:
    arms: dict[str, Any] = payload['arms']
    lines = [
        f"# Trajectory eval: `{payload['model']}`",
        '',
        '| Arm | Task success | 95% CI | Excl. derived-target | Branch pairs both right '
        '| Efficient (of successes) | Mean calls | API errors |',
        '|---|---|---|---|---|---|---|---|'
    ]

    for name, data in arms.items():
        s = data['summary']
        mean_calls = f"{s['calls'] / s['cases']:.1f}" if s['cases'] else 'n/a'
        lines.append(
            f"| {name} | {_frac(s['task_ok'], s['cases'])} | {_interval(s['task_ok'], s['cases'])} "
            f"| {_frac(s['core_ok'], s['core_cases'])} | {_frac(s['pairs_ok'], s['pairs'])} "
            f"| {_frac(s['efficient_of_ok'], s['task_ok'])} | {mean_calls} | {len(data['errors'])} |"
        )
        
    categories = sorted({c for data in arms.values() for c in data['summary']['by_category']})
    lines += [
        '',
        '## Task success by category (passed / cases)',
        '',
        '| Category | ' + ' | '.join(arms) + ' |',
        '|---|' + '---|' * len(arms)
    ]

    for category in categories:
        cells = []

        for data in arms.values():
            row = data['summary']['by_category'].get(category)
            cells.append(f"{row['task_ok']}/{row['cases']}" if row else 'n/a')

        lines.append(f"| {category} | " + " | ".join(cells) + " |")

    for name, data in arms.items():
        failed = [score for score in data['scores'] if score['failures']]
        lines += ['', f'## {name} ({len(failed)})', '']

        if not failed:
            lines.append('None.')

        for score in failed[:MAX_FAILURES_SHOWN]:
            lines.append(f"- `{score['case_id']}`: {score['failures'][0]}")

        if len(failed) > MAX_FAILURES_SHOWN:
            lines.append(f'- ...and {len(failed) - MAX_FAILURES_SHOWN} more in the JSON.')

    lines += [
        '',
        'How to read this:',
        '',
        '- *Task success* needs the right calls, in a valid order, carrying the right values, '
        'with no forbidden action, a finished run, and the right final answer.',
        '- *Branch pairs both right* is the conditional-reasoning number. A model that always acts, '
        'or never acts, gets one half of each pair and scores zero here.',
        '- *Excl. derived-target* leaves out the two cases whose destination comes from tool data. '
        'The write policy is expected to block those, so the gap between `open` and `policy` on '
        "them is the policy's cost, not a model failure.",
        '- *Efficient* counts calls, not tokens, and only among successful runs.',
        "- Intervals are wide at this sample size. Overlapping intervals mean the difference isn't meaningful."
    ]

    return '\n'.join(lines) + '\n'

# CLI
async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--model', required=True, help="`provider:model`, or a bare model name.")
    parser.add_argument('--arms', default='open,policy', help=f'Comma-separated from: {", ".join(ARMS)}.')
    parser.add_argument('--max-steps', type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument('--out', type=Path, default=None)
    args = parser.parse_args()

    arm_names = [name.strip() for name in args.arms.split(',') if name.strip()]
    unknown = [name for name in arm_names if name not in ARMS]
    if unknown:
        parser.error(f'unknown arm(s): {", ".join(unknown)}')

    provider = infer_provider(args.model)
    client = ThrottledClient(
        make_chat_client(args.model, provider), MinIntervalLimiter(resolve_rpm(provider))
    )
    cases = load_cases()

    payload: dict[str, Any] = {
        'model': args.model,
        'generated_at': datetime.now(timezone.utc).isoformat(),
        'cases': len(cases),
        'arms': {},
    }
    for name in arm_names:
        print(f"Running arm '{name}' ({len(cases)} cases)...", flush=True)
        scores, errors = await run_arm(cases, ARMS[name], client, max_steps=args.max_steps)
        payload['arms'][name] = {
            'summary': summarize(scores),
            'scores': [asdict(score) for score in scores],
            'errors': errors,
        }

    RESULTS_DIR.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    safe = re.sub(r'[^\w.-]+', '_', args.model)
    path = args.out or RESULTS_DIR / f'{safe}-{stamp}.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding='utf-8')

    print(render_report(payload))
    print(f'Wrote {path}')

if __name__ == '__main__':
    asyncio.run(main())