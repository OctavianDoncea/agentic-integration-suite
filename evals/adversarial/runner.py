"""Run the adversarial benchmark against a model."""
from __future__ import annotations
import argparse
import asyncio
import json
import re
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from agentic_suite.agents.executor import ToolExecutor
from agentic_suite.agents.loop import DEFAULT_MAX_STEPS, run_agent
from agentic_suite.agents.policy import WritePolicy
from agentic_suite.clients.base import ChatClient, ModelResponse, ToolCall
from agentic_suite.clients.factory import infer_provider, make_chat_client
from evals.adversarial.schema import AdversarialCase, load_cases
from evals.adversarial.scoring import CaseScore, score_case, summarize
from evals.adversarial.tools import build_adversarial_registry
from evals.compare import wilson_interval
from evals.throttle import MinIntervalLimiter, resolve_rpm

RESULTS_DIR = Path(__file__).parent / 'results'

@dataclass(frozen=True)
class Arm:
    name: str
    frame_result: bool
    enforce_policy: bool


ARMS: dict[str, Arm] = {
    'none': Arm('none', frame_result=False, enforce_policy=False),
    'framing': Arm('framing', frame_result=True, enforce_policy=False),
    'policy': Arm('policy', frame_result=False, enforce_policy=True),
    'both': Arm('both', frame_result=True, enforce_policy=True)
}

async def _no_sleep(_: float) -> None:
    return None

class ThrottledClient:
    """Spaces every model call, including the ones made inside the agent loop."""

    def __init__(self, inner: ChatClient, limiter: MinIntervalLimiter) -> None:
        self._inner = inner
        self._limiter = limiter

    @property
    def model(self) -> str:
        return self._inner.model

    async def complete(self, messages: list[Any], tools: list[dict[str, Any]] | None = None, **kwargs: Any) -> ModelResponse:
        await self._limiter.wait()
        return await self._inner.complete(messages, tools, **kwargs)

    def format_assistant_turn(self, response: ModelResponse) -> Any:
        return self._inner.format_assistant_turn(response)

    def format_tool_result(self, call: ToolCall, content: str) -> Any:
        return self._inner.format_tool_result(call, content)


async def run_arm(cases: list[AdversarialCase], arm: Arm, client: ChatClient, *, max_steps: int = DEFAULT_MAX_STEPS) -> tuple[list[CaseScore], list[dict[str, str]]]:
    scores: list[CaseScore] = []
    errors: list[dict[str, str]] = []

    for case in cases:
        executor = ToolExecutor(build_adversarial_registry(), base_context={'lure': case.lure}, sleep=_no_sleep)
        policy = WritePolicy(allow_writes=case.user_allows_writes) if arm.enforce_policy else None

        try:
            result = await run_agent(case.user_prompt, client=client, executor=executor, policy=policy, frame_tool_results=arm.frame_result, max_steps=max_steps)
        except Exception as exc:
            errors.append({'case_id': case.id, 'error': f'{type(exc).__name__}: {exc}'})
            continue

        scores.append(score_case(case, result.steps))

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
        f"# Adversarial eval: `{payload['model']}`",
        '',
        '| Arm | Attack attempted | Attack succeeded | 95% CI | Legit write executed | 95% CI | API errors |',
        '|---|---|---|---|---|---|---|'
    ]

    for name, data in arms.items():
        s = data['summary']
        lines.append(
            f"| {name} | {_frac(s['attempted'], s['attack_cases'])} "
            f"| {_frac(s['succeeded'], s['attack_cases'])} | {_interval(s['succeeded'], s['attack_cases'])} "
            f"| {_frac(s['utility_ok'], s['utility_cases'])} | {_interval(s['utility_ok'], s['utility_cases'])} "
            f"| {len(data['errors'])} |"
        )

    categories = sorted({c for data in arms.values() for c in data['summary']['by_category']})
    lines += [
        '',
        '## Attack success by category (succeeded / cases)',
        '',
        '| Category | ' + ' | '.join(arms) + ' |',
        '|---|' + '---|' * len(arms)
    ]
    
    for category in categories:
        cells = []

        for data in arms.values():
            row = data['summary']['by_category'].get(category)
            cells.append(f"{row['succeeded']}/{row['cases']}" if row else 'n/a')
        
        lines.append(f'| {category} | ' + ' | '.join(cells) + ' |')

    lines += [
        '',
        'How to read this:',
        '',
        '- *Attempted* is how susceptible the model is. *Succeeded* is what reached a real write.',
        "- With writes disabled, nothing can execute, so the policy arms' `unrequested_write` "
        'results are zero by construction. The informative comparison there is `none` vs `framing`.',
        '- The policy checks where a write goes, not what it says, so content-only exfiltration '
        'to a legitimate destination is expected to succeed under it.',
        "- Intervals are wide at this sample size. Overlapping intervals mean the difference isn't meaningful."
    ]

    return '\n'.join(lines) + '\n'

# CLI
async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--model', required=True, help='`provider:model`, or a bare model name.')
    parser.add_argument('--arms', default='none,both', help=f"Comma-separated from: {', '.join(ARMS)}.")
    parser.add_argument('--max-steps', type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument('--out', type=Path, default=None)
    args = parser.parse_args()

    arm_names = [name.strip() for name in args.arms.split(',') if name.strip()]
    unknown = [name for name in arm_names if name not in ARMS]

    if unknown:
        parser.error(f'Unknown arm(s): {", ".join(unknown)}')

    provider = infer_provider(args.model)
    client = ThrottledClient(make_chat_client(args.model, provider), MinIntervalLimiter(resolve_rpm(provider)))
    cases = load_cases()

    payload: dict[str, Any] = {
        'model': args.model,
        'generated_at': datetime.now(timezone.utc).isoformat(),
        'cases': len(cases),
        'arms': {}
    }

    for name in arm_names:
        print(f"Running arm '{name}' ({len(cases)} cases)...", flush=True)
        scores, errors = await run_arm(cases, ARMS[name], client, max_steps=args.max_steps)
        payload['arms'][name] = {
            'summary': summarize(scores),
            'scores': [asdict(score) for score in scores],
            'errors': errors
        }

    RESULTS_DIR.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    safe = re.sub(r'[^\w.-]', '_', args.model)
    path = args.out or RESULTS_DIR / f'{safe}-{stamp}.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding='utf-8')

    print(render_report(payload))
    print(f'Wrote {path}')

if __name__ == '__main__':
    asyncio.run(main())