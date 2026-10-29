"""Record a sweep: a CI baseline, or one run of a cross-model comparison."""
from __future__ import annotations
import argparse
import asyncio
import json
from pathlib import Path
from agentic_suite.clients.factory import PROVIDERS, infer_provider
from evals.schema import load_benchmark
from evals.test_evals import report, run_sweep

BASELINE_FULL = Path(__file__).parent / 'baseline_results.json'
BASELINE_SMOKE = Path(__file__).parent / 'baseline_smoke.json'

CI_BASELINE_PROVIDER = 'groq'

async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--model', required=True, help='`provider:model`, or a bare model name.')
    parser.add_argument(
        '--provider',
        choices=sorted(PROVIDERS),
        default=None,
        help="Defaults to the model spec's prefix, then to inferring from the name."
    )
    parser.add_argument('--smoke', action='store_true', help='Run the smoke subset only.')
    parser.add_argument('--out', type=Path, default=None)
    args = parser.parse_args()
    provider = args.provider or infer_provider(args.model)

    if args.out is None and provider != CI_BASELINE_PROVIDER:
        parser.error(
            f'{provider} runs are comparisons, not CI baselines '
            f'(those are measured on {CI_BASELINE_PROVIDER}): pass --out.'
        )

    benchmark = load_benchmark()
    cases = benchmark.smoke_subset() if args.smoke else benchmark.cases
    summary = await run_sweep(cases, args.model, provider=provider)
    print(report(summary))

    path = args.out or (BASELINE_SMOKE if args.smoke else BASELINE_FULL)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary.to_dict(), indent=2), encoding="utf-8")
    print(f'\nWrote {path}')
    print(f'\nMeasured pass rate: {summary.pass_rate:.4f}')

if __name__ == '__main__':
    asyncio.run(main())