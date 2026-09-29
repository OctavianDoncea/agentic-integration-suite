from __future__ import annotations
import argparse
import asyncio
import json
from pathlib import Path
from evals.schema import load_benchmark
from evals.test_evals import report, run_sweep

BASELINE_FULL = Path(__file__).parent / 'baseline_results.json'
BASELINE_SMOKE = Path(__file__).parent / 'baseline_smoke.json'

async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--smoke', action='store_true', help='Run the smoke subset only.')
    parser.add_argument('--out', type=Path, default=None)
    args = parser.parse_args()

    benchmark = load_benchmark()
    cases = benchmark.smoke_subset() if args.smoke else benchmark.cases

    summary = await run_sweep(cases, args.model)
    print(report(summary))

    path = args.out or (BASELINE_SMOKE if args.smoke else BASELINE_FULL)
    path.write_text(json.dumps(summary.to_dict(), indent=2), encoding='utf-8')
    print(f'\nWrote {path}')
    print(f'\nMeasured pass rate: {summary.pass_rate:.4f}')

if __name__ == '__main__':
    asyncio.run(main())