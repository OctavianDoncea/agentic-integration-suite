from __future__ import annotations
import asyncio
import json
import os
from datetime import datetime, timezone
from pathlib import Path
import pytest
from agentic_suite.clients.groq_client import GroqClient, ModelResponse
from agentic_suite.config import get_settings
from agentic_suite.sdk.registry import ToolRegistry
from agentic_suite.tools.mock.calendar_tool import CalendarEventTool
from agentic_suite.tools.mock.github_tool import GitHubPRTool
from agentic_suite.tools.mock.jira_tool import JiraIssueTool
from agentic_suite.tools.mock.slack_tool import SlackMessageTool
from evals.schema import BenchmarkCase, load_benchmark
from evals.scoring import RunSummary, score_case

SYSTEM_PROMPT_PATH = Path(__file__).parent / 'data' / 'system_prompt.txt'
RESULTS_DIR = Path(__file__).parent / 'results'
MAX_CONCURRENCY = 4
REGRESSION_THRESHOLD = 0.0

def build_registry() -> ToolRegistry:
    registry = ToolRegistry()
    for tool in (JiraIssueTool, GitHubPRTool, SlackMessageTool, CalendarEventTool):
        registry.register(tool)

    return registry

def resolve_model() -> str:
    settings = get_settings()
    return os.environ.get('EVAL_MODEL') or settings.groq_model_full

async def run_case(case: BenchmarkCase, client: GroqClient, registry: ToolRegistry, system_prompt: str, semaphore: asyncio.Semaphore) -> ModelResponse:
    async with semaphore:
        return await client.complete(
            messages = [
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': case.user_prompt},
            ],
            tools = registry.get_schema_for_all()
        )

async def run_sweep(cases: list[BenchmarkCase], model: str) -> RunSummary:
    registry = build_registry()
    client = GroqClient(model=model)
    system_prompt = SYSTEM_PROMPT_PATH.read_text(encoding='utf-8')
    semaphore = asyncio.Semaphore(MAX_CONCURRENCY)
    responses = await asyncio.gather(*(run_case(c, client, registry, system_prompt, semaphore) for c in cases), return_exceptions=True)
    results = []

    for case, response in zip(cases, responses, strict=True):
        if isinstance(response, BaseException):
            from evals.scoring import CaseResult

            results.append(CaseResult(case_id=case.id, category=case.category.value, failure_reason=f'API error: {type(response).__name__}: {response}'))
            continue
        results.append(score_case(case, response, registry))

    return RunSummary(model=model, results=results)

def write_results(summary: RunSummary, label: str) -> Path:
    RESULTS_DIR.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    path = RESULTS_DIR / f'{label}-{summary.model}-{stamp}.json'
    path.write_text(json.dumps(summary.to_dict(), indent=2), encoding='utf-8')

    return path

def report(summary: RunSummary) -> str:
    lines = [
        '',
        f'Model:                {summary.model}',
        f'Cases:                {summary.total}',
        f'Passed:               {summary.passed} ({summary.pass_rate:.1%})',
        f'Selection accuracy:   {summary.selection_accuracy:.1%}',
        f'Argument precision:   {summary.argument_precision:.1%}',
        '',
        'By category:'
    ]
    for category, stats in summary.by_category().items():
        lines.append(f"  {category:<20} {stats['passed']}/{stats['total']} ({stats['pass_rate']:.1%})")

    failures = [r for r in summary.results if not r.passed]
    if failures:
        lines += ['', 'Failures:']
        lines += [f'  {r.case_id:<16} {r.failure_reason}' for r in failures]

    return '\n'.join(lines)

@pytest.mark.smoke
async def test_smoke_subset():
    cases = load_benchmark().smoke_subset()
    summary = await run_sweep(cases, resolve_model())
    path = write_results(summary, 'smoke')

    print(report(summary))
    print(f'\nResults: {path}')

    assert summary.pass_rate >= REGRESSION_THRESHOLD, report(summary)

@pytest.mark.sweep
async def test_full_sweep():
    cases = load_benchmark().cases
    summary = await run_sweep(cases, resolve_model())
    path = write_results(summary, 'full')

    print(report(summary))
    print(f'\nResults: {path}')

    assert summary.pass_rate >= REGRESSION_THRESHOLD, report(summary)