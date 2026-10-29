from __future__ import annotations
import asyncio
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
import pytest
from agentic_suite.agents.prompt import load_system_prompt
from agentic_suite.clients.base import ChatClient, ModelResponse
from agentic_suite.clients.factory import bare_model, infer_provider, make_chat_client
from agentic_suite.config import get_settings
from agentic_suite.middleware.retry import with_retry
from agentic_suite.sdk.registry import ToolRegistry
from agentic_suite.tools.mock.calendar_tool import CalendarEventTool
from agentic_suite.tools.mock.github_tool import GitHubPRTool
from agentic_suite.tools.mock.jira_tool import JiraIssueTool
from agentic_suite.tools.mock.slack_tool import SlackMessageTool
from evals.schema import BenchmarkCase, load_benchmark
from evals.scoring import CaseResult, RunSummary, score_case
from evals.throttle import MinIntervalLimiter, resolve_rpm

RESULTS_DIR = Path(__file__).parent / "results"
MAX_CONCURRENCY = 1
MAX_ATTEMPTS = 5
MAX_WAIT_SECONDS = 60.0

def build_registry() -> ToolRegistry:
    registry = ToolRegistry()
    for tool in (JiraIssueTool, GitHubPRTool, SlackMessageTool, CalendarEventTool):
        registry.register(tool)
    return registry

def resolve_model() -> str:
    return os.environ.get('EVAL_MODEL') or get_settings().groq_model_full

def resolve_provider() -> str | None:
    return os.environ.get('EVAL_PROVIDER') or None

async def run_case(case: BenchmarkCase, client: ChatClient, registry: ToolRegistry, system_prompt: str, semaphore: asyncio.Semaphore, limiter: MinIntervalLimiter) -> ModelResponse:
    @with_retry(max_attempts=MAX_ATTEMPTS, max_wait=MAX_WAIT_SECONDS)
    async def call() -> ModelResponse:
        await limiter.wait()
        return await client.complete(
            messages=[
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': case.user_prompt},
            ],
            tools=registry.get_schema_for_all(),
        )

    async with semaphore:
        return await call()

async def run_sweep(cases: list[BenchmarkCase], model: str, *, provider: str | None = None) -> RunSummary:
    provider = provider or infer_provider(model)
    registry = build_registry()
    client = make_chat_client(model, provider)
    system_prompt = load_system_prompt()
    semaphore = asyncio.Semaphore(MAX_CONCURRENCY)
    limiter = MinIntervalLimiter(resolve_rpm(provider))

    responses = await asyncio.gather(
        *(run_case(c, client, registry, system_prompt, semaphore, limiter) for c in cases),
        return_exceptions=True,
    )

    results: list[CaseResult] = []
    for case, response in zip(cases, responses, strict=True):
        if isinstance(response, BaseException):
            results.append(
                CaseResult(
                    case_id=case.id,
                    category=case.category.value,
                    failure_reason=f'API error: {type(response).__name__}: {response!r}',
                )
            )
            continue
        results.append(score_case(case, response, registry))

    return RunSummary(model=model, results=results)

def write_results(summary: RunSummary, label: str) -> Path:
    RESULTS_DIR.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    safe_model = re.sub(r'[^\w.-]+', '_', summary.model)
    path = RESULTS_DIR / f'{label}-{safe_model}-{stamp}.json'
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
        'By category:',
    ]
    for category, stats in summary.by_category().items():
        lines.append(
            f"  {category:<20} {stats['passed']}/{stats['total']} ({stats['pass_rate']:.1%})"
        )

    failures = [r for r in summary.results if not r.passed]
    if failures:
        lines += ['', 'Failures:']
        lines += [f'  {r.case_id:<16} {r.failure_reason}' for r in failures]
    return '\n'.join(lines)

# Sweeps
@pytest.mark.smoke
async def test_smoke_subset():
    """Runs on every push against the cheap model. Guards against gross breakage."""
    from evals.threshold import BASELINE_SMOKE_MODEL, SMOKE_THRESHOLD

    model = resolve_model()
    summary = await run_sweep(
        load_benchmark().smoke_subset(), model, provider=resolve_provider()
    )

    path = write_results(summary, 'smoke')
    print(report(summary))
    print(f'\nResults: {path}')

    if bare_model(model) != BASELINE_SMOKE_MODEL:
        pytest.skip(f'Exploratory run: threshold was measured on {BASELINE_SMOKE_MODEL}.')

    print(f'Threshold: {SMOKE_THRESHOLD:.1%}')
    assert summary.pass_rate >= SMOKE_THRESHOLD, report(summary)

@pytest.mark.sweep
async def test_full_sweep():
    """The full benchmark. Scheduled nightly and on manual dispatch, not per push."""
    from evals.threshold import BASELINE_FULL_MODEL, FULL_THRESHOLD

    model = resolve_model()
    summary = await run_sweep(load_benchmark().cases, model, provider=resolve_provider())

    path = write_results(summary, 'full')
    print(report(summary))
    print(f'\nResults: {path}')

    if bare_model(model) != BASELINE_FULL_MODEL:
        pytest.skip(f'Exploratory run: threshold was measured on {BASELINE_FULL_MODEL}.')

    print(f'Threshold: {FULL_THRESHOLD:.1%}')
    assert summary.pass_rate >= FULL_THRESHOLD, report(summary)