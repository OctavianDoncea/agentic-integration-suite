"""Dataset integrity. No model calls, these run in the free unit workflow."""

from __future__ import annotations
from datetime import datetime
from typing import Any, get_args, get_origin
import pytest
from agentic_suite.sdk.base import BaseTool
from agentic_suite.sdk.registry import ToolRegistry
from agentic_suite.tools.mock.calendar_tool import CalendarEventTool
from agentic_suite.tools.mock.github_tool import GitHubPRTool
from agentic_suite.tools.mock.jira_tool import JiraIssueTool
from agentic_suite.tools.mock.slack_tool import SlackMessageTool
from evals.schema import Behaviour, Benchmark, Category, load_benchmark

EXPECTED_DISTRIBUTION = {
    Category.DIRECT_INTENT: 15,
    Category.AMBIGUOUS: 15,
    Category.EDGE_CASE_SCHEMA: 20,
}

@pytest.fixture(scope='module')
def benchmark() -> Benchmark:
    return load_benchmark()

@pytest.fixture(scope='module')
def registry() -> ToolRegistry:
    reg = ToolRegistry()
    for tool in (JiraIssueTool, GitHubPRTool, SlackMessageTool, CalendarEventTool):
        reg.register(tool)

    return reg

def _placeholder(tool_cls: type[BaseTool], field_name: str) -> Any:
    """A value that satisfies `field_name` so partial expected payloads can be schema-checked."""
    annotation = tool_cls.model_fields[field_name].annotation
    origin = get_origin(annotation)
    if origin is list:
        return ['placeholder@example.com']
    if annotation is datetime:
        return '2026-03-14T09:30:00Z'
    if annotation is int:
        return 10
    if annotation is bool:
        return True
    literal_values = get_args(annotation)
    if literal_values and all(isinstance(v, str) for v in literal_values):
        return literal_values[0]
    if field_name == 'channel':
        return '#placeholder'
    return 'placeholder'

def _named_tool(case) -> str | None:
    return case.expected.tool_name or case.expected.default_tool_name

def test_benchmark_loads_and_validates(benchmark: Benchmark):
    assert benchmark.cases

def test_every_named_tool_is_registered(benchmark: Benchmark, registry: ToolRegistry):
    for case in benchmark.cases:
        for name in (case.expected.tool_name, case.expected.default_tool_name):
            if name:
                assert registry.is_registered(name), f'{case.id} names unknown tool {name}'

def test_expected_arguments_validate_against_their_tool(benchmark, registry):
    for case in benchmark.cases:
        for name, args in (
            (case.expected.tool_name, case.expected.arguments),
            (case.expected.default_tool_name, case.expected.default_arguments),
        ):
            if name and args is not None:
                tool_cls = registry._tools[name]
                payload = dict(args)
                for flex in case.expected.flexible_arguments:
                    payload.setdefault(flex, _placeholder(tool_cls, flex))
                tool_cls.validate_arguments(payload)

def test_flexible_arguments_are_real_fields(benchmark, registry):
    for case in benchmark.cases:
        if not case.expected.flexible_arguments:
            continue
        tool_name = _named_tool(case)
        assert tool_name, f"{case.id}: flexible_arguments without a tool"
        fields = registry._tools[tool_name].model_fields
        for arg in case.expected.flexible_arguments:
            assert arg in fields, f"{case.id}: flexible arg '{arg}' is not a field"

def test_flexible_arguments_are_not_also_exact(benchmark: Benchmark):
    for case in benchmark.cases:
        exact = set(case.expected.arguments or {}) | set(case.expected.default_arguments or {})
        flexible = set(case.expected.flexible_arguments)
        assert not (exact & flexible), f'{case.id}: {exact & flexible} declared twice'

def test_smoke_subset_is_small_and_non_empty(benchmark: Benchmark):
    assert 5 <= len(benchmark.smoke_subset()) <= 8

def test_prompts_are_distinct(benchmark: Benchmark):
    prompts = [c.user_prompt.strip().lower() for c in benchmark.cases]
    assert len(prompts) == len(set(prompts))

def test_ambiguous_cases_document_their_acceptance(benchmark: Benchmark):
    for case in benchmark.by_category(Category.AMBIGUOUS):
        assert case.expected.behaviour in {
            Behaviour.CLARIFY,
            Behaviour.CLARIFY_OR_DEFAULT,
            Behaviour.NO_TOOL_CALL,
        }
        assert case.notes, f"{case.id}: an ambiguous case must record why it's ambiguous"

def test_category_distribution(benchmark: Benchmark):
    for category, expected in EXPECTED_DISTRIBUTION.items():
        assert len(benchmark.by_category(category)) == expected

def test_total_case_count(benchmark: Benchmark):
    assert len(benchmark.cases) == 50

def test_every_tool_appears_in_the_benchmark(benchmark: Benchmark, registry: ToolRegistry):
    exercised = {name for case in benchmark.cases for name in (case.expected.tool_name, case.expected.default_tool_name) if name}
    for registered in registry.get_schema_for_all():
        assert registered['function']['name'] in exercised

def test_no_case_depends_on_the_current_date(benchmark: Benchmark):
    relative = ('tomorrow', 'today', 'next week', 'yesterday', 'this friday', 'last week')
    for case in benchmark.cases:
        if any(word in case.user_prompt.lower() for word in relative):
            args = case.expected.arguments or {}
            time_fields = {'start_time', 'date', 'due_date'}
            pinned = time_fields & set(args)
            assert not pinned, (
                f'{case.id} has a relative date in the prompt but pins {pinned}; mark those flexible or the case expires.'
            )