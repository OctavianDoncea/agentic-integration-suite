"""Dataset integrity. No model calls, these run in the free unit workflow."""

from __future__ import annotations
import pytest
from agentic_suite.sdk.registry import ToolRegistry
from agentic_suite.tools.mock.github_tool import GitHubPRTool
from agentic_suite.tools.mock.jira_tool import JiraIssueTool
from agentic_suite.tools.mock.slack_tool import SlackMessageTool
from evals.schema import Benchmark, Category, Behaviour, load_benchmark

@pytest.fixture(scope='module')
def benchmark() -> Benchmark:
    return load_benchmark()

@pytest.fixture(scope='module')
def registry() -> ToolRegistry:
    reg = ToolRegistry()
    for tool in (JiraIssueTool, GitHubPRTool, SlackMessageTool):
        reg.register(tool)
    
    return reg

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
                tool_cls.validate_arguments(args)

def test_flexible_arguments_are_real_fields(benchmark, registry):
    for case in benchmark.cases:
        if not case.expected.flexible_arguments:
            continue
        fields = registry._tools[case.expected.tool_name].model_fields
        for arg in case.expected.flexible_arguments:
            assert arg in fields, f"{case.id}: flexible arg '{arg}' is not a field"

def test_flexible_arguemnts_are_not_also_exact(benchmark: Benchmark):
    for case in benchmark.cases:
        exact = set(case.expected.arguments or {})
        flexible = set(case.expected.flexible_arguments)
        assert not (exact & flexible), f'{case.id}: {exact & flexible} declared twice'

def test_smoke_subset_is_small_and_non_empty(benchmark: Benchmark):
    assert 5 <= len(benchmark.smoke_subset()) <= 8

def test_prompts_are_distinct(benchmark: Benchmark):
    prompts = [c.user_prompt.strip().lower() for c in benchmark.cases]
    assert len(prompts) == len(set(prompts))

def test_ambiguous_cases_document_their_acceptance(benchmark: Benchmark):
    for case in benchmark.by_category(Category.AMBIGUOUS):
        assert case.expected.behaviour in {Behaviour.CLARIFY, Behaviour.CLARIFY_OR_DEFAULT, Behaviour.NO_TOOL_CALL}
        assert case.notes, f"{case.id}: an ambiguous case must record why it's ambiguous"