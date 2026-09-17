from __future__ import annotations
import pytest
from agentic_suite.clients.groq_client import ModelResponse, ToolCall
from agentic_suite.sdk.registry import ToolRegistry
from agentic_suite.tools.mock.github_tool import GitHubPRTool
from agentic_suite.tools.mock.jira_tool import JiraIssueTool
from evals.schema import Behaviour, BenchmarkCase, Category, ExpectedOutcome
from evals.scoring import RunSummary, score_case

@pytest.fixture
def registry() -> ToolRegistry:
    reg = ToolRegistry()
    reg.register(JiraIssueTool)
    reg.register(GitHubPRTool)

    return reg

def make_case(expected: ExpectedOutcome, category=Category.DIRECT_INTENT) -> BenchmarkCase:
    return BenchmarkCase(id='t-001', category=category, user_prompt='...', expected=expected, notes='test')

def tool_response(name: str, arguments: dict) -> ModelResponse:
    return ModelResponse(tool_calls=[ToolCall(name=name, arguments=arguments)])

JIRA_ARGS = {'title': 'Fix login', 'priority': 'High', 'project_key': 'ENG'}
JIRA_EXPECTED = ExpectedOutcome(behaviour=Behaviour.TOOL_CALL, tool_name='jira_issue_tool', arguments=JIRA_ARGS)

def test_exact_match_passes(registry):
    result = score_case(make_case(JIRA_EXPECTED), tool_response('jira_issue_tool', JIRA_ARGS), registry)
    assert result.passed
    assert result.selection_correct
    assert result.arguments_valid
    assert result.arguments_match

def test_wrong_tool_fails_selection(registry):
    result = score_case(make_case(JIRA_EXPECTED), tool_response('github_pr_tool', {'repo': 'a/b', 'pr_id': 1}), registry)
    assert not result.selection_correct
    assert result.arguments_match is None
    assert "expected 'jira_issue_tool'" in result.failure_reason

def test_no_tool_call_when_one_was_expected(registry):
    result = score_case(make_case(JIRA_EXPECTED), ModelResponse(text='Sure!'), registry)
    assert not result.passed
    assert 'no tool called' in result.failure_reason

def test_wrong_argument_value_fails_precision(registry):
    result = score_case(make_case(JIRA_EXPECTED), tool_response('jira_issue_tool', {**JIRA_ARGS, 'priority': 'Urgent'}), registry)

    assert result.selection_correct
    assert result.arguments_valid
    assert not result.arguments_match
    assert "'priority'" in result.failure_reason

def test_missing_argument_fails_schema_validation(registry):
    result = score_case(make_case(JIRA_EXPECTED), tool_response('jira_issue_tool', {'title': 'x'}), registry)
    assert result.selection_correct
    assert result.arguments_valid is False
    assert 'schema validation failed' in result.failure_reason

def test_type_mismatch_is_caught(registry):
    expected = ExpectedOutcome(behaviour=Behaviour.TOOL_CALL, tool_name='github_pr_tool', arguments={'repo': 'a/b', 'pr_id': 471})
    result = score_case(make_case(expected), tool_response('github_pr_tool', {'repo': 'a/b', 'pr_id': '471'}), registry)
    assert result.arguments_valid
    assert not result.arguments_match

def test_unparseable_arguments_fail(registry):
    response = ModelResponse(tool_calls=[ToolCall(
        name='jira_issue_tool',
        arguments={},
        raw_arguments="{title: 'unquoted'}",
        parse_error='Expected property name'
    )])
    result = score_case(make_case(JIRA_EXPECTED), response, registry)
    assert result.selection_correct
    assert not result.arguments_valid
    assert 'not valid JSON' in result.failure_reason

def test_flexible_argument_accepts_any_non_empty_value(registry):
    expected = ExpectedOutcome(behaviour=Behaviour.TOOL_CALL, tool_name='jira_issue_tool', arguments={'priority': 'High', 'project_key': 'ENG'}, flexible_arguments=['title'])
    for title in ('Fix login', 'Login redirect loop bug', 'Resolve authentication issues'):
        result = score_case(make_case(expected), tool_response('jira_issue_tool', {**JIRA_ARGS, 'title': title}), registry)
        assert result.passed, title

def test_empty_flexible_argument_fails(registry):
    expected = ExpectedOutcome(behaviour=Behaviour.TOOL_CALL, tool_name='jira_issue_tool', arguments={'priority': 'High', 'project_key': 'ENG'}, flexible_arguments=['title'])
    result = score_case(make_case(expected), tool_response('jira_issue_tool', {**JIRA_ARGS, 'title': ''}), registry)
    assert not result.passed

def test_no_tool_call_passes_when_no_tool_is_called(registry):
    expected = ExpectedOutcome(behaviour=Behaviour.NO_TOOL_CALL)
    result = score_case(make_case(expected, Category.AMBIGUOUS), ModelResponse(text='We generally avoid Friday deploys.'), registry)
    assert result.passed

def test_no_tool_call_fails_when_a_tool_is_called(registry):
    expected = ExpectedOutcome(behaviour=Behaviour.NO_TOOL_CALL)
    result = score_case(make_case(expected, Category.AMBIGUOUS), tool_response('jira_issue_tool', JIRA_ARGS), registry)
    assert not result.passed
    assert 'no tool applies' in result.failure_reason

def test_clarify_passes_on_a_question(registry):
    expected = ExpectedOutcome(behaviour=Behaviour.CLARIFY)
    result = score_case(make_case(expected, Category.AMBIGUOUS), ModelResponse(text='Which project should I file this under?'), registry)
    assert result.passed

def test_clarify_fails_on_a_statement(registry):
    expected = ExpectedOutcome(behaviour=Behaviour.CLARIFY)
    result = score_case(make_case(expected, Category.AMBIGUOUS), ModelResponse(text="I'll file that shortly."), registry)
    assert not result.passed

def test_clarify_fails_if_a_tool_is_called_instead(registry):
    expected = ExpectedOutcome(behaviour=Behaviour.CLARIFY)
    result = score_case(make_case(expected, Category.AMBIGUOUS), tool_response('jira_issue_tool', JIRA_ARGS), registry)
    assert not result.passed
    assert 'instead of asking' in result.failure_reason

@pytest.fixture
def clarify_or_default() -> ExpectedOutcome:
    return ExpectedOutcome(behaviour=Behaviour.CLARIFY_OR_DEFAULT, default_tool_name='jira_issue_tool', default_arguments={'project_key': 'ENG', 'priority': 'High'}, flexible_arguments=['title'])

def test_default_branch_passes(registry, clarify_or_default):
    result = score_case(make_case(clarify_or_default, Category.AMBIGUOUS), tool_response('jira_issue_tool', JIRA_ARGS), registry)
    assert result.passed

def test_clarify_branch_passes(registry, clarify_or_default):
    result = score_case(make_case(clarify_or_default, Category.AMBIGUOUS), ModelResponse(text='Which project key should I use?'), registry)
    assert result.passed

def test_wrong_default_arguments_still_fail(registry, clarify_or_default):
    result = score_case(make_case(clarify_or_default, Category.AMBIGUOUS), tool_response('jira_issue_tool', {**JIRA_ARGS, 'project_key': 'RANDOM'}), registry)
    assert not result.passed
    assert "'project_key'" in result.failure_reason

def test_neither_branch_fails(registry, clarify_or_default):
    result = score_case(make_case(clarify_or_default, Category.AMBIGUOUS), ModelResponse(text='Done.'), registry)
    assert not result.passed
    assert 'neither' in result.failure_reason

def _result(passed: bool, category='direct_intent', scored_args=True):
    from evals.scoring import CaseResult

    return CaseResult(
        case_id='x', 
        category=category,
        selection_correct=passed,
        arguments_valid=passed if scored_args else None,
        arguments_match=passed if scored_args else None
    )

def test_summary_rates():
    summary = RunSummary(model='m', results=[_result(True), _result(True), _result(False)])
    assert summary.pass_rate == pytest.approx(2/3)
    assert summary.selection_accuracy == pytest.approx(2/3)

def test_argument_precision_excludes_unscored_cases():
    summary = RunSummary(model='m', results=[_result(True), _result(False), _result(True, scored_args=False)])
    assert summary.argument_precision == pytest.approx(0.5)

def test_by_category_buckets():
    summary = RunSummary(model='m', results=[_result(True, 'direct_intent'), _result(False, 'direct_intent'), _result(True, 'ambiguous')])
    buckets = summary.by_category()
    assert buckets['direct_intent']['pass_rate'] == 0.5
    assert buckets['ambiguous']['pass_rate'] == 1.0