from __future__ import annotations
import pytest
from pydantic import ValidationError
from agentic_suite.sdk.base import BaseTool
from agentic_suite.sdk.registry import ToolRegistry, ToolNotRegisteredError
from agentic_suite.tools.mock.github_tool import GitHubPRTool
from agentic_suite.tools.mock.jira_tool import JiraIssueTool

@pytest.fixture
def registry() -> ToolRegistry:
    reg = ToolRegistry()
    reg.register(JiraIssueTool)
    reg.register(GitHubPRTool)
    return reg

def test_get_schema_for_all_returns_both_tools(registry: ToolRegistry):
    schemas = registry.get_schema_for_all()
    names = {s['function']['name'] for s in schemas}
    assert names == {'jira_issue_tool', 'git_hub_pr_tool'}

def test_is_registered(registry: ToolRegistry):
    assert registry.is_registered('jira_issue_tool')
    assert not registry.is_registered('nonexistent_tool')

async def test_execute_dispatches_to_correct_tool(registry: ToolRegistry):
    result = await registry.execute('jira_issue_tool', {'title': 'Bug', 'priority': 'High', 'project_key': 'ENG'})
    assert result['status'] == 'created'

async def test_execute_unregistered_name_raises_clear_error(registry: ToolRegistry):
    with pytest.raises(ToolNotRegisteredError) as exc_info:
        await registry.execute('nonexistent_tool', {})

    assert exc_info.value.tool_name == 'nonexistent_tool'
    assert 'jira_issue_tool' in exc_info.value.known_names

async def test_execute_invalid_arguments_raises_validation_error(registry: ToolRegistry):
    with pytest.raises(ValidationError):
        await registry.execute('jira_issue_tool', {'title': 'Bug'})

def test_register_rejects_non_tool_classes(registry: ToolRegistry):
    with pytest.raises(TypeError):
        registry.register(dict)

def test_register_same_class_twice_is_idempotent(registry: ToolRegistry):
    registry.register(JiraIssueTool)
    assert len(registry.get_schema_for_all()) == 2

def test_register_name_collision_between_different_classes_raises():
    reg = ToolRegistry()
    reg.register(JiraIssueTool)

    class JiraIssueTool2(BaseTool):
        tool_name = 'jira_issue_tool'
        x: str

        async def execute(self, **kwargs):
            return {}

    
    with pytest.raises(ValueError, match='already registered'):
        reg.register(JiraIssueTool2)

async def test_runtime_context_is_passed_to_execute_but_not_validated():
    class ContextAwareTool(BaseTool):
        value: str

        async def execute(self, **kwargs):
            return {'value': self.value, 'db_session': kwargs.get('db_session')}

    reg = ToolRegistry()
    reg.register(ContextAwareTool)
    result = await reg.execute('context_aware_tool', {'value': 'x'}, db_session='fake-session_object')
    assert result['db_session'] == 'fake-session_object'