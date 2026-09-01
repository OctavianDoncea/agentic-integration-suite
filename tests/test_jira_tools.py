from __future__ import annotations
import pytest
from agentic_suite.tools.mock.jira_tool import JiraServerError, JiraIssueTool

async def test_happy_path_returns_created_issue():
    tool = JiraIssueTool(title='Fix login bug', priority='High', project_key='ENG')
    result = await tool.execute()

    assert result['status'] == 'created'
    assert result['key'] == 'ENG-123'
    assert result['title'] == 'Fix login bug'

async def test_injected_500_raises_expected_error_shape():
    tool = JiraIssueTool(title='Fix login bug', priority='High', project_key='ENG', inject_500_error=True)
    with pytest.raises(JiraServerError) as exc_info:
        await tool.execute()

    assert exc_info.value.status == 500

def test_inject_500_error_is_not_in_the_schema():
    props = JiraIssueTool.get_schema()['function']['parameters']['properties']
    assert 'inject_500_error' not in props