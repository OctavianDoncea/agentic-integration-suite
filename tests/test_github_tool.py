from __future__ import annotations
import pytest
from agentic_suite.tools.mock.github_tool import GitHubPRTool, GitHubRateLimitError

async def test_happy_path_returns_pr_status():
    tool = GitHubPRTool(repo='owner/repo', pr_id=123)
    result = await tool.execute()

    assert result['state'] == 'open'
    assert result['pr_id'] == 123

async def test_429_injection_carries_retry_after_header():
    tool = GitHubPRTool(repo='owner/repo', pr_id=123, inject_429_rate_limit=True)

    with pytest.raises(GitHubRateLimitError) as exc_info:
        await tool.execute()

    assert exc_info.value.status == 429
    assert exc_info.value.headers['Retry-After'] == '2'

async def test_custom_retry_after_value_is_honored():
    with pytest.raises(GitHubRateLimitError) as exc_info:
        raise GitHubRateLimitError(retry_after_seconds='10')
    
    assert exc_info.value.headers['Retry-After'] == '10'