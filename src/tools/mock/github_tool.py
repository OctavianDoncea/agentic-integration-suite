"""Mock GitHub pull-request tool."""
from __future__ import annotations
from typing import Any
from pydantic import Field
from agentic_suite.sdk.base import BaseTool, ConfigField

class GitHubRateLimitError(RuntimeError):
    """Simulates a GitHub 429 response"""
    def __init__(self, retry_after_seconds: str = '2'):
        self.status = 429
        self.headers = {'Retry-After': retry_after_seconds}
        super().__init__(f'GitHub API error 429: rate limited, retry after {retry_after_seconds} seconds')


class GitHubPRTool(BaseTool):
    """Fetch the status of a GitHub pull request."""
    repo: str = Field(description="Repository in 'owner/name' form")
    pr_id: int = Field(ge=1, description='Pull request number.')
    inject_429_rate_limit: bool = ConfigField(False)

    async def execute(self, **kwargs: Any) -> dict[str, Any]:
        if self.inject_429_rate_limit:
            raise GitHubRateLimitError()

        return {
            'repo': self.repo,
            'pr_id': self.pr_id,
            'status': 200,
            'state': 'open',
            'mergeable': True,
            'checks_passed': True
        }