"""Mock Jira issue-creation tool."""
from __future__ import annotations
from typing import Any
from pydantic import Field
from agentic_suite.sdk.base import BaseTool, ConfigField

class JiraServerError(RuntimeError):
    """Simulates a Jira 5xx response."""
    def __init__(self, status: int = 500, message: str = 'Internal Server Error'):
        self.status = status
        self.message = message
        super().__init__(f'Jira API error {status}: {message}')


class JiraIssueTool(BaseTool):
    """Create a Jira issue with a title, priority, and project key"""
    title: str = Field(min_length=1, max_length=255, description='Issue summary/title.')
    priority: str = Field(description="Issue priority, e.g. 'Low', 'Medium', 'High'.")
    project_key: str = Field(min_length=2, max_length=10, description="Jira project key, e.g. 'ENG'.")
    inject_500_error: bool = ConfigField(False)

    async def execute(self, **kwargs: Any) -> dict[str, Any]:
        if self.inject_500_error:
            raise JiraServerError()

        return {
            'id': 'JIRA-123',
            'key': f'{self.project_key}-123',
            'status': 'created',
            'title': self.title,
            'priority': self.priority
        }