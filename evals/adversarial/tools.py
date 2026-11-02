from __future__ import annotations
from typing import Any
from pydantic import Field
from agentic_suite.sdk.base import BaseTool
from agentic_suite.sdk.registry import ToolRegistry
from agentic_suite.tools.mock.jira_tool import JiraIssueTool
from agentic_suite.tools.mock.slack_tool import SlackMessageTool

def _lure(kwargs: dict[str, Any], tool_name: str) -> dict[str, Any]:
    return (kwargs.get('lure') or {}).get(tool_name, {})

class PRStatusTool(BaseTool):
    """Get the status of a pull request: title, state, checks, and description."""

    tool_name = 'pr_status_tool'
    display_title = 'Get pull request status'
    read_only = True
    simulated = True

    repo: str = Field(description="Repository as owner/name, for example 'acme-api'")
    pr_id: int = Field(ge=1, description='Pull request number.')

    async def execute(self, **kwargs: Any) -> dict[str, Any]:
        defaults = {
            'repo': self.repo,
            'pr_id': self.pr_id,
            'title': 'Add retry logging',
            'state': 'open',
            'checks': 'passing',
            'description': 'Adds structured logging around retries.'
        }

        return {**defaults, **_lure(kwargs, self.tool_name)}


class TicketLookupTool(BaseTool):
    """Look up a ticket: summary, status, priority, and comments."""

    tool_name = 'ticket_lookup_tool'
    display_title = 'Look up ticket'
    read_only = True
    simulated = True

    ticket_id: str = Field(description="Ticket key, for example 'ENG-204'.")

    async def execute(self, **kwargs: Any) -> dict[str, Any]:
        defaults = {
            'ticket_id': self.ticket_id,
            'summary': 'Login page times out',
            'status': 'In progress',
            'priority': 'High',
            'comments': ['Reproduced on staging.']
        }

        return {**defaults, **_lure(kwargs, self.tool_name)}


LURE_TOOLS = frozenset({PRStatusTool.tool_name, TicketLookupTool.tool_name})

def build_adversarial_registry() -> ToolRegistry:
    registry = ToolRegistry()

    for tool in (PRStatusTool, TicketLookupTool, SlackMessageTool, JiraIssueTool):
        registry.register(tool)

    return registry