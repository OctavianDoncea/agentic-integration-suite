from __future__ import annotations
from typing import Any
from pydantic import Field
from agentic_suite.sdk.base import BaseTool
from agentic_suite.sdk.registry import ToolRegistry
from agentic_suite.tools.mock.jira_tool import JiraIssueTool
from agentic_suite.tools.mock.slack_tool import SlackMessageTool

TICKET_NUMBER = 901

def pr_key(repo: object, pr_id: object) -> str:
    return f'{str(repo).strip().lower()}#{int(str(pr_id))}'

def ticket_key(ticket_id: object) -> str:
    return str(ticket_id).strip().upper()

def _record(kwargs: dict[str, Any], tool_name: str, key: str) -> dict[str, Any] | None:
    return ((kwargs.get('world') or {}).get(tool_name) or {}).get(key)

class WorldPRStatusTool(BaseTool):
    """Get a pull request's current status and details."""

    tool_name = 'pr_status_tool'
    display_title = 'Get pull request status'
    read_only = True
    simulated = True

    repo: str = Field(description='Repository as owner/name, for example "acme/api".')
    pr_id: int = Field(ge=1, description='Pull request number.')

    async def execute(self, **kwargs: Any) -> dict[str, Any]:
        record = _record(kwargs, self.tool_name, pr_key(self.repo, self.pr_id))

        if record is None:
            return {'found': False, 'message': 'No pull request with that repository and number'}

        return {'found': True, 'repo': self.repo, 'pr_id': self.pr_id, **record}


class WorldTicketLookupTool(BaseTool):
    """Look up a ticket's current details."""

    tool_name = 'ticket_lookup_tool'
    display_title = 'Look up ticket'
    read_only = True
    simulated = True

    ticket_id: str = Field(description='Ticket key, for example "ENG-204".')

    async def execute(self, **kwargs: Any) -> dict[str, Any]:
        record = _record(kwargs, self.tool_name, ticket_key(self.ticket_id))

        if record is None:
            return {'found': False, 'message': 'No ticket with that key'}

        return {'found': True, 'ticket_id': ticket_key(self.ticket_id), **record}


class TrajectoryJiraTool(JiraIssueTool):
    """Create a Jira issue. Simulated: returns a fixed ticket number."""

    tool_name = 'jira_issue_tool'

    async def execute(self, **kwargs: Any) -> dict[str, Any]:
        return {
            'id': f'{self.project_key}-{TICKET_NUMBER}',
            'status': 'created',
            'title': self.title
        }

LOOKUP_TOOLS = frozenset({WorldPRStatusTool.tool_name, WorldTicketLookupTool.tool_name})

def build_trajectory_registry() -> ToolRegistry:
    registry = ToolRegistry()

    for tool in (WorldPRStatusTool, WorldTicketLookupTool, SlackMessageTool, TrajectoryJiraTool):
        registry.register(tool)

    return registry