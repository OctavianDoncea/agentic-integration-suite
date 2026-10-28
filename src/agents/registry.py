"""The tools the live agent can use."""

from __future__ import annotations
from agentic_suite.integrations.slack.tools import SlackPostMessageTool
from agentic_suite.sdk.registry import ToolRegistry
from agentic_suite.tools.mock.calendar_tool import CalendarEventTool
from agentic_suite.tools.mock.github_tool import GitHubPRTool
from agentic_suite.tools.mock.jira_tool import JiraIssueTool

def build_live_registry() -> ToolRegistry:
    """Real Slack plus the mocks."""

    registry = ToolRegistry()
    for tool in (SlackPostMessageTool, JiraIssueTool, GitHubPRTool, CalendarEventTool):
        registry.register(tool)

    return registry