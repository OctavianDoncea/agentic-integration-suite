"""The tool registry as an MCP server (stdio transport)."""

from __future__ import annotations
import asyncio
import importlib.metadata
import json
import logging
import sys
from collections.abc import AsyncIterator, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from typing import Any
from mcp.server import Server, ServerRequestContext
from mcp.server.stdio import stdio_server
from mcp.types import CallToolRequestParams, CallToolResult, ListToolsResult, PaginatedRequestParams, TextContent, Tool, ToolAnnotations
from sqlalchemy.orm import Session
from agentic_suite.agents.context import resolve_team_id
from agentic_suite.agents.executor import ToolExecutor, tool_error_payload
from agentic_suite.agents.registry import build_live_registry
from agentic_suite.sdk.base import BaseTool

SERVER_NAME = 'agentic-integration-suite'
SIMULATED_NOTE = ' (Simulated: returns mock data; no real system is changed.)'

logger = logging.getLogger('mcp_server')

ContextFactory = Callable[[], AbstractAsyncContextManager[dict[str, Any]]]

def _package_version() -> str:
    try:
        return importlib.metadata.version('agentic-integration-suite')
    except importlib.metadata.PackageNotFoundError:
        return '0.0.0'

# Tool listing
def tool_to_mcp(tool_cls: type[BaseTool]) -> Tool:
    """Translate a `BaseTool` subclass into an MCP `tool` definition."""

    function = tool_cls.get_schema()['function']
    description = function['description']

    if tool_cls.simulated:
        description += SIMULATED_NOTE

    hints: dict[str, Any] = {
        'read_only_hint': tool_cls.read_only,
        'open_world_hint': tool_cls.open_world
    }

    if not tool_cls.read_only:
        hints['destructive_hint'] = tool_cls.destructive
        hints['idempotent_hint'] = tool_cls.idempotent

    return Tool(name=function['name'], title=tool_cls.display_title, description=description, input_schema=function['parameters'], annotations=ToolAnnotations(**hints))

# Results
def _json_safe(value: dict[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(value, default=str))

def success_result(result: dict[str, Any]) -> CallToolResult:
    data = _json_safe(result)
    return CallToolResult(content=[TextContent(type='text', text=json.dumps(data))], structured_content=data)

def error_result(payload: dict[str, Any]) -> CallToolResult:
    data = _json_safe(payload)
    return CallToolResult(content=[TextContent(type='text', text=json.dumps(data))], structured_content=data, is_error=True)

# Server
def build_server(executor: ToolExecutor, context_factory: ContextFactory) -> Server:
    tools = [tool_to_mcp(tool_cls) for tool_cls in executor.registry.tools()]

    async def list_tools(ctx: ServerRequestContext, params: PaginatedRequestParams | None) -> ListToolsResult:
        return ListToolsResult(tools=tools)

    async def call_tool(ctx: ServerRequestContext, params: CallToolRequestParams) -> CallToolResult:
        name = params.name
        arguments = params.arguments or {}

        try:
            async with context_factory() as runtime_context:
                try:
                    result = await executor.execute(name, arguments, **runtime_context)
                except Exception as exc:
                    return error_result(tool_error_payload(exc))
        except Exception as exc:
            logger.exception('MCP call to %s failed outside the tool', name)
            return error_result({'error': 'server_error', 'type': type(exc).__name__})
        
        return success_result(result)

    return Server(SERVER_NAME, version=_package_version(), on_list_tools=list_tools, on_call_tool=call_tool)

# Runtime context
def make_database_context(session_factory: Callable[[], Session]) -> ContextFactory:
    @asynccontextmanager
    async def database_context() -> AsyncIterator[dict[str, Any]]:
        session = session_factory()

        try:
            yield {'session': session, 'team_id': resolve_team_id(session)}
            session.commit()
        except BaseException:
            session.rollback()
            raise
        finally:
            session.close()

    return database_context

# Entry point
async def _serve(server: Server) -> None:
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())

def main() -> None:
    from agentic_suite.db import get_session_factory
    from agentic_suite.logging_config import configure_logging

    configure_logging(stream=sys.stderr)
    executor = ToolExecutor(build_live_registry())
    server = build_server(executor, make_database_context(get_session_factory()))
    asyncio.run(_serve(server))

if __name__ == '__main__':
    main()