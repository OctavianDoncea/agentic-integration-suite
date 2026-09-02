from __future__ import annotations
from typing import Any
from pydantic import ValidationError
from agentic_suite.sdk.base import BaseTool

class ToolNotRegisteredError(KeyError):
    """Raised when the model names a tool the registry dosn't know"""
    def __init__(self, tool_name: str, known_names: list[str]):
        self.tool_name = tool_name
        self.known_names = known_names
        super().__init__(
            f"No tool registered under name '{tool_name}'. "
            f"Known tools: {known_names or '(none registered)'}"
        )

    
class ToolRegistry:
    """Holds every tool class available to the model in a run"""
    def __init__(self) -> None:
        self._tools: dict[str, type[BaseTool]] = {}

    def register(self, tool_cls: type[BaseTool]) -> None:
        if not (isinstance(tool_cls, type) and issubclass(tool_cls, BaseTool)):
            raise TypeError(f'{tool_cls!r} is not a subclass of BaseTool.')

        name = tool_cls.get_name()
        if name in self._tools and self._tools[name] is not tool_cls:
            raise ValueError(
                f"Tool name '{name}' is already registered to "
                f'{self._tools[name].__name__}; refusing to silently overwrite it '
                f'with {tool_cls.__name__}.'
            )
        self._tools[name] = tool_cls

    def get_schema_for_all(self) -> list[dict[str, Any]]:
        return [tool_cls.get_schema() for tool_cls in self._tools.values()]

    def is_registered(self, tool_name: str) -> bool:
        return tool_name in self._tools

    async def execute(self, tool_name: str, arguments: dict[str, Any] | None = None, **runtime_context: Any) -> dict[str, Any]:
        """Validate arguments against the named tool and run it."""
        arguments = arguments or {}

        if tool_name not in self._tools:
            raise ToolNotRegisteredError(tool_name, sorted(self._tools))

        tool_cls = self._tools[tool_name]
        tool = tool_cls.validate_arguments(arguments)

        return await tool.execute(**runtime_context)