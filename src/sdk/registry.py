from __future__ import annotations
import time
from typing import Any
from pydantic import ValidationError
from agentic_suite.sdk.base import BaseTool
from agentic_suite.middleware.circuit_breaker import CircuitOpenError

class ToolNotRegisteredError(KeyError):
    """Raised when the model names a tool the registry doesn't know"""
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

    async def execute(self, tool_name: str, arguments: dict[str, Any] | None = None, *, retry_count: int = 0, **runtime_context: Any) -> dict[str, Any]:
        """Validate arguments against the named tool, run it, and log the outcome."""

        arguments = arguments or {}
        started = time.monotonic()

        def elapsed_ms() -> float:
            return (time.monotonic() - started) * 1000

        try:
            if tool_name not in self._tools:
                raise ToolNotRegisteredError(tool_name, sorted(self._tools))

            tool = self._tools[tool_name].validate_arguments(arguments)
            result = await tool.execute(**runtime_context)

        except ToolNotRegisteredError as exc:
            log_invocation(tool_name, arguments, elapsed_ms(), 'not_registered',
                           retry_count, error_type=type(exc).__name__, error_message=str(exc))
            raise
        except ValidationError as exc:
            log_invocation(tool_name, arguments, elapsed_ms(), 'validation_error',
                           retry_count, error_type=type(exc).__name__,
                           error_message=f'{exc.error_count()} validation error(s)')
            raise
        except CircuitOpenError as exc:
            log_invocation(tool_name, arguments, elapsed_ms(), 'circuit_open',
                           retry_count, error_type=type(exc).__name__, error_message=str(exc))
            raise
        except Exception as exc:
            log_invocation(tool_name, arguments, elapsed_ms(), 'tool_error',
                           retry_count, error_type=type(exc).__name__, error_message=str(exc))
            raise

        log_invocation(tool_name, arguments, elapsed_ms(), 'success', retry_count)
        return result