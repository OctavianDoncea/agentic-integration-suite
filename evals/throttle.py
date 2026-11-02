"""Space requests to stay under a provider's per-minute limit."""
from __future__ import annotations
import asyncio
import os
import time
from collections.abc import Callable, Awaitable
from typing import Any
from agentic_suite.clients.factory import PROVIDERS
from agentic_suite.clients.base import ChatClient, ModelResponse, ToolCall

def resolve_rpm(provider: str) -> float | None:
    override = os.environ.get('EVAL_RPM', '').strip()

    if override:
        value = float(override)
        return value or None

    spec = PROVIDERS.get(provider)
    return spec.default_rpm if spec else None

class MinIntervalLimiter:
    """Ensures at least 60/rpm seconds between consecutive `wait()` returns."""

    def __init__(self, requests_per_minute: float | None, *, clock: Callable[[], float] = time.monotonic, sleep: Callable[[float], Awaitable[None]] = asyncio.sleep) -> None:
        self._interval = 60.0 / requests_per_minute if requests_per_minute else 0.0
        self._clock = clock
        self._sleep = sleep
        self._next_allowed: float | None = None
        self._lock = asyncio.Lock()

    async def wait(self) -> None:
        if not self._interval:
            return

        async with self._lock:
            now = self._clock()

            if self._next_allowed is not None and now < self._next_allowed:
                await self._sleep(self._next_allowed - now)
                now = self._next_allowed

            self._next_allowed = now + self._interval


class ThrottledClient:
    """Spaces every model call, including the ones made inside the agent loop."""

    def __init__(self, inner: ChatClient, limiter: MinIntervalLimiter) -> None:
        self._inner = inner
        self._limiter = limiter

    @property
    def model(self) -> str:
        return self._inner.model

    async def complete(self, messages: list[Any], tools: list[dict[str, Any]] | None = None, **kwargs: Any) -> ModelResponse:
        await self._limiter.wait()
        return await self._inner.complete(messages, tools, **kwargs)

    def format_assistant_turn(self, response: ModelResponse) -> Any:
        return self._inner.format_assistant_turn(response)

    def format_tool_result(self, call: ToolCall, content: Any) -> Any:
        return self._inner.format_tool_result(call, content)