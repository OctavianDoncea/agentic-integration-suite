"""One execution path for every too call, whoever asks for it."""

from __future__ import annotations
import asyncio
import time
from collections.abc import Callable, Awaitable
from typing import Any
from pydantic import ValidationError
from agentic_suite.integrations.slack.oauth import InstallationNotFoundError, ReauthRequiredError
from agentic_suite.middleware.circuit_breaker import CircuitBreaker, CircuitOpenError
from agentic_suite.middleware.retry import RetryExhaustedError, RetryTracker, with_retry
from agentic_suite.middleware.telemetry import log_invocation
from agentic_suite.sdk.registry import ToolNotRegisteredError, ToolRegistry

class ToolExecutor:
    """Runs tool calls through breaker, retry, and registry."""

    def __init__(
        self,
        registry: ToolRegistry,
        *,
        max_attempts: int = 3,
        failure_threshold: int = 3,
        cooldown_seconds: float = 30.0,
        base_context: dict[str, Any] | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[], Awaitable[None]] = asyncio.sleep
    ) -> None:
        self._registry = registry
        self._max_attempts = max_attempts
        self._failure_threshold = failure_threshold
        self._cooldown_seconds = cooldown_seconds
        self._base_context = dict(base_context or {})
        self._clock = clock
        self._sleep = sleep
        self._breakers: dict[str, CircuitBreaker] = {}

    @property
    def registry(self) -> ToolRegistry:
        return self._registry

    @property
    def breaker_names(self) -> list[str]:
        return sorted(self._breakers)

    def breaker_for(self, tool_name: str) -> CircuitBreaker:
        """One breaker per tool: a Slack outage must not shed Jira calls."""

        breaker = self._breakers.get(tool_name)
        if breaker is None:
            breaker = CircuitBreaker(name=tool_name, failure_threshold=self._failure_threshold, cooldown_seconds=self._cooldown_seconds, clock=self._clock)
            self._breakers[tool_name] = breaker

        return breaker

    async def execute(self, tool_name: str, arguments: dict[str, Any] | None = None, **runtime_context: Any) -> dict[str, Any]:
        arguments = arguments or {}
        context = {**self._base_context, **runtime_context}

        if not self._registry.is_registered(tool_name):
            return await self._registry.execute(tool_name, arguments, **context)

        tracker = RetryTracker()

        @with_retry(max_attempts=self._max_attempts, tracker=tracker, sleep=self._sleep)
        async def attempt() -> dict[str, Any]:
            return await self._registry.execute(tool_name, arguments, retry_count=tracker.retries, **context)

        try:
            return await self.breaker_for(tool_name).call(attempt)
        except CircuitOpenError as exc:
            log_invocation(tool_name, arguments, 0.0, 'circuit_open', 0, error_type=type(exc).__name__, error_message=str(exc))
            raise


def tool_error_payload(exc: BaseException) -> dict[str, Any]:
    """Turn a tool failure into a result the model can read and act on."""

    if isinstance(exc, ValidationError):
        return {
            'error': 'invalid_arguments',
            'details': [
                {
                    'field': '.'.join(str(part) for part in err['loc']) or 'root',
                    'problem': err['msg']
                }
                for err in exc.errors()[:5]
            ],
            'hint': 'Fix the listed arguments and call the tool again.'
        }

    if isinstance(exc, ToolNotRegisteredError):
        return {'error': 'unknown_tool', 'available_tools': exc.known_names}

    if isinstance(exc, CircuitOpenError):
        return {
            'error': 'service_unavailable',
            'retry_after_seconds': round(exc.seconds_until_retry, 1),
            'hint': "The service is failing. Tell the user; don't retry right now."
        }

    if isinstance(exc, RetryExhaustedError):
        return {
            'error': 'upstream_failure',
            'attempts': exc.attempts,
            'cause': type(exc.last_exception).__name__
        }

    if isinstance(exc, ReauthRequiredError):
        return {
            'error': 'reauthorization_required',
            'hint': 'The Slack app must be reinstalled in this workspace.'
        }

    if isinstance(exc, InstallationNotFoundError):
        return {
            'error': 'not_installed',
            'hint': 'No Slack workspace is connected. Install the app via /slack/install.'
        }

    payload: dict[str, Any] = {'error': 'tool_failed', 'type': type(exc).__name__}
    detail = getattr(exc, 'slack_error', None)

    if detail:
        payload['details'] = [detail]

    return payload