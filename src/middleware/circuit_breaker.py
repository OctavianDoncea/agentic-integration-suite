from __future__ import annotations
import functools
import logging
import time
from collections.abc import Awaitable, Callable
from enum import Enum
from typing import Any, TypeVar
from agentic_suite.middleware.retry import RetryExhaustedError, is_retryable

logger = logging.getLogger('middleware.circuit_breaker')

T = TypeVar('T')

DEFAULT_FAILURE_THRESHOLD = 3
DEFAULT_COOLDOWN_SECONDS = 30.0

ClockFn = Callable[[], float]

class CircuitState(str, Enum):
    CLOSED = 'closed'
    OPEN = 'open'
    HALF_OPEN = 'half-open'


class CircuitOpenError(RuntimeError):
    def __init__(self, name: str, seconds_until_retry: float):
        self.name = name
        self.seconds_until_retry = seconds_until_retry
        super().__init__(f"Circuit '{name}' is open; rejecting call. Retry in {seconds_until_retry:.1f}s.")


class CircuitBreaker:
    def __init__(
        self,
        name: str = 'default',
        *,
        failure_threshold: int = DEFAULT_FAILURE_THRESHOLD,
        cooldown_seconds: float = DEFAULT_COOLDOWN_SECONDS,
        countable: Callable[[BaseException], bool] = is_retryable,
        clock: ClockFn = time.monotonic,
    ) -> None:
        if failure_threshold < 1:
            raise ValueError('failure_threshold must be at least 1')
        if cooldown_seconds < 0:
            raise ValueError('cooldown_seconds must not be negative')

        self.name = name
        self.failure_threshold = failure_threshold
        self.cooldown_seconds = cooldown_seconds
        self._countable = countable
        self._clock = clock

        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._opened_at: float | None = None
        self._trial_in_flight = False

    @property
    def state(self) -> CircuitState:
        return self._state

    @property
    def failure_count(self) -> int:
        return self._failure_count

    def seconds_until_retry(self) -> float:
        if self._state is not CircuitState.OPEN or self._opened_at is None:
            return 0.0
        
        return max(0.0, self.cooldown_seconds - (self._clock() - self._opened_at))

    def _cooldown_elapsed(self) -> bool:
        if self._opened_at is None:
            return True

        return self._clock() - self._opened_at >= self.cooldown_seconds

    def _to_open(self) -> None:
        self._state = CircuitState.OPEN
        self._opened_at = self._clock()
        self._trial_in_flight = False
        logger.warning(f"Circuit '{self.name}' OPEN after {self._failure_count} consecutive failures; rejecting calls for {self.cooldown_seconds:.0f}s.")

    def _to_closed(self) -> None:
        if self._state is not CircuitState.CLOSED:
            logger.info(f"Circuit '{self.name}' CLOSED; dependency recovered")
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._opened_at = None
        self._trial_in_flight = False

    def _to_half_open(self) -> None:
        self._state = CircuitState.HALF_OPEN
        self._trial_in_flight = False
        logger.info(f"Circuit '{self.name}' HALF_OPEN; admitting one trial call")

    def reset(self) -> None:
        self._to_closed()

    def _admit(self) -> None:
        if self._state is CircuitState.OPEN:
            if not self._cooldown_elapsed():
                raise CircuitOpenError(self.name, self.seconds_until_retry())
            self._to_half_open()

        if self._state is CircuitState.HALF_OPEN:
            if self._trial_in_flight:
                raise CircuitOpenError(self.name, self.cooldown_seconds)
            self._trial_in_flight = True

    def _record_success(self) -> None:
        self._to_closed()

    def _record_failure(self, exc: BaseException) -> None:
        counted = exc.last_exception if isinstance(exc, RetryExhaustedError) else exc
        if not self._countable(counted):
            self._trial_in_flight = False
            logger.debug(f"Circuit '{self.name}' ignoring non-countable {type(exc).__name__}")

            return

        if self._state is CircuitState.HALF_OPEN:
            self._to_open()
            return

        self._failure_count += 1
        if self._failure_count >= self.failure_threshold:
            self._to_open()

    async def call(self, fn: Callable[..., Awaitable[T]], *args: Any, **kwargs: Any) -> T:
        self._admit()

        try:
            result = await fn(*args, **kwargs)
        except Exception as exc:
            self._record_failure(exc)
            raise

        self._record_success()
        return result

    def __call__(self, fn: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        @functools.wraps(fn)
        async def wrapper(*args: Any, **kwargs: Any) -> T:
            return await self.call(fn, *args, **kwargs)

        return wrapper

    def __repr__(self) -> str:
        return f'<CircuitBreaker name={self.name!r} state={self._state.value} failures={self._failure_count}>'