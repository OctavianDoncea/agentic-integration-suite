from __future__ import annotations
import asyncio
import functools
import logging
import random
from collections.abc import Callable, Awaitable
from typing import TypeVar, Any

logger = logging.getLogger('middleware.retry')

T = TypeVar('T')

DEFAULT_MAX_ATTEMPTS = 3
DEFAULT_BASE_DELAY = 1.0
DEFAULT_MAX_WAIT = 30.0

SleepFn = Callable[[float], Awaitable[None]]
RngFn = Callable[[], float]

class RetryExhaustedError(RuntimeError):
    """Raised when every attempt failed."""
    def __init__(self, attempts: int, last_exception: BaseException):
        self.attempts = attempts
        self.last_exception = last_exception
        super().__init__(
            f'Call failed after {attempts} attempt(s). '
            f'Last error: {type(last_exception).__name__}: {last_exception}'
        )


def get_retry_after(exc: BaseException) -> float | None:
    headers = getattr(exc, 'headers', None)
    if headers is None:
        response = getattr(exc, 'response', None)
        headers = getattr(response, 'headers', None)

    if not headers:
        return None

    raw = headers.get('Retry-After') or headers.get('retry-after')
    if raw is None:
        return None

    try:
        value = float(raw)
    except (TypeError, ValueError):
        logger.warning(f'Unparseable Retry-After value: {raw}')
        return None
    
    return max(0.0, value)

def is_retryable(exc: BaseException) -> bool:
    from agentic_suite.middleware.circuit_breaker import CircuitOpenError

    if isinstance(exc, CircuitOpenError):
        return False

    if isinstance(exc, (asyncio.TimeoutError, ConnectionError)):
        return True

    status = getattr(exc, 'status', None)
    if status is None:
        status = getattr(getattr(exc, 'response', None), 'status_code', None)

    if isinstance(status, int):
        return status == 429 or 500 <= status < 600

    return False

def compute_backoff(attempt: int, *, base_delay: float = DEFAULT_BASE_DELAY, max_wait: float = DEFAULT_MAX_WAIT, rng: RngFn = random.random) -> float:
    # attempt is 0-based (first retry is 0), so the wait is full-jitter in [0, base_delay], then doubles.
    ceiling = min(base_delay * (2 ** attempt), max_wait)
    return rng() * ceiling

def with_retry(
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    *,
    base_delay: float = DEFAULT_BASE_DELAY,
    max_wait: float = DEFAULT_MAX_WAIT,
    retryable: Callable[[BaseException], bool] = is_retryable,
    respect_retry_after: bool = True,
    sleep: SleepFn = asyncio.sleep,
    rng: RngFn = random.random,
) -> Callable[[Callable[..., Awaitable[T]]], Callable[..., Awaitable[T]]]:
    if max_attempts < 1:
        raise ValueError('max_attempts must be at least 1.')

    def decorator(fn: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        @functools.wraps(fn)
        async def wrapper(*args: Any, **kwargs: Any) -> T:
            last_exception: BaseException | None = None

            for attempt in range(max_attempts):
                try:
                    return await fn(*args, **kwargs)
                except Exception as exc:
                    last_exception = exc

                    if not retryable(exc):
                        logger.info(f'Not retrying {fn.__name__}: {type(exc).__name__} is terminal')
                        raise

                    if attempt == max_attempts - 1:
                        break

                    delay = compute_backoff(attempt, base_delay=base_delay, max_wait=max_wait, rng=rng)
                    source = 'jitter'

                    if respect_retry_after:
                        retry_after = get_retry_after(exc)
                        if retry_after is not None:
                            delay = min(retry_after, max_wait)
                            source = 'retry-after'

                    logger.info(f'Attempt {attempt+1}/{max_attempts} of {fn.__name__} failed ({type(exc).__name__}); retrying in {delay:.3f}s ({source})')
                    await sleep(delay)

            assert last_exception is not None
            raise RetryExhaustedError(max_attempts, last_exception) from last_exception

        return wrapper

    return decorator