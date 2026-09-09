from __future__ import annotations
import json
import logging
import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any, Literal

logger = logging.getLogger('telemetry.tool_invocation')

Status = Literal['success', 'tool_error', 'validation_error', 'circuit_open', 'not_registered']

SENSITIVE_KEYS = frozenset({'token', 'access_token', 'password', 'secret', 'api-key'})
SAFE_VALUE_MAX_LENGTH = 24

def redact_arguments(arguments: dict[str, Any] | None) -> dict[str, str]:
    """Summarizes arguments as `{key: type-or-short-value}`."""

    if not arguments:
        return {}

    summary: dict[str, str] = {}
    for key, value in arguments.items():
        if key.lower() in SENSITIVE_KEYS:
            summary[key] = '<redacted>'
        elif isinstance(value, bool | int | float):
            summary[key] = repr(value)
        elif isinstance(value, str):
            summary[key] = repr(value) if len(value) <= SAFE_VALUE_MAX_LENGTH else f'str(len={len(value)})'
        elif isinstance(value, list | tuple):
            summary[key] = f'{type(value).__name__}(len={len(value)})'
        elif value is None:
            summary[key] = 'None'
        else:
            summary[key] = type(value).__name__
    
    return summary

def log_invocation(tool_name: str, arguments: dict[str, Any] | None, latency_ms: float, status: Status, retry_count: int = 0, *, error_type: str | None = None, error_message: str | None = None, **extra: Any) -> None:
    """Emit one structured line describing a tool invocation."""
    record: dict[str, Any] = {
        'event': 'tool_invocation',
        'timestamp': time.time(),
        'tool_name': tool_name,
        'arguments': redact_arguments(arguments),
        'latency_ms': round(latency_ms, 2),
        'status': status,
        'retry_count': retry_count,
    }

    if error_type:
        record['error_type'] = error_type
    if error_message:
        record['error_message'] = error_message[:500]

    record.update(extra)

    level = logging.INFO if status == 'success' else logging.WARNING
    logger.log(level, json.dumps(record, default=str))

@contextmanager
def timed() -> Iterator[dict[str, float]]:
    result: dict[str, float] = {'elapsed_ms': 0.0}
    start = time.monotonic()
    try:
        yield result
    finally:
        result['elapsed_ms'] = (time.monotonic() - start) * 1000