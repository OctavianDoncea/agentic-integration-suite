"""Errors shared by all provider clients."""
from __future__ import annotations

class ProviderConfigError(RuntimeError):
    """A provider was selected without the configuration it needs."""


class ProviderHTTPError(Exception):
    """A non-2xx response from a provider."""

    def __init__(self, status_code: int, headers: dict[str, str] | None = None, body: str = ''):
        super().__init__(f'HTTP {status_code}: {body[:200]}')
        self.status_code = status_code
        self.headers = headers or {}
        self.body = body