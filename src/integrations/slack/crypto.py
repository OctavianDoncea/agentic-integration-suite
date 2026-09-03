"""Transparent encryption for token columns."""

from __future__ import annotations
from functools import lru_cache
from cryptography.fernet import Fernet, InvalidToken
from sqlalchemy import String, TypeDecorator
from agentic_suite.config import get_settings

class TokenDecryptionError(RuntimeError):
    """Raised when a stored token cannot be decrypted with the current key"""


@lru_cache(maxsize=1)
def get_fernet() -> Fernet:
    return Fernet(get_settings().slack_token_encryption_key.encode())

class EncryptedString(TypeDecorator):
    impl = String
    cache_ok = True

    def process_bind_param(self, value: str | None, dialect) -> str | None:
        if value is None:
            return None

        return get_fernet().encrypt(value.encode()).decode()

    def process_result_value(self, value: str | None, dialect) -> str | None:
        if value is None:
            return None
        try:
            return get_fernet().decrypt(value.encode()).decode()
        except InvalidToken as e:
            raise TokenDecryptionError('Stored token could not be decrypted. The encryption key likely changed; '
                'affected installations must re-authorize.'
            ) from e