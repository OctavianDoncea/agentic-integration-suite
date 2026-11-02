from __future__ import annotations
import json
from enum import Enum
from pathlib import Path
from typing import Any
from pydantic import BaseModel, Field, model_validator

CASES_PATH = Path(__file__).parent / 'cases.json'

class Category(str, Enum):
    REDIRECT = 'redirect'
    UNREQUESTED_WRITE = 'unrequested_write'
    EXFILTRATION = 'exfiltration'
    CONTROL = 'control'


class CallSpec(BaseModel):
    """Describes a tool call by its tool and (optionally) its arguments."""

    tool: str
    match: dict[str, str] = Field(default_factory=dict)
    contains: list[str] = Field(default_factory=list)


class AdversarialCase(BaseModel):
    id: str
    category: Category
    user_prompt: str
    user_allows_writes: bool
    lure: dict[str, dict[str, Any]] = Field(default_factory=dict)
    expected_write: CallSpec | None = None
    attack: CallSpec | None = None
    note: str = ''

    @model_validator(mode='after')
    def _check_shape(self) -> AdversarialCase:
        if self.category is Category.CONTROL:
            if self.attack is not None or self.expected_write is None:
                raise ValueError('control cases need an expected_write and no attack')
        elif self.attack is None:
            raise ValueError('non-control cases need an attack')
        if not self.user_allows_writes and self.expected_write is not None:
            raise ValueError("a request that disallows writes can't expect one")

        return self


def load_cases(path: Path = CASES_PATH) -> list[AdversarialCase]:
    cases = [AdversarialCase.model_validate(item) for item in json.loads(path.read_text('utf-8'))]
    ids = [case.id for case in cases]

    if len(ids) != len(set(ids)):
        raise ValueError('duplicate case ids')

    return cases