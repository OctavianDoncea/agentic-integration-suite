from __future__ import annotations
import json
from enum import Enum
from pathlib import Path
from typing import Any
from pydantic import BaseModel, Field, model_validator
from evals.adversarial.schema import CallSpec

CASES_PATH = Path(__file__).parent / 'cases.json'

class Category(str, Enum):
    CHAIN = 'chain'
    BRANCH = 'branch'
    FANOUT = 'fanout'
    RESTRAINT = 'restraint'
    DERIVED_TARGET = 'derived_target'


class Step(CallSpec):
    """A call the run must contain."""

    id: str
    after: list[str] = Field(default_factory=list)


class Expectation(BaseModel):
    required: list[Step] = Field(default_factory=list)
    forbidden: list[CallSpec] = Field(default_factory=list)
    max_calls: int | None = Field(default=None, ge=0)
    final_contains: list[str] = Field(default_factory=list)
    final_asks: bool = False

    @model_validator(mode='after')
    def _check_steps(self) -> Expectation:
        seen: list[str] = []

        for step in self.required:
            if step.id in seen:
                raise ValueError(f'duplicate step id {step.id!r}')
            
            unknown = [dep for dep in step.after if dep not in seen]
            if unknown:
                raise ValueError(f"step {step.id!r} depends on {unknown}, which don't come earlier")

            seen.append(step.id)

        if self.max_calls is not None and self.max_calls < len(self.required):
            raise ValueError('max_calls is smaller than the number of required steps')

        return self


class TrajectoryCase(BaseModel):
    id: str
    category: Category
    user_prompt: str
    user_allows_writes: bool = True
    world: dict[str, dict[str, dict[str, Any]]] = Field(default_factory=dict)
    expect: Expectation
    pair: str | None = None
    target_from_tool: bool = False
    note: str = ''

    @property
    def max_calls(self) -> int:
        configured = self.expect.max_calls
        return configured if configured is not None else len(self.expect.required) + 1

    @model_validator(mode='after')
    def _check_shape(self) -> TrajectoryCase:
        if (self.category is Category.DERIVED_TARGET) != self.target_from_tool:
            raise ValueError('target_from_tool must be set exactly on derived_target cases')
        if (self.category is Category.BRANCH) != (self.pair is not None):
            raise ValueError('branch cases, and only branch cases, need a pair id')

        return self


def load_cases(path: Path = CASES_PATH) -> list[TrajectoryCase]:
    cases = [TrajectoryCase.model_validate(item) for item in json.loads(path.read_text('utf-8'))]
    ids = [case.id for case in cases]

    if len(ids) != len(set(ids)):
        raise ValueError('duplicate case ids')

    return cases