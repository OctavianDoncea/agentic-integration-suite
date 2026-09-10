from __future__ import annotations
import json
from enum import Enum
from pathlib import Path
from typing import Any, Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator

BENCHMARK_PATH = Path(__file__).parent / 'data' / 'benchmark.json'

class Category(str, Enum):
    DIRECT_INTENT = 'direct_intent'
    AMBIGUOUS = 'ambiguous'
    EDGE_CASE_SCHEMA = 'edge_case_schema'


class Behaviour(str, Enum):
    TOOL_CALL = 'tool_call'
    CLARIFY = 'clarify'
    CLARIFY_OR_DEFAULT = 'clarify_or_default'
    NO_TOOL_CALL = 'no_tool_call'


class ExpectedOutcome(BaseModel):
    model_config = ConfigDict(extra='forbid')

    behaviour: Behaviour
    tool_name: str | None = None
    arguments: dict[str, Any] | None = None
    flexible_arguments: list[str] = Field(default_factory=list)
    default_tool_name: str | None = None
    default_arguments: dict[str, Any] | None = None

    @model_validator(mode='after')
    def _check_consistency(self) -> ExpectedOutcome:
        if self.behaviour is Behaviour.TOOL_CALL and not self.tool_name:
            raise ValueError("behaviour 'tool_call' requires a tool_name")
        if self.behaviour is Behaviour.CLARIFY_OR_DEFAULT and not self.default_tool_name:
            raise ValueError("behaviour 'clarify_or_default' requires a default_tool_name")
        if self.behaviour is Behaviour.NO_TOOL_CALL and self.tool_name:
            raise ValueError("behaviour 'no_tool_call' cannot have a tool_name")
        return self


class BenchmarkCase(BaseModel):
    model_config = ConfigDict(extra='forbid')

    id: str
    category: Category
    user_prompt: str
    expected: ExpectedOutcome
    smoke: bool = False
    notes: str = ''


class Benchmark(BaseModel):
    model_config = ConfigDict(extra='forbid')

    schema_version: Literal[1]
    description: str = ''
    cases: list[BenchmarkCase]

    @model_validator(mode='after')
    def _ids_are_unique(self) -> Benchmark:
        ids = [case.id for case in self.cases]
        duplicates = {i for i in ids if ids.count(i) > 1}
        if duplicates:
            raise ValueError(f'Duplicate case ids: {sorted(duplicates)}')
        return self

    def smoke_subset(self) -> list[BenchmarkCase]:
        return [case for case in self.cases if case.smoke]

    def by_category(self, category: Category) -> list[BenchmarkCase]:
        return [case for case in self.cases if case.category is category]


def load_benchmark(path: Path = BENCHMARK_PATH) -> Benchmark:
    return Benchmark.model_validate(json.loads(path.read_text(encoding='utf-8')))