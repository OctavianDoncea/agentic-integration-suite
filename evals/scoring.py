from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
from pydantic import ValidationError
from agentic_suite.clients.groq_client import ModelResponse
from agentic_suite.sdk.base import BaseTool
from agentic_suite.sdk.registry import ToolRegistry
from evals.schema import BenchmarkCase, Behaviour

CLARIFY_MARKERS = ('?',)

@dataclass
class CaseResult:
    case_id: str
    category: str
    selection_correct: bool = False
    arguments_valid: bool | None = None
    arguments_match: bool | None = None
    failure_reason: str | None = None
    actual_tool_name: str | None = None
    actual_arguments: dict[str, Any] = field(default_factory=dict)
    response_text: str | None = None

    @property
    def passed(self) -> bool:
        if not self.selection_correct:
            return False
        if self.arguments_valid is False or self.arguments_match is False:
            return False
        
        return True

    def to_dict(self) -> dict[str, Any]:
        return {
            'case_id': self.case_id,
            'category': self.category,
            'passed': self.passed,
            'selection_correct': self.selection_correct,
            'arguments_valid': self.arguments_valid,
            'arguments_match': self.arguments_match,
            'failure_reason': self.failure_reason,
            'actual_tool_name': self.actual_tool_name,
            'actual_arguments': self.actual_arguments,
            'response_text': (self.response_text or '')[:300] or None
        }


def looks_like_a_question(text: str | None) -> bool:
    return bool(text) and any(marker in text for marker in CLARIFY_MARKERS)

def _validate(tool_cls: type[BaseTool], arguments: dict[str, Any]) -> str | None:
    try:
        tool_cls.validate_arguments(arguments)
    except ValidationError as e:
        first = e.errors()[0]
        location = '.'.join(str(p) for p in first['loc']) or '(root)'
        return f"{location}: {first['msg']}"

    return None

def _arguments_match(expected: dict[str, Any] | None, flexible: list[str], actual: dict[str, Any]) -> str | None:
    for key, want in (expected or {}).items():
        if key not in actual:
            return f"missing argument '{key}'"
        got = actual[key]
        if got != want:
            return f"argument '{key}': expected {want!r} ({type(want).__name__}), got {got!r} ({type(got).__name__})"

    for key in flexible:
        if key not in actual:
            return f"missing flexible argument '{key}'"
        if actual[key] in (None, '', [], {}):
            return f"flexible argument '{key}' is empty"

    return None

def score_case(case: BenchmarkCase, response: ModelResponse, registry: ToolRegistry) -> CaseResult:
    expected = case.expected
    call = response.first_tool_call
    result = CaseResult(
        case_id=case.id,
        category=case.category.value,
        actual_tool_name = call.name if call else None,
        actual_arguments = call.arguments if call else {},
        response_text = response.text,
    )

    if expected.behaviour is Behaviour.NO_TOOL_CALL:
        result.selection_correct = not response.called_a_tool
        if not result.selection_correct:
            result.failure_reason = f"called '{call.name}' when no tool applies"
        return result

    if expected.behaviour is Behaviour.CLARIFY:
        if response.called_a_tool:
            result.failure_reason = f"called '{call.name}' instead of asking"
            return result
        result.selection_correct = looks_like_a_question(response.text)
        if not result.selection_correct:
            result.failure_reason = "did not ask a clarifying question"
        return result

    if expected.behaviour is Behaviour.CLARIFY_OR_DEFAULT:
        if not response.called_a_tool:
            result.selection_correct = looks_like_a_question(response.text)
            if not result.selection_correct:
                result.failure_reason = "neither called the default tool nor asked"
            return result

        target_name = expected.default_tool_name
        target_args = expected.default_arguments
    else:
        if not response.called_a_tool:
            result.failure_reason = f"no tool called; expected '{expected.tool_name}'"
            return result

        target_name = expected.tool_name
        target_args = expected.arguments

    if call.name != target_name:
        result.failure_reason = f"selected '{call.name}', expected '{target_name}'"
        return result

    result.selection_correct = True

    if not call.parsed_ok:
        result.arguments_valid = False
        result.arguments_match = False
        result.failure_reason = f'tool call arguments were not valid JSON: {call.parse_error}'
        return result

    validation_error = _validate(registry._tools[target_name], call.arguments)
    result.arguments_valid = validation_error is None
    if validation_error:
        result.arguments_match = False
        result.failure_reason = f'schema validation failed: {validation_error}'
        return result

    mismatch = _arguments_match(target_args, expected.flexible_arguments, call.arguments)
    result.arguments_match = mismatch is None
    if mismatch:
        result.failure_reason = mismatch

    return result

@dataclass
class RunSummary:
    model: str
    results: list[CaseResult]

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def passed(self) -> int:
        return sum(r.passed for r in self.results)

    @property
    def pass_rate(self) -> float:
        return self.passed / self.total if self.total else 0.0

    @property
    def selection_accuracy(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.selection_correct for r in self.results) / len(self.results)

    @property
    def argument_precision(self) -> float:
        scored = [r for r in self.results if r.arguments_match is not None]
        if not scored:
            return 0.0

        return sum(r.arguments_match for r in scored) / len(scored)

    def by_category(self) -> dict[str, dict[str, Any]]:
        buckets: dict[str, list[CaseResult]] = {}
        for result in self.results:
            buckets.setdefault(result.category, []).append(result)

        return {
            category: {
                'total': len(rs),
                'passed': sum(r.passed for r in rs),
                'pass_rate': round(sum(r.passed for r in rs) / len(rs), 4)
            }
            for category, rs in buckets.items()
        }

    def to_dict(self) -> dict[str, Any]:
        from datetime import datetime, timezone

        return {
            'model': self.model,
            'run_at': datetime.now(timezone.utc).isoformat(),
            'total': self.total,
            'passed': self.passed,
            'pass_rate': round(self.pass_rate, 4),
            'selection_accuracy': round(self.selection_accuracy, 4),
            'argument_precision': round(self.argument_precision, 4),
            'by_category': self.by_category(),
            'results': [r.to_dict() for r in self.results]
        }