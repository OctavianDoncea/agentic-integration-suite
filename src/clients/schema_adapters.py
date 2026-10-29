"""Adapt tool schemas for providers that support only a subset of JSON Schema."""
from __future__ import annotations
from typing import Any

_CONSTRAINT_TEXT: dict[str, str] = {
    'minimum': 'min {}',
    'maximum': 'max {}',
    'exclusiveMinimum': 'greater than {}',
    'exclusiveMaximum': 'less than {}',
    'minLength': 'min length {}',
    'maxLength': 'max length {}',
    'minItems': 'min items {}',
    'maxItems': 'max items {}'
}
_SCHEMA_MAPS = ('properties', '$defs', 'definitions', 'patternProperties')

def adapt_schema(node: Any) -> Any:
    """Return a copy of a JSON Schema with unsupported constraints as prose."""

    if isinstance(node, list):
        return [adapt_schema(item) for item in node]
    if not isinstance(node, dict):
        return node

    adapted: dict[str, Any] = {}
    removed: dict[str, Any] = {}
    for key, value in node.items():
        if key in _CONSTRAINT_TEXT:
            removed[key] = value
        elif key in _SCHEMA_MAPS and isinstance(value, dict):
            adapted[key] = {name: adapt_schema(sub) for name, sub in value.items()}
        else:
            adapted[key] = adapt_schema(value)

    notes = [text.format(removed[key]) for key, text in _CONSTRAINT_TEXT.items() if key in removed]
    if notes:
        note = 'Constraints: ' + ', '.join(notes) + '.'
        existing = adapted.get('description')
        adapted['description'] = f'{existing} {note}' if existing else note

    return adapted

def adapt_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{**tool, 'function': {**tool['function'], 'parameters': adapt_schema(tool['function']['parameters'])}} for tool in tools]