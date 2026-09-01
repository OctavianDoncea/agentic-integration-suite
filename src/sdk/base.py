import re
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, ClassVar
from pydantic import BaseModel, Field, ConfigDict

TOOL_CONFIG_MARKER = 'x-tool-config'

def ConfigField(default: Any = None, **kwargs: Any) -> Any:
    extra = dict(kwargs.pop('json_schema_extra', None) or {})
    extra[TOOL_CONFIG_MARKER] = True
    return Field(default, json_schema_extra=extra, **kwargs)

def _to_snake_case(name: str) -> str:
    intermediate = re.sub(r'(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub(r'([a-z0-9])([A-Z])', r'\1_\2', intermediate).lower()

class ToolDefinitionError(TypeError):
    """Raised when a subclass is not usable as a tool"""


class BaseTool(BaseModel, ABC):
    """Abstract base for every tool the model can invoke
    
    Subclasses must implement 'execute' and carry a docstring (description that the model reads when deciding whether to call the tool)
    """
    model_config = ConfigDict(extra='forbid', validate_assignment=True)

    tool_name: ClassVar[str | None] = None
    tool_description: ClassVar[str | None] = None

    @abstractmethod
    async def execute(self, **kwargs: Any) -> dict[str, Any]:
        raise NotImplementedError

    @classmethod
    def get_name(cls) -> str:
        if cls.tool_name:
            return cls.tool_name
        
        return _to_snake_case(cls.__name__)

    @classmethod
    def get_description(cls) -> str:
        if cls.tool_description:
            return ' '.join(cls.tool_description.split())
        
        doc = (cls.__doc__ or '').strip()
        if not doc:
            raise ToolDefinitionError(
                f"{cls.__name__} needs a docstring or an explicit 'tool description';"
                'the model relies on it to choose this tool.'
            )

        return ' '.join(doc.split('\n\n')[0].split())

    @classmethod
    def get_schema(cls) -> dict[str, Any]:
        raw = cls.model_json_schema(mode='validation')

        properties: dict[str, Any] = {}
        hidden: set[str] = set()

        for field_name, raw_prop in (raw.get('properties') or {}).items():
            prop = dict(raw_prop)
            if prop.pop(TOOL_CONFIG_MARKER, False):
                hidden.add(field_name)
                continue
            prop.pop('title', None)
            properties[field_name] = prop

        parameters: dict[str, Any] = {
            'type': 'object',
            'properties': properties,
            'required': [f for f in raw.get('required', []) if f not in hidden],
            'additionalProperties': False
        }

        if '$defs' in raw:
            parameters['$defs'] = raw['$defs']

        return {
            'type': 'function',
            'function': {
                'name': cls.get_name(),
                'description': cls.get_description(),
                'parameters': parameters
            }
        }

    @classmethod
    def validate_arguments(cls, arguments: dict[str, Any]) -> BaseTool:
        return cls.model_validate(arguments)