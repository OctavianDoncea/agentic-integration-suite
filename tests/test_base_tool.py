from __future__ import annotations
import pytest
from pydantic import ValidationError, Field
from agentic_suite.sdk.base import BaseTool, ConfigField, ToolDefinitionError

class DummySearchTool(BaseTool):
    """Search an internal knowledge base and return matching documents"""
    query: str = Field(description='The natural-language search query.')
    limit: int = Field(default=10, ge=1, le=50, description='Max results to return.')
    include_archived: bool = Field(default=False, description='Include archived docs.')
    fail_on_purpose: bool = ConfigField(False)

    async def execute(self, **kwargs):
        if self.fail_on_purpose:
            raise RuntimeError('injected failure')
        return {'query': self.query, 'count': self.limit}


class RenamedTool(BaseTool):
    """Docstring that should be ignored"""
    tool_name = 'custom_wire_name'
    tool_description = 'An explicitly provided description'

    value: str

    async def execute(self, **kwargs):
        return {'value': self.value}


class UndocumentedTool(BaseTool):
    value: str

    async def execute(self, **kwargs):
        return {}

def test_schema_has_correct_top_level_structure():
    schema = DummySearchTool.get_schema()
    assert schema['type'] == 'function'
    assert set(schema['function']) == {'name', 'description', 'parameters'}
    assert schema['function']['parameters']['type'] == 'object'

def test_name_is_derived_from_class_name():
    assert DummySearchTool.get_schema()['function']['name'] == 'dummy_search_tool'

def test_description_is_taken_from_docstring():
    description = DummySearchTool.get_schema()['function']['description']
    assert description.startswith('Search an internal knowledge base')

def test_class_vars_override_derived_name_and_description():
    fn = RenamedTool.get_schema()['function']
    assert fn['name'] == 'custom_wire_name'
    assert fn['description'] == 'An explicitly provided description'

def test_missing_docstring_is_a_definition_error():
    with pytest.raises(ToolDefinitionError, match='docstring'):
        UndocumentedTool.get_schema()

def test_only_defaultless_fields_are_required():
    params = DummySearchTool.get_schema()['function']['parameters']
    assert params['required'] == ['query']

def test_field_descriptions_and_constraints_survive():
    props = DummySearchTool.get_schema()['function']['parameters']['properties']
    assert props['query']['type'] == 'string'
    assert props['query']['description'] == 'The natural-language search query.'
    assert props['limit']['minimum'] == 1
    assert props['limit']['maximum'] == 50

def test_config_fields_are_hidden_from_the_model():
    params = DummySearchTool.get_schema()['function']['parameters']
    assert 'fail_on_purpose' not in params['properties']
    assert 'fail_on_purpose' not in params['required']

def test_pydantic_titles_are_stripped():
    props = DummySearchTool.get_schema()['function']['parameters']['properties']
    assert all('title' not in prop for prop in props.values())

def test_additional_properties_is_closed():
    assert DummySearchTool.get_schema()['function']['parameters']['additionalProperties'] is False

def test_validate_arguments_accepts_a_valid_payload():
    tool = DummySearchTool.validate_arguments({'query': 'onboarding docs', 'limit': 5})
    assert isinstance(tool, DummySearchTool)
    assert tool.limit == 5
    assert tool.include_archived is False

def test_validate_arguments_rejects_out_of_range_values():
    with pytest.raises(ValidationError):
        DummySearchTool.validate_arguments({'query': 'x', 'limit': 999})

def test_validate_arguments_rejects_hallucinated_fields():
    with pytest.raises(ValidationError):
        DummySearchTool.validate_arguments({'query': 'x', 'sort_by': 'relevance'})

def test_base_tool_cannot_be_instantiated():
    with pytest.raises(TypeError, match='abstract'):
        BaseTool()

def test_subclass_without_execute_cannot_be_instantiated():
    class IncompleteTool(BaseTool):
        value: str

    
    with pytest.raises(TypeError, match='abstract'):
        IncompleteTool(value='x')

async def test_execute_runs_with_validated_arguments():
    tool = DummySearchTool.validate_arguments({'query': 'quarterly report', 'limit': 3})
    assert await tool.execute() == {'query': 'quarterly report', 'count': 3}

async def test_config_field_still_functions_at_runtime():
    tool = DummySearchTool(query='x', fail_on_purpose=True)
    with pytest.raises(RuntimeError, match='injected failure'):
        await tool.execute()