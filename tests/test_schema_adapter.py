from __future__ import annotations
from agentic_suite.clients.schema_adapters import adapt_schema, adapt_tools
from agentic_suite.tools.mock.calendar_tool import CalendarEventTool
from agentic_suite.tools.mock.jira_tool import JiraIssueTool

def test_length_constraints_move_into_the_description():
    params = adapt_schema(JiraIssueTool.get_schema()['function']['parameters'])
    title = params['properties']['title']

    assert 'minLength' not in title and 'maxLength' not in title
    assert title['description'].endswith('Constraints: min length 1, max length 255.')

def test_numeric_ranges_move_and_supported_keywords_stay():
    params = adapt_schema(CalendarEventTool.get_schema()['function']['parameters'])
    props = params['properties']

    assert 'minimum' not in props['duration_minutes']
    assert 'min 5, max 400' in props['duration_minutes']['description']
    assert props['start_time']['format'] == 'date-time'
    assert props['visibility']['enum'] == ['public', 'private']
    assert params['additionalProperties'] is False

def test_required_list_is_preserved():
    original = JiraIssueTool.get_schema()['function']['parameters']
    assert adapt_schema(original)['required'] == original['required']

def test_a_field_named_like_a_keyword_survives():
    schema = {'type': 'object', 'properties': {'maximum': {'type': 'integer'}}}
    assert 'maximum' in adapt_schema(schema)['properties']

def test_adapt_tools_does_not_mutate_its_input():
    tools = [JiraIssueTool.get_schema()]
    before = repr(tools)
    adapt_tools(tools)
    assert repr(tools) == before