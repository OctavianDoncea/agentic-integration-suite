from __future__ import annotations
import pytest
from agentic_suite.tools.mock.calendar_tool import CalendarEventTool
from pydantic import ValidationError

ISO_START = '2026-03-14T09:30:00Z'

async def test_happy_path_returns_scheduled_event():
    tool = CalendarEventTool.validate_arguments({
        'title': 'Sprint planning',
        'start_time': ISO_START,
        'duration_minutes': 60,
        'attendees': ['ana@example.com', 'alex@example.com'],
    })
    result = await tool.execute()

    assert result['status'] == 'scheduled'
    assert result['id'] == 'EVT-001'
    assert result['attendee_count'] == 2

def test_empty_title_is_rejected():
    with pytest.raises(ValidationError):
        CalendarEventTool.validate_arguments({
            'title': '',
            'start_time': ISO_START,
            'duration_minutes': 30,
        })

@pytest.mark.parametrize('duration', [4, 401])
def test_duration_outside_bounds_is_rejected(duration: int):
    with pytest.raises(ValidationError):
        CalendarEventTool.validate_arguments({
            'title': 'Sync',
            'start_time': ISO_START,
            'duration_minutes': duration,
        })

@pytest.mark.parametrize('duration', [5, 400])
def test_duration_boundaries_are_accepted(duration: int):
    tool = CalendarEventTool.validate_arguments({
        'title': 'Sync',
        'start_time': ISO_START,
        'duration_minutes': duration,
    })
    assert tool.duration_minutes == duration

def test_invalid_visibility_is_rejected():
    with pytest.raises(ValidationError):
        CalendarEventTool.validate_arguments({
            'title': 'Sync',
            'start_time': ISO_START,
            'duration_minutes': 30,
            'visibility': 'secret',
        })

def test_defaults_match_schema_optional_fields():
    tool = CalendarEventTool.validate_arguments({
        'title': 'Standup',
        'start_time': ISO_START,
        'duration_minutes': 15,
    })
    assert tool.visibility == 'private'
    assert tool.send_invites is True
    assert tool.attendees == []

def test_schema_required_fields_and_visible_names():
    params = CalendarEventTool.get_schema()['function']['parameters']
    assert set(params['required']) == {'title', 'start_time', 'duration_minutes'}
    assert set(params['properties']) == {
        'title',
        'start_time',
        'duration_minutes',
        'attendees',
        'visibility',
        'send_invites',
    }