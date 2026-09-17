"""Mock calendar event-creation tool."""
from __future__ import annotations
from datetime import datetime
from typing import Any, Literal
from agentic_suite.sdk.base import BaseTool
from pydantic import Field

class CalendarEventTool(BaseTool):
    """Create a calendar event with a title, start time, duration, and optional attendees."""

    title: str = Field(min_length=1, description='Event title.')
    start_time: datetime = Field(
        description='Start time in ISO-8601 format, e.g. "2026-03-14T09:30:00Z".'
    )
    duration_minutes: int = Field(ge=5, le=400, description='Duration in minutes.')
    attendees: list[str] = Field(default_factory=list, description='Attendee email addresses.')
    visibility: Literal['public', 'private'] = Field(
        default='private', description='Event visibility.'
    )
    send_invites: bool = Field(default=True, description='Email the attendees.')

    async def execute(self, **kwargs: Any) -> dict[str, Any]:
        return {
            'id': 'EVT-001',
            'status': 'scheduled',
            'start_time': self.start_time.isoformat(),
            'attendee_count': len(self.attendees),
        }