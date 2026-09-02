from __future__ import annotations
import pytest
from pydantic import ValidationError
from agentic_suite.tools.mock.slack_tool import SlackMessageTool

async def test_valid_channel_passes():
    tool = SlackMessageTool(channel='#general', message='Hello, world!')
    result = await tool.execute()
    assert result['ok'] is True

def test_invalid_channel_raises_validation_error():
    with pytest.raises(ValidationError, match='channel must start with'):
        SlackMessageTool(channel='general', message='Hello, world!')

def test_empty_message_is_rejected():
    with pytest.raises(ValidationError):
        SlackMessageTool(channel='#general', message='')