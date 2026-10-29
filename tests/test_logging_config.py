from __future__ import annotations
import logging
import sys
from agentic_suite.logging_config import configure_logging


def test_stream_can_be_redirected_to_stderr():
    """Over stdio, stdout is the MCP protocol stream; logs must not touch it."""
    configure_logging(stream=sys.stderr)

    assert logging.getLogger().handlers[0].stream is sys.stderr
    assert logging.getLogger('telemetry').handlers[0].stream is sys.stderr