from __future__ import annotations
import logging
import sys
from typing import TextIO
from agentic_suite.config import get_settings

def configure_logging(stream: TextIO | None = None) -> None:
    stream = stream or sys.stdout
    settings = get_settings()
    
    root = logging.getLogger()
    root.setLevel(settings.log_level.upper())
    root.handlers.clear()

    app_handler = logging.StreamHandler(stream)
    app_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s-8s %(name)s: %(message)s'))
    root.addHandler(app_handler)

    telemetry = logging.getLogger('telemetry')
    telemetry.handlers.clear()
    telemetry.propagate = False

    telemetry_handler = logging.StreamHandler(stream)
    telemetry_handler.setFormatter(logging.Formatter('%(message)s'))
    telemetry.addHandler(telemetry_handler)
    telemetry.setLevel(logging.INFO)