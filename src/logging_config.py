from __future__ import annotations
import logging
import sys
from agentic_suite.config import get_settings

def configure_logging() -> None:
    settings = get_settings()
    
    root = logging.getLogger()
    root.setLevel(settings.log_level.upper())
    root.handlers.clear()

    app_handler = logging.StreamHandler(sys.stdout)
    app_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s-8s %(name)s: %(message)s'))
    root.addHandler(app_handler)

    telemetry = logging.getLogger('telemetry')
    telemetry.handlers.clear()
    telemetry.propagate = False

    telemetry_handler = logging.StreamHandler(sys.stdout)
    telemetry_handler.setFormatter(logging.Formatter('%(message)s'))
    telemetry.addHandler(telemetry_handler)
    telemetry.setLevel(logging.INFO)