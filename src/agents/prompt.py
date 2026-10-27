from __future__ import annotations
from functools import lru_cache
from importlib.resources import files

@lru_cache(maxsize=1)
def load_system_prompt() -> str:
    return files('agentic_suite.agents').joinpath('system_prompt.txt').read_text(encoding='utf-8')