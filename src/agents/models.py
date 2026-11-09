from __future__ import annotations
from datetime import datetime, timezone
from sqlalchemy import JSON, Boolean, Column, DateTime, Float, Integer, String, Text
from agentic_suite.db import Base

def _utcnow() -> datetime:
    return datetime.now(timezone.utc)

class AgentRun(Base):
    __tablename__ = 'agent_runs'

    id = Column(String(32), primary_key=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=_utcnow, index=True)
    status = Column(String(16), nullable=False, index=True)
    model = Column(String(128), nullable=False)
    provider = Column(String(32))
    team_id = Column(String(32))
    allow_writes = Column(Boolean, nullable=False)
    write_tools = Column(JSON)
    prompt_sha256 = Column(String(64), nullable=False)
    prompt_chars = Column(Integer, nullable=False)
    prompt = Column(Text)
    final_text = Column(Text)
    error_type = Column(String(128))
    steps = Column(JSON, nullable=False)
    tool_calls = Column(Integer, nullable=False, default=0)
    policy_stops = Column(Integer, nullable=False, default=0)
    model_calls = Column(Integer, nullable=False, default=0)
    prompt_tokens = Column(Integer, nullable=False, default=0)
    completion_tokens = Column(Integer, nullable=False, default=0)
    model_ms = Column(Float, nullable=False, default=0.0)
    duration_ms = Column(Float, nullable=False, default=0.0)