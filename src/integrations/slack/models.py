"""Persistence model for Slack app installations."""

from __future__ import annotations
from datetime import datetime, timezone
from sqlalchemy import Boolean, DateTime, Integer, String, func
from sqlalchemy.orm import Mapped, mapped_column
from agentic_suite.db import Base
from agentic_suite.integrations.slack.crypto import EncryptedString

def _utcnow() -> datetime:
    return datetime.now(timezone.utc)

class SlackInstallation(Base):
    """A single workspace's installation of the Slack app."""
    __tablename__ = 'slack_installations'

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    team_id: Mapped[str] = mapped_column(String(32), unique=True, nullable=False)
    team_name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    access_token: Mapped[str] = mapped_column(EncryptedString(1024), nullable=False)
    refresh_token: Mapped[str | None] = mapped_column(EncryptedString(1024), nullable=True)
    token_expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    scope: Mapped[str] = mapped_column(String(1024), nullable=False, default='')
    bot_user_id: Mapped[str | None] = mapped_column(String(32), nullable=True)
    needs_reauth: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    installed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=_utcnow, onupdate=_utcnow, server_default=func.now())

    def __repr__(self) -> str:
        return (
            f'<SlackInstallation team_id={self.team_id!r} '
            f'needs_reauth={self.needs_reauth}>'
        )