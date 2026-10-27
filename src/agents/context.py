from __future__ import annotations
from sqlalchemy.orm import Session
from agentic_suite.config import get_settings
from agentic_suite.integrations.slack.models import SlackInstallation

def resolve_team_id(session: Session, explicit: str | None = None) -> str | None:
    """Explicit > DEFAULT_SLACK_TEAM_ID > the only healthy installation.

    Returns None rather than raising when nothing matches: a request that only
    uses mock tools needs no workspace, and the Slack tool reports a missing
    one as a tool error the model can explain.
    """

    if explicit:
        return explicit

    configured = get_settings().default_slack_team_id
    if configured:
        return configured

    rows = session.query(SlackInstallation.team_id).filter_by(needs_reauth=False).limit(2).all()

    return rows[0][0] if len(rows) == 1 else None