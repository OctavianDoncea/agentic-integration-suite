"""FastAPI application: Slack install and OAuth callback endpoints."""
from __future__ import annotations
import logging
from fastapi import Depends, FastAPI, Query
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from sqlalchemy.orm import Session
from agentic_suite.config import get_settings
from agentic_suite.db import get_session
from agentic_suite.integrations.slack.oauth import InvalidOAuthStateError, TokenExchangeError, build_authorize_url, exchange_code_for_token, generate_state, persist_installation, verify_state

logger = logging.getLogger(__name__)

app = FastAPI(title='Agentic Integration Suite', description='Tool-calling SDK, resilience middleware, and evaluation harness.')

@app.get('/health', tags=['ops'])
async def health() -> dict[str, str]:
    return {'status': 'ok'}

@app.get('/slack/install', tags=['slack'])
async def slack_install() -> RedirectResponse:
    return RedirectResponse(url=build_authorize_url(generate_state()), status_code=302)

@app.get('/slack/oauth/callback', tags=['slack'])
async def slack_oauth_callback(code: str | None = Query(default=None), state: str | None = Query(default=None), error: str | None = Query(default=None), session: Session = Depends(get_session)):
    """Handles Slack's redirect back after the user approves or denies."""
    if error:
        logger.info(f'Slack OAuth declined by user: {error}')
        return HTMLResponse(
            '<h1>Installation cancelled</h1>'
            '<p>The app was not installed. You can close this window '
            "or <a href='/slack/install'>try again</a>.</p>",
            status_code=200,
        )

    try:
        verify_state(state)
    except InvalidOAuthStateError as e:
        logger.warning(f'Rejected Slack OAuth callback: {e}')
        return JSONResponse({'error': 'invalid_state', 'detail': str(e)}, status_code=400)

    if not code:
        return JSONResponse({'error': 'missing_code'}, status_code=400)

    try:
        payload = await exchange_code_for_token(code)
    except TokenExchangeError as e:
        logger.warning(f'Slack token exchange failed: {e.slack_error}')
        return JSONResponse({'error': 'token_exchange_failed', 'detail': e.slack_error}, status_code=400)

    installation = persist_installation(session, payload)
    logger.info(f'Slack app installed for team_id={installation.team_id}')

    return HTMLResponse(
        f"<h1>Installed</h1><p>Connected to <b>{installation.team_name or ''}</b> "
        f'(<code>{installation.team_id}</code>).</p>',
        status_code=200
    )