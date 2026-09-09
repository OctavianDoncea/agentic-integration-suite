"""FastAPI application: Slack install and OAuth callback endpoints, handles events"""
from __future__ import annotations
import logging
import json
from fastapi import Depends, FastAPI, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, PlainTextResponse
from sqlalchemy.orm import Session
from agentic_suite.config import get_settings
from agentic_suite.db import get_session
from agentic_suite.integrations.slack.oauth import InvalidOAuthStateError, TokenExchangeError, build_authorize_url, exchange_code_for_token, generate_state, persist_installation, verify_state
from agentic_suite.integrations.slack.signature import check_slack_signature, SIGNATURE_HEADER, TIMESTAMP_HEADER
from agentic_suite.integrations.slack.events import dispatcher, deduplicator
from agentic_suite.logging_config import configure_logging

logger = logging.getLogger(__name__)
configure_logging()

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

@app.post('/slack/events', tags=['slack'])
async def slack_events(request: Request):
    raw_body = await request.body()

    failure_reason = check_slack_signature(
        body=raw_body,
        timestamp=request.headers.get(TIMESTAMP_HEADER),
        signature=request.headers.get(SIGNATURE_HEADER),
        signing_secret=get_settings().slack_signing_secret
    )
    if failure_reason is not None:
        logger.warning(f'Rejected Slack event delivery: {failure_reason}')
        return PlainTextResponse('invalid signature', status_code=401)

    try:
        payload = json.loads(raw_body)
    except json.JSONDecodeError:
        logger.warning('Signed Slack delivery had a malformed JSON body')
        return PlainTextResponse('bad request', status_code=400)

    payload_type = payload.get('type')

    if payload_type == 'url_verification':
        return JSONResponse({'challenge': payload.get('challenge')})

    if payload_type == 'event_callback':
        retry_num = request.headers.get('X-Slack-Retry-Num')
        if retry_num:
            logger.info(f"Slack retry #{retry_num} (reason: {request.headers.get('X-Slack-Retry-Reason')})")

        if deduplicator.is_duplicate(payload.get('event_id')):
            logger.info(f'Dropped duplicate delivery of event_id={payload.get("event_id")}')
            return JSONResponse({'ok': True, 'status': 'duplicate'})

        status = dispatcher.dispatch(payload)
        return JSONResponse({'ok': True, 'status': status})

    logger.info(f'Ignoring unrecognized Slack payload type: {payload_type}')
    return JSONResponse({'ok': True, 'status': 'ignored'})