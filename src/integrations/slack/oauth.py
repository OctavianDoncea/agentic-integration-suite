"""Slack Oauth2: authorize URL, state handling, token exchange, persistence, retrieval, and revocation handling."""
from __future__ import annotations
import base64
import hmac
import secrets
import time
import httpx
from datetime import datetime, timezone, timedelta
from hashlib import sha256
from typing import Any
from urllib.parse import urlencode
from sqlalchemy.orm import Session
from agentic_suite.config import get_settings
from agentic_suite.integrations.slack.models import SlackInstallation

SLACK_AUTHORIZE_URL = 'https://slack.com/oauth/v2/authorize'
SLACK_ACCESS_URL = 'https://slack.com/api/oauth.v2.access'
REAUTH_REQUIRED_ERRORS = frozenset({'invalid_auth', 'token_revoked', 'token_expired', 'account_inactive', 'not_authed'})

class SlackOAuthError(RuntimeError):
    """Base class for OAuth-flow failures."""


class InvalidOAuthStateError(SlackOAuthError):
    """'State' parameter was missing, malformed, expired, replayed, or forged"""


class TokenExchangeError(SlackOAuthError):
    def __init__(self, slack_error: str):
        super().__init__(f'Slack token exchange failed: {slack_error}')


class InstallationNotFoundError(SlackOAuthError):
    def __init__(self, team_id: str):
        self.team_id = team_id
        super().__init__(f'No Slack installation found for team_id={team_id!r}.')


class ReauthRequiredError(SlackOAuthError):
    def __init__(self, team_id: str, slack_error: str | None = None):
        self.team_id = team_id
        self.slack_error = slack_error
        super().__init__(
            f'Slack installation for team_id={team_id!r} needs to be re-authorized'
            + (f' (Slack reported: {slack_error}).' if slack_error else '.')
        )


class OAuthStateStore:
    """Issues and verifies signed, single-use, time-limited state tokens."""
    def __init__(self) -> None:
        self._consumed: dict[str, float] = {}

    @staticmethod
    def _sign(payload: str) -> str:
        secret = get_settings().slack_client_secret.encode()
        return hmac.new(secret, payload.encode(), sha256).hexdigest()

    def generate_state(self) -> str:
        nonce = secrets.token_urlsafe(24)
        payload = f'{nonce}:{int(time.time())}'
        raw = f'{payload}:{self._sign(payload)}'
        
        return base64.urlsafe_b64encode(raw.encode()).decode().rstrip('=')

    def verify_state(self, state: str | None) -> None:
        if not state:
            raise InvalidOAuthStateError('Missing state parameter.')

        self._remove_expired()

        try:
            padded = state + '=' * (-len(state) % 4)
            raw = base64.urlsafe_b64decode(padded.encode()).decode()
            nonce, issued_at_str, signature = raw.rsplit(':', 2)
        except (ValueError, UnicodeDecodeError) as e:
            raise InvalidOAuthStateError(f'Malformed state parameter.') from e

        expected = self._sign(f'{nonce}:{issued_at_str}')

        if not hmac.compare_digest(expected, signature):
            raise InvalidOAuthStateError('State signature verification failed.')

        try:
            issued_at = int(issued_at_str)
        except ValueError as e:
            raise InvalidOAuthStateError('Malformed state parameter.') from e

        ttl = get_settings().oauth_state_ttl_seconds
        if time.time() - issued_at > ttl:
            raise InvalidOAuthStateError('State parameter has expired.')

        if nonce in self._consumed:
            raise InvalidOAuthStateError('State parameter has already been used.')

        self._consumed[nonce] = time.time()

    def _remove_expired(self) -> None:
        cutoff = time.time() - get_settings().oauth_state_ttl_seconds
        for nonce in [n for n, seen in self._consumed.items() if seen < cutoff]:
            del self._consumed[nonce]

    def clear(self) -> None:
        self._consumed.clear()


state_store = OAuthStateStore()

def generate_state() -> str:
    return state_store.generate_state()

def verify_state(state: str | None) -> None:
    state_store.verify_state(state)

def build_authorize_url(state: str) -> str:
    settings = get_settings()
    query = urlencode(
        {
            'client_id': settings.slack_client_id,
            'scope': ','.join(settings.scope_list),
            'redirect_uri': settings.slack_redirects_uri,
            'state': state,
        }
    )

    return f'{SLACK_AUTHORIZE_URL}?{query}'

async def exchange_code_for_token(code: str) -> dict[str, Any]:
    """Trade an authorization code for a bot token."""
    settings = get_settings()
    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.post(
            SLACK_ACCESS_URL,
            data={
                'client_id': settings.slack_client_id,
                'client_secret': settings.slack_client_secret,
                'code': code,
                'redirect_uri': settings.slack_redirects_uri,
            }
        )

    response.raise_for_status()
    payload = response.json()

    if not payload.get('ok'):
        raise TokenExchangeError(payload.get('error', 'unknown_error'))

    return payload

def persist_installation(session: Session, payload: dict[str, Any]) -> SlackInstallation:
    """Insert or update the installation row from a token-exchange payload."""
    team = payload.get('team') or {}
    team_id = team.get('id')

    if not team_id:
        raise TokenExchangeError('missing_team_id')

    expires_at = None
    if payload.get('expires_in'):
        expires_at = datetime.now(timezone.utc) + timedelta(seconds=payload['expires_in'])

    installation = session.query(SlackInstallation).filter_by(team_id=team_id).one_or_none()
    if installation is None:
        installation = SlackInstallation(team_id=team_id)
        session.add(installation)

    installation.team_name = team.get('name')
    installation.access_token = payload['access_token']
    installation.refresh_token = payload.get('refresh_token')
    installation.token_expires_at = expires_at
    installation.scope = payload.get('scope', '')
    installation.bot_user_id = payload.get('bot_user_id')
    installation.needs_reauth = False

    session.flush()
    return installation

def get_valid_token(session: Session, team_id: str) -> str:
    """Return a usable bot token for a workspace."""
    installation = session.query(SlackInstallation).filter_by(team_id=team_id).one_or_none()
    if installation is None:
        raise InstallationNotFoundError(team_id)

    if installation.needs_reauth:
        raise ReauthRequiredError(team_id)

    if installation.token_expires_at is not None:
        if installation.token_expires_at <= datetime.now(timezone.utc):
            mark_needs_reauth(session, team_id)
            raise ReauthRequiredError(team_id, 'token_expired')

    return installation.access_token

def mark_needs_reauth(session: Session, team_id: str) -> None:
    installation = session.query(SlackInstallation).filter_by(team_id=team_id).one_or_none()
    if installation is not None:
        installation.needs_reauth = True
        session.flush()

def raise_for_slack_error(session: Session, team_id: str, payload: dict[str, Any]) -> None:
    """Inspect a Slack Web API response and convert auth failures into flags."""
    if payload.get('ok'):
        return

    error = payload.get('error', 'unknown_error')
    if error in REAUTH_REQUIRED_ERRORS:
        mark_needs_reauth(session, team_id)
        raise ReauthRequiredError(team_id, error)

    raise SlackOAuthError(f'Slack API call failed: {error}')