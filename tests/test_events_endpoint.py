from __future__ import annotations
import hmac
import json
import logging
import time
import pytest
from collections.abc import Iterator
from hashlib import sha256
from fastapi.testclient import TestClient
from agentic_suite.integrations.slack.events import deduplicator
from agentic_suite.integrations.slack.signature import SIGNATURE_HEADER, TIMESTAMP_HEADER, MAX_TIMESTAMP_SKEW_SECONDS
from agentic_suite.main import app

SECRET = 'test-signing-secret'

@pytest.fixture
def client() -> Iterator[TestClient]:
    deduplicator.clear()
    with TestClient(app) as test_client:
        yield test_client
    deduplicator.clear()

def _signed_headers(body: bytes, timestamp: str | None = None) -> dict[str, str]:
    timestamp = timestamp or str(int(time.time()))
    basestring = b'v0:' + timestamp.encode() + b':' + body
    digest = hmac.new(SECRET.encode(), basestring, sha256).hexdigest()
    return {TIMESTAMP_HEADER: timestamp, SIGNATURE_HEADER: f'v0={digest}', 'Content-Type': 'application/json'}

def _post(client: TestClient, payload: dict, **kwargs):
    body = json.dumps(payload).encode()
    headers = kwargs.pop('headers', None) or _signed_headers(body, **kwargs)
    return client.post('/slack/events', content=body, headers=headers)

def _event_callback(event_id: str = 'Ev001', text: str = 'hello') -> dict:
    return {
        'type': 'event_callback',
        'team_id': 'T012AB3C4',
        'event_id': event_id,
        'event': {
            'type': 'message',
            'channel': 'C012AB3CD',
            'user': 'U012AB3CD',
            'text': text,
            'ts': '1700000000.000100'
        }
    }

def test_url_verification_echoes_the_challenge(client: TestClient):
    payload = {'type': 'url_verification', 'token': 'x', 'challenge': '3eZbrw1a'}
    response = _post(client, payload)

    assert response.status_code == 200
    assert response.json() == {'challenge': '3eZbrw1a'}

def test_url_verification_still_requires_a_valid_signature(client: TestClient):
    body = json.dumps({'type': 'url_verification', 'challenge': 'x'}).encode()
    response = client.post('/slack/events', content=body, headers={TIMESTAMP_HEADER: str(int(time.time())), SIGNATURE_HEADER: 'v0=bad'})
    assert response.status_code == 401

def test_valid_event_callback_is_accepted(client: TestClient):
    response = _post(client, _event_callback())
    assert response.status_code == 200
    assert response.json() == {'ok': True, 'status': 'handled'}

def test_message_event_is_logged(client: TestClient, caplog):
    with caplog.at_level(logging.INFO, logger='slack.events'):
        _post(client, _event_callback(text='deploy finished'))

    logged = [json.loads(r.message) for r in caplog.records if r.message.startswith('{')]
    assert any(
        entry['event'] == 'slack.message_received'
        and entry['channel'] == 'C012AB3CD'
        and entry['text_length'] == len('deploy finished')
        for entry in logged
    )

def test_bot_authored_messages_are_ignored(client: TestClient, caplog):
    payload = _event_callback()
    payload['event']['bot_id'] = 'B012AB3CD'

    with caplog.at_level(logging.INFO, logger='slack.events'):
        response = _post(client, payload)

    assert response.status_code == 200
    assert 'slack.message_received' not in caplog.text

def test_unknown_event_type_returns_200_not_an_error(client: TestClient):
    payload = _event_callback()
    payload['event']['type'] = 'reaction_added'

    response = _post(client, payload)
    assert response.status_code == 200
    assert response.json()['status'] == 'unhandled'

def test_unknown_payload_type_returns_200(client: TestClient):
    response = _post(client, {'type': 'something_new', 'team_id': 'T1'})
    assert response.status_code == 200
    assert response.json()['status'] == 'ignored'

def test_retried_delivery_is_deduplicated(client: TestClient):
    assert _post(client, _event_callback('Ev999')).json()['status'] == 'handled'

    retry = _post(client, _event_callback('Ev999'))
    assert retry.status_code == 200
    assert retry.json()['status'] == 'duplicate'

def test_distinct_event_ids_are_both_processed(client: TestClient):
    assert _post(client, _event_callback('EvA')).json()['status'] == 'handled'
    assert _post(client, _event_callback('EvB')).json()['status'] == 'handled'

def test_invalid_signature_is_rejected(client: TestClient):
    body = json.dumps(_event_callback()).encode()
    response = client.post('/slack/events', content=body, headers={TIMESTAMP_HEADER: str(int(time.time())), SIGNATURE_HEADER: 'v0=deadbeef'})
    assert response.status_code == 401

def test_missing_signature_headers_are_rejected(client: TestClient):
    response = client.post('/slack/events', content=b'{}')
    assert response.status_code == 401

def test_stale_delivery_is_rejected(client: TestClient):
    stale = str(int(time.time() - MAX_TIMESTAMP_SKEW_SECONDS - 60))
    response = _post(client, _event_callback(), timestamp=stale)
    assert response.status_code == 401

def test_tampered_body_is_rejected(client: TestClient):
    original = json.dumps(_event_callback()).encode()
    headers = _signed_headers(original)
    tampered = original.replace(b'hello', b'h4ck3d')
    response = client.post('/slack/events', content=tampered, headers=headers)
    assert response.status_code == 401

def test_rejection_response_leaks_no_reason(client: TestClient):
    body = json.dumps(_event_callback()).encode()
    response = client.post('/slack/events', content=body, headers={TIMESTAMP_HEADER: str(int(time.time())), SIGNATURE_HEADER: 'v0=bad'})

    for leak in ('mismatch', 'stale', 'timestamp', 'secret'):
        assert leak not in response.text.lower()

def test_signed_but_malformed_json_is_a_400(client: TestClient):
    body = b'{not json'
    response = client.post('/slack/events', content=body, headers=_signed_headers(body))
    assert response.status_code == 400