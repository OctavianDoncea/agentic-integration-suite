from __future__ import annotations
import hmac
import time
from hashlib import sha256
import pytest
from agentic_suite.integrations.slack.signature import MAX_TIMESTAMP_SKEW_SECONDS, check_slack_signature, compute_signature, verify_slack_signature

SECRET = 'test-signing-secret'
BODY = b'{"type":"event_callback","event":{"type":"message","text":"hello"}}'

def _sign(body: bytes, timestamp: str, secret: str = SECRET) -> str:
    basestring = b'v0:' + timestamp.encode() + b':' + body
    return 'v0=' + hmac.new(secret.encode(), basestring, sha256).hexdigest()

def _now() -> str:
    return str(int(time.time()))

def test_correctly_signed_payload_passes():
    timestamp = _now()
    assert verify_slack_signature(BODY, timestamp, _sign(BODY, timestamp), SECRET)

def test_tampered_body_fails():
    timestamp = _now()
    signature = _sign(BODY, timestamp)
    tampered = BODY.replace(b'hello', b'h3ll0')

    assert not verify_slack_signature(tampered, timestamp, signature, SECRET)
    assert check_slack_signature(tampered, timestamp, signature, SECRET) == 'signature_mismatch'

def test_stale_timestamp_fails_even_with_a_valid_signature():
    old = str(int(time.time() - MAX_TIMESTAMP_SKEW_SECONDS - 60))
    signature = _sign(BODY, old)

    assert not verify_slack_signature(BODY, old, signature, SECRET)
    assert check_slack_signature(BODY, old, signature, SECRET) == 'stale_timestamp'

def test_far_future_timestamp_fails():
    future = str(int(time.time() + MAX_TIMESTAMP_SKEW_SECONDS + 60))
    assert check_slack_signature(BODY, future, _sign(BODY, future), SECRET) == 'stale_timestamp'

def test_timestamp_just_inside_the_window_passes():
    recent = str(int(time.time() + MAX_TIMESTAMP_SKEW_SECONDS - 30))
    assert verify_slack_signature(BODY, recent, _sign(BODY, recent), SECRET)

def test_wrong_secret_fails():
    timestamp = _now()
    assert not verify_slack_signature(BODY, timestamp, _sign(BODY, timestamp, secret='attacker-guess'), SECRET)

def test_reserialized_body_fails():
    import json

    timestamp = _now()
    signature = _sign(BODY, timestamp)
    reserialized = json.dumps(json.loads(BODY)).encode()

    assert reserialized != BODY
    assert not verify_slack_signature(reserialized, timestamp, signature, SECRET)

@pytest.mark.parametrize(
    ('timestamp', 'signature', 'expected'),
    [
        (None, 'v0=abc', 'missing_timestamp_header'),
        ('1700000000', None, 'missing_signature_header'),
        ('not-a-number', 'v0=abc', 'malformed_timestamp')
    ]
)
def test_malformed_headers_are_rejected_with_a_reason(timestamp, signature, expected):
    assert check_slack_signature(BODY, timestamp, signature, SECRET) == expected

def test_compute_signature_matches_the_reference_implementation():
    timestamp = _now()
    assert compute_signature(BODY, timestamp, SECRET) == _sign(BODY, timestamp)