from __future__ import annotations
import hmac
import time
from hashlib import sha256

SIGNATURE_VERSION = 'v0'
SIGNATURE_HEADER = 'X-Slack-Signature'
TIMESTAMP_HEADER = 'X-Slack-Request-Timestamp'
MAX_TIMESTAMP_SKEW_SECONDS = 60 * 5

def compute_signature(body: bytes, timestamp: str, signing_secret: str) -> str:
    """Return the expected signature for a request."""
    sig_basestring = f'{SIGNATURE_VERSION}:{timestamp}:'.encode() + body
    digest = hmac.new(signing_secret.encode(), sig_basestring, sha256).hexdigest()
    return f'{SIGNATURE_VERSION}={digest}'

def check_slack_signature(body: bytes, timestamp: str | None, signature: str | None, signing_secret: str) -> str | None:
    if not signature:
        return 'missing_signature_header'
    if not timestamp:
        return 'missing_timestamp_header'

    try:
        request_time = int(timestamp)
    except ValueError:
        return 'malformed_timestamp'

    if abs(time.time() - request_time) > MAX_TIMESTAMP_SKEW_SECONDS:
        return 'stale_timestamp'

    expected = compute_signature(body, timestamp, signing_secret)

    if not hmac.compare_digest(expected, signature):
        return 'signature_mismatch'

    return None

def verify_slack_signature(body: bytes, timestamp: str, signature: str, signing_secret: str) -> bool:
    return check_slack_signature(body, timestamp, signature, signing_secret) is None