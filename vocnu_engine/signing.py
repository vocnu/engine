"""HMAC request signing shared by Rails and the worker (Vocnu's worker contract).

    X-Vocnu-Timestamp: <unix seconds>
    X-Vocnu-Signature: v1=<hex HMAC-SHA256(secret, "<timestamp>.<raw body bytes>")>
"""
import hashlib
import hmac
import time

TIMESTAMP_HEADER = 'X-Vocnu-Timestamp'
SIGNATURE_HEADER = 'X-Vocnu-Signature'
TOLERANCE_SECONDS = 300


class InvalidSignature(Exception):
    pass


def _key(secret):
    if isinstance(secret, str):
        secret = secret.encode()
    if not secret:
        raise ValueError('Empty signing secret')
    return secret


def signature(secret, timestamp, body):
    mac = hmac.new(_key(secret), f'{int(timestamp)}.'.encode() + body, hashlib.sha256)
    return 'v1=' + mac.hexdigest()


def sign(secret, body, timestamp=None):
    """Headers for a signed request with this exact raw body."""
    timestamp = int(time.time() if timestamp is None else timestamp)
    return {TIMESTAMP_HEADER: str(timestamp), SIGNATURE_HEADER: signature(secret, timestamp, body)}


def verify(secret, body, timestamp, provided, now=None, tolerance=TOLERANCE_SECONDS):
    """Raise InvalidSignature unless `provided` signs `body` at a fresh `timestamp`."""
    if not timestamp or not provided:
        raise InvalidSignature('missing signature')
    try:
        stamp = int(timestamp)
    except (TypeError, ValueError):
        raise InvalidSignature('bad timestamp') from None
    now = time.time() if now is None else now
    if abs(now - stamp) > tolerance:
        raise InvalidSignature('stale timestamp')
    if not hmac.compare_digest(signature(secret, stamp, body).encode(), str(provided).encode()):
        raise InvalidSignature('signature mismatch')
