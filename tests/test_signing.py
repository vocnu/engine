import hashlib
import hmac

import pytest

from vocnu_engine import signing


def test_signature_format_matches_the_contract():
    body = b'{"job_id":"abc"}'
    expected = 'v1=' + hmac.new(b'secret', b'1700000000.' + body, hashlib.sha256).hexdigest()
    assert signing.signature('secret', 1700000000, body) == expected
    headers = signing.sign('secret', body, timestamp=1700000000)
    assert headers == {'X-Vocnu-Timestamp': '1700000000', 'X-Vocnu-Signature': expected}


def test_round_trip_and_rejections():
    body = b'{"a":1}'
    headers = signing.sign('s3cret', body, timestamp=1000)
    signing.verify('s3cret', body, headers['X-Vocnu-Timestamp'], headers['X-Vocnu-Signature'], now=1100)
    with pytest.raises(signing.InvalidSignature):
        signing.verify('s3cret', b'{"a":2}', '1000', headers['X-Vocnu-Signature'], now=1100)
    with pytest.raises(signing.InvalidSignature):
        signing.verify('other', body, '1000', headers['X-Vocnu-Signature'], now=1100)
    with pytest.raises(signing.InvalidSignature, match='stale'):
        signing.verify('s3cret', body, '1000', headers['X-Vocnu-Signature'], now=1000 + 301)
    with pytest.raises(signing.InvalidSignature, match='missing'):
        signing.verify('s3cret', body, None, headers['X-Vocnu-Signature'], now=1100)
    with pytest.raises(signing.InvalidSignature, match='bad timestamp'):
        signing.verify('s3cret', body, 'soon', headers['X-Vocnu-Signature'], now=1100)


def test_empty_secret_is_refused():
    with pytest.raises(ValueError):
        signing.sign('', b'x')
