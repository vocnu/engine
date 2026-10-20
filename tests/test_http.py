"""Bounded retries: paid POSTs whose failure may have been billed are replayed at most once."""
import pytest
import requests

from vocnu_engine import http
from vocnu_engine.errors import ProviderError


class Response:
    def __init__(self, status):
        self.status_code, self.ok, self.headers = status, 200 <= status < 300, {}


class Session:
    def __init__(self, *outcomes):
        self.outcomes, self.calls = list(outcomes), 0

    def request(self, method, url, timeout=None, **kwargs):
        self.calls += 1
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return Response(outcome)


def post(session, **options):
    return http.request('POST', 'https://api.example.com/v1/speech', provider='Example', session=session,
                        sleep=lambda seconds: None, **options)


PAID = {'attempts': 8, 'ambiguous_retries': 1}


@pytest.mark.parametrize('failure', [503, 500, requests.ReadTimeout(), requests.ConnectionError()])
def test_a_paid_post_is_replayed_at_most_once_after_a_5xx_or_timeout(failure):
    session = Session(failure, failure, 200)
    with pytest.raises(ProviderError):
        post(session, **PAID)
    assert session.calls == 2
    session = Session(failure, 200)
    assert post(session, **PAID).ok and session.calls == 2


def test_rate_limits_are_not_billed_and_keep_every_attempt():
    session = Session(429, 429, 429, 429, 429, 200)
    assert post(session, **PAID).ok and session.calls == 6
    session = Session(*[429] * 9)
    with pytest.raises(ProviderError):
        post(session, **PAID)
    assert session.calls == 8
    # 429s in between do not reset the single replay of an ambiguous failure.
    session = Session(429, 503, 429, 503, 200)
    with pytest.raises(ProviderError):
        post(session, **PAID)
    assert session.calls == 4


def test_other_requests_keep_retrying_every_transient_failure():
    session = Session(503, requests.ReadTimeout(), 502, 200)
    assert post(session).ok and session.calls == 4


@pytest.mark.parametrize('max_wait, waited', [({}, 60.0), ({'max_wait': 5.0}, 5.0)])
def test_the_wait_before_a_new_try_is_capped_whatever_retry_after_asks(monkeypatch, max_wait, waited):
    class Full(Session):
        def request(self, method, url, timeout=None, **kwargs):
            response = super().request(method, url, timeout=timeout, **kwargs)
            response.headers = {'Retry-After': '300'} if response.status_code == 429 else {}
            return response

    waits = []
    monkeypatch.setattr(http.random, 'uniform', lambda a, b: 0.0)
    assert http.request('POST', 'https://api.example.com/v1/responses', provider='Example', session=Full(429, 200),
                        sleep=waits.append, **max_wait).ok
    assert waits == [waited]


@pytest.mark.parametrize('options', [{}, PAID])
def test_a_rejected_request_is_not_retried(options):
    session = Session(400)
    with pytest.raises(ProviderError) as raised:
        post(session, **options)
    assert session.calls == 1 and raised.value.status == 400
