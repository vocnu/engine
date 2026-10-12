import json

from vocnu_engine import signing, titles


class Response:
    def __init__(self, status, data=None):
        self.status_code = status
        self.ok = 200 <= status < 300
        self._data = data
        self.headers = {}

    def json(self):
        if self._data is None:
            raise ValueError('no JSON')
        return self._data


class Session:
    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []

    def request(self, method, url, timeout=None, **kwargs):
        self.requests.append((method, url, kwargs))
        return self.responses.pop(0)


def test_endpoint_lives_next_to_the_https_callback():
    assert titles.endpoint('https://app.vocnu.com/api/v1/worker/events') == 'https://app.vocnu.com/api/v1/worker/glossary'
    assert titles.endpoint('http://localhost:3000/api/v1/worker/events') is None
    assert titles.endpoint(None) is None


def test_sync_sends_a_signed_request_and_returns_the_decided_terms(monkeypatch):
    monkeypatch.setenv('WORKER_SHARED_SECRET', 'secret')
    session = Session([Response(200, {'title': {'key': 'bleach', 'name': 'Bleach'}, 'language': 'lt', 'terms': [
        {'source': '黒崎一護', 'target': 'Ichigo Kurosaki'}, {'source': '', 'target': 'x'}, 'junk']})])
    terms = titles.sync('https://app.vocnu.com/api/v1/worker/events', 'Bleach', 'lt',
                        [{'source': 'ホロー', 'target': 'Hollow'}], 'job-1', session=session)
    assert terms == [{'source': '黒崎一護', 'target': 'Ichigo Kurosaki'}]
    method, url, kwargs = session.requests[0]
    assert (method, url) == ('POST', 'https://app.vocnu.com/api/v1/worker/glossary')
    body = kwargs['data']
    assert json.loads(body) == {'title': 'Bleach', 'language': 'lt', 'job_id': 'job-1',
                                'terms': [{'source': 'ホロー', 'target': 'Hollow'}]}
    headers = kwargs['headers']
    signing.verify('secret', body, headers[signing.TIMESTAMP_HEADER], headers[signing.SIGNATURE_HEADER])


def test_sync_is_best_effort(monkeypatch):
    monkeypatch.setenv('WORKER_SHARED_SECRET', 'secret')
    monkeypatch.setattr('vocnu_engine.http.time.sleep', lambda s: None)
    url = 'https://app.vocnu.com/api/v1/worker/events'
    assert titles.sync(url, 'Bleach', 'lt', [], 'job-1', session=Session([Response(422)])) is None
    assert titles.sync(url, 'Bleach', 'lt', [], 'job-1', session=Session([Response(200)])) is None
    assert titles.sync(url, '', 'lt', [], 'job-1', session=Session([])) is None
    assert titles.sync(None, 'Bleach', 'lt', [], 'job-1', session=Session([])) is None


def test_title_keys_match_the_rails_library():
    assert titles.key_for('Bleach: Sennen Kessen-hen') == 'bleach-sennen-kessen-hen'
    assert titles.key_for('  Šešėlių Žaidimai!! ') == 'seseliu-zaidimai'
    assert titles.key_for('進撃の巨人') == ''
    assert len(titles.key_for('a ' * 60)) <= 80 and not titles.key_for('a ' * 60).endswith('-')


def test_local_runs_read_the_public_library_and_never_write_to_it():
    library = {'title': {'key': 'bleach', 'terms': {'lt': [{'source': '黒崎一護', 'rendering': 'Ičigas'}]}}}
    session = Session([Response(200, library)])
    decided = titles.sync(None, 'Bleach', 'lt', [], 'job-1', names_api='https://names.test/api/v1/names/',
                          session=session)
    assert decided == [{'source': '黒崎一護', 'target': 'Ičigas'}]
    assert session.requests == [('GET', 'https://names.test/api/v1/names/bleach', {})]
    # New terms stay in the job: the public library is read-only.
    assert titles.sync(None, 'Bleach', 'lt', [{'source': 'x', 'target': 'y'}], 'job-1',
                       names_api='https://names.test/api/v1/names', session=Session([])) is None


def test_a_title_the_library_lacks_or_no_library_means_no_terms():
    assert titles.sync(None, 'Unknown Film', 'lt', [], 'job-1', names_api='https://names.test/api/v1/names',
                       session=Session([Response(404)])) is None
    assert titles.sync(None, 'Bleach', 'lt', [], 'job-1', session=Session([])) is None
