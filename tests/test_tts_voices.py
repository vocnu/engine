import base64

import pytest

from vocnu_engine import tts
from vocnu_engine.costs import Receipt
from vocnu_engine.errors import ProviderError


def test_a_narrator_set_for_a_language_reads_every_line_of_it(monkeypatch):
    monkeypatch.setenv('VOCNU_NARRATOR_LT', 'narrator0000000000lt')
    narrator = {'id': 'narrator0000000000lt', 'name': 'Narrator'}
    assert tts.voice_for('male', 'lt') == tts.voice_for('female', 'lt') == narrator
    assert tts.voice_for('male', 'de') == tts.NATIVE_VOICES[('de', 'male')]
    assert tts.voice_for({'id': 'community0000000001', 'name': 'Jonas'}, 'lt')['id'] == 'community0000000001'


def test_without_a_narrator_lithuanian_has_the_stock_voices():
    assert tts.narrator_for('lt') is None
    assert tts.voice_for('male', 'lt') == tts.VOICES['male']


def test_a_narrator_must_be_an_elevenlabs_voice_id(monkeypatch):
    monkeypatch.setenv('VOCNU_NARRATOR_LT', '../../v1/user')
    with pytest.raises(ValueError):
        tts.voice_for('female', 'lt')


def test_site_languages_have_native_narrators_of_both_genders():
    for language in ('ar', 'de', 'es', 'fr', 'hi', 'id', 'it', 'ja', 'ko', 'pl', 'pt', 'ru', 'tr', 'uk', 'zh'):
        female, male = tts.voice_for('female', language), tts.voice_for('male', language)
        assert female != tts.VOICES['female'] and male != tts.VOICES['male']
        assert female['id'] != male['id']


def test_other_languages_and_voices_keep_stock_voices():
    assert tts.voice_for('male', 'nl') == tts.VOICES['male']
    assert tts.voice_for('male') == tts.VOICES['male']


def test_lithuanian_is_spoken_by_eleven_v4_and_other_languages_by_v3(monkeypatch, tmp_path):
    monkeypatch.setenv('ELEVENLABS_API_KEY', 'test-key')
    models = []

    def fake_request(method, url, **kwargs):
        models.append(kwargs['json']['model_id'])
        return FakeResponse()

    monkeypatch.setattr('vocnu_engine.tts.http.request', fake_request)
    tts.synthesize('Labas', 'lt', 'male', tmp_path, Receipt(max_usd=1))
    tts.synthesize('Hallo', 'de', 'male', tmp_path, Receipt(max_usd=1))

    assert models == ['eleven_v4', 'eleven_v3']


def test_unknown_voice_is_rejected():
    with pytest.raises(ValueError):
        tts.voice_for('robot', 'lt')


class FakeResponse:
    headers = {'character-cost': '5'}

    def json(self):
        return {'audio_base64': base64.b64encode(b'mp3').decode(), 'alignment': None, 'normalized_alignment': None}


def test_a_withdrawn_library_voice_falls_back_to_the_stock_voice(monkeypatch, tmp_path):
    monkeypatch.setenv('ELEVENLABS_API_KEY', 'test-key')
    native = tts.voice_for('female', 'de')['id']
    asked = []

    def fake_request(method, url, **kwargs):
        asked.append(url.split('/text-to-speech/')[1].split('/')[0])
        if asked[-1] == native:
            raise ProviderError('ElevenLabs rejected the request (HTTP 404).', 404)
        return FakeResponse()

    monkeypatch.setattr('vocnu_engine.tts.http.request', fake_request)
    path = tts.synthesize('Hallo', 'de', 'female', tmp_path, Receipt(max_usd=1))

    assert path.read_bytes() == b'mp3'
    assert asked == [native, tts.VOICES['female']['id']]


def test_other_provider_failures_are_not_hidden_by_the_fallback(monkeypatch, tmp_path):
    monkeypatch.setenv('ELEVENLABS_API_KEY', 'test-key')

    def fake_request(method, url, **kwargs):
        raise ProviderError('ElevenLabs is unavailable right now (HTTP 503).', 503)

    monkeypatch.setattr('vocnu_engine.tts.http.request', fake_request)
    with pytest.raises(ProviderError):
        tts.synthesize('Hallo', 'de', 'female', tmp_path, Receipt(max_usd=1))
