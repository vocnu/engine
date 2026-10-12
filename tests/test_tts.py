"""ElevenLabs speech with character timestamps: the request, parsing, the cache and alternate takes."""
import base64

import pytest

from vocnu_engine import tts
from vocnu_engine.costs import Receipt
from vocnu_engine.errors import ProviderError

ALIGNMENT = {'characters': list('Labas.'), 'character_start_times_seconds': [0.1, 0.15, 0.2, 0.25, 0.3, 0.35],
             'character_end_times_seconds': [0.15, 0.2, 0.25, 0.3, 0.35, 0.4]}


class Answer:
    def __init__(self, data, cost='6'):
        self.data, self.headers = data, {'character-cost': cost}

    def json(self):
        if isinstance(self.data, Exception):
            raise self.data
        return self.data


def speech(audio=b'ID3 speech'):
    return Answer({'audio_base64': base64.b64encode(audio).decode(), 'alignment': ALIGNMENT,
                   'normalized_alignment': None})


@pytest.fixture
def provider(monkeypatch):
    """Requests made to ElevenLabs, answered from `answers` (a speech answer once those run out)."""
    monkeypatch.setenv('ELEVENLABS_API_KEY', 'test-key')
    made, answers = [], []

    def fake_request(method, url, **kwargs):
        made.append((method, url, kwargs))
        return answers.pop(0) if answers else speech()

    monkeypatch.setattr('vocnu_engine.tts.http.request', fake_request)
    return made, answers


def test_speech_comes_with_character_timestamps_and_both_are_cached(tmp_path, provider):
    made, _ = provider
    receipt = Receipt(max_usd=1)
    path = tts.synthesize('Labas.', 'lt', 'male', tmp_path, receipt)
    method, url, options = made[0]
    voice = tts.voice_for('male', 'lt')['id']
    assert (method, url) == ('POST', f'https://api.elevenlabs.io/v1/text-to-speech/{voice}/with-timestamps')
    assert options['params'] == {'output_format': 'mp3_44100_192'}
    assert options['headers']['Accept'] == 'application/json' and options['json']['seed'] == tts.SEED
    assert (options['attempts'], options['ambiguous_retries']) == (8, 1)
    assert path.read_bytes() == b'ID3 speech'
    assert tts.alignment(path) == {'alignment': ALIGNMENT, 'normalized_alignment': None}
    assert receipt.entries[0]['credits'] == 6
    # Cached: the same line costs nothing the second time.
    assert tts.synthesize('Labas.', 'lt', 'male', tmp_path, receipt) == path
    assert len(made) == 1 and len(receipt.entries) == 1


def test_another_seed_is_a_new_take(tmp_path, provider):
    made, _ = provider
    first = tts.synthesize('Labas.', 'lt', 'male', tmp_path, Receipt(max_usd=1))
    second = tts.synthesize('Labas.', 'lt', 'male', tmp_path, Receipt(max_usd=1), seed=tts.SEED + 1)
    assert first != second and [options['json']['seed'] for _, _, options in made] == [tts.SEED, tts.SEED + 1]


@pytest.mark.parametrize('answer', [Answer(ValueError('not JSON')), Answer({'audio_base64': '%%%'}), Answer({}),
                                    Answer(['audio'])])
def test_a_malformed_answer_is_a_provider_error_and_nothing_is_cached(tmp_path, provider, answer):
    _, answers = provider
    answers.append(answer)
    receipt = Receipt(max_usd=1)
    with pytest.raises(ProviderError):
        tts.synthesize('Labas.', 'lt', 'male', tmp_path, receipt)
    assert list(tmp_path.iterdir()) == []
    assert len(receipt.entries) == 1      # ElevenLabs billed it all the same


def test_empty_audio_is_kept_so_the_take_measures_as_silent(tmp_path, provider):
    _, answers = provider
    answers.append(speech(b''))
    path = tts.synthesize('Labas.', 'lt', 'male', tmp_path, Receipt(max_usd=1))
    assert path.read_bytes() == b'' and tts.alignment(path)['alignment'] == ALIGNMENT


def test_a_take_without_timestamps_has_no_alignment(tmp_path):
    (tmp_path / 'old.mp3').write_bytes(b'mp3')
    assert tts.alignment(tmp_path / 'old.mp3') is None


def test_a_batch_passes_seeds_and_buys_a_repeated_line_once(tmp_path, monkeypatch):
    asked = []

    def fake_synthesize(text, language, voice, cache_dir, receipt, *, seed=tts.SEED, session=None):
        asked.append((text, seed))
        return tmp_path / f'{text}-{seed}.mp3'

    monkeypatch.setattr(tts, 'synthesize', fake_synthesize)
    done = []
    paths = tts.synthesize_all({1: 'Taip.', 2: 'Taip.', 3: 'Ne.', 4: ' '}, 'lt', 'male', tmp_path, Receipt(max_usd=1),
                               seeds={3: tts.SEED + 1}, on_done=done.append)
    assert sorted(asked) == [('Ne.', tts.SEED + 1), ('Taip.', tts.SEED)]
    assert paths[1] == paths[2] and set(paths) == {1, 2, 3} and sorted(done) == [1, 2, 3]
