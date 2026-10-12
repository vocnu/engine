"""ElevenLabs text-to-speech with stock voices and native narrators per language, cached by content hash.

Speech comes from the with-timestamps endpoint: the MP3 plus character timestamps, which time the narration subtitles.
Both are cached (the timestamps as JSON next to the MP3).
"""
import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import re
import threading

from . import http
from .config import secret, setting
from .costs import ELEVENLABS_USD_PER_CREDIT
from .errors import ProviderError
from .languages import get

MODEL = 'eleven_v3'
# eleven_v4 lists Lithuanian among its languages (v3 does not) and halved round-trip ASR character errors
# for the Lithuanian narrator on a demo scene (2026-10-08: 5.4 % vs 10.4 % on v3).
LANGUAGE_MODELS = {'lt': 'eleven_v4'}
OUTPUT_FORMAT = 'mp3_44100_192'
API = 'https://api.elevenlabs.io/v1/text-to-speech/{voice_id}/with-timestamps'
ALIGNMENTS = ('alignment', 'normalized_alignment')
# Premade ElevenLabs voices; eleven_v3 speaks every supported language with them. Chosen 2026-10-07 from
# eight stock voices by Lithuanian round-trip ASR (0 % character errors) and speaking rate (~14.5 chars/s).
VOICES = {
    'female': {'id': 'XrExE9yKIg1WjnnlVkGX', 'name': 'Matilda'},
    'male': {'id': 'JBFqnCBsd6RMkjVDRZzb', 'name': 'George'},
}
# Native narrators replace a stock voice for one language: ElevenLabs Voice Library narrators recorded by native
# speakers, chosen 2026-10-07 from the most-used narrative voices per language (rate 1x, notice period >= 180 days)
# by round-trip ASR of a test sentence. Library voices can be withdrawn, so synthesize() falls back to the stock
# voice when one is gone.
NATIVE_VOICES = {
    ('ar', 'female'): {'id': 'XTa3iQyMA6f1qrI4F6kZ', 'name': 'Sara'},
    ('ar', 'male'): {'id': 'R6nda3uM038xEEKi7GFl', 'name': 'Anas'},
    ('de', 'female'): {'id': '7eVMgwCnXydb3CikjV7a', 'name': 'Lea'},
    ('de', 'male'): {'id': 'FTNCalFNG5bRnkkaP5Ug', 'name': 'Otto'},
    ('es', 'female'): {'id': '2rigMbVWLdqtBSCahJFX', 'name': 'Tatiana Martin'},
    ('es', 'male'): {'id': 'l1zE9xgNpUTaQCZzpNJa', 'name': 'Alberto Rodríguez'},
    ('fr', 'female'): {'id': 'txtf1EDouKke753vN8SL', 'name': 'Jeanne'},
    ('fr', 'male'): {'id': 'aQROLel5sQbj1vuIVi6B', 'name': 'Nicolas'},
    ('hi', 'female'): {'id': '1qEiC6qsybMkmnNdVMbK', 'name': 'Monika Sogam'},
    ('hi', 'male'): {'id': 'zgqefOY5FPQ3bB7OZTVR', 'name': 'Niraj'},
    ('id', 'female'): {'id': 'o5s6XRBkPSTD4syv6mZg', 'name': 'Arunika'},
    ('id', 'male'): {'id': 'RWiGLY9uXI70QL540WNd', 'name': 'Putra'},
    ('it', 'female'): {'id': 'gfKKsLN1k0oYYN9n2dXX', 'name': 'Violetta'},
    ('it', 'male'): {'id': 'W71zT1VwIFFx3mMGH2uZ', 'name': 'MarcoTrox'},
    ('ja', 'female'): {'id': '4lOQ7A2l7HPuG7UIHiKA', 'name': 'Kyoko'},
    ('ja', 'male'): {'id': '3JDquces8E8bkmvbh6Bc', 'name': 'Otani'},
    ('ko', 'female'): {'id': 'uyVNoMrnUku1dZyVEXwD', 'name': 'Anna Kim'},
    ('ko', 'male'): {'id': 'PDoCXqBQFGsvfO0hNkEs', 'name': 'Chris'},
    ('pl', 'female'): {'id': 'N0GCuK2B0qwWozQNTS8F', 'name': 'Magdalena'},
    ('pl', 'male'): {'id': 'hIssydxXZ1WuDorjx6Ic', 'name': 'Adam'},
    ('pt', 'female'): {'id': '33B4UnXyTNbgLmdEDh5P', 'name': 'Keren'},
    ('pt', 'male'): {'id': 'CstacWqMhJQlnfLPxRG4', 'name': 'Will'},
    ('ru', 'female'): {'id': 'ymDCYd8puC7gYjxIamPt', 'name': 'Marina'},
    ('ru', 'male'): {'id': 'oKxkBkm5a8Bmrd1Whf2c', 'name': 'Prince Nur'},
    ('tr', 'female'): {'id': 'NNn9dv8zq2kUo7d3JSGG', 'name': 'Derya'},
    ('tr', 'male'): {'id': 'DpEiQMnL9WxMsymCsjPY', 'name': 'Emrullah Uzun'},
    ('uk', 'female'): {'id': '2OXYbN1uGomXXJtv9Dq6', 'name': 'Mariya Maro'},
    ('uk', 'male'): {'id': 'Ntd0iVwICtUtA6Fvx27M', 'name': 'Evgeniy Shevchenko'},
    ('zh', 'female'): {'id': 'Ca5bKgudqKJzq8YRFoAz', 'name': 'Coco Li'},
    ('zh', 'male'): {'id': '4VZIsMPtgggwNg7OXbPY', 'name': 'James Gao'},
}
# VOCNU_NARRATOR_<LANGUAGE>=<ElevenLabs voice id> (for example VOCNU_NARRATOR_LT) has one voice of your account read
# every line of that language, whichever voice a job asks for. Vocnu's hosted service reads Lithuanian this way.
# Text goes plain: stress marks from a Lithuanian accentuator made eleven_v4's stress worse (2026-10-08).
NARRATOR_SETTING = 'VOCNU_NARRATOR_{}'
VOICE_ID = re.compile(r'[A-Za-z0-9]{8,64}')
SETTINGS = {'stability': 0.5, 'similarity_boost': 0.75, 'style': 0.0, 'speed': 1.0}
SEED = 1847
GONE_STATUS = {400, 404}
PARALLEL = 4


def voice_for(name, language=None):
    """The voice for a gender ('female'/'male') and language, or a community voice given as {'id', 'name', 'gender'}."""
    if isinstance(name, dict):
        return name
    if name not in VOICES:
        raise ValueError(f'Unknown voice {name!r}')
    return narrator_for(language) or NATIVE_VOICES.get((language, name), VOICES[name])


def narrator_for(language):
    """The voice set to read every line of `language` (VOCNU_NARRATOR_<LANGUAGE>), or None."""
    if not language:
        return None
    name = NARRATOR_SETTING.format(language.upper().replace('-', '_'))
    voice_id = setting(name)
    if voice_id is None:
        return None
    if not VOICE_ID.fullmatch(voice_id):
        raise ValueError(f'{name} is not an ElevenLabs voice id')
    return {'id': voice_id, 'name': 'Narrator'}


def _stock(voice):
    return VOICES[voice.get('gender', 'female') if isinstance(voice, dict) else voice]


def estimate_usd(texts):
    return sum(len(t) for t in texts) * ELEVENLABS_USD_PER_CREDIT


def model_for(language):
    return LANGUAGE_MODELS.get(language, MODEL)


def cache_key(text, language, voice_id, seed=SEED, model=MODEL):
    material = json.dumps([model, OUTPUT_FORMAT, voice_id, language, SETTINGS, seed, text], ensure_ascii=False)
    return hashlib.sha256(material.encode()).hexdigest()


def synthesize(text, language, voice, cache_dir, receipt, *, seed=SEED, session=None):
    """Path to an MP3 for `text`; billed lines are recorded on the receipt, cache hits are free."""
    voice_id = voice_for(voice, language)['id']
    try:
        return _synthesize(text, language, voice_id, cache_dir, receipt, seed=seed, session=session)
    except ProviderError as error:
        # A library narrator its owner withdrew, or a community voice its donor took back, answers 400/404; the
        # stock voice keeps the job going.
        if voice_id == _stock(voice)['id'] or error.status not in GONE_STATUS:
            raise
        return _synthesize(text, language, _stock(voice)['id'], cache_dir, receipt, seed=seed, session=session)


def _synthesize(text, language, voice_id, cache_dir, receipt, *, seed, session):
    code = get(language).tts_code
    model = model_for(language)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f'{cache_key(text, code, voice_id, seed, model)}.mp3'
    if path.is_file() and path.stat().st_size > 0:
        return path
    body = {'text': text, 'model_id': model, 'language_code': code, 'voice_settings': SETTINGS, 'seed': seed}
    # A 429 is retried up to 8 times: concurrent jobs share the plan's concurrency limit (Creator: 5) and a refused
    # request is not billed. A 5xx or a timeout may have been billed already, so it is retried only once.
    response = http.request('POST', API.format(voice_id=voice_id), provider='ElevenLabs',
                            headers={'xi-api-key': secret('ELEVENLABS_API_KEY'), 'Accept': 'application/json'},
                            params={'output_format': OUTPUT_FORMAT}, json=body, timeout=180, session=session,
                            attempts=8, ambiguous_retries=1)
    cost = response.headers.get('character-cost', '')
    credits = int(cost) if cost.isdigit() else len(text)
    receipt.add('speech', 'elevenlabs', credits * ELEVENLABS_USD_PER_CREDIT, model=model, characters=len(text),
                credits=credits)
    audio, timestamps = parse(response)
    # The timestamps are stored first, so a cached MP3 always has its alignment beside it.
    _store(path.with_suffix('.json'), json.dumps(timestamps, ensure_ascii=False).encode())
    _store(path, audio)
    return path


def parse(response):
    """(MP3 bytes, {'alignment', 'normalized_alignment'}) from a with-timestamps answer. Empty audio stays empty: the
    take then measures as silent and is retried with another seed."""
    try:
        data = response.json()
        audio = base64.b64decode(data['audio_base64'] or '', validate=True)
    except (ValueError, KeyError, TypeError):
        raise ProviderError('ElevenLabs returned malformed speech.') from None
    return audio, {key: data[key] if isinstance(data.get(key), dict) else None for key in ALIGNMENTS}


def alignment(path):
    """The character timestamps cached with a take ({'alignment', 'normalized_alignment'}), or None."""
    try:
        data = json.loads(Path(path).with_suffix('.json').read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _store(path, data):
    partial = path.with_suffix(f'{path.suffix}.{os.getpid()}.{threading.get_ident()}.part')
    partial.write_bytes(data)
    partial.replace(path)


def synthesize_all(lines, language, voice, cache_dir, receipt, *, session=None, seeds=None, on_done=None):
    """{key: text} → {key: mp3 path}, PARALLEL requests at a time (ElevenLabs Creator allows 5).

    `seeds` ({key: seed}; SEED otherwise) asks for alternate takes of the same text. Keys with the same text and seed
    share one request, so a repeated reply is never bought twice in one batch.
    """
    seeds = seeds or {}
    groups = {}
    for key, text in lines.items():
        if text.strip():
            groups.setdefault((text, seeds.get(key, SEED)), []).append(key)
    result = {}
    with ThreadPoolExecutor(max_workers=PARALLEL) as pool:
        futures = {pool.submit(synthesize, text, language, voice, cache_dir, receipt, seed=seed, session=session): keys
                   for (text, seed), keys in groups.items()}
        for future in as_completed(futures):
            path = future.result()
            for key in futures[future]:
                result[key] = path
                if on_done:
                    on_done(key)
    return result
