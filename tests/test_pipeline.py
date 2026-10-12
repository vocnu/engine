"""Whole-job contract test with real ffmpeg but fake providers (no network, no GPU)."""
import json
from pathlib import Path
import subprocess

import pytest

from vocnu_engine import mix, pipeline, titles, translate, tts
from vocnu_engine.storage import LocalStore


class Sink:
    def __init__(self):
        self.events = []
        self.terminal_sent = False

    def progress(self, stage, progress):
        self.events.append(('progress', stage, progress))

    def completed(self, payload):
        self.events.append(('completed', payload))

    def failed(self, error, receipt):
        self.events.append(('failed', error, receipt))

    def start_heartbeat(self):
        pass

    def stop_heartbeat(self):
        pass


@pytest.fixture(scope='module')
def clip(tmp_path_factory):
    path = tmp_path_factory.mktemp('media') / 'clip.mp4'
    subprocess.run(['ffmpeg', '-v', 'error', '-f', 'lavfi', '-i', 'testsrc=size=320x240:rate=25:duration=8',
                    '-f', 'lavfi', '-i', 'sine=frequency=220:sample_rate=48000:duration=8', '-c:v', 'libx264',
                    '-pix_fmt', 'yuv420p', '-c:a', 'aac', '-shortest', str(path)], check=True)
    return path


WORDS = [{'start': 0.5, 'end': 0.9, 'word': ' Hello', 'probability': 0.9},
         {'start': 0.95, 'end': 1.5, 'word': ' there.', 'probability': 0.9},
         {'start': 3.0, 'end': 3.4, 'word': ' See', 'probability': 0.9},
         {'start': 3.45, 'end': 3.9, 'word': ' you', 'probability': 0.9},
         {'start': 3.95, 'end': 4.5, 'word': ' soon.', 'probability': 0.9}]


class Providers:
    def __init__(self):
        self.hints, self.glossaries, self.spoken, self.spans, self.synced, self.decided = [], [], [], [], [], []
        self.title = ''
        self.analysis = {'glossary': [], 'new': [], 'sung': set()}
        self.stored = None
        self.quality = {}
        self.timestamps = {}   # unit id → character alignment saved with its take
        self.evidence = {}     # 'analysis' / 'translation' → the evidence each got
        self.overlay = []      # (lines, source) the song and sign translation got
        self.overlay_kinds = {}   # line index -> kind the fake song and sign translation reports (default 'song')
        self.turns = []           # units offered for speaker turns


@pytest.fixture
def fake_providers(monkeypatch, tmp_path):
    calls = Providers()

    def fake_identify(units, *, filename, source, target, receipt, session=None):
        calls.filename = filename
        return calls.title

    def fake_analyse(units, *, source, target, receipt, session=None, decided=(), evidence=None, unheard=()):
        calls.decided.append(list(decided))
        calls.evidence['analysis'] = evidence
        receipt.add('text', 'openai', 0.001)
        return calls.analysis

    def fake_sync(callback_url, title, language, terms, job_id, *, names_api=None, session=None):
        calls.synced.append((title, language, list(terms)))
        return calls.stored(terms) if calls.stored else None

    def fake_translate(units, cues, *, source, target, budgets, receipt, session=None, speakers=None, glossary=None,
                       sung=frozenset(), stats=None, evidence=None):
        calls.hints.append(speakers)
        calls.glossaries.append(glossary)
        calls.evidence['translation'] = evidence
        calls.translated = [u.id for u in units]
        if stats is not None:
            stats.update(calls.quality)
        receipt.add('text', 'openai', 0.001)
        return {u.id: {'voiceover': '' if u.id in sung else f'Eilutė {u.id}',
                       'subtitles': [f'Titras {c}' for c in u.cue_ids]} for u in units}

    def fake_tts(lines, language, voice, cache_dir, receipt, *, session=None, seeds=None, on_done=None):
        calls.spoken.extend(lines.values())
        result = {}
        for key, text in lines.items():
            path = Path(cache_dir) / f'{key}.mp3'
            path.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(['ffmpeg', '-v', 'error', '-y', '-f', 'lavfi', '-i',
                            'sine=frequency=660:sample_rate=44100:duration=0.8', '-c:a', 'libmp3lame', str(path)],
                           check=True)
            if key in calls.timestamps:
                path.with_suffix('.json').write_text(json.dumps(calls.timestamps[key]), encoding='utf-8')
            receipt.add('speech', 'elevenlabs', 0.002, characters=len(text))
            result[key] = path
            if on_done:
                on_done(key)
        return result

    real_mix = mix.mix

    def recording_mix(source, info, placements, line_paths, folder):
        calls.spans.extend((p.start, p.end) for p in placements)   # where the original dips (v4 envelope)
        return real_mix(source, info, placements, line_paths, folder)

    monkeypatch.setattr(translate, 'identify', fake_identify)
    monkeypatch.setattr(translate, 'analyse', fake_analyse)
    monkeypatch.setattr(titles, 'sync', fake_sync)
    monkeypatch.setattr(translate, 'translate', fake_translate)
    def fake_overlay(lines, *, source, target, receipt, glossary=(), heard=None, session=None):
        calls.overlay.append((list(lines), source))
        return {k: (f'Daina {k}', calls.overlay_kinds.get(k, 'song')) for k in range(len(lines))}

    monkeypatch.setattr(translate, 'shorten', lambda lines, **kw: {})
    monkeypatch.setattr(translate, 'overlay', fake_overlay)
    monkeypatch.setattr(translate, 'turns', lambda units, **kw: calls.turns.append(units) or {})
    monkeypatch.setattr(tts, 'synthesize_all', fake_tts)
    monkeypatch.setattr(mix, 'mix', recording_mix)
    return calls


def job(clip, **overrides):
    base = {'job_id': '0b9e5d3c-1111-2222-3333-444455556666', 'source': {'kind': 'file', 'path': str(clip)},
            'source_language': 'auto', 'target_language': 'lt', 'voice': 'female',
            'products': {'subtitles': True, 'voiceover_audio': True, 'voiceover_video': True,
                         'burned_subtitles': True},
            'max_duration_seconds': 9.0, 'budget_usd': 0.5}
    return {**base, **overrides}


def run(job_request, tmp_path, transcribe=lambda path, lang, hints=(): {'language': 'en', 'words': WORDS}):
    sink = Sink()
    result = pipeline.process(job_request, tmp_path / 'work', store=LocalStore(tmp_path / 'out'), sink=sink,
                              transcribe=transcribe)
    return result, sink


def test_full_job_produces_every_contract_output(clip, tmp_path, fake_providers):
    result, sink = run(job(clip), tmp_path)
    assert result['event'] == 'completed', result
    terminal = [e for e in sink.events if e[0] != 'progress']
    assert len(terminal) == 1 and terminal[0][0] == 'completed'
    payload = terminal[0][1]
    assert set(payload) == {'duration_seconds', 'source_language', 'outputs', 'receipt'}
    assert payload['source_language'] == 'en'
    kinds = {o['kind']: o for o in payload['outputs']}
    assert set(kinds) == {'subtitles_srt', 'subtitles_vtt', 'subtitles_ass', 'transcript_srt', 'voiceover_audio',
                          'voiceover_video'}
    for output in payload['outputs']:
        assert output['key'].startswith('jobs/0b9e5d3c-1111-2222-3333-444455556666/')
        path = tmp_path / 'out' / output['key']
        assert path.is_file() and path.stat().st_size == output['bytes'] > 0
    assert kinds['transcript_srt']['language'] == 'en' and kinds['voiceover_video']['language'] == 'lt'
    # With a voice-over, the delivered subtitles show what the narrator says, not the per-cue translation.
    assert 'Eilutė 1' in (tmp_path / 'out' / kinds['subtitles_srt']['key']).read_text()
    streams = json.loads(subprocess.run(['ffprobe', '-v', 'error', '-show_streams', '-of', 'json',
                                         str(tmp_path / 'out' / kinds['voiceover_video']['key'])],
                                        capture_output=True, check=True).stdout)['streams']
    audio = [s for s in streams if s['codec_type'] == 'audio']
    assert [s['tags'].get('language') for s in audio] == ['lit', 'eng']
    assert audio[0]['disposition']['default'] == 1 and audio[1]['disposition']['default'] == 0
    # Subtitles burned in are not repeated as a track; the original-language transcript stays switchable.
    subs = [stream for stream in streams if stream['codec_type'] == 'subtitle']
    assert [(t['codec_name'], t['tags'].get('language')) for t in subs] == [('mov_text', 'eng')]
    stages = [e[1] for e in sink.events if e[0] == 'progress']
    assert stages[0] == 'downloading' and stages[-1] == 'uploading'
    progress = [e[2] for e in sink.events if e[0] == 'progress']
    assert progress == sorted(progress)
    assert payload['receipt']['total_usd'] > 0
    # Both 0.8 s lines fit their gap-aware budgets at natural speed: nothing to repair.
    timing = result['stats']['timing']
    assert timing['violations'] == {'late_start': 0, 'over_budget': 0, 'past_end': 0, 'overlap': 0}
    assert (timing['sped_up_lines'], timing['retakes'], timing['missing_speech']) == (0, 0, 0)


def test_translation_gets_each_lines_speaker_from_pitch(clip, tmp_path, fake_providers):
    run(job(clip), tmp_path)
    # The test clip's audio is a steady 220 Hz tone: a higher, female-range pitch for every line.
    assert fake_providers.hints == [{1: 'female', 2: 'female'}]


def test_song_lyrics_are_subtitled_not_narrated_or_ducked(clip, tmp_path, fake_providers):
    glossary = [{'source': 'Hello', 'target': 'Labas'}]
    fake_providers.analysis = {'glossary': glossary, 'new': glossary, 'sung': {2}}
    result, _ = run(job(clip), tmp_path)
    assert result['event'] == 'completed', result
    assert fake_providers.glossaries == [glossary]
    assert fake_providers.spoken == ['Eilutė 1']
    # The original dips only under the one narrated line (the sung unit, 3.0-4.5 s, has none).
    assert len(fake_providers.spans) == 1 and fake_providers.spans[0][1] < 3.0
    subtitles = (tmp_path / 'out' / next(o['key'] for o in result['outputs'] if o['kind'] == 'subtitles_srt'))
    text = subtitles.read_text()
    assert 'Eilutė 1' in text and 'Titras 2' in text and 'Eilutė 2' not in text
    assert result['stats']['sung_units'] == 1 and result['stats']['glossary_terms'] == 1


def test_sounds_without_words_are_neither_narrated_nor_subtitled(clip, tmp_path, fake_providers):
    # Unit 2 is a scream (nonverbal): the original soundtrack keeps it; nothing is said or shown over it.
    fake_providers.analysis = {'glossary': [], 'new': [], 'sung': set(), 'silent': {2},
                               'kinds': {2: 'nonverbal'}, 'forms': {}}
    fake_providers.quality = {'review_fixes': 2, 'names_repaired': 1}
    result, _ = run(job(clip), tmp_path)
    assert result['event'] == 'completed', result
    assert fake_providers.translated == [1]
    assert fake_providers.spoken == ['Eilutė 1']
    text = output_path(result, tmp_path, 'subtitles_srt').read_text()
    assert 'Eilutė 1' in text and 'Titras 2' not in text and 'Eilutė 2' not in text
    assert result['stats']['silent_units'] == 1 and result['stats']['sung_units'] == 0
    # Review and name-check counts from the translation reach the job's stats.
    assert result['stats']['review_fixes'] == 2 and result['stats']['names_repaired'] == 1


def test_silent_units_get_no_subtitles_in_a_subtitles_only_job(clip, tmp_path, fake_providers):
    fake_providers.analysis = {'glossary': [], 'new': [], 'sung': {1}, 'silent': {2}}
    result, _ = run(job(clip, products={'subtitles': True}), tmp_path)
    assert result['event'] == 'completed', result
    target = output_path(result, tmp_path, 'subtitles_srt').read_text()
    assert 'Titras 1' in target and 'Titras 2' not in target   # lyrics subtitled, the scream not
    # The transcript keeps every recognised line.
    assert 'See you soon.' in output_path(result, tmp_path, 'transcript_srt').read_text()


def test_glossary_forms_reach_translation_but_not_rails(clip, tmp_path, fake_providers):
    new = [{'source': 'Hello', 'target': 'Labas'}]
    forms = {'Hello': {'nominative': 'Labas', 'genitive': 'Labo'}}
    fake_providers.title = 'Some Show'
    fake_providers.analysis = {'glossary': new, 'new': new, 'sung': set(), 'silent': set(), 'forms': forms}
    fake_providers.stored = lambda terms: list(terms)
    result, _ = run(job(clip, callback_url='https://app.vocnu.com/api/v1/worker/events'), tmp_path)
    assert result['event'] == 'completed', result
    assert fake_providers.synced[-1] == ('Some Show', 'lt', new)   # base renderings only
    assert fake_providers.glossaries == [[{'source': 'Hello', 'target': 'Labas', 'forms': forms['Hello']}]]


def test_narration_subtitles_follow_the_takes_character_timestamps(clip, tmp_path, fake_providers):
    chars = list('Eilutė 1')
    starts = [0.3 + 0.05 * i for i in range(len(chars))]
    fake_providers.timestamps = {1: {'alignment': {'characters': chars, 'character_start_times_seconds': starts,
                                                   'character_end_times_seconds': [s + 0.05 for s in starts]},
                                     'normalized_alignment': None}}
    result, _ = run(job(clip, products={'voiceover_audio': True, 'subtitles': True}), tmp_path)
    assert result['event'] == 'completed', result
    first = output_path(result, tmp_path, 'subtitles_srt').read_text().split('\n')[1]
    # The first line plays just after its speech (0.5 s) and its first word is spoken 0.3 s into the take.
    assert first.startswith('00:00:00,9'), first


def test_subtitles_only_job_skips_speech(clip, tmp_path, fake_providers):
    request = job(clip, products={'subtitles': True})
    result, sink = run(request, tmp_path)
    assert result['event'] == 'completed'
    assert {o['kind'] for o in result['outputs']} == {'subtitles_srt', 'subtitles_vtt', 'subtitles_ass',
                                                      'transcript_srt'}
    assert 'synthesizing' not in {e[1] for e in sink.events if e[0] == 'progress'}


def test_too_long_media_fails_before_any_paid_step(clip, tmp_path, fake_providers):
    called = []
    result, sink = run(job(clip, max_duration_seconds=5.0), tmp_path,
                       transcribe=lambda p, l, hints=(): called.append(1) or {'language': 'en', 'words': WORDS})
    assert result['error']['code'] == 'duration_mismatch' and not called
    assert result['receipt']['total_usd'] == 0


def test_budget_guard_stops_before_overspending(clip, tmp_path, fake_providers, monkeypatch):
    monkeypatch.setattr(tts, 'estimate_usd', lambda texts: 10.0)
    result, _ = run(job(clip), tmp_path)
    assert result['error'] == {'code': 'budget_exceeded', 'message': result['error']['message'], 'retryable': False}


def test_no_speech_and_invalid_requests_fail_cleanly(clip, tmp_path, fake_providers):
    result, _ = run(job(clip), tmp_path, transcribe=lambda p, l, hints=(): {'language': 'en', 'words': []})
    assert result['error']['code'] == 'no_audio'
    result, sink = run(job(clip, products={}), tmp_path)
    assert result['error']['code'] == 'internal_error'
    assert [e[0] for e in sink.events] == ['failed']


def test_a_known_title_reuses_decided_terms_and_shares_the_new_ones(clip, tmp_path, fake_providers):
    decided = [{'source': 'Hello', 'target': 'Sveiki'}]
    new = [{'source': 'soon', 'target': 'greitai'}]
    concurrent = [{'source': 'there', 'target': 'ten'}]
    fake_providers.title = 'Some Show'
    fake_providers.analysis = {'glossary': decided + new, 'new': new, 'sung': set()}
    fake_providers.stored = lambda terms: decided + list(terms) + (concurrent if terms else [])
    result, _ = run(job(clip, callback_url='https://app.vocnu.com/api/v1/worker/events'), tmp_path)
    assert result['event'] == 'completed', result
    assert fake_providers.filename == 'clip.mp4'
    assert fake_providers.synced == [('Some Show', 'lt', []), ('Some Show', 'lt', []), ('Some Show', 'lt', new)]
    assert fake_providers.decided == [decided]
    # Translation uses what Rails holds after storing: earlier decisions, this job's and a concurrent job's.
    assert fake_providers.glossaries == [decided + new + concurrent]
    assert result['stats']['title'] == 'Some Show' and result['stats']['decided_terms'] == 1


def test_unknown_titles_and_an_unreachable_glossary_keep_the_jobs_own_terms(clip, tmp_path, fake_providers):
    new = [{'source': 'soon', 'target': 'greitai'}]
    fake_providers.analysis = {'glossary': new, 'new': new, 'sung': set()}
    run(job(clip), tmp_path)
    assert fake_providers.synced == [] and fake_providers.glossaries == [new]

    fake_providers.title = 'Some Show'
    run(job(clip), tmp_path / 'again')
    assert [terms for _, _, terms in fake_providers.synced] == [[], [], new]
    assert fake_providers.glossaries[-1] == new


def test_terms_decided_by_a_concurrent_job_are_followed_before_storing(clip, tmp_path, fake_providers):
    other_job = [{'source': 'Hello', 'target': 'Sveiki'}]
    lookups = iter([[], other_job])
    fake_providers.title = 'Some Show'
    fake_providers.analysis = {'glossary': [{'source': 'soon', 'target': 'greitai'}],
                               'new': [{'source': 'soon', 'target': 'greitai'}], 'sung': set()}
    fake_providers.stored = lambda terms: next(lookups) if not terms else other_job + list(terms)
    run(job(clip), tmp_path)
    # The second lookup found the other job's decision, so the analysis ran again with it.
    assert fake_providers.decided == [[], other_job]
    assert fake_providers.glossaries == [other_job + [{'source': 'soon', 'target': 'greitai'}]]


def probe_streams(path):
    return json.loads(subprocess.run(['ffprobe', '-v', 'error', '-show_streams', '-of', 'json', str(path)],
                                     capture_output=True, check=True).stdout)['streams']


def ai_disclosed(path):
    """Whether the file says in its metadata that its voice is AI-generated (format or stream tags, any case)."""
    data = json.loads(subprocess.run(['ffprobe', '-v', 'error', '-show_format', '-show_streams', '-of', 'json',
                                      str(path)], capture_output=True, check=True).stdout)
    tags = [data.get('format', {}).get('tags', {})] + [s.get('tags', {}) for s in data.get('streams', [])]
    return any(key.lower() == 'comment' and 'AI-generated voice-over' in value
               for group in tags for key, value in group.items())


def output_path(result, tmp_path, kind):
    return tmp_path / 'out' / next(o['key'] for o in result['outputs'] if o['kind'] == kind)


@pytest.mark.parametrize('video_format, video_codec, audio_codecs, subtitle_codec', [
    ('mp4', 'h264', ['aac', 'aac'], 'mov_text'),
    ('mov', 'h264', ['aac', 'aac'], 'mov_text'),
    ('mkv', 'h264', ['aac', 'aac'], 'ass'),
    ('webm', 'vp9', ['opus', 'opus'], 'webvtt'),
    ('avi', 'h264', ['mp3', 'mp3'], None),
])
def test_each_video_format_carries_switchable_subtitles_where_the_container_can(
        clip, tmp_path, fake_providers, video_format, video_codec, audio_codecs, subtitle_codec):
    products = {'voiceover_video': True, 'voiceover_audio': True, 'subtitles': True}
    result, _ = run(job(clip, products=products, video_format=video_format), tmp_path)
    assert result['event'] == 'completed', result
    video = output_path(result, tmp_path, 'voiceover_video')
    assert video.suffix == f'.{video_format}'
    streams = probe_streams(video)
    assert [s['codec_name'] for s in streams if s['codec_type'] == 'video'] == [video_codec]
    assert [s['codec_name'] for s in streams if s['codec_type'] == 'audio'] == audio_codecs
    # The synthetic voice is disclosed: a metadata comment, and the track's name in the player's audio menu.
    assert ai_disclosed(video)
    voice_track = next(s for s in streams if s['codec_type'] == 'audio')
    names = {key.lower(): value for key, value in voice_track['tags'].items()}
    if subtitle_codec == 'mov_text':   # MP4/MOV: the handler name, which every ffmpeg version keeps
        assert names.get('handler_name') == 'AI voice-over (Lietuvių)'
    elif video_format != 'avi':
        assert names.get('title') == 'AI voice-over (Lietuvių)'
    subs = [s for s in streams if s['codec_type'] == 'subtitle']
    if subtitle_codec is None:   # AVI cannot hold text subtitles: they are the separate files
        assert subs == []
    else:
        # MP4/MOV always mark their first subtitle track enabled (ffmpeg's mov muxer); MKV and WebM leave both off.
        first = 1 if subtitle_codec == 'mov_text' else 0
        assert [(s['codec_name'], s['tags'].get('language'), s['disposition']['default']) for s in subs] == \
            [(subtitle_codec, 'lit', first), (subtitle_codec, 'eng', 0)]
    if subtitle_codec == 'ass':   # MKV keeps the styles: song and sign lines can sit at the top
        styled = subprocess.run(['ffmpeg', '-v', 'error', '-i', str(video), '-map', '0:s:0', '-f', 'ass', '-'],
                                capture_output=True, text=True, check=True).stdout
        assert 'Style: Top,' in styled


@pytest.mark.parametrize('audio_format, codec', [('m4a', 'aac'), ('mp3', 'mp3'), ('wav', 'pcm_s16le'),
                                                 ('flac', 'flac'), ('ogg', 'opus')])
def test_voiceover_audio_comes_in_the_chosen_format(clip, tmp_path, fake_providers, audio_format, codec):
    # Rails always adds subtitles to a voice-over order.
    result, _ = run(job(clip, products={'voiceover_audio': True, 'subtitles': True}, audio_format=audio_format),
                    tmp_path)
    assert result['event'] == 'completed', result
    audio = output_path(result, tmp_path, 'voiceover_audio')
    assert audio.suffix == f'.{audio_format}'
    assert [s['codec_name'] for s in probe_streams(audio)] == [codec]
    assert ai_disclosed(audio)
    assert output_path(result, tmp_path, 'subtitles_ass').read_text().startswith('[Script Info]')


def test_unknown_formats_are_rejected(clip, tmp_path, fake_providers):
    for bad in ({'video_format': 'flv'}, {'audio_format': 'aiff'}):
        result, _ = run(job(clip, **bad), tmp_path)
        assert result['error']['code'] == 'internal_error'


SIGNS_ASS = """[Script Info]
ScriptType: v4.00+

[V4+ Styles]
Format: Name, Fontname, Fontsize
Style: Song,Arial,20

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
Dialogue: 0,0:00:03.00,0:00:04.50,Song,,0,0,0,,Shooting stars in the night sky
"""


@pytest.fixture(scope='module')
def rich_clip(tmp_path_factory):
    """8 s like a Blu-ray anime episode: original English speech, a Lithuanian dub (the default track), English
    dialogue subtitles and an English signs & songs track."""
    folder = tmp_path_factory.mktemp('rich')
    (folder / 'dialogue.srt').write_text('1\n00:00:00,500 --> 00:00:01,500\nHello there.\n\n'
                                         '2\n00:00:03,000 --> 00:00:04,500\nSee you soon.\n\n'
                                         '3\n00:00:05,500 --> 00:00:06,800\nWait for me!\n')
    (folder / 'signs.ass').write_text(SIGNS_ASS)
    path = folder / 'rich.mkv'
    subprocess.run(['ffmpeg', '-v', 'error', '-f', 'lavfi', '-i', 'testsrc=size=320x240:rate=25:duration=8',
                    '-f', 'lavfi', '-i', 'sine=frequency=220:sample_rate=48000:duration=8',
                    '-f', 'lavfi', '-i', 'sine=frequency=440:sample_rate=48000:duration=8',
                    '-i', str(folder / 'dialogue.srt'), '-i', str(folder / 'signs.ass'),
                    '-map', '0:v', '-map', '1:a', '-map', '2:a', '-map', '3:s', '-map', '4:s',
                    '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-c:a', 'aac', '-c:s', 'copy',
                    '-metadata:s:a:0', 'language=eng', '-metadata:s:a:1', 'language=lit',
                    '-metadata:s:a:1', 'title=Lietuvių (įgarsinimas)', '-disposition:a:1', 'default',
                    '-disposition:a:0', '0', '-metadata:s:s:0', 'language=eng', '-metadata:s:s:1', 'language=eng',
                    '-metadata:s:s:1', 'title=Signs & Songs', str(path)], check=True)
    return path


def test_every_source_of_truth_in_the_upload_is_used(rich_clip, tmp_path, fake_providers):
    calls = []

    def transcribe(path, language, hints=()):
        calls.append((Path(path).name, language, [tuple(h) for h in hints]))
        if Path(path).name.startswith('dub-'):
            return {'language': 'lt', 'seconds': 2.0,
                    'words': [{'start': 0.6, 'end': 1.2, 'word': ' Labas', 'probability': 0.9}]}
        return {'language': 'en', 'words': WORDS}

    # The analysis hears unit 2 (3.0-4.5 s) as singing: the signs & songs track has its lyric.
    fake_providers.analysis = {'glossary': [], 'new': [], 'sung': {2}}
    result, _ = run(job(rich_clip, max_duration_seconds=9.0), tmp_path, transcribe=transcribe)
    assert result['event'] == 'completed', result
    # The original English track is the source (not the Lithuanian default); the dialogue subtitles are hints.
    assert calls[0][:2] == ('asr.opus', 'auto')
    assert [(round(a, 1), round(b, 1)) for a, b in calls[0][2]] == [(0.5, 1.5), (3.0, 4.5), (5.5, 6.8)]
    assert sorted(call[:2] for call in calls[1:]) == [('dub-2.opus', 'lt')]
    # Translation gets the subtitles and the dub at each unit; the analysis also the song line shown then. The line
    # at 5.5 s that recognition never heard is unit 3, taken from the subtitles.
    assert fake_providers.evidence['translation'] == {1: {'eng': 'Hello there.', 'lit dub': 'Labas'},
                                                      2: {'eng': 'See you soon.'}, 3: {'eng': 'Wait for me!'}}
    assert fake_providers.evidence['analysis'][2] == {'eng': 'See you soon.',
                                                      'eng songs/signs': 'Shooting stars in the night sky'}
    # The song line from the file replaces the lyric recognition caught, at the top of the picture. (Muxing may
    # shift subtitle times by a frame depending on the ffmpeg build: CI's moves them 0.02 s.)
    [(lines, language)] = fake_providers.overlay
    assert language == 'eng' and [(round(a, 1), round(b, 1), text) for a, b, text in lines] == [
        (3.0, 4.5, 'Shooting stars in the night sky')]
    assert fake_providers.spoken == ['Eilutė 1', 'Eilutė 3']
    vtt = output_path(result, tmp_path, 'subtitles_vtt').read_text()
    assert ' line:0\nDaina 0' in vtt and 'Titras 2' not in vtt
    assert 'Eilutė 1' in vtt
    assert ',Top,,0,0,0,,Daina 0' in output_path(result, tmp_path, 'subtitles_ass').read_text()
    stats = result['stats']
    assert stats['dub_tracks'] == ['lit'] and stats['overlay_lines'] == 1 and stats['subtitle_lines'] == 1
    assert [(t['language'], t['kind'], t['lines']) for t in stats['file_subtitles']] == [
        ('eng', 'dialogue', 3), ('eng', 'overlay', 1)]
    assert stats['misaligned_subtitles'] == []
    # The transcript keeps only what was recognised in the original.
    transcript = output_path(result, tmp_path, 'transcript_srt').read_text()
    assert 'Hello there.' in transcript and 'Wait for me!' not in transcript


def test_subtitles_of_another_edition_are_not_trusted_and_heard_lines_are_not_repeated():
    words = [{'start': t, 'end': t + 0.4, 'word': ' a'} for t in (1.0, 5.0, 9.0, 13.0)]
    good = {'language': 'eng', 'cues': [(0.9, 1.6, 'One.'), (5.1, 5.6, 'Two.'), (9.0, 9.5, 'Three.'),
                                        (20.0, 21.0, 'Wait!'), (24.0, 25.0, '[door slams]')]}
    shifted = {'language': 'ger', 'cues': [(31.0, 31.6, 'Eins.'), (35.1, 35.6, 'Zwei.'), (39.0, 39.5, 'Drei.')]}
    kept, dropped = pipeline._aligned([good, shifted], words)
    assert kept == [good] and dropped == ['ger']
    assert pipeline._unheard_lines(good, words) == [(20.0, 21.0, 'Wait!')]


def test_stats_say_how_long_each_stage_took(clip, tmp_path, fake_providers):
    result, _ = run(job(clip), tmp_path)
    assert result['event'] == 'completed', result
    took = result['stats']['stage_seconds']
    assert list(took) == ['downloading', 'transcribing', 'translating', 'synthesizing', 'mixing', 'rendering',
                          'uploading']
    assert all(seconds >= 0 for seconds in took.values())


def test_stage_seconds_run_from_one_stage_to_the_next():
    ticks = iter([0.0, 2.0, 2.5, 7.0, 9.0])
    progress = pipeline._Progress(Sink(), ['downloading', 'transcribing', 'uploading'], clock=lambda: next(ticks))
    progress('downloading')        # 0.0
    progress('transcribing')       # 2.0
    progress('transcribing', 0.5)  # 2.5: the same stage again
    progress('uploading')          # 7.0
    assert progress.seconds() == {'downloading': 2.0, 'transcribing': 5.0, 'uploading': 2.0}   # now: 9.0


def test_each_stage_gets_the_share_of_the_bar_its_measured_time_takes():
    rates = pipeline.STAGE_SECONDS_PER_MINUTE
    video = pipeline._Progress(Sink(), list(rates))
    total = sum(rates.values())
    assert video.start['synthesizing'] == pytest.approx((0.2 + 2.4 + 3.7) / total)
    assert video.end['uploading'] == pytest.approx(1.0)
    # Subtitles only: no speech, mixing or rendering, so translation takes most of the bar.
    subtitles = pipeline._Progress(Sink(), ['downloading', 'transcribing', 'translating', 'uploading'])
    assert subtitles.end['translating'] - subtitles.start['translating'] == pytest.approx(3.7 / 6.6)


def test_a_stage_without_progress_of_its_own_moves_on_with_time_but_never_back():
    now = [0.0]
    sink = Sink()
    progress = pipeline._Progress(sink, ['downloading', 'translating', 'uploading'], clock=lambda: now[0])
    progress.media_minutes = 10.0               # translation is expected to take 37 s
    progress('translating')
    now[0] = 18.5
    progress.tick()                             # half of the expected time: half of the stage
    now[0] = 1000.0
    progress.tick()                             # far over: held at TICK_CAP until the stage really ends
    progress('translating', 0.3)                # real progress behind the estimate is not sent
    progress('translating', 1.0)
    start, end = progress.start['translating'], progress.end['translating']
    sent = [p for kind, stage, p in sink.events if kind == 'progress' and stage == 'translating']
    assert sent == pytest.approx([start, start + (end - start) * 0.5, start + (end - start) * pipeline.TICK_CAP, end])
    progress.current = None
    progress.tick()                             # nothing running: nothing sent
    assert len(sink.events) == 4


def test_what_recognition_hears_inside_a_known_song_is_neither_translated_nor_narrated(rich_clip, tmp_path,
                                                                                         fake_providers):
    # The file's lyrics run 0.2-6.0 s (two lines with an instrumental gap); recognition heard a word at 2.15-2.35 s
    # that no dialogue subtitle shows, even with half a second of slack and a muxer shifting times by a frame
    # (Bleach S01E01: "その" -> "Tas..." in the opening). "Hello there." and "Wait for me!" have dialogue subtitles
    # (0.5-1.5 s, 5.5-6.8 s): someone speaks over the song.
    words = [{'start': 0.5, 'end': 0.9, 'word': ' Hello', 'probability': 0.9},
             {'start': 0.95, 'end': 1.5, 'word': ' there.', 'probability': 0.9},
             {'start': 2.15, 'end': 2.35, 'word': ' That...', 'probability': 0.9, 'segment_start': True},
             {'start': 5.6, 'end': 6.4, 'word': ' Wait!', 'probability': 0.9, 'segment_start': True}]
    lines = [(0.2, 1.0, 'Raise your eyes'), (2.0, 6.0, 'Wishes speak from ages old')]
    original = pipeline.file_subtitles.layers
    pipeline_layers = lambda tracks: (original(tracks)[0], ('eng', lines))   # noqa: E731
    import vocnu_engine.file_subtitles as fs
    fs_layers, fs.layers = fs.layers, pipeline_layers
    try:
        result, _ = run(job(rich_clip, max_duration_seconds=9.0), tmp_path,
                        transcribe=lambda p, l, hints=(): {'language': 'en', 'words': [] if 'dub' in str(p) else words})
    finally:
        fs.layers = fs_layers
    assert result['event'] == 'completed', result
    assert result['stats']['songs'] == [[0.2, 6.0]] and result['stats']['song_units'] == 1
    assert fake_providers.translated == [1, 3]
    assert fake_providers.spoken == ['Eilutė 1', 'Eilutė 3']
    vtt = output_path(result, tmp_path, 'subtitles_vtt').read_text()
    assert 'Titras 2' not in vtt and 'Eilutė 2' not in vtt and ' line:0\nDaina 1' in vtt



def test_units_whose_subtitles_show_two_lines_are_offered_for_speaker_turns():
    from vocnu_engine.segments import Unit
    words = [{'start': 231.7, 'end': 232.4, 'word': '持ってきて'}, {'start': 232.4, 'end': 233.0, 'word': 'やるよ'},
             {'start': 233.9, 'end': 234.6, 'word': 'ありがとう'}]
    one_line = Unit(1, 225.0, 228.0, [1], 'こんだけ脅かしときゃ', [{'start': 225.0, 'end': 228.0, 'word': 'x'},
                                                              {'start': 226.0, 'end': 228.0, 'word': 'y'}])
    two_lines = Unit(2, 231.7, 234.6, [2], '持ってきてやるよありがとう', words)
    tracks = [{'language': 'eng', 'cues': [(225.0, 228.3, "They probably won't be back."),
                                           (230.3, 233.3, "I'll bring you some fresh flowers tomorrow."),
                                           (234.1, 236.1, 'Thank you for getting rid of them.')]}]
    [(item, own)] = pipeline._turn_candidates([one_line, two_lines], tracks, [])
    assert item == {'id': 2, 'words': ['持ってきて', 'やるよ', 'ありがとう'],
                    'subtitles': ["I'll bring you some fresh flowers tomorrow.", 'Thank you for getting rid of them.']}
    assert own is words


def test_a_turn_cut_leaving_a_syllable_joins_it_to_the_next_line():
    from vocnu_engine.segments import turn_key
    words = [{'start': 0.0, 'end': 0.6, 'word': '大正解'}, {'start': 0.6, 'end': 0.7, 'word': 'め'},
             {'start': 0.7, 'end': 1.2, 'word': 'っちゃん'}]
    starts, joins = pipeline._turn_cuts(words[:2], [1], words)
    assert starts == {turn_key(words[1])} and joins == {turn_key(words[2])}
    # A whole reply that a pause separates from the next line stays a line of its own.
    apart = [dict(w) for w in words]
    apart[2]['start'] = 1.5
    assert pipeline._turn_cuts(apart[:2], [1], apart) == ({turn_key(apart[1])}, set())


def test_translation_budgets_follow_the_narrators_measured_rate():
    from vocnu_engine.languages import get
    # Lithuanian 13 characters a second with 5 % to spare; never under a short word.
    assert (get('lt').chars_for(1.0), get('lt').chars_for(4.0), get('lt').chars_for(0.3)) == (12, 49, 6)
    assert get('ja').chars_for(4.0) == 22


def test_a_community_voice_reads_the_job_and_is_credited_in_the_outputs(clip, tmp_path, fake_providers, monkeypatch):
    voices_used = []
    real_tts = pipeline.tts.synthesize_all

    def recording_tts(lines, language, voice, cache_dir, receipt, **kwargs):
        voices_used.append(voice)
        return real_tts(lines, language, voice, cache_dir, receipt, **kwargs)

    monkeypatch.setattr(pipeline.tts, 'synthesize_all', recording_tts)
    override = {'id': 'abcDEF123456', 'name': 'Jonas', 'credit': 'Jonas (Vocnu community)'}
    result, _ = run(job(clip, voice='male', voice_override=override,
                        products={'voiceover_video': True, 'subtitles': True}), tmp_path)
    assert result['event'] == 'completed', result
    assert voices_used and voices_used[0] == {'id': 'abcDEF123456', 'name': 'Jonas', 'gender': 'male'}
    video = output_path(result, tmp_path, 'voiceover_video')
    tags = {key.lower(): value for key, value in probe_streams(video)[1]['tags'].items()}
    assert tags['handler_name'] == 'AI voice-over (Lietuvių): Jonas (Vocnu community)'


def test_a_malformed_community_voice_is_refused():
    with pytest.raises(ValueError):
        pipeline.validate({**job(Path('clip.mp4')), 'voice_override': {'id': '../../x', 'name': 'Jonas'}})
