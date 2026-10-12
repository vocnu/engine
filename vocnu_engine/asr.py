"""Speech recognition with faster-whisper large-v3-turbo: CUDA float16 when a GPU is present, else CPU int8."""
import ctypes
import glob
import math
import os
import re
import site
import time
import zlib

from . import media
from .languages import from_provider, to_provider

MODEL = 'large-v3-turbo'
_loaded = {}


def _preload_cuda_libraries():
    """CTranslate2 dlopens libcublas.so.12 and cuDNN 9; load them from the nvidia-* wheels when present."""
    roots = list(site.getsitepackages())
    if site.ENABLE_USER_SITE:
        roots.append(site.getusersitepackages())
    for root in roots:
        for pattern in ('nvidia/cublas/lib/libcublasLt.so.12', 'nvidia/cublas/lib/libcublas.so.12',
                        'nvidia/cudnn/lib/libcudnn*.so.9'):
            for path in sorted(glob.glob(os.path.join(root, pattern))):
                try:
                    ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
                except OSError:
                    pass


def device():
    _preload_cuda_libraries()
    import ctranslate2
    try:
        return 'cuda' if ctranslate2.get_cuda_device_count() > 0 else 'cpu'
    except Exception:  # noqa: BLE001 - a broken CUDA install must fall back to CPU
        return 'cpu'


def load_model(download_root=None):
    """Load once per process; returns (model, device).

    The same sequential WhisperModel on both devices, float16 on a GPU and int8 on CPU: on Bleach S01E01 faster-whisper's
    batched pipeline (the GPU default before) left 11 of the 325 English subtitle lines with no Japanese word near them,
    the sequential model 7, so the GPU only makes recognition faster (the whole episode in about 32 s on a desktop GPU).
    """
    key = download_root or ''
    if key not in _loaded:
        from faster_whisper import WhisperModel
        dev = device()
        model = WhisperModel(MODEL, device=dev, compute_type='float16' if dev == 'cuda' else 'int8',
                             download_root=download_root, cpu_threads=min(8, os.cpu_count() or 4))
        _loaded[key] = (model, dev)
    return _loaded[key]


# A whole segment that is only a sound label ("[Music]", "(laughs)", "*sighs*", "（拍手）") is not dialogue: the sound
# stays in the original audio and nothing is translated or narrated (ai-voice-over v4 nonlexical_marker).
LABEL = re.compile(r'^[\s♪♫♬🎵🎶]*([\[(（【*][^\])）】*]{1,40}[\])）】*][\s.!?。！？♪♫♬🎵🎶]*)+$')


def _is_noise(text):
    stripped = text.strip().strip('♪♫♬🎵🎶*.…-–— ')
    return not stripped or bool(LABEL.match(text.strip()))


def _normal(text):
    return re.sub(r'[\W_]+', '', text.casefold())


# Text Whisper invents over music, applause or silence, learned from subtitle credits and video sign-offs in its
# training data. Credits are never spoken, so a segment that is only a credit is always dropped; a sign-off can be
# real speech, so it is dropped only when it does not sound like speech (see _hallucinated).
CREDITS = frozenset(_normal(p) for p in (
    'Subtitles by the Amara.org community', '字幕由Amara.org社区提供', '小編字幕由Amara.org社區提供',
    'Subtítulos realizados por la comunidad de Amara.org', "Sous-titres réalisés par la communauté d'Amara.org",
    "Sous-titres réalisés para la communauté d'Amara.org", 'Sottotitoli creati dalla comunità Amara.org',
    'Legendas pela comunidade Amara.org', 'Napisy stworzone przez społeczność Amara.org',
    'Ondertiteld door de Amara.org gemeenschap', 'Untertitel im Auftrag des ZDF für funk, 2017',
    'Untertitel im Auftrag des ZDF, 2017', 'Untertitelung des ZDF, 2020', 'Untertitel von Stephanie Geiges',
    'Субтитры сделал DimaTorzok', 'Субтитры создавал DimaTorzok',
    'Редактор субтитров А.Синецкая Корректор А.Егорова', 'Продолжение следует...',
    'Sous-titrage Société Radio-Canada', 'ترجمة نانسي قنقر', 'MBC 뉴스 이덕영입니다',
    '请不吝点赞 订阅 转发 打赏支持明镜与点点栏目'))
SIGN_OFFS = frozenset(_normal(p) for p in (
    'ご視聴ありがとうございました', 'ご視聴ありがとうございます', 'ご覧いただきありがとうございました',
    'チャンネル登録よろしくお願いします', 'チャンネル登録お願いします',
    'Thanks for watching!', 'Thank you for watching!', 'Thank you very much for watching!',
    'Thank you so much for watching!', 'Please subscribe to my channel.', "Don't forget to like and subscribe!",
    '谢谢观看', '感谢观看', '谢谢大家观看', '시청해 주셔서 감사합니다', '구독과 좋아요 부탁드립니다',
    '¡Gracias por ver!', 'Gracias por ver el video.', "Merci d'avoir regardé cette vidéo !", 'Merci de votre attention.',
    'Vielen Dank fürs Zuschauen!', 'Danke fürs Zuschauen!', 'Grazie per la visione!', 'Obrigado por assistir!',
    'Inscreva-se no canal!', 'Спасибо за просмотр!', 'Дякую за перегляд!', 'Dziękuję za uwagę.',
    'Dziękuję za obejrzenie!', 'Bedankt voor het kijken!', 'İzlediğiniz için teşekkür ederim.',
    'Terima kasih telah menonton!', 'شكرا للمشاهدة', 'देखने के लिए धन्यवाद'))
# A real sign-off is said at a normal pace; Whisper's inventions are often stretched over many seconds of music
# (Bleach S01E01: the 14 characters of ご視聴ありがとうございました spread over 16 s of the ending song).
MIN_SIGN_OFF_CHARS_PER_SECOND = 3.0


def _hallucinated(segment):
    """True for a segment that is nothing but a known Whisper invention (see CREDITS and SIGN_OFFS)."""
    text = _normal(segment.text)
    if text in CREDITS:
        return True
    if text not in SIGN_OFFS:
        return False
    words = [w for w in segment.words or [] if w.word.strip()]
    start, end = (words[0].start, words[-1].end) if words else (segment.start, segment.end)
    slow = len(text) / max(end - start, 0.01) < MIN_SIGN_OFF_CHARS_PER_SECOND
    return slow or segment.no_speech_prob > 0.5 or segment.avg_logprob < -0.8


# Silero VAD at 0.25 instead of the default 0.5: on Bleach S01E01 the default dropped a whole shouted scene under
# music and 26 short replies, and stretched other segments over 20 s; 0.25 recovered them on Modal's CPU too.
VAD = {'threshold': 0.25, 'min_silence_duration_ms': 500, 'speech_pad_ms': 300}
# Singing under music and quiet lines need a more sensitive VAD: at 0.25 the opening song of Bleach S01E01 (0:13-0:39)
# was not speech at all, at 0.1 it was. Only stretches recognised again (no words heard where speech is) use it, so a
# whole film is not opened up to Whisper inventing words in its music.
SENSITIVE_VAD = {**VAD, 'threshold': 0.1}
# Repair passes ported from ai-voice-over v4 (source_quality.py, content_quality.py, AUDIO-TIMING.md): a stretch
# whose transcript looks broken is recognised again on its own, never deleted on a heuristic.
LOOP_ROWS, LOOP_SECONDS = 5, 12.0        # identical adjacent rows covering this long = a repetition loop
LOOP_GAP = 3.0                           # a pause longer than this ends a run
SPARSE_SECONDS, SPARSE_LETTERS = 20.0, 1.0   # a segment this long with fewer letters per second
UNHEARD_SECONDS = 0.8                    # VAD speech this long with no recognised word ("empty ASR is not silence")
HINT_SLACK = 0.3                         # a subtitle's speech counts as heard when a word starts this close to it
UNCERTAIN_PROBABILITY = 0.35             # a segment whose words average below this is read again (v4 audio_timing)
REPAIR_WINDOW = 30.0
REPAIR_BUDGET = 0.25                     # at most this share of the media is recognised a second time


def _letters(text):
    return sum(ch.isalnum() for ch in text)


def _rows(segments, offset=0.0):
    """Segments as plain dicts (times shifted by `offset`), with Whisper's scores and words."""
    rows = []
    for segment in segments:
        rows.append({'start': segment.start + offset, 'end': segment.end + offset, 'text': segment.text,
                     'no_speech_prob': segment.no_speech_prob, 'avg_logprob': segment.avg_logprob, 'segment': segment,
                     'words': [{'start': w.start + offset, 'end': w.end + offset, 'word': w.word,
                                'probability': w.probability} for w in segment.words or []]})
    return rows


def suspect_spans(rows, speech, duration, hints=()):
    """[(start, end, reason)] stretches to recognise again: repetition loops, implausibly sparse long segments, VAD
    speech that produced no words, and `hints` (where the upload's own subtitles show a line) with no word heard."""
    spans = []
    run = []
    for row in rows + [None]:
        same = (row is not None and run and _normal(row['text']) == _normal(run[-1]['text'])
                and row['start'] - run[-1]['end'] <= LOOP_GAP)
        if same:
            run.append(row)
            continue
        if len(run) >= LOOP_ROWS and run[-1]['end'] - run[0]['start'] >= LOOP_SECONDS:
            spans.append((run[0]['start'], run[-1]['end'], 'repetition'))
        run = [row] if row is not None else []
    for row in rows:
        length = row['end'] - row['start']
        if length >= SPARSE_SECONDS and _letters(row['text']) / length < SPARSE_LETTERS:
            spans.append((row['start'], row['end'], 'sparse'))
    for row in rows:
        if row['end'] - row['start'] >= 1.0 and row['words'] and _mean_probability([row]) < UNCERTAIN_PROBABILITY:
            spans.append((row['start'], row['end'], 'uncertain'))
    heard = sorted(w['start'] for row in rows for w in row['words'])
    for start, end in speech:
        if end - start < UNHEARD_SECONDS:
            continue
        if not any(start - 0.2 <= t <= end + 0.2 for t in heard):
            spans.append((max(0.0, start - 0.3), min(duration, end + 0.3), 'unheard'))
    for start, end in hints:
        if end > start and not any(start - HINT_SLACK <= t <= end + HINT_SLACK for t in heard):
            spans.append((max(0.0, start - HINT_SLACK), min(duration, end + HINT_SLACK), 'hinted'))
    return sorted(spans)


def _windows(start, end):
    """Cut a stretch into windows of at most REPAIR_WINDOW seconds."""
    pieces = max(1, math.ceil((end - start) / REPAIR_WINDOW))
    step = (end - start) / pieces
    return [(start + i * step, start + (i + 1) * step) for i in range(pieces)]


MIN_SEGMENT_SECONDS = 0.15                # shorter "speech" is a decoding artefact (seen as repeated zero-length "ん?")


LOOPED_RATIO, LOOPED_MIN_CHARS = 2.4, 24   # Whisper's own compression-ratio limit for a repeating decode


def _looped(text):
    """True for a decode stuck in a loop ("ファーストファースト…", "ハァァァァ…": ratio about 20 on Bleach S01E01, where
    a re-reading on the GPU produced them; sentences and short chants stay under 2)."""
    data = text.strip().encode('utf-8')
    return len(text.strip()) >= LOOPED_MIN_CHARS and len(data) / len(zlib.compress(data)) > LOOPED_RATIO


def _keep(row):
    return not (_is_noise(row['text']) or (row['no_speech_prob'] > 0.8 and row['avg_logprob'] < -1.0)
                or row['end'] - row['start'] < MIN_SEGMENT_SECONDS or _hallucinated(row['segment'])
                or _looped(row['text']))


def _speech(audio, vad=VAD):
    """[(start, end)] seconds of speech by Silero VAD (by default with the same settings as recognition)."""
    from faster_whisper.vad import VadOptions, get_speech_timestamps
    return [(c['start'] / 16000, c['end'] / 16000) for c in get_speech_timestamps(audio, VadOptions(**vad))]


def _mean_probability(rows):
    probabilities = [w.get('probability', 1.0) for row in rows for w in row['words'] if w['word'].strip()]
    return sum(probabilities) / len(probabilities) if probabilities else 0.0


def _merge(spans, gap=10.0):
    """Join suspect stretches that are close together into windows of at most REPAIR_WINDOW seconds, so nearby
    short stretches cost one recognition call instead of one each (Whisper always encodes 30 s)."""
    groups = []
    for start, end, reason in sorted(spans):
        if groups and start - groups[-1][1] <= gap and max(end, groups[-1][1]) - groups[-1][0] <= REPAIR_WINDOW:
            groups[-1][1] = max(groups[-1][1], end)
            groups[-1][2].add(reason)
        else:
            groups.append([start, end, {reason}])
    return [(start, end, reasons) for start, end, reasons in groups]


def _unheard_part(row, rows):
    """What a re-read row adds: the row itself, or only its words that no word already heard overlaps (a long
    recognition segment can span a missed shout, so segment spans are not proof it was heard); None if nothing."""
    heard = [(w['start'], w['end']) for other in rows for w in other['words'] if w['word'].strip()]
    new = [w for w in row['words'] if w['word'].strip()
           and not any(a < w['end'] - 0.05 and b > w['start'] + 0.05 for a, b in heard)]
    if not new:
        return None
    if len(new) == len([w for w in row['words'] if w['word'].strip()]):
        return row
    return {**row, 'start': new[0]['start'], 'end': new[-1]['end'], 'text': ''.join(w['word'] for w in new),
            'words': new}


def _repair(model, audio, rows, options, duration, hints=()):
    """Recognise suspect stretches again on their own; returns (rows, repairs).

    A loop or a sparse segment is replaced by the new reading when it has text (otherwise the original stays: never
    delete on a heuristic). Speech that had no words (found with the sensitive VAD) and subtitle `hints` with no word
    heard are read with the sensitive VAD and only gain the new lines that do not repeat an existing one.
    """
    speech = _speech(audio, SENSITIVE_VAD)
    repairs, budget = [], REPAIR_BUDGET * duration
    single = dict(options)
    sensitive = {**single, 'vad_parameters': SENSITIVE_VAD}
    # A line the upload's subtitles show comes first, then broken readings; VAD speech without words (often music
    # at the sensitive threshold) uses what budget is left.
    order = {'hinted': 0, 'repetition': 1, 'sparse': 1, 'uncertain': 2, 'unheard': 3}
    windows = sorted(_merge(suspect_spans(rows, speech, duration, hints)),
                     key=lambda w: (min(order[r] for r in w[2]), w[0]))
    for start, end, reasons in windows:
        if end - start > budget:
            continue
        budget -= end - start
        fresh = []
        settings = sensitive if reasons <= {'unheard', 'hinted'} else single
        for a, b in _windows(start, end):
            segments, _ = model.transcribe(audio[int(a * 16000):int(b * 16000)], **settings)
            # A sign-off that only the sensitive reading finds is Whisper inventing words in music or noise (seen at
            # 0:27 and 14:05 of Bleach S01E01).
            fresh += [row for row in _rows(segments, a) if _keep(row)
                      and not (settings is sensitive and _normal(row['text']) in SIGN_OFFS)]
        if reasons & {'repetition', 'sparse', 'uncertain'}:
            old = [row for row in rows if row['start'] < end and row['end'] > start]
            if _letters(' '.join(r['text'] for r in fresh)) == 0 and old:
                continue
            # A low-confidence reading is replaced only by a more confident one.
            if reasons <= {'uncertain', 'unheard', 'hinted'} and _mean_probability(fresh) <= _mean_probability(old):
                continue
            rows = [row for row in rows if row not in old] + fresh
        else:
            added = []
            for row in fresh:
                part = _unheard_part(row, rows + added)
                if part is not None:
                    added.append(part)
            if not added:
                continue
            fresh = added
            rows = rows + fresh
        rows.sort(key=lambda row: row['start'])
        repairs.append({'reason': '+'.join(sorted(reasons)), 'start': round(float(start), 2),
                        'end': round(float(end), 2), 'segments': len(fresh)})
    return rows, repairs


CHUNK_SECONDS = 360.0      # parallel recognition: about this much audio per container


def plan_chunks(speech, duration, target=CHUNK_SECONDS):
    """[(start, end)] pieces of about `target` seconds for parallel recognition, cut in the middle of a pause between
    VAD speech regions so no word is split (preferring longer pauses near the ideal point); one piece for short media.
    """
    if duration <= target * 1.5:
        return [(0.0, duration)]
    count = math.ceil(duration / target)
    pauses = [((end + start) / 2, start - end) for (_, end), (start, _) in zip(speech, speech[1:]) if start > end]
    cuts = []
    for k in range(1, count):
        ideal = duration * k / count
        near = [p for p in pauses if abs(p[0] - ideal) <= 90 and (not cuts or p[0] > cuts[-1] + 30)]
        cuts.append(min(near, key=lambda p: abs(p[0] - ideal) - 5 * min(p[1], 2.0))[0] if near else ideal)
    bounds = [0.0] + cuts + [duration]
    return [(round(a, 3), round(b, 3)) for a, b in zip(bounds, bounds[1:]) if b - a > 0.5]


def detect_language(audio_path, *, download_root=None):
    """(language, probability) from up to three speech windows, so parallel pieces share one language."""
    from faster_whisper.vad import VadOptions
    # The WhisperModel itself: its own `.model` is the raw CTranslate2 model, whose detect_language takes other arguments.
    model, _ = load_model(download_root)
    audio = media.decode_array(audio_path, channels=1, rate=16000)
    # faster-whisper 1.2 hands vad_parameters straight to the VAD here, so it must be VadOptions, not a dict.
    language, probability, _ = model.detect_language(audio, vad_filter=True, vad_parameters=VadOptions(**VAD),
                                                     language_detection_segments=3)
    return from_provider(language), round(float(probability), 3)


def piece_hints(hints, start, end):
    """The `hints` inside one piece [start, end], shifted to the piece's own time."""
    return [(max(a, start) - start, min(b, end) - start) for a, b in hints if a < end and b > start]


def merge(parts, wall_seconds):
    """One transcript from pieces recognised in parallel: [(offset, transcript)] in time order. Times are shifted by
    each piece's offset; 'compute_seconds' adds up the recognition time of every piece (for the receipt)."""
    def shift(item, offset):
        return {**item, 'start': round(item['start'] + offset, 3), 'end': round(item['end'] + offset, 3)}

    first = parts[0][1]
    return {'language': first['language'], 'language_probability': first['language_probability'],
            'duration': round(parts[-1][0] + parts[-1][1]['duration'], 3),
            'segments': [shift(s, offset) for offset, part in parts for s in part['segments']],
            'words': sorted((shift(w, offset) for offset, part in parts for w in part['words']),
                            key=lambda word: word['start']),
            'repairs': [shift(r, offset) for offset, part in parts for r in part.get('repairs', [])],
            'device': first['device'], 'model': first['model'], 'pieces': len(parts),
            'seconds': round(wall_seconds, 2), 'compute_seconds': round(sum(p['seconds'] for _, p in parts), 2)}


def transcribe(audio_path, language=None, *, download_root=None, hints=()):
    """Words with timestamps plus the detected language (our codes, e.g. nb not no). `hints`: [(start, end)] where
    the upload's own subtitles show speech; one with no word heard is recognised again (see _repair)."""
    model, dev = load_model(download_root)
    started = time.monotonic()
    options = dict(language=to_provider(language) if language and language != 'auto' else None,
                   task='transcribe', beam_size=5, word_timestamps=True, vad_filter=True, vad_parameters=VAD,
                   temperature=(0.0, 0.2, 0.4), no_speech_threshold=0.6, log_prob_threshold=-1.0,
                   compression_ratio_threshold=2.4, condition_on_previous_text=False,
                   hallucination_silence_threshold=2.0)
    # Decode with ffmpeg ourselves: faster-whisper's PyAV path breaks across PyAV releases.
    audio = media.decode_array(audio_path, channels=1, rate=16000)
    segments, info = model.transcribe(audio, **options)
    rows = [row for row in _rows(segments) if _keep(row)]
    # Repairs keep the language the first pass detected.
    options['language'] = options['language'] or info.language
    rows, repairs = _repair(model, audio, rows, options, info.duration, [tuple(h) for h in hints or ()])
    words, kept = [], []
    for row in rows:
        kept.append({'start': round(row['start'], 3), 'end': round(row['end'], 3), 'text': row['text'].strip()})
        first = True
        for word in row['words']:
            if word['word'].strip() and not _is_noise(word['word']):
                # Whisper starts a new segment at most turn changes, even where it writes no punctuation.
                words.append({'start': round(word['start'], 3), 'end': round(word['end'], 3), 'word': word['word'],
                              'probability': round(word['probability'], 3), **({'segment_start': True} if first else {})})
                first = False
    # A recovered line can overlap a neighbour; segmenting needs the words in time order.
    words.sort(key=lambda word: word['start'])
    kept.sort(key=lambda segment: segment['start'])
    return {'language': from_provider(info.language), 'language_probability': round(info.language_probability, 3),
            'duration': round(info.duration, 3), 'segments': kept, 'words': words, 'repairs': repairs, 'device': dev,
            'model': MODEL, 'seconds': round(time.monotonic() - started, 2)}
