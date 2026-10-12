"""One job end to end, per Vocnu's worker contract.

download → transcribe → translate → synthesize → mix → render → upload, with progress events, a hard
provider budget (checked before every paid step) and a duration guard before anything is spent.
"""
import bisect
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import json
import logging
import os
from pathlib import Path
import re
import threading
import time

from . import (asr, bitmap_subtitles, file_subtitles, fitting, media, mix, render, segments, sources, speakers,
               subtitles, titles, translate, tts)
from .costs import MODAL_CORE_USD_PER_SECOND, MODAL_L4_USD_PER_SECOND, BudgetExceeded, Receipt
from .errors import WorkerError
from .languages import BY_CODE, code_for_tag, get, track_tag
from .storage import check_job_id, output_key

log = logging.getLogger('vocnu.pipeline')

PRODUCTS = ('subtitles', 'voiceover_audio', 'voiceover_video', 'burned_subtitles')
CONTENT_TYPES = {'.srt': 'application/x-subrip', '.vtt': 'text/vtt', '.ass': 'text/x-ssa',
                 '.m4a': 'audio/mp4', '.mp3': 'audio/mpeg', '.wav': 'audio/wav', '.flac': 'audio/flac',
                 '.ogg': 'audio/ogg', '.mp4': 'video/mp4', '.mkv': 'video/x-matroska', '.mov': 'video/quicktime',
                 '.webm': 'video/webm', '.avi': 'video/x-msvideo'}
# Overall progress at the start of each stage.
# Seconds of work per minute of media in each stage, measured on the live worker (2026-10-09, a 60-minute voice-over
# video: recognition 142 s, translation 219 s, speech 473 s, mixing 141 s, rendering 163 s). Each stage a job runs gets
# its share of the progress bar, so the bar moves with time and the app can tell how long is left (elapsed time /
# progress). A stage without progress of its own (one long request, or ffmpeg) moves on with the time it is expected
# to take, every TICK_SECONDS, but never past TICK_CAP of the stage before it really ends.
STAGE_SECONDS_PER_MINUTE = {'downloading': 0.2, 'transcribing': 2.4, 'translating': 3.7, 'synthesizing': 7.9,
                            'mixing': 2.4, 'rendering': 2.7, 'uploading': 0.3}
TICK_SECONDS = 10
TICK_CAP = 0.9


def validate(job):
    """Normalised copy of a job request; ValueError for anything malformed."""
    if not isinstance(job, dict):
        raise ValueError('Job must be an object')
    check_job_id(job.get('job_id'))
    source = job.get('source')
    if not isinstance(source, dict) or source.get('kind') not in ('upload', 'file'):
        raise ValueError('Invalid source')
    get(job.get('target_language'))
    source_language = job.get('source_language') or 'auto'
    if not isinstance(source_language, str) or not (source_language == 'auto' or 2 <= len(source_language) <= 3):
        raise ValueError('Invalid source_language')
    raw = job.get('products') or {}
    products = {name: bool(raw.get(name)) for name in PRODUCTS}
    if not any(products[name] for name in ('subtitles', 'voiceover_audio', 'voiceover_video')):
        raise ValueError('No product requested')
    if products['burned_subtitles'] and not products['voiceover_video']:
        raise ValueError('burned_subtitles requires voiceover_video')
    voice = job.get('voice') or 'female'
    tts.voice_for(voice)
    override = job.get('voice_override')
    if override is not None:
        # A community voice (Vocnu's ADR 0017): its ElevenLabs id, the donor's name and the credit for the outputs.
        if not isinstance(override, dict) or not re.fullmatch(r'[A-Za-z0-9]{8,64}', str(override.get('id'))) \
                or not 0 < len(str(override.get('name') or '')) <= 60 or len(str(override.get('credit') or '')) > 100:
            raise ValueError('Invalid voice_override')
    video_format = job.get('video_format') or 'mp4'
    if video_format not in render.VIDEO_FORMATS:
        raise ValueError('Invalid video_format')
    audio_format = job.get('audio_format') or 'm4a'
    if audio_format not in media.AUDIO_CODECS:
        raise ValueError('Invalid audio_format')
    max_duration = float(job.get('max_duration_seconds') or 0)
    budget = float(job.get('budget_usd') or 0)
    if max_duration <= 0 or budget <= 0 or budget > 200:
        raise ValueError('Invalid duration or budget limits')
    return {**job, 'source_language': source_language, 'products': products, 'voice': voice,
            'voice_override': override,
            'video_format': video_format, 'audio_format': audio_format,
            'max_duration_seconds': max_duration, 'budget_usd': budget}


class _Progress:
    def __init__(self, sink, stages, clock=time.monotonic):
        self.sink, self.stages, self.clock = sink, stages, clock
        self.entered = {}   # stage -> when it started
        total = sum(STAGE_SECONDS_PER_MINUTE[stage] for stage in stages)
        shares = [STAGE_SECONDS_PER_MINUTE[stage] / total for stage in stages]
        self.start = {stage: sum(shares[:i]) for i, stage in enumerate(stages)}
        self.end = {stage: self.start[stage] + share for stage, share in zip(stages, shares)}
        self.media_minutes = 0.0
        self.current, self.reached = None, {}   # the stage now running; the furthest fraction of each stage
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None

    def __call__(self, stage, fraction=0.0):
        """Report `fraction` (0–1) of `stage`, scaled onto the stages this job actually runs."""
        with self._lock:
            self.entered.setdefault(stage, self.clock())
            self.current = stage
        self._report(stage, fraction)

    def _report(self, stage, fraction):
        fraction = min(1.0, max(0.0, fraction))
        with self._lock:
            if fraction < self.reached.get(stage, 0.0):
                return
            self.reached[stage] = fraction
        self.sink.progress(stage, self.start[stage] + (self.end[stage] - self.start[stage]) * fraction)

    def tick(self):
        """Move the running stage on with the time it is expected to take, when that is further than it got."""
        with self._lock:
            stage = self.current
            if stage is None or self.media_minutes <= 0:
                return
            expected = STAGE_SECONDS_PER_MINUTE[stage] * self.media_minutes
            fraction = min(TICK_CAP, (self.clock() - self.entered[stage]) / expected)
            if fraction <= self.reached.get(stage, 0.0):
                return
        self._report(stage, fraction)

    def start_ticking(self, media_seconds):
        self.media_minutes = max(0.0, media_seconds / 60)
        if self._thread is None:
            self._thread = threading.Thread(target=self._tick_until_stopped, name='progress-ticker', daemon=True)
            self._thread.start()

    def stop(self):
        self._stop.set()

    def _tick_until_stopped(self):
        while not self._stop.wait(TICK_SECONDS):
            try:
                self.tick()
            except Exception:  # noqa: BLE001 - progress is best effort and must never break the job
                log.warning('progress tick failed', exc_info=True)

    def seconds(self):
        """{stage: seconds} each stage took, from its start to the next one's (the last one up to now)."""
        order = sorted(self.entered.items(), key=lambda item: item[1])
        ends = [when for _, when in order[1:]] + [self.clock()]
        return {stage: round(end - when, 1) for (stage, when), end in zip(order, ends)}


def _speech_hints(dialogue_tracks):
    """[(start, end)] where the upload's dialogue subtitles show speech: one per line (each is checked on its own: a
    word heard at the end of a joined stretch would hide the lines before it); overlapping lines of two tracks are
    joined."""
    spans = []
    for start, end in sorted((c[0], c[1]) for track in dialogue_tracks for c in track['cues']):
        if spans and start < spans[-1][1]:
            spans[-1][1] = max(spans[-1][1], end)
        else:
            spans.append([start, end])
    return [(round(a, 3), round(b, 3)) for a, b in spans]


SONG_GAP = 30.0   # lyric lines of one song are closer than this (Bleach's opening has a 19.5 s instrumental gap)


def _song_regions(lines):
    """[(start, end)] of each song: the upload's lyric lines less than SONG_GAP apart, joined."""
    spans = []
    for start, end, _ in sorted(lines):
        if spans and start - spans[-1][1] < SONG_GAP:
            spans[-1][1] = max(spans[-1][1], end)
        else:
            spans.append([start, end])
    return [(a, b) for a, b in spans]


def _inside(unit, regions, slack=1.0):
    return any(a - slack <= unit.start and unit.end <= b + slack for a, b in regions)


ALIGNED_SHARE = 0.6   # a dialogue subtitle track lines up when this share of its lines has recognised speech near


def _near_speech(cue, starts, slack=0.5):
    i = bisect.bisect_left(starts, cue[0] - slack)
    return i < len(starts) and starts[i] <= cue[1] + slack


def _aligned(dialogue_tracks, words):
    """(tracks that line up with the recognised speech, languages of the ones that do not)."""
    starts = sorted(w['start'] for w in words)
    kept, dropped = [], []
    for track in dialogue_tracks:
        near = sum(1 for cue in track['cues'] if _near_speech(cue, starts))
        (kept if near >= ALIGNED_SHARE * len(track['cues']) else dropped).append(track)
    return kept, [track['language'] for track in dropped]


SENTENCE_STOP = ('.', '!', '?', '…', '。', '！', '？', '؟')
_SENTENCE_ENDS = re.compile(r'[.!?…。！？؟](?=\s|$)')


def _turn_candidates(units, dialogue_tracks, dub_transcripts):
    """[(request item, the unit's words)] for units whose upload subtitles (the first aligned dialogue track) show two
    or more lines with a sentence ending before the last, or whose dub (the first one) says two sentences."""
    cues = dialogue_tracks[0]['cues'] if dialogue_tracks else []
    dub = dub_transcripts[0] if dub_transcripts else None
    found = []
    for unit in units:
        own = unit.words
        if len(own) < 2:
            continue
        lines = [c[2] for c in cues if c[0] < unit.end and c[1] > unit.start]
        split_lines = len(lines) > 1 and any(line.rstrip().rstrip('"\'”’»)').endswith(SENTENCE_STOP)
                                             for line in lines[:-1])
        said = ''
        if dub:
            spaced = segments._spaced(code_for_tag(dub[0]['tag']) or '')
            heard = [w['word'].strip() for w in dub[1].get('words') or []
                     if w['start'] < unit.end and w['end'] > unit.start]
            said = (' ' if spaced else '').join(heard)
        if split_lines or len(_SENTENCE_ENDS.findall(said)) > 1:
            found.append(({'id': unit.id, 'words': [w['word'].strip() for w in own], 'subtitles': lines,
                           **({'dub': said} if said else {})}, own))
    return found


FRAGMENT_CHARS = 3   # a piece this short at either end of a unit, run into its neighbour, is part of that line
FRAGMENT_GAP = 0.3


def _turn_cuts(own, starts, words):
    """(starts, joins) turn keys for one unit's words `own` cut before the word numbers `starts`. Recognition often
    ends a segment one syllable into the next speaker's line ("…大正解め" + "っちゃん"): a piece of at most
    FRAGMENT_CHARS at the end of the unit that runs straight into the next word continues that next line instead of
    becoming a line of its own, and a short piece at the start likewise continues the line before."""
    if not starts:
        return set(), set()
    keys, joins = {segments.turn_key(own[k]) for k in starts}, set()
    position = {segments.turn_key(w): i for i, w in enumerate(words)}

    def short(piece):
        return len(''.join(w['word'].strip() for w in piece)) <= FRAGMENT_CHARS

    after = position.get(segments.turn_key(own[-1]), len(words)) + 1
    if short(own[starts[-1]:]) and after < len(words) and words[after]['start'] - own[-1]['end'] < FRAGMENT_GAP:
        joins.add(segments.turn_key(words[after]))
    before = position.get(segments.turn_key(own[0]), 0) - 1
    if short(own[:starts[0]]) and before >= 0 and own[0]['start'] - words[before]['end'] < FRAGMENT_GAP:
        joins.add(segments.turn_key(own[0]))
    return keys, joins


UNHEARD_SLACK = 1.0   # subtitle times run up to about a second off the speech: a line with a word this close was heard


def _unheard_lines(track, words):
    """Lines of a dialogue subtitle track with no recognised word near them (sound labels left out)."""
    starts = sorted(w['start'] for w in words)
    return [(cue[0], cue[1], cue[2]) for cue in track['cues']
            if not _near_speech(cue, starts, slack=UNHEARD_SLACK) and not translate.is_nonverbal(cue[2])
            and not asr.LABEL.match(cue[2].strip())]


def _evidence(units, dialogue_tracks, dub_transcripts):
    """{unit id: {source: text}}: what the upload's dialogue subtitle tracks ("eng"; a second track of one language is
    "eng 2") and the recognised speech of its dubs ("eng dub") have for the same moment."""
    names, used = [], Counter()
    for track in dialogue_tracks:
        used[track['language']] += 1
        names.append(track['language'] + (f" {used[track['language']]}" if used[track['language']] > 1 else ''))
    dubs = [(f"{dub['tag']} dub", segments._spaced(code_for_tag(dub['tag']) or ''),
             [(w['start'], w['end'], w['word'].strip()) for w in transcript.get('words') or []])
            for dub, transcript in dub_transcripts]
    evidence = {}
    for unit in units:
        said = {name: file_subtitles.overlapping(track['cues'], unit.start, unit.end)
                for name, track in zip(names, dialogue_tracks)}
        for name, spaced, words in dubs:
            heard = [word for a, b, word in words if a < unit.end + 0.8 and b > unit.start - 0.8]
            said[name] = (' ' if spaced else '').join(heard)
        if any(said.values()):
            evidence[unit.id] = {name: text for name, text in said.items() if text}
    return evidence


def _source_name(source):
    """The uploaded file's name (the last part of its storage key), a hint for identifying the title."""
    return Path(str(source.get('key') or source.get('path') or '')).name


def _write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding='utf-8')
    return path


def process(job, workdir, *, store, sink, transcribe, compute='local', cpu_cores=4,
            asr_resource=('L4', MODAL_L4_USD_PER_SECOND)):
    """Run the job; always ends with exactly one terminal event on `sink`. Returns the event payload.

    `asr_resource` is (label, USD per second) of whatever ran speech recognition, for the receipt.
    """
    started = time.monotonic()
    receipt = Receipt(max_usd=float((job or {}).get('budget_usd') or 0) or 0.01)
    gpu_seconds = 0.0

    def charge():
        _charge_compute(receipt, compute, gpu_seconds, time.monotonic() - started, cpu_cores, asr_resource)

    progress = None
    sink.start_heartbeat()
    try:
        job = validate(job)
        receipt.max_usd = job['budget_usd']
        products, target = job['products'], job['target_language']
        override = job.get('voice_override')
        voice = ({'id': override['id'], 'name': override['name'], 'gender': job['voice']} if override
                 else job['voice'])
        credit = (override.get('credit') or override['name']) if override else None
        lang = get(target)
        wants_voice = products['voiceover_audio'] or products['voiceover_video']
        stages = ['downloading', 'transcribing', 'translating']
        if wants_voice:
            stages += ['synthesizing', 'mixing'] + (['rendering'] if products['voiceover_video'] else [])
        stages.append('uploading')
        progress = _Progress(sink, stages)
        work = Path(workdir)
        work.mkdir(parents=True, exist_ok=True)

        # 1. Download and check the duration before any paid step.
        progress('downloading')
        named = job['source_language'] if job['source_language'] != 'auto' else None
        source_path, info = sources.fetch(job['source'], work / 'source', store,
                                          avoid_languages=(target, track_tag(target)),
                                          prefer_languages=(named, track_tag(named)) if named else ())
        duration = info['duration_seconds']
        progress.start_ticking(duration)
        if duration > job['max_duration_seconds']:
            raise WorkerError('duration_mismatch', f'The media is {duration:.0f} s long, longer than quoted '
                                                   f"({job['max_duration_seconds']:.0f} s). Please request a new quote.")
        if products['voiceover_video'] and not info['has_video']:
            raise WorkerError('unsupported_media', 'This file has no video track; choose voice-over audio instead.')

        # 2. Speech recognition (GPU on Modal, local GPU/CPU in the CLI), with every other source of truth the upload
        # carries, as ai-voice-over v4 used them: its subtitle tracks (text, or Blu-ray bitmaps read with OCR) and the
        # speech of its dubs. The original speech stays primary; the rest fixes names and misheard words, shows where
        # recognition missed a line (`hints`: recognised again), and gives song lyrics and signs their subtitles.
        progress('transcribing')
        streams = info.get('streams') or []
        original = next((s for s in streams if s.get('index') == info['audio_index']), {})
        spoken = code_for_tag((original.get('tags') or {}).get('language')) or named
        subtitle_tracks = file_subtitles.read(source_path, streams, bitmap_subtitles.available_languages(),
                                              prefer=(spoken, named, target))
        dialogue_tracks, (overlay_language, overlay_lines) = file_subtitles.layers(subtitle_tracks)
        hints = _speech_hints(dialogue_tracks)
        asr_path = media.asr_audio(source_path, work / 'asr.opus', info['audio_index'], duration)
        dubs = media.dub_tracks(streams, info['audio_index'])
        asr_started = time.monotonic()
        with ThreadPoolExecutor(max_workers=1 + len(dubs)) as pool:
            main = pool.submit(transcribe, asr_path, job['source_language'], hints=hints)
            pending = []
            for dub in dubs:
                try:
                    dub_path = media.asr_audio(source_path, work / f"dub-{dub['index']}.opus", dub['index'], duration)
                except media.MediaError:
                    log.warning('dub track %s could not be extracted', dub['index'], exc_info=True)
                    continue
                pending.append((dub, pool.submit(transcribe, dub_path, code_for_tag(dub['tag']) or 'auto')))
            transcript = main.result()
            dub_transcripts = []
            for dub, future in pending:
                try:
                    dub_transcripts.append((dub, future.result()))
                except Exception:  # noqa: BLE001 - evidence is optional; the job goes on without it
                    log.warning('dub track %s recognition failed', dub['index'], exc_info=True)
        # Pieces recognised in parallel bill their summed container time, not the wall time.
        gpu_seconds = (transcript.get('compute_seconds') or time.monotonic() - asr_started) + sum(
            t.get('compute_seconds') or t.get('seconds') or 0.0 for _, t in dub_transcripts)
        words = transcript.get('words') or []
        if not words:
            raise WorkerError('no_audio', 'We could not find any speech to translate in this media.')
        source_language = transcript['language'] if job['source_language'] == 'auto' else job['source_language']

        # 3. Translation of narrator lines (time-budgeted) and per-cue subtitles.
        progress('translating')
        cues, units = segments.build(words, source_language)
        # A subtitle track from another edition (shifted times) would mislead: only tracks that line up with the
        # recognised speech are used. A dialogue line such a track shows where recognition heard nothing, even after
        # recognising it again, joins as its own unit (from the subtitle's text), so no line is skipped.
        dialogue_tracks, misaligned = _aligned(dialogue_tracks, words)
        # Speaker turns: recognition can run two characters' lines together without punctuation (Bleach S01E01, 3:51:
        # "…持ってきてやるよありがとう…" became one narrator line and one subtitle). Where the upload's subtitles or
        # dub show two lines, the model names the word that starts the second one, and the units are cut there.
        turn_starts, turn_joins = set(), set()
        candidates = _turn_candidates(units, dialogue_tracks, dub_transcripts)
        if candidates:
            try:
                receipt.check(translate.turns_estimate_usd([c for c, _ in candidates]), 'Speaker turns')
            except BudgetExceeded as error:
                log.warning('speaker turns skipped: %s', error)
            else:
                answer = translate.turns([c for c, _ in candidates], source=source_language, receipt=receipt)
                for candidate, own in candidates:
                    starts, joins = _turn_cuts(own, answer.get(candidate['id'], []), words)
                    turn_starts |= starts
                    turn_joins |= joins
            if turn_starts or turn_joins:
                cues, units = segments.build(words, source_language, frozenset(turn_starts), frozenset(turn_joins))
        unheard_lines = _unheard_lines(dialogue_tracks[0], words) if dialogue_tracks else []
        added = set()
        if unheard_lines:
            cues, units, added = segments.with_lines(cues, units, unheard_lines)
        speaker_genders = speakers.genders(speakers.load(asr_path), units)
        available = mix.windows(units, duration)
        budgets = {u.id: lang.chars_for(available[u.id]) for u in units}
        # What the upload's subtitle tracks and dubs say at each unit goes into every translation request; the
        # analysis also sees the song and sign lines shown at each unit, so lyrics are recognised as songs.
        evidence = _evidence(units, dialogue_tracks, dub_transcripts)
        evidence_chars = sum(len(text) for said in evidence.values() for text in said.values())
        shown = _evidence(units, [{'language': f'{overlay_language} songs/signs', 'cues': overlay_lines}], [])
        analysis_evidence = {uid: {**evidence.get(uid, {}), **shown.get(uid, {})} for uid in {*evidence, *shown}}
        # The independent review checks its own estimate before it runs and is skipped (logged in stats) when the
        # budget left cannot cover it, so a tight subtitles-only budget still gets its subtitles.
        receipt.check(translate.estimate_usd(units, cues, target=target, review=False, evidence_chars=evidence_chars),
                      'Translation')
        # Names and terms: a known film or series reuses what earlier jobs decided (Rails keeps a shared glossary),
        # the rest is decided once for the whole transcript, so chunks translated in parallel agree. The same pass
        # classifies every unit: only dialogue is narrated.
        title = translate.identify(units, filename=_source_name(job['source']), source=source_language,
                                   target=target, receipt=receipt)

        def lookup():
            return (titles.sync(job.get('callback_url'), title, target, [], job['job_id'],
                                names_api=job.get('names_api')) or []) if title else []

        decided = lookup()
        analysis = translate.analyse(units, source=source_language, target=target, receipt=receipt, decided=decided,
                                     evidence=analysis_evidence, unheard=added)
        glossary = analysis['glossary']
        if title and analysis['new']:
            # Episodes of one series often run at the same time: if another job decided terms meanwhile, redo the
            # analysis so this job's variants of those names follow them instead of storing competing renderings.
            latest = lookup()
            if len(latest) > len(decided):
                decided = latest
                analysis = translate.analyse(units, source=source_language, target=target, receipt=receipt,
                                             decided=decided, evidence=analysis_evidence, unheard=added)
                glossary = analysis['glossary']
            if analysis['new']:
                glossary = titles.sync(job.get('callback_url'), title, target, analysis['new'], job['job_id'],
                                       names_api=job.get('names_api')) or glossary
        # Song lyrics and on-screen text (`sung`) get subtitles but no narration. Sounds without words (screams,
        # laughs, sobs) and recognition artefacts (`silent`) get neither: the original soundtrack keeps them.
        sung, silent = analysis['sung'], analysis.get('silent') or set()
        # Case forms of names are decided per job: Rails keeps only base renderings.
        glossary = translate.with_forms(glossary, analysis.get('forms'))
        quality = {}
        # Song lyrics and signs from the upload's own subtitles, translated line by line at their authored times, at
        # the top of the picture; they replace the lyrics speech recognition caught of the same songs. Inside a song
        # the file has lyrics for, what recognition heard is the song, not dialogue (Bleach S01E01: a sung "その" at
        # 0:14 between two lyric lines of the opening became a narrated, subtitled "Tas..."), unless a dialogue
        # subtitle shows someone speaking there: it is neither translated, narrated nor subtitled.
        overlay_cues, regions = [], []
        if overlay_lines:
            try:
                receipt.check(translate.overlay_estimate_usd(overlay_lines), 'Song and sign translation')
            except BudgetExceeded as error:
                log.warning('song and sign lines skipped: %s', error)
                quality['overlay_skipped'] = str(error)
            else:
                heard = {k: ' '.join(c.text for c in cues if c.start < b and c.end > a)
                         for k, (a, b, _) in enumerate(overlay_lines)}
                translated_lines = translate.overlay(overlay_lines, source=overlay_language, target=target,
                                                     receipt=receipt, glossary=glossary, heard=heard)
                overlay_cues = [(a, b, translated_lines[k][0]) for k, (a, b, _) in enumerate(overlay_lines)
                                if k in translated_lines and translated_lines[k][0]]
                regions = _song_regions([line for k, line in enumerate(overlay_lines)
                                         if translated_lines.get(k, ('', None))[1] == 'song'])
        spoken = {uid for uid, said in evidence.items() if any(not name.endswith(' dub') for name in said)}
        in_songs = {u.id for u in units if _inside(u, regions) and u.id not in spoken}
        silent, sung = silent | in_songs, sung - in_songs
        translations = translate.translate([u for u in units if u.id not in silent], cues, source=source_language,
                                           target=target, budgets=budgets, receipt=receipt, speakers=speaker_genders,
                                           glossary=glossary, sung=sung, stats=quality, evidence=evidence)
        for unit in units:
            if unit.id in silent:
                translations[unit.id] = {'voiceover': '', 'subtitles': [''] * len(unit.cue_ids)}
        cue_text = {}
        for unit in units:
            for cue_id, text in zip(unit.cue_ids, translations[unit.id]['subtitles']):
                cue_text[cue_id] = text
        sung_cue_ids = {c for u in units if u.id in sung for c in u.cue_ids}
        replaced = {c.id for c in cues if c.id in sung_cue_ids and any(a < c.end and b > c.start
                                                                        for a, b, _ in overlay_cues)}
        target_cues = [(c.start, c.end, '' if c.id in replaced else cue_text.get(c.id, '')) for c in cues]
        sung_cues = [cue for c, cue in zip(cues, target_cues) if c.id in sung_cue_ids]
        # The transcript is what was recognised in the original; lines taken from the subtitles are not part of it.
        from_subtitles = {c for u in units if u.id in added for c in u.cue_ids}
        source_cues = [(c.start, c.end, c.text) for c in cues if c.id not in from_subtitles]
        src_lang = BY_CODE.get(source_language)
        files = {}
        sub_dir = work / 'subtitles'
        all_cues = subtitles.layered(target_cues, overlay_cues)
        target_srt = _write(sub_dir / f'subtitles.{target}.srt', subtitles.srt(all_cues, lang.line_chars, lang.spaced))
        if products['subtitles']:
            files['subtitles_srt'] = (target_srt, target)
            files['subtitles_vtt'] = (_write(sub_dir / f'subtitles.{target}.vtt',
                                             subtitles.vtt(all_cues, lang.line_chars, lang.spaced)), target)
            files['subtitles_ass'] = (_write(sub_dir / f'subtitles.{target}.ass',
                                             subtitles.ass(all_cues, lang.line_chars, lang.spaced)), target)
            files['transcript_srt'] = (_write(sub_dir / f'transcript.{source_language}.srt', subtitles.srt(
                source_cues, src_lang.line_chars if src_lang else 42, src_lang.spaced if src_lang else True)),
                source_language)
        progress('translating', 1.0)

        stats = {'cues': len(cues), 'units': len(units), 'sung_units': len(sung), 'title': title or None,
                 'decided_terms': len(decided), 'glossary_terms': len(glossary), 'silent_units': len(silent),
                 'file_subtitles': [{'language': t['language'], 'kind': t['kind'], 'bitmap': t['bitmap'],
                                     'lines': len(t['cues'])} for t in subtitle_tracks],
                 'dub_tracks': [dub['tag'] for dub, _ in dub_transcripts], 'overlay_lines': len(overlay_cues),
                 'songs': [[round(a, 1), round(b, 1)] for a, b in regions], 'song_units': len(in_songs),
                 'hinted_repairs': sum(1 for r in transcript.get('repairs') or [] if 'hinted' in r.get('reason', '')),
                 'subtitle_lines': len(added), 'misaligned_subtitles': misaligned,
                 'turn_candidates': len(candidates), 'turn_splits': len(turn_starts),
                 **quality}
        if wants_voice:
            # 4. Speech at natural speed, fitted into gap-aware timing budgets (timing.py): lines that do not fit get
            # same-text retakes (short ones) or measured rephrasing, kept only when the take is shorter and timing
            # improves; a little speed-up only as the last resort (fitting.py).
            progress('synthesizing')
            translated = {u.id: translations[u.id]['voiceover'] for u in units if translations[u.id]['voiceover']}
            done = [0]

            def tick(_):
                done[0] += 1
                if done[0] % 5 == 0 or done[0] == len(translated):
                    progress('synthesizing', 0.8 * done[0] / max(1, len(translated)))

            narration = fitting.narrate(units, translated, media_duration=duration, language=target, voice=voice,
                                        cache=work / 'tts', receipt=receipt, cues=cues, speakers=speaker_genders,
                                        glossary=glossary, on_done=tick)
            lines, placements = narration.lines, narration.placements
            stats['timing'] = narration.report
            progress('synthesizing', 1.0)

            # 5. Mix over the ducked original, normalise to -16 LUFS, encode.
            progress('mixing')
            missing = set(narration.missing)
            rephrased = {uid: text for uid, text in translated.items() if lines.get(uid, text) != text}

            def rounded(record):
                return {k: round(v, 3) if isinstance(v, float) else v for k, v in vars(record).items()}

            _write(work / 'plan.json', json.dumps({'units': [
                {'id': u.id, 'start': u.start, 'end': u.end, 'source': u.text, 'voiceover': lines.get(u.id, ''),
                 'speech_seconds': round(narration.seconds.get(u.id, 0.0), 3),
                 **({'translated': rephrased[u.id]} if u.id in rephrased else {}),
                 **({'sung': True} if u.id in sung else {}), **({'missing_speech': True} if u.id in missing else {})}
                for u in units], 'title': title, 'glossary': glossary,
                'placements': [rounded(p) for p in placements],
                'timing': {**narration.report, 'off_budget': [rounded(v) for v in narration.violations]}},
                ensure_ascii=False, indent=1))
            # Subtitles delivered with a voice-over show what the narrator says (the faithful per-cue translation
            # differs from the shortened narration, and viewers notice when text and voice disagree), timed by the
            # take's character timestamps, plus the translated lyrics of songs and the translation of any line whose
            # speech came back silent: neither is narrated.
            unspoken = {c for u in units if u.id in missing for c in u.cue_ids}
            unspoken_cues = [cue for c, cue in zip(cues, target_cues) if c.id in unspoken]
            narrated_cues = subtitles.narration_cues(
                placements, lines, lang.line_chars, lang.spaced, duration=duration, reading_cps=lang.reading_cps,
                alignments={uid: tts.alignment(path) for uid, path in narration.paths.items()})
            voice_cues = subtitles.layered(subtitles.merge(narrated_cues, sung_cues, unspoken_cues), overlay_cues)
            if products['subtitles']:
                files['subtitles_srt'] = (_write(sub_dir / f'subtitles.{target}.srt',
                                                 subtitles.srt(voice_cues, lang.line_chars, lang.spaced)), target)
                files['subtitles_vtt'] = (_write(sub_dir / f'subtitles.{target}.vtt',
                                                 subtitles.vtt(voice_cues, lang.line_chars, lang.spaced)), target)
                files['subtitles_ass'] = (_write(sub_dir / f'subtitles.{target}.ass',
                                                 subtitles.ass(voice_cues, lang.line_chars, lang.spaced)), target)
            # v4 mix: the original at its own level, dipped under each narrator line; songs have no narration.
            mixed = mix.mix(source_path, info, placements, narration.paths, work / 'mix')
            stats.update({k: mixed[k] for k in ('lines', 'sped_up_lines', 'max_tempo')})
            audio = media.encode_audio(mixed['voiceover_wav'], work / f"voiceover.{target}.{job['audio_format']}",
                                       language=track_tag(target), title=media.voice_track_title(lang.native, credit),
                                       credit=credit)
            if products['voiceover_audio']:
                files['voiceover_audio'] = (audio, target)

            # 6. Video: copy the picture, voice-over first, original second, subtitles as switchable tracks (off by
            # default); burned into the picture only on request, and then not repeated as a track.
            if products['voiceover_video']:
                progress('rendering')
                burn = None
                tracks = [(_write(work / 'tracks' / f'transcript.{source_language}.srt', subtitles.srt(
                    source_cues, src_lang.line_chars if src_lang else 42, src_lang.spaced if src_lang else True)),
                    track_tag(source_language), f'Transcript ({src_lang.native if src_lang else source_language})')]
                if products['burned_subtitles']:
                    burn = _write(work / 'burn' / 'burn.srt',
                                  subtitles.srt(voice_cues, lang.line_chars, lang.spaced))
                elif render.CONTAINERS[job['video_format']].subtitles == 'ass':
                    # MKV holds styled subtitles: song lyrics and signs keep their place at the top.
                    tracks.insert(0, (_write(work / 'tracks' / f'subtitles.{target}.ass',
                                             subtitles.ass(voice_cues, lang.line_chars, lang.spaced)),
                                      track_tag(target), f'Subtitles ({lang.native})'))
                else:
                    tracks.insert(0, (_write(work / 'tracks' / f'subtitles.{target}.srt',
                                             subtitles.srt(voice_cues, lang.line_chars, lang.spaced)),
                                      track_tag(target), f'Subtitles ({lang.native})'))
                video = render.video(source_path, mixed['voiceover_wav'], work / f"video.{target}.{job['video_format']}",
                                     info=info, target=target, source_language=source_language, burn_srt=burn,
                                     subtitle_tracks=tracks, threads=cpu_cores, credit=credit)
                files['voiceover_video'] = (video, target)

        # 7. Upload.
        progress('uploading')
        outputs = []
        for kind, (path, language) in files.items():
            key = store.upload(path, output_key(job['job_id'], path.name), CONTENT_TYPES[path.suffix])
            outputs.append({'kind': kind, 'language': language, 'key': key,
                            'content_type': CONTENT_TYPES[path.suffix], 'bytes': path.stat().st_size})
        charge()
        # Where the time went (the landing page's processing times are measured with these).
        stats['stage_seconds'] = progress.seconds()
        payload = {'duration_seconds': duration, 'source_language': source_language, 'outputs': outputs,
                   'receipt': receipt.as_dict()}
        log.info('job %s completed in %.1fs: %s', job['job_id'], time.monotonic() - started, stats)
        sink.completed(payload)
        return {'event': 'completed', **payload, 'stats': stats}
    except BudgetExceeded as error:
        return _fail(sink, receipt, WorkerError('budget_exceeded', str(error)), charge)
    except WorkerError as error:
        return _fail(sink, receipt, error, charge)
    except ValueError as error:
        log.warning('job rejected: %s', error)
        return _fail(sink, receipt, WorkerError('internal_error', 'The job request was invalid.'), charge)
    except Exception:  # noqa: BLE001 - every failure must reach Rails as one event
        log.exception('job failed unexpectedly')
        return _fail(sink, receipt, WorkerError('internal_error', 'Processing failed unexpectedly. Please try again.'),
                     charge)
    finally:
        if progress is not None:
            progress.stop()
        sink.stop_heartbeat()


def _charge_compute(receipt, compute, asr_seconds, wall_seconds, cpu_cores, asr_resource):
    if compute == 'modal':
        label, rate = asr_resource
        receipt.add('compute', 'modal', asr_seconds * rate, resource=label, seconds=round(asr_seconds, 1))
        receipt.add('compute', 'modal', wall_seconds * cpu_cores * MODAL_CORE_USD_PER_SECOND, resource='cpu',
                    cores=cpu_cores, seconds=round(wall_seconds, 1))
    else:
        receipt.add('compute', 'local', 0.0, asr_seconds=round(asr_seconds, 1), seconds=round(wall_seconds, 1))


def _fail(sink, receipt, error, charge):
    charge()
    sink.failed(error.as_dict(), receipt.as_dict())
    return {'event': 'failed', 'error': error.as_dict(), 'receipt': receipt.as_dict()}


def default_workdir(job_id):
    return Path(os.environ.get('VOCNU_WORK_DIR', '/tmp/vocnu')) / job_id
