"""Subtitle tracks inside the upload: evidence for translation, hints for speech recognition, song and sign lines.

Ported from ai-voice-over v4 (ai_job_sources.extract, ass_events.py, SUBTITLE-INPUTS.md), which read every subtitle
track of an episode (text tracks directly, Blu-ray PGS with OCR) next to the speech recognition of every audio
track: the original speech stays primary, and a subtitle track (often an official translation) fixes names and words
the recognition misheard and shows where someone speaks. Vocnu also keeps what v4 set aside: song lyrics and
on-screen text ("overlay" lines: tracks titled signs or songs, ASS styles for OP/ED/songs/signs, PGS lines in the
upper half of the picture) become translated subtitles at their authored times.
ASS events are cleaned the v4 way: vector drawings are graphics, override tags are formatting, identical overlapping
layers of one style are one event. Commentary and forced tracks are not read.
"""
import logging
import re

from . import bitmap_subtitles, media
from .languages import code_for_tag

log = logging.getLogger('vocnu.file_subtitles')

TEXT_CODECS = {'subrip', 'srt', 'ass', 'ssa', 'webvtt', 'mov_text', 'text'}
SKIP_TRACK = re.compile(r'commentary|forced|credit', re.I)
OVERLAY_TRACK = re.compile(r'sign|song|karaoke|lyric', re.I)
OVERLAY_STYLE = re.compile(r'(^|[^a-z])(op|ed|signs?|songs?|karaoke|lyrics?|title|credits?)([^a-z]|$)', re.I)
DRAWING = re.compile(r'\{[^}]*\\p[1-9][^}]*\}')
OVERRIDE = re.compile(r'\{[^}]*\}')
MAX_DIALOGUE_TRACKS = 3
MAX_OVERLAY_TRACKS = 2


def candidates(streams, ocr_languages=frozenset(), prefer=()):
    """Subtitle streams to read, best first within each kind: [{'index', 'language', 'kind', 'bitmap'}], `kind`
    'dialogue' or 'overlay' (a track titled signs/songs). Bitmap tracks only in a language the OCR has.

    `prefer` lists language codes that rank first, in order: the original's language (what is said) and the target's
    (an official translation, with its names and terms). A film with thirty subtitle tracks otherwise gave its three
    reading slots to English, English SDH and Arabic, and left out its Lithuanian translation."""
    found = []
    for stream in streams:
        codec = stream.get('codec_name')
        bitmap = codec in bitmap_subtitles.BITMAP_CODECS
        if stream.get('codec_type') != 'subtitle' or not (codec in TEXT_CODECS or bitmap):
            continue
        tags, disposition = stream.get('tags') or {}, stream.get('disposition') or {}
        title = tags.get('title', '')
        if SKIP_TRACK.search(title) or disposition.get('forced') or disposition.get('comment'):
            continue
        language = tags.get('language', 'und').lower()
        if bitmap and bitmap_subtitles.TESSERACT.get(language) not in ocr_languages:
            continue
        found.append({'index': int(stream['index']), 'language': language, 'bitmap': bitmap,
                      'kind': 'overlay' if OVERLAY_TRACK.search(title) else 'dialogue',
                      'default': bool(disposition.get('default'))})
    rank = {code: i for i, code in enumerate(dict.fromkeys(c for c in prefer if c))}
    ranked = sorted(found, key=lambda t: (rank.get(code_for_tag(t['language']), len(rank)), not t['default'],
                                          t['bitmap'], t['index']))
    return ([t for t in ranked if t['kind'] == 'dialogue'][:MAX_DIALOGUE_TRACKS]
            + [t for t in ranked if t['kind'] == 'overlay'][:MAX_OVERLAY_TRACKS])


def _seconds(stamp):
    hours, minutes, seconds = stamp.split(':')
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def parse_ass(text):
    """[(start, end, text, role)] events of an ASS/SSA document, cleaned (see the module docstring); `role` is
    'overlay' for OP/ED/song/sign/title styles, else 'dialogue'."""
    columns, events, seen = None, [], set()
    for line in text.splitlines():
        if line.startswith('Format:') and columns is None and events == [] and 'Text' in line:
            columns = [c.strip().lower() for c in line[len('Format:'):].split(',')]
            continue
        if not line.startswith('Dialogue:') or columns is None:
            continue
        values = line[len('Dialogue:'):].split(',', len(columns) - 1)
        if len(values) != len(columns):
            continue
        event = dict(zip(columns, (v.strip() for v in values)))
        raw = event.get('text', '')
        if DRAWING.search(raw):
            continue
        clean = OVERRIDE.sub('', raw).replace('\\N', ' ').replace('\\n', ' ').replace('\\h', ' ')
        clean = ' '.join(clean.split())
        if not clean:
            continue
        start, end = _seconds(event['start']), _seconds(event['end'])
        key = (round(start, 2), round(end, 2), event.get('style', ''), clean)
        if key in seen:  # the same line drawn again as another layer
            continue
        seen.add(key)
        role = 'overlay' if OVERLAY_STYLE.search(event.get('style', '') + ' ' + event.get('name', '')) else 'dialogue'
        events.append((start, end, clean, role))
    return sorted(events)


def extract(source, track):
    """[(start, end, text, role)] of one subtitle stream. Text tracks are converted to ASS by ffmpeg so styles
    survive; bitmap tracks are read with OCR, and their lines in the upper half of the picture are 'overlay'."""
    if track['bitmap']:
        language = bitmap_subtitles.TESSERACT[track['language']]
        return [(start, end, text, 'overlay' if place == 'top' else 'dialogue')
                for start, end, text, place in bitmap_subtitles.extract(source, track, language)]
    out = media.run(['ffmpeg', '-nostdin', '-v', 'error', '-i', source, '-map', f"0:{track['index']}", '-f', 'ass',
                     'pipe:1'], timeout=120)
    return parse_ass(out.decode('utf-8', errors='replace'))


def _key(text):
    return ' '.join(re.findall(r'\w+', text.casefold()))


def _duplicate(cue, cues):
    return any(_key(cue[2]) == _key(other[2]) and abs(cue[0] - other[0]) < 0.5 for other in cues)


def layers(tracks):
    """Split read tracks into dialogue evidence and overlay lines.

    `tracks`: [{'language', 'kind', 'bitmap', 'cues': [(start, end, text, role)]}]. Returns (dialogue, (language,
    overlay)): dialogue is [{'language', 'cues': [(start, end, text)]}] per dialogue track; overlay is
    [(start, end, text)] in one language (the first signs/songs track's, else that of the first track with overlay
    lines), duplicates across tracks merged. A bitmap dialogue track whose language also has a signs/songs track
    keeps its upper-half lines as dialogue unless that track repeats them (dialogue moved up for on-screen text).
    """
    overlay_tracks = [t for t in tracks if t['kind'] == 'overlay']
    languages = [t['language'] for t in overlay_tracks] + [
        t['language'] for t in tracks if t['kind'] == 'dialogue' and any(c[3] == 'overlay' for c in t['cues'])]
    language = languages[0] if languages else None
    overlay_lines = [(a, b, text) for t in overlay_tracks if t['language'] == language for a, b, text, _ in t['cues']]
    dialogue, extra = [], []
    for track in tracks:
        if track['kind'] == 'overlay':
            continue
        has_overlay_track = any(t['language'] == track['language'] for t in overlay_tracks)
        spoken = []
        for start, end, text, role in track['cues']:
            moved_up = role == 'overlay' and track['bitmap'] and has_overlay_track
            if role == 'dialogue' or (moved_up and not _duplicate((start, end, text), overlay_lines)):
                spoken.append((start, end, text))
            elif track['language'] == language:
                extra.append((start, end, text))
        if spoken:
            dialogue.append({'language': track['language'], 'cues': spoken})
    overlay = []
    for start, end, text in sorted(overlay_lines + extra):
        if not any(_key(text) == _key(o[2]) and start < o[1] and end > o[0] for o in overlay):
            overlay.append((start, end, text))
    return dialogue, (language, overlay)


def read(source, streams, ocr_languages=frozenset(), prefer=()):
    """Every readable subtitle track of the upload: [{'index', 'language', 'kind', 'bitmap', 'cues'}]; a track that
    cannot be read is skipped (evidence is optional). `prefer` as in `candidates`."""
    tracks = []
    for track in candidates(streams, ocr_languages, prefer):
        try:
            cues = extract(source, track)
        except (media.MediaError, ValueError, OSError):
            log.warning('subtitle track %s could not be read', track['index'], exc_info=True)
            continue
        if cues:
            tracks.append({**track, 'cues': cues})
    return tracks


def overlapping(cues, start, end, slack=0.5):
    """Text of the cues that overlap [start, end] (with `slack` seconds on each side), joined in time order."""
    return ' '.join(cue[2] for cue in cues if cue[0] < end + slack and cue[1] > start - slack)
