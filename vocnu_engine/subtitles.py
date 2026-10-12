"""SRT, WebVTT and ASS writers with balanced two-line wrapping, and readable narration subtitles timed by the take."""
import math

BREAK_AFTER = (',', '，', '、', '。', '；', ';', ':', '：', '!', '?', '！', '？')


def _stamp(seconds, separator):
    millis = max(0, round(seconds * 1000))
    hours, millis = divmod(millis, 3_600_000)
    minutes, millis = divmod(millis, 60_000)
    secs, millis = divmod(millis, 1000)
    return f'{hours:02d}:{minutes:02d}:{secs:02d}{separator}{millis:03d}'


def wrap(text, line_chars, spaced=True):
    """At most two lines, split near the middle; never breaks inside a word in spaced scripts."""
    text = ' '.join(text.split()) if spaced else ''.join(text.split())
    if len(text) <= line_chars:
        return [text] if text else []
    middle = len(text) / 2
    if spaced:
        spaces = [i for i, ch in enumerate(text) if ch == ' ']
        if not spaces:
            return [text]
        fitting = [i for i in spaces if i <= line_chars and len(text) - i - 1 <= line_chars]
        best = min(fitting or spaces, key=lambda i: (abs(i - middle) - (3 if text[i - 1] in BREAK_AFTER else 0)))
        return [text[:best], text[best + 1:]]
    candidates = range(1, len(text))
    best = min(candidates, key=lambda i: (abs(i - middle) - (2 if text[i - 1] in BREAK_AFTER else 0)))
    return [text[:best], text[best:]]


def _top(cue):
    """True for a cue placed at the top of the picture: (start, end, text, 'top'), used for song and sign lines."""
    return len(cue) > 3 and cue[3] == 'top'


def layered(dialogue, overlay):
    """One time-ordered list of dialogue cues (bottom) and overlay cues (song lyrics, signs: top); the two layers may
    overlap in time, each layer on its own does not."""
    return sorted([tuple(c[:3]) for c in dialogue] + [(*c[:3], 'top') for c in overlay], key=lambda c: (c[0], c[1]))


def srt(cues, line_chars=42, spaced=True):
    """`cues`: iterable of (start, end, text) or (start, end, text, place); SRT has no placement, so players stack
    cues that overlap."""
    blocks = []
    for number, cue in enumerate((c for c in cues if c[2].strip()), 1):
        start, end, text = cue[:3]
        lines = '\n'.join(wrap(text, line_chars, spaced))
        blocks.append(f'{number}\n{_stamp(start, ",")} --> {_stamp(end, ",")}\n{lines}\n')
    return '\n'.join(blocks)


def vtt(cues, line_chars=42, spaced=True):
    blocks = ['WEBVTT\n']
    for cue in cues:
        start, end, text = cue[:3]
        if text.strip():
            lines = '\n'.join(wrap(text, line_chars, spaced))
            setting = ' line:0' if _top(cue) else ''
            blocks.append(f'{_stamp(start, ".")} --> {_stamp(end, ".")}{setting}\n{lines}\n')
    return '\n'.join(blocks)


def _ass_stamp(seconds):
    centis = max(0, round(seconds * 100))
    hours, centis = divmod(centis, 360_000)
    minutes, centis = divmod(centis, 6000)
    secs, centis = divmod(centis, 100)
    return f'{hours:d}:{minutes:02d}:{secs:02d}.{centis:02d}'


# One readable default style at 1080p script resolution: white text with a black outline at the bottom centre,
# like the burned-in subtitles, and an italic Top style for song lyrics and signs. Players scale it to the video.
ASS_HEADER = """[Script Info]
ScriptType: v4.00+
PlayResX: 1920
PlayResY: 1080
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, \
Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, \
MarginV, Encoding
Style: Default,Noto Sans,64,&H00FFFFFF,&H000000FF,&H00000000,&H80000000,0,0,0,0,100,100,0,0,1,3.5,1.5,2,90,90,56,1
Style: Top,Noto Sans,58,&H00FFFFFF,&H000000FF,&H00000000,&H80000000,0,1,0,0,100,100,0,0,1,3.2,1.2,8,90,90,40,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
""".replace('\\\n', '')


def ass(cues, line_chars=42, spaced=True):
    """Advanced SubStation Alpha (styled subtitles, popular for anime) from (start, end, text) cues; song and sign
    cues (place 'top') use the italic Top style at the top of the picture."""
    events = []
    for cue in cues:
        start, end, text = cue[:3]
        if not text.strip():
            continue
        lines = [line.replace('{', '(').replace('}', ')').replace('\\', '/') for line in wrap(text, line_chars, spaced)]
        style = 'Top' if _top(cue) else 'Default'
        events.append(f'Dialogue: 0,{_ass_stamp(start)},{_ass_stamp(end)},{style},,0,0,0,,{"\\N".join(lines)}')
    return ASS_HEADER + '\n'.join(events) + ('\n' if events else '')


SENTENCE_END = ('.', '!', '?', '…', '。', '！', '？')
CLAUSE_END = (',', ';', ':', '–', '—', '，', '、', '；', '：')
# Readable narration subtitles (ai-voice-over v4 readable-lt-captions-v1). A cue starts when its words are spoken and
# stays on screen at least MIN_SECONDS, long enough to read (READING_CPS + READING_SETUP) and TAIL after its last
# word, up to MAX_TAIL when reading needs it, at most MAX_SECONDS, GAP before the next cue and never into the next
# narrator line. Pieces of one sentence at most MERGE_GAP apart become one cue when it fits two lines and
# MAX_SECONDS; different sentences and different lines (perhaps different speakers) never merge.
MIN_SECONDS = 1.2
READING_CPS = 15.0      # scripts that read slower per character use their own, lower rate (languages.reading_cps)
READING_SETUP = 0.2
TAIL = 0.5
MAX_TAIL = 1.5
MAX_SECONDS = 7.0
GAP = 0.08
MERGE_GAP = 0.55


def _pieces(text, max_chars, spaced):
    """Split one narrator line into subtitle-sized pieces: every sentence its own piece (one narrator line can hold
    two characters' sentences when recognition ran them together, and one subtitle would make them one speaker's;
    Bleach S01E01: "Rytoj tau atnešiu. Ačiū, kad juos išvijai."), then clauses and words of a long sentence. Pieces
    of one sentence are joined again by timing (_readable)."""
    text = ' '.join(text.split()) if spaced else ''.join(text.split())
    if not text:
        return []

    def split_after(source, marks):
        out, current = [], ''
        for i, ch in enumerate(source):
            current += ch
            following = source[i + 1] if i + 1 < len(source) else ''
            if ch in marks and (not spaced or following in ('', ' ')):
                out.append(current.strip())
                current = ''
        if current.strip():
            out.append(current.strip())
        return out

    pieces = []
    for sentence in split_after(text, SENTENCE_END):
        if len(sentence) <= max_chars:
            pieces.append(sentence)
            continue
        for clause in split_after(sentence, CLAUSE_END):
            while len(clause) > max_chars:      # no punctuation left: cut at the last word boundary that fits
                cut = clause.rfind(' ', 0, max_chars) if spaced else max_chars
                cut = cut if cut > 0 else max_chars
                pieces.append(clause[:cut].strip())
                clause = clause[cut:].strip()
            if clause:
                pieces.append(clause)
    return pieces


def merge(*cue_lists, gap=0.05):
    """One time-ordered cue list from several; a cue that would overlap the next one ends `gap` before it."""
    cues = sorted((c for cues in cue_lists for c in cues if c[2].strip()), key=lambda c: c[0])
    merged = []
    for i, (start, end, text) in enumerate(cues):
        if i + 1 < len(cues):
            end = min(end, cues[i + 1][0] - gap)
        if end > start:
            merged.append((start, round(end, 3), text))
    return merged


def narration_cues(placements, lines, line_chars=42, spaced=True, *, alignments=None, duration=None,
                   reading_cps=None):
    """Readable subtitles that show exactly what the narrator says, while it is said.

    `placements` carry unit_id, start and duration (after tempo), and optionally tempo and lead (seconds of the take's
    file before its speech); `lines` maps unit_id to spoken text. `alignments` maps unit_id to the take's character
    timestamps (tts.alignment), which time each piece of a long line; without them a line's time is shared out by
    character count. `duration` is the media length (cues may follow narration that runs past it, never more);
    `reading_cps` the language's reading rate, used when it is slower than READING_CPS.
    """
    ordered = sorted(placements, key=lambda p: p.start)
    rows = []
    for k, placement in enumerate(ordered):
        text = (lines.get(placement.unit_id) or '').strip()
        pieces = _pieces(text, 2 * line_chars, spaced)
        if not pieces:
            continue
        following = ordered[k + 1].start if k + 1 < len(ordered) else None
        timed = _timed(pieces, placement, (alignments or {}).get(placement.unit_id))
        rows += [{'start': start, 'end': end, 'text': piece, 'line': placement.unit_id, 'next_line': following}
                 for (start, end), piece in zip(timed, pieces)]
    return _readable(rows, line_chars, spaced, min(READING_CPS, reading_cps or READING_CPS), duration)


def _number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _char_times(pieces, timestamps):
    """[(start, end)] in the take's file for each non-space character of `pieces`, from the provider's character
    alignment (the original text first, then the normalised one), or None when neither matches the text."""
    wanted = [ch.casefold() for piece in pieces for ch in piece if not ch.isspace()]
    for key in ('alignment', 'normalized_alignment'):
        found = (timestamps or {}).get(key)
        if not isinstance(found, dict):
            continue
        chars, starts, ends = (found.get(name) for name in
                               ('characters', 'character_start_times_seconds', 'character_end_times_seconds'))
        if not (isinstance(chars, list) and isinstance(starts, list) and isinstance(ends, list)
                and len(chars) == len(starts) == len(ends)):
            continue
        rows = [(ch, a, b) for ch, a, b in zip(chars, starts, ends) if not (isinstance(ch, str) and ch.isspace())]
        if [ch.casefold() if isinstance(ch, str) else None for ch, _, _ in rows] != wanted:
            continue
        times = [(a, b) for _, a, b in rows]
        if (all(_number(a) and _number(b) and 0 <= a <= b for a, b in times)
                and all(x[0] <= y[0] for x, y in zip(times, times[1:]))):
            return times
    return None


def _timed(pieces, placement, timestamps):
    """(start, end) of each piece on the media timeline, inside the placement."""
    start, end = placement.start, placement.start + placement.duration
    times = _char_times(pieces, timestamps)
    if times:
        tempo = getattr(placement, 'tempo', 1.0) or 1.0
        lead = getattr(placement, 'lead', 0.0)

        def at(seconds):
            return min(max(start + (seconds - lead) / tempo, start), end)

        result, first = [], 0
        for piece in pieces:
            last = first + sum(1 for ch in piece if not ch.isspace()) - 1
            result.append((at(times[first][0]), at(times[last][1])))
            first = last + 1
        # Every piece must take time and start after the one before, or a cue could vanish (zero length).
        if all(b > a for a, b in result) and all(x[0] < y[0] for x, y in zip(result, result[1:])):
            return result
    total = sum(len(p) for p in pieces) or 1
    result, cursor = [], start
    for piece in pieces:
        length = placement.duration * len(piece) / total
        result.append((cursor, cursor + length))
        cursor += length
    return result


def _fits(text, line_chars, spaced):
    lines = wrap(text, line_chars, spaced)
    return len(lines) <= 2 and all(len(line) <= line_chars for line in lines)


def _readable(rows, line_chars, spaced, cps, duration):
    """Cues from timed pieces: neighbouring pieces of one line merged, display times made readable."""
    joiner = ' ' if spaced else ''
    merged = []
    for row in rows:
        last = merged[-1] if merged else None
        if (last and last['line'] == row['line'] and row['start'] - last['end'] <= MERGE_GAP
                and not last['text'].rstrip().endswith(SENTENCE_END) and row['end'] - last['start'] <= MAX_SECONDS
                and _fits(last['text'] + joiner + row['text'], line_chars, spaced)):
            last.update(text=last['text'] + joiner + row['text'], end=row['end'])
        else:
            merged.append(dict(row))
    cues = []
    for k, row in enumerate(merged):
        start, spoken = row['start'], row['end']
        wanted = max(MIN_SECONDS, len(row['text']) / cps + READING_SETUP, spoken - start + TAIL)
        following = merged[k + 1]['start'] if k + 1 < len(merged) else None
        ceiling = min(spoken + MAX_TAIL, max(spoken, start + MAX_SECONDS))
        if following is not None:
            ceiling = min(ceiling, following - GAP)
        if row['next_line'] is not None:
            ceiling = min(ceiling, row['next_line'])
        if duration is not None:
            ceiling = min(ceiling, max(duration, spoken))
        end = max(spoken, min(start + wanted, ceiling))   # dense speech: the words stay up while they are spoken
        if following is not None:
            end = min(end, following)
        cues.append((round(start, 3), round(end, 3), row['text']))
    return cues
