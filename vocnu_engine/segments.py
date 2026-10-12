"""Words → subtitle cues → voice-over units.

Cues are subtitle-sized (≤ 2 lines, ≤ 7 s), split at punctuation and pauses. Units are what the narrator
speaks in one take: the whole cues of one sentence, at most ~12 s, so a translation can be condensed per unit
while subtitles stay per cue. Separate sentences are never merged into one unit: short replies, calls and a
question with its answer are often different people (ai-voice-over v4 dropped that merging after the owner heard
replies of two characters run together). Recognition does not always mark where one speaker's line ends (Japanese
often comes without punctuation): `starts` are words where a new sentence or speaker begins (see translate.turns),
`joins` words that continue the sentence before them although recognition began a new segment there.
"""
from dataclasses import dataclass, field

from .languages import BY_CODE, from_provider

SENTENCE_END = ('.', '!', '?', '…', '。', '！', '？', '؟', '।', '۔')
SOFT_BREAK = (',', ';', ':', '，', '、', '；', '：', '—', '–')
PAUSE_SENTENCE = 1.0
MAX_CUE_SECONDS = 7.0
MAX_UNIT_SECONDS = 12.0
MIN_CUE_SECONDS = 1.0
CUE_GAP = 0.08


@dataclass
class Cue:
    id: int
    start: float
    end: float
    text: str


@dataclass
class Unit:
    id: int
    start: float
    end: float
    cue_ids: list = field(default_factory=list)
    text: str = ''
    words: list = field(default_factory=list, repr=False, compare=False)   # the recognised words it was built from

    @property
    def seconds(self):
        return self.end - self.start


def _text(words):
    return ''.join(w['word'] for w in words).strip()


def _limits(language):
    lang = BY_CODE.get(from_provider(language or ''))
    line_chars = lang.line_chars if lang else 42
    reading_cps = lang.reading_cps if lang else 17.0
    return 2 * line_chars, reading_cps


def turn_key(word):
    """How `starts` and `joins` name a word: the word itself (recognition can give neighbouring words one time)."""
    return id(word)


def sentences(words, starts=frozenset(), joins=frozenset()):
    """Split at sentence-final punctuation, a pause longer than PAUSE_SENTENCE, a new recognition segment or a word in
    `starts` (turn_key of words that begin a new sentence or speaker); a word in `joins` continues its sentence
    unless the word before it ends one."""
    result, current = [], []
    for i, word in enumerate(words):
        current.append(word)
        nxt = words[i + 1] if i + 1 < len(words) else None
        ends = word['word'].strip().endswith(SENTENCE_END)
        pause = nxt is not None and nxt['start'] - word['end'] > PAUSE_SENTENCE
        turn = nxt is not None and turn_key(nxt) in starts
        join = nxt is not None and turn_key(nxt) in joins
        if nxt is None or ends or ((pause or nxt.get('segment_start')) and not join) or turn:
            result.append(current)
            current = []
    return result


def _split_score(words, i):
    """Lower is better: cut before words[i]."""
    left, right = _text(words[:i]), _text(words[i:])
    gap = max(0.0, words[i]['start'] - words[i - 1]['end'])
    previous = words[i - 1]['word'].strip()
    score = abs(len(left) - len(right)) / max(1, len(left) + len(right))
    if previous.endswith(SENTENCE_END):
        score -= 0.6
    elif previous.endswith(SOFT_BREAK):
        score -= 0.35
    return score - min(gap, 1.0) * 0.5


def split_words(words, max_chars, max_seconds):
    """Recursively cut a run of words into pieces within the character and duration limits."""
    if len(words) <= 1:
        return [words]
    fits = len(_text(words)) <= max_chars and words[-1]['end'] - words[0]['start'] <= max_seconds
    if fits:
        return [words]
    best = min(range(1, len(words)), key=lambda i: _split_score(words, i))
    return split_words(words[:best], max_chars, max_seconds) + split_words(words[best:], max_chars, max_seconds)


def _display_end(cue_start, cue_end, chars, reading_cps, next_start):
    wanted = max(cue_end, cue_start + max(MIN_CUE_SECONDS, chars / reading_cps))
    wanted = min(wanted, cue_start + MAX_CUE_SECONDS)
    if next_start is not None:
        wanted = min(wanted, next_start - CUE_GAP)
    return max(cue_end if next_start is None else min(cue_end, next_start - CUE_GAP), wanted)


def build(words, language, starts=frozenset(), joins=frozenset()):
    """(cues, units) for the source transcript; `starts` and `joins` see `sentences`."""
    words = [w for w in words if w['word'].strip()]
    if not words:
        return [], []
    max_chars, reading_cps = _limits(language)
    pieces, sentence_of = [], []
    for s, sentence in enumerate(sentences(words, starts, joins)):
        for piece in split_words(sentence, max_chars, MAX_CUE_SECONDS):
            pieces.append(piece)
            sentence_of.append(s)
    cues = []
    for i, piece in enumerate(pieces):
        nxt = pieces[i + 1][0]['start'] if i + 1 < len(pieces) else None
        text = _text(piece)
        end = _display_end(piece[0]['start'], piece[-1]['end'], len(text), reading_cps, nxt)
        cues.append(Cue(i + 1, round(piece[0]['start'], 3), round(end, 3), text))
    # Units: the cues of one sentence, chunked to MAX_UNIT_SECONDS of speech.
    groups = []
    for cue, piece, s in zip(cues, pieces, sentence_of):
        speech_end = piece[-1]['end']
        if groups:
            last = groups[-1]
            fits = speech_end - last['start'] <= MAX_UNIT_SECONDS
            if last['sentence'] == s and fits:
                last['cues'].append(cue)
                last['words'] += piece
                last['speech_end'] = speech_end
                last['sentence'] = s
                continue
        groups.append({'start': piece[0]['start'], 'speech_end': speech_end, 'cues': [cue], 'sentence': s,
                       'words': list(piece)})
    units = [Unit(i + 1, round(g['start'], 3), round(g['speech_end'], 3), [c.id for c in g['cues']],
                  ' '.join(c.text for c in g['cues']) if _spaced(language) else ''.join(c.text for c in g['cues']),
                  g['words'])
             for i, g in enumerate(groups)]
    return cues, units


def _spaced(language):
    lang = BY_CODE.get(from_provider(language or ''))
    return lang.spaced if lang else True


def with_lines(cues, units, lines):
    """(cues, units, added ids): `lines` [(start, end, text)] (dialogue the upload's own subtitles show where speech
    recognition heard nothing) joined as one cue and one unit each, everything renumbered in time order."""
    by_id = {c.id: c for c in cues}
    groups = [(u.start, u.end, [by_id[c] for c in u.cue_ids], u.text, False) for u in units]
    groups += [(start, end, [Cue(0, start, end, text)], text, True) for start, end, text in lines]
    new_cues, new_units, added = [], [], set()
    for start, end, group_cues, text, extra in sorted(groups, key=lambda g: (g[0], g[1])):
        ids = []
        for cue in group_cues:
            new_cues.append(Cue(len(new_cues) + 1, cue.start, cue.end, cue.text))
            ids.append(len(new_cues))
        new_units.append(Unit(len(new_units) + 1, start, end, ids, text))
        if extra:
            added.add(len(new_units))
    return new_cues, new_units, added
