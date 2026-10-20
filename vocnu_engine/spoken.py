"""What the narrator says for a written line, and the take's character timestamps mapped back to the written line.

Subtitles show the written line; the narrator may need other words for some of it (owner, 2026-10-19): a model or
code the glossary gives spoken words for ("C-46" is said "cė keturiasdešimt šeši") and a name the narrator voice
mispronounced as written, respelled after a round-trip check (pronounce.py: eleven_v4 said Lithuanian "Aliaska" as
English "Alaska", "Aljaska" it says the Lithuanian way). `table` holds {written form: spoken form}; `say` swaps whole
words only, longest form first, and `written_alignment` turns the provider's timestamps of the spoken text into
timestamps of the written one, so narration subtitles stay timed by the voice.
"""
import re

ALIGNMENT = 'alignment'


def table(glossary, respellings=None):
    """{written form: spoken form}: glossary entries' spoken words ('spoken', for their rendering) and `respellings`
    ({form: spoken}) on top."""
    result = {}
    for entry in glossary or []:
        written = ' '.join(str(entry.get('target') or '').split())
        said = ' '.join(str(entry.get('spoken') or '').split())
        if written and said and said != written:
            result[written] = said
    for written, said in (respellings or {}).items():
        if written and said and said != written:
            result[written] = said
    return result


def _pattern(forms, spaced):
    alternatives = '|'.join(re.escape(form) for form in sorted(forms, key=len, reverse=True))
    return re.compile(rf'(?<!\w)(?:{alternatives})(?!\w)' if spaced else f'(?:{alternatives})')


def say(text, forms, spaced=True):
    """(spoken text, [(written start, written end, spoken start, spoken end)] of each swapped span)."""
    text = str(text or '')
    if not forms or not text:
        return text, []
    pieces, spans, cursor, length = [], [], 0, 0
    for match in _pattern(forms, spaced).finditer(text):
        said = forms[match.group(0)]
        pieces.append(text[cursor:match.start()])
        length += match.start() - cursor
        spans.append((match.start(), match.end(), length, length + len(said)))
        pieces.append(said)
        length += len(said)
        cursor = match.end()
    pieces.append(text[cursor:])
    return ''.join(pieces), spans


def spoken_text(text, forms, spaced=True):
    return say(text, forms, spaced)[0]


def written_alignment(timestamps, text, forms, spaced=True):
    """The take's timestamps ({'alignment', 'normalized_alignment'}, tts.alignment) for the written `text`: unchanged
    characters keep their times, a swapped span shares out the time its spoken words took. Unchanged when nothing was
    swapped; None when the provider's characters are not the spoken text (subtitles then share time by length)."""
    said, spans = say(text, forms, spaced)
    if not spans or not isinstance(timestamps, dict):
        return timestamps
    found = timestamps.get(ALIGNMENT)
    if not isinstance(found, dict):
        return None
    chars, starts, ends = (found.get(key) for key in
                           ('characters', 'character_start_times_seconds', 'character_end_times_seconds'))
    if not (isinstance(chars, list) and isinstance(starts, list) and isinstance(ends, list)
            and len(chars) == len(starts) == len(ends) == len(said) and ''.join(map(str, chars)) == said):
        return None
    out_chars, out_starts, out_ends = [], [], []
    written, spoken = 0, 0
    for w_start, w_end, s_start, s_end in spans + [(len(text), len(text), len(said), len(said))]:
        while written < w_start:                        # unchanged characters, one for one
            out_chars.append(text[written])
            out_starts.append(starts[spoken])
            out_ends.append(ends[spoken])
            written, spoken = written + 1, spoken + 1
        if w_end > w_start:
            first, last = starts[s_start], ends[s_end - 1]
            step = (last - first) / (w_end - w_start)
            for k in range(w_end - w_start):
                out_chars.append(text[w_start + k])
                out_starts.append(round(first + k * step, 3))
                out_ends.append(round(first + (k + 1) * step, 3))
            written, spoken = w_end, s_end
    return {ALIGNMENT: {'characters': out_chars, 'character_start_times_seconds': out_starts,
                        'character_end_times_seconds': out_ends}, 'normalized_alignment': None}
