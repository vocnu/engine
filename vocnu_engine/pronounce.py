"""Round-trip check of how the narrator says the job's names, before the bulk synthesis.

A narrator voice can read a target-language name the source language's way when the two look alike: eleven_v4 reads
Lithuanian "Aliaska" as English "Alaska" in some takes and not in others, in any case form, which the owner heard at
once (2026-10-19). So, judged by speech recognition and never by guesswork:

1. Every glossary name the narration says (its rendering and case forms; not codes with spoken words) with a form
   spelled much like its source name (LOOKALIKE: "Aliaska" and "Alaska", not "Džeikobas" and "Jacob", which a
   Lithuanian reader says as the English do anyway) is heard in the lines that say it, at most HEARD_LINES: the
   shortest line of each of its forms first, then the shortest lines left. A few lines prove little: all five heard
   "Aliaska" as written, and the narration still said "Alaskos" elsewhere (2026-10-20). They are real takes: the
   bulk synthesis later finds them in the cache.
2. The probes are recognised again in the target language, each take in a request of its own (recognise). A form is
   heard where its words match best; it leaks when that is clearly closer to its source-language name (or that name
   with the form's ending, "Alaskos") than to the form itself. Spellings of the same sounds are one key: recognition
   writes a name it knows the source language's way, "Sarah" for the narrator's "Sara" and "Barnes" for "Barns"
   (2026-10-20: both counted as leaks, and "Sara" was respelled "Sąra"), so such a name is not checked at all, and a
   misheard word that is only a little closer to the source name ("tarbankso" for "Ferbankso") is no leak.
3. For a name with a leaking form the translation model proposes respellings of its rendering in the target alphabet
   (translate.respell). A respelling is a change to the stem ("Aliaska" → "Aljaska") that every form of the name
   takes alike ("Aliaskos" → "Aljaskos"), and it must keep every sound of the target language's reading: only the
   way a sound is marked may change (a soft consonant's i or j), never a letter that is another sound or a longer
   vowel ("Sąra" for "Sara") or another syllable ("Alijaska"). Each is synthesized in the name's first PROBES lines
   and recognised; one with no probe leaking and at least half of them heard as written (recognition sometimes drops
   a word: "Aljaska" came back right in two probes and as "Tai" in the third, 2026-10-19) is a fix, and the one heard
   closest to the written forms is kept: the narrator says it in every line, the subtitles keep the written name
   (spoken.py).

Best effort and bounded: a failed step leaves the names as written, and the report says what was heard.
"""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import logging
from pathlib import Path
import re
import unicodedata

import numpy as np

from . import media, spoken, translate, tts
from .costs import BudgetExceeded
from .errors import ProviderError
from .languages import get

HEARD_LINES = 24      # lines per name heard as the narrator says them
PROBES = 5            # of those, the lines each respelling is tried in
MAX_NAMES = 40        # names checked in one job
GAP = 1.2             # seconds of silence after each take recognised
RATE = 16000
RECOGNITION_PARALLEL = 8   # takes recognised at once (the live service runs up to 8 GPU recognition containers)
FOUND = 0.5           # a form matching the heard words less than this was not found at all (recognition missed it)
LOOKALIKE = 0.7       # a name is checked when a form's spelling is at least this similar to a source name's
LEAK_MATCH = 0.8      # a form leaks when the heard words match a source-language name at least this well...
LEAK_MARGIN = 0.08    # ...and by this much better than the form itself ("Alaskos" for "Aliaskos": 1.0 against 0.875)

log = logging.getLogger('vocnu.pronounce')

# Spellings of one sound in the Latin-script languages names come from: speech recognition writes a name it knows the
# way its source language does ("Sarah", "Barnes", "Thornton", "Faith" for the narrator's "Sara", "Barns", "Tornton",
# "Feit"), so a heard word that differs from the written one only in these was said as written.
_SAME_SOUND = ((re.compile(r'th'), 't'), (re.compile(r'ph'), 'f'), (re.compile(r'ck'), 'k'), (re.compile(r'w'), 'v'),
               (re.compile(r'(?<=[aeiou])h(?![a-z])'), ''),            # Sarah, Hannah: a final h after a vowel
               (re.compile(r'(?<=[b-df-hj-np-tv-z])es(?![a-z])'), 's'),  # Barnes, James: a silent e before a final s
               (re.compile(r'ai'), 'ei'))   # Faith, Craig: the English ai of the narrator's "Feit", "Kreigas"


def _soft(text, language):
    """Lithuanian marks a soft consonant with i or j before a vowel ("lia", "lja") and writes i after č, š, ž without a
    sound of its own: one spelling for each."""
    if language == 'lt':
        text = re.sub(r'(?<=[čšž])i(?=[aąeęėouųū])', '', text)
        text = re.sub(r'(?<=[bcdfghklmnprstvzž])j(?=[aąeęėouųū])', 'i', text)
    return text


def key(text, language=None):
    """A spelling-independent key for comparing what was heard with what was written: lowercase letters without
    diacritics, one spelling of a soft consonant (Lithuanian) and of a sound written several ways (_SAME_SOUND), doubled
    letters single."""
    text = _soft(unicodedata.normalize('NFC', str(text or '')).casefold(), language)
    text = ''.join(ch for ch in unicodedata.normalize('NFKD', text) if not unicodedata.combining(ch)).replace('y', 'i')
    for pattern, same in _SAME_SOUND:
        text = pattern.sub(same, text)
    text = re.sub(r'[^a-zа-яёα-ωіїєґ]|_', '', text)
    return re.sub(r'(.)\1+', r'\1', text)


def spelling(text, language=None):
    """The letters a reader of `language` says for `text`: lowercase, every mark kept (Lithuanian ą is a longer vowel
    than a), only the ways of writing one sound unified (`_soft`). A respelling must keep them."""
    text = _soft(unicodedata.normalize('NFC', str(text or '')).casefold(), language)
    return re.sub(r'[\W\d_]', '', text)


def _distance(a, b):
    row = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        previous, row[0] = row[0], i
        for j, y in enumerate(b, 1):
            previous, row[j] = row[j], min(row[j] + 1, row[j - 1] + 1, previous + (x != y))
    return row[-1]


def similarity(a, b):
    """1 for equal keys, 0 for nothing in common."""
    return 1.0 - _distance(a, b) / max(len(a), len(b), 1)


def _heard(words, form, language):
    """(similarity, heard text) of the window of heard `words` that best matches `form`."""
    target, size = key(form, language), len(form.split())
    best = (0.0, '')
    for width in range(max(1, size - 1), size + 2):
        for start in range(0, max(0, len(words) - width) + 1):
            text = ' '.join(words[start:start + width])
            score = similarity(key(text, language), target)
            if score > best[0]:
                best = (score, text)
    return best


def _verdict(words, form, rendering, sources, language):
    """(similarity to the form, whether it leaks, heard text) for one line's heard words: it leaks when they match a
    source-language name (LEAK_MATCH) clearly better than the form (LEAK_MARGIN), not when recognition only misheard
    a sound or two."""
    score, text = _heard(words, form, language)
    heard_key, form_key = key(text, language), key(form, language)
    leaks = any(key(other, language) != form_key
                and similarity(heard_key, key(other, language)) >= max(LEAK_MATCH, score + LEAK_MARGIN)
                for other in _leaked_forms(form, rendering, sources))
    return score, leaks, text


def _leaked_forms(form, rendering, sources):
    """Source-language names the narrator may say instead of `form`: each source, and each with the form's ending
    ("Aliaskos" from "Aliaska" → "Alaskos" from "Alaska")."""
    stem = 0
    while stem < min(len(form), len(rendering)) and form[stem].casefold() == rendering[stem].casefold():
        stem += 1
    ending, base_ending = form[stem:], rendering[stem:]
    result = set()
    for source in sources:
        result.add(source)
        if ending and len(source.split()) == 1:
            root = source[:-len(base_ending)] if base_ending and source.endswith(base_ending) else source
            result.add(root + ending)
    return result


def looks_alike(form, rendering, sources, language=None):
    """Whether `form` is spelled much like a source name (or one with the form's ending) without being it: the case
    in which a narrator voice reads the target-language name the source language's way."""
    own = key(form, language)
    return any(other != own and similarity(own, other) >= LOOKALIKE
               for other in (key(name, language) for name in _leaked_forms(form, rendering, sources)))


def names(glossary, language=None, kept=False):
    """[(rendering, sources, forms)] the check looks at: glossary names (not codes with spoken words) with forms of
    letters only that a key can compare, at least one of them spelled much like a source name; with `kept`, also the
    names written as in the source and declined (Galena, Galenoje), which the final check hears too. In a language
    whose names take cases, a name without case forms is not a word of that language but one kept as in the source
    (an aircraft's "Maid in Japan"), which the narrator rightly says the original way."""
    inflects = bool(language and get(language).cases)
    found = []
    for group in translate._name_groups([entry for entry in glossary or [] if not entry.get('spoken')]):
        if inflects and not group['forms']:
            continue
        forms = []
        for form in [group['target'], *group['forms']]:
            if (form not in forms and re.search(r'[^\W\d_]', form) and not re.search(r'\d', form)
                    and key(form, language)):
                forms.append(form)
        if (any(looks_alike(form, group['target'], group['sources'], language) for form in forms)
                or kept and forms and key(group['target'], language) in {key(s, language) for s in group['sources']}):
            found.append((group['target'], group['sources'], forms))
    return found


def variant(base, option, form):
    """`form` with the change that turns `base` into `option` (a change in the stem the forms share: "Aliaska" →
    "Aljaska" makes "Aliaskos" "Aljaskos"); None when `form` does not have that stem."""
    start = 0
    while start < min(len(base), len(option)) and base[start] == option[start]:
        start += 1
    end = 0
    while end < min(len(base), len(option)) - start and base[-1 - end] == option[-1 - end]:
        end += 1
    head = base[:len(base) - end]
    if not form.startswith(head):
        return None
    return option[:len(option) - end] + form[len(head):]


def _pattern(forms, spaced):
    alternatives = '|'.join(re.escape(form) for form in sorted(forms, key=len, reverse=True))
    return re.compile(rf'(?<!\w)(?:{alternatives})(?!\w)' if spaced else f'(?:{alternatives})')


def _probes(lines, groups, spaced):
    """{name index: [(unit id, form)]}: at most HEARD_LINES lines that say each name's forms (whole words), with the
    form each says: the shortest line of every form first, in the order of the forms (the rendering first), then the
    shortest lines left. One form heard right says little about the others (2026-10-20: "Aliaskos" was heard as
    written in both its probes and "Aliaskoje" in the third, yet the narration said "Alaska", "Alaską" and
    "Alaskoje")."""
    ordered = sorted(lines.items(), key=lambda item: (len(item[1]), str(item[0])))
    result = {}
    for k, (_, _, forms) in enumerate(groups):
        pattern, said = _pattern(forms, spaced), []
        for uid, text in ordered:
            match = pattern.search(text)
            if match:
                said.append((uid, match.group(0)))
        first = {}
        for uid, form in said:
            first.setdefault(form, (uid, form))
        found = [first[form] for form in forms if form in first][:HEARD_LINES]
        found += [probe for probe in said if probe not in found][:HEARD_LINES - len(found)]
        if found:
            result[k] = found
    return result


def recognise(paths, *, transcribe, language, work, name):
    """({key: [heard words]}, recognition seconds) for {key: mp3 path}: every take recognised in a request of its own,
    RECOGNITION_PARALLEL at a time.

    Never several takes in one request: Whisper decodes the speech of a 30-second window as one text, so a name said
    right in one take makes it hear that name as written in the next. Joined with silence, 45 eleven_v4 takes of
    "Galena" and "Aliaska" lines came back with every name as written, the narrator's two real slips ("Galimoje",
    "galimą") included; one take a request, the live service's GPU recognition heard exactly those two (2026-10-20).
    """
    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    silence = np.zeros(int(GAP * RATE), dtype=np.float32)

    def hear(path):
        pcm = np.concatenate([media.decode_array(path, channels=1, rate=RATE), silence])
        wav = media.write_wav(work / f'{name}-{Path(path).stem}.wav', pcm, rate=RATE)
        # Opus, as the job's own recognition gets it: small enough to ship to a GPU container.
        audio = media.asr_audio(wav, wav.with_suffix('.opus'), 0, len(pcm) / RATE)
        wav.unlink(missing_ok=True)
        result = transcribe(audio, language)
        return ([word['word'].strip() for word in result.get('words') or []],
                float(result.get('compute_seconds') or result.get('seconds') or 0.0))

    takes = list(dict.fromkeys(paths.values()))   # a take two lines share is heard once
    with ThreadPoolExecutor(max_workers=RECOGNITION_PARALLEL) as pool:
        results = dict(zip(takes, pool.map(hear, takes)))
    return {k: results[path][0] for k, path in paths.items()}, sum(seconds for _, seconds in results.values())


class Check:
    def __init__(self, lines, glossary, *, language, voice, cache, receipt, transcribe, table, work, source=None,
                 session=None):
        self.lines, self.language, self.voice, self.cache = dict(lines), language, voice, cache
        self.receipt, self.transcribe, self.table, self.work = receipt, transcribe, dict(table or {}), Path(work)
        self.source, self.session = source, session
        self.spaced = get(language).spaced
        self.groups = names(glossary, language)[:MAX_NAMES]
        self.report = {'names': 0, 'probes': 0, 'leaks': [], 'respelled': {}, 'heard': {}, 'tried': {}}
        self.asr_seconds = 0.0

    def _speak(self, texts, what):
        """{key: mp3 path} for {key: spoken text}, after a budget check."""
        self.receipt.check(tts.estimate_usd(texts.values()), what)
        return tts.synthesize_all(texts, self.language, self.voice, self.cache, self.receipt, session=self.session)

    def _recognise(self, paths, name):
        heard, seconds = recognise(paths, transcribe=self.transcribe, language=self.language, work=self.work,
                                   name=name)
        self.asr_seconds += seconds
        return heard

    def _score(self, words, form, rendering, sources):
        return _verdict(words, form, rendering, sources, self.language)

    def run(self):
        """{written form: spoken respelling} for the forms of names the narrator says the source-language way."""
        probes = _probes(self.lines, self.groups, self.spaced)
        if not probes:
            return {}
        ids = sorted({uid for found in probes.values() for uid, _ in found}, key=str)
        texts = {uid: spoken.spoken_text(self.lines[uid], self.table, self.spaced) for uid in ids}
        self.report.update(names=len(probes), probes=len(ids))
        heard = self._recognise(self._speak(texts, 'Name check'), 'names')
        leaking = {}
        for k, found in probes.items():
            rendering, sources, _ = self.groups[k]
            results = [self._score(heard[uid], form, rendering, sources) for uid, form in found]
            for (_, form), (_, _, text) in zip(found, results):
                self.report['heard'].setdefault(form, []).append(text)
            if any(leak for _, leak, _ in results):
                leaking[k] = results
        self.report['leaks'] = sorted(self.groups[k][0] for k in leaking)
        if not leaking:
            return {}
        log.info('names said the source-language way: %s', self.report['leaks'])
        return self._respell(leaking, probes)

    def _respell(self, leaking, probes):
        """Respellings proposed by the translation model that read the same as the written forms, each synthesized in
        the name's first PROBES lines and recognised; per name the one heard closest to its forms among those with no
        probe leaking and at least half of the probes heard."""
        names = list(leaking)
        asked = []
        for n, k in enumerate(names):
            rendering, sources, _ = self.groups[k]
            heard = next(text for (_, leak, text) in leaking[k] if leak)
            asked.append({'id': n, 'word': rendering, 'heard': heard, 'source': sources[0]})
        options = translate.respell(asked, source=self.source, target=self.language, receipt=self.receipt,
                                    session=self.session)
        candidates, texts = {}, {}
        for n, k in enumerate(names):
            rendering, _, forms = self.groups[k]
            for m, option in enumerate(options.get(n, [])):
                respelled = {form: variant(rendering, option, form) for form in forms}
                respelled = {form: said for form, said in respelled.items() if said and said != form}
                if (rendering not in respelled or any(spelling(said, self.language) != spelling(form, self.language)
                                                      for form, said in respelled.items())):
                    continue   # not the same sounds by the target language's spelling: another word, not a respelling
                candidates[n, m] = respelled
                for uid, _ in probes[k][:PROBES]:
                    texts[(n, m, uid)] = spoken.spoken_text(self.lines[uid], {**self.table, **respelled}, self.spaced)
        if not texts:
            return {}
        heard_again = self._recognise(self._speak(texts, 'Name respelling'), 'respellings')
        chosen = {}
        for n, k in enumerate(names):
            rendering, sources, _ = self.groups[k]
            best, choice = 0.0, None
            for m, option in enumerate(options.get(n, [])):
                if (n, m) not in candidates:
                    continue
                scored = [self._score(heard_again[(n, m, uid)], form, rendering, sources)
                          for uid, form in probes[k][:PROBES]]
                self.report['tried'].setdefault(rendering, {})[option] = [text for _, _, text in scored]
                found = [score for score, _, _ in scored if score >= FOUND]
                if any(leak for _, leak, _ in scored) or 2 * len(found) < len(scored):
                    continue
                mean = sum(found) / len(found)
                if mean > best:
                    best, choice = mean, candidates[n, m]
            if choice:
                chosen.update(choice)
        self.report['respelled'] = dict(chosen)
        return chosen


def respellings(lines, glossary, *, language, voice, cache, receipt, transcribe, table=None, work, source=None,
                session=None):
    """({written form: spoken respelling}, report, recognition seconds); ({}, report, seconds) when nothing leaks or
    the check cannot run."""
    check = Check(lines, glossary, language=language, voice=voice, cache=cache, receipt=receipt,
                  transcribe=transcribe, table=table, work=work, source=source, session=session)
    try:
        chosen = check.run() if transcribe and check.groups else {}
    except Exception as error:  # noqa: BLE001 - best effort: a failed check leaves the names as written
        log.warning('name check stopped: %s', error, exc_info=not isinstance(error, (BudgetExceeded, ProviderError)))
        check.report['stopped'] = 'budget' if isinstance(error, BudgetExceeded) else 'error'
        chosen = check.report.get('respelled') or {}
    return chosen, check.report, check.asr_seconds


# A name heard less like its written form than this was said wrong, or not at all. Whisper heard eleven_v4's two
# slips in 22 "Galena" lines as "Galimoje" (0.75) and "galimą" (0.67), the right takes at 0.86 or better (2026-10-20).
MISHEARD = 0.8


def heard_wrong(takes, glossary, *, language, transcribe, work, name='final'):
    """The final check: ({unit id: [(form, heard text)]} of takes whose names recognition heard clearly unlike the
    written form (MISHEARD) or as a source-language name (a leak), report, recognition seconds). `takes` is
    {unit id: (written line, mp3 path)}.

    A narrator voice slips now and then: eleven_v4 said "Galenoje" as "Galemoje" in one of 22 lines, heard so by
    three recognisers, and "Aliaskos" as "Alaskos" in one of 23 (2026-10-20); a respelling cannot help there. Every
    name the check looks at, and every name written as in the source (`names(kept=True)`), is heard in the finished
    takes, each take on its own (recognise); fitting gives the lines heard wrong another take. A name heard wrong in
    more than half of three or more lines is recognition's own trouble with it (Whisper hears a hard l in most of
    the "Aliaskos" said right), not the narrator's, and is left alone. Best effort: a failure flags nothing.
    """
    report = {'lines': 0, 'misheard': {}}
    try:
        groups = names(glossary, language, kept=True)
        if not groups or not transcribe or not takes:
            return {}, report, 0.0
        spaced = get(language).spaced
        said = {}
        for uid, (text, path) in takes.items():
            found = []
            for k, (_, _, forms) in enumerate(groups):
                found += [(k, match.group(0)) for match in _pattern(forms, spaced).finditer(text)]
            if found:
                said[uid] = found
        report['lines'] = len(said)
        if not said:
            return {}, report, 0.0
        heard, seconds = recognise({uid: takes[uid][1] for uid in said}, transcribe=transcribe, language=language,
                                   work=work, name=name)
        wrong, lines, missed = {}, Counter(), Counter()
        for uid, found in said.items():
            for k, form in found:
                lines[k] += 1
                rendering, sources, _ = groups[k]
                score, leaks, text = _verdict(heard[uid], form, rendering, sources, language)
                if score < MISHEARD or leaks:
                    missed[k] += 1
                    wrong.setdefault(uid, []).append((k, form, text))
        flagged = {}
        for uid, items in wrong.items():
            kept = [(form, text) for k, form, text in items if not (lines[k] >= 3 and 2 * missed[k] > lines[k])]
            if kept:
                flagged[uid] = kept
                for form, text in kept:
                    report['misheard'].setdefault(form, []).append(text)
        return flagged, report, seconds
    except Exception as error:  # noqa: BLE001 - best effort: the takes stay as they are
        log.warning('final name check stopped: %s', error, exc_info=not isinstance(error, ProviderError))
        report['stopped'] = 'error'
        return {}, report, 0.0
