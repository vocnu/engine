"""Narrator takes that fit their timing budgets: synthesize every line at natural speed, measure it, then repair what
does not fit (ai-voice-over v4 finish.finalize with natural_retakes and causal_timing, simplified).

Up to ROUNDS times, timing.targets names the lines to shorten: the cause of a violation, not a short reply that a long
line pushed late. A short line (a call, a name, a "no") cannot be rephrased faithfully, so it first gets same-text
retakes with the next seed (+1, then +2 in a later round); any other line is rephrased by translate.shorten, which sees
the measured problem and every wording already tried with its measured length. A new take replaces the current one
only when it plays measurably shorter (MIN_GAIN) and the whole plan's timing score improves, so a shorter text with a
longer take never wins. What still does not fit after the rounds is sped up, at most MAX_TEMPO (v4 never sped up; the
owner once accepted this much), and reported. Speech is never cut.

A take that comes back empty or silent is retried once with the next seed; if it is silent again, the line is left out
(missing_speech) and the original audio plays there.

Repairs are optional: when the job budget or a provider failure stops them, the job goes on with the takes it has and
the report says so. The first synthesis is not optional; its budget check fails the job.
"""
from collections import Counter
from dataclasses import dataclass, replace
import logging
import math

from . import mix, timing, translate, tts
from .costs import BudgetExceeded
from .errors import ProviderError
from .languages import get

ROUNDS = 3
MAX_RETAKES = 2        # same-text alternates per line, for timing
MIN_GAIN = 0.01        # seconds a new take must save to replace the current one
MAX_TEMPO = 1.08       # last resort only
SPEED_UP_PASSES = 6    # each pass speeds up the next cause of what still does not fit
SHORT_CHARS = 24       # short lines: at most this many characters...
SHORT_WORDS = 3        # ...or words
CONTEXT_UNITS = 2      # source units before and after a line that the rephrasing sees

log = logging.getLogger('vocnu.fitting')


@dataclass(frozen=True)
class Take:
    text: str
    path: object
    seconds: float       # natural length of the speech, silence trimmed
    lead: float = 0.0    # where that speech starts in the file
    tempo: float = 1.0

    @property
    def played(self):
        return self.seconds / self.tempo


@dataclass
class Narration:
    lines: dict          # unit_id → what the narrator says (after any rephrasing)
    paths: dict          # unit_id → audio file of the kept take
    seconds: dict        # unit_id → natural length of the kept take
    placements: list     # timing.Placement in time order, with tempo and lead
    violations: list     # timing.Violation left after fitting
    missing: list        # unit ids whose takes stayed silent (not narrated)
    report: dict         # timing stats for the job


def short(text, spaced=True):
    """A call, name or reply too short to rephrase faithfully (word counts mean nothing without spaces)."""
    return len(text) <= SHORT_CHARS or (spaced and len(text.split()) <= SHORT_WORDS)


def narrate(units, lines, *, media_duration, language, voice, cache, receipt, cues=(), speakers=None, glossary=None,
            on_done=None):
    """Synthesize `lines` ({unit_id: text}) for `units` and fit them into their timing budgets → Narration.

    `cues` are the source cues (for cost estimates), `speakers` the likely speaker gender per unit and `glossary` the
    job's names and terms, both passed on to rephrasing. `on_done(unit_id)` follows the first synthesis.
    """
    fitter = _Fitter(units, media_duration, language, voice, cache, receipt, cues, speakers or {}, glossary)
    fitter.synthesize(lines, on_done)
    fitter.fit()
    fitter.speed_up()
    return fitter.result()


class _Fitter:
    def __init__(self, units, media_duration, language, voice, cache, receipt, cues, speakers, glossary):
        self.units = list(units)
        self.by_id = {u.id: u for u in self.units}
        self.index = {u.id: i for i, u in enumerate(self.units)}
        self.media_duration = media_duration
        self.language, self.voice, self.cache, self.receipt = language, voice, cache, receipt
        self.cues, self.speakers, self.glossary = list(cues), speakers, glossary
        self.spaced = get(language).spaced
        self.takes = {}            # unit_id → kept Take
        self.tried = {}            # unit_id → {wording: measured seconds, None when silent}
        self.offsets = {}          # (unit_id, wording) → highest seed offset asked for
        self.retakes = Counter()   # unit_id → same-text retakes asked for timing
        self.counts = Counter()
        self.missing = []
        self.stopped = None
        self.budgets = {}
        self.violations = []

    def _ordered(self, ids):
        return sorted(ids, key=self.index.__getitem__)

    def _speak(self, texts, offsets=None, what='Voice-over retakes', on_done=None):
        """{unit_id: Take, or None when silent} for new takes of `texts`, after a budget check."""
        offsets = offsets or {}
        self.receipt.check(tts.estimate_usd(texts.values()), what)
        paths = tts.synthesize_all(texts, self.language, self.voice, self.cache, self.receipt,
                                   seeds={uid: tts.SEED + offset for uid, offset in offsets.items()}, on_done=on_done)
        self.counts['takes'] += len(paths)
        takes = {}
        for uid, path in paths.items():
            text, offset = texts[uid], offsets.get(uid, 0)
            self.offsets[uid, text] = max(self.offsets.get((uid, text), 0), offset)
            found = mix.measure(path)
            takes[uid] = None if found.silent else Take(text, path, found.seconds, found.lead)
        return takes

    def synthesize(self, lines, on_done=None):
        """First takes of every line; a silent one is retried once with the next seed, then left out."""
        texts = {uid: text for uid, text in lines.items() if uid in self.by_id and text and text.strip()}
        takes = self._speak(texts, what='Voice-over speech', on_done=on_done)
        silent = {uid: texts[uid] for uid in texts if takes.get(uid) is None}
        if silent:
            log.warning('silent takes for %d lines; trying the next seed', len(silent))
            try:
                takes.update(self._speak(silent, {uid: 1 for uid in silent}))
            except (BudgetExceeded, ProviderError) as error:
                self._stop(error)
        for uid in self._ordered(texts):
            take = takes.get(uid)
            if take is None:
                self.missing.append(uid)
            else:
                self.takes[uid] = take
                self.tried[uid] = {take.text: take.seconds}
        if self.missing:
            log.warning('no speech for units %s; the original audio plays there', self.missing)
        self.budgets = timing.budgets(self.units, self.media_duration, narrated=set(self.takes))
        self.violations = self._check(self.takes)

    def _place(self, takes):
        return timing.place(self.budgets, {uid: t.played for uid, t in takes.items()},
                            {uid: t.tempo for uid, t in takes.items()})

    def _check(self, takes):
        return timing.inspect(self._place(takes), self.budgets, self.media_duration)

    def _targets(self, fixed=()):
        return timing.targets(self._place(self.takes), self.budgets, {uid: t.text for uid, t in self.takes.items()},
                              self.violations, fixed)

    def _keep(self, uid, take):
        """Keep `take` when it plays measurably shorter and the plan's timing improves without hurting another line."""
        if take is None or take.played > self.takes[uid].played - MIN_GAIN:
            return False
        trial = {**self.takes, uid: take}
        after = self._check(trial)
        if (timing.score(after) < timing.score(self.violations)
                and {v.unit_id for v in after} <= {v.unit_id for v in self.violations}):
            self.takes, self.violations = trial, after
            return True
        return False

    def fit(self):
        """Up to ROUNDS rounds of retakes (short lines) and rephrasing (the others) at the causes of violations."""
        for _ in range(ROUNDS):
            if not self.violations or self.stopped:
                return
            targets = self._targets()
            if not targets:
                return
            self.counts['rounds'] += 1
            takes = self.counts['takes']
            retake = [t for t in targets if self._retakeable(t.unit_id)]
            if self._retake(retake):
                targets = self._targets()
            asked = {t.unit_id for t in retake}
            self._rephrase([t for t in targets if t.unit_id not in asked])
            if self.counts['takes'] == takes:
                return   # nothing new to try: another round would only repeat the same request

    def _retakeable(self, uid):
        return short(self.takes[uid].text, self.spaced) and self.retakes[uid] < MAX_RETAKES

    def _retake(self, targets):
        """Same text, next seed; True when a retake was kept."""
        if not targets or self.stopped:
            return False
        texts = {t.unit_id: self.takes[t.unit_id].text for t in targets}
        offsets = {uid: self.offsets.get((uid, text), 0) + 1 for uid, text in texts.items()}
        try:
            takes = self._speak(texts, offsets)
        except (BudgetExceeded, ProviderError) as error:
            self._stop(error)
            return False
        kept = False
        for uid in self._ordered(texts):
            self.retakes[uid] += 1
            self.counts['retakes'] += 1
            if self._keep(uid, takes.get(uid)):
                self.counts['retakes_kept'] += 1
                kept = True
        return kept

    def _rephrase(self, targets):
        """New wordings from translate.shorten, synthesized; each kept only when its take pays off."""
        if not targets or self.stopped:
            return
        try:
            self.receipt.check(translate.estimate_usd([self.by_id[t.unit_id] for t in targets], self.cues) / 2,
                               'Shortening')
            answer = translate.shorten([self._request(t) for t in targets], target=self.language,
                                       receipt=self.receipt, glossary=self.glossary)
        except (BudgetExceeded, ProviderError) as error:
            self._stop(error)
            return
        self.counts['rephrase_rounds'] += 1
        answer = answer if isinstance(answer, dict) else {}
        texts = {}
        for t in targets:
            text = ' '.join(str(answer.get(t.unit_id) or '').split())
            if text and text not in self.tried[t.unit_id]:   # the same wording is never bought twice
                texts[t.unit_id] = text
        if not texts:
            return
        try:
            takes = self._speak(texts)
        except (BudgetExceeded, ProviderError) as error:
            self._stop(error)
            return
        for uid in self._ordered(texts):
            take = takes.get(uid)
            self.tried[uid][texts[uid]] = take.seconds if take else None
            if self._keep(uid, take):
                self.counts['rephrased_lines'] += 1

    def _request(self, target):
        """One line for translate.shorten: the measured problem, its context and every wording tried so far."""
        uid = target.unit_id
        take, unit, i = self.takes[uid], self.by_id[uid], self.index[uid]
        share = min(1.0, target.room / take.played) if take.played > 0 else 1.0
        problem = f'take {take.played:.1f} s, room {target.room:.1f} s'
        if target.blocked is not None:
            problem += '; it delays the lines after it'
        return {'id': uid, 'text': take.text, 'max_chars': max(6, math.floor(len(take.text) * share * 0.95)),
                'source': unit.text,
                'context_before': [u.text for u in self.units[max(0, i - CONTEXT_UNITS):i]],
                'context_after': [u.text for u in self.units[i + 1:i + 1 + CONTEXT_UNITS]],
                'speaker': self.speakers.get(uid),
                'previous': [{'text': text, 'seconds': round(seconds, 2)}
                             for text, seconds in self.tried[uid].items() if seconds is not None],
                'problem': problem}

    def speed_up(self):
        """Last resort: a cause that still does not fit plays faster, at most MAX_TEMPO (free, so also after a stop).
        A cause already at MAX_TEMPO is passed over, so the next line of the chain can help."""
        for _ in range(SPEED_UP_PASSES):
            if not self.violations:
                return
            kept = False
            for target in self._targets(fixed={uid for uid, t in self.takes.items() if t.tempo >= MAX_TEMPO}):
                take = self.takes[target.unit_id]
                tempo = round(min(MAX_TEMPO, take.seconds / target.room) if target.room > 0 else MAX_TEMPO, 3)
                if tempo > take.tempo and self._keep(target.unit_id, replace(take, tempo=tempo)):
                    kept = True
            if not kept:
                return

    def _stop(self, error):
        self.stopped = 'budget' if isinstance(error, BudgetExceeded) else 'provider_error'
        log.warning('timing repairs stopped: %s', error)

    def result(self):
        placements = [replace(p, lead=self.takes[p.unit_id].lead) for p in self._place(self.takes)]
        kinds = Counter(kind for v in self.violations for kind in v.kinds)
        tempos = [t.tempo for t in self.takes.values() if t.tempo > 1.0]
        report = {'violations': {kind: kinds[kind] for kind in timing.KINDS},
                  'lines_off_budget': len(self.violations),
                  'seconds_off_budget': round(sum(v.late + v.over for v in self.violations), 2),
                  'rounds': self.counts['rounds'], 'retakes': self.counts['retakes'],
                  'retakes_kept': self.counts['retakes_kept'], 'rephrase_rounds': self.counts['rephrase_rounds'],
                  'rephrased_lines': self.counts['rephrased_lines'], 'sped_up_lines': len(tempos),
                  'max_tempo': max(tempos, default=1.0), 'missing_speech': len(self.missing)}
        if self.stopped:
            report['stopped'] = self.stopped
        log.info('narration timing: %s', report)
        return Narration(lines={uid: t.text for uid, t in self.takes.items()},
                         paths={uid: t.path for uid, t in self.takes.items()},
                         seconds={uid: t.seconds for uid, t in self.takes.items()}, placements=placements,
                         violations=list(self.violations), missing=list(self.missing), report=report)
