"""Narrator takes that fit their timing budgets: synthesize every line at natural speed, measure it, then repair what
does not fit (ai-voice-over v4 finish.finalize with natural_retakes and causal_timing, simplified).

Up to ROUNDS times, timing.targets names the lines to shorten: the cause of a violation, not a short reply that a long
line pushed late. A short line (a call, a name, a "no") cannot be rephrased faithfully, so it first gets same-text
retakes with the next seed (+1, then +2 in a later round); any other line is rephrased by translate.shorten, which sees
the measured problem and every wording already tried with its measured length. A new take replaces the current one
only when it plays measurably shorter (MIN_GAIN) and the whole plan's timing score improves, so a shorter text with a
longer take never wins. What still does not fit after the rounds is sped up, at most MAX_TEMPO (v4 never sped up; the
owner once accepted this much), and reported. Speech is never cut.

No line is shortened before its first take, although a line bought twice is paid twice: rephrasing the lines whose
predicted length did not fit before any speech was bought, and speeding up overruns under 5 % before the repairs,
bought a 44-minute documentary a third less speech but left 284 lines and 395 s off budget, against 206 lines and
189 s in this order (2026-10-20). Once a line had been cut, the measured repairs rarely found a shorter wording (21
of the 130 lines asked in the first round), and a take already sped up left a new wording less to win.

The narrator may say other words than the written line (`spoken`, spoken.py: a code's spoken words, a respelled
name); takes are bought for the spoken text and keep the written one. `respell`, when given, checks how the narrator
says the lines' names before the synthesis and adds respellings (pronounce.py).

A take that comes back empty or silent is retried once with the next seed; if it is silent again, the line is left out
(missing_speech) and the original audio plays there.

Repairs are optional: when the job budget or a provider failure stops them, the job goes on with the takes it has and
the report says so. The first synthesis is not optional; its budget check fails the job.
"""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
import logging
import math
import time

from . import mix, spoken as speech, timing, translate, tts
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
MEASURE_PARALLEL = 6   # takes decoded and measured at once (the job container has 4 cores)

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
    lines: dict          # unit_id → what the narrator says, as written (after any rephrasing)
    paths: dict          # unit_id → audio file of the kept take
    seconds: dict        # unit_id → natural length of the kept take
    placements: list     # timing.Placement in time order, with tempo and lead
    violations: list     # timing.Violation left after fitting
    missing: list        # unit ids whose takes stayed silent (not narrated)
    report: dict         # timing stats for the job
    spoken: dict = None  # written form → what the narrator says instead (spoken.py)


def short(text, spaced=True):
    """A call, name or reply too short to rephrase faithfully (word counts mean nothing without spaces)."""
    return len(text) <= SHORT_CHARS or (spaced and len(text.split()) <= SHORT_WORDS)


def narrate(units, lines, *, media_duration, language, voice, cache, receipt, cues=(), speakers=None, glossary=None,
            on_done=None, spoken=None, respell=None, verify=None):
    """Synthesize `lines` ({unit_id: text}) for `units` and fit them into their timing budgets → Narration.

    `cues` are the source cues (for cost estimates), `speakers` the likely speaker gender per unit and `glossary` the
    job's names and terms, both passed on to rephrasing. `spoken` ({written form: spoken form}) is what the narrator
    says instead of a written form; `respell(lines, spoken)` returns more of them for the lines, before any other
    speech is bought. `verify({unit_id: (line, path)})` names the finished takes whose names were heard wrong
    (pronounce.heard_wrong); each gets one more take. `on_done(unit_id)` follows the first synthesis.
    """
    fitter = _Fitter(units, media_duration, language, voice, cache, receipt, cues, speakers or {}, glossary, spoken)
    lines = {uid: text for uid, text in lines.items() if uid in fitter.by_id and text and text.strip()}
    if respell and lines:
        started = time.monotonic()
        fitter.spoken.update(respell(dict(lines), dict(fitter.spoken)) or {})
        fitter.seconds['name_check'] += time.monotonic() - started
    fitter.synthesize(lines, on_done)
    fitter.fit()
    fitter.speed_up()
    fitter.verify(verify)
    return fitter.result()


class _Fitter:
    def __init__(self, units, media_duration, language, voice, cache, receipt, cues, speakers, glossary, spoken=None):
        self.units = list(units)
        self.by_id = {u.id: u for u in self.units}
        self.index = {u.id: i for i, u in enumerate(self.units)}
        self.media_duration = media_duration
        self.language, self.voice, self.cache, self.receipt = language, voice, cache, receipt
        self.cues, self.speakers, self.glossary = list(cues), speakers, glossary
        self.spaced = get(language).spaced
        self.spoken = dict(spoken or {})
        self.takes = {}            # unit_id → kept Take
        self.tried = {}            # unit_id → {wording: measured seconds, None when silent}
        self.offsets = {}          # (unit_id, wording) → highest seed offset asked for
        self.retakes = Counter()   # unit_id → same-text retakes asked for timing
        self.counts = Counter()
        self.seconds = Counter()   # where the time went: first takes, retakes, shortening requests, new wordings
        self.missing = []
        self.stopped = None
        self.budgets = {}
        self.violations = []

    def _ordered(self, ids):
        return sorted(ids, key=self.index.__getitem__)

    def _say(self, text):
        """What the narrator is given for a written line."""
        return speech.spoken_text(text, self.spoken, self.spaced) if self.spoken else text

    def _speak(self, texts, offsets=None, what='Voice-over retakes', on_done=None):
        """{unit_id: Take, or None when silent} for new takes of `texts`, after a budget check."""
        offsets = offsets or {}
        said = {uid: self._say(text) for uid, text in texts.items()}
        self.receipt.check(tts.estimate_usd(said.values()), what)
        paths = tts.synthesize_all(said, self.language, self.voice, self.cache, self.receipt,
                                   seeds={uid: tts.SEED + offset for uid, offset in offsets.items()}, on_done=on_done)
        self.counts['takes'] += len(paths)
        # Each take is decoded by its own ffmpeg process; one after another they took about a minute of a 44-minute
        # documentary's 758 first takes (2026-10-20).
        with ThreadPoolExecutor(MEASURE_PARALLEL) as pool:
            found = dict(zip(paths, pool.map(mix.measure, paths.values())))
        takes = {}
        for uid in paths:
            text, offset = texts[uid], offsets.get(uid, 0)
            self.offsets[uid, text] = max(self.offsets.get((uid, text), 0), offset)
            takes[uid] = None if found[uid].silent else Take(text, paths[uid], found[uid].seconds, found[uid].lead)
        return takes

    def synthesize(self, texts, on_done=None):
        """First takes of every line; a silent one is retried once with the next seed, then left out."""
        started = time.monotonic()
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
        self.seconds['first_takes'] += time.monotonic() - started

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
        """Up to ROUNDS rounds of retakes (short lines) and rephrasing (the others) at the causes of violations.

        A round's shortening request (one to four minutes for a documentary's repair round, 2026-10-20) runs while its
        short lines are retaken: it reads the takes as they were, and its wordings are tried afterwards, for the lines
        that still need them once the retakes are kept.
        """
        for _ in range(ROUNDS):
            if not self.violations or self.stopped:
                return
            targets = self._targets()
            if not targets:
                return
            self.counts['rounds'] += 1
            takes = self.counts['takes']
            retake = [t for t in targets if self._retakeable(t.unit_id)]
            asked = {t.unit_id for t in retake}
            rephrase = [t for t in targets if t.unit_id not in asked]
            with ThreadPoolExecutor(max_workers=1) as pool:
                asking = self._ask(rephrase, pool)
                if self._retake(retake) and self.violations:
                    still = {t.unit_id for t in self._targets()}
                    rephrase = [t for t in rephrase if t.unit_id in still]
                waiting = time.monotonic()
                answer = self._answer(asking)
                self.seconds['waiting_for_shortening'] += time.monotonic() - waiting
            self._try(answer, rephrase if self.violations else [])
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
        started = time.monotonic()
        try:
            takes = self._speak(texts, offsets)
        except (BudgetExceeded, ProviderError) as error:
            self._stop(error)
            return False
        finally:
            self.seconds['retakes'] += time.monotonic() - started
        kept = False
        for uid in self._ordered(texts):
            self.retakes[uid] += 1
            self.counts['retakes'] += 1
            if self._keep(uid, takes.get(uid)):
                self.counts['retakes_kept'] += 1
                kept = True
        return kept

    def _ask(self, targets, pool):
        """The shortening request for `targets`, started on `pool` (built here, from the takes as they are now); None
        when there is nothing to ask or the budget stops it."""
        if not targets or self.stopped:
            return None
        try:
            self.receipt.check(translate.estimate_usd([self.by_id[t.unit_id] for t in targets], self.cues) / 2,
                               'Shortening')
        except BudgetExceeded as error:
            self._stop(error)
            return None
        requests = [self._request(t) for t in targets]

        def ask():
            started = time.monotonic()
            try:
                return translate.shorten(requests, target=self.language, receipt=self.receipt,
                                         glossary=self.glossary)
            finally:
                self.seconds['shortening'] += time.monotonic() - started
        return pool.submit(ask)

    def _answer(self, asking):
        """{unit_id: wording} the shortening request answered; None when it failed or was not asked."""
        if asking is None:
            return None
        try:
            answer = asking.result()
        except (BudgetExceeded, ProviderError) as error:
            self._stop(error)
            return None
        self.counts['rephrase_rounds'] += 1
        return answer if isinstance(answer, dict) else {}

    def _try(self, answer, targets):
        """The new wordings of `answer` for `targets`, synthesized; each kept only when its take pays off."""
        if not answer or not targets or self.stopped:
            return
        texts = {}
        for t in targets:
            text = ' '.join(str(answer.get(t.unit_id) or '').split())
            if text and text not in self.tried[t.unit_id]:   # the same wording is never bought twice
                texts[t.unit_id] = text
        if not texts:
            return
        started = time.monotonic()
        try:
            takes = self._speak(texts)
        except (BudgetExceeded, ProviderError) as error:
            self._stop(error)
            return
        finally:
            self.seconds['new_wordings'] += time.monotonic() - started
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
        # The share applies to what the narrator says; a code written short and said long keeps its spoken words.
        said = self._say(take.text)
        chars = math.floor(len(said) * share * 0.95) - (len(said) - len(take.text))
        return {'id': uid, 'text': take.text, 'max_chars': max(6, chars),
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

    def verify(self, check):
        """Finished takes whose names `check` heard wrong get one same-text take with the next seed, played at the same
        tempo; it is kept when its names are heard as written and the timing is no worse (optional, like repairs)."""
        if not check or not self.takes or self.stopped:
            return
        started = time.monotonic()
        try:
            flagged = check({uid: (take.text, take.path) for uid, take in self.takes.items()})
            if not flagged:
                return
            self.counts['names_misheard'] += len(flagged)
            texts = {uid: self.takes[uid].text for uid in flagged if uid in self.takes}
            offsets = {uid: self.offsets.get((uid, text), 0) + 1 for uid, text in texts.items()}
            takes = self._speak(texts, offsets, what='Name retakes')
            new = {uid: replace(take, tempo=self.takes[uid].tempo) for uid, take in takes.items() if take is not None}
            still = check({uid: (take.text, take.path) for uid, take in new.items()}) if new else {}
            for uid in self._ordered(new):
                if uid in still:
                    continue
                trial = {**self.takes, uid: new[uid]}
                after = self._check(trial)
                (lines, late), (lines_before, late_before) = timing.score(after), timing.score(self.violations)
                if (lines, late) <= (lines_before, late_before + 1e-6) and \
                        {v.unit_id for v in after} <= {v.unit_id for v in self.violations}:
                    self.takes, self.violations = trial, after
                    self.counts['names_retaken'] += 1
        except (BudgetExceeded, ProviderError) as error:
            log.warning('name retakes stopped: %s', error)
        finally:
            self.seconds['final_name_check'] += time.monotonic() - started

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
                  'max_tempo': max(tempos, default=1.0), 'missing_speech': len(self.missing),
                  'takes': self.counts['takes'], 'names_misheard': self.counts['names_misheard'],
                  'names_retaken': self.counts['names_retaken'],
                  'seconds': {k: round(v, 1) for k, v in self.seconds.items()}}
        if self.stopped:
            report['stopped'] = self.stopped
        log.info('narration timing: %s', report)
        return Narration(lines={uid: t.text for uid, t in self.takes.items()},
                         paths={uid: t.path for uid, t in self.takes.items()},
                         seconds={uid: t.seconds for uid, t in self.takes.items()}, placements=placements,
                         violations=list(self.violations), missing=list(self.missing), report=report,
                         spoken=dict(self.spoken))
