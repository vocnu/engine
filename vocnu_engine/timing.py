"""When the narrator speaks: gap-aware timing budgets, placement at natural speed and checks of measured takes.

Ported from ai-voice-over v4 (gap-aware-voiceover-v1 with measured-turn-pauses-v1 and the measured-blocking-chain
repairs of causal_timing.py). Every limit comes from fixed source times, never from where an earlier line happened
to end, so a run of small overruns cannot add up to seconds of drift: a line that starts later than its slack
allows is a violation to repair, not a new reference point for the next line.

- A line is planned SOURCE_DELAY after its source speech starts, the first one too: v4's 1 s opening guard held
  back the first line of a clip that opens on speech and pushed every line after it late (Tears of Steel demo:
  the last line ran past the end of the clip).
- It may start up to SLACK_TIGHT later after a source gap of at most TIGHT_GAP (rapid dialogue), SLACK_LOOSE after
  a longer one.
- It may run past its source end by a tail that grows with the free time before the next source unit: TAIL_MIN up
  to a TIGHT_GAP gap, linearly to TAIL_MAX at OPEN_GAP and beyond; the last unit gets TAIL_MAX. Never past the
  media end, and never so late that the next line could not start in time after MIN_PAUSE.
- Narrator lines are PAUSE apart (different sentences; speakers are unknown) or PAUSE_SAME (one sentence that
  segments split into several units); the pause shrinks, down to MIN_PAUSE, only when a later deadline needs it.
- Speech plays at its natural speed and is never cut or overlapped. What still does not fit is reported (inspect)
  and repaired at its cause (targets): a short reply pushed late by a long line before it is not the line to shorten.
"""
from dataclasses import dataclass

from .segments import SENTENCE_END

SOURCE_DELAY = 0.10     # the narrator comes in just after the original speaker
TIGHT_GAP = 0.75        # source gaps up to this are rapid dialogue
OPEN_GAP = 2.5          # from this gap on, a line gets the whole TAIL_MAX
SLACK_TIGHT = 0.35      # how much later than planned a line may start after a tight gap...
SLACK_LOOSE = 0.75      # ...and after a longer one
TAIL_MIN = 0.35         # how far past its source end a line may run in rapid dialogue...
TAIL_MAX = 2.0          # ...and at most
PAUSE = 0.40            # silence between narrator lines of different sentences
PAUSE_SAME = 0.16       # silence between units of one sentence
MIN_PAUSE = 0.12        # the least silence between two narrator lines
CONTINUE_GAP = 0.35     # consecutive units closer than this, without sentence-final punctuation, are one sentence
EPSILON = 0.04          # timestamp/codec rounding, not an artistic allowance
TRIAL_SHARE = 0.2       # causal check: would this line, at a fifth of its length, let the late line meet its deadline?
KINDS = ('late_start', 'over_budget', 'past_end', 'overlap')
CLOSERS = '"\'”’»)]'


@dataclass(frozen=True)
class Budget:
    unit_id: int
    planned_start: float
    latest_start: float
    latest_end: float
    pause: float        # preferred silence before this line, after the previous narrator line

    @property
    def seconds(self):
        """How long the line may speak when it starts as planned (sets the translator's character budget)."""
        return max(0.0, self.latest_end - self.planned_start)


@dataclass
class Placement:
    unit_id: int
    start: float
    duration: float     # as played (after tempo)
    tempo: float = 1.0
    pause: float = 0.0  # least silence kept after the previous narrator line
    lead: float = 0.0   # seconds of the take's file before its speech starts (aligns provider timestamps)

    @property
    def end(self):
        return self.start + self.duration


@dataclass(frozen=True)
class Violation:
    unit_id: int
    kinds: tuple
    late: float         # seconds past the latest start
    over: float         # seconds past the latest end


@dataclass(frozen=True)
class Target:
    unit_id: int
    seconds: float      # the take as played now
    room: float         # how long it may be for the violation to go away
    blocked: int | None = None  # the later line it delays, when that is why it has to shrink


def continues(previous, unit):
    """Whether `unit` carries on the sentence of the source unit just before it: segments splits a sentence longer
    than MAX_UNIT_SECONDS into units that follow each other closely, without sentence-final punctuation between."""
    text = previous.text.rstrip().rstrip(CLOSERS)
    return unit.start - previous.end < CONTINUE_GAP and not text.endswith(SENTENCE_END)


def start_slack(before, unit):
    """How much later than planned `unit`'s line may start, given the source unit `before` it (None: first)."""
    if before is None:
        return 0.0
    return SLACK_TIGHT if unit.start - before.end <= TIGHT_GAP else SLACK_LOOSE


def tail_allowance(unit, after):
    """How far past its source end `unit`'s line may run, given the next source unit (None: the last one)."""
    if after is None:
        return TAIL_MAX
    fraction = (after.start - unit.end - TIGHT_GAP) / (OPEN_GAP - TIGHT_GAP)
    return TAIL_MIN + (TAIL_MAX - TAIL_MIN) * min(1.0, max(0.0, fraction))


def budgets(units, media_duration, narrated=None):
    """{unit_id: Budget} in time order for the narrated units (all of them by default), from source times only.

    Gaps, slack and tails look at every source unit, narrated or not (a song still takes its time); pauses and the
    next-line cap look at the next narrated line.
    """
    units = list(units)
    spoken = [i for i, unit in enumerate(units) if narrated is None or unit.id in narrated]
    starts = {}
    for i in spoken:
        planned = units[i].start + SOURCE_DELAY
        starts[i] = (planned, planned + start_slack(units[i - 1] if i else None, units[i]))
    result = {}
    for k, i in enumerate(spoken):
        unit = units[i]
        planned, latest = starts[i]
        end = min(unit.end + tail_allowance(unit, units[i + 1] if i + 1 < len(units) else None), media_duration)
        if k + 1 < len(spoken):
            end = min(end, starts[spoken[k + 1]][1] - MIN_PAUSE)
        same = k > 0 and spoken[k - 1] == i - 1 and continues(units[i - 1], unit)
        result[unit.id] = Budget(unit.id, planned, latest, end, PAUSE_SAME if same else PAUSE)
    return result


def _latest_starts(order, budgets, durations):
    """The latest start of each line that still lets it, and every line after it, meet their deadlines."""
    result, following = [0.0] * len(order), None
    for k in range(len(order) - 1, -1, -1):
        budget, seconds = budgets[order[k]], durations[order[k]]
        latest = min(budget.latest_start, budget.latest_end - seconds)
        if following is not None:
            latest = min(latest, following - seconds - MIN_PAUSE)
        result[k] = latest
        # A take that cannot fit anyway must not demand ever earlier starts from everything before it.
        following = max(budget.planned_start, latest)
    return result


def place(budgets, durations, tempos=None):
    """Placements of the lines in `durations` ({unit_id: seconds as played}) in the time order of `budgets`.

    Each line starts as planned, or after the previous one and its pause; the pause shrinks towards MIN_PAUSE only
    when a deadline further on needs the time. Lines never overlap and are never sped up here.
    """
    order = [uid for uid in budgets if uid in durations]
    latest = _latest_starts(order, budgets, durations)
    tempos = tempos or {}
    placements, previous_end = [], None
    for k, uid in enumerate(order):
        budget = budgets[uid]
        if previous_end is None:
            start, pause = budget.planned_start, 0.0
        else:
            pause = min(budget.pause, max(MIN_PAUSE, latest[k] - previous_end))
            start = max(budget.planned_start, previous_end + pause)
        placements.append(Placement(uid, start, durations[uid], tempos.get(uid, 1.0), pause))
        previous_end = start + durations[uid]
    return placements


def inspect(placements, budgets, media_duration):
    """[Violation] of placed lines against their budgets, in time order."""
    found, previous_end = [], None
    for p in placements:
        budget, end = budgets[p.unit_id], p.start + p.duration
        kinds = []
        if p.start > budget.latest_start + EPSILON:
            kinds.append('late_start')
        if end > budget.latest_end + EPSILON:
            kinds.append('over_budget')
        if end > media_duration + EPSILON:
            kinds.append('past_end')
        if previous_end is not None and p.start < previous_end + MIN_PAUSE - EPSILON:
            kinds.append('overlap')
        if kinds:
            found.append(Violation(p.unit_id, tuple(kinds), max(0.0, p.start - budget.latest_start),
                                   max(0.0, end - budget.latest_end)))
        previous_end = end
    return found


def score(violations):
    """Lower is better: (lines in violation, total seconds late or over)."""
    return len(violations), sum(v.late + v.over for v in violations)


def targets(placements, budgets, texts, violations, fixed=()):
    """[Target] whose shorter take would remove `violations`, in time order (v4 inspect + causal_timing.repairs).

    A line over its own budget is its own target when it started on time or would not fit even then. A line that is
    late or over only because earlier lines pushed it is not: of the earlier lines in the same unbroken chain that
    measurably delay it, the longest sentence is (an atomic reply or name cannot be shortened faithfully). Lines in
    `fixed` cannot get any shorter, so the next contributor is chosen instead.
    """
    index = {p.unit_id: k for k, p in enumerate(placements)}
    durations = {p.unit_id: p.duration for p in placements}
    trials = {}     # line → placements with that line cut to TRIAL_SHARE, shared by every violation it may cause
    chosen = {}
    for violation in violations:
        p = placements[index[violation.unit_id]]
        budget = budgets[p.unit_id]
        late = p.start > budget.latest_start + EPSILON
        if p.unit_id not in fixed and p.end > budget.latest_end + EPSILON and (
                p.start <= budget.planned_start + EPSILON or p.duration > budget.seconds + EPSILON):
            # When it is also late, aim at its own budget, not at the impossible one its predecessors left it.
            start = budget.planned_start if late else p.start
            chosen[p.unit_id] = Target(p.unit_id, p.duration, max(0.0, budget.latest_end - start))
    for violation in violations:
        k = index[violation.unit_id]
        p = placements[k]
        budget = budgets[p.unit_id]
        onset = p.start > budget.latest_start + EPSILON
        deficit = p.start - budget.latest_start if onset else p.end - budget.latest_end
        if deficit <= EPSILON:
            continue
        chain, cursor = ([] if onset else [k]), k   # shortening a late line cannot make it start earlier
        while cursor:
            if placements[cursor].start > placements[cursor - 1].end + placements[cursor].pause + EPSILON:
                break   # a real free gap: lines before it are not the cause
            cursor -= 1
            chain.append(cursor)
        influential = []
        for j in chain:
            uid = placements[j].unit_id
            if uid in chosen or uid in fixed:
                continue
            if uid not in trials:
                trials[uid] = place(budgets, {**durations, uid: max(0.01, durations[uid] * TRIAL_SHARE)})
            moved = trials[uid][k]
            if (p.start - moved.start if onset else p.end - moved.end) > EPSILON:
                influential.append(j)
        if not influential:
            continue
        j = max(influential, key=lambda n: (len(texts.get(placements[n].unit_id, '')), placements[n].duration))
        cause = placements[j]
        chosen[cause.unit_id] = Target(cause.unit_id, cause.duration, max(0.0, cause.duration - deficit), p.unit_id)
    return [chosen[p.unit_id] for p in placements if p.unit_id in chosen]
