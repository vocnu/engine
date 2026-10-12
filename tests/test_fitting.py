"""Fitting narrator takes into their timing budgets: retakes, rephrasing, last-resort tempo and silent takes."""
import math

import pytest

from vocnu_engine import fitting, mix, timing, translate, tts
from vocnu_engine.costs import BudgetExceeded, Receipt
from vocnu_engine.errors import ProviderError
from vocnu_engine.segments import Cue, Unit


class Studio:
    """Fake ElevenLabs, measurement and translate.shorten. Each take's length is scripted by text, or by
    (text, seed offset); 0 means the take came back silent."""

    def __init__(self, monkeypatch, lengths, rewrites=()):
        self.lengths = lengths
        self.rewrites = list(rewrites)   # one {unit_id: new wording} answer per shorten call
        self.spoken = []                 # (unit_id, text, seed offset) per synthesized take
        self.asked = []                  # the lines each shorten call received
        self.options = []                # its keyword arguments
        monkeypatch.setattr(tts, 'synthesize_all', self.synthesize_all)
        monkeypatch.setattr(mix, 'measure', self.measure)
        monkeypatch.setattr(translate, 'shorten', self.shorten)

    def synthesize_all(self, lines, language, voice, cache_dir, receipt, *, session=None, seeds=None, on_done=None):
        result = {}
        for key, text in lines.items():
            offset = (seeds or {}).get(key, tts.SEED) - tts.SEED
            self.spoken.append((key, text, offset))
            receipt.add('speech', 'elevenlabs', tts.estimate_usd([text]), characters=len(text))
            result[key] = f'{text}#{offset}'
            if on_done:
                on_done(key)
        return result

    def measure(self, path):
        text, offset = path.rsplit('#', 1)
        seconds = self.lengths.get((text, int(offset)), self.lengths.get(text, 1.0))
        return mix.Measured(seconds, 0.05, seconds <= 0)

    def shorten(self, lines, *, target, receipt, session=None, glossary=None):
        self.asked.append(lines)
        self.options.append({'target': target, 'glossary': glossary})
        answer = self.rewrites.pop(0) if self.rewrites else {}
        return {line['id']: answer.get(line['id'], line['text']) for line in lines}


def scene(*rows):
    timeline = [Unit(i + 1, start, end, [i + 1], text) for i, (start, end, text) in enumerate(rows)]
    return timeline, [Cue(u.id, u.start, u.end, u.text) for u in timeline]


def narrate(timeline, cues, lines, media_duration=30.0, budget=5.0, **options):
    return fitting.narrate(timeline, lines, media_duration=media_duration, language='lt', voice='male',
                           cache='unused', receipt=Receipt(max_usd=budget), cues=cues, **options)


def offsets(studio, unit_id):
    return [offset for uid, _, offset in studio.spoken if uid == unit_id]


REPLY = scene((1.0, 1.6, 'No!'), (1.6, 4.0, 'Then we go home together tonight.'))
REPLY_LINES = {1: 'Ne!', 2: 'Tada einam namo kartu šįvakar.'}


def test_a_short_line_gets_a_same_text_retake_with_the_next_seed(monkeypatch):
    studio = Studio(monkeypatch, {('Ne!', 0): 1.0, ('Ne!', 1): 0.7, REPLY_LINES[2]: 1.5})
    narration = narrate(*REPLY, REPLY_LINES)
    assert offsets(studio, 1) == [0, 1] and narration.paths[1] == 'Ne!#1'
    assert narration.lines == REPLY_LINES and narration.seconds[1] == 0.7
    assert narration.violations == [] and studio.asked == []     # a call is not rephrased
    assert narration.report['retakes'] == 1 and narration.report['retakes_kept'] == 1


def test_a_retake_must_be_measurably_shorter_and_a_later_round_tries_the_next_seed(monkeypatch):
    studio = Studio(monkeypatch, {('Ne!', 0): 1.0, ('Ne!', 1): 0.995, ('Ne!', 2): 0.7, REPLY_LINES[2]: 1.5})
    narration = narrate(*REPLY, REPLY_LINES)
    assert offsets(studio, 1) == [0, 1, 2]
    assert narration.paths[1] == 'Ne!#2' and narration.violations == []
    assert (narration.report['retakes'], narration.report['retakes_kept'], narration.report['rounds']) == (2, 1, 2)


LONG = scene((1.0, 3.0, 'We talked about so many things that never fit.'), (3.1, 6.0, 'Yes, let us go home now.'),
             (7.0, 8.0, 'Fine.'))
LONG_LINES = {1: 'Kalbėjome apie tiek daug dalykų, kurie niekaip netelpa.', 2: 'Taip, einam namo dabar.', 3: 'Gerai.'}


def test_a_shorter_wording_with_a_longer_take_never_replaces_the_current_one(monkeypatch):
    first, second = 'Kalbėjome apie dalykus, kurie netelpa.', 'Kalbėjome apie viską.'
    studio = Studio(monkeypatch, {LONG_LINES[1]: 3.0, first: 3.2, second: 2.0, LONG_LINES[2]: 1.0},
                    rewrites=[{1: first}, {1: second}])
    narration = narrate(*LONG, LONG_LINES, glossary=[{'source': 'home', 'target': 'namai'}], speakers={1: 'female'})
    assert narration.lines[1] == second and narration.seconds[1] == 2.0 and narration.violations == []
    assert (narration.report['rephrase_rounds'], narration.report['rephrased_lines']) == (2, 1)
    # The second request shows every wording tried so far with its measured take.
    assert studio.asked[1][0]['previous'] == [{'text': LONG_LINES[1], 'seconds': 3.0}, {'text': first, 'seconds': 3.2}]


def test_the_rephrase_request_carries_the_measured_problem_and_the_context(monkeypatch):
    studio = Studio(monkeypatch, {LONG_LINES[1]: 3.0, LONG_LINES[2]: 1.0})
    narrate(*LONG, LONG_LINES, glossary=[{'source': 'home', 'target': 'namai'}], speakers={1: 'female'})
    budget = timing.budgets(LONG[0], 30.0)[1]
    room = budget.latest_end - budget.planned_start
    assert studio.asked[0] == [{
        'id': 1, 'text': LONG_LINES[1], 'max_chars': math.floor(len(LONG_LINES[1]) * room / 3.0 * 0.95),
        'source': LONG[0][0].text, 'context_before': [], 'context_after': [LONG[0][1].text, LONG[0][2].text],
        'speaker': 'female', 'previous': [{'text': LONG_LINES[1], 'seconds': 3.0}],
        'problem': f'take 3.0 s, room {room:.1f} s'}]
    assert studio.options[0] == {'target': 'lt', 'glossary': [{'source': 'home', 'target': 'namai'}]}
    # The same answer as before buys no new take.
    assert offsets(studio, 1) == [0]


EPISODE_FIVE = scene((699.51, 704.56, 'I will hang him upside down, blindfold him, knife, water, torch, pliers, nails...'),
                     (704.56, 705.1, 'Hey.'), (705.1, 705.81, 'Did you find him?!'), (706.23, 708.02, 'I want a drink.'))
EPISODE_LINES = {1: 'Pakabinsiu žemyn galva, užrišiu akis, peilis, vanduo, degiklis, replės, nagai...', 2: 'Ei.',
                 3: 'Radai jį?!', 4: 'Noriu gerti.'}


def test_the_cause_is_repaired_not_the_reply_it_pushed_late_and_speed_up_is_the_last_resort(monkeypatch):
    studio = Studio(monkeypatch, {EPISODE_LINES[1]: 5.04, 'Ei.': 0.48, 'Radai jį?!': 0.96, 'Noriu gerti.': 0.96})
    narration = narrate(*EPISODE_FIVE, EPISODE_LINES, media_duration=800.0)
    # Only the long sentence is rephrased; it keeps its wording, so a second round would only repeat the request.
    assert [[line['id'] for line in asked] for asked in studio.asked] == [[1]]
    assert narration.report['rounds'] == narration.report['rephrase_rounds'] == 1
    assert all(offset == 0 for _, _, offset in studio.spoken)       # no retakes of the short replies
    assert 'delays the lines after it' in studio.asked[0][0]['problem']
    # No faithful shorter wording: the cause plays a little faster instead. "Ei." cannot start before its own source,
    # so that only partly helps; the remaining real reply is then the cause (v4: once upstream fits, the reply that
    # is still too long gets its own repair). The atomic "Ei." is never touched, and everything fits.
    tempos = [p.tempo for p in narration.placements]
    assert tempos[0] > 1.0 and tempos[1] == tempos[3] == 1.0 and max(tempos) <= fitting.MAX_TEMPO
    assert narration.violations == []
    assert narration.report['sped_up_lines'] == sum(t > 1.0 for t in tempos)
    assert narration.report['max_tempo'] == max(tempos)


def test_when_one_cause_is_at_the_speed_limit_the_next_one_in_the_chain_helps(monkeypatch):
    timeline, cues = scene((1.0, 3.0, 'We talked for a long while.'),
                           (3.0, 5.0, 'And then we talked for an even longer while.'), (5.0, 5.5, 'No.'))
    lines = {1: 'Mes kalbėjome labai labai ilgai.', 2: 'Tada kalbėjome dar ilgiau ir ilgiau.', 3: 'Ne.'}
    Studio(monkeypatch, {lines[1]: 2.3, lines[2]: 2.3, 'Ne.': 0.4})
    narration = narrate(timeline, cues, lines)
    assert [p.tempo for p in narration.placements] == [1.08, 1.08, 1.0]
    # Still 0.15 s too long at 1.08×: reported, and the reply is late rather than overlapped or cut.
    assert narration.report['violations'] == {'late_start': 1, 'over_budget': 1, 'past_end': 0, 'overlap': 0}
    assert narration.placements[2].start - narration.placements[1].end == pytest.approx(timing.MIN_PAUSE)


def test_speed_up_stops_at_108_and_what_still_does_not_fit_is_reported(monkeypatch):
    timeline, cues = scene((8.0, 9.0, 'This is far too long for the end.'))
    line = 'Tai daug per ilga eilutė pačiai pabaigai.'
    Studio(monkeypatch, {line: 2.5})
    narration = narrate(timeline, cues, {1: line}, media_duration=10.0)
    placement = narration.placements[0]
    assert placement.tempo == fitting.MAX_TEMPO == 1.08
    assert placement.duration == pytest.approx(2.5 / 1.08) and placement.end > 10.0    # never cut
    report = narration.report
    assert report['violations'] == {'late_start': 0, 'over_budget': 1, 'past_end': 1, 'overlap': 0}
    assert (report['sped_up_lines'], report['max_tempo'], report['lines_off_budget']) == (1, 1.08, 1)


def test_a_silent_take_is_retried_once_with_the_next_seed_then_left_out(monkeypatch):
    timeline, cues = scene((1.0, 2.0, 'Hello.'), (4.0, 5.0, 'How are you?'))
    studio = Studio(monkeypatch, {('Labas.', 0): 0.0, ('Labas.', 1): 0.8, 'Kaip sekasi?': 0.0})
    done = []
    narration = narrate(timeline, cues, {1: 'Labas.', 2: 'Kaip sekasi?'}, on_done=done.append)
    assert sorted(done) == [1, 2]
    assert offsets(studio, 1) == [0, 1] and offsets(studio, 2) == [0, 1]
    assert narration.paths == {1: 'Labas.#1'} and narration.lines == {1: 'Labas.'}
    assert narration.missing == [2] and [p.unit_id for p in narration.placements] == [1]
    assert narration.report['missing_speech'] == 1


def test_repairs_stop_at_the_job_budget_and_the_job_goes_on(monkeypatch):
    studio = Studio(monkeypatch, {LONG_LINES[1]: 3.0, LONG_LINES[2]: 1.0})
    narration = narrate(*LONG, LONG_LINES, budget=tts.estimate_usd(LONG_LINES.values()) + 0.001)
    assert studio.asked == [] and narration.report['stopped'] == 'budget'
    assert narration.report['max_tempo'] == fitting.MAX_TEMPO      # the free last resort still applies


def test_a_failing_rephrase_stops_repairs_without_failing_the_job(monkeypatch):
    Studio(monkeypatch, {LONG_LINES[1]: 3.0, LONG_LINES[2]: 1.0})

    def unavailable(lines, **options):
        raise ProviderError('OpenAI is unavailable right now (HTTP 503).', 503)

    monkeypatch.setattr(translate, 'shorten', unavailable)
    narration = narrate(*LONG, LONG_LINES)
    assert narration.report['stopped'] == 'provider_error' and narration.lines == LONG_LINES


def test_nothing_to_narrate_is_an_empty_narration(monkeypatch):
    studio = Studio(monkeypatch, {})
    narration = narrate(*LONG, {})
    assert (narration.lines, narration.placements, narration.missing, studio.spoken) == ({}, [], [], [])
    assert narration.report['violations'] == dict.fromkeys(timing.KINDS, 0)


def test_the_first_synthesis_is_not_optional(monkeypatch):
    Studio(monkeypatch, {})
    with pytest.raises(BudgetExceeded):
        narrate(*LONG, LONG_LINES, budget=0.001)
