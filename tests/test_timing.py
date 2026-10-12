"""Gap-aware narrator timing (ai-voice-over v4 gap-aware-voiceover-v1, measured-turn-pauses-v1, causal repairs)."""
import pytest

from vocnu_engine import timing
from vocnu_engine.segments import Unit


def units(*rows):
    """Units from (start, end) or (start, end, text) rows; separate sentences unless the text says otherwise."""
    return [Unit(i + 1, row[0], row[1], [i + 1], row[2] if len(row) > 2 else f'Line {i + 1}.')
            for i, row in enumerate(rows)]


def plan(timeline, durations, media_duration=30.0):
    budgets = timing.budgets(timeline, media_duration, narrated=set(durations))
    placements = timing.place(budgets, durations)
    return budgets, placements, timing.inspect(placements, budgets, media_duration)


def test_the_first_narration_follows_its_speech_like_any_other():
    budgets = timing.budgets(units((0.0, 1.04), (1.26, 3.64)), 10.0)
    assert budgets[1].planned_start == pytest.approx(0.1)
    assert budgets[2].planned_start == pytest.approx(1.36)


def test_a_clip_that_opens_on_rapid_dialogue_stays_on_time_to_its_end():
    # The Tears of Steel demo's timeline (20.6 s, speech from 0.0 s) with takes that fit their own source spans:
    # v4's 1 s opening guard made the second line start late and the last one ("Kaip nori, Tomai.") run past the
    # end of the clip.
    timeline = units((0.0, 1.04), (1.26, 3.64), (3.94, 6.74), (7.22, 11.08), (11.3, 13.3), (14.27, 15.03),
                     (15.35, 16.15), (16.27, 19.19), (19.29, 20.13))
    takes = {1: 0.9, 2: 2.1, 3: 2.5, 4: 3.4, 5: 1.5, 6: 0.7, 7: 0.7, 8: 2.6, 9: 0.9}
    _, placements, violations = plan(timeline, takes, 20.6)
    assert placements[0].start == pytest.approx(0.1)
    assert violations == [] and placements[-1].end <= 20.6


def test_lines_that_fit_play_at_natural_speed_on_time():
    _, placements, violations = plan(units((1.0, 3.0), (6.0, 8.0)), {1: 2.0, 2: 1.5}, 10.0)
    assert [(round(p.start, 3), p.tempo, p.duration) for p in placements] == [(1.1, 1.0, 2.0), (6.1, 1.0, 1.5)]
    assert violations == []


def test_start_slack_depends_on_the_source_gap_before_a_line():
    timeline = units((1.0, 2.0), (2.5, 4.0), (4.75, 6.0), (7.0, 8.0))   # gaps 0.5, 0.75, 1.0
    slack = [round(b.latest_start - b.planned_start, 3) for b in timing.budgets(timeline, 20.0).values()]
    assert slack == [0.0, 0.35, 0.35, 0.75]


@pytest.mark.parametrize('gap, tail', [(0.0, 0.35), (0.5, 0.35), (0.75, 0.35), (1.625, 1.175), (2.5, 2.0),
                                       (4.0, 2.0)])
def test_tail_allowance_grows_with_the_free_gap_up_to_two_seconds(gap, tail):
    before, after = units((1.0, 2.0), (2.0 + gap, 4.0 + gap))
    assert timing.tail_allowance(before, after) == pytest.approx(tail)


def test_the_tail_stops_at_the_media_end_and_before_the_next_line_can_start():
    assert timing.tail_allowance(units((1.0, 2.0))[0], None) == 2.0
    assert timing.budgets(units((8.0, 9.0)), 10.0)[1].latest_end == 10.0     # not 11.0
    # Touching units: the 0.35 s tail (3.35) would leave the next line, latest at 3.1 + 0.35, under 0.12 s of pause.
    budgets = timing.budgets(units((1.0, 3.0), (3.0, 5.0)), 30.0)
    assert budgets[1].latest_end == pytest.approx(3.45 - timing.MIN_PAUSE)


def test_different_sentences_keep_four_tenths_of_a_second_apart():
    _, placements, violations = plan(units((1.0, 3.0, 'We have to cross.'), (3.1, 5.0, 'Before dark.')),
                                     {1: 1.8, 2: 1.0})
    assert placements[1].start - placements[0].end == pytest.approx(timing.PAUSE) == pytest.approx(0.40)
    assert violations == []


def test_one_sentence_split_into_units_keeps_a_short_pause():
    timeline = units((1.0, 3.0, 'We have to cross'), (3.1, 5.0, 'the mountains before dark.'))
    assert timing.continues(*timeline)
    budgets, placements, violations = plan(timeline, {1: 2.2, 2: 1.0})
    assert budgets[2].pause == timing.PAUSE_SAME == 0.16
    assert placements[1].start - placements[0].end == pytest.approx(0.16)
    assert violations == []


def test_what_counts_as_the_same_sentence():
    assert not timing.continues(*units((1.0, 3.0, 'We have to cross.'), (3.1, 5.0, 'the mountains')))
    assert not timing.continues(*units((1.0, 3.0, 'He said "go."'), (3.1, 5.0, 'and left')))
    assert not timing.continues(*units((1.0, 3.0, 'We have to cross'), (3.4, 5.0, 'the mountains')))   # 0.4 s gap


def test_the_pause_shrinks_only_as_far_as_the_next_deadline_needs_and_never_below_012():
    timeline = units((1.0, 3.0), (3.0, 5.0))
    budgets, placements, violations = plan(timeline, {1: 2.23, 2: 1.0})
    assert placements[1].pause == pytest.approx(0.12)
    assert placements[1].start == pytest.approx(budgets[2].latest_start)
    assert violations == []
    # A take that ends later still keeps 0.12 s of silence; its follower is then late, and that is reported.
    budgets, placements, violations = plan(timeline, {1: 2.3, 2: 1.0})
    assert placements[1].start - placements[0].end == pytest.approx(0.12)
    assert [(v.unit_id, v.kinds) for v in violations] == [(1, ('over_budget',)), (2, ('late_start',))]


def test_small_spills_never_turn_into_unnoticed_drift():
    timeline = units(*[(1.0 + 2 * i, 3.0 + 2 * i) for i in range(8)])
    budgets, placements, violations = plan(timeline, {i + 1: 2.1 for i in range(8)})
    # Limits come from source times: each line may start 0.35 s late, however late the line before it ended.
    assert all(b.latest_start == pytest.approx(b.planned_start + 0.35) for b in list(budgets.values())[1:])
    assert placements[-1].start - timeline[-1].start > 1.0
    assert any('late_start' in v.kinds for v in violations)
    assert all(b.start >= a.end + timing.MIN_PAUSE - 1e-9 for a, b in zip(placements, placements[1:]))


def test_narration_past_the_media_end_is_reported_not_cut():
    budgets, placements, violations = plan(units((8.0, 10.0)), {1: 2.2}, 10.0)
    assert placements[0].duration == 2.2 and placements[0].end > 10.0
    assert violations[0].kinds == ('over_budget', 'past_end')
    assert timing.targets(placements, budgets, {1: 'Line 1.'}, violations) == [
        timing.Target(1, 2.2, pytest.approx(10.0 - 8.1))]


def episode_five():
    """v4 regression (Buddy Daddies S01E05): a long sentence pushes a reply over its deadline."""
    return units((699.51, 704.56, 'Pakabinsiu žemyn galva, užrišiu akis, peilis, vanduo, degiklis, replės, nagai...'),
                 (704.56, 705.1, 'Ei.'), (705.1, 705.81, 'Radai jį?!'), (706.23, 708.02, 'Noriu gerti.'))


def repair_targets(timeline, durations, media_duration=800.0):
    budgets, placements, violations = plan(timeline, durations, media_duration)
    return timing.targets(placements, budgets, {u.id: u.text for u in timeline}, violations), violations


def test_the_blocking_sentence_is_repaired_not_the_short_reply_it_delayed():
    targets, violations = repair_targets(episode_five(), {1: 5.04, 2: 0.48, 3: 0.96, 4: 0.96})
    assert [v.unit_id for v in violations] == [3]
    assert [(t.unit_id, t.blocked) for t in targets] == [(1, 3)]
    assert targets[0].room == pytest.approx(5.04 - (705.37 + 0.96 - 706.16))   # its length minus the deficit


def test_a_cause_that_cannot_get_any_shorter_is_passed_over_for_the_next_contributor():
    budgets, placements, violations = plan(episode_five(), {1: 5.04, 2: 0.48, 3: 0.96, 4: 0.96}, 800.0)
    texts = {u.id: u.text for u in episode_five()}
    found = timing.targets(placements, budgets, texts, violations, fixed={1})
    assert [(t.unit_id, t.blocked) for t in found] == [(3, 3)]     # the reply itself, never the atomic "Ei."


def test_once_upstream_fits_a_reply_that_is_too_long_itself_is_the_target():
    targets, _ = repair_targets(episode_five(), {1: 4.6, 2: 0.48, 3: 0.96, 4: 0.96})
    assert [t.unit_id for t in targets] == [3]
    assert repair_targets(episode_five(), {1: 4.6, 2: 0.48, 3: 0.88, 4: 0.96}) == ([], [])


def test_a_late_victim_is_not_rewritten_for_an_overlong_line_before_it():
    targets, violations = repair_targets(units((1.0, 2.0, 'Ilga replika.'), (3.0, 6.0, 'Ne.')), {1: 4.0, 2: 0.3}, 10.0)
    assert {v.unit_id for v in violations} == {1, 2}
    assert [(t.unit_id, t.blocked) for t in targets] == [(1, None)]
    assert targets[0].room == pytest.approx(2.0 + timing.tail_allowance(*units((1.0, 2.0), (3.0, 6.0))) - 1.1)


def test_a_real_free_gap_ends_the_causal_chain():
    timeline = units((10.0, 50.0, 'Labai ilga ankstesnė scena.'), *[(u.start, u.end, u.text) for u in episode_five()])
    targets, _ = repair_targets(timeline, {1: 39.0, 2: 5.04, 3: 0.48, 4: 0.96, 5: 0.96})
    assert [t.unit_id for t in targets] == [2]
