from vocnu_engine import subtitles, timing


def test_merge_orders_cues_and_ends_each_before_the_next():
    narration = [(0.6, 3.5, 'Eilutė'), (10.0, 12.0, 'Kita')]
    lyrics = [(3.0, 6.0, 'Daina'), (6.0, 6.0, ''), (11.0, 11.02, 'Per trumpa')]
    assert subtitles.merge(narration, lyrics) == [(0.6, 2.95, 'Eilutė'), (3.0, 6.0, 'Daina'),
                                                 (10.0, 10.95, 'Kita'), (11.0, 11.02, 'Per trumpa')]


def test_wrap_keeps_two_balanced_lines_within_limit():
    text = 'Mes turime išeiti dabar, kol dar nesutemo ir kelias matomas.'
    lines = subtitles.wrap(text, 42)
    assert len(lines) == 2
    assert all(len(line) <= 42 for line in lines)
    assert ' '.join(lines) == text


def test_short_text_is_one_line_and_whitespace_is_normalised():
    assert subtitles.wrap('  Labas,   pasauli! ', 42) == ['Labas, pasauli!']
    assert subtitles.wrap('', 42) == []


def test_unspaced_scripts_split_by_characters():
    lines = subtitles.wrap('今日はとても良い天気ですね、散歩に行きましょう', 13, spaced=False)
    assert len(lines) == 2 and ''.join(lines) == '今日はとても良い天気ですね、散歩に行きましょう'


def test_srt_and_vtt_format():
    cues = [(0.0, 1.5, 'Labas'), (61.25, 3723.004, 'Viso gero'), (5, 6, '   ')]
    assert subtitles.srt(cues) == ('1\n00:00:00,000 --> 00:00:01,500\nLabas\n\n'
                                   '2\n00:01:01,250 --> 01:02:03,004\nViso gero\n')
    vtt = subtitles.vtt(cues)
    assert vtt.startswith('WEBVTT\n')
    assert '00:01:01.250 --> 01:02:03.004\nViso gero' in vtt


class _Placement:
    def __init__(self, unit_id, start, duration):
        self.unit_id, self.start, self.duration = unit_id, start, duration


def test_narration_cues_follow_the_spoken_line_and_stay_half_a_second_after_it():
    cues = subtitles.narration_cues([_Placement(1, 2.0, 3.0)], {1: 'Prisipažink, mano roboto ranka tave gąsdina.'})
    assert cues == [(2.0, 5.5, 'Prisipažink, mano roboto ranka tave gąsdina.')]


def test_long_narration_is_split_at_sentences_with_time_shared_by_length():
    line = 'Tu mulkis, Tomai. ' + 'Celia, turim siekti aistrų ir daryti tai, ką mylime, kol dar galime tai daryti.'
    cues = subtitles.narration_cues([_Placement(1, 0.0, 6.0)], {1: line}, line_chars=42)
    assert [c[2] for c in cues][0] == 'Tu mulkis, Tomai.'
    assert ' '.join(c[2] for c in cues) == ' '.join(line.split())
    assert cues[0][0] == 0.0 and cues[1][0] == cues[0][1] == round(6.0 * 17 / 96, 3)   # shared by length
    assert 6.0 < cues[-1][1] <= 6.0 + subtitles.MAX_TAIL
    assert all(len(c[2]) <= 84 for c in cues)


def test_cjk_narration_splits_on_full_width_punctuation():
    line = '我们必须追随自己的热情。' * 10
    cues = subtitles.narration_cues([_Placement(1, 0.0, 10.0)], {1: line}, line_chars=16, spaced=False)
    assert all(len(c[2]) <= 32 for c in cues) and ''.join(c[2] for c in cues) == line


def test_ass_has_one_readable_style_and_escapes_override_blocks():
    text = subtitles.ass([(1.0, 3.5, 'Labas, aš esu Ičigas Kurosakis ir man penkiolika metų.'), (4.0, 5.0, 'Taip {ne}'),
                          (6.0, 7.0, ' ')], line_chars=42)
    assert text.startswith('[Script Info]') and 'Style: Default,Noto Sans,64' in text
    events = [line for line in text.splitlines() if line.startswith('Dialogue:')]
    assert events == ['Dialogue: 0,0:00:01.00,0:00:03.50,Default,,0,0,0,,Labas, aš esu Ičigas Kurosakis\\Nir man '
                      'penkiolika metų.', 'Dialogue: 0,0:00:04.00,0:00:05.00,Default,,0,0,0,,Taip (ne)']


def timestamps(text, words):
    """ElevenLabs-style character alignment of `text`, each word spoken over its (start, end) in the take's file."""
    chars, starts, ends = [], [], []
    for k, (token, (a, b)) in enumerate(zip(text.split(' '), words, strict=True)):
        if k:
            chars.append(' ')
            starts.append(ends[-1])
            ends.append(a)
        step = (b - a) / len(token)
        for i, ch in enumerate(token):
            chars.append(ch)
            starts.append(a + i * step)
            ends.append(a + (i + 1) * step)
    return {'alignment': {'characters': chars, 'character_start_times_seconds': starts,
                          'character_end_times_seconds': ends}, 'normalized_alignment': None}


LINE = 'Taip, taip. Einam namo dabar. Jau labai vėlu.'   # three sentences, more than 2 × 20 characters


def cues_for(words, *, start=10.0, lead=0.05, tempo=1.0, line_chars=20, **options):
    duration = (words[-1][1] + 0.06 - lead) / tempo
    placement = timing.Placement(1, start, duration, tempo=tempo, lead=lead)
    return subtitles.narration_cues([placement], {1: LINE}, line_chars, alignments={1: timestamps(LINE, words)},
                                    **options)


def test_the_takes_character_timestamps_time_each_piece():
    words = [(0.05, 0.35), (0.4, 0.8), (1.5, 1.8), (1.85, 2.1), (2.15, 2.5), (3.4, 3.6), (3.65, 3.9), (3.95, 4.3)]
    cues = cues_for(words)
    assert [c[2] for c in cues] == ['Taip, taip.', 'Einam namo dabar.', 'Jau labai vėlu.']
    assert [c[0] for c in cues] == [10.0, 11.45, 13.35]           # placement start + (word start - lead)
    sped = cues_for(words, tempo=1.08)
    assert [c[0] for c in sped] == [10.0, round(10.0 + 1.45 / 1.08, 3), round(10.0 + 3.35 / 1.08, 3)]


def test_every_sentence_of_a_line_is_its_own_cue():
    # One narrator line can hold two characters' sentences (recognition ran them together): one cue would make
    # them one speaker's, so sentences never share a cue, even 0.4 s apart and short enough for two lines.
    words = [(0.05, 0.35), (0.4, 0.8), (1.2, 1.5), (1.55, 1.8), (1.85, 2.2), (3.4, 3.6), (3.65, 3.9), (3.95, 4.3)]
    cues = cues_for(words)
    assert [c[2] for c in cues] == ['Taip, taip.', 'Einam namo dabar.', 'Jau labai vėlu.']
    assert cues[0][0] == 10.0 and cues[0][1] <= cues[1][0] and cues[2][0] == 13.35


def test_clauses_of_one_sentence_a_short_pause_apart_merge_when_they_fit():
    line = 'Taip, žinoma, einam namo.'
    placement = timing.Placement(1, 10.0, 2.25, lead=0.05)
    stamps = timestamps(line, [(0.05, 0.35), (0.4, 0.9), (1.3, 1.6), (1.65, 2.2)])
    # Two lines of 10: "Taip," and "žinoma," share a cue; the whole sentence would not fit.
    assert [c[2] for c in subtitles.narration_cues([placement], {1: line}, 10, alignments={1: stamps})] == [
        'Taip, žinoma,', 'einam namo.']


def test_a_long_pause_keeps_pieces_apart():
    words = [(0.05, 0.35), (0.4, 0.8), (1.4, 1.7), (1.75, 2.0), (2.05, 2.4), (3.4, 3.6), (3.65, 3.9), (3.95, 4.3)]
    assert len(cues_for(words)) == 3          # 0.6 s after "taip." is more than MERGE_GAP


def test_timestamps_for_other_words_fall_back_to_character_shares():
    placement = timing.Placement(1, 2.0, 3.0, lead=0.05)
    line = 'Prisipažink, mano roboto ranka tave gąsdina.'
    other = timestamps('Visai kiti žodžiai.', [(0.05, 0.4), (0.45, 0.8), (0.85, 1.5)])
    assert subtitles.narration_cues([placement], {1: line}, alignments={1: other}) == \
        subtitles.narration_cues([placement], {1: line})


def test_a_short_word_stays_up_long_enough_to_read():
    assert subtitles.narration_cues([_Placement(1, 1.0, 0.3)], {1: 'Ne!'}) == [(1.0, 2.2, 'Ne!')]


def test_longer_text_gets_reading_time_up_to_a_second_and_a_half_after_the_words():
    text = 'Tai iš tiesų neįprasta, keista ir labai ilga istorija apie mus.'   # 63 characters: 4.4 s to read
    assert subtitles.narration_cues([_Placement(1, 1.0, 2.0)], {1: text}) == [(1.0, 4.5, text)]
    medium = 'Tai neįprasta istorija.'
    assert subtitles.narration_cues([_Placement(1, 1.0, 2.0)], {1: medium}) == [(1.0, 3.5, medium)]
    # Scripts read more slowly per character use their language's rate: 9 characters at 8 per second, not 15.
    hanzi = '我们必须追随热情。'
    assert subtitles.narration_cues([_Placement(1, 1.0, 0.5)], {1: hanzi}, 16, False) == [(1.0, 2.2, hanzi)]
    assert subtitles.narration_cues([_Placement(1, 1.0, 0.5)], {1: hanzi}, 16, False, reading_cps=8.0) == \
        [(1.0, 2.325, hanzi)]


def test_a_cue_never_runs_into_the_next_narrator_line_or_past_seven_seconds():
    cues = subtitles.narration_cues([_Placement(1, 1.0, 0.4), _Placement(2, 1.6, 1.0)], {1: 'Ne!', 2: 'Taip.'})
    assert cues[0] == (1.0, 1.52, 'Ne!')       # 0.08 s before the next cue, not the 1.2 s minimum
    # The next line's first word comes after its file's lead: the cue still stops where that line starts.
    following = timing.Placement(2, 1.6, 1.0, lead=0.05)
    cues = subtitles.narration_cues([_Placement(1, 1.0, 0.4), following], {1: 'Ne!', 2: 'Taip.'},
                                    alignments={2: timestamps('Taip.', [(0.3, 0.7)])})
    assert cues[1][0] == 1.85 and cues[0][1] == 1.6
    long = 'Labai ilgas sakinys, kurį pasakotojas skaito beveik septynias sekundes be pertraukos.'
    assert subtitles.narration_cues([_Placement(1, 0.0, 6.8)], {1: long}, line_chars=60) == [(0.0, 7.0, long)]


def test_different_narrator_lines_are_never_merged():
    cues = subtitles.narration_cues([_Placement(1, 1.0, 0.5), _Placement(2, 1.65, 0.5)], {1: 'Taip!', 2: 'Ne!'})
    assert [c[2] for c in cues] == ['Taip!', 'Ne!']


def test_cues_follow_narration_that_runs_past_the_media_end_but_go_no_further():
    assert subtitles.narration_cues([_Placement(1, 9.0, 2.0)], {1: 'Labas vakaras.'}, duration=10.0) == \
        [(9.0, 11.0, 'Labas vakaras.')]
    assert subtitles.narration_cues([_Placement(1, 8.5, 1.2)], {1: 'Labas vakaras, drauge.'}, duration=10.0) == \
        [(8.5, 10.0, 'Labas vakaras, drauge.')]


def test_song_lyrics_still_merge_with_readable_narration():
    narration = subtitles.narration_cues([_Placement(1, 1.0, 0.3)], {1: 'Ne!'})
    assert subtitles.merge(narration, [(1.5, 4.0, 'Daina')]) == [(1.0, 1.45, 'Ne!'), (1.5, 4.0, 'Daina')]


def test_song_and_sign_lines_sit_at_the_top_and_may_overlap_dialogue():
    cues = subtitles.layered([(1.0, 3.0, 'Labas.')], [(2.0, 4.0, 'Pakelk akis')])
    assert cues == [(1.0, 3.0, 'Labas.'), (2.0, 4.0, 'Pakelk akis', 'top')]
    vtt = subtitles.vtt(cues)
    assert '00:00:01.000 --> 00:00:03.000\nLabas.' in vtt
    assert '00:00:02.000 --> 00:00:04.000 line:0\nPakelk akis' in vtt
    ass = subtitles.ass(cues)
    assert 'Style: Top,' in ass
    assert ',Default,,0,0,0,,Labas.' in ass and ',Top,,0,0,0,,Pakelk akis' in ass
    assert '2\n00:00:02,000 --> 00:00:04,000\nPakelk akis' in subtitles.srt(cues)
