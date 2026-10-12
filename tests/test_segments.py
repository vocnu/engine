from vocnu_engine import segments


def words_from(spec):
    """spec: list of (start, end, word) with Whisper-style leading spaces."""
    return [{'start': s, 'end': e, 'word': w, 'probability': 0.9} for s, e, w in spec]


def speech(text, start=0.0, step=0.3):
    """Evenly timed words for a sentence."""
    result, t = [], start
    for token in text.split():
        result.append((round(t, 3), round(t + step - 0.05, 3), ' ' + token))
        t += step
    return result


def test_sentences_split_on_punctuation_and_pauses():
    spec = speech('Hello there.') + speech('How are you', start=0.7) + speech('today?', start=3.5)
    parts = segments.sentences(words_from(spec))
    texts = [''.join(w['word'] for w in p).strip() for p in parts]
    assert texts == ['Hello there.', 'How are you', 'today?']


def test_long_sentence_becomes_readable_cues_within_limits():
    text = ('This is a deliberately long sentence, which keeps going for quite a while so that it '
            'cannot possibly fit into one subtitle, and therefore it must be split into several cues.')
    cues, units = segments.build(words_from(speech(text, step=0.35)), 'en')
    assert len(cues) >= 2
    for cue in cues:
        assert len(cue.text) <= 84
        assert cue.end - cue.start <= segments.MAX_CUE_SECONDS + 1e-6
    assert ' '.join(c.text for c in cues) == text
    # A single sentence under MAX_UNIT_SECONDS of speech stays one narrator unit unless it is too long.
    assert sum(len(u.cue_ids) for u in units) == len(cues)
    for unit in units:
        assert unit.end - unit.start <= segments.MAX_UNIT_SECONDS + 0.4


def test_cues_never_overlap_and_have_minimum_display_time():
    spec = speech('Yes.', start=0.0) + speech('No.', start=0.5) + speech('Maybe later then.', start=1.1)
    cues, _ = segments.build(words_from(spec), 'en')
    for a, b in zip(cues, cues[1:]):
        assert a.end <= b.start - segments.CUE_GAP + 1e-6
    assert all(c.end > c.start for c in cues)


def test_short_replies_are_never_merged_into_the_next_line():
    # "Absolutely right!" and another character's "Michan!" once became one narrator line (v4, Bleach S01E01).
    spec = speech('Yes.', start=0.0) + speech('I will come with you tomorrow morning.', start=0.5)
    cues, units = segments.build(words_from(spec), 'en')
    assert len(cues) == 2
    assert [u.text for u in units] == ['Yes.', 'I will come with you tomorrow morning.']


def test_a_new_recognition_segment_starts_a_new_line_even_without_punctuation():
    first = words_from(speech('あれは一体何でしょうか', step=0.3))
    second = words_from(speech('はい真ん中のお前', start=first[-1]['end'] + 0.2))
    second[0]['segment_start'] = True
    assert len(segments.sentences(first + second)) == 2
    _, units = segments.build(first + second, 'ja')
    assert len(units) == 2


def test_cjk_text_is_joined_without_spaces():
    spec = [(0.0, 0.3, '你好'), (0.3, 0.6, '世界'), (0.6, 0.9, '。')]
    cues, units = segments.build(words_from(spec), 'zh')
    assert cues[0].text == '你好世界。'
    assert units[0].text == '你好世界。'


def test_empty_input():
    assert segments.build([], 'en') == ([], [])


def test_lines_from_the_uploads_subtitles_join_in_time_order_as_their_own_units():
    from vocnu_engine.segments import Cue, Unit, with_lines
    cues = [Cue(1, 0.5, 1.5, 'Hello.'), Cue(2, 3.0, 3.8, 'See you'), Cue(3, 3.8, 4.5, 'soon.')]
    units = [Unit(1, 0.5, 1.5, [1], 'Hello.'), Unit(2, 3.0, 4.5, [2, 3], 'See you soon.')]
    new_cues, new_units, added = with_lines(cues, units, [(2.0, 2.6, 'Wait!'), (6.0, 7.0, 'Ichigo... Run!')])
    assert [(c.id, c.text) for c in new_cues] == [(1, 'Hello.'), (2, 'Wait!'), (3, 'See you'), (4, 'soon.'),
                                                  (5, 'Ichigo... Run!')]
    assert [(u.id, u.cue_ids, u.text) for u in new_units] == [(1, [1], 'Hello.'), (2, [2], 'Wait!'),
                                                              (3, [3, 4], 'See you soon.'), (4, [5], 'Ichigo... Run!')]
    assert added == {2, 4}



def test_a_turn_starts_at_the_named_word_even_when_neighbours_share_its_time():
    from vocnu_engine.segments import build, turn_key
    # Recognition gave three words one instant (Bleach S01E01 at 4:56): only the named word starts a sentence.
    words = [{'start': 295.6, 'end': 295.9, 'word': '俺だって', 'segment_start': True},
             {'start': 296.58, 'end': 296.58, 'word': 'や'},
             {'start': 296.58, 'end': 296.58, 'word': 'った!'},
             {'start': 296.58, 'end': 296.8, 'word': 'もう'},
             {'start': 296.8, 'end': 297.5, 'word': 'やめなよ'}]
    _, units = build(words, 'ja', starts={turn_key(words[3])})
    assert [u.text for u in units] == ['俺だってやった!', 'もうやめなよ']
    assert units[1].words == words[3:]


def test_a_join_keeps_a_word_in_the_sentence_before_it():
    from vocnu_engine.segments import build, turn_key
    words = [{'start': 0.0, 'end': 0.4, 'word': '大正解', 'segment_start': True},
             {'start': 0.4, 'end': 0.5, 'word': 'め'},
             {'start': 0.5, 'end': 0.9, 'word': 'っちゃん', 'segment_start': True}]
    _, units = build(words, 'ja', starts={turn_key(words[1])}, joins={turn_key(words[2])})
    assert [u.text for u in units] == ['大正解', 'めっちゃん']
