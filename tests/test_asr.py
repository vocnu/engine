from types import SimpleNamespace

import pytest

from vocnu_engine import asr


def segment(text, start, end, words=None, no_speech_prob=0.1, avg_logprob=-0.3):
    words = words if words is not None else [SimpleNamespace(word=text, start=start, end=end, probability=0.9)]
    return SimpleNamespace(text=text, start=start, end=end, words=words, no_speech_prob=no_speech_prob,
                           avg_logprob=avg_logprob)


def spread(pieces):
    """Words at the given (text, start, end) times, like Whisper's timestamps for an invented sign-off."""
    return [SimpleNamespace(word=text, start=start, end=end, probability=0.9) for text, start, end in pieces]


def test_sign_off_stretched_over_music_is_dropped():
    # Bleach S01E01: Whisper "heard" this over 16 s of the ending song.
    words = spread([('ご', 1328.96, 1330.36), ('視', 1336.29, 1337.69), ('聴', 1340.33, 1341.33),
                    ('ありがとうございました', 1343.74, 1345.12)])
    assert asr._hallucinated(segment('ご視聴ありがとうございました', 1328.96, 1345.12, words))


def test_sign_off_said_at_a_normal_pace_is_kept():
    assert not asr._hallucinated(segment(' Thanks for watching!', 600.0, 601.3))
    assert not asr._hallucinated(segment('ご視聴ありがとうございました。', 10.0, 12.0))


def test_doubtful_sign_off_is_dropped_by_whispers_own_scores():
    assert asr._hallucinated(segment(' Thanks for watching!', 600.0, 601.3, no_speech_prob=0.7))
    assert asr._hallucinated(segment(' Thanks for watching!', 600.0, 601.3, avg_logprob=-1.1))


def test_subtitle_credits_are_always_dropped():
    assert asr._hallucinated(segment(' Subtitles by the Amara.org community', 5.0, 7.0))
    assert asr._hallucinated(segment('字幕由Amara.org社区提供', 5.0, 7.0))


def test_ordinary_speech_and_longer_lines_are_kept():
    assert not asr._hallucinated(segment(' Thank you.', 5.0, 9.0))
    assert not asr._hallucinated(segment(" That's all for today, thanks for watching!", 5.0, 30.0))


def test_transcribe_leaves_out_invented_segments(monkeypatch):
    real = segment(' Hello there.', 1.0, 2.0)
    invented = segment(' Thank you for watching!', 50.0, 58.0,
                       spread([(' Thank', 50.0, 51.0), (' you', 53.0, 54.0), (' for', 55.0, 56.0),
                               (' watching!', 57.0, 58.0)]))

    class Model:
        def transcribe(self, audio, **options):
            return iter([real, invented]), SimpleNamespace(language='en', language_probability=0.99, duration=60.0)

    monkeypatch.setattr(asr, 'load_model', lambda download_root=None: (Model(), 'cpu'))
    monkeypatch.setattr(asr.media, 'decode_array', lambda path, channels, rate: [])
    monkeypatch.setattr(asr, '_speech', lambda audio, vad=None: [(1.0, 2.0)])
    result = asr.transcribe('clip.opus', 'en')
    assert [s['text'] for s in result['segments']] == ['Hello there.']
    assert [w['word'] for w in result['words']] == [' Hello there.']
    assert result['repairs'] == []


def row(start, end, text, words=None):
    return {'start': start, 'end': end, 'text': text, 'words': words if words is not None else
            [{'start': start, 'end': end, 'word': text}]}


def test_suspect_spans_find_loops_sparse_segments_and_unheard_speech():
    loop = [row(100 + i * 3, 102 + i * 3, 'ご視聴') for i in range(6)]
    sparse = [row(200, 224, 'あ')]
    fine = [row(300, 302, 'こんにちは'), row(303, 305, 'こんにちは')]
    speech = [(100, 118), (200, 224), (300, 305), (400, 403), (500, 500.5)]
    spans = asr.suspect_spans(loop + sparse + fine, speech, 600)
    assert [(round(a), round(b), why) for a, b, why in spans] == [
        (100, 117, 'repetition'), (200, 224, 'sparse'), (400, 403, 'unheard')]


class Recogniser:
    """A model that answers the repair pass with given segments, recording what it was asked."""

    def __init__(self, answers):
        self.answers = list(answers)
        self.calls = []

    def transcribe(self, audio, **options):
        self.calls.append((len(audio), options))
        return iter(self.answers.pop(0)), SimpleNamespace(language='ja')


def test_repair_replaces_a_loop_and_fills_unheard_speech_but_never_deletes_on_a_heuristic(monkeypatch):
    import numpy as np
    audio = np.zeros(16000 * 600, dtype=np.float32)
    loop = [dict(row(100 + i * 3, 102 + i * 3, 'ご視聴'), no_speech_prob=0.0, avg_logprob=-0.2,
                 segment=segment('ご視聴', 100 + i * 3, 102 + i * 3)) for i in range(6)]
    monkeypatch.setattr(asr, '_speech', lambda audio, vad=None: [(100, 118), (400, 403), (450, 452)])
    model = Recogniser([[segment('本当のセリフ', 1.0, 4.0)], [segment('おい!', 0.5, 1.5)], []])
    rows, repairs = asr._repair(model, audio, loop, {'language': 'ja'}, 600)
    assert [r['text'] for r in rows] == ['本当のセリフ', 'おい!']
    assert rows[0]['start'] == 101.0 and rows[1]['start'] == 400.2
    assert [(r['reason'], r['segments']) for r in repairs] == [('repetition', 1), ('unheard', 1)]


def test_sound_labels_are_not_dialogue():
    for label in ('[Music]', ' (laughs) ', '*sighs*', '（拍手）', '[音楽] ♪', '【笑い】'):
        assert asr._is_noise(label), label
    for speech in ('Hello (quietly) there.', 'ご飯、冷めちゃうよ。', '[Music] Hello there.', 'Ah!'):
        assert not asr._is_noise(speech), speech


def test_nearby_suspect_stretches_share_one_window_and_unheard_speech_never_duplicates(monkeypatch):
    import numpy as np
    assert asr._merge([(162.4, 169.4, 'unheard'), (168.8, 173.7, 'unheard'), (174.3, 183.1, 'unheard'),
                       (240.0, 242.0, 'unheard')]) == [(162.4, 183.1, {'unheard'}), (240.0, 242.0, {'unheard'})]
    audio = np.zeros(16000 * 300, dtype=np.float32)
    existing = [dict(row(170.0, 172.0, 'もうある'), no_speech_prob=0.0, avg_logprob=-0.2,
                     segment=segment('もうある', 170.0, 172.0))]
    monkeypatch.setattr(asr, '_speech', lambda audio, vad=None: [(162.4, 169.4), (174.3, 183.1)])
    # The window starts at 162.1 (0.3 s padding): new lines at 163-165 and 179-181, plus 170.1-171.9 already known.
    model = Recogniser([[segment('新しい', 0.9, 2.9), segment('もうある', 8.0, 9.8), segment('ほかの', 16.9, 18.9)]])
    rows, repairs = asr._repair(model, audio, existing, {'language': 'ja'}, 300)
    assert [r['text'] for r in rows] == ['新しい', 'もうある', 'ほかの']
    assert len(model.calls) == 1 and repairs[0]['segments'] == 2


def test_low_confidence_reading_is_replaced_only_by_a_more_confident_one(monkeypatch):
    import numpy as np
    audio = np.zeros(16000 * 100, dtype=np.float32)
    weak = dict(row(10.0, 13.0, 'ぼんやり', [{'start': 10.0, 'end': 13.0, 'word': 'ぼんやり', 'probability': 0.2}]),
                no_speech_prob=0.0, avg_logprob=-0.9, segment=segment('ぼんやり', 10.0, 13.0))
    monkeypatch.setattr(asr, '_speech', lambda audio, vad=None: [(10.0, 13.0)])
    assert [why for _, _, why in asr.suspect_spans([weak], [(10.0, 13.0)], 100)] == ['uncertain']
    better = segment('はっきり', 0.0, 3.0)
    better.words[0].probability = 0.9
    rows, _ = asr._repair(Recogniser([[better]]), audio, [weak], {'language': 'ja'}, 100)
    assert [r['text'] for r in rows] == ['はっきり']
    worse = segment('もっと曖昧', 0.0, 3.0)
    worse.words[0].probability = 0.1
    rows, _ = asr._repair(Recogniser([[worse]]), audio, [weak], {'language': 'ja'}, 100)
    assert [r['text'] for r in rows] == ['ぼんやり']


def test_zero_length_segments_are_artefacts():
    tiny = dict(row(459.5, 459.52, 'ん?'), no_speech_prob=0.0, avg_logprob=-0.2, segment=segment('ん?', 459.5, 459.52))
    real = dict(row(456.0, 456.4, 'ん?'), no_speech_prob=0.0, avg_logprob=-0.2, segment=segment('ん?', 456.0, 456.4))
    assert not asr._keep(tiny) and asr._keep(real)


def test_long_media_is_cut_in_pauses_into_pieces_of_about_six_minutes():
    assert asr.plan_chunks([(0, 100)], 400) == [(0.0, 400)]          # short: one piece
    # Speech everywhere except pauses at 355-358 s and 700-702 s.
    speech = [(0, 355), (358, 700), (702, 1380)]
    pieces = asr.plan_chunks(speech, 1380)
    assert len(pieces) == 4 and pieces[0][0] == 0.0 and pieces[-1][1] == 1380
    assert pieces[0][1] == 356.5 and pieces[1][1] == 701.0      # the pause midpoints near 345 s and 690 s
    assert all(b > a for a, b in pieces) and all(p[1] == q[0] for p, q in zip(pieces, pieces[1:]))


def test_pieces_recognised_in_parallel_merge_with_their_offsets():
    def part(word, seconds):
        return {'language': 'ja', 'language_probability': 0.99, 'duration': 360.0, 'device': 'cpu',
                'model': asr.MODEL, 'seconds': seconds, 'repairs': [{'reason': 'unheard', 'start': 1.0, 'end': 2.0}],
                'segments': [{'start': 1.0, 'end': 2.0, 'text': word}],
                'words': [{'start': 1.0, 'end': 2.0, 'word': word, 'probability': 0.9, 'segment_start': True}]}
    merged = asr.merge([(0.0, part('一', 100)), (356.5, part('二', 120))], 130)
    assert [(w['start'], w['word']) for w in merged['words']] == [(1.0, '一'), (357.5, '二')]
    assert merged['words'][1]['segment_start'] is True
    assert merged['repairs'][1]['start'] == 357.5 and merged['duration'] == 716.5
    assert (merged['seconds'], merged['compute_seconds'], merged['pieces']) == (130, 220, 2)


def test_language_detection_asks_the_whisper_model_not_its_ctranslate2_model(monkeypatch):
    class Whisper:
        model = 'raw ctranslate2 model, not to be called'

        def detect_language(self, audio, **options):
            assert options['vad_filter'] is True
            return 'ja', 0.97, []

    monkeypatch.setattr(asr.media, 'decode_array', lambda path, channels, rate: [0.0])
    for device in ('cpu', 'cuda'):
        monkeypatch.setattr(asr, 'load_model', lambda download_root=None: (Whisper(), device))
        assert asr.detect_language('clip.opus') == ('ja', 0.97)


def test_both_devices_read_with_the_same_sequential_settings(monkeypatch):
    import numpy as np

    class Model:
        def __init__(self):
            self.calls = []

        def transcribe(self, audio, **options):
            self.calls.append(options)
            return iter([segment('こんにちは', 0.5, 1.5)]), SimpleNamespace(language='ja', language_probability=0.99,
                                                                        duration=5.0)

    model = Model()
    monkeypatch.setattr(asr.media, 'decode_array', lambda path, channels, rate: np.zeros(16000 * 5, np.float32))
    monkeypatch.setattr(asr, '_repair', lambda model, audio, rows, options, duration, hints=(): (rows, []))
    for device in ('cpu', 'cuda'):
        monkeypatch.setattr(asr, 'load_model', lambda download_root=None: (model, device))
        assert asr.transcribe('clip.opus', 'ja')['device'] == device
    assert model.calls[0] == model.calls[1]
    assert 'batch_size' not in model.calls[0] and model.calls[0]['condition_on_previous_text'] is False


def test_subtitle_hints_with_no_word_heard_are_read_again_with_the_sensitive_vad(monkeypatch):
    import numpy as np
    audio = np.zeros(16000 * 100, dtype=np.float32)
    heard = [dict(row(10.0, 12.0, 'もしもし'), no_speech_prob=0.0, avg_logprob=-0.2,
                  segment=segment('もしもし', 10.0, 12.0))]
    # A subtitle over heard speech is fine; one at 40-41.5 s (a shout under music) had no word.
    assert asr.suspect_spans(heard, [], 100, hints=[(10.2, 11.8), (40.0, 41.5)]) == [(39.7, 41.8, 'hinted')]
    monkeypatch.setattr(asr, '_speech', lambda audio, vad=None: [])
    model = Recogniser([[segment('待て!', 0.5, 1.4)]])
    rows, repairs = asr._repair(model, audio, heard, {'language': 'ja', 'vad_parameters': asr.VAD}, 100,
                                hints=[(10.2, 11.8), (40.0, 41.5)])
    assert [r['text'] for r in rows] == ['もしもし', '待て!']
    assert [r['reason'] for r in repairs] == ['hinted']
    assert model.calls[0][1]['vad_parameters'] == asr.SENSITIVE_VAD


def test_hints_follow_their_piece_of_parallel_recognition():
    assert asr.piece_hints([(10.0, 12.0), (359.0, 362.0), (500.0, 501.0)], 360.0, 720.0) == [
        (0.0, 2.0), (140.0, 141.0)]


def test_a_line_missed_inside_a_long_segment_is_added_but_words_already_heard_are_not(monkeypatch):
    import numpy as np
    audio = np.zeros(16000 * 1000, dtype=np.float32)
    words = [{'start': 940.0, 'end': 941.0, 'word': 'カリン'}, {'start': 948.0, 'end': 950.0, 'word': '逃げろ'}]
    long_row = dict(row(940.0, 950.0, 'カリン 逃げろ', words), no_speech_prob=0.0, avg_logprob=-0.3,
                    segment=segment('カリン 逃げろ', 940.0, 950.0))
    monkeypatch.setattr(asr, '_speech', lambda audio, vad=None: [])
    # The window starts at 942.7: a new shout at 943.2-944.0 inside the long segment, and 'カリン' heard again.
    reread = segment('おい!', 0.5, 1.3)
    again = segment('カリン', -2.7, -1.7)
    model = Recogniser([[again, reread]])
    rows, repairs = asr._repair(model, audio, [long_row], {'language': 'ja'}, 1000, hints=[(943.0, 944.0)])
    assert [r['text'] for r in rows] == ['カリン 逃げろ', 'おい!']
    assert repairs[0]['reason'] == 'hinted' and repairs[0]['segments'] == 1


def test_lines_the_subtitles_show_come_first_in_the_repair_budget(monkeypatch):
    import numpy as np
    audio = np.zeros(16000 * 100, dtype=np.float32)
    # 25 s of VAD "speech" without words (a song at the sensitive threshold) and one subtitle line later on: the
    # budget (a quarter of 100 s) covers the line first and then has too little left for the song.
    monkeypatch.setattr(asr, '_speech', lambda audio, vad=None: [(10.0, 34.5)])
    model = Recogniser([[segment('待て!', 0.4, 1.2)]])
    rows, repairs = asr._repair(model, audio, [], {'language': 'ja'}, 100, hints=[(60.0, 61.0)])
    assert [r['reason'] for r in repairs] == ['hinted'] and [r['text'] for r in rows] == ['待て!']


def test_a_sign_off_found_only_by_the_sensitive_reading_is_dropped(monkeypatch):
    import numpy as np
    audio = np.zeros(16000 * 100, dtype=np.float32)
    monkeypatch.setattr(asr, '_speech', lambda audio, vad=None: [(27.0, 31.0)])
    invented = segment('ご視聴ありがとうございました', 0.7, 3.5)
    rows, repairs = asr._repair(Recogniser([[invented]]), audio, [], {'language': 'ja'}, 100)
    assert rows == [] and repairs == []


def test_a_decode_stuck_in_a_loop_is_not_kept_but_chants_and_sentences_are():
    def kept(text):
        return asr._keep(dict(row(10.0, 14.0, text), no_speech_prob=0.0, avg_logprob=-0.2,
                              segment=segment(text, 10.0, 14.0)))
    assert not kept('ファースト' * 44)
    assert not kept('ハァ' + 'ァ' * 120)
    assert kept('当たり前だろ。地獣すぎてこんな幼稚なコミュニケーション手段しか持ったん父親になんて誰が…')
    assert kept('ファイト！ファイト！ファイト！') and kept('No, no, no, no, no, no!')
