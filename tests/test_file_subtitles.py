import subprocess

from vocnu_engine import file_subtitles

ASS = r"""[Script Info]
ScriptType: v4.00+

[V4+ Styles]
Format: Name, Fontname, Fontsize
Style: Default,Arial,20
Style: OP-Romaji,Arial,20
Style: Sign,Arial,20

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
Dialogue: 0,0:00:01.00,0:00:03.00,Default,,0,0,0,,{\i1}Ichigo!{\i0} Wait\Nfor me.
Dialogue: 1,0:00:01.00,0:00:03.00,Default,,0,0,0,,{\i1}Ichigo!{\i0} Wait\Nfor me.
Dialogue: 0,0:00:04.00,0:00:05.00,Default,,0,0,0,,{\p1}m 0 0 l 100 0 100 100{\p0}
Dialogue: 0,0:00:06.00,0:00:08.00,OP-Romaji,,0,0,0,,Hibiku hon no naka e
Dialogue: 0,0:00:09.00,0:00:10.00,Sign,,0,0,0,,KARAKURA TOWN
Dialogue: 0,0:00:11.00,0:00:12.50,Default,,0,0,0,,Who are you, a shinigami?
"""


def test_ass_events_are_cleaned_and_song_or_sign_styles_become_overlay_lines():
    assert file_subtitles.parse_ass(ASS) == [(1.0, 3.0, 'Ichigo! Wait for me.', 'dialogue'),
                                             (6.0, 8.0, 'Hibiku hon no naka e', 'overlay'),
                                             (9.0, 10.0, 'KARAKURA TOWN', 'overlay'),
                                             (11.0, 12.5, 'Who are you, a shinigami?', 'dialogue')]


def test_candidates_read_text_and_ocr_able_bitmap_tracks_but_not_forced_or_commentary():
    streams = [
        {'index': 0, 'codec_type': 'video'},
        {'index': 4, 'codec_type': 'subtitle', 'codec_name': 'hdmv_pgs_subtitle', 'tags': {'language': 'eng'}},
        {'index': 5, 'codec_type': 'subtitle', 'codec_name': 'hdmv_pgs_subtitle',
         'tags': {'language': 'eng', 'title': 'Signs & Songs'}},
        {'index': 6, 'codec_type': 'subtitle', 'codec_name': 'subrip', 'tags': {'language': 'lit'}},
        {'index': 7, 'codec_type': 'subtitle', 'codec_name': 'ass', 'tags': {'language': 'eng', 'title': 'Full'}},
        {'index': 8, 'codec_type': 'subtitle', 'codec_name': 'subrip', 'tags': {'language': 'eng'},
         'disposition': {'forced': 1}},
        {'index': 9, 'codec_type': 'subtitle', 'codec_name': 'subrip', 'tags': {'language': 'spa'},
         'disposition': {'default': 1}},
        {'index': 10, 'codec_type': 'subtitle', 'codec_name': 'subrip',
         'tags': {'language': 'eng', 'title': 'Commentary'}},
        {'index': 11, 'codec_type': 'subtitle', 'codec_name': 'hdmv_pgs_subtitle', 'tags': {'language': 'kor'}},
    ]
    found = file_subtitles.candidates(streams, ocr_languages={'eng'})
    # Dialogue tracks: default first, text before bitmaps (at most three); then signs/songs tracks.
    assert [(t['index'], t['kind'], t['bitmap']) for t in found] == [
        (9, 'dialogue', False), (6, 'dialogue', False), (7, 'dialogue', False), (5, 'overlay', True)]
    # Without OCR, bitmap tracks are left out.
    assert all(not t['bitmap'] for t in file_subtitles.candidates(streams))


def test_the_originals_language_and_the_targets_official_translation_are_read_first():
    # A film with thirty text tracks: English, English SDH, then Arabic, Bulgarian, ... and Lithuanian at index 18.
    tags = ['eng', 'eng', 'ara', 'bul', 'cze', 'dan', 'ger', 'gre', 'spa', 'est', 'fin', 'fre', 'hrv', 'hun', 'ice',
            'ita', 'lit', 'lav']
    streams = [{'index': 0, 'codec_type': 'video'}, {'index': 1, 'codec_type': 'audio', 'tags': {'language': 'eng'}}]
    streams += [{'index': i, 'codec_type': 'subtitle', 'codec_name': 'subrip', 'tags': {'language': tag}}
                for i, tag in enumerate(tags, 2)]
    assert [t['index'] for t in file_subtitles.candidates(streams)] == [2, 3, 4]
    found = file_subtitles.candidates(streams, prefer=('en', None, 'lt'))
    assert [(t['index'], t['language']) for t in found] == [(2, 'eng'), (3, 'eng'), (18, 'lit')]
    # Two-letter tags count as the same language; the target's track still comes after the original's.
    streams[-2]['tags']['language'] = 'lt'
    assert [t['index'] for t in file_subtitles.candidates(streams, prefer=('lt', 'en'))] == [18, 2, 3]


def track(language, kind, cues, bitmap=True):
    return {'language': language, 'kind': kind, 'bitmap': bitmap, 'cues': cues}


def test_layers_split_dialogue_from_song_and_sign_lines():
    song = (2.1, 5.8, 'Raise your eyes', 'overlay')
    full = track('eng', 'dialogue', [song, (149.1, 151.9, 'I sense a strong Spirit Energy...', 'dialogue'),
                                     (162.7, 164.0, 'What the hell?', 'dialogue'),
                                     # moved up for on-screen text, not on the signs track: still dialogue
                                     (300.0, 301.0, 'Over here!', 'overlay')])
    signs = track('eng', 'overlay', [(2.13, 5.8, 'Raise your eyes', 'overlay'),
                                     (129.3, 131.7, 'WE TREMBLE IN AWE', 'overlay')])
    german = track('ger', 'overlay', [(129.3, 131.7, 'WIR ERZITTERN', 'overlay')])
    dialogue, (language, overlay) = file_subtitles.layers([full, signs, german])
    assert dialogue == [{'language': 'eng', 'cues': [(149.1, 151.9, 'I sense a strong Spirit Energy...'),
                                                     (162.7, 164.0, 'What the hell?'),
                                                     (300.0, 301.0, 'Over here!')]}]
    # One language of overlay lines, the song shown once although both English tracks carry it.
    assert language == 'eng'
    assert [text for _, _, text in overlay] == ['Raise your eyes', 'WE TREMBLE IN AWE']


def test_without_a_signs_track_upper_lines_of_a_bitmap_track_are_overlay():
    full = track('eng', 'dialogue', [(2.1, 5.8, 'Raise your eyes', 'overlay'), (7.0, 8.0, 'Hey!', 'dialogue')])
    dialogue, (language, overlay) = file_subtitles.layers([full])
    assert dialogue == [{'language': 'eng', 'cues': [(7.0, 8.0, 'Hey!')]}]
    assert (language, overlay) == ('eng', [(2.1, 5.8, 'Raise your eyes')])


def test_tracks_are_read_from_a_real_file(tmp_path):
    srt = tmp_path / 'subs.srt'
    srt.write_text('1\n00:00:01,000 --> 00:00:02,500\nKurosaki Ichigo.\n\n2\n00:00:03,000 --> 00:00:04,000\nShinigami!\n')
    clip = tmp_path / 'clip.mkv'
    subprocess.run(['ffmpeg', '-v', 'error', '-f', 'lavfi', '-i', 'sine=frequency=220:duration=5', '-i', str(srt),
                    '-map', '0:a', '-map', '1:s', '-c:a', 'aac', '-c:s', 'srt', '-metadata:s:s:0', 'language=lit',
                    str(clip)], check=True)
    streams = [{'index': 0, 'codec_type': 'audio'},
               {'index': 1, 'codec_type': 'subtitle', 'codec_name': 'subrip', 'tags': {'language': 'lit'}}]
    # A track in the target language is evidence too (v4 read every track).
    [found] = file_subtitles.read(clip, streams)
    assert (found['language'], found['kind']) == ('lit', 'dialogue')
    # Muxing may shift times by a frame or two depending on the ffmpeg build (CI's moves them 0.02 s).
    assert [cue[2:] for cue in found['cues']] == [('Kurosaki Ichigo.', 'dialogue'), ('Shinigami!', 'dialogue')]
    expected = [(1.0, 2.5), (3.0, 4.0)]
    assert all(abs(a - x) < 0.1 and abs(b - y) < 0.1 for (a, b, *_), (x, y) in zip(found['cues'], expected))
    assert file_subtitles.overlapping(found['cues'], 0.8, 2.0) == 'Kurosaki Ichigo.'
