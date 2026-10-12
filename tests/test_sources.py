import pytest

from vocnu_engine import sources
from vocnu_engine.errors import WorkerError
from vocnu_engine.storage import check_upload_key


@pytest.mark.parametrize('source', [
    {'kind': 'youtube', 'url': 'https://www.youtube.com/watch?v=dQw4w9WgXcQ'},
    {'kind': 'link', 'url': 'https://vimeo.com/123456'},
    {},
    None,
])
def test_only_uploads_and_local_files_are_sources(source, tmp_path):
    with pytest.raises(WorkerError) as probed:
        sources.probe(source)
    with pytest.raises(WorkerError) as fetched:
        sources.fetch(source, tmp_path)
    assert probed.value.code == fetched.value.code == 'not_found'


def test_upload_keys_must_stay_inside_uploads():
    assert check_upload_key('uploads/12/9f1c-aa/talk.mp4') == 'uploads/12/9f1c-aa/talk.mp4'
    for bad in ('jobs/1/x.mp4', 'uploads/12/../../etc/passwd', 'uploads/12/x', '/uploads/1/2/a.mp4', None):
        with pytest.raises(WorkerError):
            check_upload_key(bad)


def test_the_source_speech_is_the_original_track_not_a_dub_in_the_target_language():
    from vocnu_engine import media
    streams = [{'index': 0, 'codec_type': 'video'},
               {'index': 1, 'codec_type': 'audio', 'tags': {'language': 'eng', 'title': 'Stereo'}, 'disposition': {}},
               {'index': 2, 'codec_type': 'audio', 'tags': {'language': 'jpn'}, 'disposition': {}},
               {'index': 3, 'codec_type': 'audio', 'tags': {'language': 'lit', 'title': 'Lietuvių (įgarsinimas)'},
                'disposition': {'default': 1}},
               {'index': 4, 'codec_type': 'audio', 'tags': {'language': 'jpn', 'title': 'Commentary'},
                'disposition': {}}]
    assert media.audio_stream_index(streams) == 3
    # Bleach: English dub, Japanese original, Lithuanian dub (default) -> the Japanese original.
    assert media.audio_stream_index(streams, ('lt', 'lit')) == 2
    assert media.audio_stream_index(streams, ('lt', 'lit'), ('en', 'eng')) == 1   # the customer named English
    english_original = [streams[0], dict(streams[1], disposition={'default': 1}), streams[2]]
    assert media.audio_stream_index(english_original, ('lt', 'lit')) == 1
    only_dub = [streams[0], streams[3]]
    assert media.audio_stream_index(only_dub, ('lt', 'lit')) == 3


def test_every_other_dub_is_recognised_as_evidence_one_per_language():
    from vocnu_engine import media
    streams = [{'index': 0, 'codec_type': 'video'},
               {'index': 1, 'codec_type': 'audio', 'tags': {'language': 'eng', 'title': 'Stereo'}},
               {'index': 2, 'codec_type': 'audio', 'tags': {'language': 'jpn', 'title': 'Stereo'}},
               {'index': 3, 'codec_type': 'audio', 'tags': {'language': 'lit', 'title': 'Lietuvių (įgarsinimas)'},
                'disposition': {'default': 1}},
               {'index': 4, 'codec_type': 'audio', 'tags': {'language': 'en', 'title': 'Commentary'}},
               {'index': 5, 'codec_type': 'audio', 'tags': {'language': 'en'}}]
    # Bleach: the Japanese original, then the English and the Lithuanian dubs (the target language too, as v4 did).
    assert media.dub_tracks(streams, 2) == [{'index': 1, 'tag': 'eng'}, {'index': 3, 'tag': 'lit'}]
    assert media.dub_tracks(streams, 1) == [{'index': 2, 'tag': 'jpn'}, {'index': 3, 'tag': 'lit'}]
    assert media.dub_tracks(streams[:3], 2, limit=1) == [{'index': 1, 'tag': 'eng'}]
