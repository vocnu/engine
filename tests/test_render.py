from pathlib import Path
import subprocess

import numpy as np
import pytest

from vocnu_engine import media, render


def test_reencoding_is_needed_only_when_the_container_cannot_hold_the_picture():
    assert not render.needs_reencode('mp4', 'h264') and not render.needs_reencode('mov', 'hevc')
    assert render.needs_reencode('mp4', 'vp9') and render.needs_reencode('webm', 'h264')
    assert not render.needs_reencode('webm', 'vp9') and not render.needs_reencode('avi', 'h264')
    assert render.needs_reencode('avi', 'hevc')
    assert not render.needs_reencode('mkv', 'vp9') and not render.needs_reencode('mkv', None)
    assert render.needs_reencode('mkv', 'h264', burned=True)


def _plan(monkeypatch, names, codec, burned=False):
    """[(format, picture format or None, burns)] render.videos would run, without ffmpeg."""
    calls = []

    def fake_video(source, wav, destination, *, info, burn_srt=None, subtitle_tracks=(), picture=None, **options):
        calls.append((destination.suffix[1:], picture and picture.suffix[1:], burn_srt is not None and picture is None))

    monkeypatch.setattr(render, 'video', fake_video)
    done = []
    made = render.videos('in.mp4', 'voice.wav', [(Path(f'video.lt.{name}'), []) for name in names],
                         info={'video_codec': codec}, burn_srt=Path('burn.srt') if burned else None,
                         rendered=done.append, target='lt', source_language='en')
    assert [path.suffix[1:] for path in made] == names and done == list(range(1, len(names) + 1))
    return calls


def test_several_formats_copy_the_source_picture_where_they_can(monkeypatch):
    # MP4 and MKV both hold H.264: two remuxes, nothing encoded.
    assert _plan(monkeypatch, ['mp4', 'mkv'], 'h264') == [('mp4', None, False), ('mkv', None, False)]
    # WebM cannot: its picture is encoded (VP9) while MP4 still copies the source's.
    assert _plan(monkeypatch, ['mp4', 'webm'], 'h264') == [('mp4', None, False), ('webm', None, False)]


def test_an_encoded_picture_is_made_once_and_copied_into_every_container_that_holds_it(monkeypatch):
    # Burned subtitles: MP4 encodes them once, MOV and MKV copy that picture.
    assert _plan(monkeypatch, ['mp4', 'mkv', 'mov'], 'h264', burned=True) == [
        ('mp4', None, True), ('mov', 'mp4', False), ('mkv', 'mp4', False)]
    # WebM needs its own codec; MKV, rendered last, copies one already made instead of encoding a third.
    assert _plan(monkeypatch, ['mkv', 'webm', 'avi'], 'h264', burned=True) == [
        ('webm', None, True), ('avi', None, True), ('mkv', 'webm', False)]
    # A VP9 upload: MP4 and AVI share one H.264 encode, WebM and MKV copy the source.
    assert _plan(monkeypatch, ['mp4', 'mkv', 'webm', 'avi'], 'vp9') == [
        ('mp4', None, False), ('webm', None, False), ('avi', 'mp4', False), ('mkv', None, False)]


def onset(path, threshold=0.2):
    """Seconds into the first audio track of `path` where the burst starts."""
    pcm = media.decode_array(path, channels=1, rate=media.SR)
    return int(np.argmax(np.abs(pcm) > threshold)) / media.SR


@pytest.mark.parametrize('suffix, audio, codec', [('mp4', 'm4a', 'aac'), ('webm', 'ogg', 'opus')])
def test_the_voice_over_encoded_for_the_job_is_copied_into_the_video_in_sync(monkeypatch, tmp_path, suffix, audio,
                                                                             codec):
    source = tmp_path / f'source.{suffix}'
    picture = ['libx264', '-pix_fmt', 'yuv420p'] if suffix == 'mp4' else ['libvpx-vp9', '-deadline', 'realtime']
    subprocess.run(['ffmpeg', '-v', 'error', '-f', 'lavfi', '-i', 'testsrc=size=160x120:rate=25:duration=4',
                    '-f', 'lavfi', '-i', 'sine=frequency=220:sample_rate=48000:duration=4', '-c:v', *picture,
                    '-c:a', 'aac' if suffix == 'mp4' else 'libopus', '-shortest', str(source)], check=True)
    pcm = np.zeros(4 * media.SR, dtype=np.float32)
    pcm[2 * media.SR:2 * media.SR + media.SR // 20] = 0.5   # a burst 2.0 s in
    wav = media.write_wav(tmp_path / 'voiceover.wav', np.stack([pcm, pcm], axis=1))
    track = media.encode_audio(wav, tmp_path / f'voiceover.lt.{audio}')
    calls, run = [], media.run

    def recorded(args, **options):
        calls.append([str(a) for a in args])
        return run(args, **options)

    monkeypatch.setattr(media, 'run', recorded)
    out = render.video(source, wav, tmp_path / f'video.lt.{suffix}', info=media.inspect(source), target='lt',
                       source_language='en', voice_track=track)
    args = calls[-1]
    # Both audio tracks are copied: the voice-over from the job's audio file, the original as it was.
    assert args[args.index('-c:a:0') + 1] == 'copy' and args[args.index('-c:a:1') + 1] == 'copy'
    assert args[args.index('-i', args.index('-i') + 1) + 1] == str(track.resolve())
    streams = [s for s in media.inspect(out)['streams'] if s['codec_type'] == 'audio']
    assert [s['codec_name'] for s in streams] == [codec, codec]
    assert abs(onset(out) - onset(track)) < 0.005 and abs(onset(out) - 2.0) < 0.01


def test_a_voice_over_in_another_codec_than_the_containers_is_encoded_from_the_mix(tmp_path):
    assert render.encoded_as(tmp_path / 'voiceover.lt.m4a') == render.CONTAINERS['mp4'].voice
    assert render.encoded_as(tmp_path / 'voiceover.lt.mp3') == render.CONTAINERS['avi'].voice
    assert render.encoded_as(tmp_path / 'voiceover.lt.ogg') == render.CONTAINERS['webm'].voice
    assert render.encoded_as(tmp_path / 'voiceover.lt.wav') != render.CONTAINERS['mp4'].voice
    assert render.encoded_as(None) is None


def _clip(path, *, audio_codec, offset=0.0):
    """A 4 s clip whose audio is silent but for a burst 1.5 s into its own track, `offset` s after the picture."""
    picture = ['-c:v', 'libvpx-vp9', '-deadline', 'realtime'] if path.suffix == '.webm' else \
        ['-c:v', 'libx264', '-pix_fmt', 'yuv420p']
    subprocess.run(['ffmpeg', '-v', 'error', '-f', 'lavfi', '-i', 'testsrc=size=160x120:rate=25:duration=4',
                    '-itsoffset', str(offset), '-f', 'lavfi', '-i',
                    "aevalsrc='if(between(t,1.5,1.55),0.5*sin(2*PI*440*t),0)':s=48000:d=4", *picture,
                    '-c:a', audio_codec, str(path)], check=True)
    return path


def _original(monkeypatch, tmp_path, source):
    calls, run = [], media.run

    def recorded(args, **options):
        calls.append([str(a) for a in args])
        return run(args, **options)

    monkeypatch.setattr(media, 'run', recorded)
    out = render.original_audio(source, tmp_path / 'original.en.m4a', info=media.inspect(source),
                                source_language='en')
    args = calls[-1]
    return out, args[args.index('-c:a') + 1] if '-c:a' in args else args[args.index('-c:a:0') + 1]


def test_the_original_soundtrack_is_copied_on_its_own_when_it_is_aac(monkeypatch, tmp_path):
    out, codec = _original(monkeypatch, tmp_path, _clip(tmp_path / 'source.mp4', audio_codec='aac'))
    assert codec == 'copy'
    streams = media.inspect(out)['streams']
    assert [(s['codec_type'], s['codec_name']) for s in streams] == [('audio', 'aac')]
    tags = {key.lower(): value for key, value in streams[0]['tags'].items()}
    assert tags['language'] == 'eng' and tags['handler_name'] == 'Original'
    # Not a voice-over: nothing in it is synthetic, so it carries no AI disclosure.
    assert 'comment' not in {key.lower() for key in (media._probe_json(out, 30).get('format') or {}).get('tags', {})}
    assert abs(onset(out) - 1.5) < 0.01


def test_an_original_soundtrack_in_another_codec_is_encoded_to_aac_in_sync(monkeypatch, tmp_path):
    out, codec = _original(monkeypatch, tmp_path, _clip(tmp_path / 'source.webm', audio_codec='libopus'))
    assert codec == 'aac'
    assert [(s['codec_name'], s['channels']) for s in media.inspect(out)['streams']] == [('aac', 2)]
    assert abs(onset(out) - 1.5) < 0.01


def test_an_original_soundtrack_that_starts_after_the_picture_keeps_its_place_on_the_media_timeline(monkeypatch,
                                                                                                    tmp_path):
    # Copied as it is, the burst would play 1.5 s into the file instead of 2.0 s into the media.
    out, codec = _original(monkeypatch, tmp_path, _clip(tmp_path / 'source.mkv', audio_codec='aac', offset=0.5))
    assert codec == 'aac'
    assert abs(onset(out) - 2.0) < 0.01
