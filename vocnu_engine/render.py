"""Final video: original picture, voice-over as the default audio track, original audio as a second track, and
switchable subtitle tracks (off by default) in containers that can hold them.

The container is the customer's choice (MP4, MKV, MOV, WebM, AVI). The picture is copied whenever the container
can hold its codec; otherwise, or when subtitles are burned in, it is re-encoded, which Rails prices as a
surcharge (backend/app/services/pricing.rb keeps the same codec sets as CONTAINERS).
"""
from dataclasses import dataclass
from pathlib import Path

from . import media
from .languages import BY_CODE, from_provider, track_tag

X264 = ('-c:v', 'libx264', '-preset', 'veryfast', '-crf', '20', '-pix_fmt', 'yuv420p')
# Realtime VP9 keeps a WebM render near playback speed on four cores; quality is fine for viewing.
VP9 = ('-c:v', 'libvpx-vp9', '-deadline', 'realtime', '-cpu-used', '8', '-row-mt', '1', '-crf', '32', '-b:v', '0',
       '-pix_fmt', 'yuv420p')
SUBTITLE_STYLE = ('FontName=Noto Sans,FontSize=18,PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,'
                  'BackColour=&H80000000,BorderStyle=1,Outline=1.3,Shadow=0.6,MarginV=16')


@dataclass(frozen=True)
class Container:
    copies: frozenset | None    # video codecs it holds as they are; None: any codec
    encoder: tuple              # video encoder when it cannot (or when subtitles are burned in)
    voice: tuple                # codec and bitrate of the voice-over track (from the mixed WAV)
    original: tuple             # codec and bitrate of the original audio track; ('copy',) keeps it as is
    subtitles: str | None       # subtitle codec; None when the container cannot hold text subtitles
    flags: tuple = ()


MP4_LIKE = dict(copies=frozenset({'h264', 'hevc', 'av1', 'mpeg4'}), encoder=X264, voice=('aac', '160k'),
                original=('aac', '160k'), subtitles='mov_text', flags=('-movflags', '+faststart'))
CONTAINERS = {
    'mp4': Container(**MP4_LIKE),
    'mov': Container(**MP4_LIKE),
    'mkv': Container(copies=None, encoder=X264, voice=('aac', '160k'), original=('copy',), subtitles='ass'),
    'webm': Container(copies=frozenset({'vp8', 'vp9', 'av1'}), encoder=VP9, voice=('libopus', '128k'),
                      original=('libopus', '128k'), subtitles='webvtt'),
    'avi': Container(copies=frozenset({'h264', 'mpeg4'}), encoder=X264, voice=('libmp3lame', '192k'),
                     original=('libmp3lame', '192k'), subtitles=None),
}
VIDEO_FORMATS = tuple(CONTAINERS)


def needs_reencode(video_format, video_codec, burned=False):
    """Whether the picture must be re-encoded for this container (or because subtitles are burned in)."""
    copies = CONTAINERS[video_format].copies
    return bool(burned) or (copies is not None and video_codec not in copies)


def _audio(index, spec):
    codec, *bitrate = spec
    return [f'-c:a:{index}', codec] + ([f'-b:a:{index}', bitrate[0]] if bitrate else [])


def video(source, voiceover_wav, destination, *, info, target, source_language, burn_srt=None, subtitle_tracks=(),
          threads=0, credit=None):
    """Write the voice-over video in the container named by `destination`'s suffix.

    `subtitle_tracks` are (path, language_code, title) tuples (SRT, or ASS for MKV, which keeps their styles),
    added as switchable tracks when the container can hold them (AVI cannot: its subtitles are the separate files).
    """
    destination = Path(destination)
    container = CONTAINERS[destination.suffix.lstrip('.')]
    codec = info.get('video_codec')
    tracks = list(subtitle_tracks) if container.subtitles else []
    args = ['ffmpeg', '-nostdin', '-v', 'error', '-y', '-i', Path(source).resolve(), '-i', Path(voiceover_wav).resolve()]
    for path, _, _ in tracks:
        args += ['-i', Path(path).resolve()]
    args += ['-map', '0:v:0', '-map', '1:a:0', '-map', f"0:{info['audio_index']}"]
    for index in range(len(tracks)):
        args += ['-map', f'{index + 2}:s:0']
    if burn_srt:
        burn_srt = Path(burn_srt)
        args += ['-vf', f"subtitles={burn_srt.name}:charenc=UTF-8:force_style='{SUBTITLE_STYLE}'",
                 *container.encoder, '-threads', threads]
    elif needs_reencode(destination.suffix.lstrip('.'), codec):
        args += [*container.encoder, '-threads', threads]
    else:
        args += ['-c:v', 'copy'] + (['-tag:v', 'hvc1'] if codec == 'hevc' and container.subtitles == 'mov_text' else [])
    args += _audio(0, container.voice) + _audio(1, container.original)
    voice_title = media.voice_track_title(BY_CODE[target].native, credit)
    # MP4/MOV players name tracks after the handler; ffmpeg before 7 (Debian's, our Modal image) keeps only that.
    mp4_like = container.subtitles == 'mov_text'
    args += ['-metadata:s:a:0', f'language={track_tag(target)}', '-metadata:s:a:0', f'title={voice_title}',
             '-metadata', f'comment={media.disclosure(credit)}',
             '-metadata:s:a:1', f'language={track_tag(from_provider(source_language))}',
             '-metadata:s:a:1', 'title=Original', '-disposition:a:0', 'default', '-disposition:a:1', '0']
    if mp4_like:
        args += ['-metadata:s:a:0', f'handler_name={voice_title}', '-metadata:s:a:1', 'handler_name=Original']
    if tracks:
        args += ['-c:s', container.subtitles]
        for index, (_, language, title) in enumerate(tracks):
            args += [f'-metadata:s:s:{index}', f'language={language}', f'-metadata:s:s:{index}', f'title={title}',
                     f'-disposition:s:{index}', '0']
            if mp4_like:
                args += [f'-metadata:s:s:{index}', f'handler_name={title}']
    # Chapters (an anime's OP/ED markers) are kept; other source metadata is not.
    args += ['-map_metadata', '-1', '-map_chapters', '0', *container.flags, destination.resolve()]
    _run_in(burn_srt.parent if burn_srt else None, args)
    return destination


def _run_in(cwd, args):
    """ffmpeg's subtitles filter needs a plain relative path, so burning runs inside the subtitle folder."""
    if cwd is None:
        return media.run(args)
    import subprocess
    result = subprocess.run([str(a) for a in args], cwd=str(cwd), capture_output=True, check=False)
    if result.returncode != 0:
        tail = result.stderr.decode(errors='replace').strip()[-1500:]
        raise media.MediaError(f'ffmpeg failed ({result.returncode}): {tail}')
    return result.stdout
