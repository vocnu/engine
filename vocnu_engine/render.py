"""Final video: original picture, voice-over as the default audio track, original audio as a second track, and
switchable subtitle tracks (off by default) in containers that can hold them.

The containers are the customer's choice (MP4, MKV, MOV, WebM, AVI; one or several, each its own file). The picture
is copied whenever the container can hold its codec; otherwise, or when subtitles are burned in, it is re-encoded,
which Rails prices as a surcharge, once however many containers need it (backend/app/services/pricing.rb keeps the
same codec sets as CONTAINERS). A picture is encoded once per codec: every further container that holds it copies it.
The voice-over is copied from the job's audio file when that is already in the container's codec, and the original
audio when the container holds its codec: encoding both again was nearly all of the 154 s a 44-minute documentary
spent rendering on Modal, and on four cores its render took 2 s copied against 170 s encoded, in sync to the
millisecond (2026-10-20).
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
    encodes: str                # the codec that encoder writes
    voice: tuple                # codec and bitrate of the voice-over track (from the mixed WAV)
    original: tuple             # codec and bitrate of the original audio track; ('copy',) keeps it as is
    subtitles: str | None       # subtitle codec; None when the container cannot hold text subtitles
    flags: tuple = ()
    keeps: frozenset = frozenset()   # original audio codecs copied as they are instead of encoded as `original`


MP4_LIKE = dict(copies=frozenset({'h264', 'hevc', 'av1', 'mpeg4'}), encoder=X264, encodes='h264',
                voice=('aac', '160k'), original=('aac', '160k'), subtitles='mov_text',
                flags=('-movflags', '+faststart'), keeps=frozenset({'aac'}))
CONTAINERS = {
    'mp4': Container(**MP4_LIKE),
    'mov': Container(**MP4_LIKE),
    'mkv': Container(copies=None, encoder=X264, encodes='h264', voice=('aac', '160k'), original=('copy',),
                     subtitles='ass'),
    'webm': Container(copies=frozenset({'vp8', 'vp9', 'av1'}), encoder=VP9, encodes='vp9', voice=('libopus', '128k'),
                      original=('libopus', '128k'), subtitles='webvtt', keeps=frozenset({'opus', 'vorbis'})),
    'avi': Container(copies=frozenset({'h264', 'mpeg4'}), encoder=X264, encodes='h264', voice=('libmp3lame', '192k'),
                     original=('libmp3lame', '192k'), subtitles=None, keeps=frozenset({'mp3'})),
}
VIDEO_FORMATS = tuple(CONTAINERS)


def encoded_as(path):
    """(codec, bitrate) of an audio file the job encoded (media.AUDIO_CODECS, by its extension), or None."""
    codec = media.AUDIO_CODECS.get(Path(path).suffix.lstrip('.')) if path else None
    if not codec:
        return None
    return (codec[codec.index('-c:a') + 1],) + ((codec[codec.index('-b:a') + 1],) if '-b:a' in codec else ())


def needs_reencode(video_format, video_codec, burned=False):
    """Whether the picture must be re-encoded for this container (or because subtitles are burned in)."""
    copies = CONTAINERS[video_format].copies
    return bool(burned) or (copies is not None and video_codec not in copies)


def _audio(index, spec):
    codec, *bitrate = spec
    return [f'-c:a:{index}', codec] + ([f'-b:a:{index}', bitrate[0]] if bitrate else [])


def videos(source, voiceover_wav, renders, *, info, burn_srt=None, rendered=None, **options):
    """Write the voice-over video in each container of `renders`, (destination, subtitle_tracks) pairs named by the
    destination's suffix, and return the destinations in that order.

    Every container copies the source's picture when it can hold it. An encoded picture (burned subtitles, or a codec
    the container cannot hold) is made once per codec and copied into every further container that holds it;
    containers that hold any codec (MKV) come last, so they copy one instead of encoding their own. `rendered(n)` is
    called after each file with the number done.
    """
    codec = info.get('video_codec')
    names = [Path(destination).suffix[1:] for destination, _ in renders]
    encoded = []   # (codec, path) of each picture encoded so far
    for done, index in enumerate(sorted(range(len(renders)), key=lambda i: CONTAINERS[names[i]].copies is None), 1):
        destination, tracks = renders[index]
        container = CONTAINERS[names[index]]
        picture = None
        if needs_reencode(names[index], codec, burned=burn_srt is not None):
            picture = next((path for made, path in encoded if container.copies is None or made in container.copies),
                           None)
            if picture is None:
                encoded.append((container.encodes, Path(destination)))
        video(source, voiceover_wav, destination, info=info, burn_srt=burn_srt, subtitle_tracks=tracks, picture=picture,
              **options)
        if rendered:
            rendered(done)
    return [Path(destination) for destination, _ in renders]


def video(source, voiceover_wav, destination, *, info, target, source_language, burn_srt=None, subtitle_tracks=(),
          threads=0, credit=None, picture=None, voice_track=None):
    """Write the voice-over video in the container named by `destination`'s suffix.

    `subtitle_tracks` are (path, language_code, title) tuples (SRT, or ASS for MKV, which keeps their styles),
    added as switchable tracks when the container can hold them (AVI cannot: its subtitles are the separate files).
    `picture` is a video made for another container whose encoded picture (burned subtitles included) this one holds:
    its picture is copied instead of the source's being encoded again. `voice_track` is the same voice-over already
    encoded (the job's audio file), copied when it is in the container's voice codec.
    """
    destination = Path(destination)
    container = CONTAINERS[destination.suffix.lstrip('.')]
    codec = info.get('video_codec')
    tracks = list(subtitle_tracks) if container.subtitles else []
    burn_srt = None if picture else burn_srt   # a copied picture has them burned in already
    copied = encoded_as(voice_track) == container.voice
    voice = voice_track if copied else voiceover_wav
    args = ['ffmpeg', '-nostdin', '-v', 'error', '-y', '-i', Path(source).resolve(), '-i', Path(voice).resolve()]
    for path, _, _ in tracks:
        args += ['-i', Path(path).resolve()]
    if picture:
        args += ['-i', Path(picture).resolve()]
    args += ['-map', f'{len(tracks) + 2}:v:0' if picture else '0:v:0', '-map', '1:a:0',
             '-map', f"0:{info['audio_index']}"]
    for index in range(len(tracks)):
        args += ['-map', f'{index + 2}:s:0']
    if picture:
        args += ['-c:v', 'copy']
    elif burn_srt:
        burn_srt = Path(burn_srt)
        args += ['-vf', f"subtitles={burn_srt.name}:charenc=UTF-8:force_style='{SUBTITLE_STYLE}'",
                 *container.encoder, '-threads', threads]
    elif needs_reencode(destination.suffix.lstrip('.'), codec):
        args += [*container.encoder, '-threads', threads]
    else:
        args += ['-c:v', 'copy'] + (['-tag:v', 'hvc1'] if codec == 'hevc' and container.subtitles == 'mov_text' else [])
    original = ('copy',) if info.get('audio_codec') in container.keeps else container.original
    args += _audio(0, ('copy',) if copied else container.voice) + _audio(1, original)
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


SYNC_SLACK = 0.05   # seconds an AAC track may start off the media's start and still be copied as it is


def original_audio(source, destination, *, info, source_language):
    """The original soundtrack on its own (M4A), the track every video carries second: browsers other than Safari
    cannot switch a video's audio tracks, so the share page plays this file in step with the muted video when a viewer
    picks the original (backend ADR 0025). Its time 0 is the media's start, as the voice-over's is. AAC that starts
    there is copied as it is; anything else is encoded like the MP4's original track (AAC), in stereo, from the
    media's start."""
    streams = info.get('streams') or []
    track = next((s for s in streams if int(s['index']) == info['audio_index']), {})
    starts = [float(s['start_time']) for s in streams if s.get('start_time') not in (None, 'N/A')]
    offset = float(track.get('start_time') or 0) - min(starts, default=0.0)
    if info.get('audio_codec') in CONTAINERS['mp4'].keeps and abs(offset) <= SYNC_SLACK:
        audio = ['-c:a', 'copy']
    else:
        audio = ['-af', 'aresample=async=1:first_pts=0', '-ac', 2, *_audio(0, CONTAINERS['mp4'].original)]
    destination = Path(destination)
    media.run(['ffmpeg', '-nostdin', '-v', 'error', '-y', '-i', Path(source).resolve(), '-map',
               f"0:{info['audio_index']}", '-vn', '-sn', '-dn', *audio, '-map_metadata', '-1', '-map_chapters', '-1',
               '-metadata:s:a:0', f'language={track_tag(from_provider(source_language))}',
               '-metadata:s:a:0', 'title=Original', '-metadata:s:a:0', 'handler_name=Original',
               '-movflags', '+faststart', destination.resolve()])
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
