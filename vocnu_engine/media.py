"""ffprobe/ffmpeg helpers. Heavy lifting stays in native ffmpeg; Python only builds commands."""
import json
import math
from pathlib import Path
import re
import subprocess

import numpy as np

from .errors import WorkerError

SR = 48000
# Rails accepts up to 5 h (backend/config/vocnu.yml) plus its 5 % duration tolerance; this is the worker's own cap.
MAX_SECONDS = round(5.5 * 3600)
LOUDNESS_TARGET = -16.0
TRUE_PEAK = -1.5


class MediaError(RuntimeError):
    pass


def run(args, *, timeout=None):
    """Run a command, returning stdout; failures raise MediaError with the end of stderr."""
    args = [str(a) for a in args]
    try:
        result = subprocess.run(args, capture_output=True, timeout=timeout, check=False)
    except subprocess.TimeoutExpired:
        raise MediaError(f'{args[0]} timed out') from None
    if result.returncode != 0:
        tail = result.stderr.decode(errors='replace').strip()[-1500:]
        raise MediaError(f'{args[0]} failed ({result.returncode}): {tail}')
    return result.stdout


def _probe_json(target, timeout):
    out = run(['ffprobe', '-v', 'error', '-show_format', '-show_streams', '-of', 'json', target], timeout=timeout)
    return json.loads(out or b'{}')


# Tracks that are not the programme's speech (ai-voice-over v4 select_tracks): commentary, audio description,
# music-and-effects or karaoke stems.
NOT_SPEECH_TRACK = re.compile(r'comment|descri|\bAD\b|karaoke|instrumental|\bM ?& ?E\b|music', re.I)


def _language(stream):
    return (stream.get('tags') or {}).get('language', '').lower()


def audio_stream_index(streams, avoid_languages=(), prefer_languages=()):
    """Absolute index of the audio stream to translate, None when there is no audio.

    Ranking after ai-voice-over v4 (select_tracks): skip commentary/description/music-and-effects tracks and tracks
    already in the target language (an anime file whose default track is a Lithuanian dub must be translated from
    its original); then a track in the language the customer named as the source; then the default track; then a
    non-English track (extra tracks are mostly English dubs, as in Bleach's English/Japanese/Lithuanian files); then
    the first one.
    """
    audio = [s for s in streams if s.get('codec_type') == 'audio']
    if not audio:
        return None
    speech = [s for s in audio if not NOT_SPEECH_TRACK.search((s.get('tags') or {}).get('title', ''))] or audio
    avoid = {code.lower() for code in avoid_languages if code}
    original = [s for s in speech if _language(s) not in avoid] or speech
    prefer = {code.lower() for code in prefer_languages if code}
    chosen = (next((s for s in original if _language(s) in prefer), None)
              or next((s for s in original if (s.get('disposition') or {}).get('default')), None)
              or next((s for s in original if _language(s) not in ('eng', 'en')), None)
              or original[0])
    return int(chosen['index'])


MAX_DUB_TRACKS = 2   # v4 compared up to three soundtracks: the original and two dubs


def dub_tracks(streams, chosen_index, limit=MAX_DUB_TRACKS):
    """[{'index', 'tag'}] other speech tracks to recognise as translation evidence, as ai-voice-over v4 did with
    Bleach's English and Lithuanian dubs: language-tagged, one per language, not the chosen source's language and not
    commentary or music; English first, then file order. A dub already in the target language counts too: it is a
    human adaptation of the same lines."""
    from .languages import code_for_tag
    audio = [s for s in streams if s.get('codec_type') == 'audio']
    chosen = next((s for s in audio if int(s['index']) == chosen_index), None)

    def language(stream):
        tag = _language(stream)
        return code_for_tag(tag) or tag

    seen = {'', 'und', language(chosen) if chosen else ''}
    found = []
    for stream in sorted(audio, key=lambda s: (language(s) != 'en', int(s['index']))):
        if (int(stream['index']) == chosen_index or language(stream) in seen
                or NOT_SPEECH_TRACK.search((stream.get('tags') or {}).get('title', ''))):
            continue
        seen.add(language(stream))
        found.append({'index': int(stream['index']), 'tag': _language(stream)})
    return found[:limit]


def inspect(target, timeout=60, avoid_languages=(), prefer_languages=()):
    """Facts about a local file or URL; raises WorkerError with a probe code."""
    try:
        data = _probe_json(target, timeout)
    except MediaError:
        raise WorkerError('unsupported_media', 'The file could not be read as audio or video.') from None
    streams = data.get('streams') or []
    fmt = data.get('format') or {}
    video = next((s for s in streams if s.get('codec_type') == 'video'
                  and not (s.get('disposition') or {}).get('attached_pic')), None)
    durations = [fmt.get('duration')] + [s.get('duration') for s in streams]
    duration = max((float(d) for d in durations if d not in (None, 'N/A')), default=0.0)
    if duration <= 0:
        raise WorkerError('unsupported_media', 'The media has no measurable duration.')
    audio_index = audio_stream_index(streams, avoid_languages, prefer_languages)
    if audio_index is None:
        raise WorkerError('no_audio', 'The media has no audio track to translate.')
    if duration > MAX_SECONDS:
        raise WorkerError('too_long', 'Media longer than 5.5 hours is not supported.')
    tags = fmt.get('tags') or {}
    size = fmt.get('size')
    return {
        'duration_seconds': round(duration, 3),
        'has_audio': True,
        'has_video': video is not None,
        'title': tags.get('title') or (Path(str(target)).stem if '://' not in str(target) else None),
        'width': int(video['width']) if video and video.get('width') else None,
        'height': int(video['height']) if video and video.get('height') else None,
        'size_bytes': int(size) if size and str(size).isdigit() else None,
        'audio_index': audio_index,
        'video_codec': video.get('codec_name') if video else None,
        'audio_codec': next((s.get('codec_name') for s in streams if int(s['index']) == audio_index), None),
        'streams': streams,
    }


def public_probe(info):
    """The probe response shape from the contract."""
    keys = ('duration_seconds', 'has_audio', 'has_video', 'title', 'width', 'height', 'size_bytes', 'video_codec')
    return {key: info.get(key) for key in keys}


def asr_audio(source, destination, audio_index, duration):
    """Mono 16 kHz Opus aligned to the media timeline (small enough to ship to a GPU container)."""
    run(['ffmpeg', '-nostdin', '-v', 'error', '-y', '-i', source, '-map', f'0:{audio_index}', '-vn', '-sn', '-dn',
         '-af', 'aresample=16000:async=1:first_pts=0,apad', '-t', f'{duration:.3f}', '-ac', 1,
         '-c:a', 'libopus', '-b:a', '48k', '-application', 'voip', destination])
    return Path(destination)


def decode(source, destination, *, audio_index=None, duration=None, channels=2, rate=SR):
    """Float32 PCM memmap (frames × channels) on disk, aligned to time 0 and padded to `duration`."""
    stream = ['-map', f'0:{audio_index}'] if audio_index is not None else ['-map', '0:a:0']
    timeline = ['-af', f'aresample={rate}:async=1:first_pts=0' + (',apad' if duration else '')]
    if duration:
        timeline += ['-t', f'{duration:.3f}']
    run(['ffmpeg', '-nostdin', '-v', 'error', '-y', '-i', source, *stream, '-vn', '-sn', '-dn', *timeline,
         '-ac', channels, '-ar', rate, '-f', 'f32le', destination])
    pcm = np.memmap(destination, dtype='<f4', mode='r')
    return pcm.reshape(-1, channels) if channels > 1 else pcm


def decode_array(source, *, channels=1, rate=SR):
    """Small clips (TTS lines) decoded straight into memory."""
    out = run(['ffmpeg', '-nostdin', '-v', 'error', '-i', source, '-vn', '-ac', channels, '-ar', rate,
               '-f', 'f32le', 'pipe:1'])
    pcm = np.frombuffer(out, dtype='<f4').copy()
    return pcm.reshape(-1, channels) if channels > 1 else pcm


def write_wav(path, pcm, rate=SR, gain=1.0):
    """16-bit WAV from float PCM (frames or frames × channels, arrays or memmaps), written in chunks."""
    import wave
    channels = 1 if pcm.ndim == 1 else pcm.shape[1]
    with wave.open(str(path), 'wb') as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        step = rate * 10
        for start in range(0, len(pcm), step):
            chunk = np.clip(np.asarray(pcm[start:start + step], dtype=np.float32) * gain, -1.0, 1.0)
            handle.writeframes((chunk * 32767).astype('<i2').tobytes())
    return Path(path)


def peak(pcm, step=SR * 10):
    """Absolute peak of a large array or memmap, chunk by chunk."""
    return max((float(np.max(np.abs(pcm[i:i + step]))) for i in range(0, len(pcm), step)), default=0.0)


def loudness(path):
    """Integrated loudness (LUFS) and true peak (dBTP) measured by ffmpeg's loudnorm analysis."""
    result = subprocess.run(['ffmpeg', '-nostdin', '-hide_banner', '-i', str(path), '-af',
                             f'loudnorm=I={LOUDNESS_TARGET}:TP=-1.5:LRA=11:print_format=json', '-f', 'null', '-'],
                            capture_output=True, check=False)
    text = result.stderr.decode(errors='replace')
    match = re.search(r'\{[^{}]*"input_i"[^{}]*\}', text, re.S)
    if result.returncode != 0 or not match:
        raise MediaError('Loudness measurement failed')
    return json.loads(match.group(0))


def normalize(source, destination, *, target=LOUDNESS_TARGET, peak=TRUE_PEAK):
    """One constant gain towards `target` LUFS, capped so the true peak stays at `peak` dBTP; 48 kHz WAV.

    ffmpeg's loudnorm silently switches to dynamic mode when a linear gain would break the peak or range
    limits, and then rides the level up in every pause (heard as pumping in the voice-over mixes). A capped
    linear gain can leave a mix a dB or two under target, which is far less noticeable.
    """
    measured = loudness(source)
    gain = float(target) - float(measured['input_i'])
    headroom = float(peak) - float(measured['input_tp'])
    gain = min(gain, headroom) if math.isfinite(gain) and math.isfinite(headroom) else 0.0
    run(['ffmpeg', '-nostdin', '-v', 'error', '-y', '-i', source, '-af', f'volume={gain:.2f}dB', '-ar', SR,
         '-c:a', 'pcm_s16le', destination])
    return Path(destination)


def encode_aac(source, destination, *, bitrate='160k', language=None, title=None):
    return encode_audio(source, destination, language=language, title=title, aac_bitrate=bitrate)


# Every file with a synthetic voice says so in its metadata (EU AI Act art. 50: synthetic audio is marked in a
# machine-readable way): a comment every player and tag reader shows, the voice track named "AI voice-over".
AI_DISCLOSURE = 'Contains an AI-generated voice-over made with Vocnu (https://vocnu.com)'


def voice_track_title(language_name, credit=None):
    """The voice track's name in players' audio menus; a community voice names its donor (Vocnu's ADR 0017)."""
    return f'AI voice-over ({language_name}): {credit}' if credit else f'AI voice-over ({language_name})'


def disclosure(credit=None):
    """The metadata comment that discloses the synthetic voice (EU AI Act art. 50), with the donor's credit."""
    return f'{AI_DISCLOSURE}. Voice: {credit}' if credit else AI_DISCLOSURE


# Download formats of the voice-over audio: extension → ffmpeg codec arguments.
AUDIO_CODECS = {
    'm4a': ['-c:a', 'aac', '-b:a', '160k', '-movflags', '+faststart'],
    'mp3': ['-c:a', 'libmp3lame', '-b:a', '192k'],
    'wav': ['-c:a', 'pcm_s16le'],
    'flac': ['-c:a', 'flac'],
    'ogg': ['-c:a', 'libopus', '-b:a', '128k'],
}


def encode_audio(source, destination, *, language=None, title=None, aac_bitrate='160k', credit=None):
    """The voice-over mix in the format of `destination`'s extension (see AUDIO_CODECS)."""
    destination = Path(destination)
    codec = list(AUDIO_CODECS[destination.suffix.lstrip('.')])
    if destination.suffix == '.m4a':
        codec[codec.index('-b:a') + 1] = aac_bitrate
    meta = []
    if language:
        meta += ['-metadata:s:a:0', f'language={language}', '-metadata', f'language={language}']
    if title:
        meta += ['-metadata:s:a:0', f'title={title}', '-metadata', f'title={title}']
        if destination.suffix == '.m4a':   # see render.py: older ffmpeg names MP4 tracks only by their handler
            meta += ['-metadata:s:a:0', f'handler_name={title}']
    meta += ['-metadata', f'comment={disclosure(credit)}']
    run(['ffmpeg', '-nostdin', '-v', 'error', '-y', '-i', source, '-vn', *codec, *meta, destination])
    return destination
