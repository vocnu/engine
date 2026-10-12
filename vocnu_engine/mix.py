"""Measure narrator takes and mix them over the ducked original.

When each line plays is decided in timing.py (gap-aware budgets, natural speed) and fitting.py (repairs); this module
measures takes, levels them and mixes them over the original soundtrack. A line is never cut: narration that runs past
the end of the media extends the mix instead (a timing violation the job reports).
"""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import math
import os
from pathlib import Path

import numpy as np

from . import media, timing

SR = media.SR
# Background, exactly as the owner's ai-voice-over v4 mixed it (v4/plan.py 'mix', v4/audio.py envelope/mix_audio),
# which he found far better than Vocnu's earlier passage ducking ("v4 original sound was so good", 2026-10-08):
# the original keeps its own level (no loudness normalisation) and only dips to 0.40 (≈ -8 dB) under each narrator
# line, fading down over 0.2 s before the line, holding 0.15 s after it and coming back over 0.65 s; nearby lines'
# dips merge (the lower curve wins), so rapid dialogue stays down instead of bouncing. 0.12 had made the effects
# too quiet in v4 and 0.11 did the same in Vocnu. Songs have no narration, so they play at full level.
BACKGROUND_DURING_VOICE = 0.40
BACKGROUND_WITHOUT_VOICE = 1.0
FADE_IN = 0.2
HOLD = 0.15
FADE_OUT = 0.65
MASTER_PEAK = 0.89      # v4 masters by peak only: one linear gain so the loudest sample sits here
VOICE_RMS = 0.105       # v4: one constant gain per complete line to this active RMS, peak at most 0.8
SILENT_RMS = 0.002     # a take whose speech is quieter than this (active RMS) counts as silent
DECODE_PARALLEL = 6    # ffmpeg decodes at once while mixing (the job container has 4 cores)


@dataclass(frozen=True)
class Measured:
    seconds: float      # the take's speech, silence trimmed (60 ms of air kept either side)
    lead: float         # where that speech starts in the file
    silent: bool


def windows(units, media_duration):
    """{unit_id: seconds a narrator line may speak} from the gap-aware budgets, for translation budgets."""
    return {uid: budget.seconds for uid, budget in timing.budgets(units, media_duration).items()}


def bounds(pcm, rate=SR):
    """(first, end) sample indices of a TTS line's speech with 60 ms of air either side; None when silent."""
    peak = float(np.max(np.abs(pcm), initial=0.0))
    if peak < 3e-4:
        return None
    active = np.flatnonzero(np.abs(pcm) > max(3e-5, peak * 4e-4))
    pad = round(0.06 * rate)
    return max(0, int(active[0]) - pad), min(len(pcm), int(active[-1]) + pad + 1)


def trim(pcm, rate=SR):
    """Drop leading/trailing silence from a TTS line, keeping 60 ms of air."""
    found = bounds(pcm, rate)
    return pcm[:0] if found is None else pcm[found[0]:found[1]]


def active_rms(pcm):
    """RMS of the samples that carry the speech (above 3 % of the peak), 0 for silence."""
    peak = float(np.max(np.abs(pcm), initial=0.0))
    if peak <= 0:
        return 0.0
    active = np.abs(pcm) > max(3e-4, peak * 0.03)
    return float(np.sqrt(np.mean(pcm[active] ** 2))) if active.any() else peak


def level(pcm):
    """Constant gain per line to a common speech level (v4: active RMS target, bounded gain and peak)."""
    peak = float(np.max(np.abs(pcm), initial=0.0))
    if peak <= 0:
        return pcm
    gain = min(4.0, max(0.25, VOICE_RMS / max(active_rms(pcm), 1e-8)), 0.8 / peak)
    return (pcm * gain).astype(np.float32)


def measure(path):
    """Measured length, lead and silence of a synthesized line at natural speed (what load_line will mix)."""
    try:
        pcm = media.decode_array(path, channels=1, rate=SR)
    except media.MediaError:
        return Measured(0.0, 0.0, True)   # a take that cannot be decoded is as good as an empty one
    found = bounds(pcm)
    if found is None:
        return Measured(0.0, 0.0, True)
    first, end = found
    return Measured((end - first) / SR, first / SR, active_rms(pcm[first:end]) < SILENT_RMS)


def load_line(path, tempo=1.0):
    """Decoded, trimmed, tempo-adjusted mono speech at SR."""
    if tempo != 1.0:
        out = media.run(['ffmpeg', '-nostdin', '-v', 'error', '-i', path, '-af', f'atempo={tempo:.4f}',
                         '-ac', 1, '-ar', SR, '-f', 'f32le', 'pipe:1'])
        pcm = np.frombuffer(out, dtype='<f4').copy()
    else:
        pcm = media.decode_array(path, channels=1, rate=SR)
    return trim(pcm)


def envelope(seconds, placements, rate=100):
    """Background gain curve (v4 audio.envelope): BACKGROUND_DURING_VOICE under each line, raised-cosine fades of
    FADE_IN before it and FADE_OUT after it (+HOLD), BACKGROUND_WITHOUT_VOICE elsewhere; overlaps keep the lower."""
    count = math.ceil(seconds * rate) + 1
    env = np.full(count, BACKGROUND_WITHOUT_VOICE, np.float32)
    low, high = BACKGROUND_DURING_VOICE, BACKGROUND_WITHOUT_VOICE
    for placement in placements:
        a, b = max(0.0, placement.start), min(seconds, placement.end + HOLD)
        left, right = max(0.0, a - FADE_IN), min(seconds, b + FADE_OUT)
        i, j = math.floor(left * rate), math.ceil(right * rate)
        ts = np.arange(i, min(j + 1, count)) / rate
        gain = np.full(len(ts), low, np.float32)
        early, late = ts < a, ts > b
        gain[early] = high + (low - high) * (1 - np.cos(np.pi * np.clip((ts[early] - left) / max(a - left, 1e-6), 0, 1))) / 2
        gain[late] = low + (high - low) * (1 - np.cos(np.pi * np.clip((ts[late] - b) / max(right - b, 1e-6), 0, 1))) / 2
        env[i:i + len(gain)] = np.minimum(env[i:i + len(gain)], gain)
    return env


def _grow(path, voice, frames):
    """The narrator track memmap, extended with silence to `frames` samples."""
    voice.flush()
    os.truncate(path, frames * 4)
    return np.memmap(path, dtype='<f4', mode='r+', shape=(frames,))


def mix(source, info, placements, line_paths, folder):
    """Write voiceover.wav the v4 way: the original at its own level, dipped under each line (envelope), plus the
    narrator, then one linear gain so the peak sits at MASTER_PEAK.

    Every line is mixed whole: narration running past the end of the media makes the mix longer than the media
    (the video keeps its picture length) instead of cutting words.
    """
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    duration = info['duration_seconds']
    total = math.ceil(max([duration] + [p.end for p in placements]) * SR)
    voice = np.memmap(folder / 'narrator.f32', dtype='<f4', mode='w+', shape=(total,))
    # Every take and the original are decoded by their own ffmpeg process: several at once (one after another took
    # about half of the 129 s mix of a 23-minute episode).
    with ThreadPoolExecutor(DECODE_PARALLEL) as pool:
        decoding = pool.submit(media.decode, source, folder / 'original.f32', audio_index=info['audio_index'],
                               duration=duration, channels=2)
        takes = pool.map(lambda p: level(load_line(line_paths[p.unit_id], p.tempo)), placements)
        for p, pcm in zip(placements, takes):
            start = round(p.start * SR)
            if start + len(pcm) > len(voice):   # a sped-up take can come out a few ms longer than measured
                voice = _grow(folder / 'narrator.f32', voice, start + len(pcm))
            voice[start:start + len(pcm)] += pcm
        background = decoding.result()
    voice.flush()
    total = len(voice)
    env = envelope(total / SR, placements)
    mixed = np.memmap(folder / 'mix.f32', dtype='<f4', mode='w+', shape=(total, 2))
    step, mix_peak = 5 * SR, 0.0
    for a in range(0, total, step):
        b = min(total, a + step)
        chunk = np.zeros((b - a, 2), np.float32)
        have = max(0, min(b, len(background)) - a)
        if have:
            chunk[:have] = background[a:a + have]
        gains = np.interp(np.arange(a, b) / SR * 100, np.arange(len(env)), env).astype(np.float32)
        chunk *= gains[:, None]
        chunk += voice[a:b, None]
        mixed[a:b] = chunk
        mix_peak = max(mix_peak, float(np.max(np.abs(chunk), initial=0.0)))
    mixed.flush()
    master = min(1.0, MASTER_PEAK / max(mix_peak, 1e-8))
    voiceover = media.write_wav(folder / 'voiceover.wav', mixed, gain=master)
    for scratch in ('original.f32', 'narrator.f32', 'mix.f32'):
        (folder / scratch).unlink(missing_ok=True)
    return {'voiceover_wav': voiceover, 'master_gain': round(master, 3), 'lines': len(placements),
            'sped_up_lines': sum(1 for p in placements if p.tempo > 1.0),
            'max_tempo': max((p.tempo for p in placements), default=1.0)}
