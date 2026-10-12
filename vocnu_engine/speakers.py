"""Likely speaker gender per narrator unit, from voice pitch.

Translations into gendered languages need it for first-person forms (Lithuanian "laimingiausias būnu
mušdamas" vs "laimingiausia būnu mušdama"); the text alone often does not say who speaks. One narrator
reads every line, but each character keeps their own grammatical gender, as in Lithuanian voice-over.
A unit is labelled only when its whole pitch distribution sits in one range: shouting men, children and
music under the dialogue all push pitch up, so mixed, wide or barely voiced units get no label and the
translation falls back to the dialogue context. Measured 2026-10-08 on five demo scenes with known speakers:
no wrong labels with these limits (a median-only rule misread three male lines as female).
"""
import numpy as np

from . import media

RATE = 8000
FRAME = int(0.04 * RATE)
HOP = int(0.01 * RATE)
MIN_LAG, MAX_LAG = RATE // 400, RATE // 70   # 70-400 Hz
VOICING = 0.55
MALE_MEDIAN_MAX_HZ = 150.0     # and the 90th percentile below MALE_HIGH_MAX_HZ
MALE_HIGH_MAX_HZ = 200.0
FEMALE_LOW_MIN_HZ = 175.0      # 10th percentile; the median above FEMALE_MEDIAN_MIN_HZ
FEMALE_MEDIAN_MIN_HZ = 200.0
FEMALE_HIGH_MAX_HZ = 300.0     # 90th percentile; higher means shouting, a child or music
MIN_VOICED_FRAMES = 15


def load(path):
    """Mono 8 kHz 16-bit samples of a media file (the ASR audio is enough; ~115 MB for two hours)."""
    out = media.run(['ffmpeg', '-nostdin', '-v', 'error', '-i', path, '-map', '0:a:0', '-ac', 1, '-ar', RATE,
                     '-f', 's16le', 'pipe:1'])
    return np.frombuffer(out, dtype='<i2')


def pitches(pcm):
    """Fundamental frequency (Hz) of each voiced 40 ms frame."""
    pcm = np.asarray(pcm, dtype=np.float32)
    if len(pcm) < FRAME:
        return np.zeros(0)
    starts = np.arange(0, len(pcm) - FRAME + 1, HOP)
    frames = pcm[starts[:, None] + np.arange(FRAME)]
    frames = frames - frames.mean(axis=1, keepdims=True)
    energy = np.sqrt((frames ** 2).mean(axis=1))
    frames = frames[energy >= 0.5 * max(float(np.sqrt((pcm ** 2).mean())), 1e-6)]
    if not len(frames):
        return np.zeros(0)
    spectrum = np.fft.rfft(frames, n=2 * FRAME)
    acf = np.fft.irfft(spectrum * np.conj(spectrum))[:, :FRAME]
    acf /= np.maximum(acf[:, :1], 1e-9)
    lags = MIN_LAG + np.argmax(acf[:, MIN_LAG:MAX_LAG], axis=1)
    voiced = acf[np.arange(len(acf)), lags] > VOICING
    return RATE / lags[voiced]


def gender(pcm):
    """'male', 'female' or None for one speaker's stretch of audio."""
    f0 = pitches(pcm)
    if len(f0) < MIN_VOICED_FRAMES:
        return None
    low, median, high = np.percentile(f0, [10, 50, 90])
    if median < MALE_MEDIAN_MAX_HZ and high < MALE_HIGH_MAX_HZ:
        return 'male'
    if low > FEMALE_LOW_MIN_HZ and median > FEMALE_MEDIAN_MIN_HZ and high < FEMALE_HIGH_MAX_HZ:
        return 'female'
    return None


def genders(pcm, units):
    """{unit id: 'male' | 'female'} for units whose speaker is reasonably clear."""
    found = {}
    for unit in units:
        label = gender(pcm[int(unit.start * RATE):int(unit.end * RATE)])
        if label:
            found[unit.id] = label
    return found
