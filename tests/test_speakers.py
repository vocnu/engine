import numpy as np

from vocnu_engine import speakers
from vocnu_engine.segments import Unit

RATE = speakers.RATE


def voice(f0, seconds=1.0):
    """A buzzy voiced sound: a fundamental with a few harmonics."""
    t = np.arange(int(seconds * RATE)) / RATE
    return sum(np.sin(2 * np.pi * f0 * k * t) / k for k in range(1, 5)).astype(np.float32) * 0.2


def test_pitch_separates_lower_and_higher_voices():
    assert speakers.gender(voice(120)) == 'male'
    assert speakers.gender(voice(230)) == 'female'


def test_unclear_or_unvoiced_audio_gets_no_label():
    assert speakers.gender(voice(168)) is None                     # between the ranges
    assert speakers.gender(np.concatenate([voice(120), voice(260)])) is None   # two voices in one unit
    assert speakers.gender(voice(340)) is None                     # shouting, a child or music
    assert speakers.gender(np.zeros(RATE, dtype=np.float32)) is None
    rng = np.random.default_rng(1)
    assert speakers.gender(rng.normal(0, 0.1, RATE).astype(np.float32)) is None
    assert speakers.gender(voice(120, seconds=0.05)) is None        # too short to judge


def test_each_unit_is_judged_on_its_own_span():
    pcm = np.concatenate([voice(115), voice(240)])
    units = [Unit(1, 0.0, 1.0, [], 'He speaks.'), Unit(2, 1.0, 2.0, [], 'She answers.')]
    assert speakers.genders(pcm, units) == {1: 'male', 2: 'female'}
