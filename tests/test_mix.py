import numpy as np
import pytest

from vocnu_engine import media, mix, timing
from vocnu_engine.segments import Unit


def units(*spans):
    return [Unit(i + 1, start, end, [i + 1], 'x') for i, (start, end) in enumerate(spans)]


def place(*spans):
    """Narrator lines of a second each, starting just after each source span (as timing.place puts lines that fit)."""
    return [timing.Placement(i + 1, start + timing.SOURCE_DELAY, 1.0) for i, (start, _) in enumerate(spans)]


def test_translation_windows_are_the_gap_aware_budgets():
    timeline = units((0.0, 2.0), (2.5, 4.0), (10.0, 11.0))
    found = mix.windows(timeline, media_duration=11.5)
    assert found == {uid: budget.seconds for uid, budget in timing.budgets(timeline, 11.5).items()}
    # 1: from just after its start to 0.35 s after its end (a 0.5 s gap is rapid dialogue); 2: the whole 2 s tail
    # before a long gap; 3: the last line's 2 s tail stops at the media end.
    assert found == pytest.approx({1: 2.35 - 0.1, 2: 6.0 - 2.6, 3: 11.5 - 10.1})


def test_the_original_dips_to_v4s_level_under_each_line_and_comes_back():
    placed = place((1.0, 2.0))      # the line plays 1.1-2.1 s
    env = mix.envelope(5.0, placed)
    line = placed[0]
    assert env[int(1.5 * 100)] == np.float32(mix.BACKGROUND_DURING_VOICE)
    assert env[int((line.start - mix.FADE_IN) * 100) - 1] == 1.0              # fading starts 0.2 s before
    assert env[int((line.end + mix.HOLD + mix.FADE_OUT) * 100) + 1] == 1.0    # and is back 0.15 + 0.65 s after
    assert env[0] == 1.0 and env[-1] == 1.0


def test_lines_close_together_keep_the_original_down():
    placed = place((1.0, 2.0), (2.6, 3.6))
    env = mix.envelope(6.0, placed)
    gap = env[int((placed[0].end + 0.3) * 100):int(placed[1].start * 100)]
    # v4's curve: in a 0.6 s gap the dips overlap (the lower curve wins), so the original swells only part of the
    # way (about 0.72) instead of returning to full level between the lines.
    assert gap.max() < 0.8


def test_without_narration_the_original_plays_at_its_own_level():
    assert np.all(mix.envelope(10.0, []) == 1.0)    # songs and scenery: nothing is lowered


def test_the_background_never_jumps():
    placed = place((1.0, 2.0), (9.0, 10.0), (25.0, 26.0))
    env = mix.envelope(30.0, placed)
    # Raised-cosine fades: a 0.2 s dip and a 0.65 s return over at most 8 dB.
    assert np.abs(np.diff(20 * np.log10(env))).max() < 1.5


def _active_rms(pcm):
    active = np.abs(pcm) > np.max(np.abs(pcm)) * 0.03
    return float(np.sqrt(np.mean(pcm[active] ** 2)))


def test_trim_and_level():
    sr = mix.SR
    tone = 0.1 * np.sin(np.linspace(0, 2 * np.pi * 440, sr)).astype(np.float32)
    padded = np.concatenate([np.zeros(sr), tone, np.zeros(sr)]).astype(np.float32)
    trimmed = mix.trim(padded)
    assert sr <= len(trimmed) <= sr + 2 * round(0.06 * sr) + 1
    assert abs(_active_rms(mix.level(trimmed)) - mix.VOICE_RMS) < 0.01
    assert len(mix.trim(np.zeros(1000, np.float32))) == 0


def test_level_gain_is_bounded():
    quiet = 0.01 * np.sin(np.linspace(0, 2 * np.pi * 440, mix.SR)).astype(np.float32)
    assert _active_rms(mix.level(quiet)) == pytest.approx(_active_rms(quiet) * 4.0, rel=1e-3)
    loud = np.sign(np.sin(np.linspace(0, 2 * np.pi * 440, mix.SR))).astype(np.float32)
    assert np.max(np.abs(mix.level(loud))) <= 0.89 + 1e-6


def tone(seconds, amplitude=0.1, frequency=660.0, silence=0.0):
    """Mono float PCM: `silence` seconds, a tone, `silence` seconds."""
    t = np.arange(round(seconds * mix.SR)) / mix.SR
    gap = np.zeros(round(silence * mix.SR), np.float32)
    return np.concatenate([gap, (amplitude * np.sin(2 * np.pi * frequency * t)).astype(np.float32), gap])


def test_measure_finds_the_speech_and_where_it_starts(tmp_path):
    take = media.write_wav(tmp_path / 'take.wav', tone(1.0, silence=0.5))
    found = mix.measure(take)
    assert not found.silent
    assert found.lead == pytest.approx(0.5 - 0.06, abs=0.005)        # 60 ms of air kept before the speech
    assert found.seconds == pytest.approx(1.0 + 2 * 0.06, abs=0.005)
    assert found.seconds * mix.SR == pytest.approx(len(mix.load_line(take)), abs=1)


def test_empty_faint_or_broken_takes_measure_as_silent(tmp_path):
    silence = media.write_wav(tmp_path / 'silence.wav', np.zeros(mix.SR, np.float32))
    hiss = media.write_wav(tmp_path / 'hiss.wav', tone(1.0, amplitude=0.0015))   # active RMS ≈ 0.001
    broken = tmp_path / 'broken.mp3'
    broken.write_bytes(b'not audio')
    empty = tmp_path / 'empty.mp3'
    empty.write_bytes(b'')
    assert all(mix.measure(path).silent for path in (silence, hiss, broken, empty))
    assert not mix.measure(media.write_wav(tmp_path / 'quiet.wav', tone(1.0, amplitude=0.01))).silent


def test_narration_running_past_the_end_is_mixed_whole(tmp_path):
    source = media.write_wav(tmp_path / 'source.wav', np.stack([tone(2.0, 0.05, 220.0)] * 2, axis=1))
    line = media.write_wav(tmp_path / 'line.wav', tone(1.5))
    seconds = mix.measure(line).seconds
    placements = [timing.Placement(1, 1.5, seconds)]   # ends ≈ 3.06 s, a second after the 2 s media
    mixed = mix.mix(source, {'duration_seconds': 2.0, 'audio_index': 0}, placements, {1: line}, tmp_path / 'mix')
    out = media.decode_array(mixed['voiceover_wav'], channels=1)
    assert len(out) / mix.SR == pytest.approx(1.5 + seconds, abs=0.01)
    tail = out[int(2.2 * mix.SR):int(2.9 * mix.SR)]      # after the original ends: only the narrator
    assert np.sqrt(np.mean(tail ** 2)) > 0.05


def test_normalize_keeps_the_mix_dynamics(tmp_path):
    # Quiet bed then a loud burst: a single gain must keep their 24 dB difference (no dynamic levelling).
    t = np.arange(6 * media.SR) / media.SR
    pcm = np.where(t < 5, 0.03, 0.5) * np.sin(2 * np.pi * 440 * t)
    source = media.write_wav(tmp_path / 'in.wav', np.stack([pcm, pcm], axis=1).astype(np.float32))
    out = media.normalize(source, tmp_path / 'out.wav')
    data = np.frombuffer(media.run(['ffmpeg', '-v', 'error', '-i', out, '-ac', 1, '-f', 'f32le', 'pipe:1']), '<f4')
    quiet, loud = data[media.SR:4 * media.SR], data[int(5.2 * media.SR):int(5.8 * media.SR)]
    ratio_db = 20 * np.log10(np.sqrt(np.mean(loud ** 2)) / np.sqrt(np.mean(quiet ** 2)))
    assert abs(ratio_db - 20 * np.log10(0.5 / 0.03)) < 0.3
