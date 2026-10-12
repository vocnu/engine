import json

import pytest

from vocnu_engine import cli
from tests.test_pipeline import WORDS, clip, fake_providers  # noqa: F401 - fixtures


@pytest.fixture
def heard(monkeypatch):
    monkeypatch.setattr(cli.asr, 'transcribe', lambda path, language, hints=(): {'language': 'en', 'words': WORDS})


def test_translate_saves_a_video_with_its_subtitles_and_receipt(clip, fake_providers, heard, tmp_path, capsys):
    out = tmp_path / 'results'
    code = cli.main(['translate', str(clip), '--to', 'lt', '--out', str(out), '--work', str(tmp_path / 'work'),
                     '--budget', '0.5', '--no-names-library'])
    assert code == 0, capsys.readouterr()
    names = sorted(path.name for path in out.iterdir())
    assert 'video.lt.mp4' in names and 'subtitles.lt.srt' in names and 'receipt.json' in names
    assert json.loads((out / 'receipt.json').read_text())['event'] == 'completed'
    assert 'Spent $' in capsys.readouterr().out


def test_translate_audio_only_and_its_cache_folder_goes_after_success(clip, fake_providers, heard, tmp_path,
                                                                      monkeypatch):
    monkeypatch.setenv('XDG_CACHE_HOME', str(tmp_path / 'cache'))
    out = tmp_path / 'results'
    assert cli.main(['translate', str(clip), '--to', 'lt', '--output', 'audio', '--audio-format', 'mp3',
                     '--out', str(out), '--budget', '0.5', '--no-names-library']) == 0
    names = {path.name for path in out.iterdir()}
    assert any(name.endswith('.mp3') for name in names) and not any(name.endswith('.mp4') for name in names)
    assert list((tmp_path / 'cache' / 'vocnu-engine').iterdir()) == []


def test_every_output_includes_subtitles_like_the_app():
    assert cli.products_for('subtitles') == {'subtitles': True, 'voiceover_audio': False, 'voiceover_video': False,
                                             'burned_subtitles': False}
    assert cli.products_for('video', burn_subtitles=True)['burned_subtitles'] is True
    assert cli.products_for('audio', burn_subtitles=True)['burned_subtitles'] is False


def test_languages_lists_every_target(capsys):
    assert cli.main(['languages']) == 0
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == len(cli.LANGUAGES) and lines[1].startswith('lt ')
