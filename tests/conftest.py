import os

import pytest

from vocnu_engine import config


@pytest.fixture(scope='session')
def empty_config(tmp_path_factory):
    return tmp_path_factory.mktemp('config')


@pytest.fixture(autouse=True)
def no_private_settings(monkeypatch, empty_config):
    """Tests never read the developer's private settings (~/.config/vocnu) or narrators."""
    monkeypatch.setattr(config, 'CONFIG', empty_config)
    for name in [name for name in os.environ if name.startswith('VOCNU_NARRATOR_')]:
        monkeypatch.delenv(name)
