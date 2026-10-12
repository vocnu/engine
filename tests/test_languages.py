import pytest

from vocnu_engine import languages

# backend/config/vocnu_languages.yml (maintained on the Rails side) lists exactly these codes.
RAILS_MIRROR = ['en', 'lt', 'lv', 'et', 'pl', 'de', 'fr', 'es', 'pt', 'it', 'nl', 'sv', 'da', 'nb', 'fi', 'cs', 'sk',
                'sl', 'hr', 'bg', 'ro', 'hu', 'el', 'uk', 'ru', 'tr', 'ar', 'hi', 'id', 'ms', 'vi', 'th', 'ja', 'ko', 'zh']
# eleven_v3 language ids from GET /v1/models (2026-10-07), the subset we rely on.
ELEVEN_V3 = set('en lt lv et pl de fr es pt it nl sv da no fi cs sk sl hr bg ro hu el uk ru tr ar hi id ms vi th '
                'ja ko zh'.split())


def test_matches_the_rails_mirror_exactly():
    assert [lang.code for lang in languages.LANGUAGES] == RAILS_MIRROR
    assert 'he' not in languages.BY_CODE


def test_every_language_can_be_voiced_and_tagged():
    for lang in languages.LANGUAGES:
        assert lang.voiceover
        assert lang.tts_code in ELEVEN_V3
        assert languages.track_tag(lang.code) != 'und'
        assert lang.line_chars > 0 and lang.speech_cps > 0


def test_norwegian_maps_between_our_code_and_provider_code():
    assert languages.to_provider('nb') == 'no'
    assert languages.from_provider('no') == 'nb'
    assert languages.from_provider('en') == 'en'
    assert languages.describe('no') == 'Norwegian (nb)'


def test_unknown_code_is_rejected():
    with pytest.raises(ValueError):
        languages.get('he')
    assert languages.track_tag('xx') == 'und'


def test_case_languages_list_the_cases_of_names_nominative_first():
    with_cases = {lang.code for lang in languages.LANGUAGES if lang.cases}
    assert {'lt', 'lv', 'pl', 'cs', 'sk', 'ru', 'uk', 'fi', 'hu'} <= with_cases
    assert not languages.BY_CODE['en'].cases and not languages.BY_CODE['ja'].cases
    for code in with_cases:
        cases = languages.BY_CODE[code].cases
        assert cases[0] == 'nominative' and len(set(cases)) == len(cases)
    lithuanian = languages.BY_CODE['lt']
    assert lithuanian.cases == ('nominative', 'genitive', 'dative', 'accusative', 'instrumental', 'locative',
                                'vocative')
    assert 'vocative (šauksmininkas)' in lithuanian.note and '"Mičanai!"' in lithuanian.note
