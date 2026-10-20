"""Round-trip check of how the narrator says the job's names."""
import numpy as np
import pytest

from vocnu_engine import media, pronounce, translate, tts
from vocnu_engine.costs import Receipt

TAKE = 0.5   # seconds of every fake take


def test_keys_ignore_spelling_that_does_not_change_the_sound():
    assert pronounce.key('Aliaska', 'lt') == pronounce.key('Aljaska', 'lt') == pronounce.key('Aliąska', 'lt')
    assert pronounce.key('Šiaulių', 'lt') == pronounce.key('Šaulių', 'lt') == 'sauliu'
    assert pronounce.key('Alijaska', 'lt') != pronounce.key('Aliaska', 'lt')   # another syllable
    assert pronounce.key('Anne-Marie', 'en') == 'anemarie'
    assert pronounce.key('東京', 'ja') == ''
    assert pronounce.similarity('alaska', 'alaska') == 1.0 and pronounce.similarity('', 'x') == 0.0


@pytest.mark.parametrize('heard, written', [('Sarah', 'Sara'), ('Barnes', 'Barns'), ('Thornton', 'Tornton'),
                                            ('Sarah Severns', 'Sara Severns'), ('Philpott', 'Filpot'),
                                            ('Faith Kleida', 'Feit Kleida')])
def test_recognition_spelling_a_name_the_source_way_is_the_same_sounds(heard, written):
    # 2026-10-20: recognition wrote "Sarah" for the narrator's "Sara", the check took it for the English name, and
    # the narrator was given "Sąra".
    assert pronounce.key(heard, 'lt') == pronounce.key(written, 'lt')
    assert pronounce.key('Alaska', 'lt') != pronounce.key('Aliaska', 'lt')


def test_a_respelling_keeps_every_sound_only_the_marks_of_a_soft_consonant_may_change():
    assert pronounce.spelling('Aljaskos', 'lt') == pronounce.spelling('Aliaskos', 'lt')
    assert pronounce.spelling('Šaulių', 'lt') == pronounce.spelling('Šiaulių', 'lt')
    assert pronounce.spelling('Sąra', 'lt') != pronounce.spelling('Sara', 'lt')        # a longer vowel
    assert pronounce.spelling('Aliąska', 'lt') != pronounce.spelling('Aliaska', 'lt')
    assert pronounce.spelling('Alijaska', 'lt') != pronounce.spelling('Aliaska', 'lt')  # another syllable


def test_a_form_leaks_to_its_source_name_with_the_forms_ending():
    assert pronounce._leaked_forms('Aliaskos', 'Aliaska', ['Alaska']) == {'Alaska', 'Alaskos'}
    assert pronounce._leaked_forms('Aliaska', 'Aliaska', ['Alaska']) == {'Alaska'}
    assert pronounce._leaked_forms('Mus Pase', 'Mus Pasas', ['Moose Pass']) == {'Moose Pass'}


@pytest.mark.parametrize('form, rendering, sources, checked', [
    ('Aliaska', 'Aliaska', ['Alaska'], True), ('Aliaskoje', 'Aliaska', ['Alaska'], True),
    ('Kanadoje', 'Kanada', ['Canada'], True),
    ('Džeikobą Filpotą', 'Džeikobas Filpotas', ['Jacob Philpott'], False),   # said as in English anyway
    ('Ankoridžo', 'Ankoridžas', ['Anchorage'], False), ('Galenoje', 'Galena', ['Galena'], False),
    ('Sara', 'Sara', ['Sarah'], False), ('Barns', 'Barns', ['Barnes'], False)])   # the same sounds
def test_only_forms_spelled_much_like_their_source_name_are_checked(form, rendering, sources, checked):
    assert pronounce.looks_alike(form, rendering, sources, 'lt') is checked


def test_names_with_their_forms_are_checked_but_not_codes_with_spoken_words():
    glossary = [{'source': 'Alaska', 'target': 'Aliaska', 'forms': {'nominative': 'Aliaska', 'genitive': 'Aliaskos'}},
                {'source': 'C-46', 'target': 'C-46', 'spoken': 'cė keturiasdešimt šeši'},
                {'source': 'Route 66', 'target': 'Route 66'}, {'source': 'AK', 'target': 'Aliaska'},
                {'source': 'Galena', 'target': 'Galena', 'forms': {'nominative': 'Galena', 'locative': 'Galenoje'}},
                {'source': 'made in Japan', 'target': 'Maid in Japan'}]
    assert pronounce.names(glossary, 'lt') == [('Aliaska', ['Alaska', 'AK'], ['Aliaska', 'Aliaskos'])]
    # Without cases every name is a word of the language: one spelled like its source is checked.
    assert [rendering for rendering, _, _ in pronounce.names(glossary, 'en')] == ['Aliaska', 'Maid in Japan']


@pytest.mark.parametrize('option, form, said', [
    ('Aljaska', 'Aliaskos', 'Aljaskos'), ('Aliąska', 'Aliaską', 'Aliąską'), ('Aljaska', 'Aliaska', 'Aljaska'),
    ('Aljaska', 'Kanados', None)])
def test_a_respelling_changes_the_stem_every_form_shares(option, form, said):
    assert pronounce.variant('Aliaska', option, form) == said


class Booth:
    """Fake narrator and speech recognition: each take is TAKE seconds of tone, and recognition hears the take's
    text with the narrator's accent applied (`accent`: {written word or whole line: what is heard}). The take a
    request holds is known by its file name (pronounce.recognise names it after the take); `requests` keeps each
    request's take text and audio length."""

    def __init__(self, monkeypatch, tmp_path, accent, options=None):
        self.accent, self.options, self.texts, self.requests = accent, options or {}, [], []
        self.folder = tmp_path / 'takes'
        self.folder.mkdir()
        self.asked = []
        monkeypatch.setattr(tts, 'synthesize_all', self.synthesize_all)
        monkeypatch.setattr(translate, 'respell', self.respell)

    def synthesize_all(self, lines, language, voice, cache_dir, receipt, *, session=None, seeds=None, on_done=None):
        result = {}
        for key, text in lines.items():
            path = self.folder / f'{len(self.texts)}.wav'
            media.write_wav(path, np.full(int(TAKE * 16000), 0.1, dtype=np.float32), rate=16000)
            receipt.add('speech', 'elevenlabs', tts.estimate_usd([text]), characters=len(text))
            self.texts.append(text)
            result[key] = path
        return result

    def transcribe(self, path, language):
        text = self.texts[int(path.stem.rsplit('-', 1)[1])]
        self.requests.append((text, len(media.decode_array(path, channels=1, rate=16000)) / 16000))
        heard = (self.accent[text].split() if text in self.accent
                 else [self.accent.get(word.strip('.,!?'), word.strip('.,!?')) for word in text.split()])
        step = TAKE / len(heard)
        words = [{'word': ' ' + word, 'start': k * step, 'end': (k + 1) * step} for k, word in enumerate(heard)]
        return {'language': language, 'words': words, 'seconds': 1.5}

    def respell(self, words, *, source, target, receipt, session=None):
        self.asked.append(words)
        return {w['id']: self.options.get(w['word'], []) for w in words}


GLOSSARY = [{'source': 'Alaska', 'target': 'Aliaska',
             'forms': {'nominative': 'Aliaska', 'genitive': 'Aliaskos', 'locative': 'Aliaskoje'}}]
LINES = {1: 'Aliaska yra didelė.', 2: 'Skridome per Aliaską ir toliau ir toliau.', 3: 'Gyvenu Aliaskoje.',
         4: 'Šiaurės Aliaskos miestai.', 5: 'Aliaska.'}


def check(booth, tmp_path, lines=LINES, budget=1.0, **options):
    return pronounce.respellings(lines, GLOSSARY, language='lt', voice='male', cache=tmp_path / 'tts',
                                 receipt=Receipt(max_usd=budget), transcribe=booth.transcribe,
                                 work=tmp_path / 'names', source='en', **options)


def test_a_name_the_narrator_says_the_english_way_gets_one_respelling_for_all_its_forms(monkeypatch, tmp_path):
    booth = Booth(monkeypatch, tmp_path, {'Aliaska': 'Alaska', 'Aliaskos': 'Alaskos', 'Alijaska': 'Alijaska'},
                  options={'Aliaska': ['Alijaska', 'Aliąska', 'Aljaska']})
    chosen, report, seconds = check(booth, tmp_path)
    # "Alijaska" adds a syllable and "Aliąska" a longer vowel: neither is tried. "Aljaska" reads the same as
    # "Aliaska" and is heard as written.
    assert chosen == {'Aliaska': 'Aljaska', 'Aliaskos': 'Aljaskos', 'Aliaskoje': 'Aljaskoje'}
    assert report['leaks'] == ['Aliaska'] and report['respelled'] == chosen
    assert set(report['tried']['Aliaska']) == {'Aljaska'}
    # The probes: the shortest line of every form, then the shortest line left ("Aliaską" is not a listed form).
    assert booth.texts[:4] == ['Aliaska yra didelė.', 'Gyvenu Aliaskoje.', 'Šiaurės Aliaskos miestai.', 'Aliaska.']
    assert booth.asked == [[{'id': 0, 'word': 'Aliaska', 'heard': 'Alaska', 'source': 'Alaska'}]]
    assert report['heard'] == {'Aliaska': ['Alaska', 'Alaska'], 'Aliaskos': ['Alaskos'], 'Aliaskoje': ['Aliaskoje']}
    assert seconds == 12.0   # four probes and four respelled probes, each take recognised on its own


def test_names_the_narrator_says_right_are_left_alone(monkeypatch, tmp_path):
    booth = Booth(monkeypatch, tmp_path, {})
    chosen, report, _ = check(booth, tmp_path)
    assert chosen == {} and report['leaks'] == [] and booth.asked == []
    assert report['names'] == 1 and report['probes'] == 4


def test_every_form_is_heard_not_only_the_shortest_lines(monkeypatch, tmp_path):
    # The narrator says "Aliaskos" right in many short lines and "Aliaska" wrong in the one long line that says it.
    lines = {k: f'Daug Aliaskos {word}.' for k, word in enumerate(['ežerų', 'kalnų', 'upių', 'miestų', 'kelių', 'oro'])}
    lines[9] = 'Aliaska yra didžiausia ir šalčiausia valstija, kurioje gyvena mažai žmonių.'
    booth = Booth(monkeypatch, tmp_path, {'Aliaska': 'Alaska'}, options={'Aliaska': ['Aljaska']})
    chosen, report, _ = check(booth, tmp_path, lines=lines)
    assert report['probes'] == len(lines) and lines[9] in booth.texts[:len(lines)]
    assert report['leaks'] == ['Aliaska'] and chosen['Aliaska'] == 'Aljaska'


def test_a_name_said_wrong_in_one_line_of_many_is_caught_and_respellings_are_tried_in_a_few(monkeypatch, tmp_path):
    words = ['ežerų', 'kalnų', 'upių', 'miestų', 'kelių', 'oro uostų', 'gyventojų', 'žvejų kaimų']
    lines = {k: f'Daug Aliaskos {word}.' for k, word in enumerate(words)}
    # The narrator says the name right in the five shortest lines and wrong in a longer one.
    booth = Booth(monkeypatch, tmp_path, {'Daug Aliaskos žvejų kaimų.': 'Daug Alaskos žvejų kaimų'},
                  options={'Aliaska': ['Aljaska']})
    chosen, report, _ = check(booth, tmp_path, lines=lines)
    assert report['probes'] == len(lines) and report['leaks'] == ['Aliaska']
    assert chosen == {'Aliaska': 'Aljaska', 'Aliaskos': 'Aljaskos', 'Aliaskoje': 'Aljaskoje'}
    # The respelling is tried in the name's first lines only.
    assert len(report['tried']['Aliaska']['Aljaska']) == pronounce.PROBES
    assert len(booth.texts) == len(lines) + pronounce.PROBES


def test_a_respelling_that_still_leaks_or_is_not_heard_is_not_used(monkeypatch, tmp_path):
    booth = Booth(monkeypatch, tmp_path, {'Aliaska': 'Alaska', 'Aljaska': 'Alaska', 'Aliąska': 'Kanados'},
                  options={'Aliaska': ['Aljaska', 'Aliąska']})
    chosen, report, _ = check(booth, tmp_path)
    assert chosen == {} and report['respelled'] == {} and report['leaks'] == ['Aliaska']


def test_a_respelling_heard_right_where_recognition_found_it_is_used(monkeypatch, tmp_path):
    # Recognition misses the respelled name in one of the four probes; the others are heard as written.
    booth = Booth(monkeypatch, tmp_path, {'Aliaska': 'Alaska', 'Aljaskoje': 'Kanados'}, options={'Aliaska': ['Aljaska']})
    chosen, report, _ = check(booth, tmp_path)
    assert report['tried']['Aliaska']['Aljaska'] == ['Aljaska', 'Aljaskos', 'Kanados', 'Aljaska']
    assert chosen == {'Aliaska': 'Aljaska', 'Aliaskos': 'Aljaskos', 'Aliaskoje': 'Aljaskoje'}


def test_the_check_stops_at_the_budget_and_the_names_stay_as_written(monkeypatch, tmp_path):
    booth = Booth(monkeypatch, tmp_path, {'Aliaska': 'Alaska'}, options={'Aliaska': ['Aljaska']})
    chosen, report, _ = check(booth, tmp_path, budget=0.0001)
    assert chosen == {} and report['stopped'] == 'budget' and booth.texts == []


def test_nothing_is_checked_without_recognition_or_names(monkeypatch, tmp_path):
    booth = Booth(monkeypatch, tmp_path, {})
    assert pronounce.respellings(LINES, GLOSSARY, language='lt', voice='male', cache=tmp_path, receipt=Receipt(1.0),
                                 transcribe=None, work=tmp_path)[0] == {}
    assert check(booth, tmp_path, lines={1: 'Be vardų.'})[0] == {} and booth.texts == []


@pytest.mark.parametrize('table, said', [({}, 'Aliaska.'), ({'Aliaska': 'Aljaska'}, 'Aljaska.')])
def test_probes_are_said_as_the_narrator_will_say_them(monkeypatch, tmp_path, table, said):
    booth = Booth(monkeypatch, tmp_path, {})
    check(booth, tmp_path, lines={5: 'Aliaska.'}, table=table)
    assert booth.texts == [said]


def test_a_failing_recognition_leaves_the_names_as_written(monkeypatch, tmp_path):
    Booth(monkeypatch, tmp_path, {'Aliaska': 'Alaska'})

    def broken(path, language):
        raise RuntimeError('the GPU container went away')

    chosen, report, _ = pronounce.respellings(LINES, GLOSSARY, language='lt', voice='male', cache=tmp_path / 'tts',
                                              receipt=Receipt(max_usd=1.0), transcribe=broken, work=tmp_path / 'n')
    assert chosen == {} and report['stopped'] == 'error'


GALENA = [{'source': 'Galena', 'target': 'Galena',
           'forms': {'nominative': 'Galena', 'genitive': 'Galenos', 'accusative': 'Galeną', 'locative': 'Galenoje'}}]


def finished(monkeypatch, tmp_path, lines, accent, glossary=GALENA):
    """The final check of finished takes of `lines` (a line that says no name last), heard with `accent`."""
    booth = Booth(monkeypatch, tmp_path, accent)
    paths = tts.synthesize_all(lines, 'lt', 'male', tmp_path / 'tts', Receipt(max_usd=1.0))
    return pronounce.heard_wrong({uid: (text, paths[uid]) for uid, text in lines.items()}, glossary, language='lt',
                                 transcribe=booth.transcribe, work=tmp_path / 'final')


def test_the_final_check_flags_a_line_where_the_narrator_slipped_on_a_name_kept_as_in_the_source(monkeypatch, tmp_path):
    lines = {1: 'Galenoje sunku gyventi.', 2: 'Skrendame į Galeną.', 3: 'Galenos žmonės laukia.', 4: 'Be vardų.'}
    flagged, report, seconds = finished(monkeypatch, tmp_path, lines, {'Galenoje': 'Galimoje'})
    assert flagged == {1: [('Galenoje', 'Galimoje')]} and seconds == 4.5   # the three takes that say a name
    assert report == {'lines': 3, 'misheard': {'Galenoje': ['Galimoje']}}


def test_every_take_is_recognised_on_its_own_and_a_take_two_lines_share_once(monkeypatch, tmp_path):
    # Takes joined in one request let recognition hear a slip as the name the other takes said right.
    booth = Booth(monkeypatch, tmp_path, {'Galenoje': 'Galimoje'})
    paths = tts.synthesize_all({1: 'Galenoje šalta.', 2: 'Skrendame į Galeną.'}, 'lt', 'male', tmp_path / 'tts',
                               Receipt(max_usd=1.0))
    heard, seconds = pronounce.recognise({1: paths[1], 2: paths[2], 3: paths[1]}, transcribe=booth.transcribe,
                                         language='lt', work=tmp_path / 'final', name='final-0')
    assert heard == {1: ['Galimoje', 'šalta'], 2: ['Skrendame', 'į', 'Galeną'], 3: ['Galimoje', 'šalta']}
    assert sorted(text for text, _ in booth.requests) == ['Galenoje šalta.', 'Skrendame į Galeną.']
    assert all(abs(length - TAKE - pronounce.GAP) < 0.01 for _, length in booth.requests) and seconds == 3.0


def test_the_final_check_flags_a_name_said_the_source_language_way(monkeypatch, tmp_path):
    # "Alaskos" is a sound away from "Aliaskos", and it is the English name with the form's ending.
    lines = {1: 'Aliaska yra didelė.', 2: 'Gyvenu Aliaskoje.', 3: 'Šiaurės Aliaskos miestai.'}
    flagged, report, _ = finished(monkeypatch, tmp_path, lines, {lines[3]: 'Šiaurės Alaskos miestai'}, GLOSSARY)
    assert flagged == {3: [('Aliaskos', 'Alaskos')]}


def test_a_name_heard_wrong_in_most_lines_is_recognitions_trouble_and_left_alone(monkeypatch, tmp_path):
    lines = {1: 'Galenoje sunku gyventi.', 2: 'Skrendame į Galeną.', 3: 'Galenos žmonės laukia.'}
    flagged, report, _ = finished(monkeypatch, tmp_path, lines, {'Galenoje': 'Galimoje', 'Galeną': 'galimą'})
    assert flagged == {} and report == {'lines': 3, 'misheard': {}}


def test_a_failing_final_check_flags_nothing(monkeypatch, tmp_path):
    booth = Booth(monkeypatch, tmp_path, {})
    path = booth.synthesize_all({1: 'Galenoje šalta.'}, 'lt', 'male', tmp_path, Receipt(max_usd=1.0))[1]

    def broken(path, language):
        raise RuntimeError('the GPU container went away')

    flagged, report, _ = pronounce.heard_wrong({1: ('Galenoje šalta.', path)}, GALENA, language='lt',
                                               transcribe=broken, work=tmp_path / 'final')
    assert flagged == {} and report['stopped'] == 'error'
