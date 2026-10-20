"""What the narrator says for a written line, and its timestamps mapped back to the written line."""
from vocnu_engine import spoken


def test_the_table_holds_the_glossarys_spoken_words_and_the_respellings_on_top():
    glossary = [{'source': 'C-46', 'target': 'C-46', 'spoken': 'cė keturiasdešimt šeši'},
                {'source': 'Alaska', 'target': 'Aliaska'}, {'source': 'X', 'target': 'X', 'spoken': 'X'}]
    assert spoken.table(glossary, {'Aliaska': 'Aljaska', 'Aliaskos': 'Aliaskos'}) == {
        'C-46': 'cė keturiasdešimt šeši', 'Aliaska': 'Aljaska'}


def test_whole_words_are_swapped_longest_form_first():
    forms = {'Boeing 737': 'Boeing septyni šimtai trisdešimt septyni', 'Boeing': 'Boingas', 'Aliaska': 'Aljaska'}
    said, spans = spoken.say('Boeing 737 skrido virš Aliaskos, o Boeing liko Aliaska.', forms)
    assert said == 'Boeing septyni šimtai trisdešimt septyni skrido virš Aliaskos, o Boingas liko Aljaska.'
    assert [(a, b) for a, b, _, _ in spans] == [(0, 10), (35, 41), (47, 54)]
    assert spoken.say('Be vardų.', forms) == ('Be vardų.', [])
    assert spoken.spoken_text('Aliaska', {}) == 'Aliaska'


def alignment(text, step=0.1):
    return {'alignment': {'characters': list(text),
                          'character_start_times_seconds': [round(k * step, 3) for k in range(len(text))],
                          'character_end_times_seconds': [round((k + 1) * step, 3) for k in range(len(text))]},
            'normalized_alignment': None}


def test_the_takes_timestamps_are_mapped_back_to_the_written_line():
    text, forms = 'C-46 ten.', {'C-46': 'cė keturiasdešimt'}
    said = spoken.spoken_text(text, forms)
    mapped = spoken.written_alignment(alignment(said), text, forms)['alignment']
    assert ''.join(mapped['characters']) == text
    # The written code shares the time its spoken words took; the rest keeps its own times.
    assert mapped['character_start_times_seconds'][0] == 0.0
    assert mapped['character_end_times_seconds'][3] == round(len('cė keturiasdešimt') * 0.1, 3)
    assert mapped['character_start_times_seconds'][5:] == [round((len(said) - 4 + k) * 0.1, 3) for k in range(4)]


def test_timestamps_stay_as_they_are_without_a_swap_and_are_dropped_when_they_do_not_match():
    timestamps = alignment('Labas.')
    assert spoken.written_alignment(timestamps, 'Labas.', {'Aliaska': 'Aljaska'}) is timestamps
    assert spoken.written_alignment(alignment('Aliaska.'), 'Aliaska.', {'Aliaska': 'Aljaska'}) is None
    assert spoken.written_alignment(None, 'Aliaska.', {'Aliaska': 'Aljaska'}) is None
