"""Supported target languages: the source of truth that Rails mirrors in backend/config/vocnu_languages.yml.

Every language here can be voiced by ElevenLabs eleven_v3 (checked against GET /v1/models on 2026-10-07)
and recognised by Whisper. `speech_cps` is the narrator's speaking rate in characters per second (sets voice-over
length budgets): the median rate of lines of 15 characters or more in the 11 demo scenes of each voiced site
language, measured 2026-10-09 (ja 8 -> 6, zh 7 -> 5 and ar 13 -> 11 had been too generous, de, fr, es and pl too
strict); languages not measured keep their earlier rate. A per-line overhead (lines under 15 characters run at
about 8 characters a second) made budgets stricter in rapid dialogue, which cut meaning ("Neprašiau šito!" for "I
didn't ask to be born like this") without fewer lines off their timing on Bleach S01E01, so budgets stay flat and
the measured fitting repairs what does not fit. `line_chars` and `reading_cps` shape subtitles. `provider_code`
is the code ElevenLabs and Whisper use when it differs from ours (Norwegian Bokmål: nb → no).
`cases` are the grammatical cases a name takes (nominative first): for these languages each job decides every
person and place name's form per case, so the translator, the reviewer and the name check use the same forms.
"""
from dataclasses import dataclass

SEVEN_CASES = ('nominative', 'genitive', 'dative', 'accusative', 'instrumental', 'locative', 'vocative')
SIX_CASES = SEVEN_CASES[:-1]
# Finno-Ugric languages have more cases; these are the ones names commonly take.
FINNISH_CASES = ('nominative', 'genitive', 'partitive', 'essive', 'translative', 'inessive', 'elative', 'illative',
                 'adessive', 'ablative', 'allative')
ESTONIAN_CASES = FINNISH_CASES + ('terminative', 'comitative')
HUNGARIAN_CASES = ('nominative', 'accusative', 'dative', 'instrumental', 'causal', 'translative', 'terminative',
                   'essive', 'inessive', 'superessive', 'adessive', 'illative', 'sublative', 'allative', 'elative',
                   'delative', 'ablative')

# ai-voice-over v4 (Lithuanian localisation, human voice-over reference): foreign names take Lithuanian endings and
# decline; a call uses the vocative; one narrator keeps every character's own grammatical gender.
LITHUANIAN_NOTE = (
    'Lithuanian: give foreign names Lithuanian spelling and endings (Ichigo → Ičigas, Rukia → Rukija) unless the '
    'glossary fixes another rendering, and decline them. Direct address uses the vocative (šauksmininkas): '
    '"Mičanai!", "Ičigai, palauk!". A character\'s own gender shows in past tenses, participles and adjectives '
    '("buvau pavargęs" / "buvau pavargusi"), whoever reads the line. Mind look-alike verbs: liesti (to touch) is '
    'not lieti (to pour). Write everyday spoken Lithuanian: no bookish '
    'interjections such as "et" or "ak" and no slang such as "dzin" ("Whatever, Tom." → "Kaip nori, Tomai."), no '
    'word-for-word calques ("follow our passions" → "darykim, kas mums patinka", not "sekim savo aistras"), and a '
    'clause joined with "kad", never a colon ("Pripažink, kad bijai").')


@dataclass(frozen=True)
class Language:
    code: str
    name: str
    native: str
    speech_cps: float = 15.0
    line_chars: int = 42
    reading_cps: float = 17.0
    rtl: bool = False
    voiceover: bool = True
    note: str = ''  # extra instruction for the translator, the reviewer and the analysis
    provider_code: str = ''
    cases: tuple = ()  # grammatical cases of names, nominative first; empty when names barely inflect

    def chars_for(self, seconds):
        """Characters a narrator line can hold in `seconds`, with 5 % to spare; never fewer than a short word."""
        return max(6, int(seconds * self.speech_cps * 0.95))

    @property
    def tts_code(self):
        return self.provider_code or self.code

    @property
    def spaced(self):
        """False for scripts written without spaces between words."""
        return self.code not in ('ja', 'zh', 'th')


LANGUAGES = [
    Language('en', 'English', 'English', speech_cps=15.5),
    # Vocnu's Lithuanian narrator on eleven_v4: 13.0 characters a second, the median of lines of 15 characters or more
    # on Bleach S01E01 and the demos (2026-10-09).
    Language('lt', 'Lithuanian', 'Lietuvių', speech_cps=13.0, note=LITHUANIAN_NOTE, cases=SEVEN_CASES),
    Language('lv', 'Latvian', 'Latviešu', speech_cps=14.0, cases=SEVEN_CASES),
    Language('et', 'Estonian', 'Eesti', speech_cps=14.0, cases=ESTONIAN_CASES),
    Language('pl', 'Polish', 'Polski', speech_cps=16.0, cases=SEVEN_CASES),
    Language('de', 'German', 'Deutsch', speech_cps=18.0),
    Language('fr', 'French', 'Français', speech_cps=18.5),
    Language('es', 'Spanish', 'Español', speech_cps=17.0),
    Language('pt', 'Portuguese', 'Português', speech_cps=14.5),
    Language('it', 'Italian', 'Italiano', speech_cps=16.0),
    Language('nl', 'Dutch', 'Nederlands'),
    Language('sv', 'Swedish', 'Svenska'),
    Language('da', 'Danish', 'Dansk'),
    Language('nb', 'Norwegian', 'Norsk', note='Use Norwegian Bokmål.', provider_code='no'),
    Language('fi', 'Finnish', 'Suomi', speech_cps=14.0, cases=FINNISH_CASES),
    Language('cs', 'Czech', 'Čeština', speech_cps=14.0, cases=SEVEN_CASES),
    Language('sk', 'Slovak', 'Slovenčina', speech_cps=14.0, cases=SIX_CASES),
    Language('sl', 'Slovenian', 'Slovenščina', speech_cps=14.0, cases=SIX_CASES),
    Language('hr', 'Croatian', 'Hrvatski', speech_cps=14.0, cases=SEVEN_CASES),
    Language('bg', 'Bulgarian', 'Български', speech_cps=14.0),
    Language('ro', 'Romanian', 'Română'),
    Language('hu', 'Hungarian', 'Magyar', speech_cps=14.0, cases=HUNGARIAN_CASES),
    Language('el', 'Greek', 'Ελληνικά', speech_cps=14.0),
    Language('uk', 'Ukrainian', 'Українська', speech_cps=12.5, cases=SEVEN_CASES),
    Language('ru', 'Russian', 'Русский', speech_cps=15.0, cases=SIX_CASES[:-1] + ('prepositional',)),
    Language('tr', 'Turkish', 'Türkçe', speech_cps=15.0),
    Language('ar', 'Arabic', 'العربية', speech_cps=11.0, rtl=True, note='Use Modern Standard Arabic.'),
    Language('hi', 'Hindi', 'हिन्दी', speech_cps=13.5),
    Language('id', 'Indonesian', 'Bahasa Indonesia', speech_cps=14.0),
    Language('ms', 'Malay', 'Bahasa Melayu'),
    Language('vi', 'Vietnamese', 'Tiếng Việt'),
    Language('th', 'Thai', 'ไทย', speech_cps=11.0, line_chars=35, reading_cps=15.0),
    Language('ja', 'Japanese', '日本語', speech_cps=6.0, line_chars=13, reading_cps=8.0),
    Language('ko', 'Korean', '한국어', speech_cps=8.0, line_chars=16, reading_cps=12.0),
    Language('zh', 'Chinese', '中文（简体）', speech_cps=5.0, line_chars=16, reading_cps=9.0,
             note='Use Simplified Chinese.'),
]

BY_CODE = {language.code: language for language in LANGUAGES}
_FROM_PROVIDER = {language.provider_code: language.code for language in LANGUAGES if language.provider_code}

# ISO 639-2/B codes for MP4 track language tags.
ISO639_2 = {
    'en': 'eng', 'lt': 'lit', 'lv': 'lav', 'et': 'est', 'pl': 'pol', 'de': 'ger', 'fr': 'fre', 'es': 'spa',
    'pt': 'por', 'it': 'ita', 'nl': 'dut', 'sv': 'swe', 'da': 'dan', 'nb': 'nob', 'fi': 'fin', 'cs': 'cze',
    'sk': 'slo', 'sl': 'slv', 'hr': 'hrv', 'bg': 'bul', 'ro': 'rum', 'hu': 'hun', 'el': 'gre', 'uk': 'ukr',
    'ru': 'rus', 'tr': 'tur', 'ar': 'ara', 'hi': 'hin', 'id': 'ind', 'ms': 'may', 'vi': 'vie', 'th': 'tha',
    'ja': 'jpn', 'ko': 'kor', 'zh': 'chi',
}


def get(code):
    """The Language for a supported target code, or ValueError."""
    try:
        return BY_CODE[code]
    except KeyError:
        raise ValueError(f'Unsupported language: {code!r}') from None


def from_provider(code):
    """Our code for a Whisper/ElevenLabs language code (no → nb); other codes pass through."""
    return _FROM_PROVIDER.get(code, code)


def to_provider(code):
    """The Whisper/ElevenLabs code for one of ours (nb → no); unknown codes pass through."""
    language = BY_CODE.get(code)
    return language.tts_code if language else code


def describe(code):
    """Human name for prompts; also covers source languages outside the target list."""
    language = BY_CODE.get(from_provider(code))
    return f'{language.name} ({language.code})' if language else code


def track_tag(code):
    """ISO 639-2 tag for media track metadata ('und' when unknown)."""
    return ISO639_2.get(from_provider(code), 'und')


# Track tags as files write them (ISO 639-2/B and /T, or 639-1) → our code.
_FROM_TAG = {**{tag: code for code, tag in ISO639_2.items()}, **{code: code for code in ISO639_2},
             'deu': 'de', 'fra': 'fr', 'nld': 'nl', 'ces': 'cs', 'slk': 'sk', 'ron': 'ro', 'ell': 'el',
             'msa': 'ms', 'zho': 'zh', 'nor': 'nb', 'no': 'nb'}


def code_for_tag(tag):
    """Our language code for a media track's language tag ('jpn' → 'ja'), None when unknown."""
    return _FROM_TAG.get((tag or '').lower())


def as_records():
    """Plain dicts for the Rails mirror and the API."""
    return [{'code': lang.code, 'name': lang.name, 'native': lang.native, 'rtl': lang.rtl,
             'voiceover': lang.voiceover} for lang in LANGUAGES]
