"""Text normalization for BabelVox.

Expands abbreviations, numbers, dates, times, and telephone numbers
into spoken-word form so the TTS model produces natural speech.
"""
from __future__ import annotations

import re

from num2words import num2words

# ── Language mapping for num2words ─────────────────────────────────────
_LANG_MAP = {
    "english": "en", "chinese": "zh", "french": "fr", "german": "de",
    "italian": "it", "japanese": "ja", "korean": "ko", "portuguese": "pt",
    "russian": "ru", "spanish": "es",
}

# ── Abbreviation dictionaries (per-language) ──────────────────────────
_ABBREVIATIONS: dict[str, dict[str, str]] = {
    "english": {
        "Mr.": "Mister", "Mrs.": "Misses", "Ms.": "Miss", "Dr.": "Doctor",
        "Prof.": "Professor", "Jr.": "Junior", "Sr.": "Senior",
        "St.": "Saint", "Ave.": "Avenue", "Blvd.": "Boulevard",
        "vs.": "versus", "etc.": "et cetera", "approx.": "approximately",
        "dept.": "department", "govt.": "government", "inc.": "incorporated",
        "corp.": "corporation", "ltd.": "limited", "est.": "established",
        "vol.": "volume", "no.": "number", "Jan.": "January",
        "Feb.": "February", "Mar.": "March", "Apr.": "April",
        "Jun.": "June", "Jul.": "July", "Aug.": "August",
        "Sep.": "September", "Oct.": "October", "Nov.": "November",
        "Dec.": "December",
    },
    "french": {
        "M.": "Monsieur", "Mme.": "Madame", "Mlle.": "Mademoiselle",
        "Dr.": "Docteur", "Prof.": "Professeur",
    },
    "german": {
        "Hr.": "Herr", "Fr.": "Frau", "Dr.": "Doktor",
        "Prof.": "Professor", "Nr.": "Nummer", "Str.": "Strasse",
    },
}

# ── Regex patterns ────────────────────────────────────────────────────
_CURRENCY_RE = re.compile(
    r"(?P<sign>[\$\u00a3\u20ac])(?P<whole>\d{1,3}(?:,\d{3})*)(?:\.(?P<cents>\d{1,2}))?",
)
_NUMBER_RE = re.compile(r"(?<!\d)(\d{1,3}(?:,\d{3})*(?:\.\d+)?|\d{4,}(?:\.\d+)?)(?!\d)")
_ORDINAL_RE = re.compile(r"\b(\d+)(st|nd|rd|th)\b", re.IGNORECASE)
_DATE_MDY_RE = re.compile(r"\b(\d{1,2})/(\d{1,2})/(\d{2,4})\b")
_DATE_ISO_RE = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")
_TIME_RE = re.compile(r"\b(\d{1,2}):(\d{2})(?::(\d{2}))?\s*(AM|PM|am|pm|a\.m\.|p\.m\.)?\b")
_PHONE_RE = re.compile(
    r"(?<!\d)(?:\+?1[-.\s]?)?\(?(\d{3})\)?[-.\s]?(\d{3})[-.\s]?(\d{4})(?!\d)",
)
_LETTER_ACRONYM_RE = re.compile(r"\b([A-Z]\.){2,}")

_CURRENCY_NAMES = {"$": ("dollar", "dollars", "cent", "cents"),
                   "\u00a3": ("pound", "pounds", "penny", "pence"),
                   "\u20ac": ("euro", "euros", "cent", "cents")}

_MONTH_NAMES = [
    "", "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]


def _num2words_safe(n, lang="en", **kwargs):
    """num2words with fallback to English for unsupported languages."""
    try:
        return num2words(n, lang=lang, **kwargs)
    except NotImplementedError:
        return num2words(n, lang="en", **kwargs)


def expand_abbreviations(text: str, language: str = "english") -> str:
    abbrevs = _ABBREVIATIONS.get(language.lower(), {})
    if not abbrevs:
        abbrevs = _ABBREVIATIONS.get("english", {})
    for abbr, expansion in abbrevs.items():
        text = re.sub(re.escape(abbr) + r"(?=\s|$)", expansion, text)
    return text


def normalize_numbers(text: str, language: str = "english") -> str:
    lang_code = _LANG_MAP.get(language.lower(), "en")

    # Ordinals first (1st, 2nd, 3rd, 4th, ...)
    def _ordinal_repl(m):
        n = int(m.group(1))
        return _num2words_safe(n, lang=lang_code, to="ordinal")

    text = _ORDINAL_RE.sub(_ordinal_repl, text)

    # Currency
    def _currency_repl(m):
        sign = m.group("sign")
        whole = int(m.group("whole").replace(",", ""))
        cents = int(m.group("cents")) if m.group("cents") else 0
        names = _CURRENCY_NAMES.get(sign, ("dollar", "dollars", "cent", "cents"))
        parts = []
        if whole:
            word = _num2words_safe(whole, lang=lang_code)
            unit = names[0] if whole == 1 else names[1]
            parts.append(f"{word} {unit}")
        if cents:
            word = _num2words_safe(cents, lang=lang_code)
            unit = names[2] if cents == 1 else names[3]
            parts.append(f"{word} {unit}")
        if not parts:
            return f"zero {names[1]}"
        return " and ".join(parts)

    text = _CURRENCY_RE.sub(_currency_repl, text)

    # Plain numbers
    def _number_repl(m):
        raw = m.group(1).replace(",", "")
        if "." in raw:
            return _num2words_safe(float(raw), lang=lang_code)
        return _num2words_safe(int(raw), lang=lang_code)

    text = _NUMBER_RE.sub(_number_repl, text)
    return text


def normalize_dates(text: str, language: str = "english") -> str:
    lang_code = _LANG_MAP.get(language.lower(), "en")

    def _mdy_repl(m):
        month, day, year = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if year < 100:
            year += 2000 if year < 50 else 1900
        if 1 <= month <= 12:
            month_name = _MONTH_NAMES[month]
            day_word = _num2words_safe(day, lang=lang_code, to="ordinal")
            year_word = _num2words_safe(year, lang=lang_code)
            return f"{month_name} {day_word}, {year_word}"
        return m.group(0)

    text = _DATE_MDY_RE.sub(_mdy_repl, text)

    def _iso_repl(m):
        year, month, day = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if 1 <= month <= 12:
            month_name = _MONTH_NAMES[month]
            day_word = _num2words_safe(day, lang=lang_code, to="ordinal")
            year_word = _num2words_safe(year, lang=lang_code)
            return f"{month_name} {day_word}, {year_word}"
        return m.group(0)

    text = _DATE_ISO_RE.sub(_iso_repl, text)
    return text


def normalize_times(text: str, language: str = "english") -> str:
    lang_code = _LANG_MAP.get(language.lower(), "en")

    def _time_repl(m):
        hour, minute = int(m.group(1)), int(m.group(2))
        period = m.group(4) or ""
        hour_word = _num2words_safe(hour, lang=lang_code)
        parts = [hour_word]
        if minute == 0:
            parts.append("o'clock" if not period else "")
        else:
            if minute < 10:
                parts.append(f"oh {_num2words_safe(minute, lang=lang_code)}")
            else:
                parts.append(_num2words_safe(minute, lang=lang_code))
        if period:
            clean = period.replace(".", "").upper().strip()
            parts.append(" ".join(clean))
        return " ".join(p for p in parts if p)

    text = _TIME_RE.sub(_time_repl, text)
    return text


def normalize_telephone(text: str) -> str:
    def _digit_words(digits: str) -> str:
        names = {"0": "zero", "1": "one", "2": "two", "3": "three",
                 "4": "four", "5": "five", "6": "six", "7": "seven",
                 "8": "eight", "9": "nine"}
        return " ".join(names[d] for d in digits if d in names)

    def _phone_repl(m):
        area, prefix, line = m.group(1), m.group(2), m.group(3)
        return f"{_digit_words(area)}, {_digit_words(prefix)}, {_digit_words(line)}"

    text = _PHONE_RE.sub(_phone_repl, text)
    return text


def normalize_letter_acronyms(text: str) -> str:
    """Expand dotted acronyms like U.S.A. to spaced letters: U S A."""
    def _repl(m):
        letters = m.group(0).replace(".", " ").strip()
        return letters
    return _LETTER_ACRONYM_RE.sub(_repl, text)


def normalize_say_as(text: str, interpret_as: str, fmt: str = "",
                     language: str = "english") -> str:
    """Dispatcher for SSML <say-as> interpret-as values."""
    text = text.strip()
    if interpret_as == "number" or interpret_as == "cardinal":
        return normalize_numbers(text, language)
    if interpret_as == "ordinal":
        lang_code = _LANG_MAP.get(language.lower(), "en")
        try:
            return _num2words_safe(int(text), lang=lang_code, to="ordinal")
        except ValueError:
            return text
    if interpret_as == "date":
        return normalize_dates(text, language)
    if interpret_as == "time":
        return normalize_times(text, language)
    if interpret_as == "telephone":
        return normalize_telephone(text)
    if interpret_as == "characters" or interpret_as == "spell-out":
        return " ".join(text)
    return text


def normalize_text(text: str, language: str = "english") -> str:
    """Run the full normalization pipeline on plain text.

    Order matters: abbreviations first (so "Dr." doesn't get caught by
    sentence-boundary logic), then acronyms, numbers, dates, times,
    telephone numbers.
    """
    text = expand_abbreviations(text, language)
    text = normalize_letter_acronyms(text)
    text = normalize_dates(text, language)
    text = normalize_times(text, language)
    text = normalize_telephone(text)
    text = normalize_numbers(text, language)
    return text
