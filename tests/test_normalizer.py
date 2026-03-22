"""Tests for text normalization."""

from babelvox.text.normalizer import (
    expand_abbreviations,
    normalize_dates,
    normalize_letter_acronyms,
    normalize_numbers,
    normalize_say_as,
    normalize_telephone,
    normalize_text,
    normalize_times,
)


class TestExpandAbbreviations:
    def test_english_mr(self):
        assert "Mister Smith" in expand_abbreviations("Mr. Smith")

    def test_english_dr(self):
        assert "Doctor" in expand_abbreviations("Dr. Jones")

    def test_english_etc(self):
        assert "et cetera" in expand_abbreviations("cats, dogs, etc.")

    def test_english_vs(self):
        result = expand_abbreviations("cats vs. dogs")
        assert "versus" in result

    def test_french(self):
        result = expand_abbreviations("M. Dupont", language="french")
        assert "Monsieur" in result

    def test_german(self):
        result = expand_abbreviations("Dr. Mueller", language="german")
        assert "Doktor" in result

    def test_unknown_language_falls_back_to_english(self):
        result = expand_abbreviations("Mr. Smith", language="swahili")
        assert "Mister" in result

    def test_no_false_positives(self):
        result = expand_abbreviations("Hello world")
        assert result == "Hello world"


class TestNormalizeNumbers:
    def test_simple_integer(self):
        result = normalize_numbers("I have 5 cats")
        assert "five" in result

    def test_large_number(self):
        result = normalize_numbers("Population is 1000")
        assert "thousand" in result.lower() or "one thousand" in result

    def test_ordinal(self):
        result = normalize_numbers("The 3rd place")
        assert "third" in result

    def test_currency_dollars(self):
        result = normalize_numbers("It costs $4.50")
        assert "four" in result and "dollar" in result
        assert "fifty" in result and "cent" in result

    def test_currency_no_cents(self):
        result = normalize_numbers("It costs $10")
        assert "ten" in result and "dollar" in result

    def test_decimal(self):
        result = normalize_numbers("Pi is 3.14")
        assert "three" in result

    def test_comma_separated(self):
        result = normalize_numbers("Total: 1,000,000")
        assert "million" in result.lower() or "one million" in result

    def test_french_numbers(self):
        result = normalize_numbers("J'ai 5 chats", language="french")
        assert "cinq" in result

    def test_spanish_numbers(self):
        result = normalize_numbers("Tengo 3 gatos", language="spanish")
        assert "tres" in result


class TestNormalizeDates:
    def test_mdy_format(self):
        result = normalize_dates("Date: 03/19/2026")
        assert "March" in result
        assert "nineteen" in result.lower() or "nineteenth" in result

    def test_iso_format(self):
        result = normalize_dates("Date: 2026-03-19")
        assert "March" in result

    def test_two_digit_year(self):
        result = normalize_dates("Date: 12/25/99")
        assert "December" in result

    def test_no_false_positive(self):
        result = normalize_dates("Hello world")
        assert result == "Hello world"


class TestNormalizeTimes:
    def test_simple_time(self):
        result = normalize_times("Meet at 3:45 PM")
        assert "three" in result
        assert "forty" in result.lower() or "forty-five" in result.lower()

    def test_on_the_hour(self):
        result = normalize_times("Meet at 5:00")
        assert "five" in result

    def test_am_pm(self):
        result = normalize_times("Wake up at 7:30 AM")
        assert "seven" in result
        assert "A" in result and "M" in result

    def test_leading_zero_minutes(self):
        result = normalize_times("At 9:05")
        assert "nine" in result
        assert "oh" in result and "five" in result


class TestNormalizeTelephone:
    def test_standard_phone(self):
        result = normalize_telephone("Call 555-123-4567")
        assert "five five five" in result
        assert "one two three" in result

    def test_parenthesized_area(self):
        result = normalize_telephone("Call (555) 123-4567")
        assert "five five five" in result

    def test_dotted_format(self):
        result = normalize_telephone("Call 555.123.4567")
        assert "five five five" in result


class TestNormalizeLetterAcronyms:
    def test_usa(self):
        result = normalize_letter_acronyms("The U.S.A. is large")
        assert "U S A" in result

    def test_no_false_positive(self):
        result = normalize_letter_acronyms("Hello world")
        assert result == "Hello world"


class TestNormalizeSayAs:
    def test_cardinal(self):
        result = normalize_say_as("42", "cardinal")
        assert "forty" in result.lower()

    def test_ordinal(self):
        result = normalize_say_as("3", "ordinal")
        assert "third" in result

    def test_telephone(self):
        result = normalize_say_as("555-123-4567", "telephone")
        assert "five" in result

    def test_spell_out(self):
        result = normalize_say_as("hello", "spell-out")
        assert result == "h e l l o"

    def test_characters(self):
        result = normalize_say_as("ABC", "characters")
        assert result == "A B C"

    def test_unknown_returns_unchanged(self):
        result = normalize_say_as("hello", "unknown_type")
        assert result == "hello"

    def test_verbatim(self):
        result = normalize_say_as("ABC", "verbatim")
        assert result == "A B C"

    def test_fraction_half(self):
        result = normalize_say_as("1/2", "fraction")
        assert result == "one half"

    def test_fraction_three_quarters(self):
        result = normalize_say_as("3/4", "fraction")
        assert result == "three quarters"

    def test_fraction_generic(self):
        result = normalize_say_as("2/7", "fraction")
        assert "two" in result
        assert "seventh" in result

    def test_unit_kg(self):
        result = normalize_say_as("5kg", "unit")
        assert "five" in result
        assert "kilogram" in result

    def test_unit_singular(self):
        result = normalize_say_as("1m", "unit")
        assert "one" in result
        assert "meter" in result
        assert "meters" not in result

    def test_unit_unknown_suffix(self):
        result = normalize_say_as("10xyz", "unit")
        assert "ten" in result


class TestNormalizeTextPipeline:
    def test_combined(self):
        result = normalize_text("Dr. Smith has 3 cats and $4.50")
        assert "Doctor" in result
        assert "three" in result
        assert "dollar" in result

    def test_empty_string(self):
        result = normalize_text("")
        assert result == ""

    def test_already_clean(self):
        result = normalize_text("Hello world")
        assert result == "Hello world"

    def test_phone_in_sentence(self):
        result = normalize_text("Call me at 555-123-4567 today")
        assert "five five five" in result
