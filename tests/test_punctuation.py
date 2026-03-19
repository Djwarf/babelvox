"""Tests for punctuation normalization."""
from babelvox.text.punctuation import normalize_punctuation


class TestUnicodeReplacement:
    def test_smart_single_quotes(self):
        assert normalize_punctuation("\u2018hello\u2019") == "'hello'"

    def test_smart_double_quotes(self):
        assert normalize_punctuation("\u201chello\u201d") == '"hello"'

    def test_em_dash(self):
        assert " -- " in normalize_punctuation("hello\u2014world")

    def test_en_dash(self):
        assert " - " in normalize_punctuation("hello\u2013world")

    def test_ellipsis_character(self):
        assert "..." in normalize_punctuation("hello\u2026")

    def test_non_breaking_space(self):
        assert normalize_punctuation("hello\u00a0world") == "hello world"

    def test_guillemets(self):
        result = normalize_punctuation("\u00abbonjour\u00bb")
        assert result == '"bonjour"'


class TestRepeatedPunctuation:
    def test_multiple_exclamation(self):
        assert normalize_punctuation("wow!!!") == "wow!"

    def test_multiple_question(self):
        assert normalize_punctuation("really???") == "really?"

    def test_excessive_dots(self):
        assert normalize_punctuation("hmm.....") == "hmm..."

    def test_three_dots_preserved(self):
        assert normalize_punctuation("hmm...") == "hmm..."

    def test_single_preserved(self):
        assert normalize_punctuation("wow!") == "wow!"


class TestSentenceSpacing:
    def test_missing_space_after_period(self):
        result = normalize_punctuation("Hello.World")
        assert result == "Hello. World"

    def test_missing_space_after_exclamation(self):
        result = normalize_punctuation("Wow!Great")
        assert result == "Wow! Great"

    def test_existing_space_unchanged(self):
        result = normalize_punctuation("Hello. World")
        assert result == "Hello. World"

    def test_lowercase_after_period_unchanged(self):
        # Don't add space before lowercase (could be abbreviation)
        result = normalize_punctuation("e.g. this")
        assert result == "e.g. this"


class TestPassthrough:
    def test_plain_text(self):
        assert normalize_punctuation("Hello world") == "Hello world"

    def test_empty(self):
        assert normalize_punctuation("") == ""

    def test_numbers_unchanged(self):
        assert normalize_punctuation("Price: $4.50") == "Price: $4.50"
