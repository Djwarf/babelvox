"""Tests for SSML parsing."""

from babelvox.text.ssml import looks_like_ssml, parse_ssml


class TestLooksLikeSSML:
    def test_speak_tag(self):
        assert looks_like_ssml("<speak>Hello</speak>")

    def test_break_tag(self):
        assert looks_like_ssml("Hello <break time='500ms'/> world")

    def test_sub_tag(self):
        assert looks_like_ssml("The <sub alias='World Wide Web'>WWW</sub>")

    def test_plain_text(self):
        assert not looks_like_ssml("Hello world")

    def test_angle_brackets_in_math(self):
        # < in math context shouldn't trigger
        assert not looks_like_ssml("x < 5 and y > 3")


class TestParseSSMLPlainText:
    def test_no_tags(self):
        text, annotations = parse_ssml("<speak>Hello world</speak>")
        assert text == "Hello world"
        assert annotations == []

    def test_auto_wrap(self):
        text, annotations = parse_ssml("Hello world")
        assert text == "Hello world"
        assert annotations == []

    def test_whitespace_preserved(self):
        text, _ = parse_ssml("<speak>Hello  world</speak>")
        assert "Hello  world" in text


class TestBreak:
    def test_short_break(self):
        text, _ = parse_ssml("<speak>Hello<break time='200ms'/>world</speak>")
        assert "," in text
        assert "Hello" in text and "world" in text

    def test_medium_break(self):
        text, _ = parse_ssml("<speak>Hello<break time='500ms'/>world</speak>")
        assert "." in text

    def test_long_break(self):
        text, _ = parse_ssml("<speak>Hello<break time='1000ms'/>world</speak>")
        assert "..." in text

    def test_seconds(self):
        text, _ = parse_ssml("<speak>Hello<break time='2s'/>world</speak>")
        assert "..." in text

    def test_strength_strong(self):
        text, _ = parse_ssml("<speak>Hello<break strength='strong'/>world</speak>")
        assert ". " in text or "." in text

    def test_strength_weak(self):
        text, _ = parse_ssml("<speak>Hello<break strength='weak'/>world</speak>")
        assert "," in text

    def test_strength_none(self):
        text, _ = parse_ssml("<speak>Hello<break strength='none'/>world</speak>")
        assert text == "Helloworld" or text == "Hello world"


class TestSub:
    def test_alias_replacement(self):
        ssml = "<speak>The <sub alias='World Wide Web Consortium'>W3C</sub> is great</speak>"
        text, _ = parse_ssml(ssml)
        assert "World Wide Web Consortium" in text
        assert "W3C" not in text

    def test_no_alias_keeps_text(self):
        ssml = "<speak>The <sub>original</sub> text</speak>"
        text, _ = parse_ssml(ssml)
        assert "original" in text


class TestSayAs:
    def test_with_normalizer(self):
        def mock_normalizer(text, interpret_as, fmt):
            if interpret_as == "telephone":
                return "five five five, one two three four"
            return text

        ssml = "<speak>Call <say-as interpret-as='telephone'>555-1234</say-as></speak>"
        text, _ = parse_ssml(ssml, normalizer_fn=mock_normalizer)
        assert "five five five" in text

    def test_without_normalizer(self):
        ssml = "<speak>Call <say-as interpret-as='telephone'>555-1234</say-as></speak>"
        text, _ = parse_ssml(ssml, normalizer_fn=None)
        assert "555-1234" in text


class TestAnnotations:
    def test_emphasis_creates_annotation(self):
        ssml = "<speak>This is <emphasis level='strong'>important</emphasis> text</speak>"
        text, annotations = parse_ssml(ssml)
        assert "important" in text
        assert len(annotations) == 1
        assert annotations[0].type == "emphasis"
        assert annotations[0].params.get("level") == "strong"

    def test_prosody_creates_annotation(self):
        ssml = "<speak><prosody rate='fast' pitch='+2st'>Quick speech</prosody></speak>"
        text, annotations = parse_ssml(ssml)
        assert "Quick speech" in text
        assert len(annotations) == 1
        assert annotations[0].type == "prosody"
        assert annotations[0].params.get("rate") == "fast"
        assert annotations[0].params.get("pitch") == "+2st"

    def test_phoneme_creates_annotation(self):
        ssml = "<speak>I say <phoneme ph='təˈmeɪtoʊ'>tomato</phoneme></speak>"
        text, annotations = parse_ssml(ssml)
        assert "tomato" in text
        assert len(annotations) == 1
        assert annotations[0].type == "phoneme"

    def test_annotation_spans_correct(self):
        ssml = "<speak>AB<emphasis>CD</emphasis>EF</speak>"
        text, annotations = parse_ssml(ssml)
        assert text == "ABCDEF"
        assert annotations[0].start == 2
        assert annotations[0].end == 4

    def test_nested_elements(self):
        ssml = "<speak><prosody rate='slow'><emphasis>word</emphasis></prosody></speak>"
        text, annotations = parse_ssml(ssml)
        assert "word" in text
        assert len(annotations) == 2  # one for prosody, one for emphasis


class TestMalformedSSML:
    def test_unclosed_tags_returns_original(self):
        text, annotations = parse_ssml("<speak>Hello <break")
        assert "Hello" in text
        assert annotations == []

    def test_empty_speak(self):
        text, _ = parse_ssml("<speak></speak>")
        assert text == ""

    def test_non_xml_returns_original(self):
        text, _ = parse_ssml("Just plain text with no XML at all")
        assert "Just plain text" in text
