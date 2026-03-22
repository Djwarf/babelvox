"""Tests for epub reader."""
import io
import zipfile

import pytest

from babelvox.epub import (
    _detect_speaker_gender,
    _detect_style,
    parse_chapter_range,
    read_epub,
)


class TestDetectStyle:
    def test_dialogue_double_quotes(self):
        assert _detect_style('"Hello," she said.') == "dialogue"

    def test_dialogue_curly_quotes(self):
        assert _detect_style('\u201cHello,\u201d she said.') == "dialogue"

    def test_dialogue_single_quotes(self):
        assert _detect_style("'I'm quite cured of seeking pleasure.'") == "dialogue"

    def test_dialogue_curly_single_quotes(self):
        assert _detect_style('\u2018Hello,\u2019 she said.') == "dialogue"

    def test_narration_plain(self):
        assert _detect_style("The sun was setting over the hills.") == "narration"

    def test_thought_italic(self):
        assert _detect_style("I need to get out of here.", has_italic=True) == "thought"

    def test_heading_short_no_punct(self):
        assert _detect_style("Chapter One") == "heading"

    def test_not_heading_with_period(self):
        assert _detect_style("Chapter One.") != "heading"

    def test_empty(self):
        assert _detect_style("") == "narration"


class TestDetectSpeakerGender:
    def test_she_said(self):
        assert _detect_speaker_gender('"Hello," she said.') == "female"

    def test_he_replied(self):
        assert _detect_speaker_gender('"Hello," he replied.') == "male"

    def test_said_he(self):
        assert _detect_speaker_gender('"Hello," said he.') == "male"

    def test_her_voice(self):
        assert _detect_speaker_gender('"Stop!" Her voice was firm.') == "female"

    def test_no_attribution(self):
        assert _detect_speaker_gender('"Hello there."') is None

    def test_narration_no_quotes(self):
        assert _detect_speaker_gender("He walked away.") is None

    def test_single_quote_she(self):
        assert _detect_speaker_gender("'Hello,' she whispered.") == "female"


class TestParseChapterRange:
    def test_single(self):
        assert parse_chapter_range("3", 10) == [2]

    def test_range(self):
        assert parse_chapter_range("1-3", 10) == [0, 1, 2]

    def test_comma_separated(self):
        assert parse_chapter_range("1,3,5", 10) == [0, 2, 4]

    def test_mixed(self):
        assert parse_chapter_range("1-3,7", 10) == [0, 1, 2, 6]

    def test_clamps_to_total(self):
        assert parse_chapter_range("1-100", 5) == [0, 1, 2, 3, 4]

    def test_out_of_range_ignored(self):
        assert parse_chapter_range("99", 5) == []


def _make_epub(chapters_html, toc_labels=None):
    """Create a minimal epub zip in memory."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        # container.xml
        zf.writestr("META-INF/container.xml", """<?xml version="1.0"?>
<container xmlns="urn:oasis:names:tc:opendocument:xmlns:container" version="1.0">
  <rootfiles>
    <rootfile full-path="content.opf" media-type="application/oebps-package+xml"/>
  </rootfiles>
</container>""")

        # Build manifest + spine
        items = []
        spine = []
        for i, html in enumerate(chapters_html):
            item_id = f"ch{i}"
            href = f"ch{i}.xhtml"
            items.append(f'<item id="{item_id}" href="{href}" '
                         f'media-type="application/xhtml+xml"/>')
            spine.append(f'<itemref idref="{item_id}"/>')
            zf.writestr(href, html)

        opf = f"""<?xml version="1.0"?>
<package xmlns="http://www.idpf.org/2007/opf" version="2.0">
  <metadata xmlns:dc="http://purl.org/dc/elements/1.1/">
    <dc:title>Test Book</dc:title>
  </metadata>
  <manifest>
    {''.join(items)}
  </manifest>
  <spine>
    {''.join(spine)}
  </spine>
</package>"""
        zf.writestr("content.opf", opf)

        # NCX with labels
        if toc_labels:
            nav_points = []
            for i, label in enumerate(toc_labels):
                nav_points.append(f"""
<navPoint id="np{i}" playOrder="{i+1}">
  <navLabel><text>{label}</text></navLabel>
  <content src="ch{i}.xhtml"/>
</navPoint>""")
            ncx = f"""<?xml version="1.0"?>
<ncx xmlns="http://www.daisy.org/z3986/2005/ncx/">
  <navMap>{''.join(nav_points)}</navMap>
</ncx>"""
            zf.writestr("toc.ncx", ncx)

    buf.seek(0)
    return buf


def _xhtml(body_content):
    return f"""<?xml version="1.0" encoding="utf-8"?>
<html xmlns="http://www.w3.org/1999/xhtml">
<head><title>Test</title></head>
<body>{body_content}</body>
</html>"""


class TestReadEpub:
    def test_basic_chapter(self, tmp_path):
        html = _xhtml('<h1>Chapter 1</h1><p>First paragraph.</p><p>Second paragraph.</p>')
        epub_buf = _make_epub([html], ["Chapter 1"])
        epub_path = str(tmp_path / "test.epub")
        with open(epub_path, "wb") as f:
            f.write(epub_buf.read())

        chapters = read_epub(epub_path)
        assert len(chapters) == 1
        assert chapters[0].title == "Chapter 1"
        assert len(chapters[0].paragraphs) >= 2

    def test_dialogue_detection(self, tmp_path):
        html = _xhtml('<p>"Hello," she said.</p><p>He nodded silently.</p>')
        epub_buf = _make_epub([html])
        epub_path = str(tmp_path / "test.epub")
        with open(epub_path, "wb") as f:
            f.write(epub_buf.read())

        chapters = read_epub(epub_path)
        styles = [p.style for p in chapters[0].paragraphs]
        assert "dialogue" in styles
        assert "narration" in styles

    def test_thought_detection(self, tmp_path):
        html = _xhtml('<p><i>I need to think about this.</i></p>')
        epub_buf = _make_epub([html])
        epub_path = str(tmp_path / "test.epub")
        with open(epub_path, "wb") as f:
            f.write(epub_buf.read())

        chapters = read_epub(epub_path)
        assert chapters[0].paragraphs[0].style == "thought"

    def test_multiple_chapters(self, tmp_path):
        ch1 = _xhtml('<h1>One</h1><p>Content one.</p>')
        ch2 = _xhtml('<h1>Two</h1><p>Content two.</p>')
        epub_buf = _make_epub([ch1, ch2], ["Chapter 1", "Chapter 2"])
        epub_path = str(tmp_path / "test.epub")
        with open(epub_path, "wb") as f:
            f.write(epub_buf.read())

        chapters = read_epub(epub_path)
        assert len(chapters) == 2

    def test_chapter_text_property(self, tmp_path):
        html = _xhtml('<p>First.</p><p>Second.</p>')
        epub_buf = _make_epub([html])
        epub_path = str(tmp_path / "test.epub")
        with open(epub_path, "wb") as f:
            f.write(epub_buf.read())

        chapters = read_epub(epub_path)
        text = chapters[0].text
        assert "First." in text
        assert "Second." in text

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            read_epub("/nonexistent/path.epub")
