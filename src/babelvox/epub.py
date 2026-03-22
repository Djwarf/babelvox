"""Epub reader for BabelVox audiobook generation.

Parses epub files into structured chapters with paragraph-level style
detection (narration, dialogue, thought, heading) for content-adaptive
synthesis.
"""
from __future__ import annotations

import logging
import os
import zipfile
from dataclasses import dataclass, field

import defusedxml.ElementTree as ET

logger = logging.getLogger("babelvox")

_XHTML_NS = "{http://www.w3.org/1999/xhtml}"
_OPF_NS = "{http://www.idpf.org/2007/opf}"
_DC_NS = "{http://purl.org/dc/elements/1.1/}"
_CONTAINER_NS = "{urn:oasis:names:tc:opendocument:xmlns:container}"
_NCX_NS = "{http://www.daisy.org/z3986/2005/ncx/}"


@dataclass
class Paragraph:
    """A single paragraph with style metadata for adaptive synthesis."""
    text: str
    style: str = "narration"  # narration, dialogue, thought, heading
    speaker_gender: str | None = None  # "male", "female", None


@dataclass
class Chapter:
    """A chapter extracted from an epub."""
    title: str
    index: int
    paragraphs: list[Paragraph] = field(default_factory=list)

    @property
    def text(self) -> str:
        """Join all paragraphs into plain text."""
        return "\n\n".join(p.text for p in self.paragraphs if p.text.strip())

    @property
    def word_count(self) -> int:
        return sum(len(p.text.split()) for p in self.paragraphs)


def _detect_style(text: str, has_italic: bool = False) -> str:
    """Detect paragraph style from text content and formatting."""
    stripped = text.strip()
    if not stripped:
        return "narration"

    # Dialogue: starts with quote mark (double, curly, or single)
    first = stripped[0]
    if first in ('"', '\u201c'):
        if '"' in stripped[1:] or '\u201d' in stripped[1:]:
            return "dialogue"
    elif first == '\u2018':  # curly single open quote
        if '\u2019' in stripped[1:]:
            return "dialogue"
    elif first == "'":
        # Single straight quote — dialogue if there's a closing quote
        # and it's not just an apostrophe (e.g. 'twas)
        rest = stripped[1:]
        if "'" in rest and len(rest) > 3:
            return "dialogue"

    # Thought: italic-wrapped content
    if has_italic:
        return "thought"

    # Heading-like: short, no sentence-ending punctuation
    words = stripped.split()
    if len(words) <= 6 and stripped[-1:] not in '.!?,;:':
        return "heading"

    return "narration"


_FEMALE_PRONOUNS = {"she", "her", "herself"}
_MALE_PRONOUNS = {"he", "him", "his", "himself"}

# Pattern: text after closing quote mark, looking for attribution pronouns
_ATTRIBUTION_WINDOW = 60  # characters after last quote to scan


def _detect_speaker_gender(text: str) -> str | None:
    """Detect speaker gender from dialogue attribution pronouns.

    Scans text after the last closing quote for pronouns like
    'she said', 'he replied', etc.
    """
    # Find last closing quote position
    last_quote = -1
    for q in ('"', '\u201d', "'", '\u2019'):
        pos = text.rfind(q)
        if pos > last_quote:
            last_quote = pos

    if last_quote < 0:
        return None

    # Scan attribution window after the quote
    after = text[last_quote + 1:last_quote + 1 + _ATTRIBUTION_WINDOW].lower()
    words = after.split()

    for word in words:
        # Strip punctuation from word
        clean = word.strip('.,;:!?')
        if clean in _FEMALE_PRONOUNS:
            return "female"
        if clean in _MALE_PRONOUNS:
            return "male"

    return None


def _extract_text(elem) -> str:
    """Extract all text from an XML element, stripping tags."""
    return "".join(elem.itertext()).strip()


def _has_dominant_italic(elem) -> bool:
    """Check if most of the element's text is wrapped in <i> tags."""
    total = len(_extract_text(elem))
    if total == 0:
        return False
    italic_len = 0
    for child in elem.iter():
        tag = child.tag.replace(_XHTML_NS, "").lower()
        if tag == "i":
            italic_len += len(_extract_text(child))
    return italic_len > total * 0.6


def _parse_content_file(zf: zipfile.ZipFile, path: str) -> list[Paragraph]:
    """Parse a single XHTML content file into paragraphs."""
    try:
        raw = zf.read(path)
    except KeyError:
        logger.warning("Content file not found in epub: %s", path)
        return []

    try:
        root = ET.fromstring(raw)
    except ET.ParseError:
        logger.warning("Failed to parse XHTML: %s", path)
        return []

    paragraphs = []
    body = root.find(f".//{_XHTML_NS}body")
    if body is None:
        body = root

    for elem in body.iter():
        tag = elem.tag.replace(_XHTML_NS, "").lower()

        if tag in ("h1", "h2", "h3", "h4", "h5", "h6"):
            text = _extract_text(elem)
            if text:
                paragraphs.append(Paragraph(text=text, style="heading"))

        elif tag == "p":
            text = _extract_text(elem)
            if not text or len(text.strip()) < 2:
                continue
            has_italic = _has_dominant_italic(elem)
            style = _detect_style(text, has_italic)
            gender = _detect_speaker_gender(text) if style == "dialogue" else None
            paragraphs.append(Paragraph(text=text, style=style,
                                        speaker_gender=gender))

    return paragraphs


def _find_opf_path(zf: zipfile.ZipFile) -> str:
    """Find the OPF file path from META-INF/container.xml."""
    try:
        container = ET.fromstring(zf.read("META-INF/container.xml"))
        rootfile = container.find(f".//{_CONTAINER_NS}rootfile")
        if rootfile is not None:
            return rootfile.get("full-path", "")
    except (KeyError, ET.ParseError):
        pass

    # Fallback: search for .opf file
    for name in zf.namelist():
        if name.endswith(".opf"):
            return name
    return ""


def _parse_toc_titles(zf: zipfile.ZipFile, opf_dir: str) -> dict[str, str]:
    """Parse NCX table of contents for chapter titles."""
    titles: dict[str, str] = {}
    for name in zf.namelist():
        if name.endswith(".ncx"):
            try:
                ncx = ET.fromstring(zf.read(name))
                for nav_point in ncx.iter(f"{_NCX_NS}navPoint"):
                    label_elem = nav_point.find(f"{_NCX_NS}navLabel/{_NCX_NS}text")
                    content_elem = nav_point.find(f"{_NCX_NS}content")
                    if label_elem is not None and content_elem is not None:
                        src = content_elem.get("src", "")
                        # Strip fragment
                        src = src.split("#")[0]
                        label = label_elem.text or ""
                        if src and label:
                            full_src = os.path.join(opf_dir, src).replace("\\", "/")
                            titles[full_src] = label
            except ET.ParseError:
                pass
            break
    return titles


def read_epub(path: str) -> list[Chapter]:
    """Read an epub file and return a list of chapters.

    Each chapter contains paragraphs tagged with style metadata
    (narration, dialogue, thought, heading) for content-adaptive synthesis.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"epub file not found: {path}")

    with zipfile.ZipFile(path, "r") as zf:
        opf_path = _find_opf_path(zf)
        if not opf_path:
            raise ValueError("No OPF manifest found in epub")

        opf_dir = os.path.dirname(opf_path).replace("\\", "/")
        opf = ET.fromstring(zf.read(opf_path))

        # Build manifest: id → href
        manifest = {}
        for item in opf.iter(f"{_OPF_NS}item"):
            item_id = item.get("id", "")
            href = item.get("href", "")
            media = item.get("media-type", "")
            if item_id and href and "html" in media:
                full_path = os.path.join(opf_dir, href).replace("\\", "/")
                manifest[item_id] = full_path

        # Reading order from spine
        spine_ids = []
        for itemref in opf.iter(f"{_OPF_NS}itemref"):
            idref = itemref.get("idref", "")
            if idref in manifest:
                spine_ids.append(idref)

        # Chapter titles from NCX
        toc_titles = _parse_toc_titles(zf, opf_dir)

        # Parse each content file
        chapters = []
        chapter_idx = 0
        for item_id in spine_ids:
            content_path = manifest[item_id]
            paragraphs = _parse_content_file(zf, content_path)

            # Skip empty content files (cover, copyright, etc.)
            text_paras = [p for p in paragraphs if p.style != "heading" and p.text.strip()]
            if not text_paras and not any(p.style == "heading" for p in paragraphs):
                continue

            # Determine chapter title
            title = toc_titles.get(content_path, "")
            if not title:
                # Use first heading if available
                for p in paragraphs:
                    if p.style == "heading":
                        title = p.text
                        break
            if not title:
                title = f"Chapter {chapter_idx + 1}"

            chapters.append(Chapter(
                title=title,
                index=chapter_idx,
                paragraphs=paragraphs,
            ))
            chapter_idx += 1

        logger.info("Read epub: %d chapters, %d total paragraphs",
                    len(chapters),
                    sum(len(c.paragraphs) for c in chapters))
        return chapters


def parse_chapter_range(spec: str, total: int) -> list[int]:
    """Parse a chapter range spec like '1-5' or '3' or '1,3,5-7'.

    Returns zero-based indices.
    """
    indices = set()
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            start, end = part.split("-", 1)
            start = max(1, int(start))
            end = min(total, int(end))
            indices.update(range(start - 1, end))
        else:
            idx = int(part) - 1
            if 0 <= idx < total:
                indices.add(idx)
    return sorted(indices)
