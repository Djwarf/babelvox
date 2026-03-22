"""SSML (Speech Synthesis Markup Language) parser for BabelVox.

Parses a useful subset of the W3C SSML 1.1 spec:
  Immediate: <break>, <sub>, <say-as>, <emphasis>, <p>, <s>
  Deferred:  <prosody>, <phoneme>, <voice>, <mark>

Immediate tags modify text during parsing. Deferred tags create
Annotation objects for the pipeline to consume downstream.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import defusedxml.ElementTree as ET

SSML_NS = "http://www.w3.org/2001/10/synthesis"
_NS_RE = re.compile(r"\{[^}]*\}")


def _strip_ns(tag: str) -> str:
    """Remove XML namespace prefix from a tag name."""
    return _NS_RE.sub("", tag)


@dataclass
class Annotation:
    """Span-level metadata attached to a range of the output text."""
    start: int
    end: int
    type: str  # "emphasis", "prosody", "phoneme"
    params: dict[str, Any] = field(default_factory=dict)


def _parse_break(elem: ET.Element) -> str:
    """Convert a <break> element to punctuation that hints a pause."""
    time_attr = elem.get("time", "")
    strength = elem.get("strength", "")

    if time_attr:
        match = re.match(r"(\d+)\s*(ms|s)", time_attr)
        if match:
            value, unit = int(match.group(1)), match.group(2)
            ms = value if unit == "ms" else value * 1000
            if ms < 300:
                return ","
            if ms <= 700:
                return "."
            return "..."

    strength_map = {
        "none": "",
        "x-weak": ",",
        "weak": ",",
        "medium": ".",
        "strong": ". ",
        "x-strong": "... ",
    }
    return strength_map.get(strength, ".")


def _apply_emphasis(text: str, level: str) -> str:
    """Apply emphasis capitalization to text."""
    level = level.lower() if level else "moderate"
    if level == "strong":
        return text.upper()
    if level == "moderate":
        return text.title()
    if level == "reduced":
        return text.lower()
    return text  # "none" or unknown


def _walk(elem: ET.Element, text_parts: list[str],
          annotations: list[Annotation], normalizer_fn) -> None:
    """Recursively walk SSML element tree, building plain text + annotations."""
    tag = _strip_ns(elem.tag)

    if tag == "break":
        text_parts.append(_parse_break(elem))
        if elem.tail:
            text_parts.append(elem.tail)
        return

    if tag == "sub":
        alias = elem.get("alias", "")
        if alias:
            text_parts.append(alias)
        else:
            if elem.text:
                text_parts.append(elem.text)
        if elem.tail:
            text_parts.append(elem.tail)
        return

    if tag == "say-as":
        inner = "".join(elem.itertext())
        interpret_as = elem.get("interpret-as", "")
        fmt = elem.get("format", "")
        if normalizer_fn and interpret_as:
            converted = normalizer_fn(inner, interpret_as, fmt)
            text_parts.append(converted)
        else:
            text_parts.append(inner)
        if elem.tail:
            text_parts.append(elem.tail)
        return

    # <emphasis> — immediate: apply capitalization based on level
    if tag == "emphasis":
        inner_parts: list[str] = []
        if elem.text:
            inner_parts.append(elem.text)
        for child in elem:
            _walk(child, inner_parts, annotations, normalizer_fn)
        inner_text = "".join(inner_parts)
        level = elem.get("level", "moderate")
        text_parts.append(_apply_emphasis(inner_text, level))
        if elem.tail:
            text_parts.append(elem.tail)
        return

    # <p> / <s> — paragraph/sentence boundaries with trailing pause
    if tag in ("p", "s"):
        if elem.text:
            text_parts.append(elem.text)
        for child in elem:
            _walk(child, text_parts, annotations, normalizer_fn)
        # Ensure sentence-ending punctuation for natural pause
        current = "".join(text_parts)
        if current and current[-1] not in ".!?":
            text_parts.append(". ")
        else:
            text_parts.append(" ")
        if elem.tail:
            text_parts.append(elem.tail)
        return

    # <mark> — timing bookmark (self-closing)
    if tag == "mark":
        pos = sum(len(p) for p in text_parts)
        name = elem.get("name", "")
        annotations.append(Annotation(start=pos, end=pos,
                                       type="mark", params={"name": name}))
        if elem.tail:
            text_parts.append(elem.tail)
        return

    # Tags that produce annotations (deferred to pipeline)
    if tag in ("prosody", "phoneme", "voice"):
        start = sum(len(p) for p in text_parts)
        if elem.text:
            text_parts.append(elem.text)
        for child in elem:
            _walk(child, text_parts, annotations, normalizer_fn)
        end = sum(len(p) for p in text_parts)

        params: dict[str, Any] = dict(elem.attrib)
        annotations.append(Annotation(start=start, end=end,
                                       type=tag, params=params))
        if elem.tail:
            text_parts.append(elem.tail)
        return

    # Default: keep text, recurse into children
    if elem.text:
        text_parts.append(elem.text)

    for child in elem:
        _walk(child, text_parts, annotations, normalizer_fn)

    if elem.tail:
        text_parts.append(elem.tail)


def looks_like_ssml(text: str) -> bool:
    """Heuristic check for SSML content."""
    stripped = text.strip()
    return (stripped.startswith("<speak")
            or "<break" in stripped
            or "<sub " in stripped
            or "<voice" in stripped
            or "<mark" in stripped)


def parse_ssml(text: str, normalizer_fn=None) -> tuple[str, list[Annotation]]:
    """Parse SSML text into plain text and a list of annotations.

    Parameters
    ----------
    text : str
        SSML-formatted string (should be wrapped in <speak> tags).
    normalizer_fn : callable, optional
        Function(text, interpret_as, format) -> str for <say-as> handling.
        If None, <say-as> inner text is kept as-is.

    Returns
    -------
    tuple[str, list[Annotation]]
        Plain text with SSML tags removed/replaced, plus annotations
        for deferred tags (emphasis, prosody, phoneme).
    """
    # Ensure the text is wrapped in <speak>
    stripped = text.strip()
    if not stripped.startswith("<speak"):
        stripped = f"<speak>{stripped}</speak>"

    try:
        root = ET.fromstring(stripped)
    except ET.ParseError:
        # Malformed SSML — return original text unchanged
        return text, []

    text_parts: list[str] = []
    annotations: list[Annotation] = []
    _walk(root, text_parts, annotations, normalizer_fn)

    result = "".join(text_parts).strip()
    return result, annotations
