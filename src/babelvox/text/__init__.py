"""Text preprocessing pipeline for BabelVox.

Provides SSML parsing, text normalization, punctuation cleanup,
and onomatopoeia detection as a single preprocessing step.
"""
from __future__ import annotations

from babelvox.text.normalizer import normalize_say_as, normalize_text
from babelvox.text.onomatopoeia import process_onomatopoeia
from babelvox.text.punctuation import normalize_punctuation
from babelvox.text.ssml import Annotation, looks_like_ssml, parse_ssml

__all__ = [
    "preprocess_text",
    "parse_ssml",
    "Annotation",
    "normalize_text",
    "normalize_punctuation",
    "process_onomatopoeia",
]


def preprocess_text(text: str, language: str = "english",
                    is_ssml: bool = False) -> str:
    """Run the full text preprocessing pipeline.

    Parameters
    ----------
    text : str
        Raw input text, optionally containing SSML markup.
    language : str
        Language name (e.g. "english", "french").
    is_ssml : bool
        If True, force SSML parsing. If False, auto-detect.

    Returns
    -------
    str
        Cleaned plain text ready for tokenization.
    """
    if is_ssml or looks_like_ssml(text):
        def _say_as_fn(inner, interpret_as, fmt):
            return normalize_say_as(inner, interpret_as, fmt, language)

        text, _annotations = parse_ssml(text, normalizer_fn=_say_as_fn)

    text = normalize_text(text, language)
    text = normalize_punctuation(text)
    return text
