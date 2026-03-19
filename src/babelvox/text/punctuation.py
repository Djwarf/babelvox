"""Punctuation normalization for BabelVox.

Cleans up Unicode punctuation, collapses repeated punctuation, and
ensures consistent spacing around sentence-ending punctuation so the
tokenizer produces clean, predictable tokens.
"""
from __future__ import annotations

import re

# ── Unicode punctuation → ASCII equivalents ───────────────────────────
_UNICODE_MAP = {
    "\u2018": "'", "\u2019": "'",   # smart single quotes
    "\u201c": '"', "\u201d": '"',   # smart double quotes
    "\u2013": " - ",                 # en-dash
    "\u2014": " -- ",                # em-dash
    "\u2015": " -- ",                # horizontal bar
    "\u2026": "...",                  # horizontal ellipsis
    "\u00ab": '"', "\u00bb": '"',   # guillemets
    "\u2039": "'", "\u203a": "'",   # single guillemets
    "\u201a": ",",                    # single low-9 quotation
    "\u201e": '"',                   # double low-9 quotation
    "\u2010": "-", "\u2011": "-",   # hyphens
    "\u00a0": " ",                    # non-breaking space
}

_UNICODE_RE = re.compile("|".join(re.escape(k) for k in _UNICODE_MAP))

# ── Repeated punctuation patterns ─────────────────────────────────────
_MULTI_EXCL = re.compile(r"!{2,}")
_MULTI_QUEST = re.compile(r"\?{2,}")
_MULTI_PERIOD = re.compile(r"\.{4,}")

# ── Spacing after sentence-enders ─────────────────────────────────────
_SENTENCE_END_SPACE = re.compile(r"([.!?])([A-Z])")


def normalize_punctuation(text: str) -> str:
    """Normalize punctuation for cleaner tokenization.

    - Converts Unicode punctuation to ASCII equivalents
    - Collapses repeated !!! → ! and ??? → ?
    - Normalizes excessive dots (4+) to ellipsis (...)
    - Ensures space after sentence-ending punctuation before uppercase
    """
    # Unicode → ASCII
    text = _UNICODE_RE.sub(lambda m: _UNICODE_MAP[m.group()], text)

    # Collapse repeated punctuation
    text = _MULTI_EXCL.sub("!", text)
    text = _MULTI_QUEST.sub("?", text)
    text = _MULTI_PERIOD.sub("...", text)

    # Ensure space after sentence enders before new sentence
    text = _SENTENCE_END_SPACE.sub(r"\1 \2", text)

    return text
