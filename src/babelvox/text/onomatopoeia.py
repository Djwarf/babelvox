"""Onomatopoeia detection for BabelVox.

Identifies onomatopoeia in text and produces Annotation objects that
Phase 2 (prosody control) can use to adjust emphasis and expression.

In Phase 1, the text itself is not modified — only annotations are
generated for downstream consumption.
"""
from __future__ import annotations

import re

from babelvox.text.ssml import Annotation

# ── Onomatopoeia database ─────────────────────────────────────────────
# category: semantic grouping for future prosody mapping
# emphasis: suggested emphasis level ("strong", "medium", "weak")
ONOMATOPOEIA_DB: dict[str, dict[str, str]] = {
    # Impacts / explosions
    "boom": {"category": "impact", "emphasis": "strong"},
    "bang": {"category": "impact", "emphasis": "strong"},
    "crash": {"category": "impact", "emphasis": "strong"},
    "slam": {"category": "impact", "emphasis": "strong"},
    "smash": {"category": "impact", "emphasis": "strong"},
    "thud": {"category": "impact", "emphasis": "medium"},
    "thump": {"category": "impact", "emphasis": "medium"},
    "clang": {"category": "impact", "emphasis": "medium"},
    "clank": {"category": "impact", "emphasis": "medium"},
    "crack": {"category": "impact", "emphasis": "strong"},
    "pop": {"category": "impact", "emphasis": "weak"},
    "snap": {"category": "impact", "emphasis": "medium"},
    # Water / liquid
    "splash": {"category": "water", "emphasis": "medium"},
    "drip": {"category": "water", "emphasis": "weak"},
    "gurgle": {"category": "water", "emphasis": "weak"},
    "splatter": {"category": "water", "emphasis": "medium"},
    "squish": {"category": "water", "emphasis": "weak"},
    "slurp": {"category": "water", "emphasis": "medium"},
    # Fire / heat
    "sizzle": {"category": "heat", "emphasis": "medium"},
    "crackle": {"category": "heat", "emphasis": "medium"},
    "fizzle": {"category": "heat", "emphasis": "weak"},
    # Air / wind
    "whoosh": {"category": "air", "emphasis": "medium"},
    "whiz": {"category": "air", "emphasis": "medium"},
    "hiss": {"category": "air", "emphasis": "medium"},
    "whisper": {"category": "air", "emphasis": "weak"},
    "swoosh": {"category": "air", "emphasis": "medium"},
    "rustle": {"category": "air", "emphasis": "weak"},
    # Animal sounds
    "buzz": {"category": "animal", "emphasis": "medium"},
    "chirp": {"category": "animal", "emphasis": "weak"},
    "growl": {"category": "animal", "emphasis": "medium"},
    "roar": {"category": "animal", "emphasis": "strong"},
    "howl": {"category": "animal", "emphasis": "strong"},
    "squawk": {"category": "animal", "emphasis": "medium"},
    "hoot": {"category": "animal", "emphasis": "weak"},
    "meow": {"category": "animal", "emphasis": "weak"},
    "woof": {"category": "animal", "emphasis": "medium"},
    "moo": {"category": "animal", "emphasis": "medium"},
    "quack": {"category": "animal", "emphasis": "weak"},
    "ribbit": {"category": "animal", "emphasis": "weak"},
    # Human sounds
    "gasp": {"category": "human", "emphasis": "medium"},
    "sigh": {"category": "human", "emphasis": "weak"},
    "groan": {"category": "human", "emphasis": "medium"},
    "shriek": {"category": "human", "emphasis": "strong"},
    "giggle": {"category": "human", "emphasis": "weak"},
    "murmur": {"category": "human", "emphasis": "weak"},
    # Mechanical
    "click": {"category": "mechanical", "emphasis": "weak"},
    "clatter": {"category": "mechanical", "emphasis": "medium"},
    "rattle": {"category": "mechanical", "emphasis": "medium"},
    "screech": {"category": "mechanical", "emphasis": "strong"},
    "squeak": {"category": "mechanical", "emphasis": "weak"},
    "whir": {"category": "mechanical", "emphasis": "weak"},
    "beep": {"category": "mechanical", "emphasis": "weak"},
    "ding": {"category": "mechanical", "emphasis": "weak"},
    "ring": {"category": "mechanical", "emphasis": "medium"},
    "honk": {"category": "mechanical", "emphasis": "medium"},
}

# Build a single regex that matches any known onomatopoeia as a whole word
_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(w) for w in ONOMATOPOEIA_DB) + r")\b",
    re.IGNORECASE,
)


def process_onomatopoeia(text: str) -> tuple[str, list[Annotation]]:
    """Detect onomatopoeia in text and produce annotations.

    The text is returned unchanged. Annotations mark the position and
    metadata of each detected onomatopoeia for downstream prosody use.

    Returns
    -------
    tuple[str, list[Annotation]]
        Original text and a list of onomatopoeia annotations.
    """
    annotations: list[Annotation] = []
    for match in _PATTERN.finditer(text):
        word = match.group().lower()
        meta = ONOMATOPOEIA_DB.get(word, {})
        annotations.append(Annotation(
            start=match.start(),
            end=match.end(),
            type="onomatopoeia",
            params={"word": word, **meta},
        ))
    return text, annotations
