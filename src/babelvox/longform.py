"""Long-form text-to-speech synthesis for BabelVox.

Segments long text into manageable chunks, synthesizes each with
consistent voice, and concatenates with crossfade for seamless output.
Supports paragraph, sentence, natural, and chapter-based segmentation.
"""
from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger("babelvox")

# Abbreviations that shouldn't trigger sentence splits
_ABBREV_RE = re.compile(
    r'\b(?:Mr|Mrs|Ms|Dr|Prof|Jr|Sr|St|Ave|Blvd|vs|etc|approx|dept|govt'
    r'|inc|corp|ltd|est|vol|no|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Oct|Nov|Dec'
    r'|U\.S|U\.K)\.',
    re.IGNORECASE,
)

_SENTENCE_END_RE = re.compile(r'(?<=[.!?])\s+(?=[A-Z"\'])')
_CHAPTER_RE = re.compile(r'^#{1,6}\s+.+$|^---+\s*$', re.MULTILINE)


@dataclass
class Segment:
    """A chunk of text to be synthesized."""
    text: str
    index: int
    type: str = "paragraph"  # "paragraph", "sentence", "chapter_marker"


@dataclass
class SegmentResult:
    """Timing metadata for a synthesized segment."""
    segment: Segment
    start_time: float
    end_time: float
    duration: float


@dataclass
class SynthesisProgress:
    """Progress update during long-form synthesis."""
    total_segments: int
    completed_segments: int
    current_segment_index: int
    elapsed_seconds: float
    total_audio_seconds: float


@dataclass
class SynthesisResult:
    """Complete result of long-form synthesis."""
    waveform: np.ndarray
    sample_rate: int = 24000
    segments: list[SegmentResult] = field(default_factory=list)
    total_duration: float = 0.0


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences, respecting abbreviations."""
    # Replace abbreviation periods with a placeholder
    placeholders = {}
    for i, match in enumerate(_ABBREV_RE.finditer(text)):
        placeholder = f"\x00ABBR{i}\x00"
        placeholders[placeholder] = match.group()

    masked = text
    for placeholder, original in placeholders.items():
        masked = masked.replace(original, placeholder, 1)

    # Split on sentence boundaries
    parts = _SENTENCE_END_RE.split(masked)

    # Restore abbreviations
    sentences = []
    for part in parts:
        for placeholder, original in placeholders.items():
            part = part.replace(placeholder, original)
        part = part.strip()
        if part:
            sentences.append(part)
    return sentences


def segment_text(text: str, strategy: str = "natural") -> list[Segment]:
    """Split text into segments using the specified strategy.

    Strategies:
        paragraph: Split on double newlines.
        sentence: Split on sentence-ending punctuation.
        natural: Paragraphs first, split long paragraphs (>300 chars) into sentences.
        chapter: Split on markdown headers/dividers; headers become chapter_marker segments.
    """
    if strategy == "paragraph":
        parts = [p.strip() for p in text.split("\n\n")]
        return [Segment(text=p, index=i, type="paragraph")
                for i, p in enumerate(parts) if p]

    if strategy == "sentence":
        sentences = _split_sentences(text)
        return [Segment(text=s, index=i, type="sentence")
                for i, s in enumerate(sentences) if s]

    if strategy == "natural":
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        segments = []
        idx = 0
        for para in paragraphs:
            if len(para) > 300:
                for sent in _split_sentences(para):
                    if sent:
                        segments.append(Segment(text=sent, index=idx, type="sentence"))
                        idx += 1
            else:
                segments.append(Segment(text=para, index=idx, type="paragraph"))
                idx += 1
        return segments

    if strategy == "chapter":
        segments = []
        idx = 0
        last_end = 0
        for match in _CHAPTER_RE.finditer(text):
            # Text before the header
            before = text[last_end:match.start()].strip()
            if before:
                for para in before.split("\n\n"):
                    para = para.strip()
                    if para:
                        segments.append(Segment(text=para, index=idx, type="paragraph"))
                        idx += 1
            # The header itself
            header = match.group().strip()
            if header and not header.startswith("---"):
                segments.append(Segment(text=header, index=idx, type="chapter_marker"))
                idx += 1
            last_end = match.end()

        # Remaining text after last header
        remaining = text[last_end:].strip()
        if remaining:
            for para in remaining.split("\n\n"):
                para = para.strip()
                if para:
                    segments.append(Segment(text=para, index=idx, type="paragraph"))
                    idx += 1
        return segments

    raise ValueError(f"unknown segmentation strategy: {strategy!r}")


def crossfade_concat(waveforms: list[np.ndarray],
                     crossfade_samples: int = 2400) -> np.ndarray:
    """Concatenate waveforms with crossfade overlap between adjacent pairs."""
    if not waveforms:
        return np.zeros(1, dtype=np.float32)
    if len(waveforms) == 1:
        return waveforms[0]

    result_parts = [waveforms[0]]
    for i in range(1, len(waveforms)):
        prev = result_parts[-1]
        curr = waveforms[i]
        ol = min(crossfade_samples, len(prev) // 2, len(curr) // 2)

        if ol > 0:
            fade_out = np.linspace(1.0, 0.0, ol, dtype=np.float32)
            fade_in = np.linspace(0.0, 1.0, ol, dtype=np.float32)
            # Trim tail of previous, blend with head of current
            blended = prev[-ol:] * fade_out + curr[:ol] * fade_in
            result_parts[-1] = prev[:-ol]
            result_parts.append(blended)
            result_parts.append(curr[ol:])
        else:
            result_parts.append(curr)

    return np.concatenate(result_parts)


class LongFormSynthesizer:
    """Orchestrates long-form text-to-speech synthesis.

    Segments text, synthesizes each segment with consistent voice,
    and concatenates with crossfade for seamless output.
    """

    def __init__(self, tts):
        self.tts = tts

    def synthesize(self, text, strategy="natural", speaker=None,
                   speaker_embed=None, language="English", prosody=None,
                   crossfade_samples=2400, max_new_tokens=512,
                   progress_callback=None, resume_from=0,
                   seed=None):
        """Synthesize long text as a single waveform.

        Returns a SynthesisResult with the concatenated waveform,
        per-segment timestamps, and total duration.
        """
        segments = segment_text(text, strategy)

        # Resolve speaker once for voice consistency
        if speaker and self.tts.speaker_library:
            speaker_embed = self.tts.speaker_library.load(speaker).embedding
        elif speaker_embed is None and self.tts.default_speaker is not None:
            speaker_embed = self.tts.default_speaker

        waveforms = []
        segment_results = []
        t0 = time.time()
        current_time = 0.0

        logger.info("Long-form synthesis: %d segments (%s strategy)",
                     len(segments), strategy)

        for seg in segments[resume_from:]:
            if seg.type == "chapter_marker":
                segment_results.append(SegmentResult(
                    segment=seg, start_time=current_time,
                    end_time=current_time, duration=0.0))
                logger.debug("  [%d] chapter marker: %s", seg.index, seg.text)
                continue

            if seed is not None:
                np.random.seed(seed + seg.index)

            logger.debug("  [%d] synthesizing: %.60s...", seg.index, seg.text)
            wav, sr = self.tts.generate(
                text=seg.text, language=language,
                speaker_embed=speaker_embed, prosody=prosody,
                max_new_tokens=max_new_tokens)

            duration = len(wav) / sr
            segment_results.append(SegmentResult(
                segment=seg, start_time=current_time,
                end_time=current_time + duration, duration=duration))
            current_time += duration
            waveforms.append(wav)

            if progress_callback:
                progress_callback(SynthesisProgress(
                    total_segments=len(segments),
                    completed_segments=len(waveforms),
                    current_segment_index=seg.index,
                    elapsed_seconds=time.time() - t0,
                    total_audio_seconds=current_time))

        if not waveforms:
            return SynthesisResult(waveform=np.zeros(1, dtype=np.float32))

        full_wav = crossfade_concat(waveforms, crossfade_samples)
        logger.info("Long-form complete: %.1fs audio from %d segments",
                     len(full_wav) / 24000, len(waveforms))

        return SynthesisResult(
            waveform=full_wav, sample_rate=24000,
            segments=segment_results,
            total_duration=len(full_wav) / 24000)

    def synthesize_stream(self, text, strategy="natural", speaker=None,
                          speaker_embed=None, language="English",
                          prosody=None, max_new_tokens=512,
                          resume_from=0, seed=None):
        """Yield (waveform, sample_rate, progress) per segment."""
        segments = segment_text(text, strategy)

        if speaker and self.tts.speaker_library:
            speaker_embed = self.tts.speaker_library.load(speaker).embedding
        elif speaker_embed is None and self.tts.default_speaker is not None:
            speaker_embed = self.tts.default_speaker

        t0 = time.time()
        current_time = 0.0
        completed = 0

        for seg in segments[resume_from:]:
            if seg.type == "chapter_marker":
                continue

            if seed is not None:
                np.random.seed(seed + seg.index)

            wav, sr = self.tts.generate(
                text=seg.text, language=language,
                speaker_embed=speaker_embed, prosody=prosody,
                max_new_tokens=max_new_tokens)

            current_time += len(wav) / sr
            completed += 1

            progress = SynthesisProgress(
                total_segments=len(segments),
                completed_segments=completed,
                current_segment_index=seg.index,
                elapsed_seconds=time.time() - t0,
                total_audio_seconds=current_time)

            yield wav, sr, progress
