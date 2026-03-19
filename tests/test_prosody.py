"""Tests for prosody and emotion control."""
import numpy as np

from babelvox.prosody import (
    ProsodyConfig,
    adjust_sampling_for_emotion,
    apply_text_prosody,
    apply_waveform_prosody,
)

# ── Text prosody tests ────────────────────────────────────────────────

class TestApplyTextProsody:
    def test_happy_adds_exclamation(self):
        result = apply_text_prosody("Hello world", ProsodyConfig(emotion="happy"))
        assert result.endswith("!")

    def test_happy_replaces_period(self):
        result = apply_text_prosody("Hello world.", ProsodyConfig(emotion="happy"))
        assert result.endswith("!")
        assert not result.endswith(".!")

    def test_sad_adds_ellipsis(self):
        result = apply_text_prosody("Hello world", ProsodyConfig(emotion="sad"))
        assert result.endswith("...")

    def test_sad_replaces_period(self):
        result = apply_text_prosody("Hello world.", ProsodyConfig(emotion="sad"))
        assert result.endswith("...")

    def test_angry_adds_exclamation(self):
        result = apply_text_prosody("Hello world", ProsodyConfig(emotion="angry"))
        assert result.endswith("!")

    def test_angry_capitalizes_first_word(self):
        result = apply_text_prosody("hello world", ProsodyConfig(emotion="angry"))
        assert result.startswith("HELLO")

    def test_surprised_adds_exclamation(self):
        result = apply_text_prosody("Oh really", ProsodyConfig(emotion="surprised"))
        assert result.endswith("!")

    def test_neutral_unchanged(self):
        result = apply_text_prosody("Hello world.", ProsodyConfig(emotion="neutral"))
        assert result == "Hello world."

    def test_none_emotion_unchanged(self):
        result = apply_text_prosody("Hello world.", ProsodyConfig())
        assert result == "Hello world."

    def test_emphasis_words_capitalized(self):
        config = ProsodyConfig(emphasis_words=["important"])
        result = apply_text_prosody("This is important stuff.", config)
        assert "IMPORTANT" in result

    def test_emphasis_case_insensitive(self):
        config = ProsodyConfig(emphasis_words=["hello"])
        result = apply_text_prosody("Hello world", config)
        assert "HELLO" in result


# ── Sampling adjustment tests ─────────────────────────────────────────

class TestAdjustSampling:
    def test_happy_increases_temperature(self):
        temp, top_k = adjust_sampling_for_emotion("happy", 0.9, 50)
        assert temp > 0.9

    def test_sad_decreases_temperature(self):
        temp, top_k = adjust_sampling_for_emotion("sad", 0.9, 50)
        assert temp < 0.9

    def test_angry_adjusts_both(self):
        temp, top_k = adjust_sampling_for_emotion("angry", 0.9, 50)
        assert temp > 0.9
        assert top_k < 50

    def test_neutral_unchanged(self):
        temp, top_k = adjust_sampling_for_emotion("neutral", 0.9, 50)
        assert temp == 0.9
        assert top_k == 50

    def test_none_unchanged(self):
        temp, top_k = adjust_sampling_for_emotion(None, 0.9, 50)
        assert temp == 0.9
        assert top_k == 50

    def test_surprised_increases_temperature(self):
        temp, top_k = adjust_sampling_for_emotion("surprised", 0.9, 50)
        assert temp > 0.9


# ── Waveform prosody tests ───────────────────────────────────────────

class TestApplyWaveformProsody:
    def _make_wav(self, duration_secs=1.0, sr=24000):
        """Generate a sine wave for testing."""
        t = np.linspace(0, duration_secs, int(sr * duration_secs), dtype=np.float32)
        return np.sin(2 * np.pi * 440 * t)

    def test_rate_changes_length(self):
        wav = self._make_wav(1.0)
        result = apply_waveform_prosody(wav, 24000, ProsodyConfig(rate=2.0))
        # Doubling rate should roughly halve length
        assert len(result) < len(wav) * 0.7

    def test_rate_slow_increases_length(self):
        wav = self._make_wav(1.0)
        result = apply_waveform_prosody(wav, 24000, ProsodyConfig(rate=0.5))
        assert len(result) > len(wav) * 1.5

    def test_pitch_preserves_length(self):
        wav = self._make_wav(1.0)
        result = apply_waveform_prosody(wav, 24000, ProsodyConfig(pitch_semitones=4.0))
        # Pitch shift should preserve approximate length
        assert abs(len(result) - len(wav)) < 100

    def test_volume_scales_amplitude(self):
        wav = self._make_wav(0.1)
        result = apply_waveform_prosody(wav, 24000, ProsodyConfig(volume=0.5))
        # Half volume → half amplitude
        assert np.max(np.abs(result)) < np.max(np.abs(wav)) * 0.6

    def test_volume_clips_at_one(self):
        wav = self._make_wav(0.1)
        result = apply_waveform_prosody(wav, 24000, ProsodyConfig(volume=2.0))
        assert np.max(np.abs(result)) <= 1.0

    def test_no_changes_passthrough(self):
        wav = self._make_wav(0.1)
        result = apply_waveform_prosody(wav, 24000, ProsodyConfig())
        np.testing.assert_array_equal(result, wav)

    def test_combined_effects(self):
        wav = self._make_wav(1.0)
        config = ProsodyConfig(rate=1.5, pitch_semitones=2.0, volume=0.8)
        result = apply_waveform_prosody(wav, 24000, config)
        # Should be shorter (rate > 1) and quieter (volume < 1)
        assert len(result) < len(wav)
        assert np.max(np.abs(result)) <= 1.0


# ── ProsodyConfig tests ──────────────────────────────────────────────

class TestProsodyConfig:
    def test_defaults_are_identity(self):
        config = ProsodyConfig()
        assert config.rate == 1.0
        assert config.pitch_semitones == 0.0
        assert config.volume == 1.0
        assert config.emotion is None
        assert config.emphasis_words == []

    def test_custom_values(self):
        config = ProsodyConfig(rate=1.5, pitch_semitones=-2, volume=0.8,
                               emotion="happy", emphasis_words=["wow"])
        assert config.rate == 1.5
        assert config.emotion == "happy"
        assert config.emphasis_words == ["wow"]
