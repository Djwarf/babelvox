"""Tests for prosody and emotion control."""
import numpy as np

from babelvox.prosody import (
    ProsodyConfig,
    adjust_sampling_for_emotion,
    apply_text_prosody,
    apply_waveform_prosody,
    emotion_schedule,
    get_emotion_bias,
    get_emotion_subtalker_params,
)

# ── Text prosody tests ────────────────────────────────────────────────

class TestApplyTextProsody:
    def test_happy_adds_exclamation(self):
        result = apply_text_prosody("Hello world", ProsodyConfig(emotion="happy"))
        assert result.endswith("!")

    def test_happy_replaces_all_periods(self):
        result = apply_text_prosody("First. Second.", ProsodyConfig(emotion="happy"))
        assert "." not in result
        assert "!" in result

    def test_sad_adds_ellipsis(self):
        result = apply_text_prosody("Hello world", ProsodyConfig(emotion="sad"))
        assert result.endswith("...")

    def test_sad_replaces_all_periods(self):
        result = apply_text_prosody("First. Second.", ProsodyConfig(emotion="sad"))
        assert "..." in result

    def test_angry_adds_exclamation(self):
        result = apply_text_prosody("Hello world", ProsodyConfig(emotion="angry"))
        assert result.endswith("!")

    def test_angry_capitalizes_first_word(self):
        result = apply_text_prosody("hello world", ProsodyConfig(emotion="angry"))
        assert result.startswith("HELLO")

    def test_angry_replaces_all_periods(self):
        result = apply_text_prosody("First. Second.", ProsodyConfig(emotion="angry"))
        assert "!" in result

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
        temp, top_k, rep_pen, sub_temp, sub_top_k = adjust_sampling_for_emotion(
            "happy", 0.9, 50)
        assert temp > 0.9

    def test_sad_decreases_temperature(self):
        temp, top_k, rep_pen, sub_temp, sub_top_k = adjust_sampling_for_emotion(
            "sad", 0.9, 50)
        assert temp < 0.9

    def test_angry_adjusts_both(self):
        temp, top_k, rep_pen, sub_temp, sub_top_k = adjust_sampling_for_emotion(
            "angry", 0.9, 50)
        assert temp > 0.9
        assert top_k < 50

    def test_neutral_unchanged(self):
        temp, top_k, rep_pen, sub_temp, sub_top_k = adjust_sampling_for_emotion(
            "neutral", 0.9, 50)
        assert temp == 0.9
        assert top_k == 50

    def test_none_unchanged(self):
        temp, top_k, rep_pen, sub_temp, sub_top_k = adjust_sampling_for_emotion(
            None, 0.9, 50)
        assert temp == 0.9
        assert top_k == 50

    def test_surprised_increases_temperature(self):
        temp, top_k, rep_pen, sub_temp, sub_top_k = adjust_sampling_for_emotion(
            "surprised", 0.9, 50)
        assert temp > 0.9

    def test_returns_five_values(self):
        result = adjust_sampling_for_emotion("happy", 0.9, 50)
        assert len(result) == 5

    def test_sad_lowers_rep_penalty(self):
        temp, top_k, rep_pen, sub_temp, sub_top_k = adjust_sampling_for_emotion(
            "sad", 0.9, 50)
        assert rep_pen == 1.0  # no penalty = flat/monotone

    def test_happy_raises_rep_penalty(self):
        temp, top_k, rep_pen, sub_temp, sub_top_k = adjust_sampling_for_emotion(
            "happy", 0.9, 50)
        assert rep_pen > 1.05  # higher than default


# ── Emotion schedule tests ────────────────────────────────────────────

class TestEmotionSchedule:
    def test_no_emotion_returns_base(self):
        t, r, k = emotion_schedule(None, 0, 0.9, 1.05, 50)
        assert t == 0.9
        assert r == 1.05
        assert k == 50

    def test_happy_varies_over_steps(self):
        vals = [emotion_schedule("happy", s, 0.9, 1.05, 50) for s in range(40)]
        temps = [v[0] for v in vals]
        # Should not all be the same (sinusoidal variation)
        assert len(set(round(t, 4) for t in temps)) > 1

    def test_sad_is_flat(self):
        vals = [emotion_schedule("sad", s, 0.9, 1.05, 50) for s in range(20)]
        temps = [v[0] for v in vals]
        # Sad should be constant (flat)
        assert len(set(round(t, 6) for t in temps)) == 1

    def test_surprised_decays(self):
        t0, _, _ = emotion_schedule("surprised", 0, 0.9, 1.05, 50)
        t50, _, _ = emotion_schedule("surprised", 50, 0.9, 1.05, 50)
        assert t0 > t50  # initial burst > later value

    def test_angry_has_peaks(self):
        t_peak, _, _ = emotion_schedule("angry", 2, 0.9, 1.05, 50)
        t_trough, _, _ = emotion_schedule("angry", 10, 0.9, 1.05, 50)
        assert t_peak > t_trough  # step 2 is in peak, step 10 is not


# ── Emotion bias tests ────────────────────────────────────────────────

class TestEmotionBias:
    def test_none_returns_none(self):
        assert get_emotion_bias(None) is None
        assert get_emotion_bias("neutral") is None

    def test_returns_correct_shape(self):
        bias = get_emotion_bias("happy")
        assert bias.shape == (1, 1, 1024)
        assert bias.dtype == np.float32

    def test_different_emotions_different_biases(self):
        happy = get_emotion_bias("happy")
        sad = get_emotion_bias("sad")
        assert not np.allclose(happy, sad)

    def test_same_emotion_is_deterministic(self):
        a = get_emotion_bias("happy")
        b = get_emotion_bias("happy")
        np.testing.assert_array_equal(a, b)

    def test_magnitude_is_small(self):
        bias = get_emotion_bias("happy", magnitude=0.01)
        assert np.max(np.abs(bias)) < 0.1


# ── Subtalker params tests ────────────────────────────────────────────

class TestSubtalkerParams:
    def test_none_returns_defaults(self):
        assert get_emotion_subtalker_params(None) == (0.9, 50)

    def test_happy_is_brighter(self):
        sub_temp, sub_top_k = get_emotion_subtalker_params("happy")
        assert sub_temp > 0.9

    def test_sad_is_muted(self):
        sub_temp, sub_top_k = get_emotion_subtalker_params("sad")
        assert sub_temp < 0.9
        assert sub_top_k < 50

    def test_angry_is_harsh(self):
        sub_temp, sub_top_k = get_emotion_subtalker_params("angry")
        assert sub_temp > 0.9
        assert sub_top_k < 50


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

    def test_pitch_is_noop(self):
        wav = self._make_wav(1.0)
        result = apply_waveform_prosody(wav, 24000, ProsodyConfig(pitch_semitones=4.0))
        # Pitch shifting is intentionally disabled (phase vocoder artifacts)
        np.testing.assert_array_equal(result, wav)

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
        config = ProsodyConfig(rate=1.5, volume=0.8)
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
