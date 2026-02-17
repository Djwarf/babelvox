"""BabelVox: Real-time text-to-speech on Intel NPU/CPU via OpenVINO."""
from babelvox.pipeline import BabelVox
from babelvox.pipeline import mel_spectrogram_np

__all__ = ["BabelVox", "mel_spectrogram_np"]
