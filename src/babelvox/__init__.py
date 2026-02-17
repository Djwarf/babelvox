"""BabelVox: Real-time text-to-speech on Intel NPU/CPU via OpenVINO."""
from babelvox.pipeline import BabelVox
from babelvox.pipeline import download_models
from babelvox.pipeline import mel_spectrogram_np

__all__ = ["BabelVox", "download_models", "mel_spectrogram_np"]
