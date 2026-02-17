"""BabelVox: Real-time text-to-speech on Intel NPU/CPU via OpenVINO."""
from babelvox.pipeline import BabelVox
from babelvox.pipeline import download_models
from babelvox.pipeline import mel_spectrogram_np
from babelvox.server import serve

__version__ = "0.5.1"
__all__ = ["BabelVox", "download_models", "mel_spectrogram_np", "serve"]
