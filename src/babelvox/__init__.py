"""BabelVox: Real-time text-to-speech on Intel NPU/CPU via OpenVINO."""
from importlib.metadata import PackageNotFoundError, version

from babelvox.pipeline import BabelVox, download_models, mel_spectrogram_np
from babelvox.server import serve

try:
    __version__ = version("babelvox")
except PackageNotFoundError:
    __version__ = "0.0.0"

__all__ = ["BabelVox", "download_models", "mel_spectrogram_np", "serve"]
