"""Export Qwen3-TTS Speaker Encoder (ECAPA-TDNN) to OpenVINO IR format."""
import os

import numpy as np
import torch
import openvino as ov
from transformers import AutoConfig, AutoModel

from qwen_tts.core.models import Qwen3TTSConfig, Qwen3TTSForConditionalGeneration

MODEL_PATH = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
EXPORT_DIR = "openvino_export"
os.makedirs(EXPORT_DIR, exist_ok=True)

print("Loading TTS model to extract speaker encoder...")
AutoConfig.register("qwen3_tts", Qwen3TTSConfig)
AutoModel.register(Qwen3TTSConfig, Qwen3TTSForConditionalGeneration)

model = AutoModel.from_pretrained(
    MODEL_PATH,
    device_map="cpu",
    dtype=torch.float32,
    attn_implementation="eager",
)

speaker_enc = model.speaker_encoder
speaker_enc.eval()
print(f"Speaker encoder type: {type(speaker_enc).__name__}")

# Input: mel spectrogram (batch, time, mel_bins=128)
# The extract_speaker_embedding method uses:
#   mel_spectrogram(..., num_mels=128, ...).transpose(1, 2) -> (B, T, 128)
#   speaker_encoder(mels) -> (embedding_dim,)
# At 24kHz with hop_size=256: 1 second = ~93 mel frames
mel_frames = 200  # ~2 seconds
mel_bins = 128
dummy_mels = torch.randn(1, mel_frames, mel_bins, dtype=torch.float32)

print(f"\nInput shape: {dummy_mels.shape}")
print(f"  (batch=1, mel_frames={mel_frames}, mel_bins={mel_bins})")

print("Testing forward pass...")
with torch.no_grad():
    emb = speaker_enc(dummy_mels)
print(f"Output shape: {emb.shape}")
print(f"  Speaker embedding dim: {emb.shape[-1]}")

# Export via ONNX -> OpenVINO
print("\n--- Exporting via ONNX -> OpenVINO ---")
onnx_path = os.path.join(EXPORT_DIR, "speaker_encoder.onnx")

print("Running torch.onnx.export...")
torch.onnx.export(
    speaker_enc,
    (dummy_mels,),
    onnx_path,
    input_names=["mel_spectrogram"],
    output_names=["speaker_embedding"],
    dynamic_axes={"mel_spectrogram": {1: "mel_frames"}},
    opset_version=18,
)
print(f"ONNX saved to {onnx_path}")

print("Converting ONNX to OpenVINO IR...")
ov_model = ov.convert_model(onnx_path)
ir_path = os.path.join(EXPORT_DIR, "speaker_encoder.xml")
ov.save_model(ov_model, ir_path)
print(f"OpenVINO IR saved to {ir_path}")

# Validate
print("\nValidating OpenVINO model on CPU...")
core = ov.Core()
compiled = core.compile_model(ov_model, "CPU")
ov_result = compiled(dummy_mels.numpy())[0]
torch_result = emb.detach().numpy()
max_diff = np.abs(ov_result - torch_result).max()
print(f"Max diff (PyTorch vs OpenVINO CPU): {max_diff:.6f}")
if max_diff < 0.01:
    print("PASS - Speaker encoder exported successfully!")
else:
    print(f"WARNING - Large difference: {max_diff}")

# File sizes
xml_size = os.path.getsize(ir_path) / 1024 / 1024
bin_path = ir_path.replace(".xml", ".bin")
bin_size = os.path.getsize(bin_path) / 1024 / 1024
print(f"\nExport files:")
print(f"  {ir_path}: {xml_size:.1f} MB")
print(f"  {bin_path}: {bin_size:.1f} MB")
print(f"  Total: {xml_size + bin_size:.1f} MB")
