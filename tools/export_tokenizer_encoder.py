"""Export Qwen3-TTS Tokenizer Encoder (MimiModel-based) to OpenVINO IR format."""
import os

import numpy as np
import torch
from torch import nn
import openvino as ov

from qwen_tts.inference.qwen3_tts_tokenizer import Qwen3TTSTokenizer

TOKENIZER_PATH = "Qwen/Qwen3-TTS-Tokenizer-12Hz"
EXPORT_DIR = "openvino_export"
os.makedirs(EXPORT_DIR, exist_ok=True)

print("Loading tokenizer model...")
tokenizer = Qwen3TTSTokenizer.from_pretrained(
    TOKENIZER_PATH,
    device_map="cpu",
    dtype=torch.float32,
    attn_implementation="eager",
)

encoder = tokenizer.model.encoder
encoder.eval()

print(f"Encoder type: {type(encoder).__name__}")
print(f"Encoder config model_type: {encoder.config.model_type}")
print(f"Valid num quantizers: {tokenizer.model.encoder_valid_num_quantizers}")

# The encoder expects raw audio waveform
# Input: (batch, channels=1, samples)
# From the encode() method: input_values.unsqueeze(1)
sr = tokenizer.get_input_sample_rate()
print(f"Input sample rate: {sr}")

# ~2 seconds of audio
duration = 2.0
num_samples = int(sr * duration)
dummy_audio = torch.randn(1, 1, num_samples, dtype=torch.float32)

print(f"\nInput shape: {dummy_audio.shape}")
print(f"  (batch=1, channels=1, samples={num_samples})")

print("Testing encode on CPU...")
with torch.no_grad():
    enc_output = encoder.encode(input_values=dummy_audio, return_dict=True)
codes = enc_output.audio_codes
print(f"Output audio_codes shape: {codes.shape}")
print(f"  (batch, num_quantizers, code_len)")


def _patched_quantize(self, hidden_states):
    """Replace torch.cdist with manual L2 distance for ONNX compatibility.

    ||a - b||^2 = ||a||^2 + ||b||^2 - 2 * a @ b^T
    """
    hs = hidden_states.float()
    emb = self.embed.float()
    # hs: (N, D), emb: (K, D) -> dists: (N, K)
    dists = (
        hs.pow(2).sum(dim=-1, keepdim=True)
        + emb.pow(2).sum(dim=-1).unsqueeze(0)
        - 2.0 * hs @ emb.t()
    )
    return dists.argmin(dim=-1)


class EncoderForExport(nn.Module):
    """Wrapper that bypasses HuggingFace's vmap-based causal mask creation
    and replaces torch.cdist with manual L2 distance for ONNX tracing.

    Manually runs: conv encoder -> transformer layers -> downsample -> quantizer
    """

    def __init__(self, mimi_model, valid_num_quantizers):
        super().__init__()
        # Grab each submodule from the MimiModel
        self.conv_encoder = mimi_model.encoder
        self.encoder_transformer = mimi_model.encoder_transformer
        self.downsample = mimi_model.downsample
        self.quantizer = mimi_model.quantizer
        self.valid_num_quantizers = valid_num_quantizers
        self.num_quantizers = mimi_model.config.num_quantizers

        # Force eager attention on all transformer layers
        self.encoder_transformer.config._attn_implementation = "eager"
        self.encoder_transformer._attn_implementation = "eager"
        for layer in self.encoder_transformer.layers:
            layer.self_attn.config._attn_implementation = "eager"

        # Monkey-patch all codebook quantize methods to avoid torch.cdist
        import types
        for rvq in [self.quantizer.semantic_residual_vector_quantizer,
                     self.quantizer.acoustic_residual_vector_quantizer]:
            for vq_layer in rvq.layers:
                vq_layer.codebook.quantize = types.MethodType(
                    _patched_quantize, vq_layer.codebook
                )

    def _make_causal_mask(self, seq_len: int, dtype: torch.dtype, device: torch.device):
        """Simple lower-triangular causal mask. No vmap needed."""
        mask = torch.full((seq_len, seq_len), torch.finfo(dtype).min, dtype=dtype, device=device)
        mask = torch.triu(mask, diagonal=1)
        return mask.unsqueeze(0).unsqueeze(0)

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        # audio: (B, 1, T)

        # 1. Conv encoder: (B, 1, T) -> (B, hidden_size, frames)
        embeddings = self.conv_encoder(audio)

        # 2. Transformer: need (B, frames, hidden_size)
        hidden = embeddings.transpose(1, 2)
        seq_len = hidden.shape[1]

        # Pre-compute causal mask (bypasses create_causal_mask / vmap)
        causal_mask = self._make_causal_mask(seq_len, hidden.dtype, hidden.device)
        position_ids = torch.arange(seq_len, device=hidden.device).unsqueeze(0)

        # Run each transformer layer manually
        for layer in self.encoder_transformer.layers:
            layer_out = layer(
                hidden,
                attention_mask=causal_mask,
                position_ids=position_ids,
            )
            hidden = layer_out[0]

        # Back to (B, hidden_size, frames)
        embeddings = hidden.transpose(1, 2)

        # 3. Downsample
        if self.downsample is not None:
            embeddings = self.downsample(embeddings)

        # 4. Quantize -> codes (num_quantizers, B, T) -> transpose to (B, num_quantizers, T)
        codes = self.quantizer.encode(embeddings, self.num_quantizers)
        codes = codes.transpose(0, 1)

        # Return only valid quantizers
        return codes[:, :self.valid_num_quantizers]


export_encoder = EncoderForExport(encoder, tokenizer.model.encoder_valid_num_quantizers)
export_encoder.eval()

print("\nTesting wrapper...")
with torch.no_grad():
    wrapped_codes = export_encoder(dummy_audio)
print(f"Wrapper output shape: {wrapped_codes.shape}")

# Verify wrapper matches original encoder output
original_valid = codes[:, :tokenizer.model.encoder_valid_num_quantizers]
code_diff = (wrapped_codes - original_valid).abs().max().item()
print(f"Wrapper vs original max diff: {code_diff}")
assert code_diff == 0, f"Wrapper output diverges! diff={code_diff}"
print("MATCH - wrapper produces same output as original")

# Export via ONNX -> OpenVINO
print("\n--- Exporting via ONNX -> OpenVINO ---")
onnx_path = os.path.join(EXPORT_DIR, "tokenizer_encoder.onnx")

print("Running torch.onnx.export...")
torch.onnx.export(
    export_encoder,
    (dummy_audio,),
    onnx_path,
    input_names=["audio"],
    output_names=["audio_codes"],
    dynamic_axes={"audio": {2: "audio_len"}, "audio_codes": {2: "code_len"}},
    opset_version=18,
    dynamo=False,
)
print(f"ONNX saved to {onnx_path}")

print("Converting ONNX to OpenVINO IR...")
ov_model = ov.convert_model(onnx_path)
ir_path = os.path.join(EXPORT_DIR, "tokenizer_encoder.xml")
ov.save_model(ov_model, ir_path)
print(f"OpenVINO IR saved to {ir_path}")

# Validate
print("\nValidating OpenVINO model on CPU...")
core = ov.Core()
compiled = core.compile_model(ov_model, "CPU")
ov_result = compiled(dummy_audio.numpy())[0]
torch_result = wrapped_codes.detach().numpy()
max_diff = np.abs(ov_result.astype(np.int64) - torch_result.astype(np.int64)).max()
print(f"Max diff (PyTorch vs OpenVINO CPU): {max_diff}")
if max_diff == 0:
    print("PASS - Tokenizer encoder exported successfully!")
else:
    print(f"WARNING - Codes differ by {max_diff}")

# File sizes
xml_size = os.path.getsize(ir_path) / 1024 / 1024
bin_path = ir_path.replace(".xml", ".bin")
bin_size = os.path.getsize(bin_path) / 1024 / 1024
print(f"\nExport files:")
print(f"  {ir_path}: {xml_size:.1f} MB")
print(f"  {bin_path}: {bin_size:.1f} MB")
print(f"  Total: {xml_size + bin_size:.1f} MB")
