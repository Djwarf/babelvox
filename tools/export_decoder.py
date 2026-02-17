"""Export Qwen3-TTS Tokenizer Decoder to OpenVINO IR format."""
import os
import math

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

decoder = tokenizer.model.decoder
decoder.eval()

config = decoder.config
print(f"Decoder config:")
print(f"  num_quantizers: {config.num_quantizers}")
print(f"  codebook_size: {config.codebook_size}")
print(f"  decoder_dim: {config.decoder_dim}")
print(f"  latent_dim: {config.latent_dim}")
print(f"  upsample_rates: {config.upsample_rates}")
print(f"  total_upsample: {decoder.total_upsample}")


class DecoderForExport(nn.Module):
    """Wrapper that bypasses HuggingFace's vmap-based causal mask creation.

    Instead of letting the pre_transformer call create_causal_mask (which uses
    vmap and can't be traced), we pre-compute a simple causal mask and pass it
    as a dict — which the transformer accepts directly, skipping mask creation.
    """

    def __init__(self, original_decoder):
        super().__init__()
        self.quantizer = original_decoder.quantizer
        self.pre_conv = original_decoder.pre_conv
        self.pre_transformer = original_decoder.pre_transformer
        self.upsample = original_decoder.upsample
        self.decoder = original_decoder.decoder

        # Force eager attention
        self.pre_transformer.config._attn_implementation = "eager"
        for layer in self.pre_transformer.layers:
            layer.self_attn.config._attn_implementation = "eager"

    def _make_causal_mask(self, seq_len: int, dtype: torch.dtype, device: torch.device):
        """Simple lower-triangular causal mask. No vmap needed."""
        mask = torch.full((seq_len, seq_len), torch.finfo(dtype).min, dtype=dtype, device=device)
        mask = torch.triu(mask, diagonal=1)
        # Shape: (1, 1, seq_len, seq_len) for broadcast over batch and heads
        return mask.unsqueeze(0).unsqueeze(0)

    def forward(self, codes: torch.Tensor) -> torch.Tensor:
        # 1. VQ decode: codes (B, num_q, T) -> continuous embeddings
        hidden = self.quantizer.decode(codes)

        # 2. Pre-conv
        hidden = self.pre_conv(hidden).transpose(1, 2)

        # 3. Pre-transformer with pre-computed causal mask (bypasses create_causal_mask)
        seq_len = hidden.shape[1]
        causal_mask = self._make_causal_mask(seq_len, hidden.dtype, hidden.device)

        # Build the mask dict that the transformer expects
        # All layer types get the same simple causal mask
        mask_dict = {
            "full_attention": causal_mask,
            "sliding_attention": causal_mask,
        }

        hidden = self.pre_transformer(
            inputs_embeds=hidden,
            attention_mask=mask_dict,
        ).last_hidden_state

        hidden = hidden.permute(0, 2, 1)

        # 4. Upsample blocks
        for blocks in self.upsample:
            for block in blocks:
                hidden = block(hidden)

        # 5. Decoder ConvNet blocks
        wav = hidden
        for block in self.decoder:
            wav = block(wav)

        return wav.clamp(min=-1, max=1)


# Create wrapper and test
print("\nCreating export wrapper...")
export_decoder = DecoderForExport(decoder)
export_decoder.eval()

num_q = config.num_quantizers
seq_len = 50
dummy_codes = torch.randint(0, config.codebook_size, (1, num_q, seq_len), dtype=torch.long)

print(f"Input shape: {dummy_codes.shape}")

print("Testing wrapper forward pass...")
with torch.no_grad():
    wav_wrapper = export_decoder(dummy_codes)
print(f"Wrapper output shape: {wav_wrapper.shape}")

# Verify wrapper matches original
print("Verifying wrapper matches original decoder...")
with torch.no_grad():
    wav_original = decoder(dummy_codes)
max_diff = (wav_wrapper - wav_original).abs().max().item()
print(f"Max diff (wrapper vs original): {max_diff:.8f}")
assert max_diff < 1e-4, f"Wrapper output diverges from original! diff={max_diff}"
print("MATCH - wrapper produces same output")

# --- Export via ONNX -> OpenVINO ---
print("\n--- Exporting via ONNX -> OpenVINO ---")
onnx_path = os.path.join(EXPORT_DIR, "tokenizer_decoder.onnx")

print("Running torch.onnx.export...")
torch.onnx.export(
    export_decoder,
    (dummy_codes,),
    onnx_path,
    input_names=["codes"],
    output_names=["audio"],
    dynamic_axes={"codes": {2: "seq_len"}, "audio": {2: "audio_len"}},
    opset_version=17,
)
print(f"ONNX saved to {onnx_path}")

print("Converting ONNX to OpenVINO IR...")
ov_model = ov.convert_model(onnx_path)
ir_path = os.path.join(EXPORT_DIR, "tokenizer_decoder.xml")
ov.save_model(ov_model, ir_path)
print(f"OpenVINO IR saved to {ir_path}")

# Validate OpenVINO output
print("\nValidating OpenVINO model on CPU...")
core = ov.Core()
compiled = core.compile_model(ov_model, "CPU")
ov_result = compiled(dummy_codes.numpy())[0]
torch_result = wav_wrapper.detach().numpy()
max_diff_ov = np.abs(ov_result - torch_result).max()
print(f"Max diff (PyTorch vs OpenVINO CPU): {max_diff_ov:.6f}")
if max_diff_ov < 0.01:
    print("PASS - Tokenizer decoder exported successfully!")
else:
    print(f"WARNING - Large difference: {max_diff_ov}")

# Report file sizes
xml_size = os.path.getsize(ir_path) / 1024 / 1024
bin_path = ir_path.replace(".xml", ".bin")
bin_size = os.path.getsize(bin_path) / 1024 / 1024
print(f"\nExport files:")
print(f"  {ir_path}: {xml_size:.1f} MB")
print(f"  {bin_path}: {bin_size:.1f} MB")
print(f"  Total: {xml_size + bin_size:.1f} MB")
