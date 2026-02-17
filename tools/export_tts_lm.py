"""Export Qwen3-TTS Language Model (Talker + Code Predictor) to OpenVINO IR.

Phase 1: Export without KV cache (validation only).
The talker (28-layer transformer) and code predictor (5-layer transformer)
are exported as separate models. Embedding lookups and lm_head selection
stay in the Python orchestration layer.
"""
import os
import sys

import numpy as np
import torch
from torch import nn
import openvino as ov
from transformers import AutoConfig, AutoModel

from qwen_tts.core.models import Qwen3TTSConfig, Qwen3TTSForConditionalGeneration
from qwen_tts.core.models.modeling_qwen3_tts import (
    apply_multimodal_rotary_pos_emb,
    apply_rotary_pos_emb,
    rotate_half,
    eager_attention_forward,
)

MODEL_PATH = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
EXPORT_DIR = "openvino_export"
os.makedirs(EXPORT_DIR, exist_ok=True)

print("Loading TTS model...")
AutoConfig.register("qwen3_tts", Qwen3TTSConfig)
AutoModel.register(Qwen3TTSConfig, Qwen3TTSForConditionalGeneration)

model = AutoModel.from_pretrained(
    MODEL_PATH,
    device_map="cpu",
    dtype=torch.float32,
    attn_implementation="eager",
)

talker = model.talker
talker.eval()

talker_config = talker.config
cp_config = talker_config.code_predictor_config

print(f"Talker: {talker_config.num_hidden_layers} layers, "
      f"hidden={talker_config.hidden_size}, "
      f"heads={talker_config.num_attention_heads}Q/{talker_config.num_key_value_heads}KV, "
      f"vocab={talker_config.vocab_size}")
print(f"Code Predictor: {cp_config.num_hidden_layers} layers, "
      f"hidden={cp_config.hidden_size}, "
      f"heads={cp_config.num_attention_heads}Q/{cp_config.num_key_value_heads}KV, "
      f"vocab={cp_config.vocab_size}, "
      f"num_code_groups={cp_config.num_code_groups}")


# ============================================================
# Talker Wrapper
# ============================================================
class TalkerForExport(nn.Module):
    """Exports the talker's base transformer + codec_head.

    Bypasses create_causal_mask (vmap) by pre-computing a simple causal mask.
    No KV cache in this version (phase 1 - validation only).

    Input:  inputs_embeds (1, T, hidden), position_ids (3, 1, T)
    Output: logits (1, T, vocab), hidden_states (1, T, hidden)
    """

    def __init__(self, talker_model):
        super().__init__()
        base = talker_model.model
        self.layers = base.layers
        self.norm = base.norm
        self.rotary_emb = base.rotary_emb
        self.codec_head = talker_model.codec_head
        self.config = base.config

        # Force eager attention on all layers
        for layer in self.layers:
            layer.self_attn.config._attn_implementation = "eager"

    def _make_causal_mask(self, seq_len, dtype, device):
        mask = torch.full((seq_len, seq_len), torch.finfo(dtype).min, dtype=dtype, device=device)
        mask = torch.triu(mask, diagonal=1)
        return mask.unsqueeze(0).unsqueeze(0)

    def forward(self, inputs_embeds, position_ids):
        # inputs_embeds: (B, T, hidden_size)
        # position_ids: (3, B, T)

        seq_len = inputs_embeds.shape[1]
        causal_mask = self._make_causal_mask(seq_len, inputs_embeds.dtype, inputs_embeds.device)

        # Compute position embeddings (3D RoPE)
        position_embeddings = self.rotary_emb(inputs_embeds, position_ids)

        hidden_states = inputs_embeds

        for layer in self.layers:
            layer_out = layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids[0],  # text_position_ids
                past_key_values=None,
                output_attentions=False,
                use_cache=False,
                cache_position=None,
                position_embeddings=position_embeddings,
            )
            hidden_states = layer_out[0]

        hidden_states = self.norm(hidden_states)
        logits = self.codec_head(hidden_states)

        return logits, hidden_states


# ============================================================
# Code Predictor Wrapper
# ============================================================
class CodePredictorForExport(nn.Module):
    """Exports the code predictor's base transformer.

    Uses dict-based mask bypass (the model already supports this).
    No KV cache in this version (phase 1 - validation only).
    lm_head selection is done externally in Python.

    Input:  inputs_embeds (1, T, hidden) - already projected through small_to_mtp_projection
    Output: hidden_states (1, T, hidden)
    """

    def __init__(self, code_predictor):
        super().__init__()
        base = code_predictor.model
        self.layers = base.layers
        self.norm = base.norm
        self.rotary_emb = base.rotary_emb
        self.config = base.config
        self.layer_types = base.config.layer_types

        # Force eager attention on all layers
        for layer in self.layers:
            layer.self_attn.config._attn_implementation = "eager"

    def _make_causal_mask(self, seq_len, dtype, device):
        mask = torch.full((seq_len, seq_len), torch.finfo(dtype).min, dtype=dtype, device=device)
        mask = torch.triu(mask, diagonal=1)
        return mask.unsqueeze(0).unsqueeze(0)

    def forward(self, inputs_embeds):
        # inputs_embeds: (B, T, hidden_size) - already projected
        seq_len = inputs_embeds.shape[1]

        causal_mask = self._make_causal_mask(seq_len, inputs_embeds.dtype, inputs_embeds.device)
        position_ids = torch.arange(seq_len, device=inputs_embeds.device).unsqueeze(0)

        # Compute 1D RoPE
        position_embeddings = self.rotary_emb(inputs_embeds, position_ids)

        hidden_states = inputs_embeds

        for layer in self.layers:
            layer_out = layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_values=None,
                output_attentions=False,
                use_cache=False,
                cache_position=None,
                position_embeddings=position_embeddings,
            )
            hidden_states = layer_out[0]

        hidden_states = self.norm(hidden_states)
        return hidden_states


# ============================================================
# Create wrappers and test
# ============================================================
print("\nCreating export wrappers...")
talker_export = TalkerForExport(talker)
talker_export.eval()

cp_export = CodePredictorForExport(talker.code_predictor)
cp_export.eval()

# Test talker
seq_len = 20
dummy_embeds = torch.randn(1, seq_len, talker_config.hidden_size, dtype=torch.float32)
dummy_pos_ids = torch.arange(seq_len).unsqueeze(0).unsqueeze(0).expand(3, 1, -1)

print(f"\nTalker test input: embeds={dummy_embeds.shape}, pos_ids={dummy_pos_ids.shape}")
with torch.no_grad():
    talker_logits, talker_hidden = talker_export(dummy_embeds, dummy_pos_ids)
print(f"Talker output: logits={talker_logits.shape}, hidden={talker_hidden.shape}")

# Verify talker matches original
print("Verifying talker wrapper matches original...")
with torch.no_grad():
    cache_position = torch.arange(seq_len)
    orig_out = talker.model(
        input_ids=None,
        attention_mask=None,
        position_ids=dummy_pos_ids,
        past_key_values=None,
        inputs_embeds=dummy_embeds,
        use_cache=False,
        output_attentions=False,
        output_hidden_states=False,
        cache_position=cache_position,
    )
    orig_hidden = orig_out.last_hidden_state
    orig_logits = talker.codec_head(orig_hidden)

talker_diff = (talker_hidden - orig_hidden).abs().max().item()
logits_diff = (talker_logits - orig_logits).abs().max().item()
print(f"  Hidden diff: {talker_diff:.8f}")
print(f"  Logits diff: {logits_diff:.8f}")
if talker_diff < 1e-4 and logits_diff < 1e-4:
    print("  MATCH - Talker wrapper OK")
else:
    print(f"  WARNING - Talker output diverges!")
    sys.exit(1)

# Test code predictor
cp_seq_len = 10
dummy_cp_embeds = torch.randn(1, cp_seq_len, cp_config.hidden_size, dtype=torch.float32)
# Need to project through small_to_mtp_projection first for the original
dummy_cp_embeds_proj = talker.code_predictor.small_to_mtp_projection(dummy_cp_embeds)

print(f"\nCode Predictor test input: embeds={dummy_cp_embeds_proj.shape}")
with torch.no_grad():
    cp_hidden = cp_export(dummy_cp_embeds_proj)
print(f"Code Predictor output: hidden={cp_hidden.shape}")

# Verify code predictor matches original
print("Verifying code predictor wrapper matches original...")
with torch.no_grad():
    cp_cache_position = torch.arange(cp_seq_len)
    cp_orig_out = talker.code_predictor.model(
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=dummy_cp_embeds_proj,
        use_cache=False,
        output_attentions=False,
        output_hidden_states=False,
        cache_position=cp_cache_position,
    )
    cp_orig_hidden = cp_orig_out.last_hidden_state

cp_diff = (cp_hidden - cp_orig_hidden).abs().max().item()
print(f"  Hidden diff: {cp_diff:.8f}")
if cp_diff < 1e-4:
    print("  MATCH - Code Predictor wrapper OK")
else:
    print(f"  WARNING - Code Predictor output diverges!")
    sys.exit(1)


# ============================================================
# Export Talker
# ============================================================
print("\n--- Exporting Talker ---")
talker_onnx = os.path.join(EXPORT_DIR, "talker.onnx")

print("Running torch.onnx.export for talker...")
torch.onnx.export(
    talker_export,
    (dummy_embeds, dummy_pos_ids),
    talker_onnx,
    input_names=["inputs_embeds", "position_ids"],
    output_names=["logits", "hidden_states"],
    dynamic_axes={
        "inputs_embeds": {1: "seq_len"},
        "position_ids": {2: "seq_len"},
        "logits": {1: "seq_len"},
        "hidden_states": {1: "seq_len"},
    },
    opset_version=18,
    dynamo=False,
)
print(f"ONNX saved to {talker_onnx}")

print("Converting talker ONNX to OpenVINO IR...")
talker_ov = ov.convert_model(talker_onnx)
talker_ir = os.path.join(EXPORT_DIR, "talker.xml")
ov.save_model(talker_ov, talker_ir)
print(f"OpenVINO IR saved to {talker_ir}")

# Validate talker
print("Validating talker OpenVINO model on CPU...")
core = ov.Core()
talker_compiled = core.compile_model(talker_ov, "CPU")
ov_result = talker_compiled({
    "inputs_embeds": dummy_embeds.numpy(),
    "position_ids": dummy_pos_ids.numpy(),
})
ov_logits = ov_result["logits"]
ov_hidden = ov_result["hidden_states"]
torch_logits = talker_logits.detach().numpy()
torch_hidden = talker_hidden.detach().numpy()
logits_diff_ov = np.abs(ov_logits - torch_logits).max()
hidden_diff_ov = np.abs(ov_hidden - torch_hidden).max()
print(f"  Logits diff (PyTorch vs OpenVINO): {logits_diff_ov:.6f}")
print(f"  Hidden diff (PyTorch vs OpenVINO): {hidden_diff_ov:.6f}")
if logits_diff_ov < 0.01 and hidden_diff_ov < 0.01:
    print("  PASS - Talker exported successfully!")
else:
    print(f"  WARNING - Large difference")


# ============================================================
# Export Code Predictor
# ============================================================
print("\n--- Exporting Code Predictor ---")
cp_onnx = os.path.join(EXPORT_DIR, "code_predictor.onnx")

print("Running torch.onnx.export for code predictor...")
torch.onnx.export(
    cp_export,
    (dummy_cp_embeds_proj,),
    cp_onnx,
    input_names=["inputs_embeds"],
    output_names=["hidden_states"],
    dynamic_axes={
        "inputs_embeds": {1: "seq_len"},
        "hidden_states": {1: "seq_len"},
    },
    opset_version=18,
    dynamo=False,
)
print(f"ONNX saved to {cp_onnx}")

print("Converting code predictor ONNX to OpenVINO IR...")
cp_ov = ov.convert_model(cp_onnx)
cp_ir = os.path.join(EXPORT_DIR, "code_predictor.xml")
ov.save_model(cp_ov, cp_ir)
print(f"OpenVINO IR saved to {cp_ir}")

# Validate code predictor
print("Validating code predictor OpenVINO model on CPU...")
cp_compiled = core.compile_model(cp_ov, "CPU")
cp_ov_result = cp_compiled({"inputs_embeds": dummy_cp_embeds_proj.detach().numpy()})
cp_ov_hidden = cp_ov_result["hidden_states"]
cp_torch_hidden = cp_hidden.detach().numpy()
cp_diff_ov = np.abs(cp_ov_hidden - cp_torch_hidden).max()
print(f"  Hidden diff (PyTorch vs OpenVINO): {cp_diff_ov:.6f}")
if cp_diff_ov < 0.01:
    print("  PASS - Code Predictor exported successfully!")
else:
    print(f"  WARNING - Large difference")


# ============================================================
# Report file sizes
# ============================================================
print("\n--- Export Summary ---")
for name, ir_path in [("Talker", talker_ir), ("Code Predictor", cp_ir)]:
    xml_size = os.path.getsize(ir_path) / 1024 / 1024
    bin_path = ir_path.replace(".xml", ".bin")
    bin_size = os.path.getsize(bin_path) / 1024 / 1024
    print(f"  {name}: {xml_size:.1f} MB (xml) + {bin_size:.1f} MB (bin) = {xml_size + bin_size:.1f} MB")

total = sum(
    os.path.getsize(os.path.join(EXPORT_DIR, f)) / 1024 / 1024
    for f in os.listdir(EXPORT_DIR) if f.endswith(('.xml', '.bin'))
)
print(f"\n  Total all models in {EXPORT_DIR}: {total:.1f} MB")
