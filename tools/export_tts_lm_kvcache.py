"""Export Qwen3-TTS Talker with KV Cache to OpenVINO IR.

Phase 2: Exports two models:
  - talker_prefill: (1, T, 1024) -> logits + hidden + KV cache
  - talker_decode:  (1, 1, 1024) + KV cache -> logits + hidden + updated KV cache

Each decode step processes 1 new token, reusing cached key/values.
Reduces per-step compute from O(seq_len^2) to O(seq_len).

KV cache stored as stacked tensors: (num_layers, 1, num_kv_heads, kv_len, head_dim)
For NPU: decode model uses scatter to write new K/V at a fixed position in
pre-allocated buffers, keeping all tensor shapes constant.
"""
import os
import sys
import time

import numpy as np
import torch
from torch import nn
import openvino as ov
from transformers import AutoConfig, AutoModel

from qwen_tts.core.models import Qwen3TTSConfig, Qwen3TTSForConditionalGeneration
from qwen_tts.core.models.modeling_qwen3_tts import (
    apply_multimodal_rotary_pos_emb,
)

MODEL_PATH = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
EXPORT_DIR = "openvino_export"
os.makedirs(EXPORT_DIR, exist_ok=True)

# ============================================================
# Load model
# ============================================================
print("Loading TTS model...")
AutoConfig.register("qwen3_tts", Qwen3TTSConfig)
AutoModel.register(Qwen3TTSConfig, Qwen3TTSForConditionalGeneration)

model = AutoModel.from_pretrained(
    MODEL_PATH, device_map="cpu", dtype=torch.float32,
    attn_implementation="eager",
)
talker = model.talker
talker.eval()

cfg = talker.config
NUM_LAYERS = cfg.num_hidden_layers
NUM_HEADS = cfg.num_attention_heads
NUM_KV_HEADS = cfg.num_key_value_heads
HEAD_DIM = cfg.head_dim
HIDDEN = cfg.hidden_size
NUM_KV_GROUPS = NUM_HEADS // NUM_KV_HEADS

attn0 = talker.model.layers[0].self_attn
MROPE_SECTION = attn0.rope_scaling["mrope_section"]
MROPE_INTERLEAVED = attn0.rope_scaling.get("interleaved", False)

print(f"Talker: {NUM_LAYERS}L, hidden={HIDDEN}, heads={NUM_HEADS}Q/{NUM_KV_HEADS}KV, "
      f"head_dim={HEAD_DIM}, kv_groups={NUM_KV_GROUPS}")
print(f"RoPE: section={MROPE_SECTION}, interleaved={MROPE_INTERLEAVED}")


# ============================================================
# Helpers
# ============================================================
def repeat_kv(hidden_states, n_rep):
    """Expand KV heads to match Q heads for GQA."""
    if n_rep == 1:
        return hidden_states
    bsz, num_kv_heads, slen, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(
        bsz, num_kv_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(bsz, num_kv_heads * n_rep, slen, head_dim)


def manual_attention(layer, hidden_states, cos, sin, causal_mask,
                     key_states_full, value_states_full):
    """Run one transformer layer with explicit K/V (no Cache object).

    Args:
        layer: Qwen3TTSTalkerDecoderLayer
        hidden_states: (B, T, hidden)
        cos, sin: position embeddings
        causal_mask: (1, 1, T_q, T_kv) or None
        key_states_full: (B, num_kv_heads, T_kv, head_dim) - full K for attention
        value_states_full: (B, num_kv_heads, T_kv, head_dim) - full V for attention

    Returns:
        hidden_states: (B, T, hidden) - after attention + MLP
    """
    bsz, q_len, _ = hidden_states.size()

    # Pre-norm
    residual = hidden_states
    normed = layer.input_layernorm(hidden_states)

    # Q projection (only Q is from current input; K, V are from cache)
    q = layer.self_attn.q_proj(normed)
    q = layer.self_attn.q_norm(
        q.view(bsz, q_len, NUM_HEADS, HEAD_DIM)
    ).transpose(1, 2)  # (B, 16, T_q, 128)

    # Apply RoPE to Q only (K already has RoPE from when it was cached)
    # We need a dummy K for the function signature, but we won't use it
    q_rot, _ = apply_multimodal_rotary_pos_emb(
        q, q[:, :NUM_KV_HEADS],  # dummy K (won't be used)
        cos, sin, MROPE_SECTION, MROPE_INTERLEAVED
    )

    # GQA: expand KV heads
    k_exp = repeat_kv(key_states_full, NUM_KV_GROUPS)    # (B, 16, T_kv, 128)
    v_exp = repeat_kv(value_states_full, NUM_KV_GROUPS)  # (B, 16, T_kv, 128)

    # Scaled dot-product attention
    scaling = HEAD_DIM ** -0.5
    attn_w = torch.matmul(q_rot, k_exp.transpose(2, 3)) * scaling
    if causal_mask is not None:
        attn_w = attn_w + causal_mask
    attn_w = torch.softmax(attn_w, dim=-1, dtype=torch.float32).to(q.dtype)
    attn_out = torch.matmul(attn_w, v_exp)

    # Output projection
    attn_out = attn_out.transpose(1, 2).contiguous().view(bsz, q_len, -1)
    attn_out = layer.self_attn.o_proj(attn_out)

    hidden_states = residual + attn_out

    # MLP
    residual = hidden_states
    hidden_states = residual + layer.mlp(layer.post_attention_layernorm(hidden_states))

    return hidden_states


# ============================================================
# Prefill Wrapper
# ============================================================
class TalkerPrefillForExport(nn.Module):
    """Process prefill tokens and return KV cache.

    Input:  inputs_embeds (1, T, 1024), position_ids (3, 1, T)
    Output: logits (1, T, 3072), hidden_states (1, T, 1024),
            present_keys (28, 1, 8, T, 128), present_values (28, 1, 8, T, 128)
    """

    def __init__(self, talker_model):
        super().__init__()
        base = talker_model.model
        self.layers = base.layers
        self.norm = base.norm
        self.rotary_emb = base.rotary_emb
        self.codec_head = talker_model.codec_head

        for layer in self.layers:
            layer.self_attn.config._attn_implementation = "eager"

    def _make_causal_mask(self, seq_len, dtype, device):
        mask = torch.full((seq_len, seq_len), torch.finfo(dtype).min,
                          dtype=dtype, device=device)
        mask = torch.triu(mask, diagonal=1)
        return mask.unsqueeze(0).unsqueeze(0)

    def forward(self, inputs_embeds, position_ids):
        seq_len = inputs_embeds.shape[1]
        causal_mask = self._make_causal_mask(seq_len, inputs_embeds.dtype,
                                             inputs_embeds.device)
        cos, sin = self.rotary_emb(inputs_embeds, position_ids)

        hidden_states = inputs_embeds
        all_keys = []
        all_values = []

        for layer in self.layers:
            residual = hidden_states
            normed = layer.input_layernorm(hidden_states)

            bsz, q_len, _ = normed.size()

            # QKV projections
            q = layer.self_attn.q_proj(normed)
            k = layer.self_attn.k_proj(normed)
            v = layer.self_attn.v_proj(normed)

            # Reshape + norms
            q = layer.self_attn.q_norm(
                q.view(bsz, q_len, NUM_HEADS, HEAD_DIM)
            ).transpose(1, 2)   # (1, 16, T, 128)
            k = layer.self_attn.k_norm(
                k.view(bsz, q_len, NUM_KV_HEADS, HEAD_DIM)
            ).transpose(1, 2)   # (1, 8, T, 128)
            v = v.view(bsz, q_len, NUM_KV_HEADS, HEAD_DIM).transpose(1, 2)

            # Apply multimodal RoPE
            q, k = apply_multimodal_rotary_pos_emb(
                q, k, cos, sin, MROPE_SECTION, MROPE_INTERLEAVED)

            # Save post-RoPE KV for cache
            all_keys.append(k)
            all_values.append(v)

            # GQA expand + attention
            k_exp = repeat_kv(k, NUM_KV_GROUPS)
            v_exp = repeat_kv(v, NUM_KV_GROUPS)

            scaling = HEAD_DIM ** -0.5
            attn_w = torch.matmul(q, k_exp.transpose(2, 3)) * scaling
            attn_w = attn_w + causal_mask
            attn_w = torch.softmax(attn_w, dim=-1, dtype=torch.float32).to(q.dtype)
            attn_out = torch.matmul(attn_w, v_exp)

            attn_out = attn_out.transpose(1, 2).contiguous().view(bsz, q_len, -1)
            attn_out = layer.self_attn.o_proj(attn_out)

            hidden_states = residual + attn_out

            # MLP
            residual = hidden_states
            hidden_states = residual + layer.mlp(
                layer.post_attention_layernorm(hidden_states))

        hidden_states = self.norm(hidden_states)
        logits = self.codec_head(hidden_states)

        present_keys = torch.stack(all_keys)    # (28, 1, 8, T, 128)
        present_values = torch.stack(all_values)

        return logits, hidden_states, present_keys, present_values


# ============================================================
# Decode Wrapper (with KV cache scatter)
# ============================================================
class TalkerDecodeForExport(nn.Module):
    """Process 1 token with pre-allocated KV cache.

    Input:  inputs_embeds (1, 1, 1024), position_ids (3, 1, 1),
            cache_position (1,), attention_mask (1, 1, 1, max_kv_len),
            past_keys (28, 1, 8, max_kv_len, 128),
            past_values (28, 1, 8, max_kv_len, 128)
    Output: logits (1, 1, 3072), hidden_states (1, 1, 1024),
            present_keys (28, 1, 8, max_kv_len, 128),
            present_values (28, 1, 8, max_kv_len, 128)
    """

    def __init__(self, talker_model):
        super().__init__()
        base = talker_model.model
        self.layers = base.layers
        self.norm = base.norm
        self.rotary_emb = base.rotary_emb
        self.codec_head = talker_model.codec_head

        for layer in self.layers:
            layer.self_attn.config._attn_implementation = "eager"

    def forward(self, inputs_embeds, position_ids, cache_position,
                attention_mask, past_keys, past_values):
        cos, sin = self.rotary_emb(inputs_embeds, position_ids)

        hidden_states = inputs_embeds
        present_keys_list = []
        present_values_list = []

        # Scatter index: write at cache_position in dim=2 (kv_len)
        idx = cache_position.view(1, 1, 1, 1).expand(1, NUM_KV_HEADS, 1, HEAD_DIM)

        for i, layer in enumerate(self.layers):
            layer_past_k = past_keys[i]   # (1, 8, max_kv_len, 128)
            layer_past_v = past_values[i]  # same

            residual = hidden_states
            normed = layer.input_layernorm(hidden_states)

            # QKV projections for single new token
            q = layer.self_attn.q_proj(normed)
            k = layer.self_attn.k_proj(normed)
            v = layer.self_attn.v_proj(normed)

            q = layer.self_attn.q_norm(
                q.view(1, 1, NUM_HEADS, HEAD_DIM)
            ).transpose(1, 2)   # (1, 16, 1, 128)
            k = layer.self_attn.k_norm(
                k.view(1, 1, NUM_KV_HEADS, HEAD_DIM)
            ).transpose(1, 2)   # (1, 8, 1, 128)
            v = v.view(1, 1, NUM_KV_HEADS, HEAD_DIM).transpose(1, 2)

            # Apply RoPE at current position
            q, k = apply_multimodal_rotary_pos_emb(
                q, k, cos, sin, MROPE_SECTION, MROPE_INTERLEAVED)

            # Scatter new KV into cache at cache_position
            updated_k = layer_past_k.scatter(2, idx, k)
            updated_v = layer_past_v.scatter(2, idx, v)

            present_keys_list.append(updated_k)
            present_values_list.append(updated_v)

            # GQA expand full cache for attention
            k_full = repeat_kv(updated_k, NUM_KV_GROUPS)
            v_full = repeat_kv(updated_v, NUM_KV_GROUPS)

            # Attention: Q(1,16,1,128) x K(1,16,kv_len,128)
            scaling = HEAD_DIM ** -0.5
            attn_w = torch.matmul(q, k_full.transpose(2, 3)) * scaling
            attn_w = attn_w + attention_mask
            attn_w = torch.softmax(attn_w, dim=-1, dtype=torch.float32).to(q.dtype)
            attn_out = torch.matmul(attn_w, v_full)

            attn_out = attn_out.transpose(1, 2).contiguous().view(1, 1, -1)
            attn_out = layer.self_attn.o_proj(attn_out)

            hidden_states = residual + attn_out

            # MLP
            residual = hidden_states
            hidden_states = residual + layer.mlp(
                layer.post_attention_layernorm(hidden_states))

        hidden_states = self.norm(hidden_states)
        logits = self.codec_head(hidden_states)

        present_keys = torch.stack(present_keys_list)
        present_values = torch.stack(present_values_list)

        return logits, hidden_states, present_keys, present_values


# ============================================================
# Create wrappers and validate
# ============================================================
print("\nCreating export wrappers...")
prefill_export = TalkerPrefillForExport(talker)
prefill_export.eval()

decode_export = TalkerDecodeForExport(talker)
decode_export.eval()

# Test prefill
SEQ_LEN = 20
dummy_embeds = torch.randn(1, SEQ_LEN, HIDDEN, dtype=torch.float32)
dummy_pos = torch.arange(SEQ_LEN).unsqueeze(0).unsqueeze(0).expand(3, 1, -1)

print(f"\nPrefill test: embeds={dummy_embeds.shape}, pos={dummy_pos.shape}")
with torch.no_grad():
    pf_logits, pf_hidden, pf_keys, pf_values = prefill_export(dummy_embeds, dummy_pos)
print(f"Prefill output: logits={pf_logits.shape}, hidden={pf_hidden.shape}")
print(f"  KV cache: keys={pf_keys.shape}, values={pf_values.shape}")

# Verify prefill matches original talker (no KV cache)
print("\nVerifying prefill matches original...")
with torch.no_grad():
    cache_position = torch.arange(SEQ_LEN)
    orig_out = talker.model(
        input_ids=None, attention_mask=None,
        position_ids=dummy_pos, past_key_values=None,
        inputs_embeds=dummy_embeds, use_cache=False,
        output_attentions=False, output_hidden_states=False,
        cache_position=cache_position,
    )
    orig_hidden = orig_out.last_hidden_state
    orig_logits = talker.codec_head(orig_hidden)

hidden_diff = (pf_hidden - orig_hidden).abs().max().item()
logits_diff = (pf_logits - orig_logits).abs().max().item()
print(f"  Hidden diff: {hidden_diff:.8f}")
print(f"  Logits diff: {logits_diff:.8f}")
if hidden_diff < 1e-4 and logits_diff < 1e-4:
    print("  MATCH - Prefill wrapper OK")
else:
    print("  WARNING - Prefill output diverges!")
    sys.exit(1)

# Test decode with KV cache from prefill
print("\nVerifying decode with KV cache...")
new_embed = torch.randn(1, 1, HIDDEN, dtype=torch.float32)
new_pos = torch.tensor([[[SEQ_LEN]]]).expand(3, 1, 1)  # position = SEQ_LEN
cache_pos_tensor = torch.tensor([SEQ_LEN], dtype=torch.long)

# Build attention mask: allow positions 0..SEQ_LEN (inclusive)
MAX_KV_FOR_TEST = SEQ_LEN + 10
attn_mask = torch.full((1, 1, 1, MAX_KV_FOR_TEST), torch.finfo(torch.float32).min)
attn_mask[:, :, :, :SEQ_LEN + 1] = 0.0

# Pad KV cache to MAX_KV_FOR_TEST
padded_keys = torch.zeros(NUM_LAYERS, 1, NUM_KV_HEADS, MAX_KV_FOR_TEST, HEAD_DIM)
padded_values = torch.zeros(NUM_LAYERS, 1, NUM_KV_HEADS, MAX_KV_FOR_TEST, HEAD_DIM)
padded_keys[:, :, :, :SEQ_LEN, :] = pf_keys
padded_values[:, :, :, :SEQ_LEN, :] = pf_values

with torch.no_grad():
    dec_logits, dec_hidden, dec_keys, dec_values = decode_export(
        new_embed, new_pos, cache_pos_tensor, attn_mask, padded_keys, padded_values)
print(f"Decode output: logits={dec_logits.shape}, hidden={dec_hidden.shape}")

# Compare with original model on full sequence (prefill + new token)
full_embeds = torch.cat([dummy_embeds, new_embed], dim=1)
full_pos = torch.arange(SEQ_LEN + 1).unsqueeze(0).unsqueeze(0).expand(3, 1, -1)
with torch.no_grad():
    full_out = talker.model(
        input_ids=None, attention_mask=None,
        position_ids=full_pos, past_key_values=None,
        inputs_embeds=full_embeds, use_cache=False,
        output_attentions=False, output_hidden_states=False,
        cache_position=torch.arange(SEQ_LEN + 1),
    )
    full_hidden = full_out.last_hidden_state
    full_logits = talker.codec_head(full_hidden)

# Compare last position
decode_logits_diff = (dec_logits[:, 0, :] - full_logits[:, -1, :]).abs().max().item()
decode_hidden_diff = (dec_hidden[:, 0, :] - full_hidden[:, -1, :]).abs().max().item()
print(f"  Decode logits diff vs full-sequence: {decode_logits_diff:.8f}")
print(f"  Decode hidden diff vs full-sequence: {decode_hidden_diff:.8f}")
if decode_logits_diff < 1e-3 and decode_hidden_diff < 1e-3:
    print("  MATCH - Decode with KV cache OK")
else:
    print(f"  WARNING - Decode output diverges!")
    sys.exit(1)


# ============================================================
# Export Prefill
# ============================================================
print("\n--- Exporting Prefill ---")
prefill_onnx = os.path.join(EXPORT_DIR, "talker_prefill.onnx")

print("Running torch.onnx.export for prefill...")
torch.onnx.export(
    prefill_export,
    (dummy_embeds, dummy_pos),
    prefill_onnx,
    input_names=["inputs_embeds", "position_ids"],
    output_names=["logits", "hidden_states", "present_keys", "present_values"],
    dynamic_axes={
        "inputs_embeds": {1: "seq_len"},
        "position_ids": {2: "seq_len"},
        "logits": {1: "seq_len"},
        "hidden_states": {1: "seq_len"},
        "present_keys": {3: "seq_len"},
        "present_values": {3: "seq_len"},
    },
    opset_version=18,
    dynamo=False,
)
print(f"ONNX saved to {prefill_onnx}")

print("Converting prefill to OpenVINO IR...")
pf_ov = ov.convert_model(prefill_onnx)
pf_ir = os.path.join(EXPORT_DIR, "talker_prefill.xml")
ov.save_model(pf_ov, pf_ir)
print(f"Saved to {pf_ir}")

# Validate prefill
print("Validating prefill on CPU...")
core = ov.Core()
pf_compiled = core.compile_model(pf_ov, "CPU")
pf_ov_result = pf_compiled({
    "inputs_embeds": dummy_embeds.numpy(),
    "position_ids": dummy_pos.numpy(),
})
ov_logits_diff = np.abs(
    pf_ov_result["logits"] - pf_logits.detach().numpy()).max()
ov_kv_diff = np.abs(
    pf_ov_result["present_keys"] - pf_keys.detach().numpy()).max()
print(f"  Logits diff (PyTorch vs OV): {ov_logits_diff:.6f}")
print(f"  KV cache diff: {ov_kv_diff:.6f}")
if ov_logits_diff < 0.01 and ov_kv_diff < 0.01:
    print("  PASS - Prefill exported successfully!")
else:
    print("  WARNING - Large difference")


# ============================================================
# Export Decode
# ============================================================
print("\n--- Exporting Decode ---")
decode_onnx = os.path.join(EXPORT_DIR, "talker_decode.onnx")

print("Running torch.onnx.export for decode...")
torch.onnx.export(
    decode_export,
    (new_embed, new_pos, cache_pos_tensor, attn_mask, padded_keys, padded_values),
    decode_onnx,
    input_names=["inputs_embeds", "position_ids", "cache_position",
                 "attention_mask", "past_keys", "past_values"],
    output_names=["logits", "hidden_states", "present_keys", "present_values"],
    dynamic_axes={
        "attention_mask": {3: "kv_len"},
        "past_keys": {3: "kv_len"},
        "past_values": {3: "kv_len"},
        "present_keys": {3: "kv_len"},
        "present_values": {3: "kv_len"},
    },
    opset_version=18,
    dynamo=False,
)
print(f"ONNX saved to {decode_onnx}")

print("Converting decode to OpenVINO IR...")
dec_ov = ov.convert_model(decode_onnx)
dec_ir = os.path.join(EXPORT_DIR, "talker_decode.xml")
ov.save_model(dec_ov, dec_ir)
print(f"Saved to {dec_ir}")

# Validate decode
print("Validating decode on CPU...")
dec_compiled = core.compile_model(dec_ov, "CPU")
dec_ov_result = dec_compiled({
    "inputs_embeds": new_embed.numpy(),
    "position_ids": new_pos.numpy(),
    "cache_position": cache_pos_tensor.numpy(),
    "attention_mask": attn_mask.numpy(),
    "past_keys": padded_keys.numpy(),
    "past_values": padded_values.numpy(),
})
dec_ov_logits_diff = np.abs(
    dec_ov_result["logits"] - dec_logits.detach().numpy()).max()
dec_ov_kv_diff = np.abs(
    dec_ov_result["present_keys"] - dec_keys.detach().numpy()).max()
print(f"  Logits diff (PyTorch vs OV): {dec_ov_logits_diff:.6f}")
print(f"  KV cache diff: {dec_ov_kv_diff:.6f}")
if dec_ov_logits_diff < 0.01 and dec_ov_kv_diff < 0.01:
    print("  PASS - Decode exported successfully!")
else:
    print("  WARNING - Large difference")


# ============================================================
# Multi-step validation: prefill + 3 decode steps
# ============================================================
print("\n--- Multi-step validation (prefill + 3 decode) ---")
# Run on OpenVINO
kv_k = pf_ov_result["present_keys"].copy()   # (28, 1, 8, 20, 128)
kv_v = pf_ov_result["present_values"].copy()

# Pad to larger buffer for decode steps
BUF_LEN = SEQ_LEN + 10
kv_buf_k = np.zeros((NUM_LAYERS, 1, NUM_KV_HEADS, BUF_LEN, HEAD_DIM), dtype=np.float32)
kv_buf_v = np.zeros((NUM_LAYERS, 1, NUM_KV_HEADS, BUF_LEN, HEAD_DIM), dtype=np.float32)
kv_buf_k[:, :, :, :SEQ_LEN, :] = kv_k
kv_buf_v[:, :, :, :SEQ_LEN, :] = kv_v

ov_decode_logits = []
for step in range(3):
    pos = SEQ_LEN + step
    step_embed = torch.randn(1, 1, HIDDEN, dtype=torch.float32)
    step_pos = np.array([[[pos]]] * 3, dtype=np.int64)
    step_cache_pos = np.array([pos], dtype=np.int64)
    step_mask = np.full((1, 1, 1, BUF_LEN), np.finfo(np.float32).min, dtype=np.float32)
    step_mask[:, :, :, :pos + 1] = 0.0

    result = dec_compiled({
        "inputs_embeds": step_embed.numpy(),
        "position_ids": step_pos,
        "cache_position": step_cache_pos,
        "attention_mask": step_mask,
        "past_keys": kv_buf_k,
        "past_values": kv_buf_v,
    })
    ov_decode_logits.append(result["logits"][0, 0, :5])
    kv_buf_k = result["present_keys"]
    kv_buf_v = result["present_values"]

    # Build full sequence for reference
    if step == 0:
        full_seq = torch.cat([dummy_embeds, step_embed], dim=1)
    else:
        full_seq = torch.cat([full_seq, step_embed], dim=1)

# Compare last step with full-sequence reference
full_pos_ref = torch.arange(full_seq.shape[1]).unsqueeze(0).unsqueeze(0).expand(3, 1, -1)
with torch.no_grad():
    ref_out = talker.model(
        input_ids=None, attention_mask=None,
        position_ids=full_pos_ref, past_key_values=None,
        inputs_embeds=full_seq, use_cache=False,
        output_attentions=False, output_hidden_states=False,
        cache_position=torch.arange(full_seq.shape[1]),
    )
    ref_logits = talker.codec_head(ref_out.last_hidden_state)

ref_last = ref_logits[0, -1, :5].numpy()
ov_last = ov_decode_logits[-1]
multi_diff = np.abs(ref_last - ov_last).max()
print(f"  3-step decode logits diff (OV vs reference): {multi_diff:.6f}")
if multi_diff < 0.05:
    print("  PASS - Multi-step KV cache validation OK!")
else:
    print(f"  WARNING - Multi-step divergence: {multi_diff}")


# ============================================================
# Report file sizes
# ============================================================
print("\n--- Export Summary ---")
for name, ir_path in [("Prefill", pf_ir), ("Decode", dec_ir)]:
    xml_size = os.path.getsize(ir_path) / 1024 / 1024
    bin_path = ir_path.replace(".xml", ".bin")
    bin_size = os.path.getsize(bin_path) / 1024 / 1024
    print(f"  {name}: {xml_size:.1f} MB (xml) + {bin_size:.1f} MB (bin) = {xml_size + bin_size:.1f} MB")
