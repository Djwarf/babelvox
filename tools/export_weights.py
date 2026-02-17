"""Export embedding tables and projection weights as numpy arrays.

Eliminates the need to load the full 2.4GB PyTorch model at inference time.
After running this, the pipeline only needs numpy + OpenVINO + tokenizer.

Exports:
  - codec_embedding (3072, 1024)
  - text_embedding (151936, 2048) — saved as float16 to save space
  - text_projection: fc1 weight/bias (2048x2048), fc2 weight/bias (1024x2048)
  - cp_embeddings: 15 x (2048, 1024)
  - cp_lm_heads: 15 x (2048, 1024) weight only
  - config.json: token IDs and model dimensions
"""
import json
import os

import numpy as np
import torch
from transformers import AutoConfig, AutoModel

from qwen_tts.core.models import Qwen3TTSConfig, Qwen3TTSForConditionalGeneration

MODEL_PATH = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
EXPORT_DIR = os.path.join("openvino_export", "weights")
os.makedirs(EXPORT_DIR, exist_ok=True)

print("Loading TTS model...")
AutoConfig.register("qwen3_tts", Qwen3TTSConfig)
AutoModel.register(Qwen3TTSConfig, Qwen3TTSForConditionalGeneration)
model = AutoModel.from_pretrained(
    MODEL_PATH, device_map="cpu", dtype=torch.float32,
    attn_implementation="eager",
)
talker = model.talker
talker.eval()

# --- Embeddings ---
print("Exporting embeddings...")
codec_emb = talker.model.codec_embedding.weight.detach().numpy()
np.save(os.path.join(EXPORT_DIR, "codec_embedding.npy"), codec_emb)
print(f"  codec_embedding: {codec_emb.shape} ({codec_emb.nbytes / 1e6:.1f} MB)")

# text_embedding is large (151936, 2048) — save as float16
text_emb = talker.model.text_embedding.weight.detach().numpy().astype(np.float16)
np.save(os.path.join(EXPORT_DIR, "text_embedding.npy"), text_emb)
print(f"  text_embedding: {text_emb.shape} as float16 ({text_emb.nbytes / 1e6:.1f} MB)")

# --- Text projection MLP (SiLU activation) ---
print("Exporting text_projection...")
tp = talker.text_projection
np.save(os.path.join(EXPORT_DIR, "text_proj_fc1_weight.npy"),
        tp.linear_fc1.weight.detach().numpy())
np.save(os.path.join(EXPORT_DIR, "text_proj_fc1_bias.npy"),
        tp.linear_fc1.bias.detach().numpy())
np.save(os.path.join(EXPORT_DIR, "text_proj_fc2_weight.npy"),
        tp.linear_fc2.weight.detach().numpy())
np.save(os.path.join(EXPORT_DIR, "text_proj_fc2_bias.npy"),
        tp.linear_fc2.bias.detach().numpy())
print(f"  fc1: {tp.linear_fc1.weight.shape}, fc2: {tp.linear_fc2.weight.shape}")

# --- Code predictor embeddings ---
print("Exporting code predictor embeddings...")
cp_embs = talker.code_predictor.model.codec_embedding
for i, emb in enumerate(cp_embs):
    w = emb.weight.detach().numpy()
    np.save(os.path.join(EXPORT_DIR, f"cp_embedding_{i}.npy"), w)
print(f"  {len(cp_embs)} embeddings, each {cp_embs[0].weight.shape}")

# --- Code predictor lm_heads ---
print("Exporting code predictor lm_heads...")
cp_heads = talker.code_predictor.lm_head
for i, head in enumerate(cp_heads):
    w = head.weight.detach().numpy()
    np.save(os.path.join(EXPORT_DIR, f"cp_lm_head_{i}_weight.npy"), w)
print(f"  {len(cp_heads)} heads, each {cp_heads[0].weight.shape}")

# --- Config values ---
print("Exporting config...")
tc = talker.config
config = {
    "hidden_size": tc.hidden_size,
    "num_code_groups": tc.num_code_groups,
    "codec_eos_token_id": tc.codec_eos_token_id,
    "codec_think_id": tc.codec_think_id,
    "codec_nothink_id": tc.codec_nothink_id,
    "codec_think_bos_id": tc.codec_think_bos_id,
    "codec_think_eos_id": tc.codec_think_eos_id,
    "codec_pad_id": tc.codec_pad_id,
    "codec_bos_id": tc.codec_bos_id,
    "tts_bos_token_id": model.config.tts_bos_token_id,
    "tts_eos_token_id": model.config.tts_eos_token_id,
    "tts_pad_token_id": model.config.tts_pad_token_id,
    "codec_language_id": tc.codec_language_id,
}
config_path = os.path.join(EXPORT_DIR, "config.json")
with open(config_path, "w") as f:
    json.dump(config, f, indent=2)
print(f"  Saved to {config_path}")

# --- Summary ---
total_bytes = sum(
    os.path.getsize(os.path.join(EXPORT_DIR, f))
    for f in os.listdir(EXPORT_DIR)
)
print(f"\nTotal export: {total_bytes / 1e6:.1f} MB in {EXPORT_DIR}/")
print("Done. Pipeline can now run without PyTorch model loading.")
