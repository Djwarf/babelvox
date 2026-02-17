"""Compare FP16 vs INT8 model quality by measuring logit/hidden divergence.

Runs both precision models on identical inputs and reports:
  - Max absolute difference in logits and hidden states
  - Cosine similarity of output distributions
  - Top-k token agreement (do they predict the same tokens?)
"""
import os
import time

import numpy as np
import openvino as ov

EXPORT_DIR = "openvino_export"
np.random.seed(42)


def cosine_sim(a, b):
    a, b = a.flatten(), b.flatten()
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)


def load_model(core, subdir, name, device="CPU"):
    path = os.path.join(EXPORT_DIR, subdir, f"{name}.xml")
    return core.compile_model(path, device)


def compare_talker(core):
    """Compare talker outputs across precisions."""
    print("=== Talker (28-layer transformer) ===")
    fp16 = load_model(core, "fp16", "talker")
    int8 = load_model(core, "int8", "talker")

    # Fixed input
    seq_len = 20
    embeds = np.random.randn(1, seq_len, 1024).astype(np.float32) * 0.1
    pos_ids = np.arange(seq_len, dtype=np.int64).reshape(1, 1, -1)
    pos_ids = np.broadcast_to(pos_ids, (3, 1, seq_len)).copy()

    inp = {"inputs_embeds": embeds, "position_ids": pos_ids}

    r_fp16 = fp16(inp)
    r_int8 = int8(inp)

    for name in ["logits", "hidden_states"]:
        a, b = r_fp16[name], r_int8[name]
        max_diff = np.abs(a - b).max()
        mean_diff = np.abs(a - b).mean()
        cos = cosine_sim(a, b)
        print(f"  {name}:")
        print(f"    Max diff:  {max_diff:.6f}")
        print(f"    Mean diff: {mean_diff:.6f}")
        print(f"    Cosine:    {cos:.8f}")

    # Top-k agreement
    logits_fp16 = r_fp16["logits"][0, -1, :]
    logits_int8 = r_int8["logits"][0, -1, :]
    top10_fp16 = set(np.argsort(logits_fp16)[-10:])
    top10_int8 = set(np.argsort(logits_int8)[-10:])
    overlap = len(top10_fp16 & top10_int8)
    print(f"  Top-10 token overlap: {overlap}/10")
    print(f"  Argmax match: {np.argmax(logits_fp16) == np.argmax(logits_int8)}")


def compare_code_predictor(core):
    """Compare code predictor outputs across precisions."""
    print("\n=== Code Predictor (5-layer transformer) ===")
    fp16 = load_model(core, "fp16", "code_predictor")
    int8 = load_model(core, "int8", "code_predictor")

    seq_len = 5
    embeds = np.random.randn(1, seq_len, 1024).astype(np.float32) * 0.1
    inp = {"inputs_embeds": embeds}

    r_fp16 = fp16(inp)
    r_int8 = int8(inp)

    a, b = r_fp16["hidden_states"], r_int8["hidden_states"]
    print(f"  Hidden states:")
    print(f"    Max diff:  {np.abs(a - b).max():.6f}")
    print(f"    Mean diff: {np.abs(a - b).mean():.6f}")
    print(f"    Cosine:    {cosine_sim(a, b):.8f}")


def compare_speaker_encoder(core):
    """Compare speaker encoder outputs across precisions."""
    print("\n=== Speaker Encoder ===")
    fp16 = load_model(core, "fp16", "speaker_encoder")
    int8 = load_model(core, "int8", "speaker_encoder")

    mel = np.random.randn(1, 100, 128).astype(np.float32) * 0.5
    inp = {"mel_spectrogram": mel}

    r_fp16 = fp16(inp)
    r_int8 = int8(inp)

    a, b = r_fp16[0], r_int8[0]
    print(f"  Speaker embedding (1024-dim):")
    print(f"    Max diff:  {np.abs(a - b).max():.6f}")
    print(f"    Mean diff: {np.abs(a - b).mean():.6f}")
    print(f"    Cosine:    {cosine_sim(a, b):.8f}")


def compare_tokenizer(core):
    """Compare tokenizer encoder/decoder across precisions."""
    print("\n=== Tokenizer Decoder ===")
    fp16 = load_model(core, "fp16", "tokenizer_decoder")
    int8 = load_model(core, "int8", "tokenizer_decoder")

    codes = np.random.randint(0, 1024, size=(1, 16, 20)).astype(np.int64)
    inp = {"codes": codes}

    r_fp16 = fp16(inp)
    r_int8 = int8(inp)

    a, b = r_fp16[0], r_int8[0]
    print(f"  Waveform output:")
    print(f"    Max diff:  {np.abs(a - b).max():.6f}")
    print(f"    Mean diff: {np.abs(a - b).mean():.6f}")
    print(f"    Cosine:    {cosine_sim(a, b):.8f}")
    print(f"    SNR:       {10 * np.log10(np.mean(a**2) / (np.mean((a-b)**2) + 1e-12)):.1f} dB")


if __name__ == "__main__":
    core = ov.Core()
    compare_talker(core)
    compare_code_predictor(core)
    compare_speaker_encoder(core)
    compare_tokenizer(core)
