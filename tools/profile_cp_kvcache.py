"""Profile code predictor: full-recompute vs KV cache.

Runs 30 iterations of the 15-group code predictor loop and reports timing.
"""
import argparse
import os
import time

import numpy as np
import openvino as ov

EXPORT_DIR = "openvino_export"
CP_NUM_LAYERS = 5
CP_NUM_KV_HEADS = 8
CP_HEAD_DIM = 128
CP_MAX_KV_LEN = 20


def profile(precision="int8", n_iters=30):
    core = ov.Core()

    if precision == "int8":
        model_dir = os.path.join(EXPORT_DIR, "int8")
    elif precision == "fp16":
        model_dir = os.path.join(EXPORT_DIR, "fp16")
    else:
        model_dir = EXPORT_DIR

    weights_dir = os.path.join(EXPORT_DIR, "weights")
    cp_embs = [np.load(os.path.join(weights_dir, f"cp_embedding_{i}.npy")) for i in range(15)]
    cp_heads = [np.load(os.path.join(weights_dir, f"cp_lm_head_{i}_weight.npy")) for i in range(15)]
    codec_emb = np.load(os.path.join(weights_dir, "codec_embedding.npy"))

    # Load models
    print(f"Loading models ({precision})...")
    cp_full = core.compile_model(os.path.join(model_dir, "code_predictor.xml"), "CPU")
    cp_prefill = core.compile_model(os.path.join(model_dir, "cp_prefill.xml"), "CPU")
    cp_decode = core.compile_model(os.path.join(model_dir, "cp_decode.xml"), "CPU")

    # --- Profile full-recompute ---
    print(f"\nFull-recompute ({n_iters} iterations)...")
    times_full = []
    for _ in range(n_iters):
        hidden = np.random.randn(1, 1, 1024).astype(np.float32) * 0.1
        code_0_embed = codec_emb[[np.random.randint(0, 3072)]][np.newaxis, :, :]
        cp_input = np.concatenate([hidden, code_0_embed], axis=1)

        t0 = time.perf_counter()
        for g in range(15):
            result = cp_full({"inputs_embeds": cp_input})
            cp_hidden = result["hidden_states"][:, -1, :]
            cp_logits = (cp_hidden @ cp_heads[g].T).astype(np.float64)
            e = np.exp(cp_logits[0] - cp_logits[0].max())
            p = e / e.sum()
            cp_code = int(np.random.choice(len(p), p=p))
            if g < 14:
                next_e = cp_embs[g][[cp_code]][np.newaxis, :, :]
                cp_input = np.concatenate([cp_input, next_e], axis=1)
        times_full.append(time.perf_counter() - t0)

    # --- Profile KV cache ---
    print(f"KV cache ({n_iters} iterations)...")
    times_kv = []
    times_kv_prefill = []
    times_kv_decode = []
    for _ in range(n_iters):
        hidden = np.random.randn(1, 1, 1024).astype(np.float32) * 0.1
        code_0_embed = codec_emb[[np.random.randint(0, 3072)]][np.newaxis, :, :]
        cp_input = np.concatenate([hidden, code_0_embed], axis=1)

        t0 = time.perf_counter()

        # Prefill (2 tokens)
        tp = time.perf_counter()
        result = cp_prefill({"inputs_embeds": cp_input})
        cp_hidden = result["hidden_states"][:, -1, :]
        raw_keys = result["present_keys"]
        raw_values = result["present_values"]

        kv_k = np.zeros((CP_NUM_LAYERS, 1, CP_NUM_KV_HEADS, CP_MAX_KV_LEN, CP_HEAD_DIM), dtype=np.float32)
        kv_v = np.zeros_like(kv_k)
        kv_k[:, :, :, :2, :] = raw_keys
        kv_v[:, :, :, :2, :] = raw_values
        cp_pos = 2
        times_kv_prefill.append(time.perf_counter() - tp)

        # First group from prefill output
        cp_logits = (cp_hidden @ cp_heads[0].T).astype(np.float64)
        e = np.exp(cp_logits[0] - cp_logits[0].max())
        p = e / e.sum()
        cp_code = int(np.random.choice(len(p), p=p))

        # Decode 14 groups
        td_total = 0
        for g in range(1, 15):
            next_e = cp_embs[g - 1][[cp_code]][np.newaxis, :, :]

            td = time.perf_counter()
            cache_pos = np.array([cp_pos], dtype=np.int64)
            attn_mask = np.full((1, 1, 1, CP_MAX_KV_LEN),
                                np.finfo(np.float32).min, dtype=np.float32)
            attn_mask[:, :, :, :cp_pos + 1] = 0.0

            dec_result = cp_decode({
                "inputs_embeds": next_e,
                "cache_position": cache_pos,
                "attention_mask": attn_mask,
                "past_keys": kv_k,
                "past_values": kv_v,
            })
            cp_hidden = dec_result["hidden_states"][:, 0, :]
            kv_k = dec_result["present_keys"]
            kv_v = dec_result["present_values"]
            cp_pos += 1
            td_total += time.perf_counter() - td

            cp_logits = (cp_hidden @ cp_heads[g].T).astype(np.float64)
            e = np.exp(cp_logits[0] - cp_logits[0].max())
            p = e / e.sum()
            cp_code = int(np.random.choice(len(p), p=p))

        times_kv.append(time.perf_counter() - t0)
        times_kv_decode.append(td_total)

    # --- Report ---
    print(f"\n{'='*55}")
    print(f"Code Predictor timing ({precision}, {n_iters} iterations)")
    print(f"{'='*55}")
    print(f"  Full-recompute:  {np.mean(times_full)*1000:7.1f} ms  (15 sequential passes)")
    print(f"  KV cache total:  {np.mean(times_kv)*1000:7.1f} ms  (1 prefill + 14 decode)")
    print(f"    Prefill:       {np.mean(times_kv_prefill)*1000:7.1f} ms  (2-token)")
    print(f"    Decode (14x):  {np.mean(times_kv_decode)*1000:7.1f} ms  ({np.mean(times_kv_decode)/14*1000:.1f} ms each)")
    print(f"  Speedup:         {np.mean(times_full)/np.mean(times_kv):.2f}x")
    print(f"\n  Target (real-time budget for CP): ~20 ms")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--precision", default="int8", choices=["fp16", "int8", "fp32"])
    parser.add_argument("--iters", type=int, default=30)
    args = parser.parse_args()
    profile(precision=args.precision, n_iters=args.iters)
