"""Profile per-step timing breakdown for the generation loop.

Measures where time goes within each autoregressive step:
  1. Talker forward pass (NPU/CPU)
  2. Code predictor (15 iterations on CPU)
  3. Numpy overhead (embeddings, sampling, concatenation)

Runs 30 steps and reports averages.
"""
import argparse
import json
import os
import time

import numpy as np
import openvino as ov

EXPORT_DIR = "openvino_export"


def profile_talker_step(device="NPU", precision="int8", n_steps=30):
    core = ov.Core()
    seq_len = 256

    if precision == "int8":
        model_dir = os.path.join(EXPORT_DIR, "int8")
    elif precision == "fp16":
        model_dir = os.path.join(EXPORT_DIR, "fp16")
    else:
        model_dir = EXPORT_DIR

    # --- Load models ---
    print(f"Loading models ({precision} on {device})...")
    t0 = time.time()

    if device == "NPU":
        m = core.read_model(os.path.join(model_dir, "talker.xml"))
        m.reshape({
            "inputs_embeds": [1, seq_len, 1024],
            "position_ids": [3, 1, seq_len],
        })
        talker = core.compile_model(m, "NPU")
    else:
        talker = core.compile_model(os.path.join(model_dir, "talker.xml"), "CPU")

    cp = core.compile_model(os.path.join(model_dir, "code_predictor.xml"), "CPU")
    print(f"  Models loaded in {time.time()-t0:.1f}s")

    # --- Load numpy weights ---
    weights_dir = os.path.join(EXPORT_DIR, "weights")
    codec_emb = np.load(os.path.join(weights_dir, "codec_embedding.npy"))
    cp_embs = [np.load(os.path.join(weights_dir, f"cp_embedding_{i}.npy")) for i in range(15)]
    cp_heads = [np.load(os.path.join(weights_dir, f"cp_lm_head_{i}_weight.npy")) for i in range(15)]

    # --- Prepare inputs ---
    embeds = np.random.randn(1, seq_len, 1024).astype(np.float32) * 0.02
    pos_ids = np.arange(seq_len, dtype=np.int64).reshape(1, 1, -1)
    pos_ids = np.broadcast_to(pos_ids, (3, 1, seq_len)).copy()

    # Warmup
    print("Warming up...")
    for _ in range(2):
        talker({"inputs_embeds": embeds, "position_ids": pos_ids})

    # --- Profile loop ---
    print(f"\nProfiling {n_steps} steps...")
    t_talker = []
    t_cp_total = []
    t_cp_per_group = []
    t_numpy = []

    for step in range(n_steps):
        # 1) Talker forward
        t1 = time.perf_counter()
        result = talker({"inputs_embeds": embeds, "position_ids": pos_ids})
        t2 = time.perf_counter()
        t_talker.append(t2 - t1)

        logits = result["logits"][:, -1:, :]
        hidden = result["hidden_states"][:, -1:, :]

        # 2) Numpy: sample code_0
        t3 = time.perf_counter()
        raw_logits = logits[0, 0, :].astype(np.float64)
        raw_logits /= 0.9  # temperature
        top_k = min(50, len(raw_logits))
        kth = np.partition(raw_logits, -top_k)[-top_k]
        raw_logits[raw_logits < kth] = -np.inf
        e = np.exp(raw_logits - raw_logits.max())
        probs = e / e.sum()
        code_0 = int(np.random.choice(len(probs), p=probs))
        t4 = time.perf_counter()

        # 3) Code predictor: 15 groups
        code_0_embed = codec_emb[[code_0]][np.newaxis, :, :]
        cp_input = np.concatenate([hidden, code_0_embed], axis=1)

        t_cp_start = time.perf_counter()
        cp_group_times = []
        codes = [code_0]
        for g in range(15):
            tg0 = time.perf_counter()
            cp_result = cp({"inputs_embeds": cp_input})
            cp_hidden = cp_result["hidden_states"][:, -1, :]
            cp_logits = (cp_hidden @ cp_heads[g].T).astype(np.float64)
            cp_logits /= 0.9
            e = np.exp(cp_logits[0] - cp_logits[0].max())
            p = e / e.sum()
            cp_code = int(np.random.choice(len(p), p=p))
            codes.append(cp_code)
            if g < 14:
                next_e = cp_embs[g][[cp_code]][np.newaxis, :, :]
                cp_input = np.concatenate([cp_input, next_e], axis=1)
            tg1 = time.perf_counter()
            cp_group_times.append(tg1 - tg0)
        t_cp_end = time.perf_counter()

        t_cp_total.append(t_cp_end - t_cp_start)
        t_cp_per_group.append(cp_group_times)

        # 4) Numpy: build next embedding
        t5 = time.perf_counter()
        combined = codec_emb[[codes[0]]]
        for i in range(1, 16):
            combined = combined + cp_embs[i-1][[codes[i]]]
        combined = combined[np.newaxis, :, :]
        # In real pipeline: add trailing text embed, concatenate to seq_embeds
        t6 = time.perf_counter()

        t_numpy.append((t4 - t3) + (t6 - t5))

    # --- Report ---
    total_per_step = np.array(t_talker) + np.array(t_cp_total) + np.array(t_numpy)

    print(f"\n{'='*60}")
    print(f"Per-step breakdown (avg of {n_steps} steps):")
    print(f"{'='*60}")
    print(f"  Talker (NPU):      {np.mean(t_talker)*1000:7.1f} ms  ({np.mean(t_talker)/np.mean(total_per_step)*100:4.1f}%)")
    print(f"  Code predictor:    {np.mean(t_cp_total)*1000:7.1f} ms  ({np.mean(t_cp_total)/np.mean(total_per_step)*100:4.1f}%)")
    print(f"  Numpy overhead:    {np.mean(t_numpy)*1000:7.1f} ms  ({np.mean(t_numpy)/np.mean(total_per_step)*100:4.1f}%)")
    print(f"  -----------------------------")
    print(f"  Total per step:    {np.mean(total_per_step)*1000:7.1f} ms")
    print(f"  Target (real-time): {1000/12:7.1f} ms  (12 Hz codec)")
    print(f"  Gap:               {(np.mean(total_per_step) - 1/12)*1000:7.1f} ms")

    print(f"\n  Code predictor per-group breakdown (ms):")
    cp_groups = np.array(t_cp_per_group)
    for g in range(15):
        print(f"    Group {g:2d}: {np.mean(cp_groups[:, g])*1000:6.2f} ms")

    print(f"\n  Talker step times (ms): "
          f"min={np.min(t_talker)*1000:.1f}, "
          f"median={np.median(t_talker)*1000:.1f}, "
          f"max={np.max(t_talker)*1000:.1f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="NPU", choices=["CPU", "NPU"])
    parser.add_argument("--precision", default="int8", choices=["fp16", "int8", "fp32"])
    parser.add_argument("--steps", type=int, default=30)
    args = parser.parse_args()

    profile_talker_step(device=args.device, precision=args.precision, n_steps=args.steps)
