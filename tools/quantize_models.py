"""Convert OpenVINO IR models from FP32 to FP16 (and optionally INT8).

FP16 halves model size and memory bandwidth, which is the main bottleneck
on Intel NPU. INT8 requires NNCF calibration data.

Usage:
  python quantize_models.py              # FP16 conversion
  python quantize_models.py --int8       # INT8 weight compression (no calibration)
"""
import argparse
import os
import time

import openvino as ov

EXPORT_DIR = "openvino_export"
FP16_DIR = os.path.join(EXPORT_DIR, "fp16")
INT8_DIR = os.path.join(EXPORT_DIR, "int8")

MODELS = [
    "speaker_encoder",
    "talker",
    "talker_prefill",
    "talker_decode",
    "code_predictor",
    "cp_prefill",
    "cp_decode",
    "tokenizer_decoder",
    "tokenizer_encoder",
]


def convert_fp16():
    """Convert all models to FP16 using OpenVINO's compress_to_fp16."""
    os.makedirs(FP16_DIR, exist_ok=True)
    core = ov.Core()

    for name in MODELS:
        xml_path = os.path.join(EXPORT_DIR, f"{name}.xml")
        if not os.path.exists(xml_path):
            print(f"  Skipping {name} (not found)")
            continue

        t0 = time.time()
        model = core.read_model(xml_path)
        out_path = os.path.join(FP16_DIR, f"{name}.xml")
        ov.save_model(model, out_path, compress_to_fp16=True)

        # Report sizes
        orig_bin = os.path.getsize(xml_path.replace(".xml", ".bin")) / 1e6
        new_bin = os.path.getsize(out_path.replace(".xml", ".bin")) / 1e6
        ratio = new_bin / orig_bin if orig_bin > 0 else 0
        print(f"  {name}: {orig_bin:.1f} MB -> {new_bin:.1f} MB "
              f"({ratio:.1%}) [{time.time()-t0:.1f}s]")

    print(f"\nFP16 models saved to {FP16_DIR}/")


def convert_int8():
    """Convert models to INT8 using NNCF weight compression (no calibration data)."""
    try:
        import nncf
    except ImportError:
        print("ERROR: nncf not installed. Run: pip install nncf")
        return

    os.makedirs(INT8_DIR, exist_ok=True)
    core = ov.Core()

    for name in MODELS:
        xml_path = os.path.join(EXPORT_DIR, f"{name}.xml")
        if not os.path.exists(xml_path):
            print(f"  Skipping {name} (not found)")
            continue

        t0 = time.time()
        model = core.read_model(xml_path)
        compressed = nncf.compress_weights(model, mode=nncf.CompressWeightsMode.INT8_SYM)
        out_path = os.path.join(INT8_DIR, f"{name}.xml")
        ov.save_model(compressed, out_path)

        orig_bin = os.path.getsize(xml_path.replace(".xml", ".bin")) / 1e6
        new_bin = os.path.getsize(out_path.replace(".xml", ".bin")) / 1e6
        ratio = new_bin / orig_bin if orig_bin > 0 else 0
        print(f"  {name}: {orig_bin:.1f} MB -> {new_bin:.1f} MB "
              f"({ratio:.1%}) [{time.time()-t0:.1f}s]")

    print(f"\nINT8 models saved to {INT8_DIR}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quantize OpenVINO models")
    parser.add_argument("--int8", action="store_true",
                        help="Use INT8 weight compression (requires nncf)")
    args = parser.parse_args()

    print("--- FP16 Conversion ---")
    convert_fp16()

    if args.int8:
        print("\n--- INT8 Weight Compression ---")
        convert_int8()
