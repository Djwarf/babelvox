"""BabelVox command-line interface."""
import argparse
import os
import sys

import numpy as np
import soundfile as sf

from babelvox.pipeline import BabelVox


def main():
    parser = argparse.ArgumentParser(
        description="BabelVox: Real-time text-to-speech via OpenVINO")
    parser.add_argument("--device", default="CPU", choices=["CPU", "NPU"],
                        help="OpenVINO device")
    parser.add_argument("--max-tokens", type=int, default=200,
                        help="Max generation steps")
    parser.add_argument("--text",
                        default="Hello, this is a test of BabelVox "
                                "text to speech running on OpenVINO.",
                        help="Text to synthesize")
    parser.add_argument("--language", default="English",
                        help="Language for synthesis (default: English)")
    parser.add_argument("--ref-audio", default=None,
                        help="Reference audio path for voice cloning")
    parser.add_argument("--output", "-o", default="output.wav",
                        help="Output WAV file path (default: output.wav)")
    parser.add_argument("--export-dir", default=None,
                        help="Directory containing exported OpenVINO models "
                             "(auto-downloads from HuggingFace if not provided)")
    parser.add_argument("--model-path",
                        default="Qwen/Qwen3-TTS-12Hz-0.6B-Base",
                        help="HuggingFace model path (for tokenizer)")
    parser.add_argument("--max-talker-seq", type=int, default=256,
                        help="NPU: max talker sequence length (no-kv-cache mode)")
    parser.add_argument("--max-decoder-frames", type=int, default=256,
                        help="NPU: max decoder codec frames")
    parser.add_argument("--kv-cache", action="store_true",
                        help="Use KV cache models (talker_prefill + talker_decode)")
    parser.add_argument("--max-kv-len", type=int, default=256,
                        help="KV cache buffer length (max prefill + generation tokens)")
    parser.add_argument("--int8", action="store_true",
                        help="Use INT8 quantized models")
    parser.add_argument("--precision", default=None,
                        choices=["fp16", "int8", "int4", "fp32"],
                        help="Model precision (default: fp16, or int8 if --int8)")
    parser.add_argument("--cp-kv-cache", action="store_true",
                        help="Use KV cache for code predictor (reduces CP time)")
    parser.add_argument("--talker-buckets", type=str, default=None,
                        help="NPU: comma-separated bucket sizes for multi-shape talker "
                             "(e.g. '64,128,256'). Picks smallest bucket per step.")
    parser.add_argument("--serve", action="store_true",
                        help="Start HTTP server instead of generating once")
    parser.add_argument("--host", default="0.0.0.0",
                        help="Server bind address (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8765,
                        help="Server port (default: 8765)")
    args = parser.parse_args()

    precision = args.precision or ("int8" if args.int8 else "fp16")
    talker_buckets = None
    if args.talker_buckets:
        talker_buckets = [int(x.strip()) for x in args.talker_buckets.split(",")]

    tts = BabelVox(
        model_path=args.model_path,
        export_dir=args.export_dir,
        device=args.device,
        max_talker_seq=args.max_talker_seq,
        max_decoder_frames=args.max_decoder_frames,
        use_kv_cache=args.kv_cache,
        max_kv_len=args.max_kv_len,
        use_cp_kv_cache=args.cp_kv_cache,
        talker_buckets=talker_buckets,
        precision=precision,
    )

    if args.serve:
        from babelvox.server import serve
        serve(tts, host=args.host, port=args.port)
    else:
        print(f"\n--- Generating with BabelVox ({args.device}) ---")
        wav, sr = tts.generate(
            text=args.text,
            language=args.language,
            ref_audio=args.ref_audio,
            max_new_tokens=args.max_tokens,
            temperature=0.9,
            top_k=50,
        )

        # Ensure output directory exists
        out_dir = os.path.dirname(args.output)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        sf.write(args.output, wav, sr)
        print(f"\nSaved to {args.output} ({len(wav)/sr:.2f}s @ {sr}Hz)")


if __name__ == "__main__":
    main()
