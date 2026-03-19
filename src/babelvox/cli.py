"""BabelVox command-line interface."""
import argparse
import logging
import os

import soundfile as sf

from babelvox.pipeline import BabelVox

logger = logging.getLogger("babelvox")


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
    parser.add_argument("--ref-text", default=None,
                        help="Transcription of the reference audio (improves cloning)")
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
    parser.add_argument("--cache-dir", default=None,
                        help="OpenVINO model cache directory (avoids recompilation)")
    parser.add_argument("--fallback-cpu", action="store_true",
                        help="Fall back to CPU if NPU compilation fails for a model")
    parser.add_argument("--serve", action="store_true",
                        help="Start HTTP server instead of generating once")
    parser.add_argument("--host", default="0.0.0.0",
                        help="Server bind address (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8765,
                        help="Server port (default: 8765)")
    parser.add_argument("--cors-origin", default="*",
                        help="CORS Allow-Origin header (default: *)")
    parser.add_argument("--audio-dir", default=None,
                        help="Allowed directory for ref_audio paths in server mode")
    parser.add_argument("--ws-port", type=int, default=None,
                        help="WebSocket server port (enables WS alongside HTTP "
                             "when --serve is used, requires babelvox[ws])")
    parser.add_argument("--speaker", default=None,
                        help="Use a named speaker profile")
    parser.add_argument("--speaker-dir", default=None,
                        help="Speaker library directory (default: ~/.babelvox/speakers)")
    parser.add_argument("--save-speaker", default=None, metavar="NAME",
                        help="Extract speaker from --ref-audio and save as NAME")
    parser.add_argument("--list-speakers", action="store_true",
                        help="List saved speaker profiles and exit")
    parser.add_argument("--ssml", action="store_true",
                        help="Treat --text as SSML markup")
    parser.add_argument("--rate", type=float, default=1.0,
                        help="Speech rate multiplier (default: 1.0)")
    parser.add_argument("--pitch", type=float, default=0.0,
                        help="Pitch shift in semitones (default: 0)")
    parser.add_argument("--volume", type=float, default=1.0,
                        help="Volume multiplier (default: 1.0)")
    parser.add_argument("--emotion", default=None,
                        choices=["happy", "sad", "angry", "surprised", "neutral"],
                        help="Emotion style for speech")
    parser.add_argument("--longform", action="store_true",
                        help="Enable long-form synthesis mode")
    parser.add_argument("--strategy", default="natural",
                        choices=["paragraph", "sentence", "natural", "chapter"],
                        help="Text segmentation strategy (default: natural)")
    parser.add_argument("--split-output", action="store_true",
                        help="Output individual segment files (output_001.wav, ...)")
    parser.add_argument("--timestamps", action="store_true",
                        help="Write timestamps JSON alongside audio")
    parser.add_argument("--text-file", default=None, metavar="PATH",
                        help="Read text from file instead of --text")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Enable debug logging")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

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
        cache_dir=args.cache_dir,
        fallback_cpu=args.fallback_cpu,
    )

    # Speaker library setup
    from babelvox.speakers import SpeakerLibrary
    speaker_dir = args.speaker_dir or os.path.expanduser("~/.babelvox/speakers")
    tts.speaker_library = SpeakerLibrary(speaker_dir)

    if args.list_speakers:
        profiles = tts.speaker_library.list_profiles()
        if not profiles:
            print("No saved speaker profiles.")
        for p in profiles:
            print(f"  {p['name']:20s} {p.get('language', ''):10s} {p.get('description', '')}")
        return

    if args.save_speaker:
        if not args.ref_audio:
            parser.error("--save-speaker requires --ref-audio")
        profile = tts.save_speaker(args.save_speaker, args.ref_audio,
                                   language=args.language)
        print(f"Saved speaker '{profile.name}'")
        return

    if args.serve:
        from babelvox.server import serve
        if args.ws_port:
            import threading

            from babelvox.ws_server import ws_serve
            ws_thread = threading.Thread(
                target=ws_serve, args=(tts,),
                kwargs={"host": args.host, "port": args.ws_port,
                        "audio_dir": args.audio_dir},
                daemon=True)
            ws_thread.start()
        serve(tts, host=args.host, port=args.port,
              cors_origin=args.cors_origin, audio_dir=args.audio_dir)
    else:
        # Read text from file if specified
        text = args.text
        if args.text_file:
            with open(args.text_file) as f:
                text = f.read()

        prosody = None
        if args.rate != 1.0 or args.pitch != 0.0 or args.volume != 1.0 or args.emotion:
            from babelvox.prosody import ProsodyConfig
            prosody = ProsodyConfig(
                rate=args.rate, pitch_semitones=args.pitch,
                volume=args.volume, emotion=args.emotion)

        # Ensure output directory exists
        out_dir = os.path.dirname(args.output)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        if args.longform:
            from babelvox.longform import LongFormSynthesizer
            logger.info("Long-form synthesis (%s, %s)...", args.device, args.strategy)
            synth = LongFormSynthesizer(tts)
            result = synth.synthesize(
                text, strategy=args.strategy, speaker=args.speaker,
                language=args.language, prosody=prosody,
                max_new_tokens=args.max_tokens)

            sf.write(args.output, result.waveform, result.sample_rate)
            logger.info("Saved to %s (%.2fs @ %dHz, %d segments)",
                        args.output, result.total_duration, result.sample_rate,
                        len(result.segments))

            if args.split_output:
                base, ext = os.path.splitext(args.output)
                for sr_item in result.segments:
                    if sr_item.segment.type == "chapter_marker":
                        continue
                    seg_path = f"{base}_{sr_item.segment.index + 1:03d}{ext}"
                    wav_seg, _ = tts.generate(
                        text=sr_item.segment.text, language=args.language,
                        speaker=args.speaker, prosody=prosody,
                        max_new_tokens=args.max_tokens)
                    sf.write(seg_path, wav_seg, 24000)
                    logger.info("  Segment %d: %s", sr_item.segment.index + 1, seg_path)

            if args.timestamps:
                import json
                ts_path = os.path.splitext(args.output)[0] + ".timestamps.json"
                ts_data = [{
                    "index": sr_item.segment.index,
                    "type": sr_item.segment.type,
                    "text": sr_item.segment.text[:100],
                    "start_time": round(sr_item.start_time, 3),
                    "end_time": round(sr_item.end_time, 3),
                    "duration": round(sr_item.duration, 3),
                } for sr_item in result.segments]
                with open(ts_path, "w") as f:
                    json.dump(ts_data, f, indent=2)
                logger.info("Timestamps: %s", ts_path)
        else:
            logger.info("Generating with BabelVox (%s)...", args.device)
            wav, sr = tts.generate(
                text=text,
                language=args.language,
                ref_audio=args.ref_audio,
                ref_text=args.ref_text,
                speaker=args.speaker,
                max_new_tokens=args.max_tokens,
                temperature=0.9,
                top_k=50,
                ssml=args.ssml,
                prosody=prosody,
            )
            sf.write(args.output, wav, sr)
            logger.info("Saved to %s (%.2fs @ %dHz)", args.output, len(wav)/sr, sr)


if __name__ == "__main__":
    main()
