# BabelVox

Real-time text-to-speech on Intel NPU via OpenVINO. Runs Qwen3-TTS 0.6B inference entirely on Intel NPU (AI Boost), achieving **RTF=1.0x** (real-time) speech synthesis on a Lunar Lake ultrabook.

No PyTorch at runtime. Dependencies: `openvino`, `numpy`, `librosa`, `soundfile`, `scipy`, `transformers` (tokenizer only).

## Installation

```bash
pip install babelvox
```

Or from source:

```bash
git clone https://github.com/Djwarf/babelvox.git
cd babelvox
pip install -e .
```

## Quick start

Models (~2.5 GB) are **downloaded automatically** from [HuggingFace](https://huggingface.co/djwarf/babelvox-openvino-int8) on first run and cached for future use. No manual setup needed.

### As a library

```python
from babelvox import BabelVox
import soundfile as sf

# CPU — works on any machine (models auto-download on first use)
tts = BabelVox(precision="int8", use_cp_kv_cache=True)
wav, sr = tts.generate("Don't panic.", language="English")
sf.write("output.wav", wav, sr)
```

For Intel NPU (Lunar Lake or later), enable hardware acceleration:

```python
tts = BabelVox(device="NPU", precision="int8",
               use_cp_kv_cache=True, talker_buckets=[64, 128, 256])
```

### From the command line

```bash
# CPU (works anywhere, ~1.1x RTF)
babelvox --int8 --cp-kv-cache --text "Hello world" --output hello.wav

# Intel NPU (real-time, RTF=1.0x)
babelvox --device NPU --int8 --cp-kv-cache --talker-buckets "64,128,256" \
  --text "Hello, this is real-time speech synthesis on an Intel NPU." \
  --output hello.wav
```

## Features

### Voice cloning

Clone any voice from a short reference audio clip (3-10 seconds):

```python
wav, sr = tts.generate("This sounds like someone else.",
                        ref_audio="reference.wav", language="English")
```

```bash
babelvox --int8 --cp-kv-cache --ref-audio reference.wav \
  --text "This sounds like someone else." --output cloned.wav
```

### 10 languages

Chinese, English, French, German, Italian, Japanese, Korean, Portuguese, Russian, Spanish.

```python
wav, sr = tts.generate("Bonjour le monde.", language="French")
wav, sr = tts.generate("Hallo Welt.", language="German")
```

### Sampling controls

Fine-tune the generation quality and diversity:

```python
wav, sr = tts.generate(
    "Hello world",
    temperature=0.9,          # higher = more expressive, lower = more stable
    top_k=50,                 # limit sampling to top-k tokens
    top_p=1.0,                # nucleus sampling threshold
    repetition_penalty=1.05,  # discourage repeated audio patterns
    max_new_tokens=512,       # max generation steps (1 step = 1/12 sec audio)
)
```

### Pre-download models

Cache models ahead of time instead of on first use:

```python
from babelvox import download_models
path = download_models()  # downloads ~2.5 GB to HuggingFace cache
```

### API reference

**`BabelVox(model_path, export_dir, device, precision, ...)`**

| Parameter | Default | Description |
|---|---|---|
| `model_path` | `"Qwen/Qwen3-TTS-12Hz-0.6B-Base"` | HuggingFace model (tokenizer only) |
| `export_dir` | `None` (auto-download) | Path to exported OpenVINO models |
| `device` | `"CPU"` | `"CPU"` or `"NPU"` |
| `precision` | `"fp16"` | `"fp16"`, `"int8"`, `"int4"`, or `"fp32"` |
| `use_cp_kv_cache` | `False` | KV cache for code predictor (recommended) |
| `talker_buckets` | `None` | NPU bucket sizes, e.g. `[64, 128, 256]` |

**`tts.generate(text, language, ref_audio, ...)`** returns `(waveform, sample_rate)`

| Parameter | Default | Description |
|---|---|---|
| `text` | required | Text to synthesize |
| `language` | `"English"` | One of the 10 supported languages |
| `ref_audio` | `None` | Path to reference WAV for voice cloning |
| `max_new_tokens` | `512` | Max generation steps (12 steps = 1 sec audio) |
| `temperature` | `0.9` | Sampling temperature (0 = greedy) |
| `top_k` | `50` | Top-k sampling |
| `top_p` | `1.0` | Nucleus sampling threshold |
| `repetition_penalty` | `1.05` | Penalty for repeated tokens |

### Exporting models yourself (optional)

The pre-built INT8 models are downloaded automatically. If you want to export from scratch (e.g., for a different quantization), the export scripts in `tools/` require PyTorch:

```bash
pip install torch qwen-tts nncf
python tools/export_tts_lm.py
python tools/export_speaker_encoder.py
python tools/export_decoder.py
python tools/export_tokenizer_encoder.py
python tools/export_cp_kvcache.py
python tools/export_weights.py
python tools/quantize_models.py --int8
```

## Performance

### Optimization progression

| Optimization | RTF | Per-step | Notes |
|---|---|---|---|
| FP16 NPU baseline | 3.0x | 246 ms | Full-recompute, padded to 256 tokens |
| + INT8 quantization | 2.1x | 156 ms | NNCF INT8_SYM weight compression |
| + CP KV cache | 1.4x | 106 ms | Eliminates redundant code predictor recomputation |
| + Multi-bucket talker | **1.0x** | **~80 ms** | Dynamically picks smallest NPU shape per step |

RTF = Real-Time Factor. RTF=1.0x means generating 1 second of audio takes 1 second of compute.

### Where the time goes (INT8 + CP KV cache, 256-token bucket)

| Component | Device | Time | Share |
|---|---|---|---|
| Talker (28-layer transformer) | NPU | 61 ms | 57% |
| Code predictor (15 groups) | CPU | 45 ms | 43% |
| Numpy overhead (embeddings, sampling) | CPU | <1 ms | <1% |

### Multi-bucket scaling

The talker scales linearly with sequence length on NPU. Pre-compiling at multiple sizes and routing to the smallest bucket that fits dramatically reduces wasted compute:

| Bucket size | Talker time | Total (+ 45ms CP) | Effective RTF |
|---|---|---|---|
| 64 | 15 ms | 60 ms | 0.72x |
| 128 | 22 ms | 67 ms | 0.80x |
| 192 | 31 ms | 76 ms | 0.91x |
| 256 | 43 ms | 88 ms | 1.06x |

## Hardware tested

- **CPU**: Intel Core Ultra 7 258V (Lunar Lake)
- **NPU**: Intel AI Boost (~13 TOPS)
- **RAM**: 32 GB LPDDR5x
- **Device**: Samsung Galaxy Book5 Pro

## Architecture

Qwen3-TTS uses 5 model components orchestrated in an autoregressive loop:

```
Text --> Tokenizer --> Text Embeddings --> Talker (28L transformer) --> Codec code_0
                                               |
                       Speaker Embedding ------+    code_0 --> Code Predictor (5L) --> codes 1-15
                       (from reference audio)            \--> repeat 15x with KV cache
                                                                     |
                                           All 16 codes --> Tokenizer Decoder --> Waveform
```

| Component | Layers | Hidden | Heads | Device | INT8 size |
|---|---|---|---|---|---|
| **Talker** | 28 | 1024 | 16Q/8KV | NPU | 444 MB |
| **Code predictor** | 5 | 1024 | 16Q/8KV | CPU | 79 MB |
| **Tokenizer encoder** | -- | -- | -- | NPU | 48 MB |
| **Tokenizer decoder** | -- | -- | -- | NPU | 114 MB |
| **Speaker encoder** | -- | -- | -- | NPU | 9 MB |

## CLI reference

| Flag | Default | Description |
|---|---|---|
| `--device` | `CPU` | `CPU` or `NPU` |
| `--int8` | off | Use INT8 quantized models |
| `--precision` | `fp16` | `fp16`, `int8`, `int4`, or `fp32` |
| `--cp-kv-cache` | off | KV cache for code predictor (recommended) |
| `--talker-buckets` | none | Comma-separated NPU bucket sizes (e.g. `64,128,256`) |
| `--kv-cache` | off | KV cache for talker (not recommended on NPU) |
| `--max-tokens` | 200 | Maximum generation steps |
| `--max-talker-seq` | 256 | Fixed talker padding (when not using buckets) |
| `--max-decoder-frames` | 256 | Max codec frames for audio decoder |
| `--max-kv-len` | 256 | KV cache buffer size (if `--kv-cache`) |
| `--text` | demo text | Text to synthesize |
| `--language` | English | Language for synthesis |
| `--ref-audio` | none | Reference audio for voice cloning |
| `--output` / `-o` | `output.wav` | Output WAV file path |
| `--export-dir` | auto-download | Directory with exported models (downloads from HuggingFace if not set) |
| `--model-path` | `Qwen/Qwen3-TTS-12Hz-0.6B-Base` | HuggingFace model (tokenizer) |

## Acknowledgments

Based on [Qwen3-TTS](https://github.com/Qwen/Qwen3-TTS) by Alibaba Qwen Team (Apache-2.0).

## License

Apache-2.0
