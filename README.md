# BabelVox

Real-time text-to-speech on Intel NPU via OpenVINO. Runs Qwen3-TTS 0.6B inference entirely on Intel NPU (AI Boost), achieving **RTF=1.0x** (real-time) speech synthesis on a Lunar Lake ultrabook.

No PyTorch at runtime. Dependencies: `openvino`, `numpy`, `librosa`, `soundfile`, `scipy`, `transformers` (tokenizer only), `num2words`, `defusedxml`.

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

For Intel NPU (Lunar Lake or later), enable hardware acceleration and model caching:

```python
tts = BabelVox(device="NPU", precision="int8",
               use_cp_kv_cache=True, talker_buckets=[64, 128, 256],
               cache_dir="./ov_cache")
```

### From the command line

```bash
# CPU (works anywhere, ~1.1x RTF)
babelvox --int8 --cp-kv-cache --text "Hello world" --output hello.wav

# Intel NPU (real-time, RTF=1.0x)
babelvox --device NPU --int8 --cp-kv-cache --talker-buckets "64,128,256" \
  --cache-dir ./ov_cache \
  --text "Hello, this is real-time speech synthesis on an Intel NPU." \
  --output hello.wav
```

## Features

### Voice cloning

Clone any voice from a short reference audio clip (3-10 seconds). For best results, provide a transcription of the reference audio with `ref_text`:

```python
wav, sr = tts.generate(
    "This sounds like someone else.",
    ref_audio="reference.wav",
    ref_text="The words spoken in reference dot wav.",
    language="English",
)
```

```bash
babelvox --int8 --cp-kv-cache --ref-audio reference.wav \
  --ref-text "The words spoken in reference dot wav." \
  --text "This sounds like someone else." --output cloned.wav
```

### Speaker profiles

Save, load, and reuse named speaker voices across sessions:

```python
from babelvox import BabelVox, SpeakerLibrary

tts = BabelVox(precision="int8", use_cp_kv_cache=True)
tts.speaker_library = SpeakerLibrary("~/.babelvox/speakers")

# Save a speaker from reference audio
tts.save_speaker("alice", "alice.wav", language="English", gender="female")

# Use by name
wav, sr = tts.generate("Hello from Alice", speaker="alice")
```

```bash
babelvox --save-speaker alice --ref-audio alice.wav --language English
babelvox --speaker alice --text "Hello from Alice" -o hello.wav
babelvox --list-speakers
```

Mix or interpolate voices:

```python
from babelvox import mix_speakers, interpolate_speakers

lib = tts.speaker_library
a = lib.load("alice").embedding
b = lib.load("bob").embedding
mixed = mix_speakers([a, b], [0.7, 0.3])      # 70% alice, 30% bob
blended = interpolate_speakers(a, b, 0.5)      # 50/50 blend
wav, sr = tts.generate("Hello", speaker_embed=mixed)
```

**Server API:**

| Method | Path | Description |
|---|---|---|
| `GET` | `/speakers` | List all saved speaker profiles |
| `POST` | `/speakers` | Save a speaker (`{"name": "alice", "ref_audio": "..."}`) |
| `DELETE` | `/speakers/{name}` | Delete a speaker profile |
| `POST` | `/tts/batch` | Batch synthesis (`{"items": [{"text": "...", "speaker": "..."}]}`) |

Use `"speaker": "alice"` in POST /tts to synthesize with a saved voice.

### Prosody & emotion control

Control speech rate, pitch, volume, and emotion style:

```python
from babelvox import ProsodyConfig

prosody = ProsodyConfig(rate=1.2, pitch_semitones=2, volume=0.8, emotion="happy")
wav, sr = tts.generate("This is exciting news!", prosody=prosody)
```

```bash
babelvox --emotion happy --rate 1.2 --pitch 2 --text "Exciting news!" -o happy.wav
babelvox --rate 0.8 --volume 0.5 --text "Slow and quiet" -o slow.wav
```

**Emotion styles:** `happy`, `sad`, `angry`, `surprised`, `neutral`. Emotion hints are best-effort (text manipulation + sampling adjustment). Rate, pitch, and volume are guaranteed via librosa post-processing.

**Server API:**
```json
POST /tts
{"text": "Hello!", "prosody": {"emotion": "happy", "rate": 1.2, "pitch_semitones": 2}}
```

### Long-form synthesis

Synthesize books, articles, or long texts with automatic segmentation, crossfade, and progress tracking:

```python
from babelvox import LongFormSynthesizer

synth = LongFormSynthesizer(tts)
result = synth.synthesize(long_text, strategy="natural", speaker="alice")
sf.write("book.wav", result.waveform, result.sample_rate)

# Access per-segment timestamps
for seg in result.segments:
    print(f"  [{seg.start_time:.1f}s] {seg.segment.text[:50]}...")
```

```bash
# From a text file with chapter detection
babelvox --longform --strategy chapter --speaker alice \
  --text-file book.txt --timestamps -o book.wav

# Split into individual segment files
babelvox --longform --split-output --text-file article.txt -o article.wav
```

**Segmentation strategies:**

| Strategy | Behavior |
|---|---|
| `paragraph` | Split on double newlines |
| `sentence` | Split on sentence-ending punctuation (handles abbreviations) |
| `natural` (default) | Paragraphs first, split long paragraphs (>300 chars) into sentences |
| `chapter` | Split on markdown headers (`# Chapter`, `## Section`); headers become timestamps |

**Server API:**
```bash
curl -X POST http://localhost:8765/tts/longform \
  -d '{"text": "Para one.\n\nPara two.", "strategy": "natural", "speaker": "alice"}' \
  -o longform.wav
```

### Voice persistence

Without a reference audio, each `generate()` call may produce a different voice. To keep a consistent voice across multiple calls:

```python
# Extract once from reference audio, reuse for all subsequent calls
tts.default_speaker = tts.extract_speaker_embedding("voice.wav")
wav1, sr = tts.generate("First sentence.")
wav2, sr = tts.generate("Second sentence.")  # same voice

# Or pass a speaker embedding directly per call
embed = tts.extract_speaker_embedding("voice.wav")
wav, sr = tts.generate("Hello", speaker_embed=embed)
```

### Model caching

On NPU, OpenVINO compiles models at startup which can take minutes on first run. Use `cache_dir` to cache compiled models so subsequent launches are instant:

```python
tts = BabelVox(device="NPU", cache_dir="./ov_cache", precision="int8")
# First run: ~200s compile. Second run: instant.
```

```bash
babelvox --device NPU --cache-dir ./ov_cache --int8 --cp-kv-cache
```

### Streaming generation

Start playback immediately instead of waiting for the full utterance. `generate_stream()` yields waveform chunks every N codec frames (~1 second per 12 frames):

```python
import sounddevice as sd

for chunk, sr in tts.generate_stream("A long paragraph of text...",
                                      chunk_frames=12):
    sd.play(chunk, sr)
    sd.wait()
```

#### Silence-aware chunking

Split at natural pauses between words instead of fixed intervals. Produces complete words/phrases per chunk with crossfade overlap for click-free playback:

```python
for chunk, sr in tts.generate_stream(
    "A long paragraph of text...",
    split_on_silence=True,   # yield at pauses between words
    min_chunk_frames=6,      # at least 0.5s per chunk
    max_chunk_frames=48,     # at most 4s per chunk
    silence_threshold=0.02,  # RMS energy threshold for silence
    crossfade_samples=1200,  # 50ms overlap for click-free joins
):
    sd.play(chunk, sr)
    sd.wait()
```

For near-zero gap between chunks, overlap generation and playback with a thread:

```python
import threading, queue

audio_q = queue.Queue()

def producer():
    for chunk, sr in tts.generate_stream("Long text...",
                                          split_on_silence=True):
        audio_q.put((chunk, sr))
    audio_q.put(None)

threading.Thread(target=producer).start()

while True:
    item = audio_q.get()
    if item is None:
        break
    sd.play(item[0], item[1])
    sd.wait()
```

### Text preprocessing & SSML

BabelVox automatically normalizes text before synthesis — expanding abbreviations (`Dr.` → `Doctor`), numbers (`$4.50` → `four dollars and fifty cents`), dates, times, and phone numbers into spoken form. Unicode punctuation is cleaned up and repeated punctuation is collapsed.

For fine-grained control, pass SSML markup:

```python
ssml = """<speak>
  The price is <say-as interpret-as="number">$4.50</say-as>.
  <break time="500ms"/>
  Call <say-as interpret-as="telephone">555-123-4567</say-as> for details.
  <sub alias="World Wide Web Consortium">W3C</sub> approved.
</speak>"""
wav, sr = tts.generate(ssml, ssml=True)
```

```bash
babelvox --ssml --text '<speak>Hello.<break time="500ms"/>World.</speak>' -o out.wav
```

**Supported SSML tags:**

| Tag | Effect |
|---|---|
| `<break time="500ms"/>` | Insert pause (maps to punctuation) |
| `<break strength="strong"/>` | Insert pause by strength level |
| `<sub alias="...">` | Replace text with alias |
| `<say-as interpret-as="number\|date\|time\|telephone\|spell-out">` | Normalize to spoken form |
| `<emphasis>`, `<prosody>`, `<phoneme>` | Parsed into annotations (best-effort prosody hints) |

Onomatopoeia (boom, crash, sizzle, etc.) is detected and annotated for prosody emphasis.

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

### HTTP API (cross-language integration)

Run BabelVox as an HTTP server so any language (JavaScript, Go, Rust, etc.) can call it:

```bash
babelvox --serve --int8 --cp-kv-cache --port 8765
```

A built-in web demo UI is available at `http://localhost:8765/` with interactive controls for all features.

Then from any client:

```bash
curl -X POST http://localhost:8765/tts \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello world", "language": "English"}' \
  -o hello.wav
```

From JavaScript:

```javascript
const res = await fetch("http://localhost:8765/tts", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ text: "Hello world" }),
});
const blob = await res.blob();
const audio = new Audio(URL.createObjectURL(blob));
audio.play();
```

**Endpoints:**

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Built-in web demo UI |
| `POST` | `/tts` | Synthesize speech — JSON body in, WAV bytes out |
| `POST` | `/tts/batch` | Batch synthesis — multiple texts in one request |
| `POST` | `/tts/longform` | Long-form synthesis — auto-segmented, single WAV out |
| `GET` | `/tts/stream` | Streaming SSE — audio chunks as base64 events |
| `GET` | `/tts/longform/stream` | Long-form SSE — per-segment progress + audio |
| `GET` | `/speakers` | List saved speaker profiles |
| `POST` | `/speakers` | Save a new speaker profile |
| `DELETE` | `/speakers/{name}` | Delete a speaker profile |
| `GET` | `/health` | Health check — returns `{"status": "ok"}` |

**POST /tts request body:**

| Field | Required | Default | Description |
|---|---|---|---|
| `text` | yes | — | Text to synthesize |
| `language` | no | `"English"` | One of the 10 supported languages |
| `ref_audio` | no | `null` | Path to reference WAV for voice cloning |
| `ref_text` | no | `null` | Transcription of reference audio (improves cloning) |
| `max_new_tokens` | no | `512` | Max generation steps |
| `temperature` | no | `0.9` | Sampling temperature |
| `top_k` | no | `50` | Top-k sampling |
| `top_p` | no | `1.0` | Nucleus sampling threshold |
| `repetition_penalty` | no | `1.05` | Penalty for repeated tokens |
| `ssml` | no | `false` | Treat `text` as SSML markup |
| `speaker` | no | `null` | Named speaker profile |
| `prosody` | no | `null` | Prosody object: `{"rate": 1.0, "pitch_semitones": 0, "volume": 1.0, "emotion": null}` |

### Web demo UI

The server includes a built-in interactive demo at `GET /` — a single-page app with no external dependencies. It covers every API feature:

| Tab | Feature |
|---|---|
| **Basic TTS** | Text input, 10 languages, speaker select, waveform visualization |
| **Prosody & Emotion** | Rate/pitch/volume sliders, 5 emotion styles |
| **SSML** | Monospace editor with snippet buttons for all supported tags |
| **SSE Stream** | Real-time audio playback via Web Audio API as chunks arrive |
| **WebSocket** | Bidirectional streaming with connect/send/cancel controls |
| **Long-form** | Full synthesis or SSE-streamed with progress bar |
| **Batch** | Dynamic multi-item synthesis with per-item playback |
| **Speakers** | List, select, and delete saved speaker profiles |
| **Settings** | Sampling parameters, connection test, request log |

Works on both CPU and NPU — just start the server and open `http://localhost:8765/` in your browser.

### SSE streaming

Stream audio chunks in real-time via Server-Sent Events (no extra dependencies):

```bash
curl -N "http://localhost:8765/tts/stream?text=Hello+world&format=pcm_s16le"
```

Events: `start` (sample rate + format), `audio` (base64-encoded chunk), `done` (total duration), `error`.

### WebSocket streaming

For bidirectional real-time streaming with cancel support, install the `ws` extra:

```bash
pip install babelvox[ws]
babelvox --serve --int8 --cp-kv-cache --port 8765 --ws-port 8766
```

Connect and send JSON requests, receive binary audio chunks:

```javascript
const ws = new WebSocket("ws://localhost:8766");
ws.onopen = () => ws.send(JSON.stringify({ text: "Hello world", format: "pcm_s16le" }));
ws.onmessage = (e) => {
  if (typeof e.data === "string") {
    const msg = JSON.parse(e.data);
    console.log(msg.event); // "start", "done", or "error"
  } else {
    // Binary audio chunk — play with Web Audio API
  }
};
// Cancel mid-stream:
ws.send(JSON.stringify({ event: "cancel" }));
```

**Formats:** `pcm_s16le` (raw 16-bit PCM, lowest latency) or `wav_chunks` (each chunk is a complete WAV file).

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
| `cache_dir` | `None` | OpenVINO compiled model cache directory |

**`tts.generate(text, language, ref_audio, ...)`** returns `(waveform, sample_rate)`

| Parameter | Default | Description |
|---|---|---|
| `text` | required | Text to synthesize |
| `language` | `"English"` | One of the 10 supported languages |
| `ref_audio` | `None` | Path to reference WAV for voice cloning |
| `ref_text` | `None` | Transcription of the reference audio (improves cloning) |
| `speaker_embed` | `None` | Pre-extracted speaker embedding (numpy array) |
| `max_new_tokens` | `512` | Max generation steps (12 steps = 1 sec audio) |
| `temperature` | `0.9` | Sampling temperature (0 = greedy) |
| `top_k` | `50` | Top-k sampling |
| `top_p` | `1.0` | Nucleus sampling threshold |
| `repetition_penalty` | `1.05` | Penalty for repeated tokens |
| `ssml` | `False` | Treat `text` as SSML markup |
| `speaker` | `None` | Named speaker profile (requires `speaker_library` set) |
| `prosody` | `None` | `ProsodyConfig` for rate/pitch/volume/emotion control |

**`tts.generate_stream(text, language, ..., chunk_frames=12)`** — same args as `generate()`, plus:

| Parameter | Default | Description |
|---|---|---|
| `chunk_frames` | `12` | Codec frames per chunk when not using silence detection (12 = ~1 sec) |
| `split_on_silence` | `False` | Yield at natural pauses between words instead of fixed intervals |
| `min_chunk_frames` | `6` | Minimum frames before considering a silence split (~0.5s) |
| `max_chunk_frames` | `48` | Force yield after this many frames even without silence (~4s) |
| `silence_threshold` | `0.02` | RMS energy threshold for silence detection |
| `crossfade_samples` | `1200` | Overlap samples at chunk edges for click-free joins (50ms at 24kHz) |

Yields `(waveform_chunk, 24000)` tuples as audio is generated.

**`tts.extract_speaker_embedding(audio_path)`** returns numpy array `(1, 1024)`

**`tts.default_speaker`** — set to a speaker embedding for consistent voice across calls

**`tts.save_speaker(name, ref_audio, **metadata)`** — extract embedding and save as named profile

**`LongFormSynthesizer(tts).synthesize(text, strategy, speaker, ...)`** returns `SynthesisResult`

| Parameter | Default | Description |
|---|---|---|
| `text` | required | Long text to synthesize |
| `strategy` | `"natural"` | `"paragraph"`, `"sentence"`, `"natural"`, or `"chapter"` |
| `speaker` | `None` | Named speaker profile for voice consistency |
| `language` | `"English"` | Language for synthesis |
| `prosody` | `None` | `ProsodyConfig` applied to all segments |
| `crossfade_samples` | `2400` | Overlap samples between segments (100ms) |
| `progress_callback` | `None` | Called with `SynthesisProgress` after each segment |
| `resume_from` | `0` | Skip first N segments (for resume) |

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
| `--cache-dir` | none | OpenVINO compiled model cache directory |
| `--max-tokens` | 200 | Maximum generation steps |
| `--max-talker-seq` | 256 | Fixed talker padding (when not using buckets) |
| `--max-decoder-frames` | 256 | Max codec frames for audio decoder |
| `--max-kv-len` | 256 | KV cache buffer size (if `--kv-cache`) |
| `--ssml` | off | Treat `--text` as SSML markup |
| `--text` | demo text | Text to synthesize |
| `--language` | English | Language for synthesis |
| `--ref-audio` | none | Reference audio for voice cloning |
| `--ref-text` | none | Transcription of reference audio (improves cloning) |
| `--speaker` | none | Use a named speaker profile |
| `--speaker-dir` | `~/.babelvox/speakers` | Speaker library directory |
| `--save-speaker` | none | Save speaker from `--ref-audio` as named profile |
| `--list-speakers` | off | List saved speaker profiles and exit |
| `--serve` | off | Start HTTP server instead of generating once |
| `--host` | `0.0.0.0` | Server bind address |
| `--port` | `8765` | Server port |
| `--ws-port` | none | WebSocket server port (requires `babelvox[ws]`) |
| `--rate` | `1.0` | Speech rate multiplier (0.25–4.0) |
| `--pitch` | `0` | Pitch shift in semitones (-12 to +12) |
| `--volume` | `1.0` | Volume multiplier (0.0–2.0) |
| `--emotion` | none | Emotion style: happy, sad, angry, surprised, neutral |
| `--longform` | off | Enable long-form synthesis mode |
| `--strategy` | `natural` | Segmentation: paragraph, sentence, natural, chapter |
| `--text-file` | none | Read text from file instead of `--text` |
| `--split-output` | off | Write individual segment files |
| `--timestamps` | off | Write timestamps JSON alongside audio |
| `--output` / `-o` | `output.wav` | Output WAV file path |
| `--export-dir` | auto-download | Directory with exported models (downloads from HuggingFace if not set) |
| `--model-path` | `Qwen/Qwen3-TTS-12Hz-0.6B-Base` | HuggingFace model (tokenizer) |

## Acknowledgments

Based on [Qwen3-TTS](https://github.com/Qwen/Qwen3-TTS) by Alibaba Qwen Team (Apache-2.0).

## License

Apache-2.0
