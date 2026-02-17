"""BabelVox: Real-time text-to-speech inference via OpenVINO.

Orchestrates 5 exported OpenVINO IR models:
  1. speaker_encoder   - mel spectrogram -> speaker embedding (1024-dim)
  2. talker            - input embeddings -> logits + hidden states
  3. code_predictor    - input embeddings -> hidden states (lm_heads in Python)
  4. tokenizer_encoder - raw audio -> audio codes (for voice cloning reference)
  5. tokenizer_decoder - audio codes -> waveform

Embedding lookups and projection layers use pre-exported numpy weights —
no PyTorch model loading required. The heavy transformer passes run in OpenVINO.

NPU mode: Models are reshaped to fixed dimensions before compilation.
Inputs are zero-padded to the fixed size; outputs are extracted at the
correct position (causal attention prevents real tokens from seeing padding).
"""
import json
import os
import time

import numpy as np
import librosa
from librosa.filters import mel as librosa_mel_fn
import openvino as ov
from scipy.special import expit as sigmoid
from transformers import AutoTokenizer

DEFAULT_HF_REPO = "djwarf/babelvox-openvino-int8"


def download_models(repo_id=DEFAULT_HF_REPO, local_dir=None):
    """Download pre-exported OpenVINO models from HuggingFace Hub.

    Args:
        repo_id: HuggingFace repo containing the exported models.
        local_dir: Where to save. If None, uses HF cache
                   (~/.cache/huggingface/hub/).

    Returns:
        str: Path to the downloaded model directory.
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise ImportError(
            "huggingface_hub is required for auto-download. "
            "Install it with: pip install huggingface_hub"
        )
    print(f"Downloading models from {repo_id}...")
    path = snapshot_download(repo_id=repo_id, local_dir=local_dir)
    print(f"Models downloaded to {path}")
    return path


# ============================================================
# Numpy helper functions
# ============================================================
def silu(x):
    """SiLU (Swish) activation: x * sigmoid(x)."""
    return x * sigmoid(x)


def text_projection_np(x, fc1_w, fc1_b, fc2_w, fc2_b):
    """2-layer MLP: Linear -> SiLU -> Linear. Matches Qwen3TTSTalkerResizeMLP."""
    h = x @ fc1_w.T + fc1_b
    h = silu(h)
    return h @ fc2_w.T + fc2_b


def softmax(x, axis=-1):
    """Numerically stable softmax."""
    e = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e / np.sum(e, axis=axis, keepdims=True)


def mel_spectrogram_np(audio, sr=24000, n_fft=1024, hop_length=256,
                       win_length=1024, n_mels=128, fmin=0, fmax=12000):
    """Compute mel spectrogram matching the PyTorch mel_spectrogram function.

    Pipeline: reflect-pad -> STFT -> magnitude -> mel filterbank -> log compress.
    """
    padding = (n_fft - hop_length) // 2
    audio_padded = np.pad(audio, (padding, padding), mode='reflect')

    stft = librosa.stft(audio_padded, n_fft=n_fft, hop_length=hop_length,
                        win_length=win_length, window='hann', center=False)
    spec = np.sqrt(np.abs(stft) ** 2 + 1e-9)

    mel_basis = librosa_mel_fn(sr=sr, n_fft=n_fft, n_mels=n_mels,
                               fmin=fmin, fmax=fmax)
    mel_spec = mel_basis @ spec
    mel_spec = np.log(np.clip(mel_spec, a_min=1e-5, a_max=None))
    return mel_spec  # (n_mels, T)


class BabelVox:
    """OpenVINO-accelerated text-to-speech inference pipeline.

    Uses pre-exported numpy weights for embeddings/projections — no PyTorch needed.
    Supports CPU (dynamic shapes) and NPU (fixed shapes with padding).
    """

    # CP KV cache constants (code predictor: 5 layers, 8 KV heads, head_dim 128)
    CP_NUM_LAYERS = 5
    CP_NUM_KV_HEADS = 8
    CP_HEAD_DIM = 128
    CP_MAX_KV_LEN = 20  # max 16 tokens in practice (2 prefill + 14 decode)

    def __init__(self, model_path="Qwen/Qwen3-TTS-12Hz-0.6B-Base",
                 export_dir=None, device="CPU",
                 max_talker_seq=256, max_cp_seq=17,
                 max_speaker_frames=300, max_decoder_frames=256,
                 use_kv_cache=False, max_kv_len=256,
                 use_cp_kv_cache=False,
                 talker_buckets=None,
                 precision="fp16",
                 cache_dir=None):
        self.device = device
        self.is_npu = (device == "NPU")
        self.use_kv_cache = use_kv_cache
        self.max_kv_len = max_kv_len
        self.use_cp_kv_cache = use_cp_kv_cache
        self.talker_bucket_sizes = sorted(talker_buckets) if talker_buckets else None
        self.max_talker_seq = max_talker_seq
        self.max_cp_seq = max_cp_seq
        self.max_speaker_frames = max_speaker_frames
        self.max_decoder_frames = max_decoder_frames
        self.precision = precision
        self.default_speaker = None  # set to a (1, 1024) numpy array for voice persistence

        # Auto-download models if no export_dir provided or dir doesn't exist
        if export_dir is None:
            export_dir = download_models()
        elif not os.path.isdir(export_dir):
            print(f"Export dir '{export_dir}' not found, downloading from HuggingFace...")
            export_dir = download_models(local_dir=export_dir)

        # HF repo layout: int8/ and weights/ at top level
        # Local export layout: export_dir/int8/, export_dir/weights/
        if precision in ("int8", "int4", "fp16"):
            model_dir = os.path.join(export_dir, precision)
        else:
            model_dir = export_dir

        print(f"Loading OpenVINO models on {device} (precision={precision})...")
        if self.is_npu:
            if use_kv_cache:
                print(f"  NPU KV cache mode: max_kv_len={max_kv_len}, "
                      f"speaker_frames={max_speaker_frames}, "
                      f"decoder_frames={max_decoder_frames}")
            else:
                print(f"  NPU fixed shapes: talker_seq={max_talker_seq}, "
                      f"cp_seq={max_cp_seq}, speaker_frames={max_speaker_frames}, "
                      f"decoder_frames={max_decoder_frames}")

        core = ov.Core()
        if cache_dir is not None:
            core.set_property({"CACHE_DIR": cache_dir})
            print(f"  OpenVINO model cache: {cache_dir}")
        t0 = time.time()

        self.cp_on_cpu = False
        if self.is_npu:
            self._load_npu_models(core, model_dir)
        else:
            self.speaker_enc = core.compile_model(
                os.path.join(model_dir, "speaker_encoder.xml"), device)
            if self.use_kv_cache:
                self.talker_prefill = core.compile_model(
                    os.path.join(model_dir, "talker_prefill.xml"), device)
                self.talker_decode = core.compile_model(
                    os.path.join(model_dir, "talker_decode.xml"), device)
            else:
                self.talker = core.compile_model(
                    os.path.join(model_dir, "talker.xml"), device)
            if self.use_cp_kv_cache:
                self.cp_prefill = core.compile_model(
                    os.path.join(model_dir, "cp_prefill.xml"), device)
                self.cp_decode = core.compile_model(
                    os.path.join(model_dir, "cp_decode.xml"), device)
            else:
                self.code_predictor = core.compile_model(
                    os.path.join(model_dir, "code_predictor.xml"), device)
            self.tok_decoder = core.compile_model(
                os.path.join(model_dir, "tokenizer_decoder.xml"), device)

        print(f"  OpenVINO models compiled in {time.time() - t0:.1f}s")

        # Load numpy weights (embeddings + projections)
        print("Loading numpy weights...")
        t0 = time.time()
        self._load_numpy_weights(os.path.join(export_dir, "weights"))

        # Load tokenizer (lightweight — no model needed)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        print(f"  Weights + tokenizer loaded in {time.time() - t0:.1f}s")

    def _load_numpy_weights(self, weights_dir):
        """Load pre-exported embedding tables and projection weights."""
        with open(os.path.join(weights_dir, "config.json")) as f:
            cfg = json.load(f)

        self.hidden_size = cfg["hidden_size"]
        self.num_code_groups = cfg["num_code_groups"]
        self.codec_eos_id = cfg["codec_eos_token_id"]
        self.codec_think_id = cfg["codec_think_id"]
        self.codec_nothink_id = cfg["codec_nothink_id"]
        self.codec_think_bos_id = cfg["codec_think_bos_id"]
        self.codec_think_eos_id = cfg["codec_think_eos_id"]
        self.codec_pad_id = cfg["codec_pad_id"]
        self.codec_bos_id = cfg["codec_bos_id"]
        self.tts_bos_id = cfg["tts_bos_token_id"]
        self.tts_eos_id = cfg["tts_eos_token_id"]
        self.tts_pad_id = cfg["tts_pad_token_id"]
        self.codec_language_id = cfg["codec_language_id"]

        # Embeddings
        self.codec_emb = np.load(os.path.join(weights_dir, "codec_embedding.npy"))
        text_emb_f16 = np.load(os.path.join(weights_dir, "text_embedding.npy"))
        self.text_emb = text_emb_f16.astype(np.float32)

        # Text projection MLP weights (fc1 -> SiLU -> fc2)
        self.tp_fc1_w = np.load(os.path.join(weights_dir, "text_proj_fc1_weight.npy"))
        self.tp_fc1_b = np.load(os.path.join(weights_dir, "text_proj_fc1_bias.npy"))
        self.tp_fc2_w = np.load(os.path.join(weights_dir, "text_proj_fc2_weight.npy"))
        self.tp_fc2_b = np.load(os.path.join(weights_dir, "text_proj_fc2_bias.npy"))

        # Code predictor embeddings and lm_heads
        self.cp_embs = [
            np.load(os.path.join(weights_dir, f"cp_embedding_{i}.npy"))
            for i in range(self.num_code_groups - 1)
        ]
        self.cp_heads = [
            np.load(os.path.join(weights_dir, f"cp_lm_head_{i}_weight.npy"))
            for i in range(self.num_code_groups - 1)
        ]

    def _load_npu_models(self, core, export_dir):
        """Read, reshape to fixed dims, and compile each model for NPU."""
        t1 = time.time()
        m = core.read_model(os.path.join(export_dir, "speaker_encoder.xml"))
        m.reshape({"mel_spectrogram": [1, self.max_speaker_frames, 128]})
        self.speaker_enc = core.compile_model(m, "NPU")
        print(f"    Speaker encoder: {time.time()-t1:.1f}s (NPU)")

        if self.use_kv_cache:
            t1 = time.time()
            self.talker_prefill = core.compile_model(
                os.path.join(export_dir, "talker_prefill.xml"), "CPU")
            print(f"    Talker prefill:  {time.time()-t1:.1f}s (CPU)")

            t1 = time.time()
            m = core.read_model(os.path.join(export_dir, "talker_decode.xml"))
            m.reshape({
                "inputs_embeds": [1, 1, 1024],
                "position_ids": [3, 1, 1],
                "cache_position": [1],
                "attention_mask": [1, 1, 1, self.max_kv_len],
                "past_keys": [28, 1, 8, self.max_kv_len, 128],
                "past_values": [28, 1, 8, self.max_kv_len, 128],
            })
            self.talker_decode = core.compile_model(m, "NPU")
            print(f"    Talker decode:   {time.time()-t1:.1f}s (NPU, kv_len={self.max_kv_len})")
        else:
            talker_xml = os.path.join(export_dir, "talker.xml")
            if self.talker_bucket_sizes:
                # Multi-bucket: compile at each size for dynamic shape selection
                self.talker_buckets = {}
                for bsize in self.talker_bucket_sizes:
                    t1 = time.time()
                    m = core.read_model(talker_xml)
                    m.reshape({
                        "inputs_embeds": [1, bsize, 1024],
                        "position_ids": [3, 1, bsize],
                    })
                    self.talker_buckets[bsize] = core.compile_model(m, "NPU")
                    print(f"    Talker @{bsize:3d}:     {time.time()-t1:.1f}s (NPU)")
                self.max_talker_seq = max(self.talker_bucket_sizes)
            else:
                t1 = time.time()
                m = core.read_model(talker_xml)
                m.reshape({
                    "inputs_embeds": [1, self.max_talker_seq, 1024],
                    "position_ids": [3, 1, self.max_talker_seq],
                })
                self.talker = core.compile_model(m, "NPU")
                print(f"    Talker (28L):    {time.time()-t1:.1f}s (NPU)")

        t1 = time.time()
        if self.use_cp_kv_cache:
            self.cp_prefill = core.compile_model(
                os.path.join(export_dir, "cp_prefill.xml"), "CPU")
            self.cp_decode = core.compile_model(
                os.path.join(export_dir, "cp_decode.xml"), "CPU")
            print(f"    CP prefill+decode: {time.time()-t1:.1f}s (CPU - KV cache)")
        else:
            self.code_predictor = core.compile_model(
                os.path.join(export_dir, "code_predictor.xml"), "CPU")
            print(f"    Code predictor:  {time.time()-t1:.1f}s (CPU - hybrid)")
        self.cp_on_cpu = True

        t1 = time.time()
        m = core.read_model(os.path.join(export_dir, "tokenizer_decoder.xml"))
        m.reshape({"codes": [1, 16, self.max_decoder_frames]})
        self.tok_decoder = core.compile_model(m, "NPU")
        print(f"    Tok decoder:     {time.time()-t1:.1f}s (NPU)")

    # ----------------------------------------------------------
    # Numpy embedding/projection helpers
    # ----------------------------------------------------------
    def _text_proj(self, x):
        """Text projection MLP on numpy array: fc1 -> SiLU -> fc2."""
        return text_projection_np(x, self.tp_fc1_w, self.tp_fc1_b,
                                  self.tp_fc2_w, self.tp_fc2_b)

    def _codec_embed(self, ids):
        """Codec embedding lookup. ids: flat list or 1D array."""
        return self.codec_emb[ids]  # (len, 1024)

    def _text_embed(self, ids):
        """Text embedding lookup. ids: flat list or 1D array."""
        return self.text_emb[ids]  # (len, 2048)

    # ----------------------------------------------------------
    # Speaker embedding
    # ----------------------------------------------------------
    def extract_speaker_embedding(self, audio_path):
        """Extract speaker embedding from reference audio using OpenVINO."""
        audio, sr = librosa.load(audio_path, sr=24000)
        mel = mel_spectrogram_np(audio)  # (128, T)
        mel_np = mel.T[np.newaxis, :, :].astype(np.float32)  # (1, T, 128)

        if self.is_npu:
            mel_frames = mel_np.shape[1]
            padded = np.zeros((1, self.max_speaker_frames, 128), dtype=np.float32)
            n = min(mel_frames, self.max_speaker_frames)
            padded[:, :n, :] = mel_np[:, :n, :]
            result = self.speaker_enc({"mel_spectrogram": padded})
        else:
            result = self.speaker_enc({"mel_spectrogram": mel_np})

        return result[0]  # numpy (1, 1024)

    # ----------------------------------------------------------
    # Prefill embedding construction
    # ----------------------------------------------------------
    def build_prefill_embeds(self, text, language, speaker_embed,
                             ref_text=None):
        """Build the input embedding sequence for the talker prefill.

        All operations are pure numpy. Returns (1, T, 1024) arrays.
        When ref_text is provided (transcription of reference audio),
        it is prepended to the sequence for better voice cloning.
        """
        formatted = f"<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n"
        tokens = self.tokenizer(formatted, return_tensors="np", padding=True)
        input_ids = tokens["input_ids"][0]  # 1D array

        # Reference text for voice cloning (prepended to sequence)
        ref_text_embed = None
        if ref_text is not None:
            ref_formatted = f"<|im_start|>assistant\n{ref_text}<|im_end|>\n"
            ref_tokens = self.tokenizer(ref_formatted, return_tensors="np",
                                        padding=True)
            ref_ids = ref_tokens["input_ids"][0]
            ref_text_embed = self._text_proj(self._text_embed(ref_ids))

        # Language ID
        lang_key = language.lower()
        language_id = self.codec_language_id.get(lang_key)

        # Special text embeddings -> project through MLP
        special_ids = [self.tts_bos_id, self.tts_eos_id, self.tts_pad_id]
        special_embeds = self._text_proj(self._text_embed(special_ids))  # (3, 1024)
        tts_bos_embed = special_embeds[0:1]   # (1, 1024)
        tts_eos_embed = special_embeds[1:2]
        tts_pad_embed = special_embeds[2:3]

        # Codec prefill tokens
        if language_id is not None:
            codec_prefill = [self.codec_think_id, self.codec_think_bos_id,
                             language_id, self.codec_think_eos_id]
        else:
            codec_prefill = [self.codec_nothink_id, self.codec_think_bos_id,
                             self.codec_think_eos_id]

        codec_embed_0 = self._codec_embed(codec_prefill)  # (N, 1024)
        codec_embed_1 = self._codec_embed([self.codec_pad_id, self.codec_bos_id])  # (2, 1024)

        if speaker_embed is not None:
            spk = speaker_embed.reshape(1, -1)  # (1, 1024)
            codec_input = np.concatenate([codec_embed_0, spk, codec_embed_1], axis=0)
        else:
            codec_input = np.concatenate([codec_embed_0, codec_embed_1], axis=0)
        # codec_input: (C, 1024) where C = len(prefill) + [1 spk] + 2

        # Role tokens: first 3 tokens (<|im_start|>assistant\n)
        role_embed = self._text_proj(self._text_embed(input_ids[:3]))  # (3, 1024)

        # Pad + BOS overlay on codec prefill
        n_overlay = codec_input.shape[0] - 2
        pad_expanded = np.tile(tts_pad_embed, (n_overlay, 1))  # (n_overlay, 1024)
        text_overlay = np.concatenate([pad_expanded, tts_bos_embed], axis=0)
        codec_with_text = text_overlay + codec_input[:-1]  # (C-1, 1024)

        # Build talker sequence: [ref_text] [role] [codec+text overlay]
        parts = []
        if ref_text_embed is not None:
            parts.append(ref_text_embed)
        parts.append(role_embed)
        parts.append(codec_with_text)
        talker_embed = np.concatenate(parts, axis=0)

        # First text token + last codec embedding
        first_text = self._text_proj(self._text_embed(input_ids[3:4]))  # (1, 1024)
        talker_embed = np.concatenate([talker_embed,
                                       first_text + codec_input[-1:]], axis=0)

        # Trailing text
        trailing_text = self._text_proj(self._text_embed(input_ids[4:-5]))
        trailing = np.concatenate([trailing_text, tts_eos_embed], axis=0)

        # Add batch dimension: (T, 1024) -> (1, T, 1024)
        return (talker_embed[np.newaxis],
                trailing[np.newaxis],
                tts_pad_embed[np.newaxis])  # (1, 1, 1024)

    # ----------------------------------------------------------
    # OpenVINO inference helpers
    # ----------------------------------------------------------
    def _select_bucket(self, real_len):
        """Pick the smallest compiled bucket that fits real_len."""
        for bsize in self.talker_bucket_sizes:
            if bsize >= real_len:
                return bsize, self.talker_buckets[bsize]
        # Fallback: use largest bucket
        bsize = self.talker_bucket_sizes[-1]
        return bsize, self.talker_buckets[bsize]

    def _run_talker(self, seq_embeds, real_len):
        """Run talker forward pass, padding for NPU if needed."""
        pos_ids = np.arange(real_len, dtype=np.int64).reshape(1, 1, -1)
        pos_ids = np.broadcast_to(pos_ids, (3, 1, real_len)).copy()

        if self.is_npu:
            if self.talker_bucket_sizes:
                pad_len, model = self._select_bucket(real_len)
            else:
                pad_len, model = self.max_talker_seq, self.talker
            padded_e = np.zeros((1, pad_len, self.hidden_size), dtype=np.float32)
            padded_e[:, :real_len, :] = seq_embeds
            padded_p = np.zeros((3, 1, pad_len), dtype=np.int64)
            padded_p[:, :, :real_len] = pos_ids
            out = model({"inputs_embeds": padded_e, "position_ids": padded_p})
        else:
            out = self.talker({"inputs_embeds": seq_embeds, "position_ids": pos_ids})

        logits = out["logits"][:, real_len-1:real_len, :]
        hidden = out["hidden_states"][:, real_len-1:real_len, :]
        return logits, hidden

    def _run_code_predictor(self, cp_input, real_len):
        """Run code predictor forward pass (always CPU)."""
        out = self.code_predictor({"inputs_embeds": cp_input})
        return out["hidden_states"][:, real_len-1, :]  # (1, hidden)

    def _run_cp_prefill(self, cp_input):
        """Run CP prefill (2 tokens). Returns last hidden + padded KV cache."""
        result = self.cp_prefill({"inputs_embeds": cp_input})
        hidden = result["hidden_states"][:, -1, :]  # (1, 1024)

        raw_keys = result["present_keys"]    # (5, 1, 8, 2, 128)
        raw_values = result["present_values"]
        prefill_len = cp_input.shape[1]

        kv_k = np.zeros((self.CP_NUM_LAYERS, 1, self.CP_NUM_KV_HEADS,
                          self.CP_MAX_KV_LEN, self.CP_HEAD_DIM), dtype=np.float32)
        kv_v = np.zeros_like(kv_k)
        kv_k[:, :, :, :prefill_len, :] = raw_keys
        kv_v[:, :, :, :prefill_len, :] = raw_values
        return hidden, kv_k, kv_v, prefill_len

    def _run_cp_decode(self, token_embed, pos, kv_k, kv_v):
        """Run CP decode for 1 token. Returns hidden + updated KV."""
        cache_pos = np.array([pos], dtype=np.int64)
        attn_mask = np.full((1, 1, 1, self.CP_MAX_KV_LEN),
                            np.finfo(np.float32).min, dtype=np.float32)
        attn_mask[:, :, :, :pos + 1] = 0.0

        result = self.cp_decode({
            "inputs_embeds": token_embed,
            "cache_position": cache_pos,
            "attention_mask": attn_mask,
            "past_keys": kv_k,
            "past_values": kv_v,
        })
        return (result["hidden_states"][:, 0, :],  # (1, 1024)
                result["present_keys"], result["present_values"])

    def _run_talker_prefill(self, prefill_embeds, prefill_len):
        """Run talker prefill model once. Returns logits, hidden, padded KV cache."""
        pos_ids = np.arange(prefill_len, dtype=np.int64).reshape(1, 1, -1)
        pos_ids = np.broadcast_to(pos_ids, (3, 1, prefill_len)).copy()

        result = self.talker_prefill({
            "inputs_embeds": prefill_embeds,
            "position_ids": pos_ids,
        })
        logits = result["logits"][:, -1:, :]
        hidden = result["hidden_states"][:, -1:, :]

        raw_keys = result["present_keys"]
        raw_values = result["present_values"]
        kv_k = np.zeros((28, 1, 8, self.max_kv_len, 128), dtype=np.float32)
        kv_v = np.zeros((28, 1, 8, self.max_kv_len, 128), dtype=np.float32)
        kv_k[:, :, :, :prefill_len, :] = raw_keys
        kv_v[:, :, :, :prefill_len, :] = raw_values
        return logits, hidden, kv_k, kv_v

    def _run_talker_decode(self, token_embed, pos, kv_k, kv_v):
        """Run talker decode for 1 new token. Returns logits, hidden, updated KV."""
        pos_ids = np.array([[[pos]]] * 3, dtype=np.int64)
        cache_pos = np.array([pos], dtype=np.int64)
        attn_mask = np.full((1, 1, 1, self.max_kv_len),
                            np.finfo(np.float32).min, dtype=np.float32)
        attn_mask[:, :, :, :pos + 1] = 0.0

        result = self.talker_decode({
            "inputs_embeds": token_embed,
            "position_ids": pos_ids,
            "cache_position": cache_pos,
            "attention_mask": attn_mask,
            "past_keys": kv_k,
            "past_values": kv_v,
        })
        return (result["logits"], result["hidden_states"],
                result["present_keys"], result["present_values"])

    def _decode_audio(self, codes_array):
        """Decode audio codes to waveform, padding for NPU if needed."""
        num_frames = codes_array.shape[2]

        if self.is_npu:
            padded = np.zeros((1, 16, self.max_decoder_frames), dtype=np.int64)
            padded[:, :, :num_frames] = codes_array
            wav_out = self.tok_decoder({"codes": padded})
            expected_samples = num_frames * 1920
            wav = wav_out[0].squeeze()[:expected_samples]
        else:
            wav_out = self.tok_decoder({"codes": codes_array})
            wav = wav_out[0].squeeze()

        return np.clip(wav, -1.0, 1.0)

    # ----------------------------------------------------------
    # Repetition penalty
    # ----------------------------------------------------------
    def _apply_repetition_penalty(self, logits, generated_ids, penalty):
        """Apply repetition penalty to discourage repeated codec tokens."""
        if penalty <= 1.0 or len(generated_ids) == 0:
            return logits
        recent = set(generated_ids[-64:])
        for token_id in recent:
            if logits[token_id] > 0:
                logits[token_id] = logits[token_id] / penalty
            else:
                logits[token_id] = logits[token_id] * penalty
        return logits

    # ----------------------------------------------------------
    # Sampling (pure numpy)
    # ----------------------------------------------------------
    def _sample_token(self, logits, temperature=1.0, top_k=50, top_p=1.0):
        """Sample a token from logits with temperature, top-k, and top-p."""
        logits = np.array(logits, dtype=np.float64)

        # Defensive: clamp NaN/inf
        if np.any(np.isnan(logits)) or np.any(np.isinf(logits)):
            logits = np.nan_to_num(logits, nan=0.0, posinf=1e6, neginf=-1e6)

        if temperature <= 0:
            return int(np.argmax(logits))

        logits = logits / temperature

        if top_k > 0:
            top_k = min(top_k, len(logits))
            kth_val = np.partition(logits, -top_k)[-top_k]
            logits[logits < kth_val] = -np.inf

        if top_p < 1.0:
            sorted_idx = np.argsort(logits)[::-1]
            sorted_logits = logits[sorted_idx]
            probs_sorted = softmax(sorted_logits)
            cumsum = np.cumsum(probs_sorted)
            cutoff = np.searchsorted(cumsum, top_p) + 1
            sorted_logits[cutoff:] = -np.inf
            logits[sorted_idx] = sorted_logits

        probs = softmax(logits)
        return int(np.random.choice(len(probs), p=probs))

    # ----------------------------------------------------------
    # Streaming generation
    # ----------------------------------------------------------
    def generate_stream(self, text, language="English", ref_audio=None,
                        ref_text=None, speaker_embed=None,
                        max_new_tokens=512, temperature=0.9, top_k=50,
                        top_p=1.0, repetition_penalty=1.05,
                        subtalker_temperature=0.9, subtalker_top_k=50,
                        subtalker_top_p=1.0, chunk_frames=12):
        """Generate speech as a stream of waveform chunks.

        Yields (waveform_chunk, sample_rate) tuples every chunk_frames
        codec frames (~1 sec per 12 frames). Start playback on the first
        chunk while generation continues in the background.

        Args:
            chunk_frames: Number of codec frames per chunk (12 = 1 sec).
            (All other args identical to generate().)

        Yields:
            tuple: (waveform_chunk, 24000) for each chunk of audio.
        """
        # --- Setup (same as generate) ---
        if self.use_kv_cache:
            kv_limit = self.max_kv_len - 20
            decoder_limit = self.max_decoder_frames
            effective_max = min(max_new_tokens, kv_limit, decoder_limit)
            if effective_max < max_new_tokens:
                max_new_tokens = effective_max
        elif self.is_npu:
            talker_limit = self.max_talker_seq - 20
            decoder_limit = self.max_decoder_frames
            effective_max = min(max_new_tokens, talker_limit, decoder_limit)
            if effective_max < max_new_tokens:
                max_new_tokens = effective_max

        if ref_audio is not None:
            speaker_embed = self.extract_speaker_embedding(ref_audio)
        elif speaker_embed is None and self.default_speaker is not None:
            speaker_embed = self.default_speaker

        prefill_embeds, trailing_text, tts_pad_embed = self.build_prefill_embeds(
            text, language, speaker_embed, ref_text=ref_text)
        prefill_len = prefill_embeds.shape[1]

        if self.use_kv_cache:
            if prefill_len + max_new_tokens > self.max_kv_len:
                max_new_tokens = self.max_kv_len - prefill_len
        elif self.is_npu:
            if prefill_len + max_new_tokens > self.max_talker_seq:
                max_new_tokens = self.max_talker_seq - prefill_len

        all_codes = []
        generated_ids = []
        chunk_buffer = []

        if self.use_kv_cache:
            logits, hidden, kv_k, kv_v = self._run_talker_prefill(
                prefill_embeds, prefill_len)
            current_pos = prefill_len
        else:
            seq_embeds = prefill_embeds

        # --- Generation loop with streaming decode ---
        for step in range(max_new_tokens):
            if self.use_kv_cache:
                if step > 0:
                    logits, hidden, kv_k, kv_v = self._run_talker_decode(
                        combined, current_pos, kv_k, kv_v)
                    current_pos += 1
                if current_pos >= self.max_kv_len:
                    break
            else:
                real_len = seq_embeds.shape[1]
                if self.is_npu and real_len > self.max_talker_seq:
                    break
                logits, hidden = self._run_talker(seq_embeds, real_len)

            raw_logits = logits[0, 0, :] if logits.ndim == 3 else logits.squeeze()
            raw_logits = self._apply_repetition_penalty(
                raw_logits.copy(), generated_ids, repetition_penalty)
            code_0 = self._sample_token(
                raw_logits, temperature=temperature, top_k=top_k, top_p=top_p)

            if code_0 == self.codec_eos_id:
                break

            generated_ids.append(code_0)

            # Code predictor
            code_groups = [code_0]
            code_0_embed = self.codec_emb[[code_0]][np.newaxis, :, :]
            cp_input = np.concatenate([hidden, code_0_embed], axis=1)

            if self.use_cp_kv_cache:
                cp_hidden, cp_kv_k, cp_kv_v, cp_pos = self._run_cp_prefill(cp_input)
                cp_logits = cp_hidden @ self.cp_heads[0].T
                cp_code = self._sample_token(
                    cp_logits[0], temperature=subtalker_temperature,
                    top_k=subtalker_top_k, top_p=subtalker_top_p)
                code_groups.append(cp_code)
                for cp_step in range(1, self.num_code_groups - 1):
                    next_embed = self.cp_embs[cp_step - 1][[cp_code]][np.newaxis, :, :]
                    cp_hidden, cp_kv_k, cp_kv_v = self._run_cp_decode(
                        next_embed, cp_pos, cp_kv_k, cp_kv_v)
                    cp_pos += 1
                    cp_logits = cp_hidden @ self.cp_heads[cp_step].T
                    cp_code = self._sample_token(
                        cp_logits[0], temperature=subtalker_temperature,
                        top_k=subtalker_top_k, top_p=subtalker_top_p)
                    code_groups.append(cp_code)
            else:
                for cp_step in range(self.num_code_groups - 1):
                    cp_len = cp_input.shape[1]
                    cp_hidden = self._run_code_predictor(cp_input, cp_len)
                    cp_logits = cp_hidden @ self.cp_heads[cp_step].T
                    cp_code = self._sample_token(
                        cp_logits[0], temperature=subtalker_temperature,
                        top_k=subtalker_top_k, top_p=subtalker_top_p)
                    code_groups.append(cp_code)
                    if cp_step < self.num_code_groups - 2:
                        next_embed = self.cp_embs[cp_step][[cp_code]][np.newaxis, :, :]
                        cp_input = np.concatenate([cp_input, next_embed], axis=1)

            all_codes.append(code_groups)
            chunk_buffer.append(code_groups)

            # Yield a chunk when buffer is full
            if len(chunk_buffer) >= chunk_frames:
                codes = np.array(chunk_buffer, dtype=np.int64).T[np.newaxis, :, :]
                yield self._decode_audio(codes), 24000
                chunk_buffer = []

            # Build next talker input
            combined = self.codec_emb[[code_groups[0]]]
            for i in range(1, self.num_code_groups):
                combined = combined + self.cp_embs[i - 1][[code_groups[i]]]
            combined = combined[np.newaxis, :, :]

            if step < trailing_text.shape[1]:
                combined = combined + trailing_text[:, step:step+1]
            else:
                combined = combined + tts_pad_embed

            if not self.use_kv_cache:
                seq_embeds = np.concatenate([seq_embeds, combined], axis=1)

        # Flush remaining frames
        if chunk_buffer:
            codes = np.array(chunk_buffer, dtype=np.int64).T[np.newaxis, :, :]
            wav = self._decode_audio(codes)
            hit_eos = (len(all_codes) < max_new_tokens)
            if not hit_eos:
                fade_samples = min(2400, len(wav))
                fade = np.linspace(1.0, 0.0, fade_samples, dtype=np.float32)
                wav[-fade_samples:] *= fade
            yield wav, 24000

    # ----------------------------------------------------------
    # Main generation loop
    # ----------------------------------------------------------
    def generate(self, text, language="English", ref_audio=None,
                 ref_text=None, speaker_embed=None,
                 max_new_tokens=512, temperature=0.9, top_k=50, top_p=1.0,
                 repetition_penalty=1.05, subtalker_temperature=0.9,
                 subtalker_top_k=50, subtalker_top_p=1.0):
        """Generate speech from text, optionally cloning a reference voice.

        Voice can be specified via ref_audio (path), speaker_embed (numpy
        array from extract_speaker_embedding()), or self.default_speaker.
        When using ref_audio, pass ref_text (the transcription of the
        reference audio) for better voice cloning quality.

        Returns:
            tuple: (waveform_array, sample_rate) where waveform is float32
                   numpy array and sample_rate is 24000.
        """

        # Clamp max_new_tokens for device limits
        if self.use_kv_cache:
            kv_limit = self.max_kv_len - 20
            decoder_limit = self.max_decoder_frames
            effective_max = min(max_new_tokens, kv_limit, decoder_limit)
            if effective_max < max_new_tokens:
                print(f"  Clamped max_new_tokens: {max_new_tokens} -> {effective_max} (KV/decoder limits)")
                max_new_tokens = effective_max
        elif self.is_npu:
            talker_limit = self.max_talker_seq - 20
            decoder_limit = self.max_decoder_frames
            effective_max = min(max_new_tokens, talker_limit, decoder_limit)
            if effective_max < max_new_tokens:
                print(f"  Clamped max_new_tokens: {max_new_tokens} -> {effective_max} (NPU limits)")
                max_new_tokens = effective_max

        # Speaker embedding: ref_audio > explicit speaker_embed > default_speaker
        if ref_audio is not None:
            print("Extracting speaker embedding...")
            speaker_embed = self.extract_speaker_embedding(ref_audio)
        elif speaker_embed is None and self.default_speaker is not None:
            speaker_embed = self.default_speaker

        # Build prefill
        print("Building prefill embeddings...")
        prefill_embeds, trailing_text, tts_pad_embed = self.build_prefill_embeds(
            text, language, speaker_embed, ref_text=ref_text)

        prefill_len = prefill_embeds.shape[1]
        print(f"Prefill: {prefill_len} tokens, trailing text: {trailing_text.shape[1]} tokens")

        if self.use_kv_cache:
            if prefill_len + max_new_tokens > self.max_kv_len:
                max_new_tokens = self.max_kv_len - prefill_len
                print(f"  Adjusted max_new_tokens to {max_new_tokens} for KV cache fit")
        elif self.is_npu:
            if prefill_len + max_new_tokens > self.max_talker_seq:
                max_new_tokens = self.max_talker_seq - prefill_len
                print(f"  Adjusted max_new_tokens to {max_new_tokens} for prefill+gen fit")

        # Generation setup
        all_codes = []
        generated_ids = []

        if self.use_kv_cache:
            print(f"Running talker prefill ({prefill_len} tokens on CPU)...")
            t_pf = time.time()
            logits, hidden, kv_k, kv_v = self._run_talker_prefill(
                prefill_embeds, prefill_len)
            print(f"  Prefill done in {time.time() - t_pf:.2f}s")
            current_pos = prefill_len
        else:
            seq_embeds = prefill_embeds

        mode_str = "KV-cache" if self.use_kv_cache else "full-recompute"
        print(f"Generating (max {max_new_tokens} steps, {mode_str}, device={self.device})...")
        t0 = time.time()

        for step in range(max_new_tokens):
            # --- Get logits/hidden for this step ---
            if self.use_kv_cache:
                if step > 0:
                    logits, hidden, kv_k, kv_v = self._run_talker_decode(
                        combined, current_pos, kv_k, kv_v)
                    current_pos += 1
                if current_pos >= self.max_kv_len:
                    print(f"  Reached KV cache limit at step {step}")
                    break
            else:
                real_len = seq_embeds.shape[1]
                if self.is_npu and real_len > self.max_talker_seq:
                    print(f"  Reached NPU sequence limit at step {step}")
                    break
                logits, hidden = self._run_talker(seq_embeds, real_len)

            # --- Sample code_0 ---
            raw_logits = logits[0, 0, :] if logits.ndim == 3 else logits.squeeze()
            raw_logits = self._apply_repetition_penalty(
                raw_logits.copy(), generated_ids, repetition_penalty)

            code_0 = self._sample_token(
                raw_logits, temperature=temperature, top_k=top_k, top_p=top_p)

            if code_0 == self.codec_eos_id:
                print(f"  EOS at step {step}")
                break

            generated_ids.append(code_0)

            # --- Code Predictor: generate code_groups 1..15 ---
            code_groups = [code_0]
            code_0_embed = self.codec_emb[[code_0]][np.newaxis, :, :]  # (1,1,1024)
            cp_input = np.concatenate([hidden, code_0_embed], axis=1)  # (1,2,1024)

            if self.use_cp_kv_cache:
                # KV-cached CP: prefill 2 tokens, then 14 single-token decodes
                cp_hidden, cp_kv_k, cp_kv_v, cp_pos = self._run_cp_prefill(cp_input)

                cp_logits = cp_hidden @ self.cp_heads[0].T
                cp_code = self._sample_token(
                    cp_logits[0], temperature=subtalker_temperature,
                    top_k=subtalker_top_k, top_p=subtalker_top_p)
                code_groups.append(cp_code)

                for cp_step in range(1, self.num_code_groups - 1):
                    next_embed = self.cp_embs[cp_step - 1][[cp_code]]
                    next_embed = next_embed[np.newaxis, :, :]  # (1,1,1024)
                    cp_hidden, cp_kv_k, cp_kv_v = self._run_cp_decode(
                        next_embed, cp_pos, cp_kv_k, cp_kv_v)
                    cp_pos += 1

                    cp_logits = cp_hidden @ self.cp_heads[cp_step].T
                    cp_code = self._sample_token(
                        cp_logits[0], temperature=subtalker_temperature,
                        top_k=subtalker_top_k, top_p=subtalker_top_p)
                    code_groups.append(cp_code)
            else:
                # Full-recompute CP: 15 passes with growing sequence
                for cp_step in range(self.num_code_groups - 1):
                    cp_len = cp_input.shape[1]
                    cp_hidden = self._run_code_predictor(cp_input, cp_len)

                    cp_logits = cp_hidden @ self.cp_heads[cp_step].T
                    cp_code = self._sample_token(
                        cp_logits[0], temperature=subtalker_temperature,
                        top_k=subtalker_top_k, top_p=subtalker_top_p)
                    code_groups.append(cp_code)

                    if cp_step < self.num_code_groups - 2:
                        next_embed = self.cp_embs[cp_step][[cp_code]]
                        next_embed = next_embed[np.newaxis, :, :]
                        cp_input = np.concatenate([cp_input, next_embed], axis=1)

            all_codes.append(code_groups)

            # --- Build next talker input ---
            combined = self.codec_emb[[code_groups[0]]]  # (1, 1024)
            for i in range(1, self.num_code_groups):
                combined = combined + self.cp_embs[i - 1][[code_groups[i]]]
            combined = combined[np.newaxis, :, :]  # (1, 1, 1024)

            if step < trailing_text.shape[1]:
                combined = combined + trailing_text[:, step:step+1]
            else:
                combined = combined + tts_pad_embed

            if not self.use_kv_cache:
                seq_embeds = np.concatenate([seq_embeds, combined], axis=1)

            if (step + 1) % 50 == 0:
                elapsed = time.time() - t0
                print(f"  Step {step+1}/{max_new_tokens} ({elapsed:.1f}s)")

        elapsed = time.time() - t0
        n_frames = len(all_codes)
        hit_eos = (n_frames < max_new_tokens)
        print(f"Generated {n_frames} codec frames in {elapsed:.1f}s"
              f"{'' if hit_eos else ' (no EOS, hit ceiling)'}")
        if n_frames > 0:
            audio_secs = n_frames / 12.0
            print(f"  {audio_secs:.1f}s audio, RTF={elapsed/audio_secs:.1f}x")

        if n_frames == 0:
            print("WARNING: No audio generated")
            return np.zeros(1, dtype=np.float32), 24000

        # --- Decode audio codes -> waveform ---
        print("Decoding audio codes to waveform...")
        codes = np.array(all_codes, dtype=np.int64).T[np.newaxis, :, :]  # (1, 16, T)
        wav = self._decode_audio(codes)

        # Fade out to avoid garbled tail when generation hit the token ceiling
        if not hit_eos:
            fade_samples = min(2400, len(wav))  # 0.1s at 24kHz
            fade = np.linspace(1.0, 0.0, fade_samples, dtype=np.float32)
            wav[-fade_samples:] *= fade

        return wav, 24000
