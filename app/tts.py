import contextlib
import io
import os
import re
import json
import base64
import logging
from urllib.parse import urlparse
import urllib.parse
import threading
import shutil
import time
import tempfile
import platform as py_platform
import numpy as np
import soundfile as sf
from pydub import AudioSegment
import httpx
from runtime_layout import LAYOUT
from audio_validation import estimate_expected_duration_seconds
from model_downloads import ensure_hf_snapshot

DEFAULT_PAUSE_MS = 500  # Pause between different speakers
SAME_SPEAKER_PAUSE_MS = 250  # Shorter pause for same speaker continuing
TRUE_VALUES = {"1", "true", "yes", "on"}
QWEN_AUDIO_TOKENS_PER_SECOND = 12.0
QWEN_MIN_GENERATION_SECONDS = 1.75
QWEN_MAX_GENERATION_SECONDS = 45.0
QWEN_GENERATION_SECONDS_MARGIN = 0.0
QWEN_GENERATION_TOKEN_BUFFER_FACTOR = 1.0
QWEN_MIN_MAX_NEW_TOKENS = 24
QWEN_MAX_MAX_NEW_TOKENS = 768
VOXCPM2_CFG_VALUE_DEFAULT = 1.6
VOXCPM2_CFG_VALUE_MIN = 1.0
VOXCPM2_CFG_VALUE_MAX = 3.0
VOXCPM2_INFERENCE_TIMESTEPS_DEFAULT = 10
VOXCPM2_INFERENCE_TIMESTEPS_MIN = 4
VOXCPM2_INFERENCE_TIMESTEPS_MAX = 30


def sanitize_filename(name):
    """Make a string safe for use in filenames"""
    name = re.sub(r'[^\w\-]', '_', name)
    return name.lower()


def _clamp_float(value, minimum, maximum, fallback):
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = fallback
    return min(maximum, max(minimum, parsed))


def _clamp_int(value, minimum, maximum, fallback):
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = fallback
    return min(maximum, max(minimum, parsed))


def combine_audio_with_pauses(audio_segments, speakers, pause_ms=DEFAULT_PAUSE_MS, same_speaker_pause_ms=SAME_SPEAKER_PAUSE_MS):
    """Combine audio segments with pauses between them"""
    if not audio_segments:
        return None

    silence_between_speakers = AudioSegment.silent(duration=pause_ms)
    silence_same_speaker = AudioSegment.silent(duration=same_speaker_pause_ms)

    combined = audio_segments[0]
    prev_speaker = speakers[0]

    for segment, speaker in zip(audio_segments[1:], speakers[1:]):
        if speaker == prev_speaker:
            combined += silence_same_speaker + segment
        else:
            combined += silence_between_speakers + segment
        prev_speaker = speaker

    return combined


class AudioProvider:
    def __init__(self, engine):
        self.engine = engine

    @property
    def mode(self):
        raise NotImplementedError

    @property
    def local_backend(self):
        raise NotImplementedError

    def generate_voice(self, text, instruct_text, speaker, voice_config, output_path, cancel_check=None):
        raise NotImplementedError

    def generate_batch(self, chunks, voice_config, output_dir, batch_seed=-1, cancel_check=None, log_callback=None):
        raise NotImplementedError

    def generate_voice_design(self, description, sample_text, language=None, seed=-1):
        raise NotImplementedError

    def clear_clone_prompt_cache(self, speaker=None):
        raise NotImplementedError

    def unload_voice_design_model(self):
        raise NotImplementedError

    def unload(self):
        raise NotImplementedError


class QwenAudioProvider(AudioProvider):
    provider_name = "qwen3"

    @property
    def mode(self):
        return self.engine._mode

    @property
    def local_backend(self):
        if self.mode != "local":
            return None
        return self.engine._resolve_local_backend()

    def generate_voice(self, text, instruct_text, speaker, voice_config, output_path, cancel_check=None):
        return self.engine._provider_generate_voice(text, instruct_text, speaker, voice_config, output_path)

    def generate_batch(self, chunks, voice_config, output_dir, batch_seed=-1, cancel_check=None, log_callback=None):
        return self.engine._provider_generate_batch(
            chunks,
            voice_config,
            output_dir,
            batch_seed=batch_seed,
            cancel_check=cancel_check,
            log_callback=log_callback,
        )

    def generate_voice_design(self, description, sample_text, language=None, seed=-1):
        return self.engine._provider_generate_voice_design(
            description=description,
            sample_text=sample_text,
            language=language,
            seed=seed,
        )

    def clear_clone_prompt_cache(self, speaker=None):
        return self.engine._provider_clear_clone_prompt_cache(speaker=speaker)

    def unload_voice_design_model(self):
        had_design_model = self.engine._local_design_model is not None
        had_mlx_design_model = "voice_design" in self.engine._mlx_models
        self.engine._local_design_model = None
        self.engine._mlx_models.pop("voice_design", None)
        clear_cache = getattr(self.engine, "_clear_gpu_cache", None)
        if callable(clear_cache):
            clear_cache()
        return bool(had_design_model or had_mlx_design_model)

    def unload(self):
        self.engine._provider_clear_clone_prompt_cache()
        self.engine._lora_prompt_cache.clear()
        self.engine._mlx_models.clear()
        self.engine._local_custom_model = None
        self.engine._local_clone_model = None
        self.engine._local_design_model = None
        self.engine._local_lora_model = None
        self.engine._lora_adapter_path = None
        self.engine._gradio_client = None
        self.engine._external_backend = None
        self.engine._external_http_base = None
        self.engine._resolved_local_backend = None
        self.engine._e2e_qwen_sim_provider = None
        clear_cache = getattr(self.engine, "_clear_gpu_cache", None)
        if callable(clear_cache):
            clear_cache()
        return True


class TTSEngine:
    """TTS engine supporting local (qwen/mlx) and external (Gradio/HTTP) backends.

    Mode is determined by config["tts"]["mode"]:
      - "local": Loads local backend (Qwen or MLX) directly. No external server needed.
      - "external": Connects via Gradio client to a running TTS server.

    Models and clients are lazily initialized on first use.
    """

    def __init__(self, config, *, project_root=None):
        tts_config = config.get("tts", {})
        self._provider_name = self._normalize_provider_name(tts_config.get("provider", "qwen3"))
        self._mode = tts_config.get("mode", "external")
        self._local_backend_preference = (tts_config.get("local_backend", "auto") or "auto").strip().lower()
        if self._local_backend_preference not in {"auto", "qwen", "mlx"}:
            self._local_backend_preference = "auto"
        self._resolved_local_backend = None
        self._url = tts_config.get("url", "http://127.0.0.1:7860")
        self._api_key = tts_config.get("api_key") or config.get("llm", {}).get("api_key", "")
        self._device = tts_config.get("device", "auto")
        self._compile_codec_enabled = tts_config.get("compile_codec", False)
        self._voxcpm_model_id = str(tts_config.get("voxcpm_model_id") or "openbmb/VoxCPM2").strip()
        self._voxcpm_cfg_value = _clamp_float(
            tts_config.get("voxcpm_cfg_value", VOXCPM2_CFG_VALUE_DEFAULT),
            VOXCPM2_CFG_VALUE_MIN,
            VOXCPM2_CFG_VALUE_MAX,
            VOXCPM2_CFG_VALUE_DEFAULT,
        )
        self._voxcpm_inference_timesteps = _clamp_int(
            tts_config.get("voxcpm_inference_timesteps", VOXCPM2_INFERENCE_TIMESTEPS_DEFAULT),
            VOXCPM2_INFERENCE_TIMESTEPS_MIN,
            VOXCPM2_INFERENCE_TIMESTEPS_MAX,
            VOXCPM2_INFERENCE_TIMESTEPS_DEFAULT,
        )
        self._voxcpm_normalize = bool(tts_config.get("voxcpm_normalize", False))
        self._voxcpm_load_denoiser = bool(tts_config.get("voxcpm_load_denoiser", False))
        self._voxcpm_denoise_reference = bool(tts_config.get("voxcpm_denoise_reference", False))
        self._voxcpm_optimize = bool(tts_config.get("voxcpm_optimize", False)) and py_platform.system().lower() != "darwin"

        # Language setting (passed to Qwen3-TTS)
        self._language = tts_config.get("language", "English")

        # Sub-batching config
        self._sub_batch_enabled = tts_config.get("sub_batch_enabled", True)
        self._sub_batch_min_size = max(1, tts_config.get("sub_batch_min_size", 4))
        self._sub_batch_ratio = max(1.0, float(tts_config.get("sub_batch_ratio", 5)))
        self._sub_batch_max_chars = max(500, int(tts_config.get("sub_batch_max_chars", 3000)))
        self._sub_batch_max_items = int(tts_config.get("sub_batch_max_items", 0))  # 0 = auto
        self._progress_log_interval_seconds = max(
            5.0,
            float(tts_config.get("progress_log_interval_seconds", 15.0)),
        )

        # Lazy-loaded backends (guarded by _model_lock to prevent concurrent loads)
        self._model_lock = threading.Lock()
        self._custom_inference_lock = threading.Lock()
        self._clone_prompt_lock = threading.Lock()
        self._clone_inference_lock = threading.Lock()
        self._design_inference_lock = threading.Lock()
        self._lora_inference_lock = threading.Lock()
        self._local_custom_model = None
        self._local_clone_model = None
        self._local_design_model = None
        self._local_lora_model = None
        self._warmup_needed = True  # cleared after first batch warmup
        self._lora_adapter_path = None  # track which adapter is currently loaded
        self._gradio_client = None
        self._external_backend = None
        self._external_http_base = None
        self._project_root = os.path.abspath(project_root or LAYOUT.project_dir)

        # MLX local model cache
        self._mlx_models = {}

        # Clone prompt cache: speaker_name -> (ref_audio_path, reusable voice_clone_prompt)
        self._clone_prompt_cache = {}
        # LoRA clone prompt cache: adapter_path -> reusable voice_clone_prompt
        self._lora_prompt_cache = {}
        self._e2e_qwen_sim_provider = None
        self._provider = self._create_provider()

    @property
    def mode(self):
        return self._provider.mode

    @property
    def local_backend(self):
        return self._provider.local_backend

    @property
    def provider_name(self):
        return self._provider_name

    def unload(self):
        return self._provider.unload()

    def unload_voice_design_model(self):
        unload = getattr(self._provider, "unload_voice_design_model", None)
        if callable(unload):
            return bool(unload())
        return False

    @staticmethod
    def _normalize_provider_name(value):
        normalized = str(value or "qwen3").strip().lower()
        return normalized or "qwen3"

    def _create_provider(self):
        if self._provider_name == "qwen3":
            return QwenAudioProvider(self)
        if self._provider_name == "voxcpm2":
            from tts_providers.voxcpm2 import VoxCPM2AudioProvider

            return VoxCPM2AudioProvider(self)
        raise ValueError(f"Unsupported TTS provider: {self._provider_name}")

    @staticmethod
    def _env_flag(name, default=False):
        raw = os.getenv(name)
        if raw is None:
            return bool(default)
        return str(raw).strip().lower() in TRUE_VALUES

    def _maybe_init_e2e_qwen_sim_provider(self):
        if self._e2e_qwen_sim_provider is not None:
            return self._e2e_qwen_sim_provider
        if not self._env_flag("THREADSPEAK_E2E_SIM_ENABLED", default=False):
            return None
        if self._mode != "local":
            return None
        if self._resolve_local_backend() != "qwen":
            return None

        fixture_path = (os.getenv("THREADSPEAK_E2E_QWEN_FIXTURE") or "").strip()
        if not fixture_path:
            return None

        try:
            from e2e_sim.qwen_local_sim import QwenLocalSimProvider

            self._e2e_qwen_sim_provider = QwenLocalSimProvider(fixture_path)
            print(f"E2E local Qwen simulator enabled from fixture: {fixture_path}")
            return self._e2e_qwen_sim_provider
        except Exception as e:
            raise RuntimeError(f"Failed to initialize local Qwen E2E simulator: {e}") from e

    @staticmethod
    def _normalize_external_url(url):
        normalized = (url or "").strip()
        if not normalized:
            return "http://127.0.0.1:7860"

        # urlparse("localhost:42003") treats "localhost" as a scheme, which
        # breaks bare host:port inputs commonly entered in the UI.
        if "://" not in normalized:
            normalized = f"http://{normalized}"

        parsed = urlparse(normalized)
        if parsed.scheme not in {"http", "https"}:
            raise ValueError(f"Unsupported external TTS URL scheme: {parsed.scheme}")

        return normalized.rstrip("/")

    @classmethod
    def _external_url_candidates(cls, url):
        base = cls._normalize_external_url(url)
        candidates = [base]
        seen = {base}

        def add_candidate(candidate):
            normalized = candidate.rstrip("/")
            if normalized and normalized not in seen:
                seen.add(normalized)
                candidates.append(normalized)

        # Follow root redirect once to support Gradio apps mounted below "/".
        try:
            response = httpx.get(
                f"{base}/",
                follow_redirects=True,
                timeout=5.0,
            )
            final_url = str(response.url).rstrip("/")
            add_candidate(final_url)
        except Exception:
            pass

        # Common mount points seen in proxied/self-hosted Gradio deployments.
        parsed = urllib.parse.urlsplit(base)
        if parsed.path in {"", "/"}:
            for suffix in ("/gradio", "/gradio/", "/gradio_api", "/gradio_api/"):
                add_candidate(urllib.parse.urlunsplit(parsed._replace(path=suffix, query="", fragment="")))

        return candidates

    @staticmethod
    def _concat_audio(wav):
        """Concatenate audio array(s) into a single numpy array."""
        if isinstance(wav, list):
            return np.concatenate(wav) if len(wav) > 1 else wav[0]
        return wav

    @staticmethod
    def _short_uid(value):
        text = str(value or "").strip()
        if not text:
            return ""
        return text[:8]

    @staticmethod
    def _summarize_list(values, limit=4):
        items = [str(value) for value in (values or []) if str(value).strip()]
        if not items:
            return "[]"
        if len(items) <= limit:
            return "[" + ", ".join(items) + "]"
        visible = ", ".join(items[:limit])
        return f"[{visible}, +{len(items) - limit} more]"

    def _describe_batch_targets(self, chunk_ids=None, chunk_uids=None, text_lengths=None):
        parts = []
        if chunk_ids:
            parts.append(f"chunk_ids={self._summarize_list(chunk_ids)}")
        if chunk_uids:
            parts.append(f"uids={self._summarize_list([self._short_uid(uid) for uid in chunk_uids])}")
        if text_lengths:
            parts.append(f"text_chars={self._summarize_list(text_lengths)}")
            parts.append(f"total_chars={sum(int(length) for length in text_lengths)}")
        return ", ".join(parts)

    @staticmethod
    def _emit_log(message, log_callback=None):
        print(message)
        if callable(log_callback):
            try:
                log_callback(message)
            except Exception:
                pass

    @contextlib.contextmanager
    def _progress_log_context(self, message_factory, log_callback=None):
        interval = float(self._progress_log_interval_seconds or 0.0)
        if interval <= 0:
            yield
            return

        stop_event = threading.Event()
        started_at = time.time()

        def _worker():
            while not stop_event.wait(interval):
                try:
                    message = message_factory(max(0.0, time.time() - started_at))
                except Exception:
                    continue
                if message:
                    self._emit_log(message, log_callback=log_callback)

        worker = threading.Thread(
            target=_worker,
            name="threadspeak-tts-progress-log",
            daemon=True,
        )
        worker.start()
        try:
            yield
        finally:
            stop_event.set()
            worker.join(timeout=1.0)

    @staticmethod
    def _clear_gpu_cache():
        """Free GPU memory: garbage-collect Python objects, then clear CUDA cache."""
        import gc
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            return

    @staticmethod
    def _reset_compile_cache():
        """Reset torch.compile dynamo state to prevent guard accumulation.

        torch.compile(dynamic=True) accumulates shape guards across calls.
        With varying batch sizes and sequence lengths, the guard list grows
        and CPU-side guard evaluation becomes a bottleneck, causing
        progressive throughput degradation.  Resetting clears all in-memory
        guards; the next call pays a one-time recompilation cost (fast due
        to inductor disk cache) but prevents the slowdown from compounding.
        """
        try:
            import torch
            torch._dynamo.reset()
        except Exception:
            return

    def _estimate_max_batch_size(self, model, clone_prompt_tokens=0,
                                ref_text_chars=0, max_text_chars=0,
                                max_new_tokens=2048):
        """Estimate how many sequences fit in free VRAM based on KV cache math.

        Uses the talker's architecture (num_layers, num_kv_heads, head_dim) to
        calculate KV cache bytes per token, then estimates total tokens per
        sequence from clone prompt size + text length + max generation length.

        Returns max batch size (>= 1).  Falls back to a large default on CPU
        or if the model config is inaccessible.
        """
        import torch
        if not torch.cuda.is_available():
            return 9999

        try:
            config = model.model.talker.config
            num_layers = config.num_hidden_layers
            num_kv_heads = config.num_key_value_heads
            head_dim = config.hidden_size // config.num_attention_heads
        except AttributeError:
            return 9999  # can't read config, skip estimation

        dtype_bytes = 2  # bf16
        kv_per_token = num_layers * 2 * num_kv_heads * head_dim * dtype_bytes

        # Total tokens per sequence (worst case: padded to longest + full generation)
        overhead = 10  # role tokens + prefix + special tokens
        ref_text_tokens = ref_text_chars // 3 if ref_text_chars else 0
        text_tokens = max_text_chars // 3 if max_text_chars else 0
        total_tokens = overhead + clone_prompt_tokens + ref_text_tokens + text_tokens + max_new_tokens

        # Overhead factor covers prefill activations, codec, allocator fragmentation
        OVERHEAD_FACTOR = 2.0
        mem_per_seq = total_tokens * kv_per_token * OVERHEAD_FACTOR

        # Available = driver-level free + PyTorch reserved-but-unallocated
        free_driver, _ = torch.cuda.mem_get_info()
        reserved_unused = torch.cuda.memory_reserved() - torch.cuda.memory_allocated()
        free_total = free_driver + reserved_unused

        budget = int(free_total * 0.8)
        max_batch = max(1, budget // mem_per_seq)

        print(f"VRAM estimate: {free_total / 1e9:.1f}GB free, "
              f"{total_tokens} tok/seq ({clone_prompt_tokens} prompt + "
              f"{ref_text_tokens + text_tokens} text + {max_new_tokens} gen), "
              f"{mem_per_seq / 1e6:.0f}MB/seq -> max_batch={max_batch}")

        return max_batch

    def _estimate_generation_seconds_for_text(self, text):
        """Estimate a bounded generation budget for one utterance.

        Qwen3-TTS-12Hz uses acoustic tokens, so a very large fixed
        max_new_tokens allows tiny lines to overgenerate for minutes.
        Keep the budget proportional to the text and cap it globally so
        local generation fails fast instead of hanging indefinitely.
        """
        normalized = (text or "").strip()
        expected_duration = estimate_expected_duration_seconds(text=normalized)
        if expected_duration <= 0:
            expected_duration = max(0.75, len(normalized) / 24.0)

        budget_seconds = (expected_duration * 2.5) + QWEN_GENERATION_SECONDS_MARGIN
        return max(
            QWEN_MIN_GENERATION_SECONDS,
            min(QWEN_MAX_GENERATION_SECONDS, budget_seconds),
        )

    def _qwen_max_new_tokens_for_text(self, text):
        generation_seconds = self._estimate_generation_seconds_for_text(text)
        budget_tokens = int(
            generation_seconds * QWEN_AUDIO_TOKENS_PER_SECOND * QWEN_GENERATION_TOKEN_BUFFER_FACTOR
        )
        return max(
            QWEN_MIN_MAX_NEW_TOKENS,
            min(QWEN_MAX_MAX_NEW_TOKENS, budget_tokens),
        )

    def _qwen_max_new_tokens_for_texts(self, texts):
        if not texts:
            return QWEN_MIN_MAX_NEW_TOKENS
        return max(self._qwen_max_new_tokens_for_text(text) for text in texts)

    def _build_sub_batches(self, texts, max_items=None):
        """Split sorted-by-length texts into sub-batches.

        Splits on four criteria (checked in order):
        1. VRAM item limit: when max_items is set (from _estimate_max_batch_size)
        2. Total chars: when cumulative chars exceed sub_batch_max_chars
        3. Length ratio: when longest/shortest > sub_batch_ratio
        4. Minimum size: ratio splits only happen after sub_batch_min_size items

        Returns list of (start, end) index tuples.
        """
        if not self._sub_batch_enabled or len(texts) <= 1:
            return [(0, len(texts))]

        # Manual cap overrides VRAM estimate when set (take the stricter of the two)
        if self._sub_batch_max_items > 0:
            max_items = min(max_items, self._sub_batch_max_items) if max_items else self._sub_batch_max_items

        sub_batches = []
        batch_start = 0
        batch_chars = len(texts[0])

        for i in range(1, len(texts)):
            shortest = max(len(texts[batch_start]), 1)
            batch_chars += len(texts[i])
            should_split = False

            # VRAM-estimated item limit (highest priority — based on actual
            # free GPU memory and per-sequence KV cache cost)
            if max_items is not None and (i - batch_start) >= max_items:
                should_split = True
            # Chars split: too much total text risks OOM — always split
            # regardless of min_size (memory safety takes priority)
            elif batch_chars > self._sub_batch_max_chars and (i - batch_start) >= 1:
                should_split = True
            # Ratio split: large length disparity wastes padding —
            # only split after min_size items to preserve parallelism
            elif (i - batch_start) >= self._sub_batch_min_size:
                if len(texts[i]) > self._sub_batch_ratio * shortest:
                    should_split = True

            if should_split:
                sub_batches.append((batch_start, i))
                batch_start = i
                batch_chars = len(texts[i])

        sub_batches.append((batch_start, len(texts)))
        return sub_batches

    # ── Lazy initialization ──────────────────────────────────────

    def _warmup_model(self, model):
        """Run a short warmup generation to pre-tune MIOpen/GPU solvers.

        First generation after model load is ~2x slower due to MIOpen autotuning.
        This warmup pays that cost upfront so real generations run at full speed.
        """
        import time
        t0 = time.time()
        try:
            warmup_text = (
                "The ancient library stood at the crossroads of two forgotten paths, "
                "its weathered stone walls covered in ivy that had been growing for centuries."
            )
            model.generate_custom_voice(
                text=warmup_text,
                language=self._language,
                speaker="serena",
                instruct="neutral",
                non_streaming_mode=True,
                max_new_tokens=self._qwen_max_new_tokens_for_text(warmup_text),
            )
            print(f"Warmup done in {time.time()-t0:.1f}s")
        except Exception as e:
            print(f"Warmup failed (non-fatal): {e}")

    @staticmethod
    def _host_platform():
        return py_platform.system().lower(), py_platform.machine().lower()

    @staticmethod
    def _apply_mistral_regex_fix(tokenizer):
        """Apply the corrected mistral split regex to HF tokenizers backend.

        Newer transformers warns for some local tokenizer configs (including
        qwen3_tts snapshots) but the explicit fix flag is currently unstable in
        our pinned stack. Apply the same patch directly after load.
        """
        try:
            import tokenizers
        except Exception:
            return False

        if tokenizer is None or not hasattr(tokenizer, "backend_tokenizer"):
            return False

        split_pretokenizer = tokenizers.pre_tokenizers.Split(
            pattern=tokenizers.Regex(
                r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+"
            ),
            behavior="isolated",
        )

        current = tokenizer.backend_tokenizer.pre_tokenizer
        if isinstance(current, tokenizers.pre_tokenizers.Sequence):
            tokenizer.backend_tokenizer.pre_tokenizer[0] = split_pretokenizer
        else:
            if isinstance(current, tokenizers.pre_tokenizers.Metaspace):
                current = tokenizers.pre_tokenizers.ByteLevel(
                    add_prefix_space=False, use_regex=False
                )
            tokenizer.backend_tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.Sequence(
                [split_pretokenizer, current]
            )

        setattr(tokenizer, "fix_mistral_regex", True)
        return True

    @contextlib.contextmanager
    def _suppress_known_transformers_qwen3_warnings(self):
        """Suppress known non-fatal transformers warnings for local qwen3_tts."""

        class _DropMessages(logging.Filter):
            def filter(self, record):
                msg = record.getMessage()
                blocked = (
                    "incorrect regex pattern" in msg
                    or "set the `fix_mistral_regex=True` flag" in msg
                    or "using a model of type qwen3_tts to instantiate a model of type" in msg
                )
                return not blocked

        logger_names = [
            "transformers.tokenization_utils_tokenizers",
            "transformers.configuration_utils",
        ]
        filt = _DropMessages()
        loggers = [logging.getLogger(name) for name in logger_names]
        for lg in loggers:
            lg.addFilter(filt)
        try:
            yield
        finally:
            for lg in loggers:
                lg.removeFilter(filt)

    def _postprocess_mlx_qwen3_tokenizer(self, model):
        """Patch tokenizer regex for mlx qwen3_tts models when available."""
        try:
            model_type = getattr(getattr(model, "config", None), "model_type", "")
            if model_type != "qwen3_tts":
                return
            tokenizer = getattr(model, "tokenizer", None)
            if self._apply_mistral_regex_fix(tokenizer):
                print("Applied tokenizer regex fix for MLX qwen3_tts model.")
        except Exception as e:
            print(f"Tokenizer regex post-fix skipped (non-fatal): {e}")

    def _resolve_local_backend(self):
        if self._resolved_local_backend is not None:
            return self._resolved_local_backend

        if self._mode != "local":
            self._resolved_local_backend = None
            return self._resolved_local_backend

        system_name, machine = self._host_platform()
        is_apple_silicon = system_name == "darwin" and machine in {"arm64", "aarch64"}

        backend = self._local_backend_preference
        if backend == "auto":
            backend = "mlx" if is_apple_silicon else "qwen"
        elif backend == "mlx" and not is_apple_silicon:
            print("Warning: local_backend=mlx requested on non-Apple-Silicon host. Falling back to qwen.")
            backend = "qwen"

        self._resolved_local_backend = backend
        return backend

    @staticmethod
    def _mlx_voice_name(voice):
        mapping = {
            "ono_anna": "Ono_Anna",
            "uncle_fu": "Uncle_Fu",
        }
        key = str(voice or "").strip()
        if not key:
            return "Ryan"
        normalized = mapping.get(key.lower())
        return normalized or key

    def _resolve_device(self):
        """Resolve 'auto' device to the best available."""
        if self._device != "auto":
            return self._device

        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
        except ImportError:
            pass
        return "cpu"

    def _enable_rocm_optimizations(self):
        """Apply ROCm-specific optimizations. No-op on NVIDIA/CPU.

        1. FLASH_ATTENTION_TRITON_AMD_ENABLE: Lets qwen_tts whisper encoder
           use native flash attention via Triton AMD backend.
        2. MIOPEN_FIND_MODE=2: Forces MIOpen to use fast-find instead of
           exhaustive search, avoiding workspace allocation failures that
           cause fallback to slow GEMM algorithms.
        3. MIOPEN_LOG_LEVEL=4: Suppress noisy MIOpen workspace warnings.
        4. triton_key shim: Bridges pytorch-triton-rocm's get_cache_key()
           to the triton_key() that PyTorch's inductor expects.
        """
        try:
            import torch
            if not (hasattr(torch.version, "hip") and torch.version.hip):
                return  # not ROCm
        except ImportError:
            return

        # MIOpen: use fast-find to avoid workspace allocation failures
        os.environ.setdefault("MIOPEN_FIND_MODE", "2")
        # Suppress MIOpen workspace warnings
        os.environ.setdefault("MIOPEN_LOG_LEVEL", "4")

        # Flash attention via Triton AMD backend
        os.environ.setdefault("FLASH_ATTENTION_TRITON_AMD_ENABLE", "TRUE")

        # Fix triton_key compatibility for torch.compile on ROCm
        try:
            from triton.compiler import compiler as triton_compiler
            if not hasattr(triton_compiler, "triton_key"):
                import triton
                triton_compiler.triton_key = lambda: f"pytorch-triton-rocm-{triton.__version__}"
        except ImportError:
            pass

    def _compile_codec(self, model):
        """Apply torch.compile to the audio codec for faster decoding.

        The codec decoder has 136 attention modules and many small ops that
        benefit enormously from compilation.  Profiling shows the codec is
        47% of single-gen time and 85% of batch time uncompiled.  With
        torch.compile (dynamic=True, max-autotune), batch throughput
        improves from ~1.3x to ~4.3x real-time and single generation
        drops from ~14s to ~9s.

        max-autotune mode benchmarks GPU kernels to pick the fastest and
        handles varying batch sizes gracefully (unlike reduce-overhead
        which uses CUDA graphs that break on shape changes).
        """
        import torch
        try:
            codec = model.model.speech_tokenizer.model
            model.model.speech_tokenizer.model = torch.compile(
                codec, mode="max-autotune", dynamic=True,
            )
            print("Codec compiled with torch.compile (dynamic=True).")
        except Exception as e:
            print(f"Codec compilation skipped (non-fatal): {e}")

    @staticmethod
    def _resolve_local_model_path(model_id, required_files=None):
        """Check if a HuggingFace model is cached locally and return its snapshot path.

        Uses try_to_load_from_cache to find the local snapshot directory.
        Returns the local path string if cached, or None if not cached.
        """
        from huggingface_hub import try_to_load_from_cache

        def _has_required_files(snapshot_path):
            if not os.path.exists(os.path.join(snapshot_path, "config.json")):
                return False
            for requirement in required_files or ():
                if isinstance(requirement, (list, tuple, set)):
                    if not any(os.path.exists(os.path.join(snapshot_path, str(item))) for item in requirement):
                        return False
                elif not os.path.exists(os.path.join(snapshot_path, str(requirement))):
                    return False
            return True

        result = try_to_load_from_cache(model_id, "config.json")
        if isinstance(result, str):
            # result is the full path to config.json inside the snapshot dir
            candidate = os.path.dirname(result)
            if _has_required_files(candidate):
                return candidate
        repo_cache_name = f"models--{str(model_id).replace('/', '--')}"
        cache_roots = []
        hub_cache = str(os.getenv("HUGGINGFACE_HUB_CACHE") or "").strip()
        if hub_cache:
            cache_roots.append(hub_cache)
        hf_home = str(os.getenv("HF_HOME") or "").strip()
        if hf_home:
            cache_roots.append(os.path.join(hf_home, "hub"))
        cache_roots.append(os.path.join(LAYOUT.repo_root, "cache", "HF_HOME", "hub"))

        for cache_root in cache_roots:
            snapshots_dir = os.path.join(cache_root, repo_cache_name, "snapshots")
            if not os.path.isdir(snapshots_dir):
                continue
            try:
                snapshots = sorted(os.listdir(snapshots_dir), reverse=True)
            except OSError:
                continue
            for snapshot in snapshots:
                candidate = os.path.join(snapshots_dir, snapshot)
                if _has_required_files(candidate):
                    return candidate
        return None

    @staticmethod
    def _load_model(model_cls, model_id, load_kwargs):
        """Load a model, preferring local cache to avoid network issues.

        Checks if the model snapshot exists in the HF cache and loads from
        the local directory path directly, bypassing all HF Hub network calls.
        Falls back to normal download on first install when cache is empty.
        """
        local_path = TTSEngine._resolve_local_model_path(model_id)
        if local_path:
            print(f"  Loading from local cache: {local_path}")
            return model_cls.from_pretrained(local_path, **load_kwargs)
        else:
            if TTSEngine._env_flag("THREADSPEAK_DISABLE_MODEL_DOWNLOADS", default=False):
                raise RuntimeError(
                    "Model downloads are disabled by THREADSPEAK_DISABLE_MODEL_DOWNLOADS "
                    f"and no local cache exists for {model_id}."
                )
            print(f"  Model not cached locally, downloading {model_id}...")
            downloaded_path = ensure_hf_snapshot(model_id, display_name=model_id)
            return model_cls.from_pretrained(downloaded_path, **load_kwargs)

    @staticmethod
    def _import_qwen_tts_model(device):
        """Import Qwen3TTSModel while suppressing irrelevant flash-attn warnings.

        qwen-tts prints a flash-attn installation warning during import even on
        non-CUDA runtimes such as Apple Silicon and CPU, where flash-attn is not
        applicable for this session. Keep the import noise on CUDA, where the
        warning is actionable, and suppress only the known upstream message
        elsewhere.
        """
        if "cuda" in str(device).lower():
            from qwen_tts import Qwen3TTSModel
            return Qwen3TTSModel

        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            from qwen_tts import Qwen3TTSModel

        captured = buffer.getvalue()
        if captured:
            filtered = captured.replace(
                "\n********\nWarning: flash-attn is not installed. Will only run the manual PyTorch version. Please install flash-attn for faster inference.\n********\n ",
                "",
            ).strip()
            if filtered:
                print(filtered)
        return Qwen3TTSModel

    def _init_local_custom(self):
        """Load Qwen3-TTS CustomVoice model on demand."""
        sim_provider = self._maybe_init_e2e_qwen_sim_provider()
        if sim_provider is not None:
            if self._local_custom_model is None:
                self._local_custom_model = sim_provider.get_model("custom_voice")
            return self._local_custom_model

        if self._local_custom_model is not None:
            return self._local_custom_model

        with self._model_lock:
            if self._local_custom_model is not None:
                return self._local_custom_model

            self._enable_rocm_optimizations()

            import torch
            device = self._resolve_device()
            Qwen3TTSModel = self._import_qwen_tts_model(device)
            dtype = torch.bfloat16 if "cuda" in device else torch.float32

            print(f"Loading Qwen3-TTS CustomVoice model on {device} ({dtype})...")
            load_kwargs = {"dtype": dtype}
            if device != "cpu":
                load_kwargs["device_map"] = device
            self._local_custom_model = self._load_model(
                Qwen3TTSModel, "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice", load_kwargs,
            )
            if self._compile_codec_enabled:
                self._compile_codec(self._local_custom_model)
            print("CustomVoice model loaded.")
            return self._local_custom_model

    def _init_local_clone(self):
        """Load Qwen3-TTS Base model (for voice cloning) on demand."""
        sim_provider = self._maybe_init_e2e_qwen_sim_provider()
        if sim_provider is not None:
            if self._local_clone_model is None:
                self._local_clone_model = sim_provider.get_model("base")
            return self._local_clone_model

        if self._local_clone_model is not None:
            return self._local_clone_model

        with self._model_lock:
            if self._local_clone_model is not None:
                return self._local_clone_model

            self._enable_rocm_optimizations()

            import torch
            device = self._resolve_device()
            Qwen3TTSModel = self._import_qwen_tts_model(device)
            dtype = torch.bfloat16 if "cuda" in device else torch.float32

            print(f"Loading Qwen3-TTS Base model (voice cloning) on {device} ({dtype})...")
            load_kwargs = {"dtype": dtype}
            if device != "cpu":
                load_kwargs["device_map"] = device
            self._local_clone_model = self._load_model(
                Qwen3TTSModel, "Qwen/Qwen3-TTS-12Hz-1.7B-Base", load_kwargs,
            )
            if self._compile_codec_enabled:
                self._compile_codec(self._local_clone_model)
            print("Base model (voice cloning) loaded.")
            return self._local_clone_model

    def _init_local_design(self):
        """Load Qwen3-TTS VoiceDesign model on demand."""
        sim_provider = self._maybe_init_e2e_qwen_sim_provider()
        if sim_provider is not None:
            if self._local_design_model is None:
                self._local_design_model = sim_provider.get_model("voice_design")
            return self._local_design_model

        if self._local_design_model is not None:
            return self._local_design_model

        with self._model_lock:
            if self._local_design_model is not None:
                return self._local_design_model

            self._enable_rocm_optimizations()

            import torch
            device = self._resolve_device()
            Qwen3TTSModel = self._import_qwen_tts_model(device)
            dtype = torch.bfloat16 if "cuda" in device else torch.float32

            print(f"Loading Qwen3-TTS VoiceDesign model on {device} ({dtype})...")
            load_kwargs = {"dtype": dtype}
            if device != "cpu":
                load_kwargs["device_map"] = device
            self._local_design_model = self._load_model(
                Qwen3TTSModel, "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign", load_kwargs,
            )
            if self._compile_codec_enabled:
                self._compile_codec(self._local_design_model)
            print("VoiceDesign model loaded.")
            return self._local_design_model

    def _init_local_lora(self, adapter_path):
        """Load Qwen3-TTS Base model with a LoRA adapter on demand.

        Caches the model; if a different adapter is requested the old one
        is unloaded first to free VRAM.
        """
        sim_provider = self._maybe_init_e2e_qwen_sim_provider()
        if sim_provider is not None:
            if self._local_lora_model is None or self._lora_adapter_path != adapter_path:
                self._local_lora_model = sim_provider.get_model("lora")
                self._lora_adapter_path = adapter_path
            return self._local_lora_model

        if self._local_lora_model is not None and self._lora_adapter_path == adapter_path:
            return self._local_lora_model

        with self._model_lock:
            if self._local_lora_model is not None and self._lora_adapter_path == adapter_path:
                return self._local_lora_model

            # Unload previous adapter if switching
            if self._local_lora_model is not None:
                print(f"Unloading previous LoRA adapter ({self._lora_adapter_path})...")
                del self._local_lora_model
                self._local_lora_model = None
                self._lora_adapter_path = None
                self._lora_prompt_cache.clear()
                self._clear_gpu_cache()

            self._enable_rocm_optimizations()

            import torch
            from peft import PeftModel

            device = self._resolve_device()
            Qwen3TTSModel = self._import_qwen_tts_model(device)
            dtype = torch.bfloat16 if "cuda" in device else torch.float32

            print(f"Loading Qwen3-TTS Base model + LoRA adapter on {device} ({dtype})...")
            load_kwargs = {"dtype": dtype}
            if device != "cpu":
                load_kwargs["device_map"] = device

            model = self._load_model(
                Qwen3TTSModel, "Qwen/Qwen3-TTS-12Hz-1.7B-Base", load_kwargs,
            )

            # Wrap the talker with the LoRA adapter
            model.model.talker = PeftModel.from_pretrained(
                model.model.talker,
                adapter_path,
            )
            model.model.talker.eval()

            if self._compile_codec_enabled:
                self._compile_codec(model)

            self._local_lora_model = model
            self._lora_adapter_path = adapter_path
            print(f"LoRA adapter loaded from {adapter_path}")
            return model

    def _init_local_mlx_model(self, model_type):
        """Load MLX model on demand for Apple Silicon local backend."""
        model_ids = {
            "custom_voice": "mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit",
            "voice_design": "mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-8bit",
            "base": "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit",
        }
        model_id = model_ids.get(model_type)
        if not model_id:
            raise ValueError(f"Unsupported MLX model type: {model_type}")

        if model_type in self._mlx_models:
            return self._mlx_models[model_type]

        with self._model_lock:
            if model_type in self._mlx_models:
                return self._mlx_models[model_type]

            try:
                from mlx_audio.tts.utils import load_model
            except ImportError as e:
                raise RuntimeError(
                    "MLX backend is not installed. Run Install on Apple Silicon to install mlx-audio dependencies."
                ) from e

            local_path = TTSEngine._resolve_local_model_path(model_id)
            if local_path:
                load_target = local_path
            else:
                if TTSEngine._env_flag("THREADSPEAK_DISABLE_MODEL_DOWNLOADS", default=False):
                    raise RuntimeError(
                        "Model downloads are disabled by THREADSPEAK_DISABLE_MODEL_DOWNLOADS "
                        f"and no local cache exists for {model_id}."
                    )
                load_target = ensure_hf_snapshot(model_id, display_name=model_id)
            print(f"Loading MLX TTS model ({model_type}) from {load_target} ...")
            with self._suppress_known_transformers_qwen3_warnings():
                model = load_model(load_target)
            self._postprocess_mlx_qwen3_tokenizer(model)
            self._mlx_models[model_type] = model
            print(f"MLX model loaded ({model_type}).")
            return model

    @staticmethod
    def _mlx_generate_with_temp_dir(model, **kwargs):
        try:
            from mlx_audio.tts.generate import generate_audio
        except ImportError as e:
            raise RuntimeError("mlx-audio generation API unavailable in current environment.") from e

        run_temp_dir = (os.getenv("THREADSPEAK_RUN_TEMP_DIR") or "").strip() or None
        if run_temp_dir:
            os.makedirs(run_temp_dir, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="alex_mlx_tts_", dir=run_temp_dir) as temp_dir:
            generate_audio(model=model, output_path=temp_dir, **kwargs)
            output_file = os.path.join(temp_dir, "audio_000.wav")
            if not os.path.exists(output_file):
                raise RuntimeError("MLX generation failed: output file not found")
            audio, sr = sf.read(output_file)
            if getattr(audio, "ndim", 1) > 1:
                audio = audio.mean(axis=1)
            return audio.astype(np.float32), int(sr)

    def _init_external(self):
        """Initialize external backend on demand."""
        if self._external_backend == "qwen_mlx_http":
            return self._external_http_base
        if self._gradio_client is not None:
            self._external_backend = "gradio"
            return self._gradio_client

        from gradio_client import Client

        http_base = self._detect_external_http_api()
        if http_base is not None:
            self._external_backend = "qwen_mlx_http"
            self._external_http_base = http_base
            print(f"Connected to external TTS HTTP API at {http_base}.")
            return http_base

        last_error = None
        for url in self._external_url_candidates(self._url):
            try:
                print(f"Connecting to TTS server at {url}...")
                if self._provider_name == "voxcpm2":
                    self._gradio_client = Client(url, download_files=False)
                else:
                    self._gradio_client = Client(url)
                self._external_backend = "gradio"
                print("Connected to external TTS server.")
                return self._gradio_client
            except Exception as e:
                last_error = e
                self._gradio_client = None
                print(f"External TTS probe failed for {url}: {e}")

        raise ValueError(
            f"Could not connect to external TTS server starting from '{self._url}'. "
            f"Tried: {', '.join(self._external_url_candidates(self._url))}"
        ) from last_error

    def _detect_external_http_api(self):
        base = self._normalize_external_url(self._url)
        try:
            response = httpx.get(
                urllib.parse.urljoin(f"{base}/", "openapi.json"),
                timeout=5.0,
                follow_redirects=True,
            )
            if not response.is_success:
                return None
            spec = response.json()
        except Exception:
            return None

        paths = spec.get("paths", {})
        if "/api/v1/custom-voice/generate" in paths and "/api/v1/base/clone" in paths:
            return base
        return None

    def _external_headers(self):
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["X-API-Key"] = self._api_key
        return headers

    def _external_http_post(self, path, payload):
        base = self._init_external()
        if self._external_backend != "qwen_mlx_http":
            raise RuntimeError("External backend is not HTTP API mode")

        response = httpx.post(
            urllib.parse.urljoin(f"{base}/", path.lstrip("/")),
            headers=self._external_headers(),
            json=payload,
            timeout=120.0,
            follow_redirects=True,
        )
        if not response.is_success:
            detail = response.text.strip()
            try:
                detail_json = response.json()
                detail = detail_json.get("detail") or detail
            except Exception:
                pass
            raise ValueError(f"External HTTP API request failed ({response.status_code}): {detail}")
        return response.json()

    @staticmethod
    def _write_base64_audio(audio_b64, output_path):
        if not audio_b64:
            raise ValueError("External HTTP API returned empty audio payload")
        audio_bytes = base64.b64decode(audio_b64)
        with open(output_path, "wb") as f:
            f.write(audio_bytes)
        if os.path.getsize(output_path) == 0:
            raise ValueError("External HTTP API returned an empty audio file")

    @staticmethod
    def _new_voice_design_preview_path():
        previews_dir = os.path.join(LAYOUT.designed_voices_dir, "previews")
        os.makedirs(previews_dir, exist_ok=True)
        return os.path.join(previews_dir, f"preview_{int(time.time() * 1000)}.wav")

    def _resolve_project_path(self, path):
        resolved = str(path or "").strip()
        if not resolved:
            return ""
        if os.path.isabs(resolved):
            return resolved
        return os.path.join(self._project_root, resolved)

    # ── Clone prompt cache (local mode) ──────────────────────────

    def clear_clone_prompt_cache(self, speaker=None):
        return self._provider.clear_clone_prompt_cache(speaker=speaker)

    def _provider_clear_clone_prompt_cache(self, speaker=None):
        if speaker is None:
            self._clone_prompt_cache.clear()
            return
        self._clone_prompt_cache.pop(speaker, None)

    def _get_clone_prompt(self, speaker, voice_config):
        """Get or create a cached voice clone prompt for a speaker."""
        voice_data = voice_config.get(speaker, {})
        ref_audio_path = voice_data.get("ref_audio")
        ref_text = voice_data.get("generated_ref_text") or voice_data.get("ref_text")

        if not ref_audio_path or not ref_text:
            raise ValueError(f"Clone voice for '{speaker}' missing ref_audio or ref_text")
        ref_audio_path = self._resolve_project_path(ref_audio_path)
        if not os.path.exists(ref_audio_path):
            raise FileNotFoundError(f"Reference audio not found for '{speaker}': {ref_audio_path}")

        # Check cache — invalidate if ref_audio changed
        if speaker in self._clone_prompt_cache:
            cached_path, cached_prompt = self._clone_prompt_cache[speaker]
            if cached_path == ref_audio_path:
                return cached_prompt
            print(f"Voice changed for '{speaker}', rebuilding clone prompt...")

        with self._clone_prompt_lock:
            if speaker in self._clone_prompt_cache:
                cached_path, cached_prompt = self._clone_prompt_cache[speaker]
                if cached_path == ref_audio_path:
                    return cached_prompt

            model = self._init_local_clone()

            # Load reference audio as numpy array
            audio_array, sample_rate = sf.read(ref_audio_path)
            # Ensure mono
            if audio_array.ndim > 1:
                audio_array = audio_array.mean(axis=1)

            print(f"Creating clone prompt for '{speaker}'...")
            prompt = model.create_voice_clone_prompt(
                ref_audio=(audio_array, sample_rate),
                ref_text=ref_text,
            )
            self._clone_prompt_cache[speaker] = (ref_audio_path, prompt)
            print(f"Clone prompt cached for '{speaker}'.")
            return prompt

    # ── Core generation methods ──────────────────────────────────

    def generate_custom_voice(self, text, instruct_text, speaker, voice_config, output_path):
        if self._provider_name != "qwen3":
            raise NotImplementedError(f"Provider '{self._provider_name}' does not support custom voice generation")
        return self._provider_generate_custom_voice(text, instruct_text, speaker, voice_config, output_path)

    def _provider_generate_custom_voice(self, text, instruct_text, speaker, voice_config, output_path):
        """Generate audio using CustomVoice model. Returns True on success."""
        if self._mode == "local":
            if self._resolve_local_backend() == "mlx":
                return self._mlx_generate_custom(text, instruct_text, speaker, voice_config, output_path)
            return self._local_generate_custom(text, instruct_text, speaker, voice_config, output_path)
        else:
            return self._external_generate_custom(text, instruct_text, speaker, voice_config, output_path)

    def generate_clone_voice(self, text, speaker, voice_config, output_path, instruct_text=""):
        if self._provider_name != "qwen3":
            raise NotImplementedError(f"Provider '{self._provider_name}' does not support clone voice generation")
        return self._provider_generate_clone_voice(
            text,
            speaker,
            voice_config,
            output_path,
            instruct_text=instruct_text,
        )

    def _provider_generate_clone_voice(self, text, speaker, voice_config, output_path, instruct_text=""):
        """Generate audio using voice cloning. Returns True on success."""
        if self._mode == "local":
            if self._resolve_local_backend() == "mlx":
                return self._mlx_generate_clone(text, speaker, voice_config, output_path, instruct_text=instruct_text)
            return self._local_generate_clone(text, speaker, voice_config, output_path)
        else:
            return self._external_generate_clone(text, speaker, voice_config, output_path)

    def generate_voice(self, text, instruct_text, speaker, voice_config, output_path, cancel_check=None):
        return self._provider.generate_voice(
            text,
            instruct_text,
            speaker,
            voice_config,
            output_path,
            cancel_check=cancel_check,
        )

    def _provider_generate_voice(self, text, instruct_text, speaker, voice_config, output_path):
        """Generate audio using the appropriate method based on voice type config."""
        voice_data = voice_config.get(speaker)
        if not voice_data:
            print(f"Warning: No voice configuration for '{speaker}'. Skipping.")
            return False

        voice_type = voice_data.get("type", "custom")

        if voice_type == "clone":
            return self.generate_clone_voice(text, speaker, voice_config, output_path, instruct_text=instruct_text)
        elif voice_type in ("lora", "builtin_lora"):
            return self.generate_lora_voice(text, instruct_text, voice_data, output_path)
        elif voice_type == "design":
            return self.generate_design_voice(text, instruct_text, voice_data, output_path)
        else:
            return self.generate_custom_voice(text, instruct_text, speaker, voice_config, output_path)

    # ── Voice design generation ──────────────────────────────────

    def generate_voice_design(self, description, sample_text, language=None, seed=-1):
        return self._provider.generate_voice_design(
            description=description,
            sample_text=sample_text,
            language=language,
            seed=seed,
        )

    def _provider_generate_voice_design(self, description, sample_text, language=None, seed=-1):
        """Generate a voice from a text description using the VoiceDesign model.

        Args:
            description: Natural language description of the desired voice
            sample_text: Text to synthesize with the designed voice
            language: Language code (defaults to engine's configured language)
            seed: Random seed (-1 for random, >= 0 for reproducible)

        Returns:
            (wav_path, sample_rate) on success

        Raises:
            RuntimeError: If generation fails
        """
        import time

        lang = language or self._language
        print(f"VoiceDesign: generating preview for description='{description[:80]}...'"
              f"{f', seed={seed}' if seed >= 0 else ''}")

        if self._mode != "local":
            return self._external_generate_voice_design_preview(
                description=description,
                sample_text=sample_text,
                language=lang,
            )

        if self._resolve_local_backend() == "mlx":
            model = self._init_local_mlx_model("voice_design")
            t_start = time.time()
            audio, sr = self._mlx_generate_with_temp_dir(
                model,
                text=sample_text,
                instruct=description,
            )
            gen_time = time.time() - t_start
            duration = len(audio) / sr if sr else 0
            print(f"VoiceDesign [local mlx] done in {gen_time:.1f}s -> {duration:.1f}s audio")
            wav_path = self._new_voice_design_preview_path()
            self._save_wav(audio, sr, wav_path)
            return wav_path, sr

        import torch

        model = self._init_local_design()

        if seed >= 0:
            torch.manual_seed(seed)

        t_start = time.time()
        max_new_tokens = self._qwen_max_new_tokens_for_text(sample_text)
        print(
            f"VoiceDesign: generation budget {max_new_tokens} tokens "
            f"for ~{self._estimate_generation_seconds_for_text(sample_text):.1f}s max audio"
        )
        with self._design_inference_lock:
            wavs, sr = model.generate_voice_design(
                text=sample_text,
                instruct=description,
                language=lang,
                non_streaming_mode=True,
                max_new_tokens=max_new_tokens,
            )
        gen_time = time.time() - t_start

        if wavs is None or len(wavs) == 0:
            raise RuntimeError("VoiceDesign model returned no audio")

        audio = np.concatenate(wavs) if len(wavs) > 1 else wavs[0]
        duration = len(audio) / sr
        print(f"VoiceDesign: done in {gen_time:.1f}s -> {duration:.1f}s audio")

        wav_path = self._new_voice_design_preview_path()
        self._save_wav(audio, sr, wav_path)

        return wav_path, sr

    def generate_design_voice(self, text, instruct_text, voice_data, output_path):
        if self._provider_name != "qwen3":
            raise NotImplementedError(f"Provider '{self._provider_name}' does not support designed voices")
        return self._provider_generate_design_voice(text, instruct_text, voice_data, output_path)

    def _provider_generate_design_voice(self, text, instruct_text, voice_data, output_path):
        """Generate audio using VoiceDesign model with combined description + instruct.

        The voice_data 'description' field provides the base voice identity,
        and the per-line instruct_text is appended for delivery/emotion direction.
        """
        if self._mode != "local":
            return self._external_generate_design(text, instruct_text, voice_data, output_path)

        import shutil

        base_desc = (voice_data.get("description") or "").strip()
        instruct = (instruct_text or "").strip()

        if base_desc and instruct:
            description = f"{base_desc}, {instruct}"
        elif base_desc:
            description = base_desc
        elif instruct:
            description = instruct
        else:
            print("Warning: Design voice has no description or instruct. Using generic.")
            description = "A clear, natural speaking voice"

        wav_path, sr = self.generate_voice_design(description=description, sample_text=text)
        shutil.copy2(wav_path, output_path)
        return True

    # ── LoRA voice generation ────────────────────────────────────

    def generate_lora_voice(self, text, instruct_text, voice_data, output_path):
        if self._provider_name != "qwen3":
            raise NotImplementedError(f"Provider '{self._provider_name}' does not support LoRA voices")
        return self._provider_generate_lora_voice(text, instruct_text, voice_data, output_path)

    def _provider_generate_lora_voice(self, text, instruct_text, voice_data, output_path):
        """Generate audio using a LoRA-finetuned Base model.

        The adapter directory must contain:
          - PEFT adapter weights (adapter_model.safetensors / adapter_config.json)
          - ref_sample.wav (reference audio for voice cloning prompt)
          - training_meta.json (with ref_sample_text)

        The LoRA weights refine voice identity beyond what the reference alone provides.
        """
        if self._mode == "local" and self._resolve_local_backend() == "mlx":
            print("LoRA voice generation is not supported by the local MLX backend yet.")
            return False

        try:
            import torch
            import time

            adapter_path = voice_data.get("adapter_path")
            if not adapter_path:
                print(f"Error: No adapter_path in voice_data")
                return False

            # Resolve relative paths against project root
            if not os.path.isabs(adapter_path):
                root_dir = os.path.dirname(os.path.dirname(__file__))
                adapter_path = os.path.join(root_dir, adapter_path)

            if not os.path.isdir(adapter_path):
                # Auto-download built-in adapters from HF
                adapter_id = os.path.basename(adapter_path)
                if adapter_id.startswith("builtin_"):
                    print(f"Adapter {adapter_id} not downloaded, attempting auto-download...")
                    try:
                        from hf_utils import download_builtin_adapter
                        builtin_dir = os.path.dirname(adapter_path)
                        download_builtin_adapter(adapter_id, builtin_dir)
                    except Exception as e:
                        print(f"Error: Auto-download failed for {adapter_id}: {e}")
                        return False
                else:
                    print(f"Error: LoRA adapter path not found: {adapter_path}")
                    return False

            # Load reference audio and text from adapter directory
            ref_wav_path = os.path.join(adapter_path, "ref_sample.wav")
            meta_path = os.path.join(adapter_path, "training_meta.json")

            if not os.path.exists(ref_wav_path):
                print(f"Error: ref_sample.wav not found in {adapter_path}")
                return False
            if not os.path.exists(meta_path):
                print(f"Error: training_meta.json not found in {adapter_path}")
                return False

            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            ref_text = meta.get("ref_sample_text", "")
            if not ref_text:
                print(f"Error: ref_sample_text missing from training_meta.json")
                return False

            print(f"TTS [local lora] generating for adapter={os.path.basename(adapter_path)}, "
                  f"text='{text[:50]}...'")

            model = self._init_local_lora(adapter_path)

            # Build or reuse voice clone prompt for this adapter
            if adapter_path not in self._lora_prompt_cache:
                audio_array, sample_rate = sf.read(ref_wav_path)
                if audio_array.ndim > 1:
                    audio_array = audio_array.mean(axis=1)
                print(f"Creating clone prompt for LoRA adapter...")
                prompt = model.create_voice_clone_prompt(
                    ref_audio=(audio_array, sample_rate),
                    ref_text=ref_text,
                )
                self._lora_prompt_cache[adapter_path] = prompt
                print(f"Clone prompt cached for LoRA adapter.")

            prompt = self._lora_prompt_cache[adapter_path]

            # Build instruct_ids so the Base model can follow style prompts
            gen_extra = {}
            instruct = instruct_text or ""
            character_style = voice_data.get("character_style", "") or voice_data.get("default_style", "")
            if character_style:
                instruct = f"{instruct} {character_style}".strip()
            if instruct:
                instruct_formatted = f"<|im_start|>user\n{instruct}<|im_end|>\n"
                gen_extra["instruct_ids"] = model._tokenize_texts([instruct_formatted])

            t_start = time.time()
            max_new_tokens = self._qwen_max_new_tokens_for_text(text)
            print(
                f"TTS [local lora] generation budget: {max_new_tokens} tokens "
                f"for ~{self._estimate_generation_seconds_for_text(text):.1f}s max audio"
            )
            with self._lora_inference_lock:
                wavs, sr = model.generate_voice_clone(
                    text=text,
                    voice_clone_prompt=prompt,
                    non_streaming_mode=True,
                    max_new_tokens=max_new_tokens,
                    **gen_extra,
                )
            gen_time = time.time() - t_start

            if wavs is None or len(wavs) == 0:
                print(f"Error: No audio generated for: '{text[:50]}...'")
                return False

            audio = np.concatenate(wavs) if len(wavs) > 1 else wavs[0]
            duration = len(audio) / sr
            rtf = duration / gen_time if gen_time > 0 else 0
            print(f"TTS [local lora] done: {gen_time:.1f}s -> {duration:.1f}s audio ({rtf:.2f}x real-time)")
            self._save_wav(audio, sr, output_path)
            return True

        except Exception as e:
            print(f"Error generating LoRA voice: {e}")
            return False

    # ── Batch generation ─────────────────────────────────────────

    def generate_batch(self, chunks, voice_config, output_dir, batch_seed=-1, cancel_check=None, log_callback=None):
        return self._provider.generate_batch(
            chunks,
            voice_config,
            output_dir,
            batch_seed=batch_seed,
            cancel_check=cancel_check,
            log_callback=log_callback,
        )

    def _provider_generate_batch(self, chunks, voice_config, output_dir, batch_seed=-1, cancel_check=None, log_callback=None):
        """Generate multiple audio files.

        Local mode: uses native list-based batch API for custom voices.
        External mode: sequential individual calls.

        Args:
            chunks: List of dicts with 'text', 'instruct', 'speaker', 'index' keys
            voice_config: Voice configuration dict
            output_dir: Directory to save output files
            batch_seed: Single seed for all generations (-1 for random)

        Returns:
            dict with 'completed' (list of indices) and 'failed' (list of (index, error) tuples)
        """
        results = {"completed": [], "failed": []}

        if not chunks:
            return results

        def _cancel_requested():
            try:
                return bool(cancel_check and cancel_check())
            except Exception:
                return False

        if self._mode == "local" and self._resolve_local_backend() == "mlx":
            for chunk in chunks:
                if _cancel_requested():
                    print("[CANCEL] Stopping MLX local batch generation.")
                    break
                idx = chunk["index"]
                output_path = os.path.join(output_dir, f"temp_batch_{idx}.wav")
                try:
                    success = self.generate_voice(
                        chunk.get("text", ""),
                        chunk.get("instruct", ""),
                        chunk.get("speaker", ""),
                        voice_config,
                        output_path,
                    )
                    if success:
                        results["completed"].append(idx)
                    else:
                        results["failed"].append((idx, "MLX local generation failed"))
                except Exception as e:
                    results["failed"].append((idx, str(e)))
            return results

        # Reset torch.compile state to prevent progressive slowdown
        # from dynamo guard accumulation across batches
        if self._compile_codec_enabled:
            self._reset_compile_cache()

        # Separate chunks by voice type
        custom_chunks = []
        clone_chunks = []
        lora_chunks = []
        design_chunks = []

        for chunk in chunks:
            speaker = chunk.get("speaker")
            voice_data = voice_config.get(speaker, {})
            voice_type = voice_data.get("type", "custom")

            if voice_type == "clone":
                clone_chunks.append(chunk)
            elif voice_type in ("lora", "builtin_lora"):
                lora_chunks.append(chunk)
            elif voice_type == "design":
                design_chunks.append(chunk)
            else:
                custom_chunks.append(chunk)
        # Process custom voice chunks
        if custom_chunks and not _cancel_requested():
            if self._mode == "local":
                batch_results = self._local_batch_custom(
                    custom_chunks,
                    voice_config,
                    output_dir,
                    batch_seed,
                    cancel_check=cancel_check,
                    log_callback=log_callback,
                )
            else:
                batch_results = self._sequential_custom(
                    custom_chunks,
                    voice_config,
                    output_dir,
                    batch_seed,
                    cancel_check=cancel_check,
                )
            results["completed"].extend(batch_results["completed"])
            results["failed"].extend(batch_results["failed"])
            self._clear_gpu_cache()

        # Process clone voice chunks (batched by speaker in local mode)
        if clone_chunks and not _cancel_requested():
            if self._mode == "local":
                batch_results = self._local_batch_clone(
                    clone_chunks,
                    voice_config,
                    output_dir,
                    cancel_check=cancel_check,
                    log_callback=log_callback,
                )
            else:
                batch_results = {"completed": [], "failed": []}
                for chunk in clone_chunks:
                    if _cancel_requested():
                        print("[CANCEL] Stopping clone voice batch generation.")
                        break
                    idx = chunk["index"]
                    output_path = os.path.join(output_dir, f"temp_batch_{idx}.wav")
                    try:
                        success = self.generate_clone_voice(
                            chunk["text"], chunk["speaker"], voice_config, output_path
                        )
                        if success:
                            batch_results["completed"].append(idx)
                        else:
                            batch_results["failed"].append((idx, "Clone voice generation failed"))
                    except Exception as e:
                        batch_results["failed"].append((idx, str(e)))
            results["completed"].extend(batch_results["completed"])
            results["failed"].extend(batch_results["failed"])
            self._clear_gpu_cache()

        # Process LoRA voice chunks (batched by adapter in local mode)
        if lora_chunks and not _cancel_requested():
            if self._mode == "local":
                batch_results = self._local_batch_lora(
                    lora_chunks,
                    voice_config,
                    output_dir,
                    cancel_check=cancel_check,
                    log_callback=log_callback,
                )
            else:
                batch_results = {"completed": [], "failed": []}
                for chunk in lora_chunks:
                    if _cancel_requested():
                        print("[CANCEL] Stopping LoRA batch generation.")
                        break
                    idx = chunk["index"]
                    output_path = os.path.join(output_dir, f"temp_batch_{idx}.wav")
                    speaker = chunk.get("speaker")
                    voice_data = voice_config.get(speaker, {})
                    try:
                        success = self.generate_lora_voice(
                            text=chunk["text"],
                            instruct_text=chunk.get("instruct", ""),
                            voice_data=voice_data,
                            output_path=output_path,
                        )
                        if success:
                            batch_results["completed"].append(idx)
                        else:
                            batch_results["failed"].append((idx, "LoRA voice generation failed"))
                    except Exception as e:
                        batch_results["failed"].append((idx, str(e)))
            results["completed"].extend(batch_results["completed"])
            results["failed"].extend(batch_results["failed"])
            self._clear_gpu_cache()

        # Process design voice chunks (sequential — each line has unique description)
        if design_chunks and not _cancel_requested():
            for chunk in design_chunks:
                if _cancel_requested():
                    print("[CANCEL] Stopping voice design batch generation.")
                    break
                idx = chunk["index"]
                output_path = os.path.join(output_dir, f"temp_batch_{idx}.wav")
                speaker = chunk.get("speaker")
                voice_data = voice_config.get(speaker, {})
                try:
                    success = self.generate_design_voice(
                        text=chunk["text"],
                        instruct_text=chunk.get("instruct", ""),
                        voice_data=voice_data,
                        output_path=output_path,
                    )
                    if success:
                        results["completed"].append(idx)
                    else:
                        results["failed"].append((idx, "Design voice generation failed"))
                except Exception as e:
                    results["failed"].append((idx, str(e)))

        return results

    # ── Connection test ──────────────────────────────────────────

    # ── Local backend methods ────────────────────────────────────

    def _mlx_generate_custom(self, text, instruct_text, speaker, voice_config, output_path):
        """Generate custom voice audio using local MLX backend."""
        try:
            voice_data = voice_config.get(speaker) or {}
            voice_name = self._mlx_voice_name(voice_data.get("voice", "Ryan"))
            default_style = voice_data.get("default_style", "")
            instruct = instruct_text if instruct_text else (default_style if default_style else "Normal tone")
            model = self._init_local_mlx_model("custom_voice")
            audio, sr = self._mlx_generate_with_temp_dir(
                model,
                text=text,
                voice=voice_name,
                instruct=instruct,
                lang_code=self._language,
                speed=1.0,
            )
            self._save_wav(audio, sr, output_path)
            return True
        except Exception as e:
            print(f"Error generating custom voice for '{speaker}' with MLX backend: {e}")
            return False

    def _mlx_generate_clone(self, text, speaker, voice_config, output_path, instruct_text=""):
        """Generate clone voice audio using local MLX backend."""
        try:
            voice_data = voice_config.get(speaker, {})
            ref_audio_path = voice_data.get("ref_audio")
            ref_text = voice_data.get("generated_ref_text") or voice_data.get("ref_text") or "."
            character_style = (voice_data.get("character_style") or voice_data.get("default_style") or "").strip()
            line_instruct = (instruct_text or "").strip()
            if line_instruct and character_style:
                instruct = f"{line_instruct} {character_style}".strip()
            elif line_instruct:
                instruct = line_instruct
            else:
                instruct = character_style

            if not ref_audio_path:
                print(f"Warning: Clone voice for '{speaker}' missing ref_audio. Skipping.")
                return False

            ref_audio_path = self._resolve_project_path(ref_audio_path)

            if not os.path.exists(ref_audio_path):
                print(f"Warning: Reference audio not found for '{speaker}': {ref_audio_path}")
                return False

            model = self._init_local_mlx_model("base")
            kwargs = {
                "text": text,
                "ref_audio": ref_audio_path,
                "ref_text": ref_text,
                "lang_code": self._language,
            }
            if instruct:
                kwargs["instruct"] = instruct
            try:
                audio, sr = self._mlx_generate_with_temp_dir(
                    model,
                    **kwargs,
                )
            except TypeError as e:
                if "instruct" in kwargs:
                    kwargs.pop("instruct", None)
                    audio, sr = self._mlx_generate_with_temp_dir(
                        model,
                        **kwargs,
                    )
                else:
                    raise
            self._save_wav(audio, sr, output_path)
            return True
        except Exception as e:
            print(f"Error generating clone voice for '{speaker}' with MLX backend: {e}")
            return False

    def _local_generate_custom(self, text, instruct_text, speaker, voice_config, output_path):
        """Generate custom voice audio using local Qwen3-TTS model."""
        try:
            import torch

            voice_data = voice_config.get(speaker)
            if not voice_data:
                print(f"Warning: No voice configuration for '{speaker}'. Skipping.")
                return False

            voice = voice_data.get("voice", "Ryan")
            default_style = voice_data.get("default_style", "")
            seed = int(voice_data.get("seed", -1))

            instruct = instruct_text if instruct_text else (default_style if default_style else "neutral")

            import time

            print(f"TTS [local] generating with instruct='{instruct}' for text='{text[:50]}...'")

            model = self._init_local_custom()

            if seed >= 0:
                torch.manual_seed(seed)

            t_start = time.time()
            max_new_tokens = self._qwen_max_new_tokens_for_text(text)
            print(
                f"TTS [local] generation budget: {max_new_tokens} tokens "
                f"for ~{self._estimate_generation_seconds_for_text(text):.1f}s max audio"
            )
            with self._custom_inference_lock:
                wavs, sr = model.generate_custom_voice(
                    text=text,
                    language=self._language,
                    speaker=voice,
                    instruct=instruct,
                    non_streaming_mode=True,
                    max_new_tokens=max_new_tokens,
                )
            gen_time = time.time() - t_start

            if wavs is None or len(wavs) == 0:
                print(f"Error: No audio generated for: '{text[:50]}...'")
                return False

            # wavs is a list of numpy arrays; concatenate them
            audio = np.concatenate(wavs) if len(wavs) > 1 else wavs[0]
            duration = len(audio) / sr
            rtf = duration / gen_time if gen_time > 0 else 0
            print(f"TTS [local] done: {gen_time:.1f}s -> {duration:.1f}s audio ({rtf:.2f}x real-time)")
            self._save_wav(audio, sr, output_path)
            return True

        except Exception as e:
            print(f"Error generating custom voice for '{speaker}': {e}")
            return False

    def _local_generate_clone(self, text, speaker, voice_config, output_path):
        """Generate voice-cloned audio using local Qwen3-TTS Base model."""
        try:
            import torch

            voice_data = voice_config.get(speaker)
            if not voice_data:
                print(f"Warning: No voice configuration for '{speaker}'. Skipping.")
                return False

            seed = int(voice_data.get("seed", -1))

            import time

            print(f"TTS [local clone] generating for speaker='{speaker}', text='{text[:50]}...'")

            prompt = self._get_clone_prompt(speaker, voice_config)
            model = self._init_local_clone()

            if seed >= 0:
                torch.manual_seed(seed)

            t_start = time.time()
            max_new_tokens = self._qwen_max_new_tokens_for_text(text)
            print(
                f"TTS [local clone] generation budget: {max_new_tokens} tokens "
                f"for ~{self._estimate_generation_seconds_for_text(text):.1f}s max audio"
            )
            with self._clone_inference_lock:
                wavs, sr = model.generate_voice_clone(
                    text=text,
                    voice_clone_prompt=prompt,
                    non_streaming_mode=True,
                    max_new_tokens=max_new_tokens,
                )
            gen_time = time.time() - t_start

            if wavs is None or len(wavs) == 0:
                print(f"Error: No audio generated for: '{text[:50]}...'")
                return False

            audio = np.concatenate(wavs) if len(wavs) > 1 else wavs[0]
            duration = len(audio) / sr
            rtf = duration / gen_time if gen_time > 0 else 0
            print(f"TTS [local clone] done: {gen_time:.1f}s -> {duration:.1f}s audio ({rtf:.2f}x real-time)")
            self._save_wav(audio, sr, output_path)
            return True

        except Exception as e:
            print(f"Error generating clone voice for '{speaker}': {e}")
            return False

    def _local_batch_custom(self, chunks, voice_config, output_dir, batch_seed=-1, cancel_check=None, log_callback=None):
        """Batch generate custom voice using native list API with sub-batching.

        Autoregressive batch generation runs for as long as the longest sequence.
        Shorter sequences waste compute on padding. To minimize this, chunks are
        sorted by text length and split into sub-batches when the length ratio
        exceeds the configured threshold. Sub-batching can be disabled entirely
        via config, in which case everything runs as one batch.
        """
        import torch
        import time

        results = {"completed": [], "failed": []}

        texts = []
        speakers = []
        instructs = []
        indices = []
        display_ids = []

        for chunk in chunks:
            idx = chunk["index"]
            text = chunk.get("text", "")
            instruct_text = chunk.get("instruct", "")
            speaker_name = chunk.get("speaker", "")

            voice_data = voice_config.get(speaker_name, {})
            voice = voice_data.get("voice", "Ryan")
            character_style = voice_data.get("character_style", "") or voice_data.get("default_style", "")

            instruct = instruct_text if instruct_text else "neutral"
            if character_style:
                instruct = f"{instruct} {character_style}"

            texts.append(text)
            speakers.append(voice)
            instructs.append(instruct)
            indices.append(idx)
            display_ids.append(chunk.get("display_id") if chunk.get("display_id") is not None else idx)

        total_text_chars = sum(len(t) for t in texts)

        # Sort by text length to group similar-length chunks together.
        # This reduces wasted padding during autoregressive generation
        # (the LLM runs until ALL sequences finish, so short chunks
        # waste compute waiting for long ones).
        sort_order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        texts = [texts[i] for i in sort_order]
        speakers = [speakers[i] for i in sort_order]
        instructs = [instructs[i] for i in sort_order]
        indices = [indices[i] for i in sort_order]
        display_ids = [display_ids[i] for i in sort_order]

        model = self._init_local_custom()

        # Warmup on first batch to pre-tune MIOpen/GPU solvers
        if self._warmup_needed:
            print("Running batch warmup generation...")
            self._warmup_model(model)
            self._warmup_needed = False

        # Clear stale GPU cache from any prior generation to avoid
        # fragmented VRAM blocking large batch allocations (ROCm especially).
        self._clear_gpu_cache()

        max_items = self._estimate_max_batch_size(
            model, max_text_chars=len(texts[-1]),
        )
        sub_batches = self._build_sub_batches(texts, max_items=max_items)

        print(f"Batch [local]: generating {len(texts)} chunks ({total_text_chars} chars) "
              f"in {len(sub_batches)} sub-batch(es)...")

        t_total_start = time.time()
        total_audio_duration = 0.0

        for sb_idx, (start, end) in enumerate(sub_batches):
            if cancel_check and cancel_check():
                print("[CANCEL] Stopping custom voice sub-batch generation.")
                break
            sb_texts = texts[start:end]
            sb_speakers = speakers[start:end]
            sb_instructs = instructs[start:end]
            sb_indices = indices[start:end]
            sb_display_ids = display_ids[start:end]
            sb_chars = sum(len(t) for t in sb_texts)
            sb_text_lengths = [len(text) for text in sb_texts]
            sb_summary = self._describe_batch_targets(
                chunk_ids=sb_display_ids,
                chunk_uids=sb_indices,
                text_lengths=sb_text_lengths,
            )

            self._emit_log(
                f"  Sub-batch {sb_idx+1}/{len(sub_batches)} [custom]: {len(sb_texts)} chunks "
                f"({sb_chars} chars, {len(sb_texts[0])}-{len(sb_texts[-1])} chars/chunk); {sb_summary}",
                log_callback=log_callback,
            )

            try:
                if batch_seed >= 0:
                    torch.manual_seed(batch_seed)

                t_start = time.time()
                max_new_tokens = self._qwen_max_new_tokens_for_texts(sb_texts)
                self._emit_log(
                    f"  Sub-batch {sb_idx+1}/{len(sub_batches)} [custom] token budget={max_new_tokens}",
                    log_callback=log_callback,
                )
                with self._progress_log_context(
                    lambda elapsed: (
                        f"  Sub-batch {sb_idx+1}/{len(sub_batches)} [custom] active {elapsed:.1f}s; "
                        f"{sb_summary}"
                    ),
                    log_callback=log_callback,
                ):
                    wavs_list, sr = model.generate_custom_voice(
                        text=sb_texts,
                        language=[self._language] * len(sb_texts),
                        speaker=sb_speakers,
                        instruct=sb_instructs,
                        non_streaming_mode=True,
                        max_new_tokens=max_new_tokens,
                    )
                gen_time = time.time() - t_start

                if wavs_list is None:
                    for idx in sb_indices:
                        results["failed"].append((idx, "Batch returned None"))
                    continue

                sb_audio_duration = self._persist_batch_audio_outputs(
                    wavs_list,
                    sr,
                    output_dir,
                    sb_indices,
                    results,
                )

                total_audio_duration += sb_audio_duration
                sb_rtf = sb_audio_duration / gen_time if gen_time > 0 else 0
                print(f"  Sub-batch {sb_idx+1} done: {gen_time:.1f}s -> {sb_audio_duration:.1f}s audio ({sb_rtf:.2f}x RT)")

            except Exception as e:
                print(f"  Sub-batch {sb_idx+1} failed: {e}")
                for idx in sb_indices:
                    results["failed"].append((idx, f"Batch error: {e}"))

            # Free GPU memory between sub-batches to prevent VRAM exhaustion
            self._clear_gpu_cache()

        total_time = time.time() - t_total_start
        rtf = total_audio_duration / total_time if total_time > 0 else 0
        print(f"Batch total: {total_time:.1f}s -> {total_audio_duration:.1f}s audio ({rtf:.2f}x real-time)")

        return results

    def _local_batch_clone(self, chunks, voice_config, output_dir, cancel_check=None, log_callback=None):
        """Batch generate clone voices, grouped by speaker.

        Chunks sharing the same speaker (same reference audio) are batched
        together through generate_voice_clone(text=[list], ...).
        Sub-batching by text length is applied within each speaker group.
        """
        import torch
        import time

        results = {"completed": [], "failed": []}

        # Group chunks by speaker
        speaker_groups = {}
        for chunk in chunks:
            speaker = chunk.get("speaker", "")
            speaker_groups.setdefault(speaker, []).append(chunk)

        model = self._init_local_clone()

        # Warmup on first batch to pre-tune MIOpen/GPU solvers
        if self._warmup_needed:
            print("Running batch warmup generation...")
            self._warmup_model(model)
            self._warmup_needed = False

        self._clear_gpu_cache()

        t_total_start = time.time()
        total_audio_duration = 0.0

        for speaker, group in speaker_groups.items():
            if cancel_check and cancel_check():
                print("[CANCEL] Stopping clone voice grouped batch generation.")
                break
            try:
                prompt = self._get_clone_prompt(speaker, voice_config)
            except Exception as e:
                print(f"  Error building clone prompt for '{speaker}': {e}")
                for chunk in group:
                    results["failed"].append((chunk["index"], str(e)))
                continue

            texts = [c["text"] for c in group]
            indices = [c["index"] for c in group]
            display_ids = [c.get("display_id") if c.get("display_id") is not None else c["index"] for c in group]

            # Sort by text length for sub-batching efficiency
            sort_order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
            texts = [texts[i] for i in sort_order]
            indices = [indices[i] for i in sort_order]
            display_ids = [display_ids[i] for i in sort_order]

            # Estimate max batch size from VRAM + clone prompt overhead
            clone_tokens = prompt[0].ref_code.shape[0] if prompt[0].ref_code is not None else 0
            ref_text_chars = len(prompt[0].ref_text) if prompt[0].ref_text else 0
            max_items = self._estimate_max_batch_size(
                model,
                clone_tokens,
                ref_text_chars,
                len(texts[-1]),
                max_new_tokens=self._qwen_max_new_tokens_for_text(texts[-1]),
            )
            sub_batches = self._build_sub_batches(texts, max_items=max_items)

            print(f"Batch [clone] speaker='{speaker}': {len(texts)} chunks "
                  f"in {len(sub_batches)} sub-batch(es)")

            for sb_idx, (start, end) in enumerate(sub_batches):
                if cancel_check and cancel_check():
                    print("[CANCEL] Stopping clone voice sub-batch generation.")
                    break
                sb_texts = texts[start:end]
                sb_indices = indices[start:end]
                sb_display_ids = display_ids[start:end]
                sb_text_lengths = [len(text) for text in sb_texts]
                sb_summary = self._describe_batch_targets(
                    chunk_ids=sb_display_ids,
                    chunk_uids=sb_indices,
                    text_lengths=sb_text_lengths,
                )

                self._emit_log(
                    f"  Sub-batch {sb_idx+1}/{len(sub_batches)} [clone speaker='{speaker}']: "
                    f"{len(sb_texts)} chunks ({len(sb_texts[0])}-{len(sb_texts[-1])} chars/chunk); {sb_summary}",
                    log_callback=log_callback,
                )

                try:
                    t_start = time.time()
                    max_new_tokens = self._qwen_max_new_tokens_for_texts(sb_texts)
                    self._emit_log(
                        f"  Sub-batch {sb_idx+1}/{len(sub_batches)} [clone speaker='{speaker}'] "
                        f"token budget={max_new_tokens}",
                        log_callback=log_callback,
                    )
                    with self._progress_log_context(
                        lambda elapsed: (
                            f"  Sub-batch {sb_idx+1}/{len(sub_batches)} [clone speaker='{speaker}'] "
                            f"active {elapsed:.1f}s; {sb_summary}"
                        ),
                        log_callback=log_callback,
                    ):
                        wavs_list, sr = model.generate_voice_clone(
                            text=sb_texts,
                            voice_clone_prompt=prompt,
                            non_streaming_mode=True,
                            max_new_tokens=max_new_tokens,
                        )
                    gen_time = time.time() - t_start

                    if wavs_list is None:
                        for idx in sb_indices:
                            results["failed"].append((idx, "Batch returned None"))
                        continue

                    sb_audio_duration = self._persist_batch_audio_outputs(
                        wavs_list,
                        sr,
                        output_dir,
                        sb_indices,
                        results,
                    )

                    total_audio_duration += sb_audio_duration
                    sb_rtf = sb_audio_duration / gen_time if gen_time > 0 else 0
                    print(f"  Sub-batch {sb_idx+1} done: {gen_time:.1f}s -> {sb_audio_duration:.1f}s audio ({sb_rtf:.2f}x RT)")

                except Exception as e:
                    print(f"  Sub-batch {sb_idx+1} failed: {e}")
                    for idx in sb_indices:
                        results["failed"].append((idx, f"Batch error: {e}"))

                self._clear_gpu_cache()

        total_time = time.time() - t_total_start
        rtf = total_audio_duration / total_time if total_time > 0 else 0
        print(f"Batch [clone] total: {total_time:.1f}s -> {total_audio_duration:.1f}s audio ({rtf:.2f}x real-time)")

        return results

    def _local_batch_lora(self, chunks, voice_config, output_dir, cancel_check=None, log_callback=None):
        """Batch generate LoRA voices, grouped by adapter.

        Chunks sharing the same adapter are batched together through
        generate_voice_clone(text=[list], instruct_ids=[list], ...).
        Sub-batching by text length is applied within each adapter group.
        """
        import torch
        import time

        results = {"completed": [], "failed": []}
        root_dir = os.path.dirname(os.path.dirname(__file__))

        # Group chunks by adapter_path (resolved to absolute)
        adapter_groups = {}  # adapter_path -> (voice_data, [chunks])
        for chunk in chunks:
            speaker = chunk.get("speaker", "")
            voice_data = voice_config.get(speaker, {})
            adapter_path = voice_data.get("adapter_path", "")

            if not adapter_path:
                results["failed"].append((chunk["index"], "No adapter_path"))
                continue

            if not os.path.isabs(adapter_path):
                adapter_path = os.path.join(root_dir, adapter_path)

            if adapter_path not in adapter_groups:
                adapter_groups[adapter_path] = (voice_data, [])
            adapter_groups[adapter_path][1].append(chunk)

        self._clear_gpu_cache()

        # Warmup on first batch to pre-tune MIOpen/GPU solvers
        if self._warmup_needed:
            warmup_model = self._init_local_clone()
            print("Running batch warmup generation...")
            self._warmup_model(warmup_model)
            self._warmup_needed = False

        t_total_start = time.time()
        total_audio_duration = 0.0

        for adapter_path, (voice_data, group) in adapter_groups.items():
            if cancel_check and cancel_check():
                print("[CANCEL] Stopping LoRA grouped batch generation.")
                break
            if not os.path.isdir(adapter_path):
                print(f"  Error: adapter path not found: {adapter_path}")
                for chunk in group:
                    results["failed"].append((chunk["index"], f"Adapter not found: {adapter_path}"))
                continue

            # Load adapter and build/get clone prompt
            try:
                ref_wav_path = os.path.join(adapter_path, "ref_sample.wav")
                meta_path = os.path.join(adapter_path, "training_meta.json")
                if not os.path.exists(ref_wav_path) or not os.path.exists(meta_path):
                    raise FileNotFoundError(f"Missing ref_sample.wav or training_meta.json in {adapter_path}")

                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                ref_text = meta.get("ref_sample_text", "")
                if not ref_text:
                    raise ValueError("ref_sample_text missing from training_meta.json")

                model = self._init_local_lora(adapter_path)

                if adapter_path not in self._lora_prompt_cache:
                    audio_array, sample_rate = sf.read(ref_wav_path)
                    if audio_array.ndim > 1:
                        audio_array = audio_array.mean(axis=1)
                    print(f"Creating clone prompt for LoRA adapter...")
                    prompt = model.create_voice_clone_prompt(
                        ref_audio=(audio_array, sample_rate),
                        ref_text=ref_text,
                    )
                    self._lora_prompt_cache[adapter_path] = prompt
                    print(f"Clone prompt cached for LoRA adapter.")

                prompt = self._lora_prompt_cache[adapter_path]
            except Exception as e:
                print(f"  Error loading LoRA adapter {os.path.basename(adapter_path)}: {e}")
                for chunk in group:
                    results["failed"].append((chunk["index"], str(e)))
                continue

            character_style = voice_data.get("character_style", "") or voice_data.get("default_style", "")

            texts = [c["text"] for c in group]
            instructs_raw = [c.get("instruct", "") for c in group]
            indices = [c["index"] for c in group]
            display_ids = [c.get("display_id") if c.get("display_id") is not None else c["index"] for c in group]

            # Sort by text length
            sort_order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
            texts = [texts[i] for i in sort_order]
            instructs_raw = [instructs_raw[i] for i in sort_order]
            indices = [indices[i] for i in sort_order]
            display_ids = [display_ids[i] for i in sort_order]

            # Estimate max batch size from VRAM + clone prompt overhead
            clone_tokens = prompt[0].ref_code.shape[0] if prompt[0].ref_code is not None else 0
            ref_text_chars = len(prompt[0].ref_text) if prompt[0].ref_text else 0
            max_items = self._estimate_max_batch_size(
                model,
                clone_tokens,
                ref_text_chars,
                len(texts[-1]),
                max_new_tokens=self._qwen_max_new_tokens_for_text(texts[-1]),
            )
            sub_batches = self._build_sub_batches(texts, max_items=max_items)

            print(f"Batch [lora] adapter='{os.path.basename(adapter_path)}': {len(texts)} chunks "
                  f"in {len(sub_batches)} sub-batch(es)")

            for sb_idx, (start, end) in enumerate(sub_batches):
                if cancel_check and cancel_check():
                    print("[CANCEL] Stopping LoRA sub-batch generation.")
                    break
                sb_texts = texts[start:end]
                sb_instructs = instructs_raw[start:end]
                sb_indices = indices[start:end]
                sb_display_ids = display_ids[start:end]
                sb_text_lengths = [len(text) for text in sb_texts]
                sb_summary = self._describe_batch_targets(
                    chunk_ids=sb_display_ids,
                    chunk_uids=sb_indices,
                    text_lengths=sb_text_lengths,
                )

                self._emit_log(
                    f"  Sub-batch {sb_idx+1}/{len(sub_batches)} [lora adapter='{os.path.basename(adapter_path)}']: "
                    f"{len(sb_texts)} chunks ({len(sb_texts[0])}-{len(sb_texts[-1])} chars/chunk); {sb_summary}",
                    log_callback=log_callback,
                )

                try:
                    # Build instruct_ids list for this sub-batch
                    instruct_ids = []
                    for inst in sb_instructs:
                        instruct = inst or ""
                        if character_style:
                            instruct = f"{instruct} {character_style}".strip()
                        if instruct:
                            instruct_formatted = f"<|im_start|>user\n{instruct}<|im_end|>\n"
                            instruct_ids.append(model._tokenize_texts([instruct_formatted])[0])
                        else:
                            instruct_ids.append(None)

                    gen_extra = {}
                    if any(iid is not None for iid in instruct_ids):
                        gen_extra["instruct_ids"] = instruct_ids

                    t_start = time.time()
                    max_new_tokens = self._qwen_max_new_tokens_for_texts(sb_texts)
                    self._emit_log(
                        f"  Sub-batch {sb_idx+1}/{len(sub_batches)} "
                        f"[lora adapter='{os.path.basename(adapter_path)}'] token budget={max_new_tokens}",
                        log_callback=log_callback,
                    )
                    with self._progress_log_context(
                        lambda elapsed: (
                            f"  Sub-batch {sb_idx+1}/{len(sub_batches)} "
                            f"[lora adapter='{os.path.basename(adapter_path)}'] active {elapsed:.1f}s; "
                            f"{sb_summary}"
                        ),
                        log_callback=log_callback,
                    ):
                        wavs_list, sr = model.generate_voice_clone(
                            text=sb_texts,
                            voice_clone_prompt=prompt,
                            non_streaming_mode=True,
                            max_new_tokens=max_new_tokens,
                            **gen_extra,
                        )
                    gen_time = time.time() - t_start

                    if wavs_list is None:
                        for idx in sb_indices:
                            results["failed"].append((idx, "Batch returned None"))
                        continue

                    sb_audio_duration = self._persist_batch_audio_outputs(
                        wavs_list,
                        sr,
                        output_dir,
                        sb_indices,
                        results,
                    )

                    total_audio_duration += sb_audio_duration
                    sb_rtf = sb_audio_duration / gen_time if gen_time > 0 else 0
                    print(f"  Sub-batch {sb_idx+1} done: {gen_time:.1f}s -> {sb_audio_duration:.1f}s audio ({sb_rtf:.2f}x RT)")

                except Exception as e:
                    print(f"  Sub-batch {sb_idx+1} failed: {e}")
                    for idx in sb_indices:
                        results["failed"].append((idx, f"Batch error: {e}"))

                self._clear_gpu_cache()

        total_time = time.time() - t_total_start
        rtf = total_audio_duration / total_time if total_time > 0 else 0
        print(f"Batch [lora] total: {total_time:.1f}s -> {total_audio_duration:.1f}s audio ({rtf:.2f}x real-time)")

        return results

    # ── External backend methods ─────────────────────────────────

    def _external_generate_custom(self, text, instruct_text, speaker, voice_config, output_path):
        """Generate custom voice audio via external Gradio server."""
        try:
            voice_data = voice_config.get(speaker)
            if not voice_data:
                print(f"Warning: No voice configuration for '{speaker}'. Skipping.")
                return False

            voice = voice_data.get("voice", "Ryan")
            default_style = voice_data.get("default_style", "")
            seed = int(voice_data.get("seed", -1))

            instruct = instruct_text if instruct_text else (default_style if default_style else "neutral")

            print(f"TTS [external] generating with instruct='{instruct}' for text='{text[:50]}...'")

            backend = self._init_external()
            if self._external_backend == "qwen_mlx_http":
                result = self._external_http_post("/api/v1/custom-voice/generate", {
                    "text": text,
                    "language": self._language,
                    "speaker": voice,
                    "instruct": instruct,
                    "speed": 1.0,
                    "response_format": "base64",
                })
                self._write_base64_audio(result.get("audio"), output_path)
                return True

            result = backend.predict(
                text=text,
                language=self._language,
                speaker=voice,
                instruct=instruct,
                model_size="1.7B",
                seed=seed,
                api_name="/generate_custom_voice"
            )

            generated_audio_filepath = result[0]
            if not generated_audio_filepath or not os.path.exists(generated_audio_filepath):
                print(f"Error: No audio file generated for: '{text[:50]}...'")
                return False

            if os.path.getsize(generated_audio_filepath) == 0:
                print(f"Error: Generated audio file is empty for: '{text[:50]}...'")
                return False

            shutil.copy(generated_audio_filepath, output_path)
            return True

        except Exception as e:
            print(f"Error generating custom voice for '{speaker}': {e}")
            return False

    def _external_generate_design(self, text, instruct_text, voice_data, output_path):
        """Generate voice-design audio via external server."""
        try:
            base_desc = (voice_data.get("description") or "").strip()
            instruct = (instruct_text or "").strip()

            if base_desc and instruct:
                description = f"{base_desc}, {instruct}"
            elif base_desc:
                description = base_desc
            elif instruct:
                description = instruct
            else:
                description = "A clear, natural speaking voice"

            self._init_external()
            if self._external_backend == "qwen_mlx_http":
                result = self._external_http_post("/api/v1/voice-design/generate", {
                    "text": text,
                    "language": self._language,
                    "instruct": description,
                    "speed": 1.0,
                    "response_format": "base64",
                })
                self._write_base64_audio(result.get("audio"), output_path)
                return True

            raise ValueError("Voice design is only supported with the MLX HTTP external API")
        except Exception as e:
            print(f"Error generating design voice: {e}")
            return False

    def _external_generate_voice_design_preview(self, description, sample_text, language=None):
        """Generate a reusable voice preview via the external Qwen HTTP API."""
        import time

        lang = language or self._language
        t_start = time.time()

        self._init_external()
        if self._external_backend != "qwen_mlx_http":
            raise ValueError("Voice design preview is only supported with the MLX HTTP external API")

        result = self._external_http_post("/api/v1/voice-design/generate", {
            "text": sample_text,
            "language": lang,
            "instruct": description,
            "speed": 1.0,
            "response_format": "base64",
        })

        wav_path = self._new_voice_design_preview_path()
        self._write_base64_audio(result.get("audio"), wav_path)
        audio, sr = sf.read(wav_path)
        duration = len(audio) / sr if sr else 0
        gen_time = time.time() - t_start
        print(f"VoiceDesign [external] done in {gen_time:.1f}s -> {duration:.1f}s audio")
        return wav_path, sr

    def _external_generate_clone(self, text, speaker, voice_config, output_path):
        """Generate voice-cloned audio via external Gradio server."""
        try:
            from gradio_client import handle_file

            voice_data = voice_config.get(speaker)
            if not voice_data:
                print(f"Warning: No voice configuration for '{speaker}'. Skipping.")
                return False

            ref_audio = voice_data.get("ref_audio")
            ref_text = voice_data.get("generated_ref_text") or voice_data.get("ref_text")
            seed = int(voice_data.get("seed", -1))

            if not ref_audio or not ref_text:
                print(f"Warning: Clone voice for '{speaker}' missing ref_audio or ref_text. Skipping.")
                return False

            ref_audio = self._resolve_project_path(ref_audio)

            if not os.path.exists(ref_audio):
                print(f"Warning: Reference audio not found for '{speaker}': {ref_audio}")
                return False

            backend = self._init_external()

            if self._external_backend == "qwen_mlx_http":
                with open(ref_audio, "rb") as f:
                    ref_audio_b64 = base64.b64encode(f.read()).decode("ascii")
                result = self._external_http_post("/api/v1/base/clone", {
                    "text": text,
                    "language": self._language,
                    "ref_audio_base64": ref_audio_b64,
                    "ref_text": ref_text or None,
                    "x_vector_only_mode": not bool(ref_text),
                    "speed": 1.0,
                    "response_format": "base64",
                })
                self._write_base64_audio(result.get("audio"), output_path)
                return True

            result = backend.predict(
                handle_file(ref_audio),
                ref_text,
                text,
                "Auto",
                False,       # use_xvector_only
                "1.7B",
                200,         # max_chunk_chars
                0,           # chunk_gap
                seed,
                api_name="/generate_voice_clone"
            )

            generated_audio_filepath = result[0]
            if not generated_audio_filepath or not os.path.exists(generated_audio_filepath):
                print(f"Error: No audio file generated for: '{text[:50]}...'")
                return False

            if os.path.getsize(generated_audio_filepath) == 0:
                print(f"Error: Generated audio file is empty for: '{text[:50]}...'")
                return False

            shutil.copy(generated_audio_filepath, output_path)
            return True

        except Exception as e:
            print(f"Error generating clone voice for '{speaker}': {e}")
            return False

    def _sequential_custom(self, chunks, voice_config, output_dir, batch_seed=-1, cancel_check=None):
        """Sequential custom voice generation for external mode (no native batch)."""
        results = {"completed": [], "failed": []}

        for chunk in chunks:
            if cancel_check and cancel_check():
                print("[CANCEL] Stopping sequential custom generation.")
                break
            idx = chunk["index"]
            output_path = os.path.join(output_dir, f"temp_batch_{idx}.wav")
            try:
                success = self.generate_custom_voice(
                    chunk.get("text", ""),
                    chunk.get("instruct", ""),
                    chunk.get("speaker", ""),
                    voice_config,
                    output_path,
                )
                if success:
                    results["completed"].append(idx)
                    print(f"Batch chunk {idx} saved: {os.path.getsize(output_path)} bytes")
                else:
                    results["failed"].append((idx, "Custom voice generation failed"))
            except Exception as e:
                results["failed"].append((idx, str(e)))

        return results

    # ── Utility ──────────────────────────────────────────────────

    @staticmethod
    def _save_wav(audio_array, sample_rate, output_path):
        """Save a numpy audio array as a WAV file."""
        # Ensure numpy array
        if not isinstance(audio_array, np.ndarray):
            audio_array = np.array(audio_array)
        # Flatten if needed
        if audio_array.ndim > 1:
            audio_array = audio_array.flatten()
        sf.write(output_path, audio_array, sample_rate)

    def _persist_batch_audio_outputs(self, wavs_list, sample_rate, output_dir, indices, results):
        """Save batch outputs and fail any requested indices missing from the model response."""
        saved_audio_duration = 0.0
        returned = len(wavs_list or [])
        expected = len(indices or [])

        for wav, idx in zip(wavs_list or [], indices or []):
            try:
                output_path = os.path.join(output_dir, f"temp_batch_{idx}.wav")
                audio = self._concat_audio(wav)
                self._save_wav(audio, sample_rate, output_path)
                results["completed"].append(idx)
                duration = len(audio) / sample_rate
                saved_audio_duration += duration
                print(f"    Chunk {idx} saved: {os.path.getsize(output_path)} bytes ({duration:.1f}s audio)")
            except Exception as e:
                print(f"    Error saving chunk {idx}: {e}")
                results["failed"].append((idx, str(e)))

        if returned != expected:
            print(f"  Warning: batch returned {returned} audio clip(s) for {expected} requested chunk(s)")
            for idx in list(indices or [])[returned:]:
                results["failed"].append((idx, f"Batch returned {returned}/{expected} audio clips"))

        return saved_audio_duration
