"""VoxCPM2 local TTS provider.

This provider intentionally uses VoxCPM2's controllable cloning path instead of
Qwen-style clone prompts: reference audio supplies timbre, while parenthesized
control text carries voice design, style, and emotional guidance.
"""

from __future__ import annotations

import os
import shutil
import tempfile
import threading
import time
import urllib.parse
import concurrent.futures
from typing import Any

import httpx
import numpy as np
import soundfile as sf
from gradio_client import handle_file
from pydub import AudioSegment

from model_downloads import ensure_hf_snapshot


TRUE_VALUES = {"1", "true", "yes", "on"}


class VoxCPM2AudioProvider:
    provider_name = "voxcpm2"

    def __init__(self, engine):
        self.engine = engine
        self._model = None
        self._model_lock = threading.Lock()
        self._inference_lock = threading.Lock()

    @property
    def mode(self):
        return self.engine._mode

    @property
    def local_backend(self):
        if self.mode != "local":
            return None
        return "voxcpm2"

    def unload(self):
        self._model = None
        clear_cache = getattr(self.engine, "_clear_gpu_cache", None)
        if callable(clear_cache):
            clear_cache()
        return True

    def unload_voice_design_model(self):
        had_model = self._model is not None
        self._model = None
        clear_cache = getattr(self.engine, "_clear_gpu_cache", None)
        if callable(clear_cache):
            clear_cache()
        return bool(had_model)

    def clear_clone_prompt_cache(self, speaker=None):
        # VoxCPM2 uses reference_wav_path directly and does not build reusable
        # prompt objects, so there is no provider-side prompt cache to clear.
        return None

    def generate_voice_design(self, description, sample_text, language=None, seed=-1):
        description = str(description or "").strip()
        sample_text = str(sample_text or "").strip()
        if not sample_text:
            raise ValueError("VoxCPM2 voice design requires sample text")
        if not description:
            description = "A clear, natural speaking voice"

        output_path = self.engine._new_voice_design_preview_path()
        if self.mode != "local":
            self._generate_external_voice_design(
                description=description,
                text=sample_text,
                output_path=output_path,
                seed=seed,
            )
            return output_path, self._audio_sample_rate(output_path)

        self._generate_to_file(
            text=sample_text,
            output_path=output_path,
            style=description,
            seed=seed,
        )
        return output_path, self._sample_rate()

    def generate_voice(self, text, instruct_text, speaker, voice_config, output_path, cancel_check=None):
        voice_data = dict((voice_config or {}).get(speaker) or {})
        if not voice_data:
            print(f"Warning: No voice configuration for '{speaker}'. Skipping.")
            return False

        voice_type = str(voice_data.get("type") or "custom").strip().lower()
        if voice_type in {"lora", "builtin_lora"}:
            print("VoxCPM2 provider does not support LoRA voices in this version.")
            return False
        if voice_type == "clone":
            return self._generate_clone(text, instruct_text, speaker, voice_data, output_path, cancel_check=cancel_check)
        if voice_type == "design":
            return self._generate_design(text, instruct_text, voice_data, output_path, cancel_check=cancel_check)
        return self._generate_custom(text, instruct_text, voice_data, output_path, cancel_check=cancel_check)

    def generate_batch(self, chunks, voice_config, output_dir, batch_seed=-1, cancel_check=None, log_callback=None):
        results = {"completed": [], "failed": []}
        for chunk in chunks or []:
            try:
                if cancel_check and cancel_check():
                    print("[CANCEL] Stopping VoxCPM2 batch generation.")
                    break
            except Exception:
                pass

            idx = chunk.get("index")
            output_path = os.path.join(output_dir, f"temp_batch_{idx}.wav")
            try:
                ok = self.generate_voice(
                    chunk.get("text", ""),
                    chunk.get("instruct", ""),
                    chunk.get("speaker", ""),
                    voice_config,
                    output_path,
                    cancel_check=cancel_check,
                )
                if ok:
                    results["completed"].append(idx)
                else:
                    results["failed"].append((idx, "VoxCPM2 generation failed"))
            except Exception as exc:
                results["failed"].append((idx, str(exc)))
        return results

    def _generate_custom(self, text, instruct_text, voice_data, output_path, cancel_check=None):
        style = self._combine_style(
            voice_data.get("description"),
            voice_data.get("character_style"),
            voice_data.get("default_style"),
            instruct_text,
        )
        return self._generate_checked(
            text=text,
            output_path=output_path,
            style=style,
            voice_data=voice_data,
            cancel_check=cancel_check,
        )

    def _generate_design(self, text, instruct_text, voice_data, output_path, cancel_check=None):
        style = self._combine_style(
            voice_data.get("description"),
            voice_data.get("character_style"),
            voice_data.get("default_style"),
            instruct_text,
        )
        ref_audio = str(voice_data.get("ref_audio") or "").strip()
        if ref_audio:
            ref_audio_path = self.engine._resolve_project_path(ref_audio)
            if not os.path.exists(ref_audio_path):
                print(f"Warning: Reference audio not found for designed voice: {ref_audio_path}")
                return False
            return self._generate_checked(
                text=text,
                output_path=output_path,
                style=style,
                voice_data=voice_data,
                reference_wav_path=ref_audio_path,
                cancel_check=cancel_check,
            )
        return self._generate_checked(
            text=text,
            output_path=output_path,
            style=style,
            voice_data=voice_data,
            cancel_check=cancel_check,
        )

    def _generate_clone(self, text, instruct_text, speaker, voice_data, output_path, cancel_check=None):
        ref_audio_path = str(voice_data.get("ref_audio") or "").strip()
        if not ref_audio_path:
            print(f"Warning: Clone voice for '{speaker}' missing ref_audio. Skipping.")
            return False
        ref_audio_path = self.engine._resolve_project_path(ref_audio_path)
        if not os.path.exists(ref_audio_path):
            print(f"Warning: Reference audio not found for '{speaker}': {ref_audio_path}")
            return False

        style = self._combine_style(
            voice_data.get("description"),
            voice_data.get("character_style"),
            voice_data.get("default_style"),
            instruct_text,
        )
        return self._generate_checked(
            text=text,
            output_path=output_path,
            style=style,
            voice_data=voice_data,
            reference_wav_path=ref_audio_path,
            cancel_check=cancel_check,
        )

    def _generate_checked(self, *, text, output_path, style="", voice_data=None, reference_wav_path=None,
                          cancel_check=None):
        try:
            transcript = str(
                (voice_data or {}).get("generated_ref_text")
                or (voice_data or {}).get("ref_text")
                or ""
            )
            self._generate_to_file(
                text=str(text or ""),
                output_path=output_path,
                style=style,
                reference_wav_path=reference_wav_path,
                transcript=transcript,
                seed=int((voice_data or {}).get("seed", -1) or -1),
                cancel_check=cancel_check,
            )
            return True
        except Exception as exc:
            print(f"Error generating VoxCPM2 voice: {exc}")
            return False

    def _generate_to_file(self, *, text, output_path, style="", reference_wav_path=None, transcript="", seed=-1,
                          cancel_check=None):
        if self.mode != "local":
            self._generate_external_to_file(
                text=text,
                output_path=output_path,
                style=style,
                reference_wav_path=reference_wav_path,
                transcript=transcript,
                seed=seed,
                cancel_check=cancel_check,
            )
            return

        model = self._init_model()
        final_text = self.format_control_text(text, style)
        if not final_text:
            raise ValueError("VoxCPM2 generation requires non-empty text")

        if seed >= 0:
            try:
                import torch

                torch.manual_seed(int(seed))
            except Exception:
                pass
            np.random.seed(int(seed))

        kwargs = {
            "text": final_text,
            "cfg_value": float(self.engine._voxcpm_cfg_value),
            "inference_timesteps": int(self.engine._voxcpm_inference_timesteps),
            "normalize": bool(self.engine._voxcpm_normalize),
            "denoise": bool(self.engine._voxcpm_denoise_reference),
        }
        if reference_wav_path:
            kwargs["reference_wav_path"] = reference_wav_path

        print(
            "TTS [voxcpm2] generating "
            f"reference_wav_path={'yes' if reference_wav_path else 'no'} "
            f"text='{str(text or '')[:50]}...'"
        )
        start = time.time()
        with self._inference_lock:
            wav = model.generate(**kwargs)
        elapsed = time.time() - start

        if wav is None:
            raise RuntimeError("VoxCPM2 returned no audio")
        audio = np.asarray(wav, dtype=np.float32)
        if audio.size == 0:
            raise RuntimeError("VoxCPM2 returned empty audio")

        sr = self._sample_rate()
        duration = float(audio.shape[0]) / float(sr or 1)
        print(f"TTS [voxcpm2] done: {elapsed:.1f}s -> {duration:.1f}s audio")
        self.engine._save_wav(audio, sr, output_path)

    def _generate_external_to_file(self, *, text, output_path, style="", reference_wav_path=None, transcript="",
                                   seed=-1, cancel_check=None):
        if reference_wav_path:
            self._generate_external_clone(
                text=text,
                style=style,
                reference_wav_path=reference_wav_path,
                transcript=transcript,
                output_path=output_path,
                seed=seed,
                cancel_check=cancel_check,
            )
            return

        final_text = self.format_control_text(text, style)
        if not final_text:
            raise ValueError("VoxCPM2 generation requires non-empty text")

        backend = self.engine._init_external()
        print(f"TTS [voxcpm2 external] generating text='{str(text or '')[:50]}...'")
        start = time.time()
        result = self._run_external_gradio_job(
            backend,
            cancel_check=cancel_check,
            text=final_text,
            **self._external_generation_kwargs(seed=seed),
            api_name="/tts_generate",
        )
        self._materialize_generated_audio(result, output_path)
        elapsed = time.time() - start
        print(f"TTS [voxcpm2 external] done: {elapsed:.1f}s -> {self._audio_duration(output_path):.1f}s audio")

    def _generate_external_voice_design(self, *, description, text, output_path, seed=-1):
        backend = self.engine._init_external()
        print(f"VoiceDesign [voxcpm2 external] generating text='{str(text or '')[:50]}...'")
        start = time.time()
        result = backend.predict(
            description=description,
            text=text,
            **self._external_generation_kwargs(seed=seed),
            api_name="/voice_design",
        )
        self._materialize_generated_audio(result, output_path)
        elapsed = time.time() - start
        print(f"VoiceDesign [voxcpm2 external] done: {elapsed:.1f}s -> {self._audio_duration(output_path):.1f}s audio")

    def _generate_external_clone(self, *, text, style, reference_wav_path, transcript="", output_path, seed=-1,
                                 cancel_check=None):
        backend = self.engine._init_external()
        print(
            "TTS [voxcpm2 external] cloning "
            f"reference_wav_path=yes text='{str(text or '')[:50]}...'"
        )
        start = time.time()
        result = self._run_external_gradio_job(
            backend,
            cancel_check=cancel_check,
            text=str(text or ""),
            ref_audio=handle_file(reference_wav_path),
            style=str(style or ""),
            transcript=str(transcript or ""),
            **self._external_generation_kwargs(seed=seed, include_denoise=True),
            api_name="/voice_clone",
        )
        self._materialize_generated_audio(result, output_path)
        elapsed = time.time() - start
        print(f"TTS [voxcpm2 external] clone done: {elapsed:.1f}s -> {self._audio_duration(output_path):.1f}s audio")

    def _run_external_gradio_job(self, backend, *, cancel_check=None, **kwargs):
        if not cancel_check:
            return backend.predict(**kwargs)

        try:
            if cancel_check():
                raise RuntimeError("VoxCPM2 external generation cancelled before submission")
        except RuntimeError:
            raise
        except Exception:
            pass

        job = backend.submit(**kwargs)
        while True:
            try:
                return job.result(timeout=0.25)
            except concurrent.futures.TimeoutError:
                try:
                    if cancel_check():
                        try:
                            job.cancel()
                        finally:
                            raise RuntimeError("VoxCPM2 external generation cancelled")
                except RuntimeError:
                    raise
                except Exception:
                    continue

    def _external_generation_kwargs(self, *, seed=-1, include_denoise=False):
        kwargs = {
            "cfg": float(self.engine._voxcpm_cfg_value),
            "steps": int(self.engine._voxcpm_inference_timesteps),
            "fmt": "wav",
            "retry_max": 3,
            "retry_ratio": 6.0,
            "min_len": 2,
            "max_len": 4096,
            "streaming": True,
            "seed": int(seed if seed is not None else -1),
            "locked": False,
            "normalize": bool(self.engine._voxcpm_normalize),
            "retry": False,
        }
        if include_denoise:
            kwargs["denoise"] = bool(self.engine._voxcpm_denoise_reference)
        return kwargs

    @staticmethod
    def _extract_generated_audio_path(result):
        if isinstance(result, (list, tuple)):
            audio_value = result[0] if result else ""
        else:
            audio_value = result
        if isinstance(audio_value, dict):
            audio_path = audio_value.get("path") or ""
            if audio_value.get("is_stream") and audio_value.get("url"):
                audio_path = VoxCPM2AudioProvider._download_hls_audio_segment(audio_value["url"])
        else:
            audio_path = audio_value
        audio_path = str(audio_path or "").strip()
        if not audio_path:
            raise RuntimeError("VoxCPM2 external server returned no audio path")
        if not os.path.exists(audio_path):
            raise RuntimeError(f"VoxCPM2 external audio path does not exist: {audio_path}")
        if os.path.getsize(audio_path) == 0:
            raise RuntimeError(f"VoxCPM2 external audio path is empty: {audio_path}")
        return audio_path

    @classmethod
    def _materialize_generated_audio(cls, result, output_path):
        source_path = cls._extract_generated_audio_path(result)
        try:
            cls._copy_generated_audio(source_path, output_path)
        finally:
            if os.path.basename(source_path).startswith("threadspeak-voxcpm2-segment-"):
                try:
                    os.unlink(source_path)
                except OSError:
                    pass

    @classmethod
    def _copy_generated_audio(cls, source_path, output_path):
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        if cls._can_read_audio(source_path):
            try:
                source_info = sf.info(source_path)
                if output_path.lower().endswith(".wav") and source_info.format != "WAV":
                    AudioSegment.from_file(source_path).export(output_path, format="wav")
                else:
                    shutil.copy2(source_path, output_path)
                return
            except Exception:
                pass

        AudioSegment.from_file(source_path).export(output_path, format="wav")

    @staticmethod
    def _download_hls_audio_segment(playlist_url):
        playlist_response = httpx.get(str(playlist_url), timeout=30.0, follow_redirects=True)
        if not playlist_response.is_success:
            raise RuntimeError(
                f"VoxCPM2 external stream playlist request failed ({playlist_response.status_code})"
            )
        segment_url = ""
        for line in playlist_response.text.splitlines():
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                segment_url = urllib.parse.urljoin(str(playlist_url), stripped)
                break
        if not segment_url:
            raise RuntimeError("VoxCPM2 external stream playlist contained no audio segment")

        segment_response = httpx.get(segment_url, timeout=120.0, follow_redirects=True)
        if not segment_response.is_success:
            raise RuntimeError(
                f"VoxCPM2 external stream segment request failed ({segment_response.status_code})"
            )

        suffix = os.path.splitext(urllib.parse.urlparse(segment_url).path)[1] or ".aac"
        with tempfile.NamedTemporaryFile(delete=False, prefix="threadspeak-voxcpm2-segment-", suffix=suffix) as handle:
            handle.write(segment_response.content)
            return handle.name

    @staticmethod
    def _can_read_audio(path):
        try:
            sf.info(path)
            return True
        except Exception:
            return False

    @staticmethod
    def _audio_sample_rate(path):
        try:
            return int(sf.info(path).samplerate or 48000)
        except Exception:
            return 48000

    @classmethod
    def _audio_duration(cls, path):
        try:
            info = sf.info(path)
            return float(info.frames or 0) / float(info.samplerate or 1)
        except Exception:
            return 0.0

    def _init_model(self):
        if self._model is not None:
            return self._model
        with self._model_lock:
            if self._model is not None:
                return self._model

            from voxcpm import VoxCPM

            model_id = str(self.engine._voxcpm_model_id or "openbmb/VoxCPM2").strip()
            load_target = self._resolve_model_load_target(model_id)
            device = self._resolve_device()
            optimize = bool(self.engine._voxcpm_optimize) and str(device).startswith("cuda")
            print(
                f"Loading VoxCPM2 model from {load_target} "
                f"on {device} (optimize={optimize}, denoiser={bool(self.engine._voxcpm_load_denoiser)})..."
            )
            self._model = VoxCPM.from_pretrained(
                load_target,
                load_denoiser=bool(self.engine._voxcpm_load_denoiser),
                optimize=optimize,
            )
            print("VoxCPM2 model loaded.")
            return self._model

    def _resolve_model_load_target(self, model_id):
        local_path = self.engine._resolve_local_model_path(
            model_id,
            required_files=(
                "model.safetensors",
                ("audiovae.safetensors", "audiovae.pth"),
            ),
        )
        if local_path:
            print(f"  Loading VoxCPM2 from local cache: {local_path}")
            return local_path
        if self.engine._env_flag("THREADSPEAK_DISABLE_MODEL_DOWNLOADS", default=False):
            raise RuntimeError(
                "Model downloads are disabled by THREADSPEAK_DISABLE_MODEL_DOWNLOADS "
                f"and no local cache exists for {model_id}."
            )
        print(f"  VoxCPM2 model not cached locally, downloading {model_id}...")
        return ensure_hf_snapshot(
            model_id,
            display_name=model_id,
            local_path_resolver=lambda repo_id, required_files=None: self.engine._resolve_local_model_path(
                repo_id,
                required_files=required_files,
            ),
            required_files=(
                "model.safetensors",
                ("audiovae.safetensors", "audiovae.pth"),
            ),
        )

    def _resolve_device(self):
        requested = str(getattr(self.engine, "_device", "") or "auto").strip().lower()
        if requested and requested != "auto":
            return requested

        host_platform, host_arch = self.engine._host_platform()
        if host_platform == "darwin":
            if host_arch == "arm64":
                return "mps"
            print("Warning: VoxCPM2 on Intel macOS will use CPU and may be slow.")
            return "cpu"

        try:
            import torch

            if torch.cuda.is_available():
                return "cuda"
        except Exception:
            pass
        return "cpu"

    def _sample_rate(self):
        model = self._model
        tts_model = getattr(model, "tts_model", None)
        sample_rate = getattr(tts_model, "sample_rate", None)
        try:
            return int(sample_rate or 48000)
        except (TypeError, ValueError):
            return 48000

    @staticmethod
    def _combine_style(*parts: Any) -> str:
        seen = set()
        cleaned = []
        for part in parts:
            text = " ".join(str(part or "").strip().split())
            if not text:
                continue
            key = text.lower()
            if key in seen:
                continue
            seen.add(key)
            cleaned.append(text)
        return ", ".join(cleaned)

    @staticmethod
    def format_control_text(text, style=""):
        body = str(text or "").strip()
        control = " ".join(str(style or "").strip().split())
        if not control:
            return body
        return f"({control}){body}"
