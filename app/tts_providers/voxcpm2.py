"""VoxCPM2 local TTS provider.

This provider intentionally uses VoxCPM2's controllable cloning path instead of
Qwen-style clone prompts: reference audio supplies timbre, while parenthesized
control text carries voice design, style, and emotional guidance.
"""

from __future__ import annotations

import os
import threading
import time
from typing import Any

import numpy as np

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
        self._generate_to_file(
            text=sample_text,
            output_path=output_path,
            style=description,
            seed=seed,
        )
        return output_path, self._sample_rate()

    def generate_voice(self, text, instruct_text, speaker, voice_config, output_path):
        if self.mode != "local":
            print("VoxCPM2 provider supports local mode only.")
            return False

        voice_data = dict((voice_config or {}).get(speaker) or {})
        if not voice_data:
            print(f"Warning: No voice configuration for '{speaker}'. Skipping.")
            return False

        voice_type = str(voice_data.get("type") or "custom").strip().lower()
        if voice_type in {"lora", "builtin_lora"}:
            print("VoxCPM2 provider does not support LoRA voices in this version.")
            return False
        if voice_type == "clone":
            return self._generate_clone(text, instruct_text, speaker, voice_data, output_path)
        if voice_type == "design":
            return self._generate_design(text, instruct_text, voice_data, output_path)
        return self._generate_custom(text, instruct_text, voice_data, output_path)

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
                )
                if ok:
                    results["completed"].append(idx)
                else:
                    results["failed"].append((idx, "VoxCPM2 generation failed"))
            except Exception as exc:
                results["failed"].append((idx, str(exc)))
        return results

    def _generate_custom(self, text, instruct_text, voice_data, output_path):
        style = self._combine_style(
            voice_data.get("description"),
            voice_data.get("character_style"),
            voice_data.get("default_style"),
            instruct_text,
        )
        return self._generate_checked(text=text, output_path=output_path, style=style, voice_data=voice_data)

    def _generate_design(self, text, instruct_text, voice_data, output_path):
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
            )
        return self._generate_checked(text=text, output_path=output_path, style=style, voice_data=voice_data)

    def _generate_clone(self, text, instruct_text, speaker, voice_data, output_path):
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
        )

    def _generate_checked(self, *, text, output_path, style="", voice_data=None, reference_wav_path=None):
        try:
            self._generate_to_file(
                text=str(text or ""),
                output_path=output_path,
                style=style,
                reference_wav_path=reference_wav_path,
                seed=int((voice_data or {}).get("seed", -1) or -1),
            )
            return True
        except Exception as exc:
            print(f"Error generating VoxCPM2 voice: {exc}")
            return False

    def _generate_to_file(self, *, text, output_path, style="", reference_wav_path=None, seed=-1):
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
