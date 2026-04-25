import unittest
from unittest import mock
import contextlib
import concurrent.futures
import tempfile
import os
import base64
import io
import sys
import types
import wave
import numpy as np
import soundfile as sf

from tts import TTSEngine
from tts_providers.voxcpm2 import VoxCPM2AudioProvider


class NormalizeExternalUrlTests(unittest.TestCase):
    @staticmethod
    def _wav_payload(sample_rate=24000, frames=240):
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(b"\x00\x00" * frames)
        return base64.b64encode(buffer.getvalue()).decode("ascii")

    def test_defaults_empty_value(self):
        self.assertEqual(
            TTSEngine._normalize_external_url(""),
            "http://127.0.0.1:7860",
        )

    def test_preserves_http_url(self):
        self.assertEqual(
            TTSEngine._normalize_external_url("http://127.0.0.1:7860/"),
            "http://127.0.0.1:7860",
        )

    def test_adds_http_to_bare_host_port(self):
        self.assertEqual(
            TTSEngine._normalize_external_url("localhost:42003"),
            "http://localhost:42003",
        )

    def test_rejects_unsupported_scheme(self):
        with self.assertRaises(ValueError):
            TTSEngine._normalize_external_url("ftp://localhost:42003")

    @mock.patch("tts.httpx.get")
    def test_external_url_candidates_include_redirect_target(self, mock_get):
        mock_get.return_value.url = "http://localhost:42003/gradio"
        candidates = TTSEngine._external_url_candidates("localhost:42003")
        self.assertEqual(candidates[0], "http://localhost:42003")
        self.assertIn("http://localhost:42003/gradio", candidates)

    @mock.patch("tts.httpx.get")
    def test_external_url_candidates_include_common_mounts(self, mock_get):
        mock_get.side_effect = RuntimeError("offline")
        candidates = TTSEngine._external_url_candidates("localhost:42003")
        self.assertIn("http://localhost:42003/gradio", candidates)
        self.assertIn("http://localhost:42003/gradio_api", candidates)

    @mock.patch("tts.httpx.get")
    def test_detects_qwen_mlx_http_api(self, mock_get):
        mock_response = mock.Mock()
        mock_response.is_success = True
        mock_response.json.return_value = {
            "paths": {
                "/api/v1/custom-voice/generate": {},
                "/api/v1/base/clone": {},
            }
        }
        mock_get.return_value = mock_response
        engine = TTSEngine({
            "llm": {"api_key": "k"},
            "tts": {"mode": "external", "url": "localhost:42003"},
        })
        self.assertEqual(engine._detect_external_http_api(), "http://localhost:42003")

    def test_load_model_rejects_remote_download_when_runtime_downloads_are_disabled(self):
        fake_model_cls = mock.Mock()

        with mock.patch.object(TTSEngine, "_resolve_local_model_path", return_value=None):
            with mock.patch.dict(os.environ, {"THREADSPEAK_DISABLE_MODEL_DOWNLOADS": "1"}, clear=False):
                with self.assertRaises(RuntimeError) as raised:
                    TTSEngine._load_model(fake_model_cls, "Qwen/example", {})

        self.assertIn("THREADSPEAK_DISABLE_MODEL_DOWNLOADS", str(raised.exception))
        fake_model_cls.from_pretrained.assert_not_called()

    @mock.patch("tts.ensure_hf_snapshot", return_value="/cache/qwen-example")
    def test_load_model_uses_download_provider_before_from_pretrained(self, mock_download):
        fake_model_cls = mock.Mock()

        with mock.patch.object(TTSEngine, "_resolve_local_model_path", return_value=None):
            TTSEngine._load_model(fake_model_cls, "Qwen/example", {"dtype": "float32"})

        mock_download.assert_called_once_with("Qwen/example", display_name="Qwen/example")
        fake_model_cls.from_pretrained.assert_called_once_with("/cache/qwen-example", dtype="float32")

    def test_resolve_local_model_path_skips_incomplete_required_snapshot(self):
        with tempfile.TemporaryDirectory() as temp_root:
            cache_root = os.path.join(temp_root, "hub")
            incomplete = os.path.join(
                cache_root,
                "models--openbmb--VoxCPM2",
                "snapshots",
                "zzz-incomplete",
            )
            complete = os.path.join(
                cache_root,
                "models--openbmb--VoxCPM2",
                "snapshots",
                "aaa-complete",
            )
            os.makedirs(incomplete, exist_ok=True)
            os.makedirs(complete, exist_ok=True)
            for path in (
                os.path.join(incomplete, "config.json"),
                os.path.join(complete, "config.json"),
                os.path.join(complete, "model.safetensors"),
                os.path.join(complete, "audiovae.pth"),
            ):
                with open(path, "w", encoding="utf-8") as handle:
                    handle.write("{}")

            with mock.patch("huggingface_hub.try_to_load_from_cache", return_value=os.path.join(incomplete, "config.json")), \
                 mock.patch.dict(os.environ, {"HUGGINGFACE_HUB_CACHE": cache_root}, clear=False):
                resolved = TTSEngine._resolve_local_model_path(
                    "openbmb/VoxCPM2",
                    required_files=(
                        "model.safetensors",
                        ("audiovae.safetensors", "audiovae.pth"),
                    ),
                )

        self.assertEqual(resolved, complete)

    def test_write_base64_audio(self):
        payload = base64.b64encode(b"RIFFfakewav").decode("ascii")
        fd, path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        try:
            TTSEngine._write_base64_audio(payload, path)
            with open(path, "rb") as f:
                self.assertEqual(f.read(), b"RIFFfakewav")
        finally:
            if os.path.exists(path):
                os.unlink(path)

    def test_new_voice_design_preview_path_returns_wav_path(self):
        preview_path = TTSEngine._new_voice_design_preview_path()
        self.assertTrue(preview_path.endswith(".wav"))
        self.assertIn(os.path.join("designed_voices", "previews"), preview_path)
        self.assertTrue(os.path.isdir(os.path.dirname(preview_path)))

    def test_qwen_unload_voice_design_model_clears_only_design_state(self):
        engine = TTSEngine({"tts": {"mode": "local", "local_backend": "qwen"}})
        custom_model = object()
        clone_model = object()
        design_model = object()
        base_mlx_model = object()
        design_mlx_model = object()
        engine._local_custom_model = custom_model
        engine._local_clone_model = clone_model
        engine._local_design_model = design_model
        engine._mlx_models = {
            "base": base_mlx_model,
            "voice_design": design_mlx_model,
        }

        with mock.patch.object(engine, "_clear_gpu_cache") as clear_cache:
            self.assertTrue(engine.unload_voice_design_model())

        self.assertIs(engine._local_custom_model, custom_model)
        self.assertIs(engine._local_clone_model, clone_model)
        self.assertIsNone(engine._local_design_model)
        self.assertEqual(engine._mlx_models, {"base": base_mlx_model})
        clear_cache.assert_called_once()

    def test_voxcpm2_unload_voice_design_model_clears_provider_model(self):
        engine = TTSEngine({"tts": {"provider": "voxcpm2", "mode": "local"}})
        provider = engine._provider
        provider._model = object()

        with mock.patch.object(engine, "_clear_gpu_cache") as clear_cache:
            self.assertTrue(engine.unload_voice_design_model())

        self.assertIsNone(provider._model)
        clear_cache.assert_called_once()

    def test_unload_voice_design_model_is_safe_when_nothing_loaded(self):
        engine = TTSEngine({"tts": {"mode": "local", "local_backend": "qwen"}})

        with mock.patch.object(engine, "_clear_gpu_cache") as clear_cache:
            self.assertFalse(engine.unload_voice_design_model())

        self.assertIsNone(engine._local_design_model)
        self.assertEqual(engine._mlx_models, {})
        clear_cache.assert_called_once()

    def test_get_clone_prompt_resolves_relative_audio_against_project_root(self):
        with tempfile.TemporaryDirectory() as temp_root:
            clone_dir = os.path.join(temp_root, "clone_voices")
            os.makedirs(clone_dir, exist_ok=True)
            ref_path = os.path.join(clone_dir, "sample.wav")
            with wave.open(ref_path, "wb") as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(24000)
                wav_file.writeframes(b"\x00\x00" * 240)

            engine = TTSEngine({"tts": {"mode": "local", "local_backend": "qwen"}}, project_root=temp_root)
            fake_model = mock.Mock()
            fake_model.create_voice_clone_prompt.return_value = "prompt-token"

            with mock.patch.object(engine, "_init_local_clone", return_value=fake_model):
                prompt = engine._get_clone_prompt(
                    "Aerial",
                    {"Aerial": {"ref_audio": "clone_voices/sample.wav", "ref_text": "hello"}},
                )

            self.assertEqual(prompt, "prompt-token")
            fake_model.create_voice_clone_prompt.assert_called_once()

    @mock.patch.object(TTSEngine, "_init_local_design")
    @mock.patch.object(TTSEngine, "_new_voice_design_preview_path")
    @mock.patch.object(TTSEngine, "_external_http_post")
    @mock.patch.object(TTSEngine, "_init_external")
    def test_generate_voice_design_uses_external_http_api_when_configured(
        self,
        mock_init_external,
        mock_external_http_post,
        mock_preview_path,
        mock_init_local_design,
    ):
        engine = TTSEngine({
            "llm": {"api_key": "k"},
            "tts": {"mode": "external", "url": "localhost:42003", "language": "English"},
        })
        engine._external_backend = "qwen_mlx_http"
        mock_init_external.return_value = "http://localhost:42003"
        mock_external_http_post.return_value = {"audio": self._wav_payload()}

        fd, preview_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        os.unlink(preview_path)
        mock_preview_path.return_value = preview_path

        try:
            wav_path, sample_rate = engine.generate_voice_design(
                description="Calm narrator voice",
                sample_text="Hello from the external server.",
                language="English",
            )
        finally:
            if os.path.exists(preview_path):
                os.unlink(preview_path)

        self.assertEqual(wav_path, preview_path)
        self.assertEqual(sample_rate, 24000)
        mock_init_local_design.assert_not_called()
        mock_external_http_post.assert_called_once_with(
            "/api/v1/voice-design/generate",
            {
                "text": "Hello from the external server.",
                "language": "English",
                "instruct": "Calm narrator voice",
                "speed": 1.0,
                "response_format": "base64",
            },
        )


class ProviderDelegationTests(unittest.TestCase):
    def test_engine_defaults_provider_to_qwen3(self):
        engine = TTSEngine({"tts": {"mode": "local", "local_backend": "qwen"}})
        self.assertEqual(engine.provider_name, "qwen3")

    @mock.patch("tts.QwenAudioProvider")
    def test_engine_delegates_generation_calls_to_qwen_provider(self, mock_provider_cls):
        provider = mock.Mock()
        provider.generate_voice.return_value = True
        provider.generate_batch.return_value = {"completed": ["uid-1"], "failed": []}
        provider.generate_voice_design.return_value = ("/tmp/sample.wav", 24000)
        provider.mode = "local"
        provider.local_backend = "qwen"
        mock_provider_cls.return_value = provider

        engine = TTSEngine({"tts": {"mode": "local", "provider": "qwen3", "local_backend": "qwen"}})

        self.assertTrue(engine.generate_voice("Hello", "calm", "Aerial", {"Aerial": {"type": "custom"}}, "/tmp/out.wav"))
        self.assertEqual(
            engine.generate_batch([{"index": "uid-1"}], {"Aerial": {"type": "custom"}}, "/tmp", batch_seed=7),
            {"completed": ["uid-1"], "failed": []},
        )
        self.assertEqual(
            engine.generate_voice_design("calm voice", "Hello there"),
            ("/tmp/sample.wav", 24000),
        )
        provider.generate_voice.assert_called_once()
        provider.generate_batch.assert_called_once()
        provider.generate_voice_design.assert_called_once()
        self.assertEqual(engine.provider_name, "qwen3")
        self.assertEqual(engine.mode, "local")
        self.assertEqual(engine.local_backend, "qwen")

    @mock.patch("tts_providers.voxcpm2.VoxCPM2AudioProvider")
    def test_engine_delegates_generation_calls_to_voxcpm2_provider(self, mock_provider_cls):
        provider = mock.Mock()
        provider.generate_voice.return_value = True
        provider.generate_batch.return_value = {"completed": ["uid-1"], "failed": []}
        provider.generate_voice_design.return_value = ("/tmp/sample.wav", 48000)
        provider.mode = "local"
        provider.local_backend = "voxcpm2"
        mock_provider_cls.return_value = provider

        engine = TTSEngine({"tts": {"mode": "local", "provider": "voxcpm2"}})

        self.assertTrue(engine.generate_voice("Hello", "calm", "Aerial", {"Aerial": {"type": "custom"}}, "/tmp/out.wav"))
        self.assertEqual(
            engine.generate_batch([{"index": "uid-1"}], {"Aerial": {"type": "custom"}}, "/tmp", batch_seed=7),
            {"completed": ["uid-1"], "failed": []},
        )
        self.assertEqual(
            engine.generate_voice_design("calm voice", "Hello there"),
            ("/tmp/sample.wav", 48000),
        )
        provider.generate_voice.assert_called_once()
        provider.generate_batch.assert_called_once()
        provider.generate_voice_design.assert_called_once()
        self.assertEqual(engine.provider_name, "voxcpm2")
        self.assertEqual(engine.mode, "local")
        self.assertEqual(engine.local_backend, "voxcpm2")

    def test_engine_rejects_unknown_tts_provider(self):
        with self.assertRaises(ValueError):
            TTSEngine({"tts": {"mode": "local", "provider": "not-a-provider"}})

    @mock.patch("tts.py_platform.system", return_value="Darwin")
    def test_engine_clamps_voxcpm2_runtime_settings_and_disables_mac_optimize(self, _mock_system):
        engine = TTSEngine(
            {
                "tts": {
                    "mode": "local",
                    "provider": "voxcpm2",
                    "voxcpm_cfg_value": 9.0,
                    "voxcpm_inference_timesteps": 100,
                    "voxcpm_optimize": True,
                }
            }
        )

        self.assertEqual(engine._voxcpm_cfg_value, 3.0)
        self.assertEqual(engine._voxcpm_inference_timesteps, 30)
        self.assertFalse(engine._voxcpm_optimize)

    @mock.patch("tts.py_platform.system", return_value="Windows")
    def test_engine_allows_voxcpm2_optimize_setting_on_windows(self, _mock_system):
        engine = TTSEngine(
            {
                "tts": {
                    "mode": "local",
                    "provider": "voxcpm2",
                    "voxcpm_optimize": True,
                }
            }
        )

        self.assertTrue(engine._voxcpm_optimize)


class VoxCPM2ProviderTests(unittest.TestCase):
    @staticmethod
    def _write_wav(path, sample_rate=48000, frames=480):
        payload = VoxCPM2ProviderTests._wav_bytes(sample_rate=sample_rate, frames=frames)
        with open(path, "wb") as handle:
            handle.write(payload)

    @staticmethod
    def _wav_bytes(sample_rate=48000, frames=480):
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(b"\x00\x00" * frames)
        return buffer.getvalue()

    def test_format_control_text_prefixes_style(self):
        self.assertEqual(
            VoxCPM2AudioProvider.format_control_text("Hello.", "warm, calm"),
            "(warm, calm)Hello.",
        )
        self.assertEqual(
            VoxCPM2AudioProvider.format_control_text("Hello.", ""),
            "Hello.",
        )

    def test_voxcpm2_model_loading_prefers_local_cache(self):
        fake_model = mock.Mock()
        fake_model.tts_model.sample_rate = 48000
        fake_voxcpm_cls = mock.Mock()
        fake_voxcpm_cls.from_pretrained.return_value = fake_model
        fake_module = types.ModuleType("voxcpm")
        fake_module.VoxCPM = fake_voxcpm_cls

        engine = TTSEngine({"tts": {"mode": "local", "provider": "voxcpm2"}})
        provider = engine._provider

        with mock.patch.dict(sys.modules, {"voxcpm": fake_module}), \
             mock.patch.object(engine, "_resolve_local_model_path", return_value="/cache/openbmb/VoxCPM2"), \
             mock.patch.object(provider, "_resolve_device", return_value="mps"):
            loaded = provider._init_model()

        self.assertIs(loaded, fake_model)
        fake_voxcpm_cls.from_pretrained.assert_called_once_with(
            "/cache/openbmb/VoxCPM2",
            load_denoiser=False,
            optimize=False,
        )

    def test_voxcpm2_model_loading_respects_disabled_downloads(self):
        fake_module = types.ModuleType("voxcpm")
        fake_module.VoxCPM = mock.Mock()
        engine = TTSEngine({"tts": {"mode": "local", "provider": "voxcpm2"}})
        provider = engine._provider

        with mock.patch.dict(sys.modules, {"voxcpm": fake_module}), \
             mock.patch.object(engine, "_resolve_local_model_path", return_value=None), \
             mock.patch.dict(os.environ, {"THREADSPEAK_DISABLE_MODEL_DOWNLOADS": "1"}, clear=False):
            with self.assertRaises(RuntimeError) as raised:
                provider._init_model()

        self.assertIn("THREADSPEAK_DISABLE_MODEL_DOWNLOADS", str(raised.exception))
        fake_module.VoxCPM.from_pretrained.assert_not_called()

    @mock.patch("tts_providers.voxcpm2.ensure_hf_snapshot", return_value="/cache/openbmb/VoxCPM2")
    def test_voxcpm2_model_loading_uses_download_provider_before_from_pretrained(self, mock_download):
        fake_model = mock.Mock()
        fake_model.tts_model.sample_rate = 48000
        fake_voxcpm_cls = mock.Mock()
        fake_voxcpm_cls.from_pretrained.return_value = fake_model
        fake_module = types.ModuleType("voxcpm")
        fake_module.VoxCPM = fake_voxcpm_cls
        engine = TTSEngine({"tts": {"mode": "local", "provider": "voxcpm2"}})
        provider = engine._provider

        with mock.patch.dict(sys.modules, {"voxcpm": fake_module}), \
             mock.patch.object(engine, "_resolve_local_model_path", return_value=None), \
             mock.patch.object(provider, "_resolve_device", return_value="mps"):
            provider._init_model()

        mock_download.assert_called_once()
        fake_voxcpm_cls.from_pretrained.assert_called_once_with(
            "/cache/openbmb/VoxCPM2",
            load_denoiser=False,
            optimize=False,
        )

    @mock.patch.object(TTSEngine, "_host_platform", return_value=("windows", "amd64"))
    def test_voxcpm2_windows_device_auto_prefers_cuda_when_available(self, _mock_platform):
        engine = TTSEngine({"tts": {"mode": "local", "provider": "voxcpm2"}})
        provider = engine._provider
        fake_torch = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: True))

        with mock.patch.dict(sys.modules, {"torch": fake_torch}):
            self.assertEqual(provider._resolve_device(), "cuda")

    @mock.patch.object(TTSEngine, "_host_platform", return_value=("windows", "amd64"))
    def test_voxcpm2_windows_device_auto_falls_back_to_cpu_without_cuda(self, _mock_platform):
        engine = TTSEngine({"tts": {"mode": "local", "provider": "voxcpm2"}})
        provider = engine._provider
        fake_torch = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: False))

        with mock.patch.dict(sys.modules, {"torch": fake_torch}):
            self.assertEqual(provider._resolve_device(), "cpu")

    def test_voxcpm2_clone_generation_passes_reference_and_style_without_prompt_text(self):
        with tempfile.TemporaryDirectory() as temp_root:
            clone_dir = os.path.join(temp_root, "clone_voices")
            os.makedirs(clone_dir, exist_ok=True)
            ref_path = os.path.join(clone_dir, "sample.wav")
            with wave.open(ref_path, "wb") as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(24000)
                wav_file.writeframes(b"\x00\x00" * 240)

            output_path = os.path.join(temp_root, "out.wav")
            fake_model = mock.Mock()
            fake_model.tts_model.sample_rate = 48000
            fake_model.generate.return_value = np.zeros(480, dtype=np.float32)

            engine = TTSEngine(
                {
                    "tts": {
                        "mode": "local",
                        "provider": "voxcpm2",
                        "voxcpm_cfg_value": 2.25,
                        "voxcpm_inference_timesteps": 12,
                    }
                },
                project_root=temp_root,
            )
            provider = engine._provider
            provider._model = fake_model

            log_buffer = io.StringIO()
            with contextlib.redirect_stdout(log_buffer):
                ok = provider.generate_voice(
                    "Hello from the book.",
                    "urgent and afraid",
                    "Aerial",
                    {
                        "Aerial": {
                            "type": "clone",
                            "ref_audio": "clone_voices/sample.wav",
                            "description": "young woman",
                            "default_style": "warm",
                        }
                    },
                    output_path,
                )

            self.assertTrue(ok)
            self.assertTrue(os.path.exists(output_path))
            log_text = log_buffer.getvalue()
            self.assertIn("instruction_present=yes", log_text)
            self.assertIn("instruction='young woman, warm, urgent and afraid'", log_text)
            self.assertIn("final_text_has_control=yes", log_text)
            fake_model.generate.assert_called_once()
            kwargs = fake_model.generate.call_args.kwargs
            self.assertEqual(kwargs["text"], "(young woman, warm, urgent and afraid)Hello from the book.")
            self.assertEqual(kwargs["reference_wav_path"], ref_path)
            self.assertEqual(kwargs["cfg_value"], 2.25)
            self.assertEqual(kwargs["inference_timesteps"], 12)
            self.assertNotIn("prompt_text", kwargs)
            self.assertNotIn("prompt_wav_path", kwargs)

    def test_voxcpm2_local_instruction_override_uses_only_passed_instruction(self):
        with tempfile.TemporaryDirectory() as temp_root:
            clone_dir = os.path.join(temp_root, "clone_voices")
            os.makedirs(clone_dir, exist_ok=True)
            ref_path = os.path.join(clone_dir, "sample.wav")
            with wave.open(ref_path, "wb") as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(24000)
                wav_file.writeframes(b"\x00\x00" * 240)

            output_path = os.path.join(temp_root, "out.wav")
            fake_model = mock.Mock()
            fake_model.tts_model.sample_rate = 48000
            fake_model.generate.return_value = np.zeros(480, dtype=np.float32)

            engine = TTSEngine(
                {"tts": {"mode": "local", "provider": "voxcpm2"}},
                project_root=temp_root,
            )
            provider = engine._provider
            provider._model = fake_model

            log_buffer = io.StringIO()
            with contextlib.redirect_stdout(log_buffer):
                ok = provider.generate_voice(
                    "Hello from the book.",
                    "urgent and afraid",
                    "Aerial",
                    {
                        "Aerial": {
                            "type": "clone",
                            "ref_audio": "clone_voices/sample.wav",
                            "description": "young woman",
                            "character_style": "low and tense",
                            "default_style": "warm",
                        }
                    },
                    output_path,
                    instruction_override=True,
                )

            self.assertTrue(ok)
            log_text = log_buffer.getvalue()
            self.assertIn("instruction_policy=override", log_text)
            self.assertIn("instruction_present=yes", log_text)
            self.assertIn("instruction='urgent and afraid'", log_text)
            self.assertIn("final_text_has_control=yes", log_text)
            kwargs = fake_model.generate.call_args.kwargs
            self.assertEqual(kwargs["text"], "(urgent and afraid)Hello from the book.")
            self.assertEqual(kwargs["reference_wav_path"], ref_path)

    def test_voxcpm2_design_audio_can_be_reused_for_style_guided_clone(self):
        with tempfile.TemporaryDirectory() as temp_root:
            fake_model = mock.Mock()
            fake_model.tts_model.sample_rate = 48000
            fake_model.generate.return_value = np.zeros(480, dtype=np.float32)

            engine = TTSEngine(
                {"tts": {"mode": "local", "provider": "voxcpm2"}},
                project_root=temp_root,
            )
            provider = engine._provider
            provider._model = fake_model

            preview_path = os.path.join(temp_root, "designed-preview.wav")
            with mock.patch.object(engine, "_new_voice_design_preview_path", return_value=preview_path):
                generated_path, sample_rate = provider.generate_voice_design(
                    "warm narrator voice",
                    "This is the reusable reference line.",
                )

            clone_output = os.path.join(temp_root, "clone-output.wav")
            ok = provider.generate_voice(
                "Now read this with more urgency.",
                "urgent, voice tight",
                "NARRATOR",
                {
                    "NARRATOR": {
                        "type": "clone",
                        "ref_audio": preview_path,
                        "description": "warm narrator voice",
                    }
                },
                clone_output,
            )

            self.assertEqual(generated_path, preview_path)
            self.assertEqual(sample_rate, 48000)
            self.assertTrue(os.path.exists(preview_path))
            self.assertTrue(ok)
            self.assertTrue(os.path.exists(clone_output))
            self.assertEqual(fake_model.generate.call_count, 2)
            design_kwargs = fake_model.generate.call_args_list[0].kwargs
            clone_kwargs = fake_model.generate.call_args_list[1].kwargs
            self.assertEqual(
                design_kwargs["text"],
                "(warm narrator voice)This is the reusable reference line.",
            )
            self.assertNotIn("reference_wav_path", design_kwargs)
            self.assertEqual(
                clone_kwargs["text"],
                "(warm narrator voice, urgent, voice tight)Now read this with more urgency.",
            )
            self.assertEqual(clone_kwargs["reference_wav_path"], preview_path)

    def test_voxcpm2_external_custom_voice_calls_controlled_endpoint_without_spoken_instruction(self):
        with tempfile.TemporaryDirectory() as temp_root:
            output_path = os.path.join(temp_root, "out.wav")
            response = mock.Mock()
            response.is_success = True
            response.headers = {"content-type": "audio/wav"}
            response.content = self._wav_bytes()

            engine = TTSEngine(
                {
                    "tts": {
                        "mode": "external",
                        "provider": "voxcpm2",
                        "url": "http://example.invalid",
                        "voxcpm_cfg_value": 2.25,
                        "voxcpm_inference_timesteps": 12,
                        "voxcpm_normalize": True,
                    }
                },
                project_root=temp_root,
            )
            provider = engine._provider

            log_buffer = io.StringIO()
            with contextlib.redirect_stdout(log_buffer), \
                 mock.patch("tts_providers.voxcpm2.httpx.post", return_value=response) as post, \
                 mock.patch.object(provider, "_init_model") as init_model:
                ok = provider.generate_voice(
                    "Hello from the book.",
                    "urgent and afraid",
                    "Aerial",
                    {
                        "Aerial": {
                            "type": "custom",
                            "description": "young woman",
                            "default_style": "warm",
                            "seed": 42,
                        }
                    },
                    output_path,
                )

            self.assertTrue(ok)
            self.assertTrue(os.path.exists(output_path))
            log_text = log_buffer.getvalue()
            self.assertIn("TTS [voxcpm2 external] generating", log_text)
            self.assertIn("instruction_present=yes", log_text)
            self.assertIn("instruction='young woman, warm, urgent and afraid'", log_text)
            init_model.assert_not_called()
            post.assert_called_once_with(
                "http://example.invalid/voxcpm2_generate_controlled",
                json={
                    "text": "Hello from the book.",
                    "instruction": "young woman, warm, urgent and afraid",
                    "cfg": 2.25,
                    "steps": 12,
                    "fmt": "wav",
                    "retry_max": 3,
                    "retry_ratio": 6.0,
                    "min_len": 2,
                    "max_len": 4096,
                    "streaming": True,
                    "seed": 42,
                    "locked": False,
                    "normalize": True,
                    "denoise": False,
                    "retry": False,
                },
                timeout=180.0,
                follow_redirects=True,
            )

    def test_voxcpm2_external_batch_cancels_remote_gradio_job(self):
        with tempfile.TemporaryDirectory() as temp_root:
            output_path = os.path.join(temp_root, "out.wav")
            cancel_now = True

            engine = TTSEngine(
                {
                    "tts": {
                        "mode": "external",
                        "provider": "voxcpm2",
                        "url": "http://example.invalid",
                    }
                },
                project_root=temp_root,
            )
            provider = engine._provider

            def cancel_check():
                return cancel_now

            with mock.patch("tts_providers.voxcpm2.httpx.post") as post:
                results = provider.generate_batch(
                    [
                        {
                            "index": "clip-1",
                            "text": "First remote Vox line.",
                            "instruct": "tense",
                            "speaker": "Aerial",
                        },
                        {
                            "index": "clip-2",
                            "text": "Second remote Vox line.",
                            "instruct": "calm",
                            "speaker": "Aerial",
                        },
                    ],
                    {
                        "Aerial": {
                            "type": "custom",
                            "description": "young woman",
                        }
                    },
                    temp_root,
                    cancel_check=cancel_check,
                )

            post.assert_not_called()
            self.assertEqual(results["completed"], [])
            self.assertEqual(results["failed"], [])
            self.assertFalse(os.path.exists(output_path))

    def test_voxcpm2_external_voice_design_calls_voice_design_and_returns_sample_rate(self):
        with tempfile.TemporaryDirectory() as temp_root:
            generated_path = os.path.join(temp_root, "designed.wav")
            preview_path = os.path.join(temp_root, "preview.wav")
            self._write_wav(generated_path, sample_rate=44100)
            backend = mock.Mock()
            backend.predict.return_value = (generated_path, 77)

            engine = TTSEngine(
                {
                    "tts": {
                        "mode": "external",
                        "provider": "voxcpm2",
                        "url": "http://example.invalid",
                        "voxcpm_cfg_value": 1.75,
                        "voxcpm_inference_timesteps": 9,
                    }
                },
                project_root=temp_root,
            )
            provider = engine._provider

            with mock.patch.object(engine, "_init_external", return_value=backend), \
                 mock.patch.object(engine, "_new_voice_design_preview_path", return_value=preview_path), \
                 mock.patch.object(provider, "_init_model") as init_model:
                wav_path, sample_rate = provider.generate_voice_design(
                    "warm narrator voice",
                    "This is the reusable reference line.",
                    seed=13,
                )

            self.assertEqual(wav_path, preview_path)
            self.assertEqual(sample_rate, 44100)
            self.assertTrue(os.path.exists(preview_path))
            init_model.assert_not_called()
            backend.predict.assert_called_once_with(
                description="warm narrator voice",
                text="This is the reusable reference line.",
                cfg=1.75,
                steps=9,
                fmt="wav",
                retry_max=3,
                retry_ratio=6.0,
                min_len=2,
                max_len=4096,
                streaming=True,
                seed=13,
                locked=False,
                normalize=False,
                retry=False,
                api_name="/voice_design",
            )

    def test_voxcpm2_external_voice_design_materializes_gradio_stream_result(self):
        with tempfile.TemporaryDirectory() as temp_root:
            preview_path = os.path.join(temp_root, "preview.wav")
            backend = mock.Mock()
            backend.predict.return_value = (
                {
                    "path": "session/playlist.m3u8",
                    "url": "http://voxcpm.test/gradio_api/stream/session/playlist.m3u8",
                    "is_stream": True,
                    "meta": {"_type": "gradio.FileData"},
                },
                123,
            )
            playlist_response = mock.Mock()
            playlist_response.is_success = True
            playlist_response.text = "#EXTM3U\n#EXTINF:1.0,\nsegment.wav\n#EXT-X-ENDLIST\n"
            segment_response = mock.Mock()
            segment_response.is_success = True
            segment_response.content = self._wav_bytes(sample_rate=32000)

            engine = TTSEngine(
                {
                    "tts": {
                        "mode": "external",
                        "provider": "voxcpm2",
                        "url": "http://example.invalid",
                    }
                },
                project_root=temp_root,
            )
            provider = engine._provider

            with mock.patch.object(engine, "_init_external", return_value=backend), \
                 mock.patch.object(engine, "_new_voice_design_preview_path", return_value=preview_path), \
                 mock.patch("tts_providers.voxcpm2.httpx.get", side_effect=[playlist_response, segment_response]):
                wav_path, sample_rate = provider.generate_voice_design(
                    "warm narrator voice",
                    "This is the reusable reference line.",
                )

            self.assertEqual(wav_path, preview_path)
            self.assertEqual(sample_rate, 32000)
            self.assertTrue(os.path.exists(preview_path))
            self.assertEqual(sf.info(preview_path).format, "WAV")

    def test_voxcpm2_external_clone_generation_calls_voice_clone_with_reference_audio(self):
        with tempfile.TemporaryDirectory() as temp_root:
            clone_dir = os.path.join(temp_root, "clone_voices")
            os.makedirs(clone_dir, exist_ok=True)
            ref_path = os.path.join(clone_dir, "sample.wav")
            output_path = os.path.join(temp_root, "clone-output.wav")
            self._write_wav(ref_path)
            upload_response = mock.Mock()
            upload_response.raise_for_status.return_value = None
            upload_response.json.return_value = ["C:\\pinokio\\api\\VoxCPM2\\app\\temp\\sample.wav"]
            generate_response = mock.Mock()
            generate_response.is_success = True
            generate_response.headers = {"content-type": "audio/wav"}
            generate_response.content = self._wav_bytes()

            engine = TTSEngine(
                {
                    "tts": {
                        "mode": "external",
                        "provider": "voxcpm2",
                        "url": "http://example.invalid",
                        "voxcpm_denoise": True,
                        "voxcpm_denoise_reference": True,
                    }
                },
                project_root=temp_root,
            )
            provider = engine._provider

            with mock.patch("tts_providers.voxcpm2.httpx.post", side_effect=[upload_response, generate_response]) as post, \
                 mock.patch.object(provider, "_init_model") as init_model:
                ok = provider.generate_voice(
                    "Now read this with more urgency.",
                    "urgent, voice tight",
                    "NARRATOR",
                    {
                        "NARRATOR": {
                            "type": "clone",
                            "ref_audio": "clone_voices/sample.wav",
                            "ref_text": "This is the reusable reference line.",
                            "description": "warm narrator voice",
                            "seed": 5,
                        }
                    },
                    output_path,
                )

            self.assertTrue(ok)
            self.assertTrue(os.path.exists(output_path))
            init_model.assert_not_called()
            self.assertEqual(post.call_count, 2)
            self.assertEqual(post.call_args_list[0].args[0], "http://example.invalid/gradio_api/upload")
            self.assertEqual(
                post.call_args_list[1].args[0],
                "http://example.invalid/voxcpm2_generate_controlled",
            )
            self.assertEqual(
                post.call_args_list[1].kwargs["json"],
                {
                    "text": "Now read this with more urgency.",
                    "instruction": "warm narrator voice, urgent, voice tight",
                    "ref_audio": "C:\\pinokio\\api\\VoxCPM2\\app\\temp\\sample.wav",
                    "cfg": 1.6,
                    "steps": 10,
                    "fmt": "wav",
                    "retry_max": 3,
                    "retry_ratio": 6.0,
                    "min_len": 2,
                    "max_len": 4096,
                    "streaming": True,
                    "seed": 5,
                    "locked": False,
                    "normalize": False,
                    "denoise": True,
                    "retry": False,
                },
            )

    def test_voxcpm2_external_denoise_payload_uses_denoise_not_denoise_reference(self):
        with tempfile.TemporaryDirectory() as temp_root:
            clone_dir = os.path.join(temp_root, "clone_voices")
            os.makedirs(clone_dir, exist_ok=True)
            ref_path = os.path.join(clone_dir, "sample.wav")
            output_path = os.path.join(temp_root, "clone-output.wav")
            self._write_wav(ref_path)
            upload_response = mock.Mock()
            upload_response.raise_for_status.return_value = None
            upload_response.json.return_value = ["C:\\pinokio\\api\\VoxCPM2\\app\\temp\\sample.wav"]
            generate_response = mock.Mock()
            generate_response.is_success = True
            generate_response.headers = {"content-type": "audio/wav"}
            generate_response.content = self._wav_bytes()

            engine = TTSEngine(
                {
                    "tts": {
                        "mode": "external",
                        "provider": "voxcpm2",
                        "url": "http://example.invalid",
                        "voxcpm_denoise": False,
                        "voxcpm_denoise_reference": True,
                    }
                },
                project_root=temp_root,
            )
            provider = engine._provider

            with mock.patch("tts_providers.voxcpm2.httpx.post", side_effect=[upload_response, generate_response]) as post:
                ok = provider.generate_voice(
                    "Read this line.",
                    "",
                    "NARRATOR",
                    {
                        "NARRATOR": {
                            "type": "clone",
                            "ref_audio": "clone_voices/sample.wav",
                            "description": "warm narrator voice",
                        }
                    },
                    output_path,
                )

            self.assertTrue(ok)
            self.assertFalse(provider.engine._voxcpm_denoise)
            self.assertTrue(provider.engine._voxcpm_denoise_reference)
            self.assertEqual(post.call_count, 2)
            self.assertFalse(post.call_args_list[1].kwargs["json"]["denoise"])


class LocalBackendResolutionTests(unittest.TestCase):
    @mock.patch.object(TTSEngine, "_host_platform", return_value=("darwin", "arm64"))
    def test_auto_selects_mlx_on_apple_silicon(self, _mock_platform):
        engine = TTSEngine({"tts": {"mode": "local", "local_backend": "auto"}})
        self.assertEqual(engine.local_backend, "mlx")

    @mock.patch.object(TTSEngine, "_host_platform", return_value=("linux", "x86_64"))
    def test_auto_selects_qwen_on_non_macos(self, _mock_platform):
        engine = TTSEngine({"tts": {"mode": "local", "local_backend": "auto"}})
        self.assertEqual(engine.local_backend, "qwen")

    @mock.patch.object(TTSEngine, "_host_platform", return_value=("linux", "x86_64"))
    def test_explicit_mlx_falls_back_to_qwen_off_apple_silicon(self, _mock_platform):
        engine = TTSEngine({"tts": {"mode": "local", "local_backend": "mlx"}})
        self.assertEqual(engine.local_backend, "qwen")


class MlxInstructionRoutingTests(unittest.TestCase):
    def test_custom_voice_loads_instruction_capable_mlx_model(self):
        engine = TTSEngine({"tts": {"mode": "local", "local_backend": "mlx"}})
        load_calls = []
        fake_model = mock.Mock()

        def fake_load_model(model_path):
            load_calls.append(model_path)
            return fake_model

        fake_mlx_audio = types.ModuleType("mlx_audio")
        fake_tts = types.ModuleType("mlx_audio.tts")
        fake_utils = types.ModuleType("mlx_audio.tts.utils")
        fake_utils.load_model = fake_load_model

        with mock.patch.dict(
            sys.modules,
            {
                "mlx_audio": fake_mlx_audio,
                "mlx_audio.tts": fake_tts,
                "mlx_audio.tts.utils": fake_utils,
            },
        ), mock.patch.object(TTSEngine, "_resolve_local_model_path", return_value=None), \
             mock.patch("tts.ensure_hf_snapshot", return_value="/cache/mlx-custom"):
            engine._init_local_mlx_model("custom_voice")

        self.assertEqual(
            load_calls,
            ["/cache/mlx-custom"],
        )


class LocalBatchRegressionTests(unittest.TestCase):
    def test_qwen_token_budget_scales_with_text_length_and_stays_bounded(self):
        engine = TTSEngine({"tts": {"mode": "local", "local_backend": "qwen"}})

        short_budget = engine._qwen_max_new_tokens_for_text("Now,")
        medium_budget = engine._qwen_max_new_tokens_for_text(
            "This is a medium sentence with enough words to require a larger generation budget."
        )
        long_budget = engine._qwen_max_new_tokens_for_text(
            " ".join(["extended"] * 120)
        )

        self.assertEqual(short_budget, 24)
        self.assertGreater(medium_budget, short_budget)
        self.assertEqual(long_budget, 540)

    def test_persist_batch_audio_outputs_marks_missing_outputs_as_failed(self):
        engine = TTSEngine({"tts": {"mode": "local", "local_backend": "qwen"}})
        results = {"completed": [], "failed": []}

        with tempfile.TemporaryDirectory() as output_dir:
            duration = engine._persist_batch_audio_outputs(
                [np.zeros(2400, dtype=np.float32)],
                24000,
                output_dir,
                ["uid-1", "uid-2"],
                results,
            )

            self.assertAlmostEqual(duration, 0.1, places=2)
            self.assertEqual(results["completed"], ["uid-1"])
            self.assertEqual(results["failed"], [("uid-2", "Batch returned 1/2 audio clips")])
            self.assertTrue(os.path.exists(os.path.join(output_dir, "temp_batch_uid-1.wav")))

    def test_local_batch_custom_persists_outputs_for_successful_sub_batch(self):
        engine = TTSEngine({"tts": {"mode": "local", "local_backend": "qwen"}})
        chunks = [{
            "index": "uid-1",
            "text": "A calm line with enough text.",
            "instruct": "",
            "speaker": "Aerial",
        }]
        voice_config = {
            "Aerial": {
                "type": "custom",
                "voice": "Ryan",
                "character_style": "",
                "default_style": "",
            }
        }

        fake_model = mock.Mock()
        fake_model.generate_custom_voice.return_value = ([np.zeros(2400, dtype=np.float32)], 24000)

        with tempfile.TemporaryDirectory() as output_dir:
            with mock.patch.object(engine, "_init_local_custom", return_value=fake_model), \
                 mock.patch.object(engine, "_estimate_max_batch_size", return_value=8), \
                 mock.patch.object(engine, "_build_sub_batches", return_value=[(0, 1)]), \
                 mock.patch.object(engine, "_clear_gpu_cache"), \
                 mock.patch.object(engine, "_warmup_model"):
                engine._warmup_needed = False
                results = engine._local_batch_custom(chunks, voice_config, output_dir)

            self.assertEqual(results["completed"], ["uid-1"])
            self.assertEqual(results["failed"], [])
            self.assertTrue(os.path.exists(os.path.join(output_dir, "temp_batch_uid-1.wav")))

    def test_local_batch_clone_uses_dynamic_text_budget(self):
        engine = TTSEngine({"tts": {"mode": "local", "local_backend": "qwen"}})
        chunks = [{
            "index": "uid-1",
            "display_id": 21,
            "text": "Now,",
            "speaker": "Bitera",
        }]
        voice_config = {
            "Bitera": {
                "type": "clone",
                "ref_audio": "clone_voices/bitera.wav",
                "ref_text": "Reference line",
            }
        }

        prompt_item = mock.Mock()
        prompt_item.ref_code = np.zeros(12, dtype=np.int32)
        prompt_item.ref_text = "Reference line"
        prompt = [prompt_item]

        fake_model = mock.Mock()
        fake_model.generate_voice_clone.return_value = ([np.zeros(2400, dtype=np.float32)], 24000)

        with tempfile.TemporaryDirectory() as output_dir:
            with mock.patch.object(engine, "_init_local_clone", return_value=fake_model), \
                 mock.patch.object(engine, "_get_clone_prompt", return_value=prompt), \
                 mock.patch.object(engine, "_estimate_max_batch_size", return_value=8), \
                 mock.patch.object(engine, "_build_sub_batches", return_value=[(0, 1)]), \
                 mock.patch.object(engine, "_clear_gpu_cache"), \
                 mock.patch.object(engine, "_warmup_model"):
                engine._warmup_needed = False
                results = engine._local_batch_clone(chunks, voice_config, output_dir)

        self.assertEqual(results["completed"], ["uid-1"])
        self.assertEqual(results["failed"], [])
        fake_model.generate_voice_clone.assert_called_once()
        self.assertEqual(fake_model.generate_voice_clone.call_args.kwargs["max_new_tokens"], 24)

if __name__ == "__main__":
    unittest.main()
