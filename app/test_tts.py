import unittest
from unittest import mock
import tempfile
import os
import base64
import io
import wave
import numpy as np

from tts import TTSEngine


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


class LocalBatchRegressionTests(unittest.TestCase):
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

if __name__ == "__main__":
    unittest.main()
