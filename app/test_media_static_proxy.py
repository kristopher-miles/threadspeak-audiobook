import os
import tempfile
import urllib.request
import unittest
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

import api.shared as shared
from api.shared import _VoicelinesProxyStatic
from runtime_layout import LAYOUT


class MediaStaticProxyTests(unittest.TestCase):
    def tearDown(self):
        shared._shutdown_media_static_server()

    def test_voicelines_proxy_redirects_to_media_origin(self):
        with tempfile.TemporaryDirectory() as temp_root:
            voicelines_dir = os.path.join(temp_root, "voicelines")
            os.makedirs(voicelines_dir, exist_ok=True)
            with open(os.path.join(voicelines_dir, "clip.mp3"), "wb") as f:
                f.write(b"stub")

            app = FastAPI()
            app.mount("/voicelines", _VoicelinesProxyStatic(directory=voicelines_dir))
            client = TestClient(app)

            with patch("api.shared._get_media_static_origin", return_value="http://127.0.0.1:43123"):
                response = client.get("/voicelines/clip.mp3?t=abc", follow_redirects=False)

            self.assertEqual(response.status_code, 307)
            self.assertEqual(response.headers["location"], "http://127.0.0.1:43123/voicelines/clip.mp3?t=abc")

    def test_voicelines_proxy_falls_back_to_local_static_files(self):
        with tempfile.TemporaryDirectory() as temp_root:
            voicelines_dir = os.path.join(temp_root, "voicelines")
            os.makedirs(voicelines_dir, exist_ok=True)
            clip_path = os.path.join(voicelines_dir, "clip.mp3")
            with open(clip_path, "wb") as f:
                f.write(b"stub-audio")

            app = FastAPI()
            app.mount("/voicelines", _VoicelinesProxyStatic(directory=voicelines_dir))
            client = TestClient(app)

            with patch("api.shared._get_media_static_origin", return_value=None):
                response = client.get("/voicelines/clip.mp3")

            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.content, b"stub-audio")

    def test_voicelines_proxy_falls_back_to_local_static_files_when_child_server_fails(self):
        shared._shutdown_media_static_server()

        with tempfile.TemporaryDirectory() as temp_root:
            voicelines_dir = os.path.join(temp_root, "voicelines")
            os.makedirs(voicelines_dir, exist_ok=True)
            clip_path = os.path.join(voicelines_dir, "clip.mp3")
            with open(clip_path, "wb") as f:
                f.write(b"stub-audio")

            app = FastAPI()
            app.mount("/voicelines", _VoicelinesProxyStatic(directory=voicelines_dir))
            client = TestClient(app)

            with patch("api.shared.subprocess.Popen", side_effect=OSError("spawn failed")):
                response = client.get("/voicelines/clip.mp3")

            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.content, b"stub-audio")

    def test_media_static_origin_waits_until_server_is_ready(self):
        shared._shutdown_media_static_server()
        voicelines_dir = LAYOUT.voicelines_dir
        os.makedirs(voicelines_dir, exist_ok=True)
        probe_path = os.path.join(voicelines_dir, "__media_ready_probe__.mp3")
        with open(probe_path, "wb") as handle:
            handle.write(b"ready")

        try:
            origin = shared._get_media_static_origin()
            self.assertTrue(origin)

            with urllib.request.urlopen(f"{origin}/__health", timeout=2.0) as response:
                self.assertEqual(response.status, 200)
                self.assertIn(b"ok", response.read().lower())
        finally:
            if os.path.exists(probe_path):
                os.remove(probe_path)

    def test_media_static_origin_supports_range_requests_for_voicelines(self):
        shared._shutdown_media_static_server()
        voicelines_dir = LAYOUT.voicelines_dir
        os.makedirs(voicelines_dir, exist_ok=True)
        clip_path = os.path.join(voicelines_dir, "__media_range_probe__.mp3")
        with open(clip_path, "wb") as handle:
            handle.write(b"0123456789" * 100)

        try:
            origin = shared._get_media_static_origin()
            self.assertTrue(origin)

            request = urllib.request.Request(
                f"{origin}/voicelines/__media_range_probe__.mp3",
                headers={"Range": "bytes=0-9"},
            )
            with urllib.request.urlopen(request, timeout=2.0) as response:
                self.assertEqual(response.status, 206)
                self.assertEqual(response.read(), b"0123456789")
                self.assertEqual(response.headers.get("Content-Range"), "bytes 0-9/1000")
        finally:
            if os.path.exists(clip_path):
                os.remove(clip_path)


if __name__ == "__main__":
    unittest.main()
