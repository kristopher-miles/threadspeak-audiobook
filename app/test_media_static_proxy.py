import os
import tempfile
import unittest
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.shared import _VoicelinesProxyStatic


class MediaStaticProxyTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
