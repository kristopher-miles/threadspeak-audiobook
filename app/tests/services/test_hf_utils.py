import json
import os
import types
import tempfile
import unittest
from unittest import mock

import hf_utils


class FetchBuiltinManifestTests(unittest.TestCase):
    def setUp(self):
        hf_utils._manifest_cache = None
        hf_utils._manifest_cache_time = 0

    def tearDown(self):
        hf_utils._manifest_cache = None
        hf_utils._manifest_cache_time = 0

    def test_prefers_local_manifest_without_hitting_hf(self):
        expected = [{"id": "builtin_watson", "name": "Watson"}]

        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = os.path.join(tmpdir, "manifest.json")
            with open(manifest_path, "w", encoding="utf-8") as f:
                json.dump(expected, f)

            fake_hf = types.SimpleNamespace(hf_hub_download=mock.Mock())
            with mock.patch.dict("sys.modules", {"huggingface_hub": fake_hf}):
                result = hf_utils.fetch_builtin_manifest(tmpdir)

            self.assertEqual(result, expected)
            fake_hf.hf_hub_download.assert_not_called()

    def test_downloads_manifest_when_local_file_is_missing(self):
        expected = [{"id": "builtin_holmes", "name": "Holmes"}]

        with tempfile.TemporaryDirectory() as tmpdir, tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", delete=False
        ) as remote_manifest:
            json.dump(expected, remote_manifest)
            remote_manifest_path = remote_manifest.name

        try:
            fake_hf = types.SimpleNamespace(
                hf_hub_download=mock.Mock(return_value=remote_manifest_path)
            )
            with mock.patch.dict("sys.modules", {"huggingface_hub": fake_hf}):
                result = hf_utils.fetch_builtin_manifest(tmpdir)

            self.assertEqual(result, expected)
            fake_hf.hf_hub_download.assert_called_once()

            with open(os.path.join(tmpdir, "manifest.json"), "r", encoding="utf-8") as f:
                self.assertEqual(json.load(f), expected)
        finally:
            if os.path.exists(remote_manifest_path):
                os.unlink(remote_manifest_path)

    def test_skips_manifest_download_when_runtime_downloads_are_disabled(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            fake_hf = types.SimpleNamespace(hf_hub_download=mock.Mock())
            with mock.patch.dict("os.environ", {"THREADSPEAK_DISABLE_MODEL_DOWNLOADS": "1"}, clear=False):
                with mock.patch.dict("sys.modules", {"huggingface_hub": fake_hf}):
                    result = hf_utils.fetch_builtin_manifest(tmpdir)

        self.assertEqual(result, [])
        fake_hf.hf_hub_download.assert_not_called()

    def test_download_builtin_adapter_rejects_when_runtime_downloads_are_disabled(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            fake_hf = types.SimpleNamespace(hf_hub_download=mock.Mock())
            with mock.patch.dict("os.environ", {"THREADSPEAK_DISABLE_MODEL_DOWNLOADS": "1"}, clear=False):
                with mock.patch.dict("sys.modules", {"huggingface_hub": fake_hf}):
                    with self.assertRaises(RuntimeError) as raised:
                        hf_utils.download_builtin_adapter("builtin_watson", tmpdir)

        self.assertIn("THREADSPEAK_DISABLE_MODEL_DOWNLOADS", str(raised.exception))
        fake_hf.hf_hub_download.assert_not_called()

    def test_download_builtin_adapter_uses_shared_download_provider(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            def fake_download(*, repo_id, filename, display_name, local_path, record_failures=True):
                os.makedirs(os.path.dirname(local_path), exist_ok=True)
                with open(local_path, "w", encoding="utf-8") as handle:
                    handle.write(filename)
                return local_path

            with mock.patch.object(hf_utils, "download_hf_file", side_effect=fake_download) as mocked:
                path = hf_utils.download_builtin_adapter("builtin_watson", tmpdir)

        self.assertTrue(path.endswith(os.path.join("builtin_watson")))
        requested = [call.kwargs["filename"] for call in mocked.call_args_list]
        self.assertIn("watson/adapter_config.json", requested)
        self.assertIn("watson/adapter_model.safetensors", requested)
        self.assertTrue(all(call.kwargs["display_name"] == "builtin_watson LoRA adapter" for call in mocked.call_args_list))
        required_flags = {
            call.kwargs["filename"]: call.kwargs["record_failures"]
            for call in mocked.call_args_list
        }
        self.assertTrue(required_flags["watson/adapter_config.json"])
        self.assertFalse(required_flags["watson/preview_sample.wav"])


if __name__ == "__main__":
    unittest.main()
