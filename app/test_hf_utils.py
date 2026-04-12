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


if __name__ == "__main__":
    unittest.main()
