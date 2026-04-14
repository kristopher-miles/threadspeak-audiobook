import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts import isolated_debug_app as harness


class IsolatedDebugOwnershipTests(unittest.TestCase):
    def test_claim_harness_ownership_hard_fails_when_live_owner_exists(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source_root = Path(temp_dir) / "repo"
            source_root.mkdir()
            owner_path = harness._ownership_path(source_root)
            owner_path.write_text(json.dumps({
                "pid": 424242,
                "port": 4230,
                "manifest_path": "/tmp/existing/isolated-debug.json",
            }), encoding="utf-8")

            with mock.patch.object(harness, "_pid_is_running", return_value=True):
                with self.assertRaises(harness.HarnessError) as raised:
                    harness._claim_harness_ownership(source_root)

            message = str(raised.exception)
            self.assertIn("another isolated debug server is already running", message)
            self.assertIn("Stop the previous server before spawning a new one", message)
            self.assertIn("/tmp/existing/isolated-debug.json", message)

    def test_claim_harness_ownership_replaces_stale_owner_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source_root = Path(temp_dir) / "repo"
            source_root.mkdir()
            owner_path = harness._ownership_path(source_root)
            owner_path.write_text(json.dumps({
                "pid": 424242,
                "port": 4230,
                "manifest_path": "/tmp/existing/isolated-debug.json",
            }), encoding="utf-8")

            with mock.patch.object(harness, "_pid_is_running", return_value=False):
                claimed = harness._claim_harness_ownership(source_root)

            self.assertEqual(claimed, owner_path)
            payload = json.loads(owner_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["source_root"], str(source_root.resolve()))
            self.assertEqual(payload["owner_pid"], harness.os.getpid())

            harness._release_harness_ownership(source_root)
            self.assertFalse(owner_path.exists())

    def test_release_harness_ownership_respects_manifest_match(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source_root = Path(temp_dir) / "repo"
            source_root.mkdir()
            owner_path = harness._ownership_path(source_root)
            owner_path.write_text(json.dumps({
                "pid": 111,
                "manifest_path": "/tmp/expected/isolated-debug.json",
            }), encoding="utf-8")

            harness._release_harness_ownership(
                source_root,
                manifest_path="/tmp/different/isolated-debug.json",
            )
            self.assertTrue(owner_path.exists())

            harness._release_harness_ownership(
                source_root,
                manifest_path="/tmp/expected/isolated-debug.json",
            )
            self.assertFalse(owner_path.exists())


if __name__ == "__main__":
    unittest.main()
