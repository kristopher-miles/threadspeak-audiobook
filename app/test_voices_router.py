import asyncio
import json
import os
import tempfile
import unittest

from api.routers import voices_router


class _StubScriptStore:
    def list_voice_rows(self):
        return []


class _StubProjectManager:
    def __init__(self, root_dir, chunks):
        self.root_dir = root_dir
        self.script_store = _StubScriptStore()
        self._chunks = chunks

    def load_chunks(self):
        return list(self._chunks)

    def suggest_design_sample_text(self, voice_name, chunks):
        return ""

    def _normalize_speaker_name(self, name):
        return str(name or "").strip().lower()


class VoicesRouterTests(unittest.TestCase):
    def test_get_voices_prefers_chunk_snapshot_when_voice_rows_are_empty(self):
        with tempfile.TemporaryDirectory() as temp_root:
            script_path = os.path.join(temp_root, "annotated_script.json")
            with open(script_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "entries": [
                            {"speaker": "Original", "text": "Script speaker should not win."},
                        ],
                        "dictionary": [],
                    },
                    f,
                )

            original_project_manager = voices_router.project_manager
            original_root_dir = voices_router.ROOT_DIR
            original_script_path = voices_router.SCRIPT_PATH
            original_load_runtime_voice_config = voices_router._load_runtime_voice_config
            original_find_saved_voice = voices_router._find_saved_voice_option_for_speaker
            try:
                voices_router.project_manager = _StubProjectManager(
                    temp_root,
                    [
                        {
                            "id": 0,
                            "uid": "chunk-1",
                            "speaker": "Edited",
                            "text": "Chunk speaker should win.",
                            "status": "pending",
                        }
                    ],
                )
                voices_router.ROOT_DIR = temp_root
                voices_router.SCRIPT_PATH = script_path
                voices_router._load_runtime_voice_config = lambda: {}
                voices_router._find_saved_voice_option_for_speaker = lambda speaker: None

                result = asyncio.run(voices_router.get_voices())
            finally:
                voices_router.project_manager = original_project_manager
                voices_router.ROOT_DIR = original_root_dir
                voices_router.SCRIPT_PATH = original_script_path
                voices_router._load_runtime_voice_config = original_load_runtime_voice_config
                voices_router._find_saved_voice_option_for_speaker = original_find_saved_voice

        names = {row.get("name") for row in result}
        self.assertIn("Edited", names)
        self.assertNotIn("Original", names)


if __name__ == "__main__":
    unittest.main()
