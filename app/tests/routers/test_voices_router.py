import asyncio
import json
import os
import tempfile
import unittest

from api.routers import voices_router


class _StubScriptStore:
    def __init__(self, voice_rows=None):
        self._voice_rows = list(voice_rows or [])

    def list_voice_rows(self):
        return list(self._voice_rows)


class _StubProjectManager:
    def __init__(self, root_dir, chunks, voice_rows=None):
        self.root_dir = root_dir
        self.script_store = _StubScriptStore(voice_rows=voice_rows)
        self._chunks = chunks

    def load_chunks(self):
        return list(self._chunks)

    def suggest_design_sample_text(self, voice_name, chunks):
        return ""

    def _normalize_speaker_name(self, name):
        return str(name or "").strip().lower()

    def load_script_document(self):
        return {"entries": []}


class VoicesRouterTests(unittest.TestCase):
    def test_get_voices_prefers_chunk_snapshot_when_voice_rows_are_empty(self):
        with tempfile.TemporaryDirectory() as temp_root:
            original_project_manager = voices_router.project_manager
            original_root_dir = voices_router.ROOT_DIR
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
                voices_router._load_runtime_voice_config = lambda: {}
                voices_router._find_saved_voice_option_for_speaker = lambda speaker: None

                result = asyncio.run(voices_router.get_voices())
            finally:
                voices_router.project_manager = original_project_manager
                voices_router.ROOT_DIR = original_root_dir
                voices_router._load_runtime_voice_config = original_load_runtime_voice_config
                voices_router._find_saved_voice_option_for_speaker = original_find_saved_voice

        names = {row.get("name") for row in result}
        self.assertIn("Edited", names)

    def test_get_voices_hides_zero_line_profiles_but_keeps_user_created_rows(self):
        with tempfile.TemporaryDirectory() as temp_root:
            original_project_manager = voices_router.project_manager
            original_root_dir = voices_router.ROOT_DIR
            original_load_runtime_voice_config = voices_router._load_runtime_voice_config
            try:
                voices_router.project_manager = _StubProjectManager(
                    temp_root,
                    [],
                    voice_rows=[
                        {
                            "name": "Live",
                            "config": {"type": "design", "description": "current"},
                            "line_count": 2,
                            "auto_narrator_alias": False,
                            "auto_alias_target": "",
                        },
                        {
                            "name": "Dead",
                            "config": {"type": "design", "description": "stale"},
                            "line_count": 0,
                            "auto_narrator_alias": False,
                            "auto_alias_target": "",
                        },
                    ],
                )
                voices_router.ROOT_DIR = temp_root
                voices_router._load_runtime_voice_config = lambda: {
                    "Manual Zero": {
                        "type": "design",
                        "description": "manual",
                        "user_created": True,
                    }
                }

                result = asyncio.run(voices_router.get_voices())
            finally:
                voices_router.project_manager = original_project_manager
                voices_router.ROOT_DIR = original_root_dir
                voices_router._load_runtime_voice_config = original_load_runtime_voice_config

        names = {row.get("name") for row in result}
        self.assertIn("Live", names)
        self.assertIn("Manual Zero", names)
        self.assertNotIn("Dead", names)


if __name__ == "__main__":
    unittest.main()
