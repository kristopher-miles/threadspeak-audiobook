import importlib.util
import json
import os
import tempfile
import unittest

MODULE_PATH = os.path.join(os.path.dirname(__file__), "app.py")
SPEC = importlib.util.spec_from_file_location("alexandria_app_module_archive", MODULE_PATH)
app_module = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(app_module)


class ProjectArchiveHelpersTests(unittest.TestCase):
    def test_normalize_archive_path_rejects_parent_traversal(self):
        with self.assertRaises(ValueError):
            app_module._normalize_archive_path("../secret.txt")

    def test_allowed_archive_paths_cover_expected_project_content(self):
        self.assertTrue(app_module._is_allowed_project_archive_path("annotated_script.json"))
        self.assertTrue(app_module._is_allowed_project_archive_path("voicelines/chunk_001.mp3"))
        self.assertTrue(app_module._is_allowed_project_archive_path("clone_voices/manifest.json"))
        self.assertFalse(app_module._is_allowed_project_archive_path("app/config.json"))

    def test_archive_state_rewrites_uploaded_file_path_relative_to_root(self):
        with tempfile.TemporaryDirectory() as temp_root:
            uploads_dir = os.path.join(temp_root, "uploads")
            os.makedirs(uploads_dir, exist_ok=True)
            input_path = os.path.join(uploads_dir, "story.txt")
            with open(input_path, "w", encoding="utf-8") as f:
                f.write("hello")

            state_path = os.path.join(temp_root, "state.json")
            with open(state_path, "w", encoding="utf-8") as f:
                json.dump({"input_file_path": input_path, "render_prep_complete": True}, f)

            original_root = app_module.ROOT_DIR
            original_uploads = app_module.UPLOADS_DIR
            try:
                app_module.ROOT_DIR = temp_root
                app_module.UPLOADS_DIR = uploads_dir
                exported = app_module._archive_state_with_relative_paths()
            finally:
                app_module.ROOT_DIR = original_root
                app_module.UPLOADS_DIR = original_uploads

        self.assertEqual(exported["input_file_path"], "uploads/story.txt")
        self.assertTrue(exported["render_prep_complete"])

    def test_project_archive_entries_include_only_timeline_audio_and_current_voice_assets(self):
        with tempfile.TemporaryDirectory() as temp_root:
            for dirname in ("voicelines", "clone_voices", "designed_voices", "uploads"):
                os.makedirs(os.path.join(temp_root, dirname), exist_ok=True)

            files = {
                "annotated_script.json": {"entries": [], "dictionary": []},
                "voice_config.json": {
                    "Narrator": {"ref_audio": "clone_voices/current_clone.wav"},
                    "Alice": {"ref_audio": "designed_voices/current_design.wav"},
                },
                "voices.json": ["Narrator", "Alice"],
                "chunks.json": [
                    {"id": 0, "audio_path": "voicelines/live_a.mp3"},
                    {"id": 1, "audio_path": "voicelines/live_b.mp3"},
                    {"id": 2, "audio_path": None},
                ],
                "script_sanity_check.json": {"ok": True},
                "state.json": {"input_file_path": os.path.join(temp_root, "uploads", "story.txt")},
            }
            for rel, payload in files.items():
                with open(os.path.join(temp_root, rel), "w", encoding="utf-8") as f:
                    json.dump(payload, f)

            for rel in (
                "voicelines/live_a.mp3",
                "voicelines/live_b.mp3",
                "voicelines/orphan.mp3",
                "clone_voices/current_clone.wav",
                "clone_voices/orphan_clone.wav",
                "designed_voices/current_design.wav",
                "designed_voices/orphan_design.wav",
                "uploads/story.txt",
            ):
                full = os.path.join(temp_root, rel)
                os.makedirs(os.path.dirname(full), exist_ok=True)
                with open(full, "wb") as f:
                    f.write(b"test")

            with open(os.path.join(temp_root, "clone_voices", "manifest.json"), "w", encoding="utf-8") as f:
                json.dump([{"id": "clone-1", "filename": "current_clone.wav"}], f)
            with open(os.path.join(temp_root, "designed_voices", "manifest.json"), "w", encoding="utf-8") as f:
                json.dump([{"id": "design-1", "filename": "current_design.wav"}], f)

            original_root = app_module.ROOT_DIR
            original_uploads = app_module.UPLOADS_DIR
            original_chunks = app_module.CHUNKS_PATH
            original_voice_config = app_module.VOICE_CONFIG_PATH
            original_clone_manifest = getattr(app_module, "CLONE_VOICES_MANIFEST", None)
            original_design_manifest = getattr(app_module, "DESIGNED_VOICES_MANIFEST", None)
            try:
                app_module.ROOT_DIR = temp_root
                app_module.CHUNKS_PATH = os.path.join(temp_root, "chunks.json")
                app_module.VOICE_CONFIG_PATH = os.path.join(temp_root, "voice_config.json")
                app_module.UPLOADS_DIR = os.path.join(temp_root, "uploads")
                app_module.CLONE_VOICES_MANIFEST = os.path.join(temp_root, "clone_voices", "manifest.json")
                app_module.DESIGNED_VOICES_MANIFEST = os.path.join(temp_root, "designed_voices", "manifest.json")
                entries = dict(app_module._project_archive_entries())
            finally:
                app_module.ROOT_DIR = original_root
                app_module.CHUNKS_PATH = original_chunks
                app_module.VOICE_CONFIG_PATH = original_voice_config
                app_module.UPLOADS_DIR = original_uploads
                app_module.CLONE_VOICES_MANIFEST = original_clone_manifest
                app_module.DESIGNED_VOICES_MANIFEST = original_design_manifest

            self.assertIn("voicelines/live_a.mp3", entries)
            self.assertIn("voicelines/live_b.mp3", entries)
            self.assertNotIn("voicelines/orphan.mp3", entries)
            self.assertIn("clone_voices/manifest.json", entries)
            self.assertIn("clone_voices/current_clone.wav", entries)
            self.assertNotIn("clone_voices/orphan_clone.wav", entries)
            self.assertIn("designed_voices/manifest.json", entries)
            self.assertIn("designed_voices/current_design.wav", entries)
            self.assertNotIn("designed_voices/orphan_design.wav", entries)
            self.assertIn("uploads/story.txt", entries)


if __name__ == "__main__":
    unittest.main()
