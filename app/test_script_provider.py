import json
import os
import tempfile
import unittest

from project import ProjectManager


class ScriptProviderTests(unittest.TestCase):
    def _make_root(self):
        root = tempfile.mkdtemp(prefix="script-provider-")
        os.makedirs(os.path.join(root, "app"), exist_ok=True)
        os.makedirs(os.path.join(root, "voicelines"), exist_ok=True)
        return root

    def test_bootstraps_store_from_legacy_chunks_json(self):
        root = self._make_root()
        chunks_path = os.path.join(root, "chunks.json")
        with open(chunks_path, "w", encoding="utf-8") as f:
            json.dump(
                [{"id": 0, "uid": "chunk-1", "speaker": "Narrator", "text": "Hello world.", "status": "pending"}],
                f,
                indent=2,
            )

        manager = ProjectManager(root)
        try:
            chunks = manager.load_chunks()
            self.assertEqual(len(chunks), 1)
            self.assertEqual(chunks[0]["uid"], "chunk-1")
            self.assertTrue(os.path.exists(manager.chunks_db_path))
            self.assertFalse(os.path.exists(chunks_path))
        finally:
            manager.shutdown_script_store(flush=True)

    def test_bootstraps_store_from_annotated_script_when_db_missing(self):
        root = self._make_root()
        with open(os.path.join(root, "annotated_script.json"), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "entries": [
                        {"speaker": "Narrator", "text": "Hello world.", "instruct": "", "paragraph_id": "p1"}
                    ],
                    "dictionary": [],
                },
                f,
                indent=2,
            )

        manager = ProjectManager(root)
        try:
            chunks = manager.load_chunks()
            self.assertEqual(len(chunks), 1)
            self.assertEqual(chunks[0]["speaker"], "Narrator")
            self.assertEqual(chunks[0]["text"], "Hello world.")
            self.assertTrue(os.path.exists(manager.chunks_db_path))
        finally:
            manager.shutdown_script_store(flush=True)

    def test_exports_chunks_via_store(self):
        root = self._make_root()
        with open(os.path.join(root, "annotated_script.json"), "w", encoding="utf-8") as f:
            json.dump({"entries": [], "dictionary": []}, f)

        manager = ProjectManager(root)
        try:
            manager.save_chunks(
                [
                    {
                        "id": 0,
                        "uid": "chunk-1",
                        "speaker": "Narrator",
                        "text": "Hello world.",
                        "status": "done",
                        "audio_path": "voicelines/clip.mp3",
                    }
                ]
            )
            export_path = os.path.join(root, "exported_chunks.json")
            manager.export_chunks_to_path(export_path)
            with open(export_path, "r", encoding="utf-8") as f:
                exported = json.load(f)
            self.assertEqual(exported[0]["uid"], "chunk-1")
            self.assertTrue(manager.has_generated_chunk_audio())
            summary = manager.get_chunk_chapter_summary()
            self.assertEqual(summary["chunk_count"], 1)
        finally:
            manager.shutdown_script_store(flush=True)
