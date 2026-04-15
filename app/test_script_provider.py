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

    def test_does_not_bootstrap_store_from_legacy_chunks_json(self):
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
            self.assertEqual(chunks, [])
            self.assertTrue(os.path.exists(manager.chunks_db_path))
            self.assertTrue(os.path.exists(chunks_path))
        finally:
            manager.shutdown_script_store(flush=True)

    def test_does_not_bootstrap_store_from_annotated_script_when_db_missing(self):
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
            self.assertEqual(chunks, [])
            self.assertFalse(manager.script_store.has_script_entries())
            self.assertTrue(os.path.exists(manager.chunks_db_path))
        finally:
            manager.shutdown_script_store(flush=True)

    def test_exports_chunks_via_store(self):
        root = self._make_root()
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

    def test_resolve_generation_targets_includes_error_rows_when_pending_only(self):
        root = self._make_root()
        manager = ProjectManager(root)
        try:
            manager.save_chunks(
                [
                    {
                        "id": 0,
                        "uid": "pending-1",
                        "speaker": "Narrator",
                        "text": "Pending row.",
                        "status": "pending",
                    },
                    {
                        "id": 1,
                        "uid": "error-1",
                        "speaker": "Narrator",
                        "text": "Errored row should still be retried.",
                        "status": "error",
                    },
                    {
                        "id": 2,
                        "uid": "done-1",
                        "speaker": "Narrator",
                        "text": "Completed row.",
                        "status": "done",
                    },
                ]
            )

            targets = manager.resolve_generation_targets(scope_mode="project", pending_only=True)
            self.assertEqual([chunk["uid"] for chunk in targets], ["pending-1", "error-1"])
        finally:
            manager.shutdown_script_store(flush=True)

    def test_audio_coverage_summary_counts_only_valid_audio(self):
        root = self._make_root()
        manager = ProjectManager(root)
        try:
            manager.save_chunks(
                [
                    {
                        "id": 0,
                        "uid": "valid-1",
                        "speaker": "Narrator",
                        "text": "Valid audio.",
                        "status": "done",
                        "audio_path": "voicelines/valid.wav",
                        "audio_validation": {"is_valid": True, "file_size_bytes": 123, "actual_duration_sec": 1.0},
                    },
                    {
                        "id": 1,
                        "uid": "invalid-1",
                        "speaker": "Narrator",
                        "text": "Invalid audio.",
                        "status": "done",
                        "audio_path": "voicelines/invalid.wav",
                        "audio_validation": {"is_valid": False, "error": "duration mismatch"},
                    },
                    {
                        "id": 2,
                        "uid": "pending-1",
                        "speaker": "Narrator",
                        "text": "Pending audio.",
                        "status": "pending",
                        "audio_path": None,
                        "audio_validation": None,
                    },
                    {
                        "id": 3,
                        "uid": "blank-1",
                        "speaker": "Narrator",
                        "text": "   ",
                        "status": "pending",
                        "audio_path": None,
                        "audio_validation": None,
                    },
                ]
            )

            summary = manager.get_audio_coverage_summary()
            self.assertEqual(summary["total_clips"], 3)
            self.assertEqual(summary["valid_clips"], 1)
            self.assertEqual(summary["invalid_clips"], 2)
            self.assertEqual(summary["percentage"], 33)
        finally:
            manager.shutdown_script_store(flush=True)

    def test_preserve_chunk_state_does_not_reuse_same_uid_between_exact_and_fallback_matches(self):
        root = self._make_root()
        manager = ProjectManager(root)
        try:
            existing = [
                {
                    "uid": "chunk-a",
                    "speaker": "NARRATOR",
                    "text": "Repeated line.",
                    "instruct": "Neutral.",
                    "chapter": "Chapter 1",
                    "paragraph_id": "p1",
                    "status": "done",
                },
                {
                    "uid": "chunk-b",
                    "speaker": "NARRATOR",
                    "text": "Repeated line.",
                    "instruct": "Neutral.",
                    "chapter": "Chapter 1",
                    "paragraph_id": "p2",
                    "status": "pending",
                },
            ]
            rebuilt = [
                {
                    "speaker": "NARRATOR",
                    "text": "Repeated line.",
                    "instruct": "Neutral.",
                    "chapter": "Chapter 1",
                    "paragraph_id": "p1",
                },
                {
                    "speaker": "NARRATOR",
                    "text": "Repeated line.",
                    "instruct": "Neutral.",
                    "chapter": "Chapter 1",
                    "paragraph_id": "p3",
                },
            ]

            preserved = manager.script_store._preserve_chunk_state_for_entries(existing, rebuilt)
            preserved_uids = [chunk.get("uid") for chunk in preserved]
            self.assertEqual(len(set(preserved_uids)), 2)
            self.assertIn("chunk-a", preserved_uids)
            self.assertIn("chunk-b", preserved_uids)
        finally:
            manager.shutdown_script_store(flush=True)
