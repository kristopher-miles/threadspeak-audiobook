"""ProjectManager behavior tests grouped by domain."""

import json
import os
import tempfile
import time
import threading
import unittest
import zipfile
from unittest.mock import patch

import numpy as np
import soundfile as sf
from types import SimpleNamespace
from pydub import AudioSegment

import project as project_module
import project_core.mixins.chunk_store as chunk_store_module
from project import ProjectManager

class ReconcileChunkAudioStatesTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root_dir = self.temp_dir.name
        os.makedirs(os.path.join(self.root_dir, "voicelines"), exist_ok=True)
        os.makedirs(os.path.join(self.root_dir, "app"), exist_ok=True)

        with open(os.path.join(self.root_dir, "annotated_script.json"), "w", encoding="utf-8") as f:
            json.dump({"entries": [], "dictionary": []}, f)

        self.manager = ProjectManager(self.root_dir)
        self.manager.set_narrator_threshold(0)

    def tearDown(self):
        deadline = time.time() + 2.0
        while time.time() < deadline and self.manager.has_pending_audio_finalize_tasks():
            time.sleep(0.05)
        self.manager.clear_audio_finalize_tasks(cleanup_files=True)
        self.manager.flush_dirty_chunks(force=True)
        self.manager.shutdown_script_store(flush=True)
        self.temp_dir.cleanup()

    def _write_wav(self, relative_path, duration_seconds):
        full_path = os.path.join(self.root_dir, relative_path)
        sample_rate = 24000
        samples = np.zeros(int(sample_rate * duration_seconds), dtype=np.float32)
        sf.write(full_path, samples, sample_rate)
        return full_path

    def _write_wav(self, relative_path, duration_seconds):
        full_path = os.path.join(self.root_dir, relative_path)
        sample_rate = 24000
        samples = np.zeros(int(sample_rate * duration_seconds), dtype=np.float32)
        sf.write(full_path, samples, sample_rate)
        return full_path

    def test_reconciles_error_chunk_with_valid_audio(self):
        self._write_wav("voicelines/clip.wav", duration_seconds=3.0)
        chunks = [{
            "id": 0,
            "speaker": "Narrator",
            "text": "One two three four five six.",
            "instruct": "",
            "status": "error",
            "audio_path": "voicelines/clip.wav",
            "audio_validation": {"is_valid": False, "error": "stale"},
            "auto_regen_count": 1,
        }]
        self.manager.save_chunks(chunks)

        reconciled = self.manager.reconcile_chunk_audio_states()

        self.assertEqual(reconciled[0]["status"], "done")
        self.assertTrue(reconciled[0]["audio_validation"]["is_valid"])
        self.assertIsNone(reconciled[0]["audio_validation"]["error"])
        self.assertEqual(reconciled[0]["auto_regen_count"], 0)

    def test_promotes_pending_chunk_with_valid_audio(self):
        self._write_wav("voicelines/clip.wav", duration_seconds=3.0)
        chunks = [{
            "id": 0,
            "speaker": "Narrator",
            "text": "One two three four five six.",
            "instruct": "",
            "status": "pending",
            "audio_path": "voicelines/clip.wav",
            "audio_validation": None,
            "auto_regen_count": 0,
        }]
        self.manager.save_chunks(chunks)

        reconciled = self.manager.reconcile_chunk_audio_states()

        self.assertEqual(reconciled[0]["status"], "done")
        self.assertTrue(reconciled[0]["audio_validation"]["is_valid"])

    def test_get_chunk_audio_ref_promotes_legacy_voiceline_into_runtime_project(self):
        legacy_root = os.path.join(self.root_dir, "legacy-root")
        os.makedirs(os.path.join(legacy_root, "voicelines"), exist_ok=True)
        legacy_clip = os.path.join(legacy_root, "voicelines", "clip.wav")
        sample_rate = 24000
        samples = np.zeros(int(sample_rate * 3.0), dtype=np.float32)
        sf.write(legacy_clip, samples, sample_rate)

        chunks = [{
            "id": 0,
            "speaker": "Narrator",
            "text": "One two three four five six.",
            "instruct": "",
            "status": "done",
            "audio_path": "voicelines/clip.wav",
            "audio_validation": None,
            "auto_regen_count": 0,
        }]
        self.manager.save_chunks(chunks)
        self.manager._using_default_runtime_layout = True

        fake_layout = SimpleNamespace(
            legacy_path=lambda *parts: os.path.join(legacy_root, *parts),
        )
        with patch.object(chunk_store_module, "LAYOUT", fake_layout):
            payload = self.manager.get_chunk_audio_ref(0)

        runtime_clip = os.path.join(self.root_dir, "voicelines", "clip.wav")
        self.assertEqual(payload["audio_path"], "voicelines/clip.wav")
        self.assertTrue(os.path.exists(runtime_clip))

    def test_proofread_batch_commit_preserves_live_generation_token(self):
        chunks_disk = [{
            "id": 0,
            "speaker": "Narrator",
            "text": "One two three.",
            "instruct": "",
            "status": "generating",
            "generation_token": "live-token",
            "audio_path": None,
            "audio_validation": None,
        }]
        self.manager.save_chunks(chunks_disk)

        stale_snapshot = [{
            "id": 0,
            "speaker": "Narrator",
            "text": "One two three.",
            "instruct": "",
            "status": "pending",
            "audio_path": None,
            "audio_validation": None,
        }]
        pending_results = {
            0: {
                "checked": True,
                "passed": True,
                "score": 1.0,
                "audio_path": "",
                "transcript_text": "one two three",
            }
        }

        self.manager._commit_proofread_results_batch_locked(stale_snapshot, pending_results)
        merged = self.manager.load_chunks()

        self.assertEqual(merged[0].get("generation_token"), "live-token")
        self.assertEqual(merged[0].get("status"), "generating")
        self.assertTrue((merged[0].get("proofread") or {}).get("checked"))

    def test_load_chunks_chapter_uses_chapter_scoped_store_query(self):
        self.manager.save_chunks([
            {
                "id": 0,
                "speaker": "Narrator",
                "text": "Chapter one line.",
                "instruct": "",
                "chapter": "Chapter 1",
                "status": "pending",
                "audio_path": None,
                "audio_validation": None,
            },
            {
                "id": 1,
                "speaker": "Narrator",
                "text": "Chapter two line.",
                "instruct": "",
                "chapter": "Chapter 2",
                "status": "pending",
                "audio_path": None,
                "audio_validation": None,
            },
        ])

        calls = []
        original = self.manager.script_store.load_chunks

        def _recording_load_chunks(chapter=None):
            calls.append(chapter)
            return original(chapter=chapter)

        self.manager.script_store.load_chunks = _recording_load_chunks
        try:
            scoped = self.manager.load_chunks(chapter="Chapter 1")
        finally:
            self.manager.script_store.load_chunks = original

        self.assertEqual([chunk.get("chapter") for chunk in scoped], ["Chapter 1"])
        self.assertIn("Chapter 1", calls)

class ChunkRuntimeOverlayTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root_dir = self.temp_dir.name
        os.makedirs(os.path.join(self.root_dir, "voicelines"), exist_ok=True)
        os.makedirs(os.path.join(self.root_dir, "app"), exist_ok=True)

        with open(os.path.join(self.root_dir, "annotated_script.json"), "w", encoding="utf-8") as f:
            json.dump({"entries": [], "dictionary": []}, f)

        self.manager = ProjectManager(self.root_dir)
        self.manager.set_narrator_threshold(0)

    def tearDown(self):
        self.manager.flush_dirty_chunks(force=True)
        self.manager.shutdown_script_store(flush=True)
        self.temp_dir.cleanup()

    def _write_wav(self, relative_path, duration_seconds):
        full_path = os.path.join(self.root_dir, relative_path)
        sample_rate = 24000
        samples = np.zeros(int(sample_rate * duration_seconds), dtype=np.float32)
        sf.write(full_path, samples, sample_rate)
        return full_path

    def _make_chunk(self, chunk_id, uid, chapter=""):
        chunk = {
            "id": chunk_id,
            "uid": uid,
            "speaker": "Narrator",
            "text": f"Chunk {chunk_id} has enough words for validation to pass cleanly.",
            "instruct": "",
            "status": "pending",
            "audio_path": None,
            "audio_validation": None,
            "auto_regen_count": 0,
        }
        if chapter:
            chunk["chapter"] = chapter
        return chunk

    def test_load_chunks_view_reflects_runtime_overlay_before_flush(self):
        self.manager.save_chunks([self._make_chunk(0, "chunk-1")])

        self.manager.set_chunk_runtime(
            "chunk-1",
            status="done",
            audio_path="voicelines/chunk-1.mp3",
            audio_validation={"is_valid": True, "file_size_bytes": 42, "actual_duration_sec": 1.0},
            auto_regen_count=0,
            generation_token=None,
        )
        self.manager.mark_chunks_dirty(["chunk-1"])

        raw_chunks = self.manager.load_chunks_raw()
        view_chunks = self.manager.load_chunks_view()

        self.assertEqual(raw_chunks[0]["status"], "pending")
        self.assertIsNone(raw_chunks[0]["audio_path"])
        self.assertEqual(view_chunks[0]["status"], "done")
        self.assertEqual(view_chunks[0]["audio_path"], "voicelines/chunk-1.mp3")
        self.assertTrue(view_chunks[0]["audio_validation"]["is_valid"])

    def test_load_chunks_view_can_scope_to_a_single_chapter(self):
        self.manager.save_chunks([
            self._make_chunk(0, "chunk-1", chapter="Chapter A"),
            self._make_chunk(1, "chunk-2", chapter="Chapter B"),
            self._make_chunk(2, "chunk-3", chapter="Chapter A"),
        ])
        self.manager.set_chunk_runtime(
            "chunk-1",
            status="done",
            audio_path="voicelines/chapter-a.mp3",
            audio_validation={"is_valid": True},
            auto_regen_count=0,
            generation_token=None,
        )
        self.manager.set_chunk_runtime(
            "chunk-2",
            status="done",
            audio_path="voicelines/chapter-b.mp3",
            audio_validation={"is_valid": True},
            auto_regen_count=0,
            generation_token=None,
        )

        chapter_a_chunks = self.manager.load_chunks_view(chapter="Chapter A")

        self.assertEqual([chunk["uid"] for chunk in chapter_a_chunks], ["chunk-1", "chunk-3"])
        self.assertEqual(chapter_a_chunks[0]["status"], "done")
        self.assertEqual(chapter_a_chunks[0]["audio_path"], "voicelines/chapter-a.mp3")
        self.assertEqual(chapter_a_chunks[1]["status"], "pending")
        self.assertIsNone(chapter_a_chunks[1]["audio_path"])

    def test_load_chunks_view_is_not_blocked_by_slow_flush(self):
        self.manager.save_chunks([self._make_chunk(0, "chunk-1")])
        self.manager.set_chunk_runtime(
            "chunk-1",
            status="done",
            audio_path="voicelines/chunk-1.mp3",
            audio_validation={"is_valid": True},
            auto_regen_count=0,
            generation_token=None,
        )
        self.manager.mark_chunks_dirty(["chunk-1"])

        original_atomic_write = self.manager._atomic_json_write_raw
        write_started = threading.Event()
        release_write = threading.Event()

        def slow_atomic_write(data, target_path, max_retries=5):
            if os.path.abspath(target_path) == os.path.abspath(self.manager.chunks_path):
                write_started.set()
                release_write.wait(timeout=1.0)
            return original_atomic_write(data, target_path, max_retries=max_retries)

        with patch.object(self.manager, "_atomic_json_write_raw", side_effect=slow_atomic_write):
            flush_thread = threading.Thread(target=self.manager.flush_dirty_chunks, kwargs={"force": True}, daemon=True)
            flush_thread.start()
            self.assertTrue(write_started.wait(timeout=1.0))

            start = time.time()
            view_chunks = self.manager.load_chunks_view()
            elapsed = time.time() - start

            release_write.set()
            flush_thread.join(timeout=1.0)

        self.assertLess(elapsed, 0.1)
        self.assertEqual(view_chunks[0]["status"], "done")
        self.assertEqual(view_chunks[0]["audio_path"], "voicelines/chunk-1.mp3")

    def test_flush_dirty_chunks_batches_multiple_runtime_updates_into_one_manifest_write(self):
        self.manager.save_chunks([
            self._make_chunk(0, "chunk-1"),
            self._make_chunk(1, "chunk-2"),
        ])

        self.manager.set_chunk_runtime(
            "chunk-1",
            status="done",
            audio_path="voicelines/chunk-1.mp3",
            audio_validation={"is_valid": True},
            auto_regen_count=0,
            generation_token=None,
        )
        self.manager.set_chunk_runtime(
            "chunk-2",
            status="error",
            audio_path="voicelines/chunk-2.mp3",
            audio_validation={"is_valid": False, "error": "bad clip"},
            auto_regen_count=1,
            generation_token=None,
        )
        self.manager.mark_chunks_dirty(["chunk-1", "chunk-2"])

        original_atomic_write = self.manager._atomic_json_write_raw
        manifest_writes = []

        def counting_atomic_write(data, target_path, max_retries=5):
            if os.path.abspath(target_path) == os.path.abspath(self.manager.chunks_path):
                manifest_writes.append(target_path)
            return original_atomic_write(data, target_path, max_retries=max_retries)

        with patch.object(self.manager, "_atomic_json_write_raw", side_effect=counting_atomic_write):
            flushed = self.manager.flush_dirty_chunks(force=True)

        raw_chunks = self.manager.load_chunks_raw()

        self.assertEqual(flushed, 2)
        self.assertEqual(len(manifest_writes), 1)
        self.assertEqual(raw_chunks[0]["status"], "done")
        self.assertEqual(raw_chunks[1]["status"], "error")
        self.assertEqual(raw_chunks[1]["audio_validation"]["error"], "bad clip")

    def test_invalidate_chunk_audio_indices_clears_stale_runtime_overlay(self):
        clip_path = os.path.join(self.root_dir, "voicelines", "chunk-1.mp3")
        with open(clip_path, "wb") as handle:
            handle.write(b"stub-audio")

        chunk = self._make_chunk(0, "chunk-1")
        chunk.update({
            "status": "done",
            "audio_path": "voicelines/chunk-1.mp3",
            "audio_validation": {"file_size_bytes": 10, "actual_duration_sec": 1.0},
        })
        self.manager.save_chunks([chunk])
        self.manager.set_chunk_runtime(
            "chunk-1",
            status="done",
            audio_path="voicelines/chunk-1.mp3",
            audio_validation={"file_size_bytes": 10, "actual_duration_sec": 1.0},
            auto_regen_count=0,
            generation_token=None,
        )

        result = self.manager.invalidate_chunk_audio_indices([0])
        raw_chunks = self.manager.load_chunks_raw()
        view_chunks = self.manager.load_chunks_view()

        self.assertEqual(result["invalidated_clips"], 1)
        self.assertEqual(raw_chunks[0]["status"], "pending")
        self.assertIsNone(raw_chunks[0]["audio_path"])
        self.assertEqual(view_chunks[0]["status"], "pending")
        self.assertIsNone(view_chunks[0]["audio_path"])

    def test_script_store_finalize_queue_claim_complete_cycle(self):
        self.manager.save_chunks([
            self._make_chunk(0, "chunk-1"),
            self._make_chunk(1, "chunk-2"),
        ])
        task_one = self.manager.script_store.enqueue_audio_finalize_task({
            "chunk_uid": "chunk-1",
            "generation_token": "run-token",
            "temp_wav_path": "voicelines/.finalize_spool/run-token/chunk-1.wav",
            "attempt": 0,
            "speaker": "Narrator",
            "text": "alpha beta gamma",
        })
        task_two = self.manager.script_store.enqueue_audio_finalize_task({
            "chunk_uid": "chunk-2",
            "generation_token": "run-token",
            "temp_wav_path": "voicelines/.finalize_spool/run-token/chunk-2.wav",
            "attempt": 1,
            "speaker": "Narrator",
            "text": "delta epsilon zeta",
        })

        claimed_one = self.manager.script_store.claim_next_audio_finalize_task()
        claimed_two = self.manager.script_store.claim_next_audio_finalize_task()
        self.manager.script_store.complete_audio_finalize_task(claimed_one["id"])
        self.manager.script_store.fail_audio_finalize_task(claimed_two["id"], error="boom", requeue=False)

        pending = self.manager.script_store.list_audio_finalize_tasks()

        self.assertEqual(task_one["status"], "queued")
        self.assertEqual(task_two["status"], "queued")
        self.assertEqual(claimed_one["chunk_uid"], "chunk-1")
        self.assertEqual(claimed_two["chunk_uid"], "chunk-2")
        self.assertEqual(pending, [])

    def test_reset_generating_chunks_clears_runtime_overlay(self):
        self.manager.save_chunks([self._make_chunk(0, "chunk-1")])
        self.manager._claim_chunk_generation(0, generation_token="run-token")

        before_reset = self.manager.load_chunks_view()
        reset_count = self.manager.reset_generating_chunks([0], generation_token="run-token")
        after_reset = self.manager.load_chunks_view()
        raw_chunks = self.manager.load_chunks_raw()

        self.assertEqual(before_reset[0]["status"], "generating")
        self.assertEqual(reset_count, 1)
        self.assertEqual(after_reset[0]["status"], "pending")
        self.assertNotIn("generation_token", after_reset[0])
        self.assertEqual(raw_chunks[0]["status"], "pending")

    def test_get_chunk_view_returns_live_single_chunk_without_full_view_assembly(self):
        self.manager.save_chunks([
            self._make_chunk(0, "chunk-1", chapter="Chapter A"),
            self._make_chunk(1, "chunk-2", chapter="Chapter B"),
        ])
        self.manager.set_chunk_runtime(
            "chunk-2",
            status="done",
            audio_path="voicelines/chunk-2.mp3",
            audio_validation={"is_valid": True},
            auto_regen_count=0,
            generation_token=None,
        )

        live_chunk = self.manager.get_chunk_view("chunk-2")

        self.assertIsNotNone(live_chunk)
        self.assertEqual(live_chunk["uid"], "chunk-2")
        self.assertEqual(live_chunk["status"], "done")
        self.assertEqual(live_chunk["audio_path"], "voicelines/chunk-2.mp3")

    def test_groups_indices_by_resolved_speaker(self):
        chunks = [
            {"id": 0, "speaker": "Alice"},
            {"id": 1, "speaker": "Bob Alias"},
            {"id": 2, "speaker": "Alice"},
            {"id": 3, "speaker": "Bob"},
            {"id": 4, "speaker": "Narrator"},
        ]
        voice_config = {
            "Bob Alias": {"alias": "Bob"},
            "Bob": {},
            "Alice": {},
            "Narrator": {},
        }

        grouped = self.manager.group_indices_by_resolved_speaker(
            [0, 1, 2, 3, 4],
            chunks=chunks,
            voice_config=voice_config,
        )

        self.assertEqual(grouped, [0, 2, 1, 3, 4])

    def test_groups_indices_by_resolved_speaker_case_insensitive_keys(self):
        chunks = [
            {"id": 0, "speaker": "Narrator"},
            {"id": 1, "speaker": "NARRATOR"},
            {"id": 2, "speaker": "narrator"},
            {"id": 3, "speaker": "Alice"},
        ]
        voice_config = {
            "NARRATOR": {},
            "Alice": {},
        }

        grouped = self.manager.group_indices_by_resolved_speaker(
            [0, 1, 2, 3],
            chunks=chunks,
            voice_config=voice_config,
        )

        self.assertEqual(grouped, [0, 1, 2, 3])
        self.assertEqual(self.manager.resolve_voice_speaker("Narrator", voice_config), "NARRATOR")
        self.assertEqual(self.manager.resolve_voice_speaker("narrator", voice_config), "NARRATOR")

    def test_resolve_voice_speaker_applies_narrator_threshold(self):
        self.manager.set_narrator_threshold(10)
        self.manager.script_store.replace_script_document(
            entries=[
                {"speaker": "NARRATOR", "text": "Narration line."},
                {"speaker": "Bob", "text": "Hi."},
                {"speaker": "Alice", "text": "Line 1"},
                {"speaker": "Alice", "text": "Line 2"},
                {"speaker": "Alice", "text": "Line 3"},
                {"speaker": "Alice", "text": "Line 4"},
                {"speaker": "Alice", "text": "Line 5"},
                {"speaker": "Alice", "text": "Line 6"},
                {"speaker": "Alice", "text": "Line 7"},
                {"speaker": "Alice", "text": "Line 8"},
                {"speaker": "Alice", "text": "Line 9"},
                {"speaker": "Alice", "text": "Line 10"},
            ],
            dictionary=[],
            sanity_cache={"phrase_decisions": {}},
            reason="test_seed_script",
            rebuild_chunks=False,
            wait=True,
        )
        voice_config = {
            "NARRATOR": {},
            "Bob": {},
            "Alice": {},
        }
        self.manager.refresh_auto_narrator_aliases(voice_config=voice_config)

        self.assertEqual(
            self.manager.resolve_voice_speaker("Bob", voice_config),
            "NARRATOR",
        )
        self.assertEqual(
            self.manager.resolve_voice_speaker("Alice", voice_config),
            "Alice",
        )

    def test_resolve_voice_speaker_manual_alias_overrides_narrator_threshold(self):
        self.manager.set_narrator_threshold(10)
        with open(os.path.join(self.root_dir, "annotated_script.json"), "w", encoding="utf-8") as f:
            json.dump({
                "entries": [
                    {"speaker": "NARRATOR", "text": "Narration line."},
                    {"speaker": "Bob", "text": "Hi."},
                    {"speaker": "Alice", "text": "Line 1"},
                    {"speaker": "Alice", "text": "Line 2"},
                    {"speaker": "Alice", "text": "Line 3"},
                    {"speaker": "Alice", "text": "Line 4"},
                    {"speaker": "Alice", "text": "Line 5"},
                    {"speaker": "Alice", "text": "Line 6"},
                    {"speaker": "Alice", "text": "Line 7"},
                    {"speaker": "Alice", "text": "Line 8"},
                    {"speaker": "Alice", "text": "Line 9"},
                    {"speaker": "Alice", "text": "Line 10"},
                ],
                "dictionary": [],
            }, f)
        voice_config = {
            "NARRATOR": {},
            "Bob": {"alias": "Alice"},
            "Alice": {},
        }
        self.manager.refresh_auto_narrator_aliases(voice_config=voice_config)

        self.assertEqual(
            self.manager.resolve_voice_speaker("Bob", voice_config),
            "Alice",
        )

    def test_resolve_voice_speaker_manual_alias_to_thresholded_target_uses_stored_narrator_alias(self):
        self.manager.set_narrator_threshold(10)
        self.manager.script_store.replace_script_document(
            entries=[
                {"speaker": "NARRATOR", "text": "Narration line."},
                {"speaker": "Bob", "text": "Hi."},
                {"speaker": "Alice", "text": "Hello there."},
            ],
            dictionary=[],
            sanity_cache={"phrase_decisions": {}},
            reason="test_seed_script",
            rebuild_chunks=False,
            wait=True,
        )
        voice_config = {
            "NARRATOR": {},
            "Bob": {"alias": "Alice"},
            "Alice": {},
        }
        self.manager.refresh_auto_narrator_aliases(voice_config=voice_config)

        self.assertEqual(self.manager.get_auto_narrator_aliases().get("Alice"), "NARRATOR")
        self.assertEqual(
            self.manager.resolve_voice_speaker("Bob", voice_config),
            "NARRATOR",
        )

    def test_resolve_generation_speaker_routes_thresholded_lines_to_chapter_narrator_override(self):
        self.manager.set_narrator_threshold(10)
        self.manager.set_narrator_override("Chapter 1", "Alice")
        chunk = {"id": 1, "speaker": "Bob", "text": "Hi.", "chapter": "Chapter 1"}
        chapter_chunks = [
            {"id": 0, "speaker": "NARRATOR", "text": "Narration line.", "chapter": "Chapter 1"},
            chunk,
            {"id": 2, "speaker": "Alice", "text": "Line 1", "chapter": "Chapter 1"},
            {"id": 3, "speaker": "Alice", "text": "Line 2", "chapter": "Chapter 1"},
            {"id": 4, "speaker": "Alice", "text": "Line 3", "chapter": "Chapter 1"},
            {"id": 5, "speaker": "Alice", "text": "Line 4", "chapter": "Chapter 1"},
            {"id": 6, "speaker": "Alice", "text": "Line 5", "chapter": "Chapter 1"},
            {"id": 7, "speaker": "Alice", "text": "Line 6", "chapter": "Chapter 1"},
            {"id": 8, "speaker": "Alice", "text": "Line 7", "chapter": "Chapter 1"},
            {"id": 9, "speaker": "Alice", "text": "Line 8", "chapter": "Chapter 1"},
            {"id": 10, "speaker": "Alice", "text": "Line 9", "chapter": "Chapter 1"},
            {"id": 11, "speaker": "Alice", "text": "Line 10", "chapter": "Chapter 1"},
        ]
        voice_config = {
            "NARRATOR": {},
            "Bob": {},
            "Alice": {},
        }
        self.manager.refresh_auto_narrator_aliases(
            voice_config=voice_config,
            script_entries=chapter_chunks,
        )

        resolved = self.manager._resolve_generation_speaker(
            chunk,
            voice_config,
            narrator_overrides=self.manager.get_narrator_overrides(),
            narrator_name="NARRATOR",
        )

        self.assertEqual(resolved, "Alice")

    def test_resolve_generation_speaker_routes_alias_to_chapter_narrator_override(self):
        self.manager.set_narrator_override("Chapter 1", "Alice")
        chunk = {"id": 1, "speaker": "Bob", "text": "Hi.", "chapter": "Chapter 1"}
        voice_config = {
            "NARRATOR": {},
            "Bob": {"alias": "NARRATOR"},
            "Alice": {},
        }

        resolved = self.manager._resolve_generation_speaker(
            chunk,
            voice_config,
            narrator_overrides=self.manager.get_narrator_overrides(),
            narrator_name="NARRATOR",
        )

        self.assertEqual(resolved, "Alice")

    def test_validate_generation_voice_targets_reports_missing_voice_selection(self):
        chunks = [
            {
                "id": 0,
                "uid": "aerial-0",
                "speaker": "Aerial",
                "text": "Aerial line with enough words to render correctly.",
                "chapter": "Chapter 1",
                "instruct": "",
                "status": "pending",
                "audio_path": None,
                "audio_validation": None,
            }
        ]
        self.manager.save_chunks(chunks)
        self.manager._save_voice_config({
            "Aerial": {
                "type": "custom",
                "voice": "",
                "alias": "",
            }
        })

        issue = self.manager.validate_generation_voice_targets(["aerial-0"])

        self.assertIsNotNone(issue)
        self.assertEqual(issue["code"], "voice_config_required")
        self.assertEqual(issue["speaker"], "Aerial")
        self.assertEqual(issue["voice_speaker"], "Aerial")
        self.assertEqual(issue["chunk_uid"], "aerial-0")
        self.assertIn("no voice selected", issue["message"])

    def test_validate_generation_voice_targets_empty_speaker_uses_narrator_resolution(self):
        chunks = [
            {
                "id": 0,
                "uid": "blank-speaker-0",
                "speaker": "",
                "text": "Line without an explicit speaker still needs rendering.",
                "chapter": "Chapter 1",
                "instruct": "",
                "status": "pending",
                "audio_path": None,
                "audio_validation": None,
            }
        ]
        self.manager.save_chunks(chunks)
        self.manager._save_voice_config({
            "NARRATOR": {
                "type": "custom",
                "voice": "",
                "alias": "Ember",
            },
            "Ember": {
                "type": "custom",
                "voice": "ember",
                "alias": "",
            },
        })

        issue = self.manager.validate_generation_voice_targets(["blank-speaker-0"])

        self.assertIsNone(issue)

    def test_preview_voice_config_invalidation_resolves_once_per_generated_speaker(self):
        chunks = [
            {
                "id": 0,
                "uid": "a0",
                "speaker": "Bake",
                "text": "Bake line one.",
                "instruct": "",
                "chapter": "Chapter 1",
                "status": "done",
                "audio_path": "voicelines/a0.wav",
                "audio_validation": {"is_valid": True},
                "auto_regen_count": 0,
            },
            {
                "id": 1,
                "uid": "a1",
                "speaker": "Bake",
                "text": "Bake line two.",
                "instruct": "",
                "chapter": "Chapter 1",
                "status": "done",
                "audio_path": "voicelines/a1.wav",
                "audio_validation": {"is_valid": True},
                "auto_regen_count": 0,
            },
            {
                "id": 2,
                "uid": "b0",
                "speaker": "Blake",
                "text": "Blake line.",
                "instruct": "",
                "chapter": "Chapter 1",
                "status": "done",
                "audio_path": "voicelines/b0.wav",
                "audio_validation": {"is_valid": True},
                "auto_regen_count": 0,
            },
        ]
        self.manager.save_chunks(chunks)

        old_config = {
            "Bake": {"alias": ""},
            "Blake": {"alias": ""},
        }
        new_config = {
            "Bake": {"alias": "Blake"},
            "Blake": {"alias": ""},
        }

        original_resolve = self.manager.resolve_voice_speaker
        calls = []

        def tracked_resolve(speaker, voice_config, chunks=None, speaker_line_counts=None, narrator_name=None, auto_narrator_aliases=None):
            calls.append((speaker, tuple(sorted((voice_config or {}).keys()))))
            return original_resolve(
                speaker,
                voice_config,
                chunks=chunks,
                speaker_line_counts=speaker_line_counts,
                narrator_name=narrator_name,
                auto_narrator_aliases=auto_narrator_aliases,
            )

        self.manager.resolve_voice_speaker = tracked_resolve
        try:
            result = self.manager.preview_voice_config_invalidation(old_config, new_config)
        finally:
            self.manager.resolve_voice_speaker = original_resolve

        self.assertEqual(result["invalidated_clips"], 2)
        self.assertEqual(result["affected_indices"], [0, 1])
        self.assertEqual(result["affected_speakers"], ["Bake"])
        self.assertEqual(len(calls), 4)

    def test_generate_chunk_audio_uses_stored_auto_narrator_aliases(self):
        self.manager.set_narrator_threshold(10)
        self.manager.set_narrator_override("Chapter 1", "Alice")
        with open(os.path.join(self.root_dir, "annotated_script.json"), "w", encoding="utf-8") as f:
            json.dump({
                "entries": [
                    {"speaker": "NARRATOR", "text": "Narration line with enough words to validate correctly.", "chapter": "Chapter 1"},
                    *[
                        {
                            "speaker": "Jordan",
                            "text": f"Jordan line {index} has enough words for generation to validate correctly.",
                            "chapter": "Chapter 1",
                        }
                        for index in range(1, 12)
                    ],
                ],
                "dictionary": [],
            }, f)
        chunks = [
            {
                "id": 0,
                "uid": "narrator-0",
                "speaker": "NARRATOR",
                "text": "Narration line with enough words to validate correctly.",
                "chapter": "Chapter 1",
                "instruct": "",
                "status": "done",
                "audio_path": "voicelines/narrator-0.wav",
                "audio_validation": {"is_valid": True},
            }
        ]
        for index in range(1, 12):
            chunks.append({
                "id": index,
                "uid": f"jordan-{index}",
                "speaker": "Jordan",
                "text": f"Jordan line {index} has enough words for generation to validate correctly.",
                "chapter": "Chapter 1",
                "instruct": "",
                "status": "pending" if index == 11 else "done",
                "audio_path": None if index == 11 else f"voicelines/jordan-{index}.wav",
                "audio_validation": None if index == 11 else {"is_valid": True},
            })
        self.manager.save_chunks(chunks)

        class FakeEngine:
            def __init__(self, root_dir):
                self.root_dir = root_dir
                self.speakers = []

            def generate_voice(self, text, instruct, speaker, voice_config, temp_path):
                self.speakers.append(speaker)
                sample_rate = 24000
                samples = np.zeros(int(sample_rate * 2.0), dtype=np.float32)
                sf.write(temp_path, samples, sample_rate)
                return True

        fake_engine = FakeEngine(self.root_dir)
        self.manager.get_engine = lambda: fake_engine
        voice_config = {
            "NARRATOR": {},
            "Alice": {},
            "Jordan": {},
        }
        self.manager._load_voice_config = lambda: voice_config
        self.manager.refresh_auto_narrator_aliases(voice_config=voice_config)
        self.manager.resolve_generation_targets = lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("generate_chunk_audio should not recompute narrator thresholds from project targets")
        )

        success, _ = self.manager.generate_chunk_audio("jordan-11")

        self.assertTrue(success)
        self.assertEqual(fake_engine.speakers, ["Jordan"])

    def test_generate_chunk_audio_uses_neutral_instruction_only_for_exact_neutral_narrator(self):
        chunks = [
            {
                "id": 0,
                "uid": "narrator-exact",
                "speaker": "NARRATOR",
                "text": "Narration line has enough words to validate correctly.",
                "chapter": "Chapter 1",
                "instruct": "Warm dramatic narrator tone",
                "status": "pending",
            },
            {
                "id": 1,
                "uid": "narrator-title",
                "speaker": "Narrator",
                "text": "Title case narrator line has enough words to validate correctly.",
                "chapter": "Chapter 1",
                "instruct": "Keep title case instruct",
                "status": "pending",
            },
        ]
        self.manager.save_chunks(chunks)

        class FakeEngine:
            def __init__(self):
                self.calls = []

            def generate_voice(self, text, instruct, speaker, voice_config, temp_path):
                self.calls.append({"speaker": speaker, "instruct": instruct})
                sample_rate = 24000
                samples = np.zeros(int(sample_rate * 2.0), dtype=np.float32)
                sf.write(temp_path, samples, sample_rate)
                return True

        fake_engine = FakeEngine()
        self.manager.get_engine = lambda: fake_engine
        self.manager._load_voice_config = lambda: {"NARRATOR": {}, "Narrator": {}}

        exact_success, _ = self.manager.generate_chunk_audio("narrator-exact", neutral_narrator=True)
        title_success, _ = self.manager.generate_chunk_audio("narrator-title", neutral_narrator=True)

        self.assertTrue(exact_success)
        self.assertTrue(title_success)
        self.assertEqual(
            fake_engine.calls,
            [
                {"speaker": "NARRATOR", "instruct": "Neutral spoken delivery"},
                {"speaker": "NARRATOR", "instruct": "Keep title case instruct"},
            ],
        )

    def test_generate_chunk_audio_marks_exact_neutral_narrator_to_override_provider_control_text(self):
        self.manager.save_chunks([
            {
                "id": 0,
                "uid": "narrator-alias-to-ember",
                "speaker": "NARRATOR",
                "text": "Narration line has enough words to validate correctly.",
                "chapter": "Chapter 1",
                "instruct": "Protective, urgent intensity.",
                "status": "pending",
            }
        ])

        class FakeEngine:
            def __init__(self):
                self.calls = []

            def generate_voice(self, text, instruct, speaker, voice_config, temp_path, instruction_override=False):
                self.calls.append({
                    "speaker": speaker,
                    "instruct": instruct,
                    "instruction_override": instruction_override,
                })
                sample_rate = 24000
                samples = np.zeros(int(sample_rate * 2.0), dtype=np.float32)
                sf.write(temp_path, samples, sample_rate)
                return True

        fake_engine = FakeEngine()
        self.manager.get_engine = lambda: fake_engine
        self.manager._load_voice_config = lambda: {
            "NARRATOR": {"alias": "Ember"},
            "Ember": {
                "type": "clone",
                "description": "protective narrator style",
                "default_style": "urgent voice",
            },
        }

        success, _ = self.manager.generate_chunk_audio("narrator-alias-to-ember", neutral_narrator=True)

        self.assertTrue(success)
        self.assertEqual(
            fake_engine.calls,
            [{"speaker": "Ember", "instruct": "Neutral spoken delivery", "instruction_override": True}],
        )

    def test_generate_chunk_audio_keeps_alias_instruct_when_neutral_narrator_enabled(self):
        self.manager.set_narrator_threshold(0)
        self.manager.set_narrator_override("Chapter 1", "Alice")
        self.manager.save_chunks([
            {
                "id": 0,
                "uid": "alice-alias",
                "speaker": "Alice",
                "text": "Alice narration alias has enough words to validate correctly.",
                "chapter": "Chapter 1",
                "instruct": "Keep alias instruct",
                "status": "pending",
            }
        ])

        class FakeEngine:
            def __init__(self):
                self.calls = []

            def generate_voice(self, text, instruct, speaker, voice_config, temp_path):
                self.calls.append({"speaker": speaker, "instruct": instruct})
                sample_rate = 24000
                samples = np.zeros(int(sample_rate * 2.0), dtype=np.float32)
                sf.write(temp_path, samples, sample_rate)
                return True

        fake_engine = FakeEngine()
        self.manager.get_engine = lambda: fake_engine
        self.manager._load_voice_config = lambda: {"NARRATOR": {}, "Alice": {}}

        success, _ = self.manager.generate_chunk_audio("alice-alias", neutral_narrator=True)

        self.assertTrue(success)
        self.assertEqual(fake_engine.calls, [{"speaker": "Alice", "instruct": "Keep alias instruct"}])

    def test_generate_chunks_parallel_passes_cancel_check_to_engine_voice_generation(self):
        self.manager.save_chunks([
            {
                "id": 0,
                "uid": "clip-cancel-callback",
                "speaker": "Jordan",
                "text": "Jordan line has enough words to validate correctly.",
                "chapter": "Chapter 1",
                "instruct": "",
                "status": "pending",
            }
        ])

        seen_cancel_checks = []

        class FakeEngine:
            def generate_voice(self, text, instruct, speaker, voice_config, temp_path, cancel_check=None):
                seen_cancel_checks.append(cancel_check)
                return False

        self.manager.get_engine = lambda: FakeEngine()
        self.manager._load_voice_config = lambda: {"Jordan": {}}

        results = self.manager.generate_chunks_parallel(
            ["clip-cancel-callback"],
            max_workers=1,
            cancel_check=lambda: False,
        )

        self.assertEqual(results["completed"], [])
        self.assertEqual(len(results["failed"]), 1)
        self.assertEqual(len(seen_cancel_checks), 1)
        self.assertIsNotNone(seen_cancel_checks[0])

    def test_load_chunks_view_repairs_selected_chapter_narrator_narrates_flag(self):
        self.manager.save_chunks([
            {
                "id": 0,
                "uid": "chapter-1-narrator",
                "speaker": "NARRATOR",
                "text": "Opening narration line.",
                "chapter": "Chapter 1",
                "instruct": "",
                "status": "pending",
            }
        ])
        self.manager._save_voice_config({
            "Alice": {"type": "builtin", "voice": "AliceVoice", "narrates": False},
            "NARRATOR": {"type": "builtin", "voice": "NarratorVoice", "narrates": True},
        })
        self.manager.set_narrator_override("Chapter 1", "Alice")

        rows = self.manager.load_chunks_view(chapter="Chapter 1")

        self.assertEqual(len(rows), 1)
        refreshed = self.manager._load_voice_config()
        self.assertTrue(refreshed["Alice"]["narrates"])
        self.assertEqual(self.manager.get_narrator_overrides().get("Chapter 1"), "Alice")

    def test_disable_narrator_narration_rejects_without_alternate_narrator(self):
        self.manager._save_voice_config({
            "NARRATOR": {"type": "builtin", "voice": "NarratorVoice", "narrates": True},
            "Alice": {"type": "builtin", "voice": "AliceVoice", "narrates": False},
        })
        self.manager.set_narrator_override("Chapter 1", "NARRATOR")

        result = self.manager.disable_narrator_narration_and_reassign_chapters({
            "NARRATOR": {"narrates": False},
            "Alice": {"narrates": False},
        })

        self.assertEqual(result["status"], "rejected")
        self.assertEqual(result["code"], "narrator_disable_requires_other_narrator")
        refreshed = self.manager._load_voice_config()
        self.assertTrue(refreshed["NARRATOR"]["narrates"])
        self.assertNotIn("Chapter 1", self.manager.get_narrator_overrides())

    def test_disable_narrator_narration_reassigns_effective_narrator_chapters_and_invalidates_audio(self):
        self.manager.save_chunks([
            {
                "id": 0,
                "uid": "ch1-narrator",
                "speaker": "NARRATOR",
                "text": "Narrator audio in chapter one.",
                "chapter": "Chapter 1",
                "instruct": "",
                "status": "done",
                "audio_path": "voicelines/ch1-narrator.wav",
                "audio_validation": {"is_valid": True},
            },
            {
                "id": 1,
                "uid": "ch1-alice",
                "speaker": "Alice",
                "text": "Alice appears twice in chapter one.",
                "chapter": "Chapter 1",
                "instruct": "",
                "status": "pending",
            },
            {
                "id": 2,
                "uid": "ch1-alice-2",
                "speaker": "Alice",
                "text": "Alice appears twice in chapter one again.",
                "chapter": "Chapter 1",
                "instruct": "",
                "status": "pending",
            },
            {
                "id": 3,
                "uid": "ch1-blake",
                "speaker": "Blake",
                "text": "Blake appears once in chapter one.",
                "chapter": "Chapter 1",
                "instruct": "",
                "status": "pending",
            },
            {
                "id": 4,
                "uid": "ch2-narrator",
                "speaker": "NARRATOR",
                "text": "Narrator audio in chapter two.",
                "chapter": "Chapter 2",
                "instruct": "",
                "status": "done",
                "audio_path": "voicelines/ch2-narrator.wav",
                "audio_validation": {"is_valid": True},
            },
            {
                "id": 5,
                "uid": "ch2-blake",
                "speaker": "Blake",
                "text": "Blake appears twice in chapter two.",
                "chapter": "Chapter 2",
                "instruct": "",
                "status": "pending",
            },
            {
                "id": 6,
                "uid": "ch2-blake-2",
                "speaker": "Blake",
                "text": "Blake appears twice in chapter two again.",
                "chapter": "Chapter 2",
                "instruct": "",
                "status": "pending",
            },
            {
                "id": 7,
                "uid": "ch2-alice",
                "speaker": "Alice",
                "text": "Alice appears once in chapter two.",
                "chapter": "Chapter 2",
                "instruct": "",
                "status": "pending",
            },
            {
                "id": 8,
                "uid": "ch3-narrator",
                "speaker": "NARRATOR",
                "text": "Narrator only audio in chapter three.",
                "chapter": "Chapter 3",
                "instruct": "",
                "status": "done",
                "audio_path": "voicelines/ch3-narrator.wav",
                "audio_validation": {"is_valid": True},
            },
        ])
        self._write_wav("voicelines/ch1-narrator.wav", duration_seconds=2.0)
        self._write_wav("voicelines/ch2-narrator.wav", duration_seconds=2.0)
        self._write_wav("voicelines/ch3-narrator.wav", duration_seconds=2.0)
        self.manager._save_voice_config({
            "NARRATOR": {"type": "builtin", "voice": "NarratorVoice", "narrates": True},
            "Alice": {"type": "builtin", "voice": "AliceVoice", "narrates": True},
            "Blake": {"type": "builtin", "voice": "BlakeVoice", "narrates": True},
        })
        self.manager.set_narrator_override("Chapter 2", "NARRATOR")
        self.manager.set_narrator_override("Chapter 4", "Alice")

        result = self.manager.disable_narrator_narration_and_reassign_chapters({
            "NARRATOR": {"narrates": False},
            "Alice": {"narrates": True},
            "Blake": {"narrates": True},
        })

        self.assertEqual(result["status"], "saved")
        self.assertEqual(result["changed_chapters"], 3)
        self.assertEqual(result["invalidated_clips"], 3)
        self.assertEqual(result["deleted_files"], 3)
        self.assertEqual(result["chapter_assignments"]["Chapter 1"], "Alice")
        self.assertEqual(result["chapter_assignments"]["Chapter 2"], "Blake")
        self.assertEqual(result["chapter_assignments"]["Chapter 3"], "Alice")
        self.assertEqual(self.manager.get_narrator_overrides()["Chapter 1"], "Alice")
        self.assertEqual(self.manager.get_narrator_overrides()["Chapter 2"], "Blake")
        self.assertEqual(self.manager.get_narrator_overrides()["Chapter 3"], "Alice")
        self.assertEqual(self.manager.get_narrator_overrides()["Chapter 4"], "Alice")

        refreshed = self.manager._load_voice_config()
        self.assertFalse(refreshed["NARRATOR"]["narrates"])
        self.assertTrue(refreshed["Alice"]["narrates"])
        self.assertTrue(refreshed["Blake"]["narrates"])

        chunks = {chunk["uid"]: chunk for chunk in self.manager.load_chunks()}
        self.assertEqual(chunks["ch1-narrator"]["status"], "pending")
        self.assertIsNone(chunks["ch1-narrator"]["audio_path"])
        self.assertEqual(chunks["ch2-narrator"]["status"], "pending")
        self.assertIsNone(chunks["ch2-narrator"]["audio_path"])
        self.assertEqual(chunks["ch3-narrator"]["status"], "pending")
        self.assertIsNone(chunks["ch3-narrator"]["audio_path"])
        self.assertFalse(os.path.exists(os.path.join(self.root_dir, "voicelines/ch1-narrator.wav")))
        self.assertFalse(os.path.exists(os.path.join(self.root_dir, "voicelines/ch2-narrator.wav")))
        self.assertFalse(os.path.exists(os.path.join(self.root_dir, "voicelines/ch3-narrator.wav")))

    def test_rank_chapter_narration_candidates_matches_editor_mention_sorting(self):
        self.manager.save_chunks([
            {
                "id": 0,
                "uid": "chapter-6-narrator",
                "speaker": "NARRATOR",
                "text": "Ryan looks over. Blake answers. Blake waits beside Ryan while Blake nods.",
                "chapter": "Chapter 6",
                "instruct": "",
                "status": "pending",
            },
            {
                "id": 1,
                "uid": "chapter-6-ryan",
                "speaker": "Ryan",
                "text": "Ryan speaks only once as a chunk speaker.",
                "chapter": "Chapter 6",
                "instruct": "",
                "status": "pending",
            },
        ])

        ranked = self.manager.rank_chapter_narration_candidates("Chapter 6", ["NARRATOR", "Ryan", "Blake"])

        self.assertEqual(ranked, ["NARRATOR", "Blake", "Ryan"])

    def test_rank_chapter_narration_candidates_excludes_narrator_when_disabled(self):
        self.manager.save_chunks([
            {
                "id": 0,
                "uid": "chapter-6-narrator",
                "speaker": "NARRATOR",
                "text": "Ryan looks over. Blake answers. Blake waits beside Ryan while Blake nods.",
                "chapter": "Chapter 6",
                "instruct": "",
                "status": "pending",
            },
        ])

        ranked = self.manager.rank_chapter_narration_candidates(
            "Chapter 6",
            ["Blake", "Ryan"],
            include_narrator=False,
        )

        self.assertEqual(ranked, ["Blake", "Ryan"])

    def test_get_chapter_list_includes_non_default_narrator_label_only(self):
        self.manager.save_chunks([
            {
                "id": 0,
                "uid": "chapter-1-narrator",
                "speaker": "NARRATOR",
                "text": "Opening narration line.",
                "chapter": "Chapter 1",
                "instruct": "",
                "status": "pending",
            },
            {
                "id": 1,
                "uid": "chapter-2-narrator",
                "speaker": "NARRATOR",
                "text": "Second chapter narration line.",
                "chapter": "Chapter 2",
                "instruct": "",
                "status": "pending",
            },
        ])
        self.manager.set_narrator_override("Chapter 1", "Alice")

        chapters = self.manager.get_chapter_list()

        self.assertEqual(chapters[0]["chapter"], "Chapter 1")
        self.assertEqual(chapters[0]["narrator_label"], "Alice")
        self.assertEqual(chapters[1]["chapter"], "Chapter 2")
        self.assertEqual(chapters[1]["narrator_label"], "")

    def test_generate_chunks_batch_uses_stored_auto_narrator_aliases(self):
        self.manager.set_narrator_threshold(10)
        self.manager.set_narrator_override("Chapter 1", "Alice")
        with open(os.path.join(self.root_dir, "annotated_script.json"), "w", encoding="utf-8") as f:
            json.dump({
                "entries": [
                    {"speaker": "NARRATOR", "text": "Narration line with enough words to validate correctly.", "chapter": "Chapter 1"},
                    *[
                        {
                            "speaker": "Jordan",
                            "text": f"Jordan line {index} has enough words for generation to validate correctly.",
                            "chapter": "Chapter 1",
                        }
                        for index in range(1, 12)
                    ],
                ],
                "dictionary": [],
            }, f)
        chunks = [
            {
                "id": 0,
                "uid": "narrator-0",
                "speaker": "NARRATOR",
                "text": "Narration line with enough words to validate correctly.",
                "chapter": "Chapter 1",
                "instruct": "",
                "status": "done",
                "audio_path": "voicelines/narrator-0.wav",
                "audio_validation": {"is_valid": True},
            }
        ]
        for index in range(1, 12):
            chunks.append({
                "id": index,
                "uid": f"jordan-{index}",
                "speaker": "Jordan",
                "text": f"Jordan line {index} has enough words for generation to validate correctly.",
                "chapter": "Chapter 1",
                "instruct": "",
                "status": "pending" if index == 11 else "done",
                "audio_path": None if index == 11 else f"voicelines/jordan-{index}.wav",
                "audio_validation": None if index == 11 else {"is_valid": True},
            })
        self.manager.save_chunks(chunks)

        class FakeBatchEngine:
            def __init__(self, root_dir):
                self.root_dir = root_dir
                self.batch_speakers = []

            def generate_batch(self, batch_chunks, voice_config, output_dir, batch_seed, cancel_check=None):
                self.batch_speakers = [chunk["speaker"] for chunk in batch_chunks]
                sample_rate = 24000
                samples = np.zeros(int(sample_rate * 2.0), dtype=np.float32)
                for chunk in batch_chunks:
                    sf.write(os.path.join(output_dir, f"temp_batch_{chunk['index']}.wav"), samples, sample_rate)
                return {"completed": [chunk["index"] for chunk in batch_chunks], "failed": []}

        fake_engine = FakeBatchEngine(self.root_dir)
        self.manager.get_engine = lambda: fake_engine
        voice_config = {
            "NARRATOR": {},
            "Alice": {},
            "Jordan": {},
        }
        self.manager._load_voice_config = lambda: voice_config
        self.manager.refresh_auto_narrator_aliases(voice_config=voice_config)
        self.manager.resolve_generation_targets = lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("generate_chunks_batch should not recompute narrator thresholds from project targets")
        )

        results = self.manager.generate_chunks_batch(["jordan-11"], batch_group_by_type=True)

        self.assertEqual(results["failed"], [])
        self.assertEqual(results["completed"], ["jordan-11"])
        self.assertEqual(fake_engine.batch_speakers, ["Jordan"])

    def test_generate_chunks_batch_recovers_unreported_backend_temp_outputs(self):
        chunks = [
            {
                "id": 0,
                "uid": "clip-0",
                "speaker": "Jordan",
                "text": "Jordan line zero has enough words to validate correctly.",
                "chapter": "Chapter 1",
                "instruct": "",
                "status": "pending",
            },
            {
                "id": 1,
                "uid": "clip-1",
                "speaker": "Jordan",
                "text": "Jordan line one has enough words to validate correctly.",
                "chapter": "Chapter 1",
                "instruct": "",
                "status": "pending",
            },
        ]
        self.manager.save_chunks(chunks)

        class FakeBatchEngine:
            def generate_batch(self, batch_chunks, voice_config, output_dir, batch_seed, cancel_check=None):
                sample_rate = 24000
                samples = np.zeros(int(sample_rate * 2.0), dtype=np.float32)
                for chunk in batch_chunks:
                    sf.write(os.path.join(output_dir, f"temp_batch_{chunk['index']}.wav"), samples, sample_rate)
                return {"completed": [], "failed": []}

        self.manager.get_engine = lambda: FakeBatchEngine()
        self.manager._load_voice_config = lambda: {"Jordan": {}}

        results = self.manager.generate_chunks_batch(["clip-0", "clip-1"], batch_size=2)
        deadline = time.time() + 2.0
        stored = {}
        while time.time() < deadline:
            stored = {chunk["uid"]: chunk for chunk in self.manager.load_chunks()}
            if stored["clip-0"]["status"] == "done" and stored["clip-1"]["status"] == "done":
                break
            time.sleep(0.05)

        self.assertEqual(results["failed"], [])
        self.assertEqual(results["completed"], ["clip-0", "clip-1"])
        self.assertEqual(stored["clip-0"]["status"], "done")
        self.assertEqual(stored["clip-1"]["status"], "done")
        self.assertTrue(str(stored["clip-0"].get("audio_path") or "").endswith(".mp3"))
        self.assertTrue(str(stored["clip-1"].get("audio_path") or "").endswith(".mp3"))

    def test_generate_chunks_batch_marks_chunk_finalizing_before_async_saveback(self):
        chunks = [{
            "id": 0,
            "uid": "clip-finalizing",
            "speaker": "Jordan",
            "text": "Jordan line has enough words to validate correctly.",
            "chapter": "Chapter 1",
            "instruct": "",
            "status": "pending",
        }]
        self.manager.save_chunks(chunks)

        class FakeBatchEngine:
            def generate_batch(self, batch_chunks, voice_config, output_dir, batch_seed, cancel_check=None):
                sample_rate = 24000
                samples = np.zeros(int(sample_rate * 2.0), dtype=np.float32)
                for chunk in batch_chunks:
                    sf.write(os.path.join(output_dir, f"temp_batch_{chunk['index']}.wav"), samples, sample_rate)
                return {"completed": [chunk["index"] for chunk in batch_chunks], "failed": []}

        original_process = self.manager._process_audio_finalize_task
        started = threading.Event()
        release = threading.Event()

        def blocked_process(task):
            started.set()
            release.wait(timeout=2.0)
            return original_process(task)

        self.manager._process_audio_finalize_task = blocked_process
        self.manager.get_engine = lambda: FakeBatchEngine()
        self.manager._load_voice_config = lambda: {"Jordan": {}}
        try:
            results = self.manager.generate_chunks_batch(["clip-finalizing"], batch_size=1)
            deadline = time.time() + 2.0
            chunk = None
            while time.time() < deadline:
                chunk = self.manager.get_chunk_view("clip-finalizing")
                if chunk and chunk.get("status") == "finalizing":
                    break
                time.sleep(0.02)
            self.assertEqual(results["completed"], ["clip-finalizing"])
            self.assertIsNotNone(chunk)
            self.assertEqual(chunk["status"], "finalizing")
            chapter_view = {row["uid"]: row for row in self.manager.load_chunks_view()}
            self.assertEqual(chapter_view["clip-finalizing"]["status"], "finalizing")
            self.assertTrue(started.wait(timeout=2.0))
        finally:
            release.set()
            self.manager._process_audio_finalize_task = original_process
        deadline = time.time() + 2.0
        while time.time() < deadline:
            chunk = self.manager.get_chunk_view("clip-finalizing")
            if chunk and chunk.get("status") == "done":
                break
            time.sleep(0.02)

    def test_generate_chunks_batch_continues_submitting_when_finalizer_is_blocked(self):
        chunks = [
            {
                "id": 0,
                "uid": "clip-finalizing-0",
                "speaker": "Jordan",
                "text": "Jordan line zero has enough words to validate correctly.",
                "chapter": "Chapter 1",
                "instruct": "",
                "status": "pending",
            },
            {
                "id": 1,
                "uid": "clip-finalizing-1",
                "speaker": "Jordan",
                "text": "Jordan line one has enough words to validate correctly.",
                "chapter": "Chapter 1",
                "instruct": "",
                "status": "pending",
            },
        ]
        self.manager.save_chunks(chunks)

        class FakeBatchEngine:
            def generate_batch(self, batch_chunks, voice_config, output_dir, batch_seed, cancel_check=None):
                sample_rate = 24000
                samples = np.zeros(int(sample_rate * 2.0), dtype=np.float32)
                for chunk in batch_chunks:
                    sf.write(os.path.join(output_dir, f"temp_batch_{chunk['index']}.wav"), samples, sample_rate)
                return {"completed": [chunk["index"] for chunk in batch_chunks], "failed": []}

        original_process = self.manager._process_audio_finalize_task
        started = threading.Event()
        release = threading.Event()

        def blocked_process(task):
            started.set()
            release.wait(timeout=2.0)
            return original_process(task)

        self.manager._process_audio_finalize_task = blocked_process
        self.manager.get_engine = lambda: FakeBatchEngine()
        self.manager._load_voice_config = lambda: {"Jordan": {}}
        try:
            results = self.manager.generate_chunks_batch(
                ["clip-finalizing-0", "clip-finalizing-1"],
                batch_size=1,
            )
            self.assertEqual(results["failed"], [])
            self.assertEqual(results["completed"], ["clip-finalizing-0", "clip-finalizing-1"])
            self.assertTrue(started.wait(timeout=2.0))

            deadline = time.time() + 2.0
            while time.time() < deadline:
                view = {
                    row["uid"]: row
                    for row in self.manager.load_chunks_view()
                    if row["uid"] in {"clip-finalizing-0", "clip-finalizing-1"}
                }
                if (
                    view.get("clip-finalizing-0", {}).get("status") == "finalizing"
                    and view.get("clip-finalizing-1", {}).get("status") == "finalizing"
                ):
                    break
                time.sleep(0.02)

            self.assertEqual(view["clip-finalizing-0"]["status"], "finalizing")
            self.assertEqual(view["clip-finalizing-1"]["status"], "finalizing")
        finally:
            release.set()
            self.manager._process_audio_finalize_task = original_process

        deadline = time.time() + 2.0
        while time.time() < deadline:
            view = {
                row["uid"]: row
                for row in self.manager.load_chunks_view()
                if row["uid"] in {"clip-finalizing-0", "clip-finalizing-1"}
            }
            if (
                view.get("clip-finalizing-0", {}).get("status") == "done"
                and view.get("clip-finalizing-1", {}).get("status") == "done"
            ):
                break
            time.sleep(0.02)

    def test_generate_chunks_batch_only_marks_active_batch_generating(self):
        chunks = [
            {
                "id": 0,
                "uid": "clip-batch-0",
                "speaker": "Jordan",
                "text": "Jordan line zero has enough words to validate correctly.",
                "chapter": "Chapter 1",
                "instruct": "",
                "status": "pending",
            },
            {
                "id": 1,
                "uid": "clip-batch-1",
                "speaker": "Jordan",
                "text": "Jordan line one has enough words to validate correctly.",
                "chapter": "Chapter 1",
                "instruct": "",
                "status": "pending",
            },
        ]
        self.manager.save_chunks(chunks)

        first_batch_started = threading.Event()
        release_first_batch = threading.Event()
        generation_order = []

        class FakeBatchEngine:
            def generate_batch(engine_self, batch_chunks, voice_config, output_dir, batch_seed, cancel_check=None, log_callback=None):
                generation_order.append([chunk["index"] for chunk in batch_chunks])
                if len(generation_order) == 1:
                    first_batch_started.set()
                    release_first_batch.wait(timeout=2.0)
                sample_rate = 24000
                samples = np.zeros(int(sample_rate * 2.0), dtype=np.float32)
                for chunk in batch_chunks:
                    sf.write(os.path.join(output_dir, f"temp_batch_{chunk['index']}.wav"), samples, sample_rate)
                return {"completed": [chunk["index"] for chunk in batch_chunks], "failed": []}

        self.manager.get_engine = lambda: FakeBatchEngine()
        self.manager._load_voice_config = lambda: {"Jordan": {}}

        result_holder = {}

        def run_generation():
            result_holder["results"] = self.manager.generate_chunks_batch(
                ["clip-batch-0", "clip-batch-1"],
                batch_size=1,
            )

        worker = threading.Thread(target=run_generation, daemon=True)
        worker.start()

        self.assertTrue(first_batch_started.wait(timeout=2.0))
        deadline = time.time() + 2.0
        view = {}
        while time.time() < deadline:
            view = {
                row["uid"]: row
                for row in self.manager.load_chunks_view()
                if row["uid"] in {"clip-batch-0", "clip-batch-1"}
            }
            if (
                view.get("clip-batch-0", {}).get("status") == "generating"
                and view.get("clip-batch-1", {}).get("status") == "pending"
            ):
                break
            time.sleep(0.02)

        self.assertEqual(view["clip-batch-0"]["status"], "generating")
        self.assertEqual(view["clip-batch-1"]["status"], "pending")

        release_first_batch.set()
        worker.join(timeout=4.0)
        self.assertFalse(worker.is_alive())
        self.assertEqual(result_holder["results"]["failed"], [])
        self.assertEqual(result_holder["results"]["completed"], ["clip-batch-0", "clip-batch-1"])
        self.assertEqual(generation_order, [["clip-batch-0"], ["clip-batch-1"]])

        deadline = time.time() + 2.0
        while time.time() < deadline:
            view = {
                row["uid"]: row
                for row in self.manager.load_chunks_view()
                if row["uid"] in {"clip-batch-0", "clip-batch-1"}
            }
            if (
                view.get("clip-batch-0", {}).get("status") == "done"
                and view.get("clip-batch-1", {}).get("status") == "done"
            ):
                break
            time.sleep(0.02)

        self.assertEqual(view["clip-batch-0"]["status"], "done")
        self.assertEqual(view["clip-batch-1"]["status"], "done")

    def test_generate_chunks_batch_forwards_batch_log_callback_to_engine(self):
        self.manager.save_chunks([
            {
                "id": 7,
                "uid": "clip-log-7",
                "speaker": "Jordan",
                "text": "Jordan line seven is long enough to exercise batch logging.",
                "chapter": "Chapter 1",
                "instruct": "",
                "status": "pending",
            }
        ])

        forwarded = {"messages": [], "batch_chunks": None}

        class FakeBatchEngine:
            def generate_batch(engine_self, batch_chunks, voice_config, output_dir, batch_seed, cancel_check=None, log_callback=None):
                forwarded["batch_chunks"] = batch_chunks
                self.assertIsNotNone(log_callback)
                log_callback("Sub-batch 1/1 [custom] active 15.0s; chunk_ids=[7], uids=[clip-log], text_chars=[58], total_chars=58")
                sample_rate = 24000
                samples = np.zeros(int(sample_rate * 2.0), dtype=np.float32)
                for chunk in batch_chunks:
                    sf.write(os.path.join(output_dir, f"temp_batch_{chunk['index']}.wav"), samples, sample_rate)
                return {"completed": [chunk["index"] for chunk in batch_chunks], "failed": []}

        self.manager.get_engine = lambda: FakeBatchEngine()
        self.manager._load_voice_config = lambda: {"Jordan": {}}

        results = self.manager.generate_chunks_batch(
            ["clip-log-7"],
            batch_size=1,
            log_callback=forwarded["messages"].append,
        )

        self.assertEqual(results["failed"], [])
        self.assertEqual(results["completed"], ["clip-log-7"])
        self.assertEqual(
            forwarded["messages"],
            ["Sub-batch 1/1 [custom] active 15.0s; chunk_ids=[7], uids=[clip-log], text_chars=[58], total_chars=58"],
        )
        self.assertEqual(
            forwarded["batch_chunks"][0]["display_id"],
            self.manager.get_chunk_raw("clip-log-7")["id"],
        )

    def test_generate_chunks_batch_uses_neutral_instruction_only_for_exact_neutral_narrator(self):
        self.manager.save_chunks([
            {
                "id": 0,
                "uid": "batch-narrator-exact",
                "speaker": "NARRATOR",
                "text": "Exact narrator batch line has enough words to validate correctly.",
                "chapter": "Chapter 1",
                "instruct": "Drop this narrator instruct",
                "status": "pending",
            },
            {
                "id": 1,
                "uid": "batch-narrator-title",
                "speaker": "Narrator",
                "text": "Title case narrator batch line has enough words to validate correctly.",
                "chapter": "Chapter 1",
                "instruct": "Keep title case batch instruct",
                "status": "pending",
            },
        ])

        forwarded = {"batch_chunks": None}

        class FakeBatchEngine:
            def generate_batch(engine_self, batch_chunks, voice_config, output_dir, batch_seed, cancel_check=None):
                forwarded["batch_chunks"] = list(batch_chunks)
                sample_rate = 24000
                samples = np.zeros(int(sample_rate * 2.0), dtype=np.float32)
                for chunk in batch_chunks:
                    sf.write(os.path.join(output_dir, f"temp_batch_{chunk['index']}.wav"), samples, sample_rate)
                return {"completed": [chunk["index"] for chunk in batch_chunks], "failed": []}

        self.manager.get_engine = lambda: FakeBatchEngine()
        self.manager._load_voice_config = lambda: {"NARRATOR": {}, "Narrator": {}}

        results = self.manager.generate_chunks_batch(
            ["batch-narrator-exact", "batch-narrator-title"],
            batch_size=2,
            neutral_narrator=True,
        )

        self.assertEqual(results["failed"], [])
        self.assertEqual(results["completed"], ["batch-narrator-exact", "batch-narrator-title"])
        self.assertEqual(
            [
                {"speaker": chunk["speaker"], "instruct": chunk["instruct"]}
                for chunk in forwarded["batch_chunks"]
            ],
            [
                {"speaker": "NARRATOR", "instruct": "Neutral spoken delivery"},
                {"speaker": "NARRATOR", "instruct": "Keep title case batch instruct"},
            ],
        )
        self.assertEqual(
            [chunk["instruction_override"] for chunk in forwarded["batch_chunks"]],
            [True, False],
        )

    def test_enqueue_audio_finalize_task_returns_before_ledger_persist_completes(self):
        chunk = {
            "id": 0,
            "uid": "clip-async-ledger",
            "speaker": "Jordan",
            "text": "Jordan line has enough words to validate correctly.",
            "chapter": "Chapter 1",
            "instruct": "",
            "status": "generating",
            "generation_token": "run-async-ledger",
        }
        self.manager.save_chunks([chunk])

        temp_path = self.manager._spool_audio_full_path("clip-async-ledger", "run-async-ledger")
        sample_rate = 24000
        samples = np.zeros(int(sample_rate * 2.0), dtype=np.float32)
        sf.write(temp_path, samples, sample_rate)

        original_enqueue = self.manager.script_store.enqueue_audio_finalize_task
        allow_persist = threading.Event()

        def blocked_enqueue(task, *, reason="enqueue_audio_finalize_task", wait=True):
            allow_persist.wait(timeout=2.0)
            return original_enqueue(task, reason=reason, wait=wait)

        self.manager.script_store.enqueue_audio_finalize_task = blocked_enqueue
        task = None
        try:
            started = time.time()
            task = self.manager._enqueue_audio_finalize_task(
                "clip-async-ledger",
                "run-async-ledger",
                temp_path,
                speaker="Jordan",
                text=chunk["text"],
            )
            elapsed = time.time() - started
            self.assertIsNotNone(task)
            self.assertLess(elapsed, 0.2)
            self.assertEqual(self.manager.get_chunk_view("clip-async-ledger")["status"], "finalizing")
            allow_persist.set()
            persisted = task["persistence_future"].result(timeout=2.0)
            self.assertGreater(int((persisted or {}).get("id") or 0), 0)
        finally:
            allow_persist.set()
            self.manager.script_store.enqueue_audio_finalize_task = original_enqueue

        deadline = time.time() + 2.0
        while time.time() < deadline:
            live_chunk = self.manager.get_chunk_view("clip-async-ledger")
            if live_chunk and live_chunk.get("status") == "done":
                break
            time.sleep(0.02)

    def test_idle_finalizer_does_not_poll_script_store_claims(self):
        time.sleep(0.3)
        with open(self.manager.chunks_queue_log_path, "r", encoding="utf-8") as f:
            log_text = f.read()
        self.assertNotIn("claim_next_audio_finalize_task", log_text)

    def test_restores_persisted_audio_finalize_tasks_on_restart(self):
        chunk = {
            "id": 0,
            "uid": "clip-restart-finalize",
            "speaker": "Jordan",
            "text": "Jordan line has enough words to validate correctly.",
            "chapter": "Chapter 1",
            "instruct": "",
            "status": "generating",
            "generation_token": "run-restart-finalize",
        }
        self.manager.save_chunks([chunk])

        temp_path = self.manager._spool_audio_full_path("clip-restart-finalize", "run-restart-finalize")
        sample_rate = 24000
        samples = np.zeros(int(sample_rate * 2.0), dtype=np.float32)
        sf.write(temp_path, samples, sample_rate)
        relative_temp_path = os.path.relpath(temp_path, self.root_dir)

        persisted = self.manager.script_store.enqueue_audio_finalize_task(
            {
                "chunk_uid": "clip-restart-finalize",
                "generation_token": "run-restart-finalize",
                "temp_wav_path": relative_temp_path,
                "attempt": 0,
                "speaker": "Jordan",
                "text": chunk["text"],
            },
            wait=True,
        )
        self.assertGreater(int((persisted or {}).get("id") or 0), 0)

        self.manager.shutdown_script_store(flush=True)
        self.manager = ProjectManager(self.root_dir)
        self.manager.set_narrator_threshold(0)

        deadline = time.time() + 3.0
        live_chunk = None
        while time.time() < deadline:
            live_chunk = self.manager.get_chunk_view("clip-restart-finalize")
            if live_chunk and live_chunk.get("status") == "done":
                break
            time.sleep(0.05)

        self.assertIsNotNone(live_chunk)
        self.assertEqual(live_chunk["status"], "done")
        self.assertTrue(str(live_chunk.get("audio_path") or "").endswith(".mp3"))

    def test_recovers_interrupted_generating_chunk_with_valid_audio(self):
        self._write_wav("voicelines/recovered.wav", duration_seconds=3.0)
        chunks = [{
            "id": 0,
            "speaker": "Narrator",
            "text": "One two three four five six.",
            "instruct": "",
            "status": "generating",
            "audio_path": "voicelines/recovered.wav",
            "audio_validation": None,
            "auto_regen_count": 1,
            "generation_token": "abc",
        }]
        self.manager.save_chunks(chunks)

        outcome = self.manager.recover_interrupted_generating_chunks()
        recovered = self.manager.load_chunks()

        self.assertEqual(outcome, {"recovered": 1, "reset": 0})
        self.assertEqual(recovered[0]["status"], "done")
        self.assertTrue(recovered[0]["audio_validation"]["is_valid"])
        self.assertNotIn("generation_token", recovered[0])

    def test_should_request_auto_regen_retry_skips_overlong_audio_failures(self):
        self.manager._load_tts_settings = lambda: {
            "auto_regenerate_bad_clips": True,
            "auto_regenerate_bad_clip_attempts": 3,
        }

        self.assertFalse(
            self.manager._should_request_auto_regen_retry(
                0,
                error="Audio is too long for 5 words: 10.56s vs expected 2.35s (maximum 5.88s).",
            )
        )
        self.assertTrue(
            self.manager._should_request_auto_regen_retry(
                0,
                error="Generated audio file is missing or empty",
            )
        )
        self.assertFalse(
            self.manager._should_request_auto_regen_retry(
                3,
                error="Generated audio file is missing or empty",
            )
        )

    def test_sync_chunks_from_script_if_stale_rebuilds_when_script_is_newer(self):
        self.manager.script_store.replace_script_document(
            entries=[
                {"speaker": "NARRATOR", "text": "Chapter One", "instruct": "", "chapter": "Chapter 1"},
                {"speaker": "NARRATOR", "text": "First body text.", "instruct": "", "chapter": "Chapter 1"},
                {"speaker": "NARRATOR", "text": "Chapter Two", "instruct": "", "chapter": "Chapter 2"},
                {"speaker": "NARRATOR", "text": "Second body text.", "instruct": "", "chapter": "Chapter 2"},
            ],
            dictionary=[],
            sanity_cache={"phrase_decisions": {}},
            reason="test_seed_script",
            rebuild_chunks=True,
            wait=True,
        )

        stale_chunks = [{
            "id": 0,
            "uid": "oldchunk",
            "speaker": "NARRATOR",
            "text": "Old stale chunk.",
            "instruct": "",
            "status": "pending",
            "audio_path": None,
            "audio_validation": None,
            "auto_regen_count": 0,
            "chapter": "Old Chapter",
        }]
        self.manager.save_chunks(stale_chunks)
        result = self.manager.sync_chunks_from_script_if_stale()
        synced = self.manager.load_chunks()

        self.assertFalse(result["synced"])
        self.assertEqual(result["reason"], "db_transactional")
        self.assertEqual(len(synced), 1)
        self.assertEqual(synced[0]["uid"], "oldchunk")
        self.assertEqual(synced[0]["chapter"], "Old Chapter")

    def test_sync_chunks_from_script_if_stale_skips_when_chunks_are_current(self):
        self.manager.script_store.replace_script_document(
            entries=[
                {"speaker": "NARRATOR", "text": "Chapter One", "instruct": "", "chapter": "Chapter 1"},
                {"speaker": "NARRATOR", "text": "Current chunk text.", "instruct": "", "chapter": "Chapter 1"},
            ],
            dictionary=[],
            sanity_cache={"phrase_decisions": {}},
            reason="test_seed_script",
            rebuild_chunks=True,
            wait=True,
        )

        current_chunks = [{
            "id": 0,
            "uid": "currentchunk",
            "speaker": "NARRATOR",
            "text": "Current chunk text.",
            "instruct": "",
            "status": "done",
            "audio_path": "voicelines/current.mp3",
            "audio_validation": {"is_valid": True},
            "auto_regen_count": 0,
            "chapter": "Chapter 1",
        }]
        self.manager.save_chunks(current_chunks)
        result = self.manager.sync_chunks_from_script_if_stale()
        synced = self.manager.load_chunks()

        self.assertFalse(result["synced"])
        self.assertEqual(result["reason"], "db_transactional")
        self.assertEqual(synced[0]["uid"], "currentchunk")
        self.assertEqual(synced[0]["status"], "done")

    def test_sync_chunks_from_script_if_stale_refuses_to_discard_generated_audio(self):
        self.manager.script_store.replace_script_document(
            entries=[
                {"speaker": "NARRATOR", "text": "Chapter One", "instruct": "", "chapter": "Chapter 1"},
                {"speaker": "NARRATOR", "text": "Current chunk text.", "instruct": "", "chapter": "Chapter 1"},
            ],
            dictionary=[],
            sanity_cache={"phrase_decisions": {}},
            reason="test_seed_script",
            rebuild_chunks=True,
            wait=True,
        )

        current_chunks = [{
            "id": 0,
            "uid": "currentchunk",
            "speaker": "NARRATOR",
            "text": "Current chunk text.",
            "instruct": "",
            "status": "done",
            "audio_path": "voicelines/current.mp3",
            "audio_validation": {"is_valid": True},
            "auto_regen_count": 0,
            "chapter": "Chapter 1",
        }]
        self.manager.save_chunks(current_chunks)
        result = self.manager.sync_chunks_from_script_if_stale()
        synced = self.manager.load_chunks()

        self.assertFalse(result["synced"])
        self.assertEqual(result["reason"], "db_transactional")
        self.assertEqual(synced[0]["uid"], "currentchunk")
        self.assertEqual(synced[0]["status"], "done")
        self.assertEqual(synced[0]["audio_path"], "voicelines/current.mp3")

    def test_sync_chunks_from_script_if_stale_preserves_matching_chunk_uid_state(self):
        self.manager.script_store.replace_script_document(
            entries=[
                {
                    "speaker": "NARRATOR",
                    "text": "Sentence one.",
                    "instruct": "",
                    "chapter": "Chapter 1",
                    "paragraph_id": "p_0001",
                },
            ],
            dictionary=[],
            sanity_cache={"phrase_decisions": {}},
            reason="test_seed_script",
            rebuild_chunks=True,
            wait=True,
        )

        current_chunks = [{
            "id": 0,
            "uid": "preserved-uid",
            "speaker": "NARRATOR",
            "text": "Sentence one.",
            "instruct": "",
            "status": "pending",
            "audio_path": None,
            "audio_validation": None,
            "auto_regen_count": 0,
            "chapter": "Chapter 1",
            "paragraph_id": "p_0001",
        }]
        self.manager.save_chunks(current_chunks)
        result = self.manager.sync_chunks_from_script_if_stale()
        synced = self.manager.load_chunks()

        self.assertFalse(result["synced"])
        self.assertEqual(result["reason"], "db_transactional")
        self.assertEqual(len(synced), 1)
        self.assertEqual(synced[0]["uid"], "preserved-uid")
        self.assertEqual(synced[0]["text"], "Sentence one.")

    def test_sync_chunks_from_script_if_stale_preserves_sentence_level_entries_with_paragraph_ids(self):
        self.manager.script_store.replace_script_document(
            entries=[
                {
                    "speaker": "NARRATOR",
                    "text": "I'm looking for another explanation.",
                    "instruct": "",
                    "chapter": "Chapter 1",
                    "paragraph_id": "p_0001",
                },
                {
                    "speaker": "NARRATOR",
                    "text": "But it's hard to think of what else would do this.",
                    "instruct": "",
                    "chapter": "Chapter 1",
                    "paragraph_id": "p_0001",
                },
            ],
            dictionary=[],
            sanity_cache={"phrase_decisions": {}},
            reason="test_seed_script",
            rebuild_chunks=True,
            wait=True,
        )

        stale_chunks = [{
            "id": 0,
            "uid": "oldchunk",
            "speaker": "NARRATOR",
            "text": "Old stale chunk.",
            "instruct": "",
            "status": "pending",
            "audio_path": None,
            "audio_validation": None,
            "auto_regen_count": 0,
            "chapter": "Old Chapter",
        }]
        self.manager.save_chunks(stale_chunks)
        result = self.manager.sync_chunks_from_script_if_stale()
        synced = self.manager.load_chunks()

        self.assertFalse(result["synced"])
        self.assertEqual(result["reason"], "db_transactional")
        self.assertEqual(len(synced), 1)
        self.assertEqual(synced[0]["uid"], "oldchunk")
