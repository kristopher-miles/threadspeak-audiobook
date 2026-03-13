import json
import os
import tempfile
import unittest
import zipfile

import numpy as np
import soundfile as sf

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

    def tearDown(self):
        self.temp_dir.cleanup()

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

    def test_does_not_promote_pending_chunk_with_old_audio(self):
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

        self.assertEqual(reconciled[0]["status"], "pending")
        self.assertIsNone(reconciled[0]["audio_validation"])

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

    def test_resets_interrupted_generating_chunk_without_valid_audio(self):
        chunks = [{
            "id": 0,
            "speaker": "Narrator",
            "text": "One two three four five six.",
            "instruct": "",
            "status": "generating",
            "audio_path": "voicelines/missing.wav",
            "audio_validation": None,
            "auto_regen_count": 0,
            "generation_token": "abc",
        }]
        self.manager.save_chunks(chunks)

        outcome = self.manager.recover_interrupted_generating_chunks()
        recovered = self.manager.load_chunks()

        self.assertEqual(outcome, {"recovered": 0, "reset": 1})
        self.assertEqual(recovered[0]["status"], "pending")
        self.assertNotIn("generation_token", recovered[0])

    def test_collect_voice_suggestion_context_uses_story_order_and_target_chars(self):
        with open(os.path.join(self.root_dir, "state.json"), "w", encoding="utf-8") as f:
            json.dump({"input_file_path": os.path.join(self.root_dir, "story.txt")}, f)
        with open(os.path.join(self.root_dir, "story.txt"), "w", encoding="utf-8") as f:
            f.write(
                "Alice stepped into the hall.\n\n"
                "Bob answered from the stair.\n\n"
                "Alice spoke again near the window.\n\n"
                "Alice kept talking until the lamps burned low."
            )
        with open(os.path.join(self.root_dir, "app", "config.json"), "w", encoding="utf-8") as f:
            json.dump({"generation": {"chunk_size": 20}}, f)

        context = self.manager.collect_voice_suggestion_context("Alice")

        self.assertEqual(context["target_chars"], 40)
        self.assertEqual(
            [item["text"] for item in context["paragraphs"]],
            [
                "Alice stepped into the hall.",
                "Alice spoke again near the window.",
            ],
        )
        self.assertGreaterEqual(context["context_chars"], 40)

    def test_build_voice_suggestion_prompt_places_prompt_after_context(self):
        with open(os.path.join(self.root_dir, "state.json"), "w", encoding="utf-8") as f:
            json.dump({"input_file_path": os.path.join(self.root_dir, "story.txt")}, f)
        with open(os.path.join(self.root_dir, "story.txt"), "w", encoding="utf-8") as f:
            f.write("Alice laughed softly.\n\nAlice took a breath.")

        payload = self.manager.build_voice_suggestion_prompt(
            "Alice",
            'Return {"voice":"for {character_name}"}',
        )

        self.assertIn('Source paragraphs mentioning "Alice"', payload["prompt"])
        self.assertTrue(payload["prompt"].endswith('Return {"voice":"for Alice"}'))

    def test_render_prep_flag_persists_in_state(self):
        self.assertFalse(self.manager.is_render_prep_complete())

        self.assertTrue(self.manager.set_render_prep_complete(True))
        self.assertTrue(self.manager.is_render_prep_complete())

        state_path = os.path.join(self.root_dir, "state.json")
        with open(state_path, "r", encoding="utf-8") as f:
            state = json.load(f)
        self.assertTrue(state["render_prep_complete"])

    def test_auto_regen_retry_attempts_uses_positive_config_value(self):
        config_path = os.path.join(self.root_dir, "app", "config.json")
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump({"tts": {"auto_regenerate_bad_clips": True, "auto_regenerate_bad_clip_attempts": 3}}, f)

        self.assertEqual(self.manager._get_auto_regen_retry_attempts(), 3)

    def test_auto_regen_retry_attempts_disables_on_zero_or_invalid(self):
        config_path = os.path.join(self.root_dir, "app", "config.json")

        with open(config_path, "w", encoding="utf-8") as f:
            json.dump({"tts": {"auto_regenerate_bad_clips": True, "auto_regenerate_bad_clip_attempts": 0}}, f)
        self.assertEqual(self.manager._get_auto_regen_retry_attempts(), 0)

        with open(config_path, "w", encoding="utf-8") as f:
            json.dump({"tts": {"auto_regenerate_bad_clips": True, "auto_regenerate_bad_clip_attempts": "bad"}}, f)
        self.assertEqual(self.manager._get_auto_regen_retry_attempts(), 0)

        with open(config_path, "w", encoding="utf-8") as f:
            json.dump({"tts": {"auto_regenerate_bad_clips": False, "auto_regenerate_bad_clip_attempts": 5}}, f)
        self.assertEqual(self.manager._get_auto_regen_retry_attempts(), 0)

    def test_load_chunks_preserves_corrupt_file_instead_of_regenerating(self):
        script_entries = [
            {"speaker": "Narrator", "text": "Fresh script line.", "instruct": ""}
        ]
        with open(os.path.join(self.root_dir, "annotated_script.json"), "w", encoding="utf-8") as f:
            json.dump({"entries": script_entries, "dictionary": []}, f)

        chunks_path = os.path.join(self.root_dir, "chunks.json")
        with open(chunks_path, "w", encoding="utf-8") as f:
            f.write("{not valid json")

        with self.assertRaises(RuntimeError):
            self.manager.load_chunks()

        backups = [name for name in os.listdir(self.root_dir) if name.startswith("chunks.json.corrupt-")]
        self.assertTrue(backups)
        with open(chunks_path, "r", encoding="utf-8") as f:
            self.assertEqual(f.read(), "{not valid json")


class MergeAudioTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root_dir = self.temp_dir.name
        os.makedirs(os.path.join(self.root_dir, "voicelines"), exist_ok=True)
        os.makedirs(os.path.join(self.root_dir, "app"), exist_ok=True)
        with open(os.path.join(self.root_dir, "annotated_script.json"), "w", encoding="utf-8") as f:
            json.dump({"entries": [], "dictionary": []}, f)
        self.manager = ProjectManager(self.root_dir)

    def tearDown(self):
        self.temp_dir.cleanup()

    def _write_wav(self, relative_path, duration_seconds):
        full_path = os.path.join(self.root_dir, relative_path)
        sample_rate = 24000
        samples = np.zeros(int(sample_rate * duration_seconds), dtype=np.float32)
        sf.write(full_path, samples, sample_rate)
        return full_path

    def test_merge_audio_reports_progress_and_creates_mp3(self):
        self._write_wav("voicelines/clip1.wav", duration_seconds=0.5)
        self._write_wav("voicelines/clip2.wav", duration_seconds=0.5)
        self.manager.save_chunks([
            {
                "id": 0,
                "speaker": "Narrator",
                "text": "One.",
                "instruct": "",
                "chapter": "Chapter 1",
                "status": "done",
                "audio_path": "voicelines/clip1.wav",
            },
            {
                "id": 1,
                "speaker": "Narrator",
                "text": "Two.",
                "instruct": "",
                "chapter": "Chapter 2",
                "status": "done",
                "audio_path": "voicelines/clip2.wav",
            },
        ])

        progress = []
        success, output_filename = self.manager.merge_audio(progress_callback=progress.append)

        self.assertTrue(success)
        self.assertEqual(output_filename, "cloned_audiobook.mp3")
        self.assertTrue(os.path.exists(os.path.join(self.root_dir, output_filename)))
        self.assertGreater(os.path.getsize(os.path.join(self.root_dir, output_filename)), 0)
        stages = [item.get("stage") for item in progress]
        self.assertIn("preparing", stages)
        self.assertIn("assembling", stages)
        self.assertIn("exporting", stages)
        self.assertEqual(stages[-1], "complete")

    def test_optimized_export_creates_ordered_zip_parts(self):
        with open(os.path.join(self.root_dir, "state.json"), "w", encoding="utf-8") as f:
            json.dump({"input_file_path": os.path.join(self.root_dir, "My Great Book.txt")}, f)

        self._write_wav("voicelines/clip1.wav", duration_seconds=0.5)
        self._write_wav("voicelines/clip2.wav", duration_seconds=0.5)
        self._write_wav("voicelines/clip3.wav", duration_seconds=0.5)
        self.manager.save_chunks([
            {
                "id": 0,
                "speaker": "Narrator",
                "text": "One.",
                "instruct": "",
                "chapter": "Chapter 1",
                "status": "done",
                "audio_path": "voicelines/clip1.wav",
            },
            {
                "id": 1,
                "speaker": "Narrator",
                "text": "Two.",
                "instruct": "",
                "chapter": "Chapter 2",
                "status": "done",
                "audio_path": "voicelines/clip2.wav",
            },
            {
                "id": 2,
                "speaker": "Narrator",
                "text": "Three.",
                "instruct": "",
                "chapter": "Chapter 3",
                "status": "done",
                "audio_path": "voicelines/clip3.wav",
            },
        ])

        success, output_filename = self.manager.export_optimized_mp3_zip(max_part_seconds=1.4)

        self.assertTrue(success)
        self.assertEqual(output_filename, "optimized_audiobook.zip")
        zip_path = os.path.join(self.root_dir, output_filename)
        self.assertTrue(os.path.exists(zip_path))
        with zipfile.ZipFile(zip_path, "r") as zf:
            self.assertEqual(
                zf.namelist(),
                ["my-great-book-01.mp3", "my-great-book-02.mp3"],
            )

    def test_repair_legacy_chunk_order_rewrites_chunks_from_editor_order(self):
        original = [
            {
                "id": 0,
                "speaker": "Narrator",
                "text": "First",
                "instruct": "",
                "chapter": "Chapter 1",
                "status": "done",
                "audio_path": "voicelines/clip1.wav",
            },
            {
                "id": 1,
                "speaker": "Narrator",
                "text": "Second",
                "instruct": "",
                "chapter": "Chapter 1",
                "status": "done",
                "audio_path": "voicelines/clip2.wav",
            },
        ]
        self.manager.save_chunks(original)

        repaired = self.manager.repair_legacy_chunk_order([
            {
                "id": 99,
                "speaker": "Narrator",
                "text": "Second",
                "instruct": "",
                "chapter": "Chapter 1",
                "status": "done",
                "audio_path": "voicelines/clip2.wav",
            },
            {
                "id": 42,
                "speaker": "Narrator",
                "text": "Replacement First",
                "instruct": "calm",
                "chapter": "Chapter 1",
                "status": "pending",
                "audio_path": None,
            },
        ])

        self.assertEqual([chunk["id"] for chunk in repaired], [0, 1])
        self.assertEqual([chunk["text"] for chunk in repaired], ["Second", "Replacement First"])
        persisted = self.manager.load_chunks()
        self.assertEqual([chunk["text"] for chunk in persisted], ["Second", "Replacement First"])

    def test_load_chunks_backfills_stable_uids_for_legacy_rows(self):
        legacy = [
            {"id": 0, "speaker": "Narrator", "text": "One", "instruct": "", "status": "pending", "audio_path": None},
            {"id": 1, "speaker": "Narrator", "text": "Two", "instruct": "", "status": "pending", "audio_path": None},
        ]
        with open(os.path.join(self.root_dir, "chunks.json"), "w", encoding="utf-8") as f:
            json.dump(legacy, f, indent=2)

        loaded = self.manager.load_chunks()

        self.assertEqual(len(loaded), 2)
        self.assertTrue(all(chunk.get("uid") for chunk in loaded))
        self.assertNotEqual(loaded[0]["uid"], loaded[1]["uid"])

    def test_delete_and_restore_use_stable_uid(self):
        self.manager.save_chunks([
            {"id": 0, "speaker": "Narrator", "text": "One", "instruct": "", "status": "pending", "audio_path": None},
            {"id": 1, "speaker": "Narrator", "text": "Two", "instruct": "", "status": "pending", "audio_path": None},
            {"id": 2, "speaker": "Narrator", "text": "Three", "instruct": "", "status": "pending", "audio_path": None},
        ])
        initial = self.manager.load_chunks()
        deleted_uid = initial[1]["uid"]
        previous_uid = initial[0]["uid"]

        deleted, remaining, restore_after_uid = self.manager.delete_chunk(deleted_uid)
        self.assertEqual(deleted["text"], "Two")
        self.assertEqual(restore_after_uid, previous_uid)
        self.assertEqual([chunk["text"] for chunk in remaining], ["One", "Three"])

        restored = self.manager.restore_chunk(0, deleted, after_uid=restore_after_uid)
        self.assertEqual([chunk["text"] for chunk in restored], ["One", "Two", "Three"])


class DecomposeLongSegmentsTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root_dir = self.temp_dir.name
        os.makedirs(os.path.join(self.root_dir, "voicelines"), exist_ok=True)
        os.makedirs(os.path.join(self.root_dir, "app"), exist_ok=True)

        with open(os.path.join(self.root_dir, "annotated_script.json"), "w", encoding="utf-8") as f:
            json.dump({"entries": [], "dictionary": []}, f)

        self.manager = ProjectManager(self.root_dir)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_recursively_splits_unsynthesized_segments_until_within_limit(self):
        chunks = [{
            "id": 0,
            "speaker": "Narrator",
            "text": (
                "One two three four five six seven eight. "
                "Nine ten eleven twelve thirteen fourteen fifteen sixteen. "
                "Seventeen eighteen nineteen twenty twenty one twenty two twenty three twenty four. "
                "Twenty five twenty six twenty seven twenty eight twenty nine thirty thirty one thirty two."
            ),
            "instruct": "Warm and steady",
            "status": "pending",
            "audio_path": None,
            "audio_validation": None,
            "auto_regen_count": 0,
            "chapter": "Chapter 1",
        }]
        self.manager.save_chunks(chunks)

        result = self.manager.decompose_long_segments(max_words=15)
        updated = self.manager.load_chunks()

        self.assertEqual(result["changed"], 3)
        self.assertEqual(len(updated), 4)
        self.assertTrue(all(self.manager._count_words(chunk["text"]) <= 15 for chunk in updated))
        self.assertTrue(all(chunk["speaker"] == "Narrator" for chunk in updated))
        self.assertTrue(all(chunk["instruct"] == "Warm and steady" for chunk in updated))
        self.assertEqual([chunk["id"] for chunk in updated], [0, 1, 2, 3])

    def test_does_not_split_segment_that_already_has_audio(self):
        chunks = [{
            "id": 0,
            "speaker": "Narrator",
            "text": (
                "One two three four five six seven eight nine ten. "
                "Eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty."
            ),
            "instruct": "",
            "status": "done",
            "audio_path": "voicelines/existing.wav",
            "audio_validation": {"is_valid": True},
            "auto_regen_count": 0,
            "chapter": "Chapter 1",
        }]
        self.manager.save_chunks(chunks)

        result = self.manager.decompose_long_segments(max_words=5)
        updated = self.manager.load_chunks()

        self.assertEqual(result["changed"], 0)
        self.assertEqual(len(updated), 1)
        self.assertEqual(updated[0]["audio_path"], "voicelines/existing.wav")

    def test_does_not_split_without_sentence_boundaries(self):
        chunks = [{
            "id": 0,
            "speaker": "Narrator",
            "text": " ".join(f"word{i}" for i in range(30)),
            "instruct": "",
            "status": "pending",
            "audio_path": None,
            "audio_validation": None,
            "auto_regen_count": 0,
        }]
        self.manager.save_chunks(chunks)

        result = self.manager.decompose_long_segments(max_words=10)
        updated = self.manager.load_chunks()

        self.assertEqual(result["changed"], 0)
        self.assertEqual(len(updated), 1)
        self.assertEqual(updated[0]["text"], chunks[0]["text"])

    def test_limits_to_requested_chapter(self):
        chunks = [
            {
                "id": 0,
                "speaker": "Narrator",
                "text": "One two three four five six. Seven eight nine ten eleven twelve.",
                "instruct": "Calm",
                "status": "pending",
                "audio_path": None,
                "audio_validation": None,
                "auto_regen_count": 0,
                "chapter": "Chapter 1",
            },
            {
                "id": 1,
                "speaker": "Narrator",
                "text": "Alpha beta gamma delta epsilon zeta. Eta theta iota kappa lambda mu.",
                "instruct": "Calm",
                "status": "pending",
                "audio_path": None,
                "audio_validation": None,
                "auto_regen_count": 0,
                "chapter": "Chapter 2",
            },
        ]
        self.manager.save_chunks(chunks)

        result = self.manager.decompose_long_segments(chapter="Chapter 2", max_words=6)
        updated = self.manager.load_chunks()

        self.assertEqual(result["changed"], 1)
        self.assertEqual(len(updated), 3)
        self.assertEqual(updated[0]["chapter"], "Chapter 1")
        self.assertEqual(updated[0]["text"], chunks[0]["text"])
        self.assertEqual(updated[1]["chapter"], "Chapter 2")
        self.assertEqual(updated[2]["chapter"], "Chapter 2")


class MergeOrphanSegmentsTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root_dir = self.temp_dir.name
        os.makedirs(os.path.join(self.root_dir, "voicelines"), exist_ok=True)
        os.makedirs(os.path.join(self.root_dir, "app"), exist_ok=True)

        with open(os.path.join(self.root_dir, "annotated_script.json"), "w", encoding="utf-8") as f:
            json.dump({"entries": [], "dictionary": []}, f)

        self.manager = ProjectManager(self.root_dir)

    def tearDown(self):
        self.temp_dir.cleanup()

    def _chapter_intro_chunks(self, chapter, start_id=0, speaker="Narrator"):
        chunks = []
        for offset in range(5):
            chunks.append({
                "id": start_id + offset,
                "speaker": speaker,
                "text": f"Protected intro line {offset}.",
                "instruct": "",
                "status": "pending",
                "audio_path": None,
                "audio_validation": None,
                "auto_regen_count": 0,
                "chapter": chapter,
            })
        return chunks

    def test_merges_short_segment_with_adjacent_same_speaker_until_threshold(self):
        chunks = self._chapter_intro_chunks("Chapter 1")
        chunks.extend([
            {
                "id": 5,
                "speaker": "Narrator",
                "text": "Hello there friend.",
                "instruct": "",
                "status": "done",
                "audio_path": "voicelines/a.wav",
                "audio_validation": {"is_valid": True},
                "auto_regen_count": 0,
                "chapter": "Chapter 1",
            },
            {
                "id": 6,
                "speaker": "Narrator",
                "text": "We meet again tonight.",
                "instruct": "Gentle",
                "status": "done",
                "audio_path": "voicelines/b.wav",
                "audio_validation": {"is_valid": True},
                "auto_regen_count": 0,
                "chapter": "Chapter 1",
            },
            {
                "id": 7,
                "speaker": "Narrator",
                "text": "Stay close now.",
                "instruct": "",
                "status": "pending",
                "audio_path": None,
                "audio_validation": None,
                "auto_regen_count": 0,
                "chapter": "Chapter 1",
            },
        ])
        self.manager.save_chunks(chunks)

        result = self.manager.merge_orphan_segments(min_words=10)
        updated = self.manager.load_chunks()

        self.assertEqual(result["changed"], 2)
        self.assertEqual(len(updated), 6)
        self.assertEqual(updated[5]["text"], "Hello there friend. We meet again tonight. Stay close now.")
        self.assertEqual(updated[5]["instruct"], "Gentle")
        self.assertEqual(updated[5]["status"], "pending")
        self.assertIsNone(updated[5]["audio_path"])
        self.assertIsNone(updated[5]["audio_validation"])

    def test_prefers_first_non_empty_instruction_and_can_merge_forward(self):
        chunks = self._chapter_intro_chunks("Chapter 1", speaker="Alice")
        chunks.extend([
            {
                "id": 5,
                "speaker": "Alice",
                "text": "Hi there.",
                "instruct": "Bright",
                "status": "pending",
                "audio_path": None,
                "audio_validation": None,
                "auto_regen_count": 0,
                "chapter": "Chapter 1",
            },
            {
                "id": 6,
                "speaker": "Alice",
                "text": "Come inside now please.",
                "instruct": "Serious",
                "status": "pending",
                "audio_path": None,
                "audio_validation": None,
                "auto_regen_count": 0,
                "chapter": "Chapter 1",
            },
        ])
        self.manager.save_chunks(chunks)

        result = self.manager.merge_orphan_segments(min_words=10)
        updated = self.manager.load_chunks()

        self.assertEqual(result["changed"], 1)
        self.assertEqual(len(updated), 6)
        self.assertEqual(updated[5]["instruct"], "Bright")

    def test_limits_merge_to_requested_chapter(self):
        chunks = self._chapter_intro_chunks("Chapter 1")
        chunks.extend(self._chapter_intro_chunks("Chapter 2", start_id=5))
        chunks.extend([
            {
                "id": 10,
                "speaker": "Narrator",
                "text": "One two three.",
                "instruct": "",
                "status": "pending",
                "audio_path": None,
                "audio_validation": None,
                "auto_regen_count": 0,
                "chapter": "Chapter 1",
            },
            {
                "id": 11,
                "speaker": "Narrator",
                "text": "Four five six.",
                "instruct": "",
                "status": "pending",
                "audio_path": None,
                "audio_validation": None,
                "auto_regen_count": 0,
                "chapter": "Chapter 2",
            },
            {
                "id": 12,
                "speaker": "Narrator",
                "text": "Seven eight nine ten eleven.",
                "instruct": "",
                "status": "pending",
                "audio_path": None,
                "audio_validation": None,
                "auto_regen_count": 0,
                "chapter": "Chapter 2",
            },
        ])
        self.manager.save_chunks(chunks)

        result = self.manager.merge_orphan_segments(chapter="Chapter 2", min_words=10)
        updated = self.manager.load_chunks()

        self.assertEqual(result["changed"], 1)
        self.assertEqual(len(updated), 12)
        self.assertEqual(updated[10]["chapter"], "Chapter 1")
        self.assertEqual(updated[10]["text"], "One two three.")
        self.assertEqual(updated[11]["text"], "Four five six. Seven eight nine ten eleven.")

    def test_does_not_merge_across_chapter_boundaries_in_whole_project_mode(self):
        chunks = self._chapter_intro_chunks("Chapter 1")
        chunks.extend(self._chapter_intro_chunks("Chapter 2", start_id=5))
        chunks.extend([
            {
                "id": 10,
                "speaker": "Narrator",
                "text": "One two three.",
                "instruct": "",
                "status": "pending",
                "audio_path": None,
                "audio_validation": None,
                "auto_regen_count": 0,
                "chapter": "Chapter 1",
            },
            {
                "id": 11,
                "speaker": "Narrator",
                "text": "Four five six.",
                "instruct": "",
                "status": "pending",
                "audio_path": None,
                "audio_validation": None,
                "auto_regen_count": 0,
                "chapter": "Chapter 2",
            },
        ])
        self.manager.save_chunks(chunks)

        result = self.manager.merge_orphan_segments(min_words=10)
        updated = self.manager.load_chunks()

        self.assertEqual(result["changed"], 0)
        self.assertEqual(len(updated), 12)
        self.assertEqual(updated[10]["chapter"], "Chapter 1")
        self.assertEqual(updated[11]["chapter"], "Chapter 2")

    def test_preserves_exact_chapter_label_when_merging(self):
        chunks = self._chapter_intro_chunks("Chapter 12A")
        chunks.extend([
            {
                "id": 5,
                "speaker": "Narrator",
                "text": "Alpha beta gamma.",
                "instruct": "",
                "status": "done",
                "audio_path": "voicelines/a.wav",
                "audio_validation": {"is_valid": True},
                "auto_regen_count": 0,
                "chapter": "Chapter 12A",
            },
            {
                "id": 6,
                "speaker": "Narrator",
                "text": "Delta epsilon zeta eta theta iota.",
                "instruct": "",
                "status": "pending",
                "audio_path": None,
                "audio_validation": None,
                "auto_regen_count": 0,
                "chapter": "Chapter 12A",
            },
        ])
        self.manager.save_chunks(chunks)

        result = self.manager.merge_orphan_segments(min_words=10)
        updated = self.manager.load_chunks()

        self.assertEqual(result["changed"], 1)
        self.assertEqual(len(updated), 6)
        self.assertEqual(updated[5]["chapter"], "Chapter 12A")
        self.assertIsNone(updated[5]["audio_path"])

    def test_skips_first_five_samples_of_each_chapter(self):
        chunks = []
        for i in range(5):
            chunks.append({
                "id": i,
                "speaker": "Narrator",
                "text": f"Short intro {i}.",
                "instruct": "",
                "status": "pending",
                "audio_path": None,
                "audio_validation": None,
                "auto_regen_count": 0,
                "chapter": "Chapter 1",
            })
        chunks.extend([
            {
                "id": 5,
                "speaker": "Narrator",
                "text": "Tiny tail.",
                "instruct": "",
                "status": "pending",
                "audio_path": None,
                "audio_validation": None,
                "auto_regen_count": 0,
                "chapter": "Chapter 1",
            },
            {
                "id": 6,
                "speaker": "Narrator",
                "text": "This should merge with the tiny tail now.",
                "instruct": "",
                "status": "pending",
                "audio_path": None,
                "audio_validation": None,
                "auto_regen_count": 0,
                "chapter": "Chapter 1",
            },
        ])
        self.manager.save_chunks(chunks)

        result = self.manager.merge_orphan_segments(min_words=10)
        updated = self.manager.load_chunks()

        self.assertEqual(result["changed"], 1)
        self.assertEqual(len(updated), 6)
        for i in range(5):
            self.assertEqual(updated[i]["text"], f"Short intro {i}.")
        self.assertEqual(updated[5]["text"], "Tiny tail. This should merge with the tiny tail now.")


class StableAudioFilenameTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root_dir = self.temp_dir.name
        os.makedirs(os.path.join(self.root_dir, "voicelines"), exist_ok=True)
        os.makedirs(os.path.join(self.root_dir, "app"), exist_ok=True)

        with open(os.path.join(self.root_dir, "annotated_script.json"), "w", encoding="utf-8") as f:
            json.dump({"entries": [], "dictionary": []}, f)

        self.manager = ProjectManager(self.root_dir)

    def tearDown(self):
        self.temp_dir.cleanup()

    def _write_temp_wav(self, name, duration_seconds=1.5):
        path = os.path.join(self.root_dir, name)
        sample_rate = 24000
        samples = np.zeros(int(sample_rate * duration_seconds), dtype=np.float32)
        sf.write(path, samples, sample_rate)
        return path

    def test_finalize_generated_audio_uses_chunk_uid_in_filename(self):
        self.manager.save_chunks([
            {
                "id": 0,
                "uid": "chunk-alpha",
                "speaker": "Narrator",
                "text": "One two three four five six.",
                "instruct": "",
                "status": "generating",
                "audio_path": None,
            }
        ])
        temp_path = self._write_temp_wav("temp_chunk.wav")

        result = self.manager._finalize_generated_audio(
            0,
            "Narrator",
            "One two three four five six.",
            temp_path,
            chunk_uid="chunk-alpha",
        )

        self.assertEqual(result["status"], "done")
        self.assertEqual(result["audio_path"], "voicelines/voiceline_chunk-alpha_narrator.mp3")

    def test_inserted_chunk_generation_does_not_collide_with_shifted_chunk_audio(self):
        self.manager.save_chunks([
            {
                "id": 0,
                "uid": "chunk-a",
                "speaker": "Narrator",
                "text": "First line has enough words for validation to pass cleanly.",
                "instruct": "",
                "status": "done",
                "audio_path": None,
            },
            {
                "id": 1,
                "uid": "chunk-b",
                "speaker": "Narrator",
                "text": "Second line also has enough words for validation to pass cleanly.",
                "instruct": "",
                "status": "done",
                "audio_path": None,
            },
        ])

        original_temp = self._write_temp_wav("temp_original.wav")
        original = self.manager._finalize_generated_audio(
            1,
            "Narrator",
            "Second line also has enough words for validation to pass cleanly.",
            original_temp,
            chunk_uid="chunk-b",
        )

        chunks = self.manager.insert_chunk(0)
        inserted = chunks[1]
        inserted["text"] = "Inserted line also has enough words for validation to pass cleanly."
        self.manager.save_chunks(chunks)

        inserted_temp = self._write_temp_wav("temp_inserted.wav")
        generated = self.manager._finalize_generated_audio(
            1,
            "Narrator",
            "Inserted line also has enough words for validation to pass cleanly.",
            inserted_temp,
            chunk_uid=inserted["uid"],
        )

        self.assertNotEqual(original["audio_path"], generated["audio_path"])
        self.assertIn("chunk-b", original["audio_path"])
        self.assertIn(inserted["uid"], generated["audio_path"])


class InvalidateStaleAudioReferenceTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root_dir = self.temp_dir.name
        os.makedirs(os.path.join(self.root_dir, "voicelines"), exist_ok=True)
        os.makedirs(os.path.join(self.root_dir, "app"), exist_ok=True)

        with open(os.path.join(self.root_dir, "annotated_script.json"), "w", encoding="utf-8") as f:
            json.dump({"entries": [], "dictionary": []}, f)

        self.manager = ProjectManager(self.root_dir)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_invalidates_noncanonical_duplicate_legacy_audio_references(self):
        self.manager.save_chunks([
            {
                "id": 0,
                "uid": "u0",
                "speaker": "Narrator",
                "text": "One",
                "instruct": "",
                "status": "done",
                "audio_path": "voicelines/voiceline_0001_narrator.mp3",
                "audio_validation": {"is_valid": True},
                "auto_regen_count": 0,
            },
            {
                "id": 1,
                "uid": "u1",
                "speaker": "Narrator",
                "text": "Two",
                "instruct": "",
                "status": "done",
                "audio_path": "voicelines/voiceline_0001_narrator.mp3",
                "audio_validation": {"is_valid": True},
                "auto_regen_count": 0,
            },
            {
                "id": 2,
                "uid": "u2",
                "speaker": "Narrator",
                "text": "Three",
                "instruct": "",
                "status": "done",
                "audio_path": "voicelines/voiceline_0003_narrator.mp3",
                "audio_validation": {"is_valid": True},
                "auto_regen_count": 0,
            },
        ])

        result = self.manager.invalidate_stale_audio_references()
        updated = self.manager.load_chunks()

        self.assertEqual(result["invalidated"], 1)
        self.assertEqual(updated[0]["audio_path"], "voicelines/voiceline_0001_narrator.mp3")
        self.assertIsNone(updated[1]["audio_path"])
        self.assertEqual(updated[1]["status"], "pending")
        self.assertEqual(updated[2]["audio_path"], "voicelines/voiceline_0003_narrator.mp3")

    def test_invalidates_duplicate_uid_based_audio_for_nonmatching_chunks(self):
        self.manager.save_chunks([
            {
                "id": 0,
                "uid": "chunk-alpha",
                "speaker": "Narrator",
                "text": "One",
                "instruct": "",
                "status": "done",
                "audio_path": "voicelines/voiceline_chunk-alpha_narrator.mp3",
                "audio_validation": {"is_valid": True},
                "auto_regen_count": 0,
            },
            {
                "id": 1,
                "uid": "chunk-beta",
                "speaker": "Narrator",
                "text": "Two",
                "instruct": "",
                "status": "done",
                "audio_path": "voicelines/voiceline_chunk-alpha_narrator.mp3",
                "audio_validation": {"is_valid": True},
                "auto_regen_count": 0,
            },
        ])

        result = self.manager.invalidate_stale_audio_references()
        updated = self.manager.load_chunks()

        self.assertEqual(result["invalidated"], 1)
        self.assertEqual(updated[0]["audio_path"], "voicelines/voiceline_chunk-alpha_narrator.mp3")
        self.assertIsNone(updated[1]["audio_path"])

    def test_keeps_unique_legacy_audio_references_even_if_index_has_shifted(self):
        self.manager.save_chunks([
            {
                "id": 0,
                "uid": "u0",
                "speaker": "Narrator",
                "text": "Moved but still valid",
                "instruct": "",
                "status": "done",
                "audio_path": "voicelines/voiceline_0005_narrator.mp3",
                "audio_validation": {"is_valid": True},
                "auto_regen_count": 0,
            }
        ])

        result = self.manager.invalidate_stale_audio_references()
        updated = self.manager.load_chunks()

        self.assertEqual(result["invalidated"], 0)
        self.assertEqual(updated[0]["audio_path"], "voicelines/voiceline_0005_narrator.mp3")


class RepairLostAudioLinksTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root_dir = self.temp_dir.name
        os.makedirs(os.path.join(self.root_dir, "voicelines"), exist_ok=True)
        os.makedirs(os.path.join(self.root_dir, "app"), exist_ok=True)

        with open(os.path.join(self.root_dir, "annotated_script.json"), "w", encoding="utf-8") as f:
            json.dump({"entries": [], "dictionary": []}, f)

        self.manager = ProjectManager(self.root_dir)

    def tearDown(self):
        self.temp_dir.cleanup()

    def _write_wav(self, relative_path, duration_seconds):
        full_path = os.path.join(self.root_dir, relative_path)
        sample_rate = 24000
        samples = np.zeros(int(sample_rate * duration_seconds), dtype=np.float32)
        sf.write(full_path, samples, sample_rate)
        return full_path

    def test_repairs_uid_based_audio_links(self):
        chunk_uid = "1234567890abcdef1234567890abcdef"
        self._write_wav(f"voicelines/voiceline_{chunk_uid}_narrator.wav", 3.0)
        self.manager.save_chunks([
            {
                "id": 0,
                "uid": chunk_uid,
                "speaker": "Narrator",
                "text": "One two three four five six.",
                "instruct": "",
                "status": "pending",
                "audio_path": None,
                "audio_validation": None,
                "auto_regen_count": 0,
            }
        ])

        result = self.manager.repair_lost_audio_links()
        repaired = self.manager.load_chunks()

        self.assertEqual(result["relinked"], 1)
        self.assertEqual(repaired[0]["status"], "done")
        self.assertEqual(repaired[0]["audio_path"], f"voicelines/voiceline_{chunk_uid}_narrator.wav")
        self.assertTrue(repaired[0]["audio_validation"]["is_valid"])

    def test_repairs_legacy_index_audio_links(self):
        self._write_wav("voicelines/voiceline_0002_narrator.wav", 3.0)
        self.manager.save_chunks([
            {
                "id": 0,
                "uid": "u0",
                "speaker": "Narrator",
                "text": "Short intro line.",
                "instruct": "",
                "status": "pending",
                "audio_path": None,
                "audio_validation": None,
                "auto_regen_count": 0,
            },
            {
                "id": 1,
                "uid": "u1",
                "speaker": "Narrator",
                "text": "One two three four five six.",
                "instruct": "",
                "status": "pending",
                "audio_path": None,
                "audio_validation": None,
                "auto_regen_count": 0,
            },
        ])

        result = self.manager.repair_lost_audio_links()
        repaired = self.manager.load_chunks()

        self.assertEqual(result["relinked"], 1)
        self.assertIsNone(repaired[0]["audio_path"])
        self.assertEqual(repaired[1]["status"], "done")
        self.assertEqual(repaired[1]["audio_path"], "voicelines/voiceline_u1_narrator.wav")
        self.assertFalse(os.path.exists(os.path.join(self.root_dir, "voicelines/voiceline_0002_narrator.wav")))
        self.assertTrue(os.path.exists(os.path.join(self.root_dir, "voicelines/voiceline_u1_narrator.wav")))

    def test_repairs_lost_audio_links_via_asr_nearby_match(self):
        self._write_wav("voicelines/voiceline_0001_narrator.wav", 0.4)
        self.manager.save_chunks([
            {
                "id": 0,
                "uid": "u0",
                "speaker": "Narrator",
                "text": "This line should not be matched by transcription.",
                "instruct": "",
                "status": "pending",
                "audio_path": None,
                "audio_validation": None,
                "auto_regen_count": 0,
                "chapter": "Chapter One",
            },
            {
                "id": 1,
                "uid": "u1",
                "speaker": "Narrator",
                "text": "The recovered line belongs here and should be restored.",
                "instruct": "",
                "status": "pending",
                "audio_path": None,
                "audio_validation": None,
                "auto_regen_count": 0,
                "chapter": "Chapter One",
            },
        ])

        original_transcribe = self.manager.transcribe_audio_path
        original_validate = self.manager._validate_audio_path_for_chunk
        try:
            self.manager.transcribe_audio_path = lambda relative_path: {
                "text": "The recovered line belongs here and should be restored.",
                "normalized_text": self.manager._normalize_asr_text("The recovered line belongs here and should be restored."),
            }
            self.manager._validate_audio_path_for_chunk = lambda chunk, path, dictionary_entries: {
                "is_valid": False,
                "error": "Duration sanity failed in test fixture.",
            }

            result = self.manager.repair_lost_audio_links(use_asr=True)
            repaired = self.manager.load_chunks()

            self.assertEqual(result["relinked"], 0)
            self.assertEqual(result["asr_relinked"], 1)
            self.assertIsNone(repaired[0]["audio_path"])
            self.assertEqual(repaired[1]["audio_path"], "voicelines/voiceline_u1_narrator.wav")
            self.assertTrue(repaired[1]["audio_validation"]["matched_via_asr"])
        finally:
            self.manager.transcribe_audio_path = original_transcribe
            self.manager._validate_audio_path_for_chunk = original_validate

    def test_asr_similarity_score_penalizes_partial_subset_matches(self):
        partial = "She knew the stars"
        full = "She knew the stars in their multitudes, none could stray from their course and Luna not know it."
        exact = "She knew the stars in their multitudes, none could stray from their course and Luna not know it."

        partial_score = self.manager._asr_similarity_score(partial, full)
        exact_score = self.manager._asr_similarity_score(exact, full)

        self.assertLess(partial_score, 0.72)
        self.assertGreater(exact_score, 0.95)
        self.assertLess(partial_score, exact_score)

    def test_repair_lost_audio_links_does_not_auto_match_partial_asr_transcript(self):
        self._write_wav("voicelines/voiceline_0001_narrator.wav", 0.4)
        self.manager.save_chunks([
            {
                "id": 0,
                "uid": "u0",
                "speaker": "Narrator",
                "text": "She knew the stars in their multitudes, none could stray from their course and Luna not know it.",
                "instruct": "",
                "status": "pending",
                "audio_path": None,
                "audio_validation": None,
                "auto_regen_count": 0,
                "chapter": "Prologue",
            },
            {
                "id": 1,
                "uid": "u1",
                "speaker": "Narrator",
                "text": "In the stillness of her night, Luna watched the horizon and listened for change.",
                "instruct": "",
                "status": "pending",
                "audio_path": None,
                "audio_validation": None,
                "auto_regen_count": 0,
                "chapter": "Prologue",
            },
        ])

        original_transcribe = self.manager.transcribe_audio_path
        original_validate = self.manager._validate_audio_path_for_chunk
        try:
            self.manager.transcribe_audio_path = lambda relative_path: {
                "text": "She knew the stars",
                "normalized_text": self.manager._normalize_asr_text("She knew the stars"),
            }
            self.manager._validate_audio_path_for_chunk = lambda chunk, path, dictionary_entries: {
                "is_valid": False,
                "error": "Duration sanity failed in test fixture.",
            }

            result = self.manager.repair_lost_audio_links(use_asr=True)
            repaired = self.manager.load_chunks()

            self.assertEqual(result["asr_relinked"], 0)
            self.assertIsNone(repaired[0]["audio_path"])
            self.assertIsNone(repaired[1]["audio_path"])
        finally:
            self.manager.transcribe_audio_path = original_transcribe
            self.manager._validate_audio_path_for_chunk = original_validate

    def test_repair_lost_audio_links_prefers_exact_match_anywhere_in_same_chapter(self):
        self._write_wav("voicelines/voiceline_0001_narrator.wav", 0.4)
        self.manager.save_chunks([
            {
                "id": 0,
                "uid": "u0",
                "speaker": "Narrator",
                "text": "This is the legacy anchor line.",
                "instruct": "",
                "status": "pending",
                "audio_path": None,
                "audio_validation": None,
                "auto_regen_count": 0,
                "chapter": "Prologue",
            },
            {
                "id": 1,
                "uid": "u1",
                "speaker": "Narrator",
                "text": "Another nearby line that should not win.",
                "instruct": "",
                "status": "pending",
                "audio_path": None,
                "audio_validation": None,
                "auto_regen_count": 0,
                "chapter": "Prologue",
            },
            {
                "id": 2,
                "uid": "u2",
                "speaker": "Narrator",
                "text": "The exact chapter match lives farther away and should be recovered.",
                "instruct": "",
                "status": "pending",
                "audio_path": None,
                "audio_validation": None,
                "auto_regen_count": 0,
                "chapter": "Prologue",
            },
        ])

        original_transcribe = self.manager.transcribe_audio_path
        original_validate = self.manager._validate_audio_path_for_chunk
        original_settings = self.manager._load_asr_settings
        try:
            self.manager.transcribe_audio_path = lambda relative_path: {
                "text": "The exact chapter match lives farther away and should be recovered.",
                "normalized_text": self.manager._normalize_asr_text("The exact chapter match lives farther away and should be recovered."),
            }
            self.manager._validate_audio_path_for_chunk = lambda chunk, path, dictionary_entries: {
                "is_valid": False,
                "error": "Duration sanity failed in test fixture.",
            }
            self.manager._load_asr_settings = lambda: {
                "enabled": True,
                "model": "small.en",
                "language": "en",
                "device": "auto",
                "compute_type": "auto",
                "beam_size": 1,
                "repair_window": 1,
                "confidence_threshold": 0.72,
                "confidence_margin": 0.08,
            }

            result = self.manager.repair_lost_audio_links(use_asr=True)
            repaired = self.manager.load_chunks()

            self.assertEqual(result["asr_relinked"], 1)
            self.assertIsNone(repaired[0]["audio_path"])
            self.assertIsNone(repaired[1]["audio_path"])
            self.assertEqual(repaired[2]["audio_path"], "voicelines/voiceline_u2_narrator.wav")
            self.assertTrue(repaired[2]["audio_validation"]["exact_chapter_match"])
        finally:
            self.manager.transcribe_audio_path = original_transcribe
            self.manager._validate_audio_path_for_chunk = original_validate
            self.manager._load_asr_settings = original_settings

    def test_repair_lost_audio_links_requires_exact_speaker_match_for_asr(self):
        self._write_wav("voicelines/voiceline_0001_narrator.wav", 0.4)
        self.manager.save_chunks([
            {
                "id": 0,
                "uid": "u0",
                "speaker": "Voice",
                "text": "This exact text should not be assigned to the wrong speaker.",
                "instruct": "",
                "status": "pending",
                "audio_path": None,
                "audio_validation": None,
                "auto_regen_count": 0,
                "chapter": "Prologue",
            },
        ])

        original_transcribe = self.manager.transcribe_audio_path
        original_validate = self.manager._validate_audio_path_for_chunk
        try:
            self.manager.transcribe_audio_path = lambda relative_path: {
                "text": "This exact text should not be assigned to the wrong speaker.",
                "normalized_text": self.manager._normalize_asr_text("This exact text should not be assigned to the wrong speaker."),
            }
            self.manager._validate_audio_path_for_chunk = lambda chunk, path, dictionary_entries: {
                "is_valid": False,
                "error": "Duration sanity failed in test fixture.",
            }

            result = self.manager.repair_lost_audio_links(use_asr=True)
            repaired = self.manager.load_chunks()

            self.assertEqual(result["asr_relinked"], 0)
            self.assertIsNone(repaired[0]["audio_path"])
        finally:
            self.manager.transcribe_audio_path = original_transcribe
            self.manager._validate_audio_path_for_chunk = original_validate


if __name__ == "__main__":
    unittest.main()
