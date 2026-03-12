import json
import os
import tempfile
import unittest

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


if __name__ == "__main__":
    unittest.main()
