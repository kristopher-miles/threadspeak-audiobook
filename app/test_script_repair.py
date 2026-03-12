import json
import os
import tempfile
import unittest
from unittest.mock import patch

from script_repair import (
    _build_validation_failure_replacement_entries,
    _build_narrator_replacement_entries,
    _extract_entries_word_span,
    _group_entries_by_chapter,
    _merge_narrator_replacement_into_adjacent_entry,
    _remove_entry_indices,
    _removal_strictly_improves_sanity,
    _should_shortcut_short_narration_patch,
    _span_is_inside_dialogue,
    _splice_replacement,
    _whole_entry_block_for_script_span,
    RepairSupersededError,
    repair_invalid_chunks,
)


class ScriptRepairHelpersTests(unittest.TestCase):
    def test_span_is_inside_dialogue_detects_quoted_text(self):
        text = 'She frowned. "Hello there," he said. Then she left.'
        start = text.index("Hello")
        end = start + len("Hello there")
        self.assertTrue(_span_is_inside_dialogue(text, start, end))

    def test_span_is_inside_dialogue_rejects_narration(self):
        text = 'She frowned. "Hello there," he said. Then she left.'
        start = text.index("Then")
        end = start + len("Then she left")
        self.assertFalse(_span_is_inside_dialogue(text, start, end))

    def test_build_narrator_replacement_entries_uses_narrator_defaults(self):
        entries = _build_narrator_replacement_entries("Chapter One", "She swallowed.")
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["speaker"], "NARRATOR")
        self.assertEqual(entries[0]["text"], "She swallowed.")
        self.assertEqual(entries[0]["instruct"], "Neutral, even narration.")

    def test_validation_failure_prefers_narrator_only_as_fallback(self):
        entries, kind = _build_validation_failure_replacement_entries(
            "Chapter One",
            "She swallowed.",
            target_is_inside_dialogue=False,
            prefer_narrator=True,
        )
        self.assertEqual(kind, "narrator")
        self.assertEqual(entries[0]["speaker"], "NARRATOR")

    def test_validation_failure_uses_literal_for_dialogue(self):
        entries, kind = _build_validation_failure_replacement_entries(
            "Chapter One",
            "Hello there.",
            target_is_inside_dialogue=True,
            prefer_narrator=True,
        )
        self.assertEqual(kind, "literal")
        self.assertEqual(entries[0]["speaker"], "NARRATOR")
        self.assertEqual(entries[0]["text"], "Hello there.")

    def test_extract_entries_word_span_trims_boundary_entries(self):
        entries = [
            {"chapter": "Chapter One", "speaker": "NARRATOR", "text": "One two three", "instruct": ""},
            {"chapter": "Chapter One", "speaker": "NARRATOR", "text": "Four five six", "instruct": ""},
        ]
        group = _group_entries_by_chapter(entries)[0]

        sliced = _extract_entries_word_span(group["entries"], 1, 5)

        self.assertEqual([entry["text"] for entry in sliced], ["two three", "Four five"])

    def test_splice_replacement_replaces_only_target_word_span(self):
        entries = [
            {"chapter": "Chapter One", "speaker": "NARRATOR", "text": "One two three", "instruct": ""},
            {"chapter": "Chapter One", "speaker": "NARRATOR", "text": "bonus extra", "instruct": ""},
            {"chapter": "Chapter One", "speaker": "NARRATOR", "text": "four five six", "instruct": ""},
        ]
        group = _group_entries_by_chapter(entries)[0]
        replacement_chunk = {
            "script_word_start": 3,
            "script_word_end": 5,
        }
        replacement_entries = [
            {"chapter": "Chapter One", "speaker": "NARRATOR", "text": "three and a half", "instruct": ""},
        ]

        updated = _splice_replacement(entries, group, replacement_chunk, replacement_entries)

        self.assertEqual(
            [entry["text"] for entry in updated],
            ["One two three", "three and a half", "four five six"],
        )

    def test_merge_narrator_replacement_prefers_previous_narrator_entry(self):
        entries = [
            {"chapter": "Chapter One", "speaker": "NARRATOR", "text": "One two", "instruct": "warm"},
            {"chapter": "Chapter One", "speaker": "ALICE", "text": "Hello", "instruct": ""},
            {"chapter": "Chapter One", "speaker": "NARRATOR", "text": "Four five", "instruct": "cool"},
        ]
        group = _group_entries_by_chapter(entries)[0]

        updated = _merge_narrator_replacement_into_adjacent_entry(
            entries,
            group,
            {"script_word_start": 2, "script_word_end": 2},
            [{"chapter": "Chapter One", "speaker": "NARRATOR", "text": "She sighed", "instruct": "Neutral, even narration."}],
        )

        self.assertEqual(
            [(entry["speaker"], entry["text"], entry["instruct"]) for entry in updated],
            [
                ("NARRATOR", "One two She sighed", "warm"),
                ("ALICE", "Hello", ""),
                ("NARRATOR", "Four five", "cool"),
            ],
        )

    def test_merge_narrator_replacement_falls_forward_when_previous_is_not_narrator(self):
        entries = [
            {"chapter": "Chapter One", "speaker": "ALICE", "text": "Hello", "instruct": ""},
            {"chapter": "Chapter One", "speaker": "NARRATOR", "text": "Four five", "instruct": "cool"},
        ]
        group = _group_entries_by_chapter(entries)[0]

        updated = _merge_narrator_replacement_into_adjacent_entry(
            entries,
            group,
            {"script_word_start": 1, "script_word_end": 1},
            [{"chapter": "Chapter One", "speaker": "NARRATOR", "text": "She sighed", "instruct": "Neutral, even narration."}],
        )

        self.assertEqual(
            [(entry["speaker"], entry["text"], entry["instruct"]) for entry in updated],
            [
                ("ALICE", "Hello", ""),
                ("NARRATOR", "She sighed Four five", "cool"),
            ],
        )

    def test_whole_entry_block_for_script_span_requires_exact_entry_alignment(self):
        entries = [
            {"chapter": "Chapter One", "speaker": "NARRATOR", "text": "One two", "instruct": ""},
            {"chapter": "Chapter One", "speaker": "NARRATOR", "text": "bonus extra", "instruct": ""},
            {"chapter": "Chapter One", "speaker": "NARRATOR", "text": "Four five six", "instruct": ""},
        ]
        group = _group_entries_by_chapter(entries)[0]

        exact = _whole_entry_block_for_script_span(group["entries"], 2, 4)
        partial = _whole_entry_block_for_script_span(group["entries"], 1, 4)

        self.assertEqual(exact, [1])
        self.assertIsNone(partial)

    def test_remove_entry_indices_drops_only_selected_entries(self):
        entries = [
            {"chapter": "Chapter One", "speaker": "NARRATOR", "text": "One two", "instruct": ""},
            {"chapter": "Chapter One", "speaker": "NARRATOR", "text": "bonus extra", "instruct": ""},
            {"chapter": "Chapter One", "speaker": "NARRATOR", "text": "Four five six", "instruct": ""},
        ]

        updated = _remove_entry_indices(entries, [1])

        self.assertEqual([entry["text"] for entry in updated], ["One two", "Four five six"])

    def test_removal_strictly_improves_sanity_requires_inserted_reduction_without_regression(self):
        current = {"inserted_words": 2, "missing_words": 1, "invalid_chunk_count": 1}

        self.assertTrue(
            _removal_strictly_improves_sanity(
                current,
                {"inserted_words": 0, "missing_words": 1, "invalid_chunk_count": 1},
            )
        )
        self.assertFalse(
            _removal_strictly_improves_sanity(
                current,
                {"inserted_words": 1, "missing_words": 2, "invalid_chunk_count": 1},
            )
        )

    def test_short_narration_shortcut_requires_non_dialogue_and_three_words_or_less(self):
        self.assertTrue(_should_shortcut_short_narration_patch(3, False))
        self.assertFalse(_should_shortcut_short_narration_patch(4, False))
        self.assertFalse(_should_shortcut_short_narration_patch(2, True))


class ScriptRepairFlowTests(unittest.TestCase):
    def test_repair_drops_self_contained_inserted_entries_without_llm(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            source_path = os.path.join(tmpdir, "source.txt")
            with open(source_path, "w", encoding="utf-8") as f:
                f.write("One two three\n\nFour five six")

            with open(os.path.join(tmpdir, "state.json"), "w", encoding="utf-8") as f:
                json.dump({"input_file_path": source_path}, f)

            os.makedirs(os.path.join(tmpdir, "app"), exist_ok=True)
            with open(os.path.join(tmpdir, "app", "config.json"), "w", encoding="utf-8") as f:
                json.dump({
                    "llm": {
                        "base_url": "http://127.0.0.1:1234/v1",
                        "api_key": "local",
                        "model_name": "test-model",
                        "timeout": 5,
                    },
                    "generation": {
                        "chunk_size": 100,
                        "max_tokens": 256,
                    },
                    "prompts": {},
                }, f)

            script_path = os.path.join(tmpdir, "annotated_script.json")
            with open(script_path, "w", encoding="utf-8") as f:
                json.dump({
                    "entries": [
                        {"chapter": "Chapter One", "speaker": "NARRATOR", "text": "One two", "instruct": ""},
                        {"chapter": "Chapter One", "speaker": "NARRATOR", "text": "bonus extra", "instruct": ""},
                        {"chapter": "Chapter One", "speaker": "NARRATOR", "text": "Four five six", "instruct": ""},
                    ],
                    "dictionary": [],
                    "sanity_cache": {"phrase_decisions": {}},
                }, f)

            continue_calls = {"count": 0}

            def should_continue():
                continue_calls["count"] += 1
                return continue_calls["count"] <= 2

            with patch("script_repair.process_chunk", side_effect=AssertionError("LLM should not be called")):
                with self.assertRaises(RepairSupersededError):
                    repair_invalid_chunks(tmpdir, lambda *_args, **_kwargs: None, should_continue=should_continue)

            with open(script_path, "r", encoding="utf-8") as f:
                saved = json.load(f)

            self.assertEqual(
                [entry["text"] for entry in saved["entries"]],
                ["One two", "Four five six"],
            )

    def test_repair_short_non_dialogue_missing_text_as_narration_without_llm(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            source_path = os.path.join(tmpdir, "source.txt")
            with open(source_path, "w", encoding="utf-8") as f:
                f.write("One two three. She sighed. Four five six.")

            with open(os.path.join(tmpdir, "state.json"), "w", encoding="utf-8") as f:
                json.dump({"input_file_path": source_path}, f)

            os.makedirs(os.path.join(tmpdir, "app"), exist_ok=True)
            with open(os.path.join(tmpdir, "app", "config.json"), "w", encoding="utf-8") as f:
                json.dump({
                    "llm": {
                        "base_url": "http://127.0.0.1:1234/v1",
                        "api_key": "local",
                        "model_name": "test-model",
                        "timeout": 5,
                    },
                    "generation": {
                        "chunk_size": 100,
                        "max_tokens": 256,
                    },
                    "prompts": {},
                }, f)

            script_path = os.path.join(tmpdir, "annotated_script.json")
            with open(script_path, "w", encoding="utf-8") as f:
                json.dump({
                    "entries": [
                        {"chapter": "source", "speaker": "NARRATOR", "text": "One two three.", "instruct": ""},
                        {"chapter": "source", "speaker": "NARRATOR", "text": "Four five six.", "instruct": ""},
                    ],
                    "dictionary": [],
                    "sanity_cache": {"phrase_decisions": {}},
                }, f)

            continue_calls = {"count": 0}

            def should_continue():
                continue_calls["count"] += 1
                return continue_calls["count"] <= 2

            with patch("script_repair.process_chunk", side_effect=AssertionError("LLM should not be called")):
                with self.assertRaises(RepairSupersededError):
                    repair_invalid_chunks(tmpdir, lambda *_args, **_kwargs: None, should_continue=should_continue)

            with open(script_path, "r", encoding="utf-8") as f:
                saved = json.load(f)

            self.assertEqual(
                [entry["text"] for entry in saved["entries"]],
                ["One two three She sighed", "Four five six"],
            )
            self.assertEqual(saved["entries"][0]["speaker"], "NARRATOR")
            self.assertEqual(saved["entries"][0]["instruct"], "")


if __name__ == "__main__":
    unittest.main()
