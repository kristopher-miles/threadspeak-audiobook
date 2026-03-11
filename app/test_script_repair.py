import unittest

from script_repair import (
    _build_narrator_replacement_entries,
    _extract_entries_word_span,
    _group_entries_by_chapter,
    _span_is_inside_dialogue,
    _splice_replacement,
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


if __name__ == "__main__":
    unittest.main()
