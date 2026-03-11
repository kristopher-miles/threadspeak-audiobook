import unittest

from script_sanity import run_script_sanity_check


class ScriptSanityCheckTests(unittest.TestCase):
    def test_ignores_case_punctuation_and_spacing(self):
        source = {
            "type": "text",
            "title": "Book",
            "chapters": [{
                "title": "Chapter One",
                "text": "Hello, world! This is a test.",
            }],
        }
        script = {
            "entries": [{
                "chapter": "Chapter One",
                "speaker": "NARRATOR",
                "text": "hello world this   is a TEST",
                "instruct": "ignored",
            }],
            "dictionary": [],
        }

        result = run_script_sanity_check(source, script, chunk_size=100)

        self.assertEqual(result["missing_words"], 0)
        self.assertEqual(result["inserted_words"], 0)
        self.assertEqual(result["invalid_section_count"], 0)
        self.assertEqual(result["invalid_chunk_count"], 0)

    def test_localizes_insertions_and_deletions(self):
        source = {
            "type": "text",
            "title": "Book",
            "chapters": [{
                "title": "Chapter One",
                "text": "One two three four five six seven eight.",
            }],
        }
        script = {
            "entries": [{
                "chapter": "Chapter One",
                "speaker": "NARRATOR",
                "text": "One two bonus extra three four five seven eight.",
                "instruct": "",
            }],
            "dictionary": [],
        }

        result = run_script_sanity_check(source, script, chunk_size=100)

        self.assertEqual(result["missing_words"], 1)
        self.assertEqual(result["inserted_words"], 2)
        self.assertEqual(result["invalid_section_count"], 2)
        self.assertEqual(result["invalid_chunk_count"], 1)

    def test_counts_inserted_script_only_chapter(self):
        source = {
            "type": "epub",
            "title": "Book",
            "chapters": [{
                "title": "Chapter One",
                "text": "Alpha beta gamma.",
            }],
        }
        script = {
            "entries": [
                {
                    "chapter": "Chapter One",
                    "speaker": "NARRATOR",
                    "text": "Alpha beta gamma.",
                    "instruct": "",
                },
                {
                    "chapter": "Table of Contents",
                    "speaker": "NARRATOR",
                    "text": "Chapter One Chapter Two",
                    "instruct": "",
                },
            ],
            "dictionary": [],
        }

        result = run_script_sanity_check(source, script, chunk_size=100)

        self.assertEqual(result["missing_words"], 0)
        self.assertEqual(result["inserted_words"], 4)
        self.assertEqual(result["invalid_section_count"], 1)
        self.assertEqual(result["invalid_chunk_count"], 1)
        self.assertEqual(result["chapters"][1]["kind"], "inserted_chapter")

    def test_ignores_decorative_dividers_without_english_letters(self):
        source = {
            "type": "text",
            "title": "Book",
            "chapters": [{
                "title": "Chapter One",
                "text": "Alpha beta gamma.",
            }],
        }
        script = {
            "entries": [
                {"chapter": "Chapter One", "speaker": "NARRATOR", "text": "*** *** ***", "instruct": ""},
                {"chapter": "Chapter One", "speaker": "NARRATOR", "text": "Alpha beta gamma.", "instruct": ""},
            ],
            "dictionary": [],
        }

        result = run_script_sanity_check(source, script, chunk_size=100)

        self.assertEqual(result["missing_words"], 0)
        self.assertEqual(result["inserted_words"], 0)

    def test_replacement_chunks_do_not_exceed_chunk_size(self):
        source = {
            "type": "text",
            "title": "Book",
            "chapters": [{
                "title": "Chapter One",
                "text": "One two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen.",
            }],
        }
        script = {
            "entries": [{
                "chapter": "Chapter One",
                "speaker": "NARRATOR",
                "text": "One two X four five six seven eight nine ten eleven Y thirteen fourteen fifteen sixteen.",
                "instruct": "",
            }],
            "dictionary": [],
        }

        result = run_script_sanity_check(source, script, chunk_size=20)

        for chunk in result["replacement_chunks"]:
            self.assertLessEqual(chunk["source_char_end"] - chunk["source_char_start"], 20)

    def test_prunes_short_missing_attribution_phrases(self):
        source = {
            "type": "text",
            "title": "Book",
            "chapters": [{
                "title": "Chapter One",
                "text": 'Hello there, he said quietly, before leaving.',
            }],
        }
        script = {
            "entries": [{
                "chapter": "Chapter One",
                "speaker": "NARRATOR",
                "text": "Hello there before leaving",
                "instruct": "",
            }],
            "dictionary": [],
        }
        known_phrase_decisions = {
            "he said quietly": {
                "phrase": "he said quietly",
                "decision": "accepted",
                "reply": "TRUE",
                "checked_at": 0,
            },
        }

        result = run_script_sanity_check(
            source,
            script,
            chunk_size=100,
            known_phrase_decisions=known_phrase_decisions,
        )

        self.assertEqual(result["missing_words"], 0)
        self.assertEqual(result["inserted_words"], 0)
        self.assertEqual(result["attribution_pruned_sections"], 1)
        self.assertEqual(result["attribution_pruned_words"], 3)

    def test_does_not_prune_full_sentence_missing_text(self):
        source = {
            "type": "text",
            "title": "Book",
            "chapters": [{
                "title": "Chapter One",
                "text": "Hello there. She watched the rain. Then she left.",
            }],
        }
        script = {
            "entries": [{
                "chapter": "Chapter One",
                "speaker": "NARRATOR",
                "text": "Hello there. Then she left.",
                "instruct": "",
            }],
            "dictionary": [],
        }
        known_phrase_decisions = {
            "she watched the rain": {
                "phrase": "She watched the rain.",
                "decision": "accepted",
                "reply": "TRUE",
                "checked_at": 0,
            },
        }

        result = run_script_sanity_check(
            source,
            script,
            chunk_size=100,
            known_phrase_decisions=known_phrase_decisions,
        )

        self.assertGreater(result["missing_words"], 0)
        self.assertEqual(result["attribution_pruned_sections"], 0)


if __name__ == "__main__":
    unittest.main()
