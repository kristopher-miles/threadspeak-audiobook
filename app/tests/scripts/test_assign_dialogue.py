import sys
import unittest
import inspect
from types import SimpleNamespace

sys.modules.setdefault("openai", SimpleNamespace(OpenAI=object))

import scripts.assign_dialogue as assign_dialogue_module
from scripts.assign_dialogue import apply_narrated_dialogue_assignments


class NarratedDialogueAssignmentTests(unittest.TestCase):
    def test_no_llm_telemetry_log_lines_in_assign_dialogue_script(self):
        self.assertNotIn("LLM telemetry:", inspect.getsource(assign_dialogue_module))

    def test_assigns_narrator_to_all_quotes_and_clears_errors(self):
        paragraphs_doc = {
            "paragraphs": [
                {
                    "id": "p_0001",
                    "has_dialogue": True,
                    "text": '"Hello there," she said. "Goodbye now."',
                    "speakers": ["Alice", "Bob"],
                    "quote_errors": [True, True],
                    "dialogue_error": True,
                }
            ],
            "dialogue_assignment_complete": False,
            "dialogue_errors": ["p_0001"],
        }

        para_count, quote_count = apply_narrated_dialogue_assignments(paragraphs_doc)

        self.assertEqual(para_count, 1)
        self.assertEqual(quote_count, 2)
        self.assertEqual(paragraphs_doc["paragraphs"][0]["speakers"], ["NARRATOR", "NARRATOR"])
        self.assertEqual(paragraphs_doc["paragraphs"][0]["quote_errors"], [False, False])
        self.assertFalse(paragraphs_doc["paragraphs"][0]["dialogue_error"])
        self.assertTrue(paragraphs_doc["dialogue_assignment_complete"])
        self.assertEqual(paragraphs_doc["dialogue_errors"], [])

    def test_dialogue_paragraph_without_quotes_does_not_error(self):
        paragraphs_doc = {
            "paragraphs": [
                {
                    "id": "p_0002",
                    "has_dialogue": True,
                    "text": "No quoted dialogue here.",
                    "speakers": ["Unknown"],
                    "quote_errors": [True],
                    "dialogue_error": True,
                }
            ],
            "dialogue_assignment_complete": False,
            "dialogue_errors": ["p_0002"],
        }

        para_count, quote_count = apply_narrated_dialogue_assignments(paragraphs_doc)

        self.assertEqual(para_count, 1)
        self.assertEqual(quote_count, 0)
        self.assertEqual(paragraphs_doc["paragraphs"][0]["speakers"], [])
        self.assertEqual(paragraphs_doc["paragraphs"][0]["quote_errors"], [])
        self.assertFalse(paragraphs_doc["paragraphs"][0]["dialogue_error"])
        self.assertTrue(paragraphs_doc["dialogue_assignment_complete"])
        self.assertEqual(paragraphs_doc["dialogue_errors"], [])


if __name__ == "__main__":
    unittest.main()
