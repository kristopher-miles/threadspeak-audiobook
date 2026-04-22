import sys
import unittest
from types import SimpleNamespace

sys.modules.setdefault("openai", SimpleNamespace(OpenAI=object))

from scripts.assign_dialogue import apply_narrated_dialogue_assignments
from scripts.extract_temperament import split_paragraphs_for_temperament


class StructuralSilenceStageFilterTests(unittest.TestCase):
    def test_narrated_assignment_skips_structural_silence_rows(self):
        paragraphs_doc = {
            "paragraphs": [
                {
                    "id": "p_0001",
                    "has_dialogue": True,
                    "text": '"Hello there."',
                    "speakers": [],
                    "quote_errors": [],
                    "dialogue_error": False,
                },
                {
                    "id": "p_0002",
                    "has_dialogue": False,
                    "text": "***",
                    "is_structural_silence": True,
                    "speakers": ["SHOULD", "NOT", "CHANGE"],
                    "quote_errors": [True],
                    "dialogue_error": True,
                },
            ],
            "dialogue_assignment_complete": False,
            "dialogue_errors": ["p_0002"],
        }

        para_count, quote_count = apply_narrated_dialogue_assignments(paragraphs_doc)

        self.assertEqual(para_count, 1)
        self.assertEqual(quote_count, 1)
        self.assertEqual(paragraphs_doc["paragraphs"][0]["speakers"], ["NARRATOR"])
        self.assertEqual(paragraphs_doc["paragraphs"][1]["speakers"], ["SHOULD", "NOT", "CHANGE"])
        self.assertEqual(paragraphs_doc["dialogue_errors"], [])

    def test_temperament_split_excludes_structural_silence_rows(self):
        paragraphs = [
            {
                "id": "p_0001",
                "text": "Narration only.",
                "has_dialogue": False,
            },
            {
                "id": "p_0002",
                "text": "***",
                "has_dialogue": False,
                "is_structural_silence": True,
            },
            {
                "id": "p_0003",
                "text": '"Hello." she said.',
                "has_dialogue": True,
            },
        ]

        narration, dialogue = split_paragraphs_for_temperament(paragraphs)

        self.assertEqual([item[1]["id"] for item in narration], ["p_0001"])
        self.assertEqual([item[1]["id"] for item in dialogue], ["p_0003"])


if __name__ == "__main__":
    unittest.main()
