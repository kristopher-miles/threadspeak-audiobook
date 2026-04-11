import unittest
import sys
from types import SimpleNamespace

sys.modules.setdefault("openai", SimpleNamespace(OpenAI=object))

from extract_temperament import build_temperament_context


class BuildTemperamentContextTests(unittest.TestCase):
    def test_uses_only_target_paragraph_when_word_threshold_already_met(self):
        paragraphs = [
            {"text": "Earlier paragraph should not be included."},
            {"text": " ".join(f"word{i}" for i in range(160))},
            {"text": "Later paragraph should never be included."},
        ]

        context = build_temperament_context(paragraphs, 1, budget=10_000, minimum_words=150)

        self.assertEqual(context, paragraphs[1]["text"])

    def test_prepends_previous_paragraphs_until_minimum_words_reached(self):
        paragraphs = [
            {"text": "one two three four five six seven eight nine ten"},
            {"text": "eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty"},
            {"text": "twentyone twentytwo twentythree twentyfour twentyfive"},
        ]

        context = build_temperament_context(paragraphs, 2, budget=10_000, minimum_words=25)

        self.assertEqual(
            context,
            "\n\n".join([paragraphs[0]["text"], paragraphs[1]["text"], paragraphs[2]["text"]]),
        )

    def test_stops_when_next_previous_paragraph_would_exceed_budget(self):
        paragraphs = [
            {"text": "A" * 80},
            {"text": "B" * 30},
            {"text": "one two three four five"},
        ]

        budget = len(paragraphs[1]["text"]) + len(paragraphs[2]["text"]) + 2
        context = build_temperament_context(paragraphs, 2, budget=budget, minimum_words=50)

        self.assertEqual(context, "\n\n".join([paragraphs[1]["text"], paragraphs[2]["text"]]))

    def test_never_includes_following_paragraphs_and_target_stays_last(self):
        paragraphs = [
            {"text": "alpha beta gamma delta epsilon"},
            {"text": "zeta eta theta"},
            {"text": "FOLLOWING paragraph should not appear"},
        ]

        context = build_temperament_context(paragraphs, 1, budget=10_000, minimum_words=20)

        self.assertNotIn(paragraphs[2]["text"], context)
        self.assertTrue(context.endswith(paragraphs[1]["text"]))


if __name__ == "__main__":
    unittest.main()
