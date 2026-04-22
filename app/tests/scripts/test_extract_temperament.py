import unittest
import sys
import inspect
import json
import os
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

sys.modules.setdefault("openai", SimpleNamespace(OpenAI=object))

import scripts.extract_temperament as extract_temperament_module
from scripts.extract_temperament import build_temperament_context


class BuildTemperamentContextTests(unittest.TestCase):
    def test_no_llm_telemetry_log_lines_in_extract_temperament_script(self):
        self.assertNotIn("LLM telemetry:", inspect.getsource(extract_temperament_module))

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

    def test_retry_errors_mode_only_retries_failed_quote_moods_and_preserves_successes(self):
        paragraphs_doc = {
            "paragraphs": [
                {
                    "id": "p_0001",
                    "text": '"Hello." "Goodbye."',
                    "has_dialogue": True,
                    "speakers": ["Alice", "Bob"],
                    "tone": "calm",
                    "temperament_error": False,
                    "dialogue_moods": ["warm", ""],
                    "quote_mood_errors": [False, True],
                    "dialogue_mood_error": True,
                }
            ],
            "temperament_extraction_complete": True,
            "temperament_errors": [],
            "dialogue_mood_errors": ["p_0001"],
        }
        persisted = {}

        class FakeStore:
            def load_project_document(self, name):
                self.loaded = name
                return paragraphs_doc

            def stop(self):
                return None

        def fake_call_sentiment(client, runtime, system_prompt, user_msg, max_tokens):
            if "DIALOGUE:\nGoodbye." not in user_msg:
                raise AssertionError(f"unexpected retry target: {user_msg}")
            return ("sharp", "{}", "tool", True)

        with tempfile.TemporaryDirectory() as temp_root:
            config_path = os.path.join(temp_root, "config.json")
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump({"llm": {}, "generation": {"temperament_words": 150, "max_tokens": 64}}, f)

            argv = [
                "extract_temperament.py",
                "--project-root",
                temp_root,
                "--retry-errors",
                "3",
                config_path,
            ]

            with patch.object(sys, "argv", argv):
                with patch.object(extract_temperament_module, "open_project_script_store", return_value=FakeStore()):
                    with patch.object(extract_temperament_module, "_persist_paragraphs_doc", lambda path, root, doc: persisted.update(doc)):
                        with patch.object(extract_temperament_module._LLM_CLIENT_FACTORY, "create_client", return_value=object()):
                            with patch.object(extract_temperament_module, "call_sentiment", side_effect=fake_call_sentiment):
                                extract_temperament_module.main()

        self.assertEqual(persisted["paragraphs"][0]["dialogue_moods"], ["warm", "sharp"])
        self.assertEqual(persisted["paragraphs"][0]["quote_mood_errors"], [False, False])
        self.assertFalse(persisted["paragraphs"][0]["dialogue_mood_error"])
        self.assertEqual(persisted["dialogue_mood_errors"], [])


if __name__ == "__main__":
    unittest.main()
