import os
import tempfile
import unittest

from script_store import load_script_document, save_script_document


class ScriptStoreTests(unittest.TestCase):
    def test_sanity_cache_round_trips(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "annotated_script.json")
            save_script_document(
                path,
                entries=[{"speaker": "NARRATOR", "text": "Hello", "instruct": ""}],
                dictionary=[],
                sanity_cache={
                    "phrase_decisions": {
                        "he said softly": {
                            "phrase": "he said softly",
                            "decision": "accepted",
                            "reply": "TRUE",
                            "checked_at": 1,
                        },
                    },
                },
            )

            loaded = load_script_document(path)

            self.assertIn("sanity_cache", loaded)
            self.assertIn("phrase_decisions", loaded["sanity_cache"])
            self.assertEqual(
                loaded["sanity_cache"]["phrase_decisions"]["he said softly"]["decision"],
                "accepted",
            )


if __name__ == "__main__":
    unittest.main()
