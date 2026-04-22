import json
import os
import subprocess
import sys
import tempfile
import unittest

from project import ProjectManager
from source_document import is_structural_silence_text


APP_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


class StructuralSilenceClassifierTests(unittest.TestCase):
    def test_detects_nonblank_text_with_no_alphanumeric_characters(self):
        self.assertTrue(is_structural_silence_text("***"))
        self.assertTrue(is_structural_silence_text(" - - - "))
        self.assertFalse(is_structural_silence_text(""))
        self.assertFalse(is_structural_silence_text("   "))
        self.assertFalse(is_structural_silence_text("Chapter 7"))
        self.assertFalse(is_structural_silence_text("...and then"))


class CreateScriptStructuralSilenceTests(unittest.TestCase):
    def _run_script(self, *args):
        env = os.environ.copy()
        existing = env.get("PYTHONPATH")
        env["PYTHONPATH"] = APP_DIR if not existing else APP_DIR + os.pathsep + existing
        result = subprocess.run(
            [sys.executable, *args],
            cwd=APP_DIR,
            capture_output=True,
            text=True,
            env=env,
        )
        if result.returncode != 0:
            raise AssertionError(
                f"Script failed: {' '.join(args)}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
            )
        return result

    def test_create_script_converts_structural_silence_paragraphs_to_silence_chunks(self):
        with tempfile.TemporaryDirectory() as temp_root:
            os.makedirs(os.path.join(temp_root, "app"), exist_ok=True)
            config_path = os.path.join(temp_root, "config.json")
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "export": {
                            "silence_end_of_chapter_ms": 4321,
                        }
                    },
                    f,
                    indent=2,
                )

            paragraphs_path = os.path.join(temp_root, "paragraphs.json")
            with open(paragraphs_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "paragraphs": [
                            {
                                "id": "p_0001",
                                "chapter": "Chapter One",
                                "text": "Plain narration.",
                                "has_dialogue": False,
                                "tone": "",
                            },
                            {
                                "id": "p_0002",
                                "chapter": "Chapter One",
                                "text": "***",
                                "has_dialogue": False,
                                "tone": "",
                                "is_structural_silence": True,
                            },
                            {
                                "id": "p_0003",
                                "chapter": "Chapter One",
                                "text": "More narration.",
                                "has_dialogue": False,
                                "tone": "",
                            },
                        ]
                    },
                    f,
                    indent=2,
                    ensure_ascii=False,
                )

            self._run_script(
                os.path.join("scripts", "create_script.py"),
                paragraphs_path,
                os.path.join(temp_root, "annotated_script.json"),
                os.path.join(temp_root, "chunks.json"),
            )

            manager = ProjectManager(temp_root)
            try:
                script_doc = manager.load_script_document()
                chunks = manager.load_chunks()
            finally:
                manager.shutdown_script_store(flush=True)

            self.assertEqual(len(script_doc["entries"]), 3)
            self.assertEqual(len(chunks), 3)

            silence_entry = script_doc["entries"][1]
            self.assertEqual(silence_entry["type"], "silence")
            self.assertEqual(silence_entry["chapter"], "Chapter One")
            self.assertEqual(silence_entry["paragraph_id"], "p_0002")
            self.assertAlmostEqual(silence_entry["silence_duration_s"], 4.321)

            silence_chunk = chunks[1]
            self.assertEqual(silence_chunk["type"], "silence")
            self.assertEqual(silence_chunk["chapter"], "Chapter One")
            self.assertEqual(silence_chunk["paragraph_id"], "p_0002")
            self.assertAlmostEqual(silence_chunk["silence_duration_s"], 4.321)
            self.assertEqual(chunks[0]["text"], "Plain narration.")
            self.assertEqual(chunks[2]["text"], "More narration.")


if __name__ == "__main__":
    unittest.main()
