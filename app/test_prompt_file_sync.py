import os
import tempfile
import unittest

from api.routers import config_router


class PromptFileSyncTests(unittest.TestCase):
    def test_sync_prompt_files_writes_all_labeled_prompt_files(self):
        with tempfile.TemporaryDirectory() as td:
            default_path = os.path.join(td, "default_prompts.txt")
            review_path = os.path.join(td, "review_prompts.txt")
            attribution_path = os.path.join(td, "attribution_prompts.txt")
            voice_path = os.path.join(td, "voice_prompt.txt")
            dialogue_path = os.path.join(td, "dialogue_identification_system_prompt.txt")
            temperament_path = os.path.join(td, "temperament_extraction_system_prompt.txt")

            original_paths = (
                config_router._DEFAULT_PROMPTS_PATH,
                config_router._REVIEW_PROMPTS_PATH,
                config_router._ATTRIBUTION_PROMPTS_PATH,
                config_router._VOICE_PROMPT_PATH,
                config_router._DIALOGUE_PROMPT_PATH,
                config_router._TEMPERAMENT_PROMPT_PATH,
            )

            config_router._DEFAULT_PROMPTS_PATH = default_path
            config_router._REVIEW_PROMPTS_PATH = review_path
            config_router._ATTRIBUTION_PROMPTS_PATH = attribution_path
            config_router._VOICE_PROMPT_PATH = voice_path
            config_router._DIALOGUE_PROMPT_PATH = dialogue_path
            config_router._TEMPERAMENT_PROMPT_PATH = temperament_path

            try:
                prompts = {
                    "system_prompt": "script-system",
                    "user_prompt": "script-user",
                    "review_system_prompt": "review-system",
                    "review_user_prompt": "review-user",
                    "attribution_system_prompt": "attr-system",
                    "attribution_user_prompt": "attr-user",
                    "voice_prompt": "voice-template",
                    "dialogue_identification_system_prompt": "dialogue-system",
                    "temperament_extraction_system_prompt": "temperament-system",
                }
                merged = config_router._sync_prompt_files(prompts, existing_prompts={})

                self.assertEqual(merged["voice_prompt"], "voice-template")
                self.assertTrue(os.path.exists(default_path))
                self.assertTrue(os.path.exists(review_path))
                self.assertTrue(os.path.exists(attribution_path))
                self.assertTrue(os.path.exists(voice_path))
                self.assertTrue(os.path.exists(dialogue_path))
                self.assertTrue(os.path.exists(temperament_path))

                with open(default_path, "r", encoding="utf-8") as f:
                    body = f.read()
                self.assertIn("script-system", body)
                self.assertIn("---SEPARATOR---", body)
                self.assertIn("script-user", body)

                with open(dialogue_path, "r", encoding="utf-8") as f:
                    self.assertEqual(f.read().strip(), "dialogue-system")
                with open(temperament_path, "r", encoding="utf-8") as f:
                    self.assertEqual(f.read().strip(), "temperament-system")
            finally:
                (
                    config_router._DEFAULT_PROMPTS_PATH,
                    config_router._REVIEW_PROMPTS_PATH,
                    config_router._ATTRIBUTION_PROMPTS_PATH,
                    config_router._VOICE_PROMPT_PATH,
                    config_router._DIALOGUE_PROMPT_PATH,
                    config_router._TEMPERAMENT_PROMPT_PATH,
                ) = original_paths


if __name__ == "__main__":
    unittest.main()
