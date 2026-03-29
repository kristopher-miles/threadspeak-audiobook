import unittest

from factory_prompt_defaults import load_factory_default_prompts


class FactoryPromptDefaultsTests(unittest.TestCase):
    def test_load_factory_default_prompts_returns_all_prompt_keys(self):
        data = load_factory_default_prompts()
        expected = {
            "system_prompt",
            "user_prompt",
            "review_system_prompt",
            "review_user_prompt",
            "attribution_system_prompt",
            "attribution_user_prompt",
            "voice_prompt",
            "dialogue_identification_system_prompt",
            "temperament_extraction_system_prompt",
        }
        self.assertEqual(set(data.keys()), expected)
        for key in expected:
            self.assertTrue((data.get(key) or "").strip(), f"expected non-empty: {key}")


if __name__ == "__main__":
    unittest.main()
