import json
import os
import tempfile
import unittest
import asyncio

import app as app_module


class SavedVoiceReuseTests(unittest.TestCase):
    def test_narrator_alias_is_resolved_for_in_project_voice_reuse(self):
        self.assertEqual(
            app_module._resolve_voice_alias_target("NARRATOR", "Kit", {"NARRATOR", "Kit", "Maddie"}),
            "Kit",
        )
        self.assertEqual(
            app_module._resolve_voice_alias_target("CAT", "Sadie", {"CAT", "Sadie", "NARRATOR"}),
            "Sadie",
        )

    def test_finds_case_insensitive_reusable_clone_voice_by_speaker(self):
        with tempfile.TemporaryDirectory() as temp_root:
            clone_dir = os.path.join(temp_root, "clone_voices")
            designed_dir = os.path.join(temp_root, "designed_voices")
            os.makedirs(clone_dir, exist_ok=True)
            os.makedirs(designed_dir, exist_ok=True)

            clone_filename = "series_twilight.wav"
            with open(os.path.join(clone_dir, clone_filename), "wb") as f:
                f.write(b"wav")

            with open(os.path.join(clone_dir, "manifest.json"), "w", encoding="utf-8") as f:
                json.dump(
                    [
                        {
                            "id": "series_twilight",
                            "name": "Book One.Twilight Sparkle",
                            "speaker": "Twilight Sparkle",
                            "filename": clone_filename,
                            "sample_text": "Friendship is magic.",
                        }
                    ],
                    f,
                    indent=2,
                    ensure_ascii=False,
                )

            with open(os.path.join(designed_dir, "manifest.json"), "w", encoding="utf-8") as f:
                json.dump([], f)

            original_root = app_module.ROOT_DIR
            original_clone_manifest = app_module.CLONE_VOICES_MANIFEST
            original_designed_manifest = app_module.DESIGNED_VOICES_MANIFEST
            try:
                app_module.ROOT_DIR = temp_root
                app_module.CLONE_VOICES_MANIFEST = os.path.join(clone_dir, "manifest.json")
                app_module.DESIGNED_VOICES_MANIFEST = os.path.join(designed_dir, "manifest.json")

                match = app_module._find_saved_voice_option_for_speaker("twilight sparkle")
                self.assertIsNotNone(match)
                self.assertEqual(match["type"], "clone")
                self.assertEqual(match["ref_audio"], f"clone_voices/{clone_filename}")
                self.assertEqual(match["ref_text"], "Friendship is magic.")
            finally:
                app_module.ROOT_DIR = original_root
                app_module.CLONE_VOICES_MANIFEST = original_clone_manifest
                app_module.DESIGNED_VOICES_MANIFEST = original_designed_manifest

    def test_does_not_reuse_saved_voice_for_narrator(self):
        with tempfile.TemporaryDirectory() as temp_root:
            clone_dir = os.path.join(temp_root, "clone_voices")
            designed_dir = os.path.join(temp_root, "designed_voices")
            os.makedirs(clone_dir, exist_ok=True)
            os.makedirs(designed_dir, exist_ok=True)

            with open(os.path.join(clone_dir, "manifest.json"), "w", encoding="utf-8") as f:
                json.dump(
                    [{"id": "narrator_voice", "name": "Narrator", "speaker": "Narrator", "filename": "narrator.wav"}],
                    f,
                    indent=2,
                    ensure_ascii=False,
                )
            with open(os.path.join(clone_dir, "narrator.wav"), "wb") as f:
                f.write(b"wav")
            with open(os.path.join(designed_dir, "manifest.json"), "w", encoding="utf-8") as f:
                json.dump([], f)

            original_root = app_module.ROOT_DIR
            original_clone_manifest = app_module.CLONE_VOICES_MANIFEST
            original_designed_manifest = app_module.DESIGNED_VOICES_MANIFEST
            try:
                app_module.ROOT_DIR = temp_root
                app_module.CLONE_VOICES_MANIFEST = os.path.join(clone_dir, "manifest.json")
                app_module.DESIGNED_VOICES_MANIFEST = os.path.join(designed_dir, "manifest.json")

                self.assertIsNone(app_module._find_saved_voice_option_for_speaker("Narrator"))
            finally:
                app_module.ROOT_DIR = original_root
                app_module.CLONE_VOICES_MANIFEST = original_clone_manifest
                app_module.DESIGNED_VOICES_MANIFEST = original_designed_manifest

    def test_voice_processing_continues_after_single_speaker_failure(self):
        with tempfile.TemporaryDirectory() as temp_root:
            script_path = os.path.join(temp_root, "annotated_script.json")
            voice_config_path = os.path.join(temp_root, "voice_config.json")
            with open(script_path, "w", encoding="utf-8") as f:
                json.dump({"entries": []}, f)
            with open(voice_config_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "CAT": {"type": "design", "description": "cat desc", "ref_text": "cat text", "ref_audio": ""},
                        "DOG": {"type": "design", "description": "dog desc", "ref_text": "dog text", "ref_audio": ""},
                    },
                    f,
                    indent=2,
                    ensure_ascii=False,
                )

            calls = []
            logs = []

            original_root = app_module.ROOT_DIR
            original_script_path = app_module.SCRIPT_PATH
            original_voice_config_path = app_module.VOICE_CONFIG_PATH
            original_get_voices = app_module.awaitable_get_voices_sync
            original_task_current = app_module._task_is_current
            original_append_log = app_module._append_task_log
            original_finish = app_module._finish_task_run
            original_pm = app_module.project_manager
            try:
                app_module.ROOT_DIR = temp_root
                app_module.SCRIPT_PATH = script_path
                app_module.VOICE_CONFIG_PATH = voice_config_path
                app_module.awaitable_get_voices_sync = lambda: [
                    {"name": "CAT", "suggested_sample_text": "cat text"},
                    {"name": "DOG", "suggested_sample_text": "dog text"},
                ]
                app_module._task_is_current = lambda task_name, run_id: True
                app_module._append_task_log = lambda task_name, run_id, message: logs.append(message)
                app_module._finish_task_run = lambda task_name, run_id: None

                class FakeProjectManager:
                    def _normalize_speaker_name(self, value):
                        return (value or "").strip().lower()

                    def load_chunks(self):
                        return []

                    def suggest_design_sample_text(self, speaker, chunks):
                        return f"{speaker.lower()} sample"

                    def materialize_design_voice(self, speaker, description, sample_text, force, voice_config):
                        calls.append(speaker)
                        if speaker == "CAT":
                            raise RuntimeError("cat failed")
                        updated = json.loads(json.dumps(voice_config))
                        updated[speaker]["ref_audio"] = f"clone_voices/{speaker.lower()}.wav"
                        return {"voice_config": updated}

                app_module.project_manager = FakeProjectManager()

                success = app_module.run_voice_processing_task("run-1")
                self.assertFalse(success)
                self.assertEqual(calls, ["CAT", "DOG"])
                self.assertTrue(any("Failed to create voice for CAT" in message for message in logs))
                self.assertTrue(any("Created reusable voice for DOG" in message for message in logs))
            finally:
                app_module.ROOT_DIR = original_root
                app_module.SCRIPT_PATH = original_script_path
                app_module.VOICE_CONFIG_PATH = original_voice_config_path
                app_module.awaitable_get_voices_sync = original_get_voices
                app_module._task_is_current = original_task_current
                app_module._append_task_log = original_append_log
                app_module._finish_task_run = original_finish
                app_module.project_manager = original_pm

    def test_suggest_voice_description_sync_uses_config_path_without_get_config(self):
        with tempfile.TemporaryDirectory() as temp_root:
            config_path = os.path.join(temp_root, "config.json")
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "llm": {
                            "base_url": "http://localhost:11434/v1",
                            "api_key": "local",
                            "model_name": "test-model",
                            "timeout": 5,
                        },
                        "prompts": {
                            "voice_prompt": "Describe {character_name}",
                        },
                    },
                    f,
                    indent=2,
                    ensure_ascii=False,
                )

            original_config_path = app_module.CONFIG_PATH
            original_pm = app_module.project_manager
            original_openai = app_module.OpenAI
            try:
                app_module.CONFIG_PATH = config_path

                class FakeProjectManager:
                    def build_voice_suggestion_prompt(self, speaker, prompt_template):
                        return {
                            "prompt": f"{prompt_template} :: {speaker}",
                            "paragraphs": [],
                            "context_chars": 0,
                        }

                class FakeOpenAI:
                    def __init__(self, **kwargs):
                        self.chat = self
                        self.completions = self

                    def create(self, **kwargs):
                        class _Msg:
                            content = '{"voice":"Warm, measured narrator voice."}'

                        class _Choice:
                            message = _Msg()

                        class _Resp:
                            choices = [_Choice()]

                        return _Resp()

                app_module.project_manager = FakeProjectManager()
                app_module.OpenAI = FakeOpenAI

                result = app_module.suggest_voice_description_sync("Narrator")
                self.assertEqual(result["status"], "ok")
                self.assertEqual(result["speaker"], "Narrator")
                self.assertEqual(result["voice"], "Warm, measured narrator voice.")
            finally:
                app_module.CONFIG_PATH = original_config_path
                app_module.project_manager = original_pm
                app_module.OpenAI = original_openai

    def test_get_voices_deduplicates_case_variants(self):
        with tempfile.TemporaryDirectory() as temp_root:
            script_path = os.path.join(temp_root, "annotated_script.json")
            voice_config_path = os.path.join(temp_root, "voice_config.json")
            voices_path = os.path.join(temp_root, "voices.json")
            with open(script_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "entries": [
                            {"speaker": "Narrator", "text": "A"},
                            {"speaker": "NARRATOR", "text": "B"},
                        ],
                        "dictionary": [],
                    },
                    f,
                    indent=2,
                    ensure_ascii=False,
                )
            with open(voice_config_path, "w", encoding="utf-8") as f:
                json.dump({"narrator": {"type": "design", "description": "x"}}, f, indent=2, ensure_ascii=False)

            original_root = app_module.ROOT_DIR
            original_script_path = app_module.SCRIPT_PATH
            original_voice_config_path = app_module.VOICE_CONFIG_PATH
            original_voices_path = app_module.VOICES_PATH
            try:
                app_module.ROOT_DIR = temp_root
                app_module.SCRIPT_PATH = script_path
                app_module.VOICE_CONFIG_PATH = voice_config_path
                app_module.VOICES_PATH = voices_path

                voices = asyncio.run(app_module.get_voices())
                self.assertEqual(len(voices), 1)
                self.assertEqual(voices[0]["name"], "NARRATOR")
                self.assertEqual(voices[0]["line_count"], 2)
                self.assertEqual((voices[0]["config"] or {}).get("description"), "x")
            finally:
                app_module.ROOT_DIR = original_root
                app_module.SCRIPT_PATH = original_script_path
                app_module.VOICE_CONFIG_PATH = original_voice_config_path
                app_module.VOICES_PATH = original_voices_path

    def test_save_voice_config_with_invalidation_preserves_unsubmitted_voices(self):
        captured = {}

        class FakeProjectManager:
            def _normalize_speaker_name(self, value):
                return (value or "").strip().lower()

            def _load_voice_config(self):
                return {
                    "NARRATOR": {"type": "design", "description": "existing narrator"},
                    "CAT": {"type": "clone", "ref_audio": "clone_voices/cat.wav"},
                }

            def save_voice_config_with_invalidation(self, new_config, confirm_invalidation=False):
                captured["new_config"] = json.loads(json.dumps(new_config))
                captured["confirm_invalidation"] = bool(confirm_invalidation)
                return {"status": "saved", "invalidated_clips": 0}

        original_pm = app_module.project_manager
        try:
            app_module.project_manager = FakeProjectManager()
            request = app_module.VoiceConfigSaveRequest(
                config={
                    "NARRATOR": app_module.VoiceConfigItem(
                        type="design",
                        description="updated narrator",
                        ref_text="sample",
                        ref_audio="",
                        generated_ref_text="",
                        alias="",
                        seed="-1",
                    )
                },
                confirm_invalidation=False,
            )
            result = asyncio.run(app_module.save_voice_config_with_invalidation(request))
            self.assertEqual(result.get("status"), "saved")
            self.assertFalse(captured["confirm_invalidation"])
            self.assertIn("NARRATOR", captured["new_config"])
            self.assertIn("CAT", captured["new_config"])
            self.assertEqual(captured["new_config"]["NARRATOR"]["description"], "updated narrator")
            self.assertEqual(captured["new_config"]["CAT"]["ref_audio"], "clone_voices/cat.wav")
        finally:
            app_module.project_manager = original_pm


if __name__ == "__main__":
    unittest.main()
