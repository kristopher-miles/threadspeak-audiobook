import json
import os
import tempfile
import threading
import time
import unittest
import asyncio

import app as app_module
from api.routers import voice_designer_router as voice_designer_router
from llm.models import StructuredLLMResult
from project import ProjectManager


def _seed_project_manager(temp_root, *, entries=None, voice_config=None, paragraphs=None):
    os.makedirs(os.path.join(temp_root, "app"), exist_ok=True)
    manager = ProjectManager(temp_root)
    if entries is not None:
        manager.script_store.replace_script_document(
            entries=list(entries),
            dictionary=[],
            sanity_cache={"phrase_decisions": {}},
            reason="test_seed_script",
            rebuild_chunks=False,
            wait=True,
        )
    if voice_config is not None:
        manager._save_voice_config(json.loads(json.dumps(voice_config)))
    if paragraphs is not None:
        manager.save_paragraphs({"paragraphs": list(paragraphs)})
    return manager


def _shutdown_manager(manager):
    if manager is not None:
        manager.shutdown_script_store(flush=True)


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

    def test_voice_processing_same_project_reusable_voice_is_regenerated_instead_of_auto_populated(self):
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
                            "script_title": "Book One",
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
            original_get_voices = app_module.awaitable_get_voices_sync
            original_task_current = app_module._task_is_current
            original_append_log = app_module._append_task_log
            original_finish = app_module._finish_task_run
            original_pm = app_module.project_manager
            original_suggest = app_module.suggest_voice_description_sync
            manager = None
            try:
                manager = _seed_project_manager(
                    temp_root,
                    entries=[{"speaker": "Twilight Sparkle", "text": "Hello."}],
                    voice_config={},
                )
                manager._current_script_title = lambda: "Book One"
                manager.load_chunks = lambda: []
                manager.suggest_design_sample_text = lambda speaker, chunks: "Friendship is magic."
                manager.unload_tts_engine = lambda: False
                materialized = []

                def _materialize_design_voice(speaker, description, sample_text, force, voice_config):
                    materialized.append((speaker, description, sample_text))
                    updated = json.loads(json.dumps(voice_config))
                    updated.setdefault(speaker, {})
                    updated[speaker]["ref_audio"] = "clone_voices/generated.wav"
                    updated[speaker]["generated_ref_text"] = sample_text
                    manager._save_voice_config(updated)
                    return {"voice_config": updated}

                manager.materialize_design_voice = _materialize_design_voice
                app_module.ROOT_DIR = temp_root
                app_module.CLONE_VOICES_MANIFEST = os.path.join(clone_dir, "manifest.json")
                app_module.DESIGNED_VOICES_MANIFEST = os.path.join(designed_dir, "manifest.json")
                app_module.awaitable_get_voices_sync = lambda: [
                    {"name": "twilight sparkle", "suggested_sample_text": "Friendship is magic."}
                ]
                app_module._task_is_current = lambda task_name, run_id: True
                logs = []
                app_module._append_task_log = lambda task_name, run_id, message: logs.append(message)
                app_module._finish_task_run = lambda task_name, run_id: None
                app_module.project_manager = manager
                app_module.suggest_voice_description_sync = lambda speaker: {"voice": "Warm, bright voice"}
                success = app_module.run_voice_processing_task("run-1")
                self.assertTrue(success)
                self.assertEqual(
                    materialized,
                    [("twilight sparkle", "Warm, bright voice", "Friendship is magic.")],
                )
                cfg = manager._load_voice_config()
                self.assertEqual(cfg["twilight sparkle"]["type"], "design")
                self.assertEqual(cfg["twilight sparkle"]["ref_audio"], "clone_voices/generated.wav")
                self.assertEqual(cfg["twilight sparkle"]["ref_text"], "Friendship is magic.")
                self.assertEqual(cfg["twilight sparkle"]["generated_ref_text"], "Friendship is magic.")
                self.assertEqual(cfg["twilight sparkle"]["description"], "Warm, bright voice")
                self.assertFalse(any("Auto-populated twilight sparkle" in message for message in logs))
            finally:
                app_module.ROOT_DIR = original_root
                app_module.CLONE_VOICES_MANIFEST = original_clone_manifest
                app_module.DESIGNED_VOICES_MANIFEST = original_designed_manifest
                app_module.awaitable_get_voices_sync = original_get_voices
                app_module._task_is_current = original_task_current
                app_module._append_task_log = original_append_log
                app_module._finish_task_run = original_finish
                app_module.project_manager = original_pm
                app_module.suggest_voice_description_sync = original_suggest
                _shutdown_manager(manager)

    def test_voice_processing_does_not_auto_populate_voice_from_other_project(self):
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
                            "name": "Book Two.Twilight Sparkle",
                            "speaker": "Twilight Sparkle",
                            "script_title": "Book Two",
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
            original_get_voices = app_module.awaitable_get_voices_sync
            original_task_current = app_module._task_is_current
            original_append_log = app_module._append_task_log
            original_finish = app_module._finish_task_run
            original_pm = app_module.project_manager
            original_suggest = app_module.suggest_voice_description_sync
            manager = None
            try:
                manager = _seed_project_manager(
                    temp_root,
                    entries=[{"speaker": "Twilight Sparkle", "text": "Hello."}],
                    voice_config={},
                )
                manager._current_script_title = lambda: "Book One"
                manager.load_chunks = lambda: []
                manager.suggest_design_sample_text = lambda speaker, chunks: "Friendship is magic."
                app_module.ROOT_DIR = temp_root
                app_module.CLONE_VOICES_MANIFEST = os.path.join(clone_dir, "manifest.json")
                app_module.DESIGNED_VOICES_MANIFEST = os.path.join(designed_dir, "manifest.json")
                app_module.awaitable_get_voices_sync = lambda: [
                    {"name": "twilight sparkle", "suggested_sample_text": "Friendship is magic."}
                ]
                app_module._task_is_current = lambda task_name, run_id: True
                logs = []
                materialized = []
                app_module._append_task_log = lambda task_name, run_id, message: logs.append(message)
                app_module._finish_task_run = lambda task_name, run_id: None
                def _materialize_design_voice(speaker, description, sample_text, force, voice_config):
                    materialized.append((speaker, description, sample_text))
                    updated = json.loads(json.dumps(voice_config))
                    updated[speaker]["ref_audio"] = "clone_voices/generated.wav"
                    return {"voice_config": updated}

                manager.materialize_design_voice = _materialize_design_voice
                manager.unload_tts_engine = lambda: False
                app_module.project_manager = manager
                app_module.suggest_voice_description_sync = lambda speaker: {"voice": "Warm, bright voice"}

                success = app_module.run_voice_processing_task("run-1")
                self.assertTrue(success)
                self.assertEqual(materialized, [("twilight sparkle", "Warm, bright voice", "Friendship is magic.")])
                self.assertFalse(any("Auto-populated twilight sparkle" in message for message in logs))
            finally:
                app_module.ROOT_DIR = original_root
                app_module.CLONE_VOICES_MANIFEST = original_clone_manifest
                app_module.DESIGNED_VOICES_MANIFEST = original_designed_manifest
                app_module.awaitable_get_voices_sync = original_get_voices
                app_module._task_is_current = original_task_current
                app_module._append_task_log = original_append_log
                app_module._finish_task_run = original_finish
                app_module.project_manager = original_pm
                app_module.suggest_voice_description_sync = original_suggest
                _shutdown_manager(manager)

    def test_saved_voice_reuse_falls_back_to_loaded_project_name_with_exact_speaker_match(self):
        with tempfile.TemporaryDirectory() as temp_root:
            clone_dir = os.path.join(temp_root, "clone_voices")
            designed_dir = os.path.join(temp_root, "designed_voices")
            os.makedirs(clone_dir, exist_ok=True)
            os.makedirs(designed_dir, exist_ok=True)

            clone_filename = "demo_save_twilight.wav"
            with open(os.path.join(clone_dir, clone_filename), "wb") as f:
                f.write(b"wav")

            with open(os.path.join(clone_dir, "manifest.json"), "w", encoding="utf-8") as f:
                json.dump(
                    [
                        {
                            "id": "demo_save_twilight",
                            "name": "Twilight Sparkle",
                            "speaker": "Twilight Sparkle",
                            "script_title": "demo_save",
                            "filename": clone_filename,
                            "sample_text": "Friendship is magic.",
                            "description": "Warm voice",
                        }
                    ],
                    f,
                    indent=2,
                    ensure_ascii=False,
                )

            with open(os.path.join(designed_dir, "manifest.json"), "w", encoding="utf-8") as f:
                json.dump([], f)
            with open(os.path.join(temp_root, "state.json"), "w", encoding="utf-8") as f:
                json.dump({"loaded_project_name": "demo_save"}, f)

            original_root = app_module.ROOT_DIR
            original_clone_manifest = app_module.CLONE_VOICES_MANIFEST
            original_designed_manifest = app_module.DESIGNED_VOICES_MANIFEST
            original_pm = app_module.project_manager
            try:
                app_module.ROOT_DIR = temp_root
                app_module.CLONE_VOICES_MANIFEST = os.path.join(clone_dir, "manifest.json")
                app_module.DESIGNED_VOICES_MANIFEST = os.path.join(designed_dir, "manifest.json")

                class FakeProjectManager:
                    def _normalize_speaker_name(self, value):
                        return (value or "").strip().lower()

                    def _current_script_title(self):
                        return "Book One"

                app_module.project_manager = FakeProjectManager()
                result = app_module._find_saved_voice_option_for_speaker("Twilight Sparkle")
                self.assertIsNotNone(result)
                self.assertEqual(result["ref_audio"], f"clone_voices/{clone_filename}")
                self.assertEqual(result["ref_text"], "Friendship is magic.")
                self.assertEqual(result["description"], "Warm voice")
            finally:
                app_module.ROOT_DIR = original_root
                app_module.CLONE_VOICES_MANIFEST = original_clone_manifest
                app_module.DESIGNED_VOICES_MANIFEST = original_designed_manifest
                app_module.project_manager = original_pm

    def test_saved_voice_reuse_project_name_fallback_requires_exact_speaker_match(self):
        with tempfile.TemporaryDirectory() as temp_root:
            clone_dir = os.path.join(temp_root, "clone_voices")
            designed_dir = os.path.join(temp_root, "designed_voices")
            os.makedirs(clone_dir, exist_ok=True)
            os.makedirs(designed_dir, exist_ok=True)

            clone_filename = "demo_save_blake_hodges.wav"
            with open(os.path.join(clone_dir, clone_filename), "wb") as f:
                f.write(b"wav")

            with open(os.path.join(clone_dir, "manifest.json"), "w", encoding="utf-8") as f:
                json.dump(
                    [
                        {
                            "id": "demo_save_blake_hodges",
                            "name": "Blake Hodges",
                            "speaker": "Blake Hodges",
                            "script_title": "demo_save",
                            "filename": clone_filename,
                            "sample_text": "Hello there.",
                        }
                    ],
                    f,
                    indent=2,
                    ensure_ascii=False,
                )

            with open(os.path.join(designed_dir, "manifest.json"), "w", encoding="utf-8") as f:
                json.dump([], f)
            with open(os.path.join(temp_root, "state.json"), "w", encoding="utf-8") as f:
                json.dump({"loaded_project_name": "demo_save"}, f)

            original_root = app_module.ROOT_DIR
            original_clone_manifest = app_module.CLONE_VOICES_MANIFEST
            original_designed_manifest = app_module.DESIGNED_VOICES_MANIFEST
            original_pm = app_module.project_manager
            try:
                app_module.ROOT_DIR = temp_root
                app_module.CLONE_VOICES_MANIFEST = os.path.join(clone_dir, "manifest.json")
                app_module.DESIGNED_VOICES_MANIFEST = os.path.join(designed_dir, "manifest.json")

                class FakeProjectManager:
                    def _normalize_speaker_name(self, value):
                        return (value or "").strip().lower()

                    def _current_script_title(self):
                        return "Book One"

                app_module.project_manager = FakeProjectManager()
                self.assertIsNone(app_module._find_saved_voice_option_for_speaker("Blake"))
            finally:
                app_module.ROOT_DIR = original_root
                app_module.CLONE_VOICES_MANIFEST = original_clone_manifest
                app_module.DESIGNED_VOICES_MANIFEST = original_designed_manifest
                app_module.project_manager = original_pm

    def test_saved_voice_reuse_matches_project_prefixed_name_with_exact_suffix(self):
        with tempfile.TemporaryDirectory() as temp_root:
            clone_dir = os.path.join(temp_root, "clone_voices")
            designed_dir = os.path.join(temp_root, "designed_voices")
            os.makedirs(clone_dir, exist_ok=True)
            os.makedirs(designed_dir, exist_ok=True)

            clone_filename = "forbidden-places.aerial.wav"
            with open(os.path.join(clone_dir, clone_filename), "wb") as f:
                f.write(b"wav")

            with open(os.path.join(clone_dir, "manifest.json"), "w", encoding="utf-8") as f:
                json.dump(
                    [
                        {
                            "id": "forbidden_places_aerial",
                            "name": "forbidden-places.Aerial",
                            "script_title": "legacy-title",
                            "filename": clone_filename,
                            "sample_text": "Sky sample",
                            "description": "Airy voice",
                        }
                    ],
                    f,
                    indent=2,
                    ensure_ascii=False,
                )

            with open(os.path.join(designed_dir, "manifest.json"), "w", encoding="utf-8") as f:
                json.dump([], f)
            with open(os.path.join(temp_root, "state.json"), "w", encoding="utf-8") as f:
                json.dump({"loaded_project_name": "forbidden-places"}, f)

            original_root = app_module.ROOT_DIR
            original_clone_manifest = app_module.CLONE_VOICES_MANIFEST
            original_designed_manifest = app_module.DESIGNED_VOICES_MANIFEST
            original_pm = app_module.project_manager
            try:
                app_module.ROOT_DIR = temp_root
                app_module.CLONE_VOICES_MANIFEST = os.path.join(clone_dir, "manifest.json")
                app_module.DESIGNED_VOICES_MANIFEST = os.path.join(designed_dir, "manifest.json")

                class FakeProjectManager:
                    def _normalize_speaker_name(self, value):
                        return (value or "").strip().lower()

                    def _current_script_title(self):
                        return "Book One"

                app_module.project_manager = FakeProjectManager()
                result = app_module._find_saved_voice_option_for_speaker("Aerial")
                self.assertIsNotNone(result)
                self.assertEqual(result["ref_audio"], f"clone_voices/{clone_filename}")
                self.assertEqual(result["ref_text"], "Sky sample")
                self.assertEqual(result["description"], "Airy voice")
            finally:
                app_module.ROOT_DIR = original_root
                app_module.CLONE_VOICES_MANIFEST = original_clone_manifest
                app_module.DESIGNED_VOICES_MANIFEST = original_designed_manifest
                app_module.project_manager = original_pm

    def test_get_voices_does_not_auto_populate_blank_row_from_project_prefixed_saved_voice(self):
        with tempfile.TemporaryDirectory() as temp_root:
            clone_dir = os.path.join(temp_root, "clone_voices")
            designed_dir = os.path.join(temp_root, "designed_voices")
            os.makedirs(clone_dir, exist_ok=True)
            os.makedirs(designed_dir, exist_ok=True)

            clone_filename = "forbidden-places_aerial.wav"
            with open(os.path.join(clone_dir, clone_filename), "wb") as f:
                f.write(b"wav")

            with open(os.path.join(clone_dir, "manifest.json"), "w", encoding="utf-8") as f:
                json.dump(
                    [
                        {
                            "id": "forbidden_places_aerial",
                            "name": "forbidden-places.Aerial",
                            "speaker": "Aerial",
                            "script_title": "forbidden-places",
                            "filename": clone_filename,
                            "sample_text": "The horizon brightened.",
                            "description": "Airy young heroine voice",
                        }
                    ],
                    f,
                    indent=2,
                    ensure_ascii=False,
                )

            with open(os.path.join(designed_dir, "manifest.json"), "w", encoding="utf-8") as f:
                json.dump([], f)

            state_path = os.path.join(temp_root, "state.json")
            with open(state_path, "w", encoding="utf-8") as f:
                json.dump({"loaded_project_name": "forbidden-places"}, f, indent=2, ensure_ascii=False)

            original_root = app_module.ROOT_DIR
            original_clone_manifest = app_module.CLONE_VOICES_MANIFEST
            original_designed_manifest = app_module.DESIGNED_VOICES_MANIFEST
            original_pm = app_module.project_manager
            manager = None
            try:
                manager = _seed_project_manager(
                    temp_root,
                    entries=[{"speaker": "Aerial", "text": "Hello."}],
                    voice_config={},
                )
                manager.suggest_design_sample_text = lambda speaker, chunks: ""
                app_module.ROOT_DIR = temp_root
                app_module.CLONE_VOICES_MANIFEST = os.path.join(clone_dir, "manifest.json")
                app_module.DESIGNED_VOICES_MANIFEST = os.path.join(designed_dir, "manifest.json")
                app_module.project_manager = manager

                voices = asyncio.run(app_module.get_voices())
                voices_by_name = {voice["name"]: voice for voice in voices}
                self.assertEqual(voices_by_name["Aerial"]["config"].get("type"), "design")
                self.assertFalse((voices_by_name["Aerial"]["config"].get("ref_audio") or "").strip())
                self.assertFalse((voices_by_name["Aerial"]["config"].get("generated_ref_text") or "").strip())
                self.assertFalse((voices_by_name["Aerial"]["config"].get("description") or "").strip())
            finally:
                app_module.ROOT_DIR = original_root
                app_module.CLONE_VOICES_MANIFEST = original_clone_manifest
                app_module.DESIGNED_VOICES_MANIFEST = original_designed_manifest
                app_module.project_manager = original_pm
                _shutdown_manager(manager)

    def test_script_store_list_voice_rows_excludes_profile_only_speakers(self):
        with tempfile.TemporaryDirectory() as temp_root:
            manager = None
            try:
                manager = _seed_project_manager(
                    temp_root,
                    voice_config={
                        "Aerial": {"type": "design", "description": "live"},
                        "Retired": {"type": "design", "description": "stale"},
                    },
                )
                manager.save_chunks(
                    [
                        {"id": 0, "uid": "chunk-1", "speaker": "Aerial", "text": "Hello.", "status": "pending"},
                        {"id": 1, "uid": "chunk-2", "speaker": "Aerial", "text": "Again.", "status": "pending"},
                    ]
                )

                rows = manager.script_store.list_voice_rows()
                rows_by_name = {row["name"]: row for row in rows}
                self.assertEqual(set(rows_by_name.keys()), {"Aerial"})
                self.assertEqual(rows_by_name["Aerial"]["line_count"], 2)
                self.assertEqual(rows_by_name["Aerial"]["config"].get("description"), "live")
            finally:
                _shutdown_manager(manager)

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
            calls = []
            logs = []

            original_root = app_module.ROOT_DIR
            original_get_voices = app_module.awaitable_get_voices_sync
            original_task_current = app_module._task_is_current
            original_append_log = app_module._append_task_log
            original_finish = app_module._finish_task_run
            original_pm = app_module.project_manager
            manager = None
            try:
                manager = _seed_project_manager(
                    temp_root,
                    entries=[{"speaker": "CAT", "text": "meow"}, {"speaker": "DOG", "text": "woof"}],
                    voice_config={
                        "CAT": {"type": "design", "description": "cat desc", "ref_text": "cat text", "ref_audio": ""},
                        "DOG": {"type": "design", "description": "dog desc", "ref_text": "dog text", "ref_audio": ""},
                    },
                )
                manager._current_script_title = lambda: "Project"
                manager.load_chunks = lambda: []
                manager.suggest_design_sample_text = lambda speaker, chunks: f"{speaker.lower()} sample"
                def _materialize_design_voice(speaker, description, sample_text, force, voice_config):
                    calls.append(speaker)
                    if speaker == "CAT":
                        raise RuntimeError("cat failed")
                    updated = json.loads(json.dumps(voice_config))
                    updated[speaker]["ref_audio"] = f"clone_voices/{speaker.lower()}.wav"
                    return {"voice_config": updated}
                manager.materialize_design_voice = _materialize_design_voice
                app_module.ROOT_DIR = temp_root
                app_module.awaitable_get_voices_sync = lambda: [
                    {"name": "CAT", "suggested_sample_text": "cat text"},
                    {"name": "DOG", "suggested_sample_text": "dog text"},
                ]
                app_module._task_is_current = lambda task_name, run_id: True
                app_module._append_task_log = lambda task_name, run_id, message: logs.append(message)
                app_module._finish_task_run = lambda task_name, run_id: None
                app_module.project_manager = manager

                success = app_module.run_voice_processing_task("run-1")
                self.assertFalse(success)
                self.assertEqual(calls, ["CAT", "DOG"])
                self.assertTrue(any("Failed to create voice for CAT" in message for message in logs))
                self.assertTrue(any("Created reusable voice for DOG" in message for message in logs))
            finally:
                app_module.ROOT_DIR = original_root
                app_module.awaitable_get_voices_sync = original_get_voices
                app_module._task_is_current = original_task_current
                app_module._append_task_log = original_append_log
                app_module._finish_task_run = original_finish
                app_module.project_manager = original_pm
                _shutdown_manager(manager)

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
            original_llm_client_factory = app_module._LLM_CLIENT_FACTORY
            try:
                app_module.CONFIG_PATH = config_path

                class FakeProjectManager:
                    def build_voice_suggestion_prompt(self, speaker, prompt_template):
                        return {
                            "prompt": f"{prompt_template} :: {speaker}",
                            "paragraphs": [],
                            "context_chars": 0,
                        }

                class FakeClient:
                    def __init__(self):
                        self.chat = self
                        self.completions = self

                    def create(self, **kwargs):
                        class _Msg:
                            content = ""
                            tool_calls = [
                                type(
                                    "_ToolCall",
                                    (),
                                    {
                                        "function": type(
                                            "_Function",
                                            (),
                                            {"arguments": '{"voice":"Warm, measured narrator voice."}'},
                                        )()
                                    },
                                )()
                            ]

                        class _Choice:
                            message = _Msg()

                        class _Resp:
                            choices = [_Choice()]

                        return _Resp()

                class FakeLLMClientFactory:
                    def __init__(self, *args, **kwargs):
                        pass

                    def create_client(self, runtime):
                        return FakeClient()

                app_module.project_manager = FakeProjectManager()
                app_module._LLM_CLIENT_FACTORY = FakeLLMClientFactory()

                result = app_module.suggest_voice_description_sync("Narrator")
                self.assertEqual(result["status"], "ok")
                self.assertEqual(result["speaker"], "Narrator")
                self.assertEqual(result["voice"], "Warm, measured narrator voice.")
            finally:
                app_module.CONFIG_PATH = original_config_path
                app_module.project_manager = original_pm
                app_module._LLM_CLIENT_FACTORY = original_llm_client_factory

    def test_suggest_voice_description_sync_accepts_plain_text_result(self):
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
            original_llm_client_factory = app_module._LLM_CLIENT_FACTORY
            original_structured_llm = app_module._STRUCTURED_LLM_SERVICE
            try:
                app_module.CONFIG_PATH = config_path

                class FakeProjectManager:
                    def build_voice_suggestion_prompt(self, speaker, prompt_template):
                        return {
                            "prompt": f"{prompt_template} :: {speaker}",
                            "paragraphs": [],
                            "context_chars": 0,
                        }

                class FakeLLMClientFactory:
                    def create_client(self, runtime):
                        return object()

                class FakeStructuredService:
                    def run(self, **kwargs):
                        return StructuredLLMResult(
                            mode="json",
                            parsed=None,
                            text="Warm, measured narrator voice.",
                            raw_payload="Warm, measured narrator voice.",
                            tool_call_observed=False,
                        )

                app_module.project_manager = FakeProjectManager()
                app_module._LLM_CLIENT_FACTORY = FakeLLMClientFactory()
                app_module._STRUCTURED_LLM_SERVICE = FakeStructuredService()

                result = app_module.suggest_voice_description_sync("Narrator")
                self.assertEqual(result["status"], "ok")
                self.assertEqual(result["speaker"], "Narrator")
                self.assertEqual(result["voice"], "Warm, measured narrator voice.")
            finally:
                app_module.CONFIG_PATH = original_config_path
                app_module.project_manager = original_pm
                app_module._LLM_CLIENT_FACTORY = original_llm_client_factory
                app_module._STRUCTURED_LLM_SERVICE = original_structured_llm

    def test_run_voice_processing_task_suggests_all_before_generation_and_unloads(self):
        with tempfile.TemporaryDirectory() as temp_root:
            config_path = os.path.join(temp_root, "config.json")
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump({"llm": {"llm_workers": 2}}, f, indent=2, ensure_ascii=False)

            events = []
            logs = []
            state = {
                "active": 0,
                "max_active": 0,
                "completed_suggestions": 0,
            }
            lock = threading.Lock()

            original_root = app_module.ROOT_DIR
            original_config_path = app_module.CONFIG_PATH
            original_get_voices = app_module.awaitable_get_voices_sync
            original_task_current = app_module._task_is_current
            original_append_log = app_module._append_task_log
            original_finish = app_module._finish_task_run
            original_pm = app_module.project_manager
            original_suggest = app_module.suggest_voice_description_sync
            manager = None
            try:
                manager = _seed_project_manager(
                    temp_root,
                    entries=[{"speaker": "CAT", "text": "meow"}],
                    voice_config={"CAT": {"type": "design"}, "DOG": {"type": "design"}},
                )
                manager._current_script_title = lambda: "Project"
                manager.load_chunks = lambda: []
                app_module.ROOT_DIR = temp_root
                app_module.CONFIG_PATH = config_path
                app_module.awaitable_get_voices_sync = lambda: [
                    {"name": "CAT", "suggested_sample_text": "cat text"},
                    {"name": "DOG", "suggested_sample_text": "dog text"},
                ]
                app_module._task_is_current = lambda task_name, run_id: True
                app_module._append_task_log = lambda task_name, run_id, message: logs.append(message)
                app_module._finish_task_run = lambda task_name, run_id: None

                def fake_suggest(speaker):
                    with lock:
                        state["active"] += 1
                        state["max_active"] = max(state["max_active"], state["active"])
                    events.append(("suggest-start", speaker))
                    time.sleep(0.05)
                    with lock:
                        state["active"] -= 1
                        state["completed_suggestions"] += 1
                    events.append(("suggest-done", speaker))
                    return {"voice": f"{speaker} description"}
                manager.suggest_design_sample_text = lambda speaker, chunks: f"{speaker.lower()} sample"
                def _materialize_design_voice(speaker, description, sample_text, force, voice_config):
                    with lock:
                        suggestion_count = state["completed_suggestions"]
                    events.append(("generate-start", speaker, suggestion_count))
                    events.append(("generate", speaker, description, sample_text))
                    updated = json.loads(json.dumps(voice_config))
                    updated[speaker]["ref_audio"] = f"clone_voices/{speaker.lower()}.wav"
                    return {"voice_config": updated}
                manager.materialize_design_voice = _materialize_design_voice
                manager.unload_tts_engine = lambda: (events.append(("unload",)) or True)
                app_module.suggest_voice_description_sync = fake_suggest
                app_module.project_manager = manager

                success = app_module.run_voice_processing_task("run-1")
                self.assertTrue(success)
                self.assertGreaterEqual(state["max_active"], 2)
                first_generate_index = next(i for i, event in enumerate(events) if event[0] == "generate")
                last_suggest_done_index = max(i for i, event in enumerate(events) if event[0] == "suggest-done")
                self.assertGreater(first_generate_index, last_suggest_done_index)
                generate_start_events = [event for event in events if event[0] == "generate-start"]
                self.assertEqual(generate_start_events, [("generate-start", "CAT", 2), ("generate-start", "DOG", 2)])
                generate_events = [event for event in events if event[0] == "generate"]
                self.assertEqual(
                    generate_events,
                    [
                        ("generate", "CAT", "CAT description", "cat text"),
                        ("generate", "DOG", "DOG description", "dog text"),
                    ],
                )
                self.assertEqual(events[-1], ("unload",))
                self.assertTrue(any("Unloaded bulk voice generation model state." in message for message in logs))
            finally:
                app_module.ROOT_DIR = original_root
                app_module.CONFIG_PATH = original_config_path
                app_module.awaitable_get_voices_sync = original_get_voices
                app_module._task_is_current = original_task_current
                app_module._append_task_log = original_append_log
                app_module._finish_task_run = original_finish
                app_module.project_manager = original_pm
                app_module.suggest_voice_description_sync = original_suggest
                _shutdown_manager(manager)

    def test_run_voice_processing_task_respects_inferred_aliases_from_voice_payload(self):
        with tempfile.TemporaryDirectory() as temp_root:
            calls = []
            logs = []

            original_root = app_module.ROOT_DIR
            original_get_voices = app_module.awaitable_get_voices_sync
            original_task_current = app_module._task_is_current
            original_append_log = app_module._append_task_log
            original_finish = app_module._finish_task_run
            original_pm = app_module.project_manager
            manager = None
            try:
                manager = _seed_project_manager(
                    temp_root,
                    entries=[{"speaker": "Queen Novo", "text": "A"}, {"speaker": "Novo", "text": "B"}],
                    voice_config={
                        "Novo": {"type": "design", "alias": ""},
                        "Queen Novo": {"type": "design", "alias": "", "description": "queen desc", "ref_text": "queen sample"},
                    },
                )
                manager._current_script_title = lambda: "Project"
                manager.load_chunks = lambda: []
                manager.suggest_design_sample_text = lambda speaker, chunks: f"{speaker.lower()} sample"
                def _materialize_design_voice(speaker, description, sample_text, force, voice_config):
                    calls.append(speaker)
                    updated = json.loads(json.dumps(voice_config))
                    updated[speaker]["ref_audio"] = f"clone_voices/{speaker.lower().replace(' ', '_')}.wav"
                    return {"voice_config": updated}
                manager.materialize_design_voice = _materialize_design_voice
                manager.unload_tts_engine = lambda: False
                app_module.ROOT_DIR = temp_root
                app_module.awaitable_get_voices_sync = lambda: [
                    {"name": "Queen Novo", "config": {"alias": ""}, "suggested_sample_text": "queen sample", "line_count": 22, "paragraph_count": 0},
                    {"name": "Novo", "config": {"alias": ""}, "suggested_sample_text": "novo sample", "line_count": 13, "paragraph_count": 0},
                ]
                app_module._task_is_current = lambda task_name, run_id: True
                app_module._append_task_log = lambda task_name, run_id, message: logs.append(message)
                app_module._finish_task_run = lambda task_name, run_id: None
                app_module.project_manager = manager
                success = app_module.run_voice_processing_task("run-1")
                self.assertTrue(success)
                self.assertEqual(calls, ["Queen Novo"])
                self.assertTrue(any("Auto-aliased Novo to Queen Novo." in message for message in logs))
                self.assertTrue(any("Skipping Novo: aliased to Queen Novo." in message for message in logs))
                cfg = manager._load_voice_config()
                self.assertEqual(cfg["Novo"]["alias"], "Queen Novo")
            finally:
                app_module.ROOT_DIR = original_root
                app_module.awaitable_get_voices_sync = original_get_voices
                app_module._task_is_current = original_task_current
                app_module._append_task_log = original_append_log
                app_module._finish_task_run = original_finish
                app_module.project_manager = original_pm
                _shutdown_manager(manager)

    def test_run_voice_processing_task_skips_zero_line_speakers(self):
        with tempfile.TemporaryDirectory() as temp_root:
            calls = []
            logs = []

            original_root = app_module.ROOT_DIR
            original_get_voices = app_module.awaitable_get_voices_sync
            original_task_current = app_module._task_is_current
            original_append_log = app_module._append_task_log
            original_finish = app_module._finish_task_run
            original_pm = app_module.project_manager
            manager = None
            try:
                manager = _seed_project_manager(
                    temp_root,
                    entries=[{"speaker": "DOG", "text": "woof"}],
                    voice_config={
                        "DOG": {"type": "design", "description": "dog desc", "ref_text": "dog sample"},
                        "Manual": {"type": "design", "description": "manual desc", "ref_text": "manual sample", "user_created": True},
                    },
                )
                manager._current_script_title = lambda: "Project"
                manager.load_chunks = lambda: []
                manager.suggest_design_sample_text = lambda speaker, chunks: f"{speaker.lower()} sample"

                def _materialize_design_voice(speaker, description, sample_text, force, voice_config):
                    calls.append(speaker)
                    updated = json.loads(json.dumps(voice_config))
                    updated.setdefault(speaker, {})["ref_audio"] = f"clone_voices/{speaker.lower()}.wav"
                    return {"voice_config": updated}

                manager.materialize_design_voice = _materialize_design_voice
                manager.unload_tts_engine = lambda: False
                app_module.ROOT_DIR = temp_root
                app_module.awaitable_get_voices_sync = lambda: [
                    {"name": "Manual", "config": {"type": "design"}, "suggested_sample_text": "manual sample", "line_count": 0, "paragraph_count": 0},
                    {"name": "DOG", "config": {"type": "design"}, "suggested_sample_text": "dog sample", "line_count": 7, "paragraph_count": 0},
                ]
                app_module._task_is_current = lambda task_name, run_id: True
                app_module._append_task_log = lambda task_name, run_id, message: logs.append(message)
                app_module._finish_task_run = lambda task_name, run_id: None
                app_module.project_manager = manager

                success = app_module.run_voice_processing_task("run-1")
                self.assertTrue(success)
                self.assertEqual(calls, ["DOG"])
                self.assertTrue(any("Skipping Manual: no lines detected." in message for message in logs))
            finally:
                app_module.ROOT_DIR = original_root
                app_module.awaitable_get_voices_sync = original_get_voices
                app_module._task_is_current = original_task_current
                app_module._append_task_log = original_append_log
                app_module._finish_task_run = original_finish
                app_module.project_manager = original_pm
                _shutdown_manager(manager)

    def test_bulk_voice_description_suggestions_respect_llm_workers(self):
        with tempfile.TemporaryDirectory() as temp_root:
            config_path = os.path.join(temp_root, "config.json")
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump({"llm": {"llm_workers": 2}}, f, indent=2, ensure_ascii=False)

            state = {"active": 0, "max_active": 0}
            lock = threading.Lock()

            original_config_path = app_module.CONFIG_PATH
            original_suggest = app_module.suggest_voice_description_sync
            try:
                app_module.CONFIG_PATH = config_path

                def fake_suggest(speaker):
                    with lock:
                        state["active"] += 1
                        state["max_active"] = max(state["max_active"], state["active"])
                    time.sleep(0.05)
                    with lock:
                        state["active"] -= 1
                    return {"speaker": speaker, "voice": f"{speaker} voice"}

                app_module.suggest_voice_description_sync = fake_suggest

                result = app_module.suggest_voice_descriptions_batch_sync(["CAT", "DOG", "MOUSE"])
                self.assertEqual([item["speaker"] for item in result["results"]], ["CAT", "DOG", "MOUSE"])
                self.assertEqual(result["workers"], 2)
                self.assertEqual(result["failures"], [])
                self.assertGreaterEqual(state["max_active"], 2)
            finally:
                app_module.CONFIG_PATH = original_config_path
                app_module.suggest_voice_description_sync = original_suggest

    def test_unload_bulk_voice_generation_endpoint_resets_engine(self):
        original_pm = app_module.project_manager
        try:
            class FakeProjectManager:
                def unload_tts_engine(self):
                    return True

            app_module.project_manager = FakeProjectManager()
            result = asyncio.run(app_module.unload_bulk_voice_generation())
            self.assertEqual(result["status"], "unloaded")
            self.assertTrue(result["unloaded"])
        finally:
            app_module.project_manager = original_pm

    def test_get_voices_deduplicates_case_variants(self):
        with tempfile.TemporaryDirectory() as temp_root:
            original_root = app_module.ROOT_DIR
            original_pm = app_module.project_manager
            manager = None
            try:
                manager = _seed_project_manager(
                    temp_root,
                    entries=[
                        {"speaker": "Narrator", "text": "A"},
                        {"speaker": "NARRATOR", "text": "B"},
                    ],
                    voice_config={"narrator": {"type": "design", "description": "x"}},
                )
                manager.suggest_design_sample_text = lambda speaker, chunks: ""
                app_module.ROOT_DIR = temp_root
                app_module.project_manager = manager

                voices = asyncio.run(app_module.get_voices())
                self.assertEqual(len(voices), 1)
                self.assertEqual(voices[0]["name"], "NARRATOR")
                self.assertEqual(voices[0]["line_count"], 2)
                self.assertEqual((voices[0]["config"] or {}).get("description"), "x")
            finally:
                app_module.ROOT_DIR = original_root
                app_module.project_manager = original_pm
                _shutdown_manager(manager)

    def test_get_voices_does_not_infer_contained_name_alias_on_page_load(self):
        with tempfile.TemporaryDirectory() as temp_root:
            original_root = app_module.ROOT_DIR
            original_pm = app_module.project_manager
            manager = None
            try:
                manager = _seed_project_manager(
                    temp_root,
                    entries=([{"speaker": "Queen Novo", "text": "A"}] * 22)
                    + ([{"speaker": "Novo", "text": "B"}] * 13),
                    voice_config={
                        "Novo": {"type": "design", "alias": ""},
                        "Queen Novo": {"type": "design", "alias": ""},
                    },
                )
                manager.suggest_design_sample_text = lambda speaker, chunks: ""
                app_module.ROOT_DIR = temp_root
                app_module.project_manager = manager

                voices = asyncio.run(app_module.get_voices())
                voices_by_name = {voice["name"]: voice for voice in voices}
                self.assertEqual(voices_by_name["Novo"]["config"].get("alias", ""), "")
                self.assertEqual(voices_by_name["Queen Novo"]["config"].get("alias", ""), "")
            finally:
                app_module.ROOT_DIR = original_root
                app_module.project_manager = original_pm
                _shutdown_manager(manager)

    def test_get_voices_does_not_alias_word_fragments(self):
        with tempfile.TemporaryDirectory() as temp_root:
            original_root = app_module.ROOT_DIR
            original_pm = app_module.project_manager
            manager = None
            try:
                manager = _seed_project_manager(
                    temp_root,
                    entries=([{"speaker": "The Captain", "text": "A"}] * 10)
                    + ([{"speaker": "He", "text": "B"}] * 3),
                    voice_config={
                        "He": {"type": "design", "alias": ""},
                        "The Captain": {"type": "design", "alias": ""},
                    },
                )
                manager.suggest_design_sample_text = lambda speaker, chunks: ""
                app_module.ROOT_DIR = temp_root
                app_module.project_manager = manager

                voices = asyncio.run(app_module.get_voices())
                voices_by_name = {voice["name"]: voice for voice in voices}
                self.assertEqual(voices_by_name["He"]["config"].get("alias", ""), "")
                self.assertEqual(voices_by_name["The Captain"]["config"].get("alias", ""), "")
            finally:
                app_module.ROOT_DIR = original_root
                app_module.project_manager = original_pm
                _shutdown_manager(manager)

    def test_clear_uploaded_voices_clears_only_current_script_assets_and_text(self):
        with tempfile.TemporaryDirectory() as temp_root:
            clone_dir = os.path.join(temp_root, "clone_voices")
            designed_dir = os.path.join(temp_root, "designed_voices")
            os.makedirs(clone_dir, exist_ok=True)
            os.makedirs(designed_dir, exist_ok=True)

            for path in (
                os.path.join(clone_dir, "booka_cat.wav"),
                os.path.join(clone_dir, "bookb_dog.wav"),
                os.path.join(designed_dir, "booka_cat_design.wav"),
                os.path.join(designed_dir, "bookb_dog_design.wav"),
            ):
                with open(path, "wb") as f:
                    f.write(b"wav")

            with open(os.path.join(clone_dir, "manifest.json"), "w", encoding="utf-8") as f:
                json.dump(
                    [
                        {"script_title": "BookA", "speaker": "CAT", "filename": "booka_cat.wav", "generated": True},
                        {"script_title": "BookB", "speaker": "DOG", "filename": "bookb_dog.wav", "generated": True},
                    ],
                    f,
                    indent=2,
                    ensure_ascii=False,
                )
            with open(os.path.join(designed_dir, "manifest.json"), "w", encoding="utf-8") as f:
                json.dump(
                    [
                        {"script_title": "BookA", "speaker": "CAT", "filename": "booka_cat_design.wav"},
                        {"script_title": "BookB", "speaker": "DOG", "filename": "bookb_dog_design.wav"},
                    ],
                    f,
                    indent=2,
                    ensure_ascii=False,
                )

            original_root = app_module.ROOT_DIR
            original_clone_dir = app_module.CLONE_VOICES_DIR
            original_designed_dir = app_module.DESIGNED_VOICES_DIR
            original_clone_manifest = app_module.CLONE_VOICES_MANIFEST
            original_designed_manifest = app_module.DESIGNED_VOICES_MANIFEST
            original_task_running = app_module._any_project_task_running
            original_pm = app_module.project_manager
            manager = None
            try:
                manager = _seed_project_manager(
                    temp_root,
                    entries=[{"speaker": "CAT", "text": "meow"}, {"speaker": "NARRATOR", "text": "narrate"}],
                    voice_config={
                        "CAT": {
                            "type": "design",
                            "ref_audio": "clone_voices/booka_cat.wav",
                            "ref_text": "cat sample",
                            "generated_ref_text": "cat generated sample",
                        },
                        "DOG": {
                            "type": "design",
                            "ref_audio": "clone_voices/bookb_dog.wav",
                            "ref_text": "dog sample",
                            "generated_ref_text": "dog generated sample",
                        },
                    },
                )
                manager._current_script_title = lambda: "BookA"
                app_module.ROOT_DIR = temp_root
                app_module.CLONE_VOICES_DIR = clone_dir
                app_module.DESIGNED_VOICES_DIR = designed_dir
                app_module.CLONE_VOICES_MANIFEST = os.path.join(clone_dir, "manifest.json")
                app_module.DESIGNED_VOICES_MANIFEST = os.path.join(designed_dir, "manifest.json")
                app_module._any_project_task_running = lambda: None
                app_module.project_manager = manager

                result = asyncio.run(app_module.clear_uploaded_voices_for_current_script())
                self.assertEqual(result["status"], "ok")
                self.assertEqual(result["script_title"], "BookA")

                with open(os.path.join(clone_dir, "manifest.json"), "r", encoding="utf-8") as f:
                    clone_manifest = json.load(f)
                with open(os.path.join(designed_dir, "manifest.json"), "r", encoding="utf-8") as f:
                    designed_manifest = json.load(f)
                self.assertEqual(len(clone_manifest), 1)
                self.assertEqual(clone_manifest[0]["script_title"], "BookB")
                self.assertEqual(len(designed_manifest), 1)
                self.assertEqual(designed_manifest[0]["script_title"], "BookB")

                self.assertFalse(os.path.exists(os.path.join(clone_dir, "booka_cat.wav")))
                self.assertFalse(os.path.exists(os.path.join(designed_dir, "booka_cat_design.wav")))
                self.assertTrue(os.path.exists(os.path.join(clone_dir, "bookb_dog.wav")))
                self.assertTrue(os.path.exists(os.path.join(designed_dir, "bookb_dog_design.wav")))

                updated_cfg = manager._load_voice_config()
                self.assertEqual(updated_cfg["CAT"].get("ref_audio"), "")
                self.assertEqual(updated_cfg["CAT"].get("ref_text"), "")
                self.assertEqual(updated_cfg["CAT"].get("generated_ref_text"), "")
                self.assertEqual(updated_cfg["DOG"].get("ref_audio"), "clone_voices/bookb_dog.wav")
                self.assertEqual(updated_cfg["DOG"].get("ref_text"), "dog sample")
                self.assertEqual(updated_cfg["DOG"].get("generated_ref_text"), "dog generated sample")
            finally:
                app_module.ROOT_DIR = original_root
                app_module.CLONE_VOICES_DIR = original_clone_dir
                app_module.DESIGNED_VOICES_DIR = original_designed_dir
                app_module.CLONE_VOICES_MANIFEST = original_clone_manifest
                app_module.DESIGNED_VOICES_MANIFEST = original_designed_manifest
                app_module._any_project_task_running = original_task_running
                app_module.project_manager = original_pm
                _shutdown_manager(manager)

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
