import copy
import importlib.util
import os
import sqlite3
import subprocess
import sys
import tempfile
import unittest
import asyncio
import json
from pathlib import Path
from fastapi import HTTPException
from fastapi import BackgroundTasks
from project import ProjectManager
from runtime_layout import RuntimeLayout

APP_DIR = Path(__file__).resolve().parents[2]
MODULE_PATH = APP_DIR / "app.py"
SPEC = importlib.util.spec_from_file_location(
    "threadspeak_app_module_workflow_entrypoints",
    str(MODULE_PATH),
)
app_module = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(app_module)


def _quiesce_workflows():
    with app_module.processing_workflow_lock:
        app_module.process_state["processing_workflow"]["running"] = False
        app_module.process_state["processing_workflow"]["paused"] = False
        app_module.process_state["processing_workflow"]["pause_requested"] = False
    with app_module.new_mode_workflow_lock:
        app_module.process_state["new_mode_workflow"]["running"] = False
        app_module.process_state["new_mode_workflow"]["paused"] = False
        app_module.process_state["new_mode_workflow"]["pause_requested"] = False

    for stage in ("process_paragraphs", "assign_dialogue", "extract_temperament", "create_script", "voices", "proofread"):
        try:
            app_module._terminate_task_process_if_running(stage)
        except Exception:
            pass


_quiesce_workflows()


def _ensure_project_root(root):
    for dirname in ("app", "uploads", "voicelines", "clone_voices", "designed_voices", "workflow", "db", "repair", "exports"):
        os.makedirs(os.path.join(root, dirname), exist_ok=True)


def _seed_db_project(root, *, entries=None, chunks=None, voice_config=None):
    _ensure_project_root(root)
    manager = ProjectManager(root)
    try:
        if entries is not None:
            manager.script_store.replace_script_document(
                entries=entries,
                dictionary=[],
                sanity_cache={"phrase_decisions": {}},
                reason="test_seed_script",
                rebuild_chunks=True,
                wait=True,
            )
        if chunks is not None:
            manager.save_chunks(chunks)
        if voice_config is not None:
            manager._save_voice_config(voice_config)
        return manager
    except Exception:
        manager.shutdown_script_store(flush=True)
        raise


class WorkflowEntrypointAccessibilityTests(unittest.TestCase):
    def setUp(self):
        _quiesce_workflows()
        self._patches = {}
        with app_module.new_mode_workflow_lock:
            self._backup_new_mode_workflow = copy.deepcopy(app_module.process_state["new_mode_workflow"])
        with app_module.processing_workflow_lock:
            self._backup_processing_workflow = copy.deepcopy(app_module.process_state["processing_workflow"])
        with app_module.audio_queue_lock:
            self._backup_audio_queue = copy.deepcopy(app_module.audio_queue)
            self._backup_audio_current_job = copy.deepcopy(app_module.audio_current_job)

    def tearDown(self):
        for name, original in self._patches.items():
            setattr(app_module, name, original)
        with app_module.new_mode_workflow_lock:
            app_module.process_state["new_mode_workflow"] = self._backup_new_mode_workflow
        with app_module.processing_workflow_lock:
            app_module.process_state["processing_workflow"] = self._backup_processing_workflow
        with app_module.audio_queue_lock:
            app_module.audio_queue[:] = self._backup_audio_queue
            app_module.audio_current_job = self._backup_audio_current_job

    def _patch(self, name, value):
        if name not in self._patches:
            self._patches[name] = getattr(app_module, name)
        setattr(app_module, name, value)

    def test_legacy_processing_dispatch_entrypoints_are_accessible(self):
        self._patch("_run_processing_script_stage", lambda: True)
        self._patch("_run_processing_review_stage", lambda: True)
        self._patch("_run_processing_sanity_stage", lambda: True)
        self._patch("_run_processing_repair_stage", lambda: True)
        self._patch("_run_processing_voices_stage", lambda: True)
        self._patch("_run_processing_audio_stage", lambda: True)
        self._patch("_processing_workflow_is_pause_requested", lambda: False)

        for stage in ("script", "review", "sanity", "repair", "voices", "audio"):
            app_module._execute_processing_workflow_stage(stage)

    def test_new_mode_stage_entrypoints_are_accessible(self):
        # Keep task tracking deterministic while bypassing heavyweight subprocess work.
        self._patch("_start_task_run", lambda _task_name: "run-1")
        self._patch("run_voice_processing_task", lambda *args, **kwargs: True)
        self._patch("_new_mode_workflow_is_pause_requested", lambda: False)

        # Render-audio stage should short-circuit cleanly when there is no pending work.
        self._patch("_workflow_pending_audio_indices", lambda: [])
        self._patch("_refresh_audio_process_state_locked", lambda *args, **kwargs: None)
        self._patch(
            "_autosave_current_script_for_workflow",
            lambda **kwargs: {"name": "autosave-smoke", "overwrote": False},
        )

        with app_module.new_mode_workflow_lock:
            app_module.process_state["new_mode_workflow"] = app_module._new_mode_workflow_initial_state() | {
                "running": True,
                "options": {"process_voices": True, "generate_audio": True, "full_cast": True},
            }

        with app_module.audio_queue_lock:
            app_module.audio_queue.clear()
            app_module.audio_current_job = None

        stages = ("process_voices", "render_audio")
        for stage in stages:
            app_module._run_new_mode_workflow_stage(stage)

        # Autosave hooks should resolve and execute without NameError.
        app_module._maybe_autosave_after_new_mode_stage("create_script")
        app_module._maybe_autosave_after_new_mode_stage("process_voices")

    def test_new_mode_create_script_stage_passes_saved_script_max_length(self):
        captured = {}

        def fake_run_process(cmd, stage_name, run_id, relay_fn=None):
            captured["cmd"] = cmd
            captured["stage_name"] = stage_name
            captured["run_id"] = run_id
            return True

        self._patch("_start_task_run", lambda _task_name: "run-1")
        self._patch("run_process", fake_run_process)
        self._patch("_new_mode_workflow_is_pause_requested", lambda: False)

        with tempfile.TemporaryDirectory() as temp_root:
            config_path = os.path.join(temp_root, "config.json")
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump({"tts": {"script_max_length": 250}}, f)

            original_root = app_module.ROOT_DIR
            original_config = app_module.CONFIG_PATH
            try:
                app_module.ROOT_DIR = temp_root
                app_module.CONFIG_PATH = config_path

                app_module._run_new_mode_workflow_stage("create_script")
            finally:
                app_module.ROOT_DIR = original_root
                app_module.CONFIG_PATH = original_config

        self.assertEqual(captured["stage_name"], "create_script")
        self.assertEqual(captured["run_id"], "run-1")
        self.assertIn("--max-length", captured["cmd"])
        self.assertEqual(
            captured["cmd"][captured["cmd"].index("--max-length") + 1],
            "250",
        )

    def test_new_mode_assign_dialogue_stage_uses_narrated_flag_when_full_cast_disabled(self):
        captured = {}

        def fake_run_process(cmd, stage_name, run_id, relay_fn=None):
            captured["cmd"] = cmd
            captured["stage_name"] = stage_name
            captured["run_id"] = run_id
            return True

        self._patch("_start_task_run", lambda _task_name: "run-1")
        self._patch("run_process", fake_run_process)
        self._patch("_new_mode_workflow_is_pause_requested", lambda: False)

        with app_module.new_mode_workflow_lock:
            app_module.process_state["new_mode_workflow"] = app_module._new_mode_workflow_initial_state() | {
                "running": True,
                "options": {"process_voices": True, "generate_audio": False, "full_cast": False},
            }

        app_module._run_new_mode_workflow_stage("assign_dialogue")

        self.assertEqual(captured["stage_name"], "assign_dialogue")
        self.assertEqual(captured["run_id"], "run-1")
        self.assertIn("--narrated", captured["cmd"])

    def test_new_mode_assign_dialogue_stage_omits_narrated_flag_when_full_cast_enabled(self):
        captured = {}

        def fake_run_process(cmd, stage_name, run_id, relay_fn=None):
            captured["cmd"] = cmd
            captured["stage_name"] = stage_name
            captured["run_id"] = run_id
            return True

        self._patch("_start_task_run", lambda _task_name: "run-1")
        self._patch("run_process", fake_run_process)
        self._patch("_new_mode_workflow_is_pause_requested", lambda: False)

        with app_module.new_mode_workflow_lock:
            app_module.process_state["new_mode_workflow"] = app_module._new_mode_workflow_initial_state() | {
                "running": True,
                "options": {"process_voices": True, "generate_audio": False, "full_cast": True},
            }

        app_module._run_new_mode_workflow_stage("assign_dialogue")

        self.assertEqual(captured["stage_name"], "assign_dialogue")
        self.assertEqual(captured["run_id"], "run-1")
        self.assertNotIn("--narrated", captured["cmd"])

    def test_manual_stage_start_is_blocked_while_new_mode_workflow_active(self):
        with app_module.new_mode_workflow_lock:
            app_module.process_state["new_mode_workflow"] = app_module._new_mode_workflow_initial_state() | {
                "running": True,
                "paused": False,
            }

        with self.assertRaises(HTTPException) as ctx:
            asyncio.run(app_module.start_create_script(BackgroundTasks()))
        self.assertEqual(ctx.exception.status_code, 409)

    def test_reset_new_mode_can_preserve_voice_config_while_clearing_script_and_clips(self):
        with tempfile.TemporaryDirectory() as temp_root:
            voicelines_dir = os.path.join(temp_root, "voicelines")
            os.makedirs(voicelines_dir, exist_ok=True)
            os.makedirs(os.path.join(voicelines_dir, "discarded"), exist_ok=True)
            with open(os.path.join(voicelines_dir, "clip.wav"), "w", encoding="utf-8") as f:
                f.write("audio")
            with open(os.path.join(voicelines_dir, "discarded", "old.wav"), "w", encoding="utf-8") as f:
                f.write("audio")

            original_root = app_module.ROOT_DIR
            original_voicelines = app_module.VOICELINES_DIR
            original_pm = app_module.project_manager
            manager = None
            try:
                manager = _seed_db_project(
                    temp_root,
                    entries=[{"speaker": "Narrator", "text": "hello"}],
                    voice_config={"Narrator": {"alias": "kept"}},
                )
                app_module.ROOT_DIR = temp_root
                app_module.VOICELINES_DIR = voicelines_dir
                app_module.project_manager = manager

                result = asyncio.run(
                    app_module.reset_new_mode(app_module.ResetNewModeRequest(preserve_voices=True))
                )
            finally:
                app_module.ROOT_DIR = original_root
                app_module.VOICELINES_DIR = original_voicelines
                app_module.project_manager = original_pm
                if manager is not None:
                    manager.shutdown_script_store(flush=True)

            self.assertEqual(result["status"], "reset")
            self.assertTrue(result["preserved_voices"])
            self.assertFalse(bool(manager.script_store.has_script_entries()))
            self.assertTrue(bool(manager._load_voice_config()))
            self.assertEqual(os.listdir(voicelines_dir), [])

    def test_reset_new_mode_deletes_voice_config_when_not_preserved(self):
        with tempfile.TemporaryDirectory() as temp_root:
            voicelines_dir = os.path.join(temp_root, "voicelines")
            os.makedirs(voicelines_dir, exist_ok=True)

            with open(os.path.join(voicelines_dir, "clip.wav"), "w", encoding="utf-8") as f:
                f.write("audio")

            original_root = app_module.ROOT_DIR
            original_voicelines = app_module.VOICELINES_DIR
            original_pm = app_module.project_manager
            manager = None
            try:
                manager = _seed_db_project(
                    temp_root,
                    entries=[{"speaker": "Narrator", "text": "hello"}],
                    voice_config={"Narrator": {"alias": "kept"}},
                )
                app_module.ROOT_DIR = temp_root
                app_module.VOICELINES_DIR = voicelines_dir
                app_module.project_manager = manager

                result = asyncio.run(app_module.reset_new_mode())
            finally:
                app_module.ROOT_DIR = original_root
                app_module.VOICELINES_DIR = original_voicelines
                app_module.project_manager = original_pm
                if manager is not None:
                    manager.shutdown_script_store(flush=True)

            self.assertEqual(result["status"], "reset")
            self.assertFalse(result["preserved_voices"])
            self.assertFalse(bool(manager._load_voice_config()))
            self.assertEqual(os.listdir(voicelines_dir), [])

    def test_reset_new_mode_refuses_default_runtime_voicelines_under_pytest(self):
        with tempfile.TemporaryDirectory() as temp_root:
            original_root = app_module.ROOT_DIR
            original_voicelines = app_module.VOICELINES_DIR
            try:
                app_module.ROOT_DIR = temp_root
                with self.assertRaises(RuntimeError) as ctx:
                    asyncio.run(app_module.reset_new_mode())
                self.assertIn("default runtime project", str(ctx.exception))
            finally:
                app_module.ROOT_DIR = original_root
                app_module.VOICELINES_DIR = original_voicelines

    def test_assert_test_safe_runtime_target_allows_default_runtime_for_temp_clone_layout(self):
        with tempfile.TemporaryDirectory(prefix="threadspeak_temp_clone_layout_") as temp_repo_root:
            app_dir = os.path.join(temp_repo_root, "app")
            os.makedirs(app_dir, exist_ok=True)
            temp_layout = RuntimeLayout.from_app_dir(app_dir)

            original_layout = app_module.LAYOUT
            try:
                app_module.LAYOUT = temp_layout
                app_module._assert_test_safe_runtime_target(
                    "reset_project",
                    ROOT_DIR=temp_layout.project_dir,
                    VOICELINES_DIR=temp_layout.voicelines_dir,
                    UPLOADS_DIR=temp_layout.uploads_dir,
                )
            finally:
                app_module.LAYOUT = original_layout

    def test_reset_project_clears_db_runtime_artifacts_and_reinitializes_store(self):
        with tempfile.TemporaryDirectory() as temp_root:
            voicelines_dir = os.path.join(temp_root, "voicelines")
            uploads_dir = os.path.join(temp_root, "uploads")
            clone_dir = os.path.join(temp_root, "clone_voices")
            designed_dir = os.path.join(temp_root, "designed_voices")
            scripts_dir = os.path.join(temp_root, "scripts")
            saved_projects_dir = os.path.join(temp_root, "saved_projects")
            os.makedirs(voicelines_dir, exist_ok=True)
            os.makedirs(uploads_dir, exist_ok=True)
            os.makedirs(clone_dir, exist_ok=True)
            os.makedirs(designed_dir, exist_ok=True)
            os.makedirs(scripts_dir, exist_ok=True)
            os.makedirs(saved_projects_dir, exist_ok=True)

            state_path = os.path.join(temp_root, "state.json")
            chunks_db_path = os.path.join(temp_root, "chunks.sqlite3")
            queue_log_path = os.path.join(temp_root, "chunks.queue.log")
            for path, payload in (
                (state_path, {"input_file_path": os.path.join(uploads_dir, "story.txt"), "loaded_script_name": "demo"}),
                (os.path.join(temp_root, "audio_queue_state.json"), {"queue": []}),
                (os.path.join(temp_root, "processing_workflow_state.json"), {"running": True}),
                (os.path.join(temp_root, "new_mode_workflow_state.json"), {"running": True}),
            ):
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(payload, f)

            with open(os.path.join(uploads_dir, "story.txt"), "w", encoding="utf-8") as f:
                f.write("source")
            with open(os.path.join(voicelines_dir, "clip.wav"), "w", encoding="utf-8") as f:
                f.write("audio")
            with open(os.path.join(clone_dir, "voice.wav"), "w", encoding="utf-8") as f:
                f.write("audio")
            with open(os.path.join(designed_dir, "voice.wav"), "w", encoding="utf-8") as f:
                f.write("audio")
            with open(os.path.join(scripts_dir, "saved.sqlite3"), "w", encoding="utf-8") as f:
                f.write("snapshot")
            with open(os.path.join(saved_projects_dir, "saved.zip"), "w", encoding="utf-8") as f:
                f.write("archive")
            with open(queue_log_path, "w", encoding="utf-8") as f:
                f.write("queued")
            conn = sqlite3.connect(chunks_db_path)
            try:
                conn.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, value TEXT)")
                conn.execute("INSERT INTO t(value) VALUES ('ok')")
                conn.commit()
            finally:
                conn.close()
            with open(f"{chunks_db_path}-wal", "w", encoding="utf-8") as f:
                f.write("wal")
            with open(f"{chunks_db_path}-shm", "w", encoding="utf-8") as f:
                f.write("shm")

            original_root = app_module.ROOT_DIR
            original_voicelines = app_module.VOICELINES_DIR
            original_uploads = app_module.UPLOADS_DIR
            original_clone = app_module.CLONE_VOICES_DIR
            original_designed = app_module.DESIGNED_VOICES_DIR
            original_processing_path = app_module.PROCESSING_WORKFLOW_STATE_PATH
            original_new_mode_path = app_module.NEW_MODE_WORKFLOW_STATE_PATH
            original_audio_queue_path = app_module.AUDIO_QUEUE_STATE_PATH
            original_manager = app_module.project_manager
            original_audio_state = copy.deepcopy(app_module.process_state["audio"])
            try:
                class ResetStubManager:
                    def __init__(self, root_dir):
                        self.root_dir = root_dir
                        self.chunks_db_path = os.path.join(root_dir, "chunks.sqlite3")
                        self.chunks_queue_log_path = os.path.join(root_dir, "chunks.queue.log")
                        self._transcription_cache_lock = app_module.threading.Lock()
                        self._transcription_cache = {"stale": True}
                        self.engine = object()
                        self.asr_engine = object()
                        self.reload_calls = 0

                    def reload_script_store(self):
                        self.reload_calls += 1
                        with open(self.chunks_queue_log_path, "w", encoding="utf-8"):
                            pass
                        with sqlite3.connect(self.chunks_db_path) as conn:
                            conn.execute("CREATE TABLE IF NOT EXISTS chunks (id INTEGER PRIMARY KEY)")
                            conn.commit()

                    def reset_generating_chunks(self, indices=None, generation_token=None, target_status="pending"):
                        return len(list(indices or []))

                    def clear_audio_finalize_tasks(self, generation_token=None):
                        return 0

                    def unregister_audio_finalization_listener(self, generation_token):
                        return None

                stub_manager = ResetStubManager(temp_root)

                app_module.ROOT_DIR = temp_root
                app_module.VOICELINES_DIR = voicelines_dir
                app_module.UPLOADS_DIR = uploads_dir
                app_module.CLONE_VOICES_DIR = clone_dir
                app_module.DESIGNED_VOICES_DIR = designed_dir
                app_module.PROCESSING_WORKFLOW_STATE_PATH = os.path.join(temp_root, "processing_workflow_state.json")
                app_module.NEW_MODE_WORKFLOW_STATE_PATH = os.path.join(temp_root, "new_mode_workflow_state.json")
                app_module.AUDIO_QUEUE_STATE_PATH = os.path.join(temp_root, "audio_queue_state.json")
                app_module.project_manager = stub_manager

                app_module.process_state["audio"]["running"] = True
                app_module.process_state["audio"]["queue"] = [{"uid": "a"}]
                app_module.process_state["audio"]["current_job"] = {"uid": "a"}
                app_module.process_state["audio"]["recent_jobs"] = [{"uid": "a"}]
                app_module.process_state["audio"]["logs"] = ["busy"]

                result = asyncio.run(app_module.reset_project())
            finally:
                app_module.ROOT_DIR = original_root
                app_module.VOICELINES_DIR = original_voicelines
                app_module.UPLOADS_DIR = original_uploads
                app_module.CLONE_VOICES_DIR = original_clone
                app_module.DESIGNED_VOICES_DIR = original_designed
                app_module.PROCESSING_WORKFLOW_STATE_PATH = original_processing_path
                app_module.NEW_MODE_WORKFLOW_STATE_PATH = original_new_mode_path
                app_module.AUDIO_QUEUE_STATE_PATH = original_audio_queue_path
                app_module.project_manager = original_manager
                app_module.process_state["audio"].clear()
                app_module.process_state["audio"].update(original_audio_state)

            self.assertEqual(result["status"], "reset")
            self.assertTrue(os.path.exists(chunks_db_path))
            self.assertTrue(os.path.exists(queue_log_path))
            self.assertFalse(os.path.exists(f"{chunks_db_path}-wal"))
            self.assertFalse(os.path.exists(f"{chunks_db_path}-shm"))
            self.assertEqual(os.listdir(voicelines_dir), [])
            self.assertEqual(os.listdir(uploads_dir), [])
            self.assertEqual(sorted(os.listdir(clone_dir)), ["voice.wav"])
            self.assertEqual(sorted(os.listdir(designed_dir)), ["voice.wav"])
            self.assertEqual(sorted(os.listdir(scripts_dir)), ["saved.sqlite3"])
            self.assertEqual(sorted(os.listdir(saved_projects_dir)), ["saved.zip"])
            with open(state_path, "r", encoding="utf-8") as f:
                state = json.load(f)
            self.assertEqual(state, {"render_prep_complete": False})
            self.assertFalse(app_module.process_state["audio"]["running"])
            self.assertEqual(app_module.process_state["audio"]["queue"], [])
            self.assertIsNone(app_module.process_state["audio"]["current_job"])
            self.assertEqual(stub_manager.reload_calls, 1)
            self.assertIsNone(stub_manager._transcription_cache)
            self.assertIsNone(stub_manager.engine)
            self.assertIsNone(stub_manager.asr_engine)

    def test_script_info_reports_voice_state_without_script_file(self):
        with tempfile.TemporaryDirectory() as temp_root:
            voicelines_dir = os.path.join(temp_root, "voicelines")
            os.makedirs(voicelines_dir, exist_ok=True)
            with open(os.path.join(voicelines_dir, "clip.wav"), "w", encoding="utf-8") as f:
                f.write("audio")

            original_root = app_module.ROOT_DIR
            original_voicelines = app_module.VOICELINES_DIR
            original_pm = app_module.project_manager
            manager = None
            try:
                app_module.ROOT_DIR = temp_root
                app_module.VOICELINES_DIR = voicelines_dir
                manager = app_module.ProjectManager(temp_root)
                manager._save_voice_config({"Narrator": {}, "Alice": {}})
                app_module.project_manager = manager

                result = asyncio.run(app_module.get_script_info())
            finally:
                if manager is not None:
                    manager.shutdown_script_store(flush=True)
                app_module.ROOT_DIR = original_root
                app_module.VOICELINES_DIR = original_voicelines
                app_module.project_manager = original_pm

            self.assertEqual(result["entry_count"], 0)
            self.assertTrue(result["has_voice_config"])
            self.assertEqual(result["voice_count"], 2)
            self.assertTrue(result["has_voicelines"])

    def test_standalone_create_script_marks_new_mode_stage_complete(self):
        captured = {"calls": 0}

        def fake_run_create_script_task(run_id):
            captured["calls"] += 1
            captured["args"] = (run_id,)

        self._patch("_run_create_script_task", fake_run_create_script_task)

        with tempfile.TemporaryDirectory() as temp_root:
            state_path = os.path.join(temp_root, "state.json")
            with open(state_path, "w", encoding="utf-8") as f:
                json.dump({}, f)

            original_root = app_module.ROOT_DIR
            try:
                app_module.ROOT_DIR = temp_root
                with app_module.new_mode_workflow_lock:
                    app_module.process_state["new_mode_workflow"] = app_module._new_mode_workflow_initial_state()

                app_module._run_create_script_task_with_new_mode_state("run-1")

                markers = app_module._load_new_mode_stage_markers()
                completed = app_module.process_state["new_mode_workflow"]["completed_stages"]
            finally:
                app_module.ROOT_DIR = original_root

        self.assertEqual(captured["calls"], 1)
        self.assertIn("create_script", markers)
        self.assertIn("create_script", completed)

    def test_restore_new_mode_workflow_ignores_stale_completed_list_when_markers_show_complete(self):
        with tempfile.TemporaryDirectory() as temp_root:
            with open(os.path.join(temp_root, "state.json"), "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "new_mode_stage_markers": {
                            "process_paragraphs": {"completed_at": 1},
                            "assign_dialogue": {"completed_at": 2},
                            "extract_temperament": {"completed_at": 3},
                            "create_script": {"completed_at": 4},
                        }
                    },
                    f,
                )
            workflow_state_path = os.path.join(temp_root, "new_mode_workflow_state.json")
            with open(workflow_state_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "running": True,
                        "paused": False,
                        "options": {"process_voices": False, "generate_audio": False},
                        "completed_stages": ["process_paragraphs"],
                    },
                    f,
                )

            original_root = app_module.ROOT_DIR
            original_new_mode_path = app_module.NEW_MODE_WORKFLOW_STATE_PATH
            original_starter = app_module._start_new_mode_workflow_thread_locked
            started = {"count": 0}
            try:
                app_module.ROOT_DIR = temp_root
                app_module.NEW_MODE_WORKFLOW_STATE_PATH = workflow_state_path
                app_module._start_new_mode_workflow_thread_locked = lambda: started.__setitem__("count", started["count"] + 1)
                app_module._restore_new_mode_workflow_state()
                state = app_module.process_state["new_mode_workflow"]
                self.assertEqual(started["count"], 0)
                self.assertFalse(state["running"])
                self.assertIn("create_script", state.get("completed_stages", []))
            finally:
                app_module.ROOT_DIR = original_root
                app_module.NEW_MODE_WORKFLOW_STATE_PATH = original_new_mode_path
                app_module._start_new_mode_workflow_thread_locked = original_starter

    def test_restore_new_mode_workflow_forces_complete_when_script_project_exists(self):
        with tempfile.TemporaryDirectory() as temp_root:
            with open(os.path.join(temp_root, "state.json"), "w", encoding="utf-8") as f:
                json.dump({}, f)
            manager = _seed_db_project(
                temp_root,
                entries=[{"speaker": "Narrator", "text": "Hello world."}],
                chunks=[{"id": 0, "uid": "c1", "speaker": "Narrator", "text": "Hello world.", "status": "pending"}],
            )

            workflow_state_path = os.path.join(temp_root, "new_mode_workflow_state.json")
            with open(workflow_state_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "running": True,
                        "paused": False,
                        "options": {"process_voices": True, "generate_audio": True},
                        "completed_stages": [],
                    },
                    f,
                )

            original_root = app_module.ROOT_DIR
            original_new_mode_path = app_module.NEW_MODE_WORKFLOW_STATE_PATH
            original_starter = app_module._start_new_mode_workflow_thread_locked
            original_pm = app_module.project_manager
            started = {"count": 0}
            try:
                app_module.ROOT_DIR = temp_root
                app_module.NEW_MODE_WORKFLOW_STATE_PATH = workflow_state_path
                app_module.project_manager = manager
                app_module._start_new_mode_workflow_thread_locked = lambda: started.__setitem__("count", started["count"] + 1)
                app_module._restore_new_mode_workflow_state()
                state = app_module.process_state["new_mode_workflow"]
                self.assertEqual(started["count"], 0)
                self.assertFalse(state["running"])
                self.assertFalse(state["paused"])
                self.assertEqual(
                    state["completed_stages"],
                    ["process_paragraphs", "assign_dialogue", "extract_temperament", "create_script"],
                )
                self.assertEqual(state["options"], {"process_voices": False, "generate_audio": False, "full_cast": True})
                self.assertTrue(
                    any(
                        "Project script complete, Reset Project if you wish to begin generation from the beginning."
                        in entry
                        for entry in state.get("logs", [])
                    )
                )
            finally:
                app_module.ROOT_DIR = original_root
                app_module.NEW_MODE_WORKFLOW_STATE_PATH = original_new_mode_path
                app_module.project_manager = original_pm
                app_module._start_new_mode_workflow_thread_locked = original_starter
                manager.shutdown_script_store(flush=True)

    def test_pipeline_step_status_treats_existing_script_project_as_complete_without_paragraphs(self):
        with tempfile.TemporaryDirectory() as temp_root:
            manager = _seed_db_project(
                temp_root,
                entries=[{"speaker": "Narrator", "text": "Hello world."}],
                chunks=[{"id": 0, "uid": "c1", "speaker": "Narrator", "text": "Hello world.", "status": "done"}],
            )

            original_root = app_module.ROOT_DIR
            original_pm = app_module.project_manager
            try:
                app_module.ROOT_DIR = temp_root
                app_module.project_manager = manager

                result = asyncio.run(app_module.get_pipeline_step_status())
            finally:
                app_module.ROOT_DIR = original_root
                app_module.project_manager = original_pm
                manager.shutdown_script_store(flush=True)

            self.assertFalse(result["has_input_file"])
            self.assertEqual(result["process_paragraphs"], "complete")
            self.assertEqual(result["assign_dialogue"], "complete")
            self.assertEqual(result["extract_temperament"], "complete")
            self.assertEqual(result["create_script"], "complete")


class LegacyCliRedirectTests(unittest.TestCase):
    def _run_script(self, *args):
        env = dict(os.environ)
        env["PYTHONPATH"] = str(APP_DIR)
        result = subprocess.run(
            [sys.executable, *args],
            cwd=str(APP_DIR),
            env=env,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise AssertionError(
                f"Script failed: {' '.join(args)}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
            )
        return result

    def test_process_paragraphs_legacy_output_path_redirects_into_project_store(self):
        with tempfile.TemporaryDirectory() as temp_root:
            os.makedirs(os.path.join(temp_root, "app"), exist_ok=True)
            input_path = os.path.join(temp_root, "story.txt")
            output_path = os.path.join(temp_root, "paragraphs.json")
            with open(input_path, "w", encoding="utf-8") as f:
                f.write('Chapter One\n\n"Hello there."\n\nPlain narration.')

            self._run_script(
                os.path.join("scripts", "process_paragraphs.py"),
                input_path,
                output_path,
            )

            manager = ProjectManager(temp_root)
            try:
                paragraphs_doc = manager.load_paragraphs() or {}
                self.assertGreater(len(paragraphs_doc.get("paragraphs") or []), 0)
                self.assertFalse(os.path.exists(output_path))
            finally:
                manager.shutdown_script_store(flush=True)

    def test_create_script_legacy_file_args_import_into_project_store(self):
        with tempfile.TemporaryDirectory() as temp_root:
            os.makedirs(os.path.join(temp_root, "app"), exist_ok=True)
            paragraphs_path = os.path.join(temp_root, "paragraphs.json")
            script_output_path = os.path.join(temp_root, "annotated_script.json")
            chunks_output_path = os.path.join(temp_root, "chunks.json")
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
                            }
                        ]
                    },
                    f,
                    indent=2,
                    ensure_ascii=False,
                )

            self._run_script(
                os.path.join("scripts", "create_script.py"),
                paragraphs_path,
                script_output_path,
                chunks_output_path,
            )

            manager = ProjectManager(temp_root)
            try:
                script_doc = manager.load_script_document()
                chunks = manager.load_chunks()
                self.assertEqual(len(script_doc.get("entries") or []), 1)
                self.assertEqual(len(chunks), 1)
                self.assertFalse(os.path.exists(script_output_path))
                self.assertFalse(os.path.exists(chunks_output_path))
            finally:
                manager.shutdown_script_store(flush=True)

    def test_start_new_mode_workflow_uses_stage_markers_to_skip_script_pipeline(self):
        with tempfile.TemporaryDirectory() as temp_root:
            with open(os.path.join(temp_root, "state.json"), "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "new_mode_stage_markers": {
                            "process_paragraphs": {"completed_at": 1},
                            "assign_dialogue": {"completed_at": 2},
                            "extract_temperament": {"completed_at": 3},
                            "create_script": {"completed_at": 4},
                        }
                    },
                    f,
                )

            original_root = app_module.ROOT_DIR
            original_starter = app_module._start_new_mode_workflow_thread_locked
            try:
                app_module.ROOT_DIR = temp_root
                app_module._start_new_mode_workflow_thread_locked = lambda: None
                with app_module.new_mode_workflow_lock:
                    app_module.process_state["new_mode_workflow"] = app_module._new_mode_workflow_initial_state()
                result = asyncio.run(
                    app_module.start_new_mode_workflow(
                        app_module.NewModeWorkflowRequest(process_voices=False, generate_audio=False)
                    )
                )
                self.assertEqual(
                    result["completed_stages"],
                    ["process_paragraphs", "assign_dialogue", "extract_temperament", "create_script"],
                )
            finally:
                app_module.ROOT_DIR = original_root
                app_module._start_new_mode_workflow_thread_locked = original_starter

    def test_restore_job_progress_skips_chunks_with_valid_audio_even_if_status_pending(self):
        original_validate = app_module.project_manager._validate_chunk_audio
        try:
            def fake_validate(chunk, _dictionary_entries):
                if chunk.get("id") == 0:
                    return {"is_valid": True}
                return None

            app_module.project_manager._validate_chunk_audio = fake_validate
            raw_job = {"indices": [0, 1], "word_counts": {"0": 3, "1": 5}}
            chunks = [
                {"id": 0, "status": "pending", "audio_path": "voicelines/a.mp3", "text": "alpha beta gamma"},
                {"id": 1, "status": "pending", "audio_path": None, "text": "delta epsilon zeta eta theta"},
            ]
            progress = app_module._restore_job_progress_from_chunks(raw_job, chunks)
            self.assertEqual(progress["pending_indices"], [1])
            self.assertEqual(progress["processed_clips"], 1)
            self.assertEqual(chunks[0]["status"], "done")
        finally:
            app_module.project_manager._validate_chunk_audio = original_validate

    def test_cancel_audio_clears_persisted_queue_state_file_even_when_idle(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            queue_state_path = os.path.join(temp_dir, "audio_queue_state.json")
            with open(queue_state_path, "w", encoding="utf-8") as f:
                json.dump({"queue": [{"id": 1}]}, f)

            original_path = app_module.AUDIO_QUEUE_STATE_PATH
            with app_module.audio_queue_lock:
                original_queue = list(app_module.audio_queue)
                original_current = app_module.audio_current_job
            try:
                app_module.AUDIO_QUEUE_STATE_PATH = queue_state_path
                with app_module.audio_queue_lock:
                    app_module.audio_queue.clear()
                    app_module.audio_current_job = None
                result = asyncio.run(app_module.cancel_audio())
                self.assertIn(result["status"], {"not_running", "cancelled"})
                self.assertFalse(os.path.exists(queue_state_path))
            finally:
                app_module.AUDIO_QUEUE_STATE_PATH = original_path
                with app_module.audio_queue_lock:
                    app_module.audio_queue[:] = original_queue
                    app_module.audio_current_job = original_current

    def test_sync_from_script_if_stale_skips_while_audio_running(self):
        original_sync = app_module.project_manager.sync_chunks_from_script_if_stale
        with app_module.audio_queue_lock:
            original_queue = list(app_module.audio_queue)
            original_current = app_module.audio_current_job
            original_merge = bool(app_module.process_state["audio"].get("merge_running", False))
        called = {"value": False}

        try:
            def fake_sync():
                called["value"] = True
                return {"synced": True, "reason": "script_newer_than_chunks"}

            app_module.project_manager.sync_chunks_from_script_if_stale = fake_sync
            with app_module.audio_queue_lock:
                app_module.audio_queue[:] = [{"id": 999, "status": "queued"}]
                app_module.audio_current_job = None
                app_module.process_state["audio"]["merge_running"] = False

            result = asyncio.run(app_module.sync_chunks_from_script_if_stale())
            self.assertEqual(result.get("synced"), False)
            self.assertEqual(result.get("reason"), "audio_running")
            self.assertFalse(called["value"])
        finally:
            app_module.project_manager.sync_chunks_from_script_if_stale = original_sync
            with app_module.audio_queue_lock:
                app_module.audio_queue[:] = original_queue
                app_module.audio_current_job = original_current
                app_module.process_state["audio"]["merge_running"] = original_merge


if __name__ == "__main__":
    unittest.main()
