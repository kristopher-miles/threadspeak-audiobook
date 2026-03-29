import copy
import importlib.util
import os
import tempfile
import unittest
import asyncio
import json
from fastapi import HTTPException
from fastapi import BackgroundTasks

MODULE_PATH = os.path.join(os.path.dirname(__file__), "app.py")
SPEC = importlib.util.spec_from_file_location("alexandria_app_module_workflow_entrypoints", MODULE_PATH)
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
                "options": {"process_voices": True, "generate_audio": True},
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

    def test_manual_stage_start_is_blocked_while_new_mode_workflow_active(self):
        with app_module.new_mode_workflow_lock:
            app_module.process_state["new_mode_workflow"] = app_module._new_mode_workflow_initial_state() | {
                "running": True,
                "paused": False,
            }

        with self.assertRaises(HTTPException) as ctx:
            asyncio.run(app_module.start_create_script(BackgroundTasks()))
        self.assertEqual(ctx.exception.status_code, 409)

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
