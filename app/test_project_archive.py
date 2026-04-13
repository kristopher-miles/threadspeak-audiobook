import importlib.util
import asyncio
import json
import os
import sqlite3
import tempfile
import unittest
import zipfile
from fastapi import BackgroundTasks
from fastapi import HTTPException
from project import ProjectManager

MODULE_PATH = os.path.join(os.path.dirname(__file__), "app.py")
SPEC = importlib.util.spec_from_file_location("threadspeak_app_module_archive", MODULE_PATH)
app_module = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(app_module)


def _write_json(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f)


def _write_project_zip(path, files, metadata=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    manifest = {
        "kind": "threadspeak_project_archive",
        "version": app_module.PROJECT_ARCHIVE_VERSION,
        "created_at": 1,
        "entries": sorted(files.keys()),
    }
    if metadata is not None:
        manifest["metadata"] = metadata
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(app_module.PROJECT_ARCHIVE_MANIFEST_NAME, json.dumps(manifest))
        for rel, payload in files.items():
            if isinstance(payload, bytes):
                zf.writestr(rel, payload)
            else:
                zf.writestr(rel, json.dumps(payload))


def _write_sqlite_db(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE IF NOT EXISTS t (id INTEGER PRIMARY KEY, value TEXT)")
        conn.execute("INSERT INTO t(value) VALUES ('ok')")
        conn.commit()


def _sqlite_bytes():
    with tempfile.TemporaryDirectory() as temp_root:
        path = os.path.join(temp_root, "snapshot.sqlite3")
        _write_sqlite_db(path)
        with open(path, "rb") as f:
            return f.read()


def _ensure_project_root(root):
    for dirname in (
        "app",
        "scripts",
        "saved_projects",
        "uploads",
        "voicelines",
        "clone_voices",
        "designed_voices",
        "workflow",
        "db",
        "repair",
        "exports",
    ):
        os.makedirs(os.path.join(root, dirname), exist_ok=True)


def _seed_db_project(root, *, entries=None, chunks=None, voice_config=None, state=None):
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
        if state is not None:
            _write_json(os.path.join(root, "state.json"), state)
        return manager
    except Exception:
        manager.shutdown_script_store(flush=True)
        raise


def _snapshot_manager_db(manager, snapshot_path):
    app_module._copy_sqlite_database_snapshot(manager.chunks_db_path, snapshot_path)


class _StubProjectManager:
    def __init__(self):
        self.engine = object()
        self.asr_engine = object()
        self._transcription_cache_lock = app_module.threading.Lock()
        self._transcription_cache = {"stale": True}
        self.chunks_db_path = os.path.join(app_module.ROOT_DIR, "chunks.sqlite3")
        self.chunks_queue_log_path = os.path.join(app_module.ROOT_DIR, "chunks.queue.log")

    def recover_interrupted_generating_chunks(self):
        pass

    def reconcile_chunk_audio_states(self):
        pass

    def reload_script_store(self):
        return None

    def load_chunks(self):
        try:
            with open(app_module.CHUNKS_PATH, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except (OSError, ValueError, json.JSONDecodeError):
            return []
        return payload if isinstance(payload, list) else []

    def get_chunk_chapter_summary(self):
        chunks = self.load_chunks()
        ordered = []
        last_seen = None
        for chunk in chunks:
            chapter = str((chunk or {}).get("chapter") or "").strip()
            if not chapter:
                continue
            if chapter != last_seen:
                ordered.append(chapter)
                last_seen = chapter
        return {
            "chunk_count": len(chunks),
            "chapter_count": len(ordered),
            "last_chapter": ordered[-1] if ordered else None,
        }

    def has_generated_chunk_audio(self):
        return any(str((chunk or {}).get("audio_path") or "").strip() for chunk in self.load_chunks())

    def has_substantive_chunks(self):
        return any(
            str((chunk or {}).get("text") or "").strip() and str((chunk or {}).get("speaker") or "").strip()
            for chunk in self.load_chunks()
        )


class _TempProjectRuntime:
    def __init__(self, testcase, temp_root):
        self.testcase = testcase
        self.temp_root = temp_root
        self.originals = {}

    def __enter__(self):
        for name in (
            "ROOT_DIR",
            "SCRIPTS_DIR",
            "SAVED_PROJECTS_DIR",
            "UPLOADS_DIR",
            "VOICELINES_DIR",
            "CLONE_VOICES_DIR",
            "DESIGNED_VOICES_DIR",
            "CLONE_VOICES_MANIFEST",
            "DESIGNED_VOICES_MANIFEST",
            "project_manager",
            "_any_project_task_running",
        ):
            self.originals[name] = getattr(app_module, name, None)

        _ensure_project_root(self.temp_root)
        app_module.ROOT_DIR = self.temp_root
        app_module.SCRIPTS_DIR = os.path.join(self.temp_root, "scripts")
        app_module.SAVED_PROJECTS_DIR = os.path.join(self.temp_root, "saved_projects")
        app_module.UPLOADS_DIR = os.path.join(self.temp_root, "uploads")
        app_module.VOICELINES_DIR = os.path.join(self.temp_root, "voicelines")
        app_module.CLONE_VOICES_DIR = os.path.join(self.temp_root, "clone_voices")
        app_module.DESIGNED_VOICES_DIR = os.path.join(self.temp_root, "designed_voices")
        app_module.CLONE_VOICES_MANIFEST = os.path.join(self.temp_root, "clone_voices", "manifest.json")
        app_module.DESIGNED_VOICES_MANIFEST = os.path.join(self.temp_root, "designed_voices", "manifest.json")
        self.manager = _seed_db_project(self.temp_root)
        app_module.project_manager = self.manager
        app_module._any_project_task_running = lambda: None

        return self

    def __exit__(self, exc_type, exc, tb):
        manager = getattr(self, "manager", None)
        if manager is not None:
            try:
                manager.shutdown_script_store(flush=True)
            except Exception:
                pass
        for name, value in self.originals.items():
            if value is None and hasattr(app_module, name):
                delattr(app_module, name)
            elif value is not None:
                setattr(app_module, name, value)
        return False


class ProjectArchiveHelpersTests(unittest.TestCase):
    def test_project_archive_save_load_preserves_state_and_designed_voice_binding(self):
        with tempfile.TemporaryDirectory() as temp_root:
            original_values = {
                "ROOT_DIR": app_module.ROOT_DIR,
                "SAVED_PROJECTS_DIR": app_module.SAVED_PROJECTS_DIR,
                "SCRIPTS_DIR": app_module.SCRIPTS_DIR,
                "UPLOADS_DIR": app_module.UPLOADS_DIR,
                "VOICELINES_DIR": app_module.VOICELINES_DIR,
                "CLONE_VOICES_DIR": getattr(app_module, "CLONE_VOICES_DIR", os.path.join(app_module.ROOT_DIR, "clone_voices")),
                "DESIGNED_VOICES_DIR": getattr(app_module, "DESIGNED_VOICES_DIR", os.path.join(app_module.ROOT_DIR, "designed_voices")),
                "CLONE_VOICES_MANIFEST": app_module.CLONE_VOICES_MANIFEST,
                "DESIGNED_VOICES_MANIFEST": app_module.DESIGNED_VOICES_MANIFEST,
                "project_manager": app_module.project_manager,
                "_any_project_task_running": app_module._any_project_task_running,
            }
            manager = None
            try:
                _ensure_project_root(temp_root)
                app_module.ROOT_DIR = temp_root
                app_module.SAVED_PROJECTS_DIR = os.path.join(temp_root, "saved_projects")
                app_module.SCRIPTS_DIR = os.path.join(temp_root, "scripts")
                app_module.UPLOADS_DIR = os.path.join(temp_root, "uploads")
                app_module.VOICELINES_DIR = os.path.join(temp_root, "voicelines")
                app_module.CLONE_VOICES_DIR = os.path.join(temp_root, "clone_voices")
                app_module.DESIGNED_VOICES_DIR = os.path.join(temp_root, "designed_voices")
                app_module.CLONE_VOICES_MANIFEST = os.path.join(app_module.CLONE_VOICES_DIR, "manifest.json")
                app_module.DESIGNED_VOICES_MANIFEST = os.path.join(app_module.DESIGNED_VOICES_DIR, "manifest.json")
                app_module._any_project_task_running = lambda: None

                with open(os.path.join(temp_root, "state.json"), "w", encoding="utf-8") as f:
                    json.dump({"loaded_script_name": "book", "input_file_path": os.path.join(app_module.UPLOADS_DIR, "story.txt")}, f)
                with open(os.path.join(app_module.UPLOADS_DIR, "story.txt"), "w", encoding="utf-8") as f:
                    f.write("source")
                with open(app_module.CLONE_VOICES_MANIFEST, "w", encoding="utf-8") as f:
                    json.dump([], f)
                with open(app_module.DESIGNED_VOICES_MANIFEST, "w", encoding="utf-8") as f:
                    json.dump(
                        [{
                            "id": "voice-1",
                            "name": "Blake voice",
                            "description": "desc",
                            "sample_text": "hello there",
                            "filename": "book.blake.wav",
                            "script_title": "book",
                        }],
                        f,
                    )
                with open(os.path.join(app_module.DESIGNED_VOICES_DIR, "book.blake.wav"), "wb") as f:
                    f.write(b"wav")

                manager = _seed_db_project(
                    temp_root,
                    entries=[{"speaker": "Blake", "text": "hello"}],
                )
                app_module.project_manager = manager
                manager._save_voice_config(
                    {
                        "Blake": {
                            "type": "design",
                            "description": "desc",
                            "ref_text": "hello there",
                            "generated_ref_text": "hello there",
                            "ref_audio": "designed_voices/book.blake.wav",
                            "alias": "",
                            "seed": "-1",
                        },
                        "Bake": {
                            "type": "design",
                            "description": "",
                            "ref_text": "",
                            "generated_ref_text": "",
                            "ref_audio": "",
                            "alias": "Blake",
                            "seed": "-1",
                        },
                    }
                )

                result = asyncio.run(app_module.save_script(app_module.ScriptSaveRequest(name="Demo Save")))
                self.assertEqual(result["kind"], "project")

                with open(os.path.join(temp_root, "state.json"), "r", encoding="utf-8") as f:
                    saved_state = json.load(f)
                self.assertEqual(saved_state.get("loaded_script_name"), "book")
                self.assertEqual(saved_state.get("input_file_path"), os.path.join(app_module.UPLOADS_DIR, "story.txt"))

                manager._save_voice_config({})
                with open(app_module.DESIGNED_VOICES_MANIFEST, "w", encoding="utf-8") as f:
                    json.dump([], f)
                try:
                    os.remove(os.path.join(app_module.DESIGNED_VOICES_DIR, "book.blake.wav"))
                except OSError:
                    pass

                load_result = asyncio.run(app_module.load_script(app_module.ScriptLoadRequest(name="Demo Save")))
                self.assertEqual(load_result["kind"], "project")
                restored_config = app_module.project_manager._load_voice_config()
                self.assertEqual(restored_config["Blake"]["description"], "desc")
                self.assertEqual(restored_config["Blake"]["generated_ref_text"], "hello there")
                self.assertEqual(restored_config["Blake"]["ref_audio"], "designed_voices/book.blake.wav")
                self.assertEqual(restored_config["Bake"]["alias"], "Blake")
                self.assertEqual(app_module.project_manager._current_script_title(), "book")
                self.assertTrue(os.path.exists(os.path.join(app_module.DESIGNED_VOICES_DIR, "book.blake.wav")))
                reusable = app_module._find_saved_voice_option_for_speaker("Blake")
                self.assertIsNotNone(reusable)
                self.assertEqual(reusable["ref_audio"], "designed_voices/book.blake.wav")
                self.assertEqual(reusable["ref_text"], "hello there")
                self.assertEqual(reusable["generated_ref_text"], "hello there")
                self.assertEqual(reusable["description"], "desc")
            finally:
                current_manager = app_module.project_manager
                if current_manager is not None and current_manager is not original_values["project_manager"]:
                    try:
                        current_manager.shutdown_script_store()
                    except Exception:
                        pass
                for name, value in original_values.items():
                    setattr(app_module, name, value)

    def test_script_ingestion_preflight_warns_when_chunks_match_last_epub_chapter(self):
        with tempfile.TemporaryDirectory() as temp_root:
            input_path = os.path.join(temp_root, "uploads", "story.epub")
            os.makedirs(os.path.dirname(input_path), exist_ok=True)
            with open(input_path, "w", encoding="utf-8") as f:
                f.write("stub")
            with open(os.path.join(temp_root, "state.json"), "w", encoding="utf-8") as f:
                json.dump({"input_file_path": input_path}, f)

            original_root = app_module.ROOT_DIR
            original_pm = app_module.project_manager
            original_loader = app_module.load_source_document
            manager = None
            try:
                app_module.ROOT_DIR = temp_root
                manager = _seed_db_project(
                    temp_root,
                    chunks=[
                        {"id": 0, "uid": "c1", "chapter": "Chapter 1", "speaker": "Narrator", "text": "A", "status": "pending"},
                        {"id": 1, "uid": "c2", "chapter": "Epilogue: Interview", "speaker": "Narrator", "text": "B", "status": "pending"},
                    ],
                )
                app_module.project_manager = manager
                app_module.load_source_document = lambda _path: {
                    "type": "epub",
                    "chapters": [
                        {"title": "Chapter 1", "text": "A"},
                        {"title": "Epilogue: Interview", "text": "B"},
                    ],
                }
                result = app_module._script_ingestion_preflight_summary()
                self.assertTrue(result["warn"])
                self.assertEqual(result["reason"], "matching_last_chapter")
                self.assertEqual(result["last_chapter"], "Epilogue: Interview")
                self.assertEqual(result["last_source_chapter"], "Epilogue: Interview")
            finally:
                app_module.ROOT_DIR = original_root
                app_module.project_manager = original_pm
                if manager is not None:
                    manager.shutdown_script_store(flush=True)
                app_module.load_source_document = original_loader

    def test_generate_script_skip_import_marks_script_complete_without_running(self):
        with tempfile.TemporaryDirectory() as temp_root:
            input_path = os.path.join(temp_root, "uploads", "story.epub")
            os.makedirs(os.path.dirname(input_path), exist_ok=True)
            with open(input_path, "w", encoding="utf-8") as f:
                f.write("stub")
            with open(os.path.join(temp_root, "state.json"), "w", encoding="utf-8") as f:
                json.dump({"input_file_path": input_path}, f)

            original_root = app_module.ROOT_DIR
            original_run_process = app_module.run_process
            original_preflight = app_module._script_ingestion_preflight_summary
            try:
                app_module.ROOT_DIR = temp_root
                app_module._script_ingestion_preflight_summary = lambda: {"warn": True, "message": "already imported"}

                def should_not_run(*_args, **_kwargs):
                    raise AssertionError("run_process should not be called when script import is skipped")

                app_module.run_process = should_not_run
                result = asyncio.run(
                    app_module.generate_script(
                        app_module.ScriptGenerationRequest(skip_import=True),
                        BackgroundTasks(),
                    )
                )
                self.assertEqual(result["status"], "skipped")

                with open(os.path.join(temp_root, "state.json"), "r", encoding="utf-8") as f:
                    state = json.load(f)
                self.assertIn("script", state["processing_stage_markers"])
            finally:
                app_module.ROOT_DIR = original_root
                app_module.run_process = original_run_process
                app_module._script_ingestion_preflight_summary = original_preflight

    def test_start_processing_workflow_skips_persisted_completed_stages(self):
        with tempfile.TemporaryDirectory() as temp_root:
            with open(os.path.join(temp_root, "state.json"), "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "processing_stage_markers": {
                            "script": {"completed_at": 1},
                            "review": {"completed_at": 2},
                        }
                    },
                    f,
                )

            original_root = app_module.ROOT_DIR
            original_workflow_path = app_module.PROCESSING_WORKFLOW_STATE_PATH
            original_workflow_state = dict(app_module.process_state["processing_workflow"])
            original_new_mode_workflow_state = dict(app_module.process_state["new_mode_workflow"])
            original_starter = app_module._start_processing_workflow_thread_locked
            original_any_running = app_module._any_project_task_running
            try:
                app_module.ROOT_DIR = temp_root
                app_module.PROCESSING_WORKFLOW_STATE_PATH = os.path.join(temp_root, "processing_workflow_state.json")
                app_module.process_state["processing_workflow"] = app_module._new_processing_workflow_state()
                app_module.process_state["new_mode_workflow"] = app_module._new_mode_workflow_initial_state()
                app_module._start_processing_workflow_thread_locked = lambda: None
                app_module._any_project_task_running = lambda: None

                result = asyncio.run(
                    app_module.start_processing_workflow(
                        app_module.ProcessingWorkflowRequest(process_voices=False, generate_audio=False)
                    )
                )

                self.assertTrue(result["running"])
                self.assertEqual(result["completed_stages"], ["script", "review"])
                self.assertIn("Skipping already completed stages", result["logs"][-1])
            finally:
                app_module.ROOT_DIR = original_root
                app_module.PROCESSING_WORKFLOW_STATE_PATH = original_workflow_path
                app_module.process_state["processing_workflow"] = original_workflow_state
                app_module.process_state["new_mode_workflow"] = original_new_mode_workflow_state
                app_module._start_processing_workflow_thread_locked = original_starter
                app_module._any_project_task_running = original_any_running

    def test_run_generate_script_task_clears_downstream_markers_and_marks_script_complete(self):
        with tempfile.TemporaryDirectory() as temp_root:
            input_path = os.path.join(temp_root, "uploads", "story.epub")
            os.makedirs(os.path.dirname(input_path), exist_ok=True)
            with open(input_path, "w", encoding="utf-8") as f:
                f.write("story")
            with open(os.path.join(temp_root, "state.json"), "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "input_file_path": input_path,
                        "processing_stage_markers": {
                            "script": {"completed_at": 1},
                            "review": {"completed_at": 2},
                            "sanity": {"completed_at": 3},
                        },
                    },
                    f,
                )

            original_root = app_module.ROOT_DIR
            original_run_process = app_module.run_process
            try:
                app_module.ROOT_DIR = temp_root

                def fake_run_process(command, task_name, run_id):
                    self.assertEqual(task_name, "script")
                    self.assertEqual(command[-1], input_path)
                    return True

                app_module.run_process = fake_run_process
                self.assertTrue(app_module._run_generate_script_task("run-1"))

                with open(os.path.join(temp_root, "state.json"), "r", encoding="utf-8") as f:
                    state = json.load(f)
                self.assertEqual(list(state["processing_stage_markers"].keys()), ["script"])
            finally:
                app_module.ROOT_DIR = original_root
                app_module.run_process = original_run_process

    def test_load_script_marks_only_script_stage_complete(self):
        with tempfile.TemporaryDirectory() as temp_root:
            scripts_dir = os.path.join(temp_root, "scripts")
            os.makedirs(scripts_dir, exist_ok=True)
            manager = _seed_db_project(temp_root, entries=[{"speaker": "Narrator", "text": "hello"}])
            _snapshot_manager_db(manager, os.path.join(scripts_dir, "demo.sqlite3"))
            with open(os.path.join(temp_root, "state.json"), "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "render_prep_complete": True,
                        "processing_stage_markers": {
                            "script": {"completed_at": 1},
                            "review": {"completed_at": 2},
                            "sanity": {"completed_at": 3},
                        },
                    },
                    f,
                )

            original_root = app_module.ROOT_DIR
            original_scripts = app_module.SCRIPTS_DIR
            original_pm = app_module.project_manager
            original_audio_state = app_module.process_state["audio"].copy()
            with app_module.audio_queue_lock:
                original_audio_queue = list(app_module.audio_queue)
                original_audio_current = app_module.audio_current_job
            try:
                app_module.ROOT_DIR = temp_root
                app_module.SCRIPTS_DIR = scripts_dir
                app_module.project_manager = manager
                app_module.process_state["audio"]["running"] = False
                with app_module.audio_queue_lock:
                    app_module.audio_queue.clear()
                    app_module.audio_current_job = None

                result = asyncio.run(app_module.load_script(app_module.ScriptLoadRequest(name="demo")))
                self.assertEqual(result["status"], "loaded")

                with open(os.path.join(temp_root, "state.json"), "r", encoding="utf-8") as f:
                    state = json.load(f)
                self.assertFalse(state["render_prep_complete"])
                self.assertEqual(list(state["processing_stage_markers"].keys()), ["script", "voices"])
                self.assertEqual(
                    list(state["new_mode_stage_markers"].keys()),
                    ["process_paragraphs", "assign_dialogue", "extract_temperament", "create_script", "process_voices"],
                )
                self.assertTrue(os.path.exists(manager.chunks_db_path))
            finally:
                app_module.ROOT_DIR = original_root
                app_module.SCRIPTS_DIR = original_scripts
                app_module.project_manager = original_pm
                app_module.process_state["audio"].clear()
                app_module.process_state["audio"].update(original_audio_state)
                with app_module.audio_queue_lock:
                    app_module.audio_queue[:] = original_audio_queue
                    app_module.audio_current_job = original_audio_current
                manager.shutdown_script_store(flush=True)

    def test_load_script_ignores_stale_audio_running_flag_when_no_job_or_queue(self):
        with tempfile.TemporaryDirectory() as temp_root:
            scripts_dir = os.path.join(temp_root, "scripts")
            os.makedirs(scripts_dir, exist_ok=True)
            manager = _seed_db_project(temp_root, entries=[{"speaker": "Narrator", "text": "hello"}])
            _snapshot_manager_db(manager, os.path.join(scripts_dir, "demo.sqlite3"))
            with open(os.path.join(temp_root, "state.json"), "w", encoding="utf-8") as f:
                json.dump({"render_prep_complete": True}, f)

            original_root = app_module.ROOT_DIR
            original_scripts = app_module.SCRIPTS_DIR
            original_pm = app_module.project_manager
            original_audio_state = app_module.process_state["audio"].copy()
            with app_module.audio_queue_lock:
                original_audio_queue = list(app_module.audio_queue)
                original_audio_current = app_module.audio_current_job
            try:
                app_module.ROOT_DIR = temp_root
                app_module.SCRIPTS_DIR = scripts_dir
                app_module.project_manager = manager

                app_module.process_state["audio"]["running"] = True  # stale flag
                app_module.process_state["audio"]["merge_running"] = False
                with app_module.audio_queue_lock:
                    app_module.audio_queue.clear()
                    app_module.audio_current_job = None

                result = asyncio.run(app_module.load_script(app_module.ScriptLoadRequest(name="demo")))
                self.assertEqual(result["status"], "loaded")
            finally:
                app_module.ROOT_DIR = original_root
                app_module.SCRIPTS_DIR = original_scripts
                app_module.project_manager = original_pm
                app_module.process_state["audio"].clear()
                app_module.process_state["audio"].update(original_audio_state)
                with app_module.audio_queue_lock:
                    app_module.audio_queue[:] = original_audio_queue
                    app_module.audio_current_job = original_audio_current
                manager.shutdown_script_store(flush=True)

    def test_normalize_archive_path_rejects_parent_traversal(self):
        with self.assertRaises(ValueError):
            app_module._normalize_archive_path("../secret.txt")

    def test_allowed_archive_paths_cover_expected_project_content(self):
        self.assertTrue(app_module._is_allowed_project_archive_path("db/chunks.sqlite3"))
        self.assertTrue(app_module._is_allowed_project_archive_path("voicelines/chunk_001.mp3"))
        self.assertTrue(app_module._is_allowed_project_archive_path("voicelines/discarded/rejected_001.mp3"))
        self.assertTrue(app_module._is_allowed_project_archive_path("clone_voices/manifest.json"))
        self.assertTrue(app_module._is_allowed_project_archive_path("state.json"))
        self.assertFalse(app_module._is_allowed_project_archive_path("app/config.json"))

    def test_archive_state_rewrites_uploaded_file_path_relative_to_root(self):
        with tempfile.TemporaryDirectory() as temp_root:
            uploads_dir = os.path.join(temp_root, "uploads")
            os.makedirs(uploads_dir, exist_ok=True)
            input_path = os.path.join(uploads_dir, "story.txt")
            with open(input_path, "w", encoding="utf-8") as f:
                f.write("hello")

            state_path = os.path.join(temp_root, "state.json")
            with open(state_path, "w", encoding="utf-8") as f:
                json.dump({"input_file_path": input_path, "render_prep_complete": True}, f)

            original_root = app_module.ROOT_DIR
            original_uploads = app_module.UPLOADS_DIR
            try:
                app_module.ROOT_DIR = temp_root
                app_module.UPLOADS_DIR = uploads_dir
                exported = app_module._archive_state_with_relative_paths()
            finally:
                app_module.ROOT_DIR = original_root
                app_module.UPLOADS_DIR = original_uploads

        self.assertEqual(exported["input_file_path"], "uploads/story.txt")
        self.assertTrue(exported["render_prep_complete"])

    def test_project_archive_entries_include_durable_assets_only(self):
        with tempfile.TemporaryDirectory() as temp_root:
            _ensure_project_root(temp_root)
            os.makedirs(os.path.join(temp_root, "voicelines", "discarded"), exist_ok=True)

            for rel in (
                "voicelines/live_a.mp3",
                "voicelines/live_b.mp3",
                "voicelines/orphan.mp3",
                "voicelines/discarded/rejected.mp3",
                "clone_voices/current_clone.wav",
                "clone_voices/orphan_clone.wav",
                "designed_voices/current_design.wav",
                "designed_voices/orphan_design.wav",
                "uploads/story.txt",
            ):
                full = os.path.join(temp_root, rel)
                os.makedirs(os.path.dirname(full), exist_ok=True)
                with open(full, "wb") as f:
                    f.write(b"test")

            with open(os.path.join(temp_root, "clone_voices", "manifest.json"), "w", encoding="utf-8") as f:
                json.dump([{"id": "clone-1", "filename": "current_clone.wav"}], f)
            with open(os.path.join(temp_root, "designed_voices", "manifest.json"), "w", encoding="utf-8") as f:
                json.dump([{"id": "design-1", "filename": "current_design.wav"}], f)
            for rel in (
                "workflow/processing_workflow_state.json",
                "workflow/new_mode_workflow_state.json",
                "workflow/audio_queue_state.json",
                "workflow/audio_cancel_tombstone.json",
                "workflow/script_generation_checkpoint.json",
                "workflow/script_review_checkpoint.json",
                "repair/script_repair_trace.jsonl",
            ):
                full = os.path.join(temp_root, rel)
                os.makedirs(os.path.dirname(full), exist_ok=True)
                with open(full, "w", encoding="utf-8") as f:
                    f.write("stale")

            original_root = app_module.ROOT_DIR
            original_uploads = app_module.UPLOADS_DIR
            original_clone_manifest = getattr(app_module, "CLONE_VOICES_MANIFEST", None)
            original_design_manifest = getattr(app_module, "DESIGNED_VOICES_MANIFEST", None)
            original_pm = app_module.project_manager
            manager = None
            try:
                app_module.ROOT_DIR = temp_root
                app_module.UPLOADS_DIR = os.path.join(temp_root, "uploads")
                app_module.CLONE_VOICES_MANIFEST = os.path.join(temp_root, "clone_voices", "manifest.json")
                app_module.DESIGNED_VOICES_MANIFEST = os.path.join(temp_root, "designed_voices", "manifest.json")
                manager = _seed_db_project(
                    temp_root,
                    entries=[{"speaker": "Narrator", "text": "hello"}, {"speaker": "Alice", "text": "hi"}],
                    chunks=[
                        {"id": 0, "uid": "c1", "speaker": "Narrator", "text": "hello", "status": "done", "audio_path": "voicelines/live_a.mp3"},
                        {"id": 1, "uid": "c2", "speaker": "Alice", "text": "hi", "status": "done", "audio_path": "voicelines/live_b.mp3"},
                        {"id": 2, "uid": "c3", "speaker": "Narrator", "text": "later", "status": "pending", "audio_path": None},
                    ],
                    voice_config={
                        "Narrator": {"ref_audio": "clone_voices/current_clone.wav"},
                        "Alice": {"ref_audio": "designed_voices/current_design.wav"},
                    },
                    state={"input_file_path": os.path.join(temp_root, "uploads", "story.txt")},
                )
                app_module.project_manager = manager
                entries = dict(app_module._project_archive_entries())
            finally:
                app_module.ROOT_DIR = original_root
                app_module.UPLOADS_DIR = original_uploads
                app_module.CLONE_VOICES_MANIFEST = original_clone_manifest
                app_module.DESIGNED_VOICES_MANIFEST = original_design_manifest
                app_module.project_manager = original_pm
                if manager is not None:
                    manager.shutdown_script_store(flush=True)

            self.assertIn("voicelines/live_a.mp3", entries)
            self.assertIn("voicelines/live_b.mp3", entries)
            self.assertNotIn("voicelines/orphan.mp3", entries)
            self.assertNotIn("voicelines/discarded/rejected.mp3", entries)
            self.assertIn("clone_voices/manifest.json", entries)
            self.assertIn("clone_voices/current_clone.wav", entries)
            self.assertNotIn("clone_voices/orphan_clone.wav", entries)
            self.assertIn("designed_voices/manifest.json", entries)
            self.assertIn("designed_voices/current_design.wav", entries)
            self.assertNotIn("designed_voices/orphan_design.wav", entries)
            self.assertIn("uploads/story.txt", entries)
            self.assertNotIn("workflow/processing_workflow_state.json", entries)
            self.assertNotIn("workflow/new_mode_workflow_state.json", entries)
            self.assertNotIn("workflow/audio_queue_state.json", entries)
            self.assertNotIn("workflow/audio_cancel_tombstone.json", entries)
            self.assertNotIn("workflow/script_generation_checkpoint.json", entries)
            self.assertNotIn("workflow/script_review_checkpoint.json", entries)
            self.assertNotIn("repair/script_repair_trace.jsonl", entries)
            self.assertIn("db/chunks.sqlite3", {name: None for name in app_module.PROJECT_ARCHIVE_DURABLE_FILES})

    def test_restore_project_archive_restores_discarded_pool_and_transcription_cache(self):
        with tempfile.TemporaryDirectory() as temp_root:
            extract_root = os.path.join(temp_root, "extracted")
            os.makedirs(os.path.join(extract_root, "voicelines", "discarded"), exist_ok=True)
            os.makedirs(os.path.join(extract_root, "uploads"), exist_ok=True)
            os.makedirs(os.path.join(extract_root, "clone_voices"), exist_ok=True)
            os.makedirs(os.path.join(extract_root, "designed_voices"), exist_ok=True)
            _write_sqlite_db(os.path.join(extract_root, "db", "chunks.sqlite3"))
            with open(os.path.join(extract_root, "state.json"), "w", encoding="utf-8") as f:
                json.dump({"input_file_path": "uploads/story.txt"}, f)
            with open(os.path.join(extract_root, "uploads", "story.txt"), "wb") as f:
                f.write(b"story")
            with open(os.path.join(extract_root, "voicelines", "discarded", "rejected.mp3"), "wb") as f:
                f.write(b"clip")

            original_root = app_module.ROOT_DIR
            original_uploads = app_module.UPLOADS_DIR
            original_project_manager = app_module.project_manager
            try:
                app_module.ROOT_DIR = temp_root
                app_module.UPLOADS_DIR = os.path.join(temp_root, "uploads")

                class StubManager:
                    def __init__(self):
                        self.engine = object()
                        self.asr_engine = object()
                        self._transcription_cache_lock = app_module.threading.Lock()
                        self._transcription_cache = {"stale": True}
                        self.recovered = False
                        self.reconciled = False

                    def recover_interrupted_generating_chunks(self):
                        self.recovered = True

                    def reconcile_chunk_audio_states(self):
                        self.reconciled = True

                stub_manager = StubManager()
                app_module.project_manager = stub_manager

                app_module._restore_project_archive(extract_root)

                self.assertTrue(os.path.exists(os.path.join(temp_root, "chunks.sqlite3")))
                self.assertTrue(os.path.exists(os.path.join(temp_root, "voicelines", "discarded", "rejected.mp3")))
                with open(os.path.join(temp_root, "state.json"), "r", encoding="utf-8") as f:
                    restored_state = json.load(f)
                self.assertEqual(restored_state["input_file_path"], os.path.join(temp_root, "uploads", "story.txt"))
                self.assertIsNone(stub_manager._transcription_cache)
                self.assertTrue(stub_manager.recovered)
                self.assertTrue(stub_manager.reconciled)
            finally:
                app_module.ROOT_DIR = original_root
                app_module.UPLOADS_DIR = original_uploads
                app_module.project_manager = original_project_manager

    def test_project_has_generated_audio_requires_existing_chunk_audio(self):
        with tempfile.TemporaryDirectory() as temp_root:
            with _TempProjectRuntime(self, temp_root):
                app_module.project_manager.save_chunks(
                    [
                        {"id": 0, "uid": "c1", "speaker": "Narrator", "text": "hello", "status": "done", "audio_path": "voicelines/missing.mp3"},
                        {"id": 1, "uid": "c2", "speaker": "Narrator", "text": "later", "status": "pending", "audio_path": None},
                    ]
                )
                self.assertFalse(app_module._project_has_generated_audio())

                audio_path = os.path.join(temp_root, "voicelines", "existing.mp3")
                with open(audio_path, "wb") as f:
                    f.write(b"audio")
                app_module.project_manager.save_chunks(
                    [{"id": 0, "uid": "c1", "speaker": "Narrator", "text": "hello", "status": "done", "audio_path": "voicelines/existing.mp3"}]
                )
                self.assertTrue(app_module._project_has_generated_audio())

    def test_unified_save_uses_project_zip_without_audio_and_removes_script_artifacts(self):
        with tempfile.TemporaryDirectory() as temp_root:
            with _TempProjectRuntime(self, temp_root):
                app_module.project_manager.script_store.replace_script_document(
                    entries=[{"speaker": "Narrator", "text": "hello"}],
                    dictionary=[],
                    sanity_cache={"phrase_decisions": {}},
                    reason="test_seed_script",
                    rebuild_chunks=True,
                    wait=True,
                )
                app_module.project_manager._save_voice_config({"Narrator": {}})
                app_module.project_manager.save_chunks(
                    [{"id": 0, "uid": "c1", "speaker": "Narrator", "text": "hello", "status": "done", "audio_path": "voicelines/missing.mp3"}]
                )
                _write_sqlite_db(os.path.join(app_module.SCRIPTS_DIR, "demo.sqlite3"))
                stale_zip = os.path.join(app_module.SAVED_PROJECTS_DIR, "demo.zip")
                _write_project_zip(stale_zip, {"state.json": {}})

                result = asyncio.run(app_module.save_script(app_module.ScriptSaveRequest(name="Demo")))

                archive_path = os.path.join(app_module.SAVED_PROJECTS_DIR, "demo.zip")
                self.assertEqual(result["kind"], "project")
                self.assertTrue(os.path.exists(archive_path))
                self.assertFalse(os.path.exists(os.path.join(app_module.SCRIPTS_DIR, "demo.sqlite3")))
                with zipfile.ZipFile(archive_path, "r") as zf:
                    self.assertIn("db/chunks.sqlite3", zf.namelist())

    def test_unified_save_uses_project_zip_with_audio_and_removes_script_artifacts(self):
        with tempfile.TemporaryDirectory() as temp_root:
            with _TempProjectRuntime(self, temp_root):
                app_module.project_manager.script_store.replace_script_document(
                    entries=[{"speaker": "Narrator", "text": "hello"}],
                    dictionary=[],
                    sanity_cache={"phrase_decisions": {}},
                    reason="test_seed_script",
                    rebuild_chunks=True,
                    wait=True,
                )
                app_module.project_manager._save_voice_config({"Narrator": {}})
                app_module.project_manager.save_chunks(
                    [{"id": 0, "uid": "c1", "speaker": "Narrator", "text": "hello", "status": "done", "audio_path": "voicelines/live.mp3"}]
                )
                with open(os.path.join(temp_root, "voicelines", "live.mp3"), "wb") as f:
                    f.write(b"audio")
                _write_sqlite_db(os.path.join(app_module.SCRIPTS_DIR, "demo.sqlite3"))

                result = asyncio.run(app_module.save_script(app_module.ScriptSaveRequest(name="Demo")))

                archive_path = os.path.join(app_module.SAVED_PROJECTS_DIR, "demo.zip")
                self.assertEqual(result["kind"], "project")
                self.assertTrue(os.path.exists(archive_path))
                self.assertFalse(os.path.exists(os.path.join(app_module.SCRIPTS_DIR, "demo.sqlite3")))
                with zipfile.ZipFile(archive_path, "r") as zf:
                    self.assertIn(app_module.PROJECT_ARCHIVE_MANIFEST_NAME, zf.namelist())
                    self.assertIn("voicelines/live.mp3", zf.namelist())
                    self.assertIn("db/chunks.sqlite3", zf.namelist())

    def test_unified_list_deduplicates_script_when_project_zip_exists(self):
        with tempfile.TemporaryDirectory() as temp_root:
            with _TempProjectRuntime(self, temp_root):
                _write_sqlite_db(os.path.join(app_module.SCRIPTS_DIR, "demo.sqlite3"))
                _write_sqlite_db(os.path.join(app_module.SCRIPTS_DIR, "script_only.sqlite3"))
                _write_project_zip(
                    os.path.join(app_module.SAVED_PROJECTS_DIR, "demo.zip"),
                    {"db/chunks.sqlite3": _sqlite_bytes()},
                    metadata={"kind": "project", "has_audio": False, "has_voice_config": True, "chunk_count": 0, "chapter_count": 0},
                )

                projects = asyncio.run(app_module.list_saved_scripts())
                by_name = {item["name"]: item for item in projects}

                self.assertEqual(by_name["demo"]["kind"], "project")
                self.assertFalse(by_name["demo"]["has_audio"])
                self.assertTrue(by_name["demo"]["has_voice_config"])
                self.assertEqual(by_name["script_only"]["kind"], "script")
                self.assertEqual(len([item for item in projects if item["name"] == "demo"]), 1)

    def test_unified_load_prefers_project_zip_over_script_snapshot(self):
        with tempfile.TemporaryDirectory() as temp_root:
            with _TempProjectRuntime(self, temp_root):
                _write_sqlite_db(os.path.join(app_module.SCRIPTS_DIR, "demo.sqlite3"))
                _write_project_zip(
                    os.path.join(app_module.SAVED_PROJECTS_DIR, "demo.zip"),
                    {
                        "db/chunks.sqlite3": _sqlite_bytes(),
                        "state.json": {"input_file_path": "uploads/story.txt"},
                        "uploads/story.txt": b"story",
                    },
                )

                result = asyncio.run(app_module.load_script(app_module.ScriptLoadRequest(name="Demo")))

                self.assertEqual(result["kind"], "project")
                self.assertTrue(os.path.exists(app_module.project_manager.chunks_db_path))

    def test_unified_delete_removes_project_zip_and_script_companions(self):
        with tempfile.TemporaryDirectory() as temp_root:
            with _TempProjectRuntime(self, temp_root):
                _write_sqlite_db(os.path.join(app_module.SCRIPTS_DIR, "demo.sqlite3"))
                with open(os.path.join(app_module.SCRIPTS_DIR, "demo.source.txt"), "w", encoding="utf-8") as f:
                    f.write("source")
                with open(os.path.join(app_module.CLONE_VOICES_DIR, "keep.wav"), "w", encoding="utf-8") as f:
                    f.write("voice")
                _write_project_zip(os.path.join(app_module.SAVED_PROJECTS_DIR, "demo.zip"), {"state.json": {}})

                result = asyncio.run(app_module.delete_script("Demo"))

                self.assertEqual(result["status"], "deleted")
                self.assertFalse(os.path.exists(os.path.join(app_module.SAVED_PROJECTS_DIR, "demo.zip")))
                self.assertFalse(os.path.exists(os.path.join(app_module.SCRIPTS_DIR, "demo.sqlite3")))
                self.assertFalse(os.path.exists(os.path.join(app_module.SCRIPTS_DIR, "demo.source.txt")))
                self.assertTrue(os.path.exists(os.path.join(app_module.CLONE_VOICES_DIR, "keep.wav")))

    def test_restore_project_archive_merges_reusable_voice_library_and_normalizes_state(self):
        with tempfile.TemporaryDirectory() as temp_root:
            with _TempProjectRuntime(self, temp_root):
                os.makedirs(os.path.join(app_module.CLONE_VOICES_DIR), exist_ok=True)
                with open(os.path.join(app_module.CLONE_VOICES_DIR, "keep.wav"), "wb") as f:
                    f.write(b"keep")
                with open(app_module.CLONE_VOICES_MANIFEST, "w", encoding="utf-8") as f:
                    json.dump([{"id": "keep", "filename": "keep.wav", "name": "Keep"}], f)
                app_module.project_manager.script_store.replace_script_document(
                    entries=[{"speaker": "Narrator", "text": "hello"}],
                    dictionary=[],
                    sanity_cache={"phrase_decisions": {}},
                    reason="test_seed_script",
                    rebuild_chunks=True,
                    wait=True,
                )
                snapshot_path = os.path.join(temp_root, "archive.sqlite3")
                _snapshot_manager_db(app_module.project_manager, snapshot_path)
                with open(snapshot_path, "rb") as f:
                    archive_db_bytes = f.read()

                archive_path = os.path.join(app_module.SAVED_PROJECTS_DIR, "demo.zip")
                _write_project_zip(
                    archive_path,
                    {
                        "db/chunks.sqlite3": archive_db_bytes,
                        "state.json": {
                            "input_file_path": "uploads/story.txt",
                            "render_prep_complete": False,
                            "processing_stage_markers": {"review": {"completed_at": 1}},
                            "new_mode_stage_markers": {"proofread": {"completed_at": 1}},
                        },
                        "uploads/story.txt": b"story",
                        "clone_voices/manifest.json": [{"id": "imported", "filename": "imported.wav", "name": "Imported"}],
                        "clone_voices/imported.wav": b"voice",
                    },
                )

                asyncio.run(app_module.load_script(app_module.ScriptLoadRequest(name="demo")))

                self.assertTrue(os.path.exists(os.path.join(app_module.CLONE_VOICES_DIR, "keep.wav")))
                self.assertTrue(os.path.exists(os.path.join(app_module.CLONE_VOICES_DIR, "imported.wav")))
                with open(app_module.CLONE_VOICES_MANIFEST, "r", encoding="utf-8") as f:
                    manifest = json.load(f)
                ids = {entry["id"] for entry in manifest}
                self.assertEqual(ids, {"keep", "imported"})
                with open(os.path.join(app_module.ROOT_DIR, "state.json"), "r", encoding="utf-8") as f:
                    state = json.load(f)
                self.assertEqual(state["loaded_project_name"], "demo")
                self.assertEqual(state["loaded_script_name"], "story")
                self.assertEqual(state["input_file_path"], os.path.join(app_module.UPLOADS_DIR, "story.txt"))
                self.assertFalse(state["render_prep_complete"])
                self.assertEqual(list(state["processing_stage_markers"].keys()), ["script", "voices"])
                self.assertEqual(
                    list(state["new_mode_stage_markers"].keys()),
                    ["process_paragraphs", "assign_dialogue", "extract_temperament", "create_script", "process_voices"],
                )

    def test_save_script_uses_uploaded_source_name_when_request_name_is_blank(self):
        with tempfile.TemporaryDirectory() as temp_root:
            with _TempProjectRuntime(self, temp_root):
                story_path = os.path.join(app_module.UPLOADS_DIR, "My Story.txt")
                with open(story_path, "w", encoding="utf-8") as f:
                    f.write("story")
                with open(os.path.join(app_module.ROOT_DIR, "state.json"), "w", encoding="utf-8") as f:
                    json.dump({"input_file_path": story_path}, f)
                app_module.project_manager.script_store.replace_script_document(
                    entries=[{"speaker": "Narrator", "text": "hello"}],
                    dictionary=[],
                    sanity_cache={"phrase_decisions": {}},
                    reason="test_seed_script",
                    rebuild_chunks=True,
                    wait=True,
                )

                result = asyncio.run(app_module.save_script(app_module.ScriptSaveRequest(name="")))

                self.assertEqual(result["status"], "saved")
                self.assertEqual(result["name"], "my_story")
                self.assertTrue(os.path.exists(os.path.join(app_module.SAVED_PROJECTS_DIR, "my_story.zip")))

    def test_named_project_archive_restores_through_zip_validation_path(self):
        with tempfile.TemporaryDirectory() as temp_root:
            with _TempProjectRuntime(self, temp_root):
                archive_path = os.path.join(app_module.SAVED_PROJECTS_DIR, "demo.zip")
                _write_project_zip(
                    archive_path,
                    {
                        "db/chunks.sqlite3": _sqlite_bytes(),
                        "state.json": {},
                    },
                )

                app_module._restore_project_archive_zip(archive_path)

                self.assertTrue(os.path.exists(app_module.project_manager.chunks_db_path))

    def test_generate_script_rejects_duplicate_start_while_running(self):
        with tempfile.TemporaryDirectory() as temp_root:
            state_path = os.path.join(temp_root, "state.json")
            with open(state_path, "w", encoding="utf-8") as f:
                json.dump({"input_file_path": os.path.join(temp_root, "uploads", "story.txt")}, f)

            original_root = app_module.ROOT_DIR
            original_base = app_module.BASE_DIR
            original_process_state = app_module.process_state["script"].copy()
            try:
                app_module.ROOT_DIR = temp_root
                app_module.BASE_DIR = temp_root
                app_module.process_state["script"]["running"] = True
                with self.assertRaises(HTTPException) as ctx:
                    asyncio.run(app_module.generate_script(app_module.ScriptGenerationRequest(), BackgroundTasks()))
                self.assertEqual(ctx.exception.status_code, 409)
                self.assertEqual(ctx.exception.detail, "Script generation is already running.")
            finally:
                app_module.ROOT_DIR = original_root
                app_module.BASE_DIR = original_base
                app_module.process_state["script"].clear()
                app_module.process_state["script"].update(original_process_state)


if __name__ == "__main__":
    unittest.main()
