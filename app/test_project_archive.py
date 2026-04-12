import importlib.util
import asyncio
import json
import os
import tempfile
import unittest
import zipfile
from fastapi import BackgroundTasks
from fastapi import HTTPException

MODULE_PATH = os.path.join(os.path.dirname(__file__), "app.py")
SPEC = importlib.util.spec_from_file_location("threadspeak_app_module_archive", MODULE_PATH)
app_module = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(app_module)


def _write_json(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f)


def _write_project_zip(path, files):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(
            app_module.PROJECT_ARCHIVE_MANIFEST_NAME,
            json.dumps(
                {
                    "kind": "threadspeak_project_archive",
                    "version": app_module.PROJECT_ARCHIVE_VERSION,
                    "created_at": 1,
                    "entries": sorted(files.keys()),
                }
            ),
        )
        for rel, payload in files.items():
            if isinstance(payload, bytes):
                zf.writestr(rel, payload)
            else:
                zf.writestr(rel, json.dumps(payload))


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
            "SCRIPT_PATH",
            "VOICE_CONFIG_PATH",
            "VOICES_PATH",
            "CHUNKS_PATH",
            "UPLOADS_DIR",
            "CLONE_VOICES_MANIFEST",
            "DESIGNED_VOICES_MANIFEST",
            "project_manager",
            "_any_project_task_running",
        ):
            self.originals[name] = getattr(app_module, name)

        app_module.ROOT_DIR = self.temp_root
        app_module.SCRIPTS_DIR = os.path.join(self.temp_root, "scripts")
        app_module.SAVED_PROJECTS_DIR = os.path.join(self.temp_root, "saved_projects")
        app_module.SCRIPT_PATH = os.path.join(self.temp_root, "annotated_script.json")
        app_module.VOICE_CONFIG_PATH = os.path.join(self.temp_root, "voice_config.json")
        app_module.VOICES_PATH = os.path.join(self.temp_root, "voices.json")
        app_module.CHUNKS_PATH = os.path.join(self.temp_root, "chunks.json")
        app_module.UPLOADS_DIR = os.path.join(self.temp_root, "uploads")
        app_module.CLONE_VOICES_MANIFEST = os.path.join(self.temp_root, "clone_voices", "manifest.json")
        app_module.DESIGNED_VOICES_MANIFEST = os.path.join(self.temp_root, "designed_voices", "manifest.json")
        app_module.project_manager = _StubProjectManager()
        app_module._any_project_task_running = lambda: None

        for dirname in ("scripts", "saved_projects", "uploads", "voicelines", "clone_voices", "designed_voices"):
            os.makedirs(os.path.join(self.temp_root, dirname), exist_ok=True)
        return self

    def __exit__(self, exc_type, exc, tb):
        for name, value in self.originals.items():
            setattr(app_module, name, value)
        return False


class ProjectArchiveHelpersTests(unittest.TestCase):
    def test_script_ingestion_preflight_warns_when_chunks_match_last_epub_chapter(self):
        with tempfile.TemporaryDirectory() as temp_root:
            input_path = os.path.join(temp_root, "uploads", "story.epub")
            os.makedirs(os.path.dirname(input_path), exist_ok=True)
            with open(input_path, "w", encoding="utf-8") as f:
                f.write("stub")
            with open(os.path.join(temp_root, "state.json"), "w", encoding="utf-8") as f:
                json.dump({"input_file_path": input_path}, f)
            with open(os.path.join(temp_root, "chunks.json"), "w", encoding="utf-8") as f:
                json.dump(
                    [
                        {"id": 0, "chapter": "Chapter 1"},
                        {"id": 1, "chapter": "Epilogue: Interview"},
                    ],
                    f,
                )

            original_root = app_module.ROOT_DIR
            original_chunks = app_module.CHUNKS_PATH
            original_loader = app_module.load_source_document
            try:
                app_module.ROOT_DIR = temp_root
                app_module.CHUNKS_PATH = os.path.join(temp_root, "chunks.json")
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
                app_module.CHUNKS_PATH = original_chunks
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
            with open(os.path.join(scripts_dir, "demo.json"), "w", encoding="utf-8") as f:
                json.dump({"entries": [], "dictionary": []}, f)
            with open(os.path.join(scripts_dir, "demo.voice_config.json"), "w", encoding="utf-8") as f:
                json.dump({"Narrator": {}}, f)
            with open(os.path.join(temp_root, "chunks.json"), "w", encoding="utf-8") as f:
                json.dump([{"id": 0}], f)
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
            original_script = app_module.SCRIPT_PATH
            original_voice_config = app_module.VOICE_CONFIG_PATH
            original_chunks = app_module.CHUNKS_PATH
            original_audio_state = app_module.process_state["audio"].copy()
            with app_module.audio_queue_lock:
                original_audio_queue = list(app_module.audio_queue)
                original_audio_current = app_module.audio_current_job
            try:
                app_module.ROOT_DIR = temp_root
                app_module.SCRIPTS_DIR = scripts_dir
                app_module.SCRIPT_PATH = os.path.join(temp_root, "annotated_script.json")
                app_module.VOICE_CONFIG_PATH = os.path.join(temp_root, "voice_config.json")
                app_module.CHUNKS_PATH = os.path.join(temp_root, "chunks.json")
                app_module.process_state["audio"]["running"] = False
                with app_module.audio_queue_lock:
                    app_module.audio_queue.clear()
                    app_module.audio_current_job = None

                result = asyncio.run(app_module.load_script(app_module.ScriptLoadRequest(name="demo")))
                self.assertEqual(result["status"], "loaded")

                with open(os.path.join(temp_root, "state.json"), "r", encoding="utf-8") as f:
                    state = json.load(f)
                self.assertFalse(state["render_prep_complete"])
                self.assertEqual(list(state["processing_stage_markers"].keys()), ["script"])
                self.assertFalse(os.path.exists(os.path.join(temp_root, "chunks.json")))
            finally:
                app_module.ROOT_DIR = original_root
                app_module.SCRIPTS_DIR = original_scripts
                app_module.SCRIPT_PATH = original_script
                app_module.VOICE_CONFIG_PATH = original_voice_config
                app_module.CHUNKS_PATH = original_chunks
                app_module.process_state["audio"].clear()
                app_module.process_state["audio"].update(original_audio_state)
                with app_module.audio_queue_lock:
                    app_module.audio_queue[:] = original_audio_queue
                    app_module.audio_current_job = original_audio_current

    def test_load_script_ignores_stale_audio_running_flag_when_no_job_or_queue(self):
        with tempfile.TemporaryDirectory() as temp_root:
            scripts_dir = os.path.join(temp_root, "scripts")
            os.makedirs(scripts_dir, exist_ok=True)
            with open(os.path.join(scripts_dir, "demo.json"), "w", encoding="utf-8") as f:
                json.dump({"entries": [], "dictionary": []}, f)
            with open(os.path.join(temp_root, "state.json"), "w", encoding="utf-8") as f:
                json.dump({"render_prep_complete": True}, f)

            original_root = app_module.ROOT_DIR
            original_scripts = app_module.SCRIPTS_DIR
            original_script = app_module.SCRIPT_PATH
            original_chunks = app_module.CHUNKS_PATH
            original_audio_state = app_module.process_state["audio"].copy()
            with app_module.audio_queue_lock:
                original_audio_queue = list(app_module.audio_queue)
                original_audio_current = app_module.audio_current_job
            try:
                app_module.ROOT_DIR = temp_root
                app_module.SCRIPTS_DIR = scripts_dir
                app_module.SCRIPT_PATH = os.path.join(temp_root, "annotated_script.json")
                app_module.CHUNKS_PATH = os.path.join(temp_root, "chunks.json")

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
                app_module.SCRIPT_PATH = original_script
                app_module.CHUNKS_PATH = original_chunks
                app_module.process_state["audio"].clear()
                app_module.process_state["audio"].update(original_audio_state)
                with app_module.audio_queue_lock:
                    app_module.audio_queue[:] = original_audio_queue
                    app_module.audio_current_job = original_audio_current

    def test_normalize_archive_path_rejects_parent_traversal(self):
        with self.assertRaises(ValueError):
            app_module._normalize_archive_path("../secret.txt")

    def test_allowed_archive_paths_cover_expected_project_content(self):
        self.assertTrue(app_module._is_allowed_project_archive_path("annotated_script.json"))
        self.assertTrue(app_module._is_allowed_project_archive_path("voicelines/chunk_001.mp3"))
        self.assertTrue(app_module._is_allowed_project_archive_path("voicelines/discarded/rejected_001.mp3"))
        self.assertTrue(app_module._is_allowed_project_archive_path("clone_voices/manifest.json"))
        self.assertTrue(app_module._is_allowed_project_archive_path("transcription_cache.json"))
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

    def test_project_archive_entries_include_timeline_audio_voice_assets_discarded_pool_and_cache(self):
        with tempfile.TemporaryDirectory() as temp_root:
            for dirname in ("voicelines", "clone_voices", "designed_voices", "uploads"):
                os.makedirs(os.path.join(temp_root, dirname), exist_ok=True)
            os.makedirs(os.path.join(temp_root, "voicelines", "discarded"), exist_ok=True)

            files = {
                "annotated_script.json": {"entries": [], "dictionary": []},
                "voice_config.json": {
                    "Narrator": {"ref_audio": "clone_voices/current_clone.wav"},
                    "Alice": {"ref_audio": "designed_voices/current_design.wav"},
                },
                "voices.json": ["Narrator", "Alice"],
                "chunks.json": [
                    {"id": 0, "audio_path": "voicelines/live_a.mp3"},
                    {"id": 1, "audio_path": "voicelines/live_b.mp3"},
                    {"id": 2, "audio_path": None},
                ],
                "script_sanity_check.json": {"ok": True},
                "state.json": {"input_file_path": os.path.join(temp_root, "uploads", "story.txt")},
                "transcription_cache.json": {
                    "entries": [
                        {
                            "filename": "voicelines/live_a.mp3",
                            "size_bytes": 4,
                            "text": "cached transcript",
                            "normalized_text": "cached transcript",
                        }
                    ]
                },
            }
            for rel, payload in files.items():
                with open(os.path.join(temp_root, rel), "w", encoding="utf-8") as f:
                    json.dump(payload, f)

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

            original_root = app_module.ROOT_DIR
            original_uploads = app_module.UPLOADS_DIR
            original_chunks = app_module.CHUNKS_PATH
            original_voice_config = app_module.VOICE_CONFIG_PATH
            original_clone_manifest = getattr(app_module, "CLONE_VOICES_MANIFEST", None)
            original_design_manifest = getattr(app_module, "DESIGNED_VOICES_MANIFEST", None)
            try:
                app_module.ROOT_DIR = temp_root
                app_module.CHUNKS_PATH = os.path.join(temp_root, "chunks.json")
                app_module.VOICE_CONFIG_PATH = os.path.join(temp_root, "voice_config.json")
                app_module.UPLOADS_DIR = os.path.join(temp_root, "uploads")
                app_module.CLONE_VOICES_MANIFEST = os.path.join(temp_root, "clone_voices", "manifest.json")
                app_module.DESIGNED_VOICES_MANIFEST = os.path.join(temp_root, "designed_voices", "manifest.json")
                entries = dict(app_module._project_archive_entries())
            finally:
                app_module.ROOT_DIR = original_root
                app_module.CHUNKS_PATH = original_chunks
                app_module.VOICE_CONFIG_PATH = original_voice_config
                app_module.UPLOADS_DIR = original_uploads
                app_module.CLONE_VOICES_MANIFEST = original_clone_manifest
                app_module.DESIGNED_VOICES_MANIFEST = original_design_manifest

            self.assertIn("voicelines/live_a.mp3", entries)
            self.assertIn("voicelines/live_b.mp3", entries)
            self.assertNotIn("voicelines/orphan.mp3", entries)
            self.assertIn("voicelines/discarded/rejected.mp3", entries)
            self.assertIn("clone_voices/manifest.json", entries)
            self.assertIn("clone_voices/current_clone.wav", entries)
            self.assertNotIn("clone_voices/orphan_clone.wav", entries)
            self.assertIn("designed_voices/manifest.json", entries)
            self.assertIn("designed_voices/current_design.wav", entries)
            self.assertNotIn("designed_voices/orphan_design.wav", entries)
            self.assertIn("uploads/story.txt", entries)
            self.assertIn("transcription_cache.json", entries)

    def test_restore_project_archive_restores_discarded_pool_and_transcription_cache(self):
        with tempfile.TemporaryDirectory() as temp_root:
            extract_root = os.path.join(temp_root, "extracted")
            os.makedirs(os.path.join(extract_root, "voicelines", "discarded"), exist_ok=True)
            os.makedirs(os.path.join(extract_root, "uploads"), exist_ok=True)
            os.makedirs(os.path.join(extract_root, "clone_voices"), exist_ok=True)
            os.makedirs(os.path.join(extract_root, "designed_voices"), exist_ok=True)

            with open(os.path.join(extract_root, "chunks.json"), "w", encoding="utf-8") as f:
                json.dump([{"id": 0, "audio_path": None}], f)
            with open(os.path.join(extract_root, "transcription_cache.json"), "w", encoding="utf-8") as f:
                json.dump(
                    {"entries": [{"filename": "voicelines/discarded/rejected.mp3", "size_bytes": 4, "text": "cached"}]},
                    f,
                )
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

                self.assertTrue(os.path.exists(os.path.join(temp_root, "transcription_cache.json")))
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
                _write_json(
                    app_module.CHUNKS_PATH,
                    [
                        {"id": 0, "audio_path": "voicelines/missing.mp3"},
                        {"id": 1, "audio_path": None},
                    ],
                )
                self.assertFalse(app_module._project_has_generated_audio())

                audio_path = os.path.join(temp_root, "voicelines", "existing.mp3")
                with open(audio_path, "wb") as f:
                    f.write(b"audio")
                _write_json(app_module.CHUNKS_PATH, [{"id": 0, "audio_path": "voicelines/existing.mp3"}])
                self.assertTrue(app_module._project_has_generated_audio())

    def test_unified_save_uses_script_snapshot_without_audio_and_removes_same_name_zip(self):
        with tempfile.TemporaryDirectory() as temp_root:
            with _TempProjectRuntime(self, temp_root):
                _write_json(app_module.SCRIPT_PATH, {"entries": [], "dictionary": []})
                _write_json(app_module.VOICE_CONFIG_PATH, {"Narrator": {}})
                _write_json(app_module.CHUNKS_PATH, [{"id": 0, "audio_path": "voicelines/missing.mp3"}])
                stale_zip = os.path.join(app_module.SAVED_PROJECTS_DIR, "demo.zip")
                _write_project_zip(stale_zip, {"state.json": {}})

                result = asyncio.run(app_module.save_script(app_module.ScriptSaveRequest(name="Demo")))

                self.assertEqual(result["kind"], "script")
                self.assertTrue(os.path.exists(os.path.join(app_module.SCRIPTS_DIR, "demo.json")))
                self.assertTrue(os.path.exists(os.path.join(app_module.SCRIPTS_DIR, "demo.voice_config.json")))
                self.assertFalse(os.path.exists(stale_zip))

    def test_unified_save_uses_project_zip_with_audio_and_removes_script_artifacts(self):
        with tempfile.TemporaryDirectory() as temp_root:
            with _TempProjectRuntime(self, temp_root):
                _write_json(app_module.SCRIPT_PATH, {"entries": [{"text": "hello"}], "dictionary": []})
                _write_json(app_module.VOICE_CONFIG_PATH, {"Narrator": {}})
                _write_json(app_module.VOICES_PATH, ["Narrator"])
                _write_json(app_module.CHUNKS_PATH, [{"id": 0, "audio_path": "voicelines/live.mp3"}])
                with open(os.path.join(temp_root, "voicelines", "live.mp3"), "wb") as f:
                    f.write(b"audio")
                for suffix, payload in (
                    (".json", {"old": True}),
                    (".voice_config.json", {"old": True}),
                    (".paragraphs.json", {"old": True}),
                ):
                    _write_json(os.path.join(app_module.SCRIPTS_DIR, f"demo{suffix}"), payload)

                result = asyncio.run(app_module.save_script(app_module.ScriptSaveRequest(name="Demo")))

                archive_path = os.path.join(app_module.SAVED_PROJECTS_DIR, "demo.zip")
                self.assertEqual(result["kind"], "project")
                self.assertTrue(os.path.exists(archive_path))
                self.assertFalse(os.path.exists(os.path.join(app_module.SCRIPTS_DIR, "demo.json")))
                self.assertFalse(os.path.exists(os.path.join(app_module.SCRIPTS_DIR, "demo.voice_config.json")))
                with zipfile.ZipFile(archive_path, "r") as zf:
                    self.assertIn(app_module.PROJECT_ARCHIVE_MANIFEST_NAME, zf.namelist())
                    self.assertIn("voicelines/live.mp3", zf.namelist())

    def test_unified_list_deduplicates_script_when_project_zip_exists(self):
        with tempfile.TemporaryDirectory() as temp_root:
            with _TempProjectRuntime(self, temp_root):
                _write_json(os.path.join(app_module.SCRIPTS_DIR, "demo.json"), {"entries": []})
                _write_json(os.path.join(app_module.SCRIPTS_DIR, "demo.voice_config.json"), {"Narrator": {}})
                _write_json(os.path.join(app_module.SCRIPTS_DIR, "script_only.json"), {"entries": []})
                _write_project_zip(
                    os.path.join(app_module.SAVED_PROJECTS_DIR, "demo.zip"),
                    {"voice_config.json": {"Narrator": {}}, "chunks.json": []},
                )

                projects = asyncio.run(app_module.list_saved_scripts())
                by_name = {item["name"]: item for item in projects}

                self.assertEqual(by_name["demo"]["kind"], "project")
                self.assertTrue(by_name["demo"]["has_audio"])
                self.assertTrue(by_name["demo"]["has_voice_config"])
                self.assertEqual(by_name["script_only"]["kind"], "script")
                self.assertEqual(len([item for item in projects if item["name"] == "demo"]), 1)

    def test_unified_load_prefers_project_zip_over_script_snapshot(self):
        with tempfile.TemporaryDirectory() as temp_root:
            with _TempProjectRuntime(self, temp_root):
                _write_json(os.path.join(app_module.SCRIPTS_DIR, "demo.json"), {"entries": [{"text": "script"}]})
                _write_project_zip(
                    os.path.join(app_module.SAVED_PROJECTS_DIR, "demo.zip"),
                    {
                        "annotated_script.json": {"entries": [{"text": "zip"}], "dictionary": []},
                        "state.json": {"input_file_path": "uploads/story.txt"},
                        "uploads/story.txt": b"story",
                    },
                )

                result = asyncio.run(app_module.load_script(app_module.ScriptLoadRequest(name="Demo")))

                self.assertEqual(result["kind"], "project")
                with open(app_module.SCRIPT_PATH, "r", encoding="utf-8") as f:
                    restored = json.load(f)
                self.assertEqual(restored["entries"][0]["text"], "zip")

    def test_unified_delete_removes_project_zip_and_script_companions(self):
        with tempfile.TemporaryDirectory() as temp_root:
            with _TempProjectRuntime(self, temp_root):
                _write_json(os.path.join(app_module.SCRIPTS_DIR, "demo.json"), {"entries": []})
                _write_json(os.path.join(app_module.SCRIPTS_DIR, "demo.voice_config.json"), {"Narrator": {}})
                _write_json(os.path.join(app_module.SCRIPTS_DIR, "demo.paragraphs.json"), [])
                with open(os.path.join(app_module.SCRIPTS_DIR, "demo.source.txt"), "w", encoding="utf-8") as f:
                    f.write("source")
                _write_project_zip(os.path.join(app_module.SAVED_PROJECTS_DIR, "demo.zip"), {"state.json": {}})

                result = asyncio.run(app_module.delete_script("Demo"))

                self.assertEqual(result["status"], "deleted")
                self.assertFalse(os.path.exists(os.path.join(app_module.SAVED_PROJECTS_DIR, "demo.zip")))
                self.assertFalse(os.path.exists(os.path.join(app_module.SCRIPTS_DIR, "demo.json")))
                self.assertFalse(os.path.exists(os.path.join(app_module.SCRIPTS_DIR, "demo.voice_config.json")))
                self.assertFalse(os.path.exists(os.path.join(app_module.SCRIPTS_DIR, "demo.paragraphs.json")))
                self.assertFalse(os.path.exists(os.path.join(app_module.SCRIPTS_DIR, "demo.source.txt")))

    def test_named_project_archive_restores_through_zip_validation_path(self):
        with tempfile.TemporaryDirectory() as temp_root:
            with _TempProjectRuntime(self, temp_root):
                archive_path = os.path.join(app_module.SAVED_PROJECTS_DIR, "demo.zip")
                _write_project_zip(
                    archive_path,
                    {
                        "annotated_script.json": {"entries": [{"text": "restored"}], "dictionary": []},
                        "state.json": {},
                    },
                )

                app_module._restore_project_archive_zip(archive_path)

                with open(app_module.SCRIPT_PATH, "r", encoding="utf-8") as f:
                    restored = json.load(f)
                self.assertEqual(restored["entries"][0]["text"], "restored")

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
