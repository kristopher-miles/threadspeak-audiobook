from fastapi import APIRouter
from .. import shared as _shared

globals().update({k: v for k, v in vars(_shared).items() if not k.startswith("__")})

router = APIRouter()

class NavTaskRequest(BaseModel):
    tab: Optional[str] = None


class ResetNewModeRequest(BaseModel):
    preserve_voices: bool = False


class AssignDialogueRequest(BaseModel):
    full_cast: bool = True


def _ensure_new_mode_workflow_inactive(conflict_message: str = "Cannot run this step while Process Script is active. Pause or wait for new-mode workflow first."):
    state = process_state.get("new_mode_workflow") or {}
    if bool(state.get("running")) or bool(state.get("paused")):
        raise HTTPException(status_code=409, detail=conflict_message)

@router.get("/api/nav_task")
async def get_nav_task():
    return _get_nav_task_state()

@router.post("/api/nav_task/set")
async def set_nav_task(request: NavTaskRequest):
    if not request.tab:
        raise HTTPException(status_code=400, detail="Missing navigation task tab")
    return _set_nav_task_tab(request.tab)

@router.post("/api/nav_task/release")
async def release_nav_task(request: NavTaskRequest = NavTaskRequest()):
    return _release_nav_task_tab(request.tab)

@router.post("/api/reset_project")
async def reset_project():
    global audio_current_job, audio_recovery_request
    _assert_test_safe_runtime_target(
        "reset_project",
        ROOT_DIR=ROOT_DIR,
        VOICELINES_DIR=VOICELINES_DIR,
        UPLOADS_DIR=UPLOADS_DIR,
        PROCESSING_WORKFLOW_STATE_PATH=PROCESSING_WORKFLOW_STATE_PATH,
        NEW_MODE_WORKFLOW_STATE_PATH=NEW_MODE_WORKFLOW_STATE_PATH,
        AUDIO_QUEUE_STATE_PATH=AUDIO_QUEUE_STATE_PATH,
    )
    _release_nav_task_tab()
    # Hard-stop everything before nuking project artifacts.
    with task_state_lock:
        for task_name, process in list(task_processes.items()):
            try:
                if process and process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=0.5)
                    except Exception:
                        process.kill()
            except Exception:
                pass
        task_processes.clear()

        # Invalidate all active run IDs so cooperative tasks stop at the next check.
        for state in process_state.values():
            if isinstance(state, dict) and "run_id" in state:
                state["run_id"] = str(uuid.uuid4())

    with audio_queue_condition:
        now = time.time()
        while audio_queue:
            job = audio_queue.pop(0)
            job["status"] = "cancelled"
            job["finished_at"] = now
            _record_audio_recent_job_locked(job)

        process_state["audio"]["cancel"] = True
        audio_cancel_event.set()
        active_audio_job = _shared.audio_current_job
        if active_audio_job is not None:
            _abandon_audio_job_locked(
                active_audio_job,
                active_audio_job.get("run_token"),
                "Project reset requested",
                status="cancelled",
            )
        # Ensure reset always clears worker pointers, even if abandon raced.
        _shared.audio_current_job = None
        _shared.audio_recovery_request = None
        audio_current_job = None
        audio_recovery_request = None
        audio_cancel_event.clear()
        _refresh_audio_process_state_locked(persist=True)

    # Signal cancellation for thread-based tasks that poll a cancel flag.
    if "dataset_builder" in process_state and isinstance(process_state["dataset_builder"], dict):
        process_state["dataset_builder"]["cancel"] = True

    removed = []
    state_path = os.path.join(ROOT_DIR, "state.json")
    state_data = {}
    if os.path.exists(state_path):
        try:
            with open(state_path, "r", encoding="utf-8") as f:
                state_data = json.load(f)
        except (json.JSONDecodeError, ValueError, OSError):
            state_data = {}

    input_file_path = state_data.get("input_file_path") or ""
    if input_file_path:
        try:
            if os.path.commonpath([os.path.abspath(input_file_path), os.path.abspath(UPLOADS_DIR)]) == os.path.abspath(UPLOADS_DIR) and os.path.exists(input_file_path):
                os.remove(input_file_path)
                removed.append(os.path.basename(input_file_path))
        except ValueError:
            pass

    paths_to_report = [
        state_path,
        getattr(project_manager, "chunks_db_path", os.path.join(ROOT_DIR, "chunks.sqlite3")),
        f"{getattr(project_manager, 'chunks_db_path', os.path.join(ROOT_DIR, 'chunks.sqlite3'))}-wal",
        f"{getattr(project_manager, 'chunks_db_path', os.path.join(ROOT_DIR, 'chunks.sqlite3'))}-shm",
        getattr(project_manager, "chunks_queue_log_path", os.path.join(ROOT_DIR, "chunks.queue.log")),
        SCRIPT_REPAIR_TRACE_PATH,
        AUDIOBOOK_PATH,
        M4B_PATH,
        AUDIO_QUEUE_STATE_PATH,
        AUDIO_CANCEL_TOMBSTONE_PATH,
        PROCESSING_WORKFLOW_STATE_PATH,
        NEW_MODE_WORKFLOW_STATE_PATH,
        _project_export_filesystem_path("audacity_export.zip"),
        _project_export_filesystem_path("m4b_cover.jpg"),
        LAYOUT.script_generation_checkpoint_path,
        LAYOUT.script_review_checkpoint_path,
        os.path.join(ROOT_DIR, "logs", "llm_responses.log"),
        os.path.join(ROOT_DIR, "logs", "review_responses.log"),
    ]

    for path in paths_to_report:
        if os.path.exists(path):
            removed.append(os.path.basename(path))
    for dirname in (VOICELINES_DIR, UPLOADS_DIR):
        if os.path.isdir(dirname) and os.listdir(dirname):
            removed.append(f"{os.path.basename(dirname)}/*")

    _clear_project_derived_state(
        preserve_input_file=False,
        preserve_reusable_voices=True,
    )

    with audio_queue_condition:
        audio_queue.clear()
        _shared.audio_current_job = None
        _shared.audio_recovery_request = None
        audio_current_job = None
        audio_recovery_request = None
        process_state["audio"]["cancel"] = False
        audio_cancel_event.clear()
        process_state["audio"]["queue"] = []
        process_state["audio"]["current_job"] = None
        process_state["audio"]["recent_jobs"] = []
        process_state["audio"]["logs"] = []
        process_state["audio"]["running"] = False
        process_state["audio"]["merge_running"] = False
        process_state["audio"]["metrics"] = _new_audio_metrics()
        process_state["audio"]["heartbeat"] = _new_audio_heartbeat_state()
        _refresh_audio_process_state_locked(persist=False)

    with project_manager._transcription_cache_lock:
        project_manager._transcription_cache = None

    for task_name in ("script", "voices", "proofread", "review", "sanity", "repair", "audacity_export", "m4b_export",
                      "process_paragraphs", "assign_dialogue", "extract_temperament", "create_script"):
        process_state[task_name]["logs"] = []
        process_state[task_name]["running"] = False
        if "progress" in process_state[task_name]:
            process_state[task_name]["progress"] = {}

    with processing_workflow_lock:
        process_state["processing_workflow"] = _new_processing_workflow_state()
        _persist_processing_workflow_state_locked()

    with new_mode_workflow_lock:
        process_state["new_mode_workflow"] = _new_mode_workflow_initial_state()
        _persist_new_mode_workflow_state_locked()

    project_manager.engine = None
    project_manager.asr_engine = None

    logger.info("Project state reset")
    return {"status": "reset", "removed": sorted(set(removed))}

@router.post("/api/assign_dialogue")
async def start_assign_dialogue(background_tasks: BackgroundTasks, request: AssignDialogueRequest = AssignDialogueRequest()):
    _ensure_new_mode_workflow_inactive()
    _ensure_task_not_running("assign_dialogue", "Dialogue assignment is already running.")

    pdata = _load_project_paragraphs_document()
    if not pdata.get("paragraphs"):
        raise HTTPException(
            status_code=400,
            detail="Paragraph data is empty or corrupt. Re-run 'Process Paragraphs'.",
        )

    config_path = os.path.join(BASE_DIR, "config.json")
    run_id = _start_task_run("assign_dialogue")
    background_tasks.add_task(_run_assign_dialogue_task, run_id, config_path, bool(request.full_cast))
    return {"status": "started", "run_id": run_id}


@router.post("/api/extract_temperament")
async def start_extract_temperament(background_tasks: BackgroundTasks):
    _ensure_new_mode_workflow_inactive()
    _ensure_task_not_running("extract_temperament", "Temperament extraction is already running.")

    pdata = _load_project_paragraphs_document()
    if not pdata.get("paragraphs"):
        raise HTTPException(
            status_code=400,
            detail="Paragraph data is empty or corrupt. Re-run 'Process Paragraphs'.",
        )

    config_path = os.path.join(BASE_DIR, "config.json")
    run_id = _start_task_run("extract_temperament")
    background_tasks.add_task(_run_extract_temperament_task, run_id, config_path)
    return {"status": "started", "run_id": run_id}


@router.get("/api/script_info")
async def get_script_info():
    """Return a lightweight summary of the current script state."""
    if hasattr(project_manager, "_load_voice_config"):
        try:
            voice_config = project_manager._load_voice_config()
        except Exception:
            voice_config = {}
    else:
        voice_config = {}
    voice_count = len(voice_config) if isinstance(voice_config, dict) else 0
    has_voice_config = voice_count > 0

    has_voicelines = False
    if os.path.isdir(VOICELINES_DIR):
        try:
            for entry in os.listdir(VOICELINES_DIR):
                if not entry.startswith("."):
                    has_voicelines = True
                    break
        except Exception:
            pass

    if not _project_has_script_document():
        return {
            "entry_count": 0,
            "has_voice_config": has_voice_config,
            "voice_count": voice_count,
            "has_voicelines": has_voicelines,
        }
    try:
        entries = _load_project_script_document().get("entries", [])
        return {
            "entry_count": len(entries) if isinstance(entries, list) else 0,
            "has_voice_config": has_voice_config,
            "voice_count": voice_count,
            "has_voicelines": has_voicelines,
        }
    except Exception:
        return {
            "entry_count": 0,
            "has_voice_config": has_voice_config,
            "voice_count": voice_count,
            "has_voicelines": has_voicelines,
        }


def _run_create_script_task_with_new_mode_state(
    run_id: str,
):
    _run_create_script_task(run_id)
    _mark_new_mode_stage_completed_marker("create_script")
    with new_mode_workflow_lock:
        options = process_state["new_mode_workflow"].get("options") or {}
        process_state["new_mode_workflow"]["completed_stages"] = _derived_new_mode_completed_stages(options)
        _persist_new_mode_workflow_state_locked()


@router.get("/api/pipeline_step_status")
async def get_pipeline_step_status():
    """Return DB-backed completion status for the 4 new-mode pipeline steps."""

    # Check if an input file is loaded
    has_input_file = False
    state_path = os.path.join(ROOT_DIR, "state.json")
    if os.path.exists(state_path):
        try:
            with open(state_path, "r", encoding="utf-8") as f:
                state_data = json.load(f)
            input_file = state_data.get("input_file_path", "")
            has_input_file = bool(input_file and os.path.exists(input_file))
        except Exception:
            pass

    result = {
        "has_input_file": has_input_file,
        "process_paragraphs": "not_started",
        "assign_dialogue": "not_started",
        "extract_temperament": "not_started",
        "create_script": "not_started",
    }

    # A durable script project should keep the pipeline unlocked even if
    # DB-backed paragraph/script state may exist even when legacy intermediates are absent.
    if _project_script_complete_detected():
        result.update(
            {
                "process_paragraphs": "complete",
                "assign_dialogue": "complete",
                "extract_temperament": "complete",
                "create_script": "complete",
            }
        )
        return result

    pdata = _load_project_paragraphs_document()
    paragraphs = pdata.get("paragraphs", [])

    if not paragraphs:
        return result

    result["process_paragraphs"] = "complete"

    # Assign Dialogue: check first/last paragraph that has dialogue
    dialogue_paras = [p for p in paragraphs if p.get("has_dialogue")]
    if dialogue_paras:
        if "speakers" in dialogue_paras[-1]:
            result["assign_dialogue"] = "complete"
        elif "speakers" in dialogue_paras[0]:
            result["assign_dialogue"] = "incomplete"
    else:
        # No dialogue paragraphs — trivially complete
        result["assign_dialogue"] = "complete"

    if result["assign_dialogue"] != "complete":
        return result

    # Extract Temperament: check first/last paragraph for 'tone'
    if "tone" in paragraphs[-1]:
        result["extract_temperament"] = "complete"
    elif "tone" in paragraphs[0]:
        result["extract_temperament"] = "incomplete"

    if result["extract_temperament"] != "complete":
        return result

    if _project_has_script_document():
        result["create_script"] = "complete"

    return result


@router.post("/api/reset_new_mode")
async def reset_new_mode(request: ResetNewModeRequest = ResetNewModeRequest()):
    """Clear script artifacts so Create Script can start fresh."""
    _assert_test_safe_runtime_target(
        "reset_new_mode",
        ROOT_DIR=ROOT_DIR,
        VOICELINES_DIR=VOICELINES_DIR,
    )
    removed = []
    if getattr(project_manager, "script_store", None) is not None:
        project_manager.script_store.replace_script_document(
            entries=[],
            dictionary=[],
            sanity_cache={"phrase_decisions": {}},
            reason="reset_new_mode_script_clear",
            rebuild_chunks=True,
            wait=True,
        )
        project_manager.script_store.delete_project_document("paragraphs", reason="reset_new_mode", wait=True)
        project_manager.script_store.delete_project_document("script_sanity_result", reason="reset_new_mode", wait=True)
        removed.extend(["script_entries", "chunks", "paragraphs", "script_sanity_result"])

    if not request.preserve_voices:
        try:
            if hasattr(project_manager, "reset_voice_state"):
                project_manager.reset_voice_state(reason="reset_new_mode")
        except Exception:
            pass
        removed.append("voice_profiles")

    _clear_directory_contents(VOICELINES_DIR)
    removed.append("voicelines/*")

    # Also reset the in-memory task state so _ensure_task_not_running won't block
    with task_state_lock:
        state = process_state.get("create_script")
        if state:
            state["running"] = False
            state["logs"] = []
            state["progress"] = {}
    _clear_new_mode_stage_markers()
    with new_mode_workflow_lock:
        process_state["new_mode_workflow"]["completed_stages"] = []
        _persist_new_mode_workflow_state_locked()
    return {
        "status": "reset",
        "removed": removed,
        "preserved_voices": bool(request.preserve_voices),
    }


@router.post("/api/create_script")
async def start_create_script(background_tasks: BackgroundTasks):
    _ensure_new_mode_workflow_inactive()
    _ensure_task_not_running("create_script", "Script creation is already running.")

    pdata = _load_project_paragraphs_document()
    if not pdata.get("paragraphs"):
        raise HTTPException(
            status_code=400,
            detail="Paragraph data is empty or corrupt. Re-run 'Process Paragraphs'.",
        )

    run_id = _start_task_run("create_script")
    background_tasks.add_task(_run_create_script_task_with_new_mode_state, run_id)
    return {"status": "started", "run_id": run_id}


@router.post("/api/process_paragraphs")
async def start_process_paragraphs(background_tasks: BackgroundTasks):
    _ensure_new_mode_workflow_inactive()
    _ensure_task_not_running("process_paragraphs", "Paragraph processing is already running.")

    if _project_has_script_document():
        raise HTTPException(
            status_code=409,
            detail="Generating a new audiobook script requires erasing the old one first.",
        )

    # Resolve input file from state.json
    state_path = os.path.join(ROOT_DIR, "state.json")
    if not os.path.exists(state_path):
        raise HTTPException(status_code=400, detail="No input file selected. Please upload a book first.")
    with open(state_path, "r", encoding="utf-8") as f:
        state = json.load(f)
    input_file = state.get("input_file_path")
    if not input_file or not os.path.exists(input_file):
        raise HTTPException(status_code=400, detail="No input file found. Please upload a book first.")

    run_id = _start_task_run("process_paragraphs")
    background_tasks.add_task(_run_process_paragraphs_task, run_id, input_file)
    return {"status": "started", "run_id": run_id}


@router.post("/api/generate_script")
async def generate_script(request: Optional[ScriptGenerationRequest] = None, background_tasks: BackgroundTasks = None):
    request = request or ScriptGenerationRequest()
    _ensure_task_not_running("script", "Script generation is already running.")
    preflight = _script_ingestion_preflight_summary()
    if preflight.get("warn") and not request.force_reimport and not request.skip_import:
        raise HTTPException(
            status_code=409,
            detail={
                "message": preflight.get("message") or "Existing project matches the uploaded EPUB.",
                "code": "script_ingestion_conflict",
                "preflight": preflight,
            },
        )
    if request.skip_import:
        return _mark_script_stage_skipped_for_existing_project()
    if request.force_reimport:
        _clear_project_derived_state(preserve_input_file=True)

    # Get input file from state.json
    state_path = os.path.join(ROOT_DIR, "state.json")
    if not os.path.exists(state_path):
        raise HTTPException(status_code=400, detail="No input file selected")

    with open(state_path, "r", encoding="utf-8") as f:
        state = json.load(f)
        input_file = state.get("input_file_path")

    if not input_file:
         raise HTTPException(status_code=400, detail="No input file found in state")

    run_id = _start_task_run("script")
    background_tasks.add_task(_run_generate_script_task, run_id)
    return {"status": "started", "run_id": run_id}

@router.post("/api/review_script")
async def review_script(background_tasks: BackgroundTasks):
    if not _project_has_script_document():
        raise HTTPException(status_code=400, detail="No annotated script found. Generate a script first.")

    run_id = _start_task_run("review")
    background_tasks.add_task(_run_review_script_task, run_id)
    return {"status": "started", "run_id": run_id}

@router.post("/api/script_sanity_check")
async def script_sanity_check(background_tasks: BackgroundTasks):
    if not _project_has_script_document():
        raise HTTPException(status_code=400, detail="No annotated script found. Generate a script first.")

    run_id = _start_task_run("sanity")
    background_tasks.add_task(_run_sanity_task, run_id)
    return {"status": "started", "run_id": run_id}

@router.post("/api/replace_missing_chunks")
async def replace_missing_chunks(background_tasks: BackgroundTasks):
    if not _project_has_script_document():
        raise HTTPException(status_code=400, detail="No annotated script found. Generate a script first.")

    run_id = _start_task_run("repair")
    background_tasks.add_task(_run_repair_task, run_id)
    return {"status": "started", "run_id": run_id}

@router.get("/api/annotated_script")
async def get_annotated_script():
    """Return the current working annotated script entries from SQLite."""
    if not _project_has_script_document():
        raise HTTPException(status_code=404, detail="No annotated script found")
    return _load_project_script_document().get("entries", [])

@router.get("/api/script_sanity_check")
async def get_script_sanity_check():
    payload = _load_project_script_sanity_result()
    if not payload:
        raise HTTPException(status_code=404, detail="No sanity check results found")
    return payload

@router.get("/api/status/{task_name}")
async def get_status(task_name: str):
    if task_name not in process_state:
        raise HTTPException(status_code=404, detail="Task not found")
    if task_name == "audio":
        with audio_queue_lock:
            _refresh_audio_process_state_locked()
    return process_state[task_name]


@router.post("/api/processing/start")
async def start_processing_workflow(request: ProcessingWorkflowRequest):
    running_task = _any_project_task_running()
    with processing_workflow_lock:
        workflow_state = process_state["processing_workflow"]
        if workflow_state.get("running") and not workflow_state.get("paused"):
            raise HTTPException(status_code=409, detail="Processing workflow is already running.")

        if running_task and not workflow_state.get("paused"):
            raise HTTPException(status_code=409, detail=f"Cannot start processing while '{running_task}' is running.")

        options = {
            "process_voices": bool(request.process_voices),
            "generate_audio": bool(request.generate_audio),
        }
        if not workflow_state.get("paused"):
            preflight = _script_ingestion_preflight_summary()
            if preflight.get("warn") and not request.force_reimport and not request.skip_script_stage:
                raise HTTPException(
                    status_code=409,
                    detail={
                        "message": preflight.get("message") or "Existing project matches the uploaded EPUB.",
                        "code": "script_ingestion_conflict",
                        "preflight": preflight,
                    },
                )
            if request.force_reimport:
                _clear_project_derived_state(preserve_input_file=True)
            if request.skip_script_stage:
                _mark_processing_stage_completed_marker("script")

        if workflow_state.get("paused"):
            workflow_state["options"] = options
            workflow_state["running"] = True
            workflow_state["paused"] = False
            workflow_state["pause_requested"] = False
            workflow_state["last_error"] = None
            workflow_state["resume_count"] = int(workflow_state.get("resume_count", 0) or 0) + 1
            _append_processing_workflow_log_locked("Resuming processing workflow.")
        else:
            completed_stages = _derived_processing_completed_stages(options)
            process_state["processing_workflow"] = _new_processing_workflow_state() | {
                "running": True,
                "paused": False,
                "pause_requested": False,
                "options": options,
                "started_at": time.time(),
                "completed_stages": completed_stages,
            }
            _append_processing_workflow_log_locked("Starting processing workflow.")
            if completed_stages:
                labels = [PROCESSING_WORKFLOW_STAGE_LABELS.get(stage, stage) for stage in completed_stages]
                _append_processing_workflow_log_locked(
                    f"Skipping already completed stages: {', '.join(labels)}."
                )

        _start_processing_workflow_thread_locked()
        return process_state["processing_workflow"]


@router.post("/api/processing/pause")
async def pause_processing_workflow():
    with processing_workflow_lock:
        state = process_state["processing_workflow"]
        if not state.get("running"):
            if state.get("paused"):
                return {"status": "paused"}
            return {"status": "idle"}
        requested = _request_processing_workflow_pause_locked()
        stage_name = state.get("current_stage")

    if requested and stage_name:
        _request_active_stage_pause(stage_name)
    return {"status": "pause_requested", "current_stage": stage_name}


@router.post("/api/new_mode_workflow/start")
async def start_new_mode_workflow(request: NewModeWorkflowRequest):
    options = {
        "process_voices": bool(request.process_voices),
        "generate_audio": bool(request.generate_audio),
        "full_cast": bool(request.full_cast),
    }
    _initialize_new_mode_stage_markers(options=options)
    with new_mode_workflow_lock:
        state = process_state["new_mode_workflow"]
        if state.get("running") and not state.get("paused"):
            raise HTTPException(status_code=409, detail="New mode workflow is already running.")

        if state.get("paused"):
            # Resume: update options (in case toggle changed) but keep completed stages
            state["running"] = True
            state["paused"] = False
            state["pause_requested"] = False
            state["last_error"] = None
            state["options"] = options
            state["completed_stages"] = _derived_new_mode_completed_stages(options)
            _append_new_mode_workflow_log_locked("Resuming...")
        else:
            # Fresh start: derive complete stages from durable stage markers.
            completed = _derived_new_mode_completed_stages(options)
            process_state["new_mode_workflow"] = _new_mode_workflow_initial_state() | {
                "running": True,
                "started_at": time.time(),
                "options": options,
                "completed_stages": completed,
            }
            _append_new_mode_workflow_log_locked("Starting new mode workflow.")
            if completed:
                labels = [NEW_MODE_STAGE_LABELS.get(s, s) for s in completed]
                _append_new_mode_workflow_log_locked(
                    f"Skipping already complete: {', '.join(labels)}."
                )

        _start_new_mode_workflow_thread_locked()
    return dict(process_state["new_mode_workflow"])


@router.post("/api/new_mode_workflow/pause")
async def pause_new_mode_workflow():
    with new_mode_workflow_lock:
        state = process_state["new_mode_workflow"]
        if not state.get("running"):
            if state.get("paused"):
                return {"status": "paused"}
            return {"status": "idle"}
        if not state.get("pause_requested"):
            state["pause_requested"] = True
            _append_new_mode_workflow_log_locked(
                "Pause requested. Waiting for current stage to finish safely..."
            )
        stage = state.get("current_stage")

    # Stop the current stage so pause takes effect promptly
    if stage:
        if stage == "render_audio":
            _pause_audio_queue_for_workflow()
        else:
            _terminate_task_process_if_running(stage)
    return {"status": "pause_requested", "current_stage": stage}
