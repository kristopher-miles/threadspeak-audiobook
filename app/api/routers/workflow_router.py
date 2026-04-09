from fastapi import APIRouter
from .. import shared as _shared

globals().update({k: v for k, v in vars(_shared).items() if not k.startswith("__")})

router = APIRouter()

def _ensure_new_mode_workflow_inactive(conflict_message: str = "Cannot run this step while Process Script is active. Pause or wait for new-mode workflow first."):
    state = process_state.get("new_mode_workflow") or {}
    if bool(state.get("running")) or bool(state.get("paused")):
        raise HTTPException(status_code=409, detail=conflict_message)

@router.post("/api/reset_project")
async def reset_project():
    global audio_current_job, audio_recovery_request
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
        if audio_current_job is not None:
            _abandon_audio_job_locked(
                audio_current_job,
                audio_current_job.get("run_token"),
                "Project reset requested",
                status="cancelled",
            )
        # Ensure reset always clears worker pointers, even if abandon raced.
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

    files_to_remove = [
        state_path,
        SCRIPT_PATH,
        VOICES_PATH,
        VOICE_CONFIG_PATH,
        CHUNKS_PATH,
        project_manager.transcription_cache_path,
        SCRIPT_REPAIR_TRACE_PATH,
        AUDIOBOOK_PATH,
        M4B_PATH,
        AUDIO_QUEUE_STATE_PATH,
        PROCESSING_WORKFLOW_STATE_PATH,
        NEW_MODE_WORKFLOW_STATE_PATH,
        os.path.join(ROOT_DIR, "paragraphs.json"),
        SCRIPT_SANITY_PATH,
        os.path.join(ROOT_DIR, "audacity_export.zip"),
        os.path.join(ROOT_DIR, "m4b_cover.jpg"),
    ]

    for path in files_to_remove:
        if os.path.exists(path):
            os.remove(path)
            removed.append(os.path.basename(path))

    if os.path.isdir(VOICELINES_DIR):
        for entry in os.listdir(VOICELINES_DIR):
            entry_path = os.path.join(VOICELINES_DIR, entry)
            if os.path.isdir(entry_path):
                shutil.rmtree(entry_path)
            else:
                os.remove(entry_path)
    os.makedirs(VOICELINES_DIR, exist_ok=True)

    if os.path.isdir(UPLOADS_DIR):
        for entry in os.listdir(UPLOADS_DIR):
            entry_path = os.path.join(UPLOADS_DIR, entry)
            if os.path.isdir(entry_path):
                shutil.rmtree(entry_path)
            else:
                os.remove(entry_path)

    with audio_queue_condition:
        audio_queue.clear()
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

    logger.info("Project state reset")
    return {"status": "reset", "removed": removed}

@router.post("/api/assign_dialogue")
async def start_assign_dialogue(background_tasks: BackgroundTasks):
    _ensure_new_mode_workflow_inactive()
    _ensure_task_not_running("assign_dialogue", "Dialogue assignment is already running.")

    paragraphs_path = os.path.join(ROOT_DIR, "paragraphs.json")
    if not os.path.exists(paragraphs_path):
        raise HTTPException(
            status_code=400,
            detail="No paragraph data found. Run 'Process Paragraphs' first.",
        )
    try:
        with open(paragraphs_path, "r", encoding="utf-8") as f:
            pdata = json.load(f)
        if not pdata.get("paragraphs"):
            raise ValueError("empty")
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Paragraph data is empty or corrupt. Re-run 'Process Paragraphs'.",
        )

    config_path = os.path.join(BASE_DIR, "config.json")
    run_id = _start_task_run("assign_dialogue")
    background_tasks.add_task(_run_assign_dialogue_task, run_id, paragraphs_path, config_path)
    return {"status": "started", "run_id": run_id}


@router.post("/api/extract_temperament")
async def start_extract_temperament(background_tasks: BackgroundTasks):
    _ensure_new_mode_workflow_inactive()
    _ensure_task_not_running("extract_temperament", "Temperament extraction is already running.")

    paragraphs_path = os.path.join(ROOT_DIR, "paragraphs.json")
    if not os.path.exists(paragraphs_path):
        raise HTTPException(
            status_code=400,
            detail="No paragraph data found. Run 'Process Paragraphs' first.",
        )
    try:
        with open(paragraphs_path, "r", encoding="utf-8") as f:
            pdata = json.load(f)
        if not pdata.get("paragraphs"):
            raise ValueError("empty")
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Paragraph data is empty or corrupt. Re-run 'Process Paragraphs'.",
        )

    config_path = os.path.join(BASE_DIR, "config.json")
    run_id = _start_task_run("extract_temperament")
    background_tasks.add_task(_run_extract_temperament_task, run_id, paragraphs_path, config_path)
    return {"status": "started", "run_id": run_id}


@router.get("/api/script_info")
async def get_script_info():
    """Return a lightweight summary of the current script state."""
    script_path = os.path.join(ROOT_DIR, "annotated_script.json")
    if not os.path.exists(script_path):
        return {"entry_count": 0}
    try:
        with open(script_path, "r", encoding="utf-8") as f:
            doc = json.load(f)
        entries = doc.get("entries", []) if isinstance(doc, dict) else doc
        return {"entry_count": len(entries) if isinstance(entries, list) else 0}
    except Exception:
        return {"entry_count": 0}


@router.get("/api/pipeline_step_status")
async def get_pipeline_step_status():
    """Return file-based completion status for the 4 new-mode pipeline steps."""
    paragraphs_path = os.path.join(ROOT_DIR, "paragraphs.json")
    script_path = os.path.join(ROOT_DIR, "annotated_script.json")

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

    if not os.path.exists(paragraphs_path):
        return result

    try:
        with open(paragraphs_path, "r", encoding="utf-8") as f:
            pdata = json.load(f)
        paragraphs = pdata.get("paragraphs", [])
    except Exception:
        return result

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

    # Create Script: annotated_script.json must exist with entries
    if os.path.exists(script_path):
        try:
            with open(script_path, "r", encoding="utf-8") as f:
                sdata = json.load(f)
            entries = sdata.get("entries", []) if isinstance(sdata, dict) else sdata
            if isinstance(entries, list) and len(entries) > 0:
                result["create_script"] = "complete"
        except Exception:
            pass

    return result


@router.post("/api/reset_new_mode")
async def reset_new_mode():
    """Clear the script, chunks, and voice config so Create Script can start fresh."""
    removed = []
    for path in (
        os.path.join(ROOT_DIR, "annotated_script.json"),
        CHUNKS_PATH,
        VOICE_CONFIG_PATH,
    ):
        if os.path.exists(path):
            os.remove(path)
            removed.append(os.path.basename(path))
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
    return {"status": "reset", "removed": removed}


@router.post("/api/create_script")
async def start_create_script(background_tasks: BackgroundTasks):
    _ensure_new_mode_workflow_inactive()
    _ensure_task_not_running("create_script", "Script creation is already running.")

    paragraphs_path    = os.path.join(ROOT_DIR, "paragraphs.json")
    voice_config_path  = VOICE_CONFIG_PATH
    script_output_path = os.path.join(ROOT_DIR, "annotated_script.json")
    chunks_output_path = CHUNKS_PATH

    if not os.path.exists(paragraphs_path):
        raise HTTPException(
            status_code=400,
            detail="No paragraph data found. Run 'Process Paragraphs' first.",
        )
    try:
        with open(paragraphs_path, "r", encoding="utf-8") as f:
            pdata = json.load(f)
        if not pdata.get("paragraphs"):
            raise ValueError("empty")
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Paragraph data is empty or corrupt. Re-run 'Process Paragraphs'.",
        )

    run_id = _start_task_run("create_script")
    background_tasks.add_task(
        _run_create_script_task, run_id,
        paragraphs_path, voice_config_path, script_output_path, chunks_output_path,
    )
    return {"status": "started", "run_id": run_id}


@router.post("/api/process_paragraphs")
async def start_process_paragraphs(background_tasks: BackgroundTasks):
    _ensure_new_mode_workflow_inactive()
    _ensure_task_not_running("process_paragraphs", "Paragraph processing is already running.")

    # Hard-fail if an annotated script already exists with entries
    script_path = os.path.join(ROOT_DIR, "annotated_script.json")
    if os.path.exists(script_path):
        try:
            with open(script_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            entries = data.get("entries") if isinstance(data, dict) else data
            if isinstance(entries, list) and len(entries) > 0:
                raise HTTPException(
                    status_code=409,
                    detail="Generating a new audiobook script requires erasing the old one first.",
                )
        except HTTPException:
            raise
        except Exception:
            pass  # Unreadable or corrupt file — allow proceeding

    # Resolve input file from state.json
    state_path = os.path.join(ROOT_DIR, "state.json")
    if not os.path.exists(state_path):
        raise HTTPException(status_code=400, detail="No input file selected. Please upload a book first.")
    with open(state_path, "r", encoding="utf-8") as f:
        state = json.load(f)
    input_file = state.get("input_file_path")
    if not input_file or not os.path.exists(input_file):
        raise HTTPException(status_code=400, detail="No input file found. Please upload a book first.")

    output_path = os.path.join(ROOT_DIR, "paragraphs.json")
    run_id = _start_task_run("process_paragraphs")
    background_tasks.add_task(_run_process_paragraphs_task, run_id, input_file, output_path)
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
    if not os.path.exists(SCRIPT_PATH):
        raise HTTPException(status_code=400, detail="No annotated script found. Generate a script first.")

    run_id = _start_task_run("review")
    background_tasks.add_task(_run_review_script_task, run_id)
    return {"status": "started", "run_id": run_id}

@router.post("/api/script_sanity_check")
async def script_sanity_check(background_tasks: BackgroundTasks):
    if not os.path.exists(SCRIPT_PATH):
        raise HTTPException(status_code=400, detail="No annotated script found. Generate a script first.")

    run_id = _start_task_run("sanity")
    background_tasks.add_task(_run_sanity_task, run_id)
    return {"status": "started", "run_id": run_id}

@router.post("/api/replace_missing_chunks")
async def replace_missing_chunks(background_tasks: BackgroundTasks):
    if not os.path.exists(SCRIPT_PATH):
        raise HTTPException(status_code=400, detail="No annotated script found. Generate a script first.")

    run_id = _start_task_run("repair")
    background_tasks.add_task(_run_repair_task, run_id)
    return {"status": "started", "run_id": run_id}

@router.get("/api/annotated_script")
async def get_annotated_script():
    """Return the current working annotated_script.json."""
    if not os.path.exists(SCRIPT_PATH):
        raise HTTPException(status_code=404, detail="No annotated script found")
    # Backward-compatible response shape: return entries list.
    return _load_project_script_document().get("entries", [])

@router.get("/api/script_sanity_check")
async def get_script_sanity_check():
    if not os.path.exists(SCRIPT_SANITY_PATH):
        raise HTTPException(status_code=404, detail="No sanity check results found")
    with open(SCRIPT_SANITY_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

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
    options = {"process_voices": bool(request.process_voices), "generate_audio": bool(request.generate_audio)}
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
