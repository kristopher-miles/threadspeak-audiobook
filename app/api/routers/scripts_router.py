from fastapi import APIRouter
from .. import shared as _shared

globals().update({k: v for k, v in vars(_shared).items() if not k.startswith("__")})

router = APIRouter()

def _sanitize_name(name: str) -> str:
    """Make a string safe for use as a filename."""
    name = re.sub(r'[^\w\- ]', '', name).strip()
    name = re.sub(r'\s+', '_', name)
    return name.lower()

@router.get("/api/scripts")
async def list_saved_scripts():
    """List all saved scripts in the scripts/ directory."""
    scripts = []
    for f in os.listdir(SCRIPTS_DIR):
        if f.endswith(".json") and not f.endswith(".voice_config.json") and not f.endswith(".paragraphs.json"):
            name = f[:-5]  # strip .json
            filepath = os.path.join(SCRIPTS_DIR, f)
            companion = os.path.join(SCRIPTS_DIR, f"{name}.voice_config.json")
            scripts.append({
                "name": name,
                "created": os.path.getmtime(filepath),
                "has_voice_config": os.path.exists(companion)
            })
    scripts.sort(key=lambda x: x["created"], reverse=True)
    return scripts


@router.get("/api/project_archive")
async def export_project_archive(background_tasks: BackgroundTasks):
    running_task = _any_project_task_running()
    if running_task:
        raise HTTPException(status_code=409, detail=f"Cannot save a project archive while '{running_task}' is running.")

    entries = _project_archive_entries()
    manifest = _build_project_archive_manifest(entries)

    handle = tempfile.NamedTemporaryFile(prefix="threadspeak_project_", suffix=".zip", delete=False)
    temp_zip_path = handle.name
    handle.close()

    try:
        with zipfile.ZipFile(temp_zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr(PROJECT_ARCHIVE_MANIFEST_NAME, json.dumps(manifest, indent=2, ensure_ascii=False))
            for relative_path, absolute_path in entries:
                if relative_path == "state.json":
                    zf.writestr(relative_path, json.dumps(_archive_state_with_relative_paths(), indent=2, ensure_ascii=False))
                else:
                    zf.write(absolute_path, arcname=relative_path)
    except Exception:
        if os.path.exists(temp_zip_path):
            os.remove(temp_zip_path)
        raise

    archive_name = f"threadspeak_project_{time.strftime('%Y%m%d_%H%M%S')}.zip"
    background_tasks.add_task(lambda: os.path.exists(temp_zip_path) and os.remove(temp_zip_path))
    return FileResponse(temp_zip_path, filename=archive_name, media_type="application/zip")


@router.post("/api/project_archive/load")
async def load_project_archive(file: UploadFile = File(...)):
    running_task = _any_project_task_running()
    if running_task:
        raise HTTPException(status_code=409, detail=f"Cannot load a project archive while '{running_task}' is running.")

    filename = file.filename or ""
    if not filename.lower().endswith(".zip"):
        raise HTTPException(status_code=400, detail="Project archive must be a .zip file.")

    content = await file.read()
    temp_root = tempfile.mkdtemp(prefix="threadspeak_project_import_")
    zip_path = os.path.join(temp_root, "project.zip")
    extract_root = os.path.join(temp_root, "extracted")
    os.makedirs(extract_root, exist_ok=True)

    try:
        with open(zip_path, "wb") as f:
            f.write(content)

        with zipfile.ZipFile(zip_path, "r") as zf:
            names = zf.namelist()
            if PROJECT_ARCHIVE_MANIFEST_NAME not in names:
                raise HTTPException(status_code=400, detail="Archive is missing project archive manifest.")

            try:
                manifest = json.loads(zf.read(PROJECT_ARCHIVE_MANIFEST_NAME).decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as e:
                raise HTTPException(status_code=400, detail=f"Archive manifest is invalid: {e}")
            if manifest.get("kind") != "threadspeak_project_archive":
                raise HTTPException(status_code=400, detail="Archive is not a valid Threadspeak project archive.")

            for info in zf.infolist():
                if info.is_dir() or info.filename == PROJECT_ARCHIVE_MANIFEST_NAME:
                    continue
                relative_path = _normalize_archive_path(info.filename)
                if not _is_allowed_project_archive_path(relative_path):
                    raise HTTPException(status_code=400, detail=f"Archive contains unsupported path: {relative_path}")
                target_path = os.path.join(extract_root, relative_path)
                os.makedirs(os.path.dirname(target_path), exist_ok=True)
                with zf.open(info, "r") as source, open(target_path, "wb") as target:
                    shutil.copyfileobj(source, target)

        _restore_project_archive(extract_root)
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)

    return {"status": "loaded", "filename": filename}

class ScriptSaveRequest(BaseModel):
    name: str

def _script_source_prefix(name: str) -> str:
    return f"{name}.source"


def _find_saved_script_source_companion(name: str):
    prefix = _script_source_prefix(name)
    matches = []
    for entry in os.listdir(SCRIPTS_DIR):
        if not entry.startswith(prefix):
            continue
        if entry == f"{name}.json":
            continue
        full_path = os.path.join(SCRIPTS_DIR, entry)
        if os.path.isfile(full_path):
            matches.append(full_path)
    if not matches:
        return None
    matches.sort(key=lambda path: os.path.getmtime(path), reverse=True)
    return matches[0]


def _save_script_source_companion(name: str):
    source_companion = _find_saved_script_source_companion(name)
    while source_companion:
        try:
            os.remove(source_companion)
        except OSError:
            break
        source_companion = _find_saved_script_source_companion(name)

    state = _load_project_state_payload()
    input_path = os.path.abspath((state.get("input_file_path") or "").strip())
    if not input_path or not os.path.exists(input_path) or not os.path.isfile(input_path):
        return None

    ext = os.path.splitext(input_path)[1]
    companion_name = f"{_script_source_prefix(name)}{ext}"
    companion_path = os.path.join(SCRIPTS_DIR, companion_name)
    shutil.copy2(input_path, companion_path)
    return companion_path


def _delete_saved_script_artifacts(name: str):
    base = os.path.join(SCRIPTS_DIR, f"{name}.json")
    if os.path.exists(base):
        os.remove(base)
    for suffix in (".voice_config.json", ".paragraphs.json"):
        companion = os.path.join(SCRIPTS_DIR, f"{name}{suffix}")
        if os.path.exists(companion):
            os.remove(companion)
    source_companion = _find_saved_script_source_companion(name)
    while source_companion:
        try:
            os.remove(source_companion)
        except OSError:
            break
        source_companion = _find_saved_script_source_companion(name)

def _save_current_script_snapshot(name: str, *, purge_existing: bool = False):
    if not os.path.exists(SCRIPT_PATH):
        raise FileNotFoundError("No annotated script to save. Generate a script first.")

    safe_name = _sanitize_name(name)
    if not safe_name:
        raise ValueError("Invalid script name.")

    dest = os.path.join(SCRIPTS_DIR, f"{safe_name}.json")
    existed = os.path.exists(dest)
    if purge_existing and existed:
        _delete_saved_script_artifacts(safe_name)

    shutil.copy2(SCRIPT_PATH, dest)

    if os.path.exists(VOICE_CONFIG_PATH):
        shutil.copy2(VOICE_CONFIG_PATH, os.path.join(SCRIPTS_DIR, f"{safe_name}.voice_config.json"))

    paragraphs_path = os.path.join(ROOT_DIR, "paragraphs.json")
    if os.path.exists(paragraphs_path):
        shutil.copy2(paragraphs_path, os.path.join(SCRIPTS_DIR, f"{safe_name}.paragraphs.json"))
    _save_script_source_companion(safe_name)

    state = _load_project_state_payload()
    state["loaded_script_name"] = safe_name
    _save_project_state_payload(state)
    return {"name": safe_name, "overwrote": existed}

def _autosave_name_from_input_file():
    state = _load_project_state_payload()
    input_path = (state.get("input_file_path") or "").strip()
    if not input_path:
        return ""
    return _sanitize_name(os.path.splitext(os.path.basename(input_path))[0])

def _autosave_current_script_for_workflow(*, purge_existing: bool, trigger: str):
    auto_name = _autosave_name_from_input_file()
    if not auto_name:
        raise RuntimeError("Cannot auto-save script: no imported source file name found.")
    result = _save_current_script_snapshot(auto_name, purge_existing=purge_existing)
    logger.info(
        "Workflow auto-saved script '%s' (trigger=%s, purge_existing=%s)",
        result["name"],
        trigger,
        purge_existing,
    )
    return result

@router.post("/api/scripts/save")
async def save_script(request: ScriptSaveRequest):
    """Save the current annotated_script.json (and voice_config.json) under a name."""
    try:
        result = _save_current_script_snapshot(request.name, purge_existing=False)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    logger.info(f"Script saved as '{result['name']}'")
    return {"status": "saved", "name": result["name"]}

class ScriptLoadRequest(BaseModel):
    name: str


def _audio_generation_active_for_script_load() -> bool:
    """Return whether audio generation/merge is truly active.

    This recomputes audio state from queue/current-job truth and heals stale
    `process_state["audio"]["running"]` after reset/cancel races.
    """
    global audio_current_job, audio_recovery_request
    with audio_queue_lock:
        _refresh_audio_process_state_locked(persist=False)
        active = bool(
            process_state["audio"].get("merge_running")
            or audio_current_job is not None
            or audio_queue
        )
        if not active:
            return False

        # Reset can race with background job completion, leaving stale queue/current
        # pointers for a wiped project. If both script+chunks are gone, clear any
        # leftover runtime audio state so script load can proceed immediately.
        if not os.path.exists(SCRIPT_PATH) and not os.path.exists(CHUNKS_PATH):
            if audio_queue or audio_current_job is not None or process_state["audio"].get("merge_running"):
                audio_queue.clear()
                audio_current_job = None
                audio_recovery_request = None
                process_state["audio"]["cancel"] = False
                audio_cancel_event.clear()
                process_state["audio"]["merge_running"] = False
                _append_audio_log_locked("[HEAL] Cleared stale audio runtime state after project reset.")
                _refresh_audio_process_state_locked(persist=True)
            return bool(
                process_state["audio"].get("merge_running")
                or audio_current_job is not None
                or audio_queue
            )

        return True


@router.post("/api/scripts/load")
async def load_script(request: ScriptLoadRequest):
    """Load a saved script, replacing the current annotated_script.json and chunks."""
    if _audio_generation_active_for_script_load():
        raise HTTPException(status_code=409, detail="Cannot load a script while audio generation is running.")

    src = os.path.join(SCRIPTS_DIR, f"{request.name}.json")
    if not os.path.exists(src):
        raise HTTPException(status_code=404, detail=f"Saved script '{request.name}' not found.")

    shutil.copy2(src, SCRIPT_PATH)

    companion = os.path.join(SCRIPTS_DIR, f"{request.name}.voice_config.json")
    if os.path.exists(companion):
        shutil.copy2(companion, VOICE_CONFIG_PATH)

    paragraphs_path = os.path.join(ROOT_DIR, "paragraphs.json")
    paragraphs_companion = os.path.join(SCRIPTS_DIR, f"{request.name}.paragraphs.json")
    if os.path.exists(paragraphs_companion):
        shutil.copy2(paragraphs_companion, paragraphs_path)
    elif os.path.exists(paragraphs_path):
        os.remove(paragraphs_path)

    source_companion = _find_saved_script_source_companion(request.name)
    restored_input_path = None
    if source_companion:
        ext = os.path.splitext(source_companion)[1]
        upload_name = f"{request.name}{ext}"
        restored_input_path = os.path.join(UPLOADS_DIR, upload_name)
        os.makedirs(UPLOADS_DIR, exist_ok=True)
        shutil.copy2(source_companion, restored_input_path)

    # Delete chunks so they regenerate from the loaded script
    if os.path.exists(CHUNKS_PATH):
        os.remove(CHUNKS_PATH)

    state = _load_project_state_payload()
    state["render_prep_complete"] = False
    state["loaded_script_name"] = request.name
    if restored_input_path:
        state["input_file_path"] = restored_input_path
    state[PROCESSING_STAGE_MARKERS_KEY] = {"script": {"completed_at": time.time()}}
    state[NEW_MODE_STAGE_MARKERS_KEY] = {
        "create_script": {"completed_at": time.time()},
    }
    _save_project_state_payload(state)

    logger.info(f"Script '{request.name}' loaded")
    return {"status": "loaded", "name": request.name}

@router.delete("/api/scripts/{name}")
async def delete_script(name: str):
    """Delete a saved script."""
    filepath = os.path.join(SCRIPTS_DIR, f"{name}.json")
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail=f"Saved script '{name}' not found.")

    _delete_saved_script_artifacts(name)

    logger.info(f"Script '{name}' deleted")
    return {"status": "deleted", "name": name}
