from fastapi import APIRouter

from .. import shared as _shared

globals().update({k: v for k, v in vars(_shared).items() if not k.startswith("__")})

router = APIRouter()


class ScriptSaveRequest(BaseModel):
    name: str


class ScriptLoadRequest(BaseModel):
    name: str


@router.get("/api/scripts")
async def list_saved_scripts():
    """List saved projects, with full archives taking precedence over DB snapshots."""
    projects = {}

    os.makedirs(SAVED_PROJECTS_DIR, exist_ok=True)
    for f in os.listdir(SAVED_PROJECTS_DIR):
        if not f.endswith(".zip"):
            continue
        name = f[:-4]
        filepath = os.path.join(SAVED_PROJECTS_DIR, f)
        if not os.path.isfile(filepath):
            continue
        manifest = _load_project_archive_manifest(filepath) or {}
        metadata = manifest.get("metadata") if isinstance(manifest.get("metadata"), dict) else {}
        projects[name] = {
            "name": name,
            "created": float(manifest.get("created_at") or os.path.getmtime(filepath)),
            "kind": str(metadata.get("kind") or "project"),
            "has_audio": bool(metadata.get("has_audio", _project_archive_contains_entry(filepath, "voicelines/"))),
            "has_voice_config": bool(metadata.get("has_voice_config")),
            "chunk_count": int(metadata.get("chunk_count") or 0),
            "chapter_count": int(metadata.get("chapter_count") or 0),
            "last_chapter": metadata.get("last_chapter"),
        }

    os.makedirs(SCRIPTS_DIR, exist_ok=True)
    for f in os.listdir(SCRIPTS_DIR):
        if not f.endswith(".sqlite3"):
            continue
        name = f[:-8]
        if name in projects:
            continue
        filepath = os.path.join(SCRIPTS_DIR, f)
        projects[name] = {
            "name": name,
            "created": os.path.getmtime(filepath),
            "kind": "script",
            "has_audio": False,
            "has_voice_config": True,
        }

    scripts = list(projects.values())
    scripts.sort(key=lambda item: item["created"], reverse=True)
    return scripts


def _project_archive_contains_entry(zip_path: str, relative_path: str) -> bool:
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            if relative_path.endswith("/"):
                return any(name.startswith(relative_path) for name in zf.namelist())
            return relative_path in zf.namelist()
    except (OSError, zipfile.BadZipFile):
        return False


@router.get("/api/project_archive")
async def export_project_archive(background_tasks: BackgroundTasks):
    running_task = _any_project_task_running()
    if running_task:
        raise HTTPException(status_code=409, detail=f"Cannot save a project archive while '{running_task}' is running.")

    temp_zip_path = _make_runtime_temp_file("threadspeak_project_", suffix=".zip")
    try:
        _write_project_archive(temp_zip_path)
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
    temp_root = _make_runtime_temp_dir("threadspeak_project_import_")
    zip_path = os.path.join(temp_root, "project.zip")

    try:
        with open(zip_path, "wb") as f:
            f.write(content)
        inferred_name = _sanitize_name(os.path.splitext(filename)[0])
        _restore_project_archive_zip(zip_path, loaded_project_name=inferred_name)
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)

    return {"status": "loaded", "filename": filename}


@router.post("/api/scripts/save")
async def save_script(request: ScriptSaveRequest):
    """Save the current project under a name as a full-state archive."""
    running_task = _any_project_task_running()
    if running_task:
        raise HTTPException(status_code=409, detail=f"Cannot save a project while '{running_task}' is running.")

    try:
        result = _save_current_project_archive_snapshot(request.name)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    logger.info("Project '%s' saved as project", result["name"])
    return {"status": "saved", "name": result["name"], "kind": "project"}


def _audio_generation_active_for_script_load() -> bool:
    """Return whether audio generation/merge is truly active."""
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

        if not _project_has_script_document() and not bool(project_manager.load_chunks()):
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
    """Load a saved project, preferring full archives over DB snapshots."""
    if _audio_generation_active_for_script_load():
        raise HTTPException(status_code=409, detail="Cannot load a project while audio generation is running.")

    safe_name = _sanitize_name(request.name)
    if not safe_name:
        raise HTTPException(status_code=400, detail="Invalid project name.")

    archive_path = _saved_project_archive_path(safe_name)
    if os.path.exists(archive_path):
        _restore_project_archive_zip(archive_path, loaded_project_name=safe_name)
        logger.info("Project archive '%s' loaded", safe_name)
        return {"status": "loaded", "name": safe_name, "kind": "project"}

    db_snapshot = _saved_script_db_companion_path(safe_name)
    if not os.path.exists(db_snapshot):
        raise HTTPException(status_code=404, detail=f"Saved project '{safe_name}' not found.")

    _clear_project_archive_targets()
    _copy_sqlite_database_snapshot(
        db_snapshot,
        getattr(project_manager, "chunks_db_path", os.path.join(ROOT_DIR, "chunks.sqlite3")),
    )

    source_companion = _find_saved_script_source_companion(safe_name)
    restored_input_path = None
    if source_companion:
        ext = os.path.splitext(source_companion)[1]
        upload_name = f"{safe_name}{ext}"
        restored_input_path = os.path.join(UPLOADS_DIR, upload_name)
        os.makedirs(UPLOADS_DIR, exist_ok=True)
        shutil.copy2(source_companion, restored_input_path)

    if hasattr(project_manager, "reload_script_store"):
        project_manager.reload_script_store()
    restored_state = {
        "loaded_script_name": safe_name,
        "loaded_project_name": safe_name,
    }
    if restored_input_path:
        restored_state["input_file_path"] = restored_input_path
    _save_project_state_payload(
        _normalize_restored_project_state(
            restored_state,
            loaded_project_name=safe_name,
        )
    )
    if hasattr(project_manager, "log_voice_audit_event"):
        project_manager.log_voice_audit_event(
            "script_snapshot_load",
            reason="load_script_snapshot",
            snapshot_name=safe_name,
        )
    _reset_runtime_state_after_project_load()

    logger.info("Project script snapshot '%s' loaded", safe_name)
    return {"status": "loaded", "name": safe_name, "kind": "script"}


@router.delete("/api/scripts/{name}")
async def delete_script(name: str):
    """Delete a saved project in either archive or script-snapshot form."""
    safe_name = _sanitize_name(name)
    if not safe_name:
        raise HTTPException(status_code=400, detail="Invalid project name.")

    snapshot_path = _saved_script_db_companion_path(safe_name)
    had_script = os.path.exists(snapshot_path)
    had_project = _delete_saved_project_archive(safe_name)
    if not had_script and not had_project:
        raise HTTPException(status_code=404, detail=f"Saved project '{safe_name}' not found.")

    _delete_saved_script_artifacts(safe_name)

    logger.info("Project '%s' deleted", safe_name)
    return {"status": "deleted", "name": safe_name}
