from fastapi import APIRouter
from .. import shared as _shared
from pydub import AudioSegment

globals().update({k: v for k, v in vars(_shared).items() if not k.startswith("__")})

router = APIRouter()


class MergeRequest(BaseModel):
    export: Optional[ExportConfig] = None

SANITY_TRIM_FIRST_CLIP_PATH = os.path.join(ROOT_DIR, "trim_sanity_first_clip.wav")
SANITY_ASSEMBLE_FIRST5_PATH = os.path.join(ROOT_DIR, "assemble_sanity_first5.wav")
SANITY_ASSEMBLE_FIRST5_NORMALIZED_PATH = os.path.join(ROOT_DIR, "assemble_sanity_first5_normalized.wav")

def _export_download_basename():
    """Return a filesystem-safe base name for export downloads, using the saved script name when available."""
    try:
        state = _load_project_state_payload()
        name = (state.get("loaded_script_name") or "").strip()
        if not name:
            input_path = (state.get("input_file_path") or "").strip()
            if input_path:
                name = os.path.splitext(os.path.basename(input_path))[0].strip()
        if name:
            return re.sub(r"[^A-Za-z0-9_\-]+", "_", name).strip("_") or "audiobook"
    except Exception:
        pass
    return "audiobook"

@router.get("/api/audiobook")
async def get_audiobook():
    if not os.path.exists(AUDIOBOOK_PATH):
        raise HTTPException(status_code=404, detail="Audiobook not found")
    download_name = f"{_export_download_basename()}.mp3"
    return FileResponse(AUDIOBOOK_PATH, filename=download_name, media_type="audio/mpeg")

# --- Chunk Management Endpoints ---

@router.get("/api/chunks")
async def get_chunks():
    chunks = project_manager.reconcile_chunk_audio_states()
    return chunks


@router.get("/api/chunks/view")
async def get_chunks_view(chapter: Optional[str] = None):
    chunks = project_manager.load_chunks()
    chapter = (chapter or "").strip()
    if chapter:
        chunks = [chunk for chunk in chunks if (chunk.get("chapter") or "").strip() == chapter]
    return chunks


@router.post("/api/chunks/sync_from_script_if_stale")
async def sync_chunks_from_script_if_stale():
    # Never rewrite chunks while audio generation is active; doing so can clear
    # generation tokens for in-flight workers and surface as "Generation abandoned".
    with audio_queue_lock:
        if audio_current_job is not None or audio_queue or process_state["audio"].get("merge_running", False):
            return {"synced": False, "reason": "audio_running"}
    result = project_manager.sync_chunks_from_script_if_stale()
    return result

class ChunkRestoreRequest(BaseModel):
    chunk: dict
    at_index: Optional[int] = None
    after_uid: Optional[str] = None

@router.post("/api/chunks/restore")
async def restore_chunk(request: ChunkRestoreRequest):
    """Re-insert a previously deleted chunk at a specific index."""
    chunks = project_manager.restore_chunk(request.at_index or 0, request.chunk, after_uid=request.after_uid)
    if chunks is None:
        raise HTTPException(status_code=400, detail="Failed to restore chunk")
    return {"status": "ok", "total": len(chunks)}

@router.post("/api/chunks/decompose_long_segments")
async def decompose_long_segments(request: ChunkDecomposeRequest):
    running_task = _any_project_task_running()
    if running_task:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot decompose segments while {running_task} work is running",
        )

    chapter = (request.chapter or "").strip() or None
    max_words = max(int(request.max_words or 25), 1)
    result = project_manager.decompose_long_segments(chapter=chapter, max_words=max_words)
    return {"status": "ok", **result}

@router.post("/api/chunks/merge_orphans")
async def merge_orphans(request: ChunkMergeOrphansRequest):
    running_task = _any_project_task_running()
    if running_task:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot merge orphan segments while {running_task} work is running",
        )

    chapter = (request.chapter or "").strip() or None
    min_words = max(int(request.min_words or 10), 1)
    result = project_manager.merge_orphan_segments(chapter=chapter, min_words=min_words)
    return {"status": "ok", **result}

@router.post("/api/chunks/repair_legacy")
async def repair_legacy_chunks(request: ChunkRepairLegacyRequest):
    running_task = _any_project_task_running()
    if running_task:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot repair chunk order while {running_task} work is running",
        )

    repaired = project_manager.repair_legacy_chunk_order(request.chunks)
    if repaired is None:
        raise HTTPException(status_code=400, detail="Failed to repair legacy chunk order")
    return {"status": "ok", "total": len(repaired)}

@router.post("/api/chunks/reset_to_pending")
async def reset_chunks_to_pending(request: BatchGenerateRequest):
    """Force-reset the given chunks to pending status.

    Cancels any running audio job first (using the existing cancel logic so
    generation tokens are invalidated), then resets every requested chunk
    regardless of its current status.  This gives the user instant visual
    feedback before the new generation job is enqueued.
    """
    with audio_queue_condition:
        # Clear the queue and abandon any running job atomically
        now = time.time()
        while audio_queue:
            job = audio_queue.pop(0)
            job["status"] = "cancelled"
            job["finished_at"] = now
            _record_audio_recent_job_locked(job)
        if audio_current_job is not None:
            _abandon_audio_job_locked(
                audio_current_job,
                audio_current_job.get("run_token"),
                "Regenerate All reset",
                status="cancelled",
            )
        _refresh_audio_process_state_locked(persist=True)

    chunks = project_manager.load_chunks()
    resolved = []
    for ref in (request.indices or []):
        idx = project_manager.resolve_chunk_index(ref, chunks)
        if idx is not None and 0 <= idx < len(chunks):
            resolved.append(idx)

    reset_count = project_manager.force_reset_chunks_to_pending(resolved)
    return {"status": "ok", "reset": reset_count}


@router.post("/api/chunks/invalidate_stale_audio")
async def invalidate_stale_audio():
    running_task = _any_project_task_running()
    if running_task:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot invalidate stale audio while {running_task} work is running",
        )

    result = project_manager.invalidate_stale_audio_references()
    return {"status": "ok", **result}

@router.get("/api/asr/status")
async def get_asr_status():
    settings = project_manager._load_asr_settings()
    return {
        "enabled": bool(settings.get("enabled", True)),
        "model": settings.get("model", "small.en"),
        "language": settings.get("language", "en"),
        "device": settings.get("device", "auto"),
        "compute_type": settings.get("compute_type", "auto"),
        "beam_size": int(settings.get("beam_size", 1) or 1),
    }

@router.post("/api/asr/transcribe")
async def transcribe_audio_clip(request: ASRTranscribeRequest):
    try:
        result = project_manager.transcribe_audio_path(request.audio_path)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Audio file not found")
    except LocalASRUnavailableError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"ASR transcription failed: {e}")
    return {"status": "ok", **result}

@router.post("/api/chunks/repair_lost_audio")
async def repair_lost_audio(request: LostAudioRepairRequest, background_tasks: BackgroundTasks):
    running_task = _any_project_task_running()
    if running_task:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot repair lost audio while {running_task} work is running",
        )

    run_id = _start_task_run("repair")
    background_tasks.add_task(
        run_process,
        [
            sys.executable,
            "-u",
            "lost_audio_repair_runner.py",
            ROOT_DIR,
            "1" if bool(request.use_asr) else "0",
            "1" if bool(request.rejected_only) else "0",
        ],
        "repair",
        run_id,
    )
    return {"status": "started", "run_id": run_id}

@router.post("/api/proofread")
async def start_proofread(request: ProofreadRequest, background_tasks: BackgroundTasks):
    running_task = _any_project_task_running()
    if running_task:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot proofread while {running_task} work is running",
        )

    run_id = _start_task_run("proofread")
    chapter_arg = (request.chapter or "").strip() or "__ALL__"
    threshold = max(0.0, min(float(request.threshold or 0.0), 1.0))
    background_tasks.add_task(
        run_process,
        [sys.executable, "-u", "proofread_runner.py", ROOT_DIR, str(threshold), chapter_arg],
        "proofread",
        run_id,
    )
    return {"status": "started", "run_id": run_id}

@router.post("/api/proofread/auto")
async def start_proofread_auto(request: ProofreadRequest, background_tasks: BackgroundTasks):
    """Trigger a background proofread run that can run concurrently with audio generation.
    Only blocked by an already-running proofread, not by audio work."""
    _ensure_task_not_running("proofread", "Proofread is already running")
    run_id = _start_task_run("proofread")
    chapter_arg = (request.chapter or "").strip() or "__ALL__"
    threshold = max(0.0, min(float(request.threshold or 0.0), 1.0))
    background_tasks.add_task(
        run_process,
        [sys.executable, "-u", "proofread_runner.py", ROOT_DIR, str(threshold), chapter_arg],
        "proofread",
        run_id,
    )
    return {"status": "started", "run_id": run_id}

@router.post("/api/proofread/clear_failures")
async def clear_proofread_failures(request: ProofreadClearFailuresRequest):
    running_task = _any_project_task_running()
    if running_task:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot clear proofread failures while {running_task} work is running",
        )

    threshold = max(0.0, min(float(request.threshold or 0.0), 1.0))
    result = project_manager.clear_proofread_failures(
        chapter=(request.chapter or "").strip() or None,
        threshold=threshold,
    )
    return {"status": "ok", **result}

@router.post("/api/proofread/discard_selection")
async def discard_proofread_selection(request: ProofreadDiscardSelectionRequest):
    running_task = _any_project_task_running()
    if running_task:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot discard proofread selection while {running_task} work is running",
        )

    result = project_manager.discard_proofread_selection(
        chapter=(request.chapter or "").strip() or None,
    )
    return {"status": "ok", **result}

@router.post("/api/proofread/{index}/validate")
async def validate_proofread_clip(index: str, request: ProofreadValidateRequest):
    threshold = max(0.0, min(float(request.threshold or 0.0), 1.0))
    try:
        chunk = project_manager.manually_validate_proofread_clip(index, threshold=threshold)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    if chunk is None:
        raise HTTPException(status_code=404, detail="Invalid chunk id")
    return {"status": "ok", "chunk": chunk}

@router.post("/api/proofread/{index}/reject")
async def reject_proofread_clip(index: str, request: ProofreadValidateRequest):
    threshold = max(0.0, min(float(request.threshold or 0.0), 1.0))
    try:
        chunk = project_manager.manually_reject_proofread_clip(index, threshold=threshold)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    if chunk is None:
        raise HTTPException(status_code=404, detail="Invalid chunk id")
    return {"status": "ok", "chunk": chunk}

@router.post("/api/proofread/{index}/compare")
async def compare_proofread_clip(index: str, request: ProofreadCompareRequest):
    threshold = max(0.0, min(float(request.threshold or 0.0), 1.0))
    try:
        chunk = project_manager.compare_proofread_clip(index, threshold=threshold)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    if chunk is None:
        raise HTTPException(status_code=404, detail="Invalid chunk id")
    return {"status": "ok", "chunk": chunk}

@router.post("/api/render_prep_state")
async def set_render_prep_state(request: RenderPrepStateRequest):
    complete = project_manager.set_render_prep_complete(bool(request.complete))
    return {"status": "ok", "render_prep_complete": complete}

@router.post("/api/chunks/{index}")
async def update_chunk(index: str, update: ChunkUpdate):
    data = update.model_dump(exclude_unset=True)
    logger.info(f"Updating chunk {index} with data: {data}")
    chunk = project_manager.update_chunk(index, data)
    if not chunk:
        raise HTTPException(status_code=404, detail="Chunk not found")
    logger.info(f"Chunk {index} updated, instruct is now: '{chunk.get('instruct', '')}'")
    return chunk

@router.post("/api/chunks/{index}/insert")
async def insert_chunk(index: str):
    """Insert an empty chunk after the given index."""
    chunks = project_manager.insert_chunk(index)
    if chunks is None:
        raise HTTPException(status_code=404, detail="Invalid chunk index")
    return {"status": "ok", "total": len(chunks)}

@router.delete("/api/chunks/{index}")
async def delete_chunk(index: str):
    """Delete a chunk at the given index."""
    result = project_manager.delete_chunk(index)
    if result is None:
        raise HTTPException(status_code=400, detail="Cannot delete chunk (invalid index or last remaining chunk)")
    deleted, chunks, restore_after_uid = result
    return {"status": "ok", "deleted": deleted, "total": len(chunks), "restore_after_uid": restore_after_uid}

@router.post("/api/chunks/{index}/generate")
async def generate_chunk_endpoint(index: str, background_tasks: BackgroundTasks):
    chunks = project_manager.load_chunks()
    resolved_index = project_manager.resolve_chunk_index(index, chunks)
    if resolved_index is None or not (0 <= resolved_index < len(chunks)):
        raise HTTPException(status_code=404, detail="Invalid chunk id")
    if not chunks[resolved_index].get("text", "").strip():
        raise HTTPException(status_code=400, detail="Cannot generate audio for an empty line")

    def task():
        project_manager.generate_chunk_audio(resolved_index)

    background_tasks.add_task(task)
    return {"status": "started"}

@router.post("/api/chunks/{index}/regenerate")
async def regenerate_chunk_endpoint(index: str, background_tasks: BackgroundTasks):
    prepared = project_manager.prepare_chunk_for_regeneration(index)
    if prepared is None:
        raise HTTPException(status_code=404, detail="Invalid chunk id")

    chunk = prepared["chunk"]
    resolved_index = prepared["index"]
    if not chunk.get("text", "").strip():
        raise HTTPException(status_code=400, detail="Cannot generate audio for an empty line")

    def task():
        project_manager.generate_chunk_audio(resolved_index)

    background_tasks.add_task(task)
    return {"status": "started"}

@router.post("/api/merge")
async def merge_audio_endpoint(request: Optional[MergeRequest] = None, background_tasks: BackgroundTasks = None):
    background_tasks = background_tasks or BackgroundTasks()
    with audio_queue_lock:
        if audio_current_job is not None or audio_queue or process_state["audio"].get("merge_running", False):
            raise HTTPException(status_code=400, detail="Audio queue is active. Wait for queued jobs to finish or cancel them first.")

    # Reuse audio process state for merge if possible, or just background it
    # For simplicity, we just background it and frontend will assume it works
    # Or we can link it to process_state["audio"]

    def task():
        process_state["audio"]["merge_running"] = True
        process_state["audio"]["running"] = True
        process_state["audio"]["logs"] = ["Starting merge..."]
        process_state["audio"]["merge_progress"] = _new_audio_merge_progress() | {"running": True, "stage": "starting", "updated_at": time.time()}
        merge_log_state = {
            "prepare_bucket": -1,
            "export_bucket": -1,
            "normalizing_bucket": -1,
            "last_stage": None,
        }

        def _format_eta_from_percent(percent_value, elapsed_seconds):
            if percent_value <= 0:
                return "ETA --"
            if percent_value >= 99.9:
                return "ETA <1s"
            remaining = max(0.0, elapsed_seconds * (100.0 - percent_value) / percent_value)
            return f"ETA {int(round(remaining))}s"

        def _dot_meter(percent_value):
            filled = min(10, max(1, int(round(percent_value / 10.0))))
            return "." * filled + "-" * (10 - filled)

        try:
            def on_progress(progress):
                process_state["audio"]["merge_progress"] = {
                    "running": True,
                    "stage": progress.get("stage"),
                    "chapter_index": int(progress.get("chapter_index", 0) or 0),
                    "total_chapters": int(progress.get("total_chapters", 0) or 0),
                    "chapter_label": progress.get("chapter_label") or "",
                    "elapsed_seconds": float(progress.get("elapsed_seconds", 0.0) or 0.0),
                    "merged_duration_seconds": float(progress.get("merged_duration_seconds", 0.0) or 0.0),
                    "estimated_size_bytes": int(progress.get("estimated_size_bytes", 0) or 0),
                    "output_file_size_bytes": int(progress.get("output_file_size_bytes", 0) or 0),
                    "updated_at": time.time(),
                }
                stage = progress.get("stage")
                chapter_index = int(progress.get("chapter_index", 0) or 0)
                total_chapters = int(progress.get("total_chapters", 0) or 0)
                chapter_label = progress.get("chapter_label") or "Unlabeled"
                elapsed = float(progress.get("elapsed_seconds", 0.0) or 0.0)
                if stage == "preparing":
                    percent = float(progress.get("percent_complete", 0.0) or 0.0)
                    bucket = int(percent // 5)
                    if bucket > merge_log_state["prepare_bucket"] or merge_log_state["last_stage"] != stage:
                        total_items = int(progress.get("total_items", 0) or 0)
                        processed_items = int(progress.get("processed_items", 0) or 0)
                        remaining_items = int(progress.get("remaining_items", 0) or 0)
                        if bool(progress.get("trim_enabled")):
                            process_state["audio"]["logs"].append(
                                f"Trimming clips [{_dot_meter(percent)}] {int(round(percent))}% "
                                f"({processed_items}/{total_items}, {remaining_items} remaining, "
                                f"{_format_eta_from_percent(percent, elapsed)})"
                            )
                        else:
                            process_state["audio"]["logs"].append(
                                f"Preparing merge inputs: {int(round(percent))}% "
                                f"({processed_items}/{total_items}, {remaining_items} remaining, "
                                f"{_format_eta_from_percent(percent, elapsed)})"
                            )
                        merge_log_state["prepare_bucket"] = bucket
                elif stage == "assembling":
                    process_state["audio"]["logs"].append(f"Assembling chapter {chapter_index}/{total_chapters}: {chapter_label}")
                elif stage == "packing":
                    process_state["audio"]["logs"].append(f"Packing optimized part {chapter_index}/{total_chapters}: {chapter_label}")
                elif stage == "bundling":
                    process_state["audio"]["logs"].append("Writing optimized export zip...")
                elif stage == "exporting":
                    estimated_size = int(progress.get("estimated_size_bytes", 0) or 0)
                    output_size = int(progress.get("output_file_size_bytes", 0) or 0)
                    percent = 0.0
                    if estimated_size > 0:
                        percent = min(99.0, max(0.0, (output_size / estimated_size) * 100.0))
                    bucket = int(percent // 5)
                    if bucket > merge_log_state["export_bucket"] or merge_log_state["last_stage"] != stage:
                        process_state["audio"]["logs"].append(
                            f"Exporting final audiobook file... {int(round(percent))}% "
                            f"({_format_eta_from_percent(percent, elapsed)})"
                        )
                        merge_log_state["export_bucket"] = bucket
                elif stage == "normalizing":
                    label = chapter_label or ""
                    percent_match = re.search(r"(\d{1,3})%", label)
                    if percent_match:
                        percent = float(percent_match.group(1))
                        bucket = int(percent // 5)
                        if bucket > merge_log_state["normalizing_bucket"] or merge_log_state["last_stage"] != stage:
                            process_state["audio"]["logs"].append(
                                f"Applying loudness normalization... {int(round(percent))}% "
                                f"({_format_eta_from_percent(percent, elapsed)})"
                            )
                            merge_log_state["normalizing_bucket"] = bucket
                    elif merge_log_state["last_stage"] != stage:
                        process_state["audio"]["logs"].append("Applying loudness normalization...")
                merge_log_state["last_stage"] = stage

            selected_export = request.export if request and request.export is not None else _load_export_config()
            success, msg = project_manager.merge_audio(
                progress_callback=on_progress,
                log_callback=_append_audio_log_locked,
                export_config=selected_export,
            )
            if success:
                process_state["audio"]["logs"].append(f"Merge complete: {msg}")
            else:
                process_state["audio"]["logs"].append(f"Merge failed: {msg}")
        except Exception as e:
            process_state["audio"]["logs"].append(f"Merge error: {e}")
        finally:
            progress = process_state["audio"].get("merge_progress") or _new_audio_merge_progress()
            process_state["audio"]["merge_progress"] = progress | {
                "running": False,
                "stage": "complete" if process_state["audio"]["logs"] and "Merge complete" in process_state["audio"]["logs"][-1] else "idle",
                "updated_at": time.time(),
            }
            process_state["audio"]["merge_running"] = False
            process_state["audio"]["running"] = False

    background_tasks.add_task(task)
    return {"status": "started"}

@router.post("/api/merge_optimized")
async def merge_optimized_audio_endpoint(request: Optional[MergeRequest] = None, background_tasks: BackgroundTasks = None):
    background_tasks = background_tasks or BackgroundTasks()
    with audio_queue_lock:
        if audio_current_job is not None or audio_queue or process_state["audio"].get("merge_running", False):
            raise HTTPException(status_code=400, detail="Audio queue is active. Wait for queued jobs to finish or cancel them first.")

    def task():
        process_state["audio"]["merge_running"] = True
        process_state["audio"]["running"] = True
        process_state["audio"]["logs"] = ["Starting optimized export..."]
        process_state["audio"]["merge_progress"] = _new_audio_merge_progress() | {"running": True, "stage": "starting", "updated_at": time.time()}
        try:
            def on_progress(progress):
                process_state["audio"]["merge_progress"] = {
                    "running": True,
                    "stage": progress.get("stage"),
                    "chapter_index": int(progress.get("chapter_index", 0) or 0),
                    "total_chapters": int(progress.get("total_chapters", 0) or 0),
                    "chapter_label": progress.get("chapter_label") or "",
                    "elapsed_seconds": float(progress.get("elapsed_seconds", 0.0) or 0.0),
                    "merged_duration_seconds": float(progress.get("merged_duration_seconds", 0.0) or 0.0),
                    "estimated_size_bytes": int(progress.get("estimated_size_bytes", 0) or 0),
                    "output_file_size_bytes": int(progress.get("output_file_size_bytes", 0) or 0),
                    "updated_at": time.time(),
                }
                stage = progress.get("stage")
                chapter_index = int(progress.get("chapter_index", 0) or 0)
                total_chapters = int(progress.get("total_chapters", 0) or 0)
                chapter_label = progress.get("chapter_label") or "Unlabeled"
                if stage == "preparing":
                    process_state["audio"]["logs"].append(f"Preparing optimized export inputs: {chapter_label}")
                elif stage == "assembling":
                    process_state["audio"]["logs"].append(f"Exporting chapter {chapter_index}/{total_chapters}: {chapter_label}")
                elif stage == "packing":
                    process_state["audio"]["logs"].append(f"Packing optimized part {chapter_index}/{total_chapters}: {chapter_label}")
                elif stage == "bundling":
                    process_state["audio"]["logs"].append("Writing optimized export zip...")
                elif stage == "normalizing":
                    process_state["audio"]["logs"].append(f"Normalizing optimized part {chapter_index}/{total_chapters}...")

            selected_export = request.export if request and request.export is not None else _load_export_config()
            success, msg = project_manager.export_optimized_mp3_zip(
                progress_callback=on_progress,
                log_callback=_append_audio_log_locked,
                export_config=selected_export,
            )
            if success:
                process_state["audio"]["logs"].append(f"Optimized export complete: {msg}")
            else:
                process_state["audio"]["logs"].append(f"Optimized export failed: {msg}")
        except Exception as e:
            process_state["audio"]["logs"].append(f"Optimized export error: {e}")
        finally:
            progress = process_state["audio"].get("merge_progress") or _new_audio_merge_progress()
            process_state["audio"]["merge_progress"] = progress | {
                "running": False,
                "stage": "complete" if process_state["audio"]["logs"] and "Optimized export complete" in process_state["audio"]["logs"][-1] else "idle",
                "updated_at": time.time(),
            }
            process_state["audio"]["merge_running"] = False
            process_state["audio"]["running"] = False

    background_tasks.add_task(task)
    return {"status": "started"}

@router.get("/api/optimized_export")
async def get_optimized_export():
    if not os.path.exists(OPTIMIZED_EXPORT_PATH):
        raise HTTPException(status_code=404, detail="Optimized export not found. Generate it first.")
    download_name = f"{_export_download_basename()}.zip"
    return FileResponse(OPTIMIZED_EXPORT_PATH, filename=download_name, media_type="application/zip")

@router.post("/api/trim_cache/clear")
async def clear_trim_cache():
    with audio_queue_lock:
        if audio_current_job is not None or audio_queue or process_state["audio"].get("merge_running", False):
            raise HTTPException(
                status_code=400,
                detail="Audio queue is active. Wait for queued jobs to finish or cancel them first.",
            )

    cache_dir = project_manager._trim_cache_dir()
    if not os.path.isdir(cache_dir):
        return {"status": "ok", "removed_files": 0, "removed_bytes": 0, "path": cache_dir}

    removed_files = 0
    removed_bytes = 0
    for root, _dirs, files in os.walk(cache_dir):
        for name in files:
            path = os.path.join(root, name)
            removed_files += 1
            try:
                removed_bytes += os.path.getsize(path)
            except OSError:
                pass

    try:
        shutil.rmtree(cache_dir)
    except OSError as e:
        raise HTTPException(status_code=500, detail=f"Failed to clear trim cache: {e}")

    return {
        "status": "ok",
        "removed_files": removed_files,
        "removed_bytes": removed_bytes,
        "path": cache_dir,
    }


@router.post("/api/trim_sanity/first_clip")
async def trim_sanity_first_clip(request: Optional[MergeRequest] = None):
    with audio_queue_lock:
        if audio_current_job is not None or audio_queue or process_state["audio"].get("merge_running", False):
            raise HTTPException(
                status_code=400,
                detail="Audio queue is active. Wait for queued jobs to finish or cancel them first.",
            )

    chunks = project_manager.load_chunks()
    first_clip = None
    for chunk in chunks:
        rel_path = (chunk.get("audio_path") or "").strip()
        if not rel_path:
            continue
        full_path = os.path.join(ROOT_DIR, rel_path)
        if os.path.exists(full_path):
            first_clip = {
                "id": chunk.get("id"),
                "speaker": chunk.get("speaker"),
                "text": chunk.get("text"),
                "audio_path": rel_path,
                "full_path": full_path,
            }
            break
    if first_clip is None:
        raise HTTPException(status_code=404, detail="No existing clip found in project chunks.")

    selected_export = request.export if request and request.export is not None else _load_export_config()
    passed_export_payload = selected_export.model_dump()
    process_state["audio"]["logs"].append(
        f"[trim-sanity] Input export config passed to trim: {json.dumps(passed_export_payload, ensure_ascii=False, sort_keys=True)}"
    )
    print(
        f"[trim-sanity] Input export config passed to trim: {json.dumps(passed_export_payload, ensure_ascii=False, sort_keys=True)}",
        flush=True,
    )

    trim_cfg = project_manager._resolve_trim_config(selected_export)
    process_state["audio"]["logs"].append(
        f"[trim-sanity] Resolved trim args: {json.dumps(trim_cfg, ensure_ascii=False, sort_keys=True)}"
    )
    print(
        f"[trim-sanity] Resolved trim args: {json.dumps(trim_cfg, ensure_ascii=False, sort_keys=True)}",
        flush=True,
    )

    resolved_path, trim_info = project_manager._resolve_export_audio_path(first_clip["full_path"], trim_cfg)
    resolved_segment = AudioSegment.from_file(resolved_path)
    exported = resolved_segment.export(SANITY_TRIM_FIRST_CLIP_PATH, format="wav")
    if hasattr(exported, "close"):
        exported.close()

    process_state["audio"]["logs"].append(
        f"[trim-sanity] First clip: id={first_clip['id']} path={first_clip['audio_path']} "
        f"resolved_path={resolved_path} trimmed={bool(trim_info.get('trimmed'))} "
        f"lead_ms={int(trim_info.get('lead_ms', 0))} tail_ms={int(trim_info.get('tail_ms', 0))}"
    )

    return {
        "status": "ok",
        "download_url": "/api/trim_sanity/first_clip",
        "output_path": SANITY_TRIM_FIRST_CLIP_PATH,
        "clip": {
            "id": first_clip["id"],
            "speaker": first_clip["speaker"],
            "text": first_clip["text"],
            "audio_path": first_clip["audio_path"],
        },
        "trim_info": trim_info,
        "passed_export": passed_export_payload,
        "resolved_trim": trim_cfg,
    }


@router.get("/api/trim_sanity/first_clip")
async def get_trim_sanity_first_clip():
    if not os.path.exists(SANITY_TRIM_FIRST_CLIP_PATH):
        raise HTTPException(status_code=404, detail="Sanity trim clip not found. Generate it first.")
    return FileResponse(
        SANITY_TRIM_FIRST_CLIP_PATH,
        filename="trim_sanity_first_clip.wav",
        media_type="audio/wav",
    )


def _collect_first_clips_for_sanity(limit=5):
    chunks = project_manager.load_chunks()
    selected = []
    for chunk in chunks:
        rel_path = (chunk.get("audio_path") or "").strip()
        if not rel_path:
            continue
        full_path = os.path.join(ROOT_DIR, rel_path)
        if not os.path.exists(full_path):
            continue
        selected.append({
            "id": chunk.get("id"),
            "speaker": chunk.get("speaker"),
            "text": chunk.get("text"),
            "chapter": chunk.get("chapter"),
            "paragraph_id": chunk.get("paragraph_id"),
            "audio_path": rel_path,
            "full_path": full_path,
        })
        if len(selected) >= limit:
            break
    return selected


def _assemble_sanity_clips(clip_items, export_cfg_obj):
    trim_cfg = project_manager._resolve_trim_config(export_cfg_obj)
    same_ms = int(getattr(export_cfg_obj, "silence_same_speaker_ms", 250))
    between_ms = int(getattr(export_cfg_obj, "silence_between_speakers_ms", 500))
    paragraph_ms = int(getattr(export_cfg_obj, "silence_paragraph_ms", 750))

    combined = None
    details = []
    previous = None
    for item in clip_items:
        resolved_path, trim_info = project_manager._resolve_export_audio_path(item["full_path"], trim_cfg)
        segment = AudioSegment.from_file(resolved_path)
        details.append({
            "id": item["id"],
            "audio_path": item["audio_path"],
            "resolved_path": resolved_path,
            "duration_ms": len(segment),
            "trimmed": bool(trim_info.get("trimmed")),
            "lead_ms": int(trim_info.get("lead_ms", 0)),
            "tail_ms": int(trim_info.get("tail_ms", 0)),
        })
        if combined is None:
            combined = segment
            previous = item
            continue

        prev_pid = previous.get("paragraph_id")
        curr_pid = item.get("paragraph_id")
        if prev_pid and curr_pid and prev_pid != curr_pid:
            pause_ms = paragraph_ms
        elif previous.get("speaker") == item.get("speaker"):
            pause_ms = same_ms
        else:
            pause_ms = between_ms

        combined += AudioSegment.silent(duration=max(0, pause_ms), frame_rate=segment.frame_rate)
        combined += segment
        previous = item

    return combined, trim_cfg, details


@router.post("/api/assemble_sanity/first5")
async def assemble_sanity_first5(request: Optional[MergeRequest] = None):
    with audio_queue_lock:
        if audio_current_job is not None or audio_queue or process_state["audio"].get("merge_running", False):
            raise HTTPException(
                status_code=400,
                detail="Audio queue is active. Wait for queued jobs to finish or cancel them first.",
            )

    selected = _collect_first_clips_for_sanity(limit=5)
    if not selected:
        raise HTTPException(status_code=404, detail="No existing clips found in project chunks.")

    selected_export = request.export if request and request.export is not None else _load_export_config()
    passed_export_payload = selected_export.model_dump()
    process_state["audio"]["logs"].append(
        f"[assemble-sanity] Input export config: {json.dumps(passed_export_payload, ensure_ascii=False, sort_keys=True)}"
    )
    print(
        f"[assemble-sanity] Input export config: {json.dumps(passed_export_payload, ensure_ascii=False, sort_keys=True)}",
        flush=True,
    )

    combined, resolved_trim, clip_details = _assemble_sanity_clips(selected, selected_export)
    if combined is None:
        raise HTTPException(status_code=500, detail="Failed to assemble sanity clip set.")

    exported = combined.export(SANITY_ASSEMBLE_FIRST5_PATH, format="wav")
    if hasattr(exported, "close"):
        exported.close()

    process_state["audio"]["logs"].append(
        f"[assemble-sanity] Resolved trim args: {json.dumps(resolved_trim, ensure_ascii=False, sort_keys=True)}"
    )
    process_state["audio"]["logs"].append(
        f"[assemble-sanity] Assembled first {len(selected)} clips -> {SANITY_ASSEMBLE_FIRST5_PATH}"
    )

    return {
        "status": "ok",
        "download_url": "/api/assemble_sanity/first5",
        "output_path": SANITY_ASSEMBLE_FIRST5_PATH,
        "clip_count": len(selected),
        "clips": clip_details,
        "passed_export": passed_export_payload,
        "resolved_trim": resolved_trim,
    }


@router.get("/api/assemble_sanity/first5")
async def get_assemble_sanity_first5():
    if not os.path.exists(SANITY_ASSEMBLE_FIRST5_PATH):
        raise HTTPException(status_code=404, detail="Sanity assembled clip not found. Generate it first.")
    return FileResponse(
        SANITY_ASSEMBLE_FIRST5_PATH,
        filename="assemble_sanity_first5.wav",
        media_type="audio/wav",
    )


@router.post("/api/assemble_sanity/first5_normalized")
async def assemble_sanity_first5_normalized(request: Optional[MergeRequest] = None):
    with audio_queue_lock:
        if audio_current_job is not None or audio_queue or process_state["audio"].get("merge_running", False):
            raise HTTPException(
                status_code=400,
                detail="Audio queue is active. Wait for queued jobs to finish or cancel them first.",
            )

    selected = _collect_first_clips_for_sanity(limit=5)
    if not selected:
        raise HTTPException(status_code=404, detail="No existing clips found in project chunks.")

    selected_export = request.export if request and request.export is not None else _load_export_config()
    passed_export_payload = selected_export.model_dump()
    process_state["audio"]["logs"].append(
        f"[assemble-sanity-normalized] Input export config: {json.dumps(passed_export_payload, ensure_ascii=False, sort_keys=True)}"
    )
    print(
        f"[assemble-sanity-normalized] Input export config: {json.dumps(passed_export_payload, ensure_ascii=False, sort_keys=True)}",
        flush=True,
    )

    combined, resolved_trim, clip_details = _assemble_sanity_clips(selected, selected_export)
    if combined is None:
        raise HTTPException(status_code=500, detail="Failed to assemble sanity clip set.")

    exported = combined.export(SANITY_ASSEMBLE_FIRST5_NORMALIZED_PATH, format="wav")
    if hasattr(exported, "close"):
        exported.close()

    normalized, normalize_result = project_manager._call_normalize_audio_file(
        SANITY_ASSEMBLE_FIRST5_NORMALIZED_PATH,
        export_config=selected_export,
        allow_short_single_pass=True,
    )
    if not normalized:
        raise HTTPException(status_code=500, detail=f"Normalization failed: {normalize_result}")

    process_state["audio"]["logs"].append(
        f"[assemble-sanity-normalized] Resolved trim args: {json.dumps(resolved_trim, ensure_ascii=False, sort_keys=True)}"
    )
    process_state["audio"]["logs"].append(
        f"[assemble-sanity-normalized] Assembled+normalized first {len(selected)} clips -> {SANITY_ASSEMBLE_FIRST5_NORMALIZED_PATH}"
    )

    return {
        "status": "ok",
        "download_url": "/api/assemble_sanity/first5_normalized",
        "output_path": SANITY_ASSEMBLE_FIRST5_NORMALIZED_PATH,
        "clip_count": len(selected),
        "clips": clip_details,
        "passed_export": passed_export_payload,
        "resolved_trim": resolved_trim,
    }


@router.get("/api/assemble_sanity/first5_normalized")
async def get_assemble_sanity_first5_normalized():
    if not os.path.exists(SANITY_ASSEMBLE_FIRST5_NORMALIZED_PATH):
        raise HTTPException(status_code=404, detail="Normalized sanity assembled clip not found. Generate it first.")
    return FileResponse(
        SANITY_ASSEMBLE_FIRST5_NORMALIZED_PATH,
        filename="assemble_sanity_first5_normalized.wav",
        media_type="audio/wav",
    )

@router.post("/api/export_audacity")
async def export_audacity_endpoint(request: Optional[MergeRequest] = None, background_tasks: BackgroundTasks = None):
    background_tasks = background_tasks or BackgroundTasks()
    if process_state["audacity_export"]["running"]:
        raise HTTPException(status_code=400, detail="Audacity export already running")

    def task():
        process_state["audacity_export"]["running"] = True
        process_state["audacity_export"]["logs"] = ["Starting Audacity export..."]
        try:
            selected_export = request.export if request and request.export is not None else _load_export_config()
            success, msg = project_manager.export_audacity(export_config=selected_export)
            if success:
                process_state["audacity_export"]["logs"].append(f"Export complete: {msg}")
            else:
                process_state["audacity_export"]["logs"].append(f"Export failed: {msg}")
        except Exception as e:
            process_state["audacity_export"]["logs"].append(f"Export error: {e}")
        finally:
            process_state["audacity_export"]["running"] = False

    background_tasks.add_task(task)
    return {"status": "started"}

@router.get("/api/export_audacity")
async def get_audacity_export():
    zip_path = os.path.join(ROOT_DIR, "audacity_export.zip")
    if not os.path.exists(zip_path):
        raise HTTPException(status_code=404, detail="Audacity export not found. Generate it first.")
    return FileResponse(zip_path, filename="audacity_export.zip", media_type="application/zip")

class M4bExportRequest(BaseModel):
    per_chunk_chapters: bool = False
    title: str = ""
    author: str = ""
    narrator: str = ""
    year: str = ""
    description: str = ""
    export: Optional[ExportConfig] = None

@router.post("/api/merge_m4b")
async def merge_m4b_endpoint(request: M4bExportRequest, background_tasks: BackgroundTasks):
    if process_state["m4b_export"]["running"]:
        raise HTTPException(status_code=400, detail="M4B export already running")

    def task():
        process_state["m4b_export"]["running"] = True
        process_state["m4b_export"]["logs"] = ["Starting M4B export..."]
        try:
            meta = {
                "title": request.title,
                "author": request.author,
                "narrator": request.narrator,
                "year": request.year,
                "description": request.description,
                "cover_path": os.path.join(ROOT_DIR, "m4b_cover.jpg") if os.path.exists(os.path.join(ROOT_DIR, "m4b_cover.jpg")) else "",
            }
            success, msg = project_manager.merge_m4b(
                per_chunk_chapters=request.per_chunk_chapters,
                metadata=meta,
                export_config=request.export or _load_export_config(),
            )
            if success:
                process_state["m4b_export"]["logs"].append(f"Export complete: {msg}")
            else:
                process_state["m4b_export"]["logs"].append(f"Export failed: {msg}")
        except Exception as e:
            process_state["m4b_export"]["logs"].append(f"Export error: {e}")
        finally:
            process_state["m4b_export"]["running"] = False

    background_tasks.add_task(task)
    return {"status": "started"}

@router.get("/api/audiobook_m4b")
async def get_audiobook_m4b():
    if not os.path.exists(M4B_PATH):
        raise HTTPException(status_code=404, detail="M4B audiobook not found. Export it first.")
    return FileResponse(M4B_PATH, filename="audiobook.m4b", media_type="audio/mp4")

@router.post("/api/m4b_cover")
async def upload_m4b_cover(file: UploadFile = File(...)):
    """Upload a cover image for M4B export."""
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    cover_path = os.path.join(ROOT_DIR, "m4b_cover.jpg")
    content = await file.read()
    with open(cover_path, "wb") as f:
        f.write(content)
    return {"status": "uploaded", "path": cover_path}

@router.delete("/api/m4b_cover")
async def delete_m4b_cover():
    """Remove the uploaded cover image."""
    cover_path = os.path.join(ROOT_DIR, "m4b_cover.jpg")
    if os.path.exists(cover_path):
        os.remove(cover_path)
    return {"status": "removed"}

@router.post("/api/generate_batch")
async def generate_batch_endpoint(request: BatchGenerateRequest, background_tasks: BackgroundTasks):
    """Generate multiple chunks in parallel using configured worker count."""
    indices = request.indices
    if not indices:
        raise HTTPException(status_code=400, detail="No chunk indices provided")
    settings = _load_audio_worker_settings()
    return _enqueue_audio_job(
        "parallel",
        indices,
        label=request.label or f"Parallel render ({len(indices)} chunks)",
        scope=request.scope or "custom",
    ) | {"workers": settings["workers"]}

@router.post("/api/generate_batch_fast")
async def generate_batch_fast_endpoint(request: BatchGenerateRequest, background_tasks: BackgroundTasks):
    """Generate multiple chunks using batch TTS API with single seed. Faster but less flexible.
    Requires custom Qwen3-TTS with /generate_batch endpoint."""
    indices = request.indices
    if not indices:
        raise HTTPException(status_code=400, detail="No chunk indices provided")
    settings = _load_audio_worker_settings()
    return _enqueue_audio_job(
        "batch_fast",
        indices,
        label=request.label or f"Batch render ({len(indices)} chunks)",
        scope=request.scope or "custom",
    ) | {
        "batch_seed": settings["batch_seed"],
        "batch_size": settings["batch_size"],
    }

@router.post("/api/cancel_audio")
async def cancel_audio():
    """Cancel the current audio job and clear any queued jobs."""
    global audio_recovery_request
    cancelled_or_cleared = False
    with audio_queue_condition:
        cleared = len(audio_queue)
        now = time.time()
        while audio_queue:
            job = audio_queue.pop(0)
            job["status"] = "cancelled"
            job["finished_at"] = now
            _record_audio_recent_job_locked(job)
        if cleared:
            cancelled_or_cleared = True

        if audio_current_job is not None:
            process_state["audio"]["cancel"] = True
            audio_recovery_request = None
            _append_audio_log_locked(f"[CANCEL] Cancellation requested for job #{audio_current_job['id']}")
            if cleared:
                _append_audio_log_locked(f"[CANCEL] Cleared {cleared} queued job(s)")
            abandoned = _abandon_audio_job_locked(
                audio_current_job,
                audio_current_job.get("run_token"),
                "User requested cancellation",
                status="cancelled",
            )
            cancelled_or_cleared = True
            if abandoned:
                _refresh_audio_process_state_locked(persist=True)
                if os.path.exists(AUDIO_QUEUE_STATE_PATH):
                    os.remove(AUDIO_QUEUE_STATE_PATH)
                return {"status": "cancelled", "cleared_queued_jobs": cleared}
            _refresh_audio_process_state_locked(persist=True)
            if os.path.exists(AUDIO_QUEUE_STATE_PATH):
                os.remove(AUDIO_QUEUE_STATE_PATH)
            return {"status": "cancelling", "cleared_queued_jobs": cleared}

        if cleared:
            audio_recovery_request = None
            _append_audio_log_locked(f"[CANCEL] Cleared {cleared} queued job(s)")
            _refresh_audio_process_state_locked(persist=True)
            if os.path.exists(AUDIO_QUEUE_STATE_PATH):
                os.remove(AUDIO_QUEUE_STATE_PATH)
            return {"status": "cancelled", "cleared_queued_jobs": cleared}

    # Not running — still reset any stuck "generating" chunks (e.g. from a crash)
    reset_count = project_manager.reset_generating_chunks()
    with audio_queue_condition:
        audio_recovery_request = None
        process_state["audio"]["cancel"] = False
        _refresh_audio_process_state_locked(persist=True)
    if os.path.exists(AUDIO_QUEUE_STATE_PATH):
        os.remove(AUDIO_QUEUE_STATE_PATH)
    return {"status": "not_running", "reset_chunks": reset_count}

## ── Saved Scripts ──────────────────────────────────────────────
