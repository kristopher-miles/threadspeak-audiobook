from fastapi import APIRouter
from .. import shared as _shared

globals().update({k: v for k, v in vars(_shared).items() if not k.startswith("__")})

router = APIRouter()

def _load_manifest(path):
    """Load a JSON manifest file, returning [] on missing or corrupt file."""
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, ValueError):
            pass
    return []

def _save_manifest(path, manifest):
    """Write a JSON manifest file."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)


def _normalize_saved_voice_name(name: str) -> str:
    return project_manager._normalize_speaker_name(name)


def _find_saved_voice_option_for_speaker(speaker: str):
    normalized_speaker = _normalize_saved_voice_name(speaker)
    if not normalized_speaker or normalized_speaker == _normalize_saved_voice_name("NARRATOR"):
        return None
    current_script_title = _normalize_saved_voice_name(project_manager._current_script_title() or "Project")

    def _build_rel_audio(directory_name: str, entry: dict) -> str:
        filename = (entry.get("filename") or "").strip()
        return f"{directory_name}/{filename}" if filename else ""

    def _match_score(entry: dict, fields):
        for priority, field in enumerate(fields):
            candidate = _normalize_saved_voice_name(entry.get(field, ""))
            if candidate and candidate == normalized_speaker:
                return priority
        return None

    best = None

    for entry in _load_manifest(CLONE_VOICES_MANIFEST):
        rel_audio = _build_rel_audio("clone_voices", entry)
        if not rel_audio or not os.path.exists(os.path.join(ROOT_DIR, rel_audio)):
            continue
        entry_script_title = _normalize_saved_voice_name(entry.get("script_title", ""))
        if not entry_script_title or entry_script_title != current_script_title:
            continue
        score = _match_score(entry, ("speaker", "name"))
        if score is None:
            continue
        candidate = {
            "type": "clone",
            "ref_audio": rel_audio,
            "ref_text": (entry.get("sample_text") or "").strip(),
            "source_name": (entry.get("speaker") or entry.get("name") or "").strip(),
            "priority": (0, score),
        }
        if best is None or candidate["priority"] < best["priority"]:
            best = candidate

    for entry in _load_manifest(DESIGNED_VOICES_MANIFEST):
        rel_audio = _build_rel_audio("designed_voices", entry)
        if not rel_audio or not os.path.exists(os.path.join(ROOT_DIR, rel_audio)):
            continue
        entry_script_title = _normalize_saved_voice_name(entry.get("script_title", ""))
        if not entry_script_title or entry_script_title != current_script_title:
            continue
        score = _match_score(entry, ("speaker", "name"))
        if score is None:
            continue
        candidate = {
            "type": "clone",
            "ref_audio": rel_audio,
            "ref_text": (entry.get("sample_text") or "").strip(),
            "source_name": (entry.get("speaker") or entry.get("name") or "").strip(),
            "priority": (1, score),
        }
        if best is None or candidate["priority"] < best["priority"]:
            best = candidate

    if best:
        best.pop("priority", None)
    return best


def _resolve_voice_alias_target(speaker: str, alias: str, known_names):
    normalized_speaker = _normalize_saved_voice_name(speaker)
    if not normalized_speaker:
        return None

    normalized_alias = _normalize_saved_voice_name(alias)
    if not normalized_alias or normalized_alias == normalized_speaker:
        return None

    for name in known_names:
        if _normalize_saved_voice_name(name) == normalized_alias:
            return name
    return None

@router.post("/api/voice_design/preview")
async def voice_design_preview(request: VoiceDesignPreviewRequest):
    """Generate a preview voice from a text description."""
    engine = project_manager.get_engine()
    if not engine:
        raise HTTPException(status_code=500, detail="Failed to initialize TTS engine")

    try:
        wav_path, sr = await asyncio.to_thread(
            engine.generate_voice_design,
            description=request.description,
            sample_text=_apply_project_dictionary(request.sample_text),
            language=request.language,
        )
        normalized, normalize_result = await asyncio.to_thread(
            project_manager._normalize_audio_file,
            wav_path,
            _load_export_config(),
            True,
        )
        if not normalized:
            raise RuntimeError(f"Failed to normalize voice design preview: {normalize_result}")
        # Return relative URL for the static mount
        filename = os.path.basename(wav_path)
        return {"status": "ok", "audio_url": f"/designed_voices/previews/{filename}"}
    except Exception as e:
        logger.error(f"Voice design preview failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/api/voice_design/save")
async def voice_design_save(request: VoiceDesignSaveRequest):
    """Save a preview voice as a permanent designed voice."""
    previews_dir = os.path.join(DESIGNED_VOICES_DIR, "previews")
    preview_path = os.path.join(previews_dir, request.preview_file)

    if not os.path.exists(preview_path):
        raise HTTPException(status_code=404, detail="Preview file not found")

    safe_name = _sanitize_name(request.name)
    if not safe_name:
        raise HTTPException(status_code=400, detail="Invalid voice name")

    # Generate unique ID
    voice_id = f"{safe_name}_{int(time.time())}"
    dest_filename = f"{voice_id}.wav"
    dest_path = os.path.join(DESIGNED_VOICES_DIR, dest_filename)

    shutil.copy2(preview_path, dest_path)
    normalized, normalize_result = project_manager._normalize_audio_file(
        dest_path,
        export_config=_load_export_config(),
        allow_short_single_pass=True,
    )
    if not normalized:
        raise HTTPException(status_code=500, detail=f"Failed to normalize saved voice clip: {normalize_result}")

    # Update manifest
    manifest = _load_manifest(DESIGNED_VOICES_MANIFEST)
    manifest.append({
        "id": voice_id,
        "name": request.name,
        "description": request.description,
        "sample_text": request.sample_text,
        "filename": dest_filename,
        "script_title": project_manager._current_script_title(),
    })
    _save_manifest(DESIGNED_VOICES_MANIFEST, manifest)

    logger.info(f"Designed voice saved: '{request.name}' as {dest_filename}")
    return {"status": "saved", "voice_id": voice_id}

@router.get("/api/voice_design/list")
async def voice_design_list():
    """List all saved designed voices."""
    return _load_manifest(DESIGNED_VOICES_MANIFEST)

@router.delete("/api/voice_design/{voice_id}")
async def voice_design_delete(voice_id: str):
    """Delete a saved designed voice."""
    manifest = _load_manifest(DESIGNED_VOICES_MANIFEST)
    entry = next((v for v in manifest if v["id"] == voice_id), None)
    if not entry:
        raise HTTPException(status_code=404, detail="Voice not found")

    # Delete WAV file
    wav_path = os.path.join(DESIGNED_VOICES_DIR, entry["filename"])
    if os.path.exists(wav_path):
        os.remove(wav_path)

    # Remove from manifest
    manifest = [v for v in manifest if v["id"] != voice_id]
    _save_manifest(DESIGNED_VOICES_MANIFEST, manifest)

    logger.info(f"Designed voice deleted: {voice_id}")
    return {"status": "deleted", "voice_id": voice_id}
