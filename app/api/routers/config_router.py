from fastapi import APIRouter
from .. import shared as _shared

globals().update({k: v for k, v in vars(_shared).items() if not k.startswith("__")})

router = APIRouter()

@router.get("/")
async def read_index():
    return FileResponse(
        os.path.join(STATIC_DIR, "index.html"),
        headers={"Cache-Control": "no-cache, no-store, must-revalidate"}
    )

@router.get("/favicon.ico")
async def read_favicon():
    favicon_path = os.path.join(ROOT_DIR, "icon.png")
    if os.path.exists(favicon_path):
        return FileResponse(favicon_path, media_type="image/png")
    raise HTTPException(status_code=404, detail="Favicon not found")

@router.get("/api/config")
async def get_config():
    config_changed = False
    default_config = {
        "llm": {
            "base_url": "http://localhost:11434/v1",
            "api_key": "local",
            "model_name": "richardyoung/qwen3-14b-abliterated:Q8_0"
        },
        "tts": {
            "mode": "local",
            "url": "http://127.0.0.1:7860",
            "device": "auto",
            "auto_regenerate_bad_clips": False,
            "auto_regenerate_bad_clip_attempts": 3
        },
        "prompts": {
            "system_prompt": "",
            "user_prompt": "",
            "voice_prompt": ""
        },
        "proofread": {
            "certainty_threshold": 1.0
        },
        "generation": {
            "legacy_mode": False,
        },
        "export": {
            "silence_between_speakers_ms": 500,
            "silence_same_speaker_ms": 250,
            "silence_end_of_chapter_ms": 3000,
            "silence_paragraph_ms": 750,
            "normalize_enabled": True,
            "normalize_target_lufs_mono": -18.0,
            "normalize_target_lufs_stereo": -16.0,
            "normalize_true_peak_dbtp": -1.0,
            "normalize_lra": 11.0,
        },
        "ui": {
            "dark_mode": True,
        },
    }

    if not os.path.exists(CONFIG_PATH):
        sys_prompt, usr_prompt = load_default_prompts()
        default_config["prompts"]["system_prompt"] = sys_prompt
        default_config["prompts"]["user_prompt"] = usr_prompt
        try:
            rev_sys, rev_usr = load_review_prompts()
            default_config["prompts"]["review_system_prompt"] = rev_sys
            default_config["prompts"]["review_user_prompt"] = rev_usr
        except RuntimeError:
            pass
        try:
            attr_sys, attr_usr = load_attribution_prompts()
            default_config["prompts"]["attribution_system_prompt"] = attr_sys
            default_config["prompts"]["attribution_user_prompt"] = attr_usr
        except RuntimeError:
            pass
        try:
            default_config["prompts"]["voice_prompt"] = load_voice_prompt()
        except RuntimeError:
            pass
        config = default_config
    else:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            config = json.load(f)

    # Ensure prompts section exists with defaults from file
    if "prompts" not in config:
        sys_prompt, usr_prompt = load_default_prompts()
        prompts = {"system_prompt": sys_prompt, "user_prompt": usr_prompt}
        try:
            rev_sys, rev_usr = load_review_prompts()
            prompts["review_system_prompt"] = rev_sys
            prompts["review_user_prompt"] = rev_usr
        except RuntimeError:
            pass
        try:
            attr_sys, attr_usr = load_attribution_prompts()
            prompts["attribution_system_prompt"] = attr_sys
            prompts["attribution_user_prompt"] = attr_usr
        except RuntimeError:
            pass
        try:
            prompts["voice_prompt"] = load_voice_prompt()
        except RuntimeError:
            pass
        prompts["dialogue_identification_system_prompt"] = DEFAULT_DIALOGUE_IDENTIFICATION_PROMPT
        prompts["temperament_extraction_system_prompt"] = DEFAULT_TEMPERAMENT_EXTRACTION_PROMPT
        config["prompts"] = prompts
        config_changed = True
    else:
        if not config["prompts"].get("system_prompt") or not config["prompts"].get("user_prompt"):
            sys_prompt, usr_prompt = load_default_prompts()
            if not config["prompts"].get("system_prompt"):
                config["prompts"]["system_prompt"] = sys_prompt
                config_changed = True
            if not config["prompts"].get("user_prompt"):
                config["prompts"]["user_prompt"] = usr_prompt
                config_changed = True
        if not config["prompts"].get("review_system_prompt") or not config["prompts"].get("review_user_prompt"):
            try:
                rev_sys, rev_usr = load_review_prompts()
                if not config["prompts"].get("review_system_prompt"):
                    config["prompts"]["review_system_prompt"] = rev_sys
                    config_changed = True
                if not config["prompts"].get("review_user_prompt"):
                    config["prompts"]["review_user_prompt"] = rev_usr
                    config_changed = True
            except RuntimeError:
                pass  # review_prompts.txt missing or malformed — leave fields empty
        if not config["prompts"].get("attribution_system_prompt") or not config["prompts"].get("attribution_user_prompt"):
            try:
                attr_sys, attr_usr = load_attribution_prompts()
                if not config["prompts"].get("attribution_system_prompt"):
                    config["prompts"]["attribution_system_prompt"] = attr_sys
                    config_changed = True
                if not config["prompts"].get("attribution_user_prompt"):
                    config["prompts"]["attribution_user_prompt"] = attr_usr
                    config_changed = True
            except RuntimeError:
                pass
        if not config["prompts"].get("voice_prompt"):
            try:
                config["prompts"]["voice_prompt"] = load_voice_prompt()
                config_changed = True
            except RuntimeError:
                pass
        if not config["prompts"].get("dialogue_identification_system_prompt"):
            config["prompts"]["dialogue_identification_system_prompt"] = DEFAULT_DIALOGUE_IDENTIFICATION_PROMPT
            config_changed = True
        if not config["prompts"].get("temperament_extraction_system_prompt"):
            config["prompts"]["temperament_extraction_system_prompt"] = DEFAULT_TEMPERAMENT_EXTRACTION_PROMPT
            config_changed = True

    # Include current input file info if available
    if "proofread" not in config or not isinstance(config.get("proofread"), dict):
        config["proofread"] = {"certainty_threshold": 1.0}
        config_changed = True
    else:
        if config["proofread"].get("certainty_threshold") is None:
            config["proofread"]["certainty_threshold"] = 1.0
            config_changed = True

    export_defaults = default_config["export"]
    if "export" not in config or not isinstance(config.get("export"), dict):
        config["export"] = dict(export_defaults)
        config_changed = True
    else:
        for key, value in export_defaults.items():
            if config["export"].get(key) is None:
                config["export"][key] = value
                config_changed = True

    generation_defaults = default_config["generation"]
    if "generation" not in config or not isinstance(config.get("generation"), dict):
        config["generation"] = dict(generation_defaults)
        config_changed = True
    else:
        for key, value in generation_defaults.items():
            if config["generation"].get(key) is None:
                config["generation"][key] = value
                config_changed = True

    ui_defaults = default_config["ui"]
    if "ui" not in config or not isinstance(config.get("ui"), dict):
        config["ui"] = dict(ui_defaults)
        config_changed = True
    else:
        for key, value in ui_defaults.items():
            if config["ui"].get(key) is None:
                config["ui"][key] = value
                config_changed = True

    config["render_prep_complete"] = False
    state_path = os.path.join(ROOT_DIR, "state.json")
    if os.path.exists(state_path):
        try:
            with open(state_path, "r", encoding="utf-8") as sf:
                state = json.load(sf)
            input_path = state.get("input_file_path", "")
            if input_path and os.path.exists(input_path):
                config["current_file"] = os.path.basename(input_path)
            config["render_prep_complete"] = bool(state.get("render_prep_complete"))
            config["generation_mode_locked"] = bool(state.get("generation_mode_locked"))
        except (json.JSONDecodeError, ValueError):
            pass
    else:
        config["generation_mode_locked"] = False
    if "generation_mode_locked" not in config:
        config["generation_mode_locked"] = False

    if config_changed:
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)

    return config

@router.get("/api/default_prompts")
async def get_default_prompts():
    system_prompt, user_prompt = load_default_prompts()
    result = {
        "system_prompt": system_prompt,
        "user_prompt": user_prompt
    }
    try:
        review_sys, review_usr = load_review_prompts()
        result["review_system_prompt"] = review_sys
        result["review_user_prompt"] = review_usr
    except RuntimeError:
        pass
    try:
        attribution_sys, attribution_usr = load_attribution_prompts()
        result["attribution_system_prompt"] = attribution_sys
        result["attribution_user_prompt"] = attribution_usr
    except RuntimeError:
        pass
    try:
        result["voice_prompt"] = load_voice_prompt()
    except RuntimeError:
        pass
    result["dialogue_identification_system_prompt"] = DEFAULT_DIALOGUE_IDENTIFICATION_PROMPT
    result["temperament_extraction_system_prompt"] = DEFAULT_TEMPERAMENT_EXTRACTION_PROMPT
    return result

@router.post("/api/config")
async def save_config(config: AppConfig):
    payload = config.model_dump()

    existing_config = {}
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                existing_config = json.load(f)
        except (json.JSONDecodeError, ValueError):
            existing_config = {}

    prompts = payload.get("prompts") or {}
    existing_prompts = existing_config.get("prompts") or {}
    if prompts.get("voice_prompt") is None:
        if existing_prompts.get("voice_prompt") is not None:
            prompts["voice_prompt"] = existing_prompts.get("voice_prompt")
        else:
            try:
                prompts["voice_prompt"] = load_voice_prompt()
            except RuntimeError:
                pass
    payload["prompts"] = prompts
    if payload.get("ui") is None and isinstance(existing_config.get("ui"), dict):
        payload["ui"] = existing_config.get("ui")

    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    # Reset engine so it picks up new TTS settings on next use
    project_manager.engine = None
    return {"status": "saved"}

@router.post("/api/config/preferences")
async def save_preferences(update: PreferencesUpdate):
    existing_config = {}
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                existing_config = json.load(f)
        except (json.JSONDecodeError, ValueError):
            existing_config = {}

    generation_cfg = existing_config.get("generation")
    if not isinstance(generation_cfg, dict):
        generation_cfg = {}
    ui_cfg = existing_config.get("ui")
    if not isinstance(ui_cfg, dict):
        ui_cfg = {}

    if update.legacy_mode is not None:
        generation_cfg["legacy_mode"] = bool(update.legacy_mode)
    if update.dark_mode is not None:
        ui_cfg["dark_mode"] = bool(update.dark_mode)

    existing_config["generation"] = generation_cfg
    existing_config["ui"] = ui_cfg

    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(existing_config, f, indent=2, ensure_ascii=False)

    return {"status": "saved"}

@router.post("/api/generation_mode_lock")
async def set_generation_mode_lock(update: GenerationModeLockUpdate):
    state = _load_project_state_payload()
    state["generation_mode_locked"] = bool(update.locked)
    if update.trigger:
        state["generation_mode_lock_trigger"] = str(update.trigger)
    _save_project_state_payload(state)
    return {
        "status": "saved",
        "locked": bool(state.get("generation_mode_locked")),
        "trigger": state.get("generation_mode_lock_trigger"),
    }

@router.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOADS_DIR, file.filename)
    async with aiofiles.open(file_path, 'wb') as out_file:
        content = await file.read()
        await out_file.write(content)

    # Save input path to state.json to be compatible with original scripts if needed
    state_path = os.path.join(ROOT_DIR, "state.json")
    state = {}
    if os.path.exists(state_path):
        with open(state_path, "r", encoding="utf-8") as f:
            try:
                state = json.load(f)
            except (json.JSONDecodeError, ValueError):
                pass

    state["input_file_path"] = file_path
    state["render_prep_complete"] = False
    state.pop(PROCESSING_STAGE_MARKERS_KEY, None)
    with open(state_path, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)

    source_type = "text"
    chapter_count = None
    try:
        source_document = load_source_document(file_path)
        source_type = source_document.get("type", "text")
        if source_type == "epub":
            chapter_count = len(source_document.get("chapters", []))
    except Exception as e:
        logger.warning("Source inspection failed for '%s': %s", file.filename, e)

    return {
        "filename": file.filename,
        "path": file_path,
        "source_type": source_type,
        "chapter_count": chapter_count,
    }


@router.get("/api/script_ingestion/preflight")
async def get_script_ingestion_preflight():
    return _script_ingestion_preflight_summary()


