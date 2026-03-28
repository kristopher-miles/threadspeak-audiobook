from fastapi import APIRouter
from .. import shared as _shared

globals().update({k: v for k, v in vars(_shared).items() if not k.startswith("__")})

router = APIRouter()

@router.get("/api/voices")
async def get_voices():
    # Parse voices directly from the current script (no stale cache)
    voices_list = []
    line_counts: dict[str, int] = {}
    if os.path.exists(SCRIPT_PATH):
        try:
            script_data = _load_project_script_document()["entries"]
            voices_set = set()
            for entry in script_data:
                speaker = (entry.get("speaker") or entry.get("type") or "").strip()
                if speaker:
                    voices_set.add(speaker)
                    line_counts[speaker] = line_counts.get(speaker, 0) + 1
            voices_list = sorted(voices_set)
            # Update voices.json for compatibility with other tools
            with open(VOICES_PATH, "w", encoding="utf-8") as f:
                json.dump(voices_list, f, indent=2, ensure_ascii=False)
        except (json.JSONDecodeError, ValueError):
            pass

    if not voices_list:
        return []

    # Combine with config
    voice_config = {}
    if os.path.exists(VOICE_CONFIG_PATH):
        with open(VOICE_CONFIG_PATH, "r", encoding="utf-8") as f:
            voice_config = json.load(f)

    # Count paragraphs per speaker from paragraphs.json (new pipeline only)
    para_counts: dict[str, int] = {}
    paragraphs_path = os.path.join(ROOT_DIR, "paragraphs.json")
    if os.path.exists(paragraphs_path):
        try:
            with open(paragraphs_path, "r", encoding="utf-8") as f:
                para_doc = json.load(f)
            for p in para_doc.get("paragraphs", []):
                spk = (p.get("speaker") or "").strip()
                if spk:
                    para_counts[spk] = para_counts.get(spk, 0) + 1
        except Exception:
            pass

    result = []
    chunks = project_manager.load_chunks()
    for voice_name in voices_list:
        config = voice_config.get(voice_name, {})
        sample_suggestion = project_manager.suggest_design_sample_text(voice_name, chunks)
        ref_audio = (config.get("ref_audio") or "").strip()
        ref_audio_path = os.path.join(ROOT_DIR, ref_audio) if ref_audio and not os.path.isabs(ref_audio) else ref_audio
        design_clone_loaded = bool(ref_audio and ref_audio_path and os.path.exists(ref_audio_path))
        result.append({
            "name": voice_name,
            "config": config,
            "suggested_sample_text": sample_suggestion,
            "design_clone_loaded": design_clone_loaded,
            "line_count": line_counts.get(voice_name, 0),
            "paragraph_count": para_counts.get(voice_name, 0),
        })
    return result

@router.post("/api/parse_voices")
async def parse_voices(background_tasks: BackgroundTasks):
    if process_state["voices"]["running"]:
         raise HTTPException(status_code=400, detail="Voice parsing already running")

    background_tasks.add_task(run_process, [sys.executable, "-u", "parse_voices.py"], "voices")
    return {"status": "started"}


@router.get("/api/dictionary")
async def get_dictionary():
    return {"entries": _load_project_dictionary_entries()}


@router.post("/api/dictionary")
async def save_dictionary(request: DictionarySaveRequest):
    document = save_script_document(
        SCRIPT_PATH,
        dictionary=clean_dictionary_entries([entry.model_dump() for entry in request.entries]),
    )
    return {"status": "saved", "entries": document["dictionary"]}

@router.post("/api/save_voice_config")
async def save_voice_config(config_data: Dict[str, VoiceConfigItem]):
    # Read existing to preserve any fields not sent?
    # For now, we assume frontend sends full config or we just overwrite specific keys

    current_config = {}
    if os.path.exists(VOICE_CONFIG_PATH):
        with open(VOICE_CONFIG_PATH, "r", encoding="utf-8") as f:
            try:
                current_config = json.load(f)
            except (json.JSONDecodeError, ValueError):
                pass

    # Update current config with new data
    for voice_name, config in config_data.items():
        # Convert Pydantic model to dict
        current_config[voice_name] = config.model_dump()

    with open(VOICE_CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(current_config, f, indent=2, ensure_ascii=False)

    return {"status": "saved"}


@router.post("/api/voices/save_config")
async def save_voice_config_with_invalidation(request: VoiceConfigSaveRequest):
    running_task = _any_project_task_running()
    if running_task:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot update voices while {running_task} work is running",
        )

    new_config = {
        voice_name: config_item.model_dump()
        for voice_name, config_item in (request.config or {}).items()
    }
    result = project_manager.save_voice_config_with_invalidation(
        new_config,
        confirm_invalidation=bool(request.confirm_invalidation),
    )
    return result


def suggest_voice_description_sync(speaker: str):
    speaker = (speaker or "").strip()
    if not speaker:
        raise ValueError("Speaker is required")

    config = asyncio.run(get_config())
    prompts = config.get("prompts", {})
    prompt_template = (prompts.get("voice_prompt") or "").strip()
    if not prompt_template:
        prompt_template = load_voice_prompt()

    prompt_payload = project_manager.build_voice_suggestion_prompt(speaker, prompt_template)
    llm_config = config.get("llm", {})

    client = OpenAI(
        base_url=llm_config.get("base_url", "http://localhost:11434/v1"),
        api_key=llm_config.get("api_key", "local"),
        timeout=float(llm_config.get("timeout", 600)),
    )
    response = client.chat.completions.create(
        model=llm_config.get("model_name", "local-model"),
        messages=[{"role": "user", "content": prompt_payload["prompt"]}],
    )

    content = response.choices[0].message.content if response.choices else ""
    voice = _extract_voice_field(content)
    if not voice:
        raise ValueError("Model response did not include a valid JSON voice field")

    return {
        "status": "ok",
        "speaker": speaker,
        "voice": voice,
        "matched_paragraphs": len(prompt_payload["paragraphs"]),
        "context_chars": prompt_payload["context_chars"],
    }


def run_voice_processing_task(run_id: str, stop_check=None, relay_fn=None):
    success = False

    def ensure_active():
        if stop_check and stop_check():
            raise WorkflowPauseRequested()
        if not _task_is_current("voices", run_id):
            raise WorkflowPauseRequested()

    def log(message: str):
        ensure_active()
        _append_task_log("voices", run_id, message)
        if relay_fn:
            try:
                relay_fn(message)
            except Exception:
                pass

    try:
        if not os.path.exists(SCRIPT_PATH):
            raise FileNotFoundError("No annotated script found. Generate a script first.")

        chunks = project_manager.load_chunks()
        voices = awaitable_get_voices_sync()
        if not voices:
            log("No voices detected in the current script.")
            log("Task voices completed successfully.")
            return True

        voice_config = {}
        if os.path.exists(VOICE_CONFIG_PATH):
            with open(VOICE_CONFIG_PATH, "r", encoding="utf-8") as f:
                try:
                    voice_config = json.load(f)
                except (json.JSONDecodeError, ValueError):
                    voice_config = {}

        known_names = {voice["name"] for voice in voices}
        updated_config = copy.deepcopy(voice_config)
        changed = False

        for voice in voices:
            speaker = voice["name"]
            entry = updated_config.setdefault(speaker, {})
            if not entry.get("type"):
                entry["type"] = "design"
                changed = True
            ref_audio = (entry.get("ref_audio") or "").strip()
            ref_audio_path = os.path.join(ROOT_DIR, ref_audio) if ref_audio else ""
            reusable_match = _find_saved_voice_option_for_speaker(speaker)
            if reusable_match and not (ref_audio and os.path.exists(ref_audio_path)):
                entry["type"] = reusable_match["type"]
                entry["ref_audio"] = reusable_match["ref_audio"]
                if reusable_match.get("ref_text") and not (entry.get("ref_text") or "").strip():
                    entry["ref_text"] = reusable_match["ref_text"]
                changed = True
                log(
                    f"Auto-populated {speaker} from saved voice '{reusable_match.get('source_name') or reusable_match['ref_audio']}'."
                )
                continue
            if not (entry.get("ref_text") or "").strip():
                suggested = voice.get("suggested_sample_text") or project_manager.suggest_design_sample_text(speaker, chunks)
                if suggested:
                    entry["ref_text"] = suggested
                    changed = True

        if changed:
            with open(VOICE_CONFIG_PATH, "w", encoding="utf-8") as f:
                json.dump(updated_config, f, indent=2, ensure_ascii=False)

        eligible = []
        for voice in voices:
            speaker = voice["name"]
            entry = updated_config.get(speaker, {})
            alias = (entry.get("alias") or "").strip()
            alias_target = _resolve_voice_alias_target(speaker, alias, known_names)
            if alias_target:
                log(f"Skipping {speaker}: aliased to {alias_target}.")
                continue
            if entry.get("type", "design") != "design":
                log(f"Skipping {speaker}: voice type is {entry.get('type')}.")
                continue
            ref_audio = (entry.get("ref_audio") or "").strip()
            ref_audio_path = os.path.join(ROOT_DIR, ref_audio) if ref_audio else ""
            if ref_audio and os.path.exists(ref_audio_path):
                log(f"Skipping {speaker}: reusable voice already exists.")
                continue
            eligible.append(speaker)

        if not eligible:
            log("No outstanding voices needed generation.")
            log("Task voices completed successfully.")
            return True

        failures = []
        created_count = 0
        for index, speaker in enumerate(eligible, start=1):
            ensure_active()
            voice_data = updated_config.setdefault(speaker, {})
            description = (voice_data.get("description") or "").strip()
            sample_text = (voice_data.get("ref_text") or "").strip() or project_manager.suggest_design_sample_text(speaker, chunks)

            try:
                if not description:
                    log(f"[{index}/{len(eligible)}] Suggesting voice description for {speaker}...")
                    suggestion = suggest_voice_description_sync(speaker)
                    description = suggestion["voice"].strip()
                    voice_data["description"] = description
                    with open(VOICE_CONFIG_PATH, "w", encoding="utf-8") as f:
                        json.dump(updated_config, f, indent=2, ensure_ascii=False)

                if not sample_text:
                    raise ValueError(f"No sample text available for '{speaker}'")

                log(f"[{index}/{len(eligible)}] Generating reusable voice for {speaker}...")
                try:
                    materialized = project_manager.materialize_design_voice(
                        speaker=speaker,
                        description=description,
                        sample_text=sample_text,
                        force=False,
                        voice_config=updated_config,
                        export_config=_load_export_config(),
                    )
                except TypeError as e:
                    # Backward-compatible path for tests/older manager stubs that
                    # do not accept the newer export_config keyword argument.
                    if "export_config" not in str(e):
                        raise
                    materialized = project_manager.materialize_design_voice(
                        speaker=speaker,
                        description=description,
                        sample_text=sample_text,
                        force=False,
                        voice_config=updated_config,
                    )
                updated_config = materialized["voice_config"]
                created_count += 1
                log(f"[{index}/{len(eligible)}] Created reusable voice for {speaker}.")
            except WorkflowPauseRequested:
                raise
            except Exception as e:
                failures.append((speaker, str(e)))
                log(f"[{index}/{len(eligible)}] Failed to create voice for {speaker}: {e}")
                continue

        if failures:
            failure_preview = ", ".join(f"{speaker} ({message})" for speaker, message in failures[:5])
            if len(failures) > 5:
                failure_preview += f", and {len(failures) - 5} more"
            raise RuntimeError(
                f"Created {created_count} voice(s), but {len(failures)} speaker(s) failed: {failure_preview}"
            )

        log("Task voices completed successfully.")
        success = True
    except WorkflowPauseRequested:
        logger.info("Voices task interrupted")
    except Exception as e:
        logger.error(f"Error running voice processing task: {e}")
        if _task_is_current("voices", run_id):
            _append_task_log("voices", run_id, f"Error: {str(e)}")
    finally:
        _finish_task_run("voices", run_id)
    return success


def awaitable_get_voices_sync():
    return asyncio.run(get_voices())


@router.post("/api/voices/suggest_description")
async def suggest_voice_description(request: VoiceDescriptionSuggestRequest):
    speaker = (request.speaker or "").strip()
    if not speaker:
        raise HTTPException(status_code=400, detail="Speaker is required")

    try:
        return await asyncio.to_thread(suggest_voice_description_sync, speaker)
    except Exception as e:
        logger.error(f"Voice description suggestion failed for {speaker}: {e}")
        raise HTTPException(status_code=500, detail=f"Voice suggestion request failed: {e}")


@router.post("/api/voices/design_generate")
async def generate_voice_design_clone(request: VoiceDesignGenerateRequest):
    speaker = (request.speaker or "").strip()
    if not speaker:
        raise HTTPException(status_code=400, detail="Speaker is required")

    try:
        result = await asyncio.to_thread(
            project_manager.materialize_design_voice,
            speaker=speaker,
            description=request.description,
            sample_text=request.sample_text,
            force=bool(request.force),
            export_config=_load_export_config(),
        )
    except Exception as e:
        logger.error(f"Voice design clone generation failed for {speaker}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    return {
        "status": "ok",
        "speaker": speaker,
        "voice_id": result["voice_id"],
        "name": result["display_name"],
        "filename": result["filename"],
        "audio_url": f"/{result['ref_audio']}?t={int(time.time())}",
        "ref_audio": result["ref_audio"],
        "ref_text": result["ref_text"],
        "generated_ref_text": result["generated_ref_text"],
    }


@router.post("/api/voices/clear_uploaded")
async def clear_uploaded_voices_for_current_script():
    running_task = _any_project_task_running()
    if running_task:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot clear uploaded voices while {running_task} work is running",
        )

    script_title = (project_manager._current_script_title() or "").strip() or "Project"
    normalized_title = _normalize_saved_voice_name(script_title) or _normalize_saved_voice_name("Project")
    normalized_title_prefix = f"{normalized_title}."

    current_speakers = set()
    if os.path.exists(SCRIPT_PATH):
        try:
            entries = _load_project_script_document()["entries"]
            for entry in entries:
                speaker = (entry.get("speaker") or entry.get("type") or "").strip()
                normalized = _normalize_saved_voice_name(speaker)
                if normalized:
                    current_speakers.add(normalized)
        except Exception:
            pass

    clone_manifest = _load_manifest(CLONE_VOICES_MANIFEST)
    kept_entries = []
    removed_entries = []

    for entry in clone_manifest:
        entry_script_title = _normalize_saved_voice_name(entry.get("script_title", ""))
        entry_name = _normalize_saved_voice_name(entry.get("name", ""))
        entry_speaker = _normalize_saved_voice_name(entry.get("speaker", ""))
        is_generated = bool(entry.get("generated"))

        title_match = bool(entry_script_title) and entry_script_title == normalized_title
        prefixed_name_match = bool(entry_name) and entry_name.startswith(normalized_title_prefix)
        temp_speaker_match = (
            normalized_title == _normalize_saved_voice_name("Project")
            and is_generated
            and bool(entry_speaker)
            and entry_speaker in current_speakers
        )

        if title_match or prefixed_name_match or temp_speaker_match:
            removed_entries.append(entry)
        else:
            kept_entries.append(entry)

    removed_relative_paths = []
    removed_files = 0
    for entry in removed_entries:
        filename = (entry.get("filename") or "").strip()
        if not filename:
            continue
        rel_path = f"clone_voices/{filename}"
        removed_relative_paths.append(rel_path)
        abs_path = os.path.join(CLONE_VOICES_DIR, filename)
        if os.path.exists(abs_path):
            try:
                os.remove(abs_path)
                removed_files += 1
            except OSError:
                pass

    if len(kept_entries) != len(clone_manifest):
        _save_manifest(CLONE_VOICES_MANIFEST, kept_entries)

    updated_voice_config = False
    if os.path.exists(VOICE_CONFIG_PATH):
        try:
            with open(VOICE_CONFIG_PATH, "r", encoding="utf-8") as f:
                voice_config = json.load(f)
        except (json.JSONDecodeError, ValueError):
            voice_config = {}

        removed_set = set(removed_relative_paths)
        for speaker, cfg in (voice_config or {}).items():
            if not isinstance(cfg, dict):
                continue
            ref_audio = (cfg.get("ref_audio") or "").strip()
            if ref_audio and ref_audio in removed_set:
                cfg["ref_audio"] = ""
                if "generated_ref_text" in cfg:
                    cfg["generated_ref_text"] = ""
                updated_voice_config = True

        if updated_voice_config:
            with open(VOICE_CONFIG_PATH, "w", encoding="utf-8") as f:
                json.dump(voice_config, f, indent=2, ensure_ascii=False)

    if project_manager.engine and hasattr(project_manager.engine, "clear_clone_prompt_cache"):
        try:
            project_manager.engine.clear_clone_prompt_cache()
        except Exception:
            pass

    return {
        "status": "ok",
        "script_title": script_title,
        "removed_manifest_entries": len(removed_entries),
        "removed_files": removed_files,
        "updated_voice_config": updated_voice_config,
    }
