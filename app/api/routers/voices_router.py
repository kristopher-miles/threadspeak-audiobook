from concurrent.futures import ThreadPoolExecutor, as_completed
from fastapi import APIRouter
from .. import shared as _shared

globals().update({k: v for k, v in vars(_shared).items() if not k.startswith("__")})

router = APIRouter()

class NarratorThresholdRequest(BaseModel):
    value: int


def _normalized_speaker(name: str) -> str:
    return project_manager._normalize_speaker_name(name)


def _canonical_speaker_name(name: str) -> str:
    normalized = _normalized_speaker(name)
    if normalized == _normalized_speaker("NARRATOR"):
        return "NARRATOR"
    return (name or "").strip()


def _canonicalize_voice_config_keys(voice_config: Dict[str, dict]) -> Dict[str, dict]:
    """Collapse duplicate voice-config keys case-insensitively."""
    if not isinstance(voice_config, dict):
        return {}

    canonical = {}
    key_by_normalized = {}
    for raw_key, raw_value in voice_config.items():
        key = str(raw_key or "").strip()
        if not key:
            continue
        normalized = _normalized_speaker(key)
        if not normalized:
            continue
        chosen_key = key_by_normalized.get(normalized)
        if not chosen_key:
            chosen_key = _canonical_speaker_name(key)
            key_by_normalized[normalized] = chosen_key
            canonical[chosen_key] = dict(raw_value or {})
            continue

        # Merge duplicate entry into existing canonical key without clobbering
        # already-populated values.
        existing = canonical.setdefault(chosen_key, {})
        for field, value in dict(raw_value or {}).items():
            existing_value = existing.get(field)
            if existing_value in (None, "", []):
                existing[field] = value

    return canonical


def _find_config_key_case_insensitive(voice_config: Dict[str, dict], speaker: str):
    target = _normalized_speaker(speaker)
    for key in voice_config.keys():
        if _normalized_speaker(key) == target:
            return key
    return None


def _voice_alias_weight(name: str, line_counts: Dict[str, int], para_counts_by_norm: Dict[str, int]) -> int:
    line_count = int(line_counts.get(name, 0) or 0)
    if line_count > 0:
        return line_count
    return int(para_counts_by_norm.get(_normalized_speaker(name), 0) or 0)


def _name_tokens(name: str) -> List[str]:
    return [token for token in _normalized_speaker(name).split() if token]


def _contains_name_tokens(container_name: str, candidate_name: str) -> bool:
    container_tokens = _name_tokens(container_name)
    candidate_tokens = _name_tokens(candidate_name)
    candidate_len = len(candidate_tokens)
    if candidate_len == 0 or candidate_len > len(container_tokens):
        return False
    for index in range(len(container_tokens) - candidate_len + 1):
        if container_tokens[index:index + candidate_len] == candidate_tokens:
            return True
    return False


def _infer_contained_name_aliases(
    voice_names: List[str],
    voice_config: Dict[str, dict],
    line_counts: Dict[str, int],
    para_counts_by_norm: Dict[str, int],
) -> Dict[str, str]:
    suggested_aliases: Dict[str, str] = {}
    processed = set()

    for voice_name in voice_names:
        norm_a = _normalized_speaker(voice_name)
        if not norm_a or norm_a == _normalized_speaker("NARRATOR"):
            continue
        existing_a = voice_config.get(_find_config_key_case_insensitive(voice_config, voice_name) or voice_name, {})
        if (existing_a.get("alias") or "").strip():
            continue

        for other_name in voice_names:
            norm_b = _normalized_speaker(other_name)
            if not norm_b or norm_a == norm_b or norm_b == _normalized_speaker("NARRATOR"):
                continue

            pair_key = tuple(sorted((norm_a, norm_b)))
            if pair_key in processed:
                continue
            processed.add(pair_key)

            if not _contains_name_tokens(other_name, voice_name) and not _contains_name_tokens(voice_name, other_name):
                continue

            existing_b = voice_config.get(_find_config_key_case_insensitive(voice_config, other_name) or other_name, {})
            if (existing_b.get("alias") or "").strip():
                continue

            count_a = _voice_alias_weight(voice_name, line_counts, para_counts_by_norm)
            count_b = _voice_alias_weight(other_name, line_counts, para_counts_by_norm)
            if count_a <= 0 and count_b <= 0:
                continue

            alias_name, target_name = (
                (voice_name, other_name) if count_a <= count_b else (other_name, voice_name)
            )
            suggested_aliases.setdefault(alias_name, target_name)

    return suggested_aliases


@router.get("/api/voices")
async def get_voices():
    # Parse voices directly from the current script (no stale cache)
    voices_list = []
    line_counts: dict[str, int] = {}
    if os.path.exists(SCRIPT_PATH):
        try:
            script_data = _load_project_script_document()["entries"]
            voices_map = {}
            line_counts_by_norm: dict[str, int] = {}
            for entry in script_data:
                speaker = (entry.get("speaker") or entry.get("type") or "").strip()
                if speaker:
                    normalized = _normalized_speaker(speaker)
                    if not normalized:
                        continue
                    canonical_name = voices_map.get(normalized)
                    if not canonical_name:
                        canonical_name = _canonical_speaker_name(speaker)
                        voices_map[normalized] = canonical_name
                    line_counts_by_norm[normalized] = line_counts_by_norm.get(normalized, 0) + 1
            voices_list = sorted(voices_map.values(), key=lambda value: value.casefold())
            line_counts = {
                voices_map[norm]: count
                for norm, count in line_counts_by_norm.items()
                if norm in voices_map
            }
            # Update voices.json for compatibility with other tools
            with open(VOICES_PATH, "w", encoding="utf-8") as f:
                json.dump(voices_list, f, indent=2, ensure_ascii=False)
        except (json.JSONDecodeError, ValueError):
            pass

    if not voices_list:
        return []

    narrator_threshold = project_manager.get_narrator_threshold()
    narrator_name = next(
        (name for name in voices_list if _normalized_speaker(name) == _normalized_speaker("NARRATOR")),
        "",
    )

    # Combine with config
    voice_config = {}
    if os.path.exists(VOICE_CONFIG_PATH):
        with open(VOICE_CONFIG_PATH, "r", encoding="utf-8") as f:
            voice_config = _canonicalize_voice_config_keys(json.load(f))

    # Count paragraphs per speaker from paragraphs.json (new pipeline only)
    para_counts_by_norm: dict[str, int] = {}
    paragraphs_path = os.path.join(ROOT_DIR, "paragraphs.json")
    if os.path.exists(paragraphs_path):
        try:
            with open(paragraphs_path, "r", encoding="utf-8") as f:
                para_doc = json.load(f)
            for p in para_doc.get("paragraphs", []):
                spk = (p.get("speaker") or "").strip()
                if spk:
                    normalized = _normalized_speaker(spk)
                    if normalized:
                        para_counts_by_norm[normalized] = para_counts_by_norm.get(normalized, 0) + 1
        except Exception:
            pass

    result = []
    chunks = project_manager.load_chunks()
    inferred_aliases = _infer_contained_name_aliases(voices_list, voice_config, line_counts, para_counts_by_norm)
    for voice_name in voices_list:
        config_key = _find_config_key_case_insensitive(voice_config, voice_name)
        config = dict(voice_config.get(config_key, {}) if config_key else {})
        if not (config.get("alias") or "").strip():
            inferred_alias = inferred_aliases.get(voice_name)
            if inferred_alias:
                config["alias"] = inferred_alias
        sample_suggestion = project_manager.suggest_design_sample_text(voice_name, chunks)
        manual_alias = bool((config.get("alias") or "").strip())
        line_count = line_counts.get(voice_name, 0)
        auto_narrator_alias = (
            not manual_alias
            and bool(narrator_name)
            and _normalized_speaker(voice_name) != _normalized_speaker("NARRATOR")
            and line_count < narrator_threshold
        )
        ref_audio = (config.get("ref_audio") or "").strip()
        ref_audio_path = os.path.join(ROOT_DIR, ref_audio) if ref_audio and not os.path.isabs(ref_audio) else ref_audio
        design_clone_loaded = bool(ref_audio and ref_audio_path and os.path.exists(ref_audio_path))
        result.append({
            "name": voice_name,
            "config": config,
            "suggested_sample_text": sample_suggestion,
            "design_clone_loaded": design_clone_loaded,
            "line_count": line_count,
            "paragraph_count": para_counts_by_norm.get(_normalized_speaker(voice_name), 0),
            "auto_narrator_alias": auto_narrator_alias,
            "auto_alias_target": narrator_name if auto_narrator_alias else "",
        })
    return result


@router.get("/api/voices/settings")
async def get_voice_settings():
    return {"narrator_threshold": project_manager.get_narrator_threshold()}


@router.post("/api/voices/settings")
async def save_voice_settings(request: NarratorThresholdRequest):
    return {"status": "saved", "narrator_threshold": project_manager.set_narrator_threshold(request.value)}

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
        target_name = _canonical_speaker_name(voice_name)
        existing_key = _find_config_key_case_insensitive(current_config, target_name)
        current_config[existing_key or target_name] = config.model_dump()

    current_config = _canonicalize_voice_config_keys(current_config)

    with open(VOICE_CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(current_config, f, indent=2, ensure_ascii=False)

    return {"status": "saved"}


@router.post("/api/voices/save_config")
async def save_voice_config_with_invalidation(request: VoiceConfigSaveRequest):
    # Voice updates must remain available even during active processing so the
    # UI can prompt for invalidation and intentionally clear affected clips.
    incoming_config = _canonicalize_voice_config_keys({
        _canonical_speaker_name(voice_name): config_item.model_dump()
        for voice_name, config_item in (request.config or {}).items()
    })
    existing_config = _canonicalize_voice_config_keys(project_manager._load_voice_config())

    # Treat request payload as patch semantics: update submitted voices while
    # preserving unrelated voice entries already in the config file.
    new_config = copy.deepcopy(existing_config)
    for voice_name, config_item in incoming_config.items():
        target_name = _canonical_speaker_name(voice_name)
        existing_key = _find_config_key_case_insensitive(new_config, target_name)
        new_config[existing_key or target_name] = config_item
    new_config = _canonicalize_voice_config_keys(new_config)

    result = project_manager.save_voice_config_with_invalidation(
        new_config,
        confirm_invalidation=bool(request.confirm_invalidation),
    )
    return result


def _load_runtime_config() -> Dict[str, dict]:
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, ValueError):
            pass
    return {}


def _llm_worker_count(config: Dict[str, dict] | None = None) -> int:
    llm_config = (config or {}).get("llm", {})
    try:
        return max(1, int(llm_config.get("llm_workers", 1) or 1))
    except (TypeError, ValueError):
        return 1


def suggest_voice_description_sync(speaker: str):
    speaker = (speaker or "").strip()
    if not speaker:
        raise ValueError("Speaker is required")

    config = _load_runtime_config()
    prompts = config.get("prompts", {})
    prompt_template = (prompts.get("voice_prompt") or "").strip()
    if not prompt_template:
        prompt_template = load_voice_prompt()

    prompt_payload = project_manager.build_voice_suggestion_prompt(speaker, prompt_template)
    llm_config = config.get("llm", {})

    _base_url = llm_config.get("base_url", "http://localhost:11434/v1").rstrip("/")
    if not _base_url.endswith("/v1"):
        _base_url += "/v1"
    client = OpenAI(
        base_url=_base_url,
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
        "warning": prompt_payload.get("warning"),
        "context_source": prompt_payload.get("context_source"),
    }


def suggest_voice_descriptions_batch_sync(speakers: List[str], guard_fn=None, relay_fn=None):
    ordered_speakers = []
    seen = set()
    for raw_speaker in speakers or []:
        speaker = (raw_speaker or "").strip()
        if not speaker or speaker in seen:
            continue
        seen.add(speaker)
        ordered_speakers.append(speaker)

    if not ordered_speakers:
        return {"status": "ok", "workers": 0, "results": [], "failures": []}

    if guard_fn:
        guard_fn()

    config = _load_runtime_config()
    workers = min(_llm_worker_count(config), len(ordered_speakers))
    if relay_fn:
        relay_fn(
            f"Suggesting voice descriptions for {len(ordered_speakers)} speaker(s) with {workers} parallel LLM worker(s)..."
        )

    results_by_speaker = {}
    failures_by_speaker = {}
    future_map = {}

    with ThreadPoolExecutor(max_workers=workers) as executor:
        for speaker in ordered_speakers:
            if guard_fn:
                guard_fn()
            future = executor.submit(suggest_voice_description_sync, speaker)
            future_map[future] = speaker

        completed = 0
        for future in as_completed(future_map):
            if guard_fn:
                guard_fn()
            speaker = future_map[future]
            completed += 1
            try:
                result = future.result()
                if not isinstance(result, dict):
                    raise ValueError("Suggestion response must be a dictionary")
                result = dict(result)
                result.setdefault("speaker", speaker)
                results_by_speaker[speaker] = result
                if relay_fn:
                    relay_fn(f"[{completed}/{len(ordered_speakers)}] Suggested voice description for {speaker}.")
            except WorkflowPauseRequested:
                raise
            except Exception as e:
                failures_by_speaker[speaker] = str(e)
                if relay_fn:
                    relay_fn(f"[{completed}/{len(ordered_speakers)}] Failed to suggest voice for {speaker}: {e}")

    ordered_results = [results_by_speaker[speaker] for speaker in ordered_speakers if speaker in results_by_speaker]
    ordered_failures = [
        {"speaker": speaker, "error": failures_by_speaker[speaker]}
        for speaker in ordered_speakers
        if speaker in failures_by_speaker
    ]
    return {
        "status": "ok",
        "workers": workers,
        "results": ordered_results,
        "failures": ordered_failures,
    }


def _unload_bulk_voice_generation_state():
    unloaded = False
    try:
        unloaded = bool(project_manager.unload_tts_engine())
    except Exception:
        unloaded = False
    return unloaded


def run_voice_processing_task(run_id: str, stop_check=None, relay_fn=None):
    success = False
    attempted_bulk_generation = False

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
                    voice_config = _canonicalize_voice_config_keys(json.load(f))
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
            inferred_alias = ((voice.get("config") or {}).get("alias") or "").strip()
            if inferred_alias and not (entry.get("alias") or "").strip():
                entry["alias"] = inferred_alias
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

        attempted_bulk_generation = True
        failures = []
        created_count = 0
        speakers_with_suggestion_failures = set()
        to_suggest = []
        for speaker in eligible:
            voice_data = updated_config.setdefault(speaker, {})
            if not (voice_data.get("description") or "").strip():
                to_suggest.append(speaker)

        if to_suggest:
            suggestion_batch = suggest_voice_descriptions_batch_sync(
                to_suggest,
                guard_fn=ensure_active,
                relay_fn=log,
            )
            if suggestion_batch["results"]:
                for suggestion in suggestion_batch["results"]:
                    speaker = suggestion["speaker"]
                    updated_config.setdefault(speaker, {})["description"] = suggestion["voice"].strip()
                with open(VOICE_CONFIG_PATH, "w", encoding="utf-8") as f:
                    json.dump(updated_config, f, indent=2, ensure_ascii=False)
            for failure in suggestion_batch["failures"]:
                speaker = failure["speaker"]
                speakers_with_suggestion_failures.add(speaker)
                failures.append((speaker, f"suggestion failed: {failure['error']}"))

        for index, speaker in enumerate(eligible, start=1):
            ensure_active()
            voice_data = updated_config.setdefault(speaker, {})
            description = (voice_data.get("description") or "").strip()
            sample_text = (voice_data.get("ref_text") or "").strip() or project_manager.suggest_design_sample_text(speaker, chunks)

            if not description:
                if speaker not in speakers_with_suggestion_failures:
                    failures.append((speaker, "missing description"))
                log(f"[{index}/{len(eligible)}] Failed to create voice for {speaker}: missing description")
                continue

            try:
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
        if attempted_bulk_generation:
            unloaded = _unload_bulk_voice_generation_state()
            if _task_is_current("voices", run_id):
                _append_task_log(
                    "voices",
                    run_id,
                    "Unloaded bulk voice generation model state."
                    if unloaded
                    else "Bulk voice generation model state already unloaded.",
                )
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


@router.post("/api/voices/suggest_descriptions_bulk")
async def suggest_voice_descriptions_bulk(request: VoiceDescriptionSuggestBatchRequest):
    try:
        return await asyncio.to_thread(suggest_voice_descriptions_batch_sync, request.speakers)
    except Exception as e:
        logger.error(f"Bulk voice description suggestion failed: {e}")
        raise HTTPException(status_code=500, detail=f"Voice suggestion batch request failed: {e}")


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


@router.post("/api/voices/unload_bulk_generation")
async def unload_bulk_voice_generation():
    return {
        "status": "unloaded",
        "unloaded": _unload_bulk_voice_generation_state(),
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

    def _split_manifest_entries(manifest, *, is_generated_manifest=False):
        kept = []
        removed = []
        for entry in manifest:
            entry_script_title = _normalize_saved_voice_name(entry.get("script_title", ""))
            entry_name = _normalize_saved_voice_name(entry.get("name", ""))
            entry_speaker = _normalize_saved_voice_name(entry.get("speaker", ""))
            is_generated = bool(entry.get("generated"))

            title_match = bool(entry_script_title) and entry_script_title == normalized_title
            prefixed_name_match = bool(entry_name) and entry_name.startswith(normalized_title_prefix)
            temp_speaker_match = (
                normalized_title == _normalize_saved_voice_name("Project")
                and is_generated_manifest
                and is_generated
                and bool(entry_speaker)
                and entry_speaker in current_speakers
            )

            if title_match or prefixed_name_match or temp_speaker_match:
                removed.append(entry)
            else:
                kept.append(entry)
        return kept, removed

    clone_manifest = _load_manifest(CLONE_VOICES_MANIFEST)
    kept_clone_entries, removed_clone_entries = _split_manifest_entries(
        clone_manifest, is_generated_manifest=True
    )

    designed_manifest = _load_manifest(DESIGNED_VOICES_MANIFEST)
    kept_designed_entries, removed_designed_entries = _split_manifest_entries(
        designed_manifest, is_generated_manifest=False
    )

    removed_relative_paths = []
    removed_speakers = set()
    removed_files = 0

    def _remove_files_from_entries(entries, base_dir, rel_prefix):
        nonlocal removed_files
        for entry in entries:
            speaker_key = _normalize_saved_voice_name(entry.get("speaker", ""))
            if speaker_key:
                removed_speakers.add(speaker_key)
            filename = (entry.get("filename") or "").strip()
            if not filename:
                continue
            rel_path = f"{rel_prefix}/{filename}"
            removed_relative_paths.append(rel_path)
            abs_path = os.path.join(base_dir, filename)
            if os.path.exists(abs_path):
                try:
                    os.remove(abs_path)
                    removed_files += 1
                except OSError:
                    pass

    _remove_files_from_entries(removed_clone_entries, CLONE_VOICES_DIR, "clone_voices")
    _remove_files_from_entries(removed_designed_entries, DESIGNED_VOICES_DIR, "designed_voices")

    if len(kept_clone_entries) != len(clone_manifest):
        _save_manifest(CLONE_VOICES_MANIFEST, kept_clone_entries)
    if len(kept_designed_entries) != len(designed_manifest):
        _save_manifest(DESIGNED_VOICES_MANIFEST, kept_designed_entries)

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
            speaker_key = _normalize_saved_voice_name(speaker)
            should_clear_text = speaker_key in removed_speakers or (ref_audio and ref_audio in removed_set)
            if ref_audio and ref_audio in removed_set:
                cfg["ref_audio"] = ""
                updated_voice_config = True
            if should_clear_text:
                if cfg.get("ref_text"):
                    cfg["ref_text"] = ""
                    updated_voice_config = True
                if cfg.get("generated_ref_text"):
                    cfg["generated_ref_text"] = ""
                    updated_voice_config = True
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
        "removed_manifest_entries": len(removed_clone_entries) + len(removed_designed_entries),
        "removed_files": removed_files,
        "updated_voice_config": updated_voice_config,
    }
