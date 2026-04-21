from concurrent.futures import ThreadPoolExecutor, as_completed
from fastapi import APIRouter
from llm import (
    LLMClientFactory,
    LLMRuntimeConfig,
    VOICE_DESCRIPTION_CONTRACT,
    get_llm_gateway,
)
from .. import shared as _shared

globals().update({k: v for k, v in vars(_shared).items() if not k.startswith("__")})

router = APIRouter()
_LLM_CLIENT_FACTORY = LLMClientFactory()
_STRUCTURED_LLM_SERVICE = get_llm_gateway()

class NarratorThresholdRequest(BaseModel):
    value: int


class VoiceProfilePatchRequest(BaseModel):
    fields: VoiceConfigItem
    confirm_invalidation: bool = False


class VoiceProfileBatchRequest(BaseModel):
    config: Dict[str, VoiceConfigItem]
    confirm_invalidation: bool = False


class DisableNarratorNarrationRequest(BaseModel):
    config: Dict[str, VoiceConfigItem]


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


def _can_use_project_chunk_store() -> bool:
    manager_root = str(getattr(project_manager, "root_dir", "") or "").strip()
    route_root = str(ROOT_DIR or "").strip()
    if not manager_root or not route_root:
        return False
    return os.path.realpath(manager_root) == os.path.realpath(route_root)


def _compute_auto_narrator_aliases_for_route(
    voice_config: Dict[str, dict],
    line_counts: Dict[str, int],
    narrator_name: str,
) -> Dict[str, str]:
    refresh_aliases = getattr(project_manager, "refresh_auto_narrator_aliases", None)
    if callable(refresh_aliases) and _can_use_project_chunk_store():
        return refresh_aliases(
            voice_config=voice_config,
            line_counts=line_counts,
            narrator_name=narrator_name,
        )

    threshold = int(project_manager.get_narrator_threshold() or 0)
    if threshold <= 0 or not narrator_name:
        return {}

    aliases: Dict[str, str] = {}
    narrator_key = _normalized_speaker("NARRATOR")
    for voice_name, count in (line_counts or {}).items():
        if _normalized_speaker(voice_name) == narrator_key:
            continue
        config_key = _find_config_key_case_insensitive(voice_config, voice_name)
        config = voice_config.get(config_key, {}) if config_key else {}
        if str((config or {}).get("alias") or "").strip():
            continue
        if int(count or 0) < threshold:
            aliases[voice_name] = narrator_name
    return aliases


def _infer_name_aliases_from_voice_payloads(
    voices: List[Dict[str, object]],
    voice_config: Dict[str, dict],
) -> Dict[str, str]:
    voice_names = [str((voice or {}).get("name") or "").strip() for voice in (voices or []) if str((voice or {}).get("name") or "").strip()]
    line_counts = {
        name: int((voice or {}).get("line_count") or 0)
        for voice in (voices or [])
        for name in [str((voice or {}).get("name") or "").strip()]
        if name
    }
    para_counts_by_norm = {
        _normalized_speaker(str((voice or {}).get("name") or "").strip()): int((voice or {}).get("paragraph_count") or 0)
        for voice in (voices or [])
        if str((voice or {}).get("name") or "").strip()
    }
    return _infer_contained_name_aliases(voice_names, voice_config, line_counts, para_counts_by_norm)


def _merge_voice_config_patch(config_items: Dict[str, VoiceConfigItem] | None) -> Dict[str, dict]:
    incoming_config = _canonicalize_voice_config_keys({
        _canonical_speaker_name(voice_name): config_item.model_dump()
        for voice_name, config_item in (config_items or {}).items()
    })
    existing_config = _canonicalize_voice_config_keys(_load_runtime_voice_config())
    new_config = copy.deepcopy(existing_config)
    for voice_name, config_item in incoming_config.items():
        target_name = _canonical_speaker_name(voice_name)
        existing_key = _find_config_key_case_insensitive(new_config, target_name)
        new_config[existing_key or target_name] = config_item
    return _canonicalize_voice_config_keys(new_config)


def _load_runtime_voice_config() -> Dict[str, dict]:
    load_fn = getattr(project_manager, "_load_voice_config", None)
    if callable(load_fn):
        try:
            payload = load_fn()
            return payload if isinstance(payload, dict) else {}
        except Exception:
            return {}
    return {}


def _save_runtime_voice_config(config: Dict[str, dict]):
    save_fn = getattr(project_manager, "_save_voice_config", None)
    if callable(save_fn):
        save_fn(config)
        return
    raise RuntimeError("Voice config requires the project SQLite store")


@router.get("/api/voices")
async def get_voices():
    chunks = project_manager.load_chunks() if _can_use_project_chunk_store() else []
    script_store = getattr(project_manager, "script_store", None)
    voice_rows = script_store.list_voice_rows() if (_can_use_project_chunk_store() and script_store is not None) else []
    runtime_voice_config = _canonicalize_voice_config_keys(_load_runtime_voice_config())
    if not voice_rows and chunks:
        seen = {}
        for chunk in chunks:
            speaker = str((chunk or {}).get("speaker") or "").strip()
            normalized = _normalized_speaker(speaker)
            if not normalized:
                continue
            if normalized not in seen:
                canonical_name = _canonical_speaker_name(speaker)
                seen[normalized] = {
                    "name": canonical_name,
                    "config": runtime_voice_config.get(canonical_name, {}),
                    "line_count": 0,
                    "auto_narrator_alias": False,
                    "auto_alias_target": "",
                }
            seen[normalized]["line_count"] += 1
        voice_rows = list(seen.values())
    elif not voice_rows and _project_has_script_document():
        try:
            script_data = _load_project_script_document()["entries"]
            seen = {}
            for entry in script_data:
                speaker = (entry.get("speaker") or entry.get("type") or "").strip()
                normalized = _normalized_speaker(speaker)
                if not normalized or normalized in seen:
                    if normalized in seen:
                        seen[normalized]["line_count"] += 1
                    continue
                seen[normalized] = {
                    "name": _canonical_speaker_name(speaker),
                    "config": runtime_voice_config.get(_canonical_speaker_name(speaker), {}),
                    "line_count": 1,
                    "auto_narrator_alias": False,
                    "auto_alias_target": "",
                }
            voice_rows = list(seen.values())
        except (json.JSONDecodeError, ValueError):
            voice_rows = []

    if not voice_rows:
        return []

    # Count paragraphs per speaker from the persisted paragraphs document.
    para_counts_by_norm: dict[str, int] = {}
    try:
        para_doc = _load_project_paragraphs_document()
        for p in para_doc.get("paragraphs", []):
            spk = (p.get("speaker") or "").strip()
            if spk:
                normalized = _normalized_speaker(spk)
                if normalized:
                    para_counts_by_norm[normalized] = para_counts_by_norm.get(normalized, 0) + 1
    except Exception:
        pass

    result = []
    for voice in voice_rows:
        voice_name = str((voice or {}).get("name") or "").strip()
        config = dict((voice or {}).get("config") or {})
        sample_suggestion = project_manager.suggest_design_sample_text(voice_name, chunks)
        line_count = int((voice or {}).get("line_count") or 0)
        if line_count <= 0:
            continue
        auto_alias_target = str((voice or {}).get("auto_alias_target") or "")
        auto_narrator_alias = bool((voice or {}).get("auto_narrator_alias"))
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
            "auto_alias_target": auto_alias_target,
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

    background_tasks.add_task(run_process, [sys.executable, "-u", "-m", "scripts.parse_voices"], "voices")
    return {"status": "started"}


@router.get("/api/dictionary")
async def get_dictionary():
    return {"entries": _load_project_dictionary_entries()}


@router.post("/api/dictionary")
async def save_dictionary(request: DictionarySaveRequest):
    document = project_manager.script_store.replace_script_document(
        dictionary=clean_dictionary_entries([entry.model_dump() for entry in request.entries]),
        reason="save_dictionary",
        rebuild_chunks=False,
        wait=True,
    )
    return {"status": "saved", "entries": document["dictionary"]}

@router.post("/api/save_voice_config")
async def save_voice_config(config_data: Dict[str, VoiceConfigItem]):
    new_config = _merge_voice_config_patch(config_data)
    project_manager._save_voice_config(new_config)
    return {"status": "saved", "saved": len(config_data or {})}


@router.post("/api/voices/save_config")
async def save_voice_config_with_invalidation(request: VoiceConfigSaveRequest):
    new_config = _merge_voice_config_patch(request.config)
    result = project_manager.save_voice_config_with_invalidation(
        new_config,
        confirm_invalidation=bool(request.confirm_invalidation),
    )
    return result


@router.post("/api/voices/batch")
async def batch_save_voice_profiles(request: VoiceProfileBatchRequest):
    new_config = _merge_voice_config_patch(request.config)
    return project_manager.save_voice_config_with_invalidation(
        new_config,
        confirm_invalidation=bool(request.confirm_invalidation),
    )


@router.patch("/api/voices/{speaker}")
async def patch_voice_profile(speaker: str, request: VoiceProfilePatchRequest):
    return await batch_save_voice_profiles(
        VoiceProfileBatchRequest(
            config={speaker: request.fields},
            confirm_invalidation=bool(request.confirm_invalidation),
        )
    )


@router.post("/api/voices/narrator/disable")
async def disable_narrator_narration(request: DisableNarratorNarrationRequest):
    effective_config = _merge_voice_config_patch(request.config)
    result = project_manager.disable_narrator_narration_and_reassign_chapters(effective_config)
    if result.get("status") == "rejected":
        raise HTTPException(status_code=400, detail=result)
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


def _generation_max_tokens(config: Dict[str, dict] | None = None) -> int:
    generation_config = (config or {}).get("generation", {})
    try:
        return max(1, int(generation_config.get("max_tokens", 4096) or 4096))
    except (TypeError, ValueError):
        return 4096


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
    runtime = LLMRuntimeConfig.from_dict(
        llm_config,
        default_base_url="http://localhost:11434/v1",
        default_model_name="local-model",
        default_timeout=600.0,
    )
    client = _LLM_CLIENT_FACTORY.create_client(runtime)
    result = _STRUCTURED_LLM_SERVICE.run(
        client=client,
        runtime=runtime,
        messages=[{"role": "user", "content": prompt_payload["prompt"]}],
        contract=VOICE_DESCRIPTION_CONTRACT,
        max_tokens=_generation_max_tokens(config),
    )
    payload = result.parsed if isinstance(result.parsed, dict) else None
    voice = str((payload or {}).get("voice") or "").strip()
    if not voice and result.mode == "json":
        if payload:
            voice = _extract_voice_field(json.dumps(payload, ensure_ascii=False))
        if not voice:
            voice = _extract_voice_field(result.raw_payload)
        if not voice:
            voice = _extract_voice_field(result.text)
    if not voice:
        if result.mode == "tool":
            raise ValueError("Tool response did not include required 'voice' field")
        raise ValueError("Model response did not include a voice description")

    return {
        "status": "ok",
        "speaker": speaker,
        "voice": voice,
        "llm_mode": result.mode,
        "llm_tool_call_observed": bool(result.tool_call_observed),
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
        if not _project_has_script_document():
            raise FileNotFoundError("No annotated script found. Generate a script first.")

        chunks = project_manager.load_chunks()
        voices = awaitable_get_voices_sync()
        if not voices:
            log("No voices detected in the current script.")
            log("Task voices completed successfully.")
            return True

        voice_config = _canonicalize_voice_config_keys(_load_runtime_voice_config())

        known_names = {voice["name"] for voice in voices}
        updated_config = copy.deepcopy(voice_config)
        changed = False
        inferred_aliases = _infer_name_aliases_from_voice_payloads(voices, updated_config)

        for voice in voices:
            speaker = voice["name"]
            entry = updated_config.setdefault(speaker, {})
            if not entry.get("type"):
                entry["type"] = "design"
                changed = True
            inferred_alias = (inferred_aliases.get(speaker) or "").strip()
            if inferred_alias and not (entry.get("alias") or "").strip():
                entry["alias"] = inferred_alias
                changed = True
                log(f"Auto-aliased {speaker} to {inferred_alias}.")
            ref_audio = (entry.get("ref_audio") or "").strip()
            ref_audio_path = os.path.join(ROOT_DIR, ref_audio) if ref_audio else ""
            if not (entry.get("ref_text") or "").strip():
                suggested = voice.get("suggested_sample_text") or project_manager.suggest_design_sample_text(speaker, chunks)
                if suggested:
                    entry["ref_text"] = suggested
                    changed = True

        if changed:
            _save_runtime_voice_config(updated_config)

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
                _save_runtime_voice_config(updated_config)
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


@router.post("/api/voices/lmstudio_preflight_unload")
async def preflight_unload_lmstudio_models_for_voice_generation():
    return _attempt_lmstudio_unload_all_models("voices_bulk_generation")


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
    if _project_has_script_document():
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
    voice_config = _load_runtime_voice_config()
    if voice_config:
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
            _save_runtime_voice_config(voice_config)

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
