import asyncio

from fastapi import APIRouter
from llm import LMStudioModelLoadService, ToolCapabilityService, clear_llm_gateway_cache
from .. import shared as _shared
from factory_prompt_defaults import load_factory_default_prompts
from runtime_layout import LAYOUT

globals().update({k: v for k, v in vars(_shared).items() if not k.startswith("__")})

router = APIRouter()

_PROMPT_SEPARATOR = "\n\n---SEPARATOR---\n\n"
_DEFAULT_PROMPTS_PATH = LAYOUT.prompt_default_path
_REVIEW_PROMPTS_PATH = LAYOUT.prompt_review_path
_ATTRIBUTION_PROMPTS_PATH = LAYOUT.prompt_attribution_path
_VOICE_PROMPT_PATH = LAYOUT.prompt_voice_path
_DIALOGUE_PROMPT_PATH = LAYOUT.prompt_dialogue_path
_TEMPERAMENT_PROMPT_PATH = LAYOUT.prompt_temperament_path
_TOOL_CAPABILITY_TIMEOUT_SECONDS = 5
_LMSTUDIO_MODEL_LOAD_TIMEOUT_SECONDS = 120


class ToolCapabilityRequest(BaseModel):
    base_url: str
    api_key: str = ""
    model_name: str


class LMStudioModelLoadRequest(BaseModel):
    model_name: str = ""
    base_url: str = ""
    api_key: str | None = None
    context_length: int | None = None
    eval_batch_size: int | None = None
    flash_attention: bool | None = None
    num_experts: int | None = None
    offload_kv_cache_to_gpu: bool | None = None
    echo_load_config: bool = False


class LMStudioUnloadAllModelsRequest(BaseModel):
    base_url: str = ""
    api_key: str | None = None


class LMStudioListModelsRequest(BaseModel):
    base_url: str = ""
    api_key: str | None = None


def _request_json(url: str, api_key: str):
    service = ToolCapabilityService(timeout_seconds=_TOOL_CAPABILITY_TIMEOUT_SECONDS)
    return service._request_json(url, api_key)


def _lmstudio_request_json(url: str, api_key: str):
    service = LMStudioModelLoadService(timeout_seconds=_LMSTUDIO_MODEL_LOAD_TIMEOUT_SECONDS)
    return service._request_json(url, api_key)


def _post_json(url: str, payload: dict, api_key: str):
    service = LMStudioModelLoadService(timeout_seconds=_LMSTUDIO_MODEL_LOAD_TIMEOUT_SECONDS)
    return service._post_json(url, payload, api_key)


def _is_openrouter_url(base_url: str) -> bool:
    return ToolCapabilityService.is_openrouter_url(base_url)


def _normalize_lm_studio_origin(base_url: str) -> str:
    return ToolCapabilityService.normalize_lm_studio_origin(base_url)


def _model_name_matches_lm_studio(model: dict, model_name: str) -> bool:
    return ToolCapabilityService.model_name_matches_lm_studio(model, model_name)


def _tool_capability_service() -> ToolCapabilityService:
    return ToolCapabilityService(
        timeout_seconds=_TOOL_CAPABILITY_TIMEOUT_SECONDS,
        request_json_fn=_request_json,
    )


def _lmstudio_model_load_service() -> LMStudioModelLoadService:
    return LMStudioModelLoadService(
        timeout_seconds=_LMSTUDIO_MODEL_LOAD_TIMEOUT_SECONDS,
        request_json_fn=_lmstudio_request_json,
        post_json_fn=_post_json,
    )


def _verify_openrouter_tool_capability(base_url: str, api_key: str, model_name: str):
    return _tool_capability_service().verify_openrouter_tool_capability(
        base_url,
        api_key,
        model_name,
    ).to_dict()


def _verify_lm_studio_tool_capability(base_url: str, api_key: str, model_name: str):
    return _tool_capability_service().verify_lm_studio_tool_capability(
        base_url,
        api_key,
        model_name,
    ).to_dict()


def verify_tool_capability(base_url: str, api_key: str, model_name: str):
    return _tool_capability_service().verify_tool_capability(
        base_url,
        api_key,
        model_name,
    ).to_dict()


def load_lmstudio_model(
    *,
    base_url: str,
    api_key: str,
    model_name: str,
    context_length: int | None = None,
    eval_batch_size: int | None = None,
    flash_attention: bool | None = None,
    num_experts: int | None = None,
    offload_kv_cache_to_gpu: bool | None = None,
    echo_load_config: bool = False,
):
    return _lmstudio_model_load_service().load_model(
        base_url=base_url,
        api_key=api_key,
        model_name=model_name,
        context_length=context_length,
        eval_batch_size=eval_batch_size,
        flash_attention=flash_attention,
        num_experts=num_experts,
        offload_kv_cache_to_gpu=offload_kv_cache_to_gpu,
        echo_load_config=echo_load_config,
    )


def unload_all_lmstudio_models(
    *,
    base_url: str,
    api_key: str,
):
    return _lmstudio_model_load_service().unload_all_models(
        base_url=base_url,
        api_key=api_key,
    )


def list_lmstudio_models(
    *,
    base_url: str,
    api_key: str,
):
    payload = _lmstudio_model_load_service().list_models(
        base_url=base_url,
        api_key=api_key,
    )
    models = payload.get("models") if isinstance(payload, dict) else None
    if not isinstance(models, list):
        raise RuntimeError("LM Studio returned an unexpected model list.")

    normalized_models = []
    for item in models:
        if not isinstance(item, dict):
            continue
        key = str(item.get("key") or "").strip()
        display_name = str(item.get("display_name") or "").strip() or key
        if not key and not display_name:
            continue

        loaded_instance_ids = []
        for instance in item.get("loaded_instances") or []:
            if not isinstance(instance, dict):
                continue
            instance_id = str(instance.get("id") or "").strip()
            if instance_id:
                loaded_instance_ids.append(instance_id)
        capabilities = item.get("capabilities") if isinstance(item.get("capabilities"), dict) else {}
        trained_for_tool_use = capabilities.get("trained_for_tool_use")
        if trained_for_tool_use is not None:
            trained_for_tool_use = bool(trained_for_tool_use)

        normalized_models.append(
            {
                "key": key,
                "display_name": display_name,
                "loaded_instance_ids": sorted(set(loaded_instance_ids)),
                "trained_for_tool_use": trained_for_tool_use,
            }
        )
    return {"status": "ok", "models": normalized_models}


def _read_saved_llm_config() -> dict:
    if not os.path.exists(CONFIG_PATH):
        return {}
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (json.JSONDecodeError, ValueError, OSError):
        return {}

    llm_payload = payload.get("llm")
    if not isinstance(llm_payload, dict):
        return {}
    return llm_payload


def _normalize_llm_base_for_cache(raw_base_url: str) -> str:
    value = str(raw_base_url or "").strip().rstrip("/")
    if not value:
        return ""
    if not value.endswith("/v1"):
        value = f"{value}/v1"
    return value


def _llm_cache_key(llm_payload: dict | None) -> tuple[str, str]:
    payload = llm_payload if isinstance(llm_payload, dict) else {}
    base_url = _normalize_llm_base_for_cache(payload.get("base_url") or "")
    model_name = str(payload.get("model_name") or "").strip()
    return base_url, model_name


def _write_prompt_pair(path: str, system_prompt: str, user_prompt: str):
    content = f"{(system_prompt or '').strip()}{_PROMPT_SEPARATOR}{(user_prompt or '').strip()}\n"
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def _write_single_prompt(path: str, prompt: str):
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"{(prompt or '').strip()}\n")


def _sync_prompt_files(prompts: dict, existing_prompts: dict):
    merged = dict(existing_prompts or {})
    for key, value in (prompts or {}).items():
        if value is not None:
            merged[key] = value

    system_prompt = (merged.get("system_prompt") or "").strip()
    user_prompt = (merged.get("user_prompt") or "").strip()
    if not system_prompt or not user_prompt:
        try:
            default_sys, default_usr = load_default_prompts()
            system_prompt = system_prompt or default_sys
            user_prompt = user_prompt or default_usr
        except RuntimeError:
            pass

    review_system_prompt = (merged.get("review_system_prompt") or "").strip()
    review_user_prompt = (merged.get("review_user_prompt") or "").strip()
    if not review_system_prompt or not review_user_prompt:
        try:
            review_sys_default, review_usr_default = load_review_prompts()
            review_system_prompt = review_system_prompt or review_sys_default
            review_user_prompt = review_user_prompt or review_usr_default
        except RuntimeError:
            pass

    attribution_system_prompt = (merged.get("attribution_system_prompt") or "").strip()
    attribution_user_prompt = (merged.get("attribution_user_prompt") or "").strip()
    if not attribution_system_prompt or not attribution_user_prompt:
        try:
            attr_sys_default, attr_usr_default = load_attribution_prompts()
            attribution_system_prompt = attribution_system_prompt or attr_sys_default
            attribution_user_prompt = attribution_user_prompt or attr_usr_default
        except RuntimeError:
            pass

    voice_prompt = (merged.get("voice_prompt") or "").strip()
    if not voice_prompt:
        try:
            voice_prompt = load_voice_prompt()
        except RuntimeError:
            voice_prompt = ""

    dialogue_prompt = (merged.get("dialogue_identification_system_prompt") or "").strip()
    if not dialogue_prompt:
        try:
            dialogue_prompt = load_dialogue_identification_prompt()
        except RuntimeError:
            dialogue_prompt = DEFAULT_DIALOGUE_IDENTIFICATION_PROMPT

    temperament_prompt = (merged.get("temperament_extraction_system_prompt") or "").strip()
    if not temperament_prompt:
        try:
            temperament_prompt = load_temperament_extraction_prompt()
        except RuntimeError:
            temperament_prompt = DEFAULT_TEMPERAMENT_EXTRACTION_PROMPT

    if system_prompt and user_prompt:
        _write_prompt_pair(_DEFAULT_PROMPTS_PATH, system_prompt, user_prompt)
        merged["system_prompt"] = system_prompt
        merged["user_prompt"] = user_prompt
    if review_system_prompt and review_user_prompt:
        _write_prompt_pair(_REVIEW_PROMPTS_PATH, review_system_prompt, review_user_prompt)
        merged["review_system_prompt"] = review_system_prompt
        merged["review_user_prompt"] = review_user_prompt
    if attribution_system_prompt and attribution_user_prompt:
        _write_prompt_pair(_ATTRIBUTION_PROMPTS_PATH, attribution_system_prompt, attribution_user_prompt)
        merged["attribution_system_prompt"] = attribution_system_prompt
        merged["attribution_user_prompt"] = attribution_user_prompt
    if voice_prompt:
        _write_single_prompt(_VOICE_PROMPT_PATH, voice_prompt)
        merged["voice_prompt"] = voice_prompt
    if dialogue_prompt:
        _write_single_prompt(_DIALOGUE_PROMPT_PATH, dialogue_prompt)
        merged["dialogue_identification_system_prompt"] = dialogue_prompt
    if temperament_prompt:
        _write_single_prompt(_TEMPERAMENT_PROMPT_PATH, temperament_prompt)
        merged["temperament_extraction_system_prompt"] = temperament_prompt

    return merged

@router.get("/")
async def read_index():
    return FileResponse(
        os.path.join(STATIC_DIR, "index.html"),
        headers={"Cache-Control": "no-cache, no-store, must-revalidate"}
    )

@router.get("/favicon.ico")
async def read_favicon():
    return FileResponse(os.path.join(REPO_DIR, "favicon.ico"), media_type="image/x-icon")

@router.get("/api/config")
async def get_config():
    ensure_runtime_config_exists(CONFIG_PATH, CONFIG_DEFAULT_PATH)
    config_changed = False
    default_config = {
        "llm": {
            "base_url": "",
            "api_key": "",
            "model_name": "",
            "llm_workers": 1
        },
        "tts": {
            "mode": "local",
            "local_backend": "auto",
            "url": "http://127.0.0.1:7860",
            "device": "auto",
            "language": "English",
            "parallel_workers": 4,
            "batch_seed": None,
            "compile_codec": False,
            "batch_group_by_type": False,
            "sub_batch_enabled": True,
            "sub_batch_min_size": 4,
            "sub_batch_ratio": 5.0,
            "sub_batch_max_chars": 3000,
            "sub_batch_max_items": 0,
            "script_max_length": 250,
            "auto_regenerate_bad_clips": True,
            "auto_regenerate_bad_clip_attempts": 3
        },
        "prompts": {
            "system_prompt": "",
            "user_prompt": "",
            "voice_prompt": ""
        },
        "proofread": {
            "certainty_threshold": 0.75
        },
        "generation": {
            "legacy_mode": False,
            "temperament_words": 150,
            "script_error_retry_attempts": 3,
        },
        "export": {
            "silence_between_speakers_ms": 500,
            "silence_same_speaker_ms": 250,
            "silence_end_of_chapter_ms": 3000,
            "silence_paragraph_ms": 750,
            "trim_clip_silence_enabled": True,
            "trim_silence_threshold_dbfs": -50.0,
            "trim_min_silence_len_ms": 150,
            "trim_keep_padding_ms": 40,
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

    if not isinstance(config, dict):
        config = {}
        config_changed = True

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
        try:
            prompts["dialogue_identification_system_prompt"] = load_dialogue_identification_prompt()
        except RuntimeError:
            prompts["dialogue_identification_system_prompt"] = DEFAULT_DIALOGUE_IDENTIFICATION_PROMPT
        try:
            prompts["temperament_extraction_system_prompt"] = load_temperament_extraction_prompt()
        except RuntimeError:
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
            try:
                config["prompts"]["dialogue_identification_system_prompt"] = load_dialogue_identification_prompt()
            except RuntimeError:
                config["prompts"]["dialogue_identification_system_prompt"] = DEFAULT_DIALOGUE_IDENTIFICATION_PROMPT
            config_changed = True
        if not config["prompts"].get("temperament_extraction_system_prompt"):
            try:
                config["prompts"]["temperament_extraction_system_prompt"] = load_temperament_extraction_prompt()
            except RuntimeError:
                config["prompts"]["temperament_extraction_system_prompt"] = DEFAULT_TEMPERAMENT_EXTRACTION_PROMPT
            config_changed = True

    # Include current input file info if available
    if "proofread" not in config or not isinstance(config.get("proofread"), dict):
        config["proofread"] = {"certainty_threshold": 0.75}
        config_changed = True
    else:
        if config["proofread"].get("certainty_threshold") is None:
            config["proofread"]["certainty_threshold"] = 0.75
            config_changed = True
    if "tts" not in config or not isinstance(config.get("tts"), dict):
        config["tts"] = dict(default_config["tts"])
        config_changed = True
    else:
        for key, value in default_config["tts"].items():
            if config["tts"].get(key) is None:
                config["tts"][key] = value
                config_changed = True
        if not config["tts"].get("local_backend"):
            config["tts"]["local_backend"] = "auto"
            config_changed = True

    if "llm" not in config or not isinstance(config.get("llm"), dict):
        config["llm"] = dict(default_config["llm"])
        config_changed = True
    else:
        for key, value in default_config["llm"].items():
            if config["llm"].get(key) is None:
                config["llm"][key] = value
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
            else:
                if config.get("current_file") is not None:
                    config_changed = True
                config["current_file"] = None
            config["render_prep_complete"] = bool(state.get("render_prep_complete"))
            config["generation_mode_locked"] = bool(state.get("generation_mode_locked"))
        except (json.JSONDecodeError, ValueError):
            pass
    else:
        if config.get("current_file") is not None:
            config_changed = True
        config["current_file"] = None
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
    try:
        result["dialogue_identification_system_prompt"] = load_dialogue_identification_prompt()
    except RuntimeError:
        result["dialogue_identification_system_prompt"] = DEFAULT_DIALOGUE_IDENTIFICATION_PROMPT
    try:
        result["temperament_extraction_system_prompt"] = load_temperament_extraction_prompt()
    except RuntimeError:
        result["temperament_extraction_system_prompt"] = DEFAULT_TEMPERAMENT_EXTRACTION_PROMPT
    return result


@router.get("/api/factory_default_prompts")
async def get_factory_default_prompts():
    return load_factory_default_prompts()

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
    old_llm_key = _llm_cache_key(existing_config.get("llm"))

    prompts = payload.get("prompts") or {}
    existing_prompts = existing_config.get("prompts") or {}
    payload["prompts"] = _sync_prompt_files(prompts, existing_prompts)
    if payload.get("ui") is None and isinstance(existing_config.get("ui"), dict):
        payload["ui"] = existing_config.get("ui")

    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    llm_cache_cleared = False
    if _llm_cache_key(payload.get("llm")) != old_llm_key:
        clear_llm_gateway_cache()
        llm_cache_cleared = True
    # Reset engine so it picks up new TTS settings on next use
    project_manager.engine = None
    return {"status": "saved", "llm_cache_cleared": llm_cache_cleared}


@router.post("/api/config/setup")
async def save_setup_config(update: SetupConfigUpdate):
    existing_config = {}
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                existing_config = json.load(f)
        except (json.JSONDecodeError, ValueError):
            existing_config = {}
    old_llm_key = _llm_cache_key(existing_config.get("llm"))

    engine_reset = False

    if update.llm is not None:
        existing_config["llm"] = {
            **(existing_config.get("llm") or {}),
            **update.llm.model_dump(),
        }

    if update.tts is not None:
        new_tts = update.tts.model_dump()
        old_tts = existing_config.get("tts") or {}
        hidden_local_backend = str(old_tts.get("local_backend") or "").strip()
        # The Setup tab does not expose local_backend yet, so keep the stored value
        # when autosave posts the UI default back.
        if hidden_local_backend and str(new_tts.get("local_backend") or "").strip() == "auto":
            new_tts["local_backend"] = hidden_local_backend
        tts_changed = any(new_tts.get(k) != old_tts.get(k) for k in new_tts)
        existing_config["tts"] = {**old_tts, **new_tts}
        if tts_changed:
            project_manager.engine = None
            engine_reset = True

    if update.prompts is not None:
        prompts = update.prompts.model_dump()
        existing_prompts = existing_config.get("prompts") or {}
        existing_config["prompts"] = _sync_prompt_files(prompts, existing_prompts)

    if update.generation is not None:
        existing_config["generation"] = {
            **(existing_config.get("generation") or {}),
            **update.generation.model_dump(),
        }

    if update.proofread is not None:
        existing_config["proofread"] = {
            **(existing_config.get("proofread") or {}),
            **update.proofread,
        }

    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(existing_config, f, indent=2, ensure_ascii=False)
    llm_cache_cleared = False
    if _llm_cache_key(existing_config.get("llm")) != old_llm_key:
        clear_llm_gateway_cache()
        llm_cache_cleared = True

    return {
        "status": "saved",
        "engine_reset": engine_reset,
        "llm_cache_cleared": llm_cache_cleared,
    }


@router.post("/api/config/export")
async def save_export_config(export: ExportConfig):
    payload = export.model_dump()
    existing_config = {}
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                existing_config = json.load(f)
        except (json.JSONDecodeError, ValueError):
            existing_config = {}

    if not isinstance(existing_config, dict):
        existing_config = {}
    existing_export = existing_config.get("export")
    if not isinstance(existing_export, dict):
        existing_export = {}

    existing_export.update(payload)
    existing_config["export"] = existing_export

    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(existing_config, f, indent=2, ensure_ascii=False)

    return {"status": "saved", "export": existing_export}

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


@router.post("/api/config/verify_tool_capability")
async def verify_model_tool_capability(request: ToolCapabilityRequest):
    return await asyncio.to_thread(
        verify_tool_capability,
        request.base_url,
        request.api_key,
        request.model_name,
    )


@router.post("/api/config/lmstudio/load_model")
async def load_lmstudio_model_endpoint(request: LMStudioModelLoadRequest):
    saved = _read_saved_llm_config()
    base_url = (request.base_url or "").strip() or str(saved.get("base_url") or "").strip()
    api_key = request.api_key if request.api_key is not None else str(saved.get("api_key") or "")
    model_name = (request.model_name or "").strip() or str(saved.get("model_name") or "").strip()

    if not model_name:
        raise HTTPException(status_code=400, detail="Model name is required.")
    if not base_url:
        raise HTTPException(status_code=400, detail="LLM base URL is required.")

    try:
        return await asyncio.to_thread(
            load_lmstudio_model,
            base_url=base_url,
            api_key=api_key,
            model_name=model_name,
            context_length=request.context_length,
            eval_batch_size=request.eval_batch_size,
            flash_attention=request.flash_attention,
            num_experts=request.num_experts,
            offload_kv_cache_to_gpu=request.offload_kv_cache_to_gpu,
            echo_load_config=request.echo_load_config,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Failed to load LM Studio model: {exc}") from exc


@router.post("/api/config/lmstudio/unload_all_models")
async def unload_all_lmstudio_models_endpoint(request: LMStudioUnloadAllModelsRequest):
    saved = _read_saved_llm_config()
    base_url = (request.base_url or "").strip() or str(saved.get("base_url") or "").strip()
    api_key = request.api_key if request.api_key is not None else str(saved.get("api_key") or "")

    if not base_url:
        raise HTTPException(status_code=400, detail="LLM base URL is required.")

    try:
        return await asyncio.to_thread(
            unload_all_lmstudio_models,
            base_url=base_url,
            api_key=api_key,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Failed to unload LM Studio models: {exc}") from exc


@router.post("/api/config/lmstudio/list_models")
async def list_lmstudio_models_endpoint(request: LMStudioListModelsRequest):
    saved = _read_saved_llm_config()
    base_url = (request.base_url or "").strip() or str(saved.get("base_url") or "").strip()
    api_key = request.api_key if request.api_key is not None else str(saved.get("api_key") or "")

    if not base_url:
        raise HTTPException(status_code=400, detail="LLM base URL is required.")

    try:
        return await asyncio.to_thread(
            list_lmstudio_models,
            base_url=base_url,
            api_key=api_key,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Failed to list LM Studio models: {exc}") from exc


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
    state.pop(NEW_MODE_STAGE_MARKERS_KEY, None)
    with open(state_path, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)

    source_type = "text"
    chapter_count = None
    try:
        source_document = load_source_document(file_path)
        source_type = source_document.get("type", "text")
        if source_type in {"epub", "docx"}:
            chapter_count = len(source_document.get("chapters", []))
    except Exception as e:
        logger.warning("Source inspection failed for '%s': %s", file.filename, e)

    return {
        "filename": file.filename,
        "path": file_path,
        "source_type": source_type,
        "chapter_count": chapter_count,
    }


@router.get("/api/narrator_overrides")
async def get_narrator_overrides():
    return project_manager.get_narrator_overrides()


@router.get("/api/narrator_candidates")
async def get_narrator_candidates(chapter: str):
    voices = []
    for voice in project_manager.script_store.list_voice_rows() if getattr(project_manager, "script_store", None) is not None else []:
        name = str((voice or {}).get("name") or "").strip()
        if not name:
            continue
        if bool(((voice or {}).get("config") or {}).get("narrates")):
            voices.append(name)
    include_narrator = any(
        project_manager._normalize_speaker_name(name) == project_manager._normalize_speaker_name("NARRATOR")
        for name in voices
    )
    ordered = project_manager.rank_chapter_narration_candidates(
        chapter,
        voices,
        include_narrator=include_narrator,
    )
    return {"chapter": chapter, "voices": ordered}


class NarratorOverrideRequest(BaseModel):
    chapter: str
    voice: str
    invalidate_audio: bool = False


@router.post("/api/narrator_overrides")
async def set_narrator_override(request: NarratorOverrideRequest):
    project_manager.set_narrator_override(request.chapter, request.voice)
    invalidated_clips = 0
    deleted_files = 0
    if request.invalidate_audio:
        chunks = project_manager.load_chunks()
        narrator_norm = project_manager._normalize_speaker_name("NARRATOR")
        affected = [
            i for i, c in enumerate(chunks)
            if project_manager._normalize_speaker_name(c.get("speaker", "")) == narrator_norm
            and (c.get("chapter") or "").strip() == request.chapter
            and c.get("audio_path")
        ]
        if affected:
            result = project_manager.invalidate_chunk_audio_indices(affected)
            invalidated_clips = result.get("invalidated_clips", 0)
            deleted_files = result.get("deleted_files", 0)
    return {"status": "saved", "invalidated_clips": invalidated_clips, "deleted_files": deleted_files}


@router.get("/api/script_ingestion/preflight")
async def get_script_ingestion_preflight():
    return _script_ingestion_preflight_summary()
