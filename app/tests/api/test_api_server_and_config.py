"""API endpoint tests split by behavior domain."""

from ._common import *  # noqa: F401,F403
from . import _common as common

def test_server_reachable():
    r = get("/")
    assert_status(r, 200)
    if "text/html" not in r.headers.get("content-type", ""):
        raise TestFailure(f"Expected HTML, got {r.headers.get('content-type')}")


# ── Section 2: Config ───────────────────────────────────────

def test_get_config():
    r = get("/api/config")
    assert_status(r, 200)
    data = r.json()
    assert_key(data, "llm")
    assert_key(data, "tts")
    # current_file is optional when no source file has been selected yet.
    if "current_file" in data and data["current_file"] is not None and not isinstance(data["current_file"], str):
        raise TestFailure(f"current_file must be string or null, got {type(data['current_file']).__name__}")

def test_get_config_exposes_expected_defaults():
    r = get("/api/config")
    assert_status(r, 200)
    data = r.json()

    if (data.get("tts") or {}).get("provider") != "qwen3":
        raise TestFailure(f"Expected default tts.provider='qwen3', got {(data.get('tts') or {}).get('provider')!r}")

    if int((data.get("tts") or {}).get("script_max_length") or 0) != 250:
        raise TestFailure(f"Expected default script_max_length=250, got {(data.get('tts') or {}).get('script_max_length')!r}")

    proofread_threshold = (data.get("proofread") or {}).get("certainty_threshold")
    if float(proofread_threshold or 0.0) != 0.75:
        raise TestFailure(f"Expected default proofread certainty_threshold=0.75, got {proofread_threshold!r}")

def test_get_config_backfills_missing_defaults_with_expected_values():
    config_path = os.path.join(common.ACTIVE_APP_DIR, "config.json")

    with open(config_path, "r", encoding="utf-8") as f:
        original_config_raw = f.read()

    modified = json.loads(original_config_raw)
    modified.setdefault("tts", {})
    modified["tts"]["provider"] = None
    modified["tts"]["script_max_length"] = None
    modified["proofread"] = {"certainty_threshold": None}
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(modified, f, indent=2, ensure_ascii=False)

    try:
        r = get("/api/config")
        assert_status(r, 200)
        data = r.json()
        if (data.get("tts") or {}).get("provider") != "qwen3":
            raise TestFailure("GET /api/config did not backfill tts.provider='qwen3'")
        if int((data.get("tts") or {}).get("script_max_length") or 0) != 250:
            raise TestFailure("GET /api/config did not backfill tts.script_max_length=250")
        if float(((data.get("proofread") or {}).get("certainty_threshold") or 0.0)) != 0.75:
            raise TestFailure("GET /api/config did not backfill proofread.certainty_threshold=0.75")

        with open(config_path, "r", encoding="utf-8") as f:
            persisted = json.load(f)
        if (persisted.get("tts") or {}).get("provider") != "qwen3":
            raise TestFailure("Backfilled tts.provider was not persisted")
        if int((persisted.get("tts") or {}).get("script_max_length") or 0) != 250:
            raise TestFailure("Backfilled script_max_length was not persisted")
        if float(((persisted.get("proofread") or {}).get("certainty_threshold") or 0.0)) != 0.75:
            raise TestFailure("Backfilled proofread certainty_threshold was not persisted")
    finally:
        with open(config_path, "w", encoding="utf-8") as f:
            f.write(original_config_raw)

def test_get_config_bootstraps_local_config_from_default_template():
    config_path = os.path.join(common.ACTIVE_APP_DIR, "config.json")
    default_config_path = os.path.join(common.ACTIVE_APP_DIR, "config.default.json")

    with open(config_path, "r", encoding="utf-8") as f:
        original_config_raw = f.read()
    with open(default_config_path, "r", encoding="utf-8") as f:
        default_config = json.load(f)

    os.remove(config_path)

    try:
        r = get("/api/config")
        assert_status(r, 200)
        data = r.json()
        if not os.path.exists(config_path):
            raise TestFailure("GET /api/config did not recreate missing local config from default template")
        with open(config_path, "r", encoding="utf-8") as f:
            recreated = json.load(f)
        if recreated.get("llm", {}).get("api_key") != default_config.get("llm", {}).get("api_key"):
            raise TestFailure("Recreated local config did not come from the tracked default template")
        if data.get("llm", {}).get("base_url") != default_config.get("llm", {}).get("base_url"):
            raise TestFailure("Bootstrapped config did not preserve default llm.base_url")
        if data.get("llm", {}).get("api_key") != default_config.get("llm", {}).get("api_key"):
            raise TestFailure("Bootstrapped config did not preserve default llm.api_key")
        if data.get("llm", {}).get("model_name") != default_config.get("llm", {}).get("model_name"):
            raise TestFailure("Bootstrapped config did not preserve default llm.model_name")
        if data.get("llm", {}).get("llm_workers") != default_config.get("llm", {}).get("llm_workers"):
            raise TestFailure("Bootstrapped config did not preserve default llm.llm_workers")
        if data.get("tts", {}).get("script_max_length") != default_config.get("tts", {}).get("script_max_length"):
            raise TestFailure("Bootstrapped config did not preserve default template values")
        if data.get("tts", {}).get("provider") != default_config.get("tts", {}).get("provider"):
            raise TestFailure("Bootstrapped config did not preserve default tts.provider")
        if data.get("tts", {}).get("parallel_workers") != default_config.get("tts", {}).get("parallel_workers"):
            raise TestFailure("Bootstrapped config did not preserve default tts.parallel_workers")
        if data.get("tts", {}).get("auto_regenerate_bad_clips") != default_config.get("tts", {}).get("auto_regenerate_bad_clips"):
            raise TestFailure("Bootstrapped config did not preserve default tts.auto_regenerate_bad_clips")
        if data.get("tts", {}).get("auto_regenerate_bad_clip_attempts") != default_config.get("tts", {}).get("auto_regenerate_bad_clip_attempts"):
            raise TestFailure("Bootstrapped config did not preserve default tts.auto_regenerate_bad_clip_attempts")
    finally:
        with open(config_path, "w", encoding="utf-8") as f:
            f.write(original_config_raw)

def test_get_config_clears_stale_current_file_when_runtime_has_no_input():
    config_path = os.path.join(common.ACTIVE_APP_DIR, "config.json")
    state_path = os.path.join(common.ACTIVE_APP_DIR, "state.json")

    with open(config_path, "r", encoding="utf-8") as f:
        original_config_raw = f.read()

    original_state_raw = None
    if os.path.exists(state_path):
        with open(state_path, "r", encoding="utf-8") as f:
            original_state_raw = f.read()

    modified = json.loads(original_config_raw)
    modified["current_file"] = "stale-loaded-book.epub"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(modified, f, indent=2, ensure_ascii=False)

    with open(state_path, "w", encoding="utf-8") as f:
        json.dump({"render_prep_complete": False}, f, indent=2, ensure_ascii=False)

    try:
        r = get("/api/config")
        assert_status(r, 200)
        data = r.json()
        if data.get("current_file") is not None:
            raise TestFailure(f"Expected current_file to be cleared, got {data.get('current_file')!r}")

        with open(config_path, "r", encoding="utf-8") as f:
            persisted = json.load(f)
        if persisted.get("current_file") is not None:
            raise TestFailure("GET /api/config did not persist clearing stale current_file")
    finally:
        with open(config_path, "w", encoding="utf-8") as f:
            f.write(original_config_raw)
        if original_state_raw is None:
            if os.path.exists(state_path):
                os.remove(state_path)
        else:
            with open(state_path, "w", encoding="utf-8") as f:
                f.write(original_state_raw)

def test_save_config_roundtrip():
    # Read original
    r = get("/api/config")
    assert_status(r, 200)
    original = r.json()
    shared["original_config"] = original

    # Build test config with modified language
    test_config = {
        "llm": original["llm"],
        "tts": {**original.get("tts", {}), "language": "_test_roundtrip_lang"},
        "prompts": original.get("prompts"),
        "generation": original.get("generation"),
        "proofread": original.get("proofread"),
        "export": original.get("export"),
    }
    test_config["tts"].setdefault("mode", "external")
    test_config["tts"].setdefault("url", "http://127.0.0.1:7860")
    test_config["tts"].setdefault("device", "auto")

    # Save modified
    r = post("/api/config", json=test_config)
    assert_status(r, 200)

    # Read back and verify
    r = get("/api/config")
    assert_status(r, 200)
    readback = r.json()
    if readback.get("tts", {}).get("language") != "_test_roundtrip_lang":
        raise TestFailure("Config round-trip failed: language not persisted")

    # Verify generation section persists
    if original.get("generation") and not readback.get("generation"):
        raise TestFailure("Config round-trip failed: generation section dropped")

    # Verify export normalization settings persist
    if not readback.get("export"):
        raise TestFailure("Config round-trip failed: export section dropped")
    export = readback["export"]
    for key in (
        "trim_clip_silence_enabled",
        "trim_silence_threshold_dbfs",
        "trim_min_silence_len_ms",
        "trim_keep_padding_ms",
        "normalize_enabled",
        "normalize_target_lufs_mono",
        "normalize_target_lufs_stereo",
        "normalize_true_peak_dbtp",
        "normalize_lra",
    ):
        if key not in export:
            raise TestFailure(f"Config round-trip failed: export.{key} missing")

    # Verify review prompts persist through config save
    readback_prompts = readback.get("prompts", {})
    if original.get("prompts", {}).get("review_system_prompt"):
        if not readback_prompts.get("review_system_prompt"):
            raise TestFailure("Config round-trip failed: review_system_prompt dropped")
    if original.get("prompts", {}).get("attribution_system_prompt"):
        if not readback_prompts.get("attribution_system_prompt"):
            raise TestFailure("Config round-trip failed: attribution_system_prompt dropped")
    if original.get("prompts", {}).get("voice_prompt"):
        if not readback_prompts.get("voice_prompt"):
            raise TestFailure("Config round-trip failed: voice_prompt dropped")

    # Restore original
    restore = {
        "llm": original["llm"],
        "tts": original.get("tts", {"mode": "external", "url": "http://127.0.0.1:7860", "device": "auto"}),
        "prompts": original.get("prompts"),
        "generation": original.get("generation"),
        "proofread": original.get("proofread"),
        "export": original.get("export"),
    }
    post("/api/config", json=restore)

def test_save_export_config_roundtrip():
    r = get("/api/config")
    assert_status(r, 200)
    original = r.json()
    export_original = original.get("export") or {}

    patch_payload = {
        "silence_between_speakers_ms": int(export_original.get("silence_between_speakers_ms", 500)),
        "silence_same_speaker_ms": int(export_original.get("silence_same_speaker_ms", 250)),
        "silence_end_of_chapter_ms": int(export_original.get("silence_end_of_chapter_ms", 3000)),
        "silence_paragraph_ms": int(export_original.get("silence_paragraph_ms", 750)),
        "trim_clip_silence_enabled": False,
        "trim_silence_threshold_dbfs": float(export_original.get("trim_silence_threshold_dbfs", -50.0)),
        "trim_min_silence_len_ms": int(export_original.get("trim_min_silence_len_ms", 150)),
        "trim_keep_padding_ms": int(export_original.get("trim_keep_padding_ms", 40)),
        "normalize_enabled": bool(export_original.get("normalize_enabled", True)),
        "normalize_target_lufs_mono": float(export_original.get("normalize_target_lufs_mono", -18.0)),
        "normalize_target_lufs_stereo": float(export_original.get("normalize_target_lufs_stereo", -16.0)),
        "normalize_true_peak_dbtp": float(export_original.get("normalize_true_peak_dbtp", -1.0)),
        "normalize_lra": float(export_original.get("normalize_lra", 11.0)),
    }

    r = post("/api/config/export", json=patch_payload)
    assert_status(r, 200)
    body = r.json()
    if body.get("status") != "saved":
        raise TestFailure("Export config patch did not report success")

    r = get("/api/config")
    assert_status(r, 200)
    readback = r.json()
    if readback.get("export", {}).get("trim_clip_silence_enabled") is not False:
        raise TestFailure("Export config patch did not persist trim_clip_silence_enabled=false")
    if readback.get("export", {}).get("trim_keep_padding_ms") != patch_payload["trim_keep_padding_ms"]:
        raise TestFailure("Export config patch did not persist trim_keep_padding_ms")

    zero_padding_payload = dict(patch_payload)
    zero_padding_payload["trim_keep_padding_ms"] = 0
    r = post("/api/config/export", json=zero_padding_payload)
    assert_status(r, 200)

    r = get("/api/config")
    assert_status(r, 200)
    zero_readback = r.json()
    if zero_readback.get("export", {}).get("trim_keep_padding_ms") != 0:
        raise TestFailure("Export config patch did not preserve trim_keep_padding_ms=0")

    post("/api/config", json=original)

def test_save_review_prompts_roundtrip():
    # Read current config
    r = get("/api/config")
    assert_status(r, 200)
    original = r.json()

    # Save config with custom review prompts
    test_config = {
        "llm": original["llm"],
        "tts": original.get("tts", {"mode": "local", "url": "http://127.0.0.1:7860", "device": "auto"}),
        "prompts": {
            **(original.get("prompts") or {}),
            "review_system_prompt": f"{TEST_PREFIX}review_sys",
            "review_user_prompt": f"{TEST_PREFIX}review_usr",
        },
        "generation": original.get("generation"),
    }
    r = post("/api/config", json=test_config)
    assert_status(r, 200)

    # Read back and verify
    r = get("/api/config")
    assert_status(r, 200)
    readback = r.json()
    prompts = readback.get("prompts", {})
    if prompts.get("review_system_prompt") != f"{TEST_PREFIX}review_sys":
        raise TestFailure(f"review_system_prompt not persisted: {prompts.get('review_system_prompt')}")
    if prompts.get("review_user_prompt") != f"{TEST_PREFIX}review_usr":
        raise TestFailure(f"review_user_prompt not persisted: {prompts.get('review_user_prompt')}")

    # Restore original
    restore = {
        "llm": original["llm"],
        "tts": original.get("tts", {"mode": "local", "url": "http://127.0.0.1:7860", "device": "auto"}),
        "prompts": original.get("prompts"),
        "generation": original.get("generation"),
    }
    post("/api/config", json=restore)

def test_save_attribution_prompts_roundtrip():
    r = get("/api/config")
    assert_status(r, 200)
    original = r.json()

    test_config = {
        "llm": original["llm"],
        "tts": original.get("tts", {"mode": "local", "url": "http://127.0.0.1:7860", "device": "auto"}),
        "prompts": {
            **(original.get("prompts") or {}),
            "attribution_system_prompt": f"{TEST_PREFIX}attr_sys",
            "attribution_user_prompt": f"{TEST_PREFIX}attr_usr",
        },
        "generation": original.get("generation"),
    }
    r = post("/api/config", json=test_config)
    assert_status(r, 200)

    r = get("/api/config")
    assert_status(r, 200)
    readback = r.json()
    prompts = readback.get("prompts", {})
    if prompts.get("attribution_system_prompt") != f"{TEST_PREFIX}attr_sys":
        raise TestFailure(f"attribution_system_prompt not persisted: {prompts.get('attribution_system_prompt')}")
    if prompts.get("attribution_user_prompt") != f"{TEST_PREFIX}attr_usr":
        raise TestFailure(f"attribution_user_prompt not persisted: {prompts.get('attribution_user_prompt')}")

    restore = {
        "llm": original["llm"],
        "tts": original.get("tts", {"mode": "local", "url": "http://127.0.0.1:7860", "device": "auto"}),
        "prompts": original.get("prompts"),
        "generation": original.get("generation"),
    }
    post("/api/config", json=restore)

def test_get_default_prompts():
    r = get("/api/default_prompts")
    assert_status(r, 200)
    data = r.json()
    assert_key(data, "system_prompt")
    assert_key(data, "user_prompt")
    if not data["system_prompt"]:
        raise TestFailure("system_prompt is empty")
    assert_key(data, "review_system_prompt")
    assert_key(data, "review_user_prompt")
    assert_key(data, "attribution_system_prompt")
    assert_key(data, "attribution_user_prompt")
    assert_key(data, "voice_prompt")
    if not data["review_system_prompt"]:
        raise TestFailure("review_system_prompt is empty")
    if not data["review_user_prompt"]:
        raise TestFailure("review_user_prompt is empty")
    if not data["attribution_system_prompt"]:
        raise TestFailure("attribution_system_prompt is empty")
    if not data["attribution_user_prompt"]:
        raise TestFailure("attribution_user_prompt is empty")
    if not data["voice_prompt"]:
        raise TestFailure("voice_prompt is empty")

def test_get_config_persists_missing_voice_prompt_default():
    config_path = os.path.join(common.ACTIVE_APP_DIR, "config.json")
    with open(config_path, "r", encoding="utf-8") as f:
        original_raw = f.read()
    original = json.loads(original_raw)

    modified = json.loads(original_raw)
    prompts = modified.setdefault("prompts", {})
    prompts.pop("voice_prompt", None)
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(modified, f, indent=2, ensure_ascii=False)

    try:
        r = get("/api/config")
        assert_status(r, 200)
        data = r.json()
        if not data.get("prompts", {}).get("voice_prompt"):
            raise TestFailure("GET /api/config did not return voice_prompt")

        with open(config_path, "r", encoding="utf-8") as f:
            persisted = json.load(f)
        if not persisted.get("prompts", {}).get("voice_prompt"):
            raise TestFailure("GET /api/config did not persist backfilled voice_prompt")
    finally:
        with open(config_path, "w", encoding="utf-8") as f:
            f.write(original_raw)

def test_get_config_persists_missing_temperament_words_default():
    config_path = os.path.join(common.ACTIVE_APP_DIR, "config.json")
    with open(config_path, "r", encoding="utf-8") as f:
        original_raw = f.read()

    modified = json.loads(original_raw)
    generation = modified.setdefault("generation", {})
    generation.pop("temperament_words", None)
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(modified, f, indent=2, ensure_ascii=False)

    try:
        r = get("/api/config")
        assert_status(r, 200)
        data = r.json()
        if data.get("generation", {}).get("temperament_words") != 150:
            raise TestFailure("GET /api/config did not return generation.temperament_words=150")

        with open(config_path, "r", encoding="utf-8") as f:
            persisted = json.load(f)
        if persisted.get("generation", {}).get("temperament_words") != 150:
            raise TestFailure("GET /api/config did not persist backfilled generation.temperament_words")
    finally:
        with open(config_path, "w", encoding="utf-8") as f:
            f.write(original_raw)

def test_get_config_persists_missing_script_error_retry_attempts_default():
    config_path = os.path.join(common.ACTIVE_APP_DIR, "config.json")
    with open(config_path, "r", encoding="utf-8") as f:
        original_raw = f.read()

    modified = json.loads(original_raw)
    generation = modified.setdefault("generation", {})
    generation.pop("script_error_retry_attempts", None)
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(modified, f, indent=2, ensure_ascii=False)

    try:
        r = get("/api/config")
        assert_status(r, 200)
        data = r.json()
        if data.get("generation", {}).get("script_error_retry_attempts") != 3:
            raise TestFailure("GET /api/config did not return generation.script_error_retry_attempts=3")

        with open(config_path, "r", encoding="utf-8") as f:
            persisted = json.load(f)
        if persisted.get("generation", {}).get("script_error_retry_attempts") != 3:
            raise TestFailure("GET /api/config did not persist backfilled generation.script_error_retry_attempts")
    finally:
        with open(config_path, "w", encoding="utf-8") as f:
            f.write(original_raw)

def test_get_config_persists_missing_llm_and_tts_defaults():
    config_path = os.path.join(common.ACTIVE_APP_DIR, "config.json")
    with open(config_path, "r", encoding="utf-8") as f:
        original_raw = f.read()

    modified = json.loads(original_raw)
    modified.pop("llm", None)
    tts = modified.setdefault("tts", {})
    tts.pop("language", None)
    tts.pop("parallel_workers", None)
    tts.pop("auto_regenerate_bad_clips", None)
    tts.pop("auto_regenerate_bad_clip_attempts", None)
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(modified, f, indent=2, ensure_ascii=False)

    try:
        r = get("/api/config")
        assert_status(r, 200)
        data = r.json()
        if data.get("llm", {}).get("base_url") != "":
            raise TestFailure("GET /api/config did not backfill llm.base_url")
        if data.get("llm", {}).get("api_key") != "":
            raise TestFailure("GET /api/config did not backfill llm.api_key")
        if data.get("llm", {}).get("model_name") != "":
            raise TestFailure("GET /api/config did not backfill llm.model_name")
        if data.get("tts", {}).get("language") != "English":
            raise TestFailure("GET /api/config did not backfill tts.language")
        if int((data.get("tts") or {}).get("parallel_workers") or 0) != 4:
            raise TestFailure("GET /api/config did not backfill tts.parallel_workers=4")
        if (data.get("tts") or {}).get("auto_regenerate_bad_clips") is not True:
            raise TestFailure("GET /api/config did not backfill tts.auto_regenerate_bad_clips=true")

        with open(config_path, "r", encoding="utf-8") as f:
            persisted = json.load(f)
        if persisted.get("llm", {}).get("base_url") != "":
            raise TestFailure("GET /api/config did not persist llm defaults")
        if persisted.get("llm", {}).get("api_key") != "":
            raise TestFailure("GET /api/config did not persist llm.api_key default")
        if persisted.get("tts", {}).get("language") != "English":
            raise TestFailure("GET /api/config did not persist tts.language default")
        if int((persisted.get("tts") or {}).get("parallel_workers") or 0) != 4:
            raise TestFailure("GET /api/config did not persist tts.parallel_workers=4")
        if (persisted.get("tts") or {}).get("auto_regenerate_bad_clips") is not True:
            raise TestFailure("GET /api/config did not persist tts.auto_regenerate_bad_clips=true")
    finally:
        with open(config_path, "w", encoding="utf-8") as f:
            f.write(original_raw)

def test_save_setup_config_preserves_hidden_local_backend():
    r = get("/api/config")
    assert_status(r, 200)
    original = r.json()

    seeded = {
        "llm": original["llm"],
        "tts": {**(original.get("tts") or {}), "mode": "local", "local_backend": "qwen"},
        "prompts": original.get("prompts"),
        "generation": original.get("generation"),
        "proofread": original.get("proofread"),
        "export": original.get("export"),
        "ui": original.get("ui"),
    }
    r = post("/api/config", json=seeded)
    assert_status(r, 200)

    payload = {
        "tts": {
            "mode": "local",
            "url": seeded["tts"].get("url", "http://127.0.0.1:7860"),
            "language": seeded["tts"].get("language", "English"),
            "parallel_workers": 1,
        }
    }
    r = post("/api/config/setup", json=payload)
    assert_status(r, 200)

    try:
        r = get("/api/config")
        assert_status(r, 200)
        readback = r.json()
        if readback.get("tts", {}).get("local_backend") != "qwen":
            raise TestFailure("POST /api/config/setup overwrote hidden tts.local_backend")
    finally:
        post("/api/config", json=original)

def test_save_setup_config_roundtrip_tts_provider():
    r = get("/api/config")
    assert_status(r, 200)
    original = r.json()

    payload = {
        "tts": {
            "provider": "voxcpm2",
            "mode": original.get("tts", {}).get("mode", "local"),
            "url": original.get("tts", {}).get("url", "http://127.0.0.1:7860"),
            "language": original.get("tts", {}).get("language", "English"),
            "parallel_workers": original.get("tts", {}).get("parallel_workers", 4),
        }
    }
    r = post("/api/config/setup", json=payload)
    assert_status(r, 200)

    try:
        r = get("/api/config")
        assert_status(r, 200)
        readback = r.json()
        if readback.get("tts", {}).get("provider") != "voxcpm2":
            raise TestFailure("POST /api/config/setup did not persist tts.provider")
    finally:
        post("/api/config", json=original)

def test_save_setup_config_roundtrip_visible_voxcpm2_settings():
    r = get("/api/config")
    assert_status(r, 200)
    original = r.json()

    payload = {
        "tts": {
            "provider": "voxcpm2",
            "mode": "local",
            "url": original.get("tts", {}).get("url", "http://127.0.0.1:7860"),
            "language": original.get("tts", {}).get("language", "English"),
            "parallel_workers": 1,
            "voxcpm_model_id": "openbmb/VoxCPM2",
            "voxcpm_cfg_value": 1.35,
            "voxcpm_inference_timesteps": 14,
            "voxcpm_normalize": True,
            "voxcpm_denoise": True,
            "voxcpm_load_denoiser": True,
            "voxcpm_denoise_reference": True,
            "voxcpm_optimize": False,
        }
    }
    r = post("/api/config/setup", json=payload)
    assert_status(r, 200)

    try:
        r = get("/api/config")
        assert_status(r, 200)
        tts = r.json().get("tts", {})
        for key, expected in payload["tts"].items():
            actual = tts.get(key)
            if actual != expected:
                raise TestFailure(f"POST /api/config/setup did not persist tts.{key}: expected {expected!r}, got {actual!r}")
    finally:
        post("/api/config", json=original)

def test_save_setup_config_clamps_voxcpm2_generation_settings_on_mac():
    r = get("/api/config")
    assert_status(r, 200)
    original = r.json()

    payload = {
        "tts": {
            "provider": "voxcpm2",
            "mode": "local",
            "url": original.get("tts", {}).get("url", "http://127.0.0.1:7860"),
            "language": original.get("tts", {}).get("language", "English"),
            "parallel_workers": 1,
            "voxcpm_cfg_value": 9.0,
            "voxcpm_inference_timesteps": 100,
            "voxcpm_optimize": True,
        }
    }
    r = post("/api/config/setup", json=payload)
    assert_status(r, 200)

    try:
        r = get("/api/config")
        assert_status(r, 200)
        tts = r.json().get("tts", {})
        if float(tts.get("voxcpm_cfg_value") or 0) != 3.0:
            raise TestFailure("POST /api/config/setup did not clamp tts.voxcpm_cfg_value to 3.0")
        if int(tts.get("voxcpm_inference_timesteps") or 0) != 30:
            raise TestFailure("POST /api/config/setup did not clamp tts.voxcpm_inference_timesteps to 30")
        if sys.platform == "darwin" and tts.get("voxcpm_optimize") is not False:
            raise TestFailure("POST /api/config/setup did not disable tts.voxcpm_optimize on macOS")
    finally:
        post("/api/config", json=original)

def test_get_config_uses_voxcpm2_script_max_length_default_when_missing():
    config_path = os.path.join(common.ACTIVE_APP_DIR, "config.json")

    with open(config_path, "r", encoding="utf-8") as f:
        original_config_raw = f.read()

    modified = json.loads(original_config_raw)
    modified.setdefault("tts", {})
    modified["tts"]["provider"] = "voxcpm2"
    modified["tts"]["script_max_length"] = None
    modified["tts"]["voxcpm_cfg_value"] = None
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(modified, f, indent=2, ensure_ascii=False)

    try:
        r = get("/api/config")
        assert_status(r, 200)
        tts = r.json().get("tts", {})
        if int(tts.get("script_max_length") or 0) != 240:
            raise TestFailure(f"GET /api/config did not backfill VoxCPM2 script_max_length=240, got {tts.get('script_max_length')!r}")
        if float(tts.get("voxcpm_cfg_value") or 0) != 1.6:
            raise TestFailure("GET /api/config did not backfill VoxCPM2 cfg default to 1.6")
    finally:
        with open(config_path, "w", encoding="utf-8") as f:
            f.write(original_config_raw)

def test_save_setup_config_preserves_hidden_voxcpm2_settings():
    r = get("/api/config")
    assert_status(r, 200)
    original = r.json()

    seeded = {
        "llm": original.get("llm"),
        "tts": {
            **(original.get("tts") or {}),
            "provider": "voxcpm2",
            "voxcpm_cfg_value": 1.6,
            "voxcpm_inference_timesteps": 18,
        },
        "prompts": original.get("prompts"),
        "generation": original.get("generation"),
        "proofread": original.get("proofread"),
        "export": original.get("export"),
        "ui": original.get("ui"),
    }
    r = post("/api/config", json=seeded)
    assert_status(r, 200)

    payload = {
        "tts": {
            "provider": "voxcpm2",
            "mode": "local",
            "url": seeded["tts"].get("url", "http://127.0.0.1:7860"),
            "language": seeded["tts"].get("language", "English"),
            "parallel_workers": 1,
        }
    }
    r = post("/api/config/setup", json=payload)
    assert_status(r, 200)

    try:
        r = get("/api/config")
        assert_status(r, 200)
        readback = r.json()
        tts = readback.get("tts", {})
        if float(tts.get("voxcpm_cfg_value") or 0) != 1.6:
            raise TestFailure("POST /api/config/setup overwrote hidden tts.voxcpm_cfg_value")
        if int(tts.get("voxcpm_inference_timesteps") or 0) != 18:
            raise TestFailure("POST /api/config/setup overwrote hidden tts.voxcpm_inference_timesteps")
    finally:
        post("/api/config", json=original)

def test_save_setup_config_roundtrip_temperament_words():
    r = get("/api/config")
    assert_status(r, 200)
    original = r.json()

    payload = {
        "generation": {
            "temperament_words": 222,
        }
    }
    r = post("/api/config/setup", json=payload)
    assert_status(r, 200)

    try:
        r = get("/api/config")
        assert_status(r, 200)
        readback = r.json()
        if readback.get("generation", {}).get("temperament_words") != 222:
            raise TestFailure("POST /api/config/setup did not persist generation.temperament_words")
    finally:
        post("/api/config", json=original)

def test_save_setup_config_roundtrip_script_error_retry_attempts():
    r = get("/api/config")
    assert_status(r, 200)
    original = r.json()

    payload = {
        "generation": {
            "script_error_retry_attempts": 7,
        }
    }
    r = post("/api/config/setup", json=payload)
    assert_status(r, 200)

    try:
        r = get("/api/config")
        assert_status(r, 200)
        readback = r.json()
        if readback.get("generation", {}).get("script_error_retry_attempts") != 7:
            raise TestFailure("POST /api/config/setup did not persist generation.script_error_retry_attempts")
    finally:
        post("/api/config", json=original)


# ── Section 3: Upload ───────────────────────────────────────
