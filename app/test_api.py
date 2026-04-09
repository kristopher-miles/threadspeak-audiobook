#!/usr/bin/env python3
"""Automated API test script.

Usage:
    python test_api.py                    # Quick tests only
    python test_api.py --full             # Include TTS/LLM-dependent tests
    python test_api.py --url http://host:port
"""

import argparse
import io
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from urllib.parse import urlparse
import requests
try:
    import pytest
except Exception:  # pragma: no cover - pytest import only needed when running under pytest
    pytest = None

# ── Global state ─────────────────────────────────────────────

def _normalize_http_url(raw_url):
    url = (raw_url or "").strip()
    if not url:
        return ""
    if "://" not in url:
        url = f"http://{url}"
    return url.rstrip("/")


def _discover_base_url():
    # Explicit test override wins.
    for key in ("THREADSPEAK_TEST_URL", "BASE_URL"):
        configured = _normalize_http_url(os.getenv(key))
        if configured:
            return configured

    # Pinokio can expose the launched URL through the variable named in PINOKIO_SHARE_VAR.
    # Example: PINOKIO_SHARE_VAR=url and env[url]=http://127.0.0.1:42003
    share_var_key = (os.getenv("PINOKIO_SHARE_VAR") or "").strip()
    if share_var_key:
        shared = _normalize_http_url(os.getenv(share_var_key) or os.getenv(share_var_key.upper()))
        if shared:
            return shared

    # Pinokio local share port is a reliable fallback when no direct URL var is present.
    share_port = (os.getenv("PINOKIO_SHARE_LOCAL_PORT") or "").strip()
    if share_port.isdigit():
        return f"http://127.0.0.1:{share_port}"

    # Legacy fallback.
    return "http://127.0.0.1:4200"


BASE_URL = _discover_base_url()
FULL_MODE = (os.getenv("THREADSPEAK_TEST_FULL", "").strip().lower() in {"1", "true", "yes", "on"})
TEST_PREFIX = "_test_"
APP_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(APP_DIR)
STATE_PATH = os.path.join(REPO_DIR, "state.json")
UPLOADS_PATH = os.path.join(REPO_DIR, "uploads")
ACTIVE_APP_DIR = APP_DIR

results = {"passed": 0, "failed": 0, "skipped": 0}
failures = []
shared = {}  # state shared between dependent tests
_SERVER_PROC = None
_SERVER_TEMP_ROOT = None


def _find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _start_isolated_test_server():
    global BASE_URL, REPO_DIR, STATE_PATH, UPLOADS_PATH, ACTIVE_APP_DIR, _SERVER_PROC, _SERVER_TEMP_ROOT

    _SERVER_TEMP_ROOT = tempfile.mkdtemp(prefix="threadspeak_api_test_")
    temp_app_dir = os.path.join(_SERVER_TEMP_ROOT, "app")
    shutil.copytree(
        APP_DIR,
        temp_app_dir,
        ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache", "*.pyc", "env"),
    )
    temp_root = os.path.dirname(temp_app_dir)
    for filename in (
        "default_prompts.txt",
        "review_prompts.txt",
        "attribution_prompts.txt",
        "voice_prompt.txt",
        "dialogue_identification_system_prompt.txt",
        "temperament_extraction_system_prompt.txt",
    ):
        source = os.path.join(REPO_DIR, filename)
        if os.path.exists(source):
            shutil.copy2(source, os.path.join(temp_root, filename))

    port = _find_free_port()
    env = os.environ.copy()
    env["PINOKIO_SHARE_LOCAL"] = "false"
    env["PINOKIO_SHARE_LOCAL_PORT"] = str(port)
    env["PYTHONUNBUFFERED"] = "1"

    _SERVER_PROC = subprocess.Popen(
        [sys.executable, "app.py"],
        cwd=temp_app_dir,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    base_url = f"http://127.0.0.1:{port}"
    deadline = time.time() + 45
    while time.time() < deadline:
        if _SERVER_PROC.poll() is not None:
            output = ""
            if _SERVER_PROC.stdout:
                try:
                    output = _SERVER_PROC.stdout.read() or ""
                except Exception:
                    output = ""
            raise RuntimeError(f"Isolated test server exited early with code {_SERVER_PROC.returncode}.\n{output[-2000:]}")
        try:
            r = requests.get(f"{base_url}/", timeout=1.5)
            if r.status_code < 500:
                break
        except Exception:
            pass
        time.sleep(0.3)
    else:
        raise RuntimeError(f"Timed out waiting for isolated test server at {base_url}")

    BASE_URL = base_url
    ACTIVE_APP_DIR = temp_app_dir
    REPO_DIR = os.path.dirname(temp_app_dir)
    STATE_PATH = os.path.join(REPO_DIR, "state.json")
    UPLOADS_PATH = os.path.join(REPO_DIR, "uploads")


def _stop_isolated_test_server():
    global _SERVER_PROC, _SERVER_TEMP_ROOT

    proc = _SERVER_PROC
    _SERVER_PROC = None
    if proc is not None:
        try:
            proc.terminate()
            proc.wait(timeout=10)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass
    if _SERVER_TEMP_ROOT and os.path.isdir(_SERVER_TEMP_ROOT):
        shutil.rmtree(_SERVER_TEMP_ROOT, ignore_errors=True)
    _SERVER_TEMP_ROOT = None


if pytest is not None:
    @pytest.fixture(scope="module", autouse=True)
    def _isolated_api_server():
        use_external = (os.getenv("THREADSPEAK_TEST_USE_EXTERNAL_SERVER", "").strip().lower() in {"1", "true", "yes", "on"})
        if not use_external:
            _start_isolated_test_server()
        try:
            yield
        finally:
            try:
                cleanup()
            except Exception:
                pass
            if not use_external:
                _stop_isolated_test_server()


# ── Helpers ──────────────────────────────────────────────────

class TestFailure(Exception):
    pass


def skip_test(reason):
    if pytest is not None and "PYTEST_CURRENT_TEST" in os.environ:
        pytest.skip(reason)
    raise TestFailure(f"SKIP: {reason}")


def require_full_mode():
    if not FULL_MODE:
        skip_test("requires full suite")


def section(title):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def run_test(name, func, requires_full=False):
    if requires_full and not FULL_MODE:
        print(f"  [ SKIP ] {name} (requires --full)")
        results["skipped"] += 1
        return
    try:
        func()
        print(f"  [ PASS ] {name}")
        results["passed"] += 1
    except TestFailure as e:
        msg = str(e)
        if msg.startswith("SKIP:"):
            print(f"  [ SKIP ] {name} ({msg[5:].strip()})")
            results["skipped"] += 1
        else:
            print(f"  [ FAIL ] {name}")
            print(f"           {msg}")
            results["failed"] += 1
            failures.append((name, msg))
    except Exception as e:
        print(f"  [ FAIL ] {name}")
        print(f"           {type(e).__name__}: {e}")
        results["failed"] += 1
        failures.append((name, str(e)))


def assert_status(resp, expected=200, msg=""):
    if resp.status_code != expected:
        body = resp.text[:500]
        raise TestFailure(
            f"Expected {expected}, got {resp.status_code}. {msg}\n"
            f"           Body: {body}"
        )


def assert_key(data, key):
    if key not in data:
        raise TestFailure(f"Missing key '{key}' in: {json.dumps(data)[:300]}")


def wait_for_task(task, timeout=120, poll_interval=2):
    """Poll /api/status/{task} until it stops running or timeout is reached."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        r = requests.get(f"{BASE_URL}/api/status/{task}", timeout=10)
        if r.status_code == 200 and not r.json().get("running"):
            return True
        time.sleep(poll_interval)
    return False


def wait_for_audio_idle(timeout=120, poll_interval=2):
    """Wait until audio has no running work and no queued jobs."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = get("/api/status/audio")
            if r.status_code == 200:
                data = r.json()
                if (
                    not data.get("running")
                    and not data.get("merge_running")
                    and not data.get("current_job")
                    and not data.get("queue")
                ):
                    return True
        except Exception:
            pass
        time.sleep(poll_interval)
    return False


def get(path, **kwargs):
    return requests.get(f"{BASE_URL}{path}", timeout=30, **kwargs)


def post(path, **kwargs):
    return requests.post(f"{BASE_URL}{path}", timeout=kwargs.pop("timeout", 30), **kwargs)


def delete(path, **kwargs):
    return requests.delete(f"{BASE_URL}{path}", timeout=30, **kwargs)


def _is_local_server():
    try:
        host = (urlparse(BASE_URL).hostname or "").lower()
    except Exception:
        return False
    return host in {"127.0.0.1", "localhost"}


def _cleanup_local_upload_state():
    if not _is_local_server():
        return []
    removed = []
    if os.path.isdir(UPLOADS_PATH):
        for name in os.listdir(UPLOADS_PATH):
            if not name.startswith(TEST_PREFIX):
                continue
            path = os.path.join(UPLOADS_PATH, name)
            if os.path.isfile(path):
                try:
                    os.remove(path)
                    removed.append(f"upload {name}")
                except Exception:
                    pass
    try:
        if os.path.exists(STATE_PATH):
            with open(STATE_PATH, "r", encoding="utf-8") as f:
                state = json.load(f)
            input_path = (state.get("input_file_path") or "").strip()
            if input_path and os.path.basename(input_path).startswith(TEST_PREFIX):
                state.pop("input_file_path", None)
                state["render_prep_complete"] = False
                state.pop("processing_stage_markers", None)
                with open(STATE_PATH, "w", encoding="utf-8") as f:
                    json.dump(state, f, indent=2, ensure_ascii=False)
                removed.append("state input_file_path")
    except Exception:
        pass
    return removed


# ── Section 1: Server ───────────────────────────────────────

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
    config_path = os.path.join(ACTIVE_APP_DIR, "config.json")
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


# ── Section 3: Upload ───────────────────────────────────────

def test_upload_file():
    require_full_mode()
    content = b"Chapter One\nIt was a dark and stormy night.\nThe end."
    files = {"file": (f"{TEST_PREFIX}upload.txt", io.BytesIO(content), "text/plain")}
    r = post("/api/upload", files=files)
    assert_status(r, 200)
    data = r.json()
    assert_key(data, "filename")
    assert_key(data, "path")
    if data["filename"] != f"{TEST_PREFIX}upload.txt":
        raise TestFailure(f"Unexpected filename: {data['filename']}")


# ── Section 4: Annotated Script ─────────────────────────────

def test_get_annotated_script():
    r = get("/api/annotated_script")
    if r.status_code == 404:
        shared["has_script"] = False
        return  # acceptable — no script loaded
    assert_status(r, 200)
    data = r.json()
    if not isinstance(data, list):
        raise TestFailure(f"Expected list, got {type(data).__name__}")
    shared["has_script"] = True


# ── Section 5: Scripts CRUD ─────────────────────────────────

def test_save_script():
    if not shared.get("has_script"):
        skip_test("no annotated script loaded")
    r = post("/api/scripts/save", json={"name": f"{TEST_PREFIX}script"})
    assert_status(r, 200)
    data = r.json()
    if data.get("status") != "saved":
        raise TestFailure(f"Expected status=saved, got {data}")


def test_list_scripts():
    r = get("/api/scripts")
    assert_status(r, 200)
    data = r.json()
    if not isinstance(data, list):
        raise TestFailure(f"Expected list, got {type(data).__name__}")
    if shared.get("has_script"):
        names = [s["name"] for s in data]
        if f"{TEST_PREFIX}script" not in names:
            raise TestFailure(f"Saved script not in list: {names}")


def test_load_script():
    if not shared.get("has_script"):
        skip_test("no annotated script loaded")
    r = post("/api/scripts/load", json={"name": f"{TEST_PREFIX}script"})
    if r.status_code == 409:
        # Script load is blocked by both active and queued audio work.
        # Cancel any stale queue/current job and wait for full idle before retrying.
        post("/api/cancel_audio", json={})
        if wait_for_audio_idle(timeout=120):
            r = post("/api/scripts/load", json={"name": f"{TEST_PREFIX}script"})
    assert_status(r, 200)
    data = r.json()
    if data.get("status") != "loaded":
        raise TestFailure(f"Expected status=loaded, got {data}")


def test_delete_script():
    if not shared.get("has_script"):
        skip_test("no annotated script loaded")
    r = delete(f"/api/scripts/{TEST_PREFIX}script")
    assert_status(r, 200)
    data = r.json()
    if data.get("status") != "deleted":
        raise TestFailure(f"Expected status=deleted, got {data}")


def test_delete_script_404():
    r = delete(f"/api/scripts/{TEST_PREFIX}nonexistent_xyz")
    assert_status(r, 404)


# ── Section 6: Voices ───────────────────────────────────────

def test_get_voices():
    r = get("/api/voices")
    assert_status(r, 200)
    data = r.json()
    if not isinstance(data, list):
        raise TestFailure(f"Expected list, got {type(data).__name__}")


def test_save_voice_config():
    r = post("/api/save_voice_config", json={
        f"{TEST_PREFIX}voice": {
            "type": "custom",
            "voice": "Ryan",
            "character_style": "",
            "alias": f"{TEST_PREFIX}alias",
            "seed": "-1"
        }
    })
    assert_status(r, 200)
    data = r.json()
    if data.get("status") != "saved":
        raise TestFailure(f"Expected status=saved, got {data}")


# ── Section 7: Chunks ───────────────────────────────────────

def test_get_chunks():
    r = get("/api/chunks")
    assert_status(r, 200)
    data = r.json()
    if not isinstance(data, list):
        raise TestFailure(f"Expected list, got {type(data).__name__}")
    shared["has_chunks"] = len(data) > 0
    if data:
        shared["chunk0_original"] = {
            "text": data[0].get("text", ""),
            "instruct": data[0].get("instruct", ""),
            "speaker": data[0].get("speaker", ""),
        }
        shared["chunk0_chapter"] = (data[0].get("chapter") or "").strip()


def test_get_chunks_view():
    r = get("/api/chunks/view")
    assert_status(r, 200)
    data = r.json()
    if not isinstance(data, list):
        raise TestFailure(f"Expected list, got {type(data).__name__}")

    if shared.get("has_chunks"):
        if not data:
            raise TestFailure("Expected visible chunk list to be populated")
        chapter = shared.get("chunk0_chapter")
        if chapter:
            r = get("/api/chunks/view", params={"chapter": chapter})
            assert_status(r, 200)
            scoped = r.json()
            if not isinstance(scoped, list):
                raise TestFailure(f"Expected list, got {type(scoped).__name__}")
            if any((chunk.get("chapter") or "").strip() != chapter for chunk in scoped):
                raise TestFailure(f"Chapter-scoped view returned chunks outside {chapter!r}")


def test_update_chunk():
    if not shared.get("has_chunks"):
        skip_test("no chunks available")

    r = post("/api/chunks/0", json={
        "text": f"{TEST_PREFIX}updated_text",
        "instruct": f"{TEST_PREFIX}instruct"
    })
    assert_status(r, 200)
    data = r.json()
    if data.get("text") != f"{TEST_PREFIX}updated_text":
        raise TestFailure(f"Chunk text not updated: {data.get('text')}")
    if data.get("audio_path") is not None:
        raise TestFailure(f"Chunk audio was not invalidated: {data.get('audio_path')}")
    if data.get("status") != "pending":
        raise TestFailure(f"Chunk status was not reset: {data.get('status')}")

    # Restore original
    orig = shared.get("chunk0_original", {})
    post("/api/chunks/0", json=orig)


def test_update_chunk_404():
    r = post("/api/chunks/99999", json={"text": "nope"})
    assert_status(r, 404)


def test_insert_chunk():
    if not shared.get("has_chunks"):
        skip_test("no chunks available")

    # Get initial count
    r = get("/api/chunks")
    assert_status(r, 200)
    initial_chunks = r.json()
    initial_count = len(initial_chunks)

    # Insert after index 0
    r = post("/api/chunks/0/insert")
    assert_status(r, 200)
    data = r.json()
    if data.get("status") != "ok":
        raise TestFailure(f"Expected status=ok, got {data}")
    if data.get("total") != initial_count + 1:
        raise TestFailure(f"Expected total={initial_count + 1}, got {data.get('total')}")

    # Verify the new chunk exists at index 1 with empty text
    r = get("/api/chunks")
    assert_status(r, 200)
    chunks = r.json()
    if len(chunks) != initial_count + 1:
        raise TestFailure(f"Chunk count mismatch: expected {initial_count + 1}, got {len(chunks)}")
    if chunks[1].get("text") != "":
        raise TestFailure(f"Inserted chunk should have empty text, got: {chunks[1].get('text')}")

    # Store index for cleanup in delete test
    shared["inserted_chunk_index"] = 1


def test_insert_chunk_404():
    r = post("/api/chunks/99999/insert")
    assert_status(r, 404)


def test_delete_chunk():
    if not shared.get("has_chunks"):
        skip_test("no chunks available")

    idx = shared.get("inserted_chunk_index")
    if idx is None:
        skip_test("no inserted chunk to delete")

    # Get count before delete
    r = get("/api/chunks")
    assert_status(r, 200)
    before_count = len(r.json())

    r = delete(f"/api/chunks/{idx}")
    assert_status(r, 200)
    data = r.json()
    assert_key(data, "deleted")
    assert_key(data, "total")
    if data["total"] != before_count - 1:
        raise TestFailure(f"Expected total={before_count - 1}, got {data['total']}")

    # Save deleted chunk for restore test
    shared["deleted_chunk"] = data["deleted"]
    shared["deleted_chunk_index"] = idx


def test_delete_chunk_invalid():
    r = delete("/api/chunks/99999")
    assert_status(r, 400)


def test_restore_chunk():
    if not shared.get("deleted_chunk"):
        skip_test("no deleted chunk to restore")

    r = get("/api/chunks")
    assert_status(r, 200)
    before_count = len(r.json())

    r = post("/api/chunks/restore", json={
        "chunk": shared["deleted_chunk"],
        "at_index": shared["deleted_chunk_index"]
    })
    assert_status(r, 200)
    data = r.json()
    if data.get("status") != "ok":
        raise TestFailure(f"Expected status=ok, got {data}")
    if data.get("total") != before_count + 1:
        raise TestFailure(f"Expected total={before_count + 1}, got {data.get('total')}")

    # Clean up: delete the restored chunk so we leave chunks as we found them
    delete(f"/api/chunks/{shared['deleted_chunk_index']}")


# ── Section 8: Status Polling ────────────────────────────────

def test_status_known_tasks():
    task_names = [
        "script", "voices", "audio", "audacity_export",
        "review", "lora_training", "dataset_gen", "dataset_builder"
    ]
    for name in task_names:
        r = get(f"/api/status/{name}")
        assert_status(r, 200, msg=f"task={name}")
        data = r.json()
        if "running" not in data:
            raise TestFailure(f"Missing 'running' key for task '{name}'")
        if "logs" not in data:
            raise TestFailure(f"Missing 'logs' key for task '{name}'")


def test_status_unknown_task():
    r = get(f"/api/status/{TEST_PREFIX}fake_task")
    assert_status(r, 404)


# ── Section 9: Voice Design ─────────────────────────────────

def test_voice_design_list():
    r = get("/api/voice_design/list")
    assert_status(r, 200)
    data = r.json()
    if not isinstance(data, list):
        raise TestFailure(f"Expected list, got {type(data).__name__}")


def test_voice_design_delete_404():
    r = delete(f"/api/voice_design/{TEST_PREFIX}fake_id")
    assert_status(r, 404)


def test_voice_design_preview():
    require_full_mode()
    r = post("/api/voice_design/preview", json={
        "description": "A clear young male voice with a steady tone",
        "sample_text": "This is a test of voice design.",
    })
    assert_status(r, 200)
    data = r.json()
    assert_key(data, "audio_url")
    shared["preview_file"] = data["audio_url"].split("/")[-1]


def test_voice_design_save_and_delete():
    require_full_mode()
    preview_file = shared.get("preview_file")
    if not preview_file:
        raise TestFailure("SKIP: no preview file from previous test")

    r = post("/api/voice_design/save", json={
        "name": f"{TEST_PREFIX}voice_design",
        "description": "Test voice",
        "sample_text": "Test text",
        "preview_file": preview_file
    })
    assert_status(r, 200)
    data = r.json()
    assert_key(data, "voice_id")
    voice_id = data["voice_id"]

    # Delete it
    r = delete(f"/api/voice_design/{voice_id}")
    assert_status(r, 200)


# ── Section 9b: Clone Voices ────────────────────────────────

def test_clone_voices_list():
    r = get("/api/clone_voices/list")
    assert_status(r, 200)
    data = r.json()
    if not isinstance(data, list):
        raise TestFailure(f"Expected list, got {type(data).__name__}")


def test_clone_voices_upload_bad_format():
    files = {"file": ("test.txt", b"not audio", "text/plain")}
    r = requests.post(f"{BASE_URL}/api/clone_voices/upload", files=files)
    assert_status(r, 400)


def test_clone_voices_delete_404():
    r = delete(f"/api/clone_voices/{TEST_PREFIX}fake_id")
    assert_status(r, 404)


def test_clone_voices_upload_and_delete():
    # Create a minimal WAV file (44-byte header + silence)
    import struct
    sample_rate = 16000
    num_samples = 16000  # 1 second
    data_size = num_samples * 2
    wav_header = struct.pack('<4sI4s4sIHHIIHH4sI',
        b'RIFF', 36 + data_size, b'WAVE',
        b'fmt ', 16, 1, 1, sample_rate, sample_rate * 2, 2, 16,
        b'data', data_size)
    wav_bytes = wav_header + b'\x00' * data_size

    files = {"file": (f"{TEST_PREFIX}clone_test.wav", wav_bytes, "audio/wav")}
    r = requests.post(f"{BASE_URL}/api/clone_voices/upload", files=files)
    assert_status(r, 200)
    data = r.json()
    assert_key(data, "voice_id")
    assert_key(data, "filename")
    voice_id = data["voice_id"]

    # Verify it appears in list
    r = get("/api/clone_voices/list")
    assert_status(r, 200)
    found = any(v["id"] == voice_id for v in r.json())
    if not found:
        raise TestFailure(f"Uploaded voice {voice_id} not found in list")

    # Delete it
    r = delete(f"/api/clone_voices/{voice_id}")
    assert_status(r, 200)

    # Verify it's gone
    r = get("/api/clone_voices/list")
    found = any(v["id"] == voice_id for v in r.json())
    if found:
        raise TestFailure(f"Deleted voice {voice_id} still in list")


# ── Section 10: LoRA Datasets ───────────────────────────────

def test_lora_list_datasets():
    r = get("/api/lora/datasets")
    assert_status(r, 200)
    data = r.json()
    if not isinstance(data, list):
        raise TestFailure(f"Expected list, got {type(data).__name__}")


def test_lora_delete_dataset_404():
    r = delete(f"/api/lora/datasets/{TEST_PREFIX}fake_ds")
    assert_status(r, 404)


def test_lora_upload_bad_file():
    files = {"file": (f"{TEST_PREFIX}bad.txt", io.BytesIO(b"not a zip"), "text/plain")}
    r = post("/api/lora/upload_dataset", files=files)
    # Should fail — not a valid zip
    if r.status_code < 400:
        raise TestFailure(f"Expected error for non-zip upload, got {r.status_code}")


# ── Section 11: LoRA Models ─────────────────────────────────

def test_lora_list_models():
    r = get("/api/lora/models")
    assert_status(r, 200)
    data = r.json()
    if not isinstance(data, list):
        raise TestFailure(f"Expected list, got {type(data).__name__}")
    # Verify built-in adapters have 'downloaded' field
    for m in data:
        if m.get("builtin"):
            if "downloaded" not in m:
                raise TestFailure(f"Built-in adapter {m['id']} missing 'downloaded' field")
    shared["lora_models"] = data


def test_lora_download_invalid():
    r = post(f"/api/lora/download/{TEST_PREFIX}fake_adapter", json={})
    if r.status_code < 400:
        raise TestFailure(f"Expected error for invalid adapter, got {r.status_code}")


def test_lora_delete_model_404():
    r = delete(f"/api/lora/models/{TEST_PREFIX}fake_model")
    assert_status(r, 404)


def test_lora_train_bad_dataset():
    r = post("/api/lora/train", json={
        "name": f"{TEST_PREFIX}model",
        "dataset_id": f"{TEST_PREFIX}nonexistent_ds"
    })
    # Should fail — dataset does not exist
    if r.status_code < 400:
        raise TestFailure(f"Expected error for bad dataset, got {r.status_code}")


def test_lora_preview_404():
    r = post(f"/api/lora/preview/{TEST_PREFIX}fake_adapter")
    assert_status(r, 404)


def test_lora_preview():
    require_full_mode()
    models = shared.get("lora_models", [])
    if not models:
        raise TestFailure("SKIP: no LoRA models available")
    adapter = models[0]
    r = post(f"/api/lora/preview/{adapter['id']}", timeout=120)
    assert_status(r, 200)
    data = r.json()
    assert_key(data, "audio_url")


# ── Section 12: Dataset Builder CRUD ────────────────────────

def test_dataset_builder_list():
    r = get("/api/dataset_builder/list")
    assert_status(r, 200)
    data = r.json()
    if not isinstance(data, list):
        raise TestFailure(f"Expected list, got {type(data).__name__}")


def test_dataset_builder_create():
    r = post("/api/dataset_builder/create", json={
        "name": f"{TEST_PREFIX}builder_proj"
    })
    assert_status(r, 200)
    data = r.json()
    assert_key(data, "name")


def test_dataset_builder_update_meta():
    r = post("/api/dataset_builder/update_meta", json={
        "name": f"{TEST_PREFIX}builder_proj",
        "description": "A test voice description",
        "global_seed": "42"
    })
    assert_status(r, 200)


def test_dataset_builder_update_rows():
    r = post("/api/dataset_builder/update_rows", json={
        "name": f"{TEST_PREFIX}builder_proj",
        "rows": [
            {"emotion": "neutral", "text": "Hello world.", "seed": ""},
            {"emotion": "happy", "text": "Great to see you!", "seed": ""}
        ]
    })
    assert_status(r, 200)
    data = r.json()
    if data.get("sample_count") != 2:
        raise TestFailure(f"Expected sample_count=2, got {data.get('sample_count')}")


def test_dataset_builder_status():
    r = get(f"/api/dataset_builder/status/{TEST_PREFIX}builder_proj")
    assert_status(r, 200)
    data = r.json()
    assert_key(data, "description")
    assert_key(data, "samples")
    assert_key(data, "running")
    assert_key(data, "logs")
    if len(data["samples"]) != 2:
        raise TestFailure(f"Expected 2 samples, got {len(data['samples'])}")


def test_dataset_builder_cancel():
    r = post("/api/dataset_builder/cancel")
    assert_status(r, 200)
    data = r.json()
    if data.get("status") not in ("not_running", "cancelling"):
        raise TestFailure(f"Unexpected cancel status: {data}")


def test_dataset_builder_save_no_samples():
    r = post("/api/dataset_builder/save", json={
        "name": f"{TEST_PREFIX}builder_proj",
        "ref_index": 0
    })
    # Should fail — no completed samples
    if r.status_code < 400:
        raise TestFailure(f"Expected error for save with no samples, got {r.status_code}")


def test_dataset_builder_delete():
    r = delete(f"/api/dataset_builder/{TEST_PREFIX}builder_proj")
    assert_status(r, 200)
    data = r.json()
    if data.get("status") != "deleted":
        raise TestFailure(f"Expected status=deleted, got {data}")


def test_dataset_builder_delete_404():
    r = delete(f"/api/dataset_builder/{TEST_PREFIX}nonexistent")
    assert_status(r, 404)


# ── Section 13: Merge / Export ──────────────────────────────

def test_get_audiobook():
    r = get("/api/audiobook")
    if r.status_code == 404:
        return  # acceptable — no audiobook generated yet
    assert_status(r, 200)


def test_get_audacity_export():
    r = get("/api/export_audacity")
    if r.status_code == 404:
        return  # acceptable — no export generated yet
    assert_status(r, 200)


# ── Section 14: Full Tests — Generation ─────────────────────

def test_generate_script():
    require_full_mode()
    r = post("/api/generate_script")
    if r.status_code == 400:
        raise TestFailure("SKIP: prerequisite not met (no uploaded file or already running)")
    assert_status(r, 200)
    data = r.json()
    if data.get("status") != "started":
        raise TestFailure(f"Expected status=started, got {data}")


def test_review_script():
    require_full_mode()
    if not shared.get("has_script"):
        raise TestFailure("SKIP: no annotated script loaded")
    r = post("/api/review_script")
    if r.status_code == 400:
        raise TestFailure("SKIP: already running")
    assert_status(r, 200)
    data = r.json()
    if data.get("status") != "started":
        raise TestFailure(f"Expected status=started, got {data}")


def test_parse_voices():
    require_full_mode()
    r = post("/api/parse_voices")
    if r.status_code == 400:
        raise TestFailure("SKIP: already running")
    assert_status(r, 200)
    data = r.json()
    if data.get("status") != "started":
        raise TestFailure(f"Expected status=started, got {data}")


def test_generate_chunk():
    require_full_mode()
    if not shared.get("has_chunks"):
        raise TestFailure("SKIP: no chunks available")
    r = post("/api/chunks/0/generate")
    assert_status(r, 200)


def test_generate_batch():
    require_full_mode()
    if not shared.get("has_chunks"):
        skip_test("no chunks available")
    r = post("/api/generate_batch", json={"indices": [0]})
    if r.status_code == 400:
        skip_test("audio generation already running")
    assert_status(r, 200)
    data = r.json()
    if data.get("status") not in ("started", "queued"):
        raise TestFailure(f"Expected status=started|queued, got {data}")
    # Wait for batch to finish so subsequent tests don't conflict
    if not wait_for_task("audio", timeout=120):
        raise TestFailure("generate_batch did not complete within 120s")


def test_generate_batch_fast():
    require_full_mode()
    if not shared.get("has_chunks"):
        skip_test("no chunks available")
    # Wait for any prior generation to finish
    if not wait_for_task("audio", timeout=120):
        skip_test("prior audio generation did not finish in time")
    r = post("/api/generate_batch_fast", json={"indices": [0]})
    if r.status_code == 400:
        skip_test("audio generation already running")
    assert_status(r, 200)
    data = r.json()
    if data.get("status") not in ("started", "queued"):
        raise TestFailure(f"Expected status=started|queued, got {data}")


def test_cancel_audio():
    """Cancel endpoint works when nothing is running (resets stuck chunks)."""
    r = post("/api/cancel_audio", json={})
    assert_status(r, 200)
    data = r.json()
    if data.get("status") not in ("not_running", "cancelling"):
        raise TestFailure(f"Expected status not_running or cancelling, got {data}")


def test_export_audacity():
    require_full_mode()
    r = post("/api/export_audacity")
    if r.status_code == 400:
        raise TestFailure("SKIP: already running")
    assert_status(r, 200)
    data = r.json()
    if data.get("status") != "started":
        raise TestFailure(f"Expected status=started, got {data}")


def test_lora_test_model():
    require_full_mode()
    models = shared.get("lora_models", [])
    if not models:
        raise TestFailure("SKIP: no LoRA models available")
    adapter = models[0]
    r = post("/api/lora/test", json={
        "adapter_id": adapter["id"],
        "text": "This is a test of the LoRA voice.",
        "instruct": "Neutral, even delivery."
    }, timeout=120)
    assert_status(r, 200)
    data = r.json()
    assert_key(data, "audio_url")


def test_lora_generate_dataset():
    require_full_mode()
    r = post("/api/lora/generate_dataset", json={
        "name": f"{TEST_PREFIX}dataset",
        "description": "A clear young male voice",
        "samples": [
            {"emotion": "neutral", "text": "Hello, this is a test sample."},
            {"emotion": "happy", "text": "Great to see you today!"}
        ]
    })
    if r.status_code == 400:
        raise TestFailure("SKIP: already running or bad request")
    assert_status(r, 200)
    data = r.json()
    if data.get("status") != "started":
        raise TestFailure(f"Expected status=started, got {data}")


def test_dataset_builder_generate_sample():
    require_full_mode()
    # Create a temp project for this test
    post("/api/dataset_builder/create", json={"name": f"{TEST_PREFIX}gen_proj"})
    post("/api/dataset_builder/update_rows", json={
        "name": f"{TEST_PREFIX}gen_proj",
        "rows": [{"emotion": "neutral", "text": "Hello world.", "seed": ""}]
    })

    r = post("/api/dataset_builder/generate_sample", json={
        "description": "A clear male voice",
        "text": "Hello world.",
        "dataset_name": f"{TEST_PREFIX}gen_proj",
        "sample_index": 0,
        "seed": -1
    })
    assert_status(r, 200)
    data = r.json()
    assert_key(data, "status")

    # Cleanup
    delete(f"/api/dataset_builder/{TEST_PREFIX}gen_proj")


# ── Run all tests ────────────────────────────────────────────

def run_all_tests():
    section("Server")
    run_test("server_reachable", test_server_reachable)

    section("Config")
    run_test("get_config", test_get_config)
    run_test("save_config_roundtrip", test_save_config_roundtrip)
    run_test("save_export_config_roundtrip", test_save_export_config_roundtrip)
    run_test("save_review_prompts_roundtrip", test_save_review_prompts_roundtrip)
    run_test("save_attribution_prompts_roundtrip", test_save_attribution_prompts_roundtrip)
    run_test("get_default_prompts", test_get_default_prompts)
    run_test("get_config_persists_missing_voice_prompt_default", test_get_config_persists_missing_voice_prompt_default)

    section("Upload")
    run_test("upload_file", test_upload_file, requires_full=True)

    section("Annotated Script")
    run_test("get_annotated_script", test_get_annotated_script)

    section("Scripts CRUD")
    run_test("save_script", test_save_script)
    run_test("list_scripts", test_list_scripts)
    run_test("load_script", test_load_script)
    run_test("delete_script", test_delete_script)
    run_test("delete_script_404", test_delete_script_404)

    section("Voices")
    run_test("get_voices", test_get_voices)
    run_test("save_voice_config", test_save_voice_config)

    section("Chunks")
    run_test("get_chunks", test_get_chunks)
    run_test("update_chunk", test_update_chunk)
    run_test("update_chunk_404", test_update_chunk_404)
    run_test("insert_chunk", test_insert_chunk)
    run_test("insert_chunk_404", test_insert_chunk_404)
    run_test("delete_chunk", test_delete_chunk)
    run_test("delete_chunk_invalid", test_delete_chunk_invalid)
    run_test("restore_chunk", test_restore_chunk)

    section("Status Polling")
    run_test("status_known_tasks", test_status_known_tasks)
    run_test("status_unknown_task", test_status_unknown_task)

    section("Voice Design")
    run_test("voice_design_list", test_voice_design_list)
    run_test("voice_design_delete_404", test_voice_design_delete_404)
    run_test("voice_design_preview", test_voice_design_preview, requires_full=True)
    run_test("voice_design_save_and_delete", test_voice_design_save_and_delete, requires_full=True)

    section("Clone Voices")
    run_test("clone_voices_list", test_clone_voices_list)
    run_test("clone_voices_upload_bad_format", test_clone_voices_upload_bad_format)
    run_test("clone_voices_delete_404", test_clone_voices_delete_404)
    run_test("clone_voices_upload_and_delete", test_clone_voices_upload_and_delete)

    section("LoRA Datasets")
    run_test("lora_list_datasets", test_lora_list_datasets)
    run_test("lora_delete_dataset_404", test_lora_delete_dataset_404)
    run_test("lora_upload_bad_file", test_lora_upload_bad_file)

    section("LoRA Models")
    run_test("lora_list_models", test_lora_list_models)
    run_test("lora_download_invalid", test_lora_download_invalid)
    run_test("lora_delete_model_404", test_lora_delete_model_404)
    run_test("lora_train_bad_dataset", test_lora_train_bad_dataset)
    run_test("lora_preview_404", test_lora_preview_404)
    run_test("lora_preview", test_lora_preview, requires_full=True)

    section("Dataset Builder")
    run_test("dataset_builder_list", test_dataset_builder_list)
    run_test("dataset_builder_create", test_dataset_builder_create)
    run_test("dataset_builder_update_meta", test_dataset_builder_update_meta)
    run_test("dataset_builder_update_rows", test_dataset_builder_update_rows)
    run_test("dataset_builder_status", test_dataset_builder_status)
    run_test("dataset_builder_cancel", test_dataset_builder_cancel)
    run_test("dataset_builder_save_no_samples", test_dataset_builder_save_no_samples)
    run_test("dataset_builder_delete", test_dataset_builder_delete)
    run_test("dataset_builder_delete_404", test_dataset_builder_delete_404)

    section("Merge / Export")
    run_test("get_audiobook", test_get_audiobook)
    run_test("get_audacity_export", test_get_audacity_export)

    section("Generation (TTS/LLM)")
    run_test("generate_script", test_generate_script, requires_full=True)
    run_test("review_script", test_review_script, requires_full=True)
    run_test("parse_voices", test_parse_voices, requires_full=True)
    run_test("generate_chunk", test_generate_chunk, requires_full=True)
    run_test("generate_batch", test_generate_batch, requires_full=True)
    run_test("generate_batch_fast", test_generate_batch_fast, requires_full=True)
    run_test("cancel_audio", test_cancel_audio)
    run_test("export_audacity", test_export_audacity, requires_full=True)

    section("LoRA (TTS)")
    run_test("lora_test_model", test_lora_test_model, requires_full=True)
    run_test("lora_generate_dataset", test_lora_generate_dataset, requires_full=True)

    section("Dataset Builder Generate (TTS)")
    run_test("dataset_builder_generate_sample", test_dataset_builder_generate_sample, requires_full=True)


# ── Cleanup ──────────────────────────────────────────────────

def cleanup():
    print(f"\n--- Cleanup ---")
    items = []

    try:
        post("/api/cancel_audio", json={})
    except Exception:
        pass

    try:
        post("/api/dataset_builder/cancel")
    except Exception:
        pass

    try:
        delete(f"/api/scripts/{TEST_PREFIX}script")
        items.append("test script")
    except Exception:
        pass

    try:
        delete(f"/api/dataset_builder/{TEST_PREFIX}builder_proj")
        items.append("builder project")
    except Exception:
        pass

    try:
        delete(f"/api/dataset_builder/{TEST_PREFIX}gen_proj")
        items.append("gen project")
    except Exception:
        pass

    try:
        r = get("/api/dataset_builder/list")
        if r.status_code == 200:
            for entry in r.json():
                name = entry.get("name", "")
                if name.startswith(TEST_PREFIX):
                    delete(f"/api/dataset_builder/{name}")
                    items.append(f"builder {name}")
    except Exception:
        pass

    try:
        delete(f"/api/lora/datasets/{TEST_PREFIX}dataset")
        items.append("test dataset")
    except Exception:
        pass

    try:
        r = get("/api/voice_design/list")
        if r.status_code == 200:
            for v in r.json():
                if v.get("id", "").startswith(TEST_PREFIX):
                    delete(f"/api/voice_design/{v['id']}")
                    items.append(f"voice {v['id']}")
    except Exception:
        pass

    for removed in _cleanup_local_upload_state():
        items.append(removed)

    if items:
        print(f"  Cleaned: {', '.join(items)}")
    else:
        print(f"  Nothing to clean")


if pytest is not None:
    @pytest.fixture(scope="session", autouse=True)
    def _test_api_session_cleanup():
        cleanup()
        yield
        cleanup()


# ── Main ─────────────────────────────────────────────────────

def main():
    global BASE_URL, FULL_MODE

    parser = argparse.ArgumentParser(description="Threadspeak API test suite")
    parser.add_argument("--url", default=BASE_URL,
                        help=f"Server URL (default: {BASE_URL})")
    parser.add_argument("--full", action="store_true",
                        help="Include TTS/LLM-dependent tests")
    args = parser.parse_args()

    BASE_URL = _normalize_http_url(args.url)
    FULL_MODE = args.full

    print(f"Threadspeak API Tests")
    print(f"Server: {BASE_URL}")
    print(f"Mode:   {'FULL (includes TTS/LLM tests)' if FULL_MODE else 'QUICK (no TTS/LLM)'}")

    cleanup()

    try:
        run_all_tests()
    finally:
        cleanup()

    # Summary
    total = results["passed"] + results["failed"] + results["skipped"]
    print(f"\n{'=' * 60}")
    print(f"  RESULTS: {results['passed']} passed, {results['failed']} failed, "
          f"{results['skipped']} skipped  (total: {total})")
    print(f"{'=' * 60}")

    if failures:
        print(f"\nFailed tests:")
        for name, err in failures:
            # Truncate long error messages
            short = err.split("\n")[0][:200]
            print(f"  - {name}: {short}")

    sys.exit(1 if results["failed"] > 0 else 0)


if __name__ == "__main__":
    main()
