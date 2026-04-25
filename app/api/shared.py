import os
import sys
import gc
import errno
import copy
import json
import shutil
import logging
import asyncio
import atexit
import socket
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator, model_validator
from typing import List, Optional, Dict, Union, Literal
import re
import time
import threading
import zipfile
import tempfile
import subprocess
import aiofiles
import uuid
import urllib.request
import urllib.error
import sqlite3
from llm import (
    LLMClientFactory,
    LLMRuntimeConfig,
    LMStudioModelLoadService,
    ToolCapabilityService,
)
from chunk_events import chunk_event_broker
from audio_perf import record_audio_perf
from config_bootstrap import ensure_runtime_config_exists

# Import ProjectManager
from project import ProjectManager
from asr import LocalASRUnavailableError
from default_prompts import (
    DEFAULT_SYSTEM_PROMPT,
    DEFAULT_USER_PROMPT,
    load_default_prompts,
    DEFAULT_DIALOGUE_IDENTIFICATION_PROMPT,
    DEFAULT_TEMPERAMENT_EXTRACTION_PROMPT,
    load_dialogue_identification_prompt,
    load_temperament_extraction_prompt,
)
from review_prompts import load_review_prompts
from attribution_prompts import load_attribution_prompts
from voice_prompt import load_voice_prompt
from hf_utils import fetch_builtin_manifest, download_builtin_adapter, is_adapter_downloaded
from script_store import apply_dictionary_to_text, clean_dictionary_entries
from source_document import load_source_document
from script_sanity import build_attribution_classifier, run_script_sanity_check
from script_repair import RepairSupersededError, repair_invalid_chunks
from runtime_layout import LAYOUT, REPO_ROOT

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ThreadspeakUI")
_LLM_CLIENT_FACTORY = LLMClientFactory()
_LMSTUDIO_UNLOAD_TIMEOUT_SECONDS = 15
QWEN3_SCRIPT_MAX_LENGTH_DEFAULT = 250
VOXCPM2_SCRIPT_MAX_LENGTH_DEFAULT = 240
VOXCPM2_CFG_VALUE_DEFAULT = 1.6
VOXCPM2_CFG_VALUE_MIN = 1.0
VOXCPM2_CFG_VALUE_MAX = 3.0
VOXCPM2_INFERENCE_TIMESTEPS_DEFAULT = 10
VOXCPM2_INFERENCE_TIMESTEPS_MIN = 4
VOXCPM2_INFERENCE_TIMESTEPS_MAX = 30

app = FastAPI(title="Threadspeak Audiobook")

# Paths
BASE_DIR = LAYOUT.app_dir
REPO_DIR = REPO_ROOT
ROOT_DIR = LAYOUT.project_dir
CONFIG_PATH = os.path.join(BASE_DIR, "config.json")
CONFIG_DEFAULT_PATH = os.path.join(BASE_DIR, "config.default.json")
ensure_runtime_config_exists(CONFIG_PATH, CONFIG_DEFAULT_PATH)
AUDIOBOOK_PATH = LAYOUT.audiobook_path
M4B_PATH = LAYOUT.m4b_path
OPTIMIZED_EXPORT_PATH = LAYOUT.optimized_export_path
UPLOADS_DIR = LAYOUT.uploads_dir
SCRIPTS_DIR = LAYOUT.script_snapshots_dir
SAVED_PROJECTS_DIR = LAYOUT.project_archives_dir
AUDIO_QUEUE_STATE_PATH = LAYOUT.audio_queue_state_path
AUDIO_CANCEL_TOMBSTONE_PATH = LAYOUT.audio_cancel_tombstone_path
SCRIPT_REPAIR_TRACE_PATH = LAYOUT.script_repair_trace_path
DESIGNED_VOICES_DIR = LAYOUT.designed_voices_dir
CLONE_VOICES_DIR = LAYOUT.clone_voices_dir
LORA_MODELS_DIR = LAYOUT.lora_models_dir
LORA_DATASETS_DIR = LAYOUT.lora_datasets_dir
BUILTIN_LORA_DIR = LAYOUT.builtin_lora_dir
BUILTIN_LORA_MANIFEST = os.path.join(BUILTIN_LORA_DIR, "manifest.json")
DATASET_BUILDER_DIR = LAYOUT.dataset_builder_dir
EMOTIONS_DIR = os.path.join(LAYOUT.workspace_dir, "emotions")
EMOTIONS_AUDIO_DIR = os.path.join(EMOTIONS_DIR, "audio")
DESIGNED_VOICES_MANIFEST = os.path.join(DESIGNED_VOICES_DIR, "manifest.json")
CLONE_VOICES_MANIFEST = os.path.join(CLONE_VOICES_DIR, "manifest.json")
ALLOWED_AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg"}
PROJECT_ARCHIVE_VERSION = 6
PROJECT_ARCHIVE_MANIFEST_NAME = "project_archive_manifest.json"
PROJECT_ARCHIVE_DURABLE_FILES = {
    "db/chunks.sqlite3",
    "state.json",
    "exports/cloned_audiobook.mp3",
    "exports/optimized_audiobook.zip",
    "exports/audacity_export.zip",
    "exports/audiobook.m4b",
    "exports/m4b_cover.jpg",
}
PROJECT_ARCHIVE_LEGACY_OPTIONAL_FILES = {
    "workflow/processing_workflow_state.json",
    "workflow/new_mode_workflow_state.json",
    "workflow/audio_queue_state.json",
    "workflow/audio_cancel_tombstone.json",
    "workflow/script_generation_checkpoint.json",
    "workflow/script_review_checkpoint.json",
    "repair/script_repair_trace.jsonl",
}
PROJECT_ARCHIVE_ALLOWED_FILES = PROJECT_ARCHIVE_DURABLE_FILES | PROJECT_ARCHIVE_LEGACY_OPTIONAL_FILES
PROJECT_ARCHIVE_LEGACY_FILE_ALIASES = {}
PROJECT_ARCHIVE_ALLOWED_DIRS = {
    "uploads",
    "clone_voices",
    "designed_voices",
    "voicelines",
}

LAYOUT.ensure_base_dirs()


def tts_script_max_length_default(provider: str | None) -> int:
    return VOXCPM2_SCRIPT_MAX_LENGTH_DEFAULT if str(provider or "").strip().lower() == "voxcpm2" else QWEN3_SCRIPT_MAX_LENGTH_DEFAULT


def clamp_voxcpm_cfg_value(value) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = VOXCPM2_CFG_VALUE_DEFAULT
    return min(VOXCPM2_CFG_VALUE_MAX, max(VOXCPM2_CFG_VALUE_MIN, parsed))


def clamp_voxcpm_inference_timesteps(value) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = VOXCPM2_INFERENCE_TIMESTEPS_DEFAULT
    return min(VOXCPM2_INFERENCE_TIMESTEPS_MAX, max(VOXCPM2_INFERENCE_TIMESTEPS_MIN, parsed))

_media_static_server_lock = threading.Lock()
_media_static_server_process = None
_media_static_server_origin = None
_media_static_server_shutdown_registered = False
_MEDIA_STATIC_READY_TIMEOUT_SECONDS = 3.0
_MEDIA_STATIC_READY_POLL_SECONDS = 0.05


def _find_free_local_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _shutdown_media_static_server():
    global _media_static_server_process, _media_static_server_origin
    proc = _media_static_server_process
    _media_static_server_process = None
    _media_static_server_origin = None
    if proc is None:
        return
    try:
        proc.terminate()
        proc.wait(timeout=2)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass


def _probe_media_static_origin(origin, timeout=0.25):
    health_url = f"{origin}/__health"
    try:
        with urllib.request.urlopen(health_url, timeout=timeout) as response:
            return 200 <= int(getattr(response, "status", 0) or 0) < 300
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError, ValueError):
        return False


def _media_static_server_command(port):
    return [
        sys.executable,
        "-m",
        "uvicorn",
        "media_static_server:app",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--log-level",
        "warning",
        "--no-access-log",
        "--app-dir",
        BASE_DIR,
    ]


def _new_runtime_run_id(prefix: str = "runtime") -> str:
    label = str(prefix or "runtime").strip() or "runtime"
    return f"{label}-{uuid.uuid4().hex[:8]}"


def _make_runtime_temp_dir(prefix: str, *, run_id: Optional[str] = None) -> str:
    effective_run_id = str(run_id or _new_runtime_run_id(prefix)).strip()
    return LAYOUT.make_named_temp_dir(effective_run_id, prefix)


def _make_runtime_temp_file(prefix: str, suffix: str = "", *, run_id: Optional[str] = None, subdir: str = "tmp") -> str:
    effective_run_id = str(run_id or _new_runtime_run_id(prefix)).strip()
    directory = LAYOUT.run_subdir(effective_run_id, subdir)
    handle = tempfile.NamedTemporaryFile(prefix=prefix, suffix=suffix, dir=directory, delete=False)
    path = handle.name
    handle.close()
    return path


def _running_under_test_process():
    if "PYTEST_CURRENT_TEST" in os.environ:
        return True
    argv0 = os.path.basename(str(sys.argv[0] or "")).strip().lower()
    return "pytest" in argv0


def _layout_repo_is_ephemeral_temp_clone() -> bool:
    repo_root = os.path.realpath(os.path.abspath(str(getattr(LAYOUT, "repo_root", "") or "")))
    if not repo_root:
        return False
    temp_root = os.path.realpath(os.path.abspath(tempfile.gettempdir()))
    try:
        return os.path.commonpath([repo_root, temp_root]) == temp_root
    except ValueError:
        return False


def _assert_test_safe_runtime_target(operation: str, **paths):
    """Refuse destructive test operations that still point at the live runtime project."""
    if not _running_under_test_process():
        return
    if _layout_repo_is_ephemeral_temp_clone():
        return

    protected = {
        "ROOT_DIR": os.path.abspath(LAYOUT.project_dir),
        "VOICELINES_DIR": os.path.abspath(LAYOUT.voicelines_dir),
        "UPLOADS_DIR": os.path.abspath(LAYOUT.uploads_dir),
        "SCRIPTS_DIR": os.path.abspath(LAYOUT.script_snapshots_dir),
        "SAVED_PROJECTS_DIR": os.path.abspath(LAYOUT.project_archives_dir),
        "CLONE_VOICES_DIR": os.path.abspath(LAYOUT.clone_voices_dir),
        "DESIGNED_VOICES_DIR": os.path.abspath(LAYOUT.designed_voices_dir),
    }
    violations = []
    for label, path in paths.items():
        normalized = os.path.abspath(str(path or ""))
        protected_path = protected.get(label)
        if protected_path and normalized == protected_path:
            violations.append(f"{label}={normalized}")
    if violations:
        joined = ", ".join(violations)
        raise RuntimeError(
            f"{operation} refused to target the default runtime project during tests: {joined}"
        )


def _project_workflow_state_path(filename: str) -> str:
    if os.path.abspath(ROOT_DIR) == os.path.abspath(LAYOUT.project_dir):
        return os.path.join(LAYOUT.workflow_dir, filename)
    return os.path.join(ROOT_DIR, "workflow", filename)


def _project_repair_trace_path() -> str:
    if os.path.abspath(ROOT_DIR) == os.path.abspath(LAYOUT.project_dir):
        return SCRIPT_REPAIR_TRACE_PATH
    return os.path.join(ROOT_DIR, "repair", "script_repair_trace.jsonl")


def _get_media_static_origin():
    global _media_static_server_process, _media_static_server_origin, _media_static_server_shutdown_registered
    configured = (os.getenv("THREADSPEAK_MEDIA_STATIC_ORIGIN") or "").strip().rstrip("/")
    if configured:
        return configured

    with _media_static_server_lock:
        proc = _media_static_server_process
        if proc is not None and proc.poll() is None and _media_static_server_origin:
            return _media_static_server_origin

        port = _find_free_local_port()
        command = _media_static_server_command(port)
        try:
            proc = subprocess.Popen(
                command,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                cwd=BASE_DIR,
            )
        except Exception as exc:
            logger.warning("Failed to start media static server: %s", exc)
            _media_static_server_process = None
            _media_static_server_origin = None
            return None

        _media_static_server_process = proc
        _media_static_server_origin = f"http://127.0.0.1:{port}"
        if not _media_static_server_shutdown_registered:
            atexit.register(_shutdown_media_static_server)
            _media_static_server_shutdown_registered = True

        deadline = time.time() + _MEDIA_STATIC_READY_TIMEOUT_SECONDS
        while time.time() < deadline:
            if proc.poll() is not None:
                break
            if _probe_media_static_origin(_media_static_server_origin):
                return _media_static_server_origin
            time.sleep(_MEDIA_STATIC_READY_POLL_SECONDS)

        logger.warning("Media static server failed readiness probe: %s", _media_static_server_origin)
        _shutdown_media_static_server()
        return None


class _VoicelinesProxyStatic:
    def __init__(self, directory):
        self._fallback = StaticFiles(directory=directory)

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            origin = _get_media_static_origin()
            if origin:
                path = scope.get("path") or ""
                query_string = (scope.get("query_string") or b"").decode("utf-8", errors="ignore")
                target = f"{origin}{path}"
                if query_string:
                    target = f"{target}?{query_string}"
                response = RedirectResponse(target, status_code=307)
                await response(scope, receive, send)
                return
        await self._fallback(scope, receive, send)


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


def _sanitize_name(name: str) -> str:
    """Make a string safe for use as a filename."""
    name = re.sub(r'[^\w\- ]', "", name).strip()
    name = re.sub(r"\s+", "_", name)
    return name.lower()


def _normalize_saved_voice_name(name: str) -> str:
    return project_manager._normalize_speaker_name(name)


def _current_loaded_project_name() -> str:
    state_path = os.path.join(ROOT_DIR, "state.json")
    if not os.path.exists(state_path):
        return ""
    try:
        with open(state_path, "r", encoding="utf-8") as f:
            state = json.load(f)
    except (json.JSONDecodeError, ValueError, OSError):
        return ""
    return str(state.get("loaded_project_name") or "").strip()


def _match_project_prefixed_saved_voice_name(candidate: str, project_name: str, speaker_name: str) -> bool:
    raw_candidate = str(candidate or "").strip()
    raw_project = str(project_name or "").strip()
    raw_speaker = str(speaker_name or "").strip()
    if not raw_candidate or not raw_project or not raw_speaker:
        return False
    normalized_project = _normalize_saved_voice_name(raw_project)
    normalized_speaker = _normalize_saved_voice_name(raw_speaker)
    if not normalized_project or not normalized_speaker:
        return False
    normalized_candidate = _normalize_saved_voice_name(raw_candidate)
    for separator in (".", "_", " "):
        prefix = f"{normalized_project}{separator}"
        if normalized_candidate.startswith(prefix):
            suffix = normalized_candidate[len(prefix):]
            return suffix == normalized_speaker
    return False


def _find_saved_voice_option_for_speaker(speaker: str):
    normalized_speaker = _normalize_saved_voice_name(speaker)
    if not normalized_speaker or normalized_speaker == _normalize_saved_voice_name("NARRATOR"):
        return None
    current_script_title_fn = getattr(project_manager, "_current_script_title", None)
    current_script_title = _normalize_saved_voice_name(
        current_script_title_fn() if callable(current_script_title_fn) else "Project"
    )
    current_project_name = _normalize_saved_voice_name(_current_loaded_project_name())

    def _build_rel_audio(directory_name: str, entry: dict) -> str:
        filename = (entry.get("filename") or "").strip()
        return f"{directory_name}/{filename}" if filename else ""

    def _match_score(entry: dict, fields, *, allow_filename_fallback: bool, allow_project_prefixed_exact_match: bool, project_name: str):
        for priority, field in enumerate(fields):
            raw_candidate = entry.get(field, "")
            candidate = _normalize_saved_voice_name(raw_candidate)
            if candidate and candidate == normalized_speaker:
                return priority
            if allow_project_prefixed_exact_match and _match_project_prefixed_saved_voice_name(raw_candidate, project_name, speaker):
                return priority
        if allow_filename_fallback:
            filename = os.path.splitext(str(entry.get("filename") or "").strip())[0]
            if filename:
                filename_parts = [part for part in re.split(r"[._-]+", filename) if part]
                if filename_parts:
                    trailing = _normalize_saved_voice_name(filename_parts[-1])
                    if trailing and trailing == normalized_speaker:
                        return len(fields)
        if allow_project_prefixed_exact_match:
            filename = os.path.splitext(str(entry.get("filename") or "").strip())[0]
            if _match_project_prefixed_saved_voice_name(filename, project_name, speaker):
                return len(fields)
        return None

    title_candidates = [current_script_title]
    if current_project_name and current_project_name not in title_candidates:
        title_candidates.append(current_project_name)

    best = None

    for title_priority, candidate_title in enumerate(title_candidates):
        allow_filename_fallback = title_priority == 0
        allow_project_prefixed_exact_match = bool(current_project_name) and title_priority > 0

        for entry in _load_manifest(CLONE_VOICES_MANIFEST):
            rel_audio = _build_rel_audio("clone_voices", entry)
            if not rel_audio or not os.path.exists(os.path.join(ROOT_DIR, rel_audio)):
                continue
            entry_script_title = _normalize_saved_voice_name(entry.get("script_title", ""))
            entry_matches_loaded_project_name = allow_project_prefixed_exact_match and (
                _match_project_prefixed_saved_voice_name(entry.get("speaker", ""), current_project_name, speaker)
                or _match_project_prefixed_saved_voice_name(entry.get("name", ""), current_project_name, speaker)
                or _match_project_prefixed_saved_voice_name(
                    os.path.splitext(str(entry.get("filename") or "").strip())[0],
                    current_project_name,
                    speaker,
                )
            )
            if (not entry_script_title or entry_script_title != candidate_title) and not entry_matches_loaded_project_name:
                continue
            score = _match_score(
                entry,
                ("speaker", "name"),
                allow_filename_fallback=allow_filename_fallback,
                allow_project_prefixed_exact_match=allow_project_prefixed_exact_match,
                project_name=current_project_name,
            )
            if score is None:
                continue
            candidate = {
                "type": "clone",
                "ref_audio": rel_audio,
                "ref_text": (entry.get("sample_text") or "").strip(),
                "generated_ref_text": (entry.get("sample_text") or "").strip(),
                "description": (entry.get("description") or "").strip(),
                "source_name": (entry.get("speaker") or entry.get("name") or "").strip(),
                "priority": (title_priority, 0, score),
            }
            if best is None or candidate["priority"] < best["priority"]:
                best = candidate

        for entry in _load_manifest(DESIGNED_VOICES_MANIFEST):
            rel_audio = _build_rel_audio("designed_voices", entry)
            if not rel_audio or not os.path.exists(os.path.join(ROOT_DIR, rel_audio)):
                continue
            entry_script_title = _normalize_saved_voice_name(entry.get("script_title", ""))
            entry_matches_loaded_project_name = allow_project_prefixed_exact_match and (
                _match_project_prefixed_saved_voice_name(entry.get("speaker", ""), current_project_name, speaker)
                or _match_project_prefixed_saved_voice_name(entry.get("name", ""), current_project_name, speaker)
                or _match_project_prefixed_saved_voice_name(
                    os.path.splitext(str(entry.get("filename") or "").strip())[0],
                    current_project_name,
                    speaker,
                )
            )
            if (not entry_script_title or entry_script_title != candidate_title) and not entry_matches_loaded_project_name:
                continue
            score = _match_score(
                entry,
                ("speaker", "name"),
                allow_filename_fallback=allow_filename_fallback,
                allow_project_prefixed_exact_match=allow_project_prefixed_exact_match,
                project_name=current_project_name,
            )
            if score is None:
                continue
            candidate = {
                "type": "clone",
                "ref_audio": rel_audio,
                "ref_text": (entry.get("sample_text") or "").strip(),
                "generated_ref_text": (entry.get("sample_text") or "").strip(),
                "description": (entry.get("description") or "").strip(),
                "source_name": (entry.get("speaker") or entry.get("name") or "").strip(),
                "priority": (title_priority, 1, score),
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


def _load_project_script_document():
    load_fn = getattr(project_manager, "load_script_document", None)
    if callable(load_fn):
        try:
            return load_fn()
        except Exception:
            pass
    return {"entries": [], "dictionary": [], "sanity_cache": {"phrase_decisions": {}}}


def _project_has_script_document():
    has_fn = getattr(project_manager, "script_store", None)
    if has_fn is not None and hasattr(project_manager.script_store, "has_script_entries"):
        try:
            return bool(project_manager.script_store.has_script_entries())
        except Exception:
            return False
    return bool(_load_project_script_document().get("entries"))


def _load_project_paragraphs_document():
    load_fn = getattr(project_manager, "load_paragraphs", None)
    if callable(load_fn):
        try:
            payload = load_fn()
            return payload if isinstance(payload, dict) else {}
        except Exception:
            return {}
    return {}


def _project_has_paragraphs_document():
    payload = _load_project_paragraphs_document()
    return bool(payload.get("paragraphs"))


def _load_project_script_sanity_result():
    script_store = getattr(project_manager, "script_store", None)
    if script_store is not None and hasattr(script_store, "load_project_document"):
        try:
            payload = script_store.load_project_document("script_sanity_result")
            return payload if isinstance(payload, dict) else None
        except Exception:
            return None
    return None


def _load_project_dictionary_entries():
    return _load_project_script_document()["dictionary"]


def _apply_project_dictionary(text):
    return apply_dictionary_to_text(text, _load_project_dictionary_entries())[0]


def _script_source_prefix(name: str) -> str:
    return f"{name}.source"


def _find_saved_script_source_companion(name: str):
    prefix = _script_source_prefix(name)
    matches = []
    if not os.path.isdir(SCRIPTS_DIR):
        return None
    for entry in os.listdir(SCRIPTS_DIR):
        if not entry.startswith(prefix):
            continue
        full_path = os.path.join(SCRIPTS_DIR, entry)
        if os.path.isfile(full_path):
            matches.append(full_path)
    if not matches:
        return None
    matches.sort(key=lambda path: os.path.getmtime(path), reverse=True)
    return matches[0]


def _save_script_source_companion(name: str):
    source_companion = _find_saved_script_source_companion(name)
    while source_companion:
        try:
            os.remove(source_companion)
        except OSError:
            break
        source_companion = _find_saved_script_source_companion(name)

    state = _load_project_state_payload()
    input_path = os.path.abspath((state.get("input_file_path") or "").strip())
    if not input_path or not os.path.exists(input_path) or not os.path.isfile(input_path):
        return None

    ext = os.path.splitext(input_path)[1]
    companion_name = f"{_script_source_prefix(name)}{ext}"
    companion_path = os.path.join(SCRIPTS_DIR, companion_name)
    shutil.copy2(input_path, companion_path)
    return companion_path


def _saved_script_db_companion_path(name: str):
    return os.path.join(SCRIPTS_DIR, f"{name}.sqlite3")


def _delete_saved_script_artifacts(name: str):
    for suffix in (".sqlite3", ".json"):
        companion = os.path.join(SCRIPTS_DIR, f"{name}{suffix}")
        if os.path.exists(companion):
            os.remove(companion)
    source_companion = _find_saved_script_source_companion(name)
    while source_companion:
        try:
            os.remove(source_companion)
        except OSError:
            break
        source_companion = _find_saved_script_source_companion(name)


def _resolve_project_save_name(name: str):
    requested_name = str(name or "").strip()
    if requested_name:
        safe_name = _sanitize_name(requested_name)
        if not safe_name:
            raise ValueError("Invalid project name.")
        return safe_name

    state = _load_project_state_payload()
    input_path = str(state.get("input_file_path") or "").strip()
    fallback_name = ""
    if input_path:
        fallback_name = os.path.splitext(os.path.basename(input_path))[0].strip()
    if not fallback_name:
        fallback_name = str(state.get("loaded_project_name") or "").strip()
    if not fallback_name:
        fallback_name = str(state.get("loaded_script_name") or "").strip()
    if not fallback_name:
        current_script_title = getattr(project_manager, "_current_script_title", None)
        if callable(current_script_title):
            try:
                fallback_name = str(current_script_title() or "").strip()
            except Exception:
                fallback_name = ""

    safe_name = _sanitize_name(fallback_name)
    if not safe_name:
        raise ValueError("No project name available. Import a source document or enter a project name.")
    return safe_name


def _project_has_export_outputs():
    for path in (
        AUDIOBOOK_PATH,
        OPTIMIZED_EXPORT_PATH,
        _project_export_filesystem_path("audacity_export.zip"),
        M4B_PATH,
        _project_export_filesystem_path("m4b_cover.jpg"),
    ):
        if os.path.exists(path):
            return True
    return False


def _project_has_durable_archive_state():
    if _project_has_script_document():
        return True
    has_substantive_chunks = getattr(project_manager, "has_substantive_chunks", None)
    if callable(has_substantive_chunks):
        try:
            if bool(has_substantive_chunks()):
                return True
        except Exception:
            pass
    has_voice_config = getattr(project_manager, "has_voice_config", None)
    if callable(has_voice_config):
        try:
            if bool(has_voice_config()):
                return True
        except Exception:
            pass
    if bool((_load_project_paragraphs_document() or {}).get("paragraphs")):
        return True
    if _project_has_generated_audio() or _project_has_export_outputs():
        return True
    state = _archive_state_with_relative_paths()
    return bool(str(state.get("input_file_path") or "").strip())


def _save_current_script_snapshot(name: str, *, purge_existing: bool = False):
    if not _project_has_script_document():
        raise FileNotFoundError("No script to save. Generate a script first.")

    safe_name = _sanitize_name(name)
    if not safe_name:
        raise ValueError("Invalid script name.")

    dest = _saved_script_db_companion_path(safe_name)
    existed = os.path.exists(dest)
    if purge_existing and existed:
        _delete_saved_script_artifacts(safe_name)

    db_path = getattr(project_manager, "chunks_db_path", os.path.join(ROOT_DIR, "chunks.sqlite3"))
    if not os.path.exists(db_path):
        raise FileNotFoundError("No SQLite project state to save.")
    _copy_sqlite_database_snapshot(db_path, dest)
    _save_script_source_companion(safe_name)

    state = _load_project_state_payload()
    state["loaded_script_name"] = safe_name
    _save_project_state_payload(state)
    if hasattr(project_manager, "log_voice_audit_event"):
        project_manager.log_voice_audit_event(
            "script_snapshot_write",
            reason="save_script_snapshot",
            snapshot_name=safe_name,
        )
    return {"name": safe_name, "overwrote": existed}


def _autosave_name_from_input_file():
    state = _load_project_state_payload()
    input_path = (state.get("input_file_path") or "").strip()
    if not input_path:
        return ""
    return _sanitize_name(os.path.splitext(os.path.basename(input_path))[0])


def _autosave_current_script_for_workflow(*, purge_existing: bool, trigger: str):
    auto_name = _autosave_name_from_input_file()
    if not auto_name:
        raise RuntimeError("Cannot auto-save script: no imported source file name found.")
    result = _save_current_script_snapshot(auto_name, purge_existing=purge_existing)
    logger.info(
        "Workflow auto-saved script '%s' (trigger=%s, purge_existing=%s)",
        result["name"],
        trigger,
        purge_existing,
    )
    return result


def _save_current_project_archive_snapshot(name: str):
    if not _project_has_durable_archive_state():
        raise FileNotFoundError("No durable project state to save.")

    safe_name = _resolve_project_save_name(name)

    archive_path = _saved_project_archive_path(safe_name)
    existed = os.path.exists(archive_path)
    archive_result = _write_project_archive(archive_path)
    _delete_saved_script_artifacts(safe_name)
    return {
        "name": safe_name,
        "overwrote": existed,
        "entries": archive_result["entries"],
    }


def _extract_first_json_object(text):
    depth = 0
    start = None
    in_string = False
    escaped = False

    for index, char in enumerate(text or ""):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
            continue
        if char == "{":
            if depth == 0:
                start = index
            depth += 1
        elif char == "}":
            if depth == 0:
                continue
            depth -= 1
            if depth == 0 and start is not None:
                return text[start:index + 1]
    return None


def _extract_voice_field(text):
    candidate = (text or "").strip()
    if not candidate:
        return ""

    direct_match = re.search(r'"voice"\s*:\s*"([^"]+)"', candidate, re.IGNORECASE | re.DOTALL)
    if direct_match:
        return direct_match.group(1).strip()

    json_blob = _extract_first_json_object(candidate)
    payload = None
    if json_blob:
        try:
            payload = json.loads(json_blob)
        except json.JSONDecodeError:
            payload = None

    if isinstance(payload, dict):
        voice = payload.get("voice")
        if isinstance(voice, str) and voice.strip():
            return voice.strip()

    plain = candidate
    if "```" in plain:
        plain = re.sub(r"^```(?:json)?\s*", "", plain, flags=re.IGNORECASE)
        plain = re.sub(r"\s*```$", "", plain)
    first_line = ""
    for line in plain.splitlines():
        stripped = line.strip()
        if stripped:
            first_line = stripped
            break
    if not first_line:
        return ""

    first_line = re.sub(r"(?i)^voice\s*[:=-]\s*", "", first_line).strip().strip("`").strip()
    if len(first_line) >= 2 and first_line[0] == first_line[-1] and first_line[0] in {"'", '"'}:
        first_line = first_line[1:-1].strip()
    return first_line


def _any_project_task_running():
    for name, state in process_state.items():
        if bool(state.get("running")):
            return name
    with audio_queue_lock:
        if audio_queue or audio_current_job is not None:
            return "audio"
    return None


def _ensure_task_not_running(task_name: str, conflict_message: str):
    with task_state_lock:
        if bool(process_state.get(task_name, {}).get("running")):
            raise HTTPException(status_code=409, detail=conflict_message)

# Mount static files with absolute path
STATIC_DIR = os.path.join(BASE_DIR, "static")
os.makedirs(STATIC_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Create voicelines directory if it doesn't exist to prevent startup error
VOICELINES_DIR = LAYOUT.voicelines_dir
os.makedirs(VOICELINES_DIR, exist_ok=True)
app.mount("/voicelines", _VoicelinesProxyStatic(directory=VOICELINES_DIR), name="voicelines")

# Designed voices directory for voice designer feature
app.mount("/designed_voices", StaticFiles(directory=DESIGNED_VOICES_DIR), name="designed_voices")

# Clone voices directory for user-uploaded reference audio
app.mount("/clone_voices", StaticFiles(directory=CLONE_VOICES_DIR), name="clone_voices")

# LoRA models directory for trained adapter test audio
app.mount("/lora_models", StaticFiles(directory=LORA_MODELS_DIR), name="lora_models")

# Built-in LoRA adapters directory
os.makedirs(BUILTIN_LORA_DIR, exist_ok=True)
app.mount("/builtin_lora", StaticFiles(directory=BUILTIN_LORA_DIR), name="builtin_lora")

# Dataset builder directory for preview audio
app.mount("/dataset_builder", StaticFiles(directory=DATASET_BUILDER_DIR), name="dataset_builder")

# Emotions page directory for standalone prompt-test audio
os.makedirs(EMOTIONS_AUDIO_DIR, exist_ok=True)
app.mount("/emotions_audio", StaticFiles(directory=EMOTIONS_AUDIO_DIR), name="emotions_audio")

# Initialize Project Manager
project_manager = ProjectManager(ROOT_DIR)

# Reset any chunks stuck in "generating" from a prior interrupted session
_startup_recovery = project_manager.recover_interrupted_generating_chunks()
if _startup_recovery["recovered"] or _startup_recovery["reset"]:
    print(
        "Startup: recovered "
        f"{_startup_recovery['recovered']} interrupted chunk(s) with valid audio and reset "
        f"{_startup_recovery['reset']} chunk(s) back to pending"
    )
del _startup_recovery

# CORS for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Data Models
class LLMConfig(BaseModel):
    base_url: str
    api_key: str
    model_name: str
    llm_workers: int = 1  # concurrent LLM requests

    @field_validator("base_url")
    @classmethod
    def normalize_base_url(cls, v: str) -> str:
        url = str(v or "").strip().rstrip("/")
        if not url.endswith("/v1"):
            url = url + "/v1"
        return url

    @field_validator("api_key", "model_name")
    @classmethod
    def strip_text_fields(cls, v: str) -> str:
        return str(v or "").strip()

class TTSConfig(BaseModel):
    provider: Literal["qwen3", "voxcpm2"] = "qwen3"
    mode: str = "local"  # "local" or "external"
    local_backend: str = "auto"  # local mode only: "auto", "qwen", "mlx"
    url: str = "http://127.0.0.1:7860"  # external mode only
    device: str = "auto"  # local mode: "auto", "cuda:0", "cpu", etc.
    language: str = "English"  # TTS language
    parallel_workers: int = 4  # concurrent TTS workers
    auto_regenerate_bad_clips: bool = True
    auto_regenerate_bad_clip_attempts: int = 3
    batch_seed: Optional[int] = None  # Single seed for batch mode, None/-1 = random
    compile_codec: bool = False  # torch.compile the codec for ~3-4x batch throughput (slow first run)
    sub_batch_enabled: bool = True  # split batch by text length to reduce padding waste
    sub_batch_min_size: int = 4  # minimum chunks per sub-batch before allowing a split
    sub_batch_ratio: float = 5.0  # max longest/shortest length ratio before splitting
    sub_batch_max_chars: int = 3000  # max total chars per sub-batch (lower for less VRAM)
    sub_batch_max_items: int = 0  # hard cap on sequences per sub-batch (0 = auto from VRAM estimate)
    batch_group_by_type: bool = False  # group chunks by voice type for efficient batching
    script_max_length: int = QWEN3_SCRIPT_MAX_LENGTH_DEFAULT  # Max chars per chunk in Create Script (-1 = one chunk per sentence)
    voxcpm_model_id: str = "openbmb/VoxCPM2"
    voxcpm_cfg_value: float = VOXCPM2_CFG_VALUE_DEFAULT
    voxcpm_inference_timesteps: int = VOXCPM2_INFERENCE_TIMESTEPS_DEFAULT
    voxcpm_normalize: bool = False
    voxcpm_denoise: bool = False
    voxcpm_load_denoiser: bool = False
    voxcpm_denoise_reference: bool = False
    voxcpm_optimize: bool = False

    @model_validator(mode="before")
    @classmethod
    def apply_provider_defaults(cls, values):
        if not isinstance(values, dict):
            return values
        data = dict(values)
        provider = data.get("provider", "qwen3")
        if "script_max_length" in data and data.get("script_max_length") in (None, ""):
            data["script_max_length"] = tts_script_max_length_default(provider)
        if "voxcpm_cfg_value" in data and data.get("voxcpm_cfg_value") in (None, ""):
            data["voxcpm_cfg_value"] = VOXCPM2_CFG_VALUE_DEFAULT
        if "voxcpm_inference_timesteps" in data and data.get("voxcpm_inference_timesteps") in (None, ""):
            data["voxcpm_inference_timesteps"] = VOXCPM2_INFERENCE_TIMESTEPS_DEFAULT
        return data

    @model_validator(mode="after")
    def apply_provider_script_max_length_default(self):
        if "script_max_length" not in self.model_fields_set and self.provider == "voxcpm2":
            object.__setattr__(self, "script_max_length", VOXCPM2_SCRIPT_MAX_LENGTH_DEFAULT)
        return self

    @field_validator("voxcpm_cfg_value", mode="before")
    @classmethod
    def clamp_voxcpm_cfg(cls, value):
        return clamp_voxcpm_cfg_value(value)

    @field_validator("voxcpm_inference_timesteps", mode="before")
    @classmethod
    def clamp_voxcpm_steps(cls, value):
        return clamp_voxcpm_inference_timesteps(value)

class GenerationConfig(BaseModel):
    chunk_size: int = 3000
    temperament_words: int = 150
    script_error_retry_attempts: int = 3
    max_tokens: int = 4096
    temperature: float = 0.6
    top_p: float = 0.8
    top_k: int = 20
    min_p: float = 0
    presence_penalty: float = 0.0
    banned_tokens: List[str] = []
    merge_narrators: bool = False
    orphaned_text_to_narrator_on_repair: bool = True
    legacy_mode: bool = False

class UIConfig(BaseModel):
    dark_mode: bool = True

class PromptConfig(BaseModel):
    system_prompt: Optional[str] = None
    user_prompt: Optional[str] = None
    review_system_prompt: Optional[str] = None
    review_user_prompt: Optional[str] = None
    attribution_system_prompt: Optional[str] = None
    attribution_user_prompt: Optional[str] = None
    voice_prompt: Optional[str] = None
    dialogue_identification_system_prompt: Optional[str] = None
    temperament_extraction_system_prompt: Optional[str] = None

class ExportConfig(BaseModel):
    silence_between_speakers_ms: int = 500
    silence_same_speaker_ms: int = 250
    silence_end_of_chapter_ms: int = 3000
    silence_paragraph_ms: int = 750
    trim_clip_silence_enabled: bool = True
    trim_silence_threshold_dbfs: float = -50.0
    trim_min_silence_len_ms: int = 150
    trim_keep_padding_ms: int = 40
    normalize_enabled: bool = True
    normalize_target_lufs_mono: float = -18.0
    normalize_target_lufs_stereo: float = -16.0
    normalize_true_peak_dbtp: float = -1.0
    normalize_lra: float = 11.0

class AppConfig(BaseModel):
    llm: LLMConfig
    tts: TTSConfig
    prompts: Optional[PromptConfig] = None
    generation: Optional[GenerationConfig] = None
    proofread: Optional[dict] = None
    export: Optional[ExportConfig] = None
    ui: Optional[UIConfig] = None

class SetupConfigUpdate(BaseModel):
    llm: Optional[LLMConfig] = None
    tts: Optional[TTSConfig] = None
    prompts: Optional[PromptConfig] = None
    generation: Optional[GenerationConfig] = None
    proofread: Optional[dict] = None

class PreferencesUpdate(BaseModel):
    legacy_mode: Optional[bool] = None
    dark_mode: Optional[bool] = None

class GenerationModeLockUpdate(BaseModel):
    locked: bool = True
    trigger: Optional[str] = None

class VoiceConfigItem(BaseModel):
    type: str = "custom"
    voice: Optional[str] = "Ryan"
    character_style: Optional[str] = ""
    default_style: Optional[str] = ""  # backward compat, prefer character_style
    alias: Optional[str] = ""
    seed: Optional[str] = "-1"
    ref_audio: Optional[str] = None
    ref_text: Optional[str] = None
    generated_ref_text: Optional[str] = None
    adapter_id: Optional[str] = None
    adapter_path: Optional[str] = None
    description: Optional[str] = ""  # voice description (for design type)
    narrates: Optional[bool] = None  # True = this voice acts as a narrator
    user_created: Optional[bool] = None

class ChunkUpdate(BaseModel):
    text: Optional[str] = None
    instruct: Optional[str] = None
    speaker: Optional[str] = None
    silence_duration_s: Optional[float] = None

class ChunkDecomposeRequest(BaseModel):
    chapter: Optional[str] = None
    max_words: int = 25

class ChunkMergeOrphansRequest(BaseModel):
    chapter: Optional[str] = None
    min_words: int = 10

class ChunkRepairLegacyRequest(BaseModel):
    chunks: List[dict]

class LostAudioRepairRequest(BaseModel):
    use_asr: bool = True
    rejected_only: bool = False

class ProofreadRequest(BaseModel):
    chapter: Optional[str] = None
    threshold: float = 0.75

class ProofreadClearFailuresRequest(BaseModel):
    chapter: Optional[str] = None
    threshold: float = 0.75

class ProofreadValidateRequest(BaseModel):
    threshold: float = 0.75

class ProofreadCompareRequest(BaseModel):
    threshold: float = 0.75

class ProofreadDiscardSelectionRequest(BaseModel):
    chapter: Optional[str] = None

class ASRTranscribeRequest(BaseModel):
    audio_path: str

class RenderPrepStateRequest(BaseModel):
    complete: bool = True

class ChunkGenerateRequest(BaseModel):
    neutral_narrator: bool = False

class BatchGenerateRequest(BaseModel):
    indices: List[Union[str, int]] = []
    label: Optional[str] = None
    scope: Optional[str] = None
    scope_mode: Optional[str] = None
    chapter: Optional[str] = None
    regenerate_all: bool = False
    neutral_narrator: bool = False


class DictionaryEntry(BaseModel):
    source: str = ""
    alias: str = ""


class DictionarySaveRequest(BaseModel):
    entries: List[DictionaryEntry]


class VoiceDesignGenerateRequest(BaseModel):
    speaker: str
    description: str
    sample_text: Optional[str] = None
    force: bool = False

class VoiceConfigSaveRequest(BaseModel):
    config: Dict[str, VoiceConfigItem]
    confirm_invalidation: bool = False


class VoiceDescriptionSuggestRequest(BaseModel):
    speaker: str


class VoiceDescriptionSuggestBatchRequest(BaseModel):
    speakers: List[str]

class VoiceDesignPreviewRequest(BaseModel):
    description: str
    sample_text: str
    language: Optional[str] = None

class VoiceDesignSaveRequest(BaseModel):
    name: str
    description: str
    sample_text: str
    preview_file: str

class LoraTrainingRequest(BaseModel):
    name: str
    dataset_id: str
    epochs: int = 5
    lr: float = 5e-6
    batch_size: int = 1
    lora_r: int = 32
    lora_alpha: int = 128
    gradient_accumulation_steps: int = 8

class LoraTestRequest(BaseModel):
    adapter_id: str
    text: str
    instruct: str = ""

class LoraDatasetSample(BaseModel):
    emotion: str = ""
    text: str

class LoraGenerateDatasetRequest(BaseModel):
    name: str
    description: str  # root voice description
    samples: Optional[List[LoraDatasetSample]] = None  # emotion+text pairs
    texts: Optional[List[str]] = None  # legacy: flat text list (no emotions)
    language: Optional[str] = None

class DatasetSampleGenRequest(BaseModel):
    description: str      # full voice description (root + emotion already combined by frontend)
    text: str
    dataset_name: str     # working directory name
    sample_index: int     # row number
    seed: int = -1        # -1 = random, >= 0 = manual seed

class DatasetBatchGenRequest(BaseModel):
    name: str
    description: str      # root voice description
    samples: List[LoraDatasetSample]
    indices: Optional[List[int]] = None  # which rows to generate (None = all)
    global_seed: int = -1 # -1 = random, >= 0 = same seed for all lines
    seeds: Optional[List[int]] = None  # per-line seeds (overrides global_seed)

class DatasetSaveRequest(BaseModel):
    name: str
    ref_index: int = 0    # which sample to use as ref.wav

class DatasetBuilderCreateRequest(BaseModel):
    name: str

class DatasetBuilderUpdateMetaRequest(BaseModel):
    name: str
    description: str = ""
    global_seed: str = ""

class DatasetBuilderUpdateRowsRequest(BaseModel):
    name: str
    rows: List[dict]  # [{emotion, text, seed}]


class ProcessingWorkflowRequest(BaseModel):
    process_voices: bool = True
    generate_audio: bool = False
    force_reimport: bool = False
    skip_script_stage: bool = False


class NewModeWorkflowRequest(BaseModel):
    process_voices: bool = True
    generate_audio: bool = False
    full_cast: bool = True


class ScriptGenerationRequest(BaseModel):
    force_reimport: bool = False
    skip_import: bool = False


class WorkflowPauseRequested(Exception):
    pass

# Global state for process tracking
ROLLING_AUDIO_SAMPLE_LIMIT = 50
AUDIO_HEARTBEAT_INTERVAL_SECONDS = 600
AUDIO_RECOVERY_POLL_SECONDS = 5
PROCESSING_WORKFLOW_STATE_PATH = LAYOUT.processing_workflow_state_path
NEW_MODE_WORKFLOW_STATE_PATH = LAYOUT.new_mode_workflow_state_path
NEW_MODE_STAGE_LABELS = {
    "process_paragraphs": "Process Paragraphs",
    "assign_dialogue": "Assign Dialogue",
    "extract_temperament": "Extract Temperament",
    "create_script": "Create Script",
    "process_voices": "Process Voices",
    "render_audio": "Render Audio",
    "proofread": "Proofread",
}
NEW_MODE_STAGE_ORDER = [
    "process_paragraphs",
    "assign_dialogue",
    "extract_temperament",
    "create_script",
    "process_voices",
    "render_audio",
    "proofread",
]
PROCESSING_STAGE_ORDER = ["script", "review", "sanity", "repair", "voices", "audio"]
PROCESSING_STAGE_MARKERS_KEY = "processing_stage_markers"
NEW_MODE_STAGE_MARKERS_KEY = "new_mode_stage_markers"
PROCESSING_WORKFLOW_STAGE_LABELS = {
    "script": "Generate Annotated Script",
    "review": "Review Script",
    "sanity": "Sanity Check",
    "repair": "Replace Missing Chunks",
    "voices": "Process Voices",
    "audio": "Generate Audio",
}


process_state = {
    "script": {"running": False, "logs": []},
    "voices": {"running": False, "logs": []},
    "proofread": {"running": False, "logs": [], "progress": {}},
    "audio": {
        "running": False,
        "logs": [],
        "cancel": False,
        "queue": [],
        "current_job": None,
        "recent_jobs": [],
        "merge_running": False,
        "merge_progress": {},
        "metrics": {},
        "heartbeat": {},
        "audio_coverage": {},
    },
    "audacity_export": {"running": False, "logs": []},
    "m4b_export": {"running": False, "logs": []},
    "review": {"running": False, "logs": []},
    "sanity": {"running": False, "logs": []},
    "repair": {"running": False, "logs": []},
    "lora_training": {"running": False, "logs": []},
    "dataset_gen": {"running": False, "logs": []},
    "dataset_builder": {"running": False, "logs": [], "cancel": False},
    "process_paragraphs": {"running": False, "logs": [], "progress": {}},
    "assign_dialogue": {"running": False, "logs": [], "progress": {}},
    "extract_temperament": {"running": False, "logs": [], "progress": {}},
    "create_script": {"running": False, "logs": [], "progress": {}},
    "new_mode_workflow": {
        "running": False,
        "paused": False,
        "pause_requested": False,
        "current_stage": None,
        "completed_stages": [],
        "options": {"process_voices": True, "generate_audio": False, "full_cast": True},
        "logs": [],
        "last_error": None,
        "started_at": None,
        "updated_at": None,
        "completed_at": None,
    },
    "processing_workflow": {
        "running": False,
        "paused": False,
        "pause_requested": False,
        "current_stage": None,
        "completed_stages": [],
        "options": {"process_voices": True, "generate_audio": False},
        "logs": [],
        "last_error": None,
        "started_at": None,
        "updated_at": None,
        "completed_at": None,
        "resume_count": 0,
    },
}

audio_queue_lock = threading.RLock()
audio_queue_condition = threading.Condition(audio_queue_lock)
audio_queue = []
audio_current_job = None
audio_cancel_event = threading.Event()
audio_current_runner_thread = None
audio_current_runner_token = None
audio_job_counter = 0
audio_recovery_request = None


def _force_kill_thread(thread):
    """Brutally terminate a Python thread via async exception injection."""
    try:
        import ctypes
        thread_id = getattr(thread, "ident", None)
        if not thread_id:
            return False
        result = ctypes.pythonapi.PyThreadState_SetAsyncExc(
            ctypes.c_long(thread_id),
            ctypes.py_object(SystemExit),
        )
        if result == 0:
            return False
        if result > 1:
            ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_long(thread_id), None)
            return False
        return True
    except Exception:
        return False


def _force_kill_active_audio_runner_locked():
    global audio_current_runner_thread, audio_current_runner_token
    runner = audio_current_runner_thread
    if runner is None:
        return False
    killed = _force_kill_thread(runner)
    if killed:
        _append_audio_log_locked("[CANCEL] Force-killed active audio runner thread.")
    else:
        _append_audio_log_locked("[CANCEL] Could not force-kill active audio runner thread.")
    audio_current_runner_thread = None
    audio_current_runner_token = None
    return killed


def _trim_logs(logs, limit=1000):
    while len(logs) > limit:
        logs.pop(0)


def _count_words(text):
    return len(re.findall(r"\b\w+\b", text or ""))


def _new_audio_metrics():
    return {
        "sample_window_size": ROLLING_AUDIO_SAMPLE_LIMIT,
        "samples": [],
        "processed_clips": 0,
        "successful_clips": 0,
        "error_clips": 0,
        "rolling_seconds": 0.0,
        "rolling_output_words": 0,
        "rolling_input_words": 0,
        "total_elapsed_seconds": 0.0,
        "total_output_words": 0,
        "total_input_words": 0,
        "remaining_words": 0,
        "estimated_remaining_seconds": None,
        "words_per_minute": None,
        "error_rate": 0.0,
    }


def _normalize_audio_metrics(raw_metrics=None):
    metrics = _new_audio_metrics()
    raw_metrics = raw_metrics or {}

    try:
        sample_window_size = max(1, int(raw_metrics.get("sample_window_size") or metrics["sample_window_size"]))
    except (TypeError, ValueError):
        sample_window_size = metrics["sample_window_size"]
    metrics["sample_window_size"] = sample_window_size

    normalized_samples = []
    for raw_sample in list(raw_metrics.get("samples") or [])[-sample_window_size:]:
        if not isinstance(raw_sample, dict):
            continue
        try:
            normalized_samples.append({
                "job_id": int(raw_sample.get("job_id", 0) or 0),
                "chunk_uid": str(raw_sample.get("chunk_uid") or "").strip(),
                "elapsed_seconds": max(0.0, float(raw_sample.get("elapsed_seconds", 0.0) or 0.0)),
                "input_words": max(0, int(raw_sample.get("input_words", 0) or 0)),
                "output_words": max(0, int(raw_sample.get("output_words", 0) or 0)),
                "success": bool(raw_sample.get("success")),
            })
        except (TypeError, ValueError):
            continue
    metrics["samples"] = normalized_samples
    metrics["rolling_seconds"] = sum(sample["elapsed_seconds"] for sample in normalized_samples)
    metrics["rolling_input_words"] = sum(sample["input_words"] for sample in normalized_samples)
    metrics["rolling_output_words"] = sum(sample["output_words"] for sample in normalized_samples)

    try:
        processed_clips = max(0, int(raw_metrics.get("processed_clips", 0) or 0))
    except (TypeError, ValueError):
        processed_clips = 0
    try:
        error_clips = max(0, int(raw_metrics.get("error_clips", 0) or 0))
    except (TypeError, ValueError):
        error_clips = 0
    error_clips = min(error_clips, processed_clips)
    metrics["processed_clips"] = processed_clips
    metrics["error_clips"] = error_clips
    metrics["successful_clips"] = processed_clips - error_clips

    try:
        metrics["total_elapsed_seconds"] = max(0.0, float(raw_metrics.get("total_elapsed_seconds", 0.0) or 0.0))
    except (TypeError, ValueError):
        metrics["total_elapsed_seconds"] = 0.0
    try:
        metrics["total_output_words"] = max(0, int(raw_metrics.get("total_output_words", 0) or 0))
    except (TypeError, ValueError):
        metrics["total_output_words"] = 0
    try:
        metrics["total_input_words"] = max(0, int(raw_metrics.get("total_input_words", 0) or 0))
    except (TypeError, ValueError):
        metrics["total_input_words"] = 0
    try:
        metrics["remaining_words"] = max(0, int(raw_metrics.get("remaining_words", 0) or 0))
    except (TypeError, ValueError):
        metrics["remaining_words"] = 0

    return metrics


def _default_audio_coverage_summary():
    return {
        "total_clips": 0,
        "valid_clips": 0,
        "invalid_clips": 0,
        "percentage": 0,
    }


process_state["audio"]["metrics"] = _new_audio_metrics()


def _new_audio_merge_progress():
    return {
        "running": False,
        "stage": None,
        "chapter_index": 0,
        "total_chapters": 0,
        "chapter_label": "",
        "elapsed_seconds": 0.0,
        "merged_duration_seconds": 0.0,
        "estimated_size_bytes": 0,
        "output_file_size_bytes": 0,
        "updated_at": None,
    }


process_state["audio"]["merge_progress"] = _new_audio_merge_progress()


def _new_audio_heartbeat_state():
    return {
        "interval_seconds": AUDIO_HEARTBEAT_INTERVAL_SECONDS,
        "last_check_at": None,
        "last_output_at": None,
        "last_generation_activity_at": None,
        "last_finalize_activity_at": None,
        "last_recovery_at": None,
        "recovery_count": 0,
        "last_recovery_reason": None,
    }


process_state["audio"]["heartbeat"] = _new_audio_heartbeat_state()

task_state_lock = threading.RLock()
task_processes = {}
processing_workflow_lock = threading.RLock()
processing_workflow_thread = None
new_mode_workflow_lock = threading.RLock()
new_mode_workflow_thread = None
TASK_PROGRESS_PREFIX = "__TASK_PROGRESS__:"
NAV_TASK_TABS = {"script", "voices", "editor", "proofread", "audio"}
nav_task_lock = threading.RLock()
nav_task_state = {
    "tab": None,
    "updated_at": None,
}


def _serialize_nav_task_state_locked():
    return {
        "tab": nav_task_state.get("tab"),
        "updated_at": nav_task_state.get("updated_at"),
    }


def _set_nav_task_tab(tab: str):
    if tab not in NAV_TASK_TABS:
        raise HTTPException(status_code=400, detail="Invalid navigation task tab")
    with nav_task_lock:
        nav_task_state["tab"] = tab
        nav_task_state["updated_at"] = time.time()
        return _serialize_nav_task_state_locked()


def _release_nav_task_tab(tab: Optional[str] = None):
    with nav_task_lock:
        if tab is None or nav_task_state.get("tab") == tab:
            nav_task_state["tab"] = None
            nav_task_state["updated_at"] = time.time()
        return _serialize_nav_task_state_locked()


def _get_nav_task_state():
    with nav_task_lock:
        return _serialize_nav_task_state_locked()


def _task_is_current(task_name: str, run_id: str) -> bool:
    with task_state_lock:
        return process_state.get(task_name, {}).get("run_id") == run_id


def _append_task_log(task_name: str, run_id: str, message: str) -> bool:
    with task_state_lock:
        state = process_state.get(task_name)
        if not state or state.get("run_id") != run_id:
            return False
        state["logs"].append(message)
        _trim_logs(state["logs"])
        return True


def _set_task_progress(task_name: str, run_id: str, progress: dict) -> bool:
    with task_state_lock:
        state = process_state.get(task_name)
        if not state or state.get("run_id") != run_id:
            return False
        state["progress"] = dict(progress or {})
        return True


def _status_text_snippet(text: str, limit: int = 100) -> str:
    normalized = " ".join(str(text or "").split())
    if len(normalized) <= limit:
        return normalized
    return normalized[: max(0, limit - 3)].rstrip() + "..."


def _format_invalid_text_details(missing_text: str = "", inserted_text: str = "") -> str:
    parts = []
    missing_snippet = _status_text_snippet(missing_text)
    inserted_snippet = _status_text_snippet(inserted_text)
    if missing_snippet:
        parts.append(f'missing="{missing_snippet}"')
    if inserted_snippet:
        parts.append(f'inserted="{inserted_snippet}"')
    return ", ".join(parts)


def _start_task_run(task_name: str) -> str:
    run_id = str(uuid.uuid4())
    with task_state_lock:
        previous_running = bool(process_state[task_name].get("running"))
        previous_process = task_processes.get(task_name)
        process_state[task_name]["run_id"] = run_id
        process_state[task_name]["running"] = True
        process_state[task_name]["logs"] = []
        if "progress" in process_state[task_name]:
            process_state[task_name]["progress"] = {}
        if previous_running:
            process_state[task_name]["logs"].append(
                "Restart requested. Using the latest Setup settings and attempting to continue from saved progress."
            )
            _trim_logs(process_state[task_name]["logs"])
        if previous_process and previous_process.poll() is None:
            try:
                previous_process.terminate()
            except Exception:
                pass
    return run_id


def _finish_task_run(task_name: str, run_id: str, process=None):
    with task_state_lock:
        state = process_state.get(task_name)
        if not state or state.get("run_id") != run_id:
            return
        state["running"] = False
        if "progress" in state:
            state["progress"] = dict(state.get("progress") or {}) | {
                "running": False,
                "completed_at": time.time(),
            }
        if process is not None and task_processes.get(task_name) is process:
            task_processes.pop(task_name, None)


def _register_task_process(task_name: str, run_id: str, process):
    with task_state_lock:
        if process_state.get(task_name, {}).get("run_id") != run_id:
            try:
                process.terminate()
            except Exception:
                pass
            return False
        task_processes[task_name] = process
        return True


def _format_audio_metrics_locked():
    metrics = process_state["audio"]["metrics"]
    return {
        "sample_window_size": metrics["sample_window_size"],
        "samples": list(metrics.get("samples", [])),
        "processed_clips": metrics["processed_clips"],
        "successful_clips": metrics["successful_clips"],
        "error_clips": metrics["error_clips"],
        "rolling_seconds": metrics["rolling_seconds"],
        "rolling_output_words": metrics["rolling_output_words"],
        "rolling_input_words": metrics["rolling_input_words"],
        "total_elapsed_seconds": metrics["total_elapsed_seconds"],
        "total_output_words": metrics["total_output_words"],
        "total_input_words": metrics["total_input_words"],
        "remaining_words": metrics["remaining_words"],
        "estimated_remaining_seconds": metrics["estimated_remaining_seconds"],
        "words_per_minute": metrics["words_per_minute"],
        "error_rate": metrics["error_rate"],
    }


def _format_audio_heartbeat_locked():
    heartbeat = process_state["audio"]["heartbeat"]
    return {
        "interval_seconds": heartbeat["interval_seconds"],
        "last_check_at": heartbeat["last_check_at"],
        "last_output_at": heartbeat["last_output_at"],
        "last_generation_activity_at": heartbeat.get("last_generation_activity_at"),
        "last_finalize_activity_at": heartbeat.get("last_finalize_activity_at"),
        "last_recovery_at": heartbeat["last_recovery_at"],
        "recovery_count": heartbeat["recovery_count"],
        "last_recovery_reason": heartbeat["last_recovery_reason"],
    }


def _recompute_audio_metrics_locked():
    metrics = process_state["audio"]["metrics"]
    metrics["remaining_words"] = sum(job.get("remaining_words", 0) for job in audio_queue)
    if audio_current_job is not None:
        metrics["remaining_words"] += audio_current_job.get("remaining_words", 0)

    if metrics["total_elapsed_seconds"] > 0 and metrics["total_input_words"] > 0:
        words_per_second = metrics["total_input_words"] / metrics["total_elapsed_seconds"]
        metrics["words_per_minute"] = words_per_second * 60.0
        metrics["estimated_remaining_seconds"] = metrics["remaining_words"] / words_per_second if metrics["remaining_words"] > 0 else 0.0
    else:
        metrics["words_per_minute"] = None
        metrics["estimated_remaining_seconds"] = None if metrics["remaining_words"] > 0 else 0.0

    if metrics["processed_clips"] > 0:
        metrics["error_rate"] = metrics["error_clips"] / metrics["processed_clips"]
    else:
        metrics["error_rate"] = 0.0


def _job_uids(job):
    if not job:
        return []
    return [str(uid).strip() for uid in (job.get("uids") or job.get("indices") or []) if str(uid).strip()]


def _job_pending_uids(job):
    if not job:
        return []
    return [str(uid).strip() for uid in (job.get("pending_uids") or job.get("pending_indices") or []) if str(uid).strip()]


def _job_generation_pending_uids(job):
    if not job:
        return []
    raw = job.get("generation_pending_uids")
    if raw is None:
        raw = _job_pending_uids(job)
    return [str(uid).strip() for uid in (raw or []) if str(uid).strip()]


def _job_pending_finalize_uids(job):
    if not job:
        return []
    return [str(uid).strip() for uid in (job.get("pending_finalize_uids") or []) if str(uid).strip()]


def _job_word_counts(job):
    raw = job.get("word_counts_by_uid")
    if raw is None:
        raw = job.get("word_counts") or {}
    return {str(uid).strip(): int(count or 0) for uid, count in dict(raw).items() if str(uid).strip()}


def _live_pending_finalize_uids_for_job(raw_job, candidate_uids=None):
    run_token = str((raw_job or {}).get("run_token") or "").strip()
    if not run_token:
        return []

    normalized_candidates = [str(uid).strip() for uid in (candidate_uids or []) if str(uid).strip()]
    candidate_lookup = set(normalized_candidates) if normalized_candidates else None

    try:
        tasks = project_manager.list_audio_finalize_tasks(
            generation_token=run_token,
            statuses=("queued", "processing"),
        )
    except Exception:
        return []

    live_uids = []
    seen = set()
    for task in tasks or []:
        uid = str((task or {}).get("chunk_uid") or "").strip()
        if not uid or uid in seen:
            continue
        if candidate_lookup is not None and uid not in candidate_lookup:
            continue
        seen.add(uid)
        live_uids.append(uid)

    if not live_uids:
        return []

    try:
        rows = project_manager.get_chunks_by_uids(live_uids)
    except Exception:
        return live_uids

    row_by_uid = {
        str((row or {}).get("uid") or "").strip(): row
        for row in rows or []
        if str((row or {}).get("uid") or "").strip()
    }
    return [
        uid for uid in live_uids
        if (row_by_uid.get(uid) or {}).get("status") not in {"done", "error"}
    ]


def _reconcile_audio_job_runtime_locked(job):
    if not job:
        return job

    tracked_uids = []
    for uid in _job_pending_uids(job) + _job_generation_pending_uids(job) + _job_pending_finalize_uids(job):
        normalized_uid = str(uid).strip()
        if normalized_uid and normalized_uid not in tracked_uids:
            tracked_uids.append(normalized_uid)

    live_finalize_uids = _live_pending_finalize_uids_for_job(job)
    for uid in live_finalize_uids:
        if uid not in tracked_uids:
            tracked_uids.append(uid)

    row_by_uid = {}
    if tracked_uids:
        try:
            rows = project_manager.get_chunks_by_uids(tracked_uids)
        except Exception:
            rows = []
        row_by_uid = {
            str((row or {}).get("uid") or "").strip(): row
            for row in rows or []
            if str((row or {}).get("uid") or "").strip()
        }

    def _is_pending(uid):
        row = row_by_uid.get(uid)
        if row is None:
            return False
        return str((row or {}).get("status") or "").strip() not in {"done", "error"}

    live_finalize_lookup = set(live_finalize_uids)
    generation_pending = [
        uid for uid in _job_generation_pending_uids(job)
        if uid not in live_finalize_lookup and _is_pending(uid)
    ]
    pending_lookup = set(generation_pending) | live_finalize_lookup
    for uid in _job_pending_uids(job):
        if _is_pending(uid):
            pending_lookup.add(uid)

    job["generation_pending_uids"] = generation_pending
    job["pending_finalize_uids"] = [uid for uid in live_finalize_uids if _is_pending(uid)]
    job["pending_uids"] = [
        uid for uid in _job_uids(job)
        if uid in pending_lookup and _is_pending(uid)
    ]
    if bool(job.get("generation_finished")) and job["generation_pending_uids"]:
        job["generation_finished"] = False
    return job


def _normalize_restored_audio_job_runtime(raw_job, progress):
    progress_pending = [str(uid).strip() for uid in (progress or {}).get("pending_uids", []) if str(uid).strip()]
    live_finalize_uids = _live_pending_finalize_uids_for_job(raw_job, progress_pending)
    live_finalize_lookup = set(live_finalize_uids)

    raw_generation_pending = [
        uid for uid in _job_generation_pending_uids(raw_job)
        if uid in progress_pending and uid not in live_finalize_lookup
    ]

    if bool((raw_job or {}).get("generation_finished", False)):
        generation_pending_uids = [uid for uid in progress_pending if uid not in live_finalize_lookup]
    else:
        generation_pending_uids = list(raw_generation_pending)
        for uid in progress_pending:
            if uid not in live_finalize_lookup and uid not in generation_pending_uids:
                generation_pending_uids.append(uid)

    generation_finished = (
        bool((raw_job or {}).get("generation_finished", False))
        or bool(live_finalize_uids)
    ) and not generation_pending_uids
    run_token = str((raw_job or {}).get("run_token") or "").strip() or None
    if not live_finalize_uids:
        run_token = None

    return {
        "generation_pending_uids": generation_pending_uids,
        "pending_finalize_uids": live_finalize_uids,
        "generation_finished": generation_finished,
        "run_token": run_token,
    }


def _effective_parallel_workers(settings):
    configured_workers = max(1, int((settings or {}).get("workers", 1) or 1))
    tts_cfg = dict((settings or {}).get("tts_cfg") or {})
    if str(tts_cfg.get("mode") or "").strip().lower() != "local":
        return configured_workers

    backend_hint = str(tts_cfg.get("local_backend") or "").strip().lower()
    try:
        engine = project_manager.get_engine()
    except Exception:
        engine = None

    engine_mode = str(
        getattr(engine, "mode", None)
        or getattr(engine, "_mode", None)
        or tts_cfg.get("mode")
        or ""
    ).strip().lower()
    resolved_backend = None
    resolver = getattr(engine, "_resolve_local_backend", None)
    if callable(resolver):
        try:
            resolved_backend = resolver()
        except Exception:
            resolved_backend = None
    if resolved_backend is None:
        resolved_backend = getattr(engine, "local_backend", None) or backend_hint

    if engine_mode == "local" and str(resolved_backend or "").strip().lower() == "mlx":
        return 1
    return configured_workers


def _resolve_uid_ordinals(uids):
    rows = project_manager.get_chunks_by_uids(uids)
    return {
        str((row or {}).get("uid") or "").strip(): int((row or {}).get("id") or 0)
        for row in rows
        if str((row or {}).get("uid") or "").strip()
    }


def _serialize_audio_job(job):
    _reconcile_audio_job_runtime_locked(job)
    uids = _job_uids(job)
    pending_uids = _job_pending_uids(job)
    generation_pending_uids = _job_generation_pending_uids(job)
    pending_finalize_uids = _job_pending_finalize_uids(job)
    ordinals = _resolve_uid_ordinals(uids) if uids else {}
    return {
        "id": job["id"],
        "corr_id": job.get("corr_id"),
        "kind": job["kind"],
        "status": job["status"],
        "label": job["label"],
        "scope": job["scope"],
        "scope_mode": job.get("scope_mode"),
        "chapter": job.get("chapter"),
        "neutral_narrator": bool(job.get("neutral_narrator", False)),
        "uids": uids,
        "pending_uids": pending_uids,
        "generation_pending_uids": generation_pending_uids,
        "pending_finalize_uids": pending_finalize_uids,
        "indices": [ordinals[uid] for uid in uids if uid in ordinals],
        "total_chunks": job["total_chunks"],
        "total_words": job.get("total_words", 0),
        "remaining_words": job.get("remaining_words", 0),
        "processed_clips": job.get("processed_clips", 0),
        "error_clips": job.get("error_clips", 0),
        "generation_finished": bool(job.get("generation_finished", False)),
        "finalized_clips": int(job.get("finalized_clips", 0) or 0),
        "finalizer_failures": int(job.get("finalizer_failures", 0) or 0),
        "retry_uids": [str(uid).strip() for uid in (job.get("retry_uids") or []) if str(uid).strip()],
        "pending_indices": [ordinals[uid] for uid in pending_uids if uid in ordinals],
        "recovery_count": job.get("recovery_count", 0),
        "queued_at": job.get("queued_at"),
        "started_at": job.get("started_at"),
        "finished_at": job.get("finished_at"),
        "last_output_at": job.get("last_output_at"),
        "last_generation_activity_at": job.get("last_generation_activity_at"),
        "last_finalize_activity_at": job.get("last_finalize_activity_at"),
        "run_token": job.get("run_token"),
    }


def _serialize_audio_job_checkpoint(job):
    return {
        "id": job["id"],
        "corr_id": job.get("corr_id"),
        "kind": job["kind"],
        "status": job["status"],
        "label": job["label"],
        "scope": job["scope"],
        "scope_mode": job.get("scope_mode"),
        "chapter": job.get("chapter"),
        "neutral_narrator": bool(job.get("neutral_narrator", False)),
        "uids": _job_uids(job),
        "pending_uids": _job_pending_uids(job),
        "generation_pending_uids": _job_generation_pending_uids(job),
        "pending_finalize_uids": _job_pending_finalize_uids(job),
        "total_chunks": job["total_chunks"],
        "total_words": job.get("total_words", 0),
        "remaining_words": job.get("remaining_words", 0),
        "processed_clips": job.get("processed_clips", 0),
        "error_clips": job.get("error_clips", 0),
        "generation_finished": bool(job.get("generation_finished", False)),
        "finalized_clips": int(job.get("finalized_clips", 0) or 0),
        "finalizer_failures": int(job.get("finalizer_failures", 0) or 0),
        "retry_uids": [str(uid).strip() for uid in (job.get("retry_uids") or []) if str(uid).strip()],
        "recovery_count": job.get("recovery_count", 0),
        "queued_at": job.get("queued_at"),
        "started_at": job.get("started_at"),
        "finished_at": job.get("finished_at"),
        "last_output_at": job.get("last_output_at"),
        "last_generation_activity_at": job.get("last_generation_activity_at"),
        "last_finalize_activity_at": job.get("last_finalize_activity_at"),
        "run_token": job.get("run_token"),
    }


def _serialize_audio_job_summary(job):
    pending_uids = _job_pending_uids(job)
    generation_pending_uids = _job_generation_pending_uids(job)
    pending_finalize_uids = _job_pending_finalize_uids(job)
    retry_uids = [str(uid).strip() for uid in (job.get("retry_uids") or []) if str(uid).strip()]
    return {
        "id": job["id"],
        "corr_id": job.get("corr_id"),
        "kind": job["kind"],
        "status": job["status"],
        "label": job["label"],
        "scope": job["scope"],
        "scope_mode": job.get("scope_mode"),
        "chapter": job.get("chapter"),
        "neutral_narrator": bool(job.get("neutral_narrator", False)),
        "total_chunks": job["total_chunks"],
        "total_words": job.get("total_words", 0),
        "remaining_words": job.get("remaining_words", 0),
        "processed_clips": job.get("processed_clips", 0),
        "error_clips": job.get("error_clips", 0),
        "generation_finished": bool(job.get("generation_finished", False)),
        "finalized_clips": int(job.get("finalized_clips", 0) or 0),
        "finalizer_failures": int(job.get("finalizer_failures", 0) or 0),
        "recovery_count": job.get("recovery_count", 0),
        "pending_chunks": len(pending_uids),
        "generation_pending_chunks": len(generation_pending_uids),
        "pending_finalize_chunks": len(pending_finalize_uids),
        "retry_chunks": len(retry_uids),
        "queued_at": job.get("queued_at"),
        "started_at": job.get("started_at"),
        "finished_at": job.get("finished_at"),
        "last_output_at": job.get("last_output_at"),
        "last_generation_activity_at": job.get("last_generation_activity_at"),
        "last_finalize_activity_at": job.get("last_finalize_activity_at"),
        "run_token": job.get("run_token"),
    }


def _is_transient_windows_file_error(exc):
    if os.name != "nt":
        return False
    if getattr(exc, "winerror", None) in {5, 32}:
        return True
    text = str(exc)
    return "Access is denied" in text or "being used by another process" in text


def _atomic_json_write(path, data, max_retries=8):
    for attempt in range(max_retries):
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=os.path.dirname(path) or ".",
                prefix=f".{os.path.basename(path)}.",
                suffix=".tmp",
                delete=False,
            ) as f:
                tmp_path = f.name
                json.dump(data, f, indent=2, ensure_ascii=False)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, path)
            return
        except OSError as exc:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass
            if attempt < max_retries - 1 and _is_transient_windows_file_error(exc):
                time.sleep(0.05 * (2 ** attempt))
                continue
            raise


def _append_script_repair_trace(run_id: str, event_type: str, payload: Optional[dict] = None):
    record = {
        "ts": time.time(),
        "run_id": run_id,
        "event": event_type,
        "payload": payload or {},
    }
    with open(SCRIPT_REPAIR_TRACE_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _clear_script_repair_trace():
    if os.path.exists(SCRIPT_REPAIR_TRACE_PATH):
        os.remove(SCRIPT_REPAIR_TRACE_PATH)


def _new_processing_workflow_state():
    return {
        "running": False,
        "paused": False,
        "pause_requested": False,
        "current_stage": None,
        "completed_stages": [],
        "options": {"process_voices": True, "generate_audio": False},
        "logs": [],
        "last_error": None,
        "started_at": None,
        "updated_at": None,
        "completed_at": None,
        "resume_count": 0,
    }


def _has_retired_processing_pause_text(logs):
    for line in logs if isinstance(logs, list) else []:
        text = str(line or "").lower()
        if "hard-killed all active tasks" in text or "debug pause probe" in text:
            return True
    return False


def _sanitize_restored_processing_workflow_state(restored):
    """Normalize stale legacy workflow snapshots that should not be resumed/rendered."""
    current_stage = str(restored.get("current_stage") or "").strip()
    stale_retired_pause_snapshot = (
        bool(restored.get("paused"))
        and not bool(restored.get("running"))
        and not current_stage
        and _has_retired_processing_pause_text(restored.get("logs"))
    )
    if not stale_retired_pause_snapshot:
        return False

    options = restored.get("options")
    resume_count = int(restored.get("resume_count", 0) or 0)
    sanitized = _new_processing_workflow_state()
    if isinstance(options, dict):
        sanitized["options"] = options
    sanitized["resume_count"] = resume_count
    restored.clear()
    restored.update(sanitized)
    return True


def _load_project_state_payload():
    state_path = os.path.join(ROOT_DIR, "state.json")
    if not os.path.exists(state_path):
        return {}
    try:
        with open(state_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except (OSError, json.JSONDecodeError, ValueError):
        return {}
    if not isinstance(payload, dict):
        return {}
    for key in ("narrator_threshold", "narrator_overrides", "auto_narrator_aliases"):
        payload.pop(key, None)
    return payload


def _save_project_state_payload(state):
    payload = dict(state or {})
    for key in ("narrator_threshold", "narrator_overrides", "auto_narrator_aliases"):
        payload.pop(key, None)
    _atomic_json_write(os.path.join(ROOT_DIR, "state.json"), payload)


def _load_processing_stage_markers():
    state = _load_project_state_payload()
    markers = state.get(PROCESSING_STAGE_MARKERS_KEY)
    return dict(markers) if isinstance(markers, dict) else {}


def _load_new_mode_stage_markers():
    state = _load_project_state_payload()
    markers = state.get(NEW_MODE_STAGE_MARKERS_KEY)
    return dict(markers) if isinstance(markers, dict) else {}


def _save_processing_stage_markers(markers, state=None):
    payload = dict(state) if isinstance(state, dict) else _load_project_state_payload()
    cleaned = {stage: value for stage, value in dict(markers).items() if stage in PROCESSING_STAGE_ORDER}
    if cleaned:
        payload[PROCESSING_STAGE_MARKERS_KEY] = cleaned
    else:
        payload.pop(PROCESSING_STAGE_MARKERS_KEY, None)
    _save_project_state_payload(payload)


def _save_new_mode_stage_markers(markers, state=None):
    payload = dict(state) if isinstance(state, dict) else _load_project_state_payload()
    cleaned = {stage: value for stage, value in dict(markers).items() if stage in NEW_MODE_STAGE_ORDER}
    if cleaned:
        payload[NEW_MODE_STAGE_MARKERS_KEY] = cleaned
    else:
        payload.pop(NEW_MODE_STAGE_MARKERS_KEY, None)
    _save_project_state_payload(payload)


def _mark_processing_stage_completed_marker(stage_name):
    if stage_name not in PROCESSING_STAGE_ORDER:
        return
    state = _load_project_state_payload()
    markers = state.get(PROCESSING_STAGE_MARKERS_KEY)
    markers = dict(markers) if isinstance(markers, dict) else {}
    markers[stage_name] = {
        "completed_at": time.time(),
    }
    _save_processing_stage_markers(markers, state=state)


def _mark_new_mode_stage_completed_marker(stage_name):
    if stage_name not in NEW_MODE_STAGE_ORDER:
        return
    state = _load_project_state_payload()
    markers = state.get(NEW_MODE_STAGE_MARKERS_KEY)
    markers = dict(markers) if isinstance(markers, dict) else {}
    markers[stage_name] = {
        "completed_at": time.time(),
    }
    _save_new_mode_stage_markers(markers, state=state)


def _clear_processing_stage_markers(stage_names=None):
    state = _load_project_state_payload()
    markers = state.get(PROCESSING_STAGE_MARKERS_KEY)
    markers = dict(markers) if isinstance(markers, dict) else {}
    if stage_names is None:
        changed = bool(markers)
        markers = {}
    else:
        changed = False
        for stage_name in stage_names:
            if stage_name in markers:
                markers.pop(stage_name, None)
                changed = True
    if changed:
        _save_processing_stage_markers(markers, state=state)


def _clear_new_mode_stage_markers(stage_names=None):
    state = _load_project_state_payload()
    markers = state.get(NEW_MODE_STAGE_MARKERS_KEY)
    markers = dict(markers) if isinstance(markers, dict) else {}
    if stage_names is None:
        changed = bool(markers)
        markers = {}
    else:
        changed = False
        for stage_name in stage_names:
            if stage_name in markers:
                markers.pop(stage_name, None)
                changed = True
    if changed:
        _save_new_mode_stage_markers(markers, state=state)


def _clear_processing_stage_and_downstream(stage_name, include_self=True):
    if stage_name not in PROCESSING_STAGE_ORDER:
        return
    start_index = PROCESSING_STAGE_ORDER.index(stage_name) + (0 if include_self else 1)
    _clear_processing_stage_markers(PROCESSING_STAGE_ORDER[start_index:])


def _clear_new_mode_stage_and_downstream(stage_name, include_self=True):
    if stage_name not in NEW_MODE_STAGE_ORDER:
        return
    start_index = NEW_MODE_STAGE_ORDER.index(stage_name) + (0 if include_self else 1)
    _clear_new_mode_stage_markers(NEW_MODE_STAGE_ORDER[start_index:])


def _derived_processing_completed_stages(options=None):
    options = options or {}
    allowed = set(_processing_workflow_stage_sequence(options))
    markers = _load_processing_stage_markers()
    completed = []

    for stage_name in PROCESSING_STAGE_ORDER:
        if stage_name not in allowed:
            continue
        if stage_name == "script":
            if markers.get(stage_name) or _project_has_script_document():
                completed.append(stage_name)
            continue
        if stage_name == "review" and markers.get(stage_name):
            completed.append(stage_name)
            continue
        if stage_name in ("sanity", "repair") and markers.get(stage_name) and _project_has_script_document() and _load_project_script_sanity_result():
            completed.append(stage_name)
            continue
        if stage_name == "voices" and markers.get(stage_name) and getattr(project_manager, "has_voice_config", lambda: False)():
            completed.append(stage_name)
            continue
        if stage_name == "audio" and markers.get(stage_name) and bool(_load_project_chunks_snapshot()):
            completed.append(stage_name)

    return completed


def _chunk_chapter_summary():
    try:
        chunks = _load_project_chunks_snapshot()
        ordered_chapters = []
        last_seen = None
        for chunk in chunks:
            chapter = str((chunk or {}).get("chapter") or "").strip()
            if not chapter:
                continue
            if chapter != last_seen:
                ordered_chapters.append(chapter)
                last_seen = chapter
        return {
            "chunk_count": len(chunks),
            "chapter_count": len(ordered_chapters),
            "last_chapter": ordered_chapters[-1] if ordered_chapters else None,
        }
    except Exception:
        return {
            "chunk_count": 0,
            "chapter_count": 0,
        "last_chapter": None,
        }


def _load_project_chunks_snapshot():
    if hasattr(project_manager, "load_chunks"):
        try:
            chunks = project_manager.load_chunks()
            if isinstance(chunks, list):
                return chunks
        except Exception:
            pass
    return []


def _script_ingestion_preflight_summary():
    state = _load_project_state_payload()
    input_path = (state.get("input_file_path") or "").strip()
    if not input_path or not os.path.exists(input_path):
        return {
            "warn": False,
            "reason": "no_input",
            "message": "",
        }

    chunk_summary = _chunk_chapter_summary()
    if chunk_summary["chunk_count"] <= 0:
        return {
            "warn": False,
            "reason": "no_chunks",
            "message": "",
            **chunk_summary,
        }

    try:
        source_document = load_source_document(input_path)
    except Exception as e:
        logger.warning("Script ingestion preflight could not inspect source document: %s", e)
        return {
            "warn": False,
            "reason": "source_inspection_failed",
            "message": "",
            **chunk_summary,
        }

    chapters = source_document.get("chapters") or []
    last_source_chapter = str((chapters[-1].get("title") if chapters else None) or "").strip() or None
    source_type = source_document.get("type")
    source_chapter_count = len(chapters)
    last_chunk_chapter = chunk_summary["last_chapter"]
    matches_existing = (
        source_type == "epub"
        and bool(last_source_chapter)
        and bool(last_chunk_chapter)
        and last_source_chapter == last_chunk_chapter
    )

    message = ""
    if matches_existing:
        message = (
            "This project already appears to contain a full import of the current EPUB. "
            f"The last existing chapter is \"{last_chunk_chapter}\" and the last EPUB chapter is also "
            f"\"{last_source_chapter}\". Re-importing will delete the current generated project state and "
            "rebuild it from the source document."
        )

    return {
        "warn": matches_existing,
        "reason": "matching_last_chapter" if matches_existing else "no_match",
        "message": message,
        "source_type": source_type,
        "source_chapter_count": source_chapter_count,
        "last_source_chapter": last_source_chapter,
        **chunk_summary,
    }


def _mark_script_stage_skipped_for_existing_project():
    _mark_processing_stage_completed_marker("script")
    return {"status": "skipped", "skipped_stage": "script"}


def _clear_directory_contents(directory):
    if not os.path.isdir(directory):
        os.makedirs(directory, exist_ok=True)
        return
    for entry in os.listdir(directory):
        entry_path = os.path.join(directory, entry)
        if os.path.isdir(entry_path):
            shutil.rmtree(entry_path)
        else:
            _remove_file_with_retries(entry_path)


def _remove_file_with_retries(path, *, retries=100, delay_seconds=0.1):
    """
    Remove a file with bounded retries for transient lock contention.

    Windows can briefly hold SQLite/media file handles after worker teardown.
    Retries are effectively no-op on normal paths and keep Linux/macOS behavior unchanged.
    """
    normalized = os.path.abspath(str(path or ""))
    last_error = None
    for attempt in range(max(1, int(retries))):
        try:
            os.remove(normalized)
            return True
        except FileNotFoundError:
            return False
        except PermissionError as exc:
            last_error = exc
        except OSError as exc:
            # Retry only common access/busy-denied cases; re-raise everything else.
            if exc.errno not in (errno.EACCES, errno.EBUSY, errno.EPERM):
                raise
            last_error = exc
        if attempt >= retries - 1:
            break
        gc.collect()
        time.sleep(delay_seconds)
    if last_error is not None:
        raise last_error
    return False


def _shutdown_live_project_managers_for_db_path(db_path):
    target_db = os.path.abspath(str(db_path or ""))
    if not target_db:
        return
    for obj in gc.get_objects():
        try:
            if not isinstance(obj, ProjectManager):
                continue
            candidate = os.path.abspath(str(getattr(obj, "chunks_db_path", "") or ""))
            if candidate != target_db:
                continue
            try:
                obj.shutdown_script_store(flush=True)
            except Exception:
                pass
        except Exception:
            continue


def _clear_project_derived_state(preserve_input_file=True, preserve_reusable_voices=True):
    _assert_test_safe_runtime_target(
        "_clear_project_derived_state",
        ROOT_DIR=ROOT_DIR,
        VOICELINES_DIR=VOICELINES_DIR,
        UPLOADS_DIR=UPLOADS_DIR,
        CLONE_VOICES_DIR=CLONE_VOICES_DIR,
        DESIGNED_VOICES_DIR=DESIGNED_VOICES_DIR,
    )
    state = _load_project_state_payload()
    input_file_path = (state.get("input_file_path") or "").strip()
    _shutdown_media_static_server()
    manager_supports_shutdown = hasattr(project_manager, "shutdown_script_store")
    if manager_supports_shutdown:
        try:
            project_manager.shutdown_script_store(flush=True)
        except Exception:
            pass
    manager_root = os.path.abspath(getattr(project_manager, "root_dir", ROOT_DIR))
    current_root = os.path.abspath(ROOT_DIR)
    if manager_root == current_root:
        chunks_db_path = project_manager.chunks_db_path
        chunks_queue_log_path = project_manager.chunks_queue_log_path
    else:
        chunks_db_path = getattr(project_manager, "chunks_db_path", os.path.join(ROOT_DIR, "chunks.sqlite3"))
        chunks_queue_log_path = getattr(project_manager, "chunks_queue_log_path", os.path.join(ROOT_DIR, "chunks.queue.log"))
    _shutdown_live_project_managers_for_db_path(chunks_db_path)

    files_to_remove = [
        f"{chunks_db_path}-wal",
        f"{chunks_db_path}-shm",
        chunks_queue_log_path,
        _project_repair_trace_path(),
        AUDIOBOOK_PATH,
        M4B_PATH,
        _project_workflow_state_path("audio_queue_state.json"),
        _project_workflow_state_path("audio_cancel_tombstone.json"),
        _project_workflow_state_path("processing_workflow_state.json"),
        _project_workflow_state_path("new_mode_workflow_state.json"),
        _project_workflow_state_path("script_generation_checkpoint.json"),
        _project_workflow_state_path("script_review_checkpoint.json"),
        LAYOUT.audacity_export_path,
        LAYOUT.m4b_cover_path,
        os.path.join(ROOT_DIR, "logs", "llm_responses.log"),
        os.path.join(ROOT_DIR, "logs", "review_responses.log"),
    ]
    if manager_supports_shutdown:
        files_to_remove.insert(0, chunks_db_path)
    for path in files_to_remove:
        if os.path.exists(path):
            _remove_file_with_retries(path)

    _clear_directory_contents(VOICELINES_DIR)
    _clear_directory_contents(os.path.join(LAYOUT.exports_dir, "_wip"))
    if not preserve_reusable_voices:
        _clear_directory_contents(DESIGNED_VOICES_DIR)
        _clear_directory_contents(CLONE_VOICES_DIR)
    if not preserve_input_file:
        _clear_directory_contents(UPLOADS_DIR)

    new_state = {}
    if preserve_input_file and input_file_path:
        new_state["input_file_path"] = input_file_path
    new_state["render_prep_complete"] = False
    _save_project_state_payload(new_state)

    with project_manager._transcription_cache_lock:
        project_manager._transcription_cache = None
    project_manager.engine = None
    project_manager.asr_engine = None
    project_manager.reload_script_store()


def _persist_processing_workflow_state_locked():
    _atomic_json_write(PROCESSING_WORKFLOW_STATE_PATH, process_state["processing_workflow"])


def _refresh_processing_workflow_updated_at_locked():
    process_state["processing_workflow"]["updated_at"] = time.time()


def _append_processing_workflow_log_locked(message):
    process_state["processing_workflow"]["logs"].append(message)
    _trim_logs(process_state["processing_workflow"]["logs"])
    _refresh_processing_workflow_updated_at_locked()
    _persist_processing_workflow_state_locked()


def _set_processing_workflow_state_locked(**updates):
    process_state["processing_workflow"].update(updates)
    _refresh_processing_workflow_updated_at_locked()
    _persist_processing_workflow_state_locked()


def _processing_workflow_stage_sequence(options=None):
    options = options or process_state["processing_workflow"].get("options") or {}
    stages = ["script", "review", "sanity", "repair"]
    if options.get("process_voices", True):
        stages.append("voices")
    if options.get("generate_audio", False):
        stages.append("audio")
    return stages


def _processing_workflow_is_pause_requested():
    with processing_workflow_lock:
        return bool(process_state["processing_workflow"].get("pause_requested"))


def _ensure_processing_workflow_not_paused():
    if _processing_workflow_is_pause_requested():
        raise WorkflowPauseRequested()


def _mark_processing_workflow_stage_complete(stage_name):
    _mark_processing_stage_completed_marker(stage_name)
    with processing_workflow_lock:
        completed = list(process_state["processing_workflow"].get("completed_stages") or [])
        if stage_name not in completed:
            completed.append(stage_name)
        process_state["processing_workflow"]["completed_stages"] = completed
        process_state["processing_workflow"]["current_stage"] = stage_name
        _append_processing_workflow_log_locked(
            f"{PROCESSING_WORKFLOW_STAGE_LABELS.get(stage_name, stage_name)} completed."
        )


def _set_processing_workflow_paused(stage_name=None):
    with processing_workflow_lock:
        process_state["processing_workflow"]["running"] = False
        process_state["processing_workflow"]["paused"] = True
        process_state["processing_workflow"]["pause_requested"] = False
        if stage_name is not None:
            process_state["processing_workflow"]["current_stage"] = stage_name
        _append_processing_workflow_log_locked("Processing paused. Resume to continue from the current stage.")


def _set_processing_workflow_failed(stage_name, message):
    with processing_workflow_lock:
        process_state["processing_workflow"]["running"] = False
        process_state["processing_workflow"]["paused"] = False
        process_state["processing_workflow"]["pause_requested"] = False
        process_state["processing_workflow"]["current_stage"] = stage_name
        process_state["processing_workflow"]["last_error"] = message
        _append_processing_workflow_log_locked(message)


def _set_processing_workflow_completed():
    with processing_workflow_lock:
        process_state["processing_workflow"]["running"] = False
        process_state["processing_workflow"]["paused"] = False
        process_state["processing_workflow"]["pause_requested"] = False
        process_state["processing_workflow"]["current_stage"] = None
        process_state["processing_workflow"]["last_error"] = None
        process_state["processing_workflow"]["completed_at"] = time.time()
        _append_processing_workflow_log_locked("Processing workflow completed successfully.")


def _request_processing_workflow_pause_locked():
    state = process_state["processing_workflow"]
    if not state.get("running"):
        return False
    if state.get("pause_requested"):
        return False
    state["pause_requested"] = True
    _append_processing_workflow_log_locked("Pause requested. Waiting for the current stage to stop safely.")
    return True


def _terminate_task_process_if_running(task_name):
    with task_state_lock:
        process = task_processes.get(task_name)
    if process is None:
        return False
    if process.poll() is not None:
        return False
    try:
        process.terminate()
        return True
    except Exception:
        return False


def _pause_audio_queue_for_workflow():
    global audio_recovery_request
    with audio_queue_condition:
        cleared = len(audio_queue)
        now = time.time()
        while audio_queue:
            job = audio_queue.pop(0)
            job["status"] = "cancelled"
            job["finished_at"] = now
            _record_audio_recent_job_locked(job)

        if audio_current_job is not None:
            process_state["audio"]["cancel"] = True
            audio_cancel_event.set()
            audio_recovery_request = None
            _append_audio_log_locked(f"[WORKFLOW] Pause requested for audio job #{audio_current_job['id']}")
            if cleared:
                _append_audio_log_locked(f"[WORKFLOW] Removed {cleared} queued audio job(s)")

            # Force-abandon the active job immediately so workflow cancellation
            # does not leave stale running state behind.
            abandoned = _abandon_audio_job_locked(
                audio_current_job,
                audio_current_job.get("run_token"),
                "Workflow pause requested",
                status="cancelled",
            )
            if abandoned:
                return True

            # Fallback to cooperative cancellation when a race prevents immediate abandon.
            _refresh_audio_process_state_locked(persist=True)
            return True

        if cleared:
            audio_recovery_request = None
            _append_audio_log_locked(f"[WORKFLOW] Removed {cleared} queued audio job(s)")
            _refresh_audio_process_state_locked(persist=True)
            return True
    return False


def _request_active_stage_pause(stage_name):
    if stage_name in ("script", "review"):
        return _terminate_task_process_if_running(stage_name)
    if stage_name == "audio":
        return _pause_audio_queue_for_workflow()
    return False


def _normalize_archive_path(path: str) -> str:
    normalized = (path or "").replace("\\", "/").strip("/")
    if not normalized:
        return ""
    parts = [part for part in normalized.split("/") if part and part != "."]
    if any(part == ".." for part in parts):
        raise ValueError(f"Unsafe archive path: {path}")
    return "/".join(parts)


def _archive_relative_file_target(path: str) -> str:
    normalized = _normalize_archive_path(path)
    return PROJECT_ARCHIVE_LEGACY_FILE_ALIASES.get(normalized, normalized)


def _project_archive_filesystem_path(path: str) -> str:
    normalized = _archive_relative_file_target(path)
    if normalized == "db/chunks.sqlite3":
        return getattr(project_manager, "chunks_db_path", os.path.join(ROOT_DIR, "chunks.sqlite3"))
    if os.path.abspath(ROOT_DIR) == os.path.abspath(LAYOUT.project_dir):
        return os.path.join(ROOT_DIR, normalized)
    reverse_aliases = {target: legacy for legacy, target in PROJECT_ARCHIVE_LEGACY_FILE_ALIASES.items()}
    legacy_relative = reverse_aliases.get(normalized, normalized)
    return os.path.join(ROOT_DIR, legacy_relative)


def _project_archive_source_path(extracted_dir: str, path: str) -> str:
    normalized = _archive_relative_file_target(path)
    source_path = os.path.join(extracted_dir, normalized)
    if os.path.exists(source_path):
        return source_path
    reverse_aliases = {target: legacy for legacy, target in PROJECT_ARCHIVE_LEGACY_FILE_ALIASES.items()}
    legacy_relative = reverse_aliases.get(normalized)
    if legacy_relative:
        legacy_path = os.path.join(extracted_dir, legacy_relative)
        if os.path.exists(legacy_path):
            return legacy_path
    return source_path


def _is_allowed_project_archive_path(path: str) -> bool:
    normalized = _archive_relative_file_target(path)
    if not normalized or normalized == PROJECT_ARCHIVE_MANIFEST_NAME:
        return True
    if normalized in PROJECT_ARCHIVE_ALLOWED_FILES:
        return True
    first = normalized.split("/", 1)[0]
    return first in PROJECT_ARCHIVE_ALLOWED_DIRS


def _project_export_filesystem_path(filename: str) -> str:
    if os.path.abspath(ROOT_DIR) == os.path.abspath(LAYOUT.project_dir):
        return os.path.join(LAYOUT.exports_dir, filename)
    return os.path.join(ROOT_DIR, filename)


def _archive_state_with_relative_paths():
    state = {}
    state_path = os.path.join(ROOT_DIR, "state.json")
    if os.path.exists(state_path):
        with open(state_path, "r", encoding="utf-8") as f:
            try:
                state = json.load(f)
            except (json.JSONDecodeError, ValueError):
                state = {}

    input_file_path = state.get("input_file_path")
    if input_file_path:
        try:
            absolute_input = os.path.abspath(input_file_path)
            if os.path.commonpath([absolute_input, os.path.abspath(UPLOADS_DIR)]) == os.path.abspath(UPLOADS_DIR):
                state["input_file_path"] = os.path.relpath(absolute_input, ROOT_DIR).replace(os.sep, "/")
            else:
                state.pop("input_file_path", None)
        except ValueError:
            state.pop("input_file_path", None)
    return state


def _ensure_voice_compat_exports():
    return None


def _project_archive_metadata():
    summary = {}
    get_summary = getattr(project_manager, "get_chunk_chapter_summary", None)
    if callable(get_summary):
        try:
            summary = dict(get_summary() or {})
        except Exception:
            summary = {}
    if not summary:
        summary = _chunk_chapter_summary()

    has_audio_fn = getattr(project_manager, "has_generated_chunk_audio", None)
    if callable(has_audio_fn):
        try:
            has_audio = bool(has_audio_fn())
        except Exception:
            has_audio = _project_has_generated_audio()
    else:
        has_audio = _project_has_generated_audio()

    has_voice_fn = getattr(project_manager, "has_voice_config", None)
    if callable(has_voice_fn):
        try:
            has_voice_config = bool(has_voice_fn())
        except Exception:
            has_voice_config = False
    else:
        has_voice_config = False

    return {
        "kind": "project",
        "has_audio": has_audio,
        "has_voice_config": has_voice_config,
        "chunk_count": int(summary.get("chunk_count") or 0),
        "chapter_count": int(summary.get("chapter_count") or 0),
        "last_chapter": summary.get("last_chapter"),
    }


def _load_project_archive_manifest(zip_path: str):
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            if PROJECT_ARCHIVE_MANIFEST_NAME not in zf.namelist():
                return None
            with zf.open(PROJECT_ARCHIVE_MANIFEST_NAME) as manifest_file:
                payload = json.load(manifest_file)
    except (OSError, zipfile.BadZipFile, json.JSONDecodeError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def _copy_sqlite_database_snapshot(source_path: str, target_path: str):
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    with sqlite3.connect(source_path, timeout=30.0) as source_conn:
        with sqlite3.connect(target_path, timeout=30.0) as target_conn:
            source_conn.backup(target_conn)


def _prepare_project_archive_sqlite_snapshot():
    db_path = getattr(project_manager, "chunks_db_path", os.path.join(ROOT_DIR, "chunks.sqlite3"))
    if not db_path or not os.path.exists(db_path):
        return None

    script_store = getattr(project_manager, "script_store", None)
    manager_root = os.path.abspath(getattr(project_manager, "root_dir", ROOT_DIR))
    current_root = os.path.abspath(ROOT_DIR)
    if manager_root == current_root and script_store is not None:
        try:
            script_store.flush(timeout=5.0)
        except Exception:
            pass

    temp_dir = _make_runtime_temp_dir("threadspeak_archive_db_")
    backup_path = os.path.join(temp_dir, "chunks.sqlite3")
    try:
        _copy_sqlite_database_snapshot(db_path, backup_path)
    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise
    return temp_dir, backup_path


def _project_archive_entries():
    entries = {}

    def add_relative_path(relative_path: str):
        normalized = _archive_relative_file_target(relative_path)
        if not normalized or not _is_allowed_project_archive_path(normalized):
            return
        absolute_path = _project_archive_filesystem_path(normalized)
        if os.path.exists(absolute_path):
            entries[normalized] = absolute_path

    for relative_path in sorted(PROJECT_ARCHIVE_DURABLE_FILES - {"db/chunks.sqlite3"}):
        add_relative_path(relative_path)

    state = _archive_state_with_relative_paths()
    input_file_path = (state.get("input_file_path") or "").strip()
    if input_file_path:
        add_relative_path(input_file_path)

    chunks = _load_project_chunks_snapshot()

    for chunk in chunks:
        audio_path = (chunk.get("audio_path") or "").strip()
        if audio_path:
            add_relative_path(audio_path)

    voice_assets = set()
    for manifest_path in (CLONE_VOICES_MANIFEST, DESIGNED_VOICES_MANIFEST):
        if os.path.exists(manifest_path):
            relative_manifest = os.path.relpath(manifest_path, ROOT_DIR).replace(os.sep, "/")
            add_relative_path(relative_manifest)
            for entry in _load_manifest(manifest_path):
                filename = (entry.get("filename") or "").strip()
                if filename:
                    voice_assets.add(f"{os.path.dirname(relative_manifest).replace(os.sep, '/')}/{filename}")

    voice_config = {}
    load_voice_config = getattr(project_manager, "_load_voice_config", None)
    if callable(load_voice_config):
        try:
            payload = load_voice_config()
            if isinstance(payload, dict):
                voice_config = payload
        except Exception:
            voice_config = {}
    for config in voice_config.values():
        if not isinstance(config, dict):
            continue
        ref_audio = (config.get("ref_audio") or "").strip()
        if ref_audio:
            voice_assets.add(ref_audio)

    for relative_path in sorted(voice_assets):
        add_relative_path(relative_path)

    return sorted(entries.items())


def _build_project_archive_manifest(entries):
    return {
        "kind": "threadspeak_project_archive",
        "version": PROJECT_ARCHIVE_VERSION,
        "created_at": time.time(),
        "metadata": _project_archive_metadata(),
        "entries": [relative_path for relative_path, _ in entries],
    }


def _saved_project_archive_path(name: str) -> str:
    safe_name = _sanitize_name(name)
    if not safe_name:
        raise ValueError("Invalid project name.")
    os.makedirs(SAVED_PROJECTS_DIR, exist_ok=True)
    return os.path.join(SAVED_PROJECTS_DIR, f"{safe_name}.zip")


def _delete_saved_project_archive(name: str) -> bool:
    try:
        archive_path = _saved_project_archive_path(name)
    except ValueError:
        return False
    if os.path.exists(archive_path):
        os.remove(archive_path)
        return True
    return False


def _project_has_generated_audio() -> bool:
    try:
        chunks = _load_project_chunks_snapshot()
    except Exception:
        return False

    root = os.path.abspath(ROOT_DIR)
    for chunk in chunks:
        if not isinstance(chunk, dict):
            continue
        audio_path = (chunk.get("audio_path") or "").strip()
        if not audio_path:
            continue
        try:
            normalized = _normalize_archive_path(audio_path)
        except ValueError:
            continue
        full_path = os.path.abspath(os.path.join(ROOT_DIR, normalized))
        try:
            if os.path.commonpath([root, full_path]) != root:
                continue
        except ValueError:
            continue
        if os.path.isfile(full_path):
            return True
    return False


def _write_project_archive(zip_path: str):
    sqlite_snapshot = None
    try:
        entries = dict(_project_archive_entries())
        sqlite_snapshot = _prepare_project_archive_sqlite_snapshot()
        if sqlite_snapshot is not None:
            _, sqlite_backup_path = sqlite_snapshot
            entries["db/chunks.sqlite3"] = sqlite_backup_path
        sorted_entries = sorted(entries.items())
        manifest = _build_project_archive_manifest(sorted_entries)
    except Exception:
        if sqlite_snapshot is not None:
            shutil.rmtree(sqlite_snapshot[0], ignore_errors=True)
        raise
    os.makedirs(os.path.dirname(zip_path), exist_ok=True)
    temp_zip_path = f"{zip_path}.tmp"
    if os.path.exists(temp_zip_path):
        os.remove(temp_zip_path)

    try:
        with zipfile.ZipFile(temp_zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr(PROJECT_ARCHIVE_MANIFEST_NAME, json.dumps(manifest, indent=2, ensure_ascii=False))
            for relative_path, absolute_path in sorted_entries:
                if relative_path == "state.json":
                    zf.writestr(relative_path, json.dumps(_archive_state_with_relative_paths(), indent=2, ensure_ascii=False))
                else:
                    zf.write(absolute_path, arcname=relative_path)
        os.replace(temp_zip_path, zip_path)
    finally:
        if os.path.exists(temp_zip_path):
            os.remove(temp_zip_path)
        if sqlite_snapshot is not None:
            shutil.rmtree(sqlite_snapshot[0], ignore_errors=True)
    if hasattr(project_manager, "log_voice_audit_event"):
        project_manager.log_voice_audit_event(
            "project_archive_write",
            reason="write_project_archive",
            archive_path=zip_path,
        )
    return {"entries": manifest["entries"], "path": zip_path}


def _clear_project_archive_targets():
    _assert_test_safe_runtime_target(
        "_clear_project_archive_targets",
        ROOT_DIR=ROOT_DIR,
        VOICELINES_DIR=VOICELINES_DIR,
        UPLOADS_DIR=UPLOADS_DIR,
        SCRIPTS_DIR=SCRIPTS_DIR,
        SAVED_PROJECTS_DIR=SAVED_PROJECTS_DIR,
        CLONE_VOICES_DIR=CLONE_VOICES_DIR,
        DESIGNED_VOICES_DIR=DESIGNED_VOICES_DIR,
    )
    _shutdown_media_static_server()
    if hasattr(project_manager, "shutdown_script_store"):
        try:
            project_manager.shutdown_script_store(flush=True)
        except Exception:
            pass

    logs_dir = os.path.join(ROOT_DIR, "logs")
    removable_files = [
        getattr(project_manager, "chunks_db_path", os.path.join(ROOT_DIR, "chunks.sqlite3")),
        f"{getattr(project_manager, 'chunks_db_path', os.path.join(ROOT_DIR, 'chunks.sqlite3'))}-wal",
        f"{getattr(project_manager, 'chunks_db_path', os.path.join(ROOT_DIR, 'chunks.sqlite3'))}-shm",
        getattr(project_manager, "chunks_queue_log_path", os.path.join(ROOT_DIR, "chunks.queue.log")),
        os.path.join(ROOT_DIR, "state.json"),
        AUDIOBOOK_PATH,
        OPTIMIZED_EXPORT_PATH,
        _project_export_filesystem_path("audacity_export.zip"),
        M4B_PATH,
        _project_export_filesystem_path("m4b_cover.jpg"),
        _project_workflow_state_path("audio_queue_state.json"),
        _project_workflow_state_path("audio_cancel_tombstone.json"),
        _project_workflow_state_path("processing_workflow_state.json"),
        _project_workflow_state_path("new_mode_workflow_state.json"),
        _project_workflow_state_path("script_generation_checkpoint.json"),
        _project_workflow_state_path("script_review_checkpoint.json"),
        _project_repair_trace_path(),
        os.path.join(logs_dir, "llm_responses.log"),
        os.path.join(logs_dir, "review_responses.log"),
    ]
    removable_dirs = [UPLOADS_DIR, VOICELINES_DIR]
    primary_db_path = getattr(project_manager, "chunks_db_path", os.path.join(ROOT_DIR, "chunks.sqlite3"))
    _shutdown_live_project_managers_for_db_path(primary_db_path)

    for absolute_path in removable_files:
        if os.path.exists(absolute_path):
            _remove_file_with_retries(absolute_path)

    for absolute_dir in removable_dirs:
        if os.path.isdir(absolute_dir):
            shutil.rmtree(absolute_dir)
        os.makedirs(absolute_dir, exist_ok=True)


def _archive_stage_marker(stage_name, completed_at):
    return {
        stage_name: {
            "completed_at": float(completed_at or time.time()),
        }
    }


def _upsert_manifest_entries(existing_entries, incoming_entries):
    merged = []
    index_by_key = {}

    def entry_key(entry):
        if not isinstance(entry, dict):
            return None
        for field in ("id", "filename", "name", "speaker"):
            value = str(entry.get(field) or "").strip()
            if value:
                return f"{field}:{value}"
        return None

    for entry in list(existing_entries or []):
        if not isinstance(entry, dict):
            continue
        key = entry_key(entry)
        if key is None or key not in index_by_key:
            index_by_key[key] = len(merged)
            merged.append(dict(entry))

    for entry in list(incoming_entries or []):
        if not isinstance(entry, dict):
            continue
        key = entry_key(entry)
        if key is not None and key in index_by_key:
            merged[index_by_key[key]] = dict(entry)
            continue
        index_by_key[key] = len(merged)
        merged.append(dict(entry))
    return merged


def _merge_reusable_voice_library(extracted_dir: str, dirname: str):
    source_dir = os.path.join(extracted_dir, dirname)
    target_dir = os.path.join(ROOT_DIR, dirname)
    if not os.path.isdir(source_dir):
        return

    os.makedirs(target_dir, exist_ok=True)
    for current_root, _, filenames in os.walk(source_dir):
        for filename in filenames:
            if filename == "manifest.json":
                continue
            source_path = os.path.join(current_root, filename)
            relative_path = os.path.relpath(source_path, source_dir)
            target_path = os.path.join(target_dir, relative_path)
            os.makedirs(os.path.dirname(target_path), exist_ok=True)
            shutil.copy2(source_path, target_path)

    source_manifest_path = os.path.join(source_dir, "manifest.json")
    target_manifest_path = os.path.join(target_dir, "manifest.json")
    if not os.path.exists(source_manifest_path):
        return
    merged_manifest = _upsert_manifest_entries(
        _load_manifest(target_manifest_path),
        _load_manifest(source_manifest_path),
    )
    _save_manifest(target_manifest_path, merged_manifest)


def _normalize_restored_project_state(restored_state=None, *, loaded_project_name: str = ""):
    payload = dict(restored_state or {})
    now = time.time()

    input_file_path = str(payload.get("input_file_path") or "").strip()
    if input_file_path and not os.path.exists(input_file_path):
        input_file_path = ""
    if input_file_path:
        input_file_path = os.path.normpath(input_file_path)

    normalized_project_name = str(loaded_project_name or payload.get("loaded_project_name") or "").strip()
    normalized_script_name = str(payload.get("loaded_script_name") or "").strip()
    if not normalized_script_name:
        if input_file_path:
            normalized_script_name = os.path.splitext(os.path.basename(input_file_path))[0].strip()
        elif normalized_project_name:
            normalized_script_name = normalized_project_name

    has_script = _project_has_script_document()
    paragraphs_doc = _load_project_paragraphs_document()
    has_paragraphs = bool((paragraphs_doc or {}).get("paragraphs"))
    has_voice_config = bool(getattr(project_manager, "has_voice_config", lambda: False)())
    has_audio = _project_has_generated_audio()
    has_exports = _project_has_export_outputs()

    process_markers = {}
    if has_script:
        process_markers.update(_archive_stage_marker("script", now))
    if has_voice_config:
        process_markers.update(_archive_stage_marker("voices", now))
    if has_audio or has_exports:
        process_markers.update(_archive_stage_marker("audio", now))

    new_mode_markers = {}
    if has_script:
        for stage_name in ("process_paragraphs", "assign_dialogue", "extract_temperament", "create_script"):
            new_mode_markers.update(_archive_stage_marker(stage_name, now))
    else:
        if has_paragraphs:
            new_mode_markers.update(_archive_stage_marker("process_paragraphs", now))
        if (paragraphs_doc or {}).get("dialogue_assignment_complete"):
            new_mode_markers.update(_archive_stage_marker("assign_dialogue", now))
        if (paragraphs_doc or {}).get("temperament_extraction_complete"):
            new_mode_markers.update(_archive_stage_marker("extract_temperament", now))
    if has_voice_config:
        new_mode_markers.update(_archive_stage_marker("process_voices", now))
    if has_audio or has_exports:
        new_mode_markers.update(_archive_stage_marker("render_audio", now))

    if input_file_path:
        payload["input_file_path"] = input_file_path
    else:
        payload.pop("input_file_path", None)
    if normalized_project_name:
        payload["loaded_project_name"] = normalized_project_name
    else:
        payload.pop("loaded_project_name", None)
    if normalized_script_name:
        payload["loaded_script_name"] = normalized_script_name
    else:
        payload.pop("loaded_script_name", None)

    payload["render_prep_complete"] = bool(has_audio or has_exports)
    if process_markers:
        payload[PROCESSING_STAGE_MARKERS_KEY] = process_markers
    else:
        payload.pop(PROCESSING_STAGE_MARKERS_KEY, None)
    if new_mode_markers:
        payload[NEW_MODE_STAGE_MARKERS_KEY] = new_mode_markers
    else:
        payload.pop(NEW_MODE_STAGE_MARKERS_KEY, None)
    return payload


def _reset_runtime_state_after_project_load():
    global audio_current_job, audio_recovery_request
    with audio_queue_condition:
        audio_queue.clear()
        audio_current_job = None
        audio_recovery_request = None
        process_state["audio"]["cancel"] = False
        audio_cancel_event.clear()
        process_state["audio"]["queue"] = []
        process_state["audio"]["current_job"] = None
        process_state["audio"]["recent_jobs"] = []
        process_state["audio"]["logs"] = []
        process_state["audio"]["running"] = False
        process_state["audio"]["merge_running"] = False
        process_state["audio"]["merge_progress"] = _new_audio_merge_progress()
        process_state["audio"]["metrics"] = _new_audio_metrics()
        process_state["audio"]["heartbeat"] = _new_audio_heartbeat_state()
        _refresh_audio_process_state_locked(persist=False)

    with processing_workflow_lock:
        process_state["processing_workflow"] = _new_processing_workflow_state()
        _persist_processing_workflow_state_locked()

    for task_name in ("script", "voices", "proofread", "review", "sanity", "repair", "audacity_export", "m4b_export"):
        process_state[task_name]["logs"] = []
        process_state[task_name]["running"] = False
        if "progress" in process_state[task_name]:
            process_state[task_name]["progress"] = {}

    project_manager.engine = None
    project_manager.asr_engine = None
    with project_manager._transcription_cache_lock:
        project_manager._transcription_cache = None
    project_manager.recover_interrupted_generating_chunks()
    project_manager.reconcile_chunk_audio_states()


def _restore_project_archive(extracted_dir: str, *, loaded_project_name: str = ""):
    _clear_project_archive_targets()

    restored_state = {}
    for relative_path in sorted(PROJECT_ARCHIVE_DURABLE_FILES):
        source_path = _project_archive_source_path(extracted_dir, relative_path)
        target_path = _project_archive_filesystem_path(relative_path)
        if not os.path.exists(source_path):
            continue
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        if relative_path == "state.json":
            with open(source_path, "r", encoding="utf-8") as f:
                restored_state = json.load(f)
            input_file_path = (restored_state.get("input_file_path") or "").strip()
            if input_file_path:
                restored_state["input_file_path"] = os.path.normpath(
                    os.path.join(ROOT_DIR, input_file_path)
                )
        else:
            shutil.copy2(source_path, target_path)

    for dirname in ("uploads", "voicelines"):
        source_dir = os.path.join(extracted_dir, dirname)
        target_dir = os.path.join(ROOT_DIR, dirname)
        if not os.path.isdir(source_dir):
            continue
        if os.path.isdir(target_dir):
            shutil.rmtree(target_dir)
        shutil.copytree(source_dir, target_dir)

    for dirname in ("clone_voices", "designed_voices"):
        _merge_reusable_voice_library(extracted_dir, dirname)

    if hasattr(project_manager, "reload_script_store"):
        project_manager.reload_script_store()
    _save_project_state_payload(
        _normalize_restored_project_state(
            restored_state,
            loaded_project_name=loaded_project_name,
        )
    )
    if hasattr(project_manager, "log_voice_audit_event"):
        project_manager.log_voice_audit_event(
            "project_archive_restore",
            reason="restore_project_archive",
            loaded_project_name=loaded_project_name,
        )
    _reset_runtime_state_after_project_load()


def _restore_project_archive_zip(zip_path: str, *, loaded_project_name: str = ""):
    temp_root = _make_runtime_temp_dir("threadspeak_project_import_")
    extract_root = os.path.join(temp_root, "extracted")
    os.makedirs(extract_root, exist_ok=True)

    try:
        try:
            zf_context = zipfile.ZipFile(zip_path, "r")
        except zipfile.BadZipFile:
            raise HTTPException(status_code=400, detail="Project archive is not a valid .zip file.")

        with zf_context as zf:
            names = zf.namelist()
            if PROJECT_ARCHIVE_MANIFEST_NAME not in names:
                raise HTTPException(status_code=400, detail="Archive is missing project archive manifest.")

            try:
                manifest = json.loads(zf.read(PROJECT_ARCHIVE_MANIFEST_NAME).decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as e:
                raise HTTPException(status_code=400, detail=f"Archive manifest is invalid: {e}")
            if manifest.get("kind") != "threadspeak_project_archive":
                raise HTTPException(status_code=400, detail="Archive is not a valid Threadspeak project archive.")

            for info in zf.infolist():
                if info.is_dir() or info.filename == PROJECT_ARCHIVE_MANIFEST_NAME:
                    continue
                relative_path = _normalize_archive_path(info.filename)
                if not _is_allowed_project_archive_path(relative_path):
                    raise HTTPException(status_code=400, detail=f"Archive contains unsupported path: {relative_path}")
                target_path = os.path.join(extract_root, _archive_relative_file_target(relative_path))
                os.makedirs(os.path.dirname(target_path), exist_ok=True)
                with zf.open(info, "r") as source, open(target_path, "wb") as target:
                    shutil.copyfileobj(source, target)

        _restore_project_archive(extract_root, loaded_project_name=loaded_project_name)
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)


def _persist_audio_queue_state_locked():
    if not audio_queue and audio_current_job is None and not process_state["audio"].get("merge_running", False):
        if os.path.exists(AUDIO_QUEUE_STATE_PATH):
            try:
                os.remove(AUDIO_QUEUE_STATE_PATH)
            except OSError as exc:
                logger.warning(f"Failed to clear idle audio queue state: {exc}")
        return

    payload = {
        "job_counter": audio_job_counter,
        "queue": [_serialize_audio_job_checkpoint(job) | {"word_counts_by_uid": _job_word_counts(job)} for job in audio_queue],
        "current_job": (_serialize_audio_job_checkpoint(audio_current_job) | {"word_counts_by_uid": _job_word_counts(audio_current_job)}) if audio_current_job else None,
        "metrics": _format_audio_metrics_locked(),
        "heartbeat": _format_audio_heartbeat_locked(),
        "updated_at": time.time(),
    }
    try:
        _atomic_json_write(AUDIO_QUEUE_STATE_PATH, payload)
    except OSError as exc:
        logger.warning(f"Failed to persist audio queue state: {exc}")


def _refresh_audio_process_state_locked(persist=False):
    refresh_started = time.perf_counter()

    serialize_started = time.perf_counter()
    process_state["audio"]["queue"] = [_serialize_audio_job_summary(job) for job in audio_queue]
    process_state["audio"]["current_job"] = _serialize_audio_job_summary(audio_current_job) if audio_current_job else None
    serialize_ms = (time.perf_counter() - serialize_started) * 1000.0

    metrics_started = time.perf_counter()
    process_state["audio"]["running"] = audio_current_job is not None or process_state["audio"].get("merge_running", False)
    _recompute_audio_metrics_locked()
    process_state["audio"]["metrics"] = _format_audio_metrics_locked()
    process_state["audio"]["heartbeat"] = _format_audio_heartbeat_locked()
    process_state["audio"]["merge_progress"] = dict(process_state["audio"].get("merge_progress") or _new_audio_merge_progress())
    metrics_ms = (time.perf_counter() - metrics_started) * 1000.0

    coverage_started = time.perf_counter()
    try:
        coverage = project_manager.get_audio_coverage_summary()
    except Exception:
        coverage = _default_audio_coverage_summary()
    process_state["audio"]["audio_coverage"] = dict(coverage or _default_audio_coverage_summary())
    coverage_ms = (time.perf_counter() - coverage_started) * 1000.0

    persist_ms = 0.0
    if persist:
        persist_started = time.perf_counter()
        _persist_audio_queue_state_locked()
        persist_ms = (time.perf_counter() - persist_started) * 1000.0

    publish_started = time.perf_counter()
    chunk_event_broker.publish("audio_status", {
        "running": process_state["audio"]["running"],
        "queue": list(process_state["audio"]["queue"]),
        "current_job": dict(process_state["audio"]["current_job"]) if process_state["audio"]["current_job"] else None,
        "recent_jobs": list(process_state["audio"].get("recent_jobs") or []),
        "metrics": dict(process_state["audio"]["metrics"]),
        "heartbeat": dict(process_state["audio"]["heartbeat"]),
        "merge_running": bool(process_state["audio"].get("merge_running")),
        "merge_progress": dict(process_state["audio"].get("merge_progress") or {}),
        "audio_coverage": dict(process_state["audio"].get("audio_coverage") or _default_audio_coverage_summary()),
    })
    publish_ms = (time.perf_counter() - publish_started) * 1000.0
    record_audio_perf(
        "refresh_audio_process_state",
        persist=bool(persist),
        queue_len=len(audio_queue),
        current_job_id=(audio_current_job or {}).get("id"),
        serialize_ms=round(serialize_ms, 3),
        metrics_ms=round(metrics_ms, 3),
        coverage_ms=round(coverage_ms, 3),
        persist_ms=round(persist_ms, 3),
        publish_ms=round(publish_ms, 3),
        total_ms=round((time.perf_counter() - refresh_started) * 1000.0, 3),
    )


def _write_audio_cancel_tombstone_locked(reason, job=None):
    payload = {
        "requested_at": time.time(),
        "reason": reason,
        "job_id": (job or {}).get("id"),
        "corr_id": (job or {}).get("corr_id"),
        "run_token": (job or {}).get("run_token"),
    }
    _atomic_json_write(AUDIO_CANCEL_TOMBSTONE_PATH, payload)


def _clear_audio_cancel_tombstone_locked():
    if not os.path.exists(AUDIO_CANCEL_TOMBSTONE_PATH):
        return False
    try:
        os.remove(AUDIO_CANCEL_TOMBSTONE_PATH)
    except OSError:
        return False
    return True


def _hard_wipe_audio_runtime_locked(reason):
    """Immediately clear runtime audio work and persisted queue state."""
    global audio_current_job, audio_recovery_request, audio_current_runner_thread, audio_current_runner_token

    active_job = audio_current_job
    cleared_queued_jobs = len(audio_queue)
    reset_count = 0
    if active_job is not None:
        reset_count = project_manager.reset_generating_chunks(
            indices=_job_uids(active_job),
            generation_token=active_job.get("run_token"),
        )
        project_manager.clear_audio_finalize_tasks(generation_token=active_job.get("run_token"))
    else:
        reset_count = project_manager.reset_generating_chunks()
        project_manager.clear_audio_finalize_tasks()

    _write_audio_cancel_tombstone_locked(reason, active_job)

    now = time.time()
    while audio_queue:
        job = audio_queue.pop(0)
        job["status"] = "cancelled"
        job["finished_at"] = now
        _record_audio_recent_job_locked(job)

    if active_job is not None:
        active_job["status"] = "cancelled"
        active_job["finished_at"] = now
        _record_audio_recent_job_locked(active_job)

    _force_kill_active_audio_runner_locked()
    if active_job is not None:
        project_manager.unregister_audio_finalization_listener(active_job.get("run_token"))

    audio_current_job = None
    audio_current_runner_thread = None
    audio_current_runner_token = None
    audio_recovery_request = None
    process_state["audio"]["cancel"] = False
    audio_cancel_event.clear()
    _refresh_audio_process_state_locked(persist=False)
    audio_queue_condition.notify_all()

    if os.path.exists(AUDIO_QUEUE_STATE_PATH):
        try:
            os.remove(AUDIO_QUEUE_STATE_PATH)
        except OSError:
            pass

    return {
        "cleared_queued_jobs": cleared_queued_jobs,
        "had_active_job": active_job is not None,
        "reset_chunks": reset_count,
    }


def _append_audio_log(message):
    with audio_queue_lock:
        _append_audio_log_locked(message)


def _append_audio_log_locked(message):
    process_state["audio"]["logs"].append(message)
    _trim_logs(process_state["audio"]["logs"])


def _read_runtime_llm_settings() -> Dict[str, object]:
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError, ValueError):
        return {}
    llm_payload = payload.get("llm") if isinstance(payload, dict) else None
    return llm_payload if isinstance(llm_payload, dict) else {}


def _attempt_lmstudio_unload_all_models(action_label: str) -> Dict[str, object]:
    action = str(action_label or "runtime").strip() or "runtime"
    llm_settings = _read_runtime_llm_settings()
    base_url = str(llm_settings.get("base_url") or "").strip()
    api_key = str(llm_settings.get("api_key") or "")

    if not base_url:
        message = f"LM Studio unload-all skipped before {action}: no LLM base URL is configured."
        logger.info(message)
        return {"status": "skipped", "reason": "missing_base_url", "message": message}

    if ToolCapabilityService.is_openrouter_url(base_url):
        message = f"LM Studio unload-all skipped before {action}: OpenRouter endpoint configured."
        logger.info(message)
        return {"status": "skipped", "reason": "openrouter", "message": message}

    try:
        result = LMStudioModelLoadService(
            timeout_seconds=_LMSTUDIO_UNLOAD_TIMEOUT_SECONDS
        ).unload_all_models(base_url=base_url, api_key=api_key)
        unloaded = int(result.get("total_loaded_instances") or 0) if isinstance(result, dict) else 0
        message = f"LM Studio unload-all succeeded before {action}: unloaded {unloaded} instance(s)."
        logger.info(message)
        return {
            "status": "ok",
            "reason": "unloaded",
            "message": message,
            "details": result if isinstance(result, dict) else {},
        }
    except Exception as exc:
        message = f"LM Studio unload-all skipped before {action}: {exc}"
        logger.info(message)
        return {"status": "skipped", "reason": "unreachable", "message": message}


def _load_export_config() -> ExportConfig:
    """Load ExportConfig from config.json, falling back to defaults for missing fields."""
    try:
        if os.path.exists(CONFIG_PATH):
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                raw = json.load(f)
            export_raw = raw.get("export") or {}
            return ExportConfig(**{k: v for k, v in export_raw.items() if k in ExportConfig.model_fields})
    except Exception:
        pass
    return ExportConfig()


def _record_audio_recent_job_locked(job):
    process_state["audio"]["recent_jobs"].insert(0, _serialize_audio_job_summary(job))
    del process_state["audio"]["recent_jobs"][10:]


def _record_audio_sample_locked(job, chunk_uid, elapsed_seconds, input_words, output_words, success):
    metrics = process_state["audio"]["metrics"]
    # Older refresh paths may have serialized metrics without the rolling sample
    # buffer. Recreate it so timing estimates continue updating instead of
    # breaking on the next queue refresh.
    metrics.setdefault("samples", [])
    now = time.time()
    sample = {
        "job_id": job["id"],
        "chunk_uid": chunk_uid,
        "elapsed_seconds": max(0.0, float(elapsed_seconds)),
        "input_words": max(0, int(input_words)),
        "output_words": max(0, int(output_words)),
        "success": bool(success),
    }
    metrics["samples"].append(sample)
    metrics["rolling_seconds"] += sample["elapsed_seconds"]
    metrics["rolling_input_words"] += sample["input_words"]
    metrics["rolling_output_words"] += sample["output_words"]
    while len(metrics["samples"]) > metrics["sample_window_size"]:
        removed = metrics["samples"].pop(0)
        metrics["rolling_seconds"] -= removed["elapsed_seconds"]
        metrics["rolling_input_words"] -= removed["input_words"]
        metrics["rolling_output_words"] -= removed["output_words"]

    metrics["processed_clips"] += 1
    metrics["total_elapsed_seconds"] += sample["elapsed_seconds"]
    metrics["total_input_words"] += sample["input_words"]
    metrics["total_output_words"] += sample["output_words"]
    if success:
        metrics["successful_clips"] += 1
    else:
        metrics["error_clips"] += 1

    job["processed_clips"] = job.get("processed_clips", 0) + 1
    pending_uids = _job_pending_uids(job)
    if chunk_uid in pending_uids:
        pending_uids.remove(chunk_uid)
        job["pending_uids"] = pending_uids
    if not success:
        job["error_clips"] = job.get("error_clips", 0) + 1
    job["remaining_words"] = max(0, job.get("remaining_words", 0) - sample["input_words"])
    job["last_output_at"] = now
    process_state["audio"]["heartbeat"]["last_output_at"] = now
    _recompute_audio_metrics_locked()


def _mark_audio_generation_activity_locked(job, when=None):
    now = when or time.time()
    job["last_generation_activity_at"] = now
    process_state["audio"]["heartbeat"]["last_generation_activity_at"] = now


def _mark_audio_finalize_activity_locked(job, when=None):
    now = when or time.time()
    job["last_finalize_activity_at"] = now
    process_state["audio"]["heartbeat"]["last_finalize_activity_at"] = now


def _record_audio_finalize_submission_locked(job, chunk_uid):
    normalized_uid = str(chunk_uid or "").strip()
    if not normalized_uid:
        return
    generation_pending = _job_generation_pending_uids(job)
    if normalized_uid in generation_pending:
        generation_pending.remove(normalized_uid)
    job["generation_pending_uids"] = generation_pending

    pending_finalize = _job_pending_finalize_uids(job)
    if normalized_uid not in pending_finalize:
        pending_finalize.append(normalized_uid)
    job["pending_finalize_uids"] = pending_finalize
    _mark_audio_generation_activity_locked(job)


def _record_audio_finalize_result_locked(job, chunk_uid, success, meta=None):
    meta = dict(meta or {})
    normalized_uid = str(chunk_uid or "").strip()
    pending_finalize = _job_pending_finalize_uids(job)
    if normalized_uid in pending_finalize:
        pending_finalize.remove(normalized_uid)
    job["pending_finalize_uids"] = pending_finalize
    job["finalized_clips"] = int(job.get("finalized_clips", 0) or 0) + 1
    if not success:
        job["finalizer_failures"] = int(job.get("finalizer_failures", 0) or 0) + 1
        if meta.get("retry_requested"):
            retry_uids = [str(uid).strip() for uid in (job.get("retry_uids") or []) if str(uid).strip()]
            if normalized_uid and normalized_uid not in retry_uids:
                retry_uids.append(normalized_uid)
            job["retry_uids"] = retry_uids
    _mark_audio_finalize_activity_locked(job)


def _audio_job_ready_to_finish_locked(job):
    _reconcile_audio_job_runtime_locked(job)
    return bool(
        job
        and job.get("generation_finished")
        and not _job_generation_pending_uids(job)
        and not _job_pending_finalize_uids(job)
    )


def _restore_job_progress_from_chunks(raw_job, chunks=None):
    if chunks is not None and not raw_job.get("uids") and not raw_job.get("pending_uids") and not raw_job.get("word_counts_by_uid"):
        legacy_indices = [int(idx) for idx in (raw_job.get("indices") or []) if str(idx).strip()]
        reconciled_indices = [idx for idx in legacy_indices if 0 <= idx < len(chunks)]
        word_counts = {
            str(idx): int(value or 0)
            for idx, value in dict(raw_job.get("word_counts") or {}).items()
            if str(idx).strip()
        }
        dictionary_entries = project_manager.load_dictionary_entries()
        pending_indices = []
        processed_clips = 0
        error_clips = 0

        for idx in reconciled_indices:
            chunk = chunks[idx]
            status = chunk.get("status")
            if status == "done":
                processed_clips += 1
                continue

            validation = project_manager._validate_chunk_audio(chunk, dictionary_entries)
            if validation and validation.get("is_valid"):
                processed_clips += 1
                chunk["status"] = "done"
                chunk["audio_validation"] = validation
                chunk["auto_regen_count"] = 0
                continue

            if status == "error":
                processed_clips += 1
                error_clips += 1
                continue

            pending_indices.append(idx)

        total_words = sum(word_counts.get(str(idx), 0) for idx in reconciled_indices)
        remaining_words = sum(word_counts.get(str(idx), 0) for idx in pending_indices)
        return {
            "indices": reconciled_indices,
            "pending_indices": pending_indices,
            "word_counts": word_counts,
            "total_chunks": len(reconciled_indices),
            "total_words": total_words,
            "remaining_words": remaining_words,
            "processed_clips": processed_clips,
            "error_clips": error_clips,
        }

    uids = _job_uids(raw_job)
    if not uids:
        legacy_indices = [idx for idx in (raw_job.get("indices") or [])]
        resolved_uids = []
        for chunk_ref in legacy_indices:
            chunk = project_manager.get_chunk_raw(chunk_ref)
            if chunk is None:
                continue
            uid = str(chunk.get("uid") or "").strip()
            if uid:
                resolved_uids.append(uid)
        uids = resolved_uids

    word_counts = _job_word_counts(raw_job)
    dictionary_entries = project_manager.load_dictionary_entries()
    chunks = project_manager.get_chunks_by_uids(uids)
    chunks_by_uid = {
        str((chunk or {}).get("uid") or "").strip(): chunk
        for chunk in chunks
        if str((chunk or {}).get("uid") or "").strip()
    }

    reconciled_uids = [uid for uid in uids if uid in chunks_by_uid]
    pending_uids = []
    processed_clips = 0
    error_clips = 0

    for uid in reconciled_uids:
        chunk = chunks_by_uid[uid]
        status = chunk.get("status")
        if status == "done":
            processed_clips += 1
            continue

        # Status can drift during crashes/cancels; trust the on-disk audio when valid.
        validation = project_manager._validate_chunk_audio(chunk, dictionary_entries)
        if validation and validation.get("is_valid"):
            processed_clips += 1
            chunk["status"] = "done"
            chunk["audio_validation"] = validation
            chunk["auto_regen_count"] = 0
            continue

        if status == "error":
            processed_clips += 1
            error_clips += 1
            continue

        pending_uids.append(uid)

    total_words = sum(word_counts.get(uid, 0) for uid in reconciled_uids)
    remaining_words = sum(word_counts.get(uid, 0) for uid in pending_uids)

    return {
        "uids": reconciled_uids,
        "pending_uids": pending_uids,
        "word_counts_by_uid": word_counts,
        "total_chunks": len(reconciled_uids),
        "total_words": total_words,
        "remaining_words": remaining_words,
        "processed_clips": processed_clips,
        "error_clips": error_clips,
    }


def _load_audio_worker_settings():
    workers = 2
    batch_seed = -1
    batch_size = 4
    batch_group_by_type = False
    tts_cfg = {}
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                cfg = json.load(f)
                tts_cfg = cfg.get("tts", {})
                workers = max(1, tts_cfg.get("parallel_workers", 2))
                seed_val = tts_cfg.get("batch_seed")
                if seed_val is not None and seed_val != "":
                    batch_seed = int(seed_val)
                batch_size = max(1, tts_cfg.get("parallel_workers", 4))
                batch_group_by_type = tts_cfg.get("batch_group_by_type", False)
        except (json.JSONDecodeError, ValueError):
            pass
    return {
        "workers": workers,
        "batch_seed": batch_seed,
        "batch_size": batch_size,
        "batch_group_by_type": batch_group_by_type,
        "tts_cfg": tts_cfg,
    }


def _enqueue_audio_job(kind, indices, label=None, scope=None, scope_mode=None, chapter=None, neutral_narrator=False):
    global audio_job_counter

    with audio_queue_condition:
        _clear_audio_cancel_tombstone_locked()
        audio_job_counter += 1
        if audio_current_job is None and not audio_queue:
            process_state["audio"]["logs"] = []
            process_state["audio"]["recent_jobs"] = []

        valid_rows = []
        seen_uids = set()
        for chunk_ref in indices:
            chunk = project_manager.get_chunk_raw(chunk_ref)
            if chunk is None:
                continue
            uid = str(chunk.get("uid") or "").strip()
            text = chunk.get("text", "")
            if not uid or uid in seen_uids or not text or not text.strip():
                continue
            seen_uids.add(uid)
            valid_rows.append(chunk)
        if not valid_rows:
            _append_audio_log_locked("[QUEUE] Rejecting enqueue request: no valid non-empty indices after resolution.")
            raise HTTPException(status_code=400, detail="No non-empty chunk indices provided")
        valid_uids = [chunk.get("uid") for chunk in valid_rows]
        word_counts = {
            chunk.get("uid"): _count_words(chunk.get("text", ""))
            for chunk in valid_rows
            if chunk.get("uid")
        }

        job = {
            "id": audio_job_counter,
            "corr_id": f"audio-{audio_job_counter:05d}-{uuid.uuid4().hex[:8]}",
            "kind": kind,
            "uids": valid_uids,
            "pending_uids": list(valid_uids),
            "generation_pending_uids": list(valid_uids),
            "pending_finalize_uids": [],
            "word_counts_by_uid": word_counts,
            "total_chunks": len(valid_uids),
            "total_words": sum(word_counts.values()),
            "remaining_words": sum(word_counts.values()),
            "processed_clips": 0,
            "error_clips": 0,
            "generation_finished": False,
            "finalized_clips": 0,
            "finalizer_failures": 0,
            "retry_uids": [],
            "recovery_count": 0,
            "label": label or f"Audio Job {audio_job_counter}",
            "scope": scope or "custom",
            "scope_mode": scope_mode,
            "chapter": chapter,
            "neutral_narrator": bool(neutral_narrator),
            "status": "queued",
            "queued_at": time.time(),
            "started_at": None,
            "finished_at": None,
            "last_output_at": None,
            "last_generation_activity_at": None,
            "last_finalize_activity_at": None,
            "run_token": None,
        }
        audio_queue.append(job)
        _append_audio_log_locked(
            f"[QUEUE] Job #{job['id']} ({job['corr_id']}) queued: {job['label']} ({job['total_chunks']} chunks, scope={job['scope']})"
        )
        _refresh_audio_process_state_locked(persist=True)
        queue_position = len(audio_queue)
        audio_queue_condition.notify()
        return {
            "status": "queued",
            "job_id": job["id"],
            "queue_position": queue_position,
            "total_chunks": job["total_chunks"],
            "total_words": job["total_words"],
            "label": job["label"],
            "scope": job["scope"],
            "estimated_remaining_seconds": process_state["audio"]["metrics"]["estimated_remaining_seconds"],
        }


def _clone_audio_job_for_retry(job, pending_uids, reason):
    global audio_job_counter

    word_counts_by_uid = _job_word_counts(job)
    pending_uids = [uid for uid in pending_uids if uid in word_counts_by_uid]
    if not pending_uids:
        return None

    audio_job_counter += 1
    word_counts = {uid: word_counts_by_uid[uid] for uid in pending_uids if uid in word_counts_by_uid}
    retry_count = job.get("recovery_count", 0) + 1
    return {
        "id": audio_job_counter,
        "corr_id": f"audio-{audio_job_counter:05d}-{uuid.uuid4().hex[:8]}",
        "kind": job["kind"],
        "uids": list(pending_uids),
        "pending_uids": list(pending_uids),
        "generation_pending_uids": list(pending_uids),
        "pending_finalize_uids": [],
        "word_counts_by_uid": word_counts,
        "total_chunks": len(pending_uids),
        "total_words": sum(word_counts.values()),
        "remaining_words": sum(word_counts.values()),
        "processed_clips": 0,
        "error_clips": 0,
        "generation_finished": False,
        "finalized_clips": 0,
        "finalizer_failures": 0,
        "retry_uids": [],
        "recovery_count": retry_count,
        "label": f"{job['label']} (resume {retry_count})",
        "scope": job["scope"],
        "scope_mode": job.get("scope_mode"),
        "chapter": job.get("chapter"),
        "neutral_narrator": bool(job.get("neutral_narrator", False)),
        "status": "queued",
        "queued_at": time.time(),
        "started_at": None,
        "finished_at": None,
        "last_output_at": None,
        "last_generation_activity_at": None,
        "last_finalize_activity_at": None,
        "run_token": None,
        "resume_of": job["id"],
        "resume_reason": reason,
    }


def _queue_follow_on_retry_job_locked(job):
    retry_uids = [uid for uid in (job.get("retry_uids") or []) if uid in _job_word_counts(job)]
    if not retry_uids:
        return None
    retry_job = _clone_audio_job_for_retry(job, retry_uids, "finalizer invalid clip retry")
    if retry_job is None:
        return None
    retry_job["label"] = f"{job['label']} (invalid clip retry)"
    retry_job["scope"] = "custom"
    retry_job["resume_of"] = job["id"]
    audio_queue.append(retry_job)
    _append_audio_log_locked(
        f"[QUEUE] Follow-on retry job #{retry_job['id']} ({retry_job['corr_id']}) queued for {len(retry_uids)} invalid clip(s)"
    )
    return retry_job


def _restore_audio_queue_state():
    global audio_job_counter

    if not os.path.exists(AUDIO_QUEUE_STATE_PATH):
        return

    try:
        with open(AUDIO_QUEUE_STATE_PATH, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except (OSError, json.JSONDecodeError, ValueError) as e:
        logger.warning(f"Failed to restore audio queue state: {e}")
        return

    # A cancellation can race with process shutdown. In that case, a stale
    # running state may still be persisted. A tombstone explicitly blocks
    # restart-time job resurrection after any cancel request.
    if os.path.exists(AUDIO_CANCEL_TOMBSTONE_PATH):
        try:
            with open(AUDIO_CANCEL_TOMBSTONE_PATH, "r", encoding="utf-8") as f:
                tombstone = json.load(f)
        except Exception:
            tombstone = {}
        try:
            os.remove(AUDIO_CANCEL_TOMBSTONE_PATH)
        except OSError:
            pass
        try:
            os.remove(AUDIO_QUEUE_STATE_PATH)
        except OSError:
            pass
        logger.info(
            "Discarded persisted audio queue state due to cancel tombstone (reason=%r, job_id=%r, corr_id=%r).",
            tombstone.get("reason"),
            tombstone.get("job_id"),
            tombstone.get("corr_id"),
        )
        return

    # Guard: if there are no valid chunks (project was reset or deleted while the
    # server was down), discard the stale queue state rather than resuming generation
    # against a non-existent or empty project.
    try:
        has_valid_chunks = project_manager.has_substantive_chunks()
    except Exception:
        has_valid_chunks = False

    if not has_valid_chunks:
        try:
            os.remove(AUDIO_QUEUE_STATE_PATH)
        except OSError:
            pass
        logger.info("Discarded stale audio queue state — no valid project chunks found.")
        return

    with audio_queue_condition:
        project_manager.recover_interrupted_generating_chunks()
        project_manager.reconcile_chunk_audio_states()
        process_state["audio"]["metrics"] = _normalize_audio_metrics(payload.get("metrics"))
        process_state["audio"]["heartbeat"] = _new_audio_heartbeat_state()
        process_state["audio"]["logs"] = []
        process_state["audio"]["recent_jobs"] = []
        audio_job_counter = max(audio_job_counter, int(payload.get("job_counter", 0) or 0))

        saved_heartbeat = payload.get("heartbeat") or {}
        process_state["audio"]["heartbeat"]["last_check_at"] = saved_heartbeat.get("last_check_at")
        process_state["audio"]["heartbeat"]["last_output_at"] = saved_heartbeat.get("last_output_at")
        process_state["audio"]["heartbeat"]["last_generation_activity_at"] = saved_heartbeat.get("last_generation_activity_at")
        process_state["audio"]["heartbeat"]["last_finalize_activity_at"] = saved_heartbeat.get("last_finalize_activity_at")
        process_state["audio"]["heartbeat"]["last_recovery_at"] = saved_heartbeat.get("last_recovery_at")
        process_state["audio"]["heartbeat"]["recovery_count"] = int(saved_heartbeat.get("recovery_count", 0) or 0)
        process_state["audio"]["heartbeat"]["last_recovery_reason"] = saved_heartbeat.get("last_recovery_reason")

        restored_jobs = []
        for raw_job in payload.get("queue", []):
            progress = _restore_job_progress_from_chunks(raw_job)
            if not progress["pending_uids"]:
                continue
            normalized_runtime = _normalize_restored_audio_job_runtime(raw_job, progress)
            restored_jobs.append({
                "id": int(raw_job.get("id", 0) or 0),
                "corr_id": (raw_job.get("corr_id") or f"audio-{int(raw_job.get('id', 0) or 0):05d}-{uuid.uuid4().hex[:8]}"),
                "kind": raw_job.get("kind", "parallel"),
                "uids": progress["uids"],
                "pending_uids": progress["pending_uids"],
                "generation_pending_uids": normalized_runtime["generation_pending_uids"],
                "pending_finalize_uids": normalized_runtime["pending_finalize_uids"],
                "word_counts_by_uid": progress["word_counts_by_uid"],
                "total_chunks": progress["total_chunks"],
                "total_words": progress["total_words"],
                "remaining_words": progress["remaining_words"],
                "processed_clips": progress["processed_clips"],
                "error_clips": progress["error_clips"],
                "generation_finished": normalized_runtime["generation_finished"],
                "finalized_clips": int(raw_job.get("finalized_clips", 0) or progress["processed_clips"]),
                "finalizer_failures": int(raw_job.get("finalizer_failures", 0) or 0),
                "retry_uids": [uid for uid in (raw_job.get("retry_uids") or []) if uid in progress["pending_uids"]],
                "recovery_count": int(raw_job.get("recovery_count", 0) or 0),
                "label": raw_job.get("label", "Recovered audio job"),
                "scope": raw_job.get("scope", "custom"),
                "scope_mode": raw_job.get("scope_mode"),
                "chapter": raw_job.get("chapter"),
                "neutral_narrator": bool(raw_job.get("neutral_narrator", False)),
                "status": "queued",
                "queued_at": time.time(),
                "started_at": None,
                "finished_at": None,
                "last_output_at": None,
                "last_generation_activity_at": raw_job.get("last_generation_activity_at"),
                "last_finalize_activity_at": raw_job.get("last_finalize_activity_at"),
                "run_token": normalized_runtime["run_token"],
            })

        raw_current = payload.get("current_job")
        resumed_job = None
        if raw_current:
            progress = _restore_job_progress_from_chunks(raw_current)
            if progress["pending_uids"]:
                normalized_runtime = _normalize_restored_audio_job_runtime(raw_current, progress)
                resumed_job = {
                    "id": int(raw_current.get("id", 0) or 0),
                    "corr_id": (raw_current.get("corr_id") or f"audio-{int(raw_current.get('id', 0) or 0):05d}-{uuid.uuid4().hex[:8]}"),
                    "kind": raw_current.get("kind", "parallel"),
                    "uids": progress["uids"],
                    "pending_uids": progress["pending_uids"],
                    "generation_pending_uids": normalized_runtime["generation_pending_uids"],
                    "pending_finalize_uids": normalized_runtime["pending_finalize_uids"],
                    "word_counts_by_uid": progress["word_counts_by_uid"],
                    "total_chunks": progress["total_chunks"],
                    "total_words": progress["total_words"],
                    "remaining_words": progress["remaining_words"],
                    "processed_clips": progress["processed_clips"],
                    "error_clips": progress["error_clips"],
                    "generation_finished": normalized_runtime["generation_finished"],
                    "finalized_clips": int(raw_current.get("finalized_clips", 0) or progress["processed_clips"]),
                    "finalizer_failures": int(raw_current.get("finalizer_failures", 0) or 0),
                    "retry_uids": [uid for uid in (raw_current.get("retry_uids") or []) if uid in progress["pending_uids"]],
                    "recovery_count": int(raw_current.get("recovery_count", 0) or 0) + 1,
                    "label": f"{raw_current.get('label', 'Recovered audio job')} (resumed after restart)",
                    "scope": raw_current.get("scope", "custom"),
                    "scope_mode": raw_current.get("scope_mode"),
                    "chapter": raw_current.get("chapter"),
                    "neutral_narrator": bool(raw_current.get("neutral_narrator", False)),
                    "status": "queued",
                    "queued_at": time.time(),
                    "started_at": None,
                    "finished_at": None,
                    "last_output_at": None,
                    "last_generation_activity_at": raw_current.get("last_generation_activity_at"),
                    "last_finalize_activity_at": raw_current.get("last_finalize_activity_at"),
                    "run_token": normalized_runtime["run_token"],
                }
                restored_jobs.insert(0, resumed_job)
                _append_audio_log_locked(
                    f"[RECOVER] Restored interrupted job from disk with {len(progress['pending_uids'])} pending chunk(s)"
                )

        audio_queue[:] = restored_jobs
        restored_metrics = _normalize_audio_metrics(payload.get("metrics"))
        processed_from_jobs = sum(max(0, int(job.get("processed_clips", 0) or 0)) for job in restored_jobs)
        error_from_jobs = sum(max(0, int(job.get("error_clips", 0) or 0)) for job in restored_jobs)
        restored_metrics["processed_clips"] = max(restored_metrics["processed_clips"], processed_from_jobs)
        restored_metrics["error_clips"] = min(
            restored_metrics["processed_clips"],
            max(restored_metrics["error_clips"], error_from_jobs),
        )
        restored_metrics["successful_clips"] = restored_metrics["processed_clips"] - restored_metrics["error_clips"]
        process_state["audio"]["metrics"] = restored_metrics
        _recompute_audio_metrics_locked()
        if restored_jobs:
            _refresh_audio_process_state_locked(persist=True)
            audio_queue_condition.notify_all()


def _request_audio_recovery_locked(reason):
    global audio_recovery_request
    if audio_current_job is None:
        return False
    if audio_recovery_request is not None:
        return False
    audio_recovery_request = {
        "job_id": audio_current_job["id"],
        "run_token": audio_current_job.get("run_token"),
        "reason": reason,
        "requested_at": time.time(),
    }
    _append_audio_log_locked(f"[WATCHDOG] {reason}")
    _refresh_audio_process_state_locked(persist=True)
    audio_queue_condition.notify_all()
    return True


def _perform_audio_recovery_locked(job, run_token, reason):
    global audio_current_job, audio_recovery_request

    if audio_current_job is None:
        return False
    if audio_current_job["id"] != job["id"] or audio_current_job.get("run_token") != run_token:
        return False

    pending_uids = list(_job_pending_uids(job))
    project_manager.reset_generating_chunks(indices=_job_uids(job), generation_token=run_token)
    project_manager.clear_audio_finalize_tasks(generation_token=run_token)
    project_manager.unregister_audio_finalization_listener(run_token)

    job["status"] = "stalled"
    job["finished_at"] = time.time()
    _record_audio_recent_job_locked(job)

    heartbeat = process_state["audio"]["heartbeat"]
    heartbeat["last_recovery_at"] = job["finished_at"]
    heartbeat["last_recovery_reason"] = reason
    heartbeat["recovery_count"] += 1

    retry_job = _clone_audio_job_for_retry(job, pending_uids, reason)
    if retry_job is not None:
        audio_queue.insert(0, retry_job)
        _append_audio_log_locked(
            f"[RECOVER] Re-queued {len(retry_job['uids'])} stalled chunk(s) from job #{job['id']} ({job.get('corr_id')}) to the front of the queue as #{retry_job['id']} ({retry_job.get('corr_id')})"
        )
    else:
        _append_audio_log_locked(f"[RECOVER] No remaining chunks to re-queue for job #{job['id']} ({job.get('corr_id')})")

    audio_current_job = None
    process_state["audio"]["cancel"] = False
    audio_recovery_request = None
    _refresh_audio_process_state_locked(persist=True)
    audio_queue_condition.notify_all()
    return True


def _abandon_audio_job_locked(job, run_token, reason, *, status="cancelled"):
    global audio_current_job, audio_recovery_request, audio_current_runner_thread, audio_current_runner_token

    if audio_current_job is None:
        _append_audio_log_locked("[CANCEL] Abandon requested but no active job exists.")
        return False
    if audio_current_job["id"] != job["id"] or audio_current_job.get("run_token") != run_token:
        _append_audio_log_locked(
            f"[CANCEL] Abandon skipped due to token/job mismatch requested_job={job.get('id')} requested_token={run_token}"
        )
        return False

    _append_audio_log_locked(
        f"[CANCEL] Abandoning job #{job['id']} ({job.get('corr_id')}) status={status} reason='{reason}' run_token={run_token} pending={len(_job_pending_uids(job))}"
    )
    reset_count = project_manager.reset_generating_chunks(
        indices=_job_uids(job),
        generation_token=run_token,
    )
    project_manager.clear_audio_finalize_tasks(generation_token=run_token)
    project_manager.unregister_audio_finalization_listener(run_token)
    job["status"] = status
    job["finished_at"] = time.time()
    _record_audio_recent_job_locked(job)
    _append_audio_log_locked(
        f"[CANCEL] Abandoned job #{job['id']} ({job.get('corr_id')}) and reset {reset_count} generating chunk(s) to pending"
    )

    audio_current_job = None
    audio_recovery_request = None
    audio_current_runner_thread = None
    audio_current_runner_token = None
    process_state["audio"]["cancel"] = False
    audio_cancel_event.clear()
    _refresh_audio_process_state_locked(persist=True)
    audio_queue_condition.notify_all()
    return True


def _audio_job_runner(job, settings, run_token, result_holder, done_event):
    job_prefix = f"[JOB {job['id']}|{job.get('corr_id', 'no-cid')}]"
    generation_pending_uids = _job_generation_pending_uids(job)
    if generation_pending_uids:
        execution_uids = list(generation_pending_uids)
    elif job.get("generation_finished"):
        execution_uids = []
    else:
        execution_uids = list(_job_uids(job))
    started_at_by_uid = {}
    generation_elapsed_by_uid = {}

    def is_active():
        with audio_queue_lock:
            return (
                audio_current_job is not None
                and audio_current_job["id"] == job["id"]
                and audio_current_job.get("run_token") == run_token
            )

    def _remember_started_at(uid, started_at):
        normalized_uid = str(uid or "").strip()
        if not normalized_uid:
            return
        try:
            normalized_started_at = float(started_at)
        except (TypeError, ValueError):
            return
        if normalized_started_at <= 0:
            return
        started_at_by_uid[normalized_uid] = normalized_started_at

    def _remember_generation_elapsed(uid, elapsed_seconds):
        normalized_uid = str(uid or "").strip()
        if not normalized_uid:
            return
        try:
            normalized_elapsed = float(elapsed_seconds)
        except (TypeError, ValueError):
            return
        generation_elapsed_by_uid[normalized_uid] = max(0.0, normalized_elapsed)

    def _consume_generation_elapsed(uid, fallback_elapsed_seconds):
        normalized_uid = str(uid or "").strip()
        if not normalized_uid:
            return max(0.0, float(fallback_elapsed_seconds or 0.0))
        stored = generation_elapsed_by_uid.pop(normalized_uid, None)
        if stored is not None:
            return stored
        return max(0.0, float(fallback_elapsed_seconds or 0.0))

    def _consume_elapsed_seconds(uid, fallback_elapsed_seconds):
        normalized_uid = str(uid or "").strip()
        started_at = started_at_by_uid.pop(normalized_uid, None)
        if started_at is None:
            return max(0.0, float(fallback_elapsed_seconds or 0.0))
        return max(0.0, time.time() - started_at)

    def item_started_callback(uid, started_at):
        callback_started = time.perf_counter()
        refresh_ms = 0.0
        with audio_queue_lock:
            if not is_active():
                return
            _remember_started_at(uid, started_at)
            _mark_audio_generation_activity_locked(job)
            refresh_started = time.perf_counter()
            _refresh_audio_process_state_locked(persist=False)
            refresh_ms = (time.perf_counter() - refresh_started) * 1000.0
        record_audio_perf(
            "audio_callback_item_started",
            job_id=job["id"],
            uid=uid,
            refresh_ms=round(refresh_ms, 3),
            total_ms=round((time.perf_counter() - callback_started) * 1000.0, 3),
        )

    def progress_callback(completed, failed, total):
        callback_started = time.perf_counter()
        refresh_ms = 0.0
        with audio_queue_lock:
            if not is_active():
                return
            _mark_audio_generation_activity_locked(job)
            _append_audio_log_locked(
                f"{job_prefix} Progress: {completed + failed}/{total} ({completed} submitted, {failed} failed)"
            )
            refresh_started = time.perf_counter()
            _refresh_audio_process_state_locked(persist=False)
            refresh_ms = (time.perf_counter() - refresh_started) * 1000.0
        record_audio_perf(
            "audio_callback_progress",
            job_id=job["id"],
            completed=int(completed or 0),
            failed=int(failed or 0),
            total=int(total or 0),
            refresh_ms=round(refresh_ms, 3),
            total_ms=round((time.perf_counter() - callback_started) * 1000.0, 3),
        )

    def item_callback(uid, success, elapsed_seconds, input_words, output_words):
        callback_started = time.perf_counter()
        refresh_ms = 0.0
        with audio_queue_lock:
            if not is_active():
                return
            generation_pending = _job_generation_pending_uids(job)
            if uid in generation_pending:
                generation_pending.remove(uid)
            job["generation_pending_uids"] = generation_pending
            _mark_audio_generation_activity_locked(job)
            if success:
                _remember_generation_elapsed(uid, elapsed_seconds)
                refresh_started = time.perf_counter()
                _refresh_audio_process_state_locked(persist=False)
                refresh_ms = (time.perf_counter() - refresh_started) * 1000.0
            else:
                effective_elapsed = _consume_elapsed_seconds(uid, elapsed_seconds)
                _record_audio_sample_locked(job, uid, effective_elapsed, input_words, output_words, success)
                refresh_started = time.perf_counter()
                _refresh_audio_process_state_locked(persist=False)
                refresh_ms = (time.perf_counter() - refresh_started) * 1000.0
        record_audio_perf(
            "audio_callback_item",
            job_id=job["id"],
            uid=uid,
            success=bool(success),
            refresh_ms=round(refresh_ms, 3),
            total_ms=round((time.perf_counter() - callback_started) * 1000.0, 3),
        )
        if success:
            return

    def submission_callback(uid, task):
        callback_started = time.perf_counter()
        refresh_ms = 0.0
        with audio_queue_lock:
            if not is_active():
                return
            _record_audio_finalize_submission_locked(job, uid)
            refresh_started = time.perf_counter()
            _refresh_audio_process_state_locked(persist=False)
            refresh_ms = (time.perf_counter() - refresh_started) * 1000.0
            audio_queue_condition.notify_all()
        record_audio_perf(
            "audio_callback_submission",
            job_id=job["id"],
            uid=uid,
            task_id=(task or {}).get("id"),
            refresh_ms=round(refresh_ms, 3),
            total_ms=round((time.perf_counter() - callback_started) * 1000.0, 3),
        )

    def finalization_item_callback(uid, success, elapsed_seconds, input_words, output_words, meta):
        callback_started = time.perf_counter()
        refresh_ms = 0.0
        with audio_queue_lock:
            if not is_active():
                return
            _record_audio_finalize_result_locked(job, uid, success, meta)
            effective_elapsed = _consume_generation_elapsed(uid, 0.0) + max(0.0, float(elapsed_seconds or 0.0))
            started_at_by_uid.pop(str(uid or "").strip(), None)
            _record_audio_sample_locked(job, uid, effective_elapsed, input_words, output_words, success)
            refresh_started = time.perf_counter()
            _refresh_audio_process_state_locked(persist=False)
            refresh_ms = (time.perf_counter() - refresh_started) * 1000.0
            audio_queue_condition.notify_all()
        record_audio_perf(
            "audio_callback_finalization_result",
            job_id=job["id"],
            uid=uid,
            success=bool(success),
            refresh_ms=round(refresh_ms, 3),
            total_ms=round((time.perf_counter() - callback_started) * 1000.0, 3),
        )

    def activity_callback(kind, uid):
        callback_started = time.perf_counter()
        refresh_ms = 0.0
        with audio_queue_lock:
            if not is_active():
                return
            if kind == "finalizer_started":
                _mark_audio_finalize_activity_locked(job)
            else:
                _mark_audio_generation_activity_locked(job)
            refresh_started = time.perf_counter()
            _refresh_audio_process_state_locked(persist=False)
            refresh_ms = (time.perf_counter() - refresh_started) * 1000.0
        record_audio_perf(
            "audio_callback_activity",
            job_id=job["id"],
            uid=uid,
            kind=kind,
            refresh_ms=round(refresh_ms, 3),
            total_ms=round((time.perf_counter() - callback_started) * 1000.0, 3),
        )

    def log_callback(message):
        callback_started = time.perf_counter()
        refresh_ms = 0.0
        with audio_queue_lock:
            if not is_active():
                return
            _mark_audio_generation_activity_locked(job)
            _append_audio_log_locked(str(message or ""))
            refresh_started = time.perf_counter()
            _refresh_audio_process_state_locked(persist=False)
            refresh_ms = (time.perf_counter() - refresh_started) * 1000.0
        record_audio_perf(
            "audio_callback_log",
            job_id=job["id"],
            refresh_ms=round(refresh_ms, 3),
            total_ms=round((time.perf_counter() - callback_started) * 1000.0, 3),
        )

    def cancel_check():
        with audio_queue_lock:
            if not is_active():
                return True
            return process_state["audio"]["cancel"] or audio_cancel_event.is_set()

    try:
        unloaded_voice_design = project_manager.unload_voice_design_model()
        if unloaded_voice_design:
            _append_audio_log(f"{job_prefix} Unloaded voice-design model state before audio generation.")
        else:
            _append_audio_log(f"{job_prefix} No loaded voice-design model state found before audio generation.")
    except BaseException as e:
        result_holder["error"] = f"Voice design model unload failed before audio generation: {e}"
        _append_audio_log(f"{job_prefix} {result_holder['error']}")
        done_event.set()
        return

    try:
        project_manager.register_audio_finalization_listener(
            run_token,
            submission_callback=submission_callback,
            item_callback=finalization_item_callback,
            activity_callback=activity_callback,
        )
        if not execution_uids:
            result_holder["results"] = {"completed": [], "failed": [], "cancelled": 0}
        elif job["kind"] == "parallel":
            effective_workers = _effective_parallel_workers(settings)
            if effective_workers != settings["workers"]:
                _append_audio_log(
                    f"{job_prefix} Using {effective_workers} worker for local MLX stability (configured {settings['workers']})"
                )
            result_holder["results"] = project_manager.generate_chunks_parallel(
                execution_uids,
                effective_workers,
                progress_callback,
                cancel_check=cancel_check,
                item_callback=item_callback,
                generation_token=run_token,
                item_started_callback=item_started_callback,
                neutral_narrator=bool(job.get("neutral_narrator", False)),
            )
        else:
            result_holder["results"] = project_manager.generate_chunks_batch(
                execution_uids,
                settings["batch_seed"],
                settings["batch_size"],
                progress_callback,
                batch_group_by_type=settings["batch_group_by_type"],
                cancel_check=cancel_check,
                item_callback=item_callback,
                generation_token=run_token,
                item_started_callback=item_started_callback,
                log_callback=log_callback,
                neutral_narrator=bool(job.get("neutral_narrator", False)),
            )
    except BaseException as e:
        result_holder["error"] = str(e)
    finally:
        done_event.set()


def _prepare_job_indices_for_execution_locked(job, settings):
    tts_cfg = settings.get("tts_cfg") or {}
    if tts_cfg.get("mode") != "external":
        return

    chunks = project_manager.get_chunks_by_uids(_job_uids(job))
    if not chunks:
        return

    reordered = project_manager.group_indices_by_resolved_speaker(_job_uids(job), chunks=chunks)
    if reordered == _job_uids(job):
        return

    pending_lookup = set(_job_generation_pending_uids(job))
    job["uids"] = reordered
    job["pending_uids"] = [uid for uid in reordered if uid in pending_lookup or uid in set(_job_pending_finalize_uids(job))]
    job["generation_pending_uids"] = [uid for uid in reordered if uid in pending_lookup]
    _append_audio_log_locked(
        f"[JOB {job['id']}] Reordered external job by speaker for clone/cache locality"
    )


def _audio_queue_worker():
    global audio_current_job, audio_recovery_request, audio_current_runner_thread, audio_current_runner_token

    while True:
        with audio_queue_condition:
            while not audio_queue:
                if audio_current_job is not None:
                    project_manager.unregister_audio_finalization_listener(audio_current_job.get("run_token"))
                audio_current_job = None
                audio_current_runner_thread = None
                audio_current_runner_token = None
                process_state["audio"]["cancel"] = False
                audio_cancel_event.clear()
                audio_recovery_request = None
                _refresh_audio_process_state_locked(persist=True)
                audio_queue_condition.wait()

            job = audio_queue.pop(0)
            audio_current_job = job
            job["status"] = "running"
            job["started_at"] = time.time()
            job["run_token"] = job.get("run_token") or uuid.uuid4().hex
            job.setdefault("generation_pending_uids", list(_job_pending_uids(job)))
            job.setdefault("pending_finalize_uids", [])
            job.setdefault("generation_finished", False)
            job.setdefault("finalized_clips", 0)
            job.setdefault("finalizer_failures", 0)
            job.setdefault("retry_uids", [])
            job["last_generation_activity_at"] = time.time()
            job["last_finalize_activity_at"] = None
            process_state["audio"]["cancel"] = False
            audio_cancel_event.clear()
            audio_recovery_request = None
            audio_current_runner_thread = None
            audio_current_runner_token = None
            _refresh_audio_process_state_locked(persist=True)

        settings = _load_audio_worker_settings()
        with audio_queue_condition:
            if audio_current_job is job:
                _prepare_job_indices_for_execution_locked(job, settings)
                _refresh_audio_process_state_locked(persist=True)
        job_prefix = f"[JOB {job['id']}|{job.get('corr_id', 'no-cid')}]"

        _append_audio_log(
            f"{job_prefix} Starting {job['kind']} generation for {job['total_chunks']} chunks ({job['label']})"
        )

        result_holder = {}
        done_event = threading.Event()
        runner = threading.Thread(
            target=_audio_job_runner,
            args=(job, settings, job["run_token"], result_holder, done_event),
            daemon=True,
            name=f"audio-job-{job['id']}",
        )
        runner.start()
        with audio_queue_condition:
            if audio_current_job is job and audio_current_job.get("run_token") == job.get("run_token"):
                audio_current_runner_thread = runner
                audio_current_runner_token = job.get("run_token")

        abandoned = False
        while not done_event.wait(AUDIO_RECOVERY_POLL_SECONDS):
            with audio_queue_condition:
                if (
                    audio_recovery_request is not None
                    and audio_recovery_request.get("job_id") == job["id"]
                    and audio_recovery_request.get("run_token") == job.get("run_token")
                ):
                    abandoned = _perform_audio_recovery_locked(job, job.get("run_token"), audio_recovery_request["reason"])
                    if abandoned:
                        break

        if abandoned:
            continue

        with audio_queue_condition:
            if audio_current_job is None or audio_current_job["id"] != job["id"] or audio_current_job.get("run_token") != job.get("run_token"):
                continue
            job["generation_finished"] = True
            if "error" in result_holder:
                job["generation_pending_uids"] = []
            _mark_audio_generation_activity_locked(job)
            _refresh_audio_process_state_locked(persist=True)

        while True:
            with audio_queue_condition:
                if audio_current_job is None or audio_current_job["id"] != job["id"] or audio_current_job.get("run_token") != job.get("run_token"):
                    abandoned = True
                    break
                if _audio_job_ready_to_finish_locked(job):
                    break
                if (
                    audio_recovery_request is not None
                    and audio_recovery_request.get("job_id") == job["id"]
                    and audio_recovery_request.get("run_token") == job.get("run_token")
                ):
                    abandoned = _perform_audio_recovery_locked(job, job.get("run_token"), audio_recovery_request["reason"])
                    break
                audio_queue_condition.wait(timeout=AUDIO_RECOVERY_POLL_SECONDS)

        if abandoned:
            project_manager.unregister_audio_finalization_listener(job.get("run_token"))
            continue

        with audio_queue_condition:
            if audio_current_job is None or audio_current_job["id"] != job["id"] or audio_current_job.get("run_token") != job.get("run_token"):
                project_manager.unregister_audio_finalization_listener(job.get("run_token"))
                continue

            results = result_holder.get("results", {"completed": [], "failed": [], "cancelled": 0})
            cancelled = results.get("cancelled", 0)
            if "error" in result_holder:
                logger.error(f"Audio queue job error: {result_holder['error']}")
                _append_audio_log_locked(f"{job_prefix} Batch generation error: {result_holder['error']}")
                for uid in list(_job_generation_pending_uids(job)):
                    word_count = _job_word_counts(job).get(uid, 0)
                    effective_elapsed = _consume_elapsed_seconds(uid, 0.0)
                    _record_audio_sample_locked(job, uid, effective_elapsed, word_count, 0, False)
                job["generation_pending_uids"] = []
                if not job.get("retry_uids"):
                    job["retry_uids"] = []
                job["status"] = "completed_with_errors"
            elif cancelled:
                job["status"] = "cancelled"
            elif int(job.get("error_clips", 0) or 0) > 0 or int(job.get("finalizer_failures", 0) or 0) > 0:
                job["status"] = "completed_with_errors"
            else:
                job["status"] = "completed"

            msg = (
                f"{job_prefix} Complete: {int(job.get('processed_clips', 0) or 0) - int(job.get('error_clips', 0) or 0)} succeeded, "
                f"{int(job.get('error_clips', 0) or 0)} failed"
            )
            if cancelled:
                msg += f", {cancelled} cancelled"
            _append_audio_log_locked(msg)
            if results["failed"]:
                for idx, err in results["failed"]:
                    _append_audio_log_locked(f"{job_prefix} Chunk {idx} failed: {err}")
            if job["status"] != "cancelled":
                _queue_follow_on_retry_job_locked(job)

            job["finished_at"] = time.time()
            _record_audio_recent_job_locked(job)
            project_manager.unregister_audio_finalization_listener(job.get("run_token"))
            audio_current_job = None
            audio_current_runner_thread = None
            audio_current_runner_token = None
            audio_recovery_request = None
            process_state["audio"]["cancel"] = False
            audio_cancel_event.clear()
            _refresh_audio_process_state_locked(persist=True)
            audio_queue_condition.notify_all()


audio_worker_thread = threading.Thread(target=_audio_queue_worker, daemon=True, name="audio-queue-worker")
audio_worker_thread.start()


def _audio_heartbeat_daemon():
    while True:
        time.sleep(AUDIO_HEARTBEAT_INTERVAL_SECONDS)
        with audio_queue_condition:
            heartbeat = process_state["audio"]["heartbeat"]
            heartbeat["last_check_at"] = time.time()
            current = audio_current_job
            if current is None:
                _refresh_audio_process_state_locked(persist=True)
                continue
            last_activity_at = max(
                value
                for value in [
                    current.get("last_output_at"),
                    current.get("last_generation_activity_at"),
                    current.get("last_finalize_activity_at"),
                    current.get("started_at"),
                ]
                if value is not None
            )
            if last_activity_at is None:
                _refresh_audio_process_state_locked(persist=True)
                continue
            idle_seconds = time.time() - last_activity_at
            if idle_seconds >= AUDIO_HEARTBEAT_INTERVAL_SECONDS:
                _request_audio_recovery_locked(
                    f"Job #{current['id']} produced no audio for {int(idle_seconds)}s; scheduling automatic recovery"
                )
            else:
                _refresh_audio_process_state_locked(persist=True)


audio_heartbeat_thread = threading.Thread(target=_audio_heartbeat_daemon, daemon=True, name="audio-heartbeat")
audio_heartbeat_thread.start()
_restore_audio_queue_state()

def run_process(command: List[str], task_name: str, run_id: str, relay_fn=None):
    """Run a subprocess and capture logs.

    relay_fn: optional callable(str) called for each logged line in addition
              to the task-specific log.  Used by workflow runners to mirror
              subprocess output into a combined workflow log stream.
    """
    logger.info(f"Starting task {task_name}: {' '.join(command)}")
    success = False

    def _log(message: str):
        _append_task_log(task_name, run_id, message)
        if relay_fn:
            try:
                relay_fn(message)
            except Exception:
                pass

    try:
        env = os.environ.copy()
        env.setdefault("PYTHONIOENCODING", "utf-8")
        env.setdefault("PYTHONUTF8", "1")
        env.setdefault("PYTHONUNBUFFERED", "1")
        env["THREADSPEAK_RUN_ID"] = str(run_id or "")
        env["THREADSPEAK_RUN_DIR"] = LAYOUT.run_dir(run_id)
        env["THREADSPEAK_RUN_TEMP_DIR"] = LAYOUT.run_temp_dir(run_id)
        env["THREADSPEAK_RUN_LOGS_DIR"] = LAYOUT.run_logs_dir(run_id)
        env["THREADSPEAK_RUN_EXPORTS_DIR"] = LAYOUT.run_exports_dir(run_id)
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            cwd=BASE_DIR,
            bufsize=1,
            universal_newlines=True,
            env=env,
        )
        if not _register_task_process(task_name, run_id, process):
            return

        for line in process.stdout:
            if not _task_is_current(task_name, run_id):
                try:
                    process.terminate()
                except Exception:
                    pass
                break
            log_line = line.strip()
            if log_line:
                if log_line.startswith(TASK_PROGRESS_PREFIX):
                    try:
                        progress = json.loads(log_line[len(TASK_PROGRESS_PREFIX):])
                        _set_task_progress(task_name, run_id, progress)
                        message = (progress or {}).get("message")
                        if message:
                            _log(str(message))
                    except (json.JSONDecodeError, ValueError, TypeError):
                        _log(log_line)
                    continue
                _log(log_line)

        process.wait()
        if not _task_is_current(task_name, run_id):
            return
        return_code = process.returncode

        if return_code == 0:
            _log(f"Task {task_name} completed successfully.")
            success = True
        else:
            _log(f"Task {task_name} failed with return code {return_code}.")

    except Exception as e:
        logger.error(f"Error running {task_name}: {e}")
        _log(f"Error: {str(e)}")
    finally:
        _finish_task_run(task_name, run_id, locals().get("process"))
    return success


def run_script_sanity_task(run_id: str, stop_check=None):
    success = False

    def ensure_active():
        if stop_check and stop_check():
            raise WorkflowPauseRequested()
        if not _task_is_current("sanity", run_id):
            raise WorkflowPauseRequested()

    def log(message: str):
        ensure_active()
        return _append_task_log("sanity", run_id, message)

    progress_state = {
        "prepared_logged": False,
        "last_logged_current": 0,
    }

    def on_attribution_progress(event: str, payload: dict):
        if stop_check and stop_check():
            return
        if not _task_is_current("sanity", run_id):
            return
        total = int(payload.get("total") or payload.get("candidates") or 0)
        current = int(payload.get("current") or 0)
        cache_hits = int(payload.get("cache_hits") or 0)
        queries = int(payload.get("queries") or 0)
        if event == "prepared":
            if total <= 0:
                log("Attribution check: no candidate omissions detected.")
            else:
                log(f"Attribution check: evaluating {total} candidate omissions.")
            progress_state["prepared_logged"] = True
            return
        if total <= 0:
            return
        should_log = current == total or current == 1 or (current - progress_state["last_logged_current"]) >= 10
        if should_log:
            decision = payload.get("decision")
            suffix = f", latest_decision={decision}" if decision else ""
            log(
                f"Attribution progress: {current}/{total} checked "
                f"(model_queries={queries}, cache_hits={cache_hits}{suffix})"
            )
            progress_state["last_logged_current"] = current

    def persist_attribution_decision(_phrase_key: str, _decision: dict, phrase_decisions: dict):
        if stop_check and stop_check():
            return
        if not _task_is_current("sanity", run_id):
            return
        project_manager.script_store.replace_script_document(
            entries=script_document.get("entries"),
            dictionary=script_document.get("dictionary", []),
            sanity_cache={"phrase_decisions": phrase_decisions},
            reason="script_sanity_progress",
            rebuild_chunks=False,
            wait=True,
        )

    try:
        ensure_active()
        if getattr(project_manager, "script_store", None) is not None:
            project_manager.script_store.delete_project_document(
                "script_sanity_result",
                reason="script_sanity_reset",
                wait=True,
            )

        ensure_active()
        if not _project_has_script_document():
            raise FileNotFoundError("No annotated script found. Generate a script first.")

        state_path = os.path.join(ROOT_DIR, "state.json")
        if not os.path.exists(state_path):
            raise FileNotFoundError("No source file selected.")

        with open(state_path, "r", encoding="utf-8") as f:
            state = json.load(f)
        input_file = state.get("input_file_path")
        if not input_file or not os.path.exists(input_file):
            raise FileNotFoundError("Original uploaded source could not be found.")

        config = {}
        if os.path.exists(CONFIG_PATH):
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                config = json.load(f)
        chunk_size = int(config.get("generation", {}).get("chunk_size", 3000))
        generation_max_tokens = int(config.get("generation", {}).get("max_tokens", 4096) or 4096)
        llm_config = config.get("llm", {})
        prompts_config = config.get("prompts", {})
        if not prompts_config.get("attribution_system_prompt") or not prompts_config.get("attribution_user_prompt"):
            try:
                attr_sys, attr_usr = load_attribution_prompts()
                prompts_config = {
                    **prompts_config,
                    "attribution_system_prompt": prompts_config.get("attribution_system_prompt") or attr_sys,
                    "attribution_user_prompt": prompts_config.get("attribution_user_prompt") or attr_usr,
                }
            except RuntimeError:
                pass

        log(f"Loading source document: {os.path.basename(input_file)}")
        source_document = load_source_document(input_file)
        script_document = _load_project_script_document()
        log(
            f"Comparing {len(source_document.get('chapters', []))} source chapter(s) "
            f"against {len(script_document.get('entries', []))} script entr{'y' if len(script_document.get('entries', [])) == 1 else 'ies'}"
        )

        attribution_resolver = None
        attribution_system_prompt = prompts_config.get("attribution_system_prompt")
        attribution_user_prompt = prompts_config.get("attribution_user_prompt")
        if attribution_system_prompt and attribution_user_prompt:
            ensure_active()
            llm_runtime = LLMRuntimeConfig.from_dict(
                llm_config,
                default_base_url="http://localhost:11434/v1",
                default_model_name="local-model",
                default_timeout=600.0,
            )
            client = _LLM_CLIENT_FACTORY.create_client(llm_runtime)
            attribution_resolver = build_attribution_classifier(
                client,
                llm_runtime.model_name,
                attribution_system_prompt,
                attribution_user_prompt,
                max_tokens=generation_max_tokens,
                runtime=llm_runtime,
            )

        result = run_script_sanity_check(
            source_document,
            script_document,
            chunk_size,
            attribution_resolver=attribution_resolver,
            known_phrase_decisions=(script_document.get("sanity_cache") or {}).get("phrase_decisions"),
            attribution_progress=on_attribution_progress,
            attribution_decision_persist=persist_attribution_decision,
        )
        if not _task_is_current("sanity", run_id):
            raise WorkflowPauseRequested()

        updated_sanity_cache = {
            "phrase_decisions": result.get("attribution_phrase_decisions", {}),
        }
        project_manager.script_store.replace_script_document(
            entries=script_document.get("entries"),
            dictionary=script_document.get("dictionary", []),
            sanity_cache=updated_sanity_cache,
            reason="script_sanity_complete",
            rebuild_chunks=False,
            wait=True,
        )
        project_manager.script_store.replace_project_document(
            "script_sanity_result",
            result,
            reason="script_sanity_complete",
            wait=True,
        )

        mismatched = [chapter for chapter in result["chapters"] if chapter["missing_words"] or chapter["inserted_words"]]
        if not mismatched:
            log("All chapters match the original source after normalization.")
        else:
            for chapter in mismatched:
                title = chapter.get("chapter_title") or chapter.get("source_title") or chapter.get("script_title") or f"Chapter {chapter.get('chapter_index')}"
                log(
                    f'Chapter "{title}": missing_words={chapter["missing_words"]}, '
                    f'inserted_words={chapter["inserted_words"]}, '
                    f'invalid_sections={len(chapter["invalid_sections"])}, '
                    f'invalid_chunks={chapter["invalid_chunk_count"]}'
                )
                for section_index, section in enumerate(chapter.get("invalid_sections") or [], start=1):
                    details = _format_invalid_text_details(
                        section.get("source_text") or "",
                        section.get("inserted_text") or "",
                    )
                    if details:
                        log(f"  section {section_index}: {details}")

        log(f"Dialogue-attribution candidates: {result['attribution_candidates']}")
        log(f"Dialogue-attribution cache hits: {result['attribution_cache_hits']}")
        log(f"Dialogue-attribution model queries: {result['attribution_model_queries']}")
        log(f"Dialogue-attribution sections pruned: {result['attribution_pruned_sections']}")
        log(f"Dialogue-attribution words pruned: {result['attribution_pruned_words']}")
        model_decisions = [d for d in (result.get("attribution_decisions") or []) if d.get("source") == "model"]
        if model_decisions:
            mode_labels = sorted({str(d.get("llm_mode") or "").strip() for d in model_decisions if str(d.get("llm_mode") or "").strip()})
            tool_observed_values = sorted({bool(d.get("llm_tool_call_observed")) for d in model_decisions})
            if mode_labels:
                log(f"Dialogue-attribution llm modes: {', '.join(mode_labels)}")
            log(f"Dialogue-attribution tool_call_observed values: {', '.join(str(v) for v in tool_observed_values)}")
        for decision in result.get("attribution_decisions") or []:
            if decision.get("decision") == "rejected":
                phrase = decision.get("source_text") or ""
                log(f'Confirmed genuine deletion: "{phrase}"')

        log(f"Missing words total: {result['missing_words']}")
        log(f"Inserted words total: {result['inserted_words']}")
        log(f"Invalid sections total: {result['invalid_section_count']}")
        log(f"Invalid chunks total: {result['invalid_chunk_count']}")
        log("Task sanity completed successfully.")
        success = True
    except WorkflowPauseRequested:
        logger.info("Sanity task interrupted")
    except Exception as e:
        logger.error(f"Error running script sanity check: {e}")
        if _task_is_current("sanity", run_id):
            log(f"Error: {str(e)}")
    finally:
        _finish_task_run("sanity", run_id)
    return success


def run_script_repair_task(run_id: str, stop_check=None):
    success = False

    def ensure_active():
        if stop_check and stop_check():
            raise WorkflowPauseRequested()
        if not _task_is_current("repair", run_id):
            raise WorkflowPauseRequested()

    def log(message: str):
        ensure_active()
        if not _append_task_log("repair", run_id, message):
            raise RepairSupersededError()

    try:
        ensure_active()
        _append_script_repair_trace(
            run_id,
            "repair_run_started",
            {"entry_count": len(_load_project_script_document().get("entries") or [])},
        )
        if getattr(project_manager, "script_store", None) is not None:
            project_manager.script_store.delete_project_document(
                "script_sanity_result",
                reason="repair_reset_sanity",
                wait=True,
            )

        result = repair_invalid_chunks(
            ROOT_DIR,
            log,
            should_continue=lambda: _task_is_current("repair", run_id) and not (stop_check and stop_check()),
            trace=lambda event_type, payload: _append_script_repair_trace(run_id, event_type, payload),
        )
        if not _task_is_current("repair", run_id):
            raise WorkflowPauseRequested()
        final_sanity = result["final_sanity"]

        project_manager.script_store.replace_project_document(
            "script_sanity_result",
            final_sanity,
            reason="repair_final_sanity",
            wait=True,
        )

        log(f"Initial invalid chunks: {result['initial_invalid_chunks']}")
        log(f"Initial missing words: {result['initial_missing_words']}")
        log(f"Initial inserted words: {result['initial_inserted_words']}")
        if result.get("repaired_headings"):
            log(f"Restored chapter headings from source metadata: {result['repaired_headings']}")
        log(f"Repair passes completed: {result['repaired_targets']}")

        if final_sanity["invalid_chunk_count"] == 0:
            log("Parity guarantee succeeded: script matches the source input.")
        else:
            log(
                "Parity guarantee incomplete: "
                f"outstanding_invalid_chunks={final_sanity['invalid_chunk_count']}, "
                f"missing_words={final_sanity['missing_words']}, "
                f"inserted_words={final_sanity['inserted_words']}"
            )

        log("Task repair completed successfully.")
        _append_script_repair_trace(
            run_id,
            "repair_run_completed",
            {
                "initial_invalid_chunks": int(result["initial_invalid_chunks"]),
                "initial_missing_words": int(result["initial_missing_words"]),
                "initial_inserted_words": int(result["initial_inserted_words"]),
                "repaired_targets": int(result["repaired_targets"]),
                "repaired_headings": int(result.get("repaired_headings") or 0),
                "failed_targets": int(result.get("failed_targets") or 0),
                "final_invalid_chunks": int(final_sanity["invalid_chunk_count"]),
                "final_missing_words": int(final_sanity["missing_words"]),
                "final_inserted_words": int(final_sanity["inserted_words"]),
            },
        )
        success = True
    except WorkflowPauseRequested:
        _append_script_repair_trace(run_id, "repair_run_interrupted", {})
        logger.info("Repair task interrupted")
    except RepairSupersededError:
        _append_script_repair_trace(run_id, "repair_run_superseded", {})
        logger.info("Repair task superseded by a newer request")
    except Exception as e:
        _append_script_repair_trace(run_id, "repair_run_error", {"error": str(e)})
        logger.error(f"Error running script repair: {e}")
        if _task_is_current("repair", run_id):
            log(f"Error: {str(e)}")
    finally:
        _finish_task_run("repair", run_id)
    return success


def run_lost_audio_repair_task(run_id: str, use_asr: bool):
    success = False
    last_progress_message = None

    def ensure_active():
        if not _task_is_current("repair", run_id):
            raise WorkflowPauseRequested()

    def log(message: str):
        ensure_active()
        if not _append_task_log("repair", run_id, message):
            raise RepairSupersededError()

    def on_progress(update: dict):
        nonlocal last_progress_message
        ensure_active()
        message = str((update or {}).get("message") or "").strip()
        if not message or message == last_progress_message:
            return
        last_progress_message = message
        log(message)

    try:
        log(f"Starting lost audio repair (ASR {'enabled' if use_asr else 'disabled'})...")
        result = project_manager.repair_lost_audio_links(use_asr=use_asr, progress_callback=on_progress)
        log(
            f"Relinked {result.get('relinked', 0)} clip(s) from exact transcript matches. "
            f"Processed {result.get('total_candidates', 0)} candidate file(s) and "
            f"recovered {result.get('discarded_retry_relinked', 0)} rejected clip(s)."
        )
        log(
            f"Discarded {result.get('unmatched_files', 0)} unmatched clip(s), "
            f"{result.get('invalid_candidates', 0)} ambiguous clip(s), and "
            f"{result.get('duplicate_matches', 0)} duplicate-target clip(s)."
        )
        for error in result.get("asr_errors", []) or []:
            log(f"ASR note: {error}")
        log("Task repair completed successfully.")
        success = True
    except WorkflowPauseRequested:
        logger.info("Lost audio repair task interrupted")
    except RepairSupersededError:
        logger.info("Lost audio repair task superseded by a newer request")
    except Exception as e:
        logger.error(f"Error running lost audio repair: {e}")
        if _task_is_current("repair", run_id):
            log(f"Error: {str(e)}")
    finally:
        _finish_task_run("repair", run_id)
    return success


def _build_assign_dialogue_command(
    config_path: str,
    *,
    full_cast: bool = True,
    retry_errors: Optional[int] = None,
) -> list[str]:
    command = [
        sys.executable,
        "-u",
        "-m",
        "scripts.assign_dialogue",
        "--project-root",
        ROOT_DIR,
        config_path,
    ]
    if not full_cast:
        command.append("--narrated")
    if retry_errors is not None:
        command.extend(["--retry-errors", str(max(0, int(retry_errors)))])
    return command


def _run_assign_dialogue_task(run_id: str, config_path: str, full_cast: bool = True):
    run_process(
        _build_assign_dialogue_command(config_path, full_cast=full_cast),
        "assign_dialogue",
        run_id,
    )


def _build_extract_temperament_command(config_path: str, *, retry_errors: Optional[int] = None) -> list[str]:
    command = [
        sys.executable,
        "-u",
        "-m",
        "scripts.extract_temperament",
        "--project-root",
        ROOT_DIR,
        config_path,
    ]
    if retry_errors is not None:
        command.extend(["--retry-errors", str(max(0, int(retry_errors)))])
    return command


def _run_extract_temperament_task(run_id: str, config_path: str):
    run_process(
        _build_extract_temperament_command(config_path),
        "extract_temperament",
        run_id,
    )


def _run_create_script_task(run_id: str):
    # ── Error correction: retry dialogue-error paragraphs before building script ──
    try:
        pdata = _load_project_paragraphs_document()
        dialogue_errors = pdata.get("dialogue_errors", []) if isinstance(pdata, dict) else []
    except Exception:
        dialogue_errors = []

    retry_attempts = 0
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        tts = cfg.get("tts", {})
        if tts.get("auto_regenerate_bad_clips", False):
            retry_attempts = max(0, int(tts.get("auto_regenerate_bad_clip_attempts", 3) or 0))
    except Exception:
        pass

    script_max_length = _load_script_max_length()

    if dialogue_errors and retry_attempts > 0:
        run_process(
            [sys.executable, "-u", "-m", "scripts.assign_dialogue",
             "--project-root", ROOT_DIR, CONFIG_PATH, "--retry-errors", str(retry_attempts)],
            "create_script",
            run_id,
        )

    # ── Build the script ──────────────────────────────────────────────────────
    success = run_process(
        [sys.executable, "-u", "-m", "scripts.create_script",
         "--project-root", ROOT_DIR,
         "--max-length", str(script_max_length)],
        "create_script",
        run_id,
    )
    if success:
        if hasattr(project_manager, "sync_missing_voice_profiles_from_chunks"):
            project_manager.sync_missing_voice_profiles_from_chunks(
                reason="create_script_seed_voice_profiles",
            )
        if hasattr(project_manager, "log_voice_audit_event"):
            project_manager.log_voice_audit_event(
                "create_script_voice_seed_complete",
                reason="create_script_seed_voice_profiles",
            )


def _load_script_max_length() -> int:
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        tts = cfg.get("tts", {})
        return int(tts.get("script_max_length", 250))
    except Exception:
        return 250


def _load_script_error_retry_attempts(config_path: Optional[str] = None) -> int:
    effective_config_path = str(config_path or CONFIG_PATH).strip() or CONFIG_PATH
    try:
        with open(effective_config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        generation = cfg.get("generation", {}) if isinstance(cfg, dict) else {}
        return max(0, int((generation or {}).get("script_error_retry_attempts", 3) or 0))
    except Exception:
        return 3


def _load_stage_residual_error_ids(stage_name: str) -> list[str]:
    try:
        pdata = _load_project_paragraphs_document()
    except Exception:
        return []
    if not isinstance(pdata, dict):
        return []
    if stage_name == "assign_dialogue":
        ids = [str(item).strip() for item in (pdata.get("dialogue_errors") or []) if str(item).strip()]
        if not ids:
            ids = [
                str(item.get("id") or "").strip()
                for item in (pdata.get("paragraphs") or [])
                if isinstance(item, dict) and item.get("dialogue_error") and str(item.get("id") or "").strip()
            ]
        return ids
    if stage_name == "extract_temperament":
        ids = []
        for key in ("temperament_errors", "dialogue_mood_errors"):
            ids.extend(str(item).strip() for item in (pdata.get(key) or []) if str(item).strip())
        if not ids:
            for item in (pdata.get("paragraphs") or []):
                if not isinstance(item, dict):
                    continue
                item_id = str(item.get("id") or "").strip()
                if item_id and (item.get("temperament_error") or item.get("dialogue_mood_error")):
                    ids.append(item_id)
        return sorted(set(ids))
    return []


def _maybe_run_new_mode_stage_error_heal(
    stage_name: str,
    *,
    config_path: str,
    run_id: str,
    relay,
) -> None:
    if stage_name not in {"assign_dialogue", "extract_temperament"}:
        return

    retry_attempts = _load_script_error_retry_attempts(config_path)
    if retry_attempts <= 0:
        return

    initial_error_ids = _load_stage_residual_error_ids(stage_name)
    if not initial_error_ids:
        return

    relay(
        f"Auto-heal: retrying {len(initial_error_ids)} residual {NEW_MODE_STAGE_LABELS.get(stage_name, stage_name)} error(s) "
        f"with up to {retry_attempts} attempt(s) each."
    )

    if stage_name == "assign_dialogue":
        full_cast = bool((process_state["new_mode_workflow"].get("options") or {}).get("full_cast", True))
        command = _build_assign_dialogue_command(
            config_path,
            full_cast=full_cast,
            retry_errors=retry_attempts,
        )
    else:
        command = _build_extract_temperament_command(config_path, retry_errors=retry_attempts)

    success = run_process(command, stage_name, run_id, relay_fn=relay)
    if not success:
        raise RuntimeError(
            f"{NEW_MODE_STAGE_LABELS.get(stage_name, stage_name)} automatic error-heal retry failed."
        )

    remaining_error_ids = _load_stage_residual_error_ids(stage_name)
    relay(
        f"Auto-heal complete for {NEW_MODE_STAGE_LABELS.get(stage_name, stage_name)}. "
        f"Remaining residual errors: {len(remaining_error_ids)}."
    )


def _run_process_paragraphs_task(run_id: str, input_file: str):
    run_process(
        [sys.executable, "-u", "-m", "scripts.process_paragraphs", input_file, "--project-root", ROOT_DIR],
        "process_paragraphs",
        run_id,
    )


def _run_generate_script_task(run_id: str):
    state = _load_project_state_payload()
    input_file = state.get("input_file_path")
    if not input_file:
        raise FileNotFoundError("No input file found in state")
    _clear_processing_stage_and_downstream("script")
    config_path = os.path.join(ROOT_DIR, "scripts", "config.json")

    legacy_mode = False
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        generation_cfg = cfg.get("generation", {}) if isinstance(cfg, dict) else {}
        legacy_mode = bool((generation_cfg or {}).get("legacy_mode", False))
    except Exception:
        # Preserve legacy behavior when config is unavailable.
        legacy_mode = True

    if legacy_mode:
        success = run_process(
            [sys.executable, "-u", "-m", "scripts.generate_script", input_file],
            "script",
            run_id,
        )
    else:
        success = run_process(
            [sys.executable, "-u", "-m", "scripts.process_paragraphs", input_file, "--project-root", ROOT_DIR],
            "script",
            run_id,
        )
        if success:
            success = run_process(
                [sys.executable, "-u", "-m", "scripts.assign_dialogue", "--project-root", ROOT_DIR, config_path],
                "script",
                run_id,
            )
        if success:
            success = run_process(
                [sys.executable, "-u", "-m", "scripts.extract_temperament", "--project-root", ROOT_DIR, config_path],
                "script",
                run_id,
            )
        if success:
            script_max_length = _load_script_max_length()
            success = run_process(
                [
                    sys.executable,
                    "-u",
                    "-m",
                    "scripts.create_script",
                    "--project-root",
                    ROOT_DIR,
                    "--max-length",
                    str(script_max_length),
                ],
                "script",
                run_id,
            )
            if success and hasattr(project_manager, "sync_missing_voice_profiles_from_chunks"):
                project_manager.sync_missing_voice_profiles_from_chunks(
                    reason="create_script_seed_voice_profiles",
                )
            if success and hasattr(project_manager, "log_voice_audit_event"):
                project_manager.log_voice_audit_event(
                    "create_script_voice_seed_complete",
                    reason="create_script_seed_voice_profiles",
                )

    if success:
        project_manager.reload_script_store()
        _mark_processing_stage_completed_marker("script")
    return success


def _run_review_script_task(run_id: str):
    _clear_processing_stage_and_downstream("review")
    success = run_process([sys.executable, "-u", "-m", "scripts.review_script"], "review", run_id)
    if success:
        _mark_processing_stage_completed_marker("review")
    return success


def _run_sanity_task(run_id: str, stop_check=None):
    _clear_processing_stage_and_downstream("sanity")
    success = run_script_sanity_task(run_id, stop_check=stop_check)
    if success:
        _mark_processing_stage_completed_marker("sanity")
    return success


def _run_repair_task(run_id: str, stop_check=None):
    _clear_processing_stage_and_downstream("repair")
    success = run_script_repair_task(run_id, stop_check=stop_check)
    if success:
        _mark_processing_stage_completed_marker("repair")
    return success


def _invoke_voice_processing_task(run_id: str, stop_check=None, relay_fn=None):
    """Route to the voices module implementation without exporting a conflicting symbol."""
    from .routers.voices_router import run_voice_processing_task as _run_voice_processing_task

    return _run_voice_processing_task(run_id, stop_check=stop_check, relay_fn=relay_fn)


def _run_voices_task(run_id: str, stop_check=None):
    _clear_processing_stage_and_downstream("voices")
    success = _invoke_voice_processing_task(run_id, stop_check=stop_check)
    if success:
        _mark_processing_stage_completed_marker("voices")
    return success


def _run_processing_script_stage():
    state_path = os.path.join(ROOT_DIR, "state.json")
    if not os.path.exists(state_path):
        raise FileNotFoundError("No input file selected")

    with open(state_path, "r", encoding="utf-8") as f:
        state = json.load(f)
    input_file = state.get("input_file_path")
    if not input_file:
        raise FileNotFoundError("No input file found in state")

    run_id = _start_task_run("script")
    return _run_generate_script_task(run_id)


def _run_processing_review_stage():
    if not _project_has_script_document():
        raise FileNotFoundError("No script found. Generate a script first.")
    run_id = _start_task_run("review")
    return _run_review_script_task(run_id)


def _run_processing_sanity_stage():
    if not _project_has_script_document():
        raise FileNotFoundError("No script found. Generate a script first.")
    run_id = _start_task_run("sanity")
    return _run_sanity_task(run_id, stop_check=_processing_workflow_is_pause_requested)


def _run_processing_repair_stage():
    if not _project_has_script_document():
        raise FileNotFoundError("No script found. Generate a script first.")
    run_id = _start_task_run("repair")
    return _run_repair_task(run_id, stop_check=_processing_workflow_is_pause_requested)


def _run_processing_voices_stage():
    run_id = _start_task_run("voices")
    return _run_voices_task(run_id, stop_check=_processing_workflow_is_pause_requested)


def _auto_prepare_segments_for_processing():
    if project_manager.is_render_prep_complete():
        return
    project_manager.merge_orphan_segments(chapter=None, min_words=10)
    project_manager.decompose_long_segments(chapter=None, max_words=25)
    project_manager.set_render_prep_complete(True)


def _workflow_pending_audio_indices():
    chunks = project_manager.load_chunks()
    return [
        chunk.get("id", index)
        for index, chunk in enumerate(chunks)
        if (chunk.get("text") or "").strip() and chunk.get("status") != "done"
    ]


def _run_processing_audio_stage():
    _ensure_processing_workflow_not_paused()
    _auto_prepare_segments_for_processing()

    with audio_queue_lock:
        _refresh_audio_process_state_locked()
        has_existing_audio_work = bool(audio_current_job is not None or audio_queue)

    if not has_existing_audio_work:
        indices = _workflow_pending_audio_indices()
        if not indices:
            with processing_workflow_lock:
                _append_processing_workflow_log_locked("No pending audio segments needed generation.")
            return True

        settings = _load_audio_worker_settings()
        kind = "parallel" if settings["tts_cfg"].get("mode") == "external" else "batch_fast"
        label = "Workflow audio generation"
        scope = "workflow"
        if kind == "parallel":
            _enqueue_audio_job(kind, indices, label=label, scope=scope)
        else:
            _enqueue_audio_job(kind, indices, label=label, scope=scope)

    while True:
        if _processing_workflow_is_pause_requested():
            _pause_audio_queue_for_workflow()
            raise WorkflowPauseRequested()
        with audio_queue_lock:
            _refresh_audio_process_state_locked()
            has_audio_work = bool(audio_current_job is not None or audio_queue)
        if not has_audio_work:
            break
        time.sleep(1)

    remaining = _workflow_pending_audio_indices()
    if remaining:
        raise RuntimeError(f"Audio generation stopped with {len(remaining)} pending segment(s) remaining.")
    _mark_processing_stage_completed_marker("audio")
    return True


def _execute_processing_workflow_stage(stage_name):
    with processing_workflow_lock:
        process_state["processing_workflow"]["current_stage"] = stage_name
        process_state["processing_workflow"]["last_error"] = None
        _append_processing_workflow_log_locked(
            f"Starting {PROCESSING_WORKFLOW_STAGE_LABELS.get(stage_name, stage_name)}..."
        )

    if stage_name == "script":
        success = _run_processing_script_stage()
    elif stage_name == "review":
        success = _run_processing_review_stage()
    elif stage_name == "sanity":
        success = _run_processing_sanity_stage()
    elif stage_name == "repair":
        success = _run_processing_repair_stage()
    elif stage_name == "voices":
        success = _run_processing_voices_stage()
    elif stage_name == "audio":
        success = _run_processing_audio_stage()
    else:
        raise RuntimeError(f"Unknown processing stage: {stage_name}")

    if _processing_workflow_is_pause_requested():
        raise WorkflowPauseRequested()
    if not success:
        raise RuntimeError(f"{PROCESSING_WORKFLOW_STAGE_LABELS.get(stage_name, stage_name)} failed.")

def _maybe_autosave_after_legacy_stage(stage_name: str):
    if stage_name == "repair":
        result = _autosave_current_script_for_workflow(
            purge_existing=False,
            trigger="legacy_after_repair",
        )
        with processing_workflow_lock:
            _append_processing_workflow_log_locked(
                f"Auto-saved script '{result['name']}' after Replace Missing Chunks."
            )
    elif stage_name == "voices":
        result = _autosave_current_script_for_workflow(
            purge_existing=True,
            trigger="legacy_after_voices",
        )
        with processing_workflow_lock:
            _append_processing_workflow_log_locked(
                f"Auto-saved script '{result['name']}' after Process Voices (before audio generation)."
            )


def _processing_workflow_runner():
    global processing_workflow_thread

    try:
        while True:
            with processing_workflow_lock:
                state = copy.deepcopy(process_state["processing_workflow"])
            if not state.get("running"):
                break

            stages = _processing_workflow_stage_sequence(state.get("options"))
            pending_stage = next((stage for stage in stages if stage not in (state.get("completed_stages") or [])), None)
            if pending_stage is None:
                _set_processing_workflow_completed()
                break

            try:
                _execute_processing_workflow_stage(pending_stage)
                _mark_processing_workflow_stage_complete(pending_stage)
                _maybe_autosave_after_legacy_stage(pending_stage)
            except WorkflowPauseRequested:
                _set_processing_workflow_paused(pending_stage)
                break
            except Exception as e:
                logger.error("Processing workflow failed during %s: %s", pending_stage, e)
                _set_processing_workflow_failed(
                    pending_stage,
                    f"{PROCESSING_WORKFLOW_STAGE_LABELS.get(pending_stage, pending_stage)} failed: {e}",
                )
                break
    finally:
        with processing_workflow_lock:
            processing_workflow_thread = None


def _start_processing_workflow_thread_locked():
    global processing_workflow_thread
    if processing_workflow_thread is not None and processing_workflow_thread.is_alive():
        return
    processing_workflow_thread = threading.Thread(
        target=_processing_workflow_runner,
        daemon=True,
        name="processing-workflow",
    )
    processing_workflow_thread.start()


def _restore_processing_workflow_state():
    if not os.path.exists(PROCESSING_WORKFLOW_STATE_PATH):
        return

    try:
        with open(PROCESSING_WORKFLOW_STATE_PATH, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except (OSError, json.JSONDecodeError, ValueError) as e:
        logger.warning(f"Failed to restore processing workflow state: {e}")
        return

    restored = _new_processing_workflow_state()
    restored.update(payload)
    process_state["processing_workflow"] = restored
    if _sanitize_restored_processing_workflow_state(process_state["processing_workflow"]):
        logger.info("Discarded stale legacy processing workflow pause snapshot during restore.")
        with processing_workflow_lock:
            _refresh_processing_workflow_updated_at_locked()
            _persist_processing_workflow_state_locked()

    if restored.get("running") and not restored.get("paused"):
        with processing_workflow_lock:
            process_state["processing_workflow"]["resume_count"] = int(
                process_state["processing_workflow"].get("resume_count", 0) or 0
            ) + 1
            _append_processing_workflow_log_locked("Recovered processing workflow after app restart.")
            _start_processing_workflow_thread_locked()


_restore_processing_workflow_state()


# ── New-mode workflow ──────────────────────────────────────────────────────────

def _new_mode_workflow_initial_state():
    return {
        "running": False,
        "paused": False,
        "pause_requested": False,
        "current_stage": None,
        "completed_stages": [],
        "options": {"process_voices": True, "generate_audio": False, "full_cast": True},
        "logs": [],
        "last_error": None,
        "started_at": None,
        "updated_at": None,
        "completed_at": None,
    }


def _new_mode_stage_sequence(options=None) -> list:
    options = options or process_state["new_mode_workflow"].get("options") or {}
    stages = ["process_paragraphs", "assign_dialogue", "extract_temperament", "create_script"]
    if options.get("process_voices", True):
        stages.append("process_voices")
    if options.get("generate_audio", False):
        stages.append("render_audio")
        stages.append("proofread")
    return stages


def _persist_new_mode_workflow_state_locked():
    _atomic_json_write(NEW_MODE_WORKFLOW_STATE_PATH, process_state["new_mode_workflow"])


def _append_new_mode_workflow_log_locked(message: str):
    process_state["new_mode_workflow"]["logs"].append(message)
    _trim_logs(process_state["new_mode_workflow"]["logs"])
    process_state["new_mode_workflow"]["updated_at"] = time.time()
    _persist_new_mode_workflow_state_locked()


def _new_mode_workflow_is_pause_requested() -> bool:
    with new_mode_workflow_lock:
        return bool(process_state["new_mode_workflow"].get("pause_requested"))


def _derived_new_mode_completed_stages(options=None) -> list:
    """Derive complete stages for new-mode workflow (marker-authoritative)."""
    options = options or {}
    allowed = set(_new_mode_stage_sequence(options))
    markers = _load_new_mode_stage_markers()
    return [stage for stage in NEW_MODE_STAGE_ORDER if stage in allowed and markers.get(stage)]


def _derived_new_mode_completed_stages_from_files(options=None) -> list:
    """Derive completed new-mode stages from persisted DB-backed project state."""
    options = options or {}
    allowed = set(_new_mode_stage_sequence(options))
    pdata = _load_project_paragraphs_document()

    completed = []
    if pdata.get("paragraphs"):
        completed.append("process_paragraphs")
    if pdata.get("dialogue_assignment_complete"):
        completed.append("assign_dialogue")
    if pdata.get("temperament_extraction_complete"):
        completed.append("extract_temperament")
    if _project_has_script_document() and bool(_load_project_chunks_snapshot()):
        completed.append("create_script")

    # process_voices: all voice entries in the runtime store have a ref_audio file
    if options.get("process_voices", True):
        try:
            vc = project_manager._load_voice_config()
            if isinstance(vc, dict) and vc:
                all_have_audio = all(
                    bool((v.get("ref_audio") or "").strip())
                    and os.path.exists(os.path.join(ROOT_DIR, v["ref_audio"].strip()))
                    for v in vc.values()
                    if isinstance(v, dict) and not (v.get("alias") or "").strip()
                )
                if all_have_audio:
                    completed.append("process_voices")
        except Exception:
            pass

    return [stage for stage in completed if stage in allowed]


def _project_script_complete_detected() -> bool:
    """Return True when a generated script project already exists in SQLite."""
    if not _project_has_script_document():
        return False

    try:
        for chunk in _load_project_chunks_snapshot():
            if not isinstance(chunk, dict):
                continue
            has_text = bool(str(chunk.get("text") or "").strip())
            has_speaker = bool(str(chunk.get("speaker") or "").strip())
            if has_text and has_speaker:
                return True
        return False
    except Exception:
        return False


def _initialize_new_mode_stage_markers(options=None, legacy_completed_stages=None):
    options = options or {}
    markers = _load_new_mode_stage_markers()
    if markers:
        return markers

    completed = []
    if legacy_completed_stages:
        completed = [stage for stage in legacy_completed_stages if stage in NEW_MODE_STAGE_ORDER]
    if not completed:
        completed = _derived_new_mode_completed_stages_from_files(options)

    if completed:
        now = time.time()
        migrated = {stage: {"completed_at": now} for stage in completed}
        _save_new_mode_stage_markers(migrated)
        return migrated
    return {}


def _set_new_mode_workflow_paused(stage_name=None):
    with new_mode_workflow_lock:
        process_state["new_mode_workflow"]["running"] = False
        process_state["new_mode_workflow"]["paused"] = True
        process_state["new_mode_workflow"]["pause_requested"] = False
        if stage_name is not None:
            process_state["new_mode_workflow"]["current_stage"] = stage_name
        _append_new_mode_workflow_log_locked(
            "Processing paused. Click 'Process Script' again to resume."
        )


def _set_new_mode_workflow_failed(stage_name: str, message: str):
    with new_mode_workflow_lock:
        process_state["new_mode_workflow"]["running"] = False
        process_state["new_mode_workflow"]["paused"] = False
        process_state["new_mode_workflow"]["pause_requested"] = False
        process_state["new_mode_workflow"]["current_stage"] = stage_name
        process_state["new_mode_workflow"]["last_error"] = message
        _append_new_mode_workflow_log_locked(f"ERROR: {message}")


def _set_new_mode_workflow_completed():
    with new_mode_workflow_lock:
        options = process_state["new_mode_workflow"].get("options") or {}
        process_state["new_mode_workflow"]["completed_stages"] = _derived_new_mode_completed_stages(options)
        process_state["new_mode_workflow"]["running"] = False
        process_state["new_mode_workflow"]["paused"] = False
        process_state["new_mode_workflow"]["pause_requested"] = False
        process_state["new_mode_workflow"]["current_stage"] = None
        process_state["new_mode_workflow"]["last_error"] = None
        process_state["new_mode_workflow"]["completed_at"] = time.time()
        if options.get("generate_audio"):
            _append_new_mode_workflow_log_locked(
                "All steps complete. Book is ready for export."
            )
        else:
            _append_new_mode_workflow_log_locked(
                "All steps complete. Script is ready in the Editor tab."
            )


def _run_new_mode_workflow_stage(stage_name: str):
    """Run a single new-mode stage as a subprocess, blocking until done.

    Raises WorkflowPauseRequested if pause was requested.
    Raises RuntimeError if the subprocess fails.
    """
    with new_mode_workflow_lock:
        process_state["new_mode_workflow"]["current_stage"] = stage_name
        process_state["new_mode_workflow"]["last_error"] = None
        _append_new_mode_workflow_log_locked(
            f"=== {NEW_MODE_STAGE_LABELS.get(stage_name, stage_name)} ==="
        )

    def relay(message: str):
        with new_mode_workflow_lock:
            _append_new_mode_workflow_log_locked(message)

    config_path = os.path.join(BASE_DIR, "config.json")

    if stage_name == "process_voices":
        # Voices task uses the "voices" task slot, not the stage name
        run_id = _start_task_run("voices")
        success = _invoke_voice_processing_task(
            run_id,
            stop_check=_new_mode_workflow_is_pause_requested,
            relay_fn=relay,
        )
    elif stage_name == "render_audio":
        # Audio generation uses the audio queue directly — no subprocess slot needed
        with audio_queue_lock:
            _refresh_audio_process_state_locked()
            has_existing_audio_work = bool(audio_current_job is not None or audio_queue)

        if not has_existing_audio_work:
            indices = _workflow_pending_audio_indices()
            if not indices:
                with new_mode_workflow_lock:
                    _append_new_mode_workflow_log_locked("No pending audio segments. Skipping render.")
                return
            settings = _load_audio_worker_settings()
            kind = "parallel" if settings["tts_cfg"].get("mode") == "external" else "batch_fast"
            _enqueue_audio_job(kind, indices, label="Workflow audio generation", scope="workflow")

        while True:
            if _new_mode_workflow_is_pause_requested():
                _pause_audio_queue_for_workflow()
                raise WorkflowPauseRequested()
            with audio_queue_lock:
                _refresh_audio_process_state_locked()
                has_audio_work = bool(audio_current_job is not None or audio_queue)
            if not has_audio_work:
                break
            time.sleep(1)

        remaining = _workflow_pending_audio_indices()
        if remaining:
            raise RuntimeError(
                f"Audio generation stopped with {len(remaining)} pending segment(s) remaining."
            )
        return
    else:
        run_id = _start_task_run(stage_name)
        if stage_name == "process_paragraphs":
            state = _load_project_state_payload()
            input_file = (state.get("input_file_path") or "").strip()
            if not input_file or not os.path.exists(input_file):
                raise RuntimeError("No input file found. Please upload a book first.")
            success = run_process(
                [sys.executable, "-u", "-m", "scripts.process_paragraphs", input_file, "--project-root", ROOT_DIR],
                stage_name, run_id, relay_fn=relay,
            )
        elif stage_name == "assign_dialogue":
            full_cast = bool((process_state["new_mode_workflow"].get("options") or {}).get("full_cast", True))
            success = run_process(
                _build_assign_dialogue_command(config_path, full_cast=full_cast),
                stage_name, run_id, relay_fn=relay,
            )
        elif stage_name == "extract_temperament":
            success = run_process(
                [sys.executable, "-u", "-m", "scripts.extract_temperament", "--project-root", ROOT_DIR, config_path],
                stage_name, run_id, relay_fn=relay,
            )
        elif stage_name == "create_script":
            script_max_length = _load_script_max_length()
            success = run_process(
                [sys.executable, "-u", "-m", "scripts.create_script",
                 "--project-root", ROOT_DIR,
                 "--max-length", str(script_max_length)],
                stage_name, run_id, relay_fn=relay,
            )
            if success and hasattr(project_manager, "sync_missing_voice_profiles_from_chunks"):
                project_manager.sync_missing_voice_profiles_from_chunks(
                    reason="new_mode_create_script_seed_voice_profiles",
                )
        elif stage_name == "proofread":
            success = run_process(
                [sys.executable, "-u", "-m", "scripts.proofread_runner", ROOT_DIR, "0.8", "__ALL__"],
                "proofread", run_id, relay_fn=relay,
            )
        else:
            raise RuntimeError(f"Unknown new-mode stage: {stage_name}")

        if success:
            _maybe_run_new_mode_stage_error_heal(
                stage_name,
                config_path=config_path,
                run_id=run_id,
                relay=relay,
            )

    if _new_mode_workflow_is_pause_requested():
        raise WorkflowPauseRequested()
    if not success:
        raise RuntimeError(
            f"{NEW_MODE_STAGE_LABELS.get(stage_name, stage_name)} failed."
        )

def _maybe_autosave_after_new_mode_stage(stage_name: str):
    if stage_name == "create_script":
        result = _autosave_current_script_for_workflow(
            purge_existing=False,
            trigger="new_mode_after_create_script",
        )
        with new_mode_workflow_lock:
            _append_new_mode_workflow_log_locked(
                f"Auto-saved script '{result['name']}' after Create Script."
            )
    elif stage_name == "process_voices":
        result = _autosave_current_script_for_workflow(
            purge_existing=True,
            trigger="new_mode_after_process_voices",
        )
        with new_mode_workflow_lock:
            _append_new_mode_workflow_log_locked(
                f"Auto-saved script '{result['name']}' after Process Voices (before audio generation)."
            )


def _new_mode_workflow_runner():
    global new_mode_workflow_thread
    try:
        while True:
            with new_mode_workflow_lock:
                state = dict(process_state["new_mode_workflow"])
            if not state.get("running"):
                break

            options = state.get("options") or {}
            completed = _derived_new_mode_completed_stages(options)
            with new_mode_workflow_lock:
                process_state["new_mode_workflow"]["completed_stages"] = completed
                _persist_new_mode_workflow_state_locked()
            pending = next(
                (s for s in _new_mode_stage_sequence(options) if s not in set(completed)),
                None,
            )
            if pending is None:
                _set_new_mode_workflow_completed()
                break

            try:
                _run_new_mode_workflow_stage(pending)
                with new_mode_workflow_lock:
                    _mark_new_mode_stage_completed_marker(pending)
                    done = _derived_new_mode_completed_stages(options)
                    process_state["new_mode_workflow"]["completed_stages"] = done
                    _append_new_mode_workflow_log_locked(
                        f"{NEW_MODE_STAGE_LABELS.get(pending, pending)} complete."
                    )
                _maybe_autosave_after_new_mode_stage(pending)
            except WorkflowPauseRequested:
                _set_new_mode_workflow_paused(pending)
                break
            except Exception as e:
                logger.error("New-mode workflow failed at %s: %s", pending, e)
                _set_new_mode_workflow_failed(pending, str(e))
                break
    finally:
        with new_mode_workflow_lock:
            new_mode_workflow_thread = None


def _start_new_mode_workflow_thread_locked():
    global new_mode_workflow_thread
    if new_mode_workflow_thread is not None and new_mode_workflow_thread.is_alive():
        return
    new_mode_workflow_thread = threading.Thread(
        target=_new_mode_workflow_runner,
        daemon=True,
        name="new-mode-workflow",
    )
    new_mode_workflow_thread.start()


def _restore_new_mode_workflow_state():
    if not os.path.exists(NEW_MODE_WORKFLOW_STATE_PATH):
        return
    try:
        with open(NEW_MODE_WORKFLOW_STATE_PATH, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except (OSError, json.JSONDecodeError, ValueError) as e:
        logger.warning("Failed to restore new mode workflow state: %s", e)
        return
    restored = _new_mode_workflow_initial_state()
    restored.update(payload)
    options = restored.get("options") or {}
    _initialize_new_mode_stage_markers(
        options=options,
        legacy_completed_stages=restored.get("completed_stages") or [],
    )
    if _project_script_complete_detected():
        script_complete_stages = ["process_paragraphs", "assign_dialogue", "extract_temperament", "create_script"]
        markers = _load_new_mode_stage_markers()
        now = time.time()
        for stage in script_complete_stages:
            markers.setdefault(stage, {"completed_at": now})
        _save_new_mode_stage_markers(markers)

        restored["options"] = {"process_voices": False, "generate_audio": False, "full_cast": True}
        restored["completed_stages"] = script_complete_stages
        restored["running"] = False
        restored["paused"] = False
        restored["pause_requested"] = False
        restored["current_stage"] = None
        restored["last_error"] = None
        restored["completed_at"] = restored.get("completed_at") or now
        process_state["new_mode_workflow"] = restored
        message = "Project script complete, Reset Project if you wish to begin generation from the beginning."
        logger.info(message)
        with new_mode_workflow_lock:
            _append_new_mode_workflow_log_locked(message)
        return

    restored["completed_stages"] = _derived_new_mode_completed_stages(options)
    process_state["new_mode_workflow"] = restored
    if restored.get("running") and not restored.get("paused"):
        stage_order = _new_mode_stage_sequence(options)
        completed = set(_derived_new_mode_completed_stages(options))
        pending = [stage for stage in stage_order if stage not in completed]

        if not pending:
            with new_mode_workflow_lock:
                process_state["new_mode_workflow"]["running"] = False
                process_state["new_mode_workflow"]["paused"] = False
                process_state["new_mode_workflow"]["pause_requested"] = False
                process_state["new_mode_workflow"]["current_stage"] = None
                process_state["new_mode_workflow"]["last_error"] = None
                process_state["new_mode_workflow"]["completed_at"] = (
                    process_state["new_mode_workflow"].get("completed_at") or time.time()
                )
                _append_new_mode_workflow_log_locked(
                    "Recovered workflow was already complete; no stages resumed."
                )
            return

        # Do not auto-resume into process_paragraphs unless an input file exists.
        if pending and pending[0] == "process_paragraphs":
            state = _load_project_state_payload()
            input_file = (state.get("input_file_path") or "").strip()
            if not input_file or not os.path.exists(input_file):
                with new_mode_workflow_lock:
                    process_state["new_mode_workflow"]["running"] = False
                    process_state["new_mode_workflow"]["paused"] = False
                    process_state["new_mode_workflow"]["last_error"] = (
                        "Recovered workflow requires an input file, but none was found. Upload a book and start Process Script."
                    )
                    _append_new_mode_workflow_log_locked(
                        "Recovery skipped: missing input file for Process Paragraphs."
                    )
                return

        with new_mode_workflow_lock:
            _append_new_mode_workflow_log_locked("Recovered workflow after app restart.")
            _start_new_mode_workflow_thread_locked()


_restore_new_mode_workflow_state()

# Endpoints
