import os
import sys
import gc
import copy
import json
import shutil
import logging
import asyncio
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Union
import re
import time
import threading
import zipfile
import tempfile
import subprocess
import aiofiles
import uuid
from openai import OpenAI

# Import ProjectManager
from project import ProjectManager
from asr import LocalASRUnavailableError
from default_prompts import DEFAULT_SYSTEM_PROMPT, DEFAULT_USER_PROMPT, load_default_prompts, DEFAULT_DIALOGUE_IDENTIFICATION_PROMPT, DEFAULT_TEMPERAMENT_EXTRACTION_PROMPT
from review_prompts import load_review_prompts
from attribution_prompts import load_attribution_prompts
from voice_prompt import load_voice_prompt
from hf_utils import fetch_builtin_manifest, download_builtin_adapter, is_adapter_downloaded
from script_store import apply_dictionary_to_text, clean_dictionary_entries, load_script_document, save_script_document
from source_document import load_source_document
from script_sanity import build_attribution_classifier, run_script_sanity_check
from script_repair import RepairSupersededError, repair_invalid_chunks

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("AlexandriaUI")

app = FastAPI(title="Alexandria Audiobook")

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
CONFIG_PATH = os.path.join(BASE_DIR, "config.json")
VOICES_PATH = os.path.join(ROOT_DIR, "voices.json")
VOICE_CONFIG_PATH = os.path.join(ROOT_DIR, "voice_config.json")
SCRIPT_PATH = os.path.join(ROOT_DIR, "annotated_script.json")
AUDIOBOOK_PATH = os.path.join(ROOT_DIR, "cloned_audiobook.mp3")
M4B_PATH = os.path.join(ROOT_DIR, "audiobook.m4b")
OPTIMIZED_EXPORT_PATH = os.path.join(ROOT_DIR, "optimized_audiobook.zip")
UPLOADS_DIR = os.path.join(BASE_DIR, "uploads")
SCRIPTS_DIR = os.path.join(ROOT_DIR, "scripts")
CHUNKS_PATH = os.path.join(ROOT_DIR, "chunks.json")
AUDIO_QUEUE_STATE_PATH = os.path.join(ROOT_DIR, "audio_queue_state.json")
SCRIPT_SANITY_PATH = os.path.join(ROOT_DIR, "script_sanity_check.json")
SCRIPT_REPAIR_TRACE_PATH = os.path.join(ROOT_DIR, "script_repair_trace.jsonl")
DESIGNED_VOICES_DIR = os.path.join(ROOT_DIR, "designed_voices")
CLONE_VOICES_DIR = os.path.join(ROOT_DIR, "clone_voices")
LORA_MODELS_DIR = os.path.join(ROOT_DIR, "lora_models")
LORA_DATASETS_DIR = os.path.join(ROOT_DIR, "lora_datasets")
BUILTIN_LORA_DIR = os.path.join(ROOT_DIR, "builtin_lora")
BUILTIN_LORA_MANIFEST = os.path.join(BUILTIN_LORA_DIR, "manifest.json")
DATASET_BUILDER_DIR = os.path.join(ROOT_DIR, "dataset_builder")
PROJECT_ARCHIVE_VERSION = 2
PROJECT_ARCHIVE_MANIFEST_NAME = "project_archive_manifest.json"
PROJECT_ARCHIVE_ALLOWED_FILES = {
    "annotated_script.json",
    "paragraphs.json",
    "voice_config.json",
    "voices.json",
    "chunks.json",
    "script_sanity_check.json",
    "state.json",
    "transcription_cache.json",
}
PROJECT_ARCHIVE_ALLOWED_DIRS = {
    "uploads",
    "clone_voices",
    "designed_voices",
    "voicelines",
}

os.makedirs(UPLOADS_DIR, exist_ok=True)
os.makedirs(SCRIPTS_DIR, exist_ok=True)
os.makedirs(DESIGNED_VOICES_DIR, exist_ok=True)
os.makedirs(CLONE_VOICES_DIR, exist_ok=True)
os.makedirs(LORA_MODELS_DIR, exist_ok=True)
os.makedirs(LORA_DATASETS_DIR, exist_ok=True)
os.makedirs(DATASET_BUILDER_DIR, exist_ok=True)


def _load_project_script_document():
    if not os.path.exists(SCRIPT_PATH):
        return {"entries": [], "dictionary": []}
    return load_script_document(SCRIPT_PATH)


def _load_project_dictionary_entries():
    return _load_project_script_document()["dictionary"]


def _apply_project_dictionary(text):
    return apply_dictionary_to_text(text, _load_project_dictionary_entries())[0]


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
    if not json_blob:
        return ""

    try:
        payload = json.loads(json_blob)
    except json.JSONDecodeError:
        return ""

    voice = payload.get("voice")
    return voice.strip() if isinstance(voice, str) else ""


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
VOICELINES_DIR = os.path.join(ROOT_DIR, "voicelines")
os.makedirs(VOICELINES_DIR, exist_ok=True)
app.mount("/voicelines", StaticFiles(directory=VOICELINES_DIR), name="voicelines")

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

class TTSConfig(BaseModel):
    mode: str = "local"  # "local" or "external"
    url: str = "http://127.0.0.1:7860"  # external mode only
    device: str = "auto"  # local mode: "auto", "cuda:0", "cpu", etc.
    language: str = "English"  # TTS language
    parallel_workers: int = 2  # concurrent TTS workers
    auto_regenerate_bad_clips: bool = False
    auto_regenerate_bad_clip_attempts: int = 3
    batch_seed: Optional[int] = None  # Single seed for batch mode, None/-1 = random
    compile_codec: bool = False  # torch.compile the codec for ~3-4x batch throughput (slow first run)
    sub_batch_enabled: bool = True  # split batch by text length to reduce padding waste
    sub_batch_min_size: int = 4  # minimum chunks per sub-batch before allowing a split
    sub_batch_ratio: float = 5.0  # max longest/shortest length ratio before splitting
    sub_batch_max_chars: int = 3000  # max total chars per sub-batch (lower for less VRAM)
    sub_batch_max_items: int = 0  # hard cap on sequences per sub-batch (0 = auto from VRAM estimate)
    batch_group_by_type: bool = False  # group chunks by voice type for efficient batching

class GenerationConfig(BaseModel):
    chunk_size: int = 3000
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

class ChunkUpdate(BaseModel):
    text: Optional[str] = None
    instruct: Optional[str] = None
    speaker: Optional[str] = None

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
    threshold: float = 1.0

class ProofreadClearFailuresRequest(BaseModel):
    chapter: Optional[str] = None
    threshold: float = 1.0

class ProofreadValidateRequest(BaseModel):
    threshold: float = 1.0

class ProofreadCompareRequest(BaseModel):
    threshold: float = 1.0

class ProofreadDiscardSelectionRequest(BaseModel):
    chapter: Optional[str] = None

class ASRTranscribeRequest(BaseModel):
    audio_path: str

class RenderPrepStateRequest(BaseModel):
    complete: bool = True

class BatchGenerateRequest(BaseModel):
    indices: List[Union[str, int]]
    label: Optional[str] = None
    scope: Optional[str] = None


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


class ScriptGenerationRequest(BaseModel):
    force_reimport: bool = False
    skip_import: bool = False


class WorkflowPauseRequested(Exception):
    pass

# Global state for process tracking
ROLLING_AUDIO_SAMPLE_LIMIT = 50
AUDIO_HEARTBEAT_INTERVAL_SECONDS = 600
AUDIO_RECOVERY_POLL_SECONDS = 5
PROCESSING_WORKFLOW_STATE_PATH = os.path.join(ROOT_DIR, "processing_workflow_state.json")
NEW_MODE_WORKFLOW_STATE_PATH = os.path.join(ROOT_DIR, "new_mode_workflow_state.json")
NEW_MODE_STAGE_LABELS = {
    "process_paragraphs": "Process Paragraphs",
    "assign_dialogue": "Assign Dialogue",
    "extract_temperament": "Extract Temperament",
    "create_script": "Create Script",
    "process_voices": "Process Voices",
    "render_audio": "Render Audio",
    "proofread": "Proofread",
}
PROCESSING_STAGE_ORDER = ["script", "review", "sanity", "repair", "voices", "audio"]
PROCESSING_STAGE_MARKERS_KEY = "processing_stage_markers"
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
        "options": {"process_voices": True},
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
audio_job_counter = 0
audio_recovery_request = None


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
        "last_recovery_at": heartbeat["last_recovery_at"],
        "recovery_count": heartbeat["recovery_count"],
        "last_recovery_reason": heartbeat["last_recovery_reason"],
    }


def _recompute_audio_metrics_locked():
    metrics = process_state["audio"]["metrics"]
    metrics["remaining_words"] = sum(job.get("remaining_words", 0) for job in audio_queue)
    if audio_current_job is not None:
        metrics["remaining_words"] += audio_current_job.get("remaining_words", 0)

    if metrics["rolling_seconds"] > 0 and metrics["rolling_output_words"] > 0:
        words_per_second = metrics["rolling_output_words"] / metrics["rolling_seconds"]
        metrics["words_per_minute"] = words_per_second * 60.0
        metrics["estimated_remaining_seconds"] = metrics["remaining_words"] / words_per_second if metrics["remaining_words"] > 0 else 0.0
    else:
        metrics["words_per_minute"] = None
        metrics["estimated_remaining_seconds"] = None if metrics["remaining_words"] > 0 else 0.0

    if metrics["processed_clips"] > 0:
        metrics["error_rate"] = metrics["error_clips"] / metrics["processed_clips"]
    else:
        metrics["error_rate"] = 0.0


def _serialize_audio_job(job):
    return {
        "id": job["id"],
        "kind": job["kind"],
        "status": job["status"],
        "label": job["label"],
        "scope": job["scope"],
        "indices": list(job["indices"]),
        "total_chunks": job["total_chunks"],
        "total_words": job.get("total_words", 0),
        "remaining_words": job.get("remaining_words", 0),
        "processed_clips": job.get("processed_clips", 0),
        "error_clips": job.get("error_clips", 0),
        "pending_indices": list(job.get("pending_indices", [])),
        "recovery_count": job.get("recovery_count", 0),
        "queued_at": job["queued_at"],
        "started_at": job.get("started_at"),
        "finished_at": job.get("finished_at"),
        "last_output_at": job.get("last_output_at"),
    }


def _atomic_json_write(path, data):
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
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


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


def _load_project_state_payload():
    state_path = os.path.join(ROOT_DIR, "state.json")
    if not os.path.exists(state_path):
        return {}
    try:
        with open(state_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except (OSError, json.JSONDecodeError, ValueError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _save_project_state_payload(state):
    _atomic_json_write(os.path.join(ROOT_DIR, "state.json"), state)


def _load_processing_stage_markers():
    state = _load_project_state_payload()
    markers = state.get(PROCESSING_STAGE_MARKERS_KEY)
    return dict(markers) if isinstance(markers, dict) else {}


def _save_processing_stage_markers(markers, state=None):
    payload = dict(state) if isinstance(state, dict) else _load_project_state_payload()
    cleaned = {stage: value for stage, value in dict(markers).items() if stage in PROCESSING_STAGE_ORDER}
    if cleaned:
        payload[PROCESSING_STAGE_MARKERS_KEY] = cleaned
    else:
        payload.pop(PROCESSING_STAGE_MARKERS_KEY, None)
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


def _clear_processing_stage_and_downstream(stage_name, include_self=True):
    if stage_name not in PROCESSING_STAGE_ORDER:
        return
    start_index = PROCESSING_STAGE_ORDER.index(stage_name) + (0 if include_self else 1)
    _clear_processing_stage_markers(PROCESSING_STAGE_ORDER[start_index:])


def _derived_processing_completed_stages(options=None):
    options = options or {}
    allowed = set(_processing_workflow_stage_sequence(options))
    markers = _load_processing_stage_markers()
    completed = []

    for stage_name in PROCESSING_STAGE_ORDER:
        if stage_name not in allowed:
            continue
        if stage_name == "script":
            if os.path.exists(SCRIPT_PATH):
                completed.append(stage_name)
            continue
        if stage_name == "review" and markers.get(stage_name) and os.path.exists(SCRIPT_PATH):
            completed.append(stage_name)
            continue
        if stage_name in ("sanity", "repair") and markers.get(stage_name) and os.path.exists(SCRIPT_PATH) and os.path.exists(SCRIPT_SANITY_PATH):
            completed.append(stage_name)
            continue
        if stage_name == "voices" and markers.get(stage_name) and os.path.exists(VOICE_CONFIG_PATH):
            completed.append(stage_name)
            continue
        if stage_name == "audio" and markers.get(stage_name) and os.path.exists(CHUNKS_PATH):
            completed.append(stage_name)

    return completed


def _chunk_chapter_summary():
    if not os.path.exists(CHUNKS_PATH):
        return {
            "chunk_count": 0,
            "chapter_count": 0,
            "last_chapter": None,
        }
    try:
        with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
            chunks = json.load(f)
    except (OSError, json.JSONDecodeError, ValueError):
        return {
            "chunk_count": 0,
            "chapter_count": 0,
            "last_chapter": None,
        }

    ordered_chapters = []
    last_seen = None
    for chunk in chunks if isinstance(chunks, list) else []:
        chapter = str((chunk or {}).get("chapter") or "").strip()
        if not chapter:
            continue
        if chapter != last_seen:
            ordered_chapters.append(chapter)
            last_seen = chapter

    return {
        "chunk_count": len(chunks) if isinstance(chunks, list) else 0,
        "chapter_count": len(ordered_chapters),
        "last_chapter": ordered_chapters[-1] if ordered_chapters else None,
    }


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
            os.remove(entry_path)


def _clear_project_derived_state(preserve_input_file=True):
    state = _load_project_state_payload()
    input_file_path = (state.get("input_file_path") or "").strip()

    files_to_remove = [
        SCRIPT_PATH,
        VOICES_PATH,
        VOICE_CONFIG_PATH,
        CHUNKS_PATH,
        project_manager.transcription_cache_path,
        AUDIOBOOK_PATH,
        M4B_PATH,
        AUDIO_QUEUE_STATE_PATH,
        PROCESSING_WORKFLOW_STATE_PATH,
        NEW_MODE_WORKFLOW_STATE_PATH,
        os.path.join(ROOT_DIR, "paragraphs.json"),
        SCRIPT_SANITY_PATH,
        os.path.join(ROOT_DIR, "audacity_export.zip"),
        os.path.join(ROOT_DIR, "m4b_cover.jpg"),
    ]
    for path in files_to_remove:
        if os.path.exists(path):
            os.remove(path)

    _clear_directory_contents(VOICELINES_DIR)
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
            audio_recovery_request = None
            _append_audio_log_locked(f"[WORKFLOW] Pause requested for audio job #{audio_current_job['id']}")
            if cleared:
                _append_audio_log_locked(f"[WORKFLOW] Removed {cleared} queued audio job(s)")
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


def _is_allowed_project_archive_path(path: str) -> bool:
    normalized = _normalize_archive_path(path)
    if not normalized or normalized == PROJECT_ARCHIVE_MANIFEST_NAME:
        return True
    if normalized in PROJECT_ARCHIVE_ALLOWED_FILES:
        return True
    first = normalized.split("/", 1)[0]
    return first in PROJECT_ARCHIVE_ALLOWED_DIRS


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


def _project_archive_entries():
    entries = {}

    def add_relative_path(relative_path: str):
        normalized = _normalize_archive_path(relative_path)
        if not normalized or not _is_allowed_project_archive_path(normalized):
            return
        absolute_path = os.path.join(ROOT_DIR, normalized)
        if os.path.exists(absolute_path):
            entries[normalized] = absolute_path

    for relative_path in sorted(PROJECT_ARCHIVE_ALLOWED_FILES):
        add_relative_path(relative_path)

    state = _archive_state_with_relative_paths()
    input_file_path = (state.get("input_file_path") or "").strip()
    if input_file_path:
        add_relative_path(input_file_path)

    chunks = []
    if os.path.exists(CHUNKS_PATH):
        try:
            with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
                chunks = json.load(f)
        except (json.JSONDecodeError, ValueError, OSError):
            chunks = []

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

    if os.path.exists(VOICE_CONFIG_PATH):
        try:
            with open(VOICE_CONFIG_PATH, "r", encoding="utf-8") as f:
                voice_config = json.load(f)
        except (json.JSONDecodeError, ValueError, OSError):
            voice_config = {}
        if isinstance(voice_config, dict):
            for config in voice_config.values():
                if not isinstance(config, dict):
                    continue
                ref_audio = (config.get("ref_audio") or "").strip()
                if ref_audio:
                    voice_assets.add(ref_audio)

    for relative_path in sorted(voice_assets):
        add_relative_path(relative_path)

    discarded_dir = os.path.join(ROOT_DIR, "voicelines", "discarded")
    if os.path.isdir(discarded_dir):
        for current_root, _, filenames in os.walk(discarded_dir):
            for filename in filenames:
                absolute_path = os.path.join(current_root, filename)
                relative_path = os.path.relpath(absolute_path, ROOT_DIR).replace(os.sep, "/")
                add_relative_path(relative_path)

    return sorted(entries.items())


def _build_project_archive_manifest(entries):
    return {
        "kind": "alexandria_project_archive",
        "version": PROJECT_ARCHIVE_VERSION,
        "created_at": time.time(),
        "entries": [relative_path for relative_path, _ in entries],
    }


def _clear_project_archive_targets():
    removable_files = [
        "annotated_script.json",
        "voice_config.json",
        "voices.json",
        "chunks.json",
        "script_sanity_check.json",
        "state.json",
        "transcription_cache.json",
        "cloned_audiobook.mp3",
        "optimized_audiobook.zip",
        "audacity_export.zip",
        "audiobook.m4b",
        "audio_queue_state.json",
        "processing_workflow_state.json",
    ]
    removable_dirs = ["uploads", "clone_voices", "designed_voices", "voicelines"]

    for relative_path in removable_files:
        absolute_path = os.path.join(ROOT_DIR, relative_path)
        if os.path.exists(absolute_path):
            os.remove(absolute_path)

    for dirname in removable_dirs:
        absolute_dir = os.path.join(ROOT_DIR, dirname)
        if os.path.isdir(absolute_dir):
            shutil.rmtree(absolute_dir)
        os.makedirs(absolute_dir, exist_ok=True)


def _reset_runtime_state_after_project_load():
    global audio_current_job, audio_recovery_request
    with audio_queue_condition:
        audio_queue.clear()
        audio_current_job = None
        audio_recovery_request = None
        process_state["audio"]["cancel"] = False
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


def _restore_project_archive(extracted_dir: str):
    _clear_project_archive_targets()

    for relative_path in sorted(PROJECT_ARCHIVE_ALLOWED_FILES):
        source_path = os.path.join(extracted_dir, relative_path)
        target_path = os.path.join(ROOT_DIR, relative_path)
        if not os.path.exists(source_path):
            continue
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        if relative_path == "state.json":
            with open(source_path, "r", encoding="utf-8") as f:
                state = json.load(f)
            input_file_path = (state.get("input_file_path") or "").strip()
            if input_file_path:
                state["input_file_path"] = os.path.join(ROOT_DIR, input_file_path)
            with open(target_path, "w", encoding="utf-8") as f:
                json.dump(state, f, indent=2, ensure_ascii=False)
        else:
            shutil.copy2(source_path, target_path)

    for dirname in sorted(PROJECT_ARCHIVE_ALLOWED_DIRS):
        source_dir = os.path.join(extracted_dir, dirname)
        target_dir = os.path.join(ROOT_DIR, dirname)
        if not os.path.isdir(source_dir):
            continue
        if os.path.isdir(target_dir):
            shutil.rmtree(target_dir)
        shutil.copytree(source_dir, target_dir)

    _reset_runtime_state_after_project_load()


def _persist_audio_queue_state_locked():
    payload = {
        "job_counter": audio_job_counter,
        "queue": [_serialize_audio_job(job) | {"word_counts": job.get("word_counts", {})} for job in audio_queue],
        "current_job": (_serialize_audio_job(audio_current_job) | {"word_counts": audio_current_job.get("word_counts", {})}) if audio_current_job else None,
        "metrics": _format_audio_metrics_locked(),
        "heartbeat": _format_audio_heartbeat_locked(),
        "updated_at": time.time(),
    }
    _atomic_json_write(AUDIO_QUEUE_STATE_PATH, payload)


def _refresh_audio_process_state_locked(persist=False):
    process_state["audio"]["queue"] = [_serialize_audio_job(job) for job in audio_queue]
    process_state["audio"]["current_job"] = _serialize_audio_job(audio_current_job) if audio_current_job else None
    process_state["audio"]["running"] = audio_current_job is not None or process_state["audio"].get("merge_running", False)
    process_state["audio"]["metrics"] = _format_audio_metrics_locked()
    process_state["audio"]["heartbeat"] = _format_audio_heartbeat_locked()
    process_state["audio"]["merge_progress"] = dict(process_state["audio"].get("merge_progress") or _new_audio_merge_progress())
    if persist:
        _persist_audio_queue_state_locked()


def _append_audio_log(message):
    with audio_queue_lock:
        _append_audio_log_locked(message)


def _append_audio_log_locked(message):
    process_state["audio"]["logs"].append(message)
    _trim_logs(process_state["audio"]["logs"])


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
    process_state["audio"]["recent_jobs"].insert(0, _serialize_audio_job(job))
    del process_state["audio"]["recent_jobs"][10:]


def _record_audio_sample_locked(job, chunk_index, elapsed_seconds, input_words, output_words, success):
    metrics = process_state["audio"]["metrics"]
    # Older refresh paths may have serialized metrics without the rolling sample
    # buffer. Recreate it so timing estimates continue updating instead of
    # breaking on the next queue refresh.
    metrics.setdefault("samples", [])
    now = time.time()
    sample = {
        "job_id": job["id"],
        "chunk_index": chunk_index,
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
    if chunk_index in job.get("pending_indices", []):
        job["pending_indices"].remove(chunk_index)
    if not success:
        job["error_clips"] = job.get("error_clips", 0) + 1
    job["remaining_words"] = max(0, job.get("remaining_words", 0) - sample["input_words"])
    job["last_output_at"] = now
    process_state["audio"]["heartbeat"]["last_output_at"] = now
    _recompute_audio_metrics_locked()


def _restore_job_progress_from_chunks(raw_job, chunks):
    indices = [int(idx) for idx in raw_job.get("indices", [])]
    word_counts = {int(k): int(v) for k, v in (raw_job.get("word_counts") or {}).items()}

    reconciled_indices = [idx for idx in indices if 0 <= idx < len(chunks)]
    pending_indices = []
    processed_clips = 0
    error_clips = 0

    for idx in reconciled_indices:
        status = chunks[idx].get("status")
        if status == "done":
            processed_clips += 1
        elif status == "error":
            processed_clips += 1
            error_clips += 1
        else:
            pending_indices.append(idx)

    total_words = sum(word_counts.get(idx, 0) for idx in reconciled_indices)
    remaining_words = sum(word_counts.get(idx, 0) for idx in pending_indices)

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


def _seed_audio_metrics_from_jobs_locked(*jobs):
    metrics = _new_audio_metrics()
    for job in jobs:
        if not job:
            continue
        metrics["processed_clips"] += int(job.get("processed_clips", 0) or 0)
        metrics["error_clips"] += int(job.get("error_clips", 0) or 0)
    metrics["successful_clips"] = metrics["processed_clips"] - metrics["error_clips"]
    process_state["audio"]["metrics"] = metrics
    _recompute_audio_metrics_locked()


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


def _enqueue_audio_job(kind, indices, label=None, scope=None):
    global audio_job_counter

    with audio_queue_condition:
        audio_job_counter += 1
        if audio_current_job is None and not audio_queue:
            process_state["audio"]["logs"] = []
            process_state["audio"]["recent_jobs"] = []
            process_state["audio"]["metrics"] = _new_audio_metrics()

        chunks = project_manager.load_chunks()
        valid_indices = []
        word_counts = {}
        for chunk_ref in indices:
            idx = project_manager.resolve_chunk_index(chunk_ref, chunks)
            if idx is not None and 0 <= idx < len(chunks):
                text = chunks[idx].get("text", "")
                if text and text.strip():
                    valid_indices.append(idx)
                    word_counts[idx] = _count_words(text)
        if not valid_indices:
            raise HTTPException(status_code=400, detail="No non-empty chunk indices provided")

        job = {
            "id": audio_job_counter,
            "kind": kind,
            "indices": valid_indices,
            "pending_indices": list(valid_indices),
            "word_counts": word_counts,
            "total_chunks": len(valid_indices),
            "total_words": sum(word_counts.values()),
            "remaining_words": sum(word_counts.values()),
            "processed_clips": 0,
            "error_clips": 0,
            "recovery_count": 0,
            "label": label or f"Audio Job {audio_job_counter}",
            "scope": scope or "custom",
            "status": "queued",
            "queued_at": time.time(),
            "started_at": None,
            "finished_at": None,
            "last_output_at": None,
            "run_token": None,
        }
        audio_queue.append(job)
        _append_audio_log_locked(
            f"[QUEUE] Job #{job['id']} queued: {job['label']} ({job['total_chunks']} chunks, scope={job['scope']})"
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


def _clone_audio_job_for_retry(job, pending_indices, reason):
    global audio_job_counter

    pending_indices = [idx for idx in pending_indices if idx in job.get("word_counts", {})]
    if not pending_indices:
        return None

    audio_job_counter += 1
    word_counts = {idx: job["word_counts"][idx] for idx in pending_indices if idx in job["word_counts"]}
    retry_count = job.get("recovery_count", 0) + 1
    return {
        "id": audio_job_counter,
        "kind": job["kind"],
        "indices": list(pending_indices),
        "pending_indices": list(pending_indices),
        "word_counts": word_counts,
        "total_chunks": len(pending_indices),
        "total_words": sum(word_counts.values()),
        "remaining_words": sum(word_counts.values()),
        "processed_clips": 0,
        "error_clips": 0,
        "recovery_count": retry_count,
        "label": f"{job['label']} (resume {retry_count})",
        "scope": job["scope"],
        "status": "queued",
        "queued_at": time.time(),
        "started_at": None,
        "finished_at": None,
        "last_output_at": None,
        "run_token": None,
        "resume_of": job["id"],
        "resume_reason": reason,
    }


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

    with audio_queue_condition:
        project_manager.recover_interrupted_generating_chunks()
        chunks = project_manager.load_chunks()
        process_state["audio"]["metrics"] = _new_audio_metrics()
        process_state["audio"]["heartbeat"] = _new_audio_heartbeat_state()
        process_state["audio"]["logs"] = []
        process_state["audio"]["recent_jobs"] = []
        audio_job_counter = max(audio_job_counter, int(payload.get("job_counter", 0) or 0))

        saved_heartbeat = payload.get("heartbeat") or {}
        process_state["audio"]["heartbeat"]["last_check_at"] = saved_heartbeat.get("last_check_at")
        process_state["audio"]["heartbeat"]["last_output_at"] = saved_heartbeat.get("last_output_at")
        process_state["audio"]["heartbeat"]["last_recovery_at"] = saved_heartbeat.get("last_recovery_at")
        process_state["audio"]["heartbeat"]["recovery_count"] = int(saved_heartbeat.get("recovery_count", 0) or 0)
        process_state["audio"]["heartbeat"]["last_recovery_reason"] = saved_heartbeat.get("last_recovery_reason")

        restored_jobs = []
        for raw_job in payload.get("queue", []):
            progress = _restore_job_progress_from_chunks(raw_job, chunks)
            if not progress["pending_indices"]:
                continue
            restored_jobs.append({
                "id": int(raw_job.get("id", 0) or 0),
                "kind": raw_job.get("kind", "parallel"),
                "indices": progress["indices"],
                "pending_indices": progress["pending_indices"],
                "word_counts": progress["word_counts"],
                "total_chunks": progress["total_chunks"],
                "total_words": progress["total_words"],
                "remaining_words": progress["remaining_words"],
                "processed_clips": progress["processed_clips"],
                "error_clips": progress["error_clips"],
                "recovery_count": int(raw_job.get("recovery_count", 0) or 0),
                "label": raw_job.get("label", "Recovered audio job"),
                "scope": raw_job.get("scope", "custom"),
                "status": "queued",
                "queued_at": time.time(),
                "started_at": None,
                "finished_at": None,
                "last_output_at": None,
                "run_token": None,
            })

        raw_current = payload.get("current_job")
        resumed_job = None
        if raw_current:
            progress = _restore_job_progress_from_chunks(raw_current, chunks)
            if progress["pending_indices"]:
                resumed_job = {
                    "id": int(raw_current.get("id", 0) or 0),
                    "kind": raw_current.get("kind", "parallel"),
                    "indices": progress["indices"],
                    "pending_indices": progress["pending_indices"],
                    "word_counts": progress["word_counts"],
                    "total_chunks": progress["total_chunks"],
                    "total_words": progress["total_words"],
                    "remaining_words": progress["remaining_words"],
                    "processed_clips": progress["processed_clips"],
                    "error_clips": progress["error_clips"],
                    "recovery_count": int(raw_current.get("recovery_count", 0) or 0) + 1,
                    "label": f"{raw_current.get('label', 'Recovered audio job')} (resumed after restart)",
                    "scope": raw_current.get("scope", "custom"),
                    "status": "queued",
                    "queued_at": time.time(),
                    "started_at": None,
                    "finished_at": None,
                    "last_output_at": None,
                    "run_token": None,
                }
                restored_jobs.insert(0, resumed_job)
                _append_audio_log_locked(
                    f"[RECOVER] Restored interrupted job from disk with {len(progress['pending_indices'])} pending chunk(s)"
                )

        audio_queue[:] = restored_jobs
        _seed_audio_metrics_from_jobs_locked(*restored_jobs)
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

    pending_indices = list(job.get("pending_indices", []))
    project_manager.reset_generating_chunks(indices=job.get("indices"), generation_token=run_token)

    job["status"] = "stalled"
    job["finished_at"] = time.time()
    _record_audio_recent_job_locked(job)

    heartbeat = process_state["audio"]["heartbeat"]
    heartbeat["last_recovery_at"] = job["finished_at"]
    heartbeat["last_recovery_reason"] = reason
    heartbeat["recovery_count"] += 1

    retry_job = _clone_audio_job_for_retry(job, pending_indices, reason)
    if retry_job is not None:
        audio_queue.insert(0, retry_job)
        _append_audio_log_locked(
            f"[RECOVER] Re-queued {len(retry_job['indices'])} stalled chunk(s) from job #{job['id']} to the front of the queue"
        )
    else:
        _append_audio_log_locked(f"[RECOVER] No remaining chunks to re-queue for job #{job['id']}")

    audio_current_job = None
    process_state["audio"]["cancel"] = False
    audio_recovery_request = None
    _refresh_audio_process_state_locked(persist=True)
    audio_queue_condition.notify_all()
    return True


def _abandon_audio_job_locked(job, run_token, reason, *, status="cancelled"):
    global audio_current_job, audio_recovery_request

    if audio_current_job is None:
        return False
    if audio_current_job["id"] != job["id"] or audio_current_job.get("run_token") != run_token:
        return False

    reset_count = project_manager.reset_generating_chunks(
        indices=job.get("indices"),
        generation_token=run_token,
    )
    job["status"] = status
    job["finished_at"] = time.time()
    _record_audio_recent_job_locked(job)
    _append_audio_log_locked(
        f"[CANCEL] Abandoned job #{job['id']} and reset {reset_count} generating chunk(s) to pending"
    )

    audio_current_job = None
    audio_recovery_request = None
    process_state["audio"]["cancel"] = False
    _refresh_audio_process_state_locked(persist=True)
    audio_queue_condition.notify_all()
    return True


def _audio_job_runner(job, settings, run_token, result_holder, done_event):
    job_prefix = f"[JOB {job['id']}]"

    def is_active():
        with audio_queue_lock:
            return (
                audio_current_job is not None
                and audio_current_job["id"] == job["id"]
                and audio_current_job.get("run_token") == run_token
            )

    def progress_callback(completed, failed, total):
        if is_active():
            _append_audio_log(
                f"{job_prefix} Progress: {completed + failed}/{total} ({completed} done, {failed} failed)"
            )

    def item_callback(idx, success, elapsed_seconds, input_words, output_words):
        with audio_queue_lock:
            if not is_active():
                return
            _record_audio_sample_locked(job, idx, elapsed_seconds, input_words, output_words, success)
            _refresh_audio_process_state_locked(persist=True)

    def cancel_check():
        with audio_queue_lock:
            if not is_active():
                return True
            return process_state["audio"]["cancel"]

    try:
        if job["kind"] == "parallel":
            result_holder["results"] = project_manager.generate_chunks_parallel(
                job["indices"],
                settings["workers"],
                progress_callback,
                cancel_check=cancel_check,
                item_callback=item_callback,
                generation_token=run_token,
            )
        else:
            result_holder["results"] = project_manager.generate_chunks_batch(
                job["indices"],
                settings["batch_seed"],
                settings["batch_size"],
                progress_callback,
                batch_group_by_type=settings["batch_group_by_type"],
                cancel_check=cancel_check,
                item_callback=item_callback,
                generation_token=run_token,
            )
    except Exception as e:
        result_holder["error"] = str(e)
    finally:
        done_event.set()


def _prepare_job_indices_for_execution_locked(job, settings):
    tts_cfg = settings.get("tts_cfg") or {}
    if tts_cfg.get("mode") != "external":
        return

    chunks = project_manager.load_chunks()
    if not chunks:
        return

    reordered = project_manager.group_indices_by_resolved_speaker(job.get("indices", []), chunks=chunks)
    if reordered == job.get("indices", []):
        return

    pending_lookup = set(job.get("pending_indices", []))
    job["indices"] = reordered
    job["pending_indices"] = [idx for idx in reordered if idx in pending_lookup]
    _append_audio_log_locked(
        f"[JOB {job['id']}] Reordered external job by speaker for clone/cache locality"
    )


def _audio_queue_worker():
    global audio_current_job, audio_recovery_request

    while True:
        with audio_queue_condition:
            while not audio_queue:
                audio_current_job = None
                process_state["audio"]["cancel"] = False
                audio_recovery_request = None
                _refresh_audio_process_state_locked(persist=True)
                audio_queue_condition.wait()

            job = audio_queue.pop(0)
            audio_current_job = job
            job["status"] = "running"
            job["started_at"] = time.time()
            job["run_token"] = uuid.uuid4().hex
            process_state["audio"]["cancel"] = False
            audio_recovery_request = None
            _refresh_audio_process_state_locked(persist=True)

        settings = _load_audio_worker_settings()
        with audio_queue_condition:
            if audio_current_job is job:
                _prepare_job_indices_for_execution_locked(job, settings)
                _refresh_audio_process_state_locked(persist=True)
        job_prefix = f"[JOB {job['id']}]"

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

            if "error" in result_holder:
                logger.error(f"Audio queue job error: {result_holder['error']}")
                job["status"] = "failed"
                _append_audio_log_locked(f"{job_prefix} Batch generation error: {result_holder['error']}")
            else:
                results = result_holder.get("results", {"completed": [], "failed": [], "cancelled": 0})
                completed = len(results["completed"])
                failed = len(results["failed"])
                cancelled = results.get("cancelled", 0)

                if cancelled:
                    job["status"] = "cancelled"
                elif failed:
                    job["status"] = "completed_with_errors"
                else:
                    job["status"] = "completed"

                msg = f"{job_prefix} Complete: {completed} succeeded, {failed} failed"
                if cancelled:
                    msg += f", {cancelled} cancelled"
                _append_audio_log_locked(msg)
                if results["failed"]:
                    for idx, err in results["failed"]:
                        _append_audio_log_locked(f"{job_prefix} Chunk {idx} failed: {err}")

            job["finished_at"] = time.time()
            _record_audio_recent_job_locked(job)
            audio_current_job = None
            audio_recovery_request = None
            process_state["audio"]["cancel"] = False
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
            last_activity_at = current.get("last_output_at") or current.get("started_at")
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
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
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
        save_script_document(
            SCRIPT_PATH,
            entries=script_document.get("entries"),
            dictionary=script_document.get("dictionary", []),
            sanity_cache={"phrase_decisions": phrase_decisions},
        )

    try:
        ensure_active()
        if os.path.exists(SCRIPT_SANITY_PATH):
            os.remove(SCRIPT_SANITY_PATH)

        ensure_active()
        if not os.path.exists(SCRIPT_PATH):
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
            client = OpenAI(
                base_url=llm_config.get("base_url", "http://localhost:11434/v1"),
                api_key=llm_config.get("api_key", "local"),
                timeout=float(llm_config.get("timeout", 600)),
            )
            attribution_resolver = build_attribution_classifier(
                client,
                llm_config.get("model_name", "local-model"),
                attribution_system_prompt,
                attribution_user_prompt,
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
        save_script_document(
            SCRIPT_PATH,
            entries=script_document.get("entries"),
            dictionary=script_document.get("dictionary", []),
            sanity_cache=updated_sanity_cache,
        )

        with open(SCRIPT_SANITY_PATH, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

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
        _append_script_repair_trace(run_id, "repair_run_started", {"script_path": SCRIPT_PATH})
        if os.path.exists(SCRIPT_SANITY_PATH):
            os.remove(SCRIPT_SANITY_PATH)

        result = repair_invalid_chunks(
            ROOT_DIR,
            log,
            should_continue=lambda: _task_is_current("repair", run_id) and not (stop_check and stop_check()),
            trace=lambda event_type, payload: _append_script_repair_trace(run_id, event_type, payload),
        )
        if not _task_is_current("repair", run_id):
            raise WorkflowPauseRequested()
        final_sanity = result["final_sanity"]

        with open(SCRIPT_SANITY_PATH, "w", encoding="utf-8") as f:
            json.dump(final_sanity, f, indent=2, ensure_ascii=False)

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


def _run_assign_dialogue_task(run_id: str, paragraphs_path: str, config_path: str):
    run_process(
        [sys.executable, "-u", "assign_dialogue.py", paragraphs_path, config_path],
        "assign_dialogue",
        run_id,
    )


def _run_extract_temperament_task(run_id: str, paragraphs_path: str, config_path: str):
    run_process(
        [sys.executable, "-u", "extract_temperament.py", paragraphs_path, config_path],
        "extract_temperament",
        run_id,
    )


def _run_create_script_task(run_id: str, paragraphs_path: str, voice_config_path: str,
                            script_output_path: str, chunks_output_path: str):
    # ── Error correction: retry dialogue-error paragraphs before building script ──
    try:
        with open(paragraphs_path, "r", encoding="utf-8") as f:
            pdata = json.load(f)
        dialogue_errors = pdata.get("dialogue_errors", [])
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

    if dialogue_errors and retry_attempts > 0:
        run_process(
            [sys.executable, "-u", "assign_dialogue.py",
             paragraphs_path, CONFIG_PATH, "--retry-errors", str(retry_attempts)],
            "create_script",
            run_id,
        )

    # ── Build the script ──────────────────────────────────────────────────────
    run_process(
        [sys.executable, "-u", "create_script.py",
         paragraphs_path, voice_config_path, script_output_path, chunks_output_path],
        "create_script",
        run_id,
    )


def _run_process_paragraphs_task(run_id: str, input_file: str, output_path: str):
    run_process(
        [sys.executable, "-u", "process_paragraphs.py", input_file, output_path],
        "process_paragraphs",
        run_id,
    )


def _run_generate_script_task(run_id: str):
    state = _load_project_state_payload()
    input_file = state.get("input_file_path")
    if not input_file:
        raise FileNotFoundError("No input file found in state")
    _clear_processing_stage_and_downstream("script")
    success = run_process([sys.executable, "-u", "generate_script.py", input_file], "script", run_id)
    if success:
        _mark_processing_stage_completed_marker("script")
    return success


def _run_review_script_task(run_id: str):
    _clear_processing_stage_and_downstream("review")
    success = run_process([sys.executable, "-u", "review_script.py"], "review", run_id)
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


def _run_voices_task(run_id: str, stop_check=None):
    _clear_processing_stage_and_downstream("voices")
    success = run_voice_processing_task(run_id, stop_check=stop_check)
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
    if not os.path.exists(SCRIPT_PATH):
        raise FileNotFoundError("No annotated script found. Generate a script first.")
    run_id = _start_task_run("review")
    return _run_review_script_task(run_id)


def _run_processing_sanity_stage():
    if not os.path.exists(SCRIPT_PATH):
        raise FileNotFoundError("No annotated script found. Generate a script first.")
    run_id = _start_task_run("sanity")
    return _run_sanity_task(run_id, stop_check=_processing_workflow_is_pause_requested)


def _run_processing_repair_stage():
    if not os.path.exists(SCRIPT_PATH):
        raise FileNotFoundError("No annotated script found. Generate a script first.")
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
        "options": {"process_voices": True, "generate_audio": False},
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
    """Detect already-complete stages from file state (used on fresh start)."""
    options = options or {}
    paragraphs_path = os.path.join(ROOT_DIR, "paragraphs.json")
    chunks_path = CHUNKS_PATH
    script_path = os.path.join(ROOT_DIR, "annotated_script.json")

    pdata: dict = {}
    if os.path.exists(paragraphs_path):
        try:
            with open(paragraphs_path, "r", encoding="utf-8") as f:
                pdata = json.load(f)
        except Exception:
            pass

    completed = []
    if pdata.get("paragraphs"):
        completed.append("process_paragraphs")
    if pdata.get("dialogue_assignment_complete"):
        completed.append("assign_dialogue")
    if pdata.get("temperament_extraction_complete"):
        completed.append("extract_temperament")
    if os.path.exists(script_path) and os.path.exists(chunks_path):
        completed.append("create_script")

    # process_voices: all voice entries in voice_config have a ref_audio file
    if options.get("process_voices", True) and os.path.exists(VOICE_CONFIG_PATH):
        try:
            with open(VOICE_CONFIG_PATH, "r", encoding="utf-8") as f:
                vc = json.load(f)
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

    return completed


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
        process_state["new_mode_workflow"]["running"] = False
        process_state["new_mode_workflow"]["paused"] = False
        process_state["new_mode_workflow"]["pause_requested"] = False
        process_state["new_mode_workflow"]["current_stage"] = None
        process_state["new_mode_workflow"]["last_error"] = None
        process_state["new_mode_workflow"]["completed_at"] = time.time()
        options = process_state["new_mode_workflow"].get("options") or {}
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

    paragraphs_path = os.path.join(ROOT_DIR, "paragraphs.json")
    config_path = os.path.join(BASE_DIR, "config.json")

    if stage_name == "process_voices":
        # Voices task uses the "voices" task slot, not the stage name
        run_id = _start_task_run("voices")
        success = run_voice_processing_task(
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
            input_file = state.get("input_file_path", "")
            success = run_process(
                [sys.executable, "-u", "process_paragraphs.py", input_file, paragraphs_path],
                stage_name, run_id, relay_fn=relay,
            )
        elif stage_name == "assign_dialogue":
            success = run_process(
                [sys.executable, "-u", "assign_dialogue.py", paragraphs_path, config_path],
                stage_name, run_id, relay_fn=relay,
            )
        elif stage_name == "extract_temperament":
            success = run_process(
                [sys.executable, "-u", "extract_temperament.py", paragraphs_path, config_path],
                stage_name, run_id, relay_fn=relay,
            )
        elif stage_name == "create_script":
            script_path = os.path.join(ROOT_DIR, "annotated_script.json")
            success = run_process(
                [sys.executable, "-u", "create_script.py",
                 paragraphs_path, VOICE_CONFIG_PATH, script_path, CHUNKS_PATH],
                stage_name, run_id, relay_fn=relay,
            )
        elif stage_name == "proofread":
            success = run_process(
                [sys.executable, "-u", "proofread_runner.py", ROOT_DIR, "0.8", "__ALL__"],
                "proofread", run_id, relay_fn=relay,
            )
        else:
            raise RuntimeError(f"Unknown new-mode stage: {stage_name}")

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
            completed = list(state.get("completed_stages") or [])
            pending = next(
                (s for s in _new_mode_stage_sequence(options) if s not in completed),
                None,
            )
            if pending is None:
                _set_new_mode_workflow_completed()
                break

            try:
                _run_new_mode_workflow_stage(pending)
                with new_mode_workflow_lock:
                    done = list(process_state["new_mode_workflow"].get("completed_stages") or [])
                    if pending not in done:
                        done.append(pending)
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
    process_state["new_mode_workflow"] = restored
    if restored.get("running") and not restored.get("paused"):
        with new_mode_workflow_lock:
            _append_new_mode_workflow_log_locked("Recovered workflow after app restart.")
            _start_new_mode_workflow_thread_locked()


_restore_new_mode_workflow_state()

# Endpoints

@app.get("/")
async def read_index():
    return FileResponse(
        os.path.join(STATIC_DIR, "index.html"),
        headers={"Cache-Control": "no-cache, no-store, must-revalidate"}
    )

@app.get("/favicon.ico")
async def read_favicon():
    favicon_path = os.path.join(ROOT_DIR, "icon.png")
    if os.path.exists(favicon_path):
        return FileResponse(favicon_path, media_type="image/png")
    raise HTTPException(status_code=404, detail="Favicon not found")

@app.get("/api/config")
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

@app.get("/api/default_prompts")
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

@app.post("/api/config")
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

@app.post("/api/config/preferences")
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

@app.post("/api/generation_mode_lock")
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

@app.post("/api/upload")
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


@app.get("/api/script_ingestion/preflight")
async def get_script_ingestion_preflight():
    return _script_ingestion_preflight_summary()


@app.post("/api/reset_project")
async def reset_project():
    running_task = _any_project_task_running()
    if running_task:
        raise HTTPException(status_code=409, detail=f"Cannot reset while '{running_task}' is running.")

    removed = []
    state_path = os.path.join(ROOT_DIR, "state.json")
    state_data = {}
    if os.path.exists(state_path):
        try:
            with open(state_path, "r", encoding="utf-8") as f:
                state_data = json.load(f)
        except (json.JSONDecodeError, ValueError, OSError):
            state_data = {}

    input_file_path = state_data.get("input_file_path") or ""
    if input_file_path:
        try:
            if os.path.commonpath([os.path.abspath(input_file_path), os.path.abspath(UPLOADS_DIR)]) == os.path.abspath(UPLOADS_DIR) and os.path.exists(input_file_path):
                os.remove(input_file_path)
                removed.append(os.path.basename(input_file_path))
        except ValueError:
            pass

    files_to_remove = [
        state_path,
        SCRIPT_PATH,
        VOICES_PATH,
        VOICE_CONFIG_PATH,
        CHUNKS_PATH,
        project_manager.transcription_cache_path,
        SCRIPT_REPAIR_TRACE_PATH,
        AUDIOBOOK_PATH,
        M4B_PATH,
        AUDIO_QUEUE_STATE_PATH,
        PROCESSING_WORKFLOW_STATE_PATH,
        NEW_MODE_WORKFLOW_STATE_PATH,
        os.path.join(ROOT_DIR, "paragraphs.json"),
        SCRIPT_SANITY_PATH,
        os.path.join(ROOT_DIR, "audacity_export.zip"),
        os.path.join(ROOT_DIR, "m4b_cover.jpg"),
    ]

    for path in files_to_remove:
        if os.path.exists(path):
            os.remove(path)
            removed.append(os.path.basename(path))

    if os.path.isdir(VOICELINES_DIR):
        for entry in os.listdir(VOICELINES_DIR):
            entry_path = os.path.join(VOICELINES_DIR, entry)
            if os.path.isdir(entry_path):
                shutil.rmtree(entry_path)
            else:
                os.remove(entry_path)
    os.makedirs(VOICELINES_DIR, exist_ok=True)

    if os.path.isdir(UPLOADS_DIR):
        for entry in os.listdir(UPLOADS_DIR):
            entry_path = os.path.join(UPLOADS_DIR, entry)
            if os.path.isdir(entry_path):
                shutil.rmtree(entry_path)
            else:
                os.remove(entry_path)

    with audio_queue_condition:
        audio_queue.clear()
        process_state["audio"]["cancel"] = False
        process_state["audio"]["queue"] = []
        process_state["audio"]["current_job"] = None
        process_state["audio"]["recent_jobs"] = []
        process_state["audio"]["logs"] = []
        process_state["audio"]["running"] = False
        process_state["audio"]["merge_running"] = False
        process_state["audio"]["metrics"] = _new_audio_metrics()
        process_state["audio"]["heartbeat"] = _new_audio_heartbeat_state()

    with project_manager._transcription_cache_lock:
        project_manager._transcription_cache = None

    for task_name in ("script", "voices", "proofread", "review", "sanity", "repair", "audacity_export", "m4b_export",
                      "process_paragraphs", "assign_dialogue", "extract_temperament", "create_script"):
        process_state[task_name]["logs"] = []
        process_state[task_name]["running"] = False
        if "progress" in process_state[task_name]:
            process_state[task_name]["progress"] = {}

    with processing_workflow_lock:
        process_state["processing_workflow"] = _new_processing_workflow_state()
        _persist_processing_workflow_state_locked()

    with new_mode_workflow_lock:
        process_state["new_mode_workflow"] = _new_mode_workflow_initial_state()
        _persist_new_mode_workflow_state_locked()

    project_manager.engine = None

    logger.info("Project state reset")
    return {"status": "reset", "removed": removed}

@app.post("/api/assign_dialogue")
async def start_assign_dialogue(background_tasks: BackgroundTasks):
    _ensure_task_not_running("assign_dialogue", "Dialogue assignment is already running.")

    paragraphs_path = os.path.join(ROOT_DIR, "paragraphs.json")
    if not os.path.exists(paragraphs_path):
        raise HTTPException(
            status_code=400,
            detail="No paragraph data found. Run 'Process Paragraphs' first.",
        )
    try:
        with open(paragraphs_path, "r", encoding="utf-8") as f:
            pdata = json.load(f)
        if not pdata.get("paragraphs"):
            raise ValueError("empty")
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Paragraph data is empty or corrupt. Re-run 'Process Paragraphs'.",
        )

    config_path = os.path.join(BASE_DIR, "config.json")
    run_id = _start_task_run("assign_dialogue")
    background_tasks.add_task(_run_assign_dialogue_task, run_id, paragraphs_path, config_path)
    return {"status": "started", "run_id": run_id}


@app.post("/api/extract_temperament")
async def start_extract_temperament(background_tasks: BackgroundTasks):
    _ensure_task_not_running("extract_temperament", "Temperament extraction is already running.")

    paragraphs_path = os.path.join(ROOT_DIR, "paragraphs.json")
    if not os.path.exists(paragraphs_path):
        raise HTTPException(
            status_code=400,
            detail="No paragraph data found. Run 'Process Paragraphs' first.",
        )
    try:
        with open(paragraphs_path, "r", encoding="utf-8") as f:
            pdata = json.load(f)
        if not pdata.get("paragraphs"):
            raise ValueError("empty")
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Paragraph data is empty or corrupt. Re-run 'Process Paragraphs'.",
        )

    config_path = os.path.join(BASE_DIR, "config.json")
    run_id = _start_task_run("extract_temperament")
    background_tasks.add_task(_run_extract_temperament_task, run_id, paragraphs_path, config_path)
    return {"status": "started", "run_id": run_id}


@app.get("/api/script_info")
async def get_script_info():
    """Return a lightweight summary of the current script state."""
    script_path = os.path.join(ROOT_DIR, "annotated_script.json")
    if not os.path.exists(script_path):
        return {"entry_count": 0}
    try:
        with open(script_path, "r", encoding="utf-8") as f:
            doc = json.load(f)
        entries = doc.get("entries", []) if isinstance(doc, dict) else doc
        return {"entry_count": len(entries) if isinstance(entries, list) else 0}
    except Exception:
        return {"entry_count": 0}


@app.post("/api/reset_new_mode")
async def reset_new_mode():
    """Clear the script, chunks, and voice config so Create Script can start fresh."""
    removed = []
    for path in (
        os.path.join(ROOT_DIR, "annotated_script.json"),
        CHUNKS_PATH,
        VOICE_CONFIG_PATH,
    ):
        if os.path.exists(path):
            os.remove(path)
            removed.append(os.path.basename(path))
    # Also reset the in-memory task state so _ensure_task_not_running won't block
    with task_state_lock:
        state = process_state.get("create_script")
        if state:
            state["running"] = False
            state["logs"] = []
            state["progress"] = {}
    return {"status": "reset", "removed": removed}


@app.post("/api/create_script")
async def start_create_script(background_tasks: BackgroundTasks):
    _ensure_task_not_running("create_script", "Script creation is already running.")

    paragraphs_path    = os.path.join(ROOT_DIR, "paragraphs.json")
    voice_config_path  = VOICE_CONFIG_PATH
    script_output_path = os.path.join(ROOT_DIR, "annotated_script.json")
    chunks_output_path = CHUNKS_PATH

    if not os.path.exists(paragraphs_path):
        raise HTTPException(
            status_code=400,
            detail="No paragraph data found. Run 'Process Paragraphs' first.",
        )
    try:
        with open(paragraphs_path, "r", encoding="utf-8") as f:
            pdata = json.load(f)
        if not pdata.get("paragraphs"):
            raise ValueError("empty")
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Paragraph data is empty or corrupt. Re-run 'Process Paragraphs'.",
        )

    run_id = _start_task_run("create_script")
    background_tasks.add_task(
        _run_create_script_task, run_id,
        paragraphs_path, voice_config_path, script_output_path, chunks_output_path,
    )
    return {"status": "started", "run_id": run_id}


@app.post("/api/process_paragraphs")
async def start_process_paragraphs(background_tasks: BackgroundTasks):
    _ensure_task_not_running("process_paragraphs", "Paragraph processing is already running.")

    # Hard-fail if an annotated script already exists with entries
    script_path = os.path.join(ROOT_DIR, "annotated_script.json")
    if os.path.exists(script_path):
        try:
            with open(script_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            entries = data.get("entries") if isinstance(data, dict) else data
            if isinstance(entries, list) and len(entries) > 0:
                raise HTTPException(
                    status_code=409,
                    detail="Generating a new audiobook script requires erasing the old one first.",
                )
        except HTTPException:
            raise
        except Exception:
            pass  # Unreadable or corrupt file — allow proceeding

    # Resolve input file from state.json
    state_path = os.path.join(ROOT_DIR, "state.json")
    if not os.path.exists(state_path):
        raise HTTPException(status_code=400, detail="No input file selected. Please upload a book first.")
    with open(state_path, "r", encoding="utf-8") as f:
        state = json.load(f)
    input_file = state.get("input_file_path")
    if not input_file or not os.path.exists(input_file):
        raise HTTPException(status_code=400, detail="No input file found. Please upload a book first.")

    output_path = os.path.join(ROOT_DIR, "paragraphs.json")
    run_id = _start_task_run("process_paragraphs")
    background_tasks.add_task(_run_process_paragraphs_task, run_id, input_file, output_path)
    return {"status": "started", "run_id": run_id}


@app.post("/api/generate_script")
async def generate_script(request: ScriptGenerationRequest, background_tasks: BackgroundTasks):
    _ensure_task_not_running("script", "Script generation is already running.")
    preflight = _script_ingestion_preflight_summary()
    if preflight.get("warn") and not request.force_reimport and not request.skip_import:
        raise HTTPException(
            status_code=409,
            detail={
                "message": preflight.get("message") or "Existing project matches the uploaded EPUB.",
                "code": "script_ingestion_conflict",
                "preflight": preflight,
            },
        )
    if request.skip_import:
        return _mark_script_stage_skipped_for_existing_project()
    if request.force_reimport:
        _clear_project_derived_state(preserve_input_file=True)

    # Get input file from state.json
    state_path = os.path.join(ROOT_DIR, "state.json")
    if not os.path.exists(state_path):
        raise HTTPException(status_code=400, detail="No input file selected")

    with open(state_path, "r", encoding="utf-8") as f:
        state = json.load(f)
        input_file = state.get("input_file_path")

    if not input_file:
         raise HTTPException(status_code=400, detail="No input file found in state")

    run_id = _start_task_run("script")
    background_tasks.add_task(_run_generate_script_task, run_id)
    return {"status": "started", "run_id": run_id}

@app.post("/api/review_script")
async def review_script(background_tasks: BackgroundTasks):
    if not os.path.exists(SCRIPT_PATH):
        raise HTTPException(status_code=400, detail="No annotated script found. Generate a script first.")

    run_id = _start_task_run("review")
    background_tasks.add_task(_run_review_script_task, run_id)
    return {"status": "started", "run_id": run_id}

@app.post("/api/script_sanity_check")
async def script_sanity_check(background_tasks: BackgroundTasks):
    if not os.path.exists(SCRIPT_PATH):
        raise HTTPException(status_code=400, detail="No annotated script found. Generate a script first.")

    run_id = _start_task_run("sanity")
    background_tasks.add_task(_run_sanity_task, run_id)
    return {"status": "started", "run_id": run_id}

@app.post("/api/replace_missing_chunks")
async def replace_missing_chunks(background_tasks: BackgroundTasks):
    if not os.path.exists(SCRIPT_PATH):
        raise HTTPException(status_code=400, detail="No annotated script found. Generate a script first.")

    run_id = _start_task_run("repair")
    background_tasks.add_task(_run_repair_task, run_id)
    return {"status": "started", "run_id": run_id}

@app.get("/api/annotated_script")
async def get_annotated_script():
    """Return the current working annotated_script.json."""
    if not os.path.exists(SCRIPT_PATH):
        raise HTTPException(status_code=404, detail="No annotated script found")
    return _load_project_script_document()

@app.get("/api/script_sanity_check")
async def get_script_sanity_check():
    if not os.path.exists(SCRIPT_SANITY_PATH):
        raise HTTPException(status_code=404, detail="No sanity check results found")
    with open(SCRIPT_SANITY_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

@app.get("/api/status/{task_name}")
async def get_status(task_name: str):
    if task_name not in process_state:
        raise HTTPException(status_code=404, detail="Task not found")
    if task_name == "audio":
        with audio_queue_lock:
            _refresh_audio_process_state_locked()
    return process_state[task_name]


@app.post("/api/processing/start")
async def start_processing_workflow(request: ProcessingWorkflowRequest):
    running_task = _any_project_task_running()
    with processing_workflow_lock:
        workflow_state = process_state["processing_workflow"]
        if workflow_state.get("running") and not workflow_state.get("paused"):
            raise HTTPException(status_code=409, detail="Processing workflow is already running.")

        if running_task and not workflow_state.get("paused"):
            raise HTTPException(status_code=409, detail=f"Cannot start processing while '{running_task}' is running.")

        options = {
            "process_voices": bool(request.process_voices),
            "generate_audio": bool(request.generate_audio),
        }
        if not workflow_state.get("paused"):
            preflight = _script_ingestion_preflight_summary()
            if preflight.get("warn") and not request.force_reimport and not request.skip_script_stage:
                raise HTTPException(
                    status_code=409,
                    detail={
                        "message": preflight.get("message") or "Existing project matches the uploaded EPUB.",
                        "code": "script_ingestion_conflict",
                        "preflight": preflight,
                    },
                )
            if request.force_reimport:
                _clear_project_derived_state(preserve_input_file=True)
            if request.skip_script_stage:
                _mark_processing_stage_completed_marker("script")

        if workflow_state.get("paused"):
            workflow_state["options"] = options
            workflow_state["running"] = True
            workflow_state["paused"] = False
            workflow_state["pause_requested"] = False
            workflow_state["last_error"] = None
            workflow_state["resume_count"] = int(workflow_state.get("resume_count", 0) or 0) + 1
            _append_processing_workflow_log_locked("Resuming processing workflow.")
        else:
            completed_stages = _derived_processing_completed_stages(options)
            process_state["processing_workflow"] = _new_processing_workflow_state() | {
                "running": True,
                "paused": False,
                "pause_requested": False,
                "options": options,
                "started_at": time.time(),
                "completed_stages": completed_stages,
            }
            _append_processing_workflow_log_locked("Starting processing workflow.")
            if completed_stages:
                labels = [PROCESSING_WORKFLOW_STAGE_LABELS.get(stage, stage) for stage in completed_stages]
                _append_processing_workflow_log_locked(
                    f"Skipping already completed stages: {', '.join(labels)}."
                )

        _start_processing_workflow_thread_locked()
        return process_state["processing_workflow"]


@app.post("/api/processing/pause")
async def pause_processing_workflow():
    with processing_workflow_lock:
        state = process_state["processing_workflow"]
        if not state.get("running"):
            if state.get("paused"):
                return {"status": "paused"}
            return {"status": "idle"}
        requested = _request_processing_workflow_pause_locked()
        stage_name = state.get("current_stage")

    if requested and stage_name:
        _request_active_stage_pause(stage_name)
    return {"status": "pause_requested", "current_stage": stage_name}


@app.post("/api/new_mode_workflow/start")
async def start_new_mode_workflow(request: NewModeWorkflowRequest):
    options = {"process_voices": bool(request.process_voices), "generate_audio": bool(request.generate_audio)}
    with new_mode_workflow_lock:
        state = process_state["new_mode_workflow"]
        if state.get("running") and not state.get("paused"):
            raise HTTPException(status_code=409, detail="New mode workflow is already running.")

        if state.get("paused"):
            # Resume: update options (in case toggle changed) but keep completed stages
            state["running"] = True
            state["paused"] = False
            state["pause_requested"] = False
            state["last_error"] = None
            state["options"] = options
            _append_new_mode_workflow_log_locked("Resuming...")
        else:
            # Fresh start: detect already-complete stages from file state
            completed = _derived_new_mode_completed_stages(options)
            process_state["new_mode_workflow"] = _new_mode_workflow_initial_state() | {
                "running": True,
                "started_at": time.time(),
                "options": options,
                "completed_stages": completed,
            }
            _append_new_mode_workflow_log_locked("Starting new mode workflow.")
            if completed:
                labels = [NEW_MODE_STAGE_LABELS.get(s, s) for s in completed]
                _append_new_mode_workflow_log_locked(
                    f"Skipping already complete: {', '.join(labels)}."
                )

        _start_new_mode_workflow_thread_locked()
    return dict(process_state["new_mode_workflow"])


@app.post("/api/new_mode_workflow/pause")
async def pause_new_mode_workflow():
    with new_mode_workflow_lock:
        state = process_state["new_mode_workflow"]
        if not state.get("running"):
            if state.get("paused"):
                return {"status": "paused"}
            return {"status": "idle"}
        if not state.get("pause_requested"):
            state["pause_requested"] = True
            _append_new_mode_workflow_log_locked(
                "Pause requested. Waiting for current stage to finish safely..."
            )
        stage = state.get("current_stage")

    # Stop the current stage so pause takes effect promptly
    if stage:
        if stage == "render_audio":
            _pause_audio_queue_for_workflow()
        else:
            _terminate_task_process_if_running(stage)
    return {"status": "pause_requested", "current_stage": stage}


@app.get("/api/voices")
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

@app.post("/api/parse_voices")
async def parse_voices(background_tasks: BackgroundTasks):
    if process_state["voices"]["running"]:
         raise HTTPException(status_code=400, detail="Voice parsing already running")

    background_tasks.add_task(run_process, [sys.executable, "-u", "parse_voices.py"], "voices")
    return {"status": "started"}


@app.get("/api/dictionary")
async def get_dictionary():
    return {"entries": _load_project_dictionary_entries()}


@app.post("/api/dictionary")
async def save_dictionary(request: DictionarySaveRequest):
    document = save_script_document(
        SCRIPT_PATH,
        dictionary=clean_dictionary_entries([entry.model_dump() for entry in request.entries]),
    )
    return {"status": "saved", "entries": document["dictionary"]}

@app.post("/api/save_voice_config")
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


@app.post("/api/voices/save_config")
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
                materialized = project_manager.materialize_design_voice(
                    speaker=speaker,
                    description=description,
                    sample_text=sample_text,
                    force=False,
                    voice_config=updated_config,
                    export_config=_load_export_config(),
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


@app.post("/api/voices/suggest_description")
async def suggest_voice_description(request: VoiceDescriptionSuggestRequest):
    speaker = (request.speaker or "").strip()
    if not speaker:
        raise HTTPException(status_code=400, detail="Speaker is required")

    try:
        return await asyncio.to_thread(suggest_voice_description_sync, speaker)
    except Exception as e:
        logger.error(f"Voice description suggestion failed for {speaker}: {e}")
        raise HTTPException(status_code=500, detail=f"Voice suggestion request failed: {e}")


@app.post("/api/voices/design_generate")
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


@app.post("/api/voices/clear_uploaded")
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

@app.get("/api/audiobook")
async def get_audiobook():
    if not os.path.exists(AUDIOBOOK_PATH):
        raise HTTPException(status_code=404, detail="Audiobook not found")
    download_name = f"{_export_download_basename()}.mp3"
    return FileResponse(AUDIOBOOK_PATH, filename=download_name, media_type="audio/mpeg")

# --- Chunk Management Endpoints ---

@app.get("/api/chunks")
async def get_chunks():
    chunks = project_manager.reconcile_chunk_audio_states()
    return chunks


@app.post("/api/chunks/sync_from_script_if_stale")
async def sync_chunks_from_script_if_stale():
    result = project_manager.sync_chunks_from_script_if_stale()
    return result

class ChunkRestoreRequest(BaseModel):
    chunk: dict
    at_index: Optional[int] = None
    after_uid: Optional[str] = None

@app.post("/api/chunks/restore")
async def restore_chunk(request: ChunkRestoreRequest):
    """Re-insert a previously deleted chunk at a specific index."""
    chunks = project_manager.restore_chunk(request.at_index or 0, request.chunk, after_uid=request.after_uid)
    if chunks is None:
        raise HTTPException(status_code=400, detail="Failed to restore chunk")
    return {"status": "ok", "total": len(chunks)}

@app.post("/api/chunks/decompose_long_segments")
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

@app.post("/api/chunks/merge_orphans")
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

@app.post("/api/chunks/repair_legacy")
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

@app.post("/api/chunks/reset_to_pending")
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


@app.post("/api/chunks/invalidate_stale_audio")
async def invalidate_stale_audio():
    running_task = _any_project_task_running()
    if running_task:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot invalidate stale audio while {running_task} work is running",
        )

    result = project_manager.invalidate_stale_audio_references()
    return {"status": "ok", **result}

@app.get("/api/asr/status")
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

@app.post("/api/asr/transcribe")
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

@app.post("/api/chunks/repair_lost_audio")
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

@app.post("/api/proofread")
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

@app.post("/api/proofread/auto")
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

@app.post("/api/proofread/clear_failures")
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

@app.post("/api/proofread/discard_selection")
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

@app.post("/api/proofread/{index}/validate")
async def validate_proofread_clip(index: str, request: ProofreadValidateRequest):
    threshold = max(0.0, min(float(request.threshold or 0.0), 1.0))
    try:
        chunk = project_manager.manually_validate_proofread_clip(index, threshold=threshold)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    if chunk is None:
        raise HTTPException(status_code=404, detail="Invalid chunk id")
    return {"status": "ok", "chunk": chunk}

@app.post("/api/proofread/{index}/reject")
async def reject_proofread_clip(index: str, request: ProofreadValidateRequest):
    threshold = max(0.0, min(float(request.threshold or 0.0), 1.0))
    try:
        chunk = project_manager.manually_reject_proofread_clip(index, threshold=threshold)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    if chunk is None:
        raise HTTPException(status_code=404, detail="Invalid chunk id")
    return {"status": "ok", "chunk": chunk}

@app.post("/api/proofread/{index}/compare")
async def compare_proofread_clip(index: str, request: ProofreadCompareRequest):
    threshold = max(0.0, min(float(request.threshold or 0.0), 1.0))
    try:
        chunk = project_manager.compare_proofread_clip(index, threshold=threshold)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    if chunk is None:
        raise HTTPException(status_code=404, detail="Invalid chunk id")
    return {"status": "ok", "chunk": chunk}

@app.post("/api/render_prep_state")
async def set_render_prep_state(request: RenderPrepStateRequest):
    complete = project_manager.set_render_prep_complete(bool(request.complete))
    return {"status": "ok", "render_prep_complete": complete}

@app.post("/api/chunks/{index}")
async def update_chunk(index: str, update: ChunkUpdate):
    data = update.model_dump(exclude_unset=True)
    logger.info(f"Updating chunk {index} with data: {data}")
    chunk = project_manager.update_chunk(index, data)
    if not chunk:
        raise HTTPException(status_code=404, detail="Chunk not found")
    logger.info(f"Chunk {index} updated, instruct is now: '{chunk.get('instruct', '')}'")
    return chunk

@app.post("/api/chunks/{index}/insert")
async def insert_chunk(index: str):
    """Insert an empty chunk after the given index."""
    chunks = project_manager.insert_chunk(index)
    if chunks is None:
        raise HTTPException(status_code=404, detail="Invalid chunk index")
    return {"status": "ok", "total": len(chunks)}

@app.delete("/api/chunks/{index}")
async def delete_chunk(index: str):
    """Delete a chunk at the given index."""
    result = project_manager.delete_chunk(index)
    if result is None:
        raise HTTPException(status_code=400, detail="Cannot delete chunk (invalid index or last remaining chunk)")
    deleted, chunks, restore_after_uid = result
    return {"status": "ok", "deleted": deleted, "total": len(chunks), "restore_after_uid": restore_after_uid}

@app.post("/api/chunks/{index}/generate")
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

@app.post("/api/chunks/{index}/regenerate")
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

@app.post("/api/merge")
async def merge_audio_endpoint(background_tasks: BackgroundTasks):
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
                    process_state["audio"]["logs"].append(f"Preparing merge inputs: {chapter_label}")
                elif stage == "assembling":
                    process_state["audio"]["logs"].append(f"Assembling chapter {chapter_index}/{total_chapters}: {chapter_label}")
                elif stage == "packing":
                    process_state["audio"]["logs"].append(f"Packing optimized part {chapter_index}/{total_chapters}: {chapter_label}")
                elif stage == "bundling":
                    process_state["audio"]["logs"].append("Writing optimized export zip...")
                elif stage == "exporting":
                    process_state["audio"]["logs"].append("Exporting final audiobook file...")
                elif stage == "normalizing":
                    process_state["audio"]["logs"].append("Applying loudness normalization...")

            success, msg = project_manager.merge_audio(
                progress_callback=on_progress,
                log_callback=_append_audio_log_locked,
                export_config=_load_export_config(),
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

@app.post("/api/merge_optimized")
async def merge_optimized_audio_endpoint(background_tasks: BackgroundTasks):
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

            success, msg = project_manager.export_optimized_mp3_zip(
                progress_callback=on_progress,
                log_callback=_append_audio_log_locked,
                export_config=_load_export_config(),
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

@app.get("/api/optimized_export")
async def get_optimized_export():
    if not os.path.exists(OPTIMIZED_EXPORT_PATH):
        raise HTTPException(status_code=404, detail="Optimized export not found. Generate it first.")
    download_name = f"{_export_download_basename()}.zip"
    return FileResponse(OPTIMIZED_EXPORT_PATH, filename=download_name, media_type="application/zip")

@app.post("/api/export_audacity")
async def export_audacity_endpoint(background_tasks: BackgroundTasks):
    if process_state["audacity_export"]["running"]:
        raise HTTPException(status_code=400, detail="Audacity export already running")

    def task():
        process_state["audacity_export"]["running"] = True
        process_state["audacity_export"]["logs"] = ["Starting Audacity export..."]
        try:
            success, msg = project_manager.export_audacity()
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

@app.get("/api/export_audacity")
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

@app.post("/api/merge_m4b")
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
                export_config=_load_export_config(),
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

@app.get("/api/audiobook_m4b")
async def get_audiobook_m4b():
    if not os.path.exists(M4B_PATH):
        raise HTTPException(status_code=404, detail="M4B audiobook not found. Export it first.")
    return FileResponse(M4B_PATH, filename="audiobook.m4b", media_type="audio/mp4")

@app.post("/api/m4b_cover")
async def upload_m4b_cover(file: UploadFile = File(...)):
    """Upload a cover image for M4B export."""
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    cover_path = os.path.join(ROOT_DIR, "m4b_cover.jpg")
    content = await file.read()
    with open(cover_path, "wb") as f:
        f.write(content)
    return {"status": "uploaded", "path": cover_path}

@app.delete("/api/m4b_cover")
async def delete_m4b_cover():
    """Remove the uploaded cover image."""
    cover_path = os.path.join(ROOT_DIR, "m4b_cover.jpg")
    if os.path.exists(cover_path):
        os.remove(cover_path)
    return {"status": "removed"}

@app.post("/api/generate_batch")
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

@app.post("/api/generate_batch_fast")
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

@app.post("/api/cancel_audio")
async def cancel_audio():
    """Cancel the current audio job and clear any queued jobs."""
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
            if abandoned:
                return {"status": "cancelled", "cleared_queued_jobs": cleared}
            _refresh_audio_process_state_locked(persist=True)
            return {"status": "cancelling", "cleared_queued_jobs": cleared}

        if cleared:
            audio_recovery_request = None
            _append_audio_log_locked(f"[CANCEL] Cleared {cleared} queued job(s)")
            _refresh_audio_process_state_locked(persist=True)
            return {"status": "cancelled", "cleared_queued_jobs": cleared}

    # Not running — still reset any stuck "generating" chunks (e.g. from a crash)
    reset_count = project_manager.reset_generating_chunks()
    return {"status": "not_running", "reset_chunks": reset_count}

## ── Saved Scripts ──────────────────────────────────────────────

def _sanitize_name(name: str) -> str:
    """Make a string safe for use as a filename."""
    name = re.sub(r'[^\w\- ]', '', name).strip()
    name = re.sub(r'\s+', '_', name)
    return name.lower()

@app.get("/api/scripts")
async def list_saved_scripts():
    """List all saved scripts in the scripts/ directory."""
    scripts = []
    for f in os.listdir(SCRIPTS_DIR):
        if f.endswith(".json") and not f.endswith(".voice_config.json") and not f.endswith(".paragraphs.json"):
            name = f[:-5]  # strip .json
            filepath = os.path.join(SCRIPTS_DIR, f)
            companion = os.path.join(SCRIPTS_DIR, f"{name}.voice_config.json")
            scripts.append({
                "name": name,
                "created": os.path.getmtime(filepath),
                "has_voice_config": os.path.exists(companion)
            })
    scripts.sort(key=lambda x: x["created"], reverse=True)
    return scripts


@app.get("/api/project_archive")
async def export_project_archive(background_tasks: BackgroundTasks):
    running_task = _any_project_task_running()
    if running_task:
        raise HTTPException(status_code=409, detail=f"Cannot save a project archive while '{running_task}' is running.")

    entries = _project_archive_entries()
    manifest = _build_project_archive_manifest(entries)

    handle = tempfile.NamedTemporaryFile(prefix="alexandria_project_", suffix=".zip", delete=False)
    temp_zip_path = handle.name
    handle.close()

    try:
        with zipfile.ZipFile(temp_zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr(PROJECT_ARCHIVE_MANIFEST_NAME, json.dumps(manifest, indent=2, ensure_ascii=False))
            for relative_path, absolute_path in entries:
                if relative_path == "state.json":
                    zf.writestr(relative_path, json.dumps(_archive_state_with_relative_paths(), indent=2, ensure_ascii=False))
                else:
                    zf.write(absolute_path, arcname=relative_path)
    except Exception:
        if os.path.exists(temp_zip_path):
            os.remove(temp_zip_path)
        raise

    archive_name = f"alexandria_project_{time.strftime('%Y%m%d_%H%M%S')}.zip"
    background_tasks.add_task(lambda: os.path.exists(temp_zip_path) and os.remove(temp_zip_path))
    return FileResponse(temp_zip_path, filename=archive_name, media_type="application/zip")


@app.post("/api/project_archive/load")
async def load_project_archive(file: UploadFile = File(...)):
    running_task = _any_project_task_running()
    if running_task:
        raise HTTPException(status_code=409, detail=f"Cannot load a project archive while '{running_task}' is running.")

    filename = file.filename or ""
    if not filename.lower().endswith(".zip"):
        raise HTTPException(status_code=400, detail="Project archive must be a .zip file.")

    content = await file.read()
    temp_root = tempfile.mkdtemp(prefix="alexandria_project_import_")
    zip_path = os.path.join(temp_root, "project.zip")
    extract_root = os.path.join(temp_root, "extracted")
    os.makedirs(extract_root, exist_ok=True)

    try:
        with open(zip_path, "wb") as f:
            f.write(content)

        with zipfile.ZipFile(zip_path, "r") as zf:
            names = zf.namelist()
            if PROJECT_ARCHIVE_MANIFEST_NAME not in names:
                raise HTTPException(status_code=400, detail="Archive is missing project archive manifest.")

            try:
                manifest = json.loads(zf.read(PROJECT_ARCHIVE_MANIFEST_NAME).decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as e:
                raise HTTPException(status_code=400, detail=f"Archive manifest is invalid: {e}")
            if manifest.get("kind") != "alexandria_project_archive":
                raise HTTPException(status_code=400, detail="Archive is not a valid Alexandria project archive.")

            for info in zf.infolist():
                if info.is_dir() or info.filename == PROJECT_ARCHIVE_MANIFEST_NAME:
                    continue
                relative_path = _normalize_archive_path(info.filename)
                if not _is_allowed_project_archive_path(relative_path):
                    raise HTTPException(status_code=400, detail=f"Archive contains unsupported path: {relative_path}")
                target_path = os.path.join(extract_root, relative_path)
                os.makedirs(os.path.dirname(target_path), exist_ok=True)
                with zf.open(info, "r") as source, open(target_path, "wb") as target:
                    shutil.copyfileobj(source, target)

        _restore_project_archive(extract_root)
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)

    return {"status": "loaded", "filename": filename}

class ScriptSaveRequest(BaseModel):
    name: str

def _delete_saved_script_artifacts(name: str):
    base = os.path.join(SCRIPTS_DIR, f"{name}.json")
    if os.path.exists(base):
        os.remove(base)
    for suffix in (".voice_config.json", ".paragraphs.json"):
        companion = os.path.join(SCRIPTS_DIR, f"{name}{suffix}")
        if os.path.exists(companion):
            os.remove(companion)

def _save_current_script_snapshot(name: str, *, purge_existing: bool = False):
    if not os.path.exists(SCRIPT_PATH):
        raise FileNotFoundError("No annotated script to save. Generate a script first.")

    safe_name = _sanitize_name(name)
    if not safe_name:
        raise ValueError("Invalid script name.")

    dest = os.path.join(SCRIPTS_DIR, f"{safe_name}.json")
    existed = os.path.exists(dest)
    if purge_existing and existed:
        _delete_saved_script_artifacts(safe_name)

    shutil.copy2(SCRIPT_PATH, dest)

    if os.path.exists(VOICE_CONFIG_PATH):
        shutil.copy2(VOICE_CONFIG_PATH, os.path.join(SCRIPTS_DIR, f"{safe_name}.voice_config.json"))

    paragraphs_path = os.path.join(ROOT_DIR, "paragraphs.json")
    if os.path.exists(paragraphs_path):
        shutil.copy2(paragraphs_path, os.path.join(SCRIPTS_DIR, f"{safe_name}.paragraphs.json"))

    state = _load_project_state_payload()
    state["loaded_script_name"] = safe_name
    _save_project_state_payload(state)
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

@app.post("/api/scripts/save")
async def save_script(request: ScriptSaveRequest):
    """Save the current annotated_script.json (and voice_config.json) under a name."""
    try:
        result = _save_current_script_snapshot(request.name, purge_existing=False)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    logger.info(f"Script saved as '{result['name']}'")
    return {"status": "saved", "name": result["name"]}

class ScriptLoadRequest(BaseModel):
    name: str

@app.post("/api/scripts/load")
async def load_script(request: ScriptLoadRequest):
    """Load a saved script, replacing the current annotated_script.json and chunks."""
    if process_state["audio"]["running"]:
        raise HTTPException(status_code=409, detail="Cannot load a script while audio generation is running.")

    src = os.path.join(SCRIPTS_DIR, f"{request.name}.json")
    if not os.path.exists(src):
        raise HTTPException(status_code=404, detail=f"Saved script '{request.name}' not found.")

    shutil.copy2(src, SCRIPT_PATH)

    companion = os.path.join(SCRIPTS_DIR, f"{request.name}.voice_config.json")
    if os.path.exists(companion):
        shutil.copy2(companion, VOICE_CONFIG_PATH)

    paragraphs_path = os.path.join(ROOT_DIR, "paragraphs.json")
    paragraphs_companion = os.path.join(SCRIPTS_DIR, f"{request.name}.paragraphs.json")
    if os.path.exists(paragraphs_companion):
        shutil.copy2(paragraphs_companion, paragraphs_path)
    elif os.path.exists(paragraphs_path):
        os.remove(paragraphs_path)

    # Delete chunks so they regenerate from the loaded script
    if os.path.exists(CHUNKS_PATH):
        os.remove(CHUNKS_PATH)

    state = _load_project_state_payload()
    state["render_prep_complete"] = False
    state["loaded_script_name"] = request.name
    state[PROCESSING_STAGE_MARKERS_KEY] = {"script": {"completed_at": time.time()}}
    _save_project_state_payload(state)

    logger.info(f"Script '{request.name}' loaded")
    return {"status": "loaded", "name": request.name}

@app.delete("/api/scripts/{name}")
async def delete_script(name: str):
    """Delete a saved script."""
    filepath = os.path.join(SCRIPTS_DIR, f"{name}.json")
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail=f"Saved script '{name}' not found.")

    os.remove(filepath)
    for suffix in (".voice_config.json", ".paragraphs.json"):
        companion = os.path.join(SCRIPTS_DIR, f"{name}{suffix}")
        if os.path.exists(companion):
            os.remove(companion)

    logger.info(f"Script '{name}' deleted")
    return {"status": "deleted", "name": name}

## ── Voice Designer ──────────────────────────────────────────────

DESIGNED_VOICES_MANIFEST = os.path.join(DESIGNED_VOICES_DIR, "manifest.json")

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

@app.post("/api/voice_design/preview")
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

@app.post("/api/voice_design/save")
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
    })
    _save_manifest(DESIGNED_VOICES_MANIFEST, manifest)

    logger.info(f"Designed voice saved: '{request.name}' as {dest_filename}")
    return {"status": "saved", "voice_id": voice_id}

@app.get("/api/voice_design/list")
async def voice_design_list():
    """List all saved designed voices."""
    return _load_manifest(DESIGNED_VOICES_MANIFEST)

@app.delete("/api/voice_design/{voice_id}")
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

## ── Clone Voice Uploads ───────────────────────────────────────

CLONE_VOICES_MANIFEST = os.path.join(CLONE_VOICES_DIR, "manifest.json")
ALLOWED_AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg"}

@app.get("/api/clone_voices/list")
async def clone_voices_list():
    """List all uploaded clone voices."""
    return _load_manifest(CLONE_VOICES_MANIFEST)

@app.post("/api/clone_voices/upload")
async def clone_voices_upload(file: UploadFile = File(...)):
    """Upload an audio file for voice cloning."""
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_AUDIO_EXTS:
        raise HTTPException(status_code=400, detail=f"Unsupported format. Use: {', '.join(ALLOWED_AUDIO_EXTS)}")

    base_name = os.path.splitext(file.filename)[0]
    safe_name = _sanitize_name(base_name)
    if not safe_name:
        raise HTTPException(status_code=400, detail="Invalid filename")

    voice_id = f"{safe_name}_{int(time.time())}"
    dest_filename = f"{voice_id}{ext}"
    dest_path = os.path.join(CLONE_VOICES_DIR, dest_filename)

    async with aiofiles.open(dest_path, "wb") as out_file:
        content = await file.read()
        await out_file.write(content)

    manifest = _load_manifest(CLONE_VOICES_MANIFEST)
    manifest.append({
        "id": voice_id,
        "name": base_name,
        "filename": dest_filename,
    })
    _save_manifest(CLONE_VOICES_MANIFEST, manifest)

    logger.info(f"Clone voice uploaded: '{base_name}' as {dest_filename}")
    return {"status": "uploaded", "voice_id": voice_id, "filename": dest_filename}

@app.delete("/api/clone_voices/{voice_id}")
async def clone_voices_delete(voice_id: str):
    """Delete an uploaded clone voice."""
    manifest = _load_manifest(CLONE_VOICES_MANIFEST)
    entry = next((v for v in manifest if v["id"] == voice_id), None)
    if not entry:
        raise HTTPException(status_code=404, detail="Clone voice not found")

    wav_path = os.path.join(CLONE_VOICES_DIR, entry["filename"])
    if os.path.exists(wav_path):
        os.remove(wav_path)

    manifest = [v for v in manifest if v["id"] != voice_id]
    _save_manifest(CLONE_VOICES_MANIFEST, manifest)

    logger.info(f"Clone voice deleted: {voice_id}")
    return {"status": "deleted", "voice_id": voice_id}

## ── LoRA Training ──────────────────────────────────────────────

LORA_MODELS_MANIFEST = os.path.join(LORA_MODELS_DIR, "manifest.json")

def _load_builtin_lora_manifest():
    """Load built-in LoRA manifest from HF (with local fallback). Returns ALL entries with download status."""
    entries = fetch_builtin_manifest(BUILTIN_LORA_DIR)
    result = []
    for entry in entries:
        entry = dict(entry)  # avoid mutating cached list
        local_id = entry["id"] if entry["id"].startswith("builtin_") else f"builtin_{entry['id']}"
        downloaded = is_adapter_downloaded(local_id, BUILTIN_LORA_DIR)
        entry["id"] = local_id
        entry["builtin"] = True
        entry["downloaded"] = downloaded
        entry["adapter_path"] = f"builtin_lora/{local_id}" if downloaded else None
        result.append(entry)
    return result

@app.post("/api/lora/upload_dataset")
async def lora_upload_dataset(file: UploadFile = File(...)):
    """Upload a ZIP containing WAV files and metadata.jsonl."""
    if not file.filename.endswith(".zip"):
        raise HTTPException(status_code=400, detail="File must be a .zip archive")

    # Derive dataset name from ZIP filename
    dataset_name = re.sub(r'[^\w\- ]', '', os.path.splitext(file.filename)[0]).strip()
    dataset_name = re.sub(r'\s+', '_', dataset_name).lower()
    if not dataset_name:
        raise HTTPException(status_code=400, detail="Invalid dataset name from filename")

    dataset_dir = os.path.join(LORA_DATASETS_DIR, dataset_name)
    if os.path.exists(dataset_dir):
        raise HTTPException(status_code=400, detail=f"Dataset '{dataset_name}' already exists")

    # Save ZIP temporarily, then extract
    tmp_path = os.path.join(LORA_DATASETS_DIR, f"_tmp_{dataset_name}.zip")
    try:
        async with aiofiles.open(tmp_path, "wb") as out_file:
            content = await file.read()
            await out_file.write(content)

        os.makedirs(dataset_dir, exist_ok=True)
        with zipfile.ZipFile(tmp_path, "r") as zf:
            zf.extractall(dataset_dir)

        # Check for metadata.jsonl (may be inside a subdirectory)
        metadata_path = os.path.join(dataset_dir, "metadata.jsonl")
        if not os.path.exists(metadata_path):
            # Check one level deep
            for entry in os.listdir(dataset_dir):
                candidate = os.path.join(dataset_dir, entry, "metadata.jsonl")
                if os.path.isdir(os.path.join(dataset_dir, entry)) and os.path.exists(candidate):
                    # Move contents up
                    nested = os.path.join(dataset_dir, entry)
                    for item in os.listdir(nested):
                        shutil.move(os.path.join(nested, item), os.path.join(dataset_dir, item))
                    os.rmdir(nested)
                    metadata_path = os.path.join(dataset_dir, "metadata.jsonl")
                    break

        if not os.path.exists(metadata_path):
            shutil.rmtree(dataset_dir)
            raise HTTPException(status_code=400, detail="ZIP must contain metadata.jsonl")

        # Count samples
        sample_count = 0
        with open(metadata_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    sample_count += 1

        logger.info(f"LoRA dataset uploaded: '{dataset_name}' ({sample_count} samples)")
        return {"status": "uploaded", "dataset_id": dataset_name, "sample_count": sample_count}

    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

@app.post("/api/lora/generate_dataset")
async def lora_generate_dataset(request: LoraGenerateDatasetRequest, background_tasks: BackgroundTasks):
    """Generate a LoRA training dataset using Voice Designer.

    Generates multiple audio samples with the same voice description,
    saving them as a ready-to-train dataset.
    """
    if process_state["dataset_gen"]["running"]:
        raise HTTPException(status_code=400, detail="Dataset generation already running")

    # Build unified sample list from either format
    sample_list = []
    if request.samples:
        for s in request.samples:
            if s.text.strip():
                sample_list.append({"emotion": s.emotion.strip(), "text": s.text.strip()})
    elif request.texts:
        for t in request.texts:
            if t.strip():
                sample_list.append({"emotion": "", "text": t.strip()})

    if not sample_list:
        raise HTTPException(status_code=400, detail="Provide at least one sample text")

    safe_name = _sanitize_name(request.name)
    if not safe_name:
        raise HTTPException(status_code=400, detail="Invalid dataset name")

    dataset_dir = os.path.join(LORA_DATASETS_DIR, safe_name)
    if os.path.exists(dataset_dir):
        raise HTTPException(status_code=400, detail=f"Dataset '{safe_name}' already exists")

    total = len(sample_list)
    root_description = request.description.strip()

    def task():
        process_state["dataset_gen"]["running"] = True
        process_state["dataset_gen"]["logs"] = [
            f"Generating {total} samples with VoiceDesign..."
        ]
        try:
            engine = project_manager.get_engine()
            if not engine:
                process_state["dataset_gen"]["logs"].append("Error: TTS engine not initialized")
                return

            os.makedirs(dataset_dir, exist_ok=True)
            metadata_lines = []
            completed = 0

            for i, sample in enumerate(sample_list):
                text = sample["text"]
                emotion = sample["emotion"]
                # Build full description: root + emotion if provided
                description = f"{root_description}, {emotion}" if emotion else root_description

                process_state["dataset_gen"]["logs"].append(
                    f"[{i+1}/{total}] {('[' + emotion + '] ' if emotion else '')}\"{ text[:60]}{'...' if len(text) > 60 else ''}\""
                )
                try:
                    wav_path, sr = engine.generate_voice_design(
                        description=description,
                        sample_text=_apply_project_dictionary(text),
                        language=request.language,
                    )
                    # Copy to dataset dir with sequential name
                    dest_filename = f"sample_{i:03d}.wav"
                    dest_path = os.path.join(dataset_dir, dest_filename)
                    shutil.copy2(wav_path, dest_path)

                    # Save first successful sample as ref.wav for consistent speaker embedding
                    if completed == 0:
                        shutil.copy2(wav_path, os.path.join(dataset_dir, "ref.wav"))

                    metadata_lines.append(json.dumps({
                        "audio_filepath": dest_filename,
                        "text": text,
                        "ref_audio": "ref.wav",
                    }, ensure_ascii=False))
                    completed += 1
                    process_state["dataset_gen"]["logs"].append(
                        f"  Saved {dest_filename}"
                    )
                except Exception as e:
                    process_state["dataset_gen"]["logs"].append(
                        f"  Failed: {e}"
                    )

            # Write metadata.jsonl
            metadata_path = os.path.join(dataset_dir, "metadata.jsonl")
            with open(metadata_path, "w", encoding="utf-8") as f:
                f.write("\n".join(metadata_lines) + "\n")

            process_state["dataset_gen"]["logs"].append(
                f"Dataset '{safe_name}' complete: {completed}/{total} samples generated."
            )
            logger.info(f"LoRA dataset generated: '{safe_name}' ({completed} samples)")

        except Exception as e:
            process_state["dataset_gen"]["logs"].append(f"Error: {e}")
            logger.error(f"Dataset generation error: {e}")
            # Clean up partial dataset on failure
            if os.path.exists(dataset_dir):
                shutil.rmtree(dataset_dir)
        finally:
            process_state["dataset_gen"]["running"] = False

    background_tasks.add_task(task)
    return {"status": "started", "dataset_id": safe_name, "total": total}

@app.get("/api/lora/datasets")
async def lora_list_datasets():
    """List uploaded LoRA training datasets."""
    datasets = []
    if not os.path.exists(LORA_DATASETS_DIR):
        return datasets

    for name in sorted(os.listdir(LORA_DATASETS_DIR)):
        dataset_dir = os.path.join(LORA_DATASETS_DIR, name)
        if not os.path.isdir(dataset_dir):
            continue
        metadata_path = os.path.join(dataset_dir, "metadata.jsonl")
        sample_count = 0
        if os.path.exists(metadata_path):
            with open(metadata_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        sample_count += 1
        datasets.append({"dataset_id": name, "sample_count": sample_count})
    return datasets

@app.delete("/api/lora/datasets/{dataset_id}")
async def lora_delete_dataset(dataset_id: str):
    """Delete an uploaded dataset."""
    dataset_dir = os.path.join(LORA_DATASETS_DIR, dataset_id)
    if not os.path.isdir(dataset_dir):
        raise HTTPException(status_code=404, detail="Dataset not found")

    shutil.rmtree(dataset_dir)
    logger.info(f"LoRA dataset deleted: {dataset_id}")
    return {"status": "deleted", "dataset_id": dataset_id}

@app.post("/api/lora/train")
async def lora_start_training(request: LoraTrainingRequest, background_tasks: BackgroundTasks):
    """Start LoRA training as a subprocess."""
    if process_state["lora_training"]["running"]:
        raise HTTPException(status_code=400, detail="LoRA training already running")

    # Validate dataset exists
    dataset_dir = os.path.join(LORA_DATASETS_DIR, request.dataset_id)
    if not os.path.isdir(dataset_dir):
        raise HTTPException(status_code=400, detail=f"Dataset '{request.dataset_id}' not found")

    # Build output directory
    safe_name = _sanitize_name(request.name)
    if not safe_name:
        raise HTTPException(status_code=400, detail="Invalid adapter name")

    adapter_id = f"{safe_name}_{int(time.time())}"
    output_dir = os.path.join(LORA_MODELS_DIR, adapter_id)

    # Unload TTS engine to free GPU
    if project_manager.engine is not None:
        logger.info("Unloading TTS engine for LoRA training...")
        project_manager.engine = None
        gc.collect()

    # Build subprocess command
    command = [
        sys.executable, "-u", "train_lora.py",
        "--data_dir", dataset_dir,
        "--output_dir", output_dir,
        "--epochs", str(request.epochs),
        "--lr", str(request.lr),
        "--batch_size", str(request.batch_size),
        "--lora_r", str(request.lora_r),
        "--lora_alpha", str(request.lora_alpha),
        "--gradient_accumulation_steps", str(request.gradient_accumulation_steps),
    ]
    run_id = _start_task_run("lora_training")

    def on_training_complete():
        """After training subprocess finishes, update manifest if adapter was saved."""
        run_process(command, "lora_training", run_id)

        # Check if training produced an adapter
        if os.path.isdir(output_dir) and os.path.exists(os.path.join(output_dir, "training_meta.json")):
            try:
                with open(os.path.join(output_dir, "training_meta.json"), "r") as f:
                    meta = json.load(f)

                manifest = _load_manifest(LORA_MODELS_MANIFEST)
                manifest.append({
                    "id": adapter_id,
                    "name": request.name,
                    "dataset_id": request.dataset_id,
                    "epochs": meta.get("epochs", request.epochs),
                    "final_loss": meta.get("final_loss"),
                    "sample_count": meta.get("num_samples"),
                    "lora_r": meta.get("lora_r"),
                    "lr": meta.get("lr"),
                    "created": time.time(),
                })
                _save_manifest(LORA_MODELS_MANIFEST, manifest)
                logger.info(f"LoRA adapter registered: {adapter_id}")
            except Exception as e:
                logger.error(f"Failed to update LoRA manifest: {e}")

    background_tasks.add_task(on_training_complete)
    return {"status": "started", "adapter_id": adapter_id, "run_id": run_id}

@app.get("/api/lora/models")
async def lora_list_models():
    """List all LoRA adapters (built-in + user-trained)."""
    models = _load_builtin_lora_manifest() + _load_manifest(LORA_MODELS_MANIFEST)
    for m in models:
        is_builtin = m.get("builtin", False)
        is_downloaded = m.get("downloaded", True)  # user-trained are always downloaded

        if not is_downloaded:
            m["preview_audio_url"] = None
            continue

        if is_builtin:
            adapter_dir = os.path.join(BUILTIN_LORA_DIR, m["id"])
            url_prefix = f"/builtin_lora/{m['id']}"
        else:
            adapter_dir = os.path.join(LORA_MODELS_DIR, m["id"])
            url_prefix = f"/lora_models/{m['id']}"
        preview_path = os.path.join(adapter_dir, "preview_sample.wav")
        m["preview_audio_url"] = f"{url_prefix}/preview_sample.wav" if os.path.exists(preview_path) else None
    return models

@app.delete("/api/lora/models/{adapter_id}")
async def lora_delete_model(adapter_id: str):
    """Delete a trained LoRA adapter. Built-in adapters cannot be deleted."""
    builtin = _load_builtin_lora_manifest()
    if any(m["id"] == adapter_id for m in builtin):
        raise HTTPException(status_code=403, detail="Built-in adapters cannot be deleted")
    manifest = _load_manifest(LORA_MODELS_MANIFEST)
    entry = next((m for m in manifest if m["id"] == adapter_id), None)
    if not entry:
        raise HTTPException(status_code=404, detail="Adapter not found")

    # Delete adapter directory
    adapter_dir = os.path.join(LORA_MODELS_DIR, adapter_id)
    if os.path.isdir(adapter_dir):
        shutil.rmtree(adapter_dir)

    # Remove from manifest
    manifest = [m for m in manifest if m["id"] != adapter_id]
    _save_manifest(LORA_MODELS_MANIFEST, manifest)

    logger.info(f"LoRA adapter deleted: {adapter_id}")
    return {"status": "deleted", "adapter_id": adapter_id}

@app.post("/api/lora/download/{adapter_id}")
async def lora_download_builtin(adapter_id: str):
    """Download a built-in LoRA adapter from HuggingFace."""
    manifest = fetch_builtin_manifest(BUILTIN_LORA_DIR)
    hf_name = adapter_id.replace("builtin_", "", 1)
    entry = next((e for e in manifest if e["id"] == hf_name or e["id"] == adapter_id), None)
    if not entry:
        raise HTTPException(status_code=404, detail=f"Unknown built-in adapter: {adapter_id}")

    if is_adapter_downloaded(adapter_id, BUILTIN_LORA_DIR):
        return {"status": "already_downloaded", "adapter_id": adapter_id}

    try:
        download_builtin_adapter(adapter_id, BUILTIN_LORA_DIR)
        logger.info(f"Built-in adapter downloaded: {adapter_id}")
        return {"status": "downloaded", "adapter_id": adapter_id}
    except Exception as e:
        logger.error(f"Download failed for {adapter_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/lora/test")
async def lora_test_model(request: LoraTestRequest):
    """Generate test audio using a LoRA adapter (built-in or user-trained)."""
    # Check both manifests
    builtin = _load_builtin_lora_manifest()
    user_trained = _load_manifest(LORA_MODELS_MANIFEST)
    all_adapters = builtin + user_trained
    entry = next((m for m in all_adapters if m["id"] == request.adapter_id), None)
    if not entry:
        raise HTTPException(status_code=404, detail="Adapter not found")

    is_builtin = entry.get("builtin", False)
    if is_builtin:
        adapter_dir = os.path.join(BUILTIN_LORA_DIR, request.adapter_id)
        audio_url_prefix = f"/builtin_lora/{request.adapter_id}"
    else:
        adapter_dir = os.path.join(LORA_MODELS_DIR, request.adapter_id)
        audio_url_prefix = f"/lora_models/{request.adapter_id}"

    if not os.path.isdir(adapter_dir) and is_builtin:
        try:
            download_builtin_adapter(request.adapter_id, BUILTIN_LORA_DIR)
            adapter_dir = os.path.join(BUILTIN_LORA_DIR, request.adapter_id)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Auto-download failed: {e}")
    elif not os.path.isdir(adapter_dir):
        raise HTTPException(status_code=404, detail="Adapter files not found")

    engine = project_manager.get_engine()
    if not engine:
        raise HTTPException(status_code=500, detail="Failed to initialize TTS engine")

    try:
        output_filename = f"test_{request.adapter_id}_{int(time.time())}.wav"
        output_path = os.path.join(adapter_dir, output_filename)

        voice_data = {
            "type": "lora",
            "adapter_id": request.adapter_id,
            "adapter_path": adapter_dir,
        }
        voice_config = {"_lora_test_": voice_data}
        engine.generate_voice(
            text=_apply_project_dictionary(request.text),
            instruct_text=request.instruct or "",
            speaker="_lora_test_",
            voice_config=voice_config,
            output_path=output_path,
        )

        return {
            "status": "ok",
            "audio_url": f"{audio_url_prefix}/{output_filename}",
        }
    except Exception as e:
        logger.error(f"LoRA test generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

LORA_PREVIEW_TEXT = "The ancient library stood at the crossroads of two forgotten paths, its weathered stone walls covered in ivy that had been growing for centuries."

@app.post("/api/lora/preview/{adapter_id}")
async def lora_preview(adapter_id: str):
    """Generate or return cached preview audio for a LoRA adapter."""
    builtin = _load_builtin_lora_manifest()
    user_trained = _load_manifest(LORA_MODELS_MANIFEST)
    all_adapters = builtin + user_trained
    entry = next((m for m in all_adapters if m["id"] == adapter_id), None)
    if not entry:
        raise HTTPException(status_code=404, detail="Adapter not found")

    is_builtin = entry.get("builtin", False)
    if is_builtin:
        adapter_dir = os.path.join(BUILTIN_LORA_DIR, adapter_id)
        url_prefix = f"/builtin_lora/{adapter_id}"
    else:
        adapter_dir = os.path.join(LORA_MODELS_DIR, adapter_id)
        url_prefix = f"/lora_models/{adapter_id}"

    if not os.path.isdir(adapter_dir) and is_builtin:
        try:
            download_builtin_adapter(adapter_id, BUILTIN_LORA_DIR)
            adapter_dir = os.path.join(BUILTIN_LORA_DIR, adapter_id)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Auto-download failed: {e}")
    elif not os.path.isdir(adapter_dir):
        raise HTTPException(status_code=404, detail="Adapter files not found")

    preview_path = os.path.join(adapter_dir, "preview_sample.wav")

    # Return cached if exists
    if os.path.exists(preview_path):
        return {"status": "cached", "audio_url": f"{url_prefix}/preview_sample.wav"}

    # Generate preview
    engine = project_manager.get_engine()
    if not engine:
        raise HTTPException(status_code=500, detail="Failed to initialize TTS engine")

    try:
        voice_data = {
            "type": "lora",
            "adapter_id": adapter_id,
            "adapter_path": adapter_dir,
        }
        voice_config = {"_lora_preview_": voice_data}
        engine.generate_voice(
            text=_apply_project_dictionary(LORA_PREVIEW_TEXT),
            instruct_text="",
            speaker="_lora_preview_",
            voice_config=voice_config,
            output_path=preview_path,
        )
        return {"status": "generated", "audio_url": f"{url_prefix}/preview_sample.wav"}
    except Exception as e:
        logger.error(f"LoRA preview generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

## ── Dataset Builder ──────────────────────────────────────────

def _load_builder_state(name):
    """Load project state from dataset builder working directory."""
    state_path = os.path.join(DATASET_BUILDER_DIR, name, "state.json")
    if os.path.exists(state_path):
        try:
            with open(state_path, "r", encoding="utf-8") as f:
                state = json.load(f)
            # Ensure new fields exist for backward compat
            state.setdefault("description", "")
            state.setdefault("global_seed", "")
            state.setdefault("samples", [])
            return state
        except Exception:
            pass
    return {"description": "", "global_seed": "", "samples": []}

def _save_builder_state(name, state):
    """Save per-sample state to dataset builder working directory."""
    work_dir = os.path.join(DATASET_BUILDER_DIR, name)
    os.makedirs(work_dir, exist_ok=True)
    with open(os.path.join(work_dir, "state.json"), "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)

@app.get("/api/dataset_builder/list")
async def dataset_builder_list():
    """List existing dataset builder projects."""
    projects = []
    if os.path.isdir(DATASET_BUILDER_DIR):
        for name in sorted(os.listdir(DATASET_BUILDER_DIR)):
            state_path = os.path.join(DATASET_BUILDER_DIR, name, "state.json")
            if os.path.isfile(state_path):
                state = _load_builder_state(name)
                samples = state.get("samples", [])
                projects.append({
                    "name": name,
                    "description": state.get("description", ""),
                    "sample_count": len(samples),
                    "done_count": sum(1 for s in samples if s.get("status") == "done"),
                })
    return projects

@app.post("/api/dataset_builder/create")
async def dataset_builder_create(request: DatasetBuilderCreateRequest):
    """Create a new dataset builder project."""
    safe_name = _sanitize_name(request.name)
    if not safe_name:
        raise HTTPException(status_code=400, detail="Invalid dataset name")
    work_dir = os.path.join(DATASET_BUILDER_DIR, safe_name)
    if os.path.exists(work_dir):
        raise HTTPException(status_code=400, detail=f"Project '{safe_name}' already exists")
    _save_builder_state(safe_name, {"description": "", "global_seed": "", "samples": []})
    return {"name": safe_name}

@app.post("/api/dataset_builder/update_meta")
async def dataset_builder_update_meta(request: DatasetBuilderUpdateMetaRequest):
    """Update project description and global seed without touching samples."""
    safe_name = _sanitize_name(request.name)
    work_dir = os.path.join(DATASET_BUILDER_DIR, safe_name)
    if not os.path.exists(work_dir):
        raise HTTPException(status_code=404, detail="Project not found")
    state = _load_builder_state(safe_name)
    state["description"] = request.description
    state["global_seed"] = request.global_seed
    _save_builder_state(safe_name, state)
    return {"status": "ok"}

@app.post("/api/dataset_builder/update_rows")
async def dataset_builder_update_rows(request: DatasetBuilderUpdateRowsRequest):
    """Update row definitions, preserving existing generation status/audio."""
    safe_name = _sanitize_name(request.name)
    work_dir = os.path.join(DATASET_BUILDER_DIR, safe_name)
    if not os.path.exists(work_dir):
        raise HTTPException(status_code=404, detail="Project not found")
    state = _load_builder_state(safe_name)
    existing = state.get("samples", [])
    # Merge: keep status/audio_url from existing samples where text unchanged
    new_samples = []
    for i, row in enumerate(request.rows):
        sample = {
            "emotion": row.get("emotion", ""),
            "text": row.get("text", "").strip(),
            "seed": row.get("seed", ""),
            "status": "pending",
            "audio_url": None,
        }
        if i < len(existing):
            old = existing[i]
            # Preserve generation state if text unchanged (trimmed comparison)
            if old.get("text", "").strip() == sample["text"]:
                sample["status"] = old.get("status", "pending")
                sample["audio_url"] = old.get("audio_url")
        new_samples.append(sample)
    state["samples"] = new_samples
    _save_builder_state(safe_name, state)
    return {"status": "ok", "sample_count": len(new_samples)}

@app.post("/api/dataset_builder/generate_sample")
async def dataset_builder_generate_sample(request: DatasetSampleGenRequest):
    """Generate a single dataset sample using VoiceDesign."""
    engine = project_manager.get_engine()
    if not engine:
        raise HTTPException(status_code=500, detail="Failed to initialize TTS engine")

    work_dir = os.path.join(DATASET_BUILDER_DIR, request.dataset_name)
    os.makedirs(work_dir, exist_ok=True)

    try:
        wav_path, sr = engine.generate_voice_design(
            description=request.description,
            sample_text=_apply_project_dictionary(request.text),
            seed=request.seed,
        )

        dest_filename = f"sample_{request.sample_index:03d}.wav"
        dest_path = os.path.join(work_dir, dest_filename)
        shutil.copy2(wav_path, dest_path)

        # Update state (cache-bust URL so browser loads fresh audio on regen)
        cache_bust = int(time.time())
        audio_url = f"/dataset_builder/{request.dataset_name}/{dest_filename}?t={cache_bust}"
        state = _load_builder_state(request.dataset_name)
        samples = state.get("samples", [])
        # Ensure list is large enough
        while len(samples) <= request.sample_index:
            samples.append({"status": "pending"})
        existing_sample = samples[request.sample_index] if request.sample_index < len(samples) else {}
        samples[request.sample_index] = {
            **existing_sample,
            "status": "done",
            "audio_url": audio_url,
            "text": request.text.strip(),
            "description": request.description,
        }
        state["samples"] = samples
        _save_builder_state(request.dataset_name, state)

        return {
            "status": "done",
            "sample_index": request.sample_index,
            "audio_url": audio_url,
        }
    except Exception as e:
        logger.error(f"Dataset builder sample generation failed: {e}")
        # Mark as error in state
        state = _load_builder_state(request.dataset_name)
        samples = state.get("samples", [])
        while len(samples) <= request.sample_index:
            samples.append({"status": "pending"})
        samples[request.sample_index] = {"status": "error", "error": str(e)}
        state["samples"] = samples
        _save_builder_state(request.dataset_name, state)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/dataset_builder/generate_batch")
async def dataset_builder_generate_batch(request: DatasetBatchGenRequest):
    """Batch generate dataset samples as a background task."""
    if process_state["dataset_builder"]["running"]:
        raise HTTPException(status_code=400, detail="Dataset generation already running")

    if not request.samples or len(request.samples) == 0:
        raise HTTPException(status_code=400, detail="No samples provided")

    safe_name = _sanitize_name(request.name)
    if not safe_name:
        raise HTTPException(status_code=400, detail="Invalid dataset name")

    work_dir = os.path.join(DATASET_BUILDER_DIR, safe_name)
    os.makedirs(work_dir, exist_ok=True)
    root_desc = request.description.strip()

    # Determine which indices to generate
    if request.indices is not None:
        to_generate = request.indices
    else:
        to_generate = list(range(len(request.samples)))

    total = len(to_generate)

    # Snapshot request data for the thread (request object may not survive)
    samples_snapshot = [(s.emotion.strip(), s.text.strip()) for s in request.samples]
    global_seed = request.global_seed
    per_seeds = request.seeds

    def task():
        process_state["dataset_builder"]["running"] = True
        process_state["dataset_builder"]["logs"] = []
        process_state["dataset_builder"]["cancel"] = False

        engine = project_manager.get_engine()
        if not engine:
            process_state["dataset_builder"]["logs"].append("[ERROR] Failed to initialize TTS engine")
            process_state["dataset_builder"]["running"] = False
            return

        state = _load_builder_state(safe_name)
        samples_state = state.get("samples", [])
        # Ensure list is large enough for all samples
        while len(samples_state) < len(samples_snapshot):
            samples_state.append({"status": "pending"})

        completed = 0
        for i, idx in enumerate(to_generate):
            if process_state["dataset_builder"]["cancel"]:
                process_state["dataset_builder"]["logs"].append(f"[CANCEL] Stopped at {completed}/{total}")
                break

            emotion, text = samples_snapshot[idx]
            description = f"{root_desc}, {emotion}" if emotion else root_desc

            # Mark as generating (preserve existing fields like emotion, seed)
            existing_s = samples_state[idx] if idx < len(samples_state) else {}
            samples_state[idx] = {**existing_s, "status": "generating", "text": text, "emotion": emotion, "description": description}
            state["samples"] = samples_state
            _save_builder_state(safe_name, state)

            process_state["dataset_builder"]["logs"].append(
                f"[{i+1}/{total}] {('[' + emotion + '] ' if emotion else '')}\"{text[:60]}{'...' if len(text) > 60 else ''}\""
            )

            try:
                # Resolve seed: per-line > global > random
                seed = -1
                if per_seeds and idx < len(per_seeds) and per_seeds[idx] >= 0:
                    seed = per_seeds[idx]
                elif global_seed >= 0:
                    seed = global_seed

                wav_path, sr = engine.generate_voice_design(
                    description=description,
                    sample_text=_apply_project_dictionary(text),
                    seed=seed,
                )
                dest_filename = f"sample_{idx:03d}.wav"
                dest_path = os.path.join(work_dir, dest_filename)
                shutil.copy2(wav_path, dest_path)

                samples_state[idx] = {
                    **samples_state[idx],
                    "status": "done",
                    "audio_url": f"/dataset_builder/{safe_name}/{dest_filename}?t={int(time.time())}",
                    "text": text,
                    "emotion": emotion,
                    "description": description,
                }
                completed += 1
            except Exception as e:
                logger.error(f"Dataset builder sample {idx} failed: {e}")
                process_state["dataset_builder"]["logs"].append(f"  Error: {e}")
                samples_state[idx] = {**samples_state[idx], "status": "error", "error": str(e), "text": text, "emotion": emotion}

            state["samples"] = samples_state
            _save_builder_state(safe_name, state)

        process_state["dataset_builder"]["logs"].append(
            f"[DONE] Generated {completed}/{total} samples"
        )
        process_state["dataset_builder"]["running"] = False

    threading.Thread(target=task, daemon=True).start()
    return {"status": "started", "dataset_name": safe_name, "total": total}

@app.post("/api/dataset_builder/cancel")
async def dataset_builder_cancel():
    """Cancel ongoing batch dataset generation."""
    if process_state["dataset_builder"]["running"]:
        process_state["dataset_builder"]["cancel"] = True
        return {"status": "cancelling"}
    return {"status": "not_running"}

@app.get("/api/dataset_builder/status/{name}")
async def dataset_builder_status(name: str):
    """Get per-sample generation status for a dataset builder project."""
    state = _load_builder_state(name)
    return {
        "description": state.get("description", ""),
        "global_seed": state.get("global_seed", ""),
        "samples": state.get("samples", []),
        "running": process_state["dataset_builder"]["running"],
        "logs": process_state["dataset_builder"]["logs"],
    }

@app.post("/api/dataset_builder/save")
async def dataset_builder_save(request: DatasetSaveRequest):
    """Finalize dataset builder project as a training dataset."""
    safe_name = _sanitize_name(request.name)
    if not safe_name:
        raise HTTPException(status_code=400, detail="Invalid dataset name")

    work_dir = os.path.join(DATASET_BUILDER_DIR, safe_name)
    if not os.path.exists(work_dir):
        raise HTTPException(status_code=404, detail="Dataset builder project not found")

    state = _load_builder_state(safe_name)
    samples = state.get("samples", [])

    # Collect completed samples
    done_samples = [(i, s) for i, s in enumerate(samples) if s.get("status") == "done"]
    if not done_samples:
        raise HTTPException(status_code=400, detail="No completed samples to save")

    # Check ref_index is valid
    ref_idx = request.ref_index
    ref_sample = next((s for i, s in done_samples if i == ref_idx), None)
    if ref_sample is None:
        # Fall back to first completed sample
        ref_idx = done_samples[0][0]
        ref_sample = done_samples[0][1]

    # Create training dataset directory
    dataset_dir = os.path.join(LORA_DATASETS_DIR, safe_name)
    if os.path.exists(dataset_dir):
        raise HTTPException(status_code=400, detail=f"Dataset '{safe_name}' already exists in training datasets")

    os.makedirs(dataset_dir, exist_ok=True)

    try:
        metadata_lines = []
        for i, sample in done_samples:
            src_filename = f"sample_{i:03d}.wav"
            src_path = os.path.join(work_dir, src_filename)
            if not os.path.exists(src_path):
                continue

            dest_filename = f"sample_{i:03d}.wav"
            shutil.copy2(src_path, os.path.join(dataset_dir, dest_filename))

            metadata_lines.append(json.dumps({
                "audio_filepath": dest_filename,
                "text": sample.get("text", ""),
                "ref_audio": "ref.wav",
            }, ensure_ascii=False))

        # Copy ref sample and save its text for correct clone prompt alignment
        ref_src = os.path.join(work_dir, f"sample_{ref_idx:03d}.wav")
        if os.path.exists(ref_src):
            shutil.copy2(ref_src, os.path.join(dataset_dir, "ref.wav"))
        ref_text = ref_sample.get("text", "")
        with open(os.path.join(dataset_dir, "ref_text.txt"), "w", encoding="utf-8") as f:
            f.write(ref_text)

        # Write metadata
        with open(os.path.join(dataset_dir, "metadata.jsonl"), "w", encoding="utf-8") as f:
            f.write("\n".join(metadata_lines) + "\n")

        sample_count = len(metadata_lines)
        logger.info(f"Dataset saved: '{safe_name}' ({sample_count} samples, ref=sample_{ref_idx:03d})")

        return {
            "status": "saved",
            "dataset_id": safe_name,
            "sample_count": sample_count,
        }
    except Exception as e:
        # Clean up on failure
        if os.path.exists(dataset_dir):
            shutil.rmtree(dataset_dir, ignore_errors=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/dataset_builder/{name}")
async def dataset_builder_delete(name: str):
    """Discard a dataset builder working project."""
    work_dir = os.path.join(DATASET_BUILDER_DIR, name)
    if not os.path.exists(work_dir):
        raise HTTPException(status_code=404, detail="Dataset builder project not found")
    shutil.rmtree(work_dir, ignore_errors=True)
    logger.info(f"Dataset builder project discarded: {name}")
    return {"status": "deleted", "name": name}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=4200, access_log=False)
