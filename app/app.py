import os
import sys
import gc
import json
import shutil
import logging
import asyncio
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict
import re
import time
import threading
import zipfile
import subprocess
import aiofiles
import uuid
from openai import OpenAI

# Import ProjectManager
from project import ProjectManager
from default_prompts import DEFAULT_SYSTEM_PROMPT, DEFAULT_USER_PROMPT, load_default_prompts
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
UPLOADS_DIR = os.path.join(BASE_DIR, "uploads")
SCRIPTS_DIR = os.path.join(ROOT_DIR, "scripts")
CHUNKS_PATH = os.path.join(ROOT_DIR, "chunks.json")
AUDIO_QUEUE_STATE_PATH = os.path.join(ROOT_DIR, "audio_queue_state.json")
SCRIPT_SANITY_PATH = os.path.join(ROOT_DIR, "script_sanity_check.json")
DESIGNED_VOICES_DIR = os.path.join(ROOT_DIR, "designed_voices")
CLONE_VOICES_DIR = os.path.join(ROOT_DIR, "clone_voices")
LORA_MODELS_DIR = os.path.join(ROOT_DIR, "lora_models")
LORA_DATASETS_DIR = os.path.join(ROOT_DIR, "lora_datasets")
BUILTIN_LORA_DIR = os.path.join(ROOT_DIR, "builtin_lora")
BUILTIN_LORA_MANIFEST = os.path.join(BUILTIN_LORA_DIR, "manifest.json")
DATASET_BUILDER_DIR = os.path.join(ROOT_DIR, "dataset_builder")

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
    auto_regenerate_bad_clips: bool = False  # retry invalid clips once immediately
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

class PromptConfig(BaseModel):
    system_prompt: Optional[str] = None
    user_prompt: Optional[str] = None
    review_system_prompt: Optional[str] = None
    review_user_prompt: Optional[str] = None
    attribution_system_prompt: Optional[str] = None
    attribution_user_prompt: Optional[str] = None
    voice_prompt: Optional[str] = None

class AppConfig(BaseModel):
    llm: LLMConfig
    tts: TTSConfig
    prompts: Optional[PromptConfig] = None
    generation: Optional[GenerationConfig] = None

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

class RenderPrepStateRequest(BaseModel):
    complete: bool = True

class BatchGenerateRequest(BaseModel):
    indices: List[int]
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

# Global state for process tracking
ROLLING_AUDIO_SAMPLE_LIMIT = 50
AUDIO_HEARTBEAT_INTERVAL_SECONDS = 600
AUDIO_RECOVERY_POLL_SECONDS = 5


process_state = {
    "script": {"running": False, "logs": []},
    "voices": {"running": False, "logs": []},
    "audio": {
        "running": False,
        "logs": [],
        "cancel": False,
        "queue": [],
        "current_job": None,
        "recent_jobs": [],
        "merge_running": False,
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
    "dataset_builder": {"running": False, "logs": [], "cancel": False}
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
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp_path, path)


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
    if persist:
        _persist_audio_queue_state_locked()


def _append_audio_log(message):
    with audio_queue_lock:
        _append_audio_log_locked(message)


def _append_audio_log_locked(message):
    process_state["audio"]["logs"].append(message)
    _trim_logs(process_state["audio"]["logs"])


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
        for idx in indices:
            if 0 <= idx < len(chunks):
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

def run_process(command: List[str], task_name: str, run_id: str):
    """Run a subprocess and capture logs."""
    logger.info(f"Starting task {task_name}: {' '.join(command)}")

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
                _append_task_log(task_name, run_id, log_line)

        process.wait()
        if not _task_is_current(task_name, run_id):
            return
        return_code = process.returncode

        if return_code == 0:
            _append_task_log(task_name, run_id, f"Task {task_name} completed successfully.")
        else:
            _append_task_log(task_name, run_id, f"Task {task_name} failed with return code {return_code}.")

    except Exception as e:
        logger.error(f"Error running {task_name}: {e}")
        _append_task_log(task_name, run_id, f"Error: {str(e)}")
    finally:
        _finish_task_run(task_name, run_id, locals().get("process"))


def run_script_sanity_task(run_id: str):
    def log(message: str):
        return _append_task_log("sanity", run_id, message)

    progress_state = {
        "prepared_logged": False,
        "last_logged_current": 0,
    }

    def on_attribution_progress(event: str, payload: dict):
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
        if not _task_is_current("sanity", run_id):
            return
        save_script_document(
            SCRIPT_PATH,
            entries=script_document.get("entries"),
            dictionary=script_document.get("dictionary", []),
            sanity_cache={"phrase_decisions": phrase_decisions},
        )

    try:
        if os.path.exists(SCRIPT_SANITY_PATH):
            os.remove(SCRIPT_SANITY_PATH)

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
            return

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
    except Exception as e:
        logger.error(f"Error running script sanity check: {e}")
        if _task_is_current("sanity", run_id):
            log(f"Error: {str(e)}")
    finally:
        _finish_task_run("sanity", run_id)


def run_script_repair_task(run_id: str):
    def log(message: str):
        if not _append_task_log("repair", run_id, message):
            raise RepairSupersededError()

    try:
        if os.path.exists(SCRIPT_SANITY_PATH):
            os.remove(SCRIPT_SANITY_PATH)

        result = repair_invalid_chunks(
            ROOT_DIR,
            log,
            should_continue=lambda: _task_is_current("repair", run_id),
        )
        if not _task_is_current("repair", run_id):
            return
        final_sanity = result["final_sanity"]

        with open(SCRIPT_SANITY_PATH, "w", encoding="utf-8") as f:
            json.dump(final_sanity, f, indent=2, ensure_ascii=False)

        log(f"Initial invalid chunks: {result['initial_invalid_chunks']}")
        log(f"Initial missing words: {result['initial_missing_words']}")
        log(f"Initial inserted words: {result['initial_inserted_words']}")
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
    except RepairSupersededError:
        logger.info("Repair task superseded by a newer request")
    except Exception as e:
        logger.error(f"Error running script repair: {e}")
        if _task_is_current("repair", run_id):
            log(f"Error: {str(e)}")
    finally:
        _finish_task_run("repair", run_id)

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
            "auto_regenerate_bad_clips": False
        },
        "prompts": {
            "system_prompt": "",
            "user_prompt": "",
            "voice_prompt": ""
        }
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

    # Include current input file info if available
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
        except (json.JSONDecodeError, ValueError):
            pass

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

    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    # Reset engine so it picks up new TTS settings on next use
    project_manager.engine = None
    return {"status": "saved"}

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
        AUDIOBOOK_PATH,
        M4B_PATH,
        AUDIO_QUEUE_STATE_PATH,
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

    for task_name in ("script", "voices", "review", "sanity", "repair", "audacity_export", "m4b_export"):
        process_state[task_name]["logs"] = []
        process_state[task_name]["running"] = False

    project_manager.engine = None

    logger.info("Project state reset")
    return {"status": "reset", "removed": removed}

@app.post("/api/generate_script")
async def generate_script(background_tasks: BackgroundTasks):
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
    background_tasks.add_task(run_process, [sys.executable, "-u", "generate_script.py", input_file], "script", run_id)
    return {"status": "started", "run_id": run_id}

@app.post("/api/review_script")
async def review_script(background_tasks: BackgroundTasks):
    if not os.path.exists(SCRIPT_PATH):
        raise HTTPException(status_code=400, detail="No annotated script found. Generate a script first.")

    run_id = _start_task_run("review")
    background_tasks.add_task(run_process, [sys.executable, "-u", "review_script.py"], "review", run_id)
    return {"status": "started", "run_id": run_id}

@app.post("/api/script_sanity_check")
async def script_sanity_check(background_tasks: BackgroundTasks):
    if not os.path.exists(SCRIPT_PATH):
        raise HTTPException(status_code=400, detail="No annotated script found. Generate a script first.")

    run_id = _start_task_run("sanity")
    background_tasks.add_task(run_script_sanity_task, run_id)
    return {"status": "started", "run_id": run_id}

@app.post("/api/replace_missing_chunks")
async def replace_missing_chunks(background_tasks: BackgroundTasks):
    if not os.path.exists(SCRIPT_PATH):
        raise HTTPException(status_code=400, detail="No annotated script found. Generate a script first.")

    run_id = _start_task_run("repair")
    background_tasks.add_task(run_script_repair_task, run_id)
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

@app.get("/api/voices")
async def get_voices():
    # Parse voices directly from the current script (no stale cache)
    voices_list = []
    if os.path.exists(SCRIPT_PATH):
        try:
            script_data = _load_project_script_document()["entries"]
            voices_set = set()
            for entry in script_data:
                speaker = (entry.get("speaker") or entry.get("type") or "").strip()
                if speaker:
                    voices_set.add(speaker)
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


@app.post("/api/voices/suggest_description")
async def suggest_voice_description(request: VoiceDescriptionSuggestRequest):
    speaker = (request.speaker or "").strip()
    if not speaker:
        raise HTTPException(status_code=400, detail="Speaker is required")

    config = await get_config()
    prompts = config.get("prompts", {})
    prompt_template = (prompts.get("voice_prompt") or "").strip()
    if not prompt_template:
        try:
            prompt_template = load_voice_prompt()
        except RuntimeError as e:
            raise HTTPException(status_code=500, detail=str(e))

    try:
        prompt_payload = project_manager.build_voice_suggestion_prompt(speaker, prompt_template)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    llm_config = config.get("llm", {})
    try:
        def run_request():
            client = OpenAI(
                base_url=llm_config.get("base_url", "http://localhost:11434/v1"),
                api_key=llm_config.get("api_key", "local"),
                timeout=float(llm_config.get("timeout", 600)),
            )
            return client.chat.completions.create(
                model=llm_config.get("model_name", "local-model"),
                messages=[{"role": "user", "content": prompt_payload["prompt"]}],
            )

        response = await asyncio.to_thread(run_request)
    except Exception as e:
        logger.error(f"Voice description suggestion failed for {speaker}: {e}")
        raise HTTPException(status_code=500, detail=f"Voice suggestion request failed: {e}")

    content = response.choices[0].message.content if response.choices else ""
    voice = _extract_voice_field(content)
    if not voice:
        raise HTTPException(status_code=500, detail="Model response did not include a valid JSON voice field")

    return {
        "status": "ok",
        "speaker": speaker,
        "voice": voice,
        "matched_paragraphs": len(prompt_payload["paragraphs"]),
        "context_chars": prompt_payload["context_chars"],
    }


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

@app.get("/api/audiobook")
async def get_audiobook():
    if not os.path.exists(AUDIOBOOK_PATH):
        raise HTTPException(status_code=404, detail="Audiobook not found")
    return FileResponse(AUDIOBOOK_PATH, filename="audiobook.mp3", media_type="audio/mpeg")

# --- Chunk Management Endpoints ---

@app.get("/api/chunks")
async def get_chunks():
    chunks = project_manager.reconcile_chunk_audio_states()
    return chunks

class ChunkRestoreRequest(BaseModel):
    chunk: dict
    at_index: int

@app.post("/api/chunks/restore")
async def restore_chunk(request: ChunkRestoreRequest):
    """Re-insert a previously deleted chunk at a specific index."""
    chunks = project_manager.restore_chunk(request.at_index, request.chunk)
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

@app.post("/api/render_prep_state")
async def set_render_prep_state(request: RenderPrepStateRequest):
    complete = project_manager.set_render_prep_complete(bool(request.complete))
    return {"status": "ok", "render_prep_complete": complete}

@app.post("/api/chunks/{index}")
async def update_chunk(index: int, update: ChunkUpdate):
    data = update.model_dump(exclude_unset=True)
    logger.info(f"Updating chunk {index} with data: {data}")
    chunk = project_manager.update_chunk(index, data)
    if not chunk:
        raise HTTPException(status_code=404, detail="Chunk not found")
    logger.info(f"Chunk {index} updated, instruct is now: '{chunk.get('instruct', '')}'")
    return chunk

@app.post("/api/chunks/{index}/insert")
async def insert_chunk(index: int):
    """Insert an empty chunk after the given index."""
    chunks = project_manager.insert_chunk(index)
    if chunks is None:
        raise HTTPException(status_code=404, detail="Invalid chunk index")
    return {"status": "ok", "total": len(chunks)}

@app.delete("/api/chunks/{index}")
async def delete_chunk(index: int):
    """Delete a chunk at the given index."""
    result = project_manager.delete_chunk(index)
    if result is None:
        raise HTTPException(status_code=400, detail="Cannot delete chunk (invalid index or last remaining chunk)")
    deleted, chunks = result
    return {"status": "ok", "deleted": deleted, "total": len(chunks)}

@app.post("/api/chunks/{index}/generate")
async def generate_chunk_endpoint(index: int, background_tasks: BackgroundTasks):
    chunks = project_manager.load_chunks()
    if not (0 <= index < len(chunks)):
        raise HTTPException(status_code=404, detail="Invalid chunk index")
    if not chunks[index].get("text", "").strip():
        raise HTTPException(status_code=400, detail="Cannot generate audio for an empty line")

    def task():
        project_manager.generate_chunk_audio(index)

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
        try:
            success, msg = project_manager.merge_audio()
            if success:
                process_state["audio"]["logs"].append(f"Merge complete: {msg}")
            else:
                process_state["audio"]["logs"].append(f"Merge failed: {msg}")
        except Exception as e:
            process_state["audio"]["logs"].append(f"Merge error: {e}")
        finally:
            process_state["audio"]["merge_running"] = False
            process_state["audio"]["running"] = False

    background_tasks.add_task(task)
    return {"status": "started"}

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
            success, msg = project_manager.merge_m4b(per_chunk_chapters=request.per_chunk_chapters, metadata=meta)
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
        if f.endswith(".json") and not f.endswith(".voice_config.json"):
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

class ScriptSaveRequest(BaseModel):
    name: str

@app.post("/api/scripts/save")
async def save_script(request: ScriptSaveRequest):
    """Save the current annotated_script.json (and voice_config.json) under a name."""
    if not os.path.exists(SCRIPT_PATH):
        raise HTTPException(status_code=404, detail="No annotated script to save. Generate a script first.")

    safe_name = _sanitize_name(request.name)
    if not safe_name:
        raise HTTPException(status_code=400, detail="Invalid script name.")

    dest = os.path.join(SCRIPTS_DIR, f"{safe_name}.json")
    shutil.copy2(SCRIPT_PATH, dest)

    if os.path.exists(VOICE_CONFIG_PATH):
        shutil.copy2(VOICE_CONFIG_PATH, os.path.join(SCRIPTS_DIR, f"{safe_name}.voice_config.json"))

    logger.info(f"Script saved as '{safe_name}'")
    return {"status": "saved", "name": safe_name}

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

    # Delete chunks so they regenerate from the loaded script
    if os.path.exists(CHUNKS_PATH):
        os.remove(CHUNKS_PATH)

    logger.info(f"Script '{request.name}' loaded")
    return {"status": "loaded", "name": request.name}

@app.delete("/api/scripts/{name}")
async def delete_script(name: str):
    """Delete a saved script."""
    filepath = os.path.join(SCRIPTS_DIR, f"{name}.json")
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail=f"Saved script '{name}' not found.")

    os.remove(filepath)
    companion = os.path.join(SCRIPTS_DIR, f"{name}.voice_config.json")
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
