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
from pydantic import BaseModel, field_validator
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
from script_store import apply_dictionary_to_text, clean_dictionary_entries, load_script_document, save_script_document
from source_document import load_source_document
from script_sanity import build_attribution_classifier, run_script_sanity_check
from script_repair import RepairSupersededError, repair_invalid_chunks

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ThreadspeakUI")

app = FastAPI(title="Threadspeak Audiobook")

# Paths
# shared.py lives in app/api, but runtime paths must stay compatible with the
# original app.py location in app/.
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
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
AUDIO_CANCEL_TOMBSTONE_PATH = os.path.join(ROOT_DIR, "audio_cancel_tombstone.json")
SCRIPT_SANITY_PATH = os.path.join(ROOT_DIR, "script_sanity_check.json")
SCRIPT_REPAIR_TRACE_PATH = os.path.join(ROOT_DIR, "script_repair_trace.jsonl")
DESIGNED_VOICES_DIR = os.path.join(ROOT_DIR, "designed_voices")
CLONE_VOICES_DIR = os.path.join(ROOT_DIR, "clone_voices")
LORA_MODELS_DIR = os.path.join(ROOT_DIR, "lora_models")
LORA_DATASETS_DIR = os.path.join(ROOT_DIR, "lora_datasets")
BUILTIN_LORA_DIR = os.path.join(ROOT_DIR, "builtin_lora")
BUILTIN_LORA_MANIFEST = os.path.join(BUILTIN_LORA_DIR, "manifest.json")
DATASET_BUILDER_DIR = os.path.join(ROOT_DIR, "dataset_builder")
DESIGNED_VOICES_MANIFEST = os.path.join(DESIGNED_VOICES_DIR, "manifest.json")
CLONE_VOICES_MANIFEST = os.path.join(CLONE_VOICES_DIR, "manifest.json")
ALLOWED_AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg"}
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


def _load_project_script_document():
    if not os.path.exists(SCRIPT_PATH):
        return {"entries": [], "dictionary": []}
    return load_script_document(SCRIPT_PATH)


def _load_project_dictionary_entries():
    return _load_project_script_document()["dictionary"]


def _apply_project_dictionary(text):
    return apply_dictionary_to_text(text, _load_project_dictionary_entries())[0]


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
    llm_workers: int = 1  # concurrent LLM requests

    @field_validator("base_url")
    @classmethod
    def normalize_base_url(cls, v: str) -> str:
        url = v.rstrip("/")
        if not url.endswith("/v1"):
            url = url + "/v1"
        return url

class TTSConfig(BaseModel):
    mode: str = "local"  # "local" or "external"
    local_backend: str = "auto"  # local mode only: "auto", "qwen", "mlx"
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
    script_max_length: int = 100  # Max chars per chunk in Create Script (-1 = one chunk per sentence)

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
        "corr_id": job.get("corr_id"),
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
            # Persisted markers are authoritative for restore/resume workflows;
            # fall back to file existence for legacy projects.
            if markers.get(stage_name) or os.path.exists(SCRIPT_PATH):
                completed.append(stage_name)
            continue
        if stage_name == "review" and markers.get(stage_name):
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
            indices=active_job.get("indices"),
            generation_token=active_job.get("run_token"),
        )
    else:
        reset_count = project_manager.reset_generating_chunks()

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
    dictionary_entries = project_manager.load_dictionary_entries()

    reconciled_indices = [idx for idx in indices if 0 <= idx < len(chunks)]
    pending_indices = []
    processed_clips = 0
    error_clips = 0

    for idx in reconciled_indices:
        chunk = chunks[idx]
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
        _clear_audio_cancel_tombstone_locked()
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
            _append_audio_log_locked("[QUEUE] Rejecting enqueue request: no valid non-empty indices after resolution.")
            raise HTTPException(status_code=400, detail="No non-empty chunk indices provided")

        job = {
            "id": audio_job_counter,
            "corr_id": f"audio-{audio_job_counter:05d}-{uuid.uuid4().hex[:8]}",
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
        "corr_id": f"audio-{audio_job_counter:05d}-{uuid.uuid4().hex[:8]}",
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
        preloaded_chunks = project_manager.load_chunks()
        has_valid_chunks = any(c.get("text") or c.get("voice") for c in preloaded_chunks)
    except Exception:
        preloaded_chunks = []
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
        chunks = preloaded_chunks
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
                "corr_id": (raw_job.get("corr_id") or f"audio-{int(raw_job.get('id', 0) or 0):05d}-{uuid.uuid4().hex[:8]}"),
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
                    "corr_id": (raw_current.get("corr_id") or f"audio-{int(raw_current.get('id', 0) or 0):05d}-{uuid.uuid4().hex[:8]}"),
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
        if restored_jobs:
            # Persist any status repairs made while reconciling/restoring.
            project_manager.save_chunks(chunks)
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
            f"[RECOVER] Re-queued {len(retry_job['indices'])} stalled chunk(s) from job #{job['id']} ({job.get('corr_id')}) to the front of the queue as #{retry_job['id']} ({retry_job.get('corr_id')})"
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
        f"[CANCEL] Abandoning job #{job['id']} ({job.get('corr_id')}) status={status} reason='{reason}' run_token={run_token} pending={len(job.get('pending_indices', []))}"
    )
    reset_count = project_manager.reset_generating_chunks(
        indices=job.get("indices"),
        generation_token=run_token,
    )
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
            return process_state["audio"]["cancel"] or audio_cancel_event.is_set()

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
    except BaseException as e:
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
    global audio_current_job, audio_recovery_request, audio_current_runner_thread, audio_current_runner_token

    while True:
        with audio_queue_condition:
            while not audio_queue:
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
            job["run_token"] = uuid.uuid4().hex
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
            _base_url = llm_config.get("base_url", "http://localhost:11434/v1").rstrip("/")
            if not _base_url.endswith("/v1"):
                _base_url += "/v1"
            client = OpenAI(
                base_url=_base_url,
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
        script_max_length = int(tts.get("script_max_length", 100))
    except Exception:
        script_max_length = 100

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
         paragraphs_path, voice_config_path, script_output_path, chunks_output_path,
         "--max-length", str(script_max_length)],
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
    """Derive complete stages for new-mode workflow (marker-authoritative)."""
    options = options or {}
    allowed = set(_new_mode_stage_sequence(options))
    markers = _load_new_mode_stage_markers()
    return [stage for stage in NEW_MODE_STAGE_ORDER if stage in allowed and markers.get(stage)]


def _derived_new_mode_completed_stages_from_files(options=None) -> list:
    """Best-effort one-time migration source for missing marker state."""
    options = options or {}
    allowed = set(_new_mode_stage_sequence(options))
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

    return [stage for stage in completed if stage in allowed]


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

    paragraphs_path = os.path.join(ROOT_DIR, "paragraphs.json")
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
