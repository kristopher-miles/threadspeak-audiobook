import os
import json
import atexit
import shutil
import subprocess
import inspect
import threading
import queue
import concurrent.futures
import zipfile
import io
import re
import time
import copy
import tempfile
import uuid
import hashlib
from types import SimpleNamespace
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from tts import (
    TTSEngine,
    combine_audio_with_pauses,
    sanitize_filename,
    DEFAULT_PAUSE_MS,
    SAME_SPEAKER_PAUSE_MS
)
from audio_validation import get_audio_duration_seconds, validate_audio_clip
from audio_validation import estimate_expected_duration_seconds
from asr import LocalASREngine, LocalASRUnavailableError
from pydub import AudioSegment
from pydub.silence import detect_nonsilent
from ffmpeg_utils import configure_pydub, get_ffmpeg_exe, get_ffprobe_exe
from script_store import (
    apply_dictionary_to_text,
)
from source_document import load_source_document, iter_document_paragraphs
from script_provider import create_script_store

from project_core.constants import (
    MAX_CHUNK_CHARS,
    PROOFREAD_DURATION_OUTLIER_SECONDS,
    PROOFREAD_LONG_CHUNK_WORD_THRESHOLD,
    PROOFREAD_LONG_CHUNK_DURATION_OUTLIER_SECONDS,
    PROOFREAD_SHORT_AUDIO_FORCE_ASR_SECONDS,
    PROOFREAD_BATCH_COMMIT_SIZE,
    REPAIR_BATCH_COMMIT_SIZE,
    CHAPTER_HEADING_RE,
    COMMON_PROOFREAD_ABBREVIATIONS,
    TRIM_CACHE_VERSION,
    TRIM_SILENCE_THRESHOLD_DBFS,
    TRIM_MIN_SILENCE_LEN_MS,
    TRIM_KEEP_PADDING_MS,
    VOICE_AUDIT_LOG_ENABLED_DEFAULT,
)
from project_core.chunking import (
    _coerce_bool,
    get_speaker,
    _is_structural_text,
    _extract_chapter_name,
    _build_chunk,
    group_into_chunks,
    script_entries_to_chunks,
)
from project_core.mixins.io_state import ProjectIOStateMixin
from project_core.mixins.runtime_state import ProjectRuntimeStateMixin
from project_core.mixins.chunk_store import ProjectChunkStoreMixin
from project_core.mixins.chunk_editing import ProjectChunkEditingMixin
from project_core.mixins.voice import ProjectVoiceMixin
from project_core.mixins.proofread_asr import ProjectProofreadASRMixin
from project_core.mixins.audio_repair import ProjectAudioRepairMixin
from project_core.mixins.audio_export import ProjectAudioExportMixin
from project_core.mixins.generation import ProjectGenerationMixin
from runtime_layout import LAYOUT

configure_pydub(AudioSegment)

class ProjectManager(
    ProjectIOStateMixin,
    ProjectRuntimeStateMixin,
    ProjectChunkStoreMixin,
    ProjectChunkEditingMixin,
    ProjectVoiceMixin,
    ProjectProofreadASRMixin,
    ProjectAudioRepairMixin,
    ProjectAudioExportMixin,
    ProjectGenerationMixin,
):
    DEFAULT_NARRATOR_THRESHOLD = 10
    _HEADING_RE = re.compile(
        r'^(chapter|part|book|volume|prologue|epilogue|introduction|conclusion|act|section)\b',
        re.IGNORECASE
    )

    def __init__(self, root_dir, *, config_path=None):
        self.root_dir = root_dir
        using_default_layout = os.path.abspath(root_dir) == os.path.abspath(LAYOUT.project_dir)
        self._using_default_runtime_layout = using_default_layout
        self.chunks_db_path = LAYOUT.chunks_db_path if using_default_layout else os.path.join(root_dir, "chunks.sqlite3")
        self.chunks_queue_log_path = LAYOUT.chunks_queue_log_path if using_default_layout else os.path.join(root_dir, "chunks.queue.log")
        self.voice_audit_log_path = LAYOUT.voice_audit_log_path if using_default_layout else os.path.join(root_dir, "voice_state.audit.jsonl")
        # Internal chunk persistence still routes through this marker path, but
        # writes are intercepted and committed to SQLite instead of creating a
        # project-side chunk file.
        self.chunks_path = os.path.join(self.root_dir, ".chunk_store_state")
        self.backups_dir = LAYOUT.backups_dir if using_default_layout else os.path.join(root_dir, "backups")
        self.chunks_backups_dir = LAYOUT.chunk_backups_dir if using_default_layout else os.path.join(self.backups_dir, "chunks")
        self.voicelines_dir = LAYOUT.voicelines_dir if using_default_layout else os.path.join(root_dir, "voicelines")
        self.exports_dir = LAYOUT.exports_dir if using_default_layout else root_dir
        self.audio_finalize_spool_dir = (
            os.path.join(LAYOUT.runs_dir, "audio-finalize", "spool")
            if using_default_layout else os.path.join(self.voicelines_dir, ".finalize_spool")
        )
        default_config_path = os.path.join(root_dir, "app", "config.json")
        self.config_path = config_path or (os.path.join(LAYOUT.app_dir, "config.json") if using_default_layout else default_config_path)

        # Ensure voicelines dir exists
        os.makedirs(self.voicelines_dir, exist_ok=True)
        os.makedirs(self.audio_finalize_spool_dir, exist_ok=True)

        self.engine = None
        self.asr_engine = None
        self._chunks_lock = threading.RLock()  # Thread-safe durable chunk writes
        self._chunks_snapshot_lock = threading.Lock()
        self._chunks_snapshot = None
        self._chunk_runtime_lock = threading.Lock()
        self._chunk_runtime = {}
        self._dirty_chunk_uids = set()
        self._chunks_flush_lock = threading.Lock()
        self._chunks_flush_condition = threading.Condition()
        self._transcription_cache_lock = threading.Lock()
        self._transcription_cache = None
        self.voice_audit_logging_enabled = VOICE_AUDIT_LOG_ENABLED_DEFAULT
        runtime_settings = self._load_chunk_runtime_settings()
        self._finalizer_workers = runtime_settings["finalizer_workers"]
        self._chunk_state_flush_interval_s = runtime_settings["chunk_state_flush_ms"] / 1000.0
        self._chunk_state_flush_batch_size = runtime_settings["chunk_state_flush_batch_size"]
        self._audio_finalize_listener_lock = threading.Lock()
        self._audio_finalize_listeners = {}
        self._audio_finalize_queue = queue.Queue()
        self._audio_finalize_tasks_lock = threading.Lock()
        self._audio_finalize_tasks = {}
        self._audio_finalize_persist_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix=f"audio-finalize-ledger-{os.path.basename(self.root_dir)}",
        )
        self._audio_finalize_threads = []
        self.script_store = None
        self._init_script_store()
        self._restore_audio_finalize_tasks_from_store()
        self._chunks_flush_thread = threading.Thread(
            target=self._chunks_flush_loop,
            daemon=True,
            name=f"chunk-flush-{os.path.basename(self.root_dir)}",
        )
        self._chunks_flush_thread.start()
        for worker_index in range(self._finalizer_workers):
            thread = threading.Thread(
                target=self._audio_finalize_worker_loop,
                daemon=True,
                name=f"audio-finalizer-{os.path.basename(self.root_dir)}-{worker_index + 1}",
            )
            thread.start()
            self._audio_finalize_threads.append(thread)
        atexit.register(self.flush_dirty_chunks, True)
        atexit.register(self.shutdown_script_store)

    def _init_script_store(self):
        self.script_store = create_script_store(
            root_dir=self.root_dir,
            db_path=self.chunks_db_path,
            queue_log_path=self.chunks_queue_log_path,
            state_path=os.path.join(self.root_dir, "state.json"),
            archive_dir=self.chunks_backups_dir,
            voice_audit_log_path=self.voice_audit_log_path,
        )
        self.script_store.start()

    def shutdown_script_store(self, flush=True):
        executor = getattr(self, "_audio_finalize_persist_executor", None)
        if executor is not None:
            executor.shutdown(wait=flush, cancel_futures=False)
            self._audio_finalize_persist_executor = None
        if self.script_store is None:
            return
        self.script_store.stop(flush=flush)

    def reload_script_store(self, clear_runtime=True):
        self.shutdown_script_store(flush=True)
        if clear_runtime:
            with self._chunks_snapshot_lock:
                self._chunks_snapshot = None
            with self._chunk_runtime_lock:
                self._chunk_runtime = {}
                self._dirty_chunk_uids.clear()
        self._init_script_store()
        self._audio_finalize_persist_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix=f"audio-finalize-ledger-{os.path.basename(self.root_dir)}",
        )
        self._restore_audio_finalize_tasks_from_store()
