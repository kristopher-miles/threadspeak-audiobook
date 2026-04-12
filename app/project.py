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
    load_script_document,
)
from source_document import load_source_document, iter_document_paragraphs

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

    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.script_path = os.path.join(root_dir, "annotated_script.json")
        self.chunks_path = os.path.join(root_dir, "chunks.json")
        self.backups_dir = os.path.join(root_dir, "backups")
        self.chunks_backups_dir = os.path.join(self.backups_dir, "chunks")
        self.chunks_latest_backup_path = os.path.join(self.chunks_backups_dir, "chunks.latest.json")
        self.chunks_best_backup_path = os.path.join(self.chunks_backups_dir, "chunks.most_audio.json")
        self.voicelines_dir = os.path.join(root_dir, "voicelines")
        self.voice_config_path = os.path.join(root_dir, "voice_config.json")
        self.config_path = os.path.join(root_dir, "app", "config.json")
        self.transcription_cache_path = os.path.join(root_dir, "transcription_cache.json")
        self.paragraphs_path = os.path.join(root_dir, "paragraphs.json")

        # Ensure voicelines dir exists
        os.makedirs(self.voicelines_dir, exist_ok=True)
        os.makedirs(self.chunks_backups_dir, exist_ok=True)

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
        runtime_settings = self._load_chunk_runtime_settings()
        self._postprocess_workers = runtime_settings["saveback_workers"]
        self._chunk_state_flush_interval_s = runtime_settings["chunk_state_flush_ms"] / 1000.0
        self._chunk_state_flush_batch_size = runtime_settings["chunk_state_flush_batch_size"]
        self._postprocess_queue: queue.Queue = queue.Queue(maxsize=self._postprocess_workers * 2)
        self._postprocess_threads = []
        self._chunks_flush_thread = threading.Thread(
            target=self._chunks_flush_loop,
            daemon=True,
            name=f"chunk-flush-{os.path.basename(self.root_dir)}",
        )
        self._chunks_flush_thread.start()
        for worker_index in range(self._postprocess_workers):
            thread = threading.Thread(
                target=self._postprocess_worker_loop,
                daemon=True,
                name=f"postprocess-{os.path.basename(self.root_dir)}-{worker_index + 1}",
            )
            thread.start()
            self._postprocess_threads.append(thread)
        atexit.register(self.flush_dirty_chunks, True)
