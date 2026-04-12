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

MAX_CHUNK_CHARS = 500
PROOFREAD_DURATION_OUTLIER_SECONDS = 10.0
PROOFREAD_LONG_CHUNK_WORD_THRESHOLD = 25
PROOFREAD_LONG_CHUNK_DURATION_OUTLIER_SECONDS = 25.0
PROOFREAD_SHORT_AUDIO_FORCE_ASR_SECONDS = 2.0
PROOFREAD_BATCH_COMMIT_SIZE = 25
REPAIR_BATCH_COMMIT_SIZE = 25
CHAPTER_HEADING_RE = re.compile(
    r'^(chapter|part|book|volume|prologue|epilogue|introduction|conclusion|act|section)\b',
    re.IGNORECASE,
)
COMMON_PROOFREAD_ABBREVIATIONS = {
    "adm": ("admiral",),
    "approx": ("approximately",),
    "capt": ("captain",),
    "cmdr": ("commander",),
    "col": ("colonel",),
    "dept": ("department",),
    "dr": ("doctor",),
    "etc": ("et", "cetera"),
    "gen": ("general",),
    "gov": ("governor",),
    "jr": ("junior",),
    "lt": ("lieutenant",),
    "mr": ("mister",),
    "mrs": ("missus",),
    "ms": ("miss",),
    "no": ("number",),
    "pres": ("president",),
    "prof": ("professor",),
    "rep": ("representative",),
    "rev": ("reverend",),
    "sen": ("senator",),
    "sgt": ("sergeant",),
    "sr": ("senior",),
    "vs": ("versus",),
}
TRIM_CACHE_VERSION = 4
TRIM_SILENCE_THRESHOLD_DBFS = -50.0
TRIM_MIN_SILENCE_LEN_MS = 150
TRIM_KEEP_PADDING_MS = 40

configure_pydub(AudioSegment)

def _coerce_bool(value, default=False):
    if isinstance(value, bool):
        return value
    if value is None:
        return bool(default)
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"1", "true", "yes", "on"}:
            return True
        if text in {"0", "false", "no", "off", ""}:
            return False
    return bool(value)

def get_speaker(entry):
    """Get speaker from entry, checking both 'speaker' and 'type' fields."""
    return entry.get("speaker") or entry.get("type") or ""


def _is_structural_text(text):
    """Check if text is a title, chapter heading, dedication, or other structural fragment."""
    stripped = text.strip()
    if not stripped:
        return True
    # Very short and not a full sentence (no sentence-ending punctuation)
    if len(stripped) < 80 and not stripped[-1] in '.!?':
        return True
    return False


def _extract_chapter_name(entry):
    chapter = (entry.get("chapter") or "").strip()
    if chapter:
        return chapter

    text = (entry.get("text") or "").strip()
    if text and CHAPTER_HEADING_RE.match(text):
        return text

    return None


def _build_chunk(speaker, text, instruct, chapter=None, paragraph_id=None):
    chunk = {
        "speaker": speaker,
        "text": text,
        "instruct": instruct,
        "uid": uuid.uuid4().hex,
    }
    if chapter:
        chunk["chapter"] = chapter
    if paragraph_id:
        chunk["paragraph_id"] = paragraph_id
    return chunk


def group_into_chunks(script_entries, max_chars=MAX_CHUNK_CHARS):
    """Group consecutive entries by same speaker into chunks up to max_chars"""
    if not script_entries:
        return []

    chunks = []
    current_speaker = get_speaker(script_entries[0])
    current_text = script_entries[0].get("text", "")
    current_instruct = script_entries[0].get("instruct", "")
    current_chapter = _extract_chapter_name(script_entries[0])
    current_paragraph_id = script_entries[0].get("paragraph_id")

    for entry in script_entries[1:]:
        speaker = get_speaker(entry)
        text = entry.get("text", "")
        instruct = entry.get("instruct", "")
        entry_chapter = _extract_chapter_name(entry)
        effective_chapter = entry_chapter or current_chapter
        entry_paragraph_id = entry.get("paragraph_id")

        # Don't merge structural text (titles, chapter headings, dedications)
        if (speaker == current_speaker and instruct == current_instruct
                and effective_chapter == current_chapter
                and not _is_structural_text(current_text)
                and not _is_structural_text(text)):
            combined = current_text + " " + text
            if len(combined) <= max_chars:
                current_text = combined
                # Track the latest paragraph_id so the chunk reflects where it ends
                current_paragraph_id = entry_paragraph_id or current_paragraph_id
            else:
                chunks.append(_build_chunk(current_speaker, current_text, current_instruct, current_chapter, current_paragraph_id))
                current_text = text
                current_instruct = instruct
                current_chapter = effective_chapter
                current_paragraph_id = entry_paragraph_id
        else:
            chunks.append(_build_chunk(current_speaker, current_text, current_instruct, current_chapter, current_paragraph_id))
            current_speaker = speaker
            current_text = text
            current_instruct = instruct
            current_chapter = effective_chapter
            current_paragraph_id = entry_paragraph_id

    # Don't forget the last chunk
    chunks.append(_build_chunk(current_speaker, current_text, current_instruct, current_chapter, current_paragraph_id))

    return chunks


def script_entries_to_chunks(script_entries, max_chars=MAX_CHUNK_CHARS):
    """Build chunks from script entries.

    For sentence-level scripts produced by create_script.py (which include
    paragraph_id), preserve a strict 1:1 mapping between entries and chunks.
    Legacy scripts without paragraph_id keep the historical merge behavior.
    """
    if not script_entries:
        return []

    has_paragraph_ids = any(bool(entry.get("paragraph_id")) for entry in script_entries)
    if not has_paragraph_ids:
        return group_into_chunks(script_entries, max_chars=max_chars)

    chunks = []
    for entry in script_entries:
        chunks.append(
            _build_chunk(
                get_speaker(entry),
                entry.get("text", ""),
                entry.get("instruct", ""),
                _extract_chapter_name(entry),
                entry.get("paragraph_id"),
            )
        )
    return chunks

class ProjectManager:
    DEFAULT_NARRATOR_THRESHOLD = 10

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

    def _load_chunk_runtime_settings(self):
        tts_settings = self._load_tts_settings()

        def _coerce_positive_int(value, default):
            try:
                parsed = int(value)
            except (TypeError, ValueError):
                return default
            return parsed if parsed > 0 else default

        return {
            "saveback_workers": _coerce_positive_int(tts_settings.get("saveback_workers", 2), 2),
            "chunk_state_flush_ms": _coerce_positive_int(tts_settings.get("chunk_state_flush_ms", 250), 250),
            "chunk_state_flush_batch_size": _coerce_positive_int(
                tts_settings.get("chunk_state_flush_batch_size", 25),
                25,
            ),
        }

    def _copy_chunks_snapshot(self):
        with self._chunks_snapshot_lock:
            if self._chunks_snapshot is None:
                return None
            return copy.deepcopy(self._chunks_snapshot)

    def _set_chunks_snapshot(self, chunks):
        with self._chunks_snapshot_lock:
            self._chunks_snapshot = copy.deepcopy(chunks)

    def _copy_chunk_runtime(self):
        with self._chunk_runtime_lock:
            return copy.deepcopy(self._chunk_runtime)

    @staticmethod
    def _runtime_chunk_fields():
        return (
            "status",
            "generation_token",
            "audio_path",
            "audio_validation",
            "auto_regen_count",
            "updated_at",
        )

    def _merge_runtime_chunk(self, chunk, runtime_chunk):
        if not runtime_chunk:
            return chunk
        merged = dict(chunk)
        for field in self._runtime_chunk_fields():
            if field in runtime_chunk:
                merged[field] = copy.deepcopy(runtime_chunk[field])
        if merged.get("status") != "generating":
            merged.pop("generation_token", None)
        return merged

    def set_chunk_runtime(self, uid, **fields):
        if not uid:
            return {}
        with self._chunk_runtime_lock:
            runtime_chunk = self._chunk_runtime.setdefault(uid, {})
            for key, value in fields.items():
                if key == "generation_token":
                    if value is None:
                        runtime_chunk.pop("generation_token", None)
                    else:
                        runtime_chunk["generation_token"] = value
                else:
                    runtime_chunk[key] = copy.deepcopy(value)
            runtime_chunk["updated_at"] = time.time()
            return copy.deepcopy(runtime_chunk)

    def clear_chunk_runtime(self, uid, fields=None):
        if not uid:
            return False
        with self._chunk_runtime_lock:
            runtime_chunk = self._chunk_runtime.get(uid)
            if runtime_chunk is None:
                return False
            if fields is None:
                self._chunk_runtime.pop(uid, None)
            else:
                for field in fields:
                    runtime_chunk.pop(field, None)
                if not runtime_chunk:
                    self._chunk_runtime.pop(uid, None)
            self._dirty_chunk_uids.discard(uid)
            return True

    def mark_chunks_dirty(self, uids):
        dirty_count = 0
        queued = False
        with self._chunk_runtime_lock:
            for uid in uids:
                if not uid or uid not in self._chunk_runtime:
                    continue
                self._dirty_chunk_uids.add(uid)
                queued = True
            dirty_count = len(self._dirty_chunk_uids)
        if queued:
            with self._chunks_flush_condition:
                self._chunks_flush_condition.notify_all()
        if dirty_count >= self._chunk_state_flush_batch_size:
            self.flush_dirty_chunks(force=False)
        return dirty_count

    def _chunks_flush_loop(self):
        while True:
            with self._chunks_flush_condition:
                while not self._dirty_chunk_uids:
                    self._chunks_flush_condition.wait()
                if len(self._dirty_chunk_uids) < self._chunk_state_flush_batch_size:
                    self._chunks_flush_condition.wait(timeout=self._chunk_state_flush_interval_s)
            self.flush_dirty_chunks(force=False)

    def load_paragraphs(self):
        """Return the paragraphs.json document, or None if it does not exist yet."""
        if not os.path.exists(self.paragraphs_path):
            return None
        with open(self.paragraphs_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def save_paragraphs(self, data: dict):
        """Atomically write paragraphs data to paragraphs.json."""
        tmp = self.paragraphs_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        os.replace(tmp, self.paragraphs_path)

    def _load_voice_config(self):
        if os.path.exists(self.voice_config_path):
            try:
                with open(self.voice_config_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, ValueError):
                return {}
        return {}

    def _save_voice_config(self, voice_config):
        with open(self.voice_config_path, "w", encoding="utf-8") as f:
            json.dump(voice_config, f, indent=2, ensure_ascii=False)

    def _load_app_config(self):
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, ValueError):
                return {}
        return {}

    def _load_transcription_cache_locked(self):
        if self._transcription_cache is not None:
            return self._transcription_cache

        cache = {}
        if os.path.exists(self.transcription_cache_path):
            try:
                with open(self.transcription_cache_path, "r", encoding="utf-8") as f:
                    payload = json.load(f)
                entries = payload.get("entries", []) if isinstance(payload, dict) else payload
                for entry in entries or []:
                    if not isinstance(entry, dict):
                        continue
                    key = str(entry.get("key") or "").strip()
                    if not key:
                        continue
                    cache[key] = dict(entry)
            except (json.JSONDecodeError, ValueError, OSError):
                cache = {}

        self._transcription_cache = cache
        return self._transcription_cache

    def _save_transcription_cache_locked(self):
        entries = list((self._transcription_cache or {}).values())
        self._atomic_json_write({"entries": entries}, self.transcription_cache_path)

    @staticmethod
    def _transcription_cache_key(filename, size_bytes):
        return f"{filename}|{int(size_bytes)}"

    def _lookup_cached_transcription(self, relative_audio_path):
        full_path = os.path.join(self.root_dir, relative_audio_path)
        if not os.path.exists(full_path):
            return None

        filename = os.path.basename(relative_audio_path)
        size_bytes = os.path.getsize(full_path)
        key = self._transcription_cache_key(filename, size_bytes)
        with self._transcription_cache_lock:
            cache = self._load_transcription_cache_locked()
            entry = cache.get(key)
            if not entry:
                return None
            return {
                "text": entry.get("text", ""),
                "normalized_text": entry.get("normalized_text", ""),
                "cached": True,
                "filename": filename,
                "size_bytes": size_bytes,
            }

    def _store_cached_transcription(self, relative_audio_path, result):
        full_path = os.path.join(self.root_dir, relative_audio_path)
        if not os.path.exists(full_path):
            return

        filename = os.path.basename(relative_audio_path)
        size_bytes = os.path.getsize(full_path)
        entry = {
            "key": self._transcription_cache_key(filename, size_bytes),
            "filename": filename,
            "size_bytes": size_bytes,
            "text": result.get("text", ""),
            "normalized_text": result.get("normalized_text") or self._normalize_asr_text(result.get("text", "")),
            "updated_at": time.time(),
        }
        with self._transcription_cache_lock:
            cache = self._load_transcription_cache_locked()
            cache[entry["key"]] = entry
            self._save_transcription_cache_locked()

    def _copy_cached_transcription_key(self, source_relative_audio_path, target_relative_audio_path):
        source_full_path = os.path.join(self.root_dir, source_relative_audio_path)
        target_full_path = os.path.join(self.root_dir, target_relative_audio_path)
        if not os.path.exists(target_full_path):
            return

        source_filename = os.path.basename(source_relative_audio_path)
        target_size_bytes = os.path.getsize(target_full_path)
        source_size_bytes = os.path.getsize(source_full_path) if os.path.exists(source_full_path) else target_size_bytes
        source_key = self._transcription_cache_key(source_filename, source_size_bytes)

        target_filename = os.path.basename(target_relative_audio_path)
        target_key = self._transcription_cache_key(target_filename, target_size_bytes)

        with self._transcription_cache_lock:
            cache = self._load_transcription_cache_locked()
            entry = cache.get(source_key)
            if not entry:
                return
            cache[target_key] = {
                **dict(entry),
                "key": target_key,
                "filename": target_filename,
                "size_bytes": target_size_bytes,
                "updated_at": time.time(),
            }
            self._save_transcription_cache_locked()

    def _current_script_title(self):
        state_path = os.path.join(self.root_dir, "state.json")
        if os.path.exists(state_path):
            try:
                with open(state_path, "r", encoding="utf-8") as f:
                    state = json.load(f)
                if state.get("loaded_script_name"):
                    return state["loaded_script_name"].strip()
                input_path = state.get("input_file_path") or ""
                if input_path:
                    return os.path.splitext(os.path.basename(input_path))[0].strip()
            except (json.JSONDecodeError, ValueError, OSError):
                pass
        return "Project"

    def _load_state(self):
        state_path = os.path.join(self.root_dir, "state.json")
        if not os.path.exists(state_path):
            return {}
        try:
            with open(state_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, ValueError, OSError):
            return {}

    def _save_state(self, state):
        state_path = os.path.join(self.root_dir, "state.json")
        with open(state_path, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2, ensure_ascii=False)

    @staticmethod
    def _new_chunk_uid():
        return uuid.uuid4().hex

    def _ensure_chunk_uids(self, chunks):
        changed = False
        seen = set()
        for chunk in chunks:
            uid = str(chunk.get("uid") or "").strip()
            if not uid or uid in seen:
                uid = self._new_chunk_uid()
                chunk["uid"] = uid
                changed = True
            seen.add(uid)
        return changed

    def resolve_chunk_index(self, chunk_ref, chunks=None):
        chunks = chunks if chunks is not None else self.load_chunks()
        ref = "" if chunk_ref is None else str(chunk_ref)

        for index, chunk in enumerate(chunks):
            if str(chunk.get("uid") or "") == ref:
                return index

        try:
            numeric_ref = int(ref)
        except (TypeError, ValueError):
            return None

        if 0 <= numeric_ref < len(chunks):
            return numeric_ref

        for index, chunk in enumerate(chunks):
            if chunk.get("id") == numeric_ref:
                return index
        return None

    def is_render_prep_complete(self):
        return bool(self._load_state().get("render_prep_complete"))

    def set_render_prep_complete(self, complete=True):
        state = self._load_state()
        state["render_prep_complete"] = bool(complete)
        self._save_state(state)
        return state["render_prep_complete"]

    @staticmethod
    def _escape_concat_path(path):
        return path.replace("\\", "\\\\").replace("'", r"'\''")

    @classmethod
    def _write_concat_line(cls, handle, path):
        handle.write(f"file '{cls._escape_concat_path(path)}'\n")

    def _resolve_trim_config(self, export_config=None):
        cfg = export_config or self._resolve_export_normalization_config(None)
        return {
            "enabled": _coerce_bool(getattr(cfg, "trim_clip_silence_enabled", True), default=True),
            "silence_threshold_dbfs": float(getattr(cfg, "trim_silence_threshold_dbfs", TRIM_SILENCE_THRESHOLD_DBFS)),
            "min_silence_len_ms": max(1, int(getattr(cfg, "trim_min_silence_len_ms", TRIM_MIN_SILENCE_LEN_MS))),
            "keep_padding_ms": max(0, int(getattr(cfg, "trim_keep_padding_ms", TRIM_KEEP_PADDING_MS))),
        }

    def _trim_cache_dir(self):
        return os.path.join(self.root_dir, "voicelines", ".trim_cache")

    def _build_trim_cache_key(self, full_path, trim_cfg):
        stat = os.stat(full_path)
        rel_path = os.path.relpath(full_path, self.root_dir)
        payload = "|".join([
            str(TRIM_CACHE_VERSION),
            rel_path,
            str(stat.st_size),
            str(getattr(stat, "st_mtime_ns", int(stat.st_mtime * 1_000_000_000))),
            str(trim_cfg["silence_threshold_dbfs"]),
            str(trim_cfg["min_silence_len_ms"]),
            str(trim_cfg["keep_padding_ms"]),
        ])
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()

    @staticmethod
    def _trim_audio_segment_boundaries(segment, trim_cfg):
        if len(segment) <= 0:
            return segment, 0, 0, False

        ranges = detect_nonsilent(
            segment,
            min_silence_len=trim_cfg["min_silence_len_ms"],
            silence_thresh=trim_cfg["silence_threshold_dbfs"],
        )
        if not ranges:
            return segment, 0, 0, False

        first_start = int(ranges[0][0])
        last_end = int(ranges[-1][1])

        keep = trim_cfg["keep_padding_ms"]
        trim_start = max(0, first_start - keep)

        # Trailing keep-padding can reintroduce codec-tail artifacts (especially
        # from MP3 decoder padding) into the trimmed output. Preserve only the
        # leading keep-padding; keep the trailing cut at the detected nonsilent end.
        trim_end = min(len(segment), last_end)
        if trim_end <= trim_start:
            return segment, 0, 0, False

        frame_rate = int(segment.frame_rate or 0)
        channels = max(1, int(segment.channels or 1))
        frame_width = int(segment.frame_width or 0)
        if frame_rate <= 0 or frame_width <= 0:
            lead_removed = trim_start
            tail_removed = len(segment) - trim_end
            return segment[trim_start:trim_end], lead_removed, tail_removed, (lead_removed > 0 or tail_removed > 0)

        samples = segment.get_array_of_samples()
        total_frames = len(samples) // channels
        start_frame = max(0, min(total_frames, int(round(trim_start * frame_rate / 1000.0))))
        end_frame = max(start_frame, min(total_frames, int(round(trim_end * frame_rate / 1000.0))))
        if end_frame <= start_frame:
            return segment, 0, 0, False

        # detect_nonsilent() boundaries are RMS-window based. Snap start to a nearby
        # zero crossing so transitions from inserted silence don't hard-click.
        max_search_frames = max(1, int(round(frame_rate * 0.008)))  # up to ~8ms
        if start_frame < total_frames - 1:
            search_limit = min(total_frames - 1, start_frame + max_search_frames)
            prev = int(samples[start_frame * channels])
            best_frame = start_frame
            best_abs = abs(prev)
            for frame in range(start_frame + 1, search_limit + 1):
                current = int(samples[frame * channels])
                current_abs = abs(current)
                if current_abs < best_abs:
                    best_abs = current_abs
                    best_frame = frame
                if current == 0 or (prev <= 0 < current) or (prev >= 0 > current):
                    start_frame = frame
                    break
                prev = current
            else:
                start_frame = best_frame

        # detect_nonsilent() uses windowed RMS and can end at a non-zero crossing.
        # Snap the cut point to the next nearby zero crossing to avoid hard-edge pops.
        search_limit = min(total_frames - 1, end_frame + max_search_frames)
        if end_frame < total_frames - 1:
            prev = int(samples[(end_frame - 1) * channels]) if end_frame > 0 else int(samples[end_frame * channels])
            best_frame = end_frame
            best_abs = abs(prev)
            for frame in range(end_frame, search_limit + 1):
                current = int(samples[frame * channels])
                current_abs = abs(current)
                if current_abs < best_abs:
                    best_abs = current_abs
                    best_frame = frame
                if current == 0 or (prev <= 0 < current) or (prev >= 0 > current):
                    end_frame = min(total_frames, frame + 1)
                    break
                prev = current
            else:
                end_frame = min(total_frames, best_frame + 1)

        start_byte = start_frame * frame_width
        end_byte = end_frame * frame_width
        trimmed_segment = segment._spawn(segment._data[start_byte:end_byte])

        lead_removed = int(round((start_frame * 1000.0) / frame_rate))
        tail_removed = int(round(((total_frames - end_frame) * 1000.0) / frame_rate))
        return trimmed_segment, lead_removed, tail_removed, (lead_removed > 0 or tail_removed > 0)

    def _resolve_export_audio_path(self, full_path, trim_cfg):
        if not trim_cfg["enabled"]:
            return full_path, {"cache_hit": False, "trimmed": False, "lead_ms": 0, "tail_ms": 0}

        cache_dir = self._trim_cache_dir()
        os.makedirs(cache_dir, exist_ok=True)
        cache_key = self._build_trim_cache_key(full_path, trim_cfg)
        cache_path = os.path.join(cache_dir, f"{cache_key}.mp3")

        if os.path.exists(cache_path):
            try:
                if os.path.getsize(cache_path) > 0:
                    return cache_path, {"cache_hit": True, "trimmed": False, "lead_ms": 0, "tail_ms": 0}
            except OSError:
                pass

        source_segment = AudioSegment.from_file(full_path)
        trimmed_segment, lead_ms, tail_ms, changed = self._trim_audio_segment_boundaries(source_segment, trim_cfg)

        # Safety guard: trimming must never increase duration. If it does,
        # discard the generated segment and cache the original audio instead.
        if len(trimmed_segment) > len(source_segment):
            print(
                f"[trim] ERROR: Trim result longer than original for {full_path} "
                f"(original={len(source_segment)}ms, trimmed={len(trimmed_segment)}ms). "
                "Discarding trimmed output and caching original clip.",
                flush=True,
            )
            trimmed_segment = source_segment
            lead_ms = 0
            tail_ms = 0
            changed = False

        exported = trimmed_segment.export(cache_path, format="mp3", bitrate="128k")
        if hasattr(exported, "close"):
            exported.close()
        if not os.path.exists(cache_path) or os.path.getsize(cache_path) <= 0:
            raise RuntimeError(f"Trim cache export produced invalid file: {cache_path}")

        return cache_path, {
            "cache_hit": False,
            "trimmed": bool(changed),
            "lead_ms": int(lead_ms),
            "tail_ms": int(tail_ms),
        }

    def _collect_merge_timeline(self, progress_callback=None, merge_started_at=None, export_config=None, log_callback=None, temp_dir=None):
        chunks = self.load_chunks()
        timeline = []
        timeline_size_bytes = 0
        trim_cfg = self._resolve_trim_config(export_config)
        trim_stats = {
            "enabled": trim_cfg["enabled"],
            "processed": 0,
            "cache_hits": 0,
            "trimmed_clips": 0,
            "lead_ms_removed": 0,
            "tail_ms_removed": 0,
            "fallback_originals": 0,
        }
        # Build ordered list of (kind, chunk[, full_path]) preserving chunk order
        ordered_items = []
        for chunk in chunks:
            if chunk.get("type") == "silence":
                ordered_items.append(("silence", chunk))
            elif chunk.get("status") == "done":
                path = chunk.get("audio_path")
                if not path:
                    continue
                full_path = os.path.join(self.root_dir, path)
                if not os.path.exists(full_path):
                    continue
                ordered_items.append(("audio", chunk, full_path))
        eligible_chunks = [(c, p) for kind, c, p in (x for x in ordered_items if x[0] == "audio")]
        total_candidates = len(eligible_chunks)

        if progress_callback:
            progress_callback({
                "stage": "preparing",
                "chapter_index": 0,
                "total_chapters": 0,
                "chapter_label": "Scanning completed clips",
                "elapsed_seconds": 0.0,
                "merged_duration_seconds": 0.0,
                "estimated_size_bytes": 0,
                "output_file_size_bytes": 0,
                "total_items": total_candidates,
                "processed_items": 0,
                "remaining_items": total_candidates,
                "percent_complete": 0.0,
                "eta_seconds": None,
                "trim_enabled": bool(trim_cfg["enabled"]),
            })

        # Pre-process audio clips (trim/index), then build final ordered timeline
        last_prepare_bucket = -1
        prepare_started_at = merge_started_at or time.time()
        clip_times = []
        # Map uid -> resolved_path for audio chunks after trim processing
        resolved_audio_paths = {}
        for processed_index, (chunk, full_path) in enumerate(eligible_chunks, start=1):
            clip_start = time.time()
            resolved_path = full_path
            if trim_cfg["enabled"]:
                try:
                    resolved_path, trim_info = self._resolve_export_audio_path(full_path, trim_cfg)
                    trim_stats["processed"] += 1
                    if trim_info["cache_hit"]:
                        trim_stats["cache_hits"] += 1
                    if trim_info["trimmed"]:
                        trim_stats["trimmed_clips"] += 1
                        trim_stats["lead_ms_removed"] += int(trim_info["lead_ms"])
                        trim_stats["tail_ms_removed"] += int(trim_info["tail_ms"])
                except Exception:
                    trim_stats["fallback_originals"] += 1
                    resolved_path = full_path
            clip_times.append(time.time() - clip_start)
            resolved_audio_paths[chunk["uid"]] = resolved_path
            if progress_callback and total_candidates > 0:
                percent_complete = (processed_index / total_candidates) * 100.0
                progress_bucket = int(percent_complete // 5)
                should_emit = (
                    processed_index == 1
                    or processed_index == total_candidates
                    or progress_bucket > last_prepare_bucket
                )
                if should_emit:
                    elapsed = max(0.0, time.time() - prepare_started_at)
                    rate = processed_index / elapsed if elapsed > 0 else 0.0
                    remaining_items = max(total_candidates - processed_index, 0)
                    avg_clip_s = sum(clip_times) / len(clip_times) if clip_times else 0.0
                    eta_seconds = avg_clip_s * remaining_items
                    if trim_cfg["enabled"]:
                        chapter_label = (
                            f"Trimming clips: {int(round(percent_complete))}% "
                            f"({processed_index}/{total_candidates}, {remaining_items} remaining)"
                        )
                    else:
                        chapter_label = (
                            f"Indexing clips: {int(round(percent_complete))}% "
                            f"({processed_index}/{total_candidates}, {remaining_items} remaining)"
                        )
                        eta_seconds = (remaining_items / rate) if rate > 0 and remaining_items > 0 else 0.0
                    progress_callback({
                        "stage": "preparing",
                        "chapter_index": 0,
                        "total_chapters": 0,
                        "chapter_label": chapter_label,
                        "elapsed_seconds": elapsed,
                        "merged_duration_seconds": 0.0,
                        "estimated_size_bytes": timeline_size_bytes,
                        "output_file_size_bytes": 0,
                        "total_items": total_candidates,
                        "processed_items": processed_index,
                        "remaining_items": remaining_items,
                        "percent_complete": percent_complete,
                        "eta_seconds": round(eta_seconds, 1),
                        "trim_enabled": bool(trim_cfg["enabled"]),
                    })
                    if trim_cfg["enabled"]:
                        eta_m, eta_s = divmod(int(eta_seconds), 60)
                        eta_str = f"{eta_m}m{eta_s:02d}s" if eta_m else f"{eta_s}s"
                        print(
                            f"[trim] {processed_index}/{total_candidates} clips"
                            f" — avg {avg_clip_s * 1000:.0f}ms/clip"
                            f" — ETA {eta_str}",
                            flush=True,
                        )
                    last_prepare_bucket = progress_bucket

        # Build final timeline in chunk order, interleaving silence blocks
        for entry in ordered_items:
            kind = entry[0]
            chunk = entry[1]
            if kind == "silence":
                duration_ms = int(float(chunk.get("silence_duration_s", 1.0)) * 1000)
                item = {
                    "chunk": chunk,
                    "full_path": None,  # WAV generated later with correct audio format
                    "file_size_bytes": 0,
                    "is_silence_block": True,
                    "silence_duration_ms": duration_ms,
                }
                timeline.append(item)
            else:
                # audio chunk
                resolved_path = resolved_audio_paths.get(chunk["uid"])
                if resolved_path is None:
                    continue
                item = {
                    "chunk": chunk,
                    "full_path": resolved_path,
                    "file_size_bytes": os.path.getsize(resolved_path),
                }
                timeline.append(item)
                timeline_size_bytes += item["file_size_bytes"]

        if log_callback and trim_cfg["enabled"]:
            total_removed_ms = trim_stats["lead_ms_removed"] + trim_stats["tail_ms_removed"]
            log_callback(
                "Trim cache: "
                f"{trim_stats['trimmed_clips']} clip(s) trimmed, "
                f"{trim_stats['cache_hits']} cache hit(s), "
                f"{total_removed_ms}ms total removed "
                f"(lead {trim_stats['lead_ms_removed']}ms, tail {trim_stats['tail_ms_removed']}ms), "
                f"{trim_stats['fallback_originals']} fallback(s)"
            )

        return timeline

    def _group_timeline_by_chapter(self, timeline):
        chapter_groups = []
        current_label = None
        current_items = []

        for item in timeline:
            chunk = item["chunk"]
            chapter_label = (chunk.get("chapter") or "").strip() or "Unlabeled"
            if current_items and chapter_label != current_label:
                chapter_groups.append((current_label, current_items))
                current_items = [item]
                current_label = chapter_label
            else:
                if not current_items:
                    current_label = chapter_label
                current_items.append(item)

        if current_items:
            chapter_groups.append((current_label, current_items))

        return chapter_groups

    def _emit_merge_progress(
        self,
        progress_callback,
        merge_started_at,
        *,
        stage,
        chapter_index,
        total_chapters,
        chapter_label,
        estimated_size_bytes,
        output_path=None,
        merged_duration_seconds=0.0,
    ):
        if not progress_callback:
            return
        progress_callback({
            "stage": stage,
            "chapter_index": chapter_index,
            "total_chapters": total_chapters,
            "chapter_label": chapter_label,
            "elapsed_seconds": time.time() - merge_started_at,
            "merged_duration_seconds": merged_duration_seconds,
            "estimated_size_bytes": estimated_size_bytes,
            "output_file_size_bytes": os.path.getsize(output_path) if output_path and os.path.exists(output_path) else 0,
        })

    def _create_silence_assets(self, temp_dir, export_config=None, reference_audio_path=None):
        default_ms = getattr(export_config, "silence_between_speakers_ms", DEFAULT_PAUSE_MS)
        same_ms = getattr(export_config, "silence_same_speaker_ms", SAME_SPEAKER_PAUSE_MS)
        chapter_end_ms = getattr(export_config, "silence_end_of_chapter_ms", 3000)
        paragraph_ms = getattr(export_config, "silence_paragraph_ms", 750)

        reference_frame_rate = None
        reference_channels = None
        reference_sample_width = None
        if reference_audio_path and os.path.exists(reference_audio_path):
            try:
                reference_segment = AudioSegment.from_file(reference_audio_path)
                reference_frame_rate = int(reference_segment.frame_rate or 0) or None
                reference_channels = int(reference_segment.channels or 0) or None
                reference_sample_width = int(reference_segment.sample_width or 0) or None
            except Exception:
                reference_frame_rate = None
                reference_channels = None
                reference_sample_width = None

        # Silence files must be MP3 so they are the same codec as the audio clips.
        # FFmpeg's concat demuxer skips files whose codec differs from the first
        # entry in the concat list, which caused all silence to be dropped silently.
        default_silence_path = os.path.join(temp_dir, "pause_default.mp3")
        same_silence_path = os.path.join(temp_dir, "pause_same_speaker.mp3")
        chapter_end_silence_path = os.path.join(temp_dir, "pause_chapter_end.mp3")
        paragraph_silence_path = os.path.join(temp_dir, "pause_paragraph.mp3")

        def _write(path, duration_ms):
            seg = AudioSegment.silent(duration=max(0, duration_ms))
            if reference_frame_rate:
                seg = seg.set_frame_rate(reference_frame_rate)
            if reference_channels:
                seg = seg.set_channels(reference_channels)
            if reference_sample_width:
                seg = seg.set_sample_width(reference_sample_width)
            exp = seg.export(path, format="mp3", parameters=["-q:a", "2"])
            if hasattr(exp, "close"):
                exp.close()

        _write(default_silence_path, default_ms)
        _write(same_silence_path, same_ms)
        _write(chapter_end_silence_path, chapter_end_ms)
        _write(paragraph_silence_path, paragraph_ms)

        return {
            "default_path": default_silence_path,
            "same_path": same_silence_path,
            "chapter_end_path": chapter_end_silence_path,
            "paragraph_path": paragraph_silence_path,
            "default_size_bytes": os.path.getsize(default_silence_path),
            "same_size_bytes": os.path.getsize(same_silence_path),
            "chapter_end_size_bytes": os.path.getsize(chapter_end_silence_path),
            "paragraph_size_bytes": os.path.getsize(paragraph_silence_path),
            # Reference format for use when generating additional silence WAVs
            "_write": _write,
        }

    @staticmethod
    def _pick_silence(prev_item, curr_item, silence_assets, *, is_chapter_boundary=False):
        """Return (pause_path, pause_size_bytes) for the gap between two timeline items."""
        if is_chapter_boundary:
            return silence_assets["chapter_end_path"], silence_assets["chapter_end_size_bytes"]
        prev_pid = prev_item["chunk"].get("paragraph_id")
        curr_pid = curr_item["chunk"].get("paragraph_id")
        if prev_pid and curr_pid and prev_pid != curr_pid:
            return silence_assets["paragraph_path"], silence_assets["paragraph_size_bytes"]
        if prev_item["chunk"]["speaker"] == curr_item["chunk"]["speaker"]:
            return silence_assets["same_path"], silence_assets["same_size_bytes"]
        return silence_assets["default_path"], silence_assets["default_size_bytes"]

    def _run_ffmpeg_concat(self, concat_path, output_path, codec_args, progress_tick=None):
        command = [
            get_ffmpeg_exe(),
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-progress",
            "pipe:1",
            "-nostats",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            concat_path,
            "-vn",
            *codec_args,
            output_path,
        ]
        if os.path.exists(output_path):
            try:
                os.remove(output_path)
            except OSError:
                pass

        def _progress_callback(_out_time_seconds, _phase_percent):
            if progress_tick:
                progress_tick()

        returncode, stdout_text, stderr_text = self._run_ffmpeg_with_progress(
            command,
            duration_seconds=None,
            progress_callback=_progress_callback if progress_tick else None,
        )
        if returncode != 0:
            message = (stderr_text or stdout_text or "").strip()
            if message:
                message = message[-600:]
            return False, message or f"ffmpeg exited with code {returncode}"

        if not os.path.exists(output_path):
            return False, "ffmpeg completed without creating an output file"

        output_size = os.path.getsize(output_path)
        if output_size <= 0:
            try:
                os.remove(output_path)
            except OSError:
                pass
            return False, "ffmpeg produced an empty output file"

        # Very short/silent MP3 outputs can be smaller than 1KB while still valid.
        # Validate via ffprobe duration/stream metadata instead of a hard size floor.
        summary = self._ffprobe_audio_summary(output_path)
        if not summary.get("ok"):
            try:
                os.remove(output_path)
            except OSError:
                pass
            return False, f"ffmpeg produced an unreadable file ({output_size} bytes): {summary.get('error')}"

        try:
            duration_seconds = float(summary.get("duration") or 0.0)
        except (TypeError, ValueError):
            duration_seconds = 0.0
        if duration_seconds <= 0.0 or not summary.get("codec"):
            try:
                os.remove(output_path)
            except OSError:
                pass
            return (
                False,
                f"ffmpeg produced an invalid file ({output_size} bytes): "
                f"codec={summary.get('codec')} duration={summary.get('duration')}",
            )

        return True, output_path

    @staticmethod
    def _parse_concat_entries(concat_path):
        entries = []
        try:
            with open(concat_path, "r", encoding="utf-8") as handle:
                for line in handle:
                    stripped = line.strip()
                    if not stripped.startswith("file '") or not stripped.endswith("'"):
                        continue
                    raw = stripped[6:-1]
                    path = raw.replace(r"'\''", "'").replace("\\\\", "\\")
                    entries.append(path)
        except OSError:
            return []
        return entries

    @staticmethod
    def _ffprobe_audio_summary(path):
        cmd = [
            get_ffprobe_exe(),
            "-v",
            "error",
            "-show_entries",
            "stream=codec_name,sample_rate,channels,sample_fmt,bits_per_sample:format=format_name,duration,bit_rate,size",
            "-of",
            "json",
            path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
        if result.returncode != 0:
            err = (result.stderr or result.stdout or "").strip()
            try:
                segment = AudioSegment.from_file(path)
                ext = os.path.splitext(path)[1].lstrip(".").lower()
                return {
                    "ok": True,
                    "codec": ext or "audio",
                    "sample_rate": str(segment.frame_rate),
                    "channels": segment.channels,
                    "sample_fmt": None,
                    "bits_per_sample": segment.sample_width * 8,
                    "format_name": ext or None,
                    "duration": str(segment.duration_seconds),
                    "bit_rate": None,
                    "size": str(os.path.getsize(path)) if os.path.exists(path) else None,
                }
            except Exception:
                return {"ok": False, "error": err[-500:] if err else f"ffprobe exit {result.returncode}"}
        try:
            payload = json.loads(result.stdout or "{}")
        except json.JSONDecodeError:
            return {"ok": False, "error": "ffprobe returned invalid JSON"}

        stream = (payload.get("streams") or [{}])[0]
        fmt = payload.get("format") or {}
        return {
            "ok": True,
            "codec": stream.get("codec_name"),
            "sample_rate": stream.get("sample_rate"),
            "channels": stream.get("channels"),
            "sample_fmt": stream.get("sample_fmt"),
            "bits_per_sample": stream.get("bits_per_sample"),
            "format_name": fmt.get("format_name"),
            "duration": fmt.get("duration"),
            "bit_rate": fmt.get("bit_rate"),
            "size": fmt.get("size"),
        }

    def _log_concat_failure_diagnostics(self, concat_path, output_path, failure_message, log_callback=None):
        if not log_callback:
            return
        log_callback(f"[diag] MP3 concat export failed for {output_path}")
        log_callback(f"[diag] ffmpeg error: {failure_message}")
        entries = self._parse_concat_entries(concat_path)
        log_callback(f"[diag] concat file entries: {len(entries)}")
        if not entries:
            return

        sample_indices = sorted(set([0, 1, 2, max(len(entries) - 2, 0), len(entries) - 1]))
        probed = []
        for idx in sample_indices:
            if not (0 <= idx < len(entries)):
                continue
            src = entries[idx]
            summary = self._ffprobe_audio_summary(src)
            probed.append((idx, src, summary))
            basename = os.path.basename(src)
            if summary.get("ok"):
                log_callback(
                    f"[diag] input[{idx}] {basename}: "
                    f"codec={summary.get('codec')}, sr={summary.get('sample_rate')}, "
                    f"ch={summary.get('channels')}, fmt={summary.get('sample_fmt')}, "
                    f"bps={summary.get('bits_per_sample')}, dur={summary.get('duration')}, "
                    f"br={summary.get('bit_rate')}, size={summary.get('size')}"
                )
            else:
                log_callback(f"[diag] input[{idx}] {basename}: ffprobe error: {summary.get('error')}")

        layouts = set()
        codecs = set()
        formats = set()
        for _idx, _src, summary in probed:
            if not summary.get("ok"):
                continue
            layouts.add((summary.get("sample_rate"), summary.get("channels"), summary.get("sample_fmt"), summary.get("bits_per_sample")))
            codecs.add(summary.get("codec"))
            formats.add(summary.get("format_name"))
        log_callback(f"[diag] sampled input codecs={sorted(c for c in codecs if c)} formats={sorted(f for f in formats if f)}")
        log_callback(f"[diag] sampled input layouts={sorted(layouts)}")

    def _export_concat_mp3(self, concat_path, output_path, progress_tick=None, log_callback=None):
        mp3_success, mp3_result = self._run_ffmpeg_concat(
            concat_path,
            output_path,
            ["-c:a", "libmp3lame", "-q:a", "2"],
            progress_tick=progress_tick,
        )
        if mp3_success:
            return True, mp3_result
        self._log_concat_failure_diagnostics(concat_path, output_path, mp3_result, log_callback=log_callback)
        return False, f"MP3 concat export failed: {mp3_result}"

    def _optimized_export_part_basename(self, part_index):
        title = self._current_script_title().strip() or "Project"
        safe_title = re.sub(r"[^A-Za-z0-9]+", "-", title).strip("-").lower() or "project"
        return f"{safe_title}-{part_index:02d}.mp3"

    @staticmethod
    def _is_normalization_enabled(export_config):
        if export_config is None:
            return True
        return bool(getattr(export_config, "normalize_enabled", True))

    @staticmethod
    def _extract_json_object(text):
        if not text:
            return None
        start = text.find("{")
        while start != -1:
            depth = 0
            for idx in range(start, len(text)):
                char = text[idx]
                if char == "{":
                    depth += 1
                elif char == "}":
                    depth -= 1
                    if depth == 0:
                        candidate = text[start:idx + 1]
                        try:
                            return json.loads(candidate)
                        except json.JSONDecodeError:
                            break
            start = text.find("{", start + 1)
        return None

    def _detect_channel_count(self, input_path):
        try:
            return int(AudioSegment.from_file(input_path).channels or 1)
        except Exception:
            return 1

    @staticmethod
    def _loudnorm_codec_args_for_path(path, *, sample_rate=None, channels=None):
        ext = os.path.splitext(path)[1].lower()
        args = []
        if ext == ".wav":
            args.extend(["-c:a", "pcm_s16le"])
        else:
            args.extend(["-c:a", "libmp3lame", "-q:a", "2"])
        if sample_rate:
            args.extend(["-ar", str(int(sample_rate))])
        if channels:
            args.extend(["-ac", str(int(channels))])
        return args

    @staticmethod
    def _run_ffmpeg_with_progress(command, duration_seconds=None, progress_callback=None):
        """Run ffmpeg and emit estimated progress derived from out_time_ms."""
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        stderr_lines = []
        last_percent = -1.0
        safe_duration = max(float(duration_seconds or 0.0), 0.001)

        while True:
            line = process.stderr.readline()
            if not line:
                if process.poll() is not None:
                    break
                continue

            stderr_lines.append(line)
            stripped = line.strip()
            if not stripped:
                continue

            if progress_callback and stripped.startswith("out_time_ms="):
                try:
                    out_time_seconds = max(0.0, int(stripped.split("=", 1)[1]) / 1_000_000.0)
                except (TypeError, ValueError):
                    continue
                phase_percent = min(100.0, (out_time_seconds / safe_duration) * 100.0)
                if phase_percent >= last_percent + 1.0 or phase_percent >= 99.9:
                    last_percent = phase_percent
                    progress_callback(out_time_seconds, phase_percent)

        stdout_text = ""
        if process.stdout:
            try:
                stdout_text = process.stdout.read() or ""
            except Exception:
                stdout_text = ""

        stderr_text = "".join(stderr_lines)
        if progress_callback:
            progress_callback(float(duration_seconds or 0.0), 100.0)
        return process.returncode, stdout_text, stderr_text

    def _resolve_export_normalization_config(self, export_config=None):
        if export_config is not None:
            return export_config
        app_config = self._load_app_config()
        export_raw = app_config.get("export", {}) if isinstance(app_config, dict) else {}
        if not isinstance(export_raw, dict):
            export_raw = {}
        return SimpleNamespace(**export_raw)

    def _normalize_audio_file(
        self,
        input_path,
        export_config=None,
        allow_short_single_pass=False,
        short_seconds_threshold=12.0,
        progress_callback=None,
        log_callback=None,
    ):
        """Apply two-pass EBU R128 loudness normalization in-place."""
        export_config = self._resolve_export_normalization_config(export_config)
        if not self._is_normalization_enabled(export_config):
            return True, input_path

        if not os.path.exists(input_path):
            return False, f"Normalization input does not exist: {input_path}"

        duration_seconds = 0.0
        input_sample_rate = 48000
        input_channels = 1
        try:
            input_segment = AudioSegment.from_file(input_path)
            duration_seconds = float(input_segment.duration_seconds or 0.0)
            input_sample_rate = int(input_segment.frame_rate or 48000)
            input_channels = max(1, int(input_segment.channels or 1))
        except Exception:
            duration_seconds = 0.0
        is_short_clip = duration_seconds > 0 and duration_seconds <= float(short_seconds_threshold)

        channels = input_channels
        target_lufs = float(
            getattr(export_config, "normalize_target_lufs_stereo", -16.0)
            if channels > 1 else getattr(export_config, "normalize_target_lufs_mono", -18.0)
        )
        target_tp = float(getattr(export_config, "normalize_true_peak_dbtp", -1.0))
        target_lra = float(getattr(export_config, "normalize_lra", 11.0))

        base_filter = (
            f"loudnorm=I={target_lufs}:TP={target_tp}:LRA={target_lra}:"
            "print_format=json"
        )
        ext = os.path.splitext(input_path)[1]

        progress_log_state = {}
        progress_emit_state = {"overall_percent": -1.0}

        def _emit_progress(phase, out_time_seconds=0.0, phase_percent=0.0):
            if progress_callback:
                phase_percent = max(0.0, min(100.0, float(phase_percent or 0.0)))
                if phase == "pass1-measure":
                    overall_percent = phase_percent * 0.5
                elif phase == "pass2-apply":
                    overall_percent = 50.0 + (phase_percent * 0.5)
                else:
                    overall_percent = phase_percent
                should_emit = (
                    overall_percent >= 99.9
                    or (overall_percent - progress_emit_state["overall_percent"]) >= 1.0
                )
                if should_emit:
                    progress_emit_state["overall_percent"] = overall_percent
                    progress_callback({
                        "phase": phase,
                        "processed_seconds": float(out_time_seconds or 0.0),
                        "phase_percent": phase_percent,
                        "overall_percent": overall_percent,
                    })
            if log_callback:
                bucket = int(float(phase_percent or 0.0) // 20) * 20
                previous_bucket = progress_log_state.get(phase, -10)
                if bucket > previous_bucket and bucket <= 100:
                    progress_log_state[phase] = bucket
                    readable_phase = {
                        "pass1-measure": "Normalization pass 1/2",
                        "pass2-apply": "Normalization pass 2/2",
                        "single-pass": "Normalization single-pass fallback",
                    }.get(phase, "Normalization")
                    log_callback(f"{readable_phase}: {bucket}%")

        def _single_pass_normalize(reason):
            pass1_filter = (
                f"loudnorm=I={target_lufs}:TP={target_tp}:LRA={target_lra}:"
                "linear=false:print_format=summary"
            )
            temp_output = f"{input_path}.normalized{ext}"
            cmd = [
                get_ffmpeg_exe(),
                "-y",
                "-hide_banner",
                "-progress",
                "pipe:2",
                "-nostats",
                "-loglevel",
                "error",
                "-i",
                input_path,
                "-vn",
                "-af",
                pass1_filter,
                *self._loudnorm_codec_args_for_path(
                    input_path,
                    sample_rate=input_sample_rate,
                    channels=input_channels,
                ),
                temp_output,
            ]
            if log_callback:
                log_callback("Normalization fallback: running single-pass loudnorm for short clip...")
            returncode, stdout_text, stderr_text = self._run_ffmpeg_with_progress(
                cmd,
                duration_seconds=duration_seconds,
                progress_callback=lambda out_time, pct: _emit_progress("single-pass", out_time, pct),
            )
            if returncode != 0:
                try:
                    if os.path.exists(temp_output):
                        os.remove(temp_output)
                except OSError:
                    pass
                stderr_tail = (stderr_text or stdout_text or "").strip()[-600:]
                return False, f"{reason}; short-clip loudnorm fallback failed: {stderr_tail or f'exit {returncode}'}"
            if not os.path.exists(temp_output) or os.path.getsize(temp_output) < 1024:
                try:
                    if os.path.exists(temp_output):
                        os.remove(temp_output)
                except OSError:
                    pass
                return False, f"{reason}; short-clip loudnorm fallback produced invalid output"
            os.replace(temp_output, input_path)
            return True, input_path

        measure_cmd = [
            get_ffmpeg_exe(),
            "-y",
            "-hide_banner",
            "-progress",
            "pipe:2",
            "-nostats",
            "-loglevel",
            "info",
            "-i",
            input_path,
            "-vn",
            "-af",
            base_filter,
            "-f",
            "null",
            "-",
        ]
        if log_callback:
            log_callback("Normalization pass 1/2: measuring integrated loudness...")
        returncode, stdout_text, stderr_text = self._run_ffmpeg_with_progress(
            measure_cmd,
            duration_seconds=duration_seconds,
            progress_callback=lambda out_time, pct: _emit_progress("pass1-measure", out_time, pct),
        )
        if returncode != 0:
            if allow_short_single_pass and is_short_clip:
                return _single_pass_normalize("loudnorm pass 1 failed")
            stderr_tail = (stderr_text or stdout_text or "").strip()[-600:]
            return False, f"loudnorm pass 1 failed: {stderr_tail or f'exit {returncode}'}"

        measure_text = f"{stderr_text}\n{stdout_text}"
        measured = self._extract_json_object(measure_text)
        required_keys = (
            "input_i",
            "input_tp",
            "input_lra",
            "input_thresh",
            "target_offset",
        )
        if not measured or any(key not in measured for key in required_keys):
            if allow_short_single_pass and is_short_clip:
                return _single_pass_normalize("loudnorm pass 1 did not produce valid measurement JSON")
            return False, "loudnorm pass 1 did not produce valid measurement JSON"

        try:
            measured_i = float(measured["input_i"])
            measured_tp = float(measured["input_tp"])
            measured_lra = float(measured["input_lra"])
            measured_thresh = float(measured["input_thresh"])
            target_offset = float(measured["target_offset"])
        except (TypeError, ValueError):
            if allow_short_single_pass and is_short_clip:
                return _single_pass_normalize("loudnorm pass 1 returned non-numeric values")
            return False, "loudnorm pass 1 returned invalid measurement values"

        pass2_filter = (
            f"loudnorm=I={target_lufs}:TP={target_tp}:LRA={target_lra}:"
            f"measured_I={measured_i}:measured_TP={measured_tp}:"
            f"measured_LRA={measured_lra}:measured_thresh={measured_thresh}:"
            f"offset={target_offset}:linear=true:print_format=summary"
        )
        temp_output = f"{input_path}.normalized{ext}"
        normalize_cmd = [
            get_ffmpeg_exe(),
            "-y",
            "-hide_banner",
            "-progress",
            "pipe:2",
            "-nostats",
            "-loglevel",
            "error",
            "-i",
            input_path,
            "-vn",
            "-af",
            pass2_filter,
            *self._loudnorm_codec_args_for_path(
                input_path,
                sample_rate=input_sample_rate,
                channels=input_channels,
            ),
            temp_output,
        ]
        if log_callback:
            log_callback("Normalization pass 2/2: applying measured loudness correction...")
        returncode, stdout_text, stderr_text = self._run_ffmpeg_with_progress(
            normalize_cmd,
            duration_seconds=duration_seconds,
            progress_callback=lambda out_time, pct: _emit_progress("pass2-apply", out_time, pct),
        )
        if returncode != 0:
            if allow_short_single_pass and is_short_clip:
                return _single_pass_normalize("loudnorm pass 2 failed")
            try:
                if os.path.exists(temp_output):
                    os.remove(temp_output)
            except OSError:
                pass
            stderr_tail = (stderr_text or stdout_text or "").strip()[-600:]
            return False, f"loudnorm pass 2 failed: {stderr_tail or f'exit {returncode}'}"

        if not os.path.exists(temp_output) or os.path.getsize(temp_output) < 1024:
            try:
                if os.path.exists(temp_output):
                    os.remove(temp_output)
            except OSError:
                pass
            return False, "loudnorm pass 2 produced an invalid output file"

        os.replace(temp_output, input_path)
        return True, input_path

    def _call_normalize_audio_file(self, input_path, export_config=None, progress_callback=None, log_callback=None, **kwargs):
        """
        Backward-compatible wrapper around `_normalize_audio_file`.
        Some tests monkeypatch `_normalize_audio_file` with a narrower signature.
        """
        normalize_fn = self._normalize_audio_file
        call_kwargs = {}
        try:
            accepted = set(inspect.signature(normalize_fn).parameters.keys())
        except (TypeError, ValueError):
            accepted = set()

        if "export_config" in accepted:
            call_kwargs["export_config"] = export_config
        if "progress_callback" in accepted and progress_callback is not None:
            call_kwargs["progress_callback"] = progress_callback
        if "log_callback" in accepted and log_callback is not None:
            call_kwargs["log_callback"] = log_callback
        for key, value in kwargs.items():
            if key in accepted:
                call_kwargs[key] = value

        return normalize_fn(input_path, **call_kwargs)

    def _load_generation_settings(self):
        return self._load_app_config().get("generation", {})

    def load_source_document(self):
        input_path = (self._load_state().get("input_file_path") or "").strip()
        if not input_path or not os.path.exists(input_path):
            raise ValueError("No uploaded source document found")
        return load_source_document(input_path)

    @staticmethod
    def _split_text_sentences(text):
        parts = re.split(r'(?<=[.!?])\s+', (text or "").strip())
        return [part.strip() for part in parts if part.strip()]

    @staticmethod
    def _count_words(text):
        return len(re.findall(r"\b\w+\b", text or ""))

    @classmethod
    def _split_long_chunk_text(cls, text):
        sentences = cls._split_text_sentences(text)
        if len(sentences) < 2:
            return None

        sentence_word_counts = [cls._count_words(sentence) for sentence in sentences]
        total_words = sum(sentence_word_counts)
        if total_words <= 1:
            return None

        best_split_index = None
        best_diff = None
        left_words = 0

        for index in range(1, len(sentences)):
            left_words += sentence_word_counts[index - 1]
            right_words = total_words - left_words
            if left_words <= 0 or right_words <= 0:
                continue
            diff = abs(left_words - right_words)
            if best_diff is None or diff < best_diff:
                best_diff = diff
                best_split_index = index

        if best_split_index is None:
            return None

        left_text = " ".join(sentences[:best_split_index]).strip()
        right_text = " ".join(sentences[best_split_index:]).strip()
        if not left_text or not right_text:
            return None
        return left_text, right_text

    @classmethod
    def _speakers_match(cls, left, right):
        left_name = cls._normalize_speaker_name(left)
        right_name = cls._normalize_speaker_name(right)
        return bool(left_name) and left_name == right_name

    @staticmethod
    def _chunk_in_scope(chunk, chapter=None):
        if not chapter:
            return True
        return (chunk.get("chapter") or "").strip() == chapter

    @staticmethod
    def _chapter_key(chunk):
        return (chunk.get("chapter") or "").strip()

    @classmethod
    def _chunks_share_chapter(cls, first_chunk, second_chunk):
        return cls._chapter_key(first_chunk) == cls._chapter_key(second_chunk)

    @staticmethod
    def _join_chunk_text(*parts):
        return " ".join(part.strip() for part in parts if (part or "").strip()).strip()

    def _merge_adjacent_chunks(self, first_chunk, second_chunk):
        merged = copy.deepcopy(first_chunk)
        merged["text"] = self._join_chunk_text(first_chunk.get("text", ""), second_chunk.get("text", ""))
        merged["instruct"] = (first_chunk.get("instruct") or "").strip() or (second_chunk.get("instruct") or "").strip()
        merged["status"] = "pending"
        merged["audio_path"] = None
        merged["audio_validation"] = None
        merged["auto_regen_count"] = 0
        merged.pop("generation_token", None)
        self._clear_proofread_state(merged)
        return merged

    @staticmethod
    def _paragraph_mentions_speaker(paragraph_text, speaker):
        normalized_paragraph = re.sub(r"\s+", " ", paragraph_text or "").casefold()
        normalized_speaker = re.sub(r"\s+", " ", speaker or "").strip().casefold()
        if not normalized_paragraph or not normalized_speaker:
            return False
        return normalized_speaker in normalized_paragraph

    def suggest_design_sample_text(self, speaker, chunks=None):
        chunks = chunks if chunks is not None else self.load_chunks()
        speaker_key = self._normalize_speaker_name(speaker)
        matching_texts = [
            (chunk.get("text") or "").strip()
            for chunk in chunks
            if self._normalize_speaker_name(chunk.get("speaker")) == speaker_key and (chunk.get("text") or "").strip()
        ]
        if not matching_texts:
            return ""

        sorted_texts = sorted(matching_texts, key=lambda text: len(re.findall(r"\b\w+\b", text)), reverse=True)
        base_sentences = self._split_text_sentences(sorted_texts[0]) or [sorted_texts[0]]
        selected = list(base_sentences)

        while len(re.findall(r"\b\w+\b", " ".join(selected))) > 50 and len(selected) > 1:
            candidate_without_last = selected[:-1]
            candidate_without_first = selected[1:]
            first_words = len(re.findall(r"\b\w+\b", " ".join(candidate_without_first)))
            last_words = len(re.findall(r"\b\w+\b", " ".join(candidate_without_last)))
            selected = candidate_without_first if first_words >= last_words else candidate_without_last

        used = {sentence.strip() for sentence in selected if sentence.strip()}
        total_words = len(re.findall(r"\b\w+\b", " ".join(selected)))
        if total_words >= 30:
            return " ".join(selected).strip()

        extra_sentences = []
        for text in sorted_texts:
            for sentence in self._split_text_sentences(text) or [text]:
                sentence = sentence.strip()
                if sentence and sentence not in used:
                    extra_sentences.append(sentence)
                    used.add(sentence)

        for sentence in extra_sentences:
            selected.append(sentence)
            total_words = len(re.findall(r"\b\w+\b", " ".join(selected)))
            if total_words >= 30:
                break

        return " ".join(selected).strip()

    def collect_voice_suggestion_context(self, speaker, target_chars=None):
        generation_settings = self._load_generation_settings()
        chunk_size = int(generation_settings.get("chunk_size") or 3000)
        target_chars = max(int(target_chars or (chunk_size * 2)), 1)

        source_error = None
        selected = []
        total_chars = 0

        try:
            source_document = self.load_source_document()
            for paragraph in iter_document_paragraphs(source_document):
                if not self._paragraph_mentions_speaker(paragraph["text"], speaker):
                    continue
                selected.append(paragraph)
                total_chars += len(paragraph["text"])
                if total_chars >= target_chars:
                    break
            context_source = "source_document"
        except Exception as e:
            source_error = str(e)
            context_source = "chunks_fallback"
            chunks = self.load_chunks()
            speaker_key = self._normalize_speaker_name(speaker)
            fallback_items = []
            for chunk in chunks:
                text = (chunk.get("text") or "").strip()
                if not text:
                    continue
                chunk_speaker = self._normalize_speaker_name(chunk.get("speaker"))
                matches = chunk_speaker == speaker_key or self._paragraph_mentions_speaker(text, speaker)
                if not matches:
                    continue
                fallback_items.append({
                    "text": text,
                    "chapter": (chunk.get("chapter") or "").strip(),
                })
                total_chars += len(text)
                if total_chars >= target_chars:
                    break
            selected = fallback_items
            if source_error:
                print(
                    f"Voice suggestion context fallback for '{speaker}': source unavailable ({source_error}); using generated chunks."
                )

        return {
            "speaker": speaker,
            "target_chars": target_chars,
            "context_chars": total_chars,
            "paragraphs": selected,
            "context_source": context_source,
            "source_error": source_error,
            "warning": (
                f"Source document unavailable: {source_error}. Using generated chunks as fallback context."
                if source_error
                else None
            ),
        }

    def build_voice_suggestion_prompt(self, speaker, prompt_template):
        context = self.collect_voice_suggestion_context(speaker)
        paragraphs = context["paragraphs"]
        source_error = context.get("source_error")
        context_source = context.get("context_source")
        warning = context.get("warning")

        if paragraphs:
            context_blocks = []
            for item in paragraphs:
                chapter = (item.get("chapter") or "").strip()
                text = item["text"]
                context_blocks.append(f"[{chapter}] {text}" if chapter else text)
            if context_source == "chunks_fallback":
                context_prefix = (
                    f'Source document unavailable ({source_error}). '
                    f'Fallback context from generated chunks mentioning "{speaker}":\n\n'
                    + "\n\n".join(context_blocks)
                )
            else:
                context_prefix = (
                    f'Source paragraphs mentioning "{speaker}":\n\n' +
                    "\n\n".join(context_blocks)
                )
        else:
            if context_source == "chunks_fallback":
                context_prefix = (
                    f'Source document unavailable ({source_error}). '
                    f'No generated chunks mentioning "{speaker}" were found for fallback context.'
                )
            else:
                context_prefix = f'No source paragraphs mentioning "{speaker}" were found in the uploaded story.'

        rendered_prompt = (prompt_template or "").replace("{character_name}", speaker)
        return {
            **context,
            "warning": warning,
            "prompt": f"{context_prefix}\n\n{rendered_prompt}".strip(),
        }

    def _upsert_clone_manifest_entry(self, entry):
        manifest_path = os.path.join(self.root_dir, "clone_voices", "manifest.json")
        manifest = []
        if os.path.exists(manifest_path):
            try:
                with open(manifest_path, "r", encoding="utf-8") as f:
                    manifest = json.load(f)
            except (json.JSONDecodeError, ValueError):
                manifest = []

        replaced = False
        for index, existing in enumerate(manifest):
            if existing.get("id") == entry["id"]:
                manifest[index] = entry
                replaced = True
                break
        if not replaced:
            manifest.append(entry)

        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)

    def materialize_design_voice(self, speaker, description=None, sample_text=None, force=False, voice_config=None, export_config=None):
        voice_config = copy.deepcopy(voice_config) if voice_config is not None else self._load_voice_config()
        voice_data = voice_config.setdefault(speaker, {})
        voice_data["type"] = "design"

        description = (description if description is not None else voice_data.get("description") or "").strip()
        if not description:
            raise ValueError(f"Voice design for '{speaker}' is missing a base description")

        chunks = self.load_chunks()
        sample_text = (sample_text if sample_text is not None else voice_data.get("ref_text") or "").strip()
        if not sample_text:
            sample_text = self.suggest_design_sample_text(speaker, chunks)
        if not sample_text:
            raise ValueError(f"No sample text available for '{speaker}'")
        generated_ref_text, _ = apply_dictionary_to_text(sample_text, self.load_dictionary_entries())

        script_title = self._current_script_title()
        display_name = f"{script_title}.{speaker}"
        safe_name = sanitize_filename(f"{script_title}.{speaker}")
        voice_id = safe_name or sanitize_filename(speaker) or "designed_clone"
        filename = f"{voice_id}.wav"
        rel_audio_path = f"clone_voices/{filename}"
        abs_audio_path = os.path.join(self.root_dir, rel_audio_path)
        os.makedirs(os.path.dirname(abs_audio_path), exist_ok=True)

        audio_exists = os.path.exists(abs_audio_path)
        if force or not audio_exists:
            engine = self.get_engine()
            if not engine:
                raise RuntimeError("TTS engine not initialized")
            wav_path, _ = engine.generate_voice_design(description=description, sample_text=generated_ref_text)
            shutil.copy2(wav_path, abs_audio_path)
            normalized, normalize_result = self._call_normalize_audio_file(
                abs_audio_path,
                export_config=export_config,
                allow_short_single_pass=True,
            )
            if not normalized:
                raise RuntimeError(f"Failed to normalize generated design voice for '{speaker}': {normalize_result}")
            if hasattr(engine, "clear_clone_prompt_cache"):
                engine.clear_clone_prompt_cache(speaker)

        voice_data["description"] = description
        voice_data["ref_text"] = sample_text
        voice_data["generated_ref_text"] = generated_ref_text
        voice_data["ref_audio"] = rel_audio_path
        self._save_voice_config(voice_config)

        self._upsert_clone_manifest_entry({
            "id": voice_id,
            "name": display_name,
            "filename": filename,
            "generated": True,
            "speaker": speaker,
            "script_title": script_title,
            "sample_text": sample_text,
            "description": description,
        })

        runtime_data = copy.deepcopy(voice_data)
        runtime_data["type"] = "clone"
        return {
            "voice_config": voice_config,
            "runtime_voice_data": runtime_data,
            "voice_id": voice_id,
            "display_name": display_name,
            "filename": filename,
            "ref_audio": rel_audio_path,
            "ref_text": sample_text,
            "generated_ref_text": generated_ref_text,
        }

    def prepare_runtime_voice_config(self, voice_config, speakers, force_design_refresh=False):
        runtime_config = copy.deepcopy(voice_config)
        normalized_lookup = {}
        for name in runtime_config.keys():
            normalized = self._normalize_speaker_name(name)
            if normalized and normalized not in normalized_lookup:
                normalized_lookup[normalized] = name

        def ensure_runtime_voice(speaker, visiting=None):
            visiting = visiting or set()
            normalized = self._normalize_speaker_name(speaker)
            actual_speaker = normalized_lookup.get(normalized, speaker)
            if actual_speaker in visiting:
                raise ValueError(f"Voice fallback loop detected for '{actual_speaker}'")
            visiting = set(visiting)
            visiting.add(actual_speaker)

            voice_data = runtime_config.get(actual_speaker, {})
            if voice_data.get("type") != "design":
                return actual_speaker

            try:
                materialized = self.materialize_design_voice(
                    actual_speaker,
                    description=voice_data.get("description"),
                    sample_text=voice_data.get("ref_text"),
                    force=force_design_refresh or not voice_data.get("ref_audio"),
                    voice_config=runtime_config,
                )
                runtime_config.update(materialized["voice_config"])
                runtime_config[actual_speaker] = materialized["runtime_voice_data"]
                return actual_speaker
            except Exception:
                if normalized == self._normalize_speaker_name("NARRATOR"):
                    raise

                narrator_name = normalized_lookup.get(self._normalize_speaker_name("NARRATOR"))
                if not narrator_name:
                    raise ValueError(f"Voice design for '{actual_speaker}' is unavailable and narrator has no voice configured")

                ensure_runtime_voice(narrator_name, visiting)
                narrator_runtime = runtime_config.get(narrator_name, {})
                if narrator_runtime.get("type") == "design":
                    raise ValueError("Narrator voice is unavailable; cannot fall back from design voice")
                runtime_config[actual_speaker] = copy.deepcopy(narrator_runtime)
                return actual_speaker

        for speaker in speakers:
            ensure_runtime_voice(speaker)

        return runtime_config

    @staticmethod
    def _normalize_speaker_name(name):
        return re.sub(r"\s+", " ", (name or "").strip()).casefold()

    def get_narrator_threshold(self):
        state = self._load_state()
        raw = state.get("narrator_threshold", self.DEFAULT_NARRATOR_THRESHOLD)
        try:
            value = int(raw)
        except (TypeError, ValueError):
            value = self.DEFAULT_NARRATOR_THRESHOLD
        return max(0, value)

    def set_narrator_threshold(self, value):
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            parsed = self.DEFAULT_NARRATOR_THRESHOLD
        parsed = max(0, parsed)
        state = self._load_state()
        state["narrator_threshold"] = parsed
        self._save_state(state)
        return parsed

    def get_narrator_overrides(self):
        """Return dict of chapter_name -> voice_name for narrator substitution."""
        return self._load_state().get("narrator_overrides", {})

    def _apply_narrator_override(self, speaker, chapter, narrator_overrides):
        """Return the effective speaker, substituting the chapter narrator override if applicable."""
        if self._normalize_speaker_name(speaker) == self._normalize_speaker_name("NARRATOR"):
            override = narrator_overrides.get(chapter)
            if override and self._normalize_speaker_name(override) != self._normalize_speaker_name("NARRATOR"):
                return override
        return speaker

    def set_narrator_override(self, chapter: str, voice: str):
        """Set or clear the narrator voice for a specific chapter."""
        state = self._load_state()
        overrides = state.get("narrator_overrides", {})
        if voice and self._normalize_speaker_name(voice) != self._normalize_speaker_name("NARRATOR"):
            overrides[chapter] = voice
        else:
            overrides.pop(chapter, None)
        state["narrator_overrides"] = overrides
        self._save_state(state)

    @staticmethod
    def _count_speaker_lines(chunks, speaker):
        speaker_key = ProjectManager._normalize_speaker_name(speaker)
        if not speaker_key:
            return 0
        count = 0
        for chunk in chunks or []:
            text = (chunk.get("text") or "").strip()
            if not text:
                continue
            if ProjectManager._normalize_speaker_name(chunk.get("speaker")) == speaker_key:
                count += 1
        return count

    def _resolve_narrator_name(self, voice_config, chunks=None):
        narrator_key = self._normalize_speaker_name("NARRATOR")
        for name in voice_config.keys():
            if self._normalize_speaker_name(name) == narrator_key:
                return name
        for chunk in chunks or []:
            speaker = (chunk.get("speaker") or "").strip()
            if self._normalize_speaker_name(speaker) == narrator_key:
                return speaker
        return None

    def resolve_voice_speaker(self, speaker, voice_config, chunks=None):
        """Resolve a speaker alias to a configured target speaker.

        Aliases are case-insensitive and can chain, but any invalid target,
        self-alias, or loop falls back to the original speaker.
        """
        original = speaker or ""
        if not original:
            return original

        lookup = {}
        for name in voice_config.keys():
            normalized = self._normalize_speaker_name(name)
            if normalized and normalized not in lookup:
                lookup[normalized] = name

        # Canonicalize the starting speaker to the configured key when a
        # case-variant exists (e.g. "Narrator" -> "NARRATOR").
        current = lookup.get(self._normalize_speaker_name(original), original)
        manual_alias_used = False
        seen = {self._normalize_speaker_name(current)}
        while current:
            voice_data = voice_config.get(current, {})
            alias = (voice_data.get("alias") or "").strip()
            if not alias:
                break

            target = lookup.get(self._normalize_speaker_name(alias))
            if not target or self._normalize_speaker_name(target) == self._normalize_speaker_name(current):
                break
            manual_alias_used = True

            target_key = self._normalize_speaker_name(target)
            if target_key in seen:
                print(f"Alias loop detected for speaker '{original}'. Falling back to original speaker.")
                return original

            seen.add(target_key)
            current = target

        resolved = current or original
        if manual_alias_used:
            return resolved

        threshold = self.get_narrator_threshold()
        if threshold <= 0:
            return resolved

        resolved_key = self._normalize_speaker_name(resolved)
        narrator_key = self._normalize_speaker_name("NARRATOR")
        if resolved_key == narrator_key:
            return resolved

        try:
            chunk_list = chunks if chunks is not None else self.load_chunks()
        except Exception:
            chunk_list = []

        narrator_name = self._resolve_narrator_name(voice_config, chunk_list)
        if not narrator_name:
            return resolved
        if self._normalize_speaker_name(narrator_name) == resolved_key:
            return resolved

        line_count = self._count_speaker_lines(chunk_list, resolved)
        if line_count < threshold:
            return narrator_name

        return resolved

    @staticmethod
    def _voice_entry_from_config(voice_config, speaker):
        if not isinstance(voice_config, dict):
            return {}
        entry = voice_config.get(speaker, {})
        return entry if isinstance(entry, dict) else {}

    @staticmethod
    def _voice_ref_audio_from_config(voice_config, speaker):
        entry = ProjectManager._voice_entry_from_config(voice_config, speaker)
        return str(entry.get("ref_audio") or "").strip()

    def preview_voice_config_invalidation(self, old_config, new_config):
        """Preview how many generated clips must be invalidated for a voice-config change.

        A clip is considered affected when:
        1) its resolved alias target changes, or
        2) the resolved target keeps the same name but its ref_audio changes.
        """
        old_config = old_config if isinstance(old_config, dict) else {}
        new_config = new_config if isinstance(new_config, dict) else {}
        chunks = self.load_chunks()

        affected_indices = []
        affected_speakers = set()
        changed_reasons = Counter()

        for index, chunk in enumerate(chunks):
            audio_path = str((chunk or {}).get("audio_path") or "").strip()
            if not audio_path:
                continue

            speaker = str((chunk or {}).get("speaker") or "").strip()
            if not speaker:
                continue

            old_resolved = self.resolve_voice_speaker(speaker, old_config, chunks=chunks)
            new_resolved = self.resolve_voice_speaker(speaker, new_config, chunks=chunks)

            alias_changed = self._normalize_speaker_name(old_resolved) != self._normalize_speaker_name(new_resolved)
            old_ref_audio = self._voice_ref_audio_from_config(old_config, old_resolved)
            new_ref_audio = self._voice_ref_audio_from_config(new_config, new_resolved)
            ref_audio_changed = old_ref_audio != new_ref_audio and (old_ref_audio or new_ref_audio)

            if not (alias_changed or ref_audio_changed):
                continue

            affected_indices.append(index)
            affected_speakers.add(speaker)
            if alias_changed:
                changed_reasons["alias"] += 1
            if ref_audio_changed:
                changed_reasons["ref_audio"] += 1

        return {
            "invalidated_clips": len(affected_indices),
            "affected_indices": affected_indices,
            "affected_speakers": sorted(affected_speakers),
            "reason_counts": dict(changed_reasons),
        }

    def invalidate_chunk_audio_indices(self, indices):
        """Delete audio files and clear chunk audio refs for selected chunk indices."""
        target_indices = {int(i) for i in (indices or []) if isinstance(i, int) or str(i).isdigit()}
        if not target_indices:
            return {"invalidated_clips": 0, "deleted_files": 0}

        deleted_files = 0
        cleared = 0
        files_to_delete = set()

        with self._chunks_lock:
            if not os.path.exists(self.chunks_path):
                return {"invalidated_clips": 0, "deleted_files": 0}

            with open(self.chunks_path, "r", encoding="utf-8") as f:
                chunks = json.load(f)

            for index in sorted(target_indices):
                if not (0 <= index < len(chunks)):
                    continue
                chunk = chunks[index]
                audio_path = str(chunk.get("audio_path") or "").strip()
                if not audio_path:
                    continue
                files_to_delete.add(audio_path)
                chunk["audio_path"] = None
                chunk["audio_validation"] = None
                chunk["status"] = "pending"
                chunk["auto_regen_count"] = 0
                chunk.pop("generation_token", None)
                self._clear_proofread_state(chunk)
                cleared += 1

            if cleared > 0:
                self._atomic_json_write(chunks, self.chunks_path)

        root_abs = os.path.abspath(self.root_dir)
        for relative_path in files_to_delete:
            full_path = os.path.abspath(os.path.join(self.root_dir, relative_path))
            if not (full_path == root_abs or full_path.startswith(root_abs + os.sep)):
                continue
            if not os.path.exists(full_path):
                continue
            try:
                os.remove(full_path)
                deleted_files += 1
            except OSError:
                pass

        return {"invalidated_clips": cleared, "deleted_files": deleted_files}

    def save_voice_config_with_invalidation(self, new_config, confirm_invalidation=False):
        new_config = copy.deepcopy(new_config) if isinstance(new_config, dict) else {}
        old_config = self._load_voice_config()
        preview = self.preview_voice_config_invalidation(old_config, new_config)

        if preview["invalidated_clips"] > 0 and not confirm_invalidation:
            return {"status": "confirmation_required", **preview}

        self._save_voice_config(new_config)
        if preview["invalidated_clips"] <= 0:
            return {"status": "saved", "invalidated_clips": 0, "deleted_files": 0, **preview}

        applied = self.invalidate_chunk_audio_indices(preview["affected_indices"])
        return {"status": "saved", **preview, **applied}

    def get_engine(self):
        if self.engine:
            return self.engine

        try:
            self.engine = TTSEngine(self._load_app_config())
            backend = self.engine.local_backend if self.engine.mode == "local" else "external"
            print(f"TTS engine initialized (mode={self.engine.mode}, backend={backend})")
            return self.engine
        except Exception as e:
            print(f"Failed to initialize TTS engine: {e}")
            return None

    def unload_tts_engine(self):
        engine = self.engine
        self.engine = None
        if not engine:
            return False

        try:
            clear_cache = getattr(engine, "_clear_gpu_cache", None)
            if callable(clear_cache):
                clear_cache()
        except Exception:
            pass

        return True

    def _load_asr_settings(self):
        config = self._load_app_config()
        settings = dict(config.get("asr", {}) or {})
        settings.setdefault("enabled", True)
        settings.setdefault("model", "small.en")
        settings.setdefault("language", "en")
        settings.setdefault("device", "auto")
        settings.setdefault("compute_type", "auto")
        settings.setdefault("beam_size", 1)
        settings.setdefault("repair_window", 12)
        settings.setdefault("confidence_threshold", 0.72)
        settings.setdefault("confidence_margin", 0.08)
        cpu_count = max(os.cpu_count() or 1, 1)
        settings.setdefault("parallel_workers", cpu_count)
        settings.setdefault("cpu_threads", 1)
        return settings

    def get_asr_engine(self):
        settings = self._load_asr_settings()
        if not settings.get("enabled", True):
            raise LocalASRUnavailableError("Local ASR is disabled in app/config.json.")
        if self.asr_engine:
            return self.asr_engine

        self.asr_engine = LocalASREngine(
            model_size=settings.get("model", "small.en"),
            device=settings.get("device", "auto"),
            compute_type=settings.get("compute_type", "auto"),
            language=settings.get("language", "en"),
            beam_size=settings.get("beam_size", 1),
            cpu_threads=settings.get("cpu_threads", 1),
            num_workers=settings.get("parallel_workers", max(os.cpu_count() or 1, 1)),
        )
        return self.asr_engine

    @staticmethod
    def _clear_proofread_state(chunk):
        if isinstance(chunk, dict):
            chunk.pop("proofread", None)

    def _allowed_proofread_speaker_slugs(self, speaker, voice_config=None):
        voice_config = voice_config if voice_config is not None else self._load_voice_config()
        allowed = set()
        raw_speaker = (speaker or "").strip()
        if raw_speaker:
            slug = sanitize_filename(raw_speaker) or "speaker"
            allowed.add(slug)

        resolved = self.resolve_voice_speaker(raw_speaker, voice_config)
        if resolved:
            allowed.add(sanitize_filename(resolved) or "speaker")

        return {slug.lower() for slug in allowed if slug}

    @classmethod
    def _expand_common_abbreviation_words(cls, words):
        expanded = []
        replaced = False
        for word in words:
            replacement = COMMON_PROOFREAD_ABBREVIATIONS.get(word)
            if replacement:
                expanded.extend(replacement)
                replaced = True
            else:
                expanded.append(word)
        return tuple(expanded), replaced

    @staticmethod
    def _score_proofread_normalized_texts(normalized_expected, normalized_transcript):
        if not normalized_expected or not normalized_transcript:
            return {
                "score": 0.0,
                "word_precision": 0.0,
                "word_recall": 0.0,
                "word_f1": 0.0,
                "sequence_score": 0.0,
                "length_ratio": 0.0,
                "expected_word_count": len(normalized_expected.split()) if normalized_expected else 0,
                "transcript_word_count": len(normalized_transcript.split()) if normalized_transcript else 0,
            }

        expected_words = normalized_expected.split()
        transcript_words = normalized_transcript.split()
        expected_counter = Counter(expected_words)
        transcript_counter = Counter(transcript_words)
        overlap = sum((expected_counter & transcript_counter).values())
        precision = overlap / max(len(transcript_words), 1)
        recall = overlap / max(len(expected_words), 1)
        word_f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        sequence_score = SequenceMatcher(None, normalized_expected, normalized_transcript).ratio()
        length_ratio = min(len(expected_words), len(transcript_words)) / max(len(expected_words), len(transcript_words), 1)

        score = (word_f1 * 0.6) + (sequence_score * 0.25) + (length_ratio * 0.15)
        word_delta = abs(len(expected_words) - len(transcript_words))
        if word_delta >= 2:
            score *= max(0.0, 1.0 - (word_delta / max(len(expected_words), 1)))

        if normalized_expected == normalized_transcript:
            score = 1.0

        return {
            "score": round(max(0.0, min(score, 1.0)), 4),
            "word_precision": round(precision, 4),
            "word_recall": round(recall, 4),
            "word_f1": round(word_f1, 4),
            "sequence_score": round(sequence_score, 4),
            "length_ratio": round(length_ratio, 4),
            "expected_word_count": len(expected_words),
            "transcript_word_count": len(transcript_words),
        }

    @classmethod
    def _proofread_similarity_metrics(cls, expected_text, transcript_text):
        normalized_expected = cls._normalize_asr_text(expected_text)
        normalized_transcript = cls._normalize_asr_text(transcript_text)
        base_metrics = cls._score_proofread_normalized_texts(normalized_expected, normalized_transcript)

        expected_words = tuple(normalized_expected.split()) if normalized_expected else tuple()
        transcript_words = tuple(normalized_transcript.split()) if normalized_transcript else tuple()
        expanded_expected_words, expected_replaced = cls._expand_common_abbreviation_words(expected_words)
        expanded_transcript_words, transcript_replaced = cls._expand_common_abbreviation_words(transcript_words)

        if not expected_replaced and not transcript_replaced:
            return base_metrics

        expanded_metrics = cls._score_proofread_normalized_texts(
            " ".join(expanded_expected_words),
            " ".join(expanded_transcript_words),
        )

        if expanded_metrics["score"] > base_metrics["score"]:
            expanded_metrics["abbreviation_expanded_match"] = True
            return expanded_metrics

        return base_metrics

    def _build_chunk_proofread_result(self, chunk, threshold, voice_config, dictionary_entries, force_compare=False):
        audio_path = (chunk.get("audio_path") or "").strip()
        full_audio_path = os.path.join(self.root_dir, audio_path)
        expected_text, _ = apply_dictionary_to_text(chunk.get("text", ""), dictionary_entries)
        actual_duration_sec = get_audio_duration_seconds(full_audio_path)
        expected_duration_sec = estimate_expected_duration_seconds(text=expected_text)
        duration_delta_sec = abs(actual_duration_sec - expected_duration_sec)
        duration_gate_sec = self._proofread_duration_outlier_seconds_for_text(expected_text)
        checked_at = time.time()

        parsed_audio = self._parse_chunk_audio_candidate_name(os.path.basename(audio_path))
        audio_speaker_slug = (parsed_audio or {}).get("speaker_slug") or ""
        allowed_speaker_slugs = self._allowed_proofread_speaker_slugs(chunk.get("speaker", ""), voice_config)
        speaker_match = bool(audio_speaker_slug) and audio_speaker_slug in allowed_speaker_slugs

        base = {
            "checked": True,
            "checked_at": checked_at,
            "threshold": float(threshold),
            "audio_path": audio_path,
            "speaker_match": speaker_match,
            "expected_duration_sec": round(expected_duration_sec, 3),
            "actual_duration_sec": round(actual_duration_sec, 3),
            "duration_delta_sec": round(duration_delta_sec, 3),
            "transcript_text": "",
        }

        if not speaker_match:
            if force_compare:
                transcript = self.transcribe_audio_path(audio_path)
                metrics = self._proofread_similarity_metrics(expected_text, transcript.get("text", ""))
                score = metrics["score"]
                return base | metrics | {
                    "score": score,
                    "passed": False,
                    "error": "Audio filename speaker does not match the chunk speaker.",
                    "auto_failed_reason": None,
                    "transcript_text": transcript.get("text", ""),
                    "normalized_transcript": transcript.get("normalized_text") or self._normalize_asr_text(transcript.get("text", "")),
                    "forced_compare": True,
                }
            return base | {
                "score": 0.0,
                "passed": False,
                "error": "Audio filename speaker does not match the chunk speaker.",
                "auto_failed_reason": "speaker_mismatch",
            }

        if (
            not force_compare
            and
            not self._proofread_should_force_asr_for_short_audio(actual_duration_sec)
            and duration_delta_sec > duration_gate_sec
        ):
            return base | {
                "score": 0.0,
                "passed": False,
                "error": (
                    f"Audio duration differs from expected speech length by {duration_delta_sec:.1f}s."
                ),
                "auto_failed_reason": "duration_outlier",
            }

        transcript = self.transcribe_audio_path(audio_path)
        metrics = self._proofread_similarity_metrics(expected_text, transcript.get("text", ""))
        score = metrics["score"]
        return base | metrics | {
            "score": score,
            "passed": score >= float(threshold),
            "error": None if score >= float(threshold) else "Transcript confidence below threshold.",
            "auto_failed_reason": None,
            "transcript_text": transcript.get("text", ""),
            "normalized_transcript": transcript.get("normalized_text") or self._normalize_asr_text(transcript.get("text", "")),
            "forced_compare": bool(force_compare),
        }

    def _commit_proofread_result_locked(self, chunks, index, proofread_result):
        # Proofread can run in a separate process; never write an old snapshot
        # back wholesale, or we can clobber in-flight generation_token/status.
        latest = chunks
        if os.path.exists(self.chunks_path):
            try:
                with open(self.chunks_path, "r", encoding="utf-8") as f:
                    latest_loaded = json.load(f)
                if isinstance(latest_loaded, list):
                    latest = latest_loaded
            except (OSError, ValueError, json.JSONDecodeError):
                latest = chunks

        if 0 <= index < len(latest):
            latest[index]["proofread"] = proofread_result
        if 0 <= index < len(chunks):
            chunks[index]["proofread"] = proofread_result

        self._atomic_json_write(latest, self.chunks_path)
        return proofread_result

    def _commit_proofread_results_batch_locked(self, chunks, pending_results):
        if not pending_results:
            return 0
        # Proofread can run in a separate process; merge into freshest chunks
        # to avoid dropping live generation state written by the audio worker.
        latest = chunks
        if os.path.exists(self.chunks_path):
            try:
                with open(self.chunks_path, "r", encoding="utf-8") as f:
                    latest_loaded = json.load(f)
                if isinstance(latest_loaded, list):
                    latest = latest_loaded
            except (OSError, ValueError, json.JSONDecodeError):
                latest = chunks

        for index, proofread_result in pending_results.items():
            if 0 <= index < len(latest):
                latest[index]["proofread"] = proofread_result
            if 0 <= index < len(chunks):
                chunks[index]["proofread"] = proofread_result

        self._atomic_json_write(latest, self.chunks_path)
        return len(pending_results)

    @staticmethod
    def _discarded_proofread_state(existing_proofread, current_audio_path):
        proofread = dict(existing_proofread or {})
        transcript_text = (proofread.get("transcript_text") or "").strip()
        cached_audio_path = (proofread.get("audio_path") or "").strip()
        current_audio_path = (current_audio_path or "").strip()
        if transcript_text and cached_audio_path and cached_audio_path == current_audio_path:
            kept = {
                "checked": False,
                "audio_path": current_audio_path,
                "transcript_text": transcript_text,
            }
            normalized_transcript = (proofread.get("normalized_transcript") or "").strip()
            if normalized_transcript:
                kept["normalized_transcript"] = normalized_transcript
            return kept
        return None

    @staticmethod
    def _cached_chunk_transcript_text(chunk):
        chunk = chunk or {}
        proofread = chunk.get("proofread") or {}
        current_audio_path = (chunk.get("audio_path") or "").strip()
        if (proofread.get("audio_path") or "").strip() == current_audio_path:
            transcript_text = (proofread.get("transcript_text") or "").strip()
            if transcript_text:
                return transcript_text

        audio_validation = chunk.get("audio_validation") or {}
        transcript_text = (audio_validation.get("transcript_text") or "").strip()
        if transcript_text and current_audio_path:
            return transcript_text
        return ""

    @staticmethod
    def _cached_chunk_audio_validation(chunk, full_audio_path):
        chunk = chunk or {}
        audio_validation = chunk.get("audio_validation") or {}
        if not audio_validation:
            return None
        try:
            cached_size = int(audio_validation.get("file_size_bytes"))
            current_size = os.path.getsize(full_audio_path)
        except (TypeError, ValueError, OSError):
            return None
        if cached_size != current_size:
            return None
        return audio_validation

    def _manually_mark_proofread_clip(
        self,
        chunk_ref,
        threshold,
        *,
        accept: bool,
        toggle_validated_to_failure: bool = False,
        failure_error: str = "Manually rejected by user.",
    ):
        """Shared core for validate (accept=True) and reject (accept=False)."""
        with self._chunks_lock:
            if not os.path.exists(self.chunks_path):
                return None

            with open(self.chunks_path, "r", encoding="utf-8") as f:
                chunks = json.load(f)

            index = self.resolve_chunk_index(chunk_ref, chunks)
            if index is None or not (0 <= index < len(chunks)):
                return None

            chunk = chunks[index]
            audio_path = (chunk.get("audio_path") or "").strip()
            if not audio_path:
                raise ValueError("Cannot mark a clip with no audio.")

            full_audio_path = os.path.join(self.root_dir, audio_path)
            if not os.path.exists(full_audio_path):
                raise ValueError("Cannot mark a clip whose audio file is missing.")

            existing = dict(chunk.get("proofread") or {})
            now = time.time()
            proofread_state = {
                **existing,
                "checked": True,
                "checked_at": now,
                "threshold": float(threshold),
                "audio_path": audio_path,
            }
            if "speaker_match" not in proofread_state:
                proofread_state["speaker_match"] = True
            if "transcript_text" not in proofread_state:
                proofread_state["transcript_text"] = ""

            effective_accept = accept
            if (
                accept
                and toggle_validated_to_failure
                and bool(existing.get("manual_validated"))
                and bool(existing.get("passed"))
            ):
                effective_accept = False

            if effective_accept:
                proofread_state.update({
                    "score": 1.0,
                    "passed": True,
                    "error": None,
                    "manual_validated": True,
                    "manual_failed": False,
                    "validated_at": now,
                })
                proofread_state.pop("failed_at", None)
            else:
                proofread_state.update({
                    "score": 0.0,
                    "passed": False,
                    "error": failure_error,
                    "manual_validated": False,
                    "manual_failed": True,
                    "failed_at": now,
                })
                proofread_state.pop("validated_at", None)

            chunk["proofread"] = proofread_state
            self._atomic_json_write(chunks, self.chunks_path)
            return chunk

    def manually_validate_proofread_clip(self, chunk_ref, threshold=1.0):
        return self._manually_mark_proofread_clip(
            chunk_ref,
            threshold,
            accept=True,
            toggle_validated_to_failure=True,
            failure_error="Manually marked as failed by user.",
        )

    def manually_reject_proofread_clip(self, chunk_ref, threshold=1.0):
        return self._manually_mark_proofread_clip(chunk_ref, threshold, accept=False)

    def compare_proofread_clip(self, chunk_ref, threshold=1.0):
        with self._chunks_lock:
            if not os.path.exists(self.chunks_path):
                return None

            with open(self.chunks_path, "r", encoding="utf-8") as f:
                chunks = json.load(f)

            index = self.resolve_chunk_index(chunk_ref, chunks)
            if index is None or not (0 <= index < len(chunks)):
                return None

            chunk = chunks[index]
            audio_path = (chunk.get("audio_path") or "").strip()
            if not audio_path:
                raise ValueError("Cannot compare a clip with no audio.")

            full_audio_path = os.path.join(self.root_dir, audio_path)
            if not os.path.exists(full_audio_path):
                raise ValueError("Cannot compare a clip whose audio file is missing.")

            dictionary_entries = self.load_dictionary_entries()
            voice_config = self._load_voice_config()
            proofread_result = self._build_chunk_proofread_result(
                chunk,
                threshold=threshold,
                voice_config=voice_config,
                dictionary_entries=dictionary_entries,
                force_compare=True,
            )
            chunk["proofread"] = proofread_result
            self._atomic_json_write(chunks, self.chunks_path)
            return chunk

    def discard_proofread_selection(self, chapter=None):
        with self._chunks_lock:
            if not os.path.exists(self.chunks_path):
                return {
                    "discarded": 0,
                    "preserved_transcripts": 0,
                    "cleared_transcripts": 0,
                    "chapter": chapter,
                }

            with open(self.chunks_path, "r", encoding="utf-8") as f:
                chunks = json.load(f)

            discarded = 0
            preserved_transcripts = 0
            cleared_transcripts = 0

            for chunk in chunks:
                if not self._chunk_in_scope(chunk, chapter):
                    continue

                existing_proofread = chunk.get("proofread")
                if not existing_proofread:
                    continue

                replacement = self._discarded_proofread_state(existing_proofread, chunk.get("audio_path"))
                if replacement:
                    chunk["proofread"] = replacement
                    preserved_transcripts += 1
                else:
                    chunk.pop("proofread", None)
                    cleared_transcripts += 1
                discarded += 1

            if discarded:
                self._atomic_json_write(chunks, self.chunks_path)

            return {
                "discarded": discarded,
                "preserved_transcripts": preserved_transcripts,
                "cleared_transcripts": cleared_transcripts,
                "chapter": chapter,
            }

    def _reset_stale_proofread_scope_locked(self, chunks, scoped_indices):
        discarded = 0
        preserved_transcripts = 0
        cleared_transcripts = 0

        for index in scoped_indices:
            chunk = chunks[index]
            existing_proofread = chunk.get("proofread") or {}
            if not existing_proofread or not existing_proofread.get("checked"):
                continue
            if existing_proofread.get("manual_validated") and (
                (existing_proofread.get("audio_path") or "").strip() == (chunk.get("audio_path") or "").strip()
            ):
                continue

            replacement = self._discarded_proofread_state(existing_proofread, chunk.get("audio_path"))
            if replacement:
                chunk["proofread"] = replacement
                preserved_transcripts += 1
            else:
                chunk.pop("proofread", None)
                cleared_transcripts += 1
            discarded += 1

        if discarded:
            self._atomic_json_write(chunks, self.chunks_path)

        return {
            "discarded": discarded,
            "preserved_transcripts": preserved_transcripts,
            "cleared_transcripts": cleared_transcripts,
        }

    @staticmethod
    def _normalize_asr_text(text):
        text = (text or "").lower()
        # Delete apostrophes/curly-apostrophes so contractions and possessives
        # collapse into a single token instead of splitting on the apostrophe.
        # e.g. "can't" → "cant", "they're" → "theyre", "it's" → "its"
        # Without this, the regex below replaces the apostrophe with a space,
        # turning "can't" into "can t" (2 tokens) while a transcript that omits
        # the apostrophe ("cant") stays as 1 token — a false mismatch.
        text = re.sub(r"['\u2018\u2019\u02bc]", "", text)
        # Replace every remaining non-alphanumeric character with a space
        text = re.sub(r"[^a-z0-9]+", " ", text)
        return re.sub(r"\s+", " ", text).strip()

    @classmethod
    def _normalize_asr_words(cls, text):
        normalized = cls._normalize_asr_text(text)
        if not normalized:
            return tuple()
        return tuple(normalized.split())

    @classmethod
    def _proofread_duration_outlier_seconds_for_text(cls, text):
        word_count = cls._count_words(text)
        if word_count > PROOFREAD_LONG_CHUNK_WORD_THRESHOLD:
            return PROOFREAD_LONG_CHUNK_DURATION_OUTLIER_SECONDS
        return PROOFREAD_DURATION_OUTLIER_SECONDS

    @staticmethod
    def _proofread_should_force_asr_for_short_audio(actual_duration_sec):
        return float(actual_duration_sec or 0.0) <= PROOFREAD_SHORT_AUDIO_FORCE_ASR_SECONDS

    @classmethod
    def _asr_similarity_score(cls, transcript_text, chunk_text):
        normalized_transcript = cls._normalize_asr_text(transcript_text)
        normalized_chunk = cls._normalize_asr_text(chunk_text)
        if not normalized_transcript or not normalized_chunk:
            return 0.0

        transcript_words = normalized_transcript.split()
        chunk_words = normalized_chunk.split()
        transcript_set = set(transcript_words)
        chunk_set = set(chunk_words)
        overlap = len(transcript_set & chunk_set)
        if overlap <= 0:
            return 0.0

        precision = overlap / max(len(transcript_set), 1)
        recall = overlap / max(len(chunk_set), 1)
        token_score = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        seq_score = SequenceMatcher(None, normalized_transcript, normalized_chunk).ratio()
        length_ratio = min(len(transcript_words), len(chunk_words)) / max(len(transcript_words), len(chunk_words), 1)
        return (token_score * 0.5) + (seq_score * 0.35) + (length_ratio * 0.15)

    def transcribe_audio_path(self, relative_audio_path):
        cached = self._lookup_cached_transcription(relative_audio_path)
        if cached is not None:
            return cached

        full_path = os.path.join(self.root_dir, relative_audio_path)
        result = self.get_asr_engine().transcribe_file(full_path)
        result["normalized_text"] = self._normalize_asr_text(result.get("text"))
        result["cached"] = False
        self._store_cached_transcription(relative_audio_path, result)
        return result

    def transcribe_audio_paths_bulk(self, relative_audio_paths, progress_callback=None):
        paths = [path for path in relative_audio_paths if path]
        if not paths:
            return {}

        engine = self.get_asr_engine()
        max_workers = max(int(getattr(engine, "num_workers", 1) or 1), 1)
        results = {}

        def transcribe_one(path):
            return path, self.transcribe_audio_path(path)

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(transcribe_one, path): path
                for path in paths
            }
            completed = 0
            total = len(futures)
            for future in concurrent.futures.as_completed(futures):
                path = futures[future]
                completed += 1
                results[path] = future.result()[1]
                if progress_callback and (completed <= 3 or completed % 25 == 0 or completed == total):
                    progress_callback(completed, total, path)

        return results

    def _load_chunks_from_disk_locked(self):
        if os.path.exists(self.chunks_path):
            try:
                with open(self.chunks_path, "r", encoding="utf-8") as f:
                    chunks = json.load(f)
                if self._ensure_chunk_uids(chunks):
                    self._atomic_json_write(chunks, self.chunks_path)
                return chunks
            except (json.JSONDecodeError, ValueError) as e:
                backup_path = f"{self.chunks_path}.corrupt-{int(time.time())}"
                try:
                    shutil.copy2(self.chunks_path, backup_path)
                except OSError as backup_error:
                    print(f"WARNING: Failed to back up corrupted chunks.json: {backup_error}")
                    backup_path = None
                raise RuntimeError(
                    f"chunks.json is corrupted ({e})."
                    + (f" Preserved a backup at {backup_path}." if backup_path else "")
                ) from e

        if os.path.exists(self.script_path):
            script = load_script_document(self.script_path)["entries"]
            chunks = script_entries_to_chunks(script)
            for i, chunk in enumerate(chunks):
                chunk["id"] = i
                chunk["uid"] = chunk.get("uid") or self._new_chunk_uid()
                chunk["status"] = "pending"
                chunk["audio_path"] = None
                chunk["audio_validation"] = None
                chunk["auto_regen_count"] = 0
            self._atomic_json_write(chunks, self.chunks_path)
            return chunks

        return []

    def load_chunks_raw(self):
        snapshot = self._copy_chunks_snapshot()
        if snapshot is not None:
            return snapshot

        with self._chunks_lock:
            snapshot = self._copy_chunks_snapshot()
            if snapshot is None:
                snapshot = self._load_chunks_from_disk_locked()
                self._set_chunks_snapshot(snapshot)
            return copy.deepcopy(snapshot)

    def load_chunks_view(self):
        chunks = self.load_chunks_raw()
        runtime = self._copy_chunk_runtime()
        return [
            self._merge_runtime_chunk(chunk, runtime.get(chunk.get("uid")))
            for chunk in chunks
        ]

    def load_chunks(self):
        return self.load_chunks_raw()

    def sync_chunks_from_script_if_stale(self):
        """Rebuild chunks from annotated_script.json when the script is newer.

        This is a conservative sync used when navigating into the editor after a
        script-generation/review flow. It avoids overwriting user-edited chunks
        unless the script file is clearly newer than the chunk timeline.
        """
        if not os.path.exists(self.script_path):
            return {"synced": False, "reason": "no_script"}

        if not os.path.exists(self.chunks_path):
            chunks = self.load_chunks()
            return {"synced": True, "reason": "missing_chunks", "chunk_count": len(chunks)}

        script_mtime = os.path.getmtime(self.script_path)
        chunks_mtime = os.path.getmtime(self.chunks_path)
        if script_mtime <= chunks_mtime:
            return {"synced": False, "reason": "chunks_current"}

        script = load_script_document(self.script_path)["entries"]
        chunks = script_entries_to_chunks(script)
        for i, chunk in enumerate(chunks):
            chunk["id"] = i
            chunk["uid"] = chunk.get("uid") or self._new_chunk_uid()
            chunk["status"] = "pending"
            chunk["audio_path"] = None
            chunk["audio_validation"] = None
            chunk["auto_regen_count"] = 0

        self.save_chunks(chunks)
        return {
            "synced": True,
            "reason": "script_newer_than_chunks",
            "chunk_count": len(chunks),
            "script_mtime": script_mtime,
            "chunks_mtime": chunks_mtime,
        }

    def reconcile_chunk_audio_states(self):
        """Repair stale error states when a stored clip is actually valid.

        This is intended for editor/UI refreshes after interrupted or large
        generations, where a chunk may still be marked as error even though its
        audio file exists and now passes the duration sanity check.
        """
        if not os.path.exists(self.chunks_path):
            return self.load_chunks()

        with self._chunks_lock:
            with open(self.chunks_path, "r", encoding="utf-8") as f:
                chunks = json.load(f)

            dictionary_entries = self.load_dictionary_entries()
            changed = False
            immediate_commits = 0

            for chunk in chunks:
                audio_path = chunk.get("audio_path")
                if not audio_path:
                    continue

                full_audio_path = os.path.join(self.root_dir, audio_path)
                if not os.path.exists(full_audio_path):
                    continue

                try:
                    transformed_text, _ = apply_dictionary_to_text(
                        chunk.get("text", ""),
                        dictionary_entries,
                    )
                    validation = validate_audio_clip(
                        text=transformed_text,
                        actual_duration_sec=get_audio_duration_seconds(full_audio_path),
                        file_size_bytes=os.path.getsize(full_audio_path),
                    ).to_dict()
                except Exception as e:
                    print(f"Warning: failed to revalidate chunk {chunk.get('id')}: {e}")
                    continue

                if validation["is_valid"]:
                    # Any chunk with valid persisted audio should be treated as done.
                    # This prevents restart/resume from re-queueing already-rendered clips
                    # when status drifted (e.g. pending/error/generating after interruption).
                    if chunk.get("status") != "done":
                        chunk["status"] = "done"
                        changed = True
                    if chunk.get("audio_validation") != validation:
                        chunk["audio_validation"] = validation
                        changed = True
                    if int(chunk.get("auto_regen_count") or 0) != 0:
                        chunk["auto_regen_count"] = 0
                        changed = True
                elif chunk.get("audio_validation") != validation:
                    chunk["audio_validation"] = validation
                    changed = True

            if changed:
                self._atomic_json_write(chunks, self.chunks_path)

            return chunks

    def _validate_chunk_audio(self, chunk, dictionary_entries):
        audio_path = chunk.get("audio_path")
        if not audio_path:
            return None

        return self._validate_audio_path_for_chunk(chunk, audio_path, dictionary_entries)

    def _validate_audio_path_for_chunk(self, chunk, audio_path, dictionary_entries):
        full_audio_path = os.path.join(self.root_dir, audio_path)
        if not os.path.exists(full_audio_path):
            return None

        transformed_text, _ = apply_dictionary_to_text(
            chunk.get("text", ""),
            dictionary_entries,
        )
        return validate_audio_clip(
            text=transformed_text,
            actual_duration_sec=get_audio_duration_seconds(full_audio_path),
            file_size_bytes=os.path.getsize(full_audio_path),
        ).to_dict()

    @staticmethod
    def _parse_chunk_audio_candidate_name(filename):
        match = re.match(
            r"^voiceline_([0-9a-f]{32}|\d+)_([^./]+?)(?:_retry(\d+))?\.(mp3|wav)$",
            filename,
            re.IGNORECASE,
        )
        if not match:
            return None
        identifier, speaker_slug, retry_str, ext = match.groups()
        return {
            "identifier": identifier,
            "speaker_slug": (speaker_slug or "").strip().lower(),
            "retry": int(retry_str or 0),
            "ext": ext.lower(),
            "is_uid": not identifier.isdigit(),
            "legacy_index": (int(identifier) - 1) if identifier.isdigit() else None,
        }

    def _discarded_voicelines_dir(self):
        return os.path.join(self.voicelines_dir, "discarded")

    def _collect_repair_candidates_from_dir_locked(self, directory_path, relative_prefix):
        candidates = []
        scanned_files = 0
        if not os.path.isdir(directory_path):
            return candidates, scanned_files

        for entry in os.listdir(directory_path):
            full_path = os.path.join(directory_path, entry)
            if not os.path.isfile(full_path):
                continue
            scanned_files += 1
            candidates.append({
                "relative_path": f"{relative_prefix}/{entry}",
                "parsed": self._parse_chunk_audio_candidate_name(entry),
                "mtime": os.path.getmtime(full_path),
            })

        candidates.sort(
            key=lambda item: (
                -int(item["mtime"]),
                -int((item["parsed"] or {}).get("retry") or 0),
                item["relative_path"],
            )
        )
        return candidates, scanned_files

    def _move_candidate_to_discarded_locked(self, source_relative_path):
        normalized_source = str(source_relative_path or "").replace("\\", "/").strip("/")
        source_full_path = os.path.join(self.root_dir, normalized_source)
        if not os.path.exists(source_full_path):
            return None

        discarded_dir = self._discarded_voicelines_dir()
        os.makedirs(discarded_dir, exist_ok=True)

        filename = os.path.basename(normalized_source)
        stem, ext = os.path.splitext(filename)
        candidate_filename = filename
        candidate_full_path = os.path.join(discarded_dir, candidate_filename)
        counter = 1
        while os.path.exists(candidate_full_path):
            candidate_filename = f"{stem}_{counter}{ext}"
            candidate_full_path = os.path.join(discarded_dir, candidate_filename)
            counter += 1

        os.replace(source_full_path, candidate_full_path)
        return f"voicelines/discarded/{candidate_filename}"

    def _build_lost_audio_repair_text_index(self, chunks, dictionary_entries, voice_config):
        index = defaultdict(lambda: defaultdict(list))
        for chunk_index, chunk in enumerate(chunks):
            transformed_text, _ = apply_dictionary_to_text(chunk.get("text", ""), dictionary_entries)
            normalized_words = self._normalize_asr_words(transformed_text)
            if not normalized_words:
                continue
            for speaker_slug in self._allowed_proofread_speaker_slugs(chunk.get("speaker", ""), voice_config):
                index[speaker_slug][normalized_words].append(chunk_index)
        return index

    def _build_repair_match_cache(self, chunks, dictionary_entries, voice_config):
        text_index = defaultdict(lambda: defaultdict(list))
        speaker_word_frequency_cache = defaultdict(Counter)
        speaker_word_index = defaultdict(lambda: defaultdict(set))
        speaker_chunk_indices = defaultdict(list)
        chunk_entries = []

        for chunk_index, chunk in enumerate(chunks):
            transformed_text, _ = apply_dictionary_to_text(chunk.get("text", ""), dictionary_entries)
            normalized_words = self._normalize_asr_words(transformed_text)
            normalized_text = " ".join(normalized_words)
            speaker_slugs = tuple(self._allowed_proofread_speaker_slugs(chunk.get("speaker", ""), voice_config))
            entry = {
                "index": chunk_index,
                "normalized_words": normalized_words,
                "normalized_text": normalized_text,
                "chapter": (chunk.get("chapter") or "").strip().lower(),
                "speaker_slugs": speaker_slugs,
                "transformed_text": transformed_text,
            }
            chunk_entries.append(entry)
            if not normalized_words:
                continue
            for speaker_slug in speaker_slugs:
                text_index[speaker_slug][normalized_words].append(chunk_index)
                speaker_word_frequency_cache[speaker_slug].update(normalized_words)
                speaker_chunk_indices[speaker_slug].append(chunk_index)
                for word in set(normalized_words):
                    speaker_word_index[speaker_slug][word].add(chunk_index)

        return {
            "chunk_entries": chunk_entries,
            "text_index": text_index,
            "speaker_word_frequency_cache": speaker_word_frequency_cache,
            "speaker_word_index": speaker_word_index,
            "speaker_chunk_indices": speaker_chunk_indices,
        }

    def _build_speaker_word_frequency_cache(self, chunks, dictionary_entries, voice_config):
        frequency_cache = defaultdict(Counter)
        for chunk in chunks:
            transformed_text, _ = apply_dictionary_to_text(chunk.get("text", ""), dictionary_entries)
            normalized_text = self._normalize_asr_text(transformed_text)
            if not normalized_text:
                continue
            words = normalized_text.split()
            if not words:
                continue
            for speaker_slug in self._allowed_proofread_speaker_slugs(chunk.get("speaker", ""), voice_config):
                frequency_cache[speaker_slug].update(words)
        return frequency_cache

    @staticmethod
    def _drop_low_frequency_transcript_words(text, frequency_counter, drop_count=2):
        normalized_text = ProjectManager._normalize_asr_text(text)
        words = normalized_text.split()
        if not words:
            return normalized_text, tuple()

        ranked = sorted(
            enumerate(words),
            key=lambda item: (
                int(frequency_counter.get(item[1], 0)),
                item[0],
            ),
        )
        drop_indices = {index for index, _ in ranked[:max(int(drop_count or 0), 0)]}
        dropped_words = tuple(words[index] for index in sorted(drop_indices))
        trimmed_words = [word for index, word in enumerate(words) if index not in drop_indices]
        return " ".join(trimmed_words).strip(), dropped_words

    @staticmethod
    def _remove_selected_words(text, words_to_remove):
        normalized_text = ProjectManager._normalize_asr_text(text)
        if not normalized_text or not words_to_remove:
            return normalized_text

        remaining = list(normalized_text.split())
        for word in words_to_remove:
            try:
                remaining.remove(word)
            except ValueError:
                continue
        return " ".join(remaining).strip()

    def _candidate_indices_for_discarded_match(self, speaker_slug, transcript_text, claimed_indices, match_cache):
        speaker_indices = match_cache["speaker_chunk_indices"].get(speaker_slug) or []
        if not speaker_indices:
            return []

        transcript_words = self._normalize_asr_words(transcript_text)
        if not transcript_words:
            return []

        word_index = match_cache["speaker_word_index"].get(speaker_slug) or {}
        candidate_pool = None
        ranked_words = sorted(
            set(transcript_words),
            key=lambda word: (len(word_index.get(word) or ()), word),
        )
        for word in ranked_words[:4]:
            indexed = word_index.get(word)
            if not indexed:
                continue
            indexed = set(indexed)
            if candidate_pool is None:
                candidate_pool = indexed
                continue
            intersected = candidate_pool & indexed
            if intersected:
                candidate_pool = intersected
            if len(candidate_pool) <= 64:
                break

        if candidate_pool:
            filtered = [index for index in candidate_pool if index not in claimed_indices]
            if filtered:
                return filtered

        return [index for index in speaker_indices if index not in claimed_indices]

    def _best_discarded_repair_match(
        self,
        speaker_slug,
        transcript_text,
        claimed_indices,
        threshold,
        match_cache,
    ):
        if not speaker_slug or not transcript_text:
            return None

        speaker_word_frequency_cache = match_cache.get("speaker_word_frequency_cache") or {}
        speaker_frequency = speaker_word_frequency_cache.get(speaker_slug) or Counter()
        reduced_transcript_text, dropped_words = self._drop_low_frequency_transcript_words(
            transcript_text,
            speaker_frequency,
            drop_count=2,
        )
        candidate_indices = self._candidate_indices_for_discarded_match(
            speaker_slug,
            transcript_text,
            claimed_indices,
            match_cache,
        )
        if not candidate_indices:
            return None

        scored = []
        chunk_entries = match_cache.get("chunk_entries") or []
        for index in candidate_indices:
            if not (0 <= index < len(chunk_entries)):
                continue
            entry = chunk_entries[index]
            metrics = self._proofread_similarity_metrics(entry["transformed_text"], transcript_text)
            reduced_metrics = None
            if dropped_words and reduced_transcript_text:
                reduced_expected_text = self._remove_selected_words(entry["transformed_text"], dropped_words)
                if reduced_expected_text:
                    reduced_metrics = self._proofread_similarity_metrics(reduced_expected_text, reduced_transcript_text)
                    if float(reduced_metrics.get("score", 0.0) or 0.0) > float(metrics.get("score", 0.0) or 0.0):
                        metrics = dict(reduced_metrics)
                        metrics["reduced_transcript_match"] = True
                        metrics["dropped_low_frequency_words"] = list(dropped_words)
            score = float(metrics.get("score", 0.0) or 0.0)
            if score <= 0:
                continue
            scored.append((score, index, metrics))

        if not scored:
            return None

        scored.sort(key=lambda item: (-item[0], item[1]))
        best_score, best_index, best_metrics = scored[0]
        if best_score < float(threshold):
            return None
        if len(scored) > 1 and abs(scored[1][0] - best_score) < 1e-6:
            return None

        return {
            "index": best_index,
            "score": best_score,
            "metrics": best_metrics,
        }

    def _candidate_nearby_chunk_indices(self, candidate, chunks, claimed_indices, window_size):
        parsed = candidate["parsed"]
        if parsed["is_uid"]:
            target_uid = str(parsed["identifier"]).strip()
            for index, chunk in enumerate(chunks):
                if index in claimed_indices:
                    continue
                if str(chunk.get("uid") or "").strip() == target_uid:
                    return [index]
            return []

        anchor_index = parsed.get("legacy_index")
        if anchor_index is None:
            return []

        anchor_chunk = chunks[anchor_index] if 0 <= anchor_index < len(chunks) else {}
        anchor_chapter = (anchor_chunk.get("chapter") or "").strip().lower()
        ordered = []
        seen = set()

        def add_index(idx):
            if idx in seen or idx in claimed_indices or not (0 <= idx < len(chunks)):
                return
            seen.add(idx)
            ordered.append(idx)

        add_index(anchor_index)
        for offset in range(1, max(int(window_size or 0), 0) + 1):
            add_index(anchor_index - offset)
            add_index(anchor_index + offset)

        if anchor_chapter:
            chapter_matches = [
                idx for idx in ordered
                if (chunks[idx].get("chapter") or "").strip().lower() == anchor_chapter
            ]
            if chapter_matches:
                return chapter_matches + [idx for idx in ordered if idx not in chapter_matches]
        return ordered

    @staticmethod
    def _candidate_anchor_chapter(candidate, chunks):
        anchor_index = candidate["parsed"].get("legacy_index")
        if anchor_index is None or not (0 <= anchor_index < len(chunks)):
            return ""
        return (chunks[anchor_index].get("chapter") or "").strip().lower()

    def _candidate_same_chapter_indices(self, candidate, chunks, claimed_indices):
        anchor_chapter = self._candidate_anchor_chapter(candidate, chunks)
        if not anchor_chapter:
            return []
        return [
            index
            for index, chunk in enumerate(chunks)
            if index not in claimed_indices
            and (chunk.get("chapter") or "").strip().lower() == anchor_chapter
        ]

    def _find_asr_repair_match(self, candidate, chunks, claimed_indices, dictionary_entries, transcript_cache, match_cache=None):
        settings = self._load_asr_settings()
        window_size = max(int(settings.get("repair_window", 12) or 12), 1)
        threshold = float(settings.get("confidence_threshold", 0.72) or 0.72)
        margin = float(settings.get("confidence_margin", 0.08) or 0.08)
        nearby_indices = self._candidate_nearby_chunk_indices(candidate, chunks, claimed_indices, window_size)
        if not nearby_indices:
            return None

        relative_path = candidate["relative_path"]
        transcript = transcript_cache.get(relative_path)
        if transcript is None:
            transcript = self.transcribe_audio_path(relative_path)
            transcript_cache[relative_path] = transcript

        transcript_words = self._normalize_asr_words(transcript.get("normalized_text") or transcript.get("text"))
        if not transcript_words:
            return None
        transcript_text = " ".join(transcript_words)

        candidate_slug = candidate["parsed"].get("speaker_slug") or "speaker"
        chapter_indices = self._candidate_same_chapter_indices(candidate, chunks, claimed_indices)
        chunk_entries = (match_cache or {}).get("chunk_entries") or []

        exact_matches = []
        for index in chapter_indices:
            chunk = chunks[index]
            speaker_slug = sanitize_filename(chunk.get("speaker") or "") or "speaker"
            if candidate_slug != speaker_slug:
                continue
            if 0 <= index < len(chunk_entries):
                normalized_chunk_words = chunk_entries[index]["normalized_words"]
            else:
                transformed_text, _ = apply_dictionary_to_text(chunk.get("text", ""), dictionary_entries)
                normalized_chunk_words = self._normalize_asr_words(transformed_text)
            if normalized_chunk_words and normalized_chunk_words == transcript_words:
                exact_matches.append(index)

        if len(exact_matches) == 1:
            best_index = exact_matches[0]
            chunk = chunks[best_index]
            validation = self._validate_audio_path_for_chunk(chunk, relative_path, dictionary_entries)
            validation = dict(validation or {})
            validation["matched_via_asr"] = True
            validation["asr_score"] = 1.0
            validation["asr_margin"] = 1.0
            validation["transcript_text"] = transcript.get("text", "")
            validation["exact_chapter_match"] = True
            return {
                "index": best_index,
                "validation": validation,
                "score": 1.0,
                "margin": 1.0,
                "transcript_text": transcript.get("text", ""),
            }

        scored = []
        search_indices = chapter_indices or nearby_indices
        for index in search_indices:
            chunk = chunks[index]
            speaker_slug = sanitize_filename(chunk.get("speaker") or "") or "speaker"
            if candidate_slug != speaker_slug:
                continue
            if 0 <= index < len(chunk_entries):
                transformed_text = chunk_entries[index]["transformed_text"]
            else:
                transformed_text, _ = apply_dictionary_to_text(chunk.get("text", ""), dictionary_entries)
            score = self._asr_similarity_score(transcript_text, transformed_text)
            if score <= 0:
                continue

            chunk_chapter = (chunk.get("chapter") or "").strip().lower()
            anchor_chapter = self._candidate_anchor_chapter(candidate, chunks)
            if anchor_chapter and chunk_chapter and anchor_chapter == chunk_chapter:
                score += 0.05

            score = min(score, 1.0)
            scored.append((score, index))

        if not scored:
            return None

        scored.sort(reverse=True)
        best_score, best_index = scored[0]
        second_score = scored[1][0] if len(scored) > 1 else 0.0
        if best_score < threshold or (best_score - second_score) < margin:
            return None

        chunk = chunks[best_index]
        validation = self._validate_audio_path_for_chunk(chunk, relative_path, dictionary_entries)
        validation = dict(validation or {})
        validation["matched_via_asr"] = True
        validation["asr_score"] = round(best_score, 4)
        validation["asr_margin"] = round(best_score - second_score, 4)
        validation["transcript_text"] = transcript.get("text", "")
        return {
            "index": best_index,
            "validation": validation,
            "score": best_score,
            "margin": best_score - second_score,
            "transcript_text": transcript.get("text", ""),
        }

    def repair_lost_audio_links(self, use_asr=True, progress_callback=None, rejected_only=False):
        """Rebuild all chunk audio links from exact speaker+transcript matches only.

        This emergency repair intentionally ignores all existing chunk
        assignments and clip identifiers. Every clip under voicelines/ is treated
        as untrusted until its transcript matches exactly one chunk for the same
        speaker (allowing configured aliases). Anything else is moved into
        voicelines/discarded/ so later repairs do not retry it.
        """
        def emit_progress(payload):
            if progress_callback:
                progress_callback(payload)

        if not os.path.exists(self.chunks_path):
            return {
                "relinked": 0,
                "preserved": 0,
                "invalid_candidates": 0,
                "unmatched_files": 0,
                "total_candidates": 0,
                "asr_relinked": 0,
                "asr_errors": [],
                "examples": [],
            }

        with self._chunks_lock:
            with open(self.chunks_path, "r", encoding="utf-8") as f:
                chunks = json.load(f)

            repair_started_at = time.time()
            dictionary_entries = self.load_dictionary_entries()
            voice_config = self._load_voice_config()
            changed = False
            immediate_commits = 0
            relinked = 0
            preserved = 0
            invalid_candidates = 0
            unmatched_files = 0
            asr_relinked = 0
            discarded_retry_relinked = 0
            asr_errors = []
            examples = []
            duplicate_matches = 0
            claimed_indices = set()

            if not use_asr:
                emit_progress({
                    "phase": "scan",
                    "message": "Lost audio repair requires local ASR. Continuing with ASR enabled.",
                    "elapsed_seconds": round(time.time() - repair_started_at, 2),
                })

            match_cache = self._build_repair_match_cache(chunks, dictionary_entries, voice_config)
            text_index = match_cache["text_index"]

            if rejected_only:
                claimed_indices = {
                    index for index, chunk in enumerate(chunks)
                    if (chunk.get("audio_path") or "").strip()
                }
                candidates = []
                scanned_files = 0
            else:
                for chunk in chunks:
                    chunk["audio_path"] = None
                    chunk["audio_validation"] = None
                    chunk["status"] = "pending"
                    chunk["auto_regen_count"] = 0
                    chunk.pop("generation_token", None)
                    self._clear_proofread_state(chunk)
                self._atomic_json_write(chunks, self.chunks_path)
                changed = True

                candidates, scanned_files = self._collect_repair_candidates_from_dir_locked(
                    self.voicelines_dir,
                    "voicelines",
                )

            total_candidates = len(candidates)
            total_chunks = len(chunks)
            discarded_candidates = []
            discarded_scanned_files = 0
            proofread_threshold = float((self._load_app_config().get("proofread") or {}).get("certainty_threshold", 1.0) or 1.0)
            pending_checkpoint_writes = 0
            if rejected_only or total_candidates == 0:
                discarded_candidates, discarded_scanned_files = self._collect_repair_candidates_from_dir_locked(
                    self._discarded_voicelines_dir(),
                    "voicelines/discarded",
                )
            emit_progress({
                "phase": "scan",
                "message": (
                    (
                        f"Scanned {scanned_files} clip file(s). Reset {total_chunks} chunk assignment(s). "
                        f"Found {total_candidates} candidate clip(s) for full-project transcript matching."
                        if not rejected_only else
                        f"Rejected-only repair scanned {discarded_scanned_files} rejected clip(s) at certainty {proofread_threshold:.2f}."
                    )
                    + (
                        f" Falling back to {discarded_scanned_files} rejected clip(s) at certainty {proofread_threshold:.2f}."
                        if (not rejected_only) and total_candidates == 0 and discarded_candidates else ""
                    )
                ),
                "scanned_files": scanned_files,
                "total_candidates": total_candidates,
                "total_chunks": total_chunks,
                "discarded_candidates": len(discarded_candidates),
                "elapsed_seconds": round(time.time() - repair_started_at, 2),
            })

            transcribe_paths = [candidate["relative_path"] for candidate in candidates]
            if (rejected_only or not transcribe_paths) and discarded_candidates:
                transcribe_paths = [candidate["relative_path"] for candidate in discarded_candidates]
            transcript_cache = {}
            asr_started_at = time.time()
            if transcribe_paths:
                try:
                    transcript_cache = self.transcribe_audio_paths_bulk(
                        transcribe_paths,
                        progress_callback=lambda completed, total, path: emit_progress({
                            "phase": "asr_transcribe",
                            "message": (
                                f"Transcribed {completed}/{total} repair clip(s) in parallel. "
                                f"Latest: {path}"
                            ),
                            "asr_transcribed": completed,
                            "asr_transcribe_total": total,
                            "elapsed_seconds": round(time.time() - asr_started_at, 2),
                        }),
                    )
                except LocalASRUnavailableError as e:
                    asr_errors.append(str(e))
                    emit_progress({
                        "phase": "asr",
                        "message": f"ASR unavailable: {e}",
                        "asr_errors": list(asr_errors),
                        "elapsed_seconds": round(time.time() - asr_started_at, 2),
                    })
                    return {
                        "relinked": relinked,
                        "preserved": preserved,
                        "invalid_candidates": invalid_candidates,
                        "unmatched_files": unmatched_files,
                        "total_candidates": total_candidates,
                        "asr_relinked": asr_relinked,
                        "asr_enabled": True,
                        "asr_errors": asr_errors[:10],
                        "examples": examples,
                    }

            for candidate_index, candidate in enumerate(candidates, start=1):
                relative_path = candidate["relative_path"]
                parsed = candidate["parsed"]
                speaker_slug = ((parsed or {}).get("speaker_slug") or "").strip().lower()

                if not speaker_slug:
                    invalid_candidates += 1
                    discarded_path = self._move_candidate_to_discarded_locked(relative_path)
                    if discarded_path:
                        emit_progress({
                            "phase": "matching",
                            "message": f"Discarded unparseable clip {relative_path}.",
                            "processed_candidates": candidate_index,
                            "total_candidates": total_candidates,
                            "invalid_candidates": invalid_candidates,
                            "elapsed_seconds": round(time.time() - repair_started_at, 2),
                        })
                    continue

                transcript = transcript_cache.get(relative_path) or {}
                normalized_transcript_words = self._normalize_asr_words(
                    transcript.get("normalized_text") or transcript.get("text")
                )
                if not normalized_transcript_words:
                    unmatched_files += 1
                    self._move_candidate_to_discarded_locked(relative_path)
                    continue

                normalized_transcript = " ".join(normalized_transcript_words)
                matching_indices = list(text_index.get(speaker_slug, {}).get(normalized_transcript_words, []))
                if not matching_indices:
                    unmatched_files += 1
                    self._move_candidate_to_discarded_locked(relative_path)
                elif len(matching_indices) > 1:
                    invalid_candidates += 1
                    self._move_candidate_to_discarded_locked(relative_path)
                else:
                    target_index = matching_indices[0]
                    if target_index in claimed_indices:
                        duplicate_matches += 1
                        self._move_candidate_to_discarded_locked(relative_path)
                    else:
                        chunk = chunks[target_index]
                        validation = self._validate_audio_path_for_chunk(chunk, relative_path, dictionary_entries) or {}
                        validation = dict(validation)
                        validation["matched_via_asr"] = True
                        validation["repair_exact_transcript_match"] = True
                        validation["transcript_text"] = transcript.get("text", "")
                        validation["normalized_transcript"] = normalized_transcript
                        committed_audio_path = self._commit_repaired_chunk_locked(
                            chunks,
                            target_index,
                            relative_path,
                            validation,
                            write_immediately=False,
                        )
                        changed = True
                        immediate_commits += 1
                        pending_checkpoint_writes += 1
                        relinked += 1
                        asr_relinked += 1
                        claimed_indices.add(target_index)
                        if len(examples) < 50:
                            examples.append({
                                "index": target_index,
                                "speaker": chunk.get("speaker"),
                                "audio_path": committed_audio_path,
                                "repair_mode": "exact_transcript",
                            })
                        if pending_checkpoint_writes >= REPAIR_BATCH_COMMIT_SIZE:
                            self._atomic_json_write(chunks, self.chunks_path)
                            pending_checkpoint_writes = 0

                if candidate_index <= 3 or candidate_index % 25 == 0 or candidate_index == total_candidates:
                    elapsed = time.time() - repair_started_at
                    rate = candidate_index / elapsed if elapsed > 0 else 0.0
                    remaining = max(total_candidates - candidate_index, 0)
                    eta = (remaining / rate) if rate > 0 else None
                    emit_progress({
                        "phase": "matching",
                        "message": (
                            f"Matched {candidate_index}/{total_candidates} clip(s). "
                            f"Linked {relinked}, discarded unmatched {unmatched_files}, "
                            f"discarded ambiguous {invalid_candidates}, duplicate targets {duplicate_matches}. "
                            + (f"ETA {int(eta)}s." if eta is not None else "")
                        ).strip(),
                        "processed_candidates": candidate_index,
                        "total_candidates": total_candidates,
                        "relinked": relinked,
                        "invalid_candidates": invalid_candidates,
                        "unmatched_files": unmatched_files,
                        "duplicate_matches": duplicate_matches,
                        "eta_seconds": None if eta is None else round(eta, 1),
                        "elapsed_seconds": round(elapsed, 2),
                    })

            if (rejected_only or total_candidates == 0) and discarded_candidates:
                emit_progress({
                    "phase": "discarded_retry",
                    "message": (
                        (
                            f"No active clips remained in voicelines/. Re-grading {len(discarded_candidates)} rejected clip(s) "
                            f"using certainty {proofread_threshold:.2f}."
                            if not rejected_only else
                            f"Re-grading {len(discarded_candidates)} rejected clip(s) using certainty {proofread_threshold:.2f}."
                        )
                    ),
                    "discarded_candidates": len(discarded_candidates),
                    "threshold": proofread_threshold,
                    "elapsed_seconds": round(time.time() - repair_started_at, 2),
                })
                for discarded_index, candidate in enumerate(discarded_candidates, start=1):
                    relative_path = candidate["relative_path"]
                    parsed = candidate["parsed"]
                    speaker_slug = ((parsed or {}).get("speaker_slug") or "").strip().lower()
                    transcript = transcript_cache.get(relative_path) or {}
                    transcript_text = (transcript.get("text") or "").strip()
                    if not speaker_slug or not transcript_text:
                        continue

                    match = self._best_discarded_repair_match(
                        speaker_slug,
                        transcript_text,
                        claimed_indices,
                        proofread_threshold,
                        match_cache,
                    )
                    if not match:
                        if discarded_index <= 3 or discarded_index % 25 == 0 or discarded_index == len(discarded_candidates):
                            elapsed = time.time() - repair_started_at
                            rate = discarded_index / elapsed if elapsed > 0 else 0.0
                            remaining = max(len(discarded_candidates) - discarded_index, 0)
                            eta = (remaining / rate) if rate > 0 else None
                            emit_progress({
                                "phase": "discarded_retry",
                                "message": (
                                    f"Re-graded {discarded_index}/{len(discarded_candidates)} rejected clip(s). "
                                    f"Recovered {discarded_retry_relinked}, left rejected {discarded_index - discarded_retry_relinked}. "
                                    + (f"ETA {int(eta)}s." if eta is not None else "")
                                ).strip(),
                                "discarded_checked": discarded_index,
                                "discarded_candidates": len(discarded_candidates),
                                "discarded_retry_relinked": discarded_retry_relinked,
                                "eta_seconds": None if eta is None else round(eta, 1),
                                "elapsed_seconds": round(elapsed, 2),
                            })
                        continue

                    target_index = match["index"]
                    chunk = chunks[target_index]
                    normalized_transcript = transcript.get("normalized_text") or self._normalize_asr_text(transcript_text)
                    validation = self._validate_audio_path_for_chunk(chunk, relative_path, dictionary_entries) or {}
                    validation = dict(validation)
                    validation["matched_via_asr"] = True
                    validation["repair_certainty_match"] = True
                    validation["repair_certainty_threshold"] = proofread_threshold
                    validation["repair_certainty_score"] = round(match["score"], 4)
                    validation["transcript_text"] = transcript_text
                    validation["normalized_transcript"] = normalized_transcript
                    committed_audio_path = self._commit_repaired_chunk_locked(
                        chunks,
                        target_index,
                        relative_path,
                        validation,
                        write_immediately=False,
                    )
                    changed = True
                    immediate_commits += 1
                    pending_checkpoint_writes += 1
                    relinked += 1
                    asr_relinked += 1
                    discarded_retry_relinked += 1
                    claimed_indices.add(target_index)
                    if len(examples) < 50:
                        examples.append({
                            "index": target_index,
                            "speaker": chunk.get("speaker"),
                            "audio_path": committed_audio_path,
                            "repair_mode": "discarded_certainty",
                            "score": round(match["score"], 4),
                        })
                    if pending_checkpoint_writes >= REPAIR_BATCH_COMMIT_SIZE:
                        self._atomic_json_write(chunks, self.chunks_path)
                        pending_checkpoint_writes = 0

                    if discarded_index <= 3 or discarded_index % 25 == 0 or discarded_index == len(discarded_candidates):
                        elapsed = time.time() - repair_started_at
                        rate = discarded_index / elapsed if elapsed > 0 else 0.0
                        remaining = max(len(discarded_candidates) - discarded_index, 0)
                        eta = (remaining / rate) if rate > 0 else None
                        emit_progress({
                            "phase": "discarded_retry",
                            "message": (
                                f"Re-graded {discarded_index}/{len(discarded_candidates)} rejected clip(s). "
                                f"Recovered {discarded_retry_relinked}, left rejected {discarded_index - discarded_retry_relinked}. "
                                + (f"ETA {int(eta)}s." if eta is not None else "")
                            ).strip(),
                            "discarded_checked": discarded_index,
                            "discarded_candidates": len(discarded_candidates),
                            "discarded_retry_relinked": discarded_retry_relinked,
                            "eta_seconds": None if eta is None else round(eta, 1),
                            "elapsed_seconds": round(elapsed, 2),
                        })

            if pending_checkpoint_writes > 0:
                self._atomic_json_write(chunks, self.chunks_path)

            emit_progress({
                "phase": "complete",
                "message": (
                    f"Lost audio repair complete in {time.time() - repair_started_at:.1f}s. "
                    f"Linked {relinked} exact transcript match(es), discarded {unmatched_files} unmatched clip(s), "
                    f"{invalid_candidates} ambiguous clip(s), {duplicate_matches} duplicate-target clip(s), "
                    f"and recovered {discarded_retry_relinked} rejected clip(s)."
                ),
                "preserved": preserved,
                "relinked": relinked,
                "asr_relinked": asr_relinked,
                "discarded_retry_relinked": discarded_retry_relinked,
                "invalid_candidates": invalid_candidates,
                "unmatched_files": unmatched_files,
                "duplicate_matches": duplicate_matches,
                "total_candidates": total_candidates,
                "discarded_candidates": len(discarded_candidates),
                "elapsed_seconds": round(time.time() - repair_started_at, 2),
            })
            return {
                "relinked": relinked,
                "preserved": preserved,
                "invalid_candidates": invalid_candidates,
                "unmatched_files": unmatched_files,
                "duplicate_matches": duplicate_matches,
                "total_candidates": total_candidates,
                "discarded_candidates": len(discarded_candidates),
                "asr_relinked": asr_relinked,
                "discarded_retry_relinked": discarded_retry_relinked,
                "asr_enabled": True,
                "asr_errors": asr_errors[:10],
                "examples": examples,
            }

    def proofread_chunks(self, chapter=None, threshold=1.0, progress_callback=None):
        def emit_progress(payload):
            if progress_callback:
                progress_callback(payload)

        with self._chunks_lock:
            if not os.path.exists(self.chunks_path):
                return {
                    "processed": 0,
                    "skipped": 0,
                    "auto_failed": 0,
                    "passed": 0,
                    "failed": 0,
                    "chapter": chapter,
                    "threshold": float(threshold),
                }

            with open(self.chunks_path, "r", encoding="utf-8") as f:
                chunks = json.load(f)

            started_at = time.time()
            dictionary_entries = self.load_dictionary_entries()
            voice_config = self._load_voice_config()

            scoped_indices = [
                index for index, chunk in enumerate(chunks)
                if self._chunk_in_scope(chunk, chapter)
                and (chunk.get("audio_path") or "").strip()
                and os.path.exists(os.path.join(self.root_dir, chunk.get("audio_path")))
            ]
            pending_indices = [
                index for index in scoped_indices
                if not bool((chunks[index].get("proofread") or {}).get("checked"))
            ]
            auto_reset_summary = {
                "discarded": 0,
                "preserved_transcripts": 0,
                "cleared_transcripts": 0,
            }
            if scoped_indices and not pending_indices:
                auto_reset_summary = self._reset_stale_proofread_scope_locked(chunks, scoped_indices)
                pending_indices = [
                    index for index in scoped_indices
                    if not bool((chunks[index].get("proofread") or {}).get("checked"))
                ]

            processed = 0
            skipped = len(scoped_indices) - len(pending_indices)
            auto_failed = 0
            passed = 0
            failed = 0

            emit_progress({
                "phase": "scan",
                "message": (
                    f"Proofread scan found {len(scoped_indices)} clip(s) in scope, "
                    f"{len(pending_indices)} still need checking."
                    + (
                        f" Auto-reset {auto_reset_summary['discarded']} stale grade(s), "
                        f"preserving {auto_reset_summary['preserved_transcripts']} transcript cache(s)."
                        if auto_reset_summary["discarded"] > 0
                        else ""
                    )
                ),
                "chapter": chapter,
                "threshold": float(threshold),
                "scoped_clips": len(scoped_indices),
                "pending_clips": len(pending_indices),
                "auto_reset_discarded": auto_reset_summary["discarded"],
                "auto_reset_preserved_transcripts": auto_reset_summary["preserved_transcripts"],
                "processed": processed,
                "pending_total": len(pending_indices),
                "passed": passed,
                "failed": failed,
                "auto_failed": auto_failed,
                "skipped": skipped,
                "elapsed_seconds": round(time.time() - started_at, 2),
            })

            precomputed_results = {}
            asr_needed = []
            for index in pending_indices:
                chunk = chunks[index]
                audio_path = (chunk.get("audio_path") or "").strip()
                full_audio_path = os.path.join(self.root_dir, audio_path)
                expected_text, _ = apply_dictionary_to_text(chunk.get("text", ""), dictionary_entries)
                cached_validation = self._cached_chunk_audio_validation(chunk, full_audio_path)
                actual_duration_sec = (
                    float(cached_validation.get("actual_duration_sec"))
                    if cached_validation and cached_validation.get("actual_duration_sec") is not None
                    else get_audio_duration_seconds(full_audio_path)
                )
                expected_duration_sec = estimate_expected_duration_seconds(text=expected_text)
                duration_delta_sec = abs(actual_duration_sec - expected_duration_sec)
                duration_gate_sec = self._proofread_duration_outlier_seconds_for_text(expected_text)
                parsed_audio = self._parse_chunk_audio_candidate_name(os.path.basename(audio_path))
                audio_speaker_slug = (parsed_audio or {}).get("speaker_slug") or ""
                allowed_speaker_slugs = self._allowed_proofread_speaker_slugs(chunk.get("speaker", ""), voice_config)
                speaker_match = bool(audio_speaker_slug) and audio_speaker_slug in allowed_speaker_slugs
                base = {
                    "checked": True,
                    "checked_at": time.time(),
                    "threshold": float(threshold),
                    "audio_path": audio_path,
                    "speaker_match": speaker_match,
                    "expected_duration_sec": round(expected_duration_sec, 3),
                    "actual_duration_sec": round(actual_duration_sec, 3),
                    "duration_delta_sec": round(duration_delta_sec, 3),
                    "transcript_text": "",
                }
                if not speaker_match:
                    precomputed_results[index] = base | {
                        "score": 0.0,
                        "passed": False,
                        "error": "Audio filename speaker does not match the chunk speaker.",
                        "auto_failed_reason": "speaker_mismatch",
                    }
                    continue
                if (
                    not self._proofread_should_force_asr_for_short_audio(actual_duration_sec)
                    and duration_delta_sec > duration_gate_sec
                ):
                    precomputed_results[index] = base | {
                        "score": 0.0,
                        "passed": False,
                        "error": f"Audio duration differs from expected speech length by {duration_delta_sec:.1f}s.",
                        "auto_failed_reason": "duration_outlier",
                    }
                    continue
                cached_transcript_text = self._cached_chunk_transcript_text(chunk)
                if cached_transcript_text:
                    metrics = self._proofread_similarity_metrics(expected_text, cached_transcript_text)
                    score = metrics["score"]
                    precomputed_results[index] = base | metrics | {
                        "score": score,
                        "passed": score >= float(threshold),
                        "error": None if score >= float(threshold) else "Transcript confidence below threshold.",
                        "auto_failed_reason": None,
                        "transcript_text": cached_transcript_text,
                        "normalized_transcript": self._normalize_asr_text(cached_transcript_text),
                    }
                    continue
                asr_needed.append((index, expected_text, base))

            if asr_needed:
                asr_started_at = time.time()
                transcript_map = self.transcribe_audio_paths_bulk(
                    [base["audio_path"] for _, _, base in asr_needed],
                    progress_callback=lambda completed, total, path: emit_progress({
                        "phase": "proofread_transcribe",
                        "message": (
                            f"Transcribed {completed}/{total} proofread clip(s) in parallel. "
                            f"Latest: {path}"
                            + (
                                f" ETA {int((max(total - completed, 0) / (completed / max(time.time() - asr_started_at, 0.001))))}s."
                                if completed > 0 and total > completed
                                else ""
                            )
                        ).strip(),
                        "transcribed": completed,
                        "transcribe_total": total,
                        "processed": processed,
                        "pending_total": len(pending_indices),
                        "passed": passed,
                        "failed": failed,
                        "auto_failed": auto_failed,
                        "skipped": skipped,
                        "eta_seconds": (
                            round(max(total - completed, 0) / (completed / max(time.time() - asr_started_at, 0.001)), 1)
                            if completed > 0 and total > completed
                            else 0.0 if total and completed >= total
                            else None
                        ),
                        "elapsed_seconds": round(time.time() - asr_started_at, 2),
                    }),
                )
                for index, expected_text, base in asr_needed:
                    transcript = transcript_map.get(base["audio_path"], {})
                    metrics = self._proofread_similarity_metrics(expected_text, transcript.get("text", ""))
                    score = metrics["score"]
                    precomputed_results[index] = base | metrics | {
                        "score": score,
                        "passed": score >= float(threshold),
                        "error": None if score >= float(threshold) else "Transcript confidence below threshold.",
                        "auto_failed_reason": None,
                        "transcript_text": transcript.get("text", ""),
                        "normalized_transcript": transcript.get("normalized_text") or self._normalize_asr_text(transcript.get("text", "")),
                    }

            pending_write_batch = {}
            for ordinal, index in enumerate(pending_indices, start=1):
                result = precomputed_results[index]
                pending_write_batch[index] = result
                processed += 1
                if result.get("auto_failed_reason"):
                    auto_failed += 1
                if result.get("passed"):
                    passed += 1
                else:
                    failed += 1

                should_flush_batch = (
                    len(pending_write_batch) >= PROOFREAD_BATCH_COMMIT_SIZE
                    or ordinal == len(pending_indices)
                )
                if should_flush_batch:
                    self._commit_proofread_results_batch_locked(chunks, pending_write_batch)
                    pending_write_batch.clear()

                elapsed = time.time() - started_at
                rate = processed / elapsed if elapsed > 0 else 0.0
                remaining = max(len(pending_indices) - processed, 0)
                eta = (remaining / rate) if rate > 0 else None
                if processed <= 3 or processed % 25 == 0 or ordinal == len(pending_indices):
                    emit_progress({
                        "phase": "proofreading",
                        "message": (
                            f"Proofread {processed}/{len(pending_indices)} pending clip(s). "
                            f"Passed {passed}, failed {failed}, auto-failed {auto_failed}, "
                            f"skipped {skipped}. "
                            + (f"ETA {int(eta)}s." if eta is not None else "")
                        ).strip(),
                        "chapter": chapter,
                        "threshold": float(threshold),
                        "processed": processed,
                        "pending_total": len(pending_indices),
                        "passed": passed,
                        "failed": failed,
                        "auto_failed": auto_failed,
                        "skipped": skipped,
                        "eta_seconds": None if eta is None else round(eta, 1),
                        "elapsed_seconds": round(elapsed, 2),
                        "current_chunk_index": index,
                    })

            emit_progress({
                "phase": "complete",
                "message": (
                    f"Proofreading complete in {time.time() - started_at:.1f}s. "
                    f"Processed {processed}, skipped {skipped}, passed {passed}, failed {failed}."
                ),
                "chapter": chapter,
                "threshold": float(threshold),
                "processed": processed,
                "pending_total": len(pending_indices),
                "passed": passed,
                "failed": failed,
                "auto_failed": auto_failed,
                "skipped": skipped,
                "elapsed_seconds": round(time.time() - started_at, 2),
            })

            return {
                "processed": processed,
                "skipped": skipped,
                "auto_failed": auto_failed,
                "passed": passed,
                "failed": failed,
                "chapter": chapter,
                "threshold": float(threshold),
                "auto_reset_discarded": auto_reset_summary["discarded"],
            }

    def clear_proofread_failures(self, chapter=None, threshold=1.0):
        with self._chunks_lock:
            if not os.path.exists(self.chunks_path):
                return {
                    "cleared": 0,
                    "failed_candidates": 0,
                    "ungraded_with_audio": 0,
                    "chapter": chapter,
                    "threshold": float(threshold),
                }

            with open(self.chunks_path, "r", encoding="utf-8") as f:
                chunks = json.load(f)

            cleared = 0
            failed_candidates = 0
            ungraded_with_audio = 0

            for chunk in chunks:
                if not self._chunk_in_scope(chunk, chapter):
                    continue

                audio_path = (chunk.get("audio_path") or "").strip()
                if not audio_path:
                    continue

                proofread = chunk.get("proofread") or {}
                if not proofread.get("checked"):
                    ungraded_with_audio += 1
                    continue

                if proofread.get("passed") and float(proofread.get("score", 0.0) or 0.0) >= float(threshold):
                    continue

                failed_candidates += 1
                full_audio_path = os.path.join(self.root_dir, audio_path)
                if os.path.exists(full_audio_path):
                    try:
                        os.remove(full_audio_path)
                    except OSError:
                        pass

                chunk["audio_path"] = None
                chunk["audio_validation"] = None
                chunk["status"] = "pending"
                chunk["auto_regen_count"] = 0
                chunk.pop("generation_token", None)
                self._clear_proofread_state(chunk)
                cleared += 1

            if cleared:
                self._atomic_json_write(chunks, self.chunks_path)

            return {
                "cleared": cleared,
                "failed_candidates": failed_candidates,
                "ungraded_with_audio": ungraded_with_audio,
                "chapter": chapter,
                "threshold": float(threshold),
            }

    def recover_interrupted_generating_chunks(self, indices=None, generation_token=None):
        """Recover valid audio for interrupted generating chunks on restart.

        If a chunk was left in "generating" but its audio file already exists and
        validates, promote it to "done". Otherwise reset it back to "pending".
        """
        self.flush_dirty_chunks(force=True)
        outcome = {"recovered": 0, "reset": 0}

        with self._chunks_lock:
            if not os.path.exists(self.chunks_path):
                return outcome

            with open(self.chunks_path, "r", encoding="utf-8") as f:
                chunks = json.load(f)

            if indices is None:
                index_iter = range(len(chunks))
            else:
                index_iter = [index for index in indices if 0 <= index < len(chunks)]

            dictionary_entries = self.load_dictionary_entries()
            changed = False

            for index in index_iter:
                chunk = chunks[index]
                if chunk.get("status") != "generating":
                    continue
                if generation_token is not None and chunk.get("generation_token") != generation_token:
                    continue

                try:
                    validation = self._validate_chunk_audio(chunk, dictionary_entries)
                except Exception as e:
                    print(f"Warning: failed to validate interrupted chunk {chunk.get('id')}: {e}")
                    validation = None

                if validation and validation["is_valid"]:
                    chunk["status"] = "done"
                    chunk["audio_validation"] = validation
                    chunk["auto_regen_count"] = 0
                    outcome["recovered"] += 1
                else:
                    chunk["status"] = "pending"
                    outcome["reset"] += 1

                chunk.pop("generation_token", None)
                self.clear_chunk_runtime(chunk.get("uid"))
                changed = True

            if changed:
                self._atomic_json_write(chunks, self.chunks_path)

        return outcome

    def load_script_document(self):
        if not os.path.exists(self.script_path):
            return {"entries": [], "dictionary": []}
        return load_script_document(self.script_path)

    def load_dictionary_entries(self):
        return self.load_script_document()["dictionary"]

    @staticmethod
    def _count_audio_linked_chunks(chunks):
        return sum(1 for chunk in (chunks or []) if str((chunk or {}).get("audio_path") or "").strip())

    def _load_chunk_backup_audio_count(self, backup_path):
        if not os.path.exists(backup_path):
            return -1
        try:
            with open(backup_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            if not isinstance(payload, list):
                return -1
            return self._count_audio_linked_chunks(payload)
        except (OSError, ValueError, json.JSONDecodeError):
            return -1

    def _update_chunks_backups(self, chunks):
        if not isinstance(chunks, list):
            return
        self._atomic_json_write_raw(chunks, self.chunks_latest_backup_path)
        current_audio_count = self._count_audio_linked_chunks(chunks)
        best_audio_count = self._load_chunk_backup_audio_count(self.chunks_best_backup_path)
        if current_audio_count > best_audio_count:
            self._atomic_json_write_raw(chunks, self.chunks_best_backup_path)

    def _atomic_json_write_raw(self, data, target_path, max_retries=5):
        """Atomically write JSON data with retry logic for Windows file locking."""
        for attempt in range(max_retries):
            tmp_path = None
            try:
                with tempfile.NamedTemporaryFile(
                    mode="w",
                    encoding="utf-8",
                    dir=os.path.dirname(target_path) or ".",
                    prefix=f".{os.path.basename(target_path)}.",
                    suffix=".tmp",
                    delete=False,
                ) as f:
                    tmp_path = f.name
                    json.dump(data, f, indent=2, ensure_ascii=False)
                os.replace(tmp_path, target_path)
                return
            except OSError as e:
                if tmp_path and os.path.exists(tmp_path):
                    try:
                        os.remove(tmp_path)
                    except OSError:
                        pass
                if attempt < max_retries - 1 and (
                    e.errno == 5 or "Access is denied" in str(e) or "being used by another process" in str(e)
                ):
                    delay = 0.05 * (2 ** attempt)
                    time.sleep(delay)
                    continue
                raise

    def _atomic_json_write(self, data, target_path, max_retries=5):
        self._atomic_json_write_raw(data, target_path, max_retries=max_retries)
        if os.path.abspath(target_path) == os.path.abspath(self.chunks_path):
            self._set_chunks_snapshot(data)
            self._update_chunks_backups(data)

    def save_chunks(self, chunks):
        with self._chunks_lock:
            self._ensure_chunk_uids(chunks)
            self._atomic_json_write(chunks, self.chunks_path)
        chunk_uids = {chunk.get("uid") for chunk in chunks if chunk.get("uid")}
        with self._chunk_runtime_lock:
            stale_uids = [uid for uid in self._chunk_runtime if uid not in chunk_uids]
            for uid in stale_uids:
                self._chunk_runtime.pop(uid, None)
                self._dirty_chunk_uids.discard(uid)

    def _update_chunk_fields(self, index, **fields):
        """Atomically update fields on a single chunk (thread-safe read-modify-write).

        Unlike load_chunks() + modify + save_chunks(), this holds the lock for the
        entire read-modify-write cycle, preventing concurrent threads from
        overwriting each other's updates.
        """
        with self._chunks_lock:
            if not os.path.exists(self.chunks_path):
                return None
            with open(self.chunks_path, "r", encoding="utf-8") as f:
                chunks = json.load(f)
            if not (0 <= index < len(chunks)):
                return None
            chunks[index].update(fields)
            if "audio_path" in fields or "speaker" in fields or "text" in fields or "instruct" in fields:
                self._clear_proofread_state(chunks[index])
            self._atomic_json_write(chunks, self.chunks_path)
            return chunks[index]

    def _claim_chunk_generation(self, index, generation_token=None):
        chunks = self.load_chunks_raw()
        if not (0 <= index < len(chunks)):
            return None
        chunk = chunks[index]
        uid = chunk.get("uid")
        self.set_chunk_runtime(
            uid,
            status="generating",
            generation_token=generation_token,
            auto_regen_count=int(chunk.get("auto_regen_count") or 0),
        )
        return self._runtime_chunk_state(chunk)

    def _claim_chunks_generation(self, indices, generation_token=None):
        claimed = 0
        chunks = self.load_chunks_raw()
        for index in indices:
            if not (0 <= index < len(chunks)):
                continue
            chunk = chunks[index]
            self.set_chunk_runtime(
                chunk.get("uid"),
                status="generating",
                generation_token=generation_token,
                auto_regen_count=int(chunk.get("auto_regen_count") or 0),
            )
            claimed += 1
        return claimed

    def _chunk_token_matches(self, chunks, index, generation_token=None):
        if generation_token is None:
            return True
        if not (0 <= index < len(chunks)):
            return False
        return chunks[index].get("generation_token") == generation_token

    def chunk_has_generation_token(self, index, generation_token=None):
        chunks = self.load_chunks_raw()
        if not (0 <= index < len(chunks)):
            return False
        return self._chunk_token_matches_live(chunks[index], generation_token)

    def _update_chunk_fields_if_token(self, index, expected_token=None, **fields):
        chunks = self.load_chunks_raw()
        if not (0 <= index < len(chunks)):
            return None
        chunk = chunks[index]
        if not self._chunk_token_matches_live(chunk, expected_token):
            return None

        runtime_fields = dict(fields)
        if runtime_fields.get("status") != "generating" and "generation_token" not in runtime_fields:
            runtime_fields["generation_token"] = None
        runtime_chunk = self.set_chunk_runtime(chunk.get("uid"), **runtime_fields)
        self.mark_chunks_dirty([chunk.get("uid")])
        return self._merge_runtime_chunk(dict(chunk), runtime_chunk)

    def force_reset_chunks_to_pending(self, indices):
        """Force any chunk in `indices` to pending status regardless of current state.

        Clears audio_path, audio_validation, generation_token and all proofread
        state so the UI immediately reflects the reset.  Called before a
        Regenerate-All job is enqueued so the user gets instant feedback.
        """
        reset_count = 0
        with self._chunks_lock:
            if not os.path.exists(self.chunks_path):
                return reset_count
            with open(self.chunks_path, "r", encoding="utf-8") as f:
                chunks = json.load(f)
            for index in indices:
                if not (0 <= index < len(chunks)):
                    continue
                chunk = chunks[index]
                chunk["status"] = "pending"
                chunk["audio_path"] = None
                chunk["audio_validation"] = None
                chunk.pop("generation_token", None)
                self._clear_proofread_state(chunk)
                self.clear_chunk_runtime(chunk.get("uid"))
                reset_count += 1
            if reset_count:
                self._atomic_json_write(chunks, self.chunks_path)
        return reset_count

    def reset_generating_chunks(self, indices=None, generation_token=None, target_status="pending"):
        reset_count = 0
        chunks = self.load_chunks_raw()
        if indices is None:
            index_iter = range(len(chunks))
        else:
            index_iter = [index for index in indices if 0 <= index < len(chunks)]
        for index in index_iter:
            chunk = chunks[index]
            live_chunk = self._runtime_chunk_state(chunk)
            if live_chunk.get("status") != "generating":
                continue
            if generation_token is not None and live_chunk.get("generation_token") != generation_token:
                continue
            self.set_chunk_runtime(
                chunk.get("uid"),
                status=target_status,
                generation_token=None,
            )
            self.mark_chunks_dirty([chunk.get("uid")])
            reset_count += 1
        if reset_count:
            self.flush_dirty_chunks(force=True)
        return reset_count

    def insert_chunk(self, after_ref):
        """Insert an empty chunk after the given index. Returns the new chunk list."""
        with self._chunks_lock:
            if not os.path.exists(self.chunks_path):
                return None
            with open(self.chunks_path, "r", encoding="utf-8") as f:
                chunks = json.load(f)
            after_index = self.resolve_chunk_index(after_ref, chunks)
            if after_index is None or not (0 <= after_index < len(chunks)):
                return None

            # Copy speaker from the row we're splitting from
            source = chunks[after_index]
            new_chunk = {
                "id": after_index + 1,
                "uid": self._new_chunk_uid(),
                "speaker": source.get("speaker", "NARRATOR"),
                "text": "",
                "instruct": "",
                "status": "pending",
                "audio_path": None
            }
            if source.get("chapter"):
                new_chunk["chapter"] = source["chapter"]
            if source.get("paragraph_id"):
                new_chunk["paragraph_id"] = source["paragraph_id"]
            chunks.insert(after_index + 1, new_chunk)

            # Re-number all IDs
            for i, chunk in enumerate(chunks):
                chunk["id"] = i

            self._atomic_json_write(chunks, self.chunks_path)
            return new_chunk, chunks

    def insert_silence_chunk(self, after_ref, duration_s=1.0):
        """Insert a silence block after the given chunk. Returns (new_chunk, chunks) or None."""
        with self._chunks_lock:
            if not os.path.exists(self.chunks_path):
                return None
            with open(self.chunks_path, "r", encoding="utf-8") as f:
                chunks = json.load(f)
            after_index = self.resolve_chunk_index(after_ref, chunks)
            if after_index is None or not (0 <= after_index < len(chunks)):
                return None
            source = chunks[after_index]
            new_chunk = {
                "id": after_index + 1,
                "uid": self._new_chunk_uid(),
                "type": "silence",
                "silence_duration_s": float(duration_s),
                "status": "done",
            }
            if source.get("chapter"):
                new_chunk["chapter"] = source["chapter"]
            chunks.insert(after_index + 1, new_chunk)
            for i, chunk in enumerate(chunks):
                chunk["id"] = i
            self._atomic_json_write(chunks, self.chunks_path)
            return new_chunk, chunks

    def delete_chunk(self, chunk_ref):
        """Delete a chunk at the given index. Returns (deleted_chunk, updated_chunks) or None."""
        with self._chunks_lock:
            if not os.path.exists(self.chunks_path):
                return None
            with open(self.chunks_path, "r", encoding="utf-8") as f:
                chunks = json.load(f)
            index = self.resolve_chunk_index(chunk_ref, chunks)
            if index is None or not (0 <= index < len(chunks)):
                return None
            if len(chunks) <= 1:
                return None  # don't allow deleting the last chunk

            restore_after_uid = chunks[index - 1].get("uid") if index > 0 else None
            deleted = chunks.pop(index)

            # Re-number all IDs
            for i, chunk in enumerate(chunks):
                chunk["id"] = i

            self._atomic_json_write(chunks, self.chunks_path)
            return deleted, chunks, restore_after_uid

    def delete_chapter(self, chapter_name):
        """Delete all chunks belonging to a chapter and remove their audio files.
        Returns (deleted_count, updated_chunks) or None on failure.
        """
        if not chapter_name or not isinstance(chapter_name, str) or not chapter_name.strip():
            return None
        chapter_name = chapter_name.strip()
        with self._chunks_lock:
            if not os.path.exists(self.chunks_path):
                return None
            with open(self.chunks_path, "r", encoding="utf-8") as f:
                chunks = json.load(f)

            deleted = [c for c in chunks if c.get("chapter") == chapter_name]
            if not deleted:
                return None
            keep = [c for c in chunks if c.get("chapter") != chapter_name]

            # Re-number all IDs
            for i, chunk in enumerate(keep):
                chunk["id"] = i

            # Collect and delete audio files for deleted chunks
            for chunk in deleted:
                ap = chunk.get("audio_path")
                if ap:
                    full_ap = os.path.join(self.root_dir, ap)
                    if os.path.exists(full_ap):
                        try:
                            os.remove(full_ap)
                        except OSError:
                            pass

            self._atomic_json_write(keep, self.chunks_path)
            return len(deleted), keep

    def restore_chunk(self, at_index, chunk_data, after_uid=None):
        """Re-insert a chunk at a specific index. Returns the updated chunk list."""
        with self._chunks_lock:
            if not os.path.exists(self.chunks_path):
                return None
            with open(self.chunks_path, "r", encoding="utf-8") as f:
                chunks = json.load(f)

            if after_uid:
                resolved = self.resolve_chunk_index(after_uid, chunks)
                at_index = 0 if resolved is None else resolved + 1
            else:
                at_index = max(0, min(at_index, len(chunks)))
            chunks.insert(at_index, chunk_data)

            # Re-number all IDs
            for i, chunk in enumerate(chunks):
                chunk["id"] = i
            self._ensure_chunk_uids(chunks)

            self._atomic_json_write(chunks, self.chunks_path)
            return chunks

    def repair_legacy_chunk_order(self, chunks):
        """Rewrite chunks.json from the editor's current chunk order.

        This is a legacy repair tool for projects that may have suffered from
        index-based insert/delete/restore mismatches. The editor's current
        full chunk list is treated as the source of truth.
        """
        if not isinstance(chunks, list) or not chunks:
            return None

        repaired = []
        for i, chunk in enumerate(chunks):
            if not isinstance(chunk, dict):
                return None
            repaired_chunk = copy.deepcopy(chunk)
            repaired_chunk["id"] = i
            repaired_chunk["uid"] = repaired_chunk.get("uid") or self._new_chunk_uid()
            repaired.append(repaired_chunk)

        with self._chunks_lock:
            self._atomic_json_write(repaired, self.chunks_path)

        return repaired

    @staticmethod
    def _legacy_audio_index_from_path(audio_path):
        match = re.match(r"^voicelines/voiceline_(\d+)_", str(audio_path or ""))
        if not match:
            return None
        return max(int(match.group(1)) - 1, 0)

    @staticmethod
    def _stable_audio_uid_from_path(audio_path):
        match = re.match(r"^voicelines/voiceline_([A-Za-z0-9-]+)_[^/]+(?:_retry\d+)?\.(?:mp3|wav)$", str(audio_path or ""))
        if not match:
            return None
        candidate = match.group(1)
        if candidate.isdigit():
            return None
        return candidate

    def invalidate_stale_audio_references(self):
        """Clear only audio references that are provably stale.

        Multiple chunks pointing at the same file is never valid. For newer
        UID-based filenames, the matching chunk UID is the only valid owner.
        For legacy index-based filenames, the chunk currently sitting at the
        encoded index is the only defensible owner. All other claimants are
        cleared back to pending so they can be regenerated safely.
        """
        with self._chunks_lock:
            if not os.path.exists(self.chunks_path):
                return {"invalidated": 0, "duplicate_groups": 0, "kept": 0, "examples": []}

            with open(self.chunks_path, "r", encoding="utf-8") as f:
                chunks = json.load(f)

            owners = defaultdict(list)
            for index, chunk in enumerate(chunks):
                audio_path = chunk.get("audio_path")
                if audio_path:
                    owners[audio_path].append(index)

            invalidated = []
            kept = set()
            examples = []

            for audio_path, indices in owners.items():
                if len(indices) < 2:
                    continue

                canonical = set()
                stable_uid = self._stable_audio_uid_from_path(audio_path)
                if stable_uid:
                    canonical = {
                        index for index in indices
                        if str(chunks[index].get("uid") or "").strip() == stable_uid
                    }
                else:
                    legacy_index = self._legacy_audio_index_from_path(audio_path)
                    if legacy_index is not None and legacy_index in indices:
                        canonical = {legacy_index}

                stale_indices = [index for index in indices if index not in canonical]
                if not stale_indices:
                    kept.update(indices)
                    continue

                if len(examples) < 25:
                    examples.append({
                        "audio_path": audio_path,
                        "kept_indices": sorted(canonical),
                        "invalidated_indices": stale_indices,
                    })

                kept.update(canonical)
                for index in stale_indices:
                    chunk = chunks[index]
                    chunk["audio_path"] = None
                    chunk["audio_validation"] = None
                    chunk["status"] = "pending"
                    chunk["auto_regen_count"] = 0
                    chunk.pop("generation_token", None)
                    self._clear_proofread_state(chunk)
                    invalidated.append(index)

            if invalidated:
                self._atomic_json_write(chunks, self.chunks_path)

            return {
                "invalidated": len(invalidated),
                "duplicate_groups": sum(1 for indices in owners.values() if len(indices) > 1),
                "kept": len(kept),
                "examples": examples,
            }

    def decompose_long_segments(self, chapter=None, max_words=25):
        with self._chunks_lock:
            if not os.path.exists(self.chunks_path):
                return {
                    "changed": 0,
                    "total_chunks": 0,
                    "processed_scope": 0,
                    "chapter": chapter,
                    "max_words": max_words,
                }

            with open(self.chunks_path, "r", encoding="utf-8") as f:
                chunks = json.load(f)

            changed = 0

            while True:
                changed_this_pass = False
                index = 0

                while index < len(chunks):
                    chunk = chunks[index]
                    chunk_chapter = (chunk.get("chapter") or "").strip()
                    text = (chunk.get("text") or "").strip()

                    if chapter and chunk_chapter != chapter:
                        index += 1
                        continue

                    if chunk.get("audio_path"):
                        index += 1
                        continue

                    if self._count_words(text) <= max_words:
                        index += 1
                        continue

                    split_text = self._split_long_chunk_text(text)
                    if not split_text:
                        index += 1
                        continue

                    left_text, right_text = split_text
                    base_chunk = copy.deepcopy(chunk)

                    left_chunk = copy.deepcopy(base_chunk)
                    left_chunk["text"] = left_text
                    left_chunk["status"] = "pending"
                    left_chunk["audio_path"] = None
                    left_chunk["audio_validation"] = None
                    left_chunk["auto_regen_count"] = 0
                    left_chunk.pop("generation_token", None)
                    self._clear_proofread_state(left_chunk)

                    right_chunk = copy.deepcopy(base_chunk)
                    right_chunk["text"] = right_text
                    right_chunk["status"] = "pending"
                    right_chunk["audio_path"] = None
                    right_chunk["audio_validation"] = None
                    right_chunk["auto_regen_count"] = 0
                    right_chunk.pop("generation_token", None)
                    self._clear_proofread_state(right_chunk)

                    chunks[index:index + 1] = [left_chunk, right_chunk]
                    changed += 1
                    changed_this_pass = True
                    index += 2

                if not changed_this_pass:
                    break

            for i, chunk in enumerate(chunks):
                chunk["id"] = i

            if changed:
                self._atomic_json_write(chunks, self.chunks_path)

            processed_scope = 0
            for chunk in chunks:
                if chapter and (chunk.get("chapter") or "").strip() != chapter:
                    continue
                processed_scope += 1

            return {
                "changed": changed,
                "total_chunks": len(chunks),
                "processed_scope": processed_scope,
                "chapter": chapter,
                "max_words": max_words,
            }

    def merge_orphan_segments(self, chapter=None, min_words=10):
        with self._chunks_lock:
            if not os.path.exists(self.chunks_path):
                return {
                    "changed": 0,
                    "total_chunks": 0,
                    "processed_scope": 0,
                    "chapter": chapter,
                    "min_words": min_words,
                }

            with open(self.chunks_path, "r", encoding="utf-8") as f:
                chunks = json.load(f)

            chapter_chunk_counts = {}
            for chunk in chunks:
                chapter_key = self._chapter_key(chunk)
                chapter_position = chapter_chunk_counts.get(chapter_key, 0)
                chunk["_merge_protected"] = chapter_position < 5
                chapter_chunk_counts[chapter_key] = chapter_position + 1

            changed = 0
            index = 0

            while index < len(chunks):
                chunk = chunks[index]
                if not self._chunk_in_scope(chunk, chapter):
                    index += 1
                    continue

                if chunk.get("_merge_protected"):
                    index += 1
                    continue

                while self._count_words((chunks[index].get("text") or "").strip()) < min_words:
                    current_chunk = chunks[index]
                    prev_index = index - 1
                    next_index = index + 1

                    can_merge_prev = (
                        prev_index >= 0
                        and self._chunk_in_scope(chunks[prev_index], chapter)
                        and not chunks[prev_index].get("_merge_protected")
                        and self._chunks_share_chapter(chunks[prev_index], current_chunk)
                        and self._speakers_match(chunks[prev_index].get("speaker"), current_chunk.get("speaker"))
                    )
                    can_merge_next = (
                        next_index < len(chunks)
                        and self._chunk_in_scope(chunks[next_index], chapter)
                        and not chunks[next_index].get("_merge_protected")
                        and self._chunks_share_chapter(current_chunk, chunks[next_index])
                        and self._speakers_match(chunks[next_index].get("speaker"), current_chunk.get("speaker"))
                    )

                    if can_merge_prev:
                        chunks[prev_index] = self._merge_adjacent_chunks(chunks[prev_index], current_chunk)
                        del chunks[index]
                        index = prev_index
                        changed += 1
                        continue

                    if can_merge_next:
                        chunks[index] = self._merge_adjacent_chunks(current_chunk, chunks[next_index])
                        del chunks[next_index]
                        changed += 1
                        continue

                    break

                index += 1

            for chunk in chunks:
                chunk.pop("_merge_protected", None)

            for i, chunk in enumerate(chunks):
                chunk["id"] = i

            if changed:
                self._atomic_json_write(chunks, self.chunks_path)

            processed_scope = 0
            for chunk in chunks:
                if self._chunk_in_scope(chunk, chapter):
                    processed_scope += 1

            return {
                "changed": changed,
                "total_chunks": len(chunks),
                "processed_scope": processed_scope,
                "chapter": chapter,
                "min_words": min_words,
            }

    def update_chunk(self, chunk_ref, data):
        # Hold the lock for the entire read-modify-write cycle so editor saves
        # cannot overwrite in-flight generation tokens from audio workers.
        with self._chunks_lock:
            if not os.path.exists(self.chunks_path):
                return None
            with open(self.chunks_path, "r", encoding="utf-8") as f:
                chunks = json.load(f)

            index = self.resolve_chunk_index(chunk_ref, chunks)
            if index is None or not (0 <= index < len(chunks)):
                return None

            chunk = chunks[index]
            # Update fields
            if "text" in data:
                chunk["text"] = data["text"]
            if "instruct" in data:
                chunk["instruct"] = data["instruct"]
            if "speaker" in data:
                chunk["speaker"] = data["speaker"]
            if "silence_duration_s" in data:
                chunk["silence_duration_s"] = float(data["silence_duration_s"])

            # If text/instruct/speaker changed, invalidate the old audio immediately.
            # Silence chunks have no audio to invalidate.
            if chunk.get("type") != "silence" and ("text" in data or "instruct" in data or "speaker" in data):
                chunk["audio_path"] = None
                chunk["status"] = "pending"
                chunk["audio_validation"] = None
                chunk["auto_regen_count"] = 0
                chunk.pop("generation_token", None)
                self._clear_proofread_state(chunk)

            print(f"update_chunk({index}): instruct='{chunk.get('instruct', '')}', speaker='{chunk.get('speaker', '')}'")
            self._ensure_chunk_uids(chunks)
            self._atomic_json_write(chunks, self.chunks_path)
            self.clear_chunk_runtime(chunk.get("uid"))
            return chunk

    def prepare_chunk_for_regeneration(self, chunk_ref):
        with self._chunks_lock:
            if not os.path.exists(self.chunks_path):
                return None
            with open(self.chunks_path, "r", encoding="utf-8") as f:
                chunks = json.load(f)
            index = self.resolve_chunk_index(chunk_ref, chunks)
            if index is None or not (0 <= index < len(chunks)):
                return None

            chunk = chunks[index]
            audio_path = (chunk.get("audio_path") or "").strip()
            if audio_path:
                full_audio_path = os.path.join(self.root_dir, audio_path)
                if os.path.exists(full_audio_path):
                    try:
                        os.remove(full_audio_path)
                    except OSError:
                        pass

            chunk["audio_path"] = None
            chunk["audio_validation"] = None
            chunk["status"] = "pending"
            chunk["auto_regen_count"] = 0
            chunk.pop("generation_token", None)
            self._clear_proofread_state(chunk)
            self._atomic_json_write(chunks, self.chunks_path)
            self.clear_chunk_runtime(chunk.get("uid"))
            return {"index": index, "chunk": chunk}

    def _load_tts_settings(self):
        return self._load_app_config().get("tts", {})

    def _get_auto_regen_retry_attempts(self):
        tts_settings = self._load_tts_settings()
        if not tts_settings.get("auto_regenerate_bad_clips", False):
            return 0
        try:
            attempts = int(tts_settings.get("auto_regenerate_bad_clip_attempts", 3))
        except (TypeError, ValueError):
            return 0
        return attempts if attempts > 0 else 0

    @staticmethod
    def _cleanup_temp_file(temp_path):
        if os.path.exists(temp_path):
            for attempt in range(3):
                try:
                    os.remove(temp_path)
                    break
                except OSError:
                    if attempt < 2:
                        time.sleep(0.1 * (attempt + 1))
                    else:
                        print(f"Warning: Could not delete temp file {temp_path}")

    def _store_generated_audio(self, temp_path, filename_base):
        audio_path = None

        try:
            segment = AudioSegment.from_file(temp_path)
            if len(segment) == 0:
                raise RuntimeError("Generated audio has 0 duration")

            mp3_filename = f"{filename_base}.mp3"
            mp3_filepath = os.path.join(self.voicelines_dir, mp3_filename)
            segment.export(mp3_filepath, format="mp3")

            mp3_size = os.path.getsize(mp3_filepath) if os.path.exists(mp3_filepath) else 0
            if mp3_size < 1024:
                print(f"MP3 export produced invalid file ({mp3_size} bytes) — ffmpeg likely lacks MP3 encoder (libmp3lame). Falling back to WAV.")
                os.remove(mp3_filepath)
                raise RuntimeError("MP3 export produced invalid file")

            audio_path = f"voicelines/{mp3_filename}"

        except Exception as e:
            if "invalid file" not in str(e).lower():
                print(f"MP3 conversion failed (ffmpeg missing?): {e}")
            wav_filename = f"{filename_base}.wav"
            wav_filepath = os.path.join(self.voicelines_dir, wav_filename)
            shutil.copy(temp_path, wav_filepath)
            audio_path = f"voicelines/{wav_filename}"

        return audio_path

    @staticmethod
    def _chunk_audio_filename_base(chunk_uid, index, speaker, attempt=0):
        speaker_slug = sanitize_filename(speaker) or "speaker"
        stable_id = sanitize_filename((chunk_uid or "").strip()) or f"legacy_{index+1:04d}"
        filename_base = f"voiceline_{stable_id}_{speaker_slug}"
        if attempt > 0:
            filename_base = f"{filename_base}_retry{attempt}"
        return filename_base

    def _repair_target_audio_path_locked(self, chunk, index, source_relative_path):
        normalized_source = str(source_relative_path or "").replace("\\", "/").strip("/")
        source_filename = os.path.basename(normalized_source)
        _, ext = os.path.splitext(source_filename)
        ext = ext or ".mp3"

        attempt = 0
        while True:
            filename_base = self._chunk_audio_filename_base(
                chunk.get("uid"),
                index,
                chunk.get("speaker", ""),
                attempt=attempt,
            )
            candidate_relative_path = f"voicelines/{filename_base}{ext}"
            if candidate_relative_path == normalized_source:
                return candidate_relative_path

            candidate_full_path = os.path.join(self.root_dir, candidate_relative_path)
            source_full_path = os.path.join(self.root_dir, normalized_source)
            if not os.path.exists(candidate_full_path):
                os.makedirs(os.path.dirname(candidate_full_path), exist_ok=True)
                os.replace(source_full_path, candidate_full_path)
                return candidate_relative_path

            try:
                if os.path.samefile(source_full_path, candidate_full_path):
                    return candidate_relative_path
            except OSError:
                pass

            attempt += 1

    def _commit_repaired_chunk_locked(self, chunks, index, source_relative_path, validation, write_immediately=True):
        chunk = chunks[index]
        target_audio_path = self._repair_target_audio_path_locked(chunk, index, source_relative_path)
        if target_audio_path and target_audio_path != source_relative_path:
            self._copy_cached_transcription_key(source_relative_path, target_audio_path)
        chunk["audio_path"] = target_audio_path
        chunk["audio_validation"] = validation
        chunk["status"] = "done"
        chunk["auto_regen_count"] = 0
        chunk.pop("generation_token", None)
        if write_immediately:
            self._atomic_json_write(chunks, self.chunks_path)
        return target_audio_path

    # ------------------------------------------------------------------
    # Runtime overlay + postprocess workers
    # ------------------------------------------------------------------

    def _runtime_chunk_state(self, chunk):
        runtime = self._copy_chunk_runtime().get(chunk.get("uid"))
        return self._merge_runtime_chunk(dict(chunk), runtime)

    def _chunk_token_matches_live(self, chunk, expected_token=None):
        if expected_token is None:
            return True
        live_chunk = self._runtime_chunk_state(chunk)
        return live_chunk.get("generation_token") == expected_token

    def _postprocess_worker_loop(self):
        while True:
            task = self._postprocess_queue.get()
            try:
                self._process_postprocess_task(task)
            except BaseException as e:
                fut = task.get("future")
                if fut is not None and not fut.done():
                    fut.set_exception(e)
            finally:
                self._postprocess_queue.task_done()

    def _apply_runtime_patch_to_chunk(self, chunk, runtime_state):
        patched = dict(chunk)
        for field in ("status", "audio_path", "audio_validation", "auto_regen_count"):
            if field in runtime_state:
                patched[field] = copy.deepcopy(runtime_state[field])
        if "audio_path" in runtime_state:
            self._clear_proofread_state(patched)
        patched.pop("generation_token", None)
        return patched

    def flush_dirty_chunks(self, force=False):
        chunks_dir = os.path.dirname(self.chunks_path)
        if not os.path.isdir(self.root_dir) or (chunks_dir and not os.path.isdir(chunks_dir)):
            with self._chunk_runtime_lock:
                self._dirty_chunk_uids.clear()
            return 0
        with self._chunks_flush_lock:
            with self._chunk_runtime_lock:
                if not self._dirty_chunk_uids:
                    return 0
                pending_uids = list(self._dirty_chunk_uids)
                runtime_snapshot = {
                    uid: copy.deepcopy(self._chunk_runtime.get(uid, {}))
                    for uid in pending_uids
                    if uid in self._chunk_runtime
                }

            if not runtime_snapshot:
                return 0

            chunks = self.load_chunks_raw()
            dirty_versions = {
                uid: runtime_snapshot[uid].get("updated_at")
                for uid in runtime_snapshot
            }
            patched_any = False
            for chunk in chunks:
                uid = chunk.get("uid")
                runtime_state = runtime_snapshot.get(uid)
                if not runtime_state:
                    continue
                if runtime_state.get("status") == "generating" and not force:
                    continue
                patched = self._apply_runtime_patch_to_chunk(chunk, runtime_state)
                chunk.clear()
                chunk.update(patched)
                patched_any = True

            if not patched_any:
                return 0

            self._atomic_json_write_raw(chunks, self.chunks_path)
            self._set_chunks_snapshot(chunks)
            self._atomic_json_write_raw(chunks, self.chunks_latest_backup_path)
            current_audio_count = self._count_audio_linked_chunks(chunks)
            best_audio_count = self._load_chunk_backup_audio_count(self.chunks_best_backup_path)
            if current_audio_count > best_audio_count:
                self._atomic_json_write_raw(chunks, self.chunks_best_backup_path)

            flushed_count = len(runtime_snapshot)
            with self._chunk_runtime_lock:
                for uid, version in dirty_versions.items():
                    runtime_chunk = self._chunk_runtime.get(uid)
                    if runtime_chunk is None:
                        self._dirty_chunk_uids.discard(uid)
                        continue
                    if runtime_chunk.get("updated_at") == version and runtime_chunk.get("status") != "generating":
                        self._dirty_chunk_uids.discard(uid)
            return flushed_count

    def _process_postprocess_task(self, task):
        index = task["index"]
        temp_path = task["temp_path"]
        generation_token = task["generation_token"]
        fut = task["future"]

        try:
            result = self._finalize_generated_audio(
                index,
                task["speaker"],
                task["text"],
                temp_path,
                attempt=task["attempt"],
                chunk_uid=task["chunk_uid"],
            )

            if result["status"] == "error" and task.get("error_status_override"):
                updated_status = task["error_status_override"]
                keep_token = task.get("keep_token", False)
            else:
                updated_status = result["status"]
                keep_token = False

            updated_chunk = self._update_chunk_fields_if_token(
                index,
                generation_token,
                audio_path=result["audio_path"],
                audio_validation=result["audio_validation"],
                status=updated_status,
                auto_regen_count=task["attempt"],
                generation_token=generation_token if keep_token else None,
            )

            self._cleanup_temp_file(temp_path)

            if updated_chunk is None:
                fut.set_result({"status": "cancelled", "audio_path": None, "audio_validation": None, "error": "cancelled"})
                return

            fut.set_result({
                **result,
                "uid": task["chunk_uid"],
                "index": index,
                "generation_token": generation_token,
            })
        except Exception as e:
            try:
                self._update_chunk_fields_if_token(
                    index,
                    generation_token,
                    status="error",
                    audio_validation=None,
                    auto_regen_count=task["attempt"],
                    generation_token=None,
                )
            except Exception:
                pass
            self._cleanup_temp_file(temp_path)
            fut.set_exception(e)

    def _enqueue_postprocess(self, index, speaker, text, temp_path, attempt,
                             chunk_uid, generation_token,
                             error_status_override=None, keep_token=False):
        fut = concurrent.futures.Future()
        self._postprocess_queue.put({
            "index": index,
            "speaker": speaker,
            "text": text,
            "temp_path": temp_path,
            "attempt": attempt,
            "chunk_uid": chunk_uid,
            "generation_token": generation_token,
            "error_status_override": error_status_override,
            "keep_token": keep_token,
            "future": fut,
        })
        return fut

    def _finalize_generated_audio(self, index, speaker, text, temp_path, attempt=0, chunk_uid=None):
        if not os.path.exists(temp_path) or os.path.getsize(temp_path) == 0:
            return {
                "status": "error",
                "audio_path": None,
                "audio_validation": None,
                "error": "Generated audio file is missing or empty",
            }

        print(f"Generated WAV size: {os.path.getsize(temp_path)} bytes")

        filename_base = self._chunk_audio_filename_base(chunk_uid, index, speaker, attempt=attempt)
        audio_path = self._store_generated_audio(temp_path, filename_base)
        full_audio_path = os.path.join(self.root_dir, audio_path)
        validation = validate_audio_clip(
            text=text,
            actual_duration_sec=get_audio_duration_seconds(temp_path),
            file_size_bytes=os.path.getsize(full_audio_path),
        ).to_dict()

        if validation["is_valid"]:
            return {
                "status": "done",
                "audio_path": audio_path,
                "audio_validation": validation,
                "error": None,
                "proofread_cleared": True,
            }

        print(f"Chunk {index} failed audio sanity check: {validation['error']}")
        return {
            "status": "error",
            "audio_path": audio_path,
            "audio_validation": validation,
            "error": validation["error"],
            "proofread_cleared": True,
        }

    def generate_chunk_audio(self, index, attempt=0, generation_token=None):
        chunks = self.load_chunks()
        if not (0 <= index < len(chunks)):
            return False, "Invalid chunk index"

        chunk = self._claim_chunk_generation(index, generation_token)
        if not chunk:
            return False, "Invalid chunk index"

        try:
            engine = self.get_engine()
            if not engine:
                self._update_chunk_fields(index, status="error", audio_validation=None, auto_regen_count=attempt)
                return False, "TTS engine not initialized"

            speaker = chunk["speaker"]
            # Apply per-chapter narrator override: if this is a NARRATOR line and the chapter
            # has a custom narrator voice selected, substitute that voice for generation.
            speaker = self._apply_narrator_override(
                speaker, (chunk.get("chapter") or "").strip(), self.get_narrator_overrides()
            )
            voice_config = self._load_voice_config()
            resolved_speaker = self.resolve_voice_speaker(speaker, voice_config, chunks=chunks)
            voice_config = self.prepare_runtime_voice_config(voice_config, [resolved_speaker])
            text = chunk["text"]
            transformed_text, _ = apply_dictionary_to_text(text, self.load_dictionary_entries())
            instruct = chunk.get("instruct", "")
            auto_regen_retry_attempts = self._get_auto_regen_retry_attempts()

            print(
                f"Generating chunk {index}: speaker={speaker}, resolved_speaker={resolved_speaker}, "
                f"instruct='{instruct}', text='{transformed_text[:50]}...'"
            )

            # Generate to temp file (unique per chunk for parallel processing)
            temp_path = os.path.join(self.root_dir, f"temp_chunk_{index}.wav")

            success = engine.generate_voice(transformed_text, instruct, resolved_speaker, voice_config, temp_path)

            if generation_token is not None and not self.chunk_has_generation_token(index, generation_token):
                self._cleanup_temp_file(temp_path)
                return False, "Generation abandoned"

            if success:
                fut = self._enqueue_postprocess(
                    index=index,
                    speaker=speaker,
                    text=transformed_text,
                    temp_path=temp_path,
                    attempt=attempt,
                    chunk_uid=chunk.get("uid"),
                    generation_token=generation_token,
                )
                # Block this TTS thread until saveback completes; other TTS
                # threads in the executor can run their GPU work concurrently.
                try:
                    result = fut.result()
                except Exception as e:
                    return False, str(e)

                if result["status"] == "cancelled":
                    return False, "Generation abandoned"

                if result["status"] == "error" and auto_regen_retry_attempts > 0 and attempt < auto_regen_retry_attempts:
                    # Saveback wrote status="error"; _claim_chunk_generation in
                    # the recursive call will re-claim it to "generating".
                    print(f"Chunk {index} failed sanity check; auto-regenerating attempt {attempt + 1}/{auto_regen_retry_attempts}")
                    return self.generate_chunk_audio(index, attempt=attempt + 1, generation_token=generation_token)

                if generation_token is None:
                    self.flush_dirty_chunks(force=True)
                return result["status"] == "done", result["audio_path"] if result["status"] == "done" else result["error"]
            else:
                self._update_chunk_fields_if_token(
                    index,
                    generation_token,
                    status="error",
                    audio_validation=None,
                    auto_regen_count=attempt,
                    generation_token=None,
                )
                if generation_token is None:
                    self.flush_dirty_chunks(force=True)
                self._cleanup_temp_file(temp_path)
                return False, "Generation failed"

        except Exception as e:
            try:
                self._update_chunk_fields_if_token(
                    index,
                    generation_token,
                    status="error",
                    audio_validation=None,
                    auto_regen_count=attempt,
                    generation_token=None,
                )
            except Exception as update_err:
                print(f"Warning: Failed to update chunk {index} status to error: {update_err}")
            self._cleanup_temp_file(os.path.join(self.root_dir, f"temp_chunk_{index}.wav"))
            if generation_token is None:
                self.flush_dirty_chunks(force=True)
            return False, str(e)

    def merge_audio(self, progress_callback=None, log_callback=None, export_config=None, chapter=None):
        merge_started_at = time.time()
        output_filename = "cloned_audiobook.mp3"
        output_path = os.path.join(self.root_dir, output_filename)
        temp_dir = tempfile.mkdtemp(prefix="merge_audio_", dir=self.root_dir)
        timeline = self._collect_merge_timeline(
            progress_callback=progress_callback,
            merge_started_at=merge_started_at,
            export_config=export_config,
            log_callback=log_callback,
            temp_dir=temp_dir,
        )

        if chapter:
            chapter = chapter.strip()
            timeline = [
                item for item in timeline
                if item.get("is_silence_block") or
                   (item.get("chunk") or {}).get("chapter", "").strip() == chapter
            ]

        if not timeline:
            shutil.rmtree(temp_dir, ignore_errors=True)
            return False, "No audio segments found"

        chapter_groups = self._group_timeline_by_chapter(timeline)
        concat_path = os.path.join(temp_dir, "concat.txt")

        try:
            first_audio_path = next(
                (item["full_path"] for item in timeline if not item.get("is_silence_block")), None
            )
            silence_assets = self._create_silence_assets(
                temp_dir,
                export_config,
                reference_audio_path=first_audio_path,
            )

            # Generate silence block WAVs now that reference audio format is known
            _write_silence = silence_assets["_write"]
            for item in timeline:
                if item.get("is_silence_block") and item["full_path"] is None:
                    wav_path = os.path.join(temp_dir, f"silence_block_{item['chunk']['uid']}.wav")
                    _write_silence(wav_path, item["silence_duration_ms"])
                    item["full_path"] = wav_path
                    item["file_size_bytes"] = os.path.getsize(wav_path)

            estimated_size_bytes = 0
            previous_item = None
            total_same = 0
            total_diff = 0
            total_para = 0
            total_chapter_end = 0

            with open(concat_path, "w", encoding="utf-8") as concat_file:
                for chapter_index, (chapter_label, chapter_items) in enumerate(chapter_groups, start=1):
                    if previous_item is not None and chapter_items:
                        first_item = chapter_items[0]
                        if not previous_item.get("is_silence_block") and not first_item.get("is_silence_block"):
                            pause_path, pause_size = self._pick_silence(
                                previous_item, first_item, silence_assets, is_chapter_boundary=True
                            )
                            self._write_concat_line(concat_file, pause_path)
                            estimated_size_bytes += pause_size
                            total_chapter_end += 1

                    chapter_same = 0
                    chapter_diff = 0
                    chapter_para = 0
                    prev_item_in_chapter = None
                    for item in chapter_items:
                        if prev_item_in_chapter is not None:
                            if not item.get("is_silence_block") and not prev_item_in_chapter.get("is_silence_block"):
                                pause_path, pause_size = self._pick_silence(prev_item_in_chapter, item, silence_assets)
                                self._write_concat_line(concat_file, pause_path)
                                estimated_size_bytes += pause_size
                                # Tally silence type for diagnostics
                                prev_pid = prev_item_in_chapter["chunk"].get("paragraph_id")
                                curr_pid = item["chunk"].get("paragraph_id")
                                if prev_pid and curr_pid and prev_pid != curr_pid:
                                    chapter_para += 1
                                elif prev_item_in_chapter["chunk"].get("speaker") == item["chunk"].get("speaker"):
                                    chapter_same += 1
                                else:
                                    chapter_diff += 1
                        self._write_concat_line(concat_file, item["full_path"])
                        estimated_size_bytes += item["file_size_bytes"]
                        prev_item_in_chapter = item

                    previous_item = prev_item_in_chapter if prev_item_in_chapter is not None else previous_item
                    total_same += chapter_same
                    total_diff += chapter_diff
                    total_para += chapter_para
                    if log_callback:
                        log_callback(
                            f"  Chapter '{chapter_label}': {chapter_same} same-speaker, "
                            f"{chapter_diff} speaker-change, {chapter_para} paragraph silences"
                        )
                    self._emit_merge_progress(
                        progress_callback,
                        merge_started_at,
                        stage="assembling",
                        chapter_index=chapter_index,
                        total_chapters=len(chapter_groups),
                        chapter_label=chapter_label,
                        estimated_size_bytes=estimated_size_bytes,
                        output_path=output_path,
                    )

            if log_callback:
                log_callback(
                    f"Export totals: {total_chapter_end} chapter-end, {total_same} same-speaker, "
                    f"{total_diff} speaker-change, {total_para} paragraph silences"
                )

            self._emit_merge_progress(
                progress_callback,
                merge_started_at,
                stage="exporting",
                chapter_index=len(chapter_groups),
                total_chapters=len(chapter_groups),
                chapter_label=chapter_groups[-1][0],
                estimated_size_bytes=estimated_size_bytes,
                output_path=output_path,
            )

            success, export_result = self._export_concat_mp3(
                concat_path,
                output_path,
                progress_tick=lambda: self._emit_merge_progress(
                    progress_callback,
                    merge_started_at,
                    stage="exporting",
                    chapter_index=len(chapter_groups),
                    total_chapters=len(chapter_groups),
                    chapter_label=chapter_groups[-1][0],
                    estimated_size_bytes=estimated_size_bytes,
                    output_path=output_path,
                ),
                log_callback=log_callback,
            )
            if not success:
                return False, f"ffmpeg merge failed: {export_result}"

            output_path = export_result
            output_filename = os.path.basename(output_path)

            self._emit_merge_progress(
                progress_callback,
                merge_started_at,
                stage="normalizing",
                chapter_index=len(chapter_groups),
                total_chapters=len(chapter_groups),
                chapter_label=chapter_groups[-1][0],
                estimated_size_bytes=estimated_size_bytes,
                output_path=output_path,
            )
            normalized, normalize_result = self._call_normalize_audio_file(
                output_path,
                export_config=export_config,
                progress_callback=lambda info: self._emit_merge_progress(
                    progress_callback,
                    merge_started_at,
                    stage="normalizing",
                    chapter_index=len(chapter_groups),
                    total_chapters=len(chapter_groups),
                    chapter_label=(
                        f"{chapter_groups[-1][0]} "
                        f"({info.get('phase', 'normalizing')}, {int(round(info.get('overall_percent', 0.0)))}%)"
                    ),
                    estimated_size_bytes=estimated_size_bytes,
                    output_path=output_path,
                ),
                log_callback=log_callback,
            )
            if not normalized:
                return False, f"Audio normalization failed: {normalize_result}"

            self._emit_merge_progress(
                progress_callback,
                merge_started_at,
                stage="complete",
                chapter_index=len(chapter_groups),
                total_chapters=len(chapter_groups),
                chapter_label=chapter_groups[-1][0],
                estimated_size_bytes=estimated_size_bytes,
                output_path=output_path,
            )
            return True, output_filename
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def export_optimized_mp3_zip(self, progress_callback=None, log_callback=None, export_config=None, max_part_seconds=7200):
        merge_started_at = time.time()
        zip_path = os.path.join(self.root_dir, "optimized_audiobook.zip")
        temp_dir = tempfile.mkdtemp(prefix="optimized_export_", dir=self.root_dir)
        timeline = self._collect_merge_timeline(
            progress_callback=progress_callback,
            merge_started_at=merge_started_at,
            export_config=export_config,
            log_callback=log_callback,
            temp_dir=temp_dir,
        )
        if not timeline:
            shutil.rmtree(temp_dir, ignore_errors=True)
            return False, "No audio segments found"

        chapter_groups = self._group_timeline_by_chapter(timeline)
        chapter_end_ms = getattr(export_config, "silence_end_of_chapter_ms", 3000)

        try:
            first_audio_path = next(
                (item["full_path"] for item in timeline if not item.get("is_silence_block")), None
            )
            silence_assets = self._create_silence_assets(
                temp_dir,
                export_config,
                reference_audio_path=first_audio_path,
            )

            # Generate silence block WAVs now that reference audio format is known
            _write_silence = silence_assets["_write"]
            for item in timeline:
                if item.get("is_silence_block") and item["full_path"] is None:
                    wav_path = os.path.join(temp_dir, f"silence_block_{item['chunk']['uid']}.wav")
                    _write_silence(wav_path, item["silence_duration_ms"])
                    item["full_path"] = wav_path
                    item["file_size_bytes"] = os.path.getsize(wav_path)

            chapter_exports = []
            estimated_size_bytes = 0
            total_same = 0
            total_diff = 0
            total_para = 0

            for chapter_index, (chapter_label, chapter_items) in enumerate(chapter_groups, start=1):
                concat_path = os.path.join(temp_dir, f"chapter_{chapter_index:03d}.txt")
                chapter_output_path = os.path.join(temp_dir, f"chapter_{chapter_index:03d}.mp3")
                chapter_estimated_size = 0
                chapter_same = 0
                chapter_diff = 0
                chapter_para = 0
                prev_item_in_chapter = None

                with open(concat_path, "w", encoding="utf-8") as concat_file:
                    for item in chapter_items:
                        if prev_item_in_chapter is not None:
                            if not item.get("is_silence_block") and not prev_item_in_chapter.get("is_silence_block"):
                                pause_path, pause_size = self._pick_silence(prev_item_in_chapter, item, silence_assets)
                                self._write_concat_line(concat_file, pause_path)
                                chapter_estimated_size += pause_size
                                prev_pid = prev_item_in_chapter["chunk"].get("paragraph_id")
                                curr_pid = item["chunk"].get("paragraph_id")
                                if prev_pid and curr_pid and prev_pid != curr_pid:
                                    chapter_para += 1
                                elif prev_item_in_chapter["chunk"].get("speaker") == item["chunk"].get("speaker"):
                                    chapter_same += 1
                                else:
                                    chapter_diff += 1
                        self._write_concat_line(concat_file, item["full_path"])
                        chapter_estimated_size += item["file_size_bytes"]
                        prev_item_in_chapter = item

                total_same += chapter_same
                total_diff += chapter_diff
                total_para += chapter_para
                if log_callback:
                    log_callback(
                        f"  Chapter '{chapter_label}': {chapter_same} same-speaker, "
                        f"{chapter_diff} speaker-change, {chapter_para} paragraph silences"
                    )

                self._emit_merge_progress(
                    progress_callback,
                    merge_started_at,
                    stage="assembling",
                    chapter_index=chapter_index,
                    total_chapters=len(chapter_groups),
                    chapter_label=chapter_label,
                    estimated_size_bytes=estimated_size_bytes + chapter_estimated_size,
                    output_path=chapter_output_path,
                )

                success, export_result = self._export_concat_mp3(
                    concat_path,
                    chapter_output_path,
                    log_callback=log_callback,
                )
                if not success:
                    return False, f"Failed to export chapter {chapter_label}: {export_result}"

                chapter_output_path = export_result
                chapter_duration_seconds = AudioSegment.from_file(chapter_output_path).duration_seconds
                chapter_size_bytes = os.path.getsize(chapter_output_path)
                estimated_size_bytes += chapter_size_bytes
                chapter_exports.append({
                    "label": chapter_label,
                    "path": chapter_output_path,
                    "duration_seconds": chapter_duration_seconds,
                    "file_size_bytes": chapter_size_bytes,
                    "first_item": chapter_items[0],
                    "last_item": chapter_items[-1],
                    "first_speaker": chapter_items[0]["chunk"]["speaker"],
                    "last_speaker": chapter_items[-1]["chunk"]["speaker"],
                })

            if log_callback:
                total_chapter_end = max(0, len(chapter_exports) - 1)
                log_callback(
                    f"Export totals: {total_chapter_end} chapter-end, {total_same} same-speaker, "
                    f"{total_diff} speaker-change, {total_para} paragraph silences"
                )

            part_groups = []
            current_group = []
            current_duration = 0.0
            for chapter in chapter_exports:
                chapter_pause_seconds = 0.0
                if current_group:
                    chapter_pause_seconds = chapter_end_ms / 1000.0
                proposed_duration = current_duration + chapter_pause_seconds + chapter["duration_seconds"]
                if current_group and proposed_duration > max_part_seconds:
                    part_groups.append(current_group)
                    current_group = [chapter]
                    current_duration = chapter["duration_seconds"]
                else:
                    current_group.append(chapter)
                    current_duration = proposed_duration if len(current_group) > 1 else chapter["duration_seconds"]
            if current_group:
                part_groups.append(current_group)

            part_paths = []
            for part_index, part_group in enumerate(part_groups, start=1):
                part_basename = self._optimized_export_part_basename(part_index)
                part_output_path = os.path.join(temp_dir, part_basename)
                concat_path = os.path.join(temp_dir, f"part_{part_index:03d}.txt")
                part_estimated_size = 0

                with open(concat_path, "w", encoding="utf-8") as concat_file:
                    for chapter_offset, chapter in enumerate(part_group):
                        if chapter_offset > 0:
                            self._write_concat_line(concat_file, silence_assets["chapter_end_path"])
                            part_estimated_size += silence_assets["chapter_end_size_bytes"]
                        self._write_concat_line(concat_file, chapter["path"])
                        part_estimated_size += chapter["file_size_bytes"]

                self._emit_merge_progress(
                    progress_callback,
                    merge_started_at,
                    stage="packing",
                    chapter_index=part_index,
                    total_chapters=len(part_groups),
                    chapter_label=", ".join(chapter["label"] for chapter in part_group[:2]) + ("..." if len(part_group) > 2 else ""),
                    estimated_size_bytes=part_estimated_size,
                    output_path=part_output_path,
                )

                success, export_result = self._export_concat_mp3(
                    concat_path,
                    part_output_path,
                    progress_tick=lambda current_path=part_output_path, current_size=part_estimated_size, current_index=part_index: self._emit_merge_progress(
                        progress_callback,
                        merge_started_at,
                        stage="packing",
                        chapter_index=current_index,
                        total_chapters=len(part_groups),
                        chapter_label=f"Part {current_index} of {len(part_groups)}",
                        estimated_size_bytes=current_size,
                        output_path=current_path,
                    ),
                    log_callback=log_callback,
                )
                if not success:
                    return False, f"Failed to export optimized part {part_index}: {export_result}"

                part_output_path = export_result
                self._emit_merge_progress(
                    progress_callback,
                    merge_started_at,
                    stage="normalizing",
                    chapter_index=part_index,
                    total_chapters=len(part_groups),
                    chapter_label=f"Part {part_index} of {len(part_groups)}",
                    estimated_size_bytes=part_estimated_size,
                    output_path=part_output_path,
                )
                normalized, normalize_result = self._call_normalize_audio_file(
                    part_output_path,
                    export_config=export_config,
                    progress_callback=lambda info, current_path=part_output_path, current_size=part_estimated_size, current_index=part_index: self._emit_merge_progress(
                        progress_callback,
                        merge_started_at,
                        stage="normalizing",
                        chapter_index=current_index,
                        total_chapters=len(part_groups),
                        chapter_label=(
                            f"Part {current_index} "
                            f"({info.get('phase', 'normalizing')}, {int(round(info.get('overall_percent', 0.0)))}%)"
                        ),
                        estimated_size_bytes=current_size,
                        output_path=current_path,
                    ),
                    log_callback=log_callback,
                )
                if not normalized:
                    return False, f"Failed to normalize optimized part {part_index}: {normalize_result}"
                part_paths.append((part_output_path, os.path.basename(part_output_path)))

            self._emit_merge_progress(
                progress_callback,
                merge_started_at,
                stage="bundling",
                chapter_index=len(part_paths),
                total_chapters=len(part_paths),
                chapter_label="Writing optimized export zip",
                estimated_size_bytes=sum(os.path.getsize(path) for path, _ in part_paths),
                output_path=zip_path,
            )

            with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
                for part_path, part_basename in part_paths:
                    zf.write(part_path, arcname=part_basename)

            self._emit_merge_progress(
                progress_callback,
                merge_started_at,
                stage="complete",
                chapter_index=len(part_paths),
                total_chapters=len(part_paths),
                chapter_label=f"Created {len(part_paths)} optimized part{'s' if len(part_paths) != 1 else ''}",
                estimated_size_bytes=sum(os.path.getsize(path) for path, _ in part_paths),
                output_path=zip_path,
            )

            return True, os.path.basename(zip_path)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def export_audacity(self, export_config=None):
        """Export project as an Audacity-compatible zip with per-speaker WAV tracks,
        a LOF file for auto-import, and a labels file for chunk annotations."""
        timeline_items = self._collect_merge_timeline(export_config=export_config)

        # Phase 1 — Compute timeline (matching merge_audio pause logic exactly)
        timeline = []  # list of (chunk, segment, abs_start_ms)
        prev_speaker = None
        cursor_ms = 0

        prev_item = None
        for item in timeline_items:
            chunk = item["chunk"]
            if item.get("is_silence_block"):
                segment = AudioSegment.silent(duration=item["silence_duration_ms"])
                # No auto-silence before/after explicit silence blocks
            else:
                try:
                    segment = AudioSegment.from_file(item["full_path"])
                except Exception as e:
                    print(f"Error loading audio for Audacity export {item['full_path']}: {e}")
                    prev_item = item
                    continue
                if prev_item is not None and not prev_item.get("is_silence_block"):
                    speaker = chunk.get("speaker", "")
                    prev_speaker = prev_item["chunk"].get("speaker", "")
                    if speaker == prev_speaker:
                        cursor_ms += SAME_SPEAKER_PAUSE_MS
                    else:
                        cursor_ms += DEFAULT_PAUSE_MS

            timeline.append((chunk, segment, cursor_ms))
            cursor_ms += len(segment)
            prev_item = item

        if not timeline:
            return False, "No audio segments found"

        total_duration_ms = cursor_ms

        # Phase 2 — Build per-speaker WAV tracks
        speakers_ordered = []
        seen = set()
        for chunk, segment, start_ms in timeline:
            speaker = chunk.get("speaker")
            if speaker and speaker not in seen:
                speakers_ordered.append(speaker)
                seen.add(speaker)

        speaker_tracks = {}
        for speaker in speakers_ordered:
            track_cursor = 0
            track = AudioSegment.empty()

            for chunk, segment, start_ms in timeline:
                if chunk.get("speaker") != speaker:
                    continue
                # Insert silence gap from current track position to this chunk's start
                gap = start_ms - track_cursor
                if gap > 0:
                    track += AudioSegment.silent(duration=gap)
                track += segment
                track_cursor = start_ms + len(segment)

            # Pad to total duration so all tracks are equal length
            remaining = total_duration_ms - track_cursor
            if remaining > 0:
                track += AudioSegment.silent(duration=remaining)

            speaker_tracks[speaker] = track

        # Phase 3 — Build LOF and labels content
        lof_lines = []
        for speaker in speakers_ordered:
            safe_name = sanitize_filename(speaker)
            lof_lines.append(f'file "{safe_name}.wav"')
        lof_content = "\n".join(lof_lines) + "\n"

        label_lines = []
        for chunk, segment, start_ms in timeline:
            start_sec = start_ms / 1000.0
            end_sec = (start_ms + len(segment)) / 1000.0
            text_preview = chunk.get("text", "")[:80]
            label = f"[{chunk['speaker']}] {text_preview}"
            label_lines.append(f"{start_sec:.6f}\t{end_sec:.6f}\t{label}")
        labels_content = "\n".join(label_lines) + "\n"

        # Phase 4 — Zip everything
        zip_path = os.path.join(self.root_dir, "audacity_export.zip")
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("project.lof", lof_content)
            zf.writestr("labels.txt", labels_content)

            for speaker in speakers_ordered:
                safe_name = sanitize_filename(speaker)
                wav_buffer = io.BytesIO()
                speaker_tracks[speaker].export(wav_buffer, format="wav")
                zf.writestr(f"{safe_name}.wav", wav_buffer.getvalue())

        return True, zip_path

    def merge_m4b(self, per_chunk_chapters=False, metadata=None, export_config=None):
        """Merge audio chunks into an M4B audiobook with chapter markers.

        Args:
            per_chunk_chapters: If True, each chunk is a chapter. If False,
                detect chapter headings and group chunks into sections.
            metadata: Optional dict with keys: title, author, narrator, year,
                description, cover_path (absolute path to cover image).
            export_config: Optional ExportConfig with silence durations.

        Returns:
            tuple: (success: bool, message: str)
        """
        metadata = metadata or {}
        same_speaker_ms = getattr(export_config, "silence_same_speaker_ms", SAME_SPEAKER_PAUSE_MS)
        between_speakers_ms = getattr(export_config, "silence_between_speakers_ms", DEFAULT_PAUSE_MS)
        paragraph_ms = getattr(export_config, "silence_paragraph_ms", 750)
        timeline_items = self._collect_merge_timeline(export_config=export_config)

        # Phase 1 — Compute timeline (same logic as export_audacity)
        timeline = []  # list of (chunk, segment, abs_start_ms)
        prev_chunk = None
        cursor_ms = 0

        prev_item = None
        for item in timeline_items:
            chunk = item["chunk"]
            if item.get("is_silence_block"):
                segment = AudioSegment.silent(duration=item["silence_duration_ms"])
                # No auto-silence before/after explicit silence blocks
            else:
                try:
                    segment = AudioSegment.from_file(item["full_path"])
                except Exception as e:
                    print(f"Error loading audio for M4B export {item['full_path']}: {e}")
                    prev_item = item
                    continue
                if prev_item is not None and not prev_item.get("is_silence_block"):
                    prev_chunk = prev_item["chunk"]
                    prev_pid = prev_chunk.get("paragraph_id")
                    curr_pid = chunk.get("paragraph_id")
                    if prev_pid and curr_pid and prev_pid != curr_pid:
                        cursor_ms += paragraph_ms
                    elif chunk.get("speaker") == prev_chunk.get("speaker"):
                        cursor_ms += same_speaker_ms
                    else:
                        cursor_ms += between_speakers_ms

            timeline.append((chunk, segment, cursor_ms))
            cursor_ms += len(segment)
            prev_item = item

        if not timeline:
            return False, "No audio segments found"

        # Phase 2 — Build chapters
        chapters = self._build_m4b_chapters(timeline, per_chunk_chapters)
        print(f"  M4B: {len(chapters)} chapters")

        # Phase 3 — Combine audio and export to temp WAV
        # Silence blocks are already AudioSegment objects; treat them as their own "speaker"
        audio_segments = [seg for _, seg, _ in timeline]
        speakers = [chunk.get("speaker", "") for chunk, _, _ in timeline]
        final_audio = combine_audio_with_pauses(
            audio_segments, speakers,
            pause_ms=between_speakers_ms,
            same_speaker_pause_ms=same_speaker_ms,
        )

        temp_wav = os.path.join(self.root_dir, "temp_m4b_combined.wav")
        meta_path = os.path.join(self.root_dir, "temp_m4b_meta.txt")
        output_path = os.path.join(self.root_dir, "audiobook.m4b")

        try:
            final_audio.export(temp_wav, format="wav")
            normalized, normalize_result = self._call_normalize_audio_file(temp_wav, export_config=export_config)
            if not normalized:
                return False, f"Audio normalization failed: {normalize_result}"

            # Phase 4 — Write FFmpeg metadata file with book metadata
            meta_lines = [";FFMETADATA1"]
            meta_lines.append(f"title={self._escape_ffmeta(metadata.get('title') or 'Audiobook')}")
            meta_lines.append(f"artist={self._escape_ffmeta(metadata.get('author') or '')}")
            meta_lines.append(f"album_artist={self._escape_ffmeta(metadata.get('narrator') or '')}")
            meta_lines.append(f"date={self._escape_ffmeta(metadata.get('year') or '')}")
            meta_lines.append(f"comment={self._escape_ffmeta(metadata.get('description') or '')}")
            meta_lines.append("genre=Audiobook")
            meta_lines.append("")
            for title, start_ms, end_ms in chapters:
                safe_title = self._escape_ffmeta(title)
                meta_lines.append("[CHAPTER]")
                meta_lines.append("TIMEBASE=1/1000")
                meta_lines.append(f"START={start_ms}")
                meta_lines.append(f"END={end_ms}")
                meta_lines.append(f"title={safe_title}")
                meta_lines.append("")

            with open(meta_path, "w", encoding="utf-8") as f:
                f.write("\n".join(meta_lines))

            # Phase 5 — FFmpeg: WAV + chapters → M4B (AAC)
            cover_path = metadata.get("cover_path") or ""
            has_cover = cover_path and os.path.exists(cover_path)

            cmd = [get_ffmpeg_exe(), "-y", "-i", temp_wav]
            if has_cover:
                cmd += ["-i", cover_path]
            cmd += ["-i", meta_path, "-map_metadata", "2" if has_cover else "1"]
            # Map audio stream
            cmd += ["-map", "0:a"]
            if has_cover:
                # Map cover as attached picture
                cmd += ["-map", "1:v", "-c:v", "copy", "-disposition:v:0", "attached_pic"]
            cmd += [
                "-c:a", "aac",
                "-b:a", "128k",
                "-movflags", "+faststart",
                output_path
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=600)
            if result.returncode != 0:
                print(f"FFmpeg stderr: {result.stderr[-500:]}")
                return False, f"FFmpeg failed (exit {result.returncode})"

        finally:
            for tmp in [temp_wav, meta_path]:
                if os.path.exists(tmp):
                    try:
                        os.remove(tmp)
                    except OSError:
                        pass

        return True, "audiobook.m4b"

    @staticmethod
    def _escape_ffmeta(text):
        """Escape special characters for FFmpeg metadata format."""
        text = text.replace("\\", "\\\\")
        text = text.replace("=", "\\=")
        text = text.replace(";", "\\;")
        text = text.replace("#", "\\#")
        text = text.replace("\n", " ")
        return text

    # Regex for detecting chapter/section headings in chunk text
    _HEADING_RE = re.compile(
        r'^(chapter|part|book|volume|prologue|epilogue|introduction|conclusion|act|section)\b',
        re.IGNORECASE
    )

    def _build_m4b_chapters(self, timeline, per_chunk_chapters):
        """Build chapter list from timeline entries.

        Returns:
            list of (title, start_ms, end_ms) tuples
        """
        if per_chunk_chapters:
            chapters = []
            for chunk, segment, start_ms in timeline:
                end_ms = start_ms + len(segment)
                text_preview = chunk.get("text", "")[:80]
                title = f"[{chunk['speaker']}] {text_preview}"
                chapters.append((title, start_ms, end_ms))
            return chapters

        # Smart grouping: detect chapter headings
        heading_indices = []
        for i, (chunk, segment, start_ms) in enumerate(timeline):
            text = chunk.get("text", "").strip()
            # Short structural text (likely a heading) or starts with heading keyword
            if self._HEADING_RE.match(text):
                heading_indices.append(i)
            elif len(text) < 80 and '"' not in text and text and self._HEADING_RE.search(text):
                heading_indices.append(i)

        # If no headings detected, fall back to per-chunk
        if not heading_indices:
            print("  M4B: No chapter headings detected, falling back to per-chunk chapters")
            return self._build_m4b_chapters(timeline, per_chunk_chapters=True)

        chapters = []

        # Pre-heading chunks → "Introduction"
        if heading_indices[0] > 0:
            start_ms = timeline[0][2]
            last_before = heading_indices[0] - 1
            end_ms = timeline[last_before][2] + len(timeline[last_before][1])
            chapters.append(("Introduction", start_ms, end_ms))

        # Each heading starts a chapter that runs until the next heading
        for idx, head_i in enumerate(heading_indices):
            title = timeline[head_i][0].get("text", "").strip()
            # Truncate long titles
            if len(title) > 120:
                title = title[:117] + "..."

            start_ms = timeline[head_i][2]

            # End = start of next heading, or end of last chunk
            if idx + 1 < len(heading_indices):
                next_head_i = heading_indices[idx + 1]
                last_in_group = next_head_i - 1
            else:
                last_in_group = len(timeline) - 1

            end_ms = timeline[last_in_group][2] + len(timeline[last_in_group][1])
            chapters.append((title, start_ms, end_ms))

        return chapters

    def generate_chunks_parallel(self, indices, max_workers=2, progress_callback=None,
                                  cancel_check=None, item_callback=None, generation_token=None):
        """Generate multiple chunks in parallel using ThreadPoolExecutor.

        Uses individual TTS API calls with per-speaker voice settings.

        Args:
            indices: List of chunk indices to generate
            max_workers: Number of concurrent TTS workers
            progress_callback: Optional callback(completed, failed, total) for progress updates
            cancel_check: Optional callable returning True when cancellation is requested
            item_callback: Optional callback(index, success, elapsed_seconds, input_words, output_words)

        Returns:
            dict with 'completed', 'failed', and 'cancelled' keys
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        results = {"completed": [], "failed": [], "cancelled": 0}

        # Filter out empty-text chunks
        chunks = self.load_chunks()
        if chunks:
            indices = [i for i in indices if 0 <= i < len(chunks) and chunks[i].get("text", "").strip()]

        total = len(indices)

        if total == 0:
            return results

        print(f"Starting parallel generation of {total} chunks with {max_workers} workers...")
        word_counts = {
            idx: len(re.findall(r"\b\w+\b", chunks[idx].get("text", "")))
            for idx in indices if 0 <= idx < len(chunks)
        }

        def _timed_generate(idx):
            start = time.time()
            try:
                success, msg = self.generate_chunk_audio(idx, generation_token=generation_token)
                return success, msg, time.time() - start
            except Exception as e:
                return False, str(e), time.time() - start

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(_timed_generate, idx): idx
                for idx in indices
            }

            cancelled = False
            for future in as_completed(futures):
                if cancel_check and cancel_check():
                    cancelled = True
                    print("[CANCEL] Cancellation requested — stopping parallel generation")
                    executor.shutdown(wait=False, cancel_futures=True)
                    break

                idx = futures[future]
                success, msg, elapsed_seconds = future.result()
                if success:
                    results["completed"].append(idx)
                    print(f"Chunk {idx} completed: {msg}")
                    if item_callback:
                        item_callback(idx, True, elapsed_seconds, word_counts.get(idx, 0), word_counts.get(idx, 0))
                else:
                    results["failed"].append((idx, msg))
                    print(f"Chunk {idx} failed: {msg}")
                    if item_callback:
                        item_callback(idx, False, elapsed_seconds, word_counts.get(idx, 0), 0)
                if progress_callback:
                    progress_callback(len(results["completed"]), len(results["failed"]), total)

            # Reset remaining "generating" chunks to "pending"
            if cancelled:
                done_indices = set(results["completed"]) | {idx for idx, _ in results["failed"]}
                live_chunks = self.load_chunks_view()
                reset_indices = [
                    idx for idx in indices
                    if idx not in done_indices and 0 <= idx < len(live_chunks) and live_chunks[idx].get("status") == "generating"
                ]
                if reset_indices:
                    results["cancelled"] += self.reset_generating_chunks(
                        reset_indices,
                        generation_token=generation_token,
                    )

        self.flush_dirty_chunks(force=True)
        print(f"Parallel generation complete: {len(results['completed'])} succeeded, "
              f"{len(results['failed'])} failed, {results['cancelled']} cancelled")
        return results

    def _group_indices_by_voice_type(self, indices, chunks, voice_config):
        """Reorder indices so chunks with the same voice type are contiguous.

        Grouping key matches how tts.py routes batches:
        - "custom" for custom voices (all batched together)
        - "clone:{speaker}" for clone voices (batched per speaker)
        - "lora:{adapter}" for LoRA voices (batched per adapter)
        - "design" for voice design (always sequential)

        Within each group, original order is preserved.
        """
        from collections import OrderedDict
        groups = OrderedDict()
        labels = {}

        for idx in indices:
            if not (0 <= idx < len(chunks)):
                groups.setdefault("custom", []).append(idx)
                continue

            speaker = chunks[idx].get("speaker", "")
            speaker = self.resolve_voice_speaker(speaker, voice_config, chunks=chunks)
            voice_data = voice_config.get(speaker, {})
            voice_type = voice_data.get("type", "custom")

            if voice_type == "clone":
                key = f"clone:{speaker}"
            elif voice_type in ("lora", "builtin_lora"):
                adapter_id = voice_data.get("adapter_id", "")
                key = f"lora:{adapter_id}"
            elif voice_type == "design":
                key = "design"
            else:
                key = "custom"

            groups.setdefault(key, []).append(idx)

        reordered = []
        for key, group_indices in groups.items():
            print(f"  Voice group '{key}': {len(group_indices)} chunks")
            reordered.extend(group_indices)

        return reordered

    def group_indices_by_resolved_speaker(self, indices, chunks=None, voice_config=None):
        """Reorder indices so each resolved speaker is generated contiguously.

        This is primarily useful for external TTS backends where clone prompt
        reuse is much faster when all lines for the same character are rendered
        back-to-back.
        """
        from collections import OrderedDict

        chunks = chunks if chunks is not None else self.load_chunks()
        voice_config = voice_config if voice_config is not None else self._load_voice_config()
        groups = OrderedDict()
        labels = {}

        for idx in indices:
            if not (0 <= idx < len(chunks)):
                groups.setdefault("", []).append(idx)
                continue

            speaker = chunks[idx].get("speaker", "")
            resolved = self.resolve_voice_speaker(speaker, voice_config, chunks=chunks)
            group_key = self._normalize_speaker_name(resolved) or resolved
            groups.setdefault(group_key, []).append(idx)
            labels.setdefault(group_key, resolved)

        reordered = []
        for key, group_indices in groups.items():
            label = labels.get(key) or key or "<unknown>"
            print(f"  Speaker group '{label}': {len(group_indices)} chunks")
            reordered.extend(group_indices)

        return reordered

    def generate_chunks_batch(self, indices, batch_seed=-1, batch_size=4, progress_callback=None,
                               batch_group_by_type=False, cancel_check=None, item_callback=None, generation_token=None):
        """Generate multiple chunks using batch TTS API with a single seed.

        Args:
            indices: List of chunk indices to generate
            batch_seed: Single seed for all generations (-1 for random)
            batch_size: Number of chunks per batch request
            progress_callback: Optional callback(completed, failed, total) for progress updates
            batch_group_by_type: Group indices by voice type before batching for
                GPU efficiency. When False, indices are batched in sequential order.
            cancel_check: Optional callable returning True when cancellation is requested
            item_callback: Optional callback(index, success, elapsed_seconds, input_words, output_words)

        Returns:
            dict with 'completed', 'failed', and 'cancelled' keys
        """
        results = {"completed": [], "failed": [], "cancelled": 0}

        # Load chunks and voice config
        chunks = self.load_chunks()

        # Filter out empty-text chunks
        if chunks:
            indices = [i for i in indices if 0 <= i < len(chunks) and chunks[i].get("text", "").strip()]

        total = len(indices)

        if total == 0:
            return results

        print(f"Starting batch generation of {total} chunks (batch_size={batch_size}, seed={batch_seed}, "
              f"group_by_type={batch_group_by_type})...")
        word_counts = {
            idx: len(re.findall(r"\b\w+\b", chunks[idx].get("text", "")))
            for idx in indices if 0 <= idx < len(chunks)
        }
        voice_config = {}
        voice_config = self._load_voice_config()
        narrator_overrides = self.get_narrator_overrides()
        resolved_speakers = {
            self.resolve_voice_speaker(
                self._apply_narrator_override(
                    chunks[idx].get("speaker", ""),
                    (chunks[idx].get("chapter") or "").strip(),
                    narrator_overrides,
                ),
                voice_config, chunks=chunks,
            )
            for idx in indices if 0 <= idx < len(chunks)
        }
        voice_config = self.prepare_runtime_voice_config(voice_config, resolved_speakers)
        dictionary_entries = self.load_dictionary_entries()
        auto_regen_retry_attempts = self._get_auto_regen_retry_attempts()

        # Get TTS engine
        engine = self.get_engine()
        if not engine:
            for idx in indices:
                results["failed"].append((idx, "TTS engine not initialized"))
            return results

        self._claim_chunks_generation(indices, generation_token)

        # Optionally reorder indices so same voice-type chunks are contiguous.
        # This produces larger homogeneous batches (e.g. all custom voices
        # together) instead of fragmenting each batch across voice types.
        if batch_group_by_type:
            indices = self._group_indices_by_voice_type(indices, chunks, voice_config)

        # Split indices into batches
        batches = [indices[i:i + batch_size] for i in range(0, len(indices), batch_size)]
        print(f"Processing {len(batches)} batches...")

        cancelled = False
        for batch_num, batch_indices in enumerate(batches):
            if cancel_check and cancel_check():
                cancelled = True
                print(f"[CANCEL] Cancellation requested before batch {batch_num + 1}")
                break

            print(f"Batch {batch_num + 1}/{len(batches)}: {len(batch_indices)} chunks")

            # Build batch request data
            batch_chunks = []
            transformed_texts = {}
            for idx in batch_indices:
                if 0 <= idx < len(chunks):
                    chunk = chunks[idx]
                    transformed_text, _ = apply_dictionary_to_text(chunk.get("text", ""), dictionary_entries)
                    transformed_texts[idx] = transformed_text
                    effective_speaker = self._apply_narrator_override(
                        chunk.get("speaker", ""),
                        (chunk.get("chapter") or "").strip(),
                        narrator_overrides,
                    )
                    resolved_speaker = self.resolve_voice_speaker(effective_speaker, voice_config, chunks=chunks)
                    batch_chunks.append({
                        "index": idx,
                        "text": transformed_text,
                        "instruct": chunk.get("instruct", ""),
                        "speaker": resolved_speaker
                    })

            # Call batch TTS with single seed
            batch_start = time.time()
            batch_results = engine.generate_batch(
                batch_chunks,
                voice_config,
                self.root_dir,
                batch_seed,
                cancel_check=cancel_check,
            )

            # Process completed chunks - convert to MP3 and update status
            chunks = self.load_chunks()  # Reload for each batch
            if cancel_check and cancel_check():
                cancelled = True

            processed_in_batch = len(batch_results["completed"]) + len(batch_results["failed"])
            shared_elapsed = (time.time() - batch_start) / processed_in_batch if processed_in_batch > 0 else 0.0

            postprocess_futures = {}
            for idx in batch_results["completed"]:
                if cancel_check and cancel_check():
                    cancelled = True
                    temp_path = os.path.join(self.root_dir, f"temp_batch_{idx}.wav")
                    self._cleanup_temp_file(temp_path)
                    continue
                if not (0 <= idx < len(chunks)):
                    print(f"Chunk {idx} skipped: index out of range (chunks changed during generation?)")
                    results["failed"].append((idx, "Index out of range after reload"))
                    continue

                temp_path = os.path.join(self.root_dir, f"temp_batch_{idx}.wav")

                if not os.path.exists(temp_path):
                    results["failed"].append((idx, "Temp audio file not found"))
                    self._update_chunk_fields_if_token(
                        idx,
                        generation_token,
                        status="error",
                        audio_validation=None,
                        generation_token=None,
                    )
                    continue

                try:
                    if generation_token is not None and not self.chunk_has_generation_token(idx, generation_token):
                        self._cleanup_temp_file(temp_path)
                        continue
                    chunk = chunks[idx]
                    speaker = chunk.get("speaker", "unknown")
                    will_retry_on_error = auto_regen_retry_attempts > 0

                    fut = self._enqueue_postprocess(
                        index=idx,
                        speaker=speaker,
                        text=transformed_texts.get(idx, chunk.get("text", "")),
                        temp_path=temp_path,
                        attempt=0,
                        chunk_uid=chunk.get("uid"),
                        generation_token=generation_token,
                        error_status_override="pending" if will_retry_on_error else None,
                        keep_token=will_retry_on_error,
                    )
                    postprocess_futures[fut] = (idx, will_retry_on_error)
                except Exception as e:
                    print(f"Error queueing postprocess for chunk {idx}: {e}")
                    results["failed"].append((idx, str(e)))
                    self._update_chunk_fields_if_token(
                        idx,
                        generation_token,
                        status="error",
                        audio_validation=None,
                        generation_token=None,
                    )
                    self._cleanup_temp_file(temp_path)
                    if item_callback:
                        item_callback(idx, False, shared_elapsed, word_counts.get(idx, 0), 0)

            for fut in concurrent.futures.as_completed(list(postprocess_futures)):
                idx, will_retry_on_error = postprocess_futures[fut]
                try:
                    result = fut.result()
                    if result["status"] == "cancelled":
                        continue

                    chunks = self.load_chunks_view()

                    if result["status"] == "done":
                        results["completed"].append(idx)
                        print(f"Chunk {idx} completed: {chunks[idx].get('audio_path')}")
                        if item_callback:
                            item_callback(idx, True, shared_elapsed, word_counts.get(idx, 0), word_counts.get(idx, 0))
                    elif will_retry_on_error and not (cancel_check and cancel_check()):
                        print(f"Chunk {idx} failed validation in batch; retrying immediately at the front of the queue")
                        retry_start = time.time()
                        retry_success, retry_msg = self.generate_chunk_audio(idx, attempt=1, generation_token=generation_token)
                        retry_elapsed = time.time() - retry_start
                        chunks = self.load_chunks()
                        if retry_success:
                            results["completed"].append(idx)
                            print(f"Chunk {idx} completed after auto-regeneration: {retry_msg}")
                            if item_callback:
                                item_callback(idx, True, shared_elapsed + retry_elapsed, word_counts.get(idx, 0), word_counts.get(idx, 0))
                        else:
                            results["failed"].append((idx, retry_msg))
                            print(f"Chunk {idx} failed after auto-regeneration: {retry_msg}")
                            if item_callback:
                                item_callback(idx, False, shared_elapsed + retry_elapsed, word_counts.get(idx, 0), 0)
                    else:
                        results["failed"].append((idx, result["error"]))
                        print(f"Chunk {idx} failed validation: {result['error']}")
                        if item_callback:
                            item_callback(idx, False, shared_elapsed, word_counts.get(idx, 0), 0)
                except Exception as e:
                    print(f"Error processing chunk {idx}: {e}")
                    results["failed"].append((idx, str(e)))
                    self._update_chunk_fields_if_token(
                        idx,
                        generation_token,
                        status="error",
                        audio_validation=None,
                        generation_token=None,
                    )
                    self._cleanup_temp_file(temp_path)
                    if item_callback:
                        item_callback(idx, False, shared_elapsed, word_counts.get(idx, 0), 0)

            for idx, error in batch_results["failed"]:
                if 0 <= idx < len(chunks):
                    self._update_chunk_fields_if_token(
                        idx,
                        generation_token,
                        status="error",
                        audio_validation=None,
                        generation_token=None,
                    )
                results["failed"].append((idx, error))
                if item_callback:
                    item_callback(idx, False, shared_elapsed, word_counts.get(idx, 0), 0)

            chunks = self.load_chunks()
            if cancel_check and cancel_check():
                cancelled = True

            if progress_callback:
                progress_callback(len(results["completed"]), len(results["failed"]), total)

            if cancelled:
                print(f"[CANCEL] Stopping after batch {batch_num + 1}")
                break

        # Reset remaining live "generating" chunks to "pending" on cancel or completion.
        done_indices = set(results["completed"]) | {idx for idx, _ in results["failed"]}
        live_chunks = self.load_chunks_view()
        reset_indices = [
            idx for idx in indices
            if idx not in done_indices and 0 <= idx < len(live_chunks) and live_chunks[idx].get("status") == "generating"
        ]
        if reset_indices:
            results["cancelled"] += self.reset_generating_chunks(reset_indices, generation_token=generation_token)

        self.flush_dirty_chunks(force=True)

        print(f"Batch generation complete: {len(results['completed'])} succeeded, "
              f"{len(results['failed'])} failed, {results['cancelled']} cancelled")
        return results
