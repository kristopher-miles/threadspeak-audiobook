"""Filesystem-backed config/state read-write helpers.

This mixin centralizes JSON persistence for project-level files such as:
- app/config and per-project voice config,
- workflow state and generation settings, and
- transcription cache and source/paragraph documents.
"""

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
from project_core.constants import *
from project_core.chunking import _coerce_bool, get_speaker, _is_structural_text, _extract_chapter_name, _build_chunk, group_into_chunks, script_entries_to_chunks


class ProjectIOStateMixin:
        """Provide durable JSON I/O operations for project-scoped state."""
        @staticmethod
        def _prune_legacy_voice_state_fields(state):
            payload = dict(state or {})
            for key in ("narrator_threshold", "narrator_overrides", "auto_narrator_aliases"):
                payload.pop(key, None)
            return payload

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
            script_store = getattr(self, "script_store", None)
            if script_store is not None:
                return script_store.load_voice_config()
            raise RuntimeError("Voice config requires the SQLite script store")

        def _save_voice_config(self, voice_config):
            script_store = getattr(self, "script_store", None)
            if script_store is not None:
                rows = [
                    {"speaker": speaker, "config": dict(config or {})}
                    for speaker, config in dict(voice_config or {}).items()
                ]
                script_store.replace_voice_profiles(rows, reason="save_voice_config", wait=True)
                return
            raise RuntimeError("Voice config requires the SQLite script store")

        def load_voice_state_snapshot(self):
            script_store = getattr(self, "script_store", None)
            if script_store is None:
                raise RuntimeError("Voice state requires the SQLite script store")
            return script_store.load_voice_state_snapshot()

        def replace_voice_state_snapshot(self, snapshot, *, reason="replace_voice_state_snapshot"):
            script_store = getattr(self, "script_store", None)
            if script_store is None:
                raise RuntimeError("Voice state requires the SQLite script store")
            return script_store.replace_voice_state_snapshot(snapshot, reason=reason, wait=True)

        def log_voice_audit_event(self, event, **details):
            script_store = getattr(self, "script_store", None)
            if script_store is None or not hasattr(script_store, "_append_voice_audit_entry"):
                return
            script_store._append_voice_audit_entry({"event": event, **dict(details or {})})

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
                    payload = json.load(f)
            except (json.JSONDecodeError, ValueError, OSError):
                return {}
            return self._prune_legacy_voice_state_fields(payload if isinstance(payload, dict) else {})

        def _save_state(self, state):
            state_path = os.path.join(self.root_dir, "state.json")
            with open(state_path, "w", encoding="utf-8") as f:
                json.dump(self._prune_legacy_voice_state_fields(state), f, indent=2, ensure_ascii=False)

        def _load_generation_settings(self):
            return self._load_app_config().get("generation", {})

        def load_source_document(self):
            input_path = (self._load_state().get("input_file_path") or "").strip()
            if not input_path or not os.path.exists(input_path):
                raise ValueError("No uploaded source document found")
            return load_source_document(input_path)

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

        def _load_tts_settings(self):
            return self._load_app_config().get("tts", {})
