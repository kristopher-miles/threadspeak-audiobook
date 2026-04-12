"""In-memory runtime state and concurrency coordination for chunk processing.

This mixin manages:
- live runtime fields overlaid on chunk snapshots,
- generation token claims/resets for safe concurrent writes, and
- background flush/postprocess workers that persist chunk updates.
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


class ProjectRuntimeStateMixin:
        """Coordinate mutable chunk runtime state across worker threads."""
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

        def _copy_chunks_snapshot(self, chapter=None, chunk_ref=None, index=None):
            with self._chunks_snapshot_lock:
                if self._chunks_snapshot is None:
                    return None
                if index is not None:
                    if 0 <= index < len(self._chunks_snapshot):
                        return copy.deepcopy(self._chunks_snapshot[index])
                    return None
                if chunk_ref is not None:
                    index = self.resolve_chunk_index(chunk_ref, self._chunks_snapshot)
                    if index is None or not (0 <= index < len(self._chunks_snapshot)):
                        return None
                    return copy.deepcopy(self._chunks_snapshot[index])
                normalized_chapter = str(chapter or "").strip()
                if normalized_chapter:
                    return [
                        copy.deepcopy(chunk)
                        for chunk in self._chunks_snapshot
                        if (chunk.get("chapter") or "").strip() == normalized_chapter
                    ]
                return copy.deepcopy(self._chunks_snapshot)

        def _set_chunks_snapshot(self, chunks):
            with self._chunks_snapshot_lock:
                self._chunks_snapshot = copy.deepcopy(chunks)

        def _copy_chunk_runtime(self, uids=None):
            with self._chunk_runtime_lock:
                if uids is None:
                    return copy.deepcopy(self._chunk_runtime)

                subset = {}
                for uid in uids:
                    if not uid:
                        continue
                    runtime_chunk = self._chunk_runtime.get(uid)
                    if runtime_chunk is not None:
                        subset[uid] = copy.deepcopy(runtime_chunk)
                return subset

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
                chunks = self.load_chunks_raw()
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

        def _runtime_chunk_state(self, chunk, runtime_chunk=None):
            runtime = runtime_chunk
            if runtime is None:
                runtime = self._copy_chunk_runtime([chunk.get("uid")]).get(chunk.get("uid"))
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

                flushed_count = len(runtime_snapshot)
                with self._chunk_runtime_lock:
                    for uid, version in dirty_versions.items():
                        runtime_chunk = self._chunk_runtime.get(uid)
                        if runtime_chunk is None:
                            self._dirty_chunk_uids.discard(uid)
                            continue
                        if runtime_chunk.get("updated_at") == version and runtime_chunk.get("status") != "generating":
                            # Non-generating runtime overlays are only needed until
                            # they are durably reflected in chunks.json. Removing
                            # them here prevents stale runtime state from masking
                            # later direct disk edits such as narrator invalidation.
                            self._chunk_runtime.pop(uid, None)
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
