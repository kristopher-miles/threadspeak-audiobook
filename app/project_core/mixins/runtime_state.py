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
from script_store import apply_dictionary_to_text
from source_document import load_source_document, iter_document_paragraphs
from audio_perf import record_audio_perf
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
                "finalizer_workers": _coerce_positive_int(tts_settings.get("finalizer_workers", 1), 1),
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
            chunk = self.get_chunk_raw(index)
            if chunk is None:
                return None
            claimed = self.claim_generation(chunk.get("uid"), generation_token)
            return claimed if claimed is not None else None

        def _claim_chunks_generation(self, indices, generation_token=None):
            uids = []
            for chunk_ref in indices or []:
                chunk = self.get_chunk_raw(chunk_ref)
                if chunk is None:
                    continue
                uid = str(chunk.get("uid") or "").strip()
                if uid:
                    uids.append(uid)
            claimed = self.claim_generation_many(uids, generation_token)
            return len(claimed or [])

        def _chunk_token_matches(self, chunks, index, generation_token=None):
            if generation_token is None:
                return True
            if not (0 <= index < len(chunks)):
                return False
            return chunks[index].get("generation_token") == generation_token

        def chunk_has_generation_token(self, index, generation_token=None):
            chunk = self.get_chunk_raw(index)
            if chunk is None:
                return False
            if generation_token is None:
                return True
            return chunk.get("generation_token") == generation_token

        def _update_chunk_fields_if_token(self, index, expected_token=None, **fields):
            chunk = self.get_chunk_raw(index)
            if chunk is None:
                return None
            uid = chunk.get("uid")
            update_fields = dict(fields)
            clear_fields = []
            if update_fields.get("status") != "generating" and "generation_token" not in update_fields:
                clear_fields.append("generation_token")
            expected = {}
            if expected_token is not None:
                expected["generation_token"] = expected_token
            updated = self.patch_chunk_if(
                uid,
                expected=expected,
                fields=update_fields,
                clear_fields=clear_fields,
                reason="_update_chunk_fields_if_token",
            )
            return updated

        def force_reset_chunks_to_pending(self, indices):
            """Force any chunk in `indices` to pending status regardless of current state.

            Clears audio_path, audio_validation, generation_token and all proofread
            state so the UI immediately reflects the reset.  Called before a
            Regenerate-All job is enqueued so the user gets instant feedback.
            """
            reset_count = 0
            for chunk_ref in indices or []:
                chunk = self.get_chunk_raw(chunk_ref)
                if chunk is None:
                    continue
                updated = self.patch_chunk_if(
                    chunk.get("uid"),
                    fields={
                        "status": "pending",
                        "audio_path": None,
                        "audio_validation": None,
                        "auto_regen_count": 0,
                    },
                    clear_fields=["generation_token", "proofread"],
                    reason="force_reset_chunks_to_pending",
                )
                if updated is not None:
                    self.clear_chunk_runtime(chunk.get("uid"))
                    reset_count += 1
            return reset_count

        def reset_generating_chunks(self, indices=None, generation_token=None, target_status="pending"):
            if indices is None:
                targets = [
                    chunk.get("uid")
                    for chunk in self.resolve_generation_targets(scope_mode="project", chapter=None, pending_only=False)
                    if chunk.get("status") == "generating"
                ]
            else:
                targets = []
                for chunk_ref in indices:
                    chunk = self.get_chunk_raw(chunk_ref)
                    if chunk is None:
                        continue
                    targets.append(chunk.get("uid"))
            updated = self.reset_generation_rows(
                targets,
                token=generation_token,
                target_status=target_status,
                reason="reset_generating_chunks",
            )
            for chunk in updated or []:
                self.clear_chunk_runtime(chunk.get("uid"))
            return len(updated or [])

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

        def register_audio_finalization_listener(
            self,
            generation_token,
            *,
            submission_callback=None,
            item_callback=None,
            activity_callback=None,
        ):
            token = str(generation_token or "").strip()
            if not token:
                return
            with self._audio_finalize_listener_lock:
                self._audio_finalize_listeners[token] = {
                    "submission_callback": submission_callback,
                    "item_callback": item_callback,
                    "activity_callback": activity_callback,
                }

        def unregister_audio_finalization_listener(self, generation_token):
            token = str(generation_token or "").strip()
            if not token:
                return
            with self._audio_finalize_listener_lock:
                self._audio_finalize_listeners.pop(token, None)

        def _notify_audio_finalize_listener(self, generation_token, callback_name, *args):
            token = str(generation_token or "").strip()
            if not token:
                return
            with self._audio_finalize_listener_lock:
                listener = dict(self._audio_finalize_listeners.get(token) or {})
            callback = listener.get(callback_name)
            if callback is None:
                return
            try:
                callback(*args)
            except Exception as exc:
                print(f"Warning: audio finalize listener '{callback_name}' failed for {token}: {exc}")

        def _audio_finalize_task_local_id(self, task):
            local_id = str((task or {}).get("local_id") or "").strip()
            if local_id:
                return local_id
            task_id = int((task or {}).get("id") or 0)
            if task_id > 0:
                return f"persisted-{task_id}"
            return uuid.uuid4().hex

        def _audio_finalize_task_match(self, task, generation_token=None, uids=None, statuses=None):
            normalized_token = str(generation_token or "").strip()
            normalized_uids = {str(uid).strip() for uid in (uids or []) if str(uid).strip()}
            normalized_statuses = {str(status).strip() for status in (statuses or []) if str(status).strip()}

            if normalized_token and str((task or {}).get("generation_token") or "").strip() != normalized_token:
                return False
            if normalized_uids and str((task or {}).get("chunk_uid") or "").strip() not in normalized_uids:
                return False
            if normalized_statuses and str((task or {}).get("status") or "").strip() not in normalized_statuses:
                return False
            return True

        def _register_audio_finalize_task(self, task):
            normalized = dict(task or {})
            normalized["local_id"] = self._audio_finalize_task_local_id(normalized)
            normalized.setdefault("status", "queued")
            normalized.setdefault("created_at", time.time())
            normalized.setdefault("updated_at", normalized["created_at"])
            normalized.setdefault("last_error", None)
            normalized.setdefault("cancelled", False)
            with self._audio_finalize_tasks_lock:
                self._audio_finalize_tasks[normalized["local_id"]] = normalized
            return normalized

        def _get_audio_finalize_task(self, local_id):
            normalized_local_id = str(local_id or "").strip()
            if not normalized_local_id:
                return None
            with self._audio_finalize_tasks_lock:
                return self._audio_finalize_tasks.get(normalized_local_id)

        def _drop_audio_finalize_task(self, local_id):
            normalized_local_id = str(local_id or "").strip()
            if not normalized_local_id:
                return None
            with self._audio_finalize_tasks_lock:
                return self._audio_finalize_tasks.pop(normalized_local_id, None)

        def _update_audio_finalize_task_state(self, local_id, **fields):
            task = self._get_audio_finalize_task(local_id)
            if task is None:
                return None
            for key, value in fields.items():
                task[key] = value
            task["updated_at"] = time.time()
            return task

        def _snapshot_live_audio_finalize_tasks(self, generation_token=None, uids=None, statuses=None):
            with self._audio_finalize_tasks_lock:
                tasks = [
                    dict(task)
                    for task in self._audio_finalize_tasks.values()
                    if not bool(task.get("cancelled"))
                ]
            return [
                task
                for task in tasks
                if self._audio_finalize_task_match(
                    task,
                    generation_token=generation_token,
                    uids=uids,
                    statuses=statuses,
                )
            ]

        def _ensure_audio_finalize_persist_executor(self):
            executor = getattr(self, "_audio_finalize_persist_executor", None)
            if executor is not None:
                return executor
            executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix=f"audio-finalize-ledger-{os.path.basename(self.root_dir)}",
            )
            self._audio_finalize_persist_executor = executor
            return executor

        def _on_audio_finalize_task_persisted(self, local_id, future):
            try:
                persisted = future.result()
            except Exception as exc:
                task = self._update_audio_finalize_task_state(
                    local_id,
                    persistence_error=str(exc),
                    last_error=str(exc),
                )
                if task is not None:
                    print(f"Warning: audio finalize ledger persist failed for {task.get('chunk_uid')}: {exc}")
                return

            if persisted is None:
                return

            task = self._update_audio_finalize_task_state(
                local_id,
                id=int(persisted.get("id") or 0),
                created_at=persisted.get("created_at", time.time()),
                last_error=persisted.get("last_error"),
            )
            if task is None or not bool(task.get("cancelled")):
                return

            try:
                self.script_store.clear_audio_finalize_tasks(
                    generation_token=persisted.get("generation_token"),
                    uids=[persisted.get("chunk_uid")],
                    wait=True,
                )
            except Exception as exc:
                print(f"Warning: failed clearing cancelled audio finalize ledger entry {persisted.get('id')}: {exc}")

        def _persist_audio_finalize_task_async(self, task):
            task_payload = {
                "chunk_uid": task.get("chunk_uid"),
                "generation_token": task.get("generation_token"),
                "temp_wav_path": task.get("temp_wav_path"),
                "attempt": int(task.get("attempt") or 0),
                "speaker": task.get("speaker"),
                "text": task.get("text") or "",
            }
            future = self._ensure_audio_finalize_persist_executor().submit(
                self.script_store.enqueue_audio_finalize_task,
                task_payload,
            )
            future.add_done_callback(
                lambda completed_future, local_id=task.get("local_id"): self._on_audio_finalize_task_persisted(
                    local_id,
                    completed_future,
                )
            )
            task["persistence_future"] = future
            return future

        def _await_audio_finalize_task_persisted(self, task):
            if task is None:
                return None
            future = task.get("persistence_future")
            if future is None:
                return task
            persisted = future.result()
            task.pop("persistence_future", None)
            if persisted is not None:
                task["id"] = int(persisted.get("id") or 0)
                task["created_at"] = persisted.get("created_at", task.get("created_at", time.time()))
                task["last_error"] = persisted.get("last_error")
                task["updated_at"] = time.time()
            return task

        def _complete_audio_finalize_task_ledger(self, task):
            persisted_task = self._await_audio_finalize_task_persisted(task)
            task_id = int((persisted_task or {}).get("id") or 0)
            if task_id > 0:
                self.script_store.complete_audio_finalize_task(task_id, wait=True)
            return persisted_task

        def _fail_audio_finalize_task_ledger(self, task, error=None, requeue=False):
            persisted_task = self._await_audio_finalize_task_persisted(task)
            task_id = int((persisted_task or {}).get("id") or 0)
            if task_id > 0:
                self.script_store.fail_audio_finalize_task(
                    task_id,
                    error=error,
                    requeue=requeue,
                    wait=True,
                )
            return persisted_task

        def _restore_audio_finalize_tasks_from_store(self):
            if getattr(self, "script_store", None) is None:
                return 0
            restored = 0
            try:
                tasks = self.script_store.list_audio_finalize_tasks(statuses=("queued", "processing"))
            except Exception as exc:
                print(f"Warning: failed restoring audio finalize tasks from store: {exc}")
                return 0
            for persisted in tasks or []:
                local_id = f"persisted-{int((persisted or {}).get('id') or 0)}"
                if self._get_audio_finalize_task(local_id) is not None:
                    continue
                task = self._register_audio_finalize_task({
                    **dict(persisted or {}),
                    "local_id": local_id,
                    "status": "queued",
                })
                self._audio_finalize_queue.put(task)
                restored += 1
            return restored

        def _audio_finalize_worker_loop(self):
            while True:
                task = self._audio_finalize_queue.get()
                if not task:
                    continue
                local_id = self._audio_finalize_task_local_id(task)
                live_task = self._update_audio_finalize_task_state(local_id, status="processing")
                task = live_task or task
                try:
                    if bool(task.get("cancelled")):
                        self._cleanup_temp_file(
                            os.path.join(self.root_dir, str(task.get("temp_wav_path") or ""))
                            if not os.path.isabs(str(task.get("temp_wav_path") or ""))
                            else str(task.get("temp_wav_path") or "")
                        )
                        self._complete_audio_finalize_task_ledger(task)
                        continue
                    self._process_audio_finalize_task(task)
                except BaseException as e:
                    try:
                        self._fail_audio_finalize_task_ledger(task, error=str(e), requeue=False)
                    except Exception:
                        pass
                    print(f"Warning: audio finalizer failed for task {task.get('id')}: {e}")
                finally:
                    self._drop_audio_finalize_task(local_id)
                    self._audio_finalize_queue.task_done()

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
                            # they are durably reflected in the persisted chunk store. Removing
                            # them here prevents stale runtime state from masking
                            # later chunk-state edits such as narrator invalidation.
                            self._chunk_runtime.pop(uid, None)
                            self._dirty_chunk_uids.discard(uid)
                return flushed_count

        def _finalize_retry_requested(self, attempt):
            return attempt < self._get_auto_regen_retry_attempts()

        def _spool_audio_relative_path(self, chunk_uid, generation_token, attempt=0):
            token_slug = sanitize_filename(generation_token or "manual") or "manual"
            chunk_slug = sanitize_filename(chunk_uid or uuid.uuid4().hex) or uuid.uuid4().hex
            return os.path.join(
                "voicelines",
                ".finalize_spool",
                token_slug,
                f"{chunk_slug}_attempt{int(attempt or 0)}.wav",
            )

        def _spool_audio_full_path(self, chunk_uid, generation_token, attempt=0):
            relative_path = self._spool_audio_relative_path(chunk_uid, generation_token, attempt=attempt)
            full_path = os.path.join(self.root_dir, relative_path)
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            return full_path

        def _enqueue_audio_finalize_task(self, chunk_uid, generation_token, temp_path, *, attempt=0, speaker="", text=""):
            enqueue_started = time.perf_counter()
            normalized_uid = str(chunk_uid or "").strip()
            if not normalized_uid:
                return None
            normalized_path = str(temp_path or "").strip()
            if not normalized_path:
                return None
            if generation_token is not None and not self.chunk_has_generation_token(normalized_uid, generation_token):
                self._cleanup_temp_file(normalized_path)
                return None
            relative_path = os.path.relpath(normalized_path, self.root_dir)
            self.set_chunk_runtime(
                normalized_uid,
                status="finalizing",
                generation_token=generation_token,
                auto_regen_count=int(attempt or 0),
            )
            task = self._register_audio_finalize_task({
                "chunk_uid": normalized_uid,
                "generation_token": generation_token,
                "temp_wav_path": relative_path,
                "attempt": int(attempt or 0),
                "speaker": speaker,
                "text": text,
                "status": "queued",
            })
            self._persist_audio_finalize_task_async(task)
            self._audio_finalize_queue.put(task)
            print(f"Chunk {normalized_uid} submitted for finalization", flush=True)
            listener_started = time.perf_counter()
            self._notify_audio_finalize_listener(
                generation_token,
                "submission_callback",
                normalized_uid,
                task,
            )
            record_audio_perf(
                "audio_finalize_enqueue",
                generation_token=generation_token,
                uid=normalized_uid,
                queue_ms=round((listener_started - enqueue_started) * 1000.0, 3),
                listener_ms=round((time.perf_counter() - listener_started) * 1000.0, 3),
                total_ms=round((time.perf_counter() - enqueue_started) * 1000.0, 3),
            )
            return task

        def list_audio_finalize_tasks(self, generation_token=None, statuses=None):
            persisted = self.script_store.list_audio_finalize_tasks(
                generation_token=generation_token,
                statuses=statuses,
            )
            live = self._snapshot_live_audio_finalize_tasks(
                generation_token=generation_token,
                statuses=statuses,
            )

            merged = {}
            ordered_keys = []
            for task in persisted or []:
                key = ("id", int((task or {}).get("id") or 0))
                merged[key] = dict(task or {})
                ordered_keys.append(key)
            for task in live:
                task_id = int(task.get("id") or 0)
                if task_id > 0:
                    key = ("id", task_id)
                else:
                    key = (
                        "pending",
                        str(task.get("chunk_uid") or ""),
                        str(task.get("generation_token") or ""),
                        str(task.get("temp_wav_path") or ""),
                        str(task.get("local_id") or ""),
                    )
                if key not in merged:
                    ordered_keys.append(key)
                merged[key] = dict(task)
            return [merged[key] for key in ordered_keys if key in merged]

        def has_pending_audio_finalize_tasks(self, generation_token=None):
            return bool(
                self.list_audio_finalize_tasks(
                    generation_token=generation_token,
                    statuses=("queued", "processing"),
                )
            )

        def clear_audio_finalize_tasks(self, generation_token=None, uids=None, cleanup_files=True):
            tasks = self.list_audio_finalize_tasks(
                generation_token=generation_token,
                statuses=("queued", "processing"),
            )
            if uids:
                normalized_uids = {str(uid).strip() for uid in (uids or []) if str(uid).strip()}
                tasks = [task for task in tasks if str(task.get("chunk_uid") or "").strip() in normalized_uids]
            for task in tasks:
                local_id = self._audio_finalize_task_local_id(task)
                self._update_audio_finalize_task_state(local_id, cancelled=True)
            if cleanup_files:
                for task in tasks:
                    relative_temp_path = str(task.get("temp_wav_path") or "").strip()
                    if not relative_temp_path:
                        continue
                    temp_path = (
                        relative_temp_path
                        if os.path.isabs(relative_temp_path)
                        else os.path.join(self.root_dir, relative_temp_path)
                    )
                    self._cleanup_temp_file(temp_path)
            return self.script_store.clear_audio_finalize_tasks(
                generation_token=generation_token,
                uids=uids,
                wait=True,
            )

        def _process_audio_finalize_task(self, task):
            chunk_uid = str(task.get("chunk_uid") or "").strip()
            generation_token = str(task.get("generation_token") or "").strip() or None
            relative_temp_path = str(task.get("temp_wav_path") or "").strip()
            temp_path = (
                relative_temp_path
                if os.path.isabs(relative_temp_path)
                else os.path.join(self.root_dir, relative_temp_path)
            )
            attempt = int(task.get("attempt") or 0)

            self._notify_audio_finalize_listener(generation_token, "activity_callback", "finalizer_started", chunk_uid)

            chunk = self.get_chunk_raw(chunk_uid)
            if chunk is None:
                self.clear_chunk_runtime(chunk_uid)
                self._cleanup_temp_file(temp_path)
                self._complete_audio_finalize_task_ledger(task)
                return
            if generation_token is not None and chunk.get("generation_token") != generation_token:
                self.clear_chunk_runtime(chunk_uid)
                self._cleanup_temp_file(temp_path)
                self._complete_audio_finalize_task_ledger(task)
                return
            if not os.path.exists(temp_path) or os.path.getsize(temp_path) == 0:
                next_attempt = attempt + 1 if self._finalize_retry_requested(attempt) else attempt
                self._update_chunk_fields_if_token(
                    chunk_uid,
                    generation_token,
                    status="error",
                    audio_validation=None,
                    auto_regen_count=next_attempt,
                    generation_token=None,
                )
                self.clear_chunk_runtime(chunk_uid)
                self._complete_audio_finalize_task_ledger(task)
                self._notify_audio_finalize_listener(
                    generation_token,
                    "item_callback",
                    chunk_uid,
                    False,
                    0.0,
                    len(re.findall(r"\b\w+\b", task.get("text", ""))),
                    0,
                    {
                        "retry_requested": self._finalize_retry_requested(attempt),
                        "attempt": attempt,
                        "error": "Generated audio file is missing or empty",
                    },
                )
                return

            start = time.time()
            result = self._finalize_generated_audio(
                int(chunk.get("id") or 0),
                task.get("speaker") or chunk.get("speaker") or "unknown",
                task.get("text") or chunk.get("text") or "",
                temp_path,
                attempt=attempt,
                chunk_uid=chunk_uid,
            )
            elapsed = time.time() - start
            next_attempt = attempt + 1 if result["status"] == "error" and self._finalize_retry_requested(attempt) else attempt
            updated_chunk = self._update_chunk_fields_if_token(
                chunk_uid,
                generation_token,
                audio_path=result["audio_path"],
                audio_validation=result["audio_validation"],
                status=result["status"],
                auto_regen_count=next_attempt,
                generation_token=None,
            )
            self.clear_chunk_runtime(chunk_uid)
            self._cleanup_temp_file(temp_path)
            self._complete_audio_finalize_task_ledger(task)

            if updated_chunk is None:
                return

            self._notify_audio_finalize_listener(
                generation_token,
                "item_callback",
                chunk_uid,
                result["status"] == "done",
                elapsed,
                len(re.findall(r"\b\w+\b", task.get("text", ""))),
                len(re.findall(r"\b\w+\b", task.get("text", ""))) if result["status"] == "done" else 0,
                {
                    "retry_requested": result["status"] == "error" and self._finalize_retry_requested(attempt),
                    "attempt": attempt,
                    "error": result.get("error"),
                },
            )
