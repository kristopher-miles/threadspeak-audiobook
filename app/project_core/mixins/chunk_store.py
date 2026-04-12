"""Chunk persistence, indexing, and script-sync reconciliation utilities.

This mixin handles:
- loading/saving ``chunks.json`` with atomic writes and backups,
- resolving chunk references (uid/id/index) robustly for API/UI calls, and
- reconciling stale chunk state against script updates and on-disk audio.
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
import urllib.parse
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

class ProjectChunkStoreMixin:
        """Provide durable chunk storage and integrity/recovery helpers."""
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
            base_ref = "" if chunk_ref is None else str(chunk_ref).strip()
            if not base_ref:
                return None

            # Accept robustly-normalized ref variants to tolerate UI/route encoding
            # artifacts (URL-encoded values, nested JSON-quoted strings, etc.).
            ref_candidates = [base_ref]
            seen = {base_ref}

            def _push(candidate):
                candidate = "" if candidate is None else str(candidate).strip()
                if not candidate or candidate in seen:
                    return
                seen.add(candidate)
                ref_candidates.append(candidate)

            _push(urllib.parse.unquote(base_ref))

            current = base_ref
            for _ in range(3):
                trimmed = current.strip()
                if len(trimmed) >= 2 and (
                    (trimmed.startswith('"') and trimmed.endswith('"'))
                    or (trimmed.startswith("'") and trimmed.endswith("'"))
                ):
                    trimmed = trimmed[1:-1]
                    _push(trimmed)
                    current = trimmed
                    continue
                try:
                    parsed = json.loads(trimmed)
                except (TypeError, ValueError, json.JSONDecodeError):
                    break
                if isinstance(parsed, (str, int)):
                    _push(parsed)
                    current = str(parsed)
                else:
                    break

            for ref in ref_candidates:
                for index, chunk in enumerate(chunks):
                    if str(chunk.get("uid") or "").strip() == ref:
                        return index

            for ref in ref_candidates:
                try:
                    numeric_ref = int(ref)
                except (TypeError, ValueError):
                    continue

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

        @staticmethod
        def _chunk_has_generated_work(chunk):
            if not isinstance(chunk, dict):
                return False
            if str(chunk.get("audio_path") or "").strip():
                return True
            if chunk.get("status") in {"done", "generating", "error"}:
                return True
            if chunk.get("audio_validation"):
                return True
            if int(chunk.get("auto_regen_count") or 0) > 0:
                return True
            return False

        @classmethod
        def _script_sync_match_key(cls, chunk):
            if not isinstance(chunk, dict):
                return None
            paragraph_id = str(chunk.get("paragraph_id") or "").strip()
            chapter = (chunk.get("chapter") or "").strip()
            speaker = cls._normalize_speaker_name(chunk.get("speaker"))
            text = str(chunk.get("text") or "").strip()
            instruct = str(chunk.get("instruct") or "").strip()
            if paragraph_id:
                return ("paragraph", paragraph_id, chapter, speaker, text, instruct)
            if not text:
                return None
            return ("content", chapter, speaker, text, instruct)

        def _preserve_chunk_state_for_script_sync(self, existing_chunks, rebuilt_chunks):
            lookup = defaultdict(list)
            for chunk in existing_chunks or []:
                key = self._script_sync_match_key(chunk)
                if key is not None:
                    lookup[key].append(chunk)

            preserved = []
            preserved_audio = 0
            for index, chunk in enumerate(rebuilt_chunks or []):
                merged = copy.deepcopy(chunk)
                source = None
                key = self._script_sync_match_key(chunk)
                if key is not None:
                    matches = lookup.get(key) or []
                    if matches:
                        source = matches.pop(0)

                if source:
                    merged["uid"] = source.get("uid") or merged.get("uid") or self._new_chunk_uid()
                    for field in (
                        "status",
                        "audio_path",
                        "audio_validation",
                        "auto_regen_count",
                        "proofread",
                        "silence_duration_s",
                        "type",
                    ):
                        if field in source:
                            merged[field] = copy.deepcopy(source[field])
                    if source.get("generation_token"):
                        merged["generation_token"] = source["generation_token"]
                    else:
                        merged.pop("generation_token", None)
                    if str(merged.get("audio_path") or "").strip():
                        preserved_audio += 1
                else:
                    merged["uid"] = merged.get("uid") or self._new_chunk_uid()

                merged["id"] = index
                preserved.append(merged)

            return preserved, preserved_audio

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

            existing_chunks = self.load_chunks_raw()
            if any(self._chunk_has_generated_work(chunk) for chunk in existing_chunks):
                return {
                    "synced": False,
                    "reason": "generated_audio_present",
                    "chunk_count": len(existing_chunks),
                }

            script = load_script_document(self.script_path)["entries"]
            rebuilt_chunks = script_entries_to_chunks(script)
            chunks, preserved_audio = self._preserve_chunk_state_for_script_sync(existing_chunks, rebuilt_chunks)
            for i, chunk in enumerate(chunks):
                chunk["id"] = i
                chunk["uid"] = chunk.get("uid") or self._new_chunk_uid()
                chunk.setdefault("status", "pending")
                chunk.setdefault("audio_path", None)
                chunk.setdefault("audio_validation", None)
                chunk.setdefault("auto_regen_count", 0)

            self.save_chunks(chunks)
            return {
                "synced": True,
                "reason": "script_newer_than_chunks",
                "chunk_count": len(chunks),
                "preserved_audio": preserved_audio,
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
                if self._ensure_chunk_uids(chunks):
                    self._atomic_json_write(chunks, self.chunks_path)

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
