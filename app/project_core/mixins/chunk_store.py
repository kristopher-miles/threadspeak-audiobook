"""Chunk persistence, indexing, and script-sync reconciliation utilities.

This mixin handles:
- loading/saving chunk state through the SQLite script store,
- resolving chunk references (uid/id/index) robustly for API/UI calls, and
- reconciling chunk state against script updates and on-disk audio.
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
)
from chunk_events import chunk_event_broker
from source_document import load_source_document, iter_document_paragraphs
from project_core.constants import *
from project_core.chunking import _coerce_bool, get_speaker, _is_structural_text, _extract_chapter_name, _build_chunk, group_into_chunks, script_entries_to_chunks
from runtime_layout import LAYOUT

class ProjectChunkStoreMixin:
        _CHUNK_BACKUP_LATEST_KEY = "chunk_backup_latest"
        _CHUNK_BACKUP_MOST_AUDIO_KEY = "chunk_backup_most_audio"

        """Provide durable chunk storage and integrity/recovery helpers."""
        @staticmethod
        def _normalize_scope_mode(scope_mode):
            normalized = str(scope_mode or "project").strip().lower()
            return normalized if normalized in {"chapter", "project"} else "project"

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
            chunks = self.script_store.load_chunks()
            if self._ensure_chunk_uids(chunks):
                self.save_chunks(chunks)
            return chunks

        def load_chunks_raw(self, chapter=None):
            with self._chunks_lock:
                full_snapshot = self._load_chunks_from_disk_locked()
                self._set_chunks_snapshot(full_snapshot)
                return self._copy_chunks_snapshot(chapter=chapter) or []

        def load_chunks_view(self, chapter=None):
            if chapter:
                narrator_repair = self.get_chapter_narrator_voice_repair(chapter)
                if narrator_repair is not None:
                    self.ensure_chapter_narrator_voice_can_narrate(
                        chapter,
                        reason="load_chunks_view_chapter_narrator_repair",
                        repair=narrator_repair,
                    )
            chunks = self.load_chunks_raw(chapter=chapter)
            runtime = self._copy_chunk_runtime(chunk.get("uid") for chunk in chunks)
            return [
                self._merge_runtime_chunk(chunk, runtime.get(chunk.get("uid")))
                for chunk in chunks
            ]

        def load_chunks(self, chapter=None):
            return self.load_chunks_raw(chapter=chapter)

        def get_chunk_raw(self, chunk_ref):
            return self.script_store.get_chunk(chunk_ref)

        def get_chunks_by_uids(self, uids):
            return self.script_store.get_chunks_by_uids(uids)

        def get_chapter_chunks(self, chapter):
            return self.script_store.get_chapter_chunks(chapter)

        def get_chapter_list(self):
            chapters = list(self.script_store.get_chapter_list() or [])
            overrides = dict(self.get_narrator_overrides() or {})
            narrator_key = self._normalize_speaker_name("NARRATOR")
            enriched = []
            for chapter in chapters:
                entry = dict(chapter or {})
                chapter_name = str(entry.get("chapter") or "").strip()
                override = str(overrides.get(chapter_name) or "").strip()
                if override and self._normalize_speaker_name(override) != narrator_key:
                    entry["narrator_label"] = override
                else:
                    entry["narrator_label"] = ""
                enriched.append(entry)
            return enriched

        def resolve_generation_targets(self, scope_mode="project", chapter=None, pending_only=True):
            return self.script_store.resolve_generation_targets(
                scope_mode=self._normalize_scope_mode(scope_mode),
                chapter=chapter,
                pending_only=bool(pending_only),
            )

        def _resolve_chunk_audio_full_path(self, audio_path, *, promote_legacy=True):
            normalized = str(audio_path or "").strip().replace("\\", "/")
            if not normalized:
                return None

            full_audio_path = os.path.join(self.root_dir, normalized)
            if os.path.exists(full_audio_path):
                return full_audio_path

            if (
                not promote_legacy
                or not getattr(self, "_using_default_runtime_layout", False)
                or os.path.isabs(normalized)
                or not (
                    normalized == "voicelines"
                    or normalized.startswith("voicelines/")
                )
            ):
                return None

            parts = []
            for part in normalized.split("/"):
                if not part or part == ".":
                    continue
                if part == "..":
                    return None
                parts.append(part)
            if not parts:
                return None

            legacy_audio_path = LAYOUT.legacy_path(*parts)
            if not os.path.isfile(legacy_audio_path):
                return None

            os.makedirs(os.path.dirname(full_audio_path), exist_ok=True)
            shutil.copy2(legacy_audio_path, full_audio_path)
            return full_audio_path if os.path.exists(full_audio_path) else None

        def get_chunk_audio_ref(self, chunk_ref):
            payload = self.script_store.get_chunk_audio_ref(chunk_ref)
            if payload is None:
                return None
            audio_path = str(payload.get("audio_path") or "").strip()
            if audio_path:
                self._resolve_chunk_audio_full_path(audio_path)
            return payload

        def _publish_chunk_upsert(self, chunk_ref):
            chunk = self.script_store.get_chunk(chunk_ref)
            if chunk is None:
                return None
            payload = self.get_chunk_audio_ref(chunk.get("uid")) or {}
            payload.update(chunk)
            chunk_event_broker.publish("chunk_upsert", payload)
            return chunk

        def _publish_chunk_delete(self, uid, chapter=None):
            chunk_event_broker.publish("chunk_delete", {
                "uid": str(uid or "").strip(),
                "chapter": str(chapter or "").strip(),
            })

        def _publish_chapter_deleted(self, chapter):
            chunk_event_broker.publish("chapter_deleted", {
                "chapter": str(chapter or "").strip(),
            })

        def _publish_chapter_list_changed(self):
            chunk_event_broker.publish("chapter_list_changed", {
                "chapters": self.get_chapter_list(),
            })

        def get_chunk_view(self, chunk_ref, chunks=None):
            raw_chunk = None

            if chunks is not None:
                resolved_index = self.resolve_chunk_index(chunk_ref, chunks)
                if resolved_index is not None and 0 <= resolved_index < len(chunks):
                    raw_chunk = copy.deepcopy(chunks[resolved_index])
            else:
                raw_chunk = self.script_store.get_chunk(chunk_ref)

            if raw_chunk is None:
                return None

            uid = raw_chunk.get("uid")
            runtime_chunk = self._copy_chunk_runtime([uid]).get(uid)
            return self._merge_runtime_chunk(raw_chunk, runtime_chunk)

        def get_chunk_view_by_index(self, index):
            raw_chunk = self.script_store.get_chunk(index)
            if raw_chunk is None:
                return None
            uid = raw_chunk.get("uid")
            runtime_chunk = self._copy_chunk_runtime([uid]).get(uid)
            return self._merge_runtime_chunk(raw_chunk, runtime_chunk)

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
            """Script/chunk synchronization is handled transactionally inside SQLite."""
            if not self.load_script_document().get("entries"):
                return {"synced": False, "reason": "no_script"}
            chunks = self.load_chunks()
            if not chunks:
                return {"synced": False, "reason": "no_chunks"}
            return {"synced": False, "reason": "db_transactional"}

        def reconcile_chunk_audio_states(self):
            """Repair stale error states when a stored clip is actually valid.

            This is intended for editor/UI refreshes after interrupted or large
            generations, where a chunk may still be marked as error even though its
            audio file exists and now passes the duration sanity check.
            """
            chunks = self.load_chunks_raw()
            dictionary_entries = self.load_dictionary_entries()
            updates = []

            for chunk in chunks:
                audio_path = chunk.get("audio_path")
                uid = str(chunk.get("uid") or "").strip()
                if not audio_path or not uid:
                    continue

                full_audio_path = self._resolve_chunk_audio_full_path(audio_path)
                if not full_audio_path or not os.path.exists(full_audio_path):
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

                fields = {}
                if validation["is_valid"]:
                    if chunk.get("status") != "done":
                        fields["status"] = "done"
                    if chunk.get("audio_validation") != validation:
                        fields["audio_validation"] = validation
                    if int(chunk.get("auto_regen_count") or 0) != 0:
                        fields["auto_regen_count"] = 0
                elif chunk.get("audio_validation") != validation:
                    fields["audio_validation"] = validation

                if fields:
                    updates.append({
                        "uid": uid,
                        "expected": {"audio_path": audio_path},
                        "fields": fields,
                    })

            if updates:
                self._commit_chunk_updates(updates)
                return self.load_chunks_raw()
            return chunks

        def _validate_chunk_audio(self, chunk, dictionary_entries):
            audio_path = chunk.get("audio_path")
            if not audio_path:
                return None

            return self._validate_audio_path_for_chunk(chunk, audio_path, dictionary_entries)

        def _validate_audio_path_for_chunk(self, chunk, audio_path, dictionary_entries):
            full_audio_path = self._resolve_chunk_audio_full_path(audio_path)
            if not full_audio_path or not os.path.exists(full_audio_path):
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
            chunks = self.load_chunks_raw()
            if not chunks:
                return outcome

            if indices is None:
                index_iter = range(len(chunks))
            else:
                index_iter = [index for index in indices if 0 <= index < len(chunks)]

            dictionary_entries = self.load_dictionary_entries()
            updates = []
            update_kinds = {}

            for index in index_iter:
                chunk = chunks[index]
                uid = str(chunk.get("uid") or "").strip()
                if not uid or chunk.get("status") != "generating":
                    continue
                if generation_token is not None and chunk.get("generation_token") != generation_token:
                    continue

                try:
                    validation = self._validate_chunk_audio(chunk, dictionary_entries)
                except Exception as e:
                    print(f"Warning: failed to validate interrupted chunk {chunk.get('id')}: {e}")
                    validation = None

                expected = {
                    "status": "generating",
                    "audio_path": chunk.get("audio_path"),
                }
                if chunk.get("generation_token") is not None:
                    expected["generation_token"] = chunk.get("generation_token")

                if validation and validation["is_valid"]:
                    fields = {
                        "status": "done",
                        "audio_validation": validation,
                        "auto_regen_count": 0,
                    }
                    update_kinds[uid] = "recovered"
                else:
                    fields = {"status": "pending"}
                    update_kinds[uid] = "reset"

                updates.append({
                    "uid": uid,
                    "expected": expected,
                    "fields": fields,
                    "clear_fields": ["generation_token"],
                })

            if not updates:
                return outcome

            result = self._commit_chunk_updates(updates)
            for uid in result["applied"]:
                if update_kinds.get(uid) == "recovered":
                    outcome["recovered"] += 1
                elif update_kinds.get(uid) == "reset":
                    outcome["reset"] += 1
                self.clear_chunk_runtime(uid)

            return outcome

        def load_script_document(self):
            script_store = getattr(self, "script_store", None)
            if script_store is None:
                raise RuntimeError("Script document requires the SQLite script store")
            return script_store.load_script_document()

        def load_dictionary_entries(self):
            script_store = getattr(self, "script_store", None)
            if script_store is None:
                raise RuntimeError("Dictionary requires the SQLite script store")
            return script_store.load_dictionary_entries()

        @staticmethod
        def _count_audio_linked_chunks(chunks):
            return sum(1 for chunk in (chunks or []) if str((chunk or {}).get("audio_path") or "").strip())

        def _load_chunk_backup_document(self, key):
            script_store = getattr(self, "script_store", None)
            if script_store is None:
                return None
            payload = script_store.load_project_document(key)
            if not isinstance(payload, dict):
                return None
            chunks = payload.get("chunks")
            if not isinstance(chunks, list):
                return None
            return payload

        def _load_chunk_backup_audio_count(self, key):
            payload = self._load_chunk_backup_document(key)
            if not payload:
                return -1
            stored_count = payload.get("audio_linked_count")
            if isinstance(stored_count, int):
                return stored_count
            return self._count_audio_linked_chunks(payload.get("chunks") or [])

        def _update_chunks_backups(self, chunks):
            if not isinstance(chunks, list):
                return
            script_store = getattr(self, "script_store", None)
            if script_store is None:
                return
            latest_payload = {
                "chunks": copy.deepcopy(chunks),
                "audio_linked_count": self._count_audio_linked_chunks(chunks),
                "updated_at": time.time(),
            }
            script_store.replace_project_document(
                self._CHUNK_BACKUP_LATEST_KEY,
                latest_payload,
                reason="chunk_backup_latest",
                wait=True,
            )
            current_audio_count = self._count_audio_linked_chunks(chunks)
            best_audio_count = self._load_chunk_backup_audio_count(self._CHUNK_BACKUP_MOST_AUDIO_KEY)
            if current_audio_count > best_audio_count:
                script_store.replace_project_document(
                    self._CHUNK_BACKUP_MOST_AUDIO_KEY,
                    latest_payload,
                    reason="chunk_backup_most_audio",
                    wait=True,
                )

        def _atomic_json_write_raw(self, data, target_path, max_retries=5):
            """Atomically write JSON data with retry logic for Windows file locking."""
            if os.path.abspath(target_path) == os.path.abspath(self.chunks_path):
                self.script_store.replace_chunks(data, reason="atomic_json_write_raw", wait=True)
                self._set_chunks_snapshot(data)
                self._update_chunks_backups(data)
                return
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
            self._publish_chapter_list_changed()

        def _commit_chunk_updates(self, updates):
            normalized = []
            for update in updates or []:
                uid = str((update or {}).get("uid") or "").strip()
                fields = dict((update or {}).get("fields") or {})
                expected = dict((update or {}).get("expected") or {})
                clear_fields = tuple((update or {}).get("clear_fields") or ())
                if not uid or (not fields and not clear_fields):
                    continue
                normalized.append({
                    "uid": uid,
                    "fields": fields,
                    "expected": expected,
                    "clear_fields": clear_fields,
                })

            if not normalized:
                return {"changed": False, "applied": [], "skipped": []}

            applied = []
            skipped = []
            changed = False

            with self._chunks_lock:
                latest = self._load_chunks_from_disk_locked()
                latest_by_uid = {
                    str((chunk or {}).get("uid") or "").strip(): chunk
                    for chunk in latest
                    if str((chunk or {}).get("uid") or "").strip()
                }

                for update in normalized:
                    uid = update["uid"]
                    chunk = latest_by_uid.get(uid)
                    if chunk is None:
                        skipped.append(uid)
                        continue

                    expected = update["expected"]
                    if any(chunk.get(field) != value for field, value in expected.items()):
                        skipped.append(uid)
                        continue

                    chunk_changed = False
                    for field in update["clear_fields"]:
                        if field in chunk:
                            chunk.pop(field, None)
                            chunk_changed = True

                    for field, value in update["fields"].items():
                        if chunk.get(field) != value:
                            chunk[field] = copy.deepcopy(value)
                            chunk_changed = True

                    if chunk_changed:
                        applied.append(uid)
                        changed = True

                if changed:
                    self._atomic_json_write(latest, self.chunks_path)

            if changed:
                for uid in applied:
                    self._publish_chunk_upsert(uid)
            return {"changed": changed, "applied": applied, "skipped": skipped}

        def patch_chunk_if(self, uid, expected=None, fields=None, clear_fields=(), reason="patch_chunk_if"):
            result = self.script_store.patch_chunk_if(
                uid,
                expected=expected,
                fields=fields,
                clear_fields=clear_fields,
                reason=reason,
                wait=True,
            )
            if result is not None:
                self._publish_chunk_upsert(uid)
            return result

        def patch_chunks_if(self, updates, reason="patch_chunks_if"):
            result = self.script_store.patch_chunks_if(updates, reason=reason, wait=True)
            for chunk in result or []:
                self._publish_chunk_upsert(chunk.get("uid"))
            return result

        def claim_generation(self, uid, token, reason="claim_generation"):
            claimed = self.script_store.claim_generation(uid, token, reason=reason, wait=True)
            if claimed is not None:
                self._publish_chunk_upsert(uid)
            return claimed

        def claim_generation_many(self, uids, token, reason="claim_generation_many"):
            claimed = self.script_store.claim_generation_many(uids, token, reason=reason, wait=True)
            for chunk in claimed or []:
                self._publish_chunk_upsert(chunk.get("uid"))
            return claimed

        def reset_generation_rows(self, uids, token=None, target_status="pending", reason="reset_generation"):
            updated = self.script_store.reset_generation(
                uids,
                token=token,
                target_status=target_status,
                reason=reason,
                wait=True,
            )
            for chunk in updated or []:
                self._publish_chunk_upsert(chunk.get("uid"))
            return updated

        def prepare_chunk_for_regeneration_by_uid(self, uid):
            updated = self.script_store.prepare_chunk_for_regeneration(uid, wait=True)
            if updated is not None:
                self.clear_chunk_runtime(uid)
                self._publish_chunk_upsert(uid)
            return updated

        def delete_chunk_by_uid(self, uid):
            result = self.script_store.delete_chunk(uid, wait=True)
            if result is not None:
                deleted = result.get("deleted") or {}
                self.clear_chunk_runtime(deleted.get("uid"))
                self._publish_chunk_delete(deleted.get("uid"), deleted.get("chapter"))
                self._publish_chapter_list_changed()
            return result

        def delete_chapter_by_name(self, chapter):
            result = self.script_store.delete_chapter(chapter, wait=True)
            if result is not None:
                for chunk in result.get("deleted") or []:
                    self.clear_chunk_runtime(chunk.get("uid"))
                self._publish_chapter_deleted(chapter)
                self._publish_chapter_list_changed()
            return result

        def _update_chunk_fields(self, index, **fields):
            """Atomically update fields on a single chunk (thread-safe read-modify-write).

            Unlike load_chunks() + modify + save_chunks(), this holds the lock for the
            entire read-modify-write cycle, preventing concurrent threads from
            overwriting each other's updates.
            """
            with self._chunks_lock:
                chunks = self.load_chunks_raw()
                if not (0 <= index < len(chunks)):
                    return None
                chunks[index].update(fields)
                if "audio_path" in fields or "speaker" in fields or "text" in fields or "instruct" in fields:
                    self._clear_proofread_state(chunks[index])
                self._atomic_json_write(chunks, self.chunks_path)
                return chunks[index]

        def get_chunk_chapter_summary(self):
            return self.script_store.chapter_summary()

        def get_audio_coverage_summary(self):
            return self.script_store.get_audio_coverage_summary()

        def has_generated_chunk_audio(self):
            return self.script_store.has_generated_audio()

        def export_chunks_to_path(self, target_path):
            return self.script_store.export_chunks(target_path)

        def has_substantive_chunks(self):
            return self.script_store.has_substantive_chunks()
