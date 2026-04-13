import json
import os
import queue
import shutil
import sqlite3
import threading
import time
import uuid
import hashlib
from abc import ABC, abstractmethod

from project_core.chunking import script_entries_to_chunks
from project_core.constants import VOICE_AUDIT_LOG_ENABLED_DEFAULT
from script_store import load_script_document


SCRIPT_STORE_SCHEMA_VERSION = 3


class ScriptStore(ABC):
    @abstractmethod
    def start(self):
        raise NotImplementedError

    @abstractmethod
    def stop(self, flush=True):
        raise NotImplementedError

    @abstractmethod
    def load_chunks(self, chapter=None):
        raise NotImplementedError

    @abstractmethod
    def get_chunk(self, chunk_ref):
        raise NotImplementedError

    @abstractmethod
    def get_chunks_by_uids(self, uids):
        raise NotImplementedError

    @abstractmethod
    def get_chapter_chunks(self, chapter):
        raise NotImplementedError

    @abstractmethod
    def get_chapter_list(self):
        raise NotImplementedError

    @abstractmethod
    def resolve_generation_targets(self, scope_mode="project", chapter=None, pending_only=True):
        raise NotImplementedError

    @abstractmethod
    def get_chunk_audio_ref(self, chunk_ref):
        raise NotImplementedError

    @abstractmethod
    def replace_chunks(self, chunks, *, reason="replace_chunks", wait=True):
        raise NotImplementedError

    @abstractmethod
    def patch_chunks(self, patches, *, reason="patch_chunks", wait=True):
        raise NotImplementedError

    @abstractmethod
    def patch_chunk_if(self, uid, expected=None, fields=None, clear_fields=(), *, reason="patch_chunk_if", wait=True):
        raise NotImplementedError

    @abstractmethod
    def patch_chunks_if(self, updates, *, reason="patch_chunks_if", wait=True):
        raise NotImplementedError

    @abstractmethod
    def claim_generation(self, uid, token, *, reason="claim_generation", wait=True):
        raise NotImplementedError

    @abstractmethod
    def claim_generation_many(self, uids, token, *, reason="claim_generation_many", wait=True):
        raise NotImplementedError

    @abstractmethod
    def reset_generation(self, uids, token=None, target_status="pending", *, reason="reset_generation", wait=True):
        raise NotImplementedError

    @abstractmethod
    def prepare_chunk_for_regeneration(self, uid, *, reason="prepare_chunk_for_regeneration", wait=True):
        raise NotImplementedError

    @abstractmethod
    def delete_chunk(self, uid, *, reason="delete_chunk", wait=True):
        raise NotImplementedError

    @abstractmethod
    def delete_chapter(self, chapter, *, reason="delete_chapter", wait=True):
        raise NotImplementedError

    @abstractmethod
    def flush(self, timeout=None):
        raise NotImplementedError

    @abstractmethod
    def chapter_summary(self):
        raise NotImplementedError

    @abstractmethod
    def has_generated_audio(self):
        raise NotImplementedError

    @abstractmethod
    def export_chunks(self, target_path):
        raise NotImplementedError

    @abstractmethod
    def has_substantive_chunks(self):
        raise NotImplementedError

    @abstractmethod
    def clear(self):
        raise NotImplementedError

    @abstractmethod
    def load_voice_config(self):
        raise NotImplementedError

    @abstractmethod
    def has_voice_profiles(self):
        raise NotImplementedError

    @abstractmethod
    def list_voice_rows(self):
        raise NotImplementedError

    @abstractmethod
    def get_voice_profile(self, speaker_ref):
        raise NotImplementedError

    @abstractmethod
    def get_voice_profiles(self, speaker_refs):
        raise NotImplementedError

    @abstractmethod
    def replace_voice_profiles(self, rows, *, reason="replace_voice_profiles", wait=True):
        raise NotImplementedError

    @abstractmethod
    def upsert_voice_profiles(self, rows, *, reason="upsert_voice_profiles", wait=True):
        raise NotImplementedError

    @abstractmethod
    def patch_voice_profile(self, speaker_ref, fields=None, clear_fields=(), *, reason="patch_voice_profile", wait=True):
        raise NotImplementedError

    @abstractmethod
    def get_voice_settings(self):
        raise NotImplementedError

    @abstractmethod
    def set_voice_setting(self, key, value, *, reason="set_voice_setting", wait=True):
        raise NotImplementedError

    @abstractmethod
    def get_narrator_overrides(self):
        raise NotImplementedError

    @abstractmethod
    def set_narrator_override(self, chapter, voice, *, reason="set_narrator_override", wait=True):
        raise NotImplementedError

    @abstractmethod
    def replace_narrator_overrides(self, rows, *, reason="replace_narrator_overrides", wait=True):
        raise NotImplementedError

    @abstractmethod
    def get_auto_narrator_aliases(self):
        raise NotImplementedError

    @abstractmethod
    def replace_auto_narrator_aliases(self, rows, *, reason="replace_auto_narrator_aliases", wait=True):
        raise NotImplementedError

    @abstractmethod
    def refresh_auto_narrator_aliases_from_chunks(self, *, narrator_threshold=0, narrator_name="", reason="refresh_auto_narrator_aliases_from_chunks", wait=True):
        raise NotImplementedError

    @abstractmethod
    def get_voice_summary(self):
        raise NotImplementedError

    @abstractmethod
    def resolve_voice_for_chunk(self, uid):
        raise NotImplementedError

    @abstractmethod
    def export_voice_config(self, target_path):
        raise NotImplementedError

    @abstractmethod
    def export_voice_state(self, target_path):
        raise NotImplementedError

    @abstractmethod
    def load_voice_state_snapshot(self):
        raise NotImplementedError

    @abstractmethod
    def replace_voice_state_snapshot(self, snapshot, *, reason="replace_voice_state_snapshot", wait=True):
        raise NotImplementedError

    @abstractmethod
    def enqueue_audio_finalize_task(self, task, *, reason="enqueue_audio_finalize_task", wait=True):
        raise NotImplementedError

    @abstractmethod
    def claim_next_audio_finalize_task(self, *, reason="claim_next_audio_finalize_task", wait=True):
        raise NotImplementedError

    @abstractmethod
    def complete_audio_finalize_task(self, task_id, *, reason="complete_audio_finalize_task", wait=True):
        raise NotImplementedError

    @abstractmethod
    def fail_audio_finalize_task(self, task_id, error=None, requeue=False, *, reason="fail_audio_finalize_task", wait=True):
        raise NotImplementedError

    @abstractmethod
    def list_audio_finalize_tasks(self, generation_token=None, statuses=None):
        raise NotImplementedError

    @abstractmethod
    def count_audio_finalize_tasks(self, generation_token=None):
        raise NotImplementedError

    @abstractmethod
    def clear_audio_finalize_tasks(self, generation_token=None, uids=None, *, reason="clear_audio_finalize_tasks", wait=True):
        raise NotImplementedError


class SQLiteScriptStore(ScriptStore):
    _RESERVED_FIELDS = {
        "id",
        "uid",
        "speaker",
        "text",
        "instruct",
        "chapter",
        "paragraph_id",
        "type",
        "silence_duration_s",
        "status",
        "audio_path",
        "audio_validation",
        "proofread",
        "auto_regen_count",
        "generation_token",
    }

    def __init__(
        self,
        *,
        root_dir,
        db_path,
        queue_log_path,
        script_path,
        legacy_chunks_path,
        voice_config_path=None,
        state_path=None,
        archive_dir=None,
        voice_audit_log_path=None,
    ):
        self.root_dir = root_dir
        self.db_path = db_path
        self.queue_log_path = queue_log_path
        self.script_path = script_path
        self.legacy_chunks_path = legacy_chunks_path
        self.voice_config_path = voice_config_path or os.path.join(root_dir, "voice_config.json")
        self.state_path = state_path or os.path.join(root_dir, "state.json")
        self.archive_dir = archive_dir or os.path.join(root_dir, "backups", "chunks")
        self.voice_audit_log_path = voice_audit_log_path or os.path.join(root_dir, "voice_state.audit.jsonl")
        self.voice_audit_logging_enabled = VOICE_AUDIT_LOG_ENABLED_DEFAULT
        self._command_queue = queue.Queue()
        self._writer_thread = None
        self._writer_stop = threading.Event()
        self._log_lock = threading.Lock()
        self._started = False

    def start(self):
        if self._started:
            return self
        os.makedirs(os.path.dirname(self.db_path) or ".", exist_ok=True)
        os.makedirs(os.path.dirname(self.queue_log_path) or ".", exist_ok=True)
        os.makedirs(self.archive_dir, exist_ok=True)
        with open(self.queue_log_path, "w", encoding="utf-8"):
            pass
        self._initialize_db()
        self._bootstrap_if_needed()
        self._bootstrap_voice_state_if_needed()
        self._requeue_processing_audio_finalize_tasks()
        self._writer_thread = threading.Thread(
            target=self._writer_loop,
            daemon=True,
            name=f"script-store-{os.path.basename(self.root_dir)}",
        )
        self._writer_thread.start()
        self._started = True
        return self

    def stop(self, flush=True):
        if not self._started:
            return
        if flush:
            try:
                self.flush(timeout=5.0)
            except (FileNotFoundError, TimeoutError, OSError):
                pass
        self._writer_stop.set()
        self._command_queue.put(None)
        if self._writer_thread is not None:
            self._writer_thread.join(timeout=5.0)
        self._writer_thread = None
        self._started = False

    def clear(self):
        self.stop(flush=False)
        for path in (self.db_path, self.queue_log_path):
            if os.path.exists(path):
                os.remove(path)
        self._command_queue = queue.Queue()
        self._writer_stop = threading.Event()
        self.start()

    def load_chunks(self, chapter=None):
        self._bootstrap_if_needed()
        query = (
            "SELECT uid, ordinal, speaker, text, instruct, chapter, paragraph_id, "
            "chunk_type, silence_duration_s, status, audio_path, audio_validation_json, "
            "proofread_json, auto_regen_count, generation_token, extra_json "
            "FROM chunks"
        )
        params = []
        normalized_chapter = str(chapter or "").strip()
        if normalized_chapter:
            query += " WHERE chapter = ?"
            params.append(normalized_chapter)
        query += " ORDER BY ordinal ASC"
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [self._row_to_chunk(row) for row in rows]

    def get_chunk(self, chunk_ref):
        self._bootstrap_if_needed()
        resolved = self._resolve_chunk_ref(chunk_ref)
        if resolved is None:
            return None
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT uid, ordinal, speaker, text, instruct, chapter, paragraph_id,
                       chunk_type, silence_duration_s, status, audio_path,
                       audio_validation_json, proofread_json, auto_regen_count,
                       generation_token, extra_json
                FROM chunks
                WHERE uid = ?
                """,
                (resolved["uid"],),
            ).fetchone()
        return self._row_to_chunk(row) if row is not None else None

    def get_chunks_by_uids(self, uids):
        self._bootstrap_if_needed()
        normalized = [str(uid).strip() for uid in (uids or []) if str(uid).strip()]
        if not normalized:
            return []
        placeholders = ",".join("?" for _ in normalized)
        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT uid, ordinal, speaker, text, instruct, chapter, paragraph_id,
                       chunk_type, silence_duration_s, status, audio_path,
                       audio_validation_json, proofread_json, auto_regen_count,
                       generation_token, extra_json
                FROM chunks
                WHERE uid IN ({placeholders})
                ORDER BY ordinal ASC
                """,
                normalized,
            ).fetchall()
        return [self._row_to_chunk(row) for row in rows]

    def get_chapter_chunks(self, chapter):
        return self.load_chunks(chapter=chapter)

    def get_chapter_list(self):
        self._bootstrap_if_needed()
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT chapter, COUNT(*) AS chunk_count, MIN(ordinal) AS first_ordinal
                FROM chunks
                WHERE TRIM(COALESCE(chapter, '')) != ''
                GROUP BY chapter
                ORDER BY first_ordinal ASC
                """
            ).fetchall()
        return [
            {
                "chapter": row["chapter"],
                "chunk_count": int(row["chunk_count"] or 0),
            }
            for row in rows
        ]

    def resolve_generation_targets(self, scope_mode="project", chapter=None, pending_only=True):
        self._bootstrap_if_needed()
        query = (
            "SELECT uid, ordinal, speaker, text, instruct, chapter, paragraph_id, "
            "chunk_type, silence_duration_s, status, audio_path, audio_validation_json, "
            "proofread_json, auto_regen_count, generation_token, extra_json "
            "FROM chunks WHERE TRIM(COALESCE(text, '')) != ''"
        )
        params = []
        normalized_scope = str(scope_mode or "project").strip().lower()
        normalized_chapter = str(chapter or "").strip()
        if normalized_scope == "chapter" and normalized_chapter:
            query += " AND chapter = ?"
            params.append(normalized_chapter)
        if pending_only:
            # Treat errored rows as still outstanding work. A new render job should
            # retry anything not yet completed instead of leaving persistent errors.
            query += " AND COALESCE(status, 'pending') != 'done'"
        query += " ORDER BY ordinal ASC"
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [self._row_to_chunk(row) for row in rows]

    def get_chunk_audio_ref(self, chunk_ref):
        chunk = self.get_chunk(chunk_ref)
        if chunk is None:
            return None
        return {
            "uid": chunk.get("uid"),
            "id": chunk.get("id"),
            "status": chunk.get("status"),
            "audio_path": chunk.get("audio_path"),
            "audio_fingerprint": self._chunk_audio_fingerprint(chunk),
        }

    def replace_chunks(self, chunks, *, reason="replace_chunks", wait=True):
        payload = [self._normalize_chunk(chunk, index) for index, chunk in enumerate(chunks or [])]
        return self._submit_command("replace_chunks", {"chunks": payload, "reason": reason}, wait=wait)

    def patch_chunks(self, patches, *, reason="patch_chunks", wait=True):
        normalized = []
        for patch in patches or []:
            uid = str((patch or {}).get("uid") or "").strip()
            if not uid:
                continue
            fields = dict((patch or {}).get("fields") or {})
            normalized.append({"uid": uid, "fields": fields, "expected": {}, "clear_fields": ()})
        if not normalized:
            return 0
        return self._submit_command("patch_chunks", {"patches": normalized, "reason": reason}, wait=wait)

    def patch_chunk_if(self, uid, expected=None, fields=None, clear_fields=(), *, reason="patch_chunk_if", wait=True):
        updates = [{
            "uid": uid,
            "expected": dict(expected or {}),
            "fields": dict(fields or {}),
            "clear_fields": list(clear_fields or ()),
        }]
        result = self.patch_chunks_if(updates, reason=reason, wait=wait)
        if wait:
            return result[0] if result else None
        return result

    def patch_chunks_if(self, updates, *, reason="patch_chunks_if", wait=True):
        normalized = []
        for update in updates or []:
            uid = str((update or {}).get("uid") or "").strip()
            if not uid:
                continue
            normalized.append({
                "uid": uid,
                "expected": dict((update or {}).get("expected") or {}),
                "fields": dict((update or {}).get("fields") or {}),
                "clear_fields": list((update or {}).get("clear_fields") or ()),
            })
        if not normalized:
            return []
        return self._submit_command("patch_chunks", {"patches": normalized, "reason": reason}, wait=wait)

    def claim_generation(self, uid, token, *, reason="claim_generation", wait=True):
        result = self._submit_command(
            "claim_generation",
            {"uids": [str(uid).strip()], "token": token, "reason": reason},
            wait=wait,
        )
        if wait:
            return result[0] if result else None
        return result

    def claim_generation_many(self, uids, token, *, reason="claim_generation_many", wait=True):
        normalized = [str(uid).strip() for uid in (uids or []) if str(uid).strip()]
        if not normalized:
            return []
        return self._submit_command(
            "claim_generation",
            {"uids": normalized, "token": token, "reason": reason},
            wait=wait,
        )

    def reset_generation(self, uids, token=None, target_status="pending", *, reason="reset_generation", wait=True):
        normalized = [str(uid).strip() for uid in (uids or []) if str(uid).strip()]
        if not normalized:
            return []
        return self._submit_command(
            "reset_generation",
            {
                "uids": normalized,
                "token": token,
                "target_status": target_status,
                "reason": reason,
            },
            wait=wait,
        )

    def prepare_chunk_for_regeneration(self, uid, *, reason="prepare_chunk_for_regeneration", wait=True):
        normalized = str(uid or "").strip()
        if not normalized:
            return None
        return self._submit_command(
            "prepare_chunk_for_regeneration",
            {"uid": normalized, "reason": reason},
            wait=wait,
        )

    def delete_chunk(self, uid, *, reason="delete_chunk", wait=True):
        normalized = str(uid or "").strip()
        if not normalized:
            return None
        return self._submit_command(
            "delete_chunk",
            {"uid": normalized, "reason": reason},
            wait=wait,
        )

    def delete_chapter(self, chapter, *, reason="delete_chapter", wait=True):
        normalized = str(chapter or "").strip()
        if not normalized:
            return None
        return self._submit_command(
            "delete_chapter",
            {"chapter": normalized, "reason": reason},
            wait=wait,
        )

    def flush(self, timeout=None):
        return self._submit_command("barrier", {"reason": "flush"}, wait=True, timeout=timeout)

    def chapter_summary(self):
        chunks = self.load_chunks()
        ordered_chapters = []
        last_seen = None
        for chunk in chunks:
            chapter = str(chunk.get("chapter") or "").strip()
            if not chapter:
                continue
            if chapter != last_seen:
                ordered_chapters.append(chapter)
                last_seen = chapter
        return {
            "chunk_count": len(chunks),
            "chapter_count": len(ordered_chapters),
            "last_chapter": ordered_chapters[-1] if ordered_chapters else None,
        }

    def has_generated_audio(self):
        self._bootstrap_if_needed()
        with self._connect() as conn:
            row = conn.execute(
                "SELECT 1 FROM chunks WHERE audio_path IS NOT NULL AND TRIM(audio_path) != '' LIMIT 1"
            ).fetchone()
        return row is not None

    def has_substantive_chunks(self):
        self._bootstrap_if_needed()
        with self._connect() as conn:
            row = conn.execute(
                "SELECT 1 FROM chunks WHERE TRIM(COALESCE(text, '')) != '' OR TRIM(COALESCE(audio_path, '')) != '' LIMIT 1"
            ).fetchone()
        return row is not None

    def load_voice_config(self):
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT speaker_key, display_name, voice_type, voice_name, character_style,
                       default_style, alias, seed, ref_audio, ref_text, generated_ref_text,
                       adapter_id, adapter_path, description, narrates, extra_json
                FROM voice_profiles
                ORDER BY display_name COLLATE NOCASE ASC
                """
            ).fetchall()
        config = {}
        for row in rows:
            profile = self._row_to_voice_profile(row)
            config[profile["display_name"]] = profile["config"]
        return config

    def has_voice_profiles(self):
        with self._connect() as conn:
            row = conn.execute("SELECT 1 FROM voice_profiles LIMIT 1").fetchone()
        return row is not None

    def list_voice_rows(self):
        summary = self.get_voice_summary()
        line_counts = summary.get("line_counts", {})
        profiles = self.load_voice_config()
        all_names = {}
        for name in profiles.keys():
            key = self._speaker_key(name)
            if key:
                all_names[key] = name
        for name in line_counts.keys():
            key = self._speaker_key(name)
            if key and key not in all_names:
                all_names[key] = name
        auto_aliases = self.get_auto_narrator_aliases()
        rows = []
        for speaker_key, display_name in sorted(all_names.items(), key=lambda item: item[1].casefold()):
            config = dict(profiles.get(display_name) or {})
            if not config:
                config = self._default_voice_config()
            ref_audio = str(config.get("ref_audio") or "").strip()
            rows.append({
                "name": display_name,
                "speaker_key": speaker_key,
                "config": config,
                "line_count": int(line_counts.get(display_name, 0) or 0),
                "auto_alias_target": str(auto_aliases.get(display_name) or ""),
                "auto_narrator_alias": bool(auto_aliases.get(display_name)),
                "has_ref_audio": bool(ref_audio),
            })
        return rows

    def get_voice_profile(self, speaker_ref):
        target = self._speaker_key(speaker_ref)
        if not target:
            return None
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT speaker_key, display_name, voice_type, voice_name, character_style,
                       default_style, alias, seed, ref_audio, ref_text, generated_ref_text,
                       adapter_id, adapter_path, description, narrates, extra_json
                FROM voice_profiles
                WHERE speaker_key = ?
                """,
                (target,),
            ).fetchone()
        return self._row_to_voice_profile(row) if row is not None else None

    def get_voice_profiles(self, speaker_refs):
        targets = [self._speaker_key(value) for value in (speaker_refs or [])]
        normalized = [value for value in targets if value]
        if not normalized:
            return {}
        placeholders = ",".join("?" for _ in normalized)
        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT speaker_key, display_name, voice_type, voice_name, character_style,
                       default_style, alias, seed, ref_audio, ref_text, generated_ref_text,
                       adapter_id, adapter_path, description, narrates, extra_json
                FROM voice_profiles
                WHERE speaker_key IN ({placeholders})
                """,
                normalized,
            ).fetchall()
        return {
            row_profile["display_name"]: row_profile["config"]
            for row_profile in (self._row_to_voice_profile(row) for row in rows)
        }

    def replace_voice_profiles(self, rows, *, reason="replace_voice_profiles", wait=True):
        normalized = self._dedupe_voice_profile_rows(rows)
        return self._submit_command(
            "replace_voice_profiles",
            {"rows": normalized, "reason": reason},
            wait=wait,
        )

    def upsert_voice_profiles(self, rows, *, reason="upsert_voice_profiles", wait=True):
        normalized = self._dedupe_voice_profile_rows(rows)
        if not normalized:
            return []
        return self._submit_command(
            "upsert_voice_profiles",
            {"rows": normalized, "reason": reason},
            wait=wait,
        )

    def patch_voice_profile(self, speaker_ref, fields=None, clear_fields=(), *, reason="patch_voice_profile", wait=True):
        normalized = self._speaker_key(speaker_ref)
        if not normalized:
            return None
        return self._submit_command(
            "patch_voice_profile",
            {
                "speaker_key": normalized,
                "display_name": self._speaker_display_name(speaker_ref),
                "fields": dict(fields or {}),
                "clear_fields": list(clear_fields or ()),
                "reason": reason,
            },
            wait=wait,
        )

    def get_voice_settings(self):
        with self._connect() as conn:
            rows = conn.execute("SELECT key, value FROM voice_settings").fetchall()
        settings = {row["key"]: row["value"] for row in rows}
        if "narrator_threshold" not in settings:
            settings["narrator_threshold"] = "10"
        return settings

    def set_voice_setting(self, key, value, *, reason="set_voice_setting", wait=True):
        normalized_key = str(key or "").strip()
        if not normalized_key:
            return None
        return self._submit_command(
            "set_voice_setting",
            {"key": normalized_key, "value": "" if value is None else str(value), "reason": reason},
            wait=wait,
        )

    def get_narrator_overrides(self):
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT chapter, voice_name FROM chapter_narrator_overrides ORDER BY chapter COLLATE NOCASE ASC"
            ).fetchall()
        return {
            str(row["chapter"] or "").strip(): str(row["voice_name"] or "").strip()
            for row in rows
            if str(row["chapter"] or "").strip()
        }

    def set_narrator_override(self, chapter, voice, *, reason="set_narrator_override", wait=True):
        normalized_chapter = str(chapter or "").strip()
        if not normalized_chapter:
            return None
        return self._submit_command(
            "set_narrator_override",
            {
                "chapter": normalized_chapter,
                "voice_key": self._speaker_key(voice),
                "voice_name": self._speaker_display_name(voice),
                "reason": reason,
            },
            wait=wait,
        )

    def replace_narrator_overrides(self, rows, *, reason="replace_narrator_overrides", wait=True):
        normalized = []
        for row in (rows or []):
            chapter = str((row or {}).get("chapter") or "").strip()
            if not chapter:
                continue
            voice_name = self._speaker_display_name((row or {}).get("voice"))
            normalized.append({
                "chapter": chapter,
                "voice_key": self._speaker_key(voice_name),
                "voice_name": voice_name,
            })
        return self._submit_command(
            "replace_narrator_overrides",
            {"rows": normalized, "reason": reason},
            wait=wait,
        )

    def get_auto_narrator_aliases(self):
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT speaker_name, target_name FROM voice_auto_aliases ORDER BY speaker_name COLLATE NOCASE ASC"
            ).fetchall()
        return {
            str(row["speaker_name"] or "").strip(): str(row["target_name"] or "").strip()
            for row in rows
            if str(row["speaker_name"] or "").strip() and str(row["target_name"] or "").strip()
        }

    def replace_auto_narrator_aliases(self, rows, *, reason="replace_auto_narrator_aliases", wait=True):
        normalized = []
        for row in (rows or []):
            speaker_name = self._speaker_display_name((row or {}).get("speaker"))
            target_name = self._speaker_display_name((row or {}).get("target"))
            speaker_key = self._speaker_key(speaker_name)
            target_key = self._speaker_key(target_name)
            if not speaker_key or not target_key:
                continue
            normalized.append({
                "speaker_key": speaker_key,
                "speaker_name": speaker_name,
                "target_key": target_key,
                "target_name": target_name,
            })
        return self._submit_command(
            "replace_auto_narrator_aliases",
            {"rows": normalized, "reason": reason},
            wait=wait,
        )

    def refresh_auto_narrator_aliases_from_chunks(self, *, narrator_threshold=0, narrator_name="", reason="refresh_auto_narrator_aliases_from_chunks", wait=True):
        return self._submit_command(
            "refresh_auto_narrator_aliases_from_chunks",
            {
                "narrator_threshold": int(narrator_threshold or 0),
                "narrator_name": self._speaker_display_name(narrator_name),
                "reason": reason,
            },
            wait=wait,
        )

    def get_voice_summary(self):
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT LOWER(TRIM(COALESCE(speaker, ''))) AS speaker_key,
                       MIN(COALESCE(speaker, '')) AS display_name,
                       COUNT(*) AS line_count,
                       MIN(ordinal) AS first_ordinal
                FROM chunks
                WHERE TRIM(COALESCE(speaker, '')) != ''
                GROUP BY LOWER(TRIM(COALESCE(speaker, '')))
                ORDER BY first_ordinal ASC
                """
            ).fetchall()
        line_counts = {}
        ordered_names = []
        for row in rows:
            display_name = self._speaker_display_name(row["display_name"])
            if not display_name:
                continue
            ordered_names.append(display_name)
            line_counts[display_name] = int(row["line_count"] or 0)
        return {"voices": ordered_names, "line_counts": line_counts}

    def resolve_voice_for_chunk(self, uid):
        chunk = self.get_chunk(uid)
        if chunk is None:
            return None
        voice_config = self.load_voice_config()
        narrator_overrides = self.get_narrator_overrides()
        auto_narrator_aliases = self.get_auto_narrator_aliases()
        resolved_speaker = self._resolve_voice_speaker_for_store(
            str(chunk.get("speaker") or ""),
            chapter=str(chunk.get("chapter") or ""),
            voice_config=voice_config,
            narrator_overrides=narrator_overrides,
            auto_narrator_aliases=auto_narrator_aliases,
        )
        return {
            "uid": chunk.get("uid"),
            "speaker": chunk.get("speaker"),
            "chapter": chunk.get("chapter"),
            "resolved_speaker": resolved_speaker,
            "voice_profile": dict(voice_config.get(resolved_speaker) or {}),
        }

    def export_voice_config(self, target_path):
        config = self.load_voice_config()
        os.makedirs(os.path.dirname(target_path) or ".", exist_ok=True)
        tmp_path = f"{target_path}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        os.replace(tmp_path, target_path)
        return target_path

    def export_voice_state(self, target_path):
        payload = self.load_voice_state_snapshot()
        os.makedirs(os.path.dirname(target_path) or ".", exist_ok=True)
        tmp_path = f"{target_path}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        os.replace(tmp_path, target_path)
        return target_path

    def load_voice_state_snapshot(self):
        with self._connect() as conn:
            snapshot = self._load_voice_state_snapshot_tx(conn)
        return snapshot

    def replace_voice_state_snapshot(self, snapshot, *, reason="replace_voice_state_snapshot", wait=True):
        payload = dict(snapshot or {})
        return self._submit_command(
            "replace_voice_state_snapshot",
            {
                "snapshot": payload,
                "reason": reason,
            },
            wait=wait,
        )

    def enqueue_audio_finalize_task(self, task, *, reason="enqueue_audio_finalize_task", wait=True):
        normalized = self._normalize_audio_finalize_task(task)
        if normalized is None:
            return None
        return self._submit_command(
            "enqueue_audio_finalize_task",
            {"task": normalized, "reason": reason},
            wait=wait,
        )

    def claim_next_audio_finalize_task(self, *, reason="claim_next_audio_finalize_task", wait=True):
        return self._submit_command(
            "claim_next_audio_finalize_task",
            {"reason": reason},
            wait=wait,
        )

    def complete_audio_finalize_task(self, task_id, *, reason="complete_audio_finalize_task", wait=True):
        return self._submit_command(
            "complete_audio_finalize_task",
            {"task_id": int(task_id or 0), "reason": reason},
            wait=wait,
        )

    def fail_audio_finalize_task(self, task_id, error=None, requeue=False, *, reason="fail_audio_finalize_task", wait=True):
        return self._submit_command(
            "fail_audio_finalize_task",
            {
                "task_id": int(task_id or 0),
                "error": "" if error is None else str(error),
                "requeue": bool(requeue),
                "reason": reason,
            },
            wait=wait,
        )

    def list_audio_finalize_tasks(self, generation_token=None, statuses=None):
        query = (
            "SELECT id, chunk_uid, generation_token, temp_wav_path, attempt, speaker, text, "
            "status, created_at, updated_at, last_error "
            "FROM audio_finalize_queue"
        )
        params = []
        clauses = []
        normalized_token = str(generation_token or "").strip()
        if normalized_token:
            clauses.append("generation_token = ?")
            params.append(normalized_token)
        normalized_statuses = [str(status).strip() for status in (statuses or []) if str(status).strip()]
        if normalized_statuses:
            clauses.append("status IN (" + ",".join("?" for _ in normalized_statuses) + ")")
            params.extend(normalized_statuses)
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY id ASC"
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [self._row_to_audio_finalize_task(row) for row in rows]

    def count_audio_finalize_tasks(self, generation_token=None):
        query = "SELECT COUNT(*) AS count FROM audio_finalize_queue"
        params = []
        normalized_token = str(generation_token or "").strip()
        if normalized_token:
            query += " WHERE generation_token = ?"
            params.append(normalized_token)
        with self._connect() as conn:
            row = conn.execute(query, params).fetchone()
        return int(row["count"] or 0) if row is not None else 0

    def clear_audio_finalize_tasks(self, generation_token=None, uids=None, *, reason="clear_audio_finalize_tasks", wait=True):
        normalized_uids = [str(uid).strip() for uid in (uids or []) if str(uid).strip()]
        return self._submit_command(
            "clear_audio_finalize_tasks",
            {
                "generation_token": str(generation_token or "").strip(),
                "uids": normalized_uids,
                "reason": reason,
            },
            wait=wait,
        )

    def export_chunks(self, target_path):
        chunks = self.load_chunks()
        os.makedirs(os.path.dirname(target_path) or ".", exist_ok=True)
        tmp_path = f"{target_path}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(chunks, f, indent=2, ensure_ascii=False)
        os.replace(tmp_path, target_path)
        return target_path

    def _initialize_db(self):
        with self._connect() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS chunks (
                    uid TEXT PRIMARY KEY,
                    ordinal INTEGER NOT NULL UNIQUE,
                    speaker TEXT,
                    text TEXT,
                    instruct TEXT,
                    chapter TEXT,
                    paragraph_id TEXT,
                    chunk_type TEXT,
                    silence_duration_s REAL,
                    status TEXT,
                    audio_path TEXT,
                    audio_validation_json TEXT,
                    proofread_json TEXT,
                    auto_regen_count INTEGER NOT NULL DEFAULT 0,
                    generation_token TEXT,
                    extra_json TEXT
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS voice_profiles (
                    speaker_key TEXT PRIMARY KEY,
                    display_name TEXT NOT NULL,
                    voice_type TEXT,
                    voice_name TEXT,
                    character_style TEXT,
                    default_style TEXT,
                    alias TEXT,
                    seed TEXT,
                    ref_audio TEXT,
                    ref_text TEXT,
                    generated_ref_text TEXT,
                    adapter_id TEXT,
                    adapter_path TEXT,
                    description TEXT,
                    narrates INTEGER,
                    extra_json TEXT
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS voice_settings (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS chapter_narrator_overrides (
                    chapter TEXT PRIMARY KEY,
                    voice_key TEXT,
                    voice_name TEXT
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS voice_auto_aliases (
                    speaker_key TEXT PRIMARY KEY,
                    speaker_name TEXT,
                    target_key TEXT,
                    target_name TEXT
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS audio_finalize_queue (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    chunk_uid TEXT NOT NULL,
                    generation_token TEXT,
                    temp_wav_path TEXT NOT NULL,
                    attempt INTEGER NOT NULL DEFAULT 0,
                    speaker TEXT,
                    text TEXT,
                    status TEXT NOT NULL DEFAULT 'queued',
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    last_error TEXT
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_audio_finalize_queue_status_id ON audio_finalize_queue(status, id)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_audio_finalize_queue_generation_token ON audio_finalize_queue(generation_token)"
            )
            conn.execute(
                "INSERT OR REPLACE INTO metadata(key, value) VALUES('schema_version', ?)",
                (str(SCRIPT_STORE_SCHEMA_VERSION),),
            )
            conn.commit()

    def _requeue_processing_audio_finalize_tasks(self):
        with self._connect() as conn:
            with conn:
                conn.execute(
                    """
                    UPDATE audio_finalize_queue
                    SET status = 'queued', updated_at = ?, last_error = NULL
                    WHERE status = 'processing'
                    """,
                    (time.time(),),
                )

    def _connect(self):
        conn = sqlite3.connect(self.db_path, timeout=30.0, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    def _bootstrap_if_needed(self):
        with self._connect() as conn:
            row = conn.execute("SELECT COUNT(*) AS count FROM chunks").fetchone()
            if int(row["count"] or 0) > 0:
                return
        legacy_chunks = self._load_legacy_chunks()
        if legacy_chunks is not None:
            with self._connect() as conn:
                self._replace_chunks_tx(conn, legacy_chunks)
            self._archive_legacy_chunks()
            return
        if os.path.exists(self.script_path):
            script_entries = load_script_document(self.script_path)["entries"]
            chunks = script_entries_to_chunks(script_entries)
            normalized = [self._normalize_chunk(chunk, index) for index, chunk in enumerate(chunks)]
            with self._connect() as conn:
                self._replace_chunks_tx(conn, normalized)

    def _load_legacy_chunks(self):
        if not os.path.exists(self.legacy_chunks_path):
            return None
        try:
            with open(self.legacy_chunks_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except (OSError, ValueError, json.JSONDecodeError) as e:
            backup_path = f"{self.legacy_chunks_path}.corrupt-{int(time.time())}"
            try:
                shutil.copy2(self.legacy_chunks_path, backup_path)
            except OSError:
                backup_path = None
            raise RuntimeError(
                "chunks.json is corrupted"
                + (f"; preserved backup at {backup_path}" if backup_path else "")
            ) from e
        if not isinstance(payload, list):
            return None
        return [self._normalize_chunk(chunk, index) for index, chunk in enumerate(payload)]

    def _archive_legacy_chunks(self):
        if not os.path.exists(self.legacy_chunks_path):
            return
        timestamp = int(time.time())
        archive_path = os.path.join(self.archive_dir, f"chunks.imported.{timestamp}.json")
        if os.path.abspath(archive_path) == os.path.abspath(self.legacy_chunks_path):
            return
        shutil.copy2(self.legacy_chunks_path, archive_path)
        try:
            os.remove(self.legacy_chunks_path)
        except OSError:
            pass

    def _bootstrap_voice_state_if_needed(self):
        with self._connect() as conn:
            voice_profile_count = int(conn.execute("SELECT COUNT(*) AS count FROM voice_profiles").fetchone()["count"] or 0)
            settings_count = int(conn.execute("SELECT COUNT(*) AS count FROM voice_settings").fetchone()["count"] or 0)
            current_revision = self._get_voice_state_revision(conn)

        with self._connect() as conn:
            with conn:
                if settings_count <= 0:
                    conn.execute(
                        "INSERT OR REPLACE INTO voice_settings(key, value) VALUES(?, ?)",
                        ("narrator_threshold", "10"),
                    )

                seeded_rows = []
                if int(conn.execute("SELECT COUNT(*) AS count FROM voice_profiles").fetchone()["count"] or 0) <= 0:
                    seeded_rows = [
                        self._normalize_voice_profile_row({"speaker": speaker, "config": self._default_voice_config()})
                        for speaker in self.get_voice_summary().get("voices", [])
                    ]
                    self._replace_voice_profiles_tx(conn, seeded_rows)

                if current_revision <= 0 or seeded_rows:
                    self._bump_voice_state_revision_tx(conn)
                    conn.execute(
                        "INSERT OR REPLACE INTO metadata(key, value) VALUES('last_voice_bootstrap_at', ?)",
                        (str(time.time()),),
                    )

    def _submit_command(self, name, payload, *, wait, timeout=None):
        event = threading.Event() if wait else None
        envelope = {
            "name": name,
            "payload": payload,
            "event": event,
            "result": None,
            "error": None,
        }
        self._log_command(name, payload)
        self._command_queue.put(envelope)
        if not wait:
            return None
        event.wait(timeout=timeout)
        if not event.is_set():
            raise TimeoutError(f"Timed out waiting for script store command {name}")
        if envelope["error"] is not None:
            raise envelope["error"]
        return envelope["result"]

    def _writer_loop(self):
        conn = self._connect()
        try:
            while not self._writer_stop.is_set():
                command = self._command_queue.get()
                try:
                    if command is None:
                        continue
                    result = self._apply_command(conn, command["name"], command["payload"])
                    command["result"] = result
                except Exception as exc:
                    command["error"] = exc
                finally:
                    if command is not None and command.get("event") is not None:
                        command["event"].set()
                    self._command_queue.task_done()
        finally:
            conn.close()

    def _apply_command(self, conn, name, payload):
        if name == "replace_chunks":
            return self._replace_chunks_tx(conn, payload.get("chunks") or [])
        if name == "patch_chunks":
            return self._patch_chunks_tx(conn, payload.get("patches") or [])
        if name == "claim_generation":
            return self._claim_generation_tx(conn, payload.get("uids") or [], payload.get("token"))
        if name == "reset_generation":
            return self._reset_generation_tx(
                conn,
                payload.get("uids") or [],
                token=payload.get("token"),
                target_status=payload.get("target_status") or "pending",
            )
        if name == "prepare_chunk_for_regeneration":
            return self._prepare_chunk_for_regeneration_tx(conn, payload.get("uid"))
        if name == "delete_chunk":
            return self._delete_chunk_tx(conn, payload.get("uid"))
        if name == "delete_chapter":
            return self._delete_chapter_tx(conn, payload.get("chapter"))
        if name == "replace_voice_profiles":
            result = self._replace_voice_profiles_tx(conn, payload.get("rows") or [])
            self._audit_voice_state_write(conn, name, payload)
            return result
        if name == "upsert_voice_profiles":
            result = self._upsert_voice_profiles_tx(conn, payload.get("rows") or [])
            self._audit_voice_state_write(conn, name, payload)
            return result
        if name == "patch_voice_profile":
            result = self._patch_voice_profile_tx(
                conn,
                payload.get("speaker_key"),
                payload.get("display_name"),
                payload.get("fields") or {},
                payload.get("clear_fields") or (),
            )
            self._audit_voice_state_write(conn, name, payload)
            return result
        if name == "set_voice_setting":
            result = self._set_voice_setting_tx(conn, payload.get("key"), payload.get("value"))
            self._audit_voice_state_write(conn, name, payload)
            return result
        if name == "set_narrator_override":
            result = self._set_narrator_override_tx(conn, payload.get("chapter"), payload.get("voice_key"), payload.get("voice_name"))
            self._audit_voice_state_write(conn, name, payload)
            return result
        if name == "replace_narrator_overrides":
            result = self._replace_narrator_overrides_tx(conn, payload.get("rows") or [])
            self._audit_voice_state_write(conn, name, payload)
            return result
        if name == "replace_auto_narrator_aliases":
            result = self._replace_auto_narrator_aliases_tx(conn, payload.get("rows") or [])
            self._audit_voice_state_write(conn, name, payload)
            return result
        if name == "refresh_auto_narrator_aliases_from_chunks":
            result = self._refresh_auto_narrator_aliases_from_chunks_tx(
                conn,
                int(payload.get("narrator_threshold") or 0),
                payload.get("narrator_name") or "",
            )
            self._audit_voice_state_write(conn, name, payload)
            return result
        if name == "replace_voice_state_snapshot":
            result = self._replace_voice_state_snapshot_tx(conn, payload.get("snapshot") or {})
            self._audit_voice_state_write(conn, name, payload)
            return result
        if name == "enqueue_audio_finalize_task":
            return self._enqueue_audio_finalize_task_tx(conn, payload.get("task") or {})
        if name == "claim_next_audio_finalize_task":
            return self._claim_next_audio_finalize_task_tx(conn)
        if name == "complete_audio_finalize_task":
            return self._complete_audio_finalize_task_tx(conn, payload.get("task_id"))
        if name == "fail_audio_finalize_task":
            return self._fail_audio_finalize_task_tx(
                conn,
                payload.get("task_id"),
                error=payload.get("error"),
                requeue=bool(payload.get("requeue")),
            )
        if name == "clear_audio_finalize_tasks":
            return self._clear_audio_finalize_tasks_tx(
                conn,
                generation_token=payload.get("generation_token"),
                uids=payload.get("uids") or [],
            )
        if name == "barrier":
            return True
        raise ValueError(f"Unsupported script store command: {name}")

    def _replace_chunks_tx(self, conn, chunks):
        normalized = [self._normalize_chunk(chunk, index) for index, chunk in enumerate(chunks or [])]
        with conn:
            conn.execute("DELETE FROM chunks")
            for chunk in normalized:
                conn.execute(
                    """
                    INSERT INTO chunks(
                        uid, ordinal, speaker, text, instruct, chapter, paragraph_id,
                        chunk_type, silence_duration_s, status, audio_path,
                        audio_validation_json, proofread_json, auto_regen_count,
                        generation_token, extra_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    self._chunk_to_params(chunk),
                )
            conn.execute(
                "INSERT OR REPLACE INTO metadata(key, value) VALUES('last_write_at', ?)",
                (str(time.time()),),
            )
        return len(normalized)

    def _patch_chunks_tx(self, conn, patches):
        if not patches:
            return []
        updated = []
        with conn:
            for patch in patches:
                uid = patch["uid"]
                row = conn.execute(
                    """
                    SELECT uid, ordinal, speaker, text, instruct, chapter, paragraph_id,
                           chunk_type, silence_duration_s, status, audio_path,
                           audio_validation_json, proofread_json, auto_regen_count,
                           generation_token, extra_json
                    FROM chunks WHERE uid = ?
                    """,
                    (uid,),
                ).fetchone()
                if row is None:
                    continue
                chunk = self._row_to_chunk(row)
                expected = patch.get("expected") or {}
                if any(chunk.get(field) != value for field, value in expected.items()):
                    continue
                for field in patch.get("clear_fields") or ():
                    chunk.pop(field, None)
                chunk.update(patch.get("fields") or {})
                normalized = self._normalize_chunk(chunk, int(chunk.get("id") or row["ordinal"] or 0))
                conn.execute(
                    """
                    UPDATE chunks
                    SET ordinal = ?, speaker = ?, text = ?, instruct = ?, chapter = ?,
                        paragraph_id = ?, chunk_type = ?, silence_duration_s = ?, status = ?,
                        audio_path = ?, audio_validation_json = ?, proofread_json = ?,
                        auto_regen_count = ?, generation_token = ?, extra_json = ?
                    WHERE uid = ?
                    """,
                    (
                        normalized["id"],
                        normalized.get("speaker"),
                        normalized.get("text"),
                        normalized.get("instruct"),
                        normalized.get("chapter"),
                        normalized.get("paragraph_id"),
                        normalized.get("type"),
                        normalized.get("silence_duration_s"),
                        normalized.get("status"),
                        normalized.get("audio_path"),
                        json.dumps(normalized.get("audio_validation"), ensure_ascii=False)
                        if normalized.get("audio_validation") is not None else None,
                        json.dumps(normalized.get("proofread"), ensure_ascii=False)
                        if normalized.get("proofread") is not None else None,
                        int(normalized.get("auto_regen_count") or 0),
                        normalized.get("generation_token"),
                        self._encode_extra_json(normalized),
                        uid,
                    ),
                )
                updated.append(self._row_to_chunk(conn.execute(
                    """
                    SELECT uid, ordinal, speaker, text, instruct, chapter, paragraph_id,
                           chunk_type, silence_duration_s, status, audio_path,
                           audio_validation_json, proofread_json, auto_regen_count,
                           generation_token, extra_json
                    FROM chunks WHERE uid = ?
                    """,
                    (uid,),
                ).fetchone()))
            conn.execute(
                "INSERT OR REPLACE INTO metadata(key, value) VALUES('last_write_at', ?)",
                (str(time.time()),),
            )
        return updated

    def _claim_generation_tx(self, conn, uids, token):
        claimed = []
        normalized = [str(uid).strip() for uid in (uids or []) if str(uid).strip()]
        if not normalized:
            return claimed
        with conn:
            for uid in normalized:
                row = conn.execute(
                    """
                    SELECT uid, ordinal, speaker, text, instruct, chapter, paragraph_id,
                           chunk_type, silence_duration_s, status, audio_path,
                           audio_validation_json, proofread_json, auto_regen_count,
                           generation_token, extra_json
                    FROM chunks WHERE uid = ?
                    """,
                    (uid,),
                ).fetchone()
                if row is None:
                    continue
                chunk = self._row_to_chunk(row)
                existing_token = chunk.get("generation_token")
                if existing_token and existing_token != token:
                    continue
                chunk["status"] = "generating"
                chunk["generation_token"] = token
                normalized_chunk = self._normalize_chunk(chunk, chunk["id"])
                conn.execute(
                    """
                    UPDATE chunks
                    SET ordinal = ?, speaker = ?, text = ?, instruct = ?, chapter = ?,
                        paragraph_id = ?, chunk_type = ?, silence_duration_s = ?, status = ?,
                        audio_path = ?, audio_validation_json = ?, proofread_json = ?,
                        auto_regen_count = ?, generation_token = ?, extra_json = ?
                    WHERE uid = ?
                    """,
                    (
                        normalized_chunk["id"],
                        normalized_chunk.get("speaker"),
                        normalized_chunk.get("text"),
                        normalized_chunk.get("instruct"),
                        normalized_chunk.get("chapter"),
                        normalized_chunk.get("paragraph_id"),
                        normalized_chunk.get("type"),
                        normalized_chunk.get("silence_duration_s"),
                        normalized_chunk.get("status"),
                        normalized_chunk.get("audio_path"),
                        json.dumps(normalized_chunk.get("audio_validation"), ensure_ascii=False)
                        if normalized_chunk.get("audio_validation") is not None else None,
                        json.dumps(normalized_chunk.get("proofread"), ensure_ascii=False)
                        if normalized_chunk.get("proofread") is not None else None,
                        int(normalized_chunk.get("auto_regen_count") or 0),
                        normalized_chunk.get("generation_token"),
                        self._encode_extra_json(normalized_chunk),
                        uid,
                    ),
                )
                claimed.append(normalized_chunk)
            if claimed:
                conn.execute(
                    "INSERT OR REPLACE INTO metadata(key, value) VALUES('last_write_at', ?)",
                    (str(time.time()),),
                )
        return claimed

    def _reset_generation_tx(self, conn, uids, token=None, target_status="pending"):
        updated = []
        normalized = [str(uid).strip() for uid in (uids or []) if str(uid).strip()]
        if not normalized:
            return updated
        with conn:
            for uid in normalized:
                row = conn.execute(
                    """
                    SELECT uid, ordinal, speaker, text, instruct, chapter, paragraph_id,
                           chunk_type, silence_duration_s, status, audio_path,
                           audio_validation_json, proofread_json, auto_regen_count,
                           generation_token, extra_json
                    FROM chunks WHERE uid = ?
                    """,
                    (uid,),
                ).fetchone()
                if row is None:
                    continue
                chunk = self._row_to_chunk(row)
                if token is not None and chunk.get("generation_token") != token:
                    continue
                if chunk.get("status") not in {"generating", "finalizing"} and token is not None:
                    continue
                chunk["status"] = target_status
                chunk.pop("generation_token", None)
                normalized_chunk = self._normalize_chunk(chunk, chunk["id"])
                conn.execute(
                    """
                    UPDATE chunks
                    SET ordinal = ?, speaker = ?, text = ?, instruct = ?, chapter = ?,
                        paragraph_id = ?, chunk_type = ?, silence_duration_s = ?, status = ?,
                        audio_path = ?, audio_validation_json = ?, proofread_json = ?,
                        auto_regen_count = ?, generation_token = ?, extra_json = ?
                    WHERE uid = ?
                    """,
                    (
                        normalized_chunk["id"],
                        normalized_chunk.get("speaker"),
                        normalized_chunk.get("text"),
                        normalized_chunk.get("instruct"),
                        normalized_chunk.get("chapter"),
                        normalized_chunk.get("paragraph_id"),
                        normalized_chunk.get("type"),
                        normalized_chunk.get("silence_duration_s"),
                        normalized_chunk.get("status"),
                        normalized_chunk.get("audio_path"),
                        json.dumps(normalized_chunk.get("audio_validation"), ensure_ascii=False)
                        if normalized_chunk.get("audio_validation") is not None else None,
                        json.dumps(normalized_chunk.get("proofread"), ensure_ascii=False)
                        if normalized_chunk.get("proofread") is not None else None,
                        int(normalized_chunk.get("auto_regen_count") or 0),
                        normalized_chunk.get("generation_token"),
                        self._encode_extra_json(normalized_chunk),
                        uid,
                    ),
                )
                updated.append(normalized_chunk)
            if updated:
                conn.execute(
                    "INSERT OR REPLACE INTO metadata(key, value) VALUES('last_write_at', ?)",
                    (str(time.time()),),
                )
        return updated

    def _enqueue_audio_finalize_task_tx(self, conn, task):
        normalized = self._normalize_audio_finalize_task(task)
        if normalized is None:
            return None
        now = time.time()
        with conn:
            row = conn.execute(
                """
                SELECT id, chunk_uid, generation_token, temp_wav_path, attempt, speaker, text,
                       status, created_at, updated_at, last_error
                FROM audio_finalize_queue
                WHERE chunk_uid = ? AND generation_token = ? AND temp_wav_path = ?
                ORDER BY id DESC
                LIMIT 1
                """,
                (
                    normalized.get("chunk_uid"),
                    normalized.get("generation_token"),
                    normalized.get("temp_wav_path"),
                ),
            ).fetchone()
            if row is not None:
                conn.execute(
                    """
                    UPDATE audio_finalize_queue
                    SET attempt = ?, speaker = ?, text = ?, status = 'queued',
                        updated_at = ?, last_error = NULL
                    WHERE id = ?
                    """,
                    (
                        int(normalized.get("attempt") or 0),
                        normalized.get("speaker"),
                        normalized.get("text"),
                        now,
                        int(row["id"]),
                    ),
                )
                claimed_row = conn.execute(
                    """
                    SELECT id, chunk_uid, generation_token, temp_wav_path, attempt, speaker, text,
                           status, created_at, updated_at, last_error
                    FROM audio_finalize_queue
                    WHERE id = ?
                    """,
                    (int(row["id"]),),
                ).fetchone()
                return self._row_to_audio_finalize_task(claimed_row)

            conn.execute(
                """
                INSERT INTO audio_finalize_queue(
                    chunk_uid, generation_token, temp_wav_path, attempt, speaker, text,
                    status, created_at, updated_at, last_error
                ) VALUES (?, ?, ?, ?, ?, ?, 'queued', ?, ?, NULL)
                """,
                (
                    normalized.get("chunk_uid"),
                    normalized.get("generation_token"),
                    normalized.get("temp_wav_path"),
                    int(normalized.get("attempt") or 0),
                    normalized.get("speaker"),
                    normalized.get("text"),
                    now,
                    now,
                ),
            )
            task_id = int(conn.execute("SELECT last_insert_rowid() AS id").fetchone()["id"])
        with conn:
            row = conn.execute(
                """
                SELECT id, chunk_uid, generation_token, temp_wav_path, attempt, speaker, text,
                       status, created_at, updated_at, last_error
                FROM audio_finalize_queue
                WHERE id = ?
                """,
                (task_id,),
            ).fetchone()
        return self._row_to_audio_finalize_task(row)

    def _claim_next_audio_finalize_task_tx(self, conn):
        now = time.time()
        with conn:
            row = conn.execute(
                """
                SELECT id, chunk_uid, generation_token, temp_wav_path, attempt, speaker, text,
                       status, created_at, updated_at, last_error
                FROM audio_finalize_queue
                WHERE status = 'queued'
                ORDER BY id ASC
                LIMIT 1
                """
            ).fetchone()
            if row is None:
                return None
            conn.execute(
                """
                UPDATE audio_finalize_queue
                SET status = 'processing', updated_at = ?, last_error = NULL
                WHERE id = ?
                """,
                (now, int(row["id"])),
            )
            claimed_row = conn.execute(
                """
                SELECT id, chunk_uid, generation_token, temp_wav_path, attempt, speaker, text,
                       status, created_at, updated_at, last_error
                FROM audio_finalize_queue
                WHERE id = ?
                """,
                (int(row["id"]),),
            ).fetchone()
        return self._row_to_audio_finalize_task(claimed_row)

    def _complete_audio_finalize_task_tx(self, conn, task_id):
        normalized_id = int(task_id or 0)
        if normalized_id <= 0:
            return False
        with conn:
            conn.execute("DELETE FROM audio_finalize_queue WHERE id = ?", (normalized_id,))
        return True

    def _fail_audio_finalize_task_tx(self, conn, task_id, error=None, requeue=False):
        normalized_id = int(task_id or 0)
        if normalized_id <= 0:
            return False
        now = time.time()
        with conn:
            if requeue:
                conn.execute(
                    """
                    UPDATE audio_finalize_queue
                    SET status = 'queued', updated_at = ?, last_error = ?
                    WHERE id = ?
                    """,
                    (now, "" if error is None else str(error), normalized_id),
                )
            else:
                conn.execute("DELETE FROM audio_finalize_queue WHERE id = ?", (normalized_id,))
        return True

    def _clear_audio_finalize_tasks_tx(self, conn, generation_token=None, uids=None):
        clauses = []
        params = []
        normalized_token = str(generation_token or "").strip()
        normalized_uids = [str(uid).strip() for uid in (uids or []) if str(uid).strip()]
        if normalized_token:
            clauses.append("generation_token = ?")
            params.append(normalized_token)
        if normalized_uids:
            clauses.append("chunk_uid IN (" + ",".join("?" for _ in normalized_uids) + ")")
            params.extend(normalized_uids)
        query = "DELETE FROM audio_finalize_queue"
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        with conn:
            cursor = conn.execute(query, params)
        return int(cursor.rowcount or 0)

    def _prepare_chunk_for_regeneration_tx(self, conn, uid):
        normalized = str(uid or "").strip()
        if not normalized:
            return None
        result = self._patch_chunks_tx(conn, [{
            "uid": normalized,
            "expected": {},
            "fields": {
                "audio_path": None,
                "audio_validation": None,
                "status": "pending",
                "auto_regen_count": 0,
            },
            "clear_fields": ["generation_token", "proofread"],
        }])
        return result[0] if result else None

    def _delete_chunk_tx(self, conn, uid):
        normalized = str(uid or "").strip()
        if not normalized:
            return None
        with conn:
            rows = conn.execute(
                """
                SELECT uid, ordinal, speaker, text, instruct, chapter, paragraph_id,
                       chunk_type, silence_duration_s, status, audio_path,
                       audio_validation_json, proofread_json, auto_regen_count,
                       generation_token, extra_json
                FROM chunks
                ORDER BY ordinal ASC
                """
            ).fetchall()
            chunks = [self._row_to_chunk(row) for row in rows]
            target_index = next((i for i, chunk in enumerate(chunks) if chunk.get("uid") == normalized), None)
            if target_index is None or len(chunks) <= 1:
                return None
            restore_after_uid = chunks[target_index - 1].get("uid") if target_index > 0 else None
            deleted = chunks.pop(target_index)
            conn.execute("DELETE FROM chunks")
            for index, chunk in enumerate(chunks):
                normalized_chunk = self._normalize_chunk(chunk, index)
                conn.execute(
                    """
                    INSERT INTO chunks(
                        uid, ordinal, speaker, text, instruct, chapter, paragraph_id,
                        chunk_type, silence_duration_s, status, audio_path,
                        audio_validation_json, proofread_json, auto_regen_count,
                        generation_token, extra_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    self._chunk_to_params(normalized_chunk),
                )
            conn.execute(
                "INSERT OR REPLACE INTO metadata(key, value) VALUES('last_write_at', ?)",
                (str(time.time()),),
            )
        return {
            "deleted": deleted,
            "restore_after_uid": restore_after_uid,
            "total": len(chunks),
        }

    def _delete_chapter_tx(self, conn, chapter):
        normalized = str(chapter or "").strip()
        if not normalized:
            return None
        with conn:
            rows = conn.execute(
                """
                SELECT uid, ordinal, speaker, text, instruct, chapter, paragraph_id,
                       chunk_type, silence_duration_s, status, audio_path,
                       audio_validation_json, proofread_json, auto_regen_count,
                       generation_token, extra_json
                FROM chunks
                ORDER BY ordinal ASC
                """
            ).fetchall()
            chunks = [self._row_to_chunk(row) for row in rows]
            deleted = [chunk for chunk in chunks if str(chunk.get("chapter") or "").strip() == normalized]
            if not deleted:
                return None
            keep = [chunk for chunk in chunks if str(chunk.get("chapter") or "").strip() != normalized]
            conn.execute("DELETE FROM chunks")
            for index, chunk in enumerate(keep):
                normalized_chunk = self._normalize_chunk(chunk, index)
                conn.execute(
                    """
                    INSERT INTO chunks(
                        uid, ordinal, speaker, text, instruct, chapter, paragraph_id,
                        chunk_type, silence_duration_s, status, audio_path,
                        audio_validation_json, proofread_json, auto_regen_count,
                        generation_token, extra_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    self._chunk_to_params(normalized_chunk),
                )
            conn.execute(
                "INSERT OR REPLACE INTO metadata(key, value) VALUES('last_write_at', ?)",
                (str(time.time()),),
            )
        return {
            "chapter": normalized,
            "deleted": deleted,
            "deleted_count": len(deleted),
            "total": len(keep),
        }

    def _replace_voice_profiles_tx(self, conn, rows):
        normalized = [row for row in (rows or []) if row]
        with conn:
            conn.execute("DELETE FROM voice_profiles")
            for row in normalized:
                conn.execute(
                    """
                    INSERT INTO voice_profiles(
                        speaker_key, display_name, voice_type, voice_name, character_style,
                        default_style, alias, seed, ref_audio, ref_text, generated_ref_text,
                        adapter_id, adapter_path, description, narrates, extra_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    self._voice_profile_to_params(row),
                )
            self._bump_voice_state_revision_tx(conn)
            conn.execute(
                "INSERT OR REPLACE INTO metadata(key, value) VALUES('last_write_at', ?)",
                (str(time.time()),),
            )
        return len(normalized)

    def _upsert_voice_profiles_tx(self, conn, rows):
        updated = []
        with conn:
            for row in (rows or []):
                if not row:
                    continue
                conn.execute(
                    """
                    INSERT INTO voice_profiles(
                        speaker_key, display_name, voice_type, voice_name, character_style,
                        default_style, alias, seed, ref_audio, ref_text, generated_ref_text,
                        adapter_id, adapter_path, description, narrates, extra_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(speaker_key) DO UPDATE SET
                        display_name = excluded.display_name,
                        voice_type = excluded.voice_type,
                        voice_name = excluded.voice_name,
                        character_style = excluded.character_style,
                        default_style = excluded.default_style,
                        alias = excluded.alias,
                        seed = excluded.seed,
                        ref_audio = excluded.ref_audio,
                        ref_text = excluded.ref_text,
                        generated_ref_text = excluded.generated_ref_text,
                        adapter_id = excluded.adapter_id,
                        adapter_path = excluded.adapter_path,
                        description = excluded.description,
                        narrates = excluded.narrates,
                        extra_json = excluded.extra_json
                    """,
                    self._voice_profile_to_params(row),
                )
                updated.append(row)
            if updated:
                self._bump_voice_state_revision_tx(conn)
                conn.execute(
                    "INSERT OR REPLACE INTO metadata(key, value) VALUES('last_write_at', ?)",
                    (str(time.time()),),
                )
        return updated

    def _patch_voice_profile_tx(self, conn, speaker_key, display_name, fields, clear_fields):
        target_key = self._speaker_key(speaker_key)
        if not target_key:
            return None
        row = conn.execute(
            """
            SELECT speaker_key, display_name, voice_type, voice_name, character_style,
                   default_style, alias, seed, ref_audio, ref_text, generated_ref_text,
                   adapter_id, adapter_path, description, narrates, extra_json
            FROM voice_profiles
            WHERE speaker_key = ?
            """,
            (target_key,),
        ).fetchone()
        profile = self._row_to_voice_profile(row)["config"] if row is not None else self._default_voice_config()
        current_name = self._row_to_voice_profile(row)["display_name"] if row is not None else self._speaker_display_name(display_name or speaker_key)
        for field in (clear_fields or ()):
            profile.pop(field, None)
        profile.update(dict(fields or {}))
        normalized = self._normalize_voice_profile_row({"speaker": current_name, "config": profile})
        with conn:
            conn.execute(
                """
                INSERT INTO voice_profiles(
                    speaker_key, display_name, voice_type, voice_name, character_style,
                    default_style, alias, seed, ref_audio, ref_text, generated_ref_text,
                    adapter_id, adapter_path, description, narrates, extra_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(speaker_key) DO UPDATE SET
                    display_name = excluded.display_name,
                    voice_type = excluded.voice_type,
                    voice_name = excluded.voice_name,
                    character_style = excluded.character_style,
                    default_style = excluded.default_style,
                    alias = excluded.alias,
                    seed = excluded.seed,
                    ref_audio = excluded.ref_audio,
                    ref_text = excluded.ref_text,
                    generated_ref_text = excluded.generated_ref_text,
                    adapter_id = excluded.adapter_id,
                    adapter_path = excluded.adapter_path,
                    description = excluded.description,
                    narrates = excluded.narrates,
                    extra_json = excluded.extra_json
                """,
                self._voice_profile_to_params(normalized),
            )
            self._bump_voice_state_revision_tx(conn)
            conn.execute(
                "INSERT OR REPLACE INTO metadata(key, value) VALUES('last_write_at', ?)",
                (str(time.time()),),
            )
        return normalized

    def _set_voice_setting_tx(self, conn, key, value):
        normalized_key = str(key or "").strip()
        if not normalized_key:
            return None
        normalized_value = "" if value is None else str(value)
        with conn:
            conn.execute(
                "INSERT OR REPLACE INTO voice_settings(key, value) VALUES(?, ?)",
                (normalized_key, normalized_value),
            )
            self._bump_voice_state_revision_tx(conn)
            conn.execute(
                "INSERT OR REPLACE INTO metadata(key, value) VALUES('last_write_at', ?)",
                (str(time.time()),),
            )
        return normalized_value

    def _set_narrator_override_tx(self, conn, chapter, voice_key, voice_name):
        normalized_chapter = str(chapter or "").strip()
        if not normalized_chapter:
            return None
        with conn:
            if voice_name and self._speaker_key(voice_name) != self._speaker_key("NARRATOR"):
                conn.execute(
                    """
                    INSERT OR REPLACE INTO chapter_narrator_overrides(chapter, voice_key, voice_name)
                    VALUES (?, ?, ?)
                    """,
                    (normalized_chapter, self._speaker_key(voice_key or voice_name), self._speaker_display_name(voice_name)),
                )
            else:
                conn.execute("DELETE FROM chapter_narrator_overrides WHERE chapter = ?", (normalized_chapter,))
            self._bump_voice_state_revision_tx(conn)
            conn.execute(
                "INSERT OR REPLACE INTO metadata(key, value) VALUES('last_write_at', ?)",
                (str(time.time()),),
            )
        return self.get_narrator_overrides()

    def _replace_narrator_overrides_tx(self, conn, rows):
        normalized = [row for row in (rows or []) if str((row or {}).get("chapter") or "").strip()]
        with conn:
            conn.execute("DELETE FROM chapter_narrator_overrides")
            for row in normalized:
                conn.execute(
                    """
                    INSERT INTO chapter_narrator_overrides(chapter, voice_key, voice_name)
                    VALUES (?, ?, ?)
                    """,
                    (
                        str(row.get("chapter") or "").strip(),
                        self._speaker_key(row.get("voice_key") or row.get("voice_name")),
                        self._speaker_display_name(row.get("voice_name")),
                    ),
                )
            self._bump_voice_state_revision_tx(conn)
            conn.execute(
                "INSERT OR REPLACE INTO metadata(key, value) VALUES('last_write_at', ?)",
                (str(time.time()),),
            )
        return len(normalized)

    def _replace_auto_narrator_aliases_tx(self, conn, rows):
        normalized = [row for row in (rows or []) if row and row.get("speaker_key") and row.get("target_key")]
        with conn:
            conn.execute("DELETE FROM voice_auto_aliases")
            for row in normalized:
                conn.execute(
                    """
                    INSERT INTO voice_auto_aliases(speaker_key, speaker_name, target_key, target_name)
                    VALUES (?, ?, ?, ?)
                    """,
                    (
                        row.get("speaker_key"),
                        self._speaker_display_name(row.get("speaker_name")),
                        row.get("target_key"),
                        self._speaker_display_name(row.get("target_name")),
                    ),
                )
            self._bump_voice_state_revision_tx(conn)
            conn.execute(
                "INSERT OR REPLACE INTO metadata(key, value) VALUES('last_write_at', ?)",
                (str(time.time()),),
            )
        return len(normalized)

    def _replace_voice_state_snapshot_tx(self, conn, snapshot):
        payload = dict(snapshot or {})
        profiles = payload.get("profiles") or {}
        narrator_threshold = payload.get("narrator_threshold", 10)
        narrator_overrides = payload.get("narrator_overrides") or {}
        auto_aliases = payload.get("auto_narrator_aliases") or {}
        profile_rows = self._dedupe_voice_profile_rows(
            [
                {"speaker": speaker, "config": config}
                for speaker, config in dict(profiles).items()
            ]
        )
        override_rows = [
            {
                "chapter": str(chapter or "").strip(),
                "voice_key": self._speaker_key(voice),
                "voice_name": self._speaker_display_name(voice),
            }
            for chapter, voice in dict(narrator_overrides).items()
            if str(chapter or "").strip()
        ]
        alias_rows = [
            {
                "speaker_key": self._speaker_key(speaker),
                "speaker_name": self._speaker_display_name(speaker),
                "target_key": self._speaker_key(target),
                "target_name": self._speaker_display_name(target),
            }
            for speaker, target in dict(auto_aliases).items()
            if self._speaker_key(speaker) and self._speaker_key(target)
        ]
        with conn:
            conn.execute("DELETE FROM voice_profiles")
            for row in profile_rows:
                conn.execute(
                    """
                    INSERT INTO voice_profiles(
                        speaker_key, display_name, voice_type, voice_name, character_style,
                        default_style, alias, seed, ref_audio, ref_text, generated_ref_text,
                        adapter_id, adapter_path, description, narrates, extra_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    self._voice_profile_to_params(row),
                )
            conn.execute("DELETE FROM voice_settings")
            conn.execute(
                "INSERT OR REPLACE INTO voice_settings(key, value) VALUES(?, ?)",
                ("narrator_threshold", str(int(narrator_threshold or 0))),
            )
            conn.execute("DELETE FROM chapter_narrator_overrides")
            for row in override_rows:
                conn.execute(
                    """
                    INSERT INTO chapter_narrator_overrides(chapter, voice_key, voice_name)
                    VALUES (?, ?, ?)
                    """,
                    (
                        str(row.get("chapter") or "").strip(),
                        self._speaker_key(row.get("voice_key") or row.get("voice_name")),
                        self._speaker_display_name(row.get("voice_name")),
                    ),
                )
            conn.execute("DELETE FROM voice_auto_aliases")
            for row in alias_rows:
                conn.execute(
                    """
                    INSERT INTO voice_auto_aliases(speaker_key, speaker_name, target_key, target_name)
                    VALUES (?, ?, ?, ?)
                    """,
                    (
                        row.get("speaker_key"),
                        self._speaker_display_name(row.get("speaker_name")),
                        row.get("target_key"),
                        self._speaker_display_name(row.get("target_name")),
                    ),
                )
            self._bump_voice_state_revision_tx(conn)
            conn.execute(
                "INSERT OR REPLACE INTO metadata(key, value) VALUES('last_write_at', ?)",
                (str(time.time()),),
            )
        return self._load_voice_state_snapshot_tx(conn)

    def _refresh_auto_narrator_aliases_from_chunks_tx(self, conn, narrator_threshold, narrator_name):
        threshold = max(0, int(narrator_threshold or 0))
        narrator_display = self._speaker_display_name(narrator_name)
        if threshold <= 0 or not narrator_display:
            return self._replace_auto_narrator_aliases_tx(conn, [])

        voice_config = self.load_voice_config()
        line_counts = self.get_voice_summary().get("line_counts", {})
        rows = []
        narrator_key = self._speaker_key("NARRATOR")
        for speaker_name, count in line_counts.items():
            speaker_key = self._speaker_key(speaker_name)
            if not speaker_key or speaker_key == narrator_key:
                continue
            config = dict(voice_config.get(speaker_name) or {})
            if str(config.get("alias") or "").strip():
                continue
            try:
                parsed_count = int(count or 0)
            except (TypeError, ValueError):
                parsed_count = 0
            if parsed_count < threshold:
                rows.append({
                    "speaker_key": speaker_key,
                    "speaker_name": self._speaker_display_name(speaker_name),
                    "target_key": self._speaker_key(narrator_display),
                    "target_name": narrator_display,
                })
        return self._replace_auto_narrator_aliases_tx(conn, rows)

    def _chunk_to_params(self, chunk):
        return (
            chunk["uid"],
            chunk["id"],
            chunk.get("speaker"),
            chunk.get("text"),
            chunk.get("instruct"),
            chunk.get("chapter"),
            chunk.get("paragraph_id"),
            chunk.get("type"),
            chunk.get("silence_duration_s"),
            chunk.get("status"),
            chunk.get("audio_path"),
            json.dumps(chunk.get("audio_validation"), ensure_ascii=False)
            if chunk.get("audio_validation") is not None else None,
            json.dumps(chunk.get("proofread"), ensure_ascii=False)
            if chunk.get("proofread") is not None else None,
            int(chunk.get("auto_regen_count") or 0),
            chunk.get("generation_token"),
            self._encode_extra_json(chunk),
        )

    def _row_to_chunk(self, row):
        chunk = {
            "id": int(row["ordinal"]),
            "uid": row["uid"],
            "speaker": row["speaker"],
            "text": row["text"],
            "instruct": row["instruct"],
            "status": row["status"],
            "audio_path": row["audio_path"],
            "auto_regen_count": int(row["auto_regen_count"] or 0),
        }
        if row["chapter"]:
            chunk["chapter"] = row["chapter"]
        if row["paragraph_id"]:
            chunk["paragraph_id"] = row["paragraph_id"]
        if row["chunk_type"]:
            chunk["type"] = row["chunk_type"]
        if row["silence_duration_s"] is not None:
            chunk["silence_duration_s"] = row["silence_duration_s"]
        if row["audio_validation_json"]:
            chunk["audio_validation"] = json.loads(row["audio_validation_json"])
        else:
            chunk["audio_validation"] = None
        if row["proofread_json"]:
            chunk["proofread"] = json.loads(row["proofread_json"])
        if row["generation_token"]:
            chunk["generation_token"] = row["generation_token"]
        extra_json = row["extra_json"]
        if extra_json:
            chunk.update(json.loads(extra_json))
        return chunk

    def _normalize_chunk(self, chunk, index):
        normalized = dict(chunk or {})
        normalized["id"] = int(index)
        normalized["uid"] = str(normalized.get("uid") or uuid.uuid4().hex)
        normalized.setdefault("speaker", "NARRATOR")
        normalized.setdefault("text", "")
        normalized.setdefault("instruct", "")
        normalized.setdefault("status", "pending")
        normalized.setdefault("audio_path", None)
        normalized["audio_validation"] = normalized.get("audio_validation")
        normalized["auto_regen_count"] = int(normalized.get("auto_regen_count") or 0)
        if "silence_duration_s" in normalized and normalized["silence_duration_s"] is not None:
            normalized["silence_duration_s"] = float(normalized["silence_duration_s"])
        return normalized

    def _normalize_audio_finalize_task(self, task):
        payload = dict(task or {})
        chunk_uid = str(payload.get("chunk_uid") or "").strip()
        temp_wav_path = str(payload.get("temp_wav_path") or "").strip()
        if not chunk_uid or not temp_wav_path:
            return None
        return {
            "chunk_uid": chunk_uid,
            "generation_token": str(payload.get("generation_token") or "").strip() or None,
            "temp_wav_path": temp_wav_path,
            "attempt": int(payload.get("attempt") or 0),
            "speaker": str(payload.get("speaker") or "").strip() or None,
            "text": str(payload.get("text") or ""),
        }

    @staticmethod
    def _row_to_audio_finalize_task(row):
        if row is None:
            return None
        return {
            "id": int(row["id"]),
            "chunk_uid": row["chunk_uid"],
            "generation_token": row["generation_token"],
            "temp_wav_path": row["temp_wav_path"],
            "attempt": int(row["attempt"] or 0),
            "speaker": row["speaker"],
            "text": row["text"] or "",
            "status": row["status"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "last_error": row["last_error"],
        }

    @staticmethod
    def _chunk_audio_fingerprint(chunk):
        validation = (chunk or {}).get("audio_validation") or {}
        return "|".join([
            str((chunk or {}).get("audio_path") or ""),
            str(validation.get("file_size_bytes") or 0),
            str(validation.get("actual_duration_sec") or 0),
            str((chunk or {}).get("status") or ""),
        ])

    @staticmethod
    def _default_voice_config():
        return {
            "type": "design",
            "voice": "Ryan",
            "character_style": "",
            "default_style": "",
            "alias": "",
            "seed": "-1",
            "ref_audio": None,
            "ref_text": None,
            "generated_ref_text": None,
            "adapter_id": None,
            "adapter_path": None,
            "description": "",
            "narrates": False,
        }

    @staticmethod
    def _normalize_whitespace(value):
        return " ".join(str(value or "").strip().split())

    @classmethod
    def _speaker_key(cls, value):
        return cls._normalize_whitespace(value).casefold()

    @classmethod
    def _speaker_display_name(cls, value):
        display = cls._normalize_whitespace(value)
        if cls._speaker_key(display) == cls._speaker_key("NARRATOR"):
            return "NARRATOR"
        return display

    def _normalize_voice_profile_row(self, row):
        payload = dict(row or {})
        speaker = payload.get("speaker") or payload.get("display_name") or payload.get("name")
        config = dict(payload.get("config") or {})
        display_name = self._speaker_display_name(speaker)
        speaker_key = self._speaker_key(display_name)
        if not speaker_key:
            return None
        normalized = self._default_voice_config()
        normalized.update(config)
        if "narrates" in normalized:
            normalized["narrates"] = bool(normalized.get("narrates"))
        return {
            "speaker_key": speaker_key,
            "display_name": display_name,
            "config": normalized,
        }

    def _dedupe_voice_profile_rows(self, rows):
        deduped = {}
        for row in (rows or []):
            normalized = self._normalize_voice_profile_row(row)
            if not normalized:
                continue
            deduped[normalized["speaker_key"]] = normalized
        return list(deduped.values())

    def _voice_profile_to_params(self, row):
        config = dict((row or {}).get("config") or {})
        return (
            row.get("speaker_key"),
            row.get("display_name"),
            config.get("type"),
            config.get("voice"),
            config.get("character_style"),
            config.get("default_style"),
            config.get("alias"),
            config.get("seed"),
            config.get("ref_audio"),
            config.get("ref_text"),
            config.get("generated_ref_text"),
            config.get("adapter_id"),
            config.get("adapter_path"),
            config.get("description"),
            1 if bool(config.get("narrates")) else 0,
            self._encode_voice_extra_json(config),
        )

    def _row_to_voice_profile(self, row):
        if row is None:
            return None
        config = self._default_voice_config()
        config.update({
            "type": row["voice_type"] or config.get("type"),
            "voice": row["voice_name"] if row["voice_name"] is not None else config.get("voice"),
            "character_style": row["character_style"] if row["character_style"] is not None else config.get("character_style"),
            "default_style": row["default_style"] if row["default_style"] is not None else config.get("default_style"),
            "alias": row["alias"] if row["alias"] is not None else config.get("alias"),
            "seed": row["seed"] if row["seed"] is not None else config.get("seed"),
            "ref_audio": row["ref_audio"],
            "ref_text": row["ref_text"],
            "generated_ref_text": row["generated_ref_text"],
            "adapter_id": row["adapter_id"],
            "adapter_path": row["adapter_path"],
            "description": row["description"] if row["description"] is not None else config.get("description"),
            "narrates": bool(row["narrates"]) if row["narrates"] is not None else False,
        })
        if row["extra_json"]:
            config.update(json.loads(row["extra_json"]))
        return {
            "speaker_key": row["speaker_key"],
            "display_name": row["display_name"],
            "config": config,
        }

    def _get_voice_state_revision(self, conn):
        row = conn.execute(
            "SELECT value FROM metadata WHERE key = 'voice_state_revision'"
        ).fetchone()
        try:
            return int((row["value"] if row is not None else 0) or 0)
        except (TypeError, ValueError, KeyError):
            return 0

    def _bump_voice_state_revision_tx(self, conn):
        next_revision = self._get_voice_state_revision(conn) + 1
        conn.execute(
            "INSERT OR REPLACE INTO metadata(key, value) VALUES('voice_state_revision', ?)",
            (str(next_revision),),
        )
        return next_revision

    def _load_voice_state_snapshot_tx(self, conn):
        rows = conn.execute(
            """
            SELECT speaker_key, display_name, voice_type, voice_name, character_style,
                   default_style, alias, seed, ref_audio, ref_text, generated_ref_text,
                   adapter_id, adapter_path, description, narrates, extra_json
            FROM voice_profiles
            ORDER BY display_name COLLATE NOCASE ASC
            """
        ).fetchall()
        profiles = {}
        for row in rows:
            profile = self._row_to_voice_profile(row)
            profiles[profile["display_name"]] = profile["config"]
        settings = {
            row["key"]: row["value"]
            for row in conn.execute("SELECT key, value FROM voice_settings").fetchall()
        }
        try:
            narrator_threshold = int(settings.get("narrator_threshold", "10") or 10)
        except (TypeError, ValueError):
            narrator_threshold = 10
        narrator_overrides = {
            str(row["chapter"] or "").strip(): str(row["voice_name"] or "").strip()
            for row in conn.execute(
                "SELECT chapter, voice_name FROM chapter_narrator_overrides ORDER BY chapter COLLATE NOCASE ASC"
            ).fetchall()
            if str(row["chapter"] or "").strip()
        }
        auto_aliases = {
            str(row["speaker_name"] or "").strip(): str(row["target_name"] or "").strip()
            for row in conn.execute(
                "SELECT speaker_name, target_name FROM voice_auto_aliases ORDER BY speaker_name COLLATE NOCASE ASC"
            ).fetchall()
            if str(row["speaker_name"] or "").strip() and str(row["target_name"] or "").strip()
        }
        return {
            "profiles": profiles,
            "narrator_threshold": narrator_threshold,
            "narrator_overrides": narrator_overrides,
            "auto_narrator_aliases": auto_aliases,
            "revision": self._get_voice_state_revision(conn),
        }

    def _audit_voice_state_write(self, conn, command_name, payload):
        if command_name not in {
            "replace_voice_profiles",
            "upsert_voice_profiles",
            "patch_voice_profile",
            "set_voice_setting",
            "set_narrator_override",
            "replace_narrator_overrides",
            "replace_auto_narrator_aliases",
            "refresh_auto_narrator_aliases_from_chunks",
            "replace_voice_state_snapshot",
        }:
            return
        self._append_voice_audit_entry(
            {
                "event": "voice_state_write",
                "command": command_name,
                "reason": (payload or {}).get("reason"),
            },
            snapshot=self._load_voice_state_snapshot_tx(conn),
        )

    @staticmethod
    def _encode_voice_extra_json(config):
        reserved = {
            "type",
            "voice",
            "character_style",
            "default_style",
            "alias",
            "seed",
            "ref_audio",
            "ref_text",
            "generated_ref_text",
            "adapter_id",
            "adapter_path",
            "description",
            "narrates",
        }
        extra = {
            key: value
            for key, value in (config or {}).items()
            if key not in reserved
        }
        if not extra:
            return None
        return json.dumps(extra, ensure_ascii=False)

    def _resolve_voice_speaker_for_store(self, speaker, *, chapter="", voice_config=None, narrator_overrides=None, auto_narrator_aliases=None):
        voice_config = voice_config if isinstance(voice_config, dict) else self.load_voice_config()
        narrator_overrides = narrator_overrides if isinstance(narrator_overrides, dict) else self.get_narrator_overrides()
        auto_narrator_aliases = auto_narrator_aliases if isinstance(auto_narrator_aliases, dict) else self.get_auto_narrator_aliases()
        lookup = {}
        for name in voice_config.keys():
            key = self._speaker_key(name)
            if key and key not in lookup:
                lookup[key] = name
        original = self._speaker_display_name(speaker)
        if not original:
            return ""
        current = lookup.get(self._speaker_key(original), original)
        seen = {self._speaker_key(current)}
        manual_alias_used = False
        while current:
            voice_data = dict(voice_config.get(current) or {})
            alias = self._speaker_display_name(voice_data.get("alias"))
            if not alias:
                break
            target = lookup.get(self._speaker_key(alias), alias)
            if not target or self._speaker_key(target) == self._speaker_key(current):
                break
            manual_alias_used = True
            target_key = self._speaker_key(target)
            if target_key in seen:
                return original
            seen.add(target_key)
            current = target

        resolved = current or original
        if not manual_alias_used:
            stored_target = ""
            for raw_speaker, raw_target in (auto_narrator_aliases or {}).items():
                if self._speaker_key(raw_speaker) == self._speaker_key(resolved):
                    stored_target = self._speaker_display_name(raw_target)
                    break
            if stored_target and self._speaker_key(stored_target) != self._speaker_key(resolved):
                resolved = stored_target

        if self._speaker_key(resolved) == self._speaker_key("NARRATOR"):
            override = self._speaker_display_name((narrator_overrides or {}).get(str(chapter or "").strip()))
            if override and self._speaker_key(override) != self._speaker_key("NARRATOR"):
                return override
        return resolved

    def _resolve_chunk_ref(self, chunk_ref):
        base_ref = "" if chunk_ref is None else str(chunk_ref).strip()
        if not base_ref:
            return None
        with self._connect() as conn:
            row = conn.execute(
                "SELECT uid, ordinal FROM chunks WHERE uid = ?",
                (base_ref,),
            ).fetchone()
            if row is not None:
                return {"uid": row["uid"], "id": int(row["ordinal"] or 0)}
            try:
                numeric_ref = int(base_ref)
            except (TypeError, ValueError):
                return None
            row = conn.execute(
                "SELECT uid, ordinal FROM chunks WHERE ordinal = ?",
                (numeric_ref,),
            ).fetchone()
            if row is not None:
                return {"uid": row["uid"], "id": int(row["ordinal"] or 0)}
        return None

    def _encode_extra_json(self, chunk):
        extra = {
            key: value
            for key, value in (chunk or {}).items()
            if key not in self._RESERVED_FIELDS
        }
        if not extra:
            return None
        return json.dumps(extra, ensure_ascii=False)

    def _log_command(self, name, payload):
        entry = {
            "at": time.time(),
            "command": name,
            "reason": (payload or {}).get("reason"),
        }
        with self._log_lock:
            log_dir = os.path.dirname(self.queue_log_path) or "."
            if not os.path.isdir(self.root_dir):
                return
            if log_dir and not os.path.isdir(log_dir):
                return
            with open(self.queue_log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def _append_voice_audit_entry(self, entry, *, snapshot=None):
        if not self.voice_audit_logging_enabled:
            return
        if not os.path.isdir(self.root_dir):
            return
        audit_dir = os.path.dirname(self.voice_audit_log_path) or "."
        if audit_dir and not os.path.isdir(audit_dir):
            return
        snapshot_payload = snapshot or self.load_voice_state_snapshot()
        serialized_snapshot = json.dumps(snapshot_payload, sort_keys=True, ensure_ascii=False)
        record = {
            "at": time.time(),
            **dict(entry or {}),
            "voice_state_revision": snapshot_payload.get("revision", 0),
            "snapshot_hash": hashlib.sha256(serialized_snapshot.encode("utf-8")).hexdigest(),
            "snapshot": snapshot_payload,
        }
        with self._log_lock:
            with open(self.voice_audit_log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")


def create_script_store(**kwargs):
    backend = str(kwargs.pop("backend", "sqlite") or "sqlite").strip().lower()
    if backend != "sqlite":
        raise ValueError(f"Unsupported script store backend: {backend}")
    return SQLiteScriptStore(**kwargs)
