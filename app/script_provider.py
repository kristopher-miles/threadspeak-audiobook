import json
import os
import queue
import shutil
import sqlite3
import threading
import time
import uuid
from abc import ABC, abstractmethod

from project_core.chunking import script_entries_to_chunks
from script_store import load_script_document


SCRIPT_STORE_SCHEMA_VERSION = 1


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
    def replace_chunks(self, chunks, *, reason="replace_chunks", wait=True):
        raise NotImplementedError

    @abstractmethod
    def patch_chunks(self, patches, *, reason="patch_chunks", wait=True):
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
        archive_dir=None,
    ):
        self.root_dir = root_dir
        self.db_path = db_path
        self.queue_log_path = queue_log_path
        self.script_path = script_path
        self.legacy_chunks_path = legacy_chunks_path
        self.archive_dir = archive_dir or os.path.join(root_dir, "backups", "chunks")
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
            normalized.append({"uid": uid, "fields": fields})
        if not normalized:
            return 0
        return self._submit_command("patch_chunks", {"patches": normalized, "reason": reason}, wait=wait)

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
                "INSERT OR REPLACE INTO metadata(key, value) VALUES('schema_version', ?)",
                (str(SCRIPT_STORE_SCHEMA_VERSION),),
            )
            conn.commit()

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
            return 0
        updated = 0
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
                updated += 1
            conn.execute(
                "INSERT OR REPLACE INTO metadata(key, value) VALUES('last_write_at', ?)",
                (str(time.time()),),
            )
        return updated

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
            os.makedirs(os.path.dirname(self.queue_log_path) or ".", exist_ok=True)
            with open(self.queue_log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def create_script_store(**kwargs):
    backend = str(kwargs.pop("backend", "sqlite") or "sqlite").strip().lower()
    if backend != "sqlite":
        raise ValueError(f"Unsupported script store backend: {backend}")
    return SQLiteScriptStore(**kwargs)
