"""Chunk CRUD and structural editing operations used by the editor flows.

This mixin owns user-facing chunk mutations such as insert/delete/restore,
chapter deletion, stale audio-link invalidation, and light structural cleanup
(split long segments, merge short orphan segments).
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


class ProjectChunkEditingMixin:
        """Mutate chunk structure while preserving ids/uids and edit safety."""
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

        def insert_chunk(self, after_ref):
            """Insert an empty chunk after the given index. Returns the new chunk list."""
            with self._chunks_lock:
                chunks = self.load_chunks_raw()
                if not chunks:
                    return None
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
                chunks = self.load_chunks_raw()
                if not chunks:
                    return None
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
            deleted_files = []
            result = self.delete_chunk_by_uid(
                (self.get_chunk_raw(chunk_ref) or {}).get("uid")
            )
            if result is None:
                return None
            deleted = result.get("deleted") or {}
            audio_path = (deleted.get("audio_path") or "").strip()
            if audio_path:
                deleted_files.append(audio_path)

            for relative_path in deleted_files:
                full_ap = os.path.join(self.root_dir, relative_path)
                if not os.path.exists(full_ap):
                    continue
                try:
                    os.remove(full_ap)
                except OSError:
                    pass

            return deleted, self.load_chunks_raw(), result.get("restore_after_uid")

        def delete_chapter(self, chapter_name):
            """Delete all chunks belonging to a chapter and remove their audio files.
            Returns (deleted_count, updated_chunks) or None on failure.
            """
            if not chapter_name or not isinstance(chapter_name, str) or not chapter_name.strip():
                return None
            chapter_name = chapter_name.strip()
            files_to_delete = []
            result = self.delete_chapter_by_name(chapter_name)
            if result is None:
                return None
            deleted = result.get("deleted") or []
            for chunk in deleted:
                ap = chunk.get("audio_path")
                if ap:
                    files_to_delete.append(ap)

            for relative_path in files_to_delete:
                full_ap = os.path.join(self.root_dir, relative_path)
                if not os.path.exists(full_ap):
                    continue
                try:
                    os.remove(full_ap)
                except OSError:
                    pass

            return result.get("deleted_count"), self.load_chunks_raw()

        def restore_chunk(self, at_index, chunk_data, after_uid=None):
            """Re-insert a chunk at a specific index. Returns the updated chunk list."""
            with self._chunks_lock:
                chunks = self.load_chunks_raw()
                if chunks is None:
                    return None

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
                chunks = self.load_chunks_raw()
                if not chunks:
                    return {"invalidated": 0, "duplicate_groups": 0, "kept": 0, "examples": []}

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
                chunks = self.load_chunks_raw()
                if not chunks:
                    return {
                        "changed": 0,
                        "total_chunks": 0,
                        "processed_scope": 0,
                        "chapter": chapter,
                        "max_words": max_words,
                    }

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
                        left_chunk["uid"] = left_chunk.get("uid") or self._new_chunk_uid()
                        left_chunk["status"] = "pending"
                        left_chunk["audio_path"] = None
                        left_chunk["audio_validation"] = None
                        left_chunk["auto_regen_count"] = 0
                        left_chunk.pop("generation_token", None)
                        self._clear_proofread_state(left_chunk)

                        right_chunk = copy.deepcopy(base_chunk)
                        right_chunk["text"] = right_text
                        right_chunk["uid"] = self._new_chunk_uid()
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
                chunks = self.load_chunks_raw()
                if not chunks:
                    return {
                        "changed": 0,
                        "total_chunks": 0,
                        "processed_scope": 0,
                        "chapter": chapter,
                        "min_words": min_words,
                    }

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
                chunks = self.load_chunks_raw()
                if not chunks:
                    return None

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
            chunk = self.get_chunk_raw(chunk_ref)
            if chunk is None:
                return None
            updated = self.prepare_chunk_for_regeneration_by_uid(chunk.get("uid"))
            if updated is None:
                return None
            audio_path = (chunk.get("audio_path") or "").strip()
            if audio_path:
                full_audio_path = os.path.join(self.root_dir, audio_path)
                if os.path.exists(full_audio_path):
                    try:
                        os.remove(full_audio_path)
                    except OSError:
                        pass
            return {"index": updated.get("id"), "chunk": updated}
