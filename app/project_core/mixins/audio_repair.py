"""Recovery tools for repairing broken/missing chunk-to-audio associations.

This mixin scans generated/discarded voiceline assets, performs candidate
matching (filename hints plus optional ASR similarity), and safely relinks
valid clips back to chunks while preserving write integrity.
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


class ProjectAudioRepairMixin:
        """Repair chunk audio links using deterministic and ASR-assisted matching."""
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
            normalized_text = (text or "").lower()
            normalized_text = re.sub(r"['\u2018\u2019\u02bc]", "", normalized_text)
            normalized_text = re.sub(r"[^a-z0-9]+", " ", normalized_text)
            normalized_text = re.sub(r"\s+", " ", normalized_text).strip()
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
            normalized_text = (text or "").lower()
            normalized_text = re.sub(r"['\u2018\u2019\u02bc]", "", normalized_text)
            normalized_text = re.sub(r"[^a-z0-9]+", " ", normalized_text)
            normalized_text = re.sub(r"\s+", " ", normalized_text).strip()
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

            if not self.load_chunks_raw():
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
                chunks = self.load_chunks_raw()

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
