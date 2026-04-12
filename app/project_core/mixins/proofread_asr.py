"""ASR transcription, similarity scoring, and proofreading state management.

This mixin provides:
- Local ASR engine access and transcription caching hooks,
- text normalization/scoring for chunk-vs-transcript comparison, and
- commit/reset flows for proofread acceptance, rejection, and retries.
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


class ProjectProofreadASRMixin:
        """Compute and persist per-chunk proofreading outcomes from ASR output."""
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
            try:
                latest_loaded = self.load_chunks_raw()
                if isinstance(latest_loaded, list):
                    latest = latest_loaded
            except Exception:
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
            try:
                latest_loaded = self.load_chunks_raw()
                if isinstance(latest_loaded, list):
                    latest = latest_loaded
            except Exception:
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
                chunks = self.load_chunks_raw()
                if not chunks:
                    return None

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
                chunks = self.load_chunks_raw()
                if not chunks:
                    return None

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
                chunks = self.load_chunks_raw()
                if not chunks:
                    return {
                        "discarded": 0,
                        "preserved_transcripts": 0,
                        "cleared_transcripts": 0,
                        "chapter": chapter,
                    }

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

        def proofread_chunks(self, chapter=None, threshold=1.0, progress_callback=None):
            def emit_progress(payload):
                if progress_callback:
                    progress_callback(payload)

            with self._chunks_lock:
                chunks = self.load_chunks_raw()
                if not chunks:
                    return {
                        "processed": 0,
                        "skipped": 0,
                        "auto_failed": 0,
                        "passed": 0,
                        "failed": 0,
                        "chapter": chapter,
                        "threshold": float(threshold),
                    }

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
                chunks = self.load_chunks_raw()
                if not chunks:
                    return {
                        "cleared": 0,
                        "failed_candidates": 0,
                        "ungraded_with_audio": 0,
                        "chapter": chapter,
                        "threshold": float(threshold),
                    }

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
