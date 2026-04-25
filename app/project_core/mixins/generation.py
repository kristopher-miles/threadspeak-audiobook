"""Chunk audio generation orchestration and batch/parallel execution helpers.

This mixin is responsible for:
- per-chunk TTS generation and output finalization,
- retry/auto-regeneration policy application, and
- grouped/batched generation flows with progress reporting.
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


class ProjectGenerationMixin:
        """Generate and persist chunk audio through TTS pipelines."""
        def _load_generation_chunks(self, chunk_refs):
            refs = [str(chunk_ref).strip() for chunk_ref in (chunk_refs or []) if str(chunk_ref).strip()]
            if not refs:
                return []
            by_uid = {
                str((chunk or {}).get("uid") or "").strip(): chunk
                for chunk in self.get_chunks_by_uids(refs)
                if str((chunk or {}).get("uid") or "").strip()
            }
            ordered = []
            for chunk_ref in refs:
                chunk = by_uid.get(chunk_ref)
                if chunk is None:
                    chunk = self.get_chunk_raw(chunk_ref)
                if chunk is not None:
                    ordered.append(chunk)
            return ordered

        def _speaker_line_counts_for_chunks(self, chunks):
            counts = Counter()
            for chunk in chunks or []:
                text = (chunk.get("text") or "").strip()
                if not text:
                    continue
                speaker_key = self._normalize_speaker_name(chunk.get("speaker"))
                if speaker_key:
                    counts[speaker_key] += 1
            return counts

        def _resolve_generation_speaker(
            self,
            chunk,
            voice_config,
            narrator_overrides=None,
            narrator_name=None,
        ):
            chapter = (chunk.get("chapter") or "").strip()
            effective_narrator = self._apply_narrator_override(
                narrator_name or "NARRATOR",
                chapter,
                narrator_overrides or self.get_narrator_overrides(),
            )
            effective_speaker = self._apply_narrator_override(
                chunk.get("speaker", ""),
                chapter,
                narrator_overrides or self.get_narrator_overrides(),
            )
            resolved = self.resolve_voice_speaker(
                effective_speaker,
                voice_config,
                chunks=(),
                narrator_name=effective_narrator,
            )
            resolved_key = self._normalize_speaker_name(resolved)
            base_narrator_key = self._normalize_speaker_name(narrator_name or "NARRATOR")
            literal_narrator_key = self._normalize_speaker_name("NARRATOR")
            effective_narrator_key = self._normalize_speaker_name(effective_narrator)

            if resolved_key in {base_narrator_key, literal_narrator_key} and effective_narrator_key != resolved_key:
                return self.resolve_voice_speaker(
                    effective_narrator,
                    voice_config,
                    chunks=(),
                    narrator_name=effective_narrator,
                )
            return resolved

        def _get_auto_regen_retry_attempts(self):
            tts_settings = self._load_tts_settings()
            if not tts_settings.get("auto_regenerate_bad_clips", False):
                return 0
            try:
                attempts = int(tts_settings.get("auto_regenerate_bad_clip_attempts", 3))
            except (TypeError, ValueError):
                return 0
            return attempts if attempts > 0 else 0

        def _should_request_auto_regen_retry(self, attempt, error=None):
            if attempt >= self._get_auto_regen_retry_attempts():
                return False
            normalized_error = str(error or "").strip().lower()
            if "audio is too long" in normalized_error:
                return False
            return True

        @staticmethod
        def _cleanup_temp_file(temp_path):
            if os.path.exists(temp_path):
                for attempt in range(3):
                    try:
                        os.remove(temp_path)
                        break
                    except OSError:
                        if attempt < 2:
                            time.sleep(0.1 * (attempt + 1))
                        else:
                            print(f"Warning: Could not delete temp file {temp_path}")

        def _store_generated_audio(self, temp_path, filename_base):
            audio_path = None

            try:
                segment = AudioSegment.from_file(temp_path)
                if len(segment) == 0:
                    raise RuntimeError("Generated audio has 0 duration")

                mp3_filename = f"{filename_base}.mp3"
                mp3_filepath = os.path.join(self.voicelines_dir, mp3_filename)
                segment.export(mp3_filepath, format="mp3")

                mp3_size = os.path.getsize(mp3_filepath) if os.path.exists(mp3_filepath) else 0
                if mp3_size < 1024:
                    print(f"MP3 export produced invalid file ({mp3_size} bytes) — ffmpeg likely lacks MP3 encoder (libmp3lame). Falling back to WAV.")
                    os.remove(mp3_filepath)
                    raise RuntimeError("MP3 export produced invalid file")

                audio_path = f"voicelines/{mp3_filename}"

            except Exception as e:
                if "invalid file" not in str(e).lower():
                    print(f"MP3 conversion failed (ffmpeg missing?): {e}")
                wav_filename = f"{filename_base}.wav"
                wav_filepath = os.path.join(self.voicelines_dir, wav_filename)
                shutil.copy(temp_path, wav_filepath)
                audio_path = f"voicelines/{wav_filename}"

            return audio_path

        @staticmethod
        def _chunk_audio_filename_base(chunk_uid, index, speaker, attempt=0):
            speaker_slug = sanitize_filename(speaker) or "speaker"
            stable_id = sanitize_filename((chunk_uid or "").strip()) or f"legacy_{index+1:04d}"
            filename_base = f"voiceline_{stable_id}_{speaker_slug}"
            if attempt > 0:
                filename_base = f"{filename_base}_retry{attempt}"
            return filename_base

        def _finalize_generated_audio(self, index, speaker, text, temp_path, attempt=0, chunk_uid=None):
            if not os.path.exists(temp_path) or os.path.getsize(temp_path) == 0:
                return {
                    "status": "error",
                    "audio_path": None,
                    "audio_validation": None,
                    "error": "Generated audio file is missing or empty",
                }

            print(f"Generated WAV size: {os.path.getsize(temp_path)} bytes")

            filename_base = self._chunk_audio_filename_base(chunk_uid, index, speaker, attempt=attempt)
            audio_path = self._store_generated_audio(temp_path, filename_base)
            full_audio_path = os.path.join(self.root_dir, audio_path)
            validation = validate_audio_clip(
                text=text,
                actual_duration_sec=get_audio_duration_seconds(temp_path),
                file_size_bytes=os.path.getsize(full_audio_path),
            ).to_dict()

            if validation["is_valid"]:
                return {
                    "status": "done",
                    "audio_path": audio_path,
                    "audio_validation": validation,
                    "error": None,
                    "proofread_cleared": True,
                }

            print(f"Chunk {index} failed audio sanity check: {validation['error']}")
            return {
                "status": "error",
                "audio_path": audio_path,
                "audio_validation": validation,
                "error": validation["error"],
                "proofread_cleared": True,
            }

        @staticmethod
        def _effective_generation_instruct(chunk, neutral_narrator=False):
            if neutral_narrator and chunk.get("speaker") == "NARRATOR":
                return ""
            return chunk.get("instruct", "")

        def _generate_chunk_audio_internal(self, index, attempt=0, generation_token=None, async_finalize=False,
                                           neutral_narrator=False, cancel_check=None):
            chunk = self.get_chunk_raw(index)
            if chunk is None:
                return False, "Invalid chunk index"

            chunk = self._claim_chunk_generation(chunk.get("uid"), generation_token)
            if not chunk:
                return False, "Invalid chunk index"

            try:
                engine = self.get_engine()
                if not engine:
                    self.patch_chunk_if(
                        chunk.get("uid"),
                        fields={"status": "error", "audio_validation": None, "auto_regen_count": attempt},
                        clear_fields=["generation_token"],
                        reason="generate_chunk_audio_engine_missing",
                    )
                    return False, "TTS engine not initialized"

                voice_config = self._load_voice_config()
                narrator_name = self._resolve_narrator_name(voice_config, [chunk])
                speaker = chunk["speaker"]
                resolved_speaker = self._resolve_generation_speaker(
                    chunk,
                    voice_config,
                    narrator_name=narrator_name,
                )
                voice_config = self.prepare_runtime_voice_config(voice_config, [resolved_speaker])
                text = chunk["text"]
                transformed_text, _ = apply_dictionary_to_text(text, self.load_dictionary_entries())
                instruct = self._effective_generation_instruct(chunk, neutral_narrator=neutral_narrator)
                auto_regen_retry_attempts = self._get_auto_regen_retry_attempts()
                display_index = int(chunk.get("id") or 0)
                chunk_uid = chunk.get("uid")

                print(
                    f"Generating chunk {display_index}: speaker={speaker}, resolved_speaker={resolved_speaker}, "
                    f"instruct='{instruct}', text='{transformed_text[:50]}...'"
                )

                temp_path = self._spool_audio_full_path(chunk_uid, generation_token, attempt=attempt)

                generate_voice_kwargs = {}
                try:
                    generate_voice_signature = inspect.signature(engine.generate_voice)
                except (TypeError, ValueError):
                    generate_voice_signature = None
                if generate_voice_signature and "cancel_check" in generate_voice_signature.parameters:
                    generate_voice_kwargs["cancel_check"] = cancel_check
                success = engine.generate_voice(
                    transformed_text,
                    instruct,
                    resolved_speaker,
                    voice_config,
                    temp_path,
                    **generate_voice_kwargs,
                )

                if generation_token is not None and not self.chunk_has_generation_token(chunk_uid, generation_token):
                    self._cleanup_temp_file(temp_path)
                    return False, "Generation abandoned"

                if success:
                    if async_finalize:
                        task = self._enqueue_audio_finalize_task(
                            chunk_uid,
                            generation_token,
                            temp_path,
                            attempt=attempt,
                            speaker=speaker,
                            text=transformed_text,
                        )
                        if task is None:
                            return False, "Generation abandoned"
                        return True, "finalization queued"

                    result = self._finalize_generated_audio(
                        display_index,
                        speaker,
                        transformed_text,
                        temp_path,
                        attempt=attempt,
                        chunk_uid=chunk_uid,
                    )
                    updated_chunk = self._update_chunk_fields_if_token(
                        chunk_uid,
                        generation_token,
                        audio_path=result["audio_path"],
                        audio_validation=result["audio_validation"],
                        status=result["status"],
                        auto_regen_count=attempt,
                        generation_token=None,
                    )
                    self._cleanup_temp_file(temp_path)
                    if updated_chunk is None:
                        return False, "Generation abandoned"
                    should_retry = result["status"] == "error" and self._should_request_auto_regen_retry(
                        attempt,
                        error=result.get("error"),
                    )
                    if should_retry:
                        print(f"Chunk {display_index} failed sanity check; auto-regenerating attempt {attempt + 1}/{auto_regen_retry_attempts}")
                        return self._generate_chunk_audio_internal(
                            chunk_uid,
                            attempt=attempt + 1,
                            generation_token=generation_token,
                            async_finalize=False,
                            neutral_narrator=neutral_narrator,
                            cancel_check=cancel_check,
                        )
                    if generation_token is None:
                        self.flush_dirty_chunks(force=True)
                    return result["status"] == "done", result["audio_path"] if result["status"] == "done" else result["error"]
                else:
                    self._update_chunk_fields_if_token(
                        chunk_uid,
                        generation_token,
                        status="error",
                        audio_validation=None,
                        auto_regen_count=attempt,
                        generation_token=None,
                    )
                    if generation_token is None:
                        self.flush_dirty_chunks(force=True)
                    self._cleanup_temp_file(temp_path)
                    return False, "Generation failed"

            except Exception as e:
                try:
                    self._update_chunk_fields_if_token(
                        chunk_uid,
                        generation_token,
                        status="error",
                        audio_validation=None,
                        auto_regen_count=attempt,
                        generation_token=None,
                    )
                except Exception as update_err:
                    print(f"Warning: Failed to update chunk {display_index} status to error: {update_err}")
                self._cleanup_temp_file(self._spool_audio_full_path(chunk_uid, generation_token, attempt=attempt))
                if generation_token is None:
                    self.flush_dirty_chunks(force=True)
                return False, str(e)

        def generate_chunk_audio(self, index, attempt=0, generation_token=None, neutral_narrator=False,
                                 cancel_check=None):
            return self._generate_chunk_audio_internal(
                index,
                attempt=attempt,
                generation_token=generation_token,
                async_finalize=False,
                neutral_narrator=neutral_narrator,
                cancel_check=cancel_check,
            )

        def generate_chunks_parallel(self, indices, max_workers=2, progress_callback=None,
                                      cancel_check=None, item_callback=None, generation_token=None,
                                      item_started_callback=None, neutral_narrator=False):
            """Generate multiple chunks in parallel using ThreadPoolExecutor.

            Uses individual TTS API calls with per-speaker voice settings.

            Args:
                indices: List of chunk indices to generate
                max_workers: Number of concurrent TTS workers
                progress_callback: Optional callback(completed, failed, total) for progress updates
                cancel_check: Optional callable returning True when cancellation is requested
                item_callback: Optional callback(index, success, elapsed_seconds, input_words, output_words)
                item_started_callback: Optional callback(index, started_at_seconds)

            Returns:
                dict with 'completed', 'failed', and 'cancelled' keys
            """
            from concurrent.futures import ThreadPoolExecutor, as_completed

            results = {"completed": [], "failed": [], "cancelled": 0}

            target_chunks = [
                chunk for chunk in self._load_generation_chunks(indices)
                if (chunk.get("text") or "").strip()
            ]
            target_uids = [chunk.get("uid") for chunk in target_chunks if chunk.get("uid")]

            total = len(target_uids)

            if total == 0:
                return results

            print(f"Starting parallel generation of {total} chunks with {max_workers} workers...")
            word_counts = {
                chunk.get("uid"): len(re.findall(r"\b\w+\b", chunk.get("text", "")))
                for chunk in target_chunks
                if chunk.get("uid")
            }

            def _timed_generate(uid):
                start = time.time()
                if item_started_callback:
                    item_started_callback(uid, start)
                try:
                    chunk = self.get_chunk_raw(uid)
                    attempt = int((chunk or {}).get("auto_regen_count") or 0)
                    success, msg = self._generate_chunk_audio_internal(
                        uid,
                        attempt=attempt,
                        generation_token=generation_token,
                        async_finalize=True,
                        neutral_narrator=neutral_narrator,
                        cancel_check=cancel_check,
                    )
                    return success, msg, time.time() - start
                except Exception as e:
                    return False, str(e), time.time() - start

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(_timed_generate, uid): uid
                    for uid in target_uids
                }

                cancelled = False
                for future in as_completed(futures):
                    if cancel_check and cancel_check():
                        cancelled = True
                        print("[CANCEL] Cancellation requested — stopping parallel generation")
                        executor.shutdown(wait=False, cancel_futures=True)
                        break

                    uid = futures[future]
                    success, msg, elapsed_seconds = future.result()
                    if success:
                        results["completed"].append(uid)
                        if item_callback:
                            item_callback(uid, True, elapsed_seconds, word_counts.get(uid, 0), word_counts.get(uid, 0))
                    else:
                        results["failed"].append((uid, msg))
                        print(f"Chunk {uid} failed: {msg}")
                        if item_callback:
                            item_callback(uid, False, elapsed_seconds, word_counts.get(uid, 0), 0)
                    if progress_callback:
                        progress_callback(len(results["completed"]), len(results["failed"]), total)

                # Reset remaining "generating" chunks to "pending"
                if cancelled:
                    done_uids = set(results["completed"]) | {uid for uid, _ in results["failed"]}
                    reset_uids = [
                        uid for uid in target_uids
                        if uid not in done_uids and (self.get_chunk_view(uid) or {}).get("status") == "generating"
                    ]
                    if reset_uids:
                        results["cancelled"] += self.reset_generating_chunks(
                            reset_uids,
                            generation_token=generation_token,
                        )

            self.flush_dirty_chunks(force=True)
            print(f"Parallel generation complete: {len(results['completed'])} succeeded, "
                  f"{len(results['failed'])} failed, {results['cancelled']} cancelled")
            return results

        def _group_indices_by_voice_type(self, chunk_refs, chunks, voice_config):
            """Reorder chunk refs so chunks with the same voice type are contiguous.

            Grouping key matches how tts.py routes batches:
            - "custom" for custom voices (all batched together)
            - "clone:{speaker}" for clone voices (batched per speaker)
            - "lora:{adapter}" for LoRA voices (batched per adapter)
            - "design" for voice design (always sequential)

            Within each group, original order is preserved.
            """
            from collections import OrderedDict
            groups = OrderedDict()
            narrator_name = self._resolve_narrator_name(voice_config, chunks)
            chunk_by_uid = {
                str((chunk or {}).get("uid") or "").strip(): chunk
                for chunk in (chunks or [])
                if str((chunk or {}).get("uid") or "").strip()
            }
            chunk_by_position = {
                index: chunk
                for index, chunk in enumerate(chunks or [])
            }

            for chunk_ref in chunk_refs:
                chunk = chunk_by_uid.get(str(chunk_ref).strip())
                if chunk is None and isinstance(chunk_ref, int):
                    chunk = chunk_by_position.get(chunk_ref)
                if chunk is None:
                    chunk = self.get_chunk_raw(chunk_ref)
                if not chunk:
                    groups.setdefault("custom", []).append(chunk_ref)
                    continue

                speaker = self._resolve_generation_speaker(
                    chunk,
                    voice_config,
                    narrator_name=narrator_name,
                )
                voice_data = voice_config.get(speaker, {})
                voice_type = voice_data.get("type", "custom")

                if voice_type == "clone":
                    key = f"clone:{speaker}"
                elif voice_type in ("lora", "builtin_lora"):
                    adapter_id = voice_data.get("adapter_id", "")
                    key = f"lora:{adapter_id}"
                elif voice_type == "design":
                    key = "design"
                else:
                    key = "custom"

                groups.setdefault(key, []).append(chunk_ref if chunk.get("uid") is None else str(chunk.get("uid") or chunk_ref))

            reordered = []
            for key, group_refs in groups.items():
                print(f"  Voice group '{key}': {len(group_refs)} chunks")
                reordered.extend(group_refs)

            return reordered

        def group_indices_by_resolved_speaker(self, indices, chunks=None, voice_config=None):
            """Reorder chunk refs so each resolved speaker is generated contiguously.

            This is primarily useful for external TTS backends where clone prompt
            reuse is much faster when all lines for the same character are rendered
            back-to-back.
            """
            from collections import OrderedDict

            chunks = chunks if chunks is not None else self._load_generation_chunks(indices)
            voice_config = voice_config if voice_config is not None else self._load_voice_config()
            groups = OrderedDict()
            labels = {}
            narrator_name = self._resolve_narrator_name(voice_config, chunks)
            chunk_by_uid = {
                str((chunk or {}).get("uid") or "").strip(): chunk
                for chunk in (chunks or [])
                if str((chunk or {}).get("uid") or "").strip()
            }
            chunk_by_position = {
                index: chunk
                for index, chunk in enumerate(chunks or [])
            }

            for chunk_ref in indices:
                chunk = chunk_by_uid.get(str(chunk_ref).strip())
                if chunk is None and isinstance(chunk_ref, int):
                    chunk = chunk_by_position.get(chunk_ref)
                if chunk is None:
                    chunk = self.get_chunk_raw(chunk_ref)
                if not chunk:
                    groups.setdefault("", []).append(chunk_ref)
                    continue

                resolved = self._resolve_generation_speaker(
                    chunk,
                    voice_config,
                    narrator_name=narrator_name,
                )
                group_key = self._normalize_speaker_name(resolved) or resolved
                groups.setdefault(group_key, []).append(chunk_ref if chunk.get("uid") is None else str(chunk.get("uid") or chunk_ref))
                labels.setdefault(group_key, resolved)

            reordered = []
            for key, group_indices in groups.items():
                label = labels.get(key) or key or "<unknown>"
                print(f"  Speaker group '{label}': {len(group_indices)} chunks")
                reordered.extend(group_indices)

            return reordered

        def generate_chunks_batch(self, indices, batch_seed=-1, batch_size=4, progress_callback=None,
                                   batch_group_by_type=False, cancel_check=None, item_callback=None,
                                   generation_token=None, item_started_callback=None, log_callback=None,
                                   neutral_narrator=False):
            """Generate multiple chunks using batch TTS API with a single seed.

            Args:
                indices: List of chunk indices to generate
                batch_seed: Single seed for all generations (-1 for random)
                batch_size: Number of chunks per batch request
                progress_callback: Optional callback(completed, failed, total) for progress updates
                batch_group_by_type: Group indices by voice type before batching for
                    GPU efficiency. When False, indices are batched in sequential order.
                cancel_check: Optional callable returning True when cancellation is requested
                item_callback: Optional callback(index, success, elapsed_seconds, input_words, output_words)
                item_started_callback: Optional callback(index, started_at_seconds)

            Returns:
                dict with 'completed', 'failed', and 'cancelled' keys
            """
            results = {"completed": [], "failed": [], "cancelled": 0}

            target_chunks = [
                chunk for chunk in self._load_generation_chunks(indices)
                if (chunk.get("text") or "").strip()
            ]
            target_uids = [chunk.get("uid") for chunk in target_chunks if chunk.get("uid")]
            total = len(target_uids)

            if total == 0:
                return results

            print(f"Starting batch generation of {total} chunks (batch_size={batch_size}, seed={batch_seed}, "
                  f"group_by_type={batch_group_by_type})...")
            word_counts = {
                chunk.get("uid"): len(re.findall(r"\b\w+\b", chunk.get("text", "")))
                for chunk in target_chunks
                if chunk.get("uid")
            }
            voice_config = self._load_voice_config()
            narrator_overrides = self.get_narrator_overrides()
            narrator_name = self._resolve_narrator_name(voice_config, target_chunks)
            resolved_speakers = {
                self._resolve_generation_speaker(
                    chunk,
                    voice_config,
                    narrator_overrides=narrator_overrides,
                    narrator_name=narrator_name,
                )
                for chunk in target_chunks
            }
            voice_config = self.prepare_runtime_voice_config(voice_config, resolved_speakers)
            dictionary_entries = self.load_dictionary_entries()
            # Get TTS engine
            engine = self.get_engine()
            if not engine:
                for uid in target_uids:
                    results["failed"].append((uid, "TTS engine not initialized"))
                return results

            # Optionally reorder indices so same voice-type chunks are contiguous.
            # This produces larger homogeneous batches (e.g. all custom voices
            # together) instead of fragmenting each batch across voice types.
            if batch_group_by_type:
                target_uids = self._group_indices_by_voice_type(target_uids, target_chunks, voice_config)

            # Split indices into batches
            spool_run_token = generation_token or uuid.uuid4().hex
            batches = [target_uids[i:i + batch_size] for i in range(0, len(target_uids), batch_size)]
            print(f"Processing {len(batches)} batches...")

            cancelled = False
            for batch_num, batch_uids in enumerate(batches):
                if cancel_check and cancel_check():
                    cancelled = True
                    print(f"[CANCEL] Cancellation requested before batch {batch_num + 1}")
                    break

                claimed_rows = self.claim_generation_many(batch_uids, generation_token)
                claimed_uids = [
                    str((chunk or {}).get("uid") or "").strip()
                    for chunk in (claimed_rows or [])
                    if str((chunk or {}).get("uid") or "").strip()
                ]
                if not claimed_uids:
                    print(
                        f"Batch {batch_num + 1}/{len(batches)}: "
                        f"{len(batch_uids)} requested chunks skipped (none claimable)"
                    )
                    continue

                batch_uids = claimed_uids
                print(f"Batch {batch_num + 1}/{len(batches)}: {len(batch_uids)} chunks")

                # Build batch request data
                batch_chunks = []
                transformed_texts = {}
                batch_rows = {chunk.get("uid"): chunk for chunk in self._load_generation_chunks(batch_uids)}
                for uid in batch_uids:
                    chunk = batch_rows.get(uid)
                    if chunk is None:
                        continue
                    transformed_text, _ = apply_dictionary_to_text(chunk.get("text", ""), dictionary_entries)
                    transformed_texts[uid] = transformed_text
                    resolved_speaker = self._resolve_generation_speaker(
                        chunk,
                        voice_config,
                        narrator_overrides=narrator_overrides,
                        narrator_name=narrator_name,
                    )
                    batch_chunks.append({
                        "index": uid,
                        "display_id": chunk.get("id"),
                        "text": transformed_text,
                        "instruct": self._effective_generation_instruct(chunk, neutral_narrator=neutral_narrator),
                        "speaker": resolved_speaker,
                    })

                # Call batch TTS with single seed
                batch_start = time.time()
                if item_started_callback:
                    for uid in batch_uids:
                        item_started_callback(uid, batch_start)
                batch_output_dir = os.path.join(
                    self.audio_finalize_spool_dir,
                    sanitize_filename(spool_run_token) or "manual",
                    f"batch_{batch_num + 1:04d}",
                )
                os.makedirs(batch_output_dir, exist_ok=True)
                batch_generate_started = time.perf_counter()
                batch_call_kwargs = {
                    "cancel_check": cancel_check,
                }
                try:
                    generate_batch_signature = inspect.signature(engine.generate_batch)
                except (TypeError, ValueError):
                    generate_batch_signature = None
                if generate_batch_signature and "log_callback" in generate_batch_signature.parameters:
                    batch_call_kwargs["log_callback"] = log_callback
                batch_results = engine.generate_batch(
                    batch_chunks,
                    voice_config,
                    batch_output_dir,
                    batch_seed,
                    **batch_call_kwargs,
                )
                record_audio_perf(
                    "audio_batch_generate",
                    generation_token=generation_token,
                    batch_num=batch_num + 1,
                    total_batches=len(batches),
                    batch_size=len(batch_uids),
                    elapsed_ms=round((time.perf_counter() - batch_generate_started) * 1000.0, 3),
                )

                # Some backends have been observed to materialize temp audio
                # files without reporting those chunk ids in the batch result.
                # Reconcile against the temp artifacts here so saveback and job
                # progress follow the actual generated outputs.
                reported_completed = list(batch_results.get("completed") or [])
                reported_failed = list(batch_results.get("failed") or [])
                reported_terminal = set(reported_completed)
                reported_terminal.update(uid for uid, _ in reported_failed)
                for uid in batch_uids:
                    if uid in reported_terminal:
                        continue
                    temp_path = os.path.join(batch_output_dir, f"temp_batch_{uid}.wav")
                    if os.path.exists(temp_path) and os.path.getsize(temp_path) > 0:
                        print(
                            f"Batch backend did not report completion for chunk {uid}; "
                            "recovering from temp audio artifact"
                        )
                        reported_completed.append(uid)
                    else:
                        reported_failed.append((uid, "Batch backend returned no result for requested chunk"))
                batch_results = {
                    "completed": reported_completed,
                    "failed": reported_failed,
                }

                # Process completed chunks - queue async finalization for each output
                live_rows = {chunk.get("uid"): chunk for chunk in self._load_generation_chunks(batch_uids)}
                if cancel_check and cancel_check():
                    cancelled = True

                processed_in_batch = len(batch_results["completed"]) + len(batch_results["failed"])
                shared_elapsed = (time.time() - batch_start) / processed_in_batch if processed_in_batch > 0 else 0.0
                batch_postprocess_started = time.perf_counter()

                for uid in batch_results["completed"]:
                    if cancel_check and cancel_check():
                        cancelled = True
                        temp_path = os.path.join(batch_output_dir, f"temp_batch_{uid}.wav")
                        self._cleanup_temp_file(temp_path)
                        continue
                    chunk = live_rows.get(uid) or self.get_chunk_raw(uid)
                    if chunk is None:
                        print(f"Chunk {uid} skipped: row missing after generation")
                        results["failed"].append((uid, "Chunk missing after reload"))
                        continue

                    temp_path = os.path.join(batch_output_dir, f"temp_batch_{uid}.wav")

                    if not os.path.exists(temp_path):
                        results["failed"].append((uid, "Temp audio file not found"))
                        self._update_chunk_fields_if_token(
                            uid,
                            generation_token,
                            status="error",
                            audio_validation=None,
                            generation_token=None,
                        )
                        continue

                    try:
                        if generation_token is not None and not self.chunk_has_generation_token(uid, generation_token):
                            self._cleanup_temp_file(temp_path)
                            continue
                        speaker = chunk.get("speaker", "unknown")
                        attempt = int(chunk.get("auto_regen_count") or 0)
                        task = self._enqueue_audio_finalize_task(
                            chunk.get("uid"),
                            generation_token,
                            temp_path,
                            attempt=attempt,
                            speaker=speaker,
                            text=transformed_texts.get(uid, chunk.get("text", "")),
                        )
                        if task is None:
                            results["failed"].append((uid, "Generation abandoned"))
                            continue
                        results["completed"].append(uid)
                        if item_callback:
                            item_callback(uid, True, shared_elapsed, word_counts.get(uid, 0), word_counts.get(uid, 0))
                    except Exception as e:
                        print(f"Error queueing async finalization for chunk {uid}: {e}")
                        results["failed"].append((uid, str(e)))
                        self._update_chunk_fields_if_token(
                            uid,
                            generation_token,
                            status="error",
                            audio_validation=None,
                            generation_token=None,
                        )
                        self._cleanup_temp_file(temp_path)
                        if item_callback:
                            item_callback(uid, False, shared_elapsed, word_counts.get(uid, 0), 0)
                record_audio_perf(
                    "audio_batch_postprocess",
                    generation_token=generation_token,
                    batch_num=batch_num + 1,
                    total_batches=len(batches),
                    completed=len(batch_results["completed"]),
                    failed=len(batch_results["failed"]),
                    elapsed_ms=round((time.perf_counter() - batch_postprocess_started) * 1000.0, 3),
                )

                for uid, error in batch_results["failed"]:
                    if self.get_chunk_raw(uid) is not None:
                        self._update_chunk_fields_if_token(
                            uid,
                            generation_token,
                            status="error",
                            audio_validation=None,
                            generation_token=None,
                        )
                    results["failed"].append((uid, error))
                    if item_callback:
                        item_callback(uid, False, shared_elapsed, word_counts.get(uid, 0), 0)
                if cancel_check and cancel_check():
                    cancelled = True

                if progress_callback:
                    progress_callback(len(results["completed"]), len(results["failed"]), total)

                if cancelled:
                    print(f"[CANCEL] Stopping after batch {batch_num + 1}")
                    break

            # Reset remaining live "generating" chunks to "pending" on cancel or completion.
            done_uids = set(results["completed"]) | {uid for uid, _ in results["failed"]}
            reset_uids = [
                uid for uid in target_uids
                if uid not in done_uids and (self.get_chunk_view(uid) or {}).get("status") == "generating"
            ]
            if reset_uids:
                results["cancelled"] += self.reset_generating_chunks(reset_uids, generation_token=generation_token)

            self.flush_dirty_chunks(force=True)

            print(f"Batch generation complete: {len(results['completed'])} succeeded, "
                  f"{len(results['failed'])} failed, {results['cancelled']} cancelled")
            return results
