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
from script_store import (
    apply_dictionary_to_text,
    load_script_document,
)
from source_document import load_source_document, iter_document_paragraphs
from project_core.constants import *
from project_core.chunking import _coerce_bool, get_speaker, _is_structural_text, _extract_chapter_name, _build_chunk, group_into_chunks, script_entries_to_chunks


class ProjectGenerationMixin:
        """Generate and persist chunk audio through TTS pipelines."""
        def _get_auto_regen_retry_attempts(self):
            tts_settings = self._load_tts_settings()
            if not tts_settings.get("auto_regenerate_bad_clips", False):
                return 0
            try:
                attempts = int(tts_settings.get("auto_regenerate_bad_clip_attempts", 3))
            except (TypeError, ValueError):
                return 0
            return attempts if attempts > 0 else 0

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

        def generate_chunk_audio(self, index, attempt=0, generation_token=None):
            chunks = self.load_chunks()
            if not (0 <= index < len(chunks)):
                return False, "Invalid chunk index"

            chunk = self._claim_chunk_generation(index, generation_token)
            if not chunk:
                return False, "Invalid chunk index"

            try:
                engine = self.get_engine()
                if not engine:
                    self._update_chunk_fields(index, status="error", audio_validation=None, auto_regen_count=attempt)
                    return False, "TTS engine not initialized"

                speaker = chunk["speaker"]
                # Apply per-chapter narrator override: if this is a NARRATOR line and the chapter
                # has a custom narrator voice selected, substitute that voice for generation.
                speaker = self._apply_narrator_override(
                    speaker, (chunk.get("chapter") or "").strip(), self.get_narrator_overrides()
                )
                voice_config = self._load_voice_config()
                resolved_speaker = self.resolve_voice_speaker(speaker, voice_config, chunks=chunks)
                voice_config = self.prepare_runtime_voice_config(voice_config, [resolved_speaker])
                text = chunk["text"]
                transformed_text, _ = apply_dictionary_to_text(text, self.load_dictionary_entries())
                instruct = chunk.get("instruct", "")
                auto_regen_retry_attempts = self._get_auto_regen_retry_attempts()

                print(
                    f"Generating chunk {index}: speaker={speaker}, resolved_speaker={resolved_speaker}, "
                    f"instruct='{instruct}', text='{transformed_text[:50]}...'"
                )

                # Generate to temp file (unique per chunk for parallel processing)
                temp_path = os.path.join(self.root_dir, f"temp_chunk_{index}.wav")

                success = engine.generate_voice(transformed_text, instruct, resolved_speaker, voice_config, temp_path)

                if generation_token is not None and not self.chunk_has_generation_token(index, generation_token):
                    self._cleanup_temp_file(temp_path)
                    return False, "Generation abandoned"

                if success:
                    fut = self._enqueue_postprocess(
                        index=index,
                        speaker=speaker,
                        text=transformed_text,
                        temp_path=temp_path,
                        attempt=attempt,
                        chunk_uid=chunk.get("uid"),
                        generation_token=generation_token,
                    )
                    # Block this TTS thread until saveback completes; other TTS
                    # threads in the executor can run their GPU work concurrently.
                    try:
                        result = fut.result()
                    except Exception as e:
                        return False, str(e)

                    if result["status"] == "cancelled":
                        return False, "Generation abandoned"

                    if result["status"] == "error" and auto_regen_retry_attempts > 0 and attempt < auto_regen_retry_attempts:
                        # Saveback wrote status="error"; _claim_chunk_generation in
                        # the recursive call will re-claim it to "generating".
                        print(f"Chunk {index} failed sanity check; auto-regenerating attempt {attempt + 1}/{auto_regen_retry_attempts}")
                        return self.generate_chunk_audio(index, attempt=attempt + 1, generation_token=generation_token)

                    if generation_token is None:
                        self.flush_dirty_chunks(force=True)
                    return result["status"] == "done", result["audio_path"] if result["status"] == "done" else result["error"]
                else:
                    self._update_chunk_fields_if_token(
                        index,
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
                        index,
                        generation_token,
                        status="error",
                        audio_validation=None,
                        auto_regen_count=attempt,
                        generation_token=None,
                    )
                except Exception as update_err:
                    print(f"Warning: Failed to update chunk {index} status to error: {update_err}")
                self._cleanup_temp_file(os.path.join(self.root_dir, f"temp_chunk_{index}.wav"))
                if generation_token is None:
                    self.flush_dirty_chunks(force=True)
                return False, str(e)

        def generate_chunks_parallel(self, indices, max_workers=2, progress_callback=None,
                                      cancel_check=None, item_callback=None, generation_token=None):
            """Generate multiple chunks in parallel using ThreadPoolExecutor.

            Uses individual TTS API calls with per-speaker voice settings.

            Args:
                indices: List of chunk indices to generate
                max_workers: Number of concurrent TTS workers
                progress_callback: Optional callback(completed, failed, total) for progress updates
                cancel_check: Optional callable returning True when cancellation is requested
                item_callback: Optional callback(index, success, elapsed_seconds, input_words, output_words)

            Returns:
                dict with 'completed', 'failed', and 'cancelled' keys
            """
            from concurrent.futures import ThreadPoolExecutor, as_completed

            results = {"completed": [], "failed": [], "cancelled": 0}

            # Filter out empty-text chunks
            chunks = self.load_chunks()
            if chunks:
                indices = [i for i in indices if 0 <= i < len(chunks) and chunks[i].get("text", "").strip()]

            total = len(indices)

            if total == 0:
                return results

            print(f"Starting parallel generation of {total} chunks with {max_workers} workers...")
            word_counts = {
                idx: len(re.findall(r"\b\w+\b", chunks[idx].get("text", "")))
                for idx in indices if 0 <= idx < len(chunks)
            }

            def _timed_generate(idx):
                start = time.time()
                try:
                    success, msg = self.generate_chunk_audio(idx, generation_token=generation_token)
                    return success, msg, time.time() - start
                except Exception as e:
                    return False, str(e), time.time() - start

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(_timed_generate, idx): idx
                    for idx in indices
                }

                cancelled = False
                for future in as_completed(futures):
                    if cancel_check and cancel_check():
                        cancelled = True
                        print("[CANCEL] Cancellation requested — stopping parallel generation")
                        executor.shutdown(wait=False, cancel_futures=True)
                        break

                    idx = futures[future]
                    success, msg, elapsed_seconds = future.result()
                    if success:
                        results["completed"].append(idx)
                        print(f"Chunk {idx} completed: {msg}")
                        if item_callback:
                            item_callback(idx, True, elapsed_seconds, word_counts.get(idx, 0), word_counts.get(idx, 0))
                    else:
                        results["failed"].append((idx, msg))
                        print(f"Chunk {idx} failed: {msg}")
                        if item_callback:
                            item_callback(idx, False, elapsed_seconds, word_counts.get(idx, 0), 0)
                    if progress_callback:
                        progress_callback(len(results["completed"]), len(results["failed"]), total)

                # Reset remaining "generating" chunks to "pending"
                if cancelled:
                    done_indices = set(results["completed"]) | {idx for idx, _ in results["failed"]}
                    reset_indices = [
                        idx for idx in indices
                        if idx not in done_indices and (self.get_chunk_view_by_index(idx) or {}).get("status") == "generating"
                    ]
                    if reset_indices:
                        results["cancelled"] += self.reset_generating_chunks(
                            reset_indices,
                            generation_token=generation_token,
                        )

            self.flush_dirty_chunks(force=True)
            print(f"Parallel generation complete: {len(results['completed'])} succeeded, "
                  f"{len(results['failed'])} failed, {results['cancelled']} cancelled")
            return results

        def _group_indices_by_voice_type(self, indices, chunks, voice_config):
            """Reorder indices so chunks with the same voice type are contiguous.

            Grouping key matches how tts.py routes batches:
            - "custom" for custom voices (all batched together)
            - "clone:{speaker}" for clone voices (batched per speaker)
            - "lora:{adapter}" for LoRA voices (batched per adapter)
            - "design" for voice design (always sequential)

            Within each group, original order is preserved.
            """
            from collections import OrderedDict
            groups = OrderedDict()
            labels = {}

            for idx in indices:
                if not (0 <= idx < len(chunks)):
                    groups.setdefault("custom", []).append(idx)
                    continue

                speaker = chunks[idx].get("speaker", "")
                speaker = self.resolve_voice_speaker(speaker, voice_config, chunks=chunks)
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

                groups.setdefault(key, []).append(idx)

            reordered = []
            for key, group_indices in groups.items():
                print(f"  Voice group '{key}': {len(group_indices)} chunks")
                reordered.extend(group_indices)

            return reordered

        def group_indices_by_resolved_speaker(self, indices, chunks=None, voice_config=None):
            """Reorder indices so each resolved speaker is generated contiguously.

            This is primarily useful for external TTS backends where clone prompt
            reuse is much faster when all lines for the same character are rendered
            back-to-back.
            """
            from collections import OrderedDict

            chunks = chunks if chunks is not None else self.load_chunks()
            voice_config = voice_config if voice_config is not None else self._load_voice_config()
            groups = OrderedDict()
            labels = {}

            for idx in indices:
                if not (0 <= idx < len(chunks)):
                    groups.setdefault("", []).append(idx)
                    continue

                speaker = chunks[idx].get("speaker", "")
                resolved = self.resolve_voice_speaker(speaker, voice_config, chunks=chunks)
                group_key = self._normalize_speaker_name(resolved) or resolved
                groups.setdefault(group_key, []).append(idx)
                labels.setdefault(group_key, resolved)

            reordered = []
            for key, group_indices in groups.items():
                label = labels.get(key) or key or "<unknown>"
                print(f"  Speaker group '{label}': {len(group_indices)} chunks")
                reordered.extend(group_indices)

            return reordered

        def generate_chunks_batch(self, indices, batch_seed=-1, batch_size=4, progress_callback=None,
                                   batch_group_by_type=False, cancel_check=None, item_callback=None, generation_token=None):
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

            Returns:
                dict with 'completed', 'failed', and 'cancelled' keys
            """
            results = {"completed": [], "failed": [], "cancelled": 0}

            # Load chunks and voice config
            chunks = self.load_chunks()

            # Filter out empty-text chunks
            if chunks:
                indices = [i for i in indices if 0 <= i < len(chunks) and chunks[i].get("text", "").strip()]

            total = len(indices)

            if total == 0:
                return results

            print(f"Starting batch generation of {total} chunks (batch_size={batch_size}, seed={batch_seed}, "
                  f"group_by_type={batch_group_by_type})...")
            word_counts = {
                idx: len(re.findall(r"\b\w+\b", chunks[idx].get("text", "")))
                for idx in indices if 0 <= idx < len(chunks)
            }
            voice_config = {}
            voice_config = self._load_voice_config()
            narrator_overrides = self.get_narrator_overrides()
            resolved_speakers = {
                self.resolve_voice_speaker(
                    self._apply_narrator_override(
                        chunks[idx].get("speaker", ""),
                        (chunks[idx].get("chapter") or "").strip(),
                        narrator_overrides,
                    ),
                    voice_config, chunks=chunks,
                )
                for idx in indices if 0 <= idx < len(chunks)
            }
            voice_config = self.prepare_runtime_voice_config(voice_config, resolved_speakers)
            dictionary_entries = self.load_dictionary_entries()
            auto_regen_retry_attempts = self._get_auto_regen_retry_attempts()

            # Get TTS engine
            engine = self.get_engine()
            if not engine:
                for idx in indices:
                    results["failed"].append((idx, "TTS engine not initialized"))
                return results

            self._claim_chunks_generation(indices, generation_token)

            # Optionally reorder indices so same voice-type chunks are contiguous.
            # This produces larger homogeneous batches (e.g. all custom voices
            # together) instead of fragmenting each batch across voice types.
            if batch_group_by_type:
                indices = self._group_indices_by_voice_type(indices, chunks, voice_config)

            # Split indices into batches
            batches = [indices[i:i + batch_size] for i in range(0, len(indices), batch_size)]
            print(f"Processing {len(batches)} batches...")

            cancelled = False
            for batch_num, batch_indices in enumerate(batches):
                if cancel_check and cancel_check():
                    cancelled = True
                    print(f"[CANCEL] Cancellation requested before batch {batch_num + 1}")
                    break

                print(f"Batch {batch_num + 1}/{len(batches)}: {len(batch_indices)} chunks")

                # Build batch request data
                batch_chunks = []
                transformed_texts = {}
                for idx in batch_indices:
                    if 0 <= idx < len(chunks):
                        chunk = chunks[idx]
                        transformed_text, _ = apply_dictionary_to_text(chunk.get("text", ""), dictionary_entries)
                        transformed_texts[idx] = transformed_text
                        effective_speaker = self._apply_narrator_override(
                            chunk.get("speaker", ""),
                            (chunk.get("chapter") or "").strip(),
                            narrator_overrides,
                        )
                        resolved_speaker = self.resolve_voice_speaker(effective_speaker, voice_config, chunks=chunks)
                        batch_chunks.append({
                            "index": idx,
                            "text": transformed_text,
                            "instruct": chunk.get("instruct", ""),
                            "speaker": resolved_speaker
                        })

                # Call batch TTS with single seed
                batch_start = time.time()
                batch_results = engine.generate_batch(
                    batch_chunks,
                    voice_config,
                    self.root_dir,
                    batch_seed,
                    cancel_check=cancel_check,
                )

                # Process completed chunks - convert to MP3 and update status
                chunks = self.load_chunks()  # Reload for each batch
                if cancel_check and cancel_check():
                    cancelled = True

                processed_in_batch = len(batch_results["completed"]) + len(batch_results["failed"])
                shared_elapsed = (time.time() - batch_start) / processed_in_batch if processed_in_batch > 0 else 0.0

                postprocess_futures = {}
                for idx in batch_results["completed"]:
                    if cancel_check and cancel_check():
                        cancelled = True
                        temp_path = os.path.join(self.root_dir, f"temp_batch_{idx}.wav")
                        self._cleanup_temp_file(temp_path)
                        continue
                    if not (0 <= idx < len(chunks)):
                        print(f"Chunk {idx} skipped: index out of range (chunks changed during generation?)")
                        results["failed"].append((idx, "Index out of range after reload"))
                        continue

                    temp_path = os.path.join(self.root_dir, f"temp_batch_{idx}.wav")

                    if not os.path.exists(temp_path):
                        results["failed"].append((idx, "Temp audio file not found"))
                        self._update_chunk_fields_if_token(
                            idx,
                            generation_token,
                            status="error",
                            audio_validation=None,
                            generation_token=None,
                        )
                        continue

                    try:
                        if generation_token is not None and not self.chunk_has_generation_token(idx, generation_token):
                            self._cleanup_temp_file(temp_path)
                            continue
                        chunk = chunks[idx]
                        speaker = chunk.get("speaker", "unknown")
                        will_retry_on_error = auto_regen_retry_attempts > 0

                        fut = self._enqueue_postprocess(
                            index=idx,
                            speaker=speaker,
                            text=transformed_texts.get(idx, chunk.get("text", "")),
                            temp_path=temp_path,
                            attempt=0,
                            chunk_uid=chunk.get("uid"),
                            generation_token=generation_token,
                            error_status_override="pending" if will_retry_on_error else None,
                            keep_token=will_retry_on_error,
                        )
                        postprocess_futures[fut] = (idx, will_retry_on_error)
                    except Exception as e:
                        print(f"Error queueing postprocess for chunk {idx}: {e}")
                        results["failed"].append((idx, str(e)))
                        self._update_chunk_fields_if_token(
                            idx,
                            generation_token,
                            status="error",
                            audio_validation=None,
                            generation_token=None,
                        )
                        self._cleanup_temp_file(temp_path)
                        if item_callback:
                            item_callback(idx, False, shared_elapsed, word_counts.get(idx, 0), 0)

                for fut in concurrent.futures.as_completed(list(postprocess_futures)):
                    idx, will_retry_on_error = postprocess_futures[fut]
                    try:
                        result = fut.result()
                        if result["status"] == "cancelled":
                            continue

                        live_chunk = self.get_chunk_view_by_index(idx)

                        if result["status"] == "done":
                            results["completed"].append(idx)
                            print(f"Chunk {idx} completed: {(live_chunk or {}).get('audio_path')}")
                            if item_callback:
                                item_callback(idx, True, shared_elapsed, word_counts.get(idx, 0), word_counts.get(idx, 0))
                        elif will_retry_on_error and not (cancel_check and cancel_check()):
                            print(f"Chunk {idx} failed validation in batch; retrying immediately at the front of the queue")
                            retry_start = time.time()
                            retry_success, retry_msg = self.generate_chunk_audio(idx, attempt=1, generation_token=generation_token)
                            retry_elapsed = time.time() - retry_start
                            chunks = self.load_chunks()
                            if retry_success:
                                results["completed"].append(idx)
                                print(f"Chunk {idx} completed after auto-regeneration: {retry_msg}")
                                if item_callback:
                                    item_callback(idx, True, shared_elapsed + retry_elapsed, word_counts.get(idx, 0), word_counts.get(idx, 0))
                            else:
                                results["failed"].append((idx, retry_msg))
                                print(f"Chunk {idx} failed after auto-regeneration: {retry_msg}")
                                if item_callback:
                                    item_callback(idx, False, shared_elapsed + retry_elapsed, word_counts.get(idx, 0), 0)
                        else:
                            results["failed"].append((idx, result["error"]))
                            print(f"Chunk {idx} failed validation: {result['error']}")
                            if item_callback:
                                item_callback(idx, False, shared_elapsed, word_counts.get(idx, 0), 0)
                    except Exception as e:
                        print(f"Error processing chunk {idx}: {e}")
                        results["failed"].append((idx, str(e)))
                        self._update_chunk_fields_if_token(
                            idx,
                            generation_token,
                            status="error",
                            audio_validation=None,
                            generation_token=None,
                        )
                        self._cleanup_temp_file(temp_path)
                        if item_callback:
                            item_callback(idx, False, shared_elapsed, word_counts.get(idx, 0), 0)

                for idx, error in batch_results["failed"]:
                    if 0 <= idx < len(chunks):
                        self._update_chunk_fields_if_token(
                            idx,
                            generation_token,
                            status="error",
                            audio_validation=None,
                            generation_token=None,
                        )
                    results["failed"].append((idx, error))
                    if item_callback:
                        item_callback(idx, False, shared_elapsed, word_counts.get(idx, 0), 0)

                chunks = self.load_chunks()
                if cancel_check and cancel_check():
                    cancelled = True

                if progress_callback:
                    progress_callback(len(results["completed"]), len(results["failed"]), total)

                if cancelled:
                    print(f"[CANCEL] Stopping after batch {batch_num + 1}")
                    break

            # Reset remaining live "generating" chunks to "pending" on cancel or completion.
            done_indices = set(results["completed"]) | {idx for idx, _ in results["failed"]}
            reset_indices = [
                idx for idx in indices
                if idx not in done_indices and (self.get_chunk_view_by_index(idx) or {}).get("status") == "generating"
            ]
            if reset_indices:
                results["cancelled"] += self.reset_generating_chunks(reset_indices, generation_token=generation_token)

            self.flush_dirty_chunks(force=True)

            print(f"Batch generation complete: {len(results['completed'])} succeeded, "
                  f"{len(results['failed'])} failed, {results['cancelled']} cancelled")
            return results
