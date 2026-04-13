"""Export and packaging pipelines for merged audiobook output artifacts.

This mixin owns:
- concat timeline assembly from chunk audio,
- optional trimming and loudness normalization,
- final MP3/M4B export and Audacity project export,
- optimized multi-part zip packaging for long-form output.
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
from runtime_layout import LAYOUT


def _make_runtime_temp_dir(project_manager, prefix, *, run_id=None):
    if getattr(project_manager, "_using_default_runtime_layout", False):
        effective_run_id = str(run_id or f"runtime-{uuid.uuid4().hex[:8]}")
        return LAYOUT.make_named_temp_dir(effective_run_id, prefix)
    return tempfile.mkdtemp(prefix=prefix, dir=project_manager.root_dir)


class ProjectAudioExportMixin:
        """Build, normalize, and export merged audiobook deliverables."""
        @staticmethod
        def _escape_concat_path(path):
            return path.replace("\\", "\\\\").replace("'", r"'\''")

        @classmethod
        def _write_concat_line(cls, handle, path):
            handle.write(f"file '{cls._escape_concat_path(path)}'\n")

        def _resolve_trim_config(self, export_config=None):
            cfg = export_config or self._resolve_export_normalization_config(None)
            return {
                "enabled": _coerce_bool(getattr(cfg, "trim_clip_silence_enabled", True), default=True),
                "silence_threshold_dbfs": float(getattr(cfg, "trim_silence_threshold_dbfs", TRIM_SILENCE_THRESHOLD_DBFS)),
                "min_silence_len_ms": max(1, int(getattr(cfg, "trim_min_silence_len_ms", TRIM_MIN_SILENCE_LEN_MS))),
                "keep_padding_ms": max(0, int(getattr(cfg, "trim_keep_padding_ms", TRIM_KEEP_PADDING_MS))),
            }

        def _trim_cache_dir(self):
            return os.path.join(self.root_dir, "voicelines", ".trim_cache")

        def _build_trim_cache_key(self, full_path, trim_cfg):
            stat = os.stat(full_path)
            rel_path = os.path.relpath(full_path, self.root_dir)
            payload = "|".join([
                str(TRIM_CACHE_VERSION),
                rel_path,
                str(stat.st_size),
                str(getattr(stat, "st_mtime_ns", int(stat.st_mtime * 1_000_000_000))),
                str(trim_cfg["silence_threshold_dbfs"]),
                str(trim_cfg["min_silence_len_ms"]),
                str(trim_cfg["keep_padding_ms"]),
            ])
            return hashlib.sha1(payload.encode("utf-8")).hexdigest()

        @staticmethod
        def _trim_audio_segment_boundaries(segment, trim_cfg):
            if len(segment) <= 0:
                return segment, 0, 0, False

            ranges = detect_nonsilent(
                segment,
                min_silence_len=trim_cfg["min_silence_len_ms"],
                silence_thresh=trim_cfg["silence_threshold_dbfs"],
            )
            if not ranges:
                return segment, 0, 0, False

            first_start = int(ranges[0][0])
            last_end = int(ranges[-1][1])

            keep = trim_cfg["keep_padding_ms"]
            trim_start = max(0, first_start - keep)

            # Trailing keep-padding can reintroduce codec-tail artifacts (especially
            # from MP3 decoder padding) into the trimmed output. Preserve only the
            # leading keep-padding; keep the trailing cut at the detected nonsilent end.
            trim_end = min(len(segment), last_end)
            if trim_end <= trim_start:
                return segment, 0, 0, False

            frame_rate = int(segment.frame_rate or 0)
            channels = max(1, int(segment.channels or 1))
            frame_width = int(segment.frame_width or 0)
            if frame_rate <= 0 or frame_width <= 0:
                lead_removed = trim_start
                tail_removed = len(segment) - trim_end
                return segment[trim_start:trim_end], lead_removed, tail_removed, (lead_removed > 0 or tail_removed > 0)

            samples = segment.get_array_of_samples()
            total_frames = len(samples) // channels
            start_frame = max(0, min(total_frames, int(round(trim_start * frame_rate / 1000.0))))
            end_frame = max(start_frame, min(total_frames, int(round(trim_end * frame_rate / 1000.0))))
            if end_frame <= start_frame:
                return segment, 0, 0, False

            # detect_nonsilent() boundaries are RMS-window based. Snap start to a nearby
            # zero crossing so transitions from inserted silence don't hard-click.
            max_search_frames = max(1, int(round(frame_rate * 0.008)))  # up to ~8ms
            if start_frame < total_frames - 1:
                search_limit = min(total_frames - 1, start_frame + max_search_frames)
                prev = int(samples[start_frame * channels])
                best_frame = start_frame
                best_abs = abs(prev)
                for frame in range(start_frame + 1, search_limit + 1):
                    current = int(samples[frame * channels])
                    current_abs = abs(current)
                    if current_abs < best_abs:
                        best_abs = current_abs
                        best_frame = frame
                    if current == 0 or (prev <= 0 < current) or (prev >= 0 > current):
                        start_frame = frame
                        break
                    prev = current
                else:
                    start_frame = best_frame

            # detect_nonsilent() uses windowed RMS and can end at a non-zero crossing.
            # Snap the cut point to the next nearby zero crossing to avoid hard-edge pops.
            search_limit = min(total_frames - 1, end_frame + max_search_frames)
            if end_frame < total_frames - 1:
                prev = int(samples[(end_frame - 1) * channels]) if end_frame > 0 else int(samples[end_frame * channels])
                best_frame = end_frame
                best_abs = abs(prev)
                for frame in range(end_frame, search_limit + 1):
                    current = int(samples[frame * channels])
                    current_abs = abs(current)
                    if current_abs < best_abs:
                        best_abs = current_abs
                        best_frame = frame
                    if current == 0 or (prev <= 0 < current) or (prev >= 0 > current):
                        end_frame = min(total_frames, frame + 1)
                        break
                    prev = current
                else:
                    end_frame = min(total_frames, best_frame + 1)

            start_byte = start_frame * frame_width
            end_byte = end_frame * frame_width
            trimmed_segment = segment._spawn(segment._data[start_byte:end_byte])

            lead_removed = int(round((start_frame * 1000.0) / frame_rate))
            tail_removed = int(round(((total_frames - end_frame) * 1000.0) / frame_rate))
            return trimmed_segment, lead_removed, tail_removed, (lead_removed > 0 or tail_removed > 0)

        def _resolve_export_audio_path(self, full_path, trim_cfg):
            if not trim_cfg["enabled"]:
                return full_path, {"cache_hit": False, "trimmed": False, "lead_ms": 0, "tail_ms": 0}

            cache_dir = self._trim_cache_dir()
            os.makedirs(cache_dir, exist_ok=True)
            cache_key = self._build_trim_cache_key(full_path, trim_cfg)
            cache_path = os.path.join(cache_dir, f"{cache_key}.mp3")

            if os.path.exists(cache_path):
                try:
                    if os.path.getsize(cache_path) > 0:
                        return cache_path, {"cache_hit": True, "trimmed": False, "lead_ms": 0, "tail_ms": 0}
                except OSError:
                    pass

            source_segment = AudioSegment.from_file(full_path)
            trimmed_segment, lead_ms, tail_ms, changed = self._trim_audio_segment_boundaries(source_segment, trim_cfg)

            # Safety guard: trimming must never increase duration. If it does,
            # discard the generated segment and cache the original audio instead.
            if len(trimmed_segment) > len(source_segment):
                print(
                    f"[trim] ERROR: Trim result longer than original for {full_path} "
                    f"(original={len(source_segment)}ms, trimmed={len(trimmed_segment)}ms). "
                    "Discarding trimmed output and caching original clip.",
                    flush=True,
                )
                trimmed_segment = source_segment
                lead_ms = 0
                tail_ms = 0
                changed = False

            exported = trimmed_segment.export(cache_path, format="mp3", bitrate="128k")
            if hasattr(exported, "close"):
                exported.close()
            if not os.path.exists(cache_path) or os.path.getsize(cache_path) <= 0:
                raise RuntimeError(f"Trim cache export produced invalid file: {cache_path}")

            return cache_path, {
                "cache_hit": False,
                "trimmed": bool(changed),
                "lead_ms": int(lead_ms),
                "tail_ms": int(tail_ms),
            }

        def _collect_merge_timeline(self, progress_callback=None, merge_started_at=None, export_config=None, log_callback=None, temp_dir=None):
            chunks = self.load_chunks()
            timeline = []
            timeline_size_bytes = 0
            trim_cfg = self._resolve_trim_config(export_config)
            trim_stats = {
                "enabled": trim_cfg["enabled"],
                "processed": 0,
                "cache_hits": 0,
                "trimmed_clips": 0,
                "lead_ms_removed": 0,
                "tail_ms_removed": 0,
                "fallback_originals": 0,
            }
            # Build ordered list of (kind, chunk[, full_path]) preserving chunk order
            ordered_items = []
            for chunk in chunks:
                if chunk.get("type") == "silence":
                    ordered_items.append(("silence", chunk))
                elif chunk.get("status") == "done":
                    path = chunk.get("audio_path")
                    if not path:
                        continue
                    full_path = os.path.join(self.root_dir, path)
                    if not os.path.exists(full_path):
                        continue
                    ordered_items.append(("audio", chunk, full_path))
            eligible_chunks = [(c, p) for kind, c, p in (x for x in ordered_items if x[0] == "audio")]
            total_candidates = len(eligible_chunks)

            if progress_callback:
                progress_callback({
                    "stage": "preparing",
                    "chapter_index": 0,
                    "total_chapters": 0,
                    "chapter_label": "Scanning completed clips",
                    "elapsed_seconds": 0.0,
                    "merged_duration_seconds": 0.0,
                    "estimated_size_bytes": 0,
                    "output_file_size_bytes": 0,
                    "total_items": total_candidates,
                    "processed_items": 0,
                    "remaining_items": total_candidates,
                    "percent_complete": 0.0,
                    "eta_seconds": None,
                    "trim_enabled": bool(trim_cfg["enabled"]),
                })

            # Pre-process audio clips (trim/index), then build final ordered timeline
            last_prepare_bucket = -1
            prepare_started_at = merge_started_at or time.time()
            clip_times = []
            # Map uid -> resolved_path for audio chunks after trim processing
            resolved_audio_paths = {}
            for processed_index, (chunk, full_path) in enumerate(eligible_chunks, start=1):
                clip_start = time.time()
                resolved_path = full_path
                if trim_cfg["enabled"]:
                    try:
                        resolved_path, trim_info = self._resolve_export_audio_path(full_path, trim_cfg)
                        trim_stats["processed"] += 1
                        if trim_info["cache_hit"]:
                            trim_stats["cache_hits"] += 1
                        if trim_info["trimmed"]:
                            trim_stats["trimmed_clips"] += 1
                            trim_stats["lead_ms_removed"] += int(trim_info["lead_ms"])
                            trim_stats["tail_ms_removed"] += int(trim_info["tail_ms"])
                    except Exception:
                        trim_stats["fallback_originals"] += 1
                        resolved_path = full_path
                clip_times.append(time.time() - clip_start)
                resolved_audio_paths[chunk["uid"]] = resolved_path
                if progress_callback and total_candidates > 0:
                    percent_complete = (processed_index / total_candidates) * 100.0
                    progress_bucket = int(percent_complete // 5)
                    should_emit = (
                        processed_index == 1
                        or processed_index == total_candidates
                        or progress_bucket > last_prepare_bucket
                    )
                    if should_emit:
                        elapsed = max(0.0, time.time() - prepare_started_at)
                        rate = processed_index / elapsed if elapsed > 0 else 0.0
                        remaining_items = max(total_candidates - processed_index, 0)
                        avg_clip_s = sum(clip_times) / len(clip_times) if clip_times else 0.0
                        eta_seconds = avg_clip_s * remaining_items
                        if trim_cfg["enabled"]:
                            chapter_label = (
                                f"Trimming clips: {int(round(percent_complete))}% "
                                f"({processed_index}/{total_candidates}, {remaining_items} remaining)"
                            )
                        else:
                            chapter_label = (
                                f"Indexing clips: {int(round(percent_complete))}% "
                                f"({processed_index}/{total_candidates}, {remaining_items} remaining)"
                            )
                            eta_seconds = (remaining_items / rate) if rate > 0 and remaining_items > 0 else 0.0
                        progress_callback({
                            "stage": "preparing",
                            "chapter_index": 0,
                            "total_chapters": 0,
                            "chapter_label": chapter_label,
                            "elapsed_seconds": elapsed,
                            "merged_duration_seconds": 0.0,
                            "estimated_size_bytes": timeline_size_bytes,
                            "output_file_size_bytes": 0,
                            "total_items": total_candidates,
                            "processed_items": processed_index,
                            "remaining_items": remaining_items,
                            "percent_complete": percent_complete,
                            "eta_seconds": round(eta_seconds, 1),
                            "trim_enabled": bool(trim_cfg["enabled"]),
                        })
                        if trim_cfg["enabled"]:
                            eta_m, eta_s = divmod(int(eta_seconds), 60)
                            eta_str = f"{eta_m}m{eta_s:02d}s" if eta_m else f"{eta_s}s"
                            print(
                                f"[trim] {processed_index}/{total_candidates} clips"
                                f" — avg {avg_clip_s * 1000:.0f}ms/clip"
                                f" — ETA {eta_str}",
                                flush=True,
                            )
                        last_prepare_bucket = progress_bucket

            # Build final timeline in chunk order, interleaving silence blocks
            for entry in ordered_items:
                kind = entry[0]
                chunk = entry[1]
                if kind == "silence":
                    duration_ms = int(float(chunk.get("silence_duration_s", 1.0)) * 1000)
                    item = {
                        "chunk": chunk,
                        "full_path": None,  # WAV generated later with correct audio format
                        "file_size_bytes": 0,
                        "is_silence_block": True,
                        "silence_duration_ms": duration_ms,
                    }
                    timeline.append(item)
                else:
                    # audio chunk
                    resolved_path = resolved_audio_paths.get(chunk["uid"])
                    if resolved_path is None:
                        continue
                    item = {
                        "chunk": chunk,
                        "full_path": resolved_path,
                        "file_size_bytes": os.path.getsize(resolved_path),
                    }
                    timeline.append(item)
                    timeline_size_bytes += item["file_size_bytes"]

            if log_callback and trim_cfg["enabled"]:
                total_removed_ms = trim_stats["lead_ms_removed"] + trim_stats["tail_ms_removed"]
                log_callback(
                    "Trim cache: "
                    f"{trim_stats['trimmed_clips']} clip(s) trimmed, "
                    f"{trim_stats['cache_hits']} cache hit(s), "
                    f"{total_removed_ms}ms total removed "
                    f"(lead {trim_stats['lead_ms_removed']}ms, tail {trim_stats['tail_ms_removed']}ms), "
                    f"{trim_stats['fallback_originals']} fallback(s)"
                )

            return timeline

        def _group_timeline_by_chapter(self, timeline):
            chapter_groups = []
            current_label = None
            current_items = []

            for item in timeline:
                chunk = item["chunk"]
                chapter_label = (chunk.get("chapter") or "").strip() or "Unlabeled"
                if current_items and chapter_label != current_label:
                    chapter_groups.append((current_label, current_items))
                    current_items = [item]
                    current_label = chapter_label
                else:
                    if not current_items:
                        current_label = chapter_label
                    current_items.append(item)

            if current_items:
                chapter_groups.append((current_label, current_items))

            return chapter_groups

        def _emit_merge_progress(
            self,
            progress_callback,
            merge_started_at,
            *,
            stage,
            chapter_index,
            total_chapters,
            chapter_label,
            estimated_size_bytes,
            output_path=None,
            merged_duration_seconds=0.0,
        ):
            if not progress_callback:
                return
            progress_callback({
                "stage": stage,
                "chapter_index": chapter_index,
                "total_chapters": total_chapters,
                "chapter_label": chapter_label,
                "elapsed_seconds": time.time() - merge_started_at,
                "merged_duration_seconds": merged_duration_seconds,
                "estimated_size_bytes": estimated_size_bytes,
                "output_file_size_bytes": os.path.getsize(output_path) if output_path and os.path.exists(output_path) else 0,
            })

        def _create_silence_assets(self, temp_dir, export_config=None, reference_audio_path=None):
            default_ms = getattr(export_config, "silence_between_speakers_ms", DEFAULT_PAUSE_MS)
            same_ms = getattr(export_config, "silence_same_speaker_ms", SAME_SPEAKER_PAUSE_MS)
            chapter_end_ms = getattr(export_config, "silence_end_of_chapter_ms", 3000)
            paragraph_ms = getattr(export_config, "silence_paragraph_ms", 750)

            reference_frame_rate = None
            reference_channels = None
            reference_sample_width = None
            if reference_audio_path and os.path.exists(reference_audio_path):
                try:
                    reference_segment = AudioSegment.from_file(reference_audio_path)
                    reference_frame_rate = int(reference_segment.frame_rate or 0) or None
                    reference_channels = int(reference_segment.channels or 0) or None
                    reference_sample_width = int(reference_segment.sample_width or 0) or None
                except Exception:
                    reference_frame_rate = None
                    reference_channels = None
                    reference_sample_width = None

            # Silence files must be MP3 so they are the same codec as the audio clips.
            # FFmpeg's concat demuxer skips files whose codec differs from the first
            # entry in the concat list, which caused all silence to be dropped silently.
            default_silence_path = os.path.join(temp_dir, "pause_default.mp3")
            same_silence_path = os.path.join(temp_dir, "pause_same_speaker.mp3")
            chapter_end_silence_path = os.path.join(temp_dir, "pause_chapter_end.mp3")
            paragraph_silence_path = os.path.join(temp_dir, "pause_paragraph.mp3")

            def _write(path, duration_ms):
                seg = AudioSegment.silent(duration=max(0, duration_ms))
                if reference_frame_rate:
                    seg = seg.set_frame_rate(reference_frame_rate)
                if reference_channels:
                    seg = seg.set_channels(reference_channels)
                if reference_sample_width:
                    seg = seg.set_sample_width(reference_sample_width)
                exp = seg.export(path, format="mp3", parameters=["-q:a", "2"])
                if hasattr(exp, "close"):
                    exp.close()

            _write(default_silence_path, default_ms)
            _write(same_silence_path, same_ms)
            _write(chapter_end_silence_path, chapter_end_ms)
            _write(paragraph_silence_path, paragraph_ms)

            return {
                "default_path": default_silence_path,
                "same_path": same_silence_path,
                "chapter_end_path": chapter_end_silence_path,
                "paragraph_path": paragraph_silence_path,
                "default_size_bytes": os.path.getsize(default_silence_path),
                "same_size_bytes": os.path.getsize(same_silence_path),
                "chapter_end_size_bytes": os.path.getsize(chapter_end_silence_path),
                "paragraph_size_bytes": os.path.getsize(paragraph_silence_path),
                # Reference format for use when generating additional silence WAVs
                "_write": _write,
            }

        @staticmethod
        def _pick_silence(prev_item, curr_item, silence_assets, *, is_chapter_boundary=False):
            """Return (pause_path, pause_size_bytes) for the gap between two timeline items."""
            if is_chapter_boundary:
                return silence_assets["chapter_end_path"], silence_assets["chapter_end_size_bytes"]
            prev_pid = prev_item["chunk"].get("paragraph_id")
            curr_pid = curr_item["chunk"].get("paragraph_id")
            if prev_pid and curr_pid and prev_pid != curr_pid:
                return silence_assets["paragraph_path"], silence_assets["paragraph_size_bytes"]
            if prev_item["chunk"]["speaker"] == curr_item["chunk"]["speaker"]:
                return silence_assets["same_path"], silence_assets["same_size_bytes"]
            return silence_assets["default_path"], silence_assets["default_size_bytes"]

        def _run_ffmpeg_concat(self, concat_path, output_path, codec_args, progress_tick=None):
            command = [
                get_ffmpeg_exe(),
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-progress",
                "pipe:1",
                "-nostats",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                concat_path,
                "-vn",
                *codec_args,
                output_path,
            ]
            if os.path.exists(output_path):
                try:
                    os.remove(output_path)
                except OSError:
                    pass

            def _progress_callback(_out_time_seconds, _phase_percent):
                if progress_tick:
                    progress_tick()

            returncode, stdout_text, stderr_text = self._run_ffmpeg_with_progress(
                command,
                duration_seconds=None,
                progress_callback=_progress_callback if progress_tick else None,
            )
            if returncode != 0:
                message = (stderr_text or stdout_text or "").strip()
                if message:
                    message = message[-600:]
                return False, message or f"ffmpeg exited with code {returncode}"

            if not os.path.exists(output_path):
                return False, "ffmpeg completed without creating an output file"

            output_size = os.path.getsize(output_path)
            if output_size <= 0:
                try:
                    os.remove(output_path)
                except OSError:
                    pass
                return False, "ffmpeg produced an empty output file"

            # Very short/silent MP3 outputs can be smaller than 1KB while still valid.
            # Validate via ffprobe duration/stream metadata instead of a hard size floor.
            summary = self._ffprobe_audio_summary(output_path)
            if not summary.get("ok"):
                try:
                    os.remove(output_path)
                except OSError:
                    pass
                return False, f"ffmpeg produced an unreadable file ({output_size} bytes): {summary.get('error')}"

            try:
                duration_seconds = float(summary.get("duration") or 0.0)
            except (TypeError, ValueError):
                duration_seconds = 0.0
            if duration_seconds <= 0.0 or not summary.get("codec"):
                try:
                    os.remove(output_path)
                except OSError:
                    pass
                return (
                    False,
                    f"ffmpeg produced an invalid file ({output_size} bytes): "
                    f"codec={summary.get('codec')} duration={summary.get('duration')}",
                )

            return True, output_path

        @staticmethod
        def _parse_concat_entries(concat_path):
            entries = []
            try:
                with open(concat_path, "r", encoding="utf-8") as handle:
                    for line in handle:
                        stripped = line.strip()
                        if not stripped.startswith("file '") or not stripped.endswith("'"):
                            continue
                        raw = stripped[6:-1]
                        path = raw.replace(r"'\''", "'").replace("\\\\", "\\")
                        entries.append(path)
            except OSError:
                return []
            return entries

        @staticmethod
        def _ffprobe_audio_summary(path):
            cmd = [
                get_ffprobe_exe(),
                "-v",
                "error",
                "-show_entries",
                "stream=codec_name,sample_rate,channels,sample_fmt,bits_per_sample:format=format_name,duration,bit_rate,size",
                "-of",
                "json",
                path,
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
            if result.returncode != 0:
                err = (result.stderr or result.stdout or "").strip()
                try:
                    segment = AudioSegment.from_file(path)
                    ext = os.path.splitext(path)[1].lstrip(".").lower()
                    return {
                        "ok": True,
                        "codec": ext or "audio",
                        "sample_rate": str(segment.frame_rate),
                        "channels": segment.channels,
                        "sample_fmt": None,
                        "bits_per_sample": segment.sample_width * 8,
                        "format_name": ext or None,
                        "duration": str(segment.duration_seconds),
                        "bit_rate": None,
                        "size": str(os.path.getsize(path)) if os.path.exists(path) else None,
                    }
                except Exception:
                    return {"ok": False, "error": err[-500:] if err else f"ffprobe exit {result.returncode}"}
            try:
                payload = json.loads(result.stdout or "{}")
            except json.JSONDecodeError:
                return {"ok": False, "error": "ffprobe returned invalid JSON"}

            stream = (payload.get("streams") or [{}])[0]
            fmt = payload.get("format") or {}
            return {
                "ok": True,
                "codec": stream.get("codec_name"),
                "sample_rate": stream.get("sample_rate"),
                "channels": stream.get("channels"),
                "sample_fmt": stream.get("sample_fmt"),
                "bits_per_sample": stream.get("bits_per_sample"),
                "format_name": fmt.get("format_name"),
                "duration": fmt.get("duration"),
                "bit_rate": fmt.get("bit_rate"),
                "size": fmt.get("size"),
            }

        def _log_concat_failure_diagnostics(self, concat_path, output_path, failure_message, log_callback=None):
            if not log_callback:
                return
            log_callback(f"[diag] MP3 concat export failed for {output_path}")
            log_callback(f"[diag] ffmpeg error: {failure_message}")
            entries = self._parse_concat_entries(concat_path)
            log_callback(f"[diag] concat file entries: {len(entries)}")
            if not entries:
                return

            sample_indices = sorted(set([0, 1, 2, max(len(entries) - 2, 0), len(entries) - 1]))
            probed = []
            for idx in sample_indices:
                if not (0 <= idx < len(entries)):
                    continue
                src = entries[idx]
                summary = self._ffprobe_audio_summary(src)
                probed.append((idx, src, summary))
                basename = os.path.basename(src)
                if summary.get("ok"):
                    log_callback(
                        f"[diag] input[{idx}] {basename}: "
                        f"codec={summary.get('codec')}, sr={summary.get('sample_rate')}, "
                        f"ch={summary.get('channels')}, fmt={summary.get('sample_fmt')}, "
                        f"bps={summary.get('bits_per_sample')}, dur={summary.get('duration')}, "
                        f"br={summary.get('bit_rate')}, size={summary.get('size')}"
                    )
                else:
                    log_callback(f"[diag] input[{idx}] {basename}: ffprobe error: {summary.get('error')}")

            layouts = set()
            codecs = set()
            formats = set()
            for _idx, _src, summary in probed:
                if not summary.get("ok"):
                    continue
                layouts.add((summary.get("sample_rate"), summary.get("channels"), summary.get("sample_fmt"), summary.get("bits_per_sample")))
                codecs.add(summary.get("codec"))
                formats.add(summary.get("format_name"))
            log_callback(f"[diag] sampled input codecs={sorted(c for c in codecs if c)} formats={sorted(f for f in formats if f)}")
            log_callback(f"[diag] sampled input layouts={sorted(layouts)}")

        def _export_concat_mp3(self, concat_path, output_path, progress_tick=None, log_callback=None):
            mp3_success, mp3_result = self._run_ffmpeg_concat(
                concat_path,
                output_path,
                ["-c:a", "libmp3lame", "-q:a", "2"],
                progress_tick=progress_tick,
            )
            if mp3_success:
                return True, mp3_result
            self._log_concat_failure_diagnostics(concat_path, output_path, mp3_result, log_callback=log_callback)
            return False, f"MP3 concat export failed: {mp3_result}"

        def _optimized_export_part_basename(self, part_index):
            title = self._current_script_title().strip() or "Project"
            safe_title = re.sub(r"[^A-Za-z0-9]+", "-", title).strip("-").lower() or "project"
            return f"{safe_title}-{part_index:02d}.mp3"

        @staticmethod
        def _is_normalization_enabled(export_config):
            if export_config is None:
                return True
            return bool(getattr(export_config, "normalize_enabled", True))

        @staticmethod
        def _extract_json_object(text):
            if not text:
                return None
            start = text.find("{")
            while start != -1:
                depth = 0
                for idx in range(start, len(text)):
                    char = text[idx]
                    if char == "{":
                        depth += 1
                    elif char == "}":
                        depth -= 1
                        if depth == 0:
                            candidate = text[start:idx + 1]
                            try:
                                return json.loads(candidate)
                            except json.JSONDecodeError:
                                break
                start = text.find("{", start + 1)
            return None

        def _detect_channel_count(self, input_path):
            try:
                return int(AudioSegment.from_file(input_path).channels or 1)
            except Exception:
                return 1

        @staticmethod
        def _loudnorm_codec_args_for_path(path, *, sample_rate=None, channels=None):
            ext = os.path.splitext(path)[1].lower()
            args = []
            if ext == ".wav":
                args.extend(["-c:a", "pcm_s16le"])
            else:
                args.extend(["-c:a", "libmp3lame", "-q:a", "2"])
            if sample_rate:
                args.extend(["-ar", str(int(sample_rate))])
            if channels:
                args.extend(["-ac", str(int(channels))])
            return args

        @staticmethod
        def _run_ffmpeg_with_progress(command, duration_seconds=None, progress_callback=None):
            """Run ffmpeg and emit estimated progress derived from out_time_ms."""
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )
            stderr_lines = []
            last_percent = -1.0
            safe_duration = max(float(duration_seconds or 0.0), 0.001)

            while True:
                line = process.stderr.readline()
                if not line:
                    if process.poll() is not None:
                        break
                    continue

                stderr_lines.append(line)
                stripped = line.strip()
                if not stripped:
                    continue

                if progress_callback and stripped.startswith("out_time_ms="):
                    try:
                        out_time_seconds = max(0.0, int(stripped.split("=", 1)[1]) / 1_000_000.0)
                    except (TypeError, ValueError):
                        continue
                    phase_percent = min(100.0, (out_time_seconds / safe_duration) * 100.0)
                    if phase_percent >= last_percent + 1.0 or phase_percent >= 99.9:
                        last_percent = phase_percent
                        progress_callback(out_time_seconds, phase_percent)

            stdout_text = ""
            if process.stdout:
                try:
                    stdout_text = process.stdout.read() or ""
                except Exception:
                    stdout_text = ""

            stderr_text = "".join(stderr_lines)
            if progress_callback:
                progress_callback(float(duration_seconds or 0.0), 100.0)
            return process.returncode, stdout_text, stderr_text

        def _resolve_export_normalization_config(self, export_config=None):
            if export_config is not None:
                return export_config
            app_config = self._load_app_config()
            export_raw = app_config.get("export", {}) if isinstance(app_config, dict) else {}
            if not isinstance(export_raw, dict):
                export_raw = {}
            return SimpleNamespace(**export_raw)

        def _normalize_audio_file(
            self,
            input_path,
            export_config=None,
            allow_short_single_pass=False,
            short_seconds_threshold=12.0,
            progress_callback=None,
            log_callback=None,
        ):
            """Apply two-pass EBU R128 loudness normalization in-place."""
            export_config = self._resolve_export_normalization_config(export_config)
            if not self._is_normalization_enabled(export_config):
                return True, input_path

            if not os.path.exists(input_path):
                return False, f"Normalization input does not exist: {input_path}"

            duration_seconds = 0.0
            input_sample_rate = 48000
            input_channels = 1
            try:
                input_segment = AudioSegment.from_file(input_path)
                duration_seconds = float(input_segment.duration_seconds or 0.0)
                input_sample_rate = int(input_segment.frame_rate or 48000)
                input_channels = max(1, int(input_segment.channels or 1))
            except Exception:
                duration_seconds = 0.0
            is_short_clip = duration_seconds > 0 and duration_seconds <= float(short_seconds_threshold)

            channels = input_channels
            target_lufs = float(
                getattr(export_config, "normalize_target_lufs_stereo", -16.0)
                if channels > 1 else getattr(export_config, "normalize_target_lufs_mono", -18.0)
            )
            target_tp = float(getattr(export_config, "normalize_true_peak_dbtp", -1.0))
            target_lra = float(getattr(export_config, "normalize_lra", 11.0))

            base_filter = (
                f"loudnorm=I={target_lufs}:TP={target_tp}:LRA={target_lra}:"
                "print_format=json"
            )
            ext = os.path.splitext(input_path)[1]

            progress_log_state = {}
            progress_emit_state = {"overall_percent": -1.0}

            def _emit_progress(phase, out_time_seconds=0.0, phase_percent=0.0):
                if progress_callback:
                    phase_percent = max(0.0, min(100.0, float(phase_percent or 0.0)))
                    if phase == "pass1-measure":
                        overall_percent = phase_percent * 0.5
                    elif phase == "pass2-apply":
                        overall_percent = 50.0 + (phase_percent * 0.5)
                    else:
                        overall_percent = phase_percent
                    should_emit = (
                        overall_percent >= 99.9
                        or (overall_percent - progress_emit_state["overall_percent"]) >= 1.0
                    )
                    if should_emit:
                        progress_emit_state["overall_percent"] = overall_percent
                        progress_callback({
                            "phase": phase,
                            "processed_seconds": float(out_time_seconds or 0.0),
                            "phase_percent": phase_percent,
                            "overall_percent": overall_percent,
                        })
                if log_callback:
                    bucket = int(float(phase_percent or 0.0) // 20) * 20
                    previous_bucket = progress_log_state.get(phase, -10)
                    if bucket > previous_bucket and bucket <= 100:
                        progress_log_state[phase] = bucket
                        readable_phase = {
                            "pass1-measure": "Normalization pass 1/2",
                            "pass2-apply": "Normalization pass 2/2",
                            "single-pass": "Normalization single-pass fallback",
                        }.get(phase, "Normalization")
                        log_callback(f"{readable_phase}: {bucket}%")

            def _single_pass_normalize(reason):
                pass1_filter = (
                    f"loudnorm=I={target_lufs}:TP={target_tp}:LRA={target_lra}:"
                    "linear=false:print_format=summary"
                )
                temp_output = f"{input_path}.normalized{ext}"
                cmd = [
                    get_ffmpeg_exe(),
                    "-y",
                    "-hide_banner",
                    "-progress",
                    "pipe:2",
                    "-nostats",
                    "-loglevel",
                    "error",
                    "-i",
                    input_path,
                    "-vn",
                    "-af",
                    pass1_filter,
                    *self._loudnorm_codec_args_for_path(
                        input_path,
                        sample_rate=input_sample_rate,
                        channels=input_channels,
                    ),
                    temp_output,
                ]
                if log_callback:
                    log_callback("Normalization fallback: running single-pass loudnorm for short clip...")
                returncode, stdout_text, stderr_text = self._run_ffmpeg_with_progress(
                    cmd,
                    duration_seconds=duration_seconds,
                    progress_callback=lambda out_time, pct: _emit_progress("single-pass", out_time, pct),
                )
                if returncode != 0:
                    try:
                        if os.path.exists(temp_output):
                            os.remove(temp_output)
                    except OSError:
                        pass
                    stderr_tail = (stderr_text or stdout_text or "").strip()[-600:]
                    return False, f"{reason}; short-clip loudnorm fallback failed: {stderr_tail or f'exit {returncode}'}"
                if not os.path.exists(temp_output) or os.path.getsize(temp_output) < 1024:
                    try:
                        if os.path.exists(temp_output):
                            os.remove(temp_output)
                    except OSError:
                        pass
                    return False, f"{reason}; short-clip loudnorm fallback produced invalid output"
                os.replace(temp_output, input_path)
                return True, input_path

            measure_cmd = [
                get_ffmpeg_exe(),
                "-y",
                "-hide_banner",
                "-progress",
                "pipe:2",
                "-nostats",
                "-loglevel",
                "info",
                "-i",
                input_path,
                "-vn",
                "-af",
                base_filter,
                "-f",
                "null",
                "-",
            ]
            if log_callback:
                log_callback("Normalization pass 1/2: measuring integrated loudness...")
            returncode, stdout_text, stderr_text = self._run_ffmpeg_with_progress(
                measure_cmd,
                duration_seconds=duration_seconds,
                progress_callback=lambda out_time, pct: _emit_progress("pass1-measure", out_time, pct),
            )
            if returncode != 0:
                if allow_short_single_pass and is_short_clip:
                    return _single_pass_normalize("loudnorm pass 1 failed")
                stderr_tail = (stderr_text or stdout_text or "").strip()[-600:]
                return False, f"loudnorm pass 1 failed: {stderr_tail or f'exit {returncode}'}"

            measure_text = f"{stderr_text}\n{stdout_text}"
            measured = self._extract_json_object(measure_text)
            required_keys = (
                "input_i",
                "input_tp",
                "input_lra",
                "input_thresh",
                "target_offset",
            )
            if not measured or any(key not in measured for key in required_keys):
                if allow_short_single_pass and is_short_clip:
                    return _single_pass_normalize("loudnorm pass 1 did not produce valid measurement JSON")
                return False, "loudnorm pass 1 did not produce valid measurement JSON"

            try:
                measured_i = float(measured["input_i"])
                measured_tp = float(measured["input_tp"])
                measured_lra = float(measured["input_lra"])
                measured_thresh = float(measured["input_thresh"])
                target_offset = float(measured["target_offset"])
            except (TypeError, ValueError):
                if allow_short_single_pass and is_short_clip:
                    return _single_pass_normalize("loudnorm pass 1 returned non-numeric values")
                return False, "loudnorm pass 1 returned invalid measurement values"

            pass2_filter = (
                f"loudnorm=I={target_lufs}:TP={target_tp}:LRA={target_lra}:"
                f"measured_I={measured_i}:measured_TP={measured_tp}:"
                f"measured_LRA={measured_lra}:measured_thresh={measured_thresh}:"
                f"offset={target_offset}:linear=true:print_format=summary"
            )
            temp_output = f"{input_path}.normalized{ext}"
            normalize_cmd = [
                get_ffmpeg_exe(),
                "-y",
                "-hide_banner",
                "-progress",
                "pipe:2",
                "-nostats",
                "-loglevel",
                "error",
                "-i",
                input_path,
                "-vn",
                "-af",
                pass2_filter,
                *self._loudnorm_codec_args_for_path(
                    input_path,
                    sample_rate=input_sample_rate,
                    channels=input_channels,
                ),
                temp_output,
            ]
            if log_callback:
                log_callback("Normalization pass 2/2: applying measured loudness correction...")
            returncode, stdout_text, stderr_text = self._run_ffmpeg_with_progress(
                normalize_cmd,
                duration_seconds=duration_seconds,
                progress_callback=lambda out_time, pct: _emit_progress("pass2-apply", out_time, pct),
            )
            if returncode != 0:
                if allow_short_single_pass and is_short_clip:
                    return _single_pass_normalize("loudnorm pass 2 failed")
                try:
                    if os.path.exists(temp_output):
                        os.remove(temp_output)
                except OSError:
                    pass
                stderr_tail = (stderr_text or stdout_text or "").strip()[-600:]
                return False, f"loudnorm pass 2 failed: {stderr_tail or f'exit {returncode}'}"

            if not os.path.exists(temp_output) or os.path.getsize(temp_output) < 1024:
                try:
                    if os.path.exists(temp_output):
                        os.remove(temp_output)
                except OSError:
                    pass
                return False, "loudnorm pass 2 produced an invalid output file"

            os.replace(temp_output, input_path)
            return True, input_path

        def _call_normalize_audio_file(self, input_path, export_config=None, progress_callback=None, log_callback=None, **kwargs):
            """
            Backward-compatible wrapper around `_normalize_audio_file`.
            Some tests monkeypatch `_normalize_audio_file` with a narrower signature.
            """
            normalize_fn = self._normalize_audio_file
            call_kwargs = {}
            try:
                accepted = set(inspect.signature(normalize_fn).parameters.keys())
            except (TypeError, ValueError):
                accepted = set()

            if "export_config" in accepted:
                call_kwargs["export_config"] = export_config
            if "progress_callback" in accepted and progress_callback is not None:
                call_kwargs["progress_callback"] = progress_callback
            if "log_callback" in accepted and log_callback is not None:
                call_kwargs["log_callback"] = log_callback
            for key, value in kwargs.items():
                if key in accepted:
                    call_kwargs[key] = value

            return normalize_fn(input_path, **call_kwargs)

        def merge_audio(self, progress_callback=None, log_callback=None, export_config=None, chapter=None):
            merge_started_at = time.time()
            output_filename = "cloned_audiobook.mp3"
            output_path = os.path.join(self.exports_dir, output_filename)
            temp_dir = _make_runtime_temp_dir(self, "merge_audio_")
            timeline = self._collect_merge_timeline(
                progress_callback=progress_callback,
                merge_started_at=merge_started_at,
                export_config=export_config,
                log_callback=log_callback,
                temp_dir=temp_dir,
            )

            if chapter:
                chapter = chapter.strip()
                timeline = [
                    item for item in timeline
                    if item.get("is_silence_block") or
                       (item.get("chunk") or {}).get("chapter", "").strip() == chapter
                ]

            if not timeline:
                shutil.rmtree(temp_dir, ignore_errors=True)
                return False, "No audio segments found"

            chapter_groups = self._group_timeline_by_chapter(timeline)
            concat_path = os.path.join(temp_dir, "concat.txt")

            try:
                first_audio_path = next(
                    (item["full_path"] for item in timeline if not item.get("is_silence_block")), None
                )
                silence_assets = self._create_silence_assets(
                    temp_dir,
                    export_config,
                    reference_audio_path=first_audio_path,
                )

                # Generate silence block WAVs now that reference audio format is known
                _write_silence = silence_assets["_write"]
                for item in timeline:
                    if item.get("is_silence_block") and item["full_path"] is None:
                        wav_path = os.path.join(temp_dir, f"silence_block_{item['chunk']['uid']}.wav")
                        _write_silence(wav_path, item["silence_duration_ms"])
                        item["full_path"] = wav_path
                        item["file_size_bytes"] = os.path.getsize(wav_path)

                estimated_size_bytes = 0
                previous_item = None
                total_same = 0
                total_diff = 0
                total_para = 0
                total_chapter_end = 0

                with open(concat_path, "w", encoding="utf-8") as concat_file:
                    for chapter_index, (chapter_label, chapter_items) in enumerate(chapter_groups, start=1):
                        if previous_item is not None and chapter_items:
                            first_item = chapter_items[0]
                            if not previous_item.get("is_silence_block") and not first_item.get("is_silence_block"):
                                pause_path, pause_size = self._pick_silence(
                                    previous_item, first_item, silence_assets, is_chapter_boundary=True
                                )
                                self._write_concat_line(concat_file, pause_path)
                                estimated_size_bytes += pause_size
                                total_chapter_end += 1

                        chapter_same = 0
                        chapter_diff = 0
                        chapter_para = 0
                        prev_item_in_chapter = None
                        for item in chapter_items:
                            if prev_item_in_chapter is not None:
                                if not item.get("is_silence_block") and not prev_item_in_chapter.get("is_silence_block"):
                                    pause_path, pause_size = self._pick_silence(prev_item_in_chapter, item, silence_assets)
                                    self._write_concat_line(concat_file, pause_path)
                                    estimated_size_bytes += pause_size
                                    # Tally silence type for diagnostics
                                    prev_pid = prev_item_in_chapter["chunk"].get("paragraph_id")
                                    curr_pid = item["chunk"].get("paragraph_id")
                                    if prev_pid and curr_pid and prev_pid != curr_pid:
                                        chapter_para += 1
                                    elif prev_item_in_chapter["chunk"].get("speaker") == item["chunk"].get("speaker"):
                                        chapter_same += 1
                                    else:
                                        chapter_diff += 1
                            self._write_concat_line(concat_file, item["full_path"])
                            estimated_size_bytes += item["file_size_bytes"]
                            prev_item_in_chapter = item

                        previous_item = prev_item_in_chapter if prev_item_in_chapter is not None else previous_item
                        total_same += chapter_same
                        total_diff += chapter_diff
                        total_para += chapter_para
                        if log_callback:
                            log_callback(
                                f"  Chapter '{chapter_label}': {chapter_same} same-speaker, "
                                f"{chapter_diff} speaker-change, {chapter_para} paragraph silences"
                            )
                        self._emit_merge_progress(
                            progress_callback,
                            merge_started_at,
                            stage="assembling",
                            chapter_index=chapter_index,
                            total_chapters=len(chapter_groups),
                            chapter_label=chapter_label,
                            estimated_size_bytes=estimated_size_bytes,
                            output_path=output_path,
                        )

                if log_callback:
                    log_callback(
                        f"Export totals: {total_chapter_end} chapter-end, {total_same} same-speaker, "
                        f"{total_diff} speaker-change, {total_para} paragraph silences"
                    )

                self._emit_merge_progress(
                    progress_callback,
                    merge_started_at,
                    stage="exporting",
                    chapter_index=len(chapter_groups),
                    total_chapters=len(chapter_groups),
                    chapter_label=chapter_groups[-1][0],
                    estimated_size_bytes=estimated_size_bytes,
                    output_path=output_path,
                )

                success, export_result = self._export_concat_mp3(
                    concat_path,
                    output_path,
                    progress_tick=lambda: self._emit_merge_progress(
                        progress_callback,
                        merge_started_at,
                        stage="exporting",
                        chapter_index=len(chapter_groups),
                        total_chapters=len(chapter_groups),
                        chapter_label=chapter_groups[-1][0],
                        estimated_size_bytes=estimated_size_bytes,
                        output_path=output_path,
                    ),
                    log_callback=log_callback,
                )
                if not success:
                    return False, f"ffmpeg merge failed: {export_result}"

                output_path = export_result
                output_filename = os.path.basename(output_path)

                self._emit_merge_progress(
                    progress_callback,
                    merge_started_at,
                    stage="normalizing",
                    chapter_index=len(chapter_groups),
                    total_chapters=len(chapter_groups),
                    chapter_label=chapter_groups[-1][0],
                    estimated_size_bytes=estimated_size_bytes,
                    output_path=output_path,
                )
                normalized, normalize_result = self._call_normalize_audio_file(
                    output_path,
                    export_config=export_config,
                    progress_callback=lambda info: self._emit_merge_progress(
                        progress_callback,
                        merge_started_at,
                        stage="normalizing",
                        chapter_index=len(chapter_groups),
                        total_chapters=len(chapter_groups),
                        chapter_label=(
                            f"{chapter_groups[-1][0]} "
                            f"({info.get('phase', 'normalizing')}, {int(round(info.get('overall_percent', 0.0)))}%)"
                        ),
                        estimated_size_bytes=estimated_size_bytes,
                        output_path=output_path,
                    ),
                    log_callback=log_callback,
                )
                if not normalized:
                    return False, f"Audio normalization failed: {normalize_result}"

                self._emit_merge_progress(
                    progress_callback,
                    merge_started_at,
                    stage="complete",
                    chapter_index=len(chapter_groups),
                    total_chapters=len(chapter_groups),
                    chapter_label=chapter_groups[-1][0],
                    estimated_size_bytes=estimated_size_bytes,
                    output_path=output_path,
                )
                return True, output_filename
            finally:
                shutil.rmtree(temp_dir, ignore_errors=True)

        def export_optimized_mp3_zip(self, progress_callback=None, log_callback=None, export_config=None, max_part_seconds=7200):
            merge_started_at = time.time()
            zip_path = os.path.join(self.exports_dir, "optimized_audiobook.zip")
            temp_dir = _make_runtime_temp_dir(self, "optimized_export_")
            timeline = self._collect_merge_timeline(
                progress_callback=progress_callback,
                merge_started_at=merge_started_at,
                export_config=export_config,
                log_callback=log_callback,
                temp_dir=temp_dir,
            )
            if not timeline:
                shutil.rmtree(temp_dir, ignore_errors=True)
                return False, "No audio segments found"

            chapter_groups = self._group_timeline_by_chapter(timeline)
            chapter_end_ms = getattr(export_config, "silence_end_of_chapter_ms", 3000)

            try:
                first_audio_path = next(
                    (item["full_path"] for item in timeline if not item.get("is_silence_block")), None
                )
                silence_assets = self._create_silence_assets(
                    temp_dir,
                    export_config,
                    reference_audio_path=first_audio_path,
                )

                # Generate silence block WAVs now that reference audio format is known
                _write_silence = silence_assets["_write"]
                for item in timeline:
                    if item.get("is_silence_block") and item["full_path"] is None:
                        wav_path = os.path.join(temp_dir, f"silence_block_{item['chunk']['uid']}.wav")
                        _write_silence(wav_path, item["silence_duration_ms"])
                        item["full_path"] = wav_path
                        item["file_size_bytes"] = os.path.getsize(wav_path)

                chapter_exports = []
                estimated_size_bytes = 0
                total_same = 0
                total_diff = 0
                total_para = 0

                for chapter_index, (chapter_label, chapter_items) in enumerate(chapter_groups, start=1):
                    concat_path = os.path.join(temp_dir, f"chapter_{chapter_index:03d}.txt")
                    chapter_output_path = os.path.join(temp_dir, f"chapter_{chapter_index:03d}.mp3")
                    chapter_estimated_size = 0
                    chapter_same = 0
                    chapter_diff = 0
                    chapter_para = 0
                    prev_item_in_chapter = None

                    with open(concat_path, "w", encoding="utf-8") as concat_file:
                        for item in chapter_items:
                            if prev_item_in_chapter is not None:
                                if not item.get("is_silence_block") and not prev_item_in_chapter.get("is_silence_block"):
                                    pause_path, pause_size = self._pick_silence(prev_item_in_chapter, item, silence_assets)
                                    self._write_concat_line(concat_file, pause_path)
                                    chapter_estimated_size += pause_size
                                    prev_pid = prev_item_in_chapter["chunk"].get("paragraph_id")
                                    curr_pid = item["chunk"].get("paragraph_id")
                                    if prev_pid and curr_pid and prev_pid != curr_pid:
                                        chapter_para += 1
                                    elif prev_item_in_chapter["chunk"].get("speaker") == item["chunk"].get("speaker"):
                                        chapter_same += 1
                                    else:
                                        chapter_diff += 1
                            self._write_concat_line(concat_file, item["full_path"])
                            chapter_estimated_size += item["file_size_bytes"]
                            prev_item_in_chapter = item

                    total_same += chapter_same
                    total_diff += chapter_diff
                    total_para += chapter_para
                    if log_callback:
                        log_callback(
                            f"  Chapter '{chapter_label}': {chapter_same} same-speaker, "
                            f"{chapter_diff} speaker-change, {chapter_para} paragraph silences"
                        )

                    self._emit_merge_progress(
                        progress_callback,
                        merge_started_at,
                        stage="assembling",
                        chapter_index=chapter_index,
                        total_chapters=len(chapter_groups),
                        chapter_label=chapter_label,
                        estimated_size_bytes=estimated_size_bytes + chapter_estimated_size,
                        output_path=chapter_output_path,
                    )

                    success, export_result = self._export_concat_mp3(
                        concat_path,
                        chapter_output_path,
                        log_callback=log_callback,
                    )
                    if not success:
                        return False, f"Failed to export chapter {chapter_label}: {export_result}"

                    chapter_output_path = export_result
                    chapter_duration_seconds = AudioSegment.from_file(chapter_output_path).duration_seconds
                    chapter_size_bytes = os.path.getsize(chapter_output_path)
                    estimated_size_bytes += chapter_size_bytes
                    chapter_exports.append({
                        "label": chapter_label,
                        "path": chapter_output_path,
                        "duration_seconds": chapter_duration_seconds,
                        "file_size_bytes": chapter_size_bytes,
                        "first_item": chapter_items[0],
                        "last_item": chapter_items[-1],
                        "first_speaker": chapter_items[0]["chunk"]["speaker"],
                        "last_speaker": chapter_items[-1]["chunk"]["speaker"],
                    })

                if log_callback:
                    total_chapter_end = max(0, len(chapter_exports) - 1)
                    log_callback(
                        f"Export totals: {total_chapter_end} chapter-end, {total_same} same-speaker, "
                        f"{total_diff} speaker-change, {total_para} paragraph silences"
                    )

                part_groups = []
                current_group = []
                current_duration = 0.0
                for chapter in chapter_exports:
                    chapter_pause_seconds = 0.0
                    if current_group:
                        chapter_pause_seconds = chapter_end_ms / 1000.0
                    proposed_duration = current_duration + chapter_pause_seconds + chapter["duration_seconds"]
                    if current_group and proposed_duration > max_part_seconds:
                        part_groups.append(current_group)
                        current_group = [chapter]
                        current_duration = chapter["duration_seconds"]
                    else:
                        current_group.append(chapter)
                        current_duration = proposed_duration if len(current_group) > 1 else chapter["duration_seconds"]
                if current_group:
                    part_groups.append(current_group)

                part_paths = []
                for part_index, part_group in enumerate(part_groups, start=1):
                    part_basename = self._optimized_export_part_basename(part_index)
                    part_output_path = os.path.join(temp_dir, part_basename)
                    concat_path = os.path.join(temp_dir, f"part_{part_index:03d}.txt")
                    part_estimated_size = 0

                    with open(concat_path, "w", encoding="utf-8") as concat_file:
                        for chapter_offset, chapter in enumerate(part_group):
                            if chapter_offset > 0:
                                self._write_concat_line(concat_file, silence_assets["chapter_end_path"])
                                part_estimated_size += silence_assets["chapter_end_size_bytes"]
                            self._write_concat_line(concat_file, chapter["path"])
                            part_estimated_size += chapter["file_size_bytes"]

                    self._emit_merge_progress(
                        progress_callback,
                        merge_started_at,
                        stage="packing",
                        chapter_index=part_index,
                        total_chapters=len(part_groups),
                        chapter_label=", ".join(chapter["label"] for chapter in part_group[:2]) + ("..." if len(part_group) > 2 else ""),
                        estimated_size_bytes=part_estimated_size,
                        output_path=part_output_path,
                    )

                    success, export_result = self._export_concat_mp3(
                        concat_path,
                        part_output_path,
                        progress_tick=lambda current_path=part_output_path, current_size=part_estimated_size, current_index=part_index: self._emit_merge_progress(
                            progress_callback,
                            merge_started_at,
                            stage="packing",
                            chapter_index=current_index,
                            total_chapters=len(part_groups),
                            chapter_label=f"Part {current_index} of {len(part_groups)}",
                            estimated_size_bytes=current_size,
                            output_path=current_path,
                        ),
                        log_callback=log_callback,
                    )
                    if not success:
                        return False, f"Failed to export optimized part {part_index}: {export_result}"

                    part_output_path = export_result
                    self._emit_merge_progress(
                        progress_callback,
                        merge_started_at,
                        stage="normalizing",
                        chapter_index=part_index,
                        total_chapters=len(part_groups),
                        chapter_label=f"Part {part_index} of {len(part_groups)}",
                        estimated_size_bytes=part_estimated_size,
                        output_path=part_output_path,
                    )
                    normalized, normalize_result = self._call_normalize_audio_file(
                        part_output_path,
                        export_config=export_config,
                        progress_callback=lambda info, current_path=part_output_path, current_size=part_estimated_size, current_index=part_index: self._emit_merge_progress(
                            progress_callback,
                            merge_started_at,
                            stage="normalizing",
                            chapter_index=current_index,
                            total_chapters=len(part_groups),
                            chapter_label=(
                                f"Part {current_index} "
                                f"({info.get('phase', 'normalizing')}, {int(round(info.get('overall_percent', 0.0)))}%)"
                            ),
                            estimated_size_bytes=current_size,
                            output_path=current_path,
                        ),
                        log_callback=log_callback,
                    )
                    if not normalized:
                        return False, f"Failed to normalize optimized part {part_index}: {normalize_result}"
                    part_paths.append((part_output_path, os.path.basename(part_output_path)))

                self._emit_merge_progress(
                    progress_callback,
                    merge_started_at,
                    stage="bundling",
                    chapter_index=len(part_paths),
                    total_chapters=len(part_paths),
                    chapter_label="Writing optimized export zip",
                    estimated_size_bytes=sum(os.path.getsize(path) for path, _ in part_paths),
                    output_path=zip_path,
                )

                with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
                    for part_path, part_basename in part_paths:
                        zf.write(part_path, arcname=part_basename)

                self._emit_merge_progress(
                    progress_callback,
                    merge_started_at,
                    stage="complete",
                    chapter_index=len(part_paths),
                    total_chapters=len(part_paths),
                    chapter_label=f"Created {len(part_paths)} optimized part{'s' if len(part_paths) != 1 else ''}",
                    estimated_size_bytes=sum(os.path.getsize(path) for path, _ in part_paths),
                    output_path=zip_path,
                )

                return True, os.path.basename(zip_path)
            finally:
                shutil.rmtree(temp_dir, ignore_errors=True)

        def export_audacity(self, export_config=None):
            """Export project as an Audacity-compatible zip with per-speaker WAV tracks,
            a LOF file for auto-import, and a labels file for chunk annotations."""
            timeline_items = self._collect_merge_timeline(export_config=export_config)

            # Phase 1 — Compute timeline (matching merge_audio pause logic exactly)
            timeline = []  # list of (chunk, segment, abs_start_ms)
            prev_speaker = None
            cursor_ms = 0

            prev_item = None
            for item in timeline_items:
                chunk = item["chunk"]
                if item.get("is_silence_block"):
                    segment = AudioSegment.silent(duration=item["silence_duration_ms"])
                    # No auto-silence before/after explicit silence blocks
                else:
                    try:
                        segment = AudioSegment.from_file(item["full_path"])
                    except Exception as e:
                        print(f"Error loading audio for Audacity export {item['full_path']}: {e}")
                        prev_item = item
                        continue
                    if prev_item is not None and not prev_item.get("is_silence_block"):
                        speaker = chunk.get("speaker", "")
                        prev_speaker = prev_item["chunk"].get("speaker", "")
                        if speaker == prev_speaker:
                            cursor_ms += SAME_SPEAKER_PAUSE_MS
                        else:
                            cursor_ms += DEFAULT_PAUSE_MS

                timeline.append((chunk, segment, cursor_ms))
                cursor_ms += len(segment)
                prev_item = item

            if not timeline:
                return False, "No audio segments found"

            total_duration_ms = cursor_ms

            # Phase 2 — Build per-speaker WAV tracks
            speakers_ordered = []
            seen = set()
            for chunk, segment, start_ms in timeline:
                speaker = chunk.get("speaker")
                if speaker and speaker not in seen:
                    speakers_ordered.append(speaker)
                    seen.add(speaker)

            speaker_tracks = {}
            for speaker in speakers_ordered:
                track_cursor = 0
                track = AudioSegment.empty()

                for chunk, segment, start_ms in timeline:
                    if chunk.get("speaker") != speaker:
                        continue
                    # Insert silence gap from current track position to this chunk's start
                    gap = start_ms - track_cursor
                    if gap > 0:
                        track += AudioSegment.silent(duration=gap)
                    track += segment
                    track_cursor = start_ms + len(segment)

                # Pad to total duration so all tracks are equal length
                remaining = total_duration_ms - track_cursor
                if remaining > 0:
                    track += AudioSegment.silent(duration=remaining)

                speaker_tracks[speaker] = track

            # Phase 3 — Build LOF and labels content
            lof_lines = []
            for speaker in speakers_ordered:
                safe_name = sanitize_filename(speaker)
                lof_lines.append(f'file "{safe_name}.wav"')
            lof_content = "\n".join(lof_lines) + "\n"

            label_lines = []
            for chunk, segment, start_ms in timeline:
                start_sec = start_ms / 1000.0
                end_sec = (start_ms + len(segment)) / 1000.0
                text_preview = chunk.get("text", "")[:80]
                label = f"[{chunk['speaker']}] {text_preview}"
                label_lines.append(f"{start_sec:.6f}\t{end_sec:.6f}\t{label}")
            labels_content = "\n".join(label_lines) + "\n"

            # Phase 4 — Zip everything
            zip_path = os.path.join(self.exports_dir, "audacity_export.zip")
            with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
                zf.writestr("project.lof", lof_content)
                zf.writestr("labels.txt", labels_content)

                for speaker in speakers_ordered:
                    safe_name = sanitize_filename(speaker)
                    wav_buffer = io.BytesIO()
                    speaker_tracks[speaker].export(wav_buffer, format="wav")
                    zf.writestr(f"{safe_name}.wav", wav_buffer.getvalue())

            return True, zip_path

        def merge_m4b(self, per_chunk_chapters=False, metadata=None, export_config=None):
            """Merge audio chunks into an M4B audiobook with chapter markers.

            Args:
                per_chunk_chapters: If True, each chunk is a chapter. If False,
                    detect chapter headings and group chunks into sections.
                metadata: Optional dict with keys: title, author, narrator, year,
                    description, cover_path (absolute path to cover image).
                export_config: Optional ExportConfig with silence durations.

            Returns:
                tuple: (success: bool, message: str)
            """
            metadata = metadata or {}
            same_speaker_ms = getattr(export_config, "silence_same_speaker_ms", SAME_SPEAKER_PAUSE_MS)
            between_speakers_ms = getattr(export_config, "silence_between_speakers_ms", DEFAULT_PAUSE_MS)
            paragraph_ms = getattr(export_config, "silence_paragraph_ms", 750)
            timeline_items = self._collect_merge_timeline(export_config=export_config)

            # Phase 1 — Compute timeline (same logic as export_audacity)
            timeline = []  # list of (chunk, segment, abs_start_ms)
            prev_chunk = None
            cursor_ms = 0

            prev_item = None
            for item in timeline_items:
                chunk = item["chunk"]
                if item.get("is_silence_block"):
                    segment = AudioSegment.silent(duration=item["silence_duration_ms"])
                    # No auto-silence before/after explicit silence blocks
                else:
                    try:
                        segment = AudioSegment.from_file(item["full_path"])
                    except Exception as e:
                        print(f"Error loading audio for M4B export {item['full_path']}: {e}")
                        prev_item = item
                        continue
                    if prev_item is not None and not prev_item.get("is_silence_block"):
                        prev_chunk = prev_item["chunk"]
                        prev_pid = prev_chunk.get("paragraph_id")
                        curr_pid = chunk.get("paragraph_id")
                        if prev_pid and curr_pid and prev_pid != curr_pid:
                            cursor_ms += paragraph_ms
                        elif chunk.get("speaker") == prev_chunk.get("speaker"):
                            cursor_ms += same_speaker_ms
                        else:
                            cursor_ms += between_speakers_ms

                timeline.append((chunk, segment, cursor_ms))
                cursor_ms += len(segment)
                prev_item = item

            if not timeline:
                return False, "No audio segments found"

            # Phase 2 — Build chapters
            chapters = self._build_m4b_chapters(timeline, per_chunk_chapters)
            print(f"  M4B: {len(chapters)} chapters")

            # Phase 3 — Combine audio and export to temp WAV
            # Silence blocks are already AudioSegment objects; treat them as their own "speaker"
            audio_segments = [seg for _, seg, _ in timeline]
            speakers = [chunk.get("speaker", "") for chunk, _, _ in timeline]
            final_audio = combine_audio_with_pauses(
                audio_segments, speakers,
                pause_ms=between_speakers_ms,
                same_speaker_pause_ms=same_speaker_ms,
            )

            temp_dir = _make_runtime_temp_dir(self, "m4b_export_")
            temp_wav = os.path.join(temp_dir, "temp_m4b_combined.wav")
            meta_path = os.path.join(temp_dir, "temp_m4b_meta.txt")
            output_path = os.path.join(self.exports_dir, "audiobook.m4b")

            try:
                final_audio.export(temp_wav, format="wav")
                normalized, normalize_result = self._call_normalize_audio_file(temp_wav, export_config=export_config)
                if not normalized:
                    return False, f"Audio normalization failed: {normalize_result}"

                # Phase 4 — Write FFmpeg metadata file with book metadata
                meta_lines = [";FFMETADATA1"]
                meta_lines.append(f"title={self._escape_ffmeta(metadata.get('title') or 'Audiobook')}")
                meta_lines.append(f"artist={self._escape_ffmeta(metadata.get('author') or '')}")
                meta_lines.append(f"album_artist={self._escape_ffmeta(metadata.get('narrator') or '')}")
                meta_lines.append(f"date={self._escape_ffmeta(metadata.get('year') or '')}")
                meta_lines.append(f"comment={self._escape_ffmeta(metadata.get('description') or '')}")
                meta_lines.append("genre=Audiobook")
                meta_lines.append("")
                for title, start_ms, end_ms in chapters:
                    safe_title = self._escape_ffmeta(title)
                    meta_lines.append("[CHAPTER]")
                    meta_lines.append("TIMEBASE=1/1000")
                    meta_lines.append(f"START={start_ms}")
                    meta_lines.append(f"END={end_ms}")
                    meta_lines.append(f"title={safe_title}")
                    meta_lines.append("")

                with open(meta_path, "w", encoding="utf-8") as f:
                    f.write("\n".join(meta_lines))

                # Phase 5 — FFmpeg: WAV + chapters → M4B (AAC)
                cover_path = metadata.get("cover_path") or ""
                has_cover = cover_path and os.path.exists(cover_path)

                cmd = [get_ffmpeg_exe(), "-y", "-i", temp_wav]
                if has_cover:
                    cmd += ["-i", cover_path]
                cmd += ["-i", meta_path, "-map_metadata", "2" if has_cover else "1"]
                # Map audio stream
                cmd += ["-map", "0:a"]
                if has_cover:
                    # Map cover as attached picture
                    cmd += ["-map", "1:v", "-c:v", "copy", "-disposition:v:0", "attached_pic"]
                cmd += [
                    "-c:a", "aac",
                    "-b:a", "128k",
                    "-movflags", "+faststart",
                    output_path
                ]
                result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=600)
                if result.returncode != 0:
                    print(f"FFmpeg stderr: {result.stderr[-500:]}")
                    return False, f"FFmpeg failed (exit {result.returncode})"

            finally:
                shutil.rmtree(temp_dir, ignore_errors=True)

            return True, "audiobook.m4b"

        @staticmethod
        def _escape_ffmeta(text):
            """Escape special characters for FFmpeg metadata format."""
            text = text.replace("\\", "\\\\")
            text = text.replace("=", "\\=")
            text = text.replace(";", "\\;")
            text = text.replace("#", "\\#")
            text = text.replace("\n", " ")
            return text

        def _build_m4b_chapters(self, timeline, per_chunk_chapters):
            """Build chapter list from timeline entries.

            Returns:
                list of (title, start_ms, end_ms) tuples
            """
            if per_chunk_chapters:
                chapters = []
                for chunk, segment, start_ms in timeline:
                    end_ms = start_ms + len(segment)
                    text_preview = chunk.get("text", "")[:80]
                    title = f"[{chunk['speaker']}] {text_preview}"
                    chapters.append((title, start_ms, end_ms))
                return chapters

            # Smart grouping: detect chapter headings
            heading_indices = []
            for i, (chunk, segment, start_ms) in enumerate(timeline):
                text = chunk.get("text", "").strip()
                # Short structural text (likely a heading) or starts with heading keyword
                if self._HEADING_RE.match(text):
                    heading_indices.append(i)
                elif len(text) < 80 and '"' not in text and text and self._HEADING_RE.search(text):
                    heading_indices.append(i)

            # If no headings detected, fall back to per-chunk
            if not heading_indices:
                print("  M4B: No chapter headings detected, falling back to per-chunk chapters")
                return self._build_m4b_chapters(timeline, per_chunk_chapters=True)

            chapters = []

            # Pre-heading chunks → "Introduction"
            if heading_indices[0] > 0:
                start_ms = timeline[0][2]
                last_before = heading_indices[0] - 1
                end_ms = timeline[last_before][2] + len(timeline[last_before][1])
                chapters.append(("Introduction", start_ms, end_ms))

            # Each heading starts a chapter that runs until the next heading
            for idx, head_i in enumerate(heading_indices):
                title = timeline[head_i][0].get("text", "").strip()
                # Truncate long titles
                if len(title) > 120:
                    title = title[:117] + "..."

                start_ms = timeline[head_i][2]

                # End = start of next heading, or end of last chunk
                if idx + 1 < len(heading_indices):
                    next_head_i = heading_indices[idx + 1]
                    last_in_group = next_head_i - 1
                else:
                    last_in_group = len(timeline) - 1

                end_ms = timeline[last_in_group][2] + len(timeline[last_in_group][1])
                chapters.append((title, start_ms, end_ms))

            return chapters
