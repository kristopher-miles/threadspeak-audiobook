"""Voice resolution, runtime voice config prep, and invalidation logic.

This mixin covers:
- narrator/speaker resolution rules,
- design-voice suggestion/materialization helpers, and
- config-save paths that preview/apply dependent chunk invalidations.
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


class ProjectVoiceMixin:
        """Resolve speaker-to-voice mapping and manage voice-driven invalidation."""
        def _chunk_voice_line_counts(self, chunks=None):
            counts = Counter()
            canonical_names = {}
            if chunks is None and getattr(self, "script_store", None) is not None:
                summary = self.script_store.get_voice_summary()
                return Counter(summary.get("line_counts", {}))
            chunk_rows = chunks if chunks is not None else self.load_chunks()
            for chunk in chunk_rows or []:
                speaker = (chunk.get("speaker") or "").strip()
                normalized = self._normalize_speaker_name(speaker)
                if not normalized:
                    continue
                canonical_name = canonical_names.get(normalized)
                if not canonical_name:
                    canonical_name = speaker
                    canonical_names[normalized] = canonical_name
                counts[canonical_name] += 1
            return counts

        def _script_voice_line_counts(self, script_entries=None):
            counts = Counter()
            voices_map = {}
            entries = script_entries
            if entries is None:
                try:
                    entries = load_script_document(self.script_path).get("entries", [])
                except Exception:
                    entries = []

            for entry in entries or []:
                speaker = (entry.get("speaker") or entry.get("type") or "").strip()
                normalized = self._normalize_speaker_name(speaker)
                if not normalized:
                    continue
                canonical_name = voices_map.get(normalized)
                if not canonical_name:
                    canonical_name = speaker
                    voices_map[normalized] = canonical_name
                counts[canonical_name] += 1
            return counts

        @staticmethod
        def _paragraph_mentions_speaker(paragraph_text, speaker):
            normalized_paragraph = re.sub(r"\s+", " ", paragraph_text or "").casefold()
            normalized_speaker = re.sub(r"\s+", " ", speaker or "").strip().casefold()
            if not normalized_paragraph or not normalized_speaker:
                return False
            return normalized_speaker in normalized_paragraph

        def suggest_design_sample_text(self, speaker, chunks=None):
            chunks = chunks if chunks is not None else self.load_chunks()
            speaker_key = self._normalize_speaker_name(speaker)
            matching_texts = [
                (chunk.get("text") or "").strip()
                for chunk in chunks
                if self._normalize_speaker_name(chunk.get("speaker")) == speaker_key and (chunk.get("text") or "").strip()
            ]
            if not matching_texts:
                return ""

            sorted_texts = sorted(matching_texts, key=lambda text: len(re.findall(r"\b\w+\b", text)), reverse=True)
            base_sentences = self._split_text_sentences(sorted_texts[0]) or [sorted_texts[0]]
            selected = list(base_sentences)

            while len(re.findall(r"\b\w+\b", " ".join(selected))) > 50 and len(selected) > 1:
                candidate_without_last = selected[:-1]
                candidate_without_first = selected[1:]
                first_words = len(re.findall(r"\b\w+\b", " ".join(candidate_without_first)))
                last_words = len(re.findall(r"\b\w+\b", " ".join(candidate_without_last)))
                selected = candidate_without_first if first_words >= last_words else candidate_without_last

            used = {sentence.strip() for sentence in selected if sentence.strip()}
            total_words = len(re.findall(r"\b\w+\b", " ".join(selected)))
            if total_words >= 30:
                return " ".join(selected).strip()

            extra_sentences = []
            for text in sorted_texts:
                for sentence in self._split_text_sentences(text) or [text]:
                    sentence = sentence.strip()
                    if sentence and sentence not in used:
                        extra_sentences.append(sentence)
                        used.add(sentence)

            for sentence in extra_sentences:
                selected.append(sentence)
                total_words = len(re.findall(r"\b\w+\b", " ".join(selected)))
                if total_words >= 30:
                    break

            return " ".join(selected).strip()

        def collect_voice_suggestion_context(self, speaker, target_chars=None):
            generation_settings = self._load_generation_settings()
            chunk_size = int(generation_settings.get("chunk_size") or 3000)
            target_chars = max(int(target_chars or (chunk_size * 2)), 1)

            source_error = None
            selected = []
            total_chars = 0

            try:
                source_document = self.load_source_document()
                for paragraph in iter_document_paragraphs(source_document):
                    if not self._paragraph_mentions_speaker(paragraph["text"], speaker):
                        continue
                    selected.append(paragraph)
                    total_chars += len(paragraph["text"])
                    if total_chars >= target_chars:
                        break
                context_source = "source_document"
            except Exception as e:
                source_error = str(e)
                context_source = "chunks_fallback"
                chunks = self.load_chunks()
                speaker_key = self._normalize_speaker_name(speaker)
                fallback_items = []
                for chunk in chunks:
                    text = (chunk.get("text") or "").strip()
                    if not text:
                        continue
                    chunk_speaker = self._normalize_speaker_name(chunk.get("speaker"))
                    matches = chunk_speaker == speaker_key or self._paragraph_mentions_speaker(text, speaker)
                    if not matches:
                        continue
                    fallback_items.append({
                        "text": text,
                        "chapter": (chunk.get("chapter") or "").strip(),
                    })
                    total_chars += len(text)
                    if total_chars >= target_chars:
                        break
                selected = fallback_items
                if source_error:
                    print(
                        f"Voice suggestion context fallback for '{speaker}': source unavailable ({source_error}); using generated chunks."
                    )

            return {
                "speaker": speaker,
                "target_chars": target_chars,
                "context_chars": total_chars,
                "paragraphs": selected,
                "context_source": context_source,
                "source_error": source_error,
                "warning": (
                    f"Source document unavailable: {source_error}. Using generated chunks as fallback context."
                    if source_error
                    else None
                ),
            }

        def build_voice_suggestion_prompt(self, speaker, prompt_template):
            context = self.collect_voice_suggestion_context(speaker)
            paragraphs = context["paragraphs"]
            source_error = context.get("source_error")
            context_source = context.get("context_source")
            warning = context.get("warning")

            if paragraphs:
                context_blocks = []
                for item in paragraphs:
                    chapter = (item.get("chapter") or "").strip()
                    text = item["text"]
                    context_blocks.append(f"[{chapter}] {text}" if chapter else text)
                if context_source == "chunks_fallback":
                    context_prefix = (
                        f'Source document unavailable ({source_error}). '
                        f'Fallback context from generated chunks mentioning "{speaker}":\n\n'
                        + "\n\n".join(context_blocks)
                    )
                else:
                    context_prefix = (
                        f'Source paragraphs mentioning "{speaker}":\n\n' +
                        "\n\n".join(context_blocks)
                    )
            else:
                if context_source == "chunks_fallback":
                    context_prefix = (
                        f'Source document unavailable ({source_error}). '
                        f'No generated chunks mentioning "{speaker}" were found for fallback context.'
                    )
                else:
                    context_prefix = f'No source paragraphs mentioning "{speaker}" were found in the uploaded story.'

            rendered_prompt = (prompt_template or "").replace("{character_name}", speaker)
            return {
                **context,
                "warning": warning,
                "prompt": f"{context_prefix}\n\n{rendered_prompt}".strip(),
            }

        def _upsert_clone_manifest_entry(self, entry):
            manifest_path = os.path.join(self.root_dir, "clone_voices", "manifest.json")
            manifest = []
            if os.path.exists(manifest_path):
                try:
                    with open(manifest_path, "r", encoding="utf-8") as f:
                        manifest = json.load(f)
                except (json.JSONDecodeError, ValueError):
                    manifest = []

            replaced = False
            for index, existing in enumerate(manifest):
                if existing.get("id") == entry["id"]:
                    manifest[index] = entry
                    replaced = True
                    break
            if not replaced:
                manifest.append(entry)

            with open(manifest_path, "w", encoding="utf-8") as f:
                json.dump(manifest, f, indent=2, ensure_ascii=False)

        def materialize_design_voice(self, speaker, description=None, sample_text=None, force=False, voice_config=None, export_config=None):
            voice_config = copy.deepcopy(voice_config) if voice_config is not None else self._load_voice_config()
            voice_data = voice_config.setdefault(speaker, {})
            voice_data["type"] = "design"

            description = (description if description is not None else voice_data.get("description") or "").strip()
            if not description:
                raise ValueError(f"Voice design for '{speaker}' is missing a base description")

            chunks = self.load_chunks()
            sample_text = (sample_text if sample_text is not None else voice_data.get("ref_text") or "").strip()
            if not sample_text:
                sample_text = self.suggest_design_sample_text(speaker, chunks)
            if not sample_text:
                raise ValueError(f"No sample text available for '{speaker}'")
            generated_ref_text, _ = apply_dictionary_to_text(sample_text, self.load_dictionary_entries())

            script_title = self._current_script_title()
            display_name = f"{script_title}.{speaker}"
            safe_name = sanitize_filename(f"{script_title}.{speaker}")
            voice_id = safe_name or sanitize_filename(speaker) or "designed_clone"
            filename = f"{voice_id}.wav"
            rel_audio_path = f"clone_voices/{filename}"
            abs_audio_path = os.path.join(self.root_dir, rel_audio_path)
            os.makedirs(os.path.dirname(abs_audio_path), exist_ok=True)

            audio_exists = os.path.exists(abs_audio_path)
            if force or not audio_exists:
                engine = self.get_engine()
                if not engine:
                    raise RuntimeError("TTS engine not initialized")
                wav_path, _ = engine.generate_voice_design(description=description, sample_text=generated_ref_text)
                shutil.copy2(wav_path, abs_audio_path)
                normalized, normalize_result = self._call_normalize_audio_file(
                    abs_audio_path,
                    export_config=export_config,
                    allow_short_single_pass=True,
                )
                if not normalized:
                    raise RuntimeError(f"Failed to normalize generated design voice for '{speaker}': {normalize_result}")
                if hasattr(engine, "clear_clone_prompt_cache"):
                    engine.clear_clone_prompt_cache(speaker)

            voice_data["description"] = description
            voice_data["ref_text"] = sample_text
            voice_data["generated_ref_text"] = generated_ref_text
            voice_data["ref_audio"] = rel_audio_path
            self._save_voice_config(voice_config)

            self._upsert_clone_manifest_entry({
                "id": voice_id,
                "name": display_name,
                "filename": filename,
                "generated": True,
                "speaker": speaker,
                "script_title": script_title,
                "sample_text": sample_text,
                "description": description,
            })

            runtime_data = copy.deepcopy(voice_data)
            runtime_data["type"] = "clone"
            return {
                "voice_config": voice_config,
                "runtime_voice_data": runtime_data,
                "voice_id": voice_id,
                "display_name": display_name,
                "filename": filename,
                "ref_audio": rel_audio_path,
                "ref_text": sample_text,
                "generated_ref_text": generated_ref_text,
            }

        def prepare_runtime_voice_config(self, voice_config, speakers, force_design_refresh=False):
            runtime_config = copy.deepcopy(voice_config)
            normalized_lookup = {}
            for name in runtime_config.keys():
                normalized = self._normalize_speaker_name(name)
                if normalized and normalized not in normalized_lookup:
                    normalized_lookup[normalized] = name

            def ensure_runtime_voice(speaker, visiting=None):
                visiting = visiting or set()
                normalized = self._normalize_speaker_name(speaker)
                actual_speaker = normalized_lookup.get(normalized, speaker)
                if actual_speaker in visiting:
                    raise ValueError(f"Voice fallback loop detected for '{actual_speaker}'")
                visiting = set(visiting)
                visiting.add(actual_speaker)

                voice_data = runtime_config.get(actual_speaker, {})
                if voice_data.get("type") != "design":
                    return actual_speaker

                try:
                    materialized = self.materialize_design_voice(
                        actual_speaker,
                        description=voice_data.get("description"),
                        sample_text=voice_data.get("ref_text"),
                        force=force_design_refresh or not voice_data.get("ref_audio"),
                        voice_config=runtime_config,
                    )
                    runtime_config.update(materialized["voice_config"])
                    runtime_config[actual_speaker] = materialized["runtime_voice_data"]
                    return actual_speaker
                except Exception:
                    if normalized == self._normalize_speaker_name("NARRATOR"):
                        raise

                    narrator_name = normalized_lookup.get(self._normalize_speaker_name("NARRATOR"))
                    if not narrator_name:
                        raise ValueError(f"Voice design for '{actual_speaker}' is unavailable and narrator has no voice configured")

                    ensure_runtime_voice(narrator_name, visiting)
                    narrator_runtime = runtime_config.get(narrator_name, {})
                    if narrator_runtime.get("type") == "design":
                        raise ValueError("Narrator voice is unavailable; cannot fall back from design voice")
                    runtime_config[actual_speaker] = copy.deepcopy(narrator_runtime)
                    return actual_speaker

            for speaker in speakers:
                ensure_runtime_voice(speaker)

            return runtime_config

        @staticmethod
        def _normalize_speaker_name(name):
            return re.sub(r"\s+", " ", (name or "").strip()).casefold()

        def has_voice_config(self):
            script_store = getattr(self, "script_store", None)
            if script_store is not None:
                return script_store.has_voice_profiles()
            voice_config = self._load_voice_config()
            return bool(voice_config)

        def sync_missing_voice_profiles_from_chunks(self, *, reason="sync_missing_voice_profiles_from_chunks"):
            script_store = getattr(self, "script_store", None)
            if script_store is None:
                return []
            existing_config = self._load_voice_config()
            existing_keys = {self._normalize_speaker_name(name) for name in existing_config.keys()}
            missing_rows = []
            for speaker in (script_store.get_voice_summary().get("voices") or []):
                normalized = self._normalize_speaker_name(speaker)
                if not normalized or normalized in existing_keys:
                    continue
                missing_rows.append({
                    "speaker": speaker,
                    "config": self.script_store._default_voice_config(),
                })
            if not missing_rows:
                return []
            return script_store.upsert_voice_profiles(missing_rows, reason=reason, wait=True)

        def reset_voice_state(self, *, reason="reset_voice_state"):
            snapshot = {
                "profiles": {},
                "narrator_threshold": self.DEFAULT_NARRATOR_THRESHOLD,
                "narrator_overrides": {},
                "auto_narrator_aliases": {},
            }
            result = self.replace_voice_state_snapshot(snapshot, reason=reason)
            self.log_voice_audit_event("voice_state_reset", reason=reason)
            return result

        def export_voice_config_compat(self, target_path=None):
            return target_path or self.voice_config_path

        def export_voice_state_compat(self, target_path=None):
            return target_path or os.path.join(self.root_dir, "state.json")

        def import_voice_compat(self, voice_config_path=None, state_path=None, replace=False):
            return None

        def get_narrator_threshold(self):
            script_store = getattr(self, "script_store", None)
            raw = self.DEFAULT_NARRATOR_THRESHOLD
            if script_store is not None:
                raw = script_store.get_voice_settings().get("narrator_threshold", self.DEFAULT_NARRATOR_THRESHOLD)
            try:
                value = int(raw)
            except (TypeError, ValueError):
                value = self.DEFAULT_NARRATOR_THRESHOLD
            return max(0, value)

        def set_narrator_threshold(self, value):
            try:
                parsed = int(value)
            except (TypeError, ValueError):
                parsed = self.DEFAULT_NARRATOR_THRESHOLD
            parsed = max(0, parsed)
            script_store = getattr(self, "script_store", None)
            if script_store is not None:
                script_store.set_voice_setting("narrator_threshold", parsed, reason="set_narrator_threshold", wait=True)
            else:
                raise RuntimeError("Narrator threshold requires the SQLite script store")
            self.refresh_auto_narrator_aliases()
            return parsed

        def get_narrator_overrides(self):
            """Return dict of chapter_name -> voice_name for narrator substitution."""
            script_store = getattr(self, "script_store", None)
            if script_store is not None:
                return script_store.get_narrator_overrides()
            return {}

        def _apply_narrator_override(self, speaker, chapter, narrator_overrides):
            """Return the effective speaker, substituting the chapter narrator override if applicable."""
            if self._normalize_speaker_name(speaker) == self._normalize_speaker_name("NARRATOR"):
                override = narrator_overrides.get(chapter)
                if override and self._normalize_speaker_name(override) != self._normalize_speaker_name("NARRATOR"):
                    return override
            return speaker

        def set_narrator_override(self, chapter: str, voice: str):
            """Set or clear the narrator voice for a specific chapter."""
            script_store = getattr(self, "script_store", None)
            if script_store is not None:
                script_store.set_narrator_override(chapter, voice, reason="set_narrator_override", wait=True)
                return
            raise RuntimeError("Narrator overrides require the SQLite script store")

        @staticmethod
        def _count_name_mentions_in_text(text, name):
            chapter_text = str(text or "")
            candidate = str(name or "").strip()
            if not chapter_text or not candidate:
                return 0
            try:
                return len(re.findall(re.escape(candidate), chapter_text, flags=re.IGNORECASE))
            except re.error:
                return 0

        def rank_chapter_narration_candidates(self, chapter, voice_names, *, include_narrator=True):
            normalized_chapter = str(chapter or "").strip()
            if not normalized_chapter:
                return []

            narrator_name = None
            normalized_names = []
            seen = set()
            for raw_name in (voice_names or []):
                name = str(raw_name or "").strip()
                if not name:
                    continue
                normalized = self._normalize_speaker_name(name)
                if not normalized or normalized in seen:
                    continue
                seen.add(normalized)
                if normalized == self._normalize_speaker_name("NARRATOR"):
                    narrator_name = name
                    if not include_narrator:
                        continue
                if not include_narrator and normalized == self._normalize_speaker_name("NARRATOR"):
                    continue
                normalized_names.append(name)

            if not normalized_names:
                return []

            chapter_text = " ".join(
                str((chunk or {}).get("text") or "")
                for chunk in self.load_chunks(chapter=normalized_chapter)
                if str((chunk or {}).get("text") or "").strip()
            )

            ranked = sorted(
                [
                    name for name in normalized_names
                    if self._normalize_speaker_name(name) != self._normalize_speaker_name("NARRATOR")
                ],
                key=lambda voice_name: (
                    -self._count_name_mentions_in_text(chapter_text, voice_name),
                    voice_name.casefold(),
                ),
            )
            if include_narrator and narrator_name:
                return [narrator_name, *ranked]
            return ranked

        def disable_narrator_narration_and_reassign_chapters(self, draft_config=None):
            narrator_key = self._normalize_speaker_name("NARRATOR")
            print("[VOICE] Attempting narrator disable and chapter reassignment")

            persisted_config = copy.deepcopy(self._load_voice_config() or {})
            effective_config = copy.deepcopy(persisted_config)

            for speaker_name, draft_profile in dict(draft_config or {}).items():
                normalized_name = self._normalize_speaker_name(speaker_name)
                if not normalized_name:
                    continue
                canonical_name = speaker_name
                for existing_name in effective_config.keys():
                    if self._normalize_speaker_name(existing_name) == normalized_name:
                        canonical_name = existing_name
                        break
                profile = dict(effective_config.get(canonical_name) or {})
                if (draft_profile or {}).get("narrates") is not None:
                    profile["narrates"] = bool((draft_profile or {}).get("narrates"))
                effective_config[canonical_name] = profile

            narrator_name = None
            for existing_name in effective_config.keys():
                if self._normalize_speaker_name(existing_name) == narrator_key:
                    narrator_name = existing_name
                    break
            narrator_name = narrator_name or "NARRATOR"
            narrator_profile = dict(effective_config.get(narrator_name) or {})
            narrator_profile["narrates"] = False
            effective_config[narrator_name] = narrator_profile

            eligible_narrators = sorted(
                [
                    speaker_name
                    for speaker_name, profile in effective_config.items()
                    if self._normalize_speaker_name(speaker_name) != narrator_key and bool((profile or {}).get("narrates"))
                ],
                key=lambda value: value.casefold(),
            )
            if not eligible_narrators:
                print("[VOICE] Narrator disable rejected: no alternate narrators are enabled")
                return {
                    "status": "rejected",
                    "code": "narrator_disable_requires_other_narrator",
                    "message": "Enable narration on another character before disabling the narrator.",
                }

            chapter_order = []
            seen_chapters = set()
            chunks = self.load_chunks()
            for chunk in chunks:
                chapter = str((chunk or {}).get("chapter") or "").strip()
                if not chapter:
                    continue
                if chapter not in seen_chapters:
                    seen_chapters.add(chapter)
                    chapter_order.append(chapter)

            overrides = dict(self.get_narrator_overrides() or {})
            chapter_assignments = {}
            for chapter in chapter_order:
                current_narrator = str(overrides.get(chapter) or "NARRATOR").strip() or "NARRATOR"
                if self._normalize_speaker_name(current_narrator) != narrator_key:
                    continue
                ranked = self.rank_chapter_narration_candidates(
                    chapter,
                    eligible_narrators,
                    include_narrator=False,
                )
                if ranked:
                    chapter_assignments[chapter] = ranked[0]

            self._save_voice_config(effective_config)
            for chapter, voice_name in chapter_assignments.items():
                self.set_narrator_override(chapter, voice_name)

            files_to_delete = set()
            invalidated_clips = 0
            for chunk in chunks:
                chapter = str((chunk or {}).get("chapter") or "").strip()
                if chapter not in chapter_assignments:
                    continue
                if self._normalize_speaker_name((chunk or {}).get("speaker") or "") != narrator_key:
                    continue
                audio_path = str((chunk or {}).get("audio_path") or "").strip()
                if not audio_path:
                    continue
                if self.prepare_chunk_for_regeneration_by_uid((chunk or {}).get("uid")) is None:
                    continue
                invalidated_clips += 1
                files_to_delete.add(audio_path)

            deleted_files = 0
            root_abs = os.path.abspath(self.root_dir)
            for relative_path in files_to_delete:
                full_path = os.path.abspath(os.path.join(self.root_dir, relative_path))
                if not (full_path == root_abs or full_path.startswith(root_abs + os.sep)):
                    continue
                if not os.path.exists(full_path):
                    continue
                try:
                    os.remove(full_path)
                    deleted_files += 1
                except OSError:
                    pass

            print(
                f"[VOICE] Narrator disabled; reassigned {len(chapter_assignments)} chapter narrator(s), "
                f"invalidated {invalidated_clips} clip(s), deleted {deleted_files} file(s)"
            )
            return {
                "status": "saved",
                "changed_chapters": len(chapter_assignments),
                "invalidated_clips": invalidated_clips,
                "deleted_files": deleted_files,
                "chapter_assignments": chapter_assignments,
            }

        def get_chapter_narrator_voice_repair(self, chapter):
            normalized_chapter = str(chapter or "").strip()
            if not normalized_chapter:
                return None

            narrator_voice = str((self.get_narrator_overrides() or {}).get(normalized_chapter) or "").strip()
            if not narrator_voice:
                return None
            if self._normalize_speaker_name(narrator_voice) == self._normalize_speaker_name("NARRATOR"):
                return None

            voice_config = self._load_voice_config()
            canonical_voice = narrator_voice
            for existing_name in voice_config.keys():
                if self._normalize_speaker_name(existing_name) == self._normalize_speaker_name(narrator_voice):
                    canonical_voice = existing_name
                    break

            current_profile = dict(voice_config.get(canonical_voice) or {})
            if bool(current_profile.get("narrates")):
                return None

            return {
                "chapter": normalized_chapter,
                "requested_voice": narrator_voice,
                "canonical_voice": canonical_voice,
                "previous_narrates": bool(current_profile.get("narrates")),
            }

        def ensure_chapter_narrator_voice_can_narrate(self, chapter, *, reason="chapter_narrator_repair", repair=None):
            """Repair narrator eligibility for the chapter override when needed."""
            repair = repair if isinstance(repair, dict) else self.get_chapter_narrator_voice_repair(chapter)
            if not repair:
                return None

            script_store = getattr(self, "script_store", None)
            if script_store is not None:
                script_store.patch_voice_profile(
                    repair["canonical_voice"],
                    fields={"narrates": True},
                    reason=reason,
                    wait=True,
                )
            else:
                voice_config = self._load_voice_config()
                current_profile = dict(voice_config.get(repair["canonical_voice"]) or {})
                current_profile["narrates"] = True
                voice_config[repair["canonical_voice"]] = current_profile
                self._save_voice_config(voice_config)
            self.log_voice_audit_event(
                "chapter_narrator_narrates_repair",
                reason=reason,
                chapter=repair["chapter"],
                requested_voice=repair["requested_voice"],
                canonical_voice=repair["canonical_voice"],
                previous_narrates=repair["previous_narrates"],
                new_narrates=True,
            )
            return repair["canonical_voice"]

        def get_auto_narrator_aliases(self):
            script_store = getattr(self, "script_store", None)
            if script_store is not None:
                aliases = script_store.get_auto_narrator_aliases()
                return aliases if isinstance(aliases, dict) else {}
            return {}

        def compute_auto_narrator_aliases(
            self,
            voice_config=None,
            script_entries=None,
            line_counts=None,
            narrator_name=None,
        ):
            voice_config = voice_config if isinstance(voice_config, dict) else self._load_voice_config()
            if line_counts is None:
                chunk_counts = self._chunk_voice_line_counts()
                line_counts = chunk_counts or self._script_voice_line_counts(script_entries)

            threshold = self.get_narrator_threshold()
            if threshold <= 0:
                return {}

            if narrator_name is None:
                narrator_name = self._resolve_narrator_name(
                    voice_config,
                    [{"speaker": speaker} for speaker in line_counts.keys()],
                )
            if not narrator_name:
                return {}

            config_lookup = {
                self._normalize_speaker_name(name): value
                for name, value in (voice_config or {}).items()
                if self._normalize_speaker_name(name)
            }
            aliases = {}
            narrator_key = self._normalize_speaker_name("NARRATOR")
            for speaker, count in (line_counts or {}).items():
                if self._normalize_speaker_name(speaker) == narrator_key:
                    continue
                config = config_lookup.get(self._normalize_speaker_name(speaker), {})
                if str((config or {}).get("alias") or "").strip():
                    continue
                try:
                    parsed_count = int(count or 0)
                except (TypeError, ValueError):
                    parsed_count = 0
                if parsed_count < threshold:
                    aliases[speaker] = narrator_name
            return aliases

        def refresh_auto_narrator_aliases(
            self,
            voice_config=None,
            script_entries=None,
            line_counts=None,
            narrator_name=None,
        ):
            aliases = self.compute_auto_narrator_aliases(
                voice_config=voice_config,
                script_entries=script_entries,
                line_counts=line_counts,
                narrator_name=narrator_name,
            )
            script_store = getattr(self, "script_store", None)
            if script_store is not None:
                script_store.replace_auto_narrator_aliases(
                    [
                        {"speaker": speaker, "target": target}
                        for speaker, target in aliases.items()
                    ],
                    reason="refresh_auto_narrator_aliases",
                    wait=True,
                )
            else:
                raise RuntimeError("Auto narrator aliases require the SQLite script store")
            return aliases

        @staticmethod
        def _count_speaker_lines(chunks, speaker):
            speaker_key = ProjectVoiceMixin._normalize_speaker_name(speaker)
            if not speaker_key:
                return 0
            count = 0
            for chunk in chunks or []:
                text = (chunk.get("text") or "").strip()
                if not text:
                    continue
                if ProjectVoiceMixin._normalize_speaker_name(chunk.get("speaker")) == speaker_key:
                    count += 1
            return count

        def _resolve_narrator_name(self, voice_config, chunks=None):
            narrator_key = self._normalize_speaker_name("NARRATOR")
            for name in voice_config.keys():
                if self._normalize_speaker_name(name) == narrator_key:
                    return name
            for chunk in chunks or []:
                speaker = (chunk.get("speaker") or "").strip()
                if self._normalize_speaker_name(speaker) == narrator_key:
                    return speaker
            return None

        def resolve_voice_speaker(
            self,
            speaker,
            voice_config,
            chunks=None,
            speaker_line_counts=None,
            narrator_name=None,
            auto_narrator_aliases=None,
        ):
            """Resolve a speaker alias to a configured target speaker.

            Aliases are case-insensitive and can chain, but any invalid target,
            self-alias, or loop falls back to the original speaker.
            """
            original = speaker or ""
            if not original:
                return original

            lookup = {}
            for name in voice_config.keys():
                normalized = self._normalize_speaker_name(name)
                if normalized and normalized not in lookup:
                    lookup[normalized] = name

            # Canonicalize the starting speaker to the configured key when a
            # case-variant exists (e.g. "Narrator" -> "NARRATOR").
            current = lookup.get(self._normalize_speaker_name(original), original)
            manual_alias_used = False
            seen = {self._normalize_speaker_name(current)}
            while current:
                voice_data = voice_config.get(current, {})
                alias = (voice_data.get("alias") or "").strip()
                if not alias:
                    break

                target = lookup.get(self._normalize_speaker_name(alias))
                if not target or self._normalize_speaker_name(target) == self._normalize_speaker_name(current):
                    break
                manual_alias_used = True

                target_key = self._normalize_speaker_name(target)
                if target_key in seen:
                    print(f"Alias loop detected for speaker '{original}'. Falling back to original speaker.")
                    return original

                seen.add(target_key)
                current = target

            resolved = current or original
            if manual_alias_used:
                return resolved

            alias_lookup = {}
            for raw_speaker, raw_target in (
                auto_narrator_aliases if isinstance(auto_narrator_aliases, dict) else self.get_auto_narrator_aliases()
            ).items():
                normalized_speaker = self._normalize_speaker_name(raw_speaker)
                normalized_target = self._normalize_speaker_name(raw_target)
                if normalized_speaker and normalized_target:
                    alias_lookup[normalized_speaker] = raw_target

            resolved_key = self._normalize_speaker_name(resolved)
            stored_alias_target = alias_lookup.get(resolved_key)
            if stored_alias_target and self._normalize_speaker_name(stored_alias_target) != resolved_key:
                return stored_alias_target

            if speaker_line_counts is None and not chunks:
                return resolved

            threshold = self.get_narrator_threshold()
            if threshold <= 0:
                return resolved

            narrator_key = self._normalize_speaker_name("NARRATOR")
            if resolved_key == narrator_key:
                return resolved

            if narrator_name is None:
                try:
                    chunk_list = chunks if chunks is not None else self.load_chunks()
                except Exception:
                    chunk_list = []
                narrator_name = self._resolve_narrator_name(voice_config, chunk_list)
            else:
                chunk_list = chunks if chunks is not None else []
            if not narrator_name:
                return resolved
            if self._normalize_speaker_name(narrator_name) == resolved_key:
                return resolved

            if speaker_line_counts is not None:
                line_count = int(speaker_line_counts.get(resolved_key, 0) or 0)
            else:
                line_count = self._count_speaker_lines(chunk_list, resolved)
            if line_count < threshold:
                return narrator_name

            return resolved

        @staticmethod
        def _voice_entry_from_config(voice_config, speaker):
            if not isinstance(voice_config, dict):
                return {}
            entry = voice_config.get(speaker, {})
            return entry if isinstance(entry, dict) else {}

        @staticmethod
        def _voice_ref_audio_from_config(voice_config, speaker):
            entry = ProjectVoiceMixin._voice_entry_from_config(voice_config, speaker)
            return str(entry.get("ref_audio") or "").strip()

        @staticmethod
        def _voice_generation_issue_from_config(speaker, voice_data):
            profile = dict(voice_data or {})
            if not profile:
                return {
                    "code": "voice_config_required",
                    "message": f'"{speaker}" has no voice selected.',
                }

            voice_type = str(profile.get("type") or "").strip().lower() or "custom"
            if voice_type == "clone":
                ref_audio = str(profile.get("ref_audio") or "").strip()
                ref_text = str(profile.get("generated_ref_text") or profile.get("ref_text") or "").strip()
                if not ref_audio:
                    return {
                        "code": "voice_config_required",
                        "message": f'"{speaker}" is missing a reusable voice sample.',
                    }
                if not ref_text:
                    return {
                        "code": "voice_config_required",
                        "message": f'"{speaker}" is missing the saved transcript for its reusable voice sample.',
                    }
                return None

            if voice_type in {"lora", "builtin_lora"}:
                adapter_path = str(profile.get("adapter_path") or "").strip()
                if not adapter_path:
                    return {
                        "code": "voice_config_required",
                        "message": f'"{speaker}" has no LoRA voice selected.',
                    }
                return None

            if voice_type == "design":
                ref_audio = str(profile.get("ref_audio") or "").strip()
                description = str(profile.get("description") or "").strip()
                if not ref_audio and not description:
                    return {
                        "code": "voice_config_required",
                        "message": f'"{speaker}" has no voice selected.',
                    }
                if not ref_audio:
                    return {
                        "code": "voice_config_required",
                        "message": f'"{speaker}" has not finished creating its generated voice yet.',
                    }
                return None

            voice_name = str(profile.get("voice") or "").strip()
            if not voice_name:
                return {
                    "code": "voice_config_required",
                    "message": f'"{speaker}" has no voice selected.',
                }
            return None

        def validate_generation_voice_targets(self, chunk_refs):
            refs = list(chunk_refs or [])
            if not refs:
                return None

            chunk_rows = []
            for chunk_ref in refs:
                chunk = chunk_ref if isinstance(chunk_ref, dict) else self.get_chunk_raw(chunk_ref)
                if chunk is None:
                    continue
                if not str((chunk or {}).get("text") or "").strip():
                    continue
                chunk_rows.append(chunk)
            if not chunk_rows:
                return None

            voice_config = self._load_voice_config()
            narrator_overrides = self.get_narrator_overrides()
            narrator_name = self._resolve_narrator_name(voice_config, chunk_rows)
            runtime_cache = {}
            runtime_errors = {}

            for chunk in chunk_rows:
                source_speaker = str((chunk or {}).get("speaker") or "").strip()
                resolved_speaker = self._resolve_generation_speaker(
                    chunk,
                    voice_config,
                    narrator_overrides=narrator_overrides,
                    narrator_name=narrator_name,
                )
                cache_key = self._normalize_speaker_name(resolved_speaker)
                if cache_key not in runtime_cache and cache_key not in runtime_errors:
                    try:
                        prepared = self.prepare_runtime_voice_config(voice_config, [resolved_speaker])
                        runtime_cache[cache_key] = dict(prepared.get(resolved_speaker) or {})
                    except Exception as exc:
                        runtime_errors[cache_key] = str(exc).strip() or f'Unable to prepare a voice for "{resolved_speaker}".'

                if cache_key in runtime_errors:
                    issue_message = runtime_errors[cache_key]
                    issue_code = "voice_config_required"
                else:
                    issue = self._voice_generation_issue_from_config(
                        resolved_speaker,
                        runtime_cache.get(cache_key) or {},
                    )
                    if issue is None:
                        continue
                    issue_message = str(issue.get("message") or "").strip() or f'"{resolved_speaker}" has no voice selected.'
                    issue_code = str(issue.get("code") or "voice_config_required").strip() or "voice_config_required"

                if source_speaker and self._normalize_speaker_name(source_speaker) != self._normalize_speaker_name(resolved_speaker):
                    message = (
                        f'Cannot render because "{source_speaker}" resolves to "{resolved_speaker}", '
                        f'and {issue_message}'
                    )
                else:
                    message = f"Cannot render because {issue_message}"
                return {
                    "code": issue_code,
                    "message": message,
                    "speaker": source_speaker or resolved_speaker,
                    "voice_speaker": resolved_speaker or source_speaker,
                    "resolved_speaker": resolved_speaker,
                    "chunk_uid": str((chunk or {}).get("uid") or "").strip(),
                    "chunk_id": int((chunk or {}).get("id") or 0),
                    "chapter": str((chunk or {}).get("chapter") or "").strip(),
                }

            return None

        def preview_voice_config_invalidation(self, old_config, new_config):
            """Preview how many generated clips must be invalidated for a voice-config change.

            A clip is considered affected when:
            1) its resolved alias target changes, or
            2) the resolved target keeps the same name but its ref_audio changes.
            """
            old_config = old_config if isinstance(old_config, dict) else {}
            new_config = new_config if isinstance(new_config, dict) else {}
            chunks = self.load_chunks()

            affected_indices = []
            affected_speakers = set()
            changed_reasons = Counter()
            generated_indices_by_speaker = {}

            for index, chunk in enumerate(chunks):
                audio_path = str((chunk or {}).get("audio_path") or "").strip()
                if not audio_path:
                    continue

                speaker = str((chunk or {}).get("speaker") or "").strip()
                if not speaker:
                    continue

                normalized = self._normalize_speaker_name(speaker)
                if not normalized:
                    continue
                generated_indices_by_speaker.setdefault(normalized, {"speaker": speaker, "indices": []})
                generated_indices_by_speaker[normalized]["indices"].append(index)

            for speaker_group in generated_indices_by_speaker.values():
                speaker = speaker_group["speaker"]
                old_resolved = self.resolve_voice_speaker(speaker, old_config, chunks=chunks)
                new_resolved = self.resolve_voice_speaker(speaker, new_config, chunks=chunks)

                alias_changed = self._normalize_speaker_name(old_resolved) != self._normalize_speaker_name(new_resolved)
                old_ref_audio = self._voice_ref_audio_from_config(old_config, old_resolved)
                new_ref_audio = self._voice_ref_audio_from_config(new_config, new_resolved)
                ref_audio_changed = old_ref_audio != new_ref_audio and (old_ref_audio or new_ref_audio)

                if not (alias_changed or ref_audio_changed):
                    continue

                affected_indices.extend(speaker_group["indices"])
                affected_speakers.add(speaker)
                if alias_changed:
                    changed_reasons["alias"] += len(speaker_group["indices"])
                if ref_audio_changed:
                    changed_reasons["ref_audio"] += len(speaker_group["indices"])

            return {
                "invalidated_clips": len(affected_indices),
                "affected_indices": sorted(affected_indices),
                "affected_speakers": sorted(affected_speakers),
                "reason_counts": dict(changed_reasons),
            }

        def invalidate_chunk_audio_indices(self, indices):
            """Delete audio files and clear chunk audio refs for selected chunk indices."""
            target_indices = {int(i) for i in (indices or []) if isinstance(i, int) or str(i).isdigit()}
            if not target_indices:
                return {"invalidated_clips": 0, "deleted_files": 0}

            deleted_files = 0
            cleared = 0
            files_to_delete = set()
            cleared_uids = set()

            with self._chunks_lock:
                chunks = self.load_chunks_raw()
                if not chunks:
                    return {"invalidated_clips": 0, "deleted_files": 0}

                for index in sorted(target_indices):
                    if not (0 <= index < len(chunks)):
                        continue
                    chunk = chunks[index]
                    audio_path = str(chunk.get("audio_path") or "").strip()
                    if not audio_path:
                        continue
                    files_to_delete.add(audio_path)
                    chunk["audio_path"] = None
                    chunk["audio_validation"] = None
                    chunk["status"] = "pending"
                    chunk["auto_regen_count"] = 0
                    chunk.pop("generation_token", None)
                    self._clear_proofread_state(chunk)
                    if chunk.get("uid"):
                        cleared_uids.add(chunk["uid"])
                    cleared += 1

                if cleared > 0:
                    self._atomic_json_write(chunks, self.chunks_path)
                    for uid in cleared_uids:
                        self.clear_chunk_runtime(uid)

            root_abs = os.path.abspath(self.root_dir)
            for relative_path in files_to_delete:
                full_path = os.path.abspath(os.path.join(self.root_dir, relative_path))
                if not (full_path == root_abs or full_path.startswith(root_abs + os.sep)):
                    continue
                if not os.path.exists(full_path):
                    continue
                try:
                    os.remove(full_path)
                    deleted_files += 1
                except OSError:
                    pass

            return {"invalidated_clips": cleared, "deleted_files": deleted_files}

        def save_voice_config_with_invalidation(self, new_config, confirm_invalidation=False):
            new_config = copy.deepcopy(new_config) if isinstance(new_config, dict) else {}
            old_config = self._load_voice_config()
            preview = self.preview_voice_config_invalidation(old_config, new_config)

            if preview["invalidated_clips"] > 0 and not confirm_invalidation:
                return {"status": "confirmation_required", **preview}

            self._save_voice_config(new_config)
            self.refresh_auto_narrator_aliases(voice_config=new_config)
            if preview["invalidated_clips"] <= 0:
                return {"status": "saved", "invalidated_clips": 0, "deleted_files": 0, **preview}

            applied = self.invalidate_chunk_audio_indices(preview["affected_indices"])
            return {"status": "saved", **preview, **applied}

        def get_engine(self):
            if self.engine:
                return self.engine

            try:
                self.engine = TTSEngine(self._load_app_config())
                backend = self.engine.local_backend if self.engine.mode == "local" else "external"
                print(f"TTS engine initialized (mode={self.engine.mode}, backend={backend})")
                return self.engine
            except Exception as e:
                print(f"Failed to initialize TTS engine: {e}")
                return None

        def unload_tts_engine(self):
            engine = self.engine
            self.engine = None
            if not engine:
                return False

            try:
                clear_cache = getattr(engine, "_clear_gpu_cache", None)
                if callable(clear_cache):
                    clear_cache()
            except Exception:
                pass

            return True
