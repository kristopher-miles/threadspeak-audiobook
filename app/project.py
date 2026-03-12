import os
import json
import shutil
import subprocess
import threading
import zipfile
import io
import re
import time
import copy
import tempfile
import uuid
from tts import (
    TTSEngine,
    combine_audio_with_pauses,
    sanitize_filename,
    DEFAULT_PAUSE_MS,
    SAME_SPEAKER_PAUSE_MS
)
from audio_validation import get_audio_duration_seconds, validate_audio_clip
from pydub import AudioSegment
from script_store import (
    apply_dictionary_to_text,
    load_script_document,
)
from source_document import load_source_document, iter_document_paragraphs

MAX_CHUNK_CHARS = 500
CHAPTER_HEADING_RE = re.compile(
    r'^(chapter|part|book|volume|prologue|epilogue|introduction|conclusion|act|section)\b',
    re.IGNORECASE,
)

def get_speaker(entry):
    """Get speaker from entry, checking both 'speaker' and 'type' fields."""
    return entry.get("speaker") or entry.get("type") or ""


def _is_structural_text(text):
    """Check if text is a title, chapter heading, dedication, or other structural fragment."""
    stripped = text.strip()
    if not stripped:
        return True
    # Very short and not a full sentence (no sentence-ending punctuation)
    if len(stripped) < 80 and not stripped[-1] in '.!?':
        return True
    return False


def _extract_chapter_name(entry):
    chapter = (entry.get("chapter") or "").strip()
    if chapter:
        return chapter

    text = (entry.get("text") or "").strip()
    if text and CHAPTER_HEADING_RE.match(text):
        return text

    return None


def _build_chunk(speaker, text, instruct, chapter=None):
    chunk = {
        "speaker": speaker,
        "text": text,
        "instruct": instruct,
        "uid": uuid.uuid4().hex,
    }
    if chapter:
        chunk["chapter"] = chapter
    return chunk


def group_into_chunks(script_entries, max_chars=MAX_CHUNK_CHARS):
    """Group consecutive entries by same speaker into chunks up to max_chars"""
    if not script_entries:
        return []

    chunks = []
    current_speaker = get_speaker(script_entries[0])
    current_text = script_entries[0].get("text", "")
    current_instruct = script_entries[0].get("instruct", "")
    current_chapter = _extract_chapter_name(script_entries[0])

    for entry in script_entries[1:]:
        speaker = get_speaker(entry)
        text = entry.get("text", "")
        instruct = entry.get("instruct", "")
        entry_chapter = _extract_chapter_name(entry)
        effective_chapter = entry_chapter or current_chapter

        # Don't merge structural text (titles, chapter headings, dedications)
        if (speaker == current_speaker and instruct == current_instruct
                and effective_chapter == current_chapter
                and not _is_structural_text(current_text)
                and not _is_structural_text(text)):
            combined = current_text + " " + text
            if len(combined) <= max_chars:
                current_text = combined
            else:
                chunks.append(_build_chunk(current_speaker, current_text, current_instruct, current_chapter))
                current_text = text
                current_instruct = instruct
                current_chapter = effective_chapter
        else:
            chunks.append(_build_chunk(current_speaker, current_text, current_instruct, current_chapter))
            current_speaker = speaker
            current_text = text
            current_instruct = instruct
            current_chapter = effective_chapter

    # Don't forget the last chunk
    chunks.append(_build_chunk(current_speaker, current_text, current_instruct, current_chapter))

    return chunks

class ProjectManager:
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.script_path = os.path.join(root_dir, "annotated_script.json")
        self.chunks_path = os.path.join(root_dir, "chunks.json")
        self.voicelines_dir = os.path.join(root_dir, "voicelines")
        self.voice_config_path = os.path.join(root_dir, "voice_config.json")
        self.config_path = os.path.join(root_dir, "app", "config.json")

        # Ensure voicelines dir exists
        os.makedirs(self.voicelines_dir, exist_ok=True)

        self.engine = None
        self._chunks_lock = threading.Lock()  # Thread-safe file writes

    def _load_voice_config(self):
        if os.path.exists(self.voice_config_path):
            try:
                with open(self.voice_config_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, ValueError):
                return {}
        return {}

    def _save_voice_config(self, voice_config):
        with open(self.voice_config_path, "w", encoding="utf-8") as f:
            json.dump(voice_config, f, indent=2, ensure_ascii=False)

    def _current_script_title(self):
        state_path = os.path.join(self.root_dir, "state.json")
        if os.path.exists(state_path):
            try:
                with open(state_path, "r", encoding="utf-8") as f:
                    state = json.load(f)
                input_path = state.get("input_file_path") or ""
                if input_path:
                    return os.path.splitext(os.path.basename(input_path))[0].strip()
            except (json.JSONDecodeError, ValueError, OSError):
                pass
        return "Project"

    def _load_state(self):
        state_path = os.path.join(self.root_dir, "state.json")
        if not os.path.exists(state_path):
            return {}
        try:
            with open(state_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, ValueError, OSError):
            return {}

    def _save_state(self, state):
        state_path = os.path.join(self.root_dir, "state.json")
        with open(state_path, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2, ensure_ascii=False)

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
        ref = "" if chunk_ref is None else str(chunk_ref)

        for index, chunk in enumerate(chunks):
            if str(chunk.get("uid") or "") == ref:
                return index

        try:
            numeric_ref = int(ref)
        except (TypeError, ValueError):
            return None

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

    @staticmethod
    def _escape_concat_path(path):
        return path.replace("\\", "\\\\").replace("'", r"'\''")

    @classmethod
    def _write_concat_line(cls, handle, path):
        handle.write(f"file '{cls._escape_concat_path(path)}'\n")

    def _collect_merge_timeline(self, progress_callback=None, merge_started_at=None):
        chunks = self.load_chunks()
        timeline = []
        timeline_size_bytes = 0

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
            })

        for chunk in chunks:
            if chunk.get("status") != "done":
                continue
            path = chunk.get("audio_path")
            if not path:
                continue
            full_path = os.path.join(self.root_dir, path)
            if not os.path.exists(full_path):
                continue
            item = {
                "chunk": chunk,
                "full_path": full_path,
                "file_size_bytes": os.path.getsize(full_path),
            }
            timeline.append(item)
            timeline_size_bytes += item["file_size_bytes"]
            if progress_callback and len(timeline) % 100 == 0:
                progress_callback({
                    "stage": "preparing",
                    "chapter_index": 0,
                    "total_chapters": 0,
                    "chapter_label": f"Indexed {len(timeline)} clips",
                    "elapsed_seconds": max(0.0, time.time() - (merge_started_at or time.time())),
                    "merged_duration_seconds": 0.0,
                    "estimated_size_bytes": timeline_size_bytes,
                    "output_file_size_bytes": 0,
                })

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

    def _create_silence_assets(self, temp_dir):
        default_silence_path = os.path.join(temp_dir, "pause_default.mp3")
        same_silence_path = os.path.join(temp_dir, "pause_same_speaker.mp3")
        default_export = AudioSegment.silent(duration=DEFAULT_PAUSE_MS).export(default_silence_path, format="mp3")
        if hasattr(default_export, "close"):
            default_export.close()
        same_export = AudioSegment.silent(duration=SAME_SPEAKER_PAUSE_MS).export(same_silence_path, format="mp3")
        if hasattr(same_export, "close"):
            same_export.close()
        return {
            "default_path": default_silence_path,
            "same_path": same_silence_path,
            "default_size_bytes": os.path.getsize(default_silence_path),
            "same_size_bytes": os.path.getsize(same_silence_path),
        }

    def _export_concat_mp3(self, concat_path, output_path, progress_tick=None):
        command = [
            "ffmpeg",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            concat_path,
            "-vn",
            "-c:a",
            "libmp3lame",
            "-q:a",
            "2",
            output_path,
        ]
        process = subprocess.Popen(
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        while process.poll() is None:
            if progress_tick:
                progress_tick()
            time.sleep(1)

        return process.returncode == 0

    def _optimized_export_part_basename(self, part_index):
        title = self._current_script_title().strip() or "Project"
        safe_title = re.sub(r"[^A-Za-z0-9]+", "-", title).strip("-").lower() or "project"
        return f"{safe_title}-{part_index:02d}.mp3"

    def _load_generation_settings(self):
        config = {}
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, "r", encoding="utf-8") as f:
                    config = json.load(f)
            except (json.JSONDecodeError, ValueError):
                config = {}
        return config.get("generation", {})

    def load_source_document(self):
        input_path = (self._load_state().get("input_file_path") or "").strip()
        if not input_path or not os.path.exists(input_path):
            raise ValueError("No uploaded source document found")
        return load_source_document(input_path)

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
        return merged

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
        source_document = self.load_source_document()
        generation_settings = self._load_generation_settings()
        chunk_size = int(generation_settings.get("chunk_size") or 3000)
        target_chars = max(int(target_chars or (chunk_size * 2)), 1)

        selected = []
        total_chars = 0

        for paragraph in iter_document_paragraphs(source_document):
            if not self._paragraph_mentions_speaker(paragraph["text"], speaker):
                continue
            selected.append(paragraph)
            total_chars += len(paragraph["text"])
            if total_chars >= target_chars:
                break

        return {
            "speaker": speaker,
            "target_chars": target_chars,
            "context_chars": total_chars,
            "paragraphs": selected,
        }

    def build_voice_suggestion_prompt(self, speaker, prompt_template):
        context = self.collect_voice_suggestion_context(speaker)
        paragraphs = context["paragraphs"]

        if paragraphs:
            context_blocks = []
            for item in paragraphs:
                chapter = (item.get("chapter") or "").strip()
                text = item["text"]
                context_blocks.append(f"[{chapter}] {text}" if chapter else text)
            context_prefix = (
                f'Source paragraphs mentioning "{speaker}":\n\n' +
                "\n\n".join(context_blocks)
            )
        else:
            context_prefix = f'No source paragraphs mentioning "{speaker}" were found in the uploaded story.'

        rendered_prompt = (prompt_template or "").replace("{character_name}", speaker)
        return {
            **context,
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

    def materialize_design_voice(self, speaker, description=None, sample_text=None, force=False, voice_config=None):
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

    def resolve_voice_speaker(self, speaker, voice_config):
        """Resolve a speaker alias to a configured target speaker.

        Aliases are case-insensitive and can chain, but any invalid target,
        self-alias, or loop falls back to the original speaker.
        """
        original = speaker or ""
        current = original
        if not current:
            return original

        lookup = {}
        for name in voice_config.keys():
            normalized = self._normalize_speaker_name(name)
            if normalized and normalized not in lookup:
                lookup[normalized] = name

        seen = {self._normalize_speaker_name(original)}
        while current:
            voice_data = voice_config.get(current, {})
            alias = (voice_data.get("alias") or "").strip()
            if not alias:
                return current

            target = lookup.get(self._normalize_speaker_name(alias))
            if not target or self._normalize_speaker_name(target) == self._normalize_speaker_name(current):
                return current

            target_key = self._normalize_speaker_name(target)
            if target_key in seen:
                print(f"Alias loop detected for speaker '{original}'. Falling back to original speaker.")
                return original

            seen.add(target_key)
            current = target

        return original

    def get_engine(self):
        if self.engine:
            return self.engine

        # Load config
        config = {}
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, "r", encoding="utf-8") as f:
                    config = json.load(f)
            except: pass

        try:
            self.engine = TTSEngine(config)
            print(f"TTS engine initialized (mode={self.engine.mode})")
            return self.engine
        except Exception as e:
            print(f"Failed to initialize TTS engine: {e}")
            return None

    def load_chunks(self):
        if os.path.exists(self.chunks_path):
            try:
                with open(self.chunks_path, "r", encoding="utf-8") as f:
                    chunks = json.load(f)
                if self._ensure_chunk_uids(chunks):
                    self.save_chunks(chunks)
                return chunks
            except (json.JSONDecodeError, ValueError) as e:
                print(f"WARNING: chunks.json is corrupted ({e}). Regenerating from script...")
                os.remove(self.chunks_path)

        # If no chunks (or corrupted), generate from script
        if os.path.exists(self.script_path):
            script = load_script_document(self.script_path)["entries"]
            chunks = group_into_chunks(script)

            # Initialize chunk status
            for i, chunk in enumerate(chunks):
                chunk["id"] = i
                chunk["uid"] = chunk.get("uid") or self._new_chunk_uid()
                chunk["status"] = "pending" # pending, generating, done, error
                chunk["audio_path"] = None
                chunk["audio_validation"] = None
                chunk["auto_regen_count"] = 0

            self.save_chunks(chunks)
            return chunks

        return []

    def reconcile_chunk_audio_states(self):
        """Repair stale error states when a stored clip is actually valid.

        This is intended for editor/UI refreshes after interrupted or large
        generations, where a chunk may still be marked as error even though its
        audio file exists and now passes the duration sanity check.
        """
        if not os.path.exists(self.chunks_path):
            return self.load_chunks()

        with self._chunks_lock:
            with open(self.chunks_path, "r", encoding="utf-8") as f:
                chunks = json.load(f)

            dictionary_entries = self.load_dictionary_entries()
            changed = False

            for chunk in chunks:
                if chunk.get("status") != "error":
                    continue

                audio_path = chunk.get("audio_path")
                if not audio_path:
                    continue

                full_audio_path = os.path.join(self.root_dir, audio_path)
                if not os.path.exists(full_audio_path):
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

                if validation["is_valid"]:
                    chunk["status"] = "done"
                    chunk["audio_validation"] = validation
                    chunk["auto_regen_count"] = 0
                    changed = True
                elif chunk.get("audio_validation") != validation:
                    chunk["audio_validation"] = validation
                    changed = True

            if changed:
                self._atomic_json_write(chunks, self.chunks_path)

            return chunks

    def _validate_chunk_audio(self, chunk, dictionary_entries):
        audio_path = chunk.get("audio_path")
        if not audio_path:
            return None

        full_audio_path = os.path.join(self.root_dir, audio_path)
        if not os.path.exists(full_audio_path):
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
        outcome = {"recovered": 0, "reset": 0}

        with self._chunks_lock:
            if not os.path.exists(self.chunks_path):
                return outcome

            with open(self.chunks_path, "r", encoding="utf-8") as f:
                chunks = json.load(f)

            if indices is None:
                index_iter = range(len(chunks))
            else:
                index_iter = [index for index in indices if 0 <= index < len(chunks)]

            dictionary_entries = self.load_dictionary_entries()
            changed = False

            for index in index_iter:
                chunk = chunks[index]
                if chunk.get("status") != "generating":
                    continue
                if generation_token is not None and chunk.get("generation_token") != generation_token:
                    continue

                try:
                    validation = self._validate_chunk_audio(chunk, dictionary_entries)
                except Exception as e:
                    print(f"Warning: failed to validate interrupted chunk {chunk.get('id')}: {e}")
                    validation = None

                if validation and validation["is_valid"]:
                    chunk["status"] = "done"
                    chunk["audio_validation"] = validation
                    chunk["auto_regen_count"] = 0
                    outcome["recovered"] += 1
                else:
                    chunk["status"] = "pending"
                    outcome["reset"] += 1

                chunk.pop("generation_token", None)
                changed = True

            if changed:
                self._atomic_json_write(chunks, self.chunks_path)

        return outcome

    def load_script_document(self):
        if not os.path.exists(self.script_path):
            return {"entries": [], "dictionary": []}
        return load_script_document(self.script_path)

    def load_dictionary_entries(self):
        return self.load_script_document()["dictionary"]

    def _atomic_json_write(self, data, target_path, max_retries=5):
        """Atomically write JSON data with retry logic for Windows file locking."""
        tmp_path = target_path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        for attempt in range(max_retries):
            try:
                os.replace(tmp_path, target_path)
                return
            except OSError as e:
                if attempt < max_retries - 1 and (
                    e.errno == 5 or "Access is denied" in str(e) or "being used by another process" in str(e)
                ):
                    delay = 0.05 * (2 ** attempt)
                    time.sleep(delay)
                    continue
                raise

    def save_chunks(self, chunks):
        with self._chunks_lock:
            self._ensure_chunk_uids(chunks)
            self._atomic_json_write(chunks, self.chunks_path)

    def _update_chunk_fields(self, index, **fields):
        """Atomically update fields on a single chunk (thread-safe read-modify-write).

        Unlike load_chunks() + modify + save_chunks(), this holds the lock for the
        entire read-modify-write cycle, preventing concurrent threads from
        overwriting each other's updates.
        """
        with self._chunks_lock:
            if not os.path.exists(self.chunks_path):
                return None
            with open(self.chunks_path, "r", encoding="utf-8") as f:
                chunks = json.load(f)
            if not (0 <= index < len(chunks)):
                return None
            chunks[index].update(fields)
            self._atomic_json_write(chunks, self.chunks_path)
            return chunks[index]

    def _claim_chunk_generation(self, index, generation_token=None):
        with self._chunks_lock:
            if not os.path.exists(self.chunks_path):
                return None
            with open(self.chunks_path, "r", encoding="utf-8") as f:
                chunks = json.load(f)
            if not (0 <= index < len(chunks)):
                return None
            chunks[index]["status"] = "generating"
            if generation_token is not None:
                chunks[index]["generation_token"] = generation_token
            else:
                chunks[index].pop("generation_token", None)
            self._atomic_json_write(chunks, self.chunks_path)
            return chunks[index]

    def _claim_chunks_generation(self, indices, generation_token=None):
        claimed = 0
        with self._chunks_lock:
            if not os.path.exists(self.chunks_path):
                return claimed
            with open(self.chunks_path, "r", encoding="utf-8") as f:
                chunks = json.load(f)
            for index in indices:
                if not (0 <= index < len(chunks)):
                    continue
                chunks[index]["status"] = "generating"
                if generation_token is not None:
                    chunks[index]["generation_token"] = generation_token
                else:
                    chunks[index].pop("generation_token", None)
                claimed += 1
            self._atomic_json_write(chunks, self.chunks_path)
        return claimed

    def _chunk_token_matches(self, chunks, index, generation_token=None):
        if generation_token is None:
            return True
        if not (0 <= index < len(chunks)):
            return False
        return chunks[index].get("generation_token") == generation_token

    def chunk_has_generation_token(self, index, generation_token=None):
        with self._chunks_lock:
            if not os.path.exists(self.chunks_path):
                return False
            with open(self.chunks_path, "r", encoding="utf-8") as f:
                chunks = json.load(f)
            return self._chunk_token_matches(chunks, index, generation_token)

    def _update_chunk_fields_if_token(self, index, expected_token=None, **fields):
        with self._chunks_lock:
            if not os.path.exists(self.chunks_path):
                return None
            with open(self.chunks_path, "r", encoding="utf-8") as f:
                chunks = json.load(f)
            if not (0 <= index < len(chunks)):
                return None
            if not self._chunk_token_matches(chunks, index, expected_token):
                return None
            for key, value in fields.items():
                if key == "generation_token":
                    if value is None:
                        chunks[index].pop("generation_token", None)
                    else:
                        chunks[index]["generation_token"] = value
                else:
                    chunks[index][key] = value
            if fields.get("status") != "generating" and "generation_token" not in fields:
                chunks[index].pop("generation_token", None)
            self._atomic_json_write(chunks, self.chunks_path)
            return chunks[index]

    def reset_generating_chunks(self, indices=None, generation_token=None, target_status="pending"):
        reset_count = 0
        with self._chunks_lock:
            if not os.path.exists(self.chunks_path):
                return reset_count
            with open(self.chunks_path, "r", encoding="utf-8") as f:
                chunks = json.load(f)
            if indices is None:
                index_iter = range(len(chunks))
            else:
                index_iter = [index for index in indices if 0 <= index < len(chunks)]
            for index in index_iter:
                chunk = chunks[index]
                if chunk.get("status") != "generating":
                    continue
                if generation_token is not None and chunk.get("generation_token") != generation_token:
                    continue
                chunk["status"] = target_status
                chunk.pop("generation_token", None)
                reset_count += 1
            if reset_count:
                self._atomic_json_write(chunks, self.chunks_path)
        return reset_count

    def insert_chunk(self, after_ref):
        """Insert an empty chunk after the given index. Returns the new chunk list."""
        with self._chunks_lock:
            if not os.path.exists(self.chunks_path):
                return None
            with open(self.chunks_path, "r", encoding="utf-8") as f:
                chunks = json.load(f)
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
            chunks.insert(after_index + 1, new_chunk)

            # Re-number all IDs
            for i, chunk in enumerate(chunks):
                chunk["id"] = i

            self._atomic_json_write(chunks, self.chunks_path)
            return chunks

    def delete_chunk(self, chunk_ref):
        """Delete a chunk at the given index. Returns (deleted_chunk, updated_chunks) or None."""
        with self._chunks_lock:
            if not os.path.exists(self.chunks_path):
                return None
            with open(self.chunks_path, "r", encoding="utf-8") as f:
                chunks = json.load(f)
            index = self.resolve_chunk_index(chunk_ref, chunks)
            if index is None or not (0 <= index < len(chunks)):
                return None
            if len(chunks) <= 1:
                return None  # don't allow deleting the last chunk

            restore_after_uid = chunks[index - 1].get("uid") if index > 0 else None
            deleted = chunks.pop(index)

            # Re-number all IDs
            for i, chunk in enumerate(chunks):
                chunk["id"] = i

            self._atomic_json_write(chunks, self.chunks_path)
            return deleted, chunks, restore_after_uid

    def restore_chunk(self, at_index, chunk_data, after_uid=None):
        """Re-insert a chunk at a specific index. Returns the updated chunk list."""
        with self._chunks_lock:
            if not os.path.exists(self.chunks_path):
                return None
            with open(self.chunks_path, "r", encoding="utf-8") as f:
                chunks = json.load(f)

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

    def decompose_long_segments(self, chapter=None, max_words=25):
        with self._chunks_lock:
            if not os.path.exists(self.chunks_path):
                return {
                    "changed": 0,
                    "total_chunks": 0,
                    "processed_scope": 0,
                    "chapter": chapter,
                    "max_words": max_words,
                }

            with open(self.chunks_path, "r", encoding="utf-8") as f:
                chunks = json.load(f)

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
                    left_chunk["status"] = "pending"
                    left_chunk["audio_path"] = None
                    left_chunk["audio_validation"] = None
                    left_chunk["auto_regen_count"] = 0
                    left_chunk.pop("generation_token", None)

                    right_chunk = copy.deepcopy(base_chunk)
                    right_chunk["text"] = right_text
                    right_chunk["status"] = "pending"
                    right_chunk["audio_path"] = None
                    right_chunk["audio_validation"] = None
                    right_chunk["auto_regen_count"] = 0
                    right_chunk.pop("generation_token", None)

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
            if not os.path.exists(self.chunks_path):
                return {
                    "changed": 0,
                    "total_chunks": 0,
                    "processed_scope": 0,
                    "chapter": chapter,
                    "min_words": min_words,
                }

            with open(self.chunks_path, "r", encoding="utf-8") as f:
                chunks = json.load(f)

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
        chunks = self.load_chunks()
        index = self.resolve_chunk_index(chunk_ref, chunks)
        if index is not None and 0 <= index < len(chunks):
            chunk = chunks[index]
            # Update fields
            if "text" in data: chunk["text"] = data["text"]
            if "instruct" in data: chunk["instruct"] = data["instruct"]
            if "speaker" in data: chunk["speaker"] = data["speaker"]

            # If text/instruct/speaker changed, reset status (but keep old audio until regen)
            if "text" in data or "instruct" in data or "speaker" in data:
                chunk["status"] = "pending"
                chunk["audio_validation"] = None
                chunk["auto_regen_count"] = 0

            print(f"update_chunk({index}): instruct='{chunk.get('instruct', '')}', speaker='{chunk.get('speaker', '')}'")
            self.save_chunks(chunks)
            return chunk
        return None

    def _load_tts_settings(self):
        config = {}
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, "r", encoding="utf-8") as f:
                    config = json.load(f)
            except (json.JSONDecodeError, ValueError):
                pass
        return config.get("tts", {})

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

    def _finalize_generated_audio(self, index, speaker, text, temp_path, attempt=0):
        if not os.path.exists(temp_path) or os.path.getsize(temp_path) == 0:
            return {
                "status": "error",
                "audio_path": None,
                "audio_validation": None,
                "error": "Generated audio file is missing or empty",
            }

        print(f"Generated WAV size: {os.path.getsize(temp_path)} bytes")

        filename_base = f"voiceline_{index+1:04d}_{sanitize_filename(speaker)}"
        if attempt > 0:
            filename_base = f"{filename_base}_retry{attempt}"
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
            }

        print(f"Chunk {index} failed audio sanity check: {validation['error']}")
        return {
            "status": "error",
            "audio_path": audio_path,
            "audio_validation": validation,
            "error": validation["error"],
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
            voice_config = self._load_voice_config()
            resolved_speaker = self.resolve_voice_speaker(speaker, voice_config)
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
                result = self._finalize_generated_audio(index, speaker, transformed_text, temp_path, attempt=attempt)
                if result["status"] == "error" and auto_regen_retry_attempts > 0 and attempt < auto_regen_retry_attempts:
                    self._update_chunk_fields_if_token(
                        index,
                        generation_token,
                        status="pending",
                        audio_path=result["audio_path"],
                        audio_validation=result["audio_validation"],
                        auto_regen_count=attempt + 1,
                        generation_token=None,
                    )
                    self._cleanup_temp_file(temp_path)
                    print(f"Chunk {index} failed sanity check; auto-regenerating attempt {attempt + 1}/{auto_regen_retry_attempts}")
                    return self.generate_chunk_audio(index, attempt=attempt + 1, generation_token=generation_token)
                self._update_chunk_fields_if_token(
                    index,
                    generation_token,
                    status=result["status"],
                    audio_path=result["audio_path"],
                    audio_validation=result["audio_validation"],
                    auto_regen_count=attempt,
                    generation_token=None,
                )
                self._cleanup_temp_file(temp_path)
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
            return False, str(e)

    def merge_audio(self, progress_callback=None):
        merge_started_at = time.time()
        timeline = self._collect_merge_timeline(progress_callback=progress_callback, merge_started_at=merge_started_at)

        if not timeline:
            return False, "No audio segments found"

        chapter_groups = self._group_timeline_by_chapter(timeline)

        output_filename = "cloned_audiobook.mp3"
        output_path = os.path.join(self.root_dir, output_filename)
        temp_dir = tempfile.mkdtemp(prefix="merge_audio_", dir=self.root_dir)
        concat_path = os.path.join(temp_dir, "concat.txt")

        try:
            silence_assets = self._create_silence_assets(temp_dir)

            estimated_size_bytes = 0
            previous_speaker = None

            with open(concat_path, "w", encoding="utf-8") as concat_file:
                for chapter_index, (chapter_label, chapter_items) in enumerate(chapter_groups, start=1):
                    chapter_first_speaker = chapter_items[0]["chunk"]["speaker"] if chapter_items else None
                    if previous_speaker is not None and chapter_first_speaker is not None:
                        pause_path = silence_assets["same_path"] if previous_speaker == chapter_first_speaker else silence_assets["default_path"]
                        pause_size = silence_assets["same_size_bytes"] if previous_speaker == chapter_first_speaker else silence_assets["default_size_bytes"]
                        self._write_concat_line(concat_file, pause_path)
                        estimated_size_bytes += pause_size

                    chapter_previous_speaker = None
                    for item in chapter_items:
                        speaker = item["chunk"]["speaker"]
                        if chapter_previous_speaker is not None:
                            pause_path = silence_assets["same_path"] if chapter_previous_speaker == speaker else silence_assets["default_path"]
                            pause_size = silence_assets["same_size_bytes"] if chapter_previous_speaker == speaker else silence_assets["default_size_bytes"]
                            self._write_concat_line(concat_file, pause_path)
                            estimated_size_bytes += pause_size
                        self._write_concat_line(concat_file, item["full_path"])
                        estimated_size_bytes += item["file_size_bytes"]
                        chapter_previous_speaker = speaker

                    previous_speaker = chapter_previous_speaker if chapter_previous_speaker is not None else previous_speaker
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

            if not self._export_concat_mp3(
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
            ):
                return False, "ffmpeg merge failed"

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

    def export_optimized_mp3_zip(self, progress_callback=None, max_part_seconds=7200):
        merge_started_at = time.time()
        timeline = self._collect_merge_timeline(progress_callback=progress_callback, merge_started_at=merge_started_at)
        if not timeline:
            return False, "No audio segments found"

        chapter_groups = self._group_timeline_by_chapter(timeline)
        zip_path = os.path.join(self.root_dir, "optimized_audiobook.zip")
        temp_dir = tempfile.mkdtemp(prefix="optimized_export_", dir=self.root_dir)

        try:
            silence_assets = self._create_silence_assets(temp_dir)
            chapter_exports = []
            estimated_size_bytes = 0

            for chapter_index, (chapter_label, chapter_items) in enumerate(chapter_groups, start=1):
                concat_path = os.path.join(temp_dir, f"chapter_{chapter_index:03d}.txt")
                chapter_output_path = os.path.join(temp_dir, f"chapter_{chapter_index:03d}.mp3")
                chapter_estimated_size = 0
                chapter_previous_speaker = None

                with open(concat_path, "w", encoding="utf-8") as concat_file:
                    for item in chapter_items:
                        speaker = item["chunk"]["speaker"]
                        if chapter_previous_speaker is not None:
                            pause_path = silence_assets["same_path"] if chapter_previous_speaker == speaker else silence_assets["default_path"]
                            pause_size = silence_assets["same_size_bytes"] if chapter_previous_speaker == speaker else silence_assets["default_size_bytes"]
                            self._write_concat_line(concat_file, pause_path)
                            chapter_estimated_size += pause_size
                        self._write_concat_line(concat_file, item["full_path"])
                        chapter_estimated_size += item["file_size_bytes"]
                        chapter_previous_speaker = speaker

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

                if not self._export_concat_mp3(concat_path, chapter_output_path):
                    return False, f"Failed to export chapter {chapter_label}"

                chapter_duration_seconds = AudioSegment.from_file(chapter_output_path).duration_seconds
                chapter_size_bytes = os.path.getsize(chapter_output_path)
                estimated_size_bytes += chapter_size_bytes
                chapter_exports.append({
                    "label": chapter_label,
                    "path": chapter_output_path,
                    "duration_seconds": chapter_duration_seconds,
                    "file_size_bytes": chapter_size_bytes,
                    "first_speaker": chapter_items[0]["chunk"]["speaker"],
                    "last_speaker": chapter_items[-1]["chunk"]["speaker"],
                })

            part_groups = []
            current_group = []
            current_duration = 0.0
            for chapter in chapter_exports:
                chapter_pause_seconds = 0.0
                if current_group:
                    previous = current_group[-1]
                    chapter_pause_seconds = SAME_SPEAKER_PAUSE_MS / 1000.0 if previous["last_speaker"] == chapter["first_speaker"] else DEFAULT_PAUSE_MS / 1000.0
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
                            previous = part_group[chapter_offset - 1]
                            pause_path = silence_assets["same_path"] if previous["last_speaker"] == chapter["first_speaker"] else silence_assets["default_path"]
                            pause_size = silence_assets["same_size_bytes"] if previous["last_speaker"] == chapter["first_speaker"] else silence_assets["default_size_bytes"]
                            self._write_concat_line(concat_file, pause_path)
                            part_estimated_size += pause_size
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

                if not self._export_concat_mp3(
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
                ):
                    return False, f"Failed to export optimized part {part_index}"

                part_paths.append((part_output_path, part_basename))

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

    def export_audacity(self):
        """Export project as an Audacity-compatible zip with per-speaker WAV tracks,
        a LOF file for auto-import, and a labels file for chunk annotations."""
        chunks = self.load_chunks()

        # Phase 1 — Compute timeline (matching merge_audio pause logic exactly)
        timeline = []  # list of (chunk, segment, abs_start_ms)
        prev_speaker = None
        cursor_ms = 0

        for chunk in chunks:
            if chunk.get("status") != "done":
                continue
            path = chunk.get("audio_path")
            if not path:
                continue
            full_path = os.path.join(self.root_dir, path)
            if not os.path.exists(full_path):
                continue
            try:
                segment = AudioSegment.from_file(full_path)
            except Exception as e:
                print(f"Error loading audio for Audacity export {path}: {e}")
                continue

            speaker = chunk["speaker"]
            if prev_speaker is not None:
                if speaker == prev_speaker:
                    cursor_ms += SAME_SPEAKER_PAUSE_MS
                else:
                    cursor_ms += DEFAULT_PAUSE_MS

            timeline.append((chunk, segment, cursor_ms))
            cursor_ms += len(segment)
            prev_speaker = speaker

        if not timeline:
            return False, "No audio segments found"

        total_duration_ms = cursor_ms

        # Phase 2 — Build per-speaker WAV tracks
        speakers_ordered = []
        seen = set()
        for chunk, segment, start_ms in timeline:
            if chunk["speaker"] not in seen:
                speakers_ordered.append(chunk["speaker"])
                seen.add(chunk["speaker"])

        speaker_tracks = {}
        for speaker in speakers_ordered:
            track_cursor = 0
            track = AudioSegment.empty()

            for chunk, segment, start_ms in timeline:
                if chunk["speaker"] != speaker:
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
        zip_path = os.path.join(self.root_dir, "audacity_export.zip")
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("project.lof", lof_content)
            zf.writestr("labels.txt", labels_content)

            for speaker in speakers_ordered:
                safe_name = sanitize_filename(speaker)
                wav_buffer = io.BytesIO()
                speaker_tracks[speaker].export(wav_buffer, format="wav")
                zf.writestr(f"{safe_name}.wav", wav_buffer.getvalue())

        return True, zip_path

    def merge_m4b(self, per_chunk_chapters=False, metadata=None):
        """Merge audio chunks into an M4B audiobook with chapter markers.

        Args:
            per_chunk_chapters: If True, each chunk is a chapter. If False,
                detect chapter headings and group chunks into sections.
            metadata: Optional dict with keys: title, author, narrator, year,
                description, cover_path (absolute path to cover image).

        Returns:
            tuple: (success: bool, message: str)
        """
        metadata = metadata or {}
        chunks = self.load_chunks()

        # Phase 1 — Compute timeline (same logic as export_audacity)
        timeline = []  # list of (chunk, segment, abs_start_ms)
        prev_speaker = None
        cursor_ms = 0

        for chunk in chunks:
            if chunk.get("status") != "done":
                continue
            path = chunk.get("audio_path")
            if not path:
                continue
            full_path = os.path.join(self.root_dir, path)
            if not os.path.exists(full_path):
                continue
            try:
                segment = AudioSegment.from_file(full_path)
            except Exception as e:
                print(f"Error loading audio for M4B export {path}: {e}")
                continue

            speaker = chunk["speaker"]
            if prev_speaker is not None:
                if speaker == prev_speaker:
                    cursor_ms += SAME_SPEAKER_PAUSE_MS
                else:
                    cursor_ms += DEFAULT_PAUSE_MS

            timeline.append((chunk, segment, cursor_ms))
            cursor_ms += len(segment)
            prev_speaker = speaker

        if not timeline:
            return False, "No audio segments found"

        # Phase 2 — Build chapters
        chapters = self._build_m4b_chapters(timeline, per_chunk_chapters)
        print(f"  M4B: {len(chapters)} chapters")

        # Phase 3 — Combine audio and export to temp WAV
        audio_segments = [seg for _, seg, _ in timeline]
        speakers = [chunk["speaker"] for chunk, _, _ in timeline]
        final_audio = combine_audio_with_pauses(audio_segments, speakers)

        temp_wav = os.path.join(self.root_dir, "temp_m4b_combined.wav")
        meta_path = os.path.join(self.root_dir, "temp_m4b_meta.txt")
        output_path = os.path.join(self.root_dir, "audiobook.m4b")

        try:
            final_audio.export(temp_wav, format="wav")

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

            cmd = ["ffmpeg", "-y", "-i", temp_wav]
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
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            if result.returncode != 0:
                print(f"FFmpeg stderr: {result.stderr[-500:]}")
                return False, f"FFmpeg failed (exit {result.returncode})"

        finally:
            for tmp in [temp_wav, meta_path]:
                if os.path.exists(tmp):
                    try:
                        os.remove(tmp)
                    except OSError:
                        pass

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

    # Regex for detecting chapter/section headings in chunk text
    _HEADING_RE = re.compile(
        r'^(chapter|part|book|volume|prologue|epilogue|introduction|conclusion|act|section)\b',
        re.IGNORECASE
    )

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
                chunks = self.load_chunks()
                if chunks:
                    for idx in indices:
                        if idx not in done_indices and 0 <= idx < len(chunks) and chunks[idx].get("status") == "generating":
                            chunks[idx]["status"] = "pending"
                            chunks[idx].pop("generation_token", None)
                            results["cancelled"] += 1
                    self.save_chunks(chunks)

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

        for idx in indices:
            if not (0 <= idx < len(chunks)):
                groups.setdefault("custom", []).append(idx)
                continue

            speaker = chunks[idx].get("speaker", "")
            speaker = self.resolve_voice_speaker(speaker, voice_config)
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

        for idx in indices:
            if not (0 <= idx < len(chunks)):
                groups.setdefault("", []).append(idx)
                continue

            speaker = chunks[idx].get("speaker", "")
            resolved = self.resolve_voice_speaker(speaker, voice_config)
            groups.setdefault(resolved, []).append(idx)

        reordered = []
        for speaker, group_indices in groups.items():
            label = speaker or "<unknown>"
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
        resolved_speakers = {
            self.resolve_voice_speaker(chunks[idx].get("speaker", ""), voice_config)
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
                    batch_chunks.append({
                        "index": idx,
                        "text": transformed_text,
                        "instruct": chunk.get("instruct", ""),
                        "speaker": self.resolve_voice_speaker(chunk.get("speaker", ""), voice_config)
                    })

            # Call batch TTS with single seed
            batch_start = time.time()
            batch_results = engine.generate_batch(batch_chunks, voice_config, self.root_dir, batch_seed)

            # Process completed chunks - convert to MP3 and update status
            chunks = self.load_chunks()  # Reload for each batch

            processed_in_batch = len(batch_results["completed"]) + len(batch_results["failed"])
            shared_elapsed = (time.time() - batch_start) / processed_in_batch if processed_in_batch > 0 else 0.0

            for idx in batch_results["completed"]:
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
                    result = self._finalize_generated_audio(
                        idx,
                        speaker,
                        transformed_texts.get(idx, chunk.get("text", "")),
                        temp_path,
                        attempt=0,
                    )
                    updated_status = "pending" if (result["status"] == "error" and auto_regen_retry_attempts > 0) else result["status"]
                    updated_chunk = self._update_chunk_fields_if_token(
                        idx,
                        generation_token,
                        audio_path=result["audio_path"],
                        audio_validation=result["audio_validation"],
                        status=updated_status,
                        auto_regen_count=0,
                        generation_token=None if updated_status != "pending" else generation_token,
                    )
                    if updated_chunk is None:
                        self._cleanup_temp_file(temp_path)
                        continue
                    chunks = self.load_chunks()

                    if result["status"] == "done":
                        results["completed"].append(idx)
                        print(f"Chunk {idx} completed: {updated_chunk['audio_path']}")
                        if item_callback:
                            item_callback(idx, True, shared_elapsed, word_counts.get(idx, 0), word_counts.get(idx, 0))
                    elif auto_regen_retry_attempts > 0:
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

                    self._cleanup_temp_file(temp_path)

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

            if progress_callback:
                progress_callback(len(results["completed"]), len(results["failed"]), total)

        # Reset remaining "generating" chunks to "pending" on cancel or completion
        done_indices = set(results["completed"]) | {idx for idx, _ in results["failed"]}
        chunks = self.load_chunks()
        if chunks:
            for idx in indices:
                if idx not in done_indices and 0 <= idx < len(chunks) and chunks[idx].get("status") == "generating":
                    chunks[idx]["status"] = "pending"
                    chunks[idx].pop("generation_token", None)
                    results["cancelled"] += 1
            if results["cancelled"]:
                self.save_chunks(chunks)

        print(f"Batch generation complete: {len(results['completed'])} succeeded, "
              f"{len(results['failed'])} failed, {results['cancelled']} cancelled")
        return results
