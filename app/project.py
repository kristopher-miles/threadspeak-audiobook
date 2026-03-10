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

MAX_CHUNK_CHARS = 500
MAX_AUTO_REGENERATE_BAD_CLIP_ATTEMPTS = 1
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

    @staticmethod
    def _split_text_sentences(text):
        parts = re.split(r'(?<=[.!?])\s+', (text or "").strip())
        return [part.strip() for part in parts if part.strip()]

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
                    return json.load(f)
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
                chunk["status"] = "pending" # pending, generating, done, error
                chunk["audio_path"] = None
                chunk["audio_validation"] = None
                chunk["auto_regen_count"] = 0

            self.save_chunks(chunks)
            return chunks

        return []

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

    def insert_chunk(self, after_index):
        """Insert an empty chunk after the given index. Returns the new chunk list."""
        with self._chunks_lock:
            if not os.path.exists(self.chunks_path):
                return None
            with open(self.chunks_path, "r", encoding="utf-8") as f:
                chunks = json.load(f)
            if not (0 <= after_index < len(chunks)):
                return None

            # Copy speaker from the row we're splitting from
            source = chunks[after_index]
            new_chunk = {
                "id": after_index + 1,
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

    def delete_chunk(self, index):
        """Delete a chunk at the given index. Returns (deleted_chunk, updated_chunks) or None."""
        with self._chunks_lock:
            if not os.path.exists(self.chunks_path):
                return None
            with open(self.chunks_path, "r", encoding="utf-8") as f:
                chunks = json.load(f)
            if not (0 <= index < len(chunks)):
                return None
            if len(chunks) <= 1:
                return None  # don't allow deleting the last chunk

            deleted = chunks.pop(index)

            # Re-number all IDs
            for i, chunk in enumerate(chunks):
                chunk["id"] = i

            self._atomic_json_write(chunks, self.chunks_path)
            return deleted, chunks

    def restore_chunk(self, at_index, chunk_data):
        """Re-insert a chunk at a specific index. Returns the updated chunk list."""
        with self._chunks_lock:
            if not os.path.exists(self.chunks_path):
                return None
            with open(self.chunks_path, "r", encoding="utf-8") as f:
                chunks = json.load(f)

            at_index = max(0, min(at_index, len(chunks)))
            chunks.insert(at_index, chunk_data)

            # Re-number all IDs
            for i, chunk in enumerate(chunks):
                chunk["id"] = i

            self._atomic_json_write(chunks, self.chunks_path)
            return chunks

    def update_chunk(self, index, data):
        chunks = self.load_chunks()
        if 0 <= index < len(chunks):
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
            tts_settings = self._load_tts_settings()
            auto_regenerate_bad_clips = tts_settings.get("auto_regenerate_bad_clips", False)

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
                if result["status"] == "error" and auto_regenerate_bad_clips and attempt < MAX_AUTO_REGENERATE_BAD_CLIP_ATTEMPTS:
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
                    print(f"Chunk {index} failed sanity check; auto-regenerating attempt {attempt + 1}/{MAX_AUTO_REGENERATE_BAD_CLIP_ATTEMPTS}")
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

    def merge_audio(self):
        chunks = self.load_chunks()
        audio_segments = []
        speakers = []

        for chunk in chunks:
            if chunk.get("status") != "done":
                continue
            path = chunk.get("audio_path")
            if path:
                full_path = os.path.join(self.root_dir, path)
                if os.path.exists(full_path):
                    try:
                        # Auto-detect format (mp3 or wav)
                        segment = AudioSegment.from_file(full_path)
                        audio_segments.append(segment)
                        speakers.append(chunk["speaker"])
                    except Exception as e:
                        print(f"Error loading audio segment {path}: {e}")

        if not audio_segments:
            return False, "No audio segments found"

        final_audio = combine_audio_with_pauses(audio_segments, speakers)
        output_filename = "cloned_audiobook.mp3"
        output_path = os.path.join(self.root_dir, output_filename)
        final_audio.export(output_path, format="mp3")

        return True, output_filename

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
        tts_settings = self._load_tts_settings()
        auto_regenerate_bad_clips = tts_settings.get("auto_regenerate_bad_clips", False)

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
                    updated_status = "pending" if (result["status"] == "error" and auto_regenerate_bad_clips) else result["status"]
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
                    elif auto_regenerate_bad_clips:
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
