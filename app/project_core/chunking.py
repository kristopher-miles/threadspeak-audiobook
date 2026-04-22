"""Script-entry to chunk conversion helpers.

These functions are used by project initialization and script/chunk sync paths
to normalize speaker/text entries into the chunk format consumed by generation
and editor workflows.
"""

import uuid

from .constants import CHAPTER_HEADING_RE, MAX_CHUNK_CHARS


def _coerce_bool(value, default=False):
    if isinstance(value, bool):
        return value
    if value is None:
        return bool(default)
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"1", "true", "yes", "on"}:
            return True
        if text in {"0", "false", "no", "off", ""}:
            return False
    return bool(value)


def get_speaker(entry):
    """Get speaker from entry, checking both 'speaker' and 'type' fields."""
    return entry.get("speaker") or entry.get("type") or ""


def _is_structural_text(text):
    """Check if text is a title, chapter heading, dedication, or other structural fragment."""
    stripped = text.strip()
    if not stripped:
        return True
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


def _build_chunk(speaker, text, instruct, chapter=None, paragraph_id=None, *, chunk_type=None, silence_duration_s=None):
    chunk = {
        "speaker": speaker,
        "text": text,
        "instruct": instruct,
        "uid": uuid.uuid4().hex,
    }
    if chapter:
        chunk["chapter"] = chapter
    if paragraph_id:
        chunk["paragraph_id"] = paragraph_id
    if chunk_type:
        chunk["type"] = chunk_type
    if silence_duration_s is not None:
        chunk["silence_duration_s"] = float(silence_duration_s)
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
    current_paragraph_id = script_entries[0].get("paragraph_id")

    for entry in script_entries[1:]:
        speaker = get_speaker(entry)
        text = entry.get("text", "")
        instruct = entry.get("instruct", "")
        entry_chapter = _extract_chapter_name(entry)
        effective_chapter = entry_chapter or current_chapter
        entry_paragraph_id = entry.get("paragraph_id")

        if (speaker == current_speaker and instruct == current_instruct
                and effective_chapter == current_chapter
                and not _is_structural_text(current_text)
                and not _is_structural_text(text)):
            combined = current_text + " " + text
            if len(combined) <= max_chars:
                current_text = combined
                current_paragraph_id = entry_paragraph_id or current_paragraph_id
            else:
                chunks.append(_build_chunk(current_speaker, current_text, current_instruct, current_chapter, current_paragraph_id))
                current_text = text
                current_instruct = instruct
                current_chapter = effective_chapter
                current_paragraph_id = entry_paragraph_id
        else:
            chunks.append(_build_chunk(current_speaker, current_text, current_instruct, current_chapter, current_paragraph_id))
            current_speaker = speaker
            current_text = text
            current_instruct = instruct
            current_chapter = effective_chapter
            current_paragraph_id = entry_paragraph_id

    chunks.append(_build_chunk(current_speaker, current_text, current_instruct, current_chapter, current_paragraph_id))

    return chunks


def script_entries_to_chunks(script_entries, max_chars=MAX_CHUNK_CHARS):
    """Build chunks from script entries.

    For sentence-level scripts produced by create_script.py (which include
    paragraph_id), preserve a strict 1:1 mapping between entries and chunks.
    Legacy scripts without paragraph_id keep the historical merge behavior.
    """
    if not script_entries:
        return []

    has_paragraph_ids = any(bool(entry.get("paragraph_id")) for entry in script_entries)
    if not has_paragraph_ids:
        return group_into_chunks(script_entries, max_chars=max_chars)

    chunks = []
    for entry in script_entries:
        chunks.append(
            _build_chunk(
                get_speaker(entry),
                entry.get("text", ""),
                entry.get("instruct", ""),
                _extract_chapter_name(entry),
                entry.get("paragraph_id"),
                chunk_type=entry.get("type"),
                silence_duration_s=entry.get("silence_duration_s"),
            )
        )
    return chunks
