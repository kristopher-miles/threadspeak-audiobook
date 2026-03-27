#!/usr/bin/env python3
"""
Create the audiobook script from processed paragraph data.

Reads paragraphs.json (after dialogue assignment and temperament extraction)
and produces:
  1. voice_config.json  — one entry per speaker + NARRATOR (new speakers only,
                          existing entries are never overwritten), alphabetically.
  2. annotated_script.json — one entry per sentence-level fragment, with speaker
                             and instruct derived from the paragraph data.
  3. chunks.json        — written immediately after annotated_script.json so its
                          mtime is newer, preventing sync_chunks_from_script_if_stale
                          from re-merging the carefully split lines via group_into_chunks.
                          Each script entry maps 1:1 to exactly one chunk.

No LLM calls are made.  This is a pure deterministic transformation.

CLI usage:
    create_script.py <paragraphs_path> <voice_config_path> <script_output_path> <chunks_output_path>
"""
import json
import os
import re
import sys
import uuid

TASK_PROGRESS_PREFIX = "__TASK_PROGRESS__:"

# Matches straight (") or curly (\u201c / \u201d) double-quotes enclosing ≥2 chars.
QUOTE_RE = re.compile(r'["\u201c][^"\u201d]{2,}["\u201d]', re.DOTALL)

# Split after sentence-ending punctuation followed by whitespace.
SENT_RE = re.compile(r'(?<=[.!?])\s+')

# Used to detect whether a fragment contains any letter characters.
HAS_LETTER = re.compile(r'[A-Za-z]')

# Opening and closing quotation marks (straight and curly) to strip from dialogue text.
_QUOTE_CHARS = '"\u201c\u201d'

_READ_RE = re.compile(r'\bread\b', re.IGNORECASE)


def _fix_instruct(instruct: str) -> str:
    """Replace the word 'read' with 'speak' in a delivery instruction."""
    def _replace(m: re.Match) -> str:
        word = m.group(0)
        return 'Speak' if word[0].isupper() else 'speak'
    return _READ_RE.sub(_replace, instruct)


def _strip_quotes(text: str) -> str:
    """Remove leading/trailing quotation marks from a dialogue fragment."""
    return text.strip(_QUOTE_CHARS)


def _log(msg: str):
    print(msg, flush=True)


def _progress(current: int, total: int, message: str):
    print(
        TASK_PROGRESS_PREFIX + json.dumps({"current": current, "total": total, "message": message}),
        flush=True,
    )


def _atomic_write(path: str, data) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


# ── Sentence splitting ─────────────────────────────────────────────────────────

def split_sentences(text: str) -> list[str]:
    """
    Split text on sentence boundaries (after . ! ?).

    Any fragment that contains no letters (e.g. a bare '...' or trailing
    punctuation only) is merged onto the preceding fragment rather than
    becoming its own line.  If there is no preceding fragment the letterless
    fragment is discarded.
    """
    raw = [s.strip() for s in SENT_RE.split(text) if s.strip()]
    result: list[str] = []
    for part in raw:
        if HAS_LETTER.search(part):
            result.append(part)
        else:
            if result:
                result[-1] = result[-1] + " " + part
            # else: discard — no preceding fragment to attach to
    return result


# ── Paragraph → ordered segment list ──────────────────────────────────────────

def paragraph_to_segments(para: dict) -> list[dict]:
    """
    Decompose a paragraph into an ordered list of sentence-level fragments,
    tagging each as dialogue or narration.

    Dialogue spans (quoted text) are extracted first; the remaining pieces
    are narration.  Both types are further split on sentence boundaries.
    Dialogue spanning multiple sentences produces multiple fragments that all
    share the same speaker and dialogue_mood.
    """
    text = para["text"]
    segments: list[dict] = []
    last_end = 0

    for m in QUOTE_RE.finditer(text):
        # Narration before this dialogue span
        narration = text[last_end:m.start()].strip()
        if narration:
            for s in split_sentences(narration):
                segments.append({"text": s, "is_dialogue": False})

        # Dialogue span — split internally too
        for s in split_sentences(m.group(0).strip()):
            segments.append({"text": s, "is_dialogue": True})

        last_end = m.end()

    # Trailing narration after the last dialogue span
    trailing = text[last_end:].strip()
    if trailing:
        for s in split_sentences(trailing):
            segments.append({"text": s, "is_dialogue": False})

    return segments


# ── Chunk builder ──────────────────────────────────────────────────────────────

def _make_chunk(idx: int, speaker: str, text: str, instruct: str, chapter: str, paragraph_id: str = None) -> dict:
    chunk = {
        "id": idx,
        "uid": uuid.uuid4().hex,
        "speaker": speaker,
        "text": text,
        "instruct": instruct,
        "status": "pending",
        "audio_path": None,
        "audio_validation": None,
        "auto_regen_count": 0,
    }
    if chapter:
        chunk["chapter"] = chapter
    if paragraph_id:
        chunk["paragraph_id"] = paragraph_id
    return chunk


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 5:
        _log("Usage: create_script.py <paragraphs_path> <voice_config_path> "
             "<script_output_path> <chunks_output_path>")
        sys.exit(1)

    paragraphs_path    = sys.argv[1]
    voice_config_path  = sys.argv[2]
    script_output_path = sys.argv[3]
    chunks_output_path = sys.argv[4]

    # ── Load paragraphs ────────────────────────────────────────────────────────
    try:
        with open(paragraphs_path, "r", encoding="utf-8") as f:
            paragraphs_doc = json.load(f)
    except Exception as e:
        _log(f"ERROR: Could not load paragraphs file: {e}")
        sys.exit(1)

    paragraphs = paragraphs_doc.get("paragraphs", [])
    if not paragraphs:
        _log("ERROR: paragraphs.json contains no paragraphs.")
        sys.exit(1)

    # ── Load existing voice config (create if absent) ──────────────────────────
    voice_config: dict = {}
    if os.path.exists(voice_config_path):
        try:
            with open(voice_config_path, "r", encoding="utf-8") as f:
                voice_config = json.load(f)
        except Exception as e:
            _log(f"WARNING: Could not read voice_config.json ({e}). Starting fresh.")

    # ── Collect all speakers and ensure NARRATOR is included ───────────────────
    raw_speakers: set[str] = set()
    for p in paragraphs:
        if p.get("has_dialogue") and p.get("speaker"):
            raw_speakers.add(p["speaker"].strip())
    raw_speakers.add("NARRATOR")

    all_voices = sorted(raw_speakers)  # alphabetical

    new_count = 0
    for name in all_voices:
        if name not in voice_config:
            voice_config[name] = {
                "type": "design",
                "description": "",
                "ref_text": "",
                "alias": "",
                "seed": "-1",
            }
            new_count += 1

    _atomic_write(voice_config_path, voice_config)
    _log(f"Voices: created {new_count} new entries ({len(voice_config)} total).")

    # ── Group paragraphs by chapter (preserving order) ─────────────────────────
    chapters: list[tuple[str, list[dict]]] = []
    for para in paragraphs:
        ch = para.get("chapter") or ""
        if not chapters or chapters[-1][0] != ch:
            chapters.append((ch, []))
        chapters[-1][1].append(para)

    _log(f"Building script for {len(paragraphs)} paragraphs across {len(chapters)} chapter(s)...")

    entries: list[dict] = []   # for annotated_script.json
    chunks:  list[dict] = []   # for chunks.json — 1:1 with entries, no merging

    for chapter_idx, (chapter_name, paras) in enumerate(chapters):
        chapter_line_count = 0

        for para in paras:
            speaker       = (para.get("speaker") or "").strip()
            tone          = _fix_instruct((para.get("tone") or "").strip())
            dialogue_mood = _fix_instruct((para.get("dialogue_mood") or "").strip())

            para_id = para.get("id")
            if not para.get("has_dialogue"):
                # Pure narration — every sentence is NARRATOR
                for s in split_sentences(para["text"]):
                    entries.append({
                        "speaker": "NARRATOR",
                        "text": s,
                        "instruct": tone,
                        "chapter": chapter_name,
                        "paragraph_id": para_id,
                    })
                    chapter_line_count += 1
            else:
                # Mixed paragraph — preserve narration/dialogue order
                for seg in paragraph_to_segments(para):
                    if seg["is_dialogue"]:
                        entries.append({
                            "speaker": speaker or "NARRATOR",
                            "text": _strip_quotes(seg["text"]),
                            "instruct": dialogue_mood,
                            "chapter": chapter_name,
                            "paragraph_id": para_id,
                        })
                    else:
                        entries.append({
                            "speaker": "NARRATOR",
                            "text": seg["text"],
                            "instruct": tone,
                            "chapter": chapter_name,
                            "paragraph_id": para_id,
                        })
                    chapter_line_count += 1

        chapter_label = chapter_name or "(no chapter)"
        _log(f"Chapter '{chapter_label}': {chapter_line_count} lines")
        _progress(chapter_idx + 1, len(chapters),
                  f"Chapter '{chapter_label}' complete ({chapter_line_count} lines)")

    # ── Build chunks list (1:1 with entries, no merging) ──────────────────────
    for i, entry in enumerate(entries):
        chunks.append(_make_chunk(
            idx=i,
            speaker=entry["speaker"],
            text=entry["text"],
            instruct=entry["instruct"],
            chapter=entry.get("chapter", ""),
            paragraph_id=entry.get("paragraph_id"),
        ))

    # ── Write annotated_script.json first, then chunks.json ───────────────────
    # chunks.json is written LAST so its mtime is newer than annotated_script.json.
    # This prevents sync_chunks_from_script_if_stale from re-running group_into_chunks
    # and merging the carefully split lines back together.
    script_doc = {
        "entries": entries,
        "dictionary": [],
        "sanity_cache": {"phrase_decisions": {}},
    }
    _atomic_write(script_output_path, script_doc)
    _atomic_write(chunks_output_path, chunks)

    _log(f"Done. Total lines: {len(entries)} across {len(chapters)} chapter(s).")


if __name__ == "__main__":
    main()
