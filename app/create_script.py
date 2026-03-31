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
                     [--max-length N]

--max-length N  Max characters per audio chunk (default 100).
                -1 = legacy mode: split on every sentence boundary.
"""
import argparse
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

# Matches 2+ consecutive dots (ellipsis or longer run).
_ELLIPSIS_RE = re.compile(r'\.\.+')

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


def split_sentences_new(text: str) -> list[str]:
    """
    Like split_sentences() but treats sequences of 2+ dots ('...') as NOT a
    sentence boundary. Period, ! and ? are still valid sentence-end markers.
    """
    ellipses: list[str] = []

    def _protect(m: re.Match) -> str:
        ellipses.append(m.group())
        return f'\x00{len(ellipses) - 1}\x00'

    protected = _ELLIPSIS_RE.sub(_protect, text)
    raw = [s.strip() for s in SENT_RE.split(protected) if s.strip()]
    result: list[str] = []
    for part in raw:
        restored = re.sub(r'\x00(\d+)\x00', lambda m: ellipses[int(m.group(1))], part)
        if HAS_LETTER.search(restored):
            result.append(restored)
        else:
            if result:
                result[-1] = result[-1] + " " + restored
    return result


def _split_on_comma(text: str) -> list[str]:
    """Last-resort: split on commas. Comma stays with its preceding text."""
    parts = re.split(r',\s+', text)
    if len(parts) <= 1:
        return [text]
    result = [p + ',' for p in parts[:-1]] + [parts[-1]]
    return [r.strip() for r in result if r.strip()]


def _balanced_split(parts: list[str], max_length: int) -> list[str]:
    """
    Recursively split a list of consecutive text parts into chunks, choosing
    the split point that minimises |len(left) - len(right)|.
    Stops recursing when a chunk fits within max_length or has only one part.
    """
    joined = ' '.join(parts)
    if len(joined) <= max_length or len(parts) == 1:
        return [joined]

    best_idx = 1
    best_diff = float('inf')
    for i in range(1, len(parts)):
        diff = abs(len(' '.join(parts[:i])) - len(' '.join(parts[i:])))
        if diff < best_diff:
            best_diff = diff
            best_idx = i

    return (
        _balanced_split(parts[:best_idx], max_length) +
        _balanced_split(parts[best_idx:], max_length)
    )


def chunk_text(text: str, max_length: int) -> list[str]:
    """
    Split text into chunks as close as possible to max_length characters.
    - Never splits mid-sentence.
    - '...' (2+ consecutive dots) is not treated as a sentence boundary.
    - Period, ! and ? are valid sentence-end markers.
    - Falls back to comma splits only when no sentence boundaries exist.
    - Returns [text] unchanged if no valid split point exists at all.
    """
    if len(text) <= max_length:
        return [text]

    parts = split_sentences_new(text)

    if len(parts) <= 1:
        # No sentence boundaries — try comma as absolute last resort
        parts = _split_on_comma(text)
        if len(parts) <= 1:
            return [text]

    return _balanced_split(parts, max_length)


# ── Paragraph → ordered segment list ──────────────────────────────────────────

def paragraph_to_segments(para: dict, max_length: int = -1) -> list[dict]:
    """
    Decompose a paragraph into an ordered list of fragments, tagging each as
    dialogue or narration.

    When max_length < 0 (legacy): splits on every sentence boundary.
    When max_length >= 0: consolidates sentences into chunks up to max_length,
    splitting into most-equal pieces only when necessary.
    """
    split_fn = split_sentences if max_length < 0 else (lambda t: chunk_text(t, max_length))

    text = para["text"]
    segments: list[dict] = []
    last_end = 0

    for m in QUOTE_RE.finditer(text):
        # Narration before this dialogue span
        narration = text[last_end:m.start()].strip()
        if narration:
            for s in split_fn(narration):
                segments.append({"text": s, "is_dialogue": False})

        # Dialogue span — split internally too
        for s in split_fn(m.group(0).strip()):
            segments.append({"text": s, "is_dialogue": True})

        last_end = m.end()

    # Trailing narration after the last dialogue span
    trailing = text[last_end:].strip()
    if trailing:
        for s in split_fn(trailing):
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
    parser = argparse.ArgumentParser(
        description="Build annotated_script.json and chunks.json from paragraphs.json."
    )
    parser.add_argument("paragraphs_path")
    parser.add_argument("voice_config_path")
    parser.add_argument("script_output_path")
    parser.add_argument("chunks_output_path")
    parser.add_argument(
        "--max-length", type=int, default=100, dest="max_length",
        help="Max chars per audio chunk. -1 = one chunk per sentence (legacy).",
    )
    args = parser.parse_args()

    paragraphs_path    = args.paragraphs_path
    voice_config_path  = args.voice_config_path
    script_output_path = args.script_output_path
    chunks_output_path = args.chunks_output_path
    max_length         = args.max_length

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

    # Choose the splitting function based on max_length
    if max_length < 0:
        para_split_fn = split_sentences
    else:
        para_split_fn = lambda t: chunk_text(t, max_length)

    for chapter_idx, (chapter_name, paras) in enumerate(chapters):
        chapter_line_count = 0

        for para in paras:
            speaker       = (para.get("speaker") or "").strip()
            tone          = _fix_instruct((para.get("tone") or "").strip())
            dialogue_mood = _fix_instruct((para.get("dialogue_mood") or "").strip())

            para_id = para.get("id")
            if not para.get("has_dialogue"):
                # Pure narration — split according to max_length
                for s in para_split_fn(para["text"]):
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
                for seg in paragraph_to_segments(para, max_length=max_length):
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
