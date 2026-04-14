#!/usr/bin/env python3
"""
Create the audiobook script from processed paragraph data.

Reads persisted paragraph state (after dialogue assignment and temperament extraction)
and writes the resulting script/chunks into the project SQLite store.

No LLM calls are made.  This is a pure deterministic transformation.

CLI usage:
    create_script.py <paragraphs_path> <script_output_path> <chunks_output_path>
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

from stdio_utils import configure_utf8_stdio
from scripts.legacy_cli_project import infer_project_root, import_project_document_from_path
from script_provider import open_project_script_store

configure_utf8_stdio()

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


def _build_para_blocks(text: str, speakers_list: list, narration_speaker: str = "NARRATOR") -> list[dict]:
    """
    Decompose a mixed paragraph into ordered blocks, fusing same-speaker
    quoted spans into adjacent narration text.

    When a quote's assigned speaker matches narration_speaker, its text
    (WITH quotation marks) is folded into the surrounding narration so
    split_fn treats the whole run as one continuous chunk.  Quotes with a
    different speaker flush the current narration block and become their
    own block.

    Returns list of {"text": str, "speaker": str, "quote_index": int|None}.
    Narration/fused blocks have quote_index=None; dialogue blocks carry the
    quote_index needed to look up dialogue_moods.
    """
    blocks = []
    current_text = ""
    last_end = 0

    for qi, m in enumerate(QUOTE_RE.finditer(text)):
        current_text += text[last_end:m.start()]
        quote_speaker = speakers_list[qi] if qi < len(speakers_list) else narration_speaker

        if quote_speaker == narration_speaker:
            # Same speaker — fold quoted text (WITH marks) into the running block
            current_text += m.group(0)
        else:
            # Different speaker — flush accumulated narration, emit dialogue block
            if current_text.strip():
                blocks.append({"text": current_text.strip(), "speaker": narration_speaker, "quote_index": None})
            current_text = ""
            blocks.append({"text": m.group(0), "speaker": quote_speaker, "quote_index": qi})

        last_end = m.end()

    # Trailing narration after the last quote
    current_text += text[last_end:]
    if current_text.strip():
        blocks.append({"text": current_text.strip(), "speaker": narration_speaker, "quote_index": None})

    return blocks


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

    quote_match_idx = 0
    for m in QUOTE_RE.finditer(text):
        # Narration before this dialogue span
        narration = text[last_end:m.start()].strip()
        if narration:
            for s in split_fn(narration):
                segments.append({"text": s, "is_dialogue": False})

        # Dialogue span — split internally too; all sub-chunks share the same quote_index
        for s in split_fn(m.group(0).strip()):
            segments.append({"text": s, "is_dialogue": True, "quote_index": quote_match_idx})
        quote_match_idx += 1

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
        description="Build persisted script entries and chunks from persisted paragraphs."
    )
    parser.add_argument("paragraphs_path", nargs="?")
    parser.add_argument("script_output_path", nargs="?")
    parser.add_argument("chunks_output_path", nargs="?")
    parser.add_argument("--project-root", dest="project_root")
    parser.add_argument(
        "--max-length", type=int, default=100, dest="max_length",
        help="Max chars per audio chunk. -1 = one chunk per sentence (legacy).",
    )
    args = parser.parse_args()

    paragraphs_path    = args.paragraphs_path
    script_output_path = args.script_output_path
    chunks_output_path = args.chunks_output_path
    max_length         = args.max_length
    project_root       = str(args.project_root or "").strip() or infer_project_root(script_output_path, chunks_output_path, paragraphs_path)

    if not project_root and (not paragraphs_path or not script_output_path or not chunks_output_path):
        _log("Usage: create_script.py <paragraphs_path> <script_output_path> <chunks_output_path> OR --project-root <root>")
        sys.exit(1)

    if paragraphs_path and not args.project_root:
        _log(f"Legacy file-mode detected; importing paragraphs into project store at {project_root}")
        try:
            import_project_document_from_path(
                project_root,
                "paragraphs",
                paragraphs_path,
                reason="legacy_create_script_import",
            )
        except Exception as e:
            _log(f"ERROR: Could not import paragraphs file: {e}")
            sys.exit(1)

    # ── Load paragraphs ────────────────────────────────────────────────────────
    try:
        store = open_project_script_store(project_root)
        try:
            paragraphs_doc = store.load_project_document("paragraphs") or {}
        finally:
            store.stop()
    except Exception as e:
        _log(f"ERROR: Could not load paragraphs file: {e}")
        sys.exit(1)

    paragraphs = paragraphs_doc.get("paragraphs", [])
    if not paragraphs:
        _log("ERROR: persisted paragraph state contains no paragraphs.")
        sys.exit(1)

    # ── Collect all speakers and ensure NARRATOR is included ───────────────────
    raw_speakers: set[str] = set()
    for p in paragraphs:
        if p.get("has_dialogue"):
            for s in (p.get("speakers") or []):
                if s and s != "NARRATOR":
                    raw_speakers.add(s.strip())
    raw_speakers.add("NARRATOR")
    _log(f"Voices discovered in script: {len(raw_speakers)}")

    # ── Group paragraphs by chapter (preserving order) ─────────────────────────
    chapters: list[tuple[str, list[dict]]] = []
    for para in paragraphs:
        ch = para.get("chapter") or ""
        if not chapters or chapters[-1][0] != ch:
            chapters.append((ch, []))
        chapters[-1][1].append(para)

    _log(f"Building script for {len(paragraphs)} paragraphs across {len(chapters)} chapter(s)...")

    entries: list[dict] = []
    chunks:  list[dict] = []

    # Choose the splitting function based on max_length
    if max_length < 0:
        para_split_fn = split_sentences
    else:
        para_split_fn = lambda t: chunk_text(t, max_length)

    for chapter_idx, (chapter_name, paras) in enumerate(chapters):
        chapter_line_count = 0

        for para in paras:
            tone          = _fix_instruct((para.get("tone") or "").strip())
            speakers_list = para.get("speakers") or []
            moods_list    = [_fix_instruct((m or "").strip()) for m in (para.get("dialogue_moods") or [])]

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
                # Mixed paragraph — fuse same-speaker quoted spans into narration,
                # keep different-speaker quotes as their own blocks.
                for block in _build_para_blocks(para["text"], speakers_list):
                    for s in para_split_fn(block["text"]):
                        if block["speaker"] == "NARRATOR":
                            entries.append({
                                "speaker": "NARRATOR",
                                "text": s,          # quotes preserved for TTS emphasis
                                "instruct": tone,
                                "chapter": chapter_name,
                                "paragraph_id": para_id,
                            })
                        else:
                            qi = block["quote_index"]
                            instruct = moods_list[qi] if qi is not None and qi < len(moods_list) else tone
                            entries.append({
                                "speaker": block["speaker"],
                                "text": _strip_quotes(s),
                                "instruct": instruct,
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

    script_doc = {
        "entries": entries,
        "dictionary": [],
        "sanity_cache": {"phrase_decisions": {}},
    }
    store = open_project_script_store(project_root)
    try:
        store.replace_project_document(
            "paragraphs",
            {**paragraphs_doc, "create_script_complete": True},
            reason="create_script_paragraphs_marker",
            wait=True,
        )
        store.replace_script_document(
            entries=script_doc["entries"],
            dictionary=script_doc["dictionary"],
            sanity_cache=script_doc["sanity_cache"],
            reason="create_script",
            rebuild_chunks=True,
            wait=True,
        )
    finally:
        store.stop()

    _log(f"Done. Total lines: {len(entries)} across {len(chapters)} chapter(s).")


if __name__ == "__main__":
    main()
