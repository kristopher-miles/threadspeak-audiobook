#!/usr/bin/env python3
"""
Process source document into annotated paragraphs.

Each paragraph gets a stable ID (p_0001, p_0002, …), a has_dialogue flag
(heuristic: contains paired quotation marks), and a blank tone field.
Output is written to paragraphs.json.
"""
import json
import os
import re
import sys
import time

from source_document import load_source_document, iter_document_paragraphs

TASK_PROGRESS_PREFIX = "__TASK_PROGRESS__:"

# Matches straight (") or curly (\u201c/\u201d) double-quotes enclosing ≥2 chars.
# re.DOTALL lets . span newlines for multi-line quoted passages.
DIALOGUE_RE = re.compile(
    r'["\u201c][^"\u201d]{2,}["\u201d]',
    re.DOTALL,
)


def _log(msg: str):
    print(msg, flush=True)


def _progress(current: int, total: int, message: str):
    print(
        TASK_PROGRESS_PREFIX + json.dumps({"current": current, "total": total, "message": message}),
        flush=True,
    )


def main():
    if len(sys.argv) < 3:
        _log("Usage: process_paragraphs.py <input_file> <output_path>")
        sys.exit(1)

    input_file = sys.argv[1]
    output_path = sys.argv[2]

    _log(f"Loading: {os.path.basename(input_file)}")

    try:
        doc = load_source_document(input_file)
    except Exception as e:
        _log(f"ERROR: Could not load source document: {e}")
        sys.exit(1)

    title = doc.get("book_title") or doc.get("title") or "Unknown"
    chapters = doc.get("chapters") or []
    _log(f"Book: {title}  |  Chapters: {len(chapters)}")

    all_paras = list(iter_document_paragraphs(doc))
    total = len(all_paras)
    if total == 0:
        _log("ERROR: No paragraphs found in source document.")
        sys.exit(1)

    _log(f"Identified {total} paragraphs. Analysing...")

    paragraphs = []
    dialogue_count = 0

    for i, para in enumerate(all_paras):
        text = para["text"]
        has_dlg = bool(DIALOGUE_RE.search(text))
        if has_dlg:
            dialogue_count += 1
        paragraphs.append({
            "id": f"p_{i + 1:04d}",
            "chapter": para.get("chapter") or "",
            "text": text,
            "has_dialogue": has_dlg,
            "tone": "",
        })
        if (i + 1) % 100 == 0 or i + 1 == total:
            _progress(i + 1, total, f"Processed {i + 1}/{total} paragraphs...")

    _log(f"Done. Total: {total}  |  With dialogue: {dialogue_count}  |  Narration only: {total - dialogue_count}")

    result = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "source_file": os.path.basename(input_file),
        "paragraph_count": total,
        "dialogue_count": dialogue_count,
        "paragraphs": paragraphs,
    }

    tmp = output_path + ".tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        os.replace(tmp, output_path)
    except Exception as e:
        _log(f"ERROR: Could not write output file: {e}")
        sys.exit(1)

    _log(f"Saved {total} paragraphs -> {os.path.basename(output_path)}")


if __name__ == "__main__":
    main()
