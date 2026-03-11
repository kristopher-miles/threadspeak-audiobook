import copy
import json
import os
import re

from openai import OpenAI

from generate_script import process_chunk
from script_sanity import run_script_sanity_check
from script_store import load_script_document, save_script_document
from source_document import load_source_document


_WORD_RE = re.compile(r"[A-Za-z]+")
_QUOTE_RE = re.compile(r'["“”]')


def _count_words(text):
    return len(_WORD_RE.findall(text or ""))


def _tokenize_with_positions(text):
    tokens = []
    for match in _WORD_RE.finditer(text or ""):
        tokens.append({
            "word": match.group(0).lower(),
            "start": match.start(),
            "end": match.end(),
        })
    return tokens


def _normalize_title(value):
    return (value or "").strip().lower()


def _is_structural_segment(text):
    stripped = (text or "").strip()
    if not stripped:
        return False
    if len(stripped) > 160:
        return False
    lowered = stripped.lower()
    if lowered.startswith(("chapter ", "part ", "book ", "volume ", "prologue", "epilogue", "title")):
        return True
    if stripped.startswith("...") and stripped.endswith("..."):
        return True
    return False


def _group_entries_by_chapter(entries):
    groups = []
    current = None
    word_cursor = 0
    char_cursor = 0

    for entry_index, entry in enumerate(entries or []):
        chapter = (entry.get("chapter") or "").strip()
        text = (entry.get("text") or "").strip()

        if current is None or current["title"] != chapter:
            current = {
                "index": len(groups) + 1,
                "title": chapter,
                "entries": [],
                "parts": [],
            }
            groups.append(current)
            word_cursor = 0
            char_cursor = 0

        separator = "\n\n" if current["parts"] and text else ""
        char_start = char_cursor + len(separator) if text else char_cursor
        char_end = char_start + len(text) if text else char_start
        if text:
            current["parts"].append(text)
            char_cursor = char_end

        word_count = _count_words(text)
        current["entries"].append({
            "entry_index": entry_index,
            "entry": copy.deepcopy(entry),
            "word_start": word_cursor,
            "word_end": word_cursor + word_count,
            "char_start": char_start,
            "char_end": char_end,
        })
        word_cursor += word_count

    for group in groups:
        group["text"] = "\n\n".join(group["parts"]).strip()
        group["word_count"] = group["entries"][-1]["word_end"] if group["entries"] else 0

    return groups


def _find_source_chapter(source_document, title, index):
    chapters = source_document.get("chapters") or []
    normalized = _normalize_title(title)
    if normalized:
        for chapter_index, chapter in enumerate(chapters, start=1):
            if _normalize_title(chapter.get("title")) == normalized:
                return {
                    "index": chapter_index,
                    "title": (chapter.get("title") or "").strip(),
                    "text": chapter.get("text") or "",
                }
    if index and 1 <= index <= len(chapters):
        chapter = chapters[index - 1]
        return {
            "index": index,
            "title": (chapter.get("title") or "").strip(),
            "text": chapter.get("text") or "",
        }
    return None


def _find_script_group(groups, title):
    normalized = _normalize_title(title)
    for group in groups:
        if _normalize_title(group["title"]) == normalized:
            return group
    return None


def _slice_text_by_word_span(text, start_word, end_word):
    tokens = _tokenize_with_positions(text)
    if not tokens or end_word <= start_word:
        return ""

    start_word = max(0, start_word)
    end_word = min(len(tokens), end_word)
    if end_word <= start_word:
        return ""

    start_char = tokens[start_word]["start"]
    end_char = tokens[end_word - 1]["end"]
    return (text or "")[start_char:end_char].strip()


def _build_literal_replacement_entries(chapter_title, source_text):
    text = (source_text or "").strip()
    if not text:
        return []
    return [{
        "chapter": chapter_title or "",
        "speaker": "NARRATOR",
        "text": text,
        "instruct": "Neutral, clear announcement." if _is_structural_segment(text) else "",
    }]


def _build_narrator_replacement_entries(chapter_title, source_text):
    text = (source_text or "").strip()
    if not text:
        return []
    return [{
        "chapter": chapter_title or "",
        "speaker": "NARRATOR",
        "text": text,
        "instruct": "Neutral, clear announcement." if _is_structural_segment(text) else "Neutral, even narration.",
    }]


def _slice_text_by_char_span(text, start_char, end_char):
    text = text or ""
    start_char = max(0, min(len(text), int(start_char or 0)))
    end_char = max(start_char, min(len(text), int(end_char or start_char)))
    return text[start_char:end_char].strip()


def _span_is_inside_dialogue(chapter_text, start_char, end_char):
    chapter_text = chapter_text or ""
    start_char = max(0, min(len(chapter_text), int(start_char or 0)))
    end_char = max(start_char, min(len(chapter_text), int(end_char or start_char)))
    before_start = len(_QUOTE_RE.findall(chapter_text[:start_char]))
    before_end = len(_QUOTE_RE.findall(chapter_text[:end_char]))
    span_text = chapter_text[start_char:end_char]
    if _QUOTE_RE.search(span_text):
        return True
    return (before_start % 2 == 1) or (before_end % 2 == 1)


def _extract_entries_word_span(entries, start_word, end_word):
    extracted = []
    for item in entries:
        entry_start = item["word_start"]
        entry_end = item["word_end"]
        if entry_end <= start_word or entry_start >= end_word:
            continue

        entry = copy.deepcopy(item["entry"])
        entry_word_count = max(0, entry_end - entry_start)
        if entry_word_count == 0:
            continue

        relative_start = max(0, start_word - entry_start)
        relative_end = min(entry_word_count, end_word - entry_start)
        trimmed_text = _slice_text_by_word_span(entry.get("text") or "", relative_start, relative_end)
        if not trimmed_text:
            continue

        entry["text"] = trimmed_text
        extracted.append(entry)

    return extracted


def _word_index_for_char(tokens, char_position, prefer_end=False):
    if not tokens:
        return 0
    if prefer_end:
        for index, token in enumerate(tokens):
            if token["start"] >= char_position:
                return index
        return len(tokens)

    for index, token in enumerate(tokens):
        if token["end"] > char_position:
            return index
    return len(tokens)


def _expand_left_to_boundary(text, start):
    start = max(0, start)
    window = text[max(0, start - 160):start]
    last_break = max(window.rfind("\n\n"), window.rfind(". "), window.rfind("! "), window.rfind("? "))
    if last_break == -1:
        space = window.rfind(" ")
        if space == -1:
            return start
        return max(0, start - (len(window) - space - 1))
    return max(0, start - (len(window) - last_break - 1))


def _expand_right_to_boundary(text, end):
    end = min(len(text), end)
    window = text[end:min(len(text), end + 160)]
    candidates = [pos for pos in (window.find("\n\n"), window.find(". "), window.find("! "), window.find("? ")) if pos != -1]
    if not candidates:
        space = window.find(" ")
        if space == -1:
            return end
        return min(len(text), end + space)
    return min(len(text), end + min(candidates) + 1)


def _find_clean_start(text, start, latest_start):
    window_start = max(0, start - 240)
    window = text[window_start:latest_start]
    candidates = [window.rfind(marker) for marker in ("\n\n", ". ", "! ", "? ")]
    best = max(candidates)
    if best != -1:
        return window_start + best + 1
    space = window.rfind(" ")
    if space != -1:
        return window_start + space + 1
    return start


def _find_clean_end(text, earliest_end, end_limit):
    window = text[earliest_end:min(len(text), end_limit + 240)]
    candidates = [pos for pos in (window.find("\n\n"), window.find(". "), window.find("! "), window.find("? ")) if pos != -1]
    if candidates:
        return earliest_end + min(candidates) + 1
    space = window.find(" ")
    if space != -1:
        return earliest_end + space
    return min(len(text), end_limit)


def _build_centered_excerpt(chapter_text, target_start, target_end, chunk_size):
    chapter_text = chapter_text or ""
    text_length = len(chapter_text)
    target_start = max(0, min(target_start, text_length))
    target_end = max(target_start, min(target_end, text_length))

    if text_length <= chunk_size:
        return 0, text_length, chapter_text[:chunk_size]

    target_length = max(1, target_end - target_start)
    if target_length >= chunk_size:
        midpoint = (target_start + target_end) // 2
        start = max(0, midpoint - (chunk_size // 2))
        end = min(text_length, start + chunk_size)
        start = max(0, end - chunk_size)
    else:
        padding = chunk_size - target_length
        start = max(0, target_start - padding // 2)
        end = min(text_length, start + chunk_size)
        start = max(0, end - chunk_size)

    clean_start = _find_clean_start(chapter_text, start, target_start)
    if clean_start < start and (end - clean_start) > chunk_size:
        clean_start = start
    start = min(clean_start, target_start)

    end = min(text_length, start + chunk_size)
    clean_end = _find_clean_end(chapter_text, target_end, end)
    if clean_end > end:
        clean_end = end
    if clean_end <= target_end:
        clean_end = end
    end = clean_end

    if (end - start) < int(chunk_size * 0.8):
        end = min(text_length, start + chunk_size)
        start = max(0, end - chunk_size)

    return start, end, chapter_text[start:end].strip()


def _load_generation_settings(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    llm = config.get("llm", {})
    generation = config.get("generation", {})
    prompts = config.get("prompts", {})

    return {
        "base_url": llm.get("base_url", "http://localhost:11434/v1"),
        "api_key": llm.get("api_key", "local"),
        "model_name": llm.get("model_name", "local-model"),
        "timeout": float(llm.get("timeout", 600)),
        "chunk_size": int(generation.get("chunk_size", 3000)),
        "max_tokens": int(generation.get("max_tokens", 4096)),
        "temperature": float(generation.get("temperature", 0.6)),
        "top_p": float(generation.get("top_p", 0.8)),
        "top_k": int(generation.get("top_k", 20)),
        "min_p": float(generation.get("min_p", 0)),
        "presence_penalty": float(generation.get("presence_penalty", 0.0)),
        "banned_tokens": generation.get("banned_tokens", []) or [],
        "system_prompt": prompts.get("system_prompt"),
        "user_prompt": prompts.get("user_prompt"),
        "orphaned_text_to_narrator_on_repair": generation.get("orphaned_text_to_narrator_on_repair", True),
    }


def _target_signature(chapter_result, replacement_chunk):
    return (
        chapter_result.get("kind"),
        chapter_result.get("source_title") or "",
        chapter_result.get("script_title") or "",
        replacement_chunk.get("chapter_title") or "",
        int(replacement_chunk.get("source_char_start") or 0),
        int(replacement_chunk.get("source_char_end") or 0),
        int(replacement_chunk.get("script_char_start") or 0),
        int(replacement_chunk.get("script_char_end") or 0),
        int(replacement_chunk.get("missing_words") or 0),
        int(replacement_chunk.get("inserted_words") or 0),
    )


def _pick_next_target(sanity_result, failed_targets):
    for chapter in sanity_result.get("chapters") or []:
        for invalid_section in chapter.get("invalid_sections") or []:
            signature = _target_signature(chapter, invalid_section)
            if signature not in failed_targets:
                return chapter, invalid_section, signature
    return None, None, None


def _validate_generated_excerpt(chapter_title, excerpt_text, generated_entries, chunk_size):
    source_document = {
        "type": "text",
        "title": chapter_title or "excerpt",
        "chapters": [{
            "title": chapter_title or "",
            "text": excerpt_text,
        }],
    }
    script_document = {
        "entries": [dict(entry, chapter=chapter_title or "") for entry in generated_entries],
        "dictionary": [],
    }
    return run_script_sanity_check(source_document, script_document, chunk_size)


def _splice_replacement(entries, chapter_group, replacement_chunk, replacement_entries):
    prefix = _extract_entries_word_span(
        chapter_group["entries"],
        0,
        int(replacement_chunk.get("script_word_start") or 0),
    )
    suffix = _extract_entries_word_span(
        chapter_group["entries"],
        int(replacement_chunk.get("script_word_end") or 0),
        chapter_group["word_count"],
    )

    return (
        copy.deepcopy(entries[:chapter_group["entries"][0]["entry_index"]])
        + prefix
        + replacement_entries
        + suffix
        + copy.deepcopy(entries[chapter_group["entries"][-1]["entry_index"] + 1:])
    )


def repair_invalid_chunks(root_dir, log):
    state_path = os.path.join(root_dir, "state.json")
    script_path = os.path.join(root_dir, "annotated_script.json")
    chunks_path = os.path.join(root_dir, "chunks.json")
    config_path = os.path.join(root_dir, "app", "config.json")

    if not os.path.exists(state_path):
        raise FileNotFoundError("No source file selected.")
    if not os.path.exists(script_path):
        raise FileNotFoundError("No annotated script found. Generate a script first.")

    with open(state_path, "r", encoding="utf-8") as f:
        input_file = json.load(f).get("input_file_path")
    if not input_file or not os.path.exists(input_file):
        raise FileNotFoundError("Original uploaded source could not be found.")

    settings = _load_generation_settings(config_path)
    client = OpenAI(
        base_url=settings["base_url"],
        api_key=settings["api_key"],
        timeout=settings["timeout"],
    )
    source_document = load_source_document(input_file)

    repaired_targets = 0
    attempted_targets = set()
    failed_targets = set()
    initial_result = None

    while True:
        script_document = load_script_document(script_path)
        sanity_cache = script_document.get("sanity_cache") or {}
        sanity_result = run_script_sanity_check(
            source_document,
            script_document,
            settings["chunk_size"],
            known_phrase_decisions=sanity_cache.get("phrase_decisions"),
        )
        if initial_result is None:
            initial_result = sanity_result

        if sanity_result["invalid_chunk_count"] == 0:
            return {
                "initial_invalid_chunks": initial_result["invalid_chunk_count"],
                "initial_missing_words": initial_result["missing_words"],
                "initial_inserted_words": initial_result["inserted_words"],
                "repaired_targets": repaired_targets,
                "failed_targets": len(failed_targets),
                "final_sanity": sanity_result,
            }

        chapter_result, replacement_chunk, signature = _pick_next_target(sanity_result, attempted_targets)
        if chapter_result is None:
            return {
                "initial_invalid_chunks": initial_result["invalid_chunk_count"],
                "initial_missing_words": initial_result["missing_words"],
                "initial_inserted_words": initial_result["inserted_words"],
                "repaired_targets": repaired_targets,
                "failed_targets": len(failed_targets),
                "final_sanity": sanity_result,
            }
        attempted_targets.add(signature)

        chapter_title = chapter_result.get("source_title") or chapter_result.get("chapter_title") or chapter_result.get("script_title") or ""
        log(
            f'Repairing "{chapter_title or "untitled"}": '
            f'missing_words={replacement_chunk["missing_words"]}, '
            f'inserted_words={replacement_chunk["inserted_words"]}'
        )

        entries = script_document["entries"]
        chapter_groups = _group_entries_by_chapter(entries)

        if replacement_chunk["missing_words"] == 0:
            script_group = _find_script_group(chapter_groups, chapter_result.get("script_title") or chapter_result.get("chapter_title"))
            if script_group is None:
                failed_targets.add(signature)
                log(f'Could not locate script chapter for "{chapter_title}".')
                continue

            updated_entries = _splice_replacement(entries, script_group, replacement_chunk, [])
            save_script_document(
                script_path,
                entries=updated_entries,
                dictionary=script_document.get("dictionary", []),
                sanity_cache=sanity_cache,
            )
            if os.path.exists(chunks_path):
                os.remove(chunks_path)
            repaired_targets += 1
            log(f'Removed inserted text from "{chapter_title}".')
            continue

        source_chapter = _find_source_chapter(source_document, chapter_result.get("source_title") or chapter_title, chapter_result.get("chapter_index"))
        script_group = _find_script_group(chapter_groups, chapter_result.get("script_title") or chapter_title)
        if source_chapter is None or script_group is None:
            failed_targets.add(signature)
            log(f'Could not locate source/script chapter pair for "{chapter_title}".')
            continue

        excerpt_start, excerpt_end, excerpt_text = _build_centered_excerpt(
            source_chapter["text"],
            int(replacement_chunk["source_char_start"]),
            int(replacement_chunk["source_char_end"]),
            settings["chunk_size"],
        )
        if not excerpt_text:
            failed_targets.add(signature)
            log(f'Failed to build source excerpt for "{chapter_title}".')
            continue

        source_tokens = _tokenize_with_positions(source_chapter["text"])
        excerpt_word_start = _word_index_for_char(source_tokens, excerpt_start, prefer_end=False)
        excerpt_word_end = _word_index_for_char(source_tokens, excerpt_end, prefer_end=True)
        target_relative_word_start = max(0, int(replacement_chunk["source_word_start"]) - excerpt_word_start)
        target_relative_word_end = max(target_relative_word_start, int(replacement_chunk["source_word_end"]) - excerpt_word_start)
        target_source_text = _slice_text_by_char_span(
            source_chapter["text"],
            int(replacement_chunk["source_char_start"]),
            int(replacement_chunk["source_char_end"]),
        )
        target_is_inside_dialogue = _span_is_inside_dialogue(
            source_chapter["text"],
            int(replacement_chunk["source_char_start"]),
            int(replacement_chunk["source_char_end"]),
        )

        if int(replacement_chunk["source_word_start"]) == 0 and _is_structural_segment(target_source_text):
            replacement_entries = _build_literal_replacement_entries(source_chapter["title"], target_source_text)
            updated_entries = _splice_replacement(entries, script_group, replacement_chunk, replacement_entries)
            save_script_document(
                script_path,
                entries=updated_entries,
                dictionary=script_document.get("dictionary", []),
                sanity_cache=sanity_cache,
            )
            if os.path.exists(chunks_path):
                os.remove(chunks_path)
            repaired_targets += 1
            log(f'Patched structural span in "{chapter_title}" without LLM regeneration.')
            continue

        if settings.get("orphaned_text_to_narrator_on_repair", True) and not target_is_inside_dialogue:
            replacement_entries = _build_narrator_replacement_entries(source_chapter["title"], target_source_text)
            updated_entries = _splice_replacement(entries, script_group, replacement_chunk, replacement_entries)
            save_script_document(
                script_path,
                entries=updated_entries,
                dictionary=script_document.get("dictionary", []),
                sanity_cache=sanity_cache,
            )
            if os.path.exists(chunks_path):
                os.remove(chunks_path)
            repaired_targets += 1
            log(f'Patched orphaned non-dialogue text in "{chapter_title}" as NARRATOR.')
            continue

        context_entries = copy.deepcopy(entries[:script_group["entries"][0]["entry_index"]])
        context_entries.extend(
            _extract_entries_word_span(
                script_group["entries"],
                0,
                int(replacement_chunk.get("script_word_start") or 0),
            )
        )

        generated_entries = process_chunk(
            client=client,
            model_name=settings["model_name"],
            chunk={
                "text": excerpt_text,
                "chapter": source_chapter["title"],
                "chunk_index": 1,
                "chapter_chunk_count": 1,
            },
            chunk_num=1,
            total_chunks=1,
            previous_entries=context_entries,
            system_prompt=settings["system_prompt"],
            user_prompt_template=settings["user_prompt"],
            max_tokens=settings["max_tokens"],
            temperature=settings["temperature"],
            top_p=settings["top_p"],
            top_k=settings["top_k"],
            min_p=settings["min_p"],
            presence_penalty=settings["presence_penalty"],
            banned_tokens=settings["banned_tokens"],
        )
        for entry in generated_entries:
            entry["chapter"] = source_chapter["title"]

        validation = _validate_generated_excerpt(source_chapter["title"], excerpt_text, generated_entries, settings["chunk_size"])
        validated = validation["missing_words"] == 0 and validation["inserted_words"] == 0
        if not validated:
            log(
                f'Validation failed for "{chapter_title}": '
                f'missing_words={validation["missing_words"]}, inserted_words={validation["inserted_words"]}'
            )

        if not validated:
            replacement_entries = _build_literal_replacement_entries(source_chapter["title"], target_source_text)
            updated_entries = _splice_replacement(entries, script_group, replacement_chunk, replacement_entries)
            save_script_document(
                script_path,
                entries=updated_entries,
                dictionary=script_document.get("dictionary", []),
                sanity_cache=sanity_cache,
            )
            if os.path.exists(chunks_path):
                os.remove(chunks_path)
            repaired_targets += 1
            log(f'Fell back to literal source patch for "{chapter_title}" after validation failures.')
            continue

        generated_group = _group_entries_by_chapter(generated_entries)[0]
        replacement_entries = _extract_entries_word_span(
            generated_group["entries"],
            target_relative_word_start,
            target_relative_word_end,
        )
        if not replacement_entries and replacement_chunk["missing_words"] > 0:
            failed_targets.add(signature)
            log(f'Validated excerpt for "{chapter_title}" but could not isolate replacement span.')
            continue

        updated_entries = _splice_replacement(entries, script_group, replacement_chunk, replacement_entries)
        save_script_document(
            script_path,
            entries=updated_entries,
            dictionary=script_document.get("dictionary", []),
            sanity_cache=sanity_cache,
        )
        if os.path.exists(chunks_path):
            os.remove(chunks_path)
        repaired_targets += 1
        log(
            f'Patched "{chapter_title}" with {len(replacement_entries)} regenerated entr'
            f'{"y" if len(replacement_entries) == 1 else "ies"}.'
        )
