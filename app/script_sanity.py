import re
import time
from difflib import SequenceMatcher


_WORD_RE = re.compile(r"[A-Za-z]+")
_WHITESPACE_RE = re.compile(r"\s+")
_AFFIRMATIVE_RE = re.compile(r"\b(?:true|yes)\b", re.IGNORECASE)
_NEGATIVE_RE = re.compile(r"\b(?:false|no)\b", re.IGNORECASE)


def _normalize_title(value):
    return _WHITESPACE_RE.sub(" ", (value or "").strip()).lower()


def _tokenize_with_positions(text):
    tokens = []
    for match in _WORD_RE.finditer(text or ""):
        tokens.append({
            "word": match.group(0).lower(),
            "start": match.start(),
            "end": match.end(),
        })
    return tokens


def _excerpt(text, start, end, radius=80):
    text = text or ""
    start = max(0, int(start or 0))
    end = max(start, int(end or start))
    left = max(0, start - radius)
    right = min(len(text), end + radius)
    return text[left:right].strip()


def _normalize_phrase_key(text):
    return " ".join(match.group(0).lower() for match in _WORD_RE.finditer(text or ""))


def _normalize_spacing(text):
    return _WHITESPACE_RE.sub(" ", (text or "").strip())


def _contains_sentence_break(text):
    stripped = (text or "").strip()
    return any(marker in stripped for marker in (".", "!", "?", "\n\n"))


def _find_sentence_left_boundary(text, index):
    text = text or ""
    index = max(0, min(len(text), int(index or 0)))
    candidates = [pos for pos in (
        text.rfind(". ", 0, index),
        text.rfind("! ", 0, index),
        text.rfind("? ", 0, index),
        text.rfind("\n\n", 0, index),
    ) if pos != -1]
    if not candidates:
        return 0
    left = max(candidates)
    return left + 2


def _find_sentence_right_boundary(text, index):
    text = text or ""
    index = max(0, min(len(text), int(index or 0)))
    candidates = [pos for pos in (
        text.find(". ", index),
        text.find("! ", index),
        text.find("? ", index),
        text.find("\n\n", index),
    ) if pos != -1]
    if not candidates:
        return len(text)
    end = min(candidates)
    if text[end:end + 2] in (". ", "! ", "? "):
        return end + 1
    return end


def _sentence_context(text, start, end):
    text = text or ""
    left_start = _find_sentence_left_boundary(text, start)
    left_end = min(len(text), max(left_start, start))
    right_start = min(len(text), max(end, 0))
    right_end = _find_sentence_right_boundary(text, right_start)
    return (
        text[left_start:left_end].strip(),
        text[right_start:right_end].strip(),
    )


def _char_start(tokens, index, fallback=0):
    if tokens and 0 <= index < len(tokens):
        return tokens[index]["start"]
    if tokens and index >= len(tokens):
        return tokens[-1]["end"]
    return fallback


def _char_end(tokens, index, fallback=0):
    if tokens and index > 0:
        resolved = min(index - 1, len(tokens) - 1)
        return tokens[resolved]["end"]
    return fallback


def _group_script_chapters(entries):
    groups = []
    current = None

    for entry_index, entry in enumerate(entries or []):
        chapter = (entry.get("chapter") or "").strip()
        text = (entry.get("text") or "").strip()

        if current is None or current["title"] != chapter:
            current = {
                "title": chapter,
                "entry_start": entry_index,
                "entry_end": entry_index,
                "parts": [],
            }
            groups.append(current)

        current["entry_end"] = entry_index + 1
        if text:
            current["parts"].append(text)

    chapters = []
    for index, group in enumerate(groups, start=1):
        chapters.append({
            "index": index,
            "title": group["title"],
            "text": "\n\n".join(group["parts"]).strip(),
            "entry_start": group["entry_start"],
            "entry_end": group["entry_end"],
        })

    return chapters


def _pair_chapters(source_chapters, script_chapters):
    source_titles = [_normalize_title(chapter.get("title")) for chapter in source_chapters]
    script_titles = [_normalize_title(chapter.get("title")) for chapter in script_chapters]
    matcher = SequenceMatcher(None, source_titles, script_titles, autojunk=False)

    pairs = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            for offset in range(i2 - i1):
                pairs.append({
                    "kind": "paired",
                    "source": source_chapters[i1 + offset],
                    "script": script_chapters[j1 + offset],
                })
            continue

        if tag == "replace":
            common = min(i2 - i1, j2 - j1)
            for offset in range(common):
                pairs.append({
                    "kind": "paired",
                    "source": source_chapters[i1 + offset],
                    "script": script_chapters[j1 + offset],
                })
            for source_index in range(i1 + common, i2):
                pairs.append({
                    "kind": "missing_chapter",
                    "source": source_chapters[source_index],
                    "script": None,
                })
            for script_index in range(j1 + common, j2):
                pairs.append({
                    "kind": "inserted_chapter",
                    "source": None,
                    "script": script_chapters[script_index],
                })
            continue

        if tag == "delete":
            for source_index in range(i1, i2):
                pairs.append({
                    "kind": "missing_chapter",
                    "source": source_chapters[source_index],
                    "script": None,
                })
            continue

        if tag == "insert":
            for script_index in range(j1, j2):
                pairs.append({
                    "kind": "inserted_chapter",
                    "source": None,
                    "script": script_chapters[script_index],
                })

    return pairs


def _build_invalid_section(source_chapter, script_chapter, source_tokens, script_tokens, i1, i2, j1, j2):
    source_text = (source_chapter or {}).get("text") or ""
    script_text = (script_chapter or {}).get("text") or ""

    source_anchor = _char_start(source_tokens, i1, 0)
    source_end = _char_end(source_tokens, i2, source_anchor)
    script_anchor = _char_start(script_tokens, j1, 0)
    script_end = _char_end(script_tokens, j2, script_anchor)

    source_span_text = source_text[source_anchor:source_end].strip() if source_end >= source_anchor else ""
    left_context, right_context = _sentence_context(source_text, source_anchor, source_end)
    sentence_start = _find_sentence_left_boundary(source_text, source_anchor)
    sentence_end = _find_sentence_right_boundary(source_text, source_end)
    sentence_text = source_text[sentence_start:sentence_end].strip()

    return {
        "chapter_index": (source_chapter or script_chapter or {}).get("index"),
        "chapter_title": (source_chapter or script_chapter or {}).get("title") or "",
        "missing_words": max(0, i2 - i1),
        "inserted_words": max(0, j2 - j1),
        "source_word_start": i1,
        "source_word_end": i2,
        "script_word_start": j1,
        "script_word_end": j2,
        "source_char_start": source_anchor,
        "source_char_end": source_end,
        "script_char_start": script_anchor,
        "script_char_end": script_end,
        "source_excerpt": _excerpt(source_text, source_anchor, source_end),
        "script_excerpt": _excerpt(script_text, script_anchor, script_end),
        "source_text": source_span_text,
        "left_context": left_context,
        "right_context": right_context,
        "covers_full_sentence": bool(source_span_text) and _normalize_phrase_key(source_span_text) == _normalize_phrase_key(sentence_text),
    }


def _build_chapter_only_section(kind, chapter):
    text = chapter.get("text") or ""
    tokens = _tokenize_with_positions(text)
    missing_words = len(tokens) if kind == "missing_chapter" else 0
    inserted_words = len(tokens) if kind == "inserted_chapter" else 0
    char_end = len(text)
    return {
        "chapter_index": chapter.get("index"),
        "chapter_title": chapter.get("title") or "",
        "missing_words": missing_words,
        "inserted_words": inserted_words,
        "source_word_start": 0,
        "source_word_end": missing_words,
        "script_word_start": 0,
        "script_word_end": inserted_words,
        "source_char_start": 0,
        "source_char_end": char_end if kind == "missing_chapter" else 0,
        "script_char_start": 0,
        "script_char_end": char_end if kind == "inserted_chapter" else 0,
        "source_excerpt": text[:200].strip() if kind == "missing_chapter" else "",
        "script_excerpt": text[:200].strip() if kind == "inserted_chapter" else "",
        "source_text": text.strip() if kind == "missing_chapter" else "",
        "left_context": "",
        "right_context": "",
        "covers_full_sentence": kind == "missing_chapter",
    }


def _merge_replacement_chunks(sections, chunk_size):
    if not sections:
        return []

    ordered = sorted(
        sections,
        key=lambda section: (
            int(section.get("chapter_index") or 0),
            int(section.get("source_char_start") or 0),
            int(section.get("script_char_start") or 0),
        ),
    )

    merged = []
    current = None

    for section in ordered:
        if current is None:
            current = {
                "chapter_index": section["chapter_index"],
                "chapter_title": section["chapter_title"],
                "source_word_start": section["source_word_start"],
                "source_word_end": section["source_word_end"],
                "script_word_start": section["script_word_start"],
                "script_word_end": section["script_word_end"],
                "source_char_start": section["source_char_start"],
                "source_char_end": section["source_char_end"],
                "script_char_start": section["script_char_start"],
                "script_char_end": section["script_char_end"],
                "missing_words": section["missing_words"],
                "inserted_words": section["inserted_words"],
                "section_count": 1,
            }
            continue

        same_chapter = (
            current["chapter_index"] == section["chapter_index"]
            and current["chapter_title"] == section["chapter_title"]
        )
        source_gap = max(0, int(section["source_char_start"]) - int(current["source_char_end"]))
        script_gap = max(0, int(section["script_char_start"]) - int(current["script_char_end"]))
        gap = source_gap if source_gap or not script_gap else script_gap
        merged_source_size = max(current["source_char_end"], section["source_char_end"]) - min(current["source_char_start"], section["source_char_start"])
        merged_script_size = max(current["script_char_end"], section["script_char_end"]) - min(current["script_char_start"], section["script_char_start"])
        within_limit = merged_source_size <= chunk_size and merged_script_size <= chunk_size

        if same_chapter and gap < chunk_size and within_limit:
            current["source_word_start"] = min(current["source_word_start"], section["source_word_start"])
            current["source_word_end"] = max(current["source_word_end"], section["source_word_end"])
            current["script_word_start"] = min(current["script_word_start"], section["script_word_start"])
            current["script_word_end"] = max(current["script_word_end"], section["script_word_end"])
            current["source_char_end"] = max(current["source_char_end"], section["source_char_end"])
            current["script_char_end"] = max(current["script_char_end"], section["script_char_end"])
            current["missing_words"] += section["missing_words"]
            current["inserted_words"] += section["inserted_words"]
            current["section_count"] += 1
            continue

        merged.append(current)
        current = {
            "chapter_index": section["chapter_index"],
            "chapter_title": section["chapter_title"],
            "source_word_start": section["source_word_start"],
            "source_word_end": section["source_word_end"],
            "script_word_start": section["script_word_start"],
            "script_word_end": section["script_word_end"],
            "source_char_start": section["source_char_start"],
            "source_char_end": section["source_char_end"],
            "script_char_start": section["script_char_start"],
            "script_char_end": section["script_char_end"],
            "missing_words": section["missing_words"],
            "inserted_words": section["inserted_words"],
            "section_count": 1,
        }

    if current is not None:
        merged.append(current)

    return merged


def _diff_paired_chapter(source_chapter, script_chapter):
    source_text = source_chapter.get("text") or ""
    script_text = script_chapter.get("text") or ""
    source_tokens = _tokenize_with_positions(source_text)
    script_tokens = _tokenize_with_positions(script_text)
    source_words = [token["word"] for token in source_tokens]
    script_words = [token["word"] for token in script_tokens]

    matcher = SequenceMatcher(None, source_words, script_words, autojunk=False)

    invalid_sections = []
    pending = None

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            if pending is not None:
                invalid_sections.append(
                    _build_invalid_section(source_chapter, script_chapter, source_tokens, script_tokens, *pending)
                )
                pending = None
            continue

        if pending is None:
            pending = [i1, i2, j1, j2]
            continue

        if i1 == pending[1] and j1 == pending[3]:
            pending[1] = i2
            pending[3] = j2
            continue

        invalid_sections.append(
            _build_invalid_section(source_chapter, script_chapter, source_tokens, script_tokens, *pending)
        )
        pending = [i1, i2, j1, j2]

    if pending is not None:
        invalid_sections.append(
            _build_invalid_section(source_chapter, script_chapter, source_tokens, script_tokens, *pending)
        )

    return {
        "chapter_index": source_chapter.get("index"),
        "chapter_title": source_chapter.get("title") or script_chapter.get("title") or "",
        "source_word_count": len(source_tokens),
        "script_word_count": len(script_tokens),
        "missing_words": sum(section["missing_words"] for section in invalid_sections),
        "inserted_words": sum(section["inserted_words"] for section in invalid_sections),
        "invalid_sections": invalid_sections,
    }


def _iter_sections(chapter_results):
    for chapter in chapter_results:
        for section in chapter.get("invalid_sections") or []:
            yield chapter, section


def _should_classify_attribution(section):
    if int(section.get("missing_words") or 0) <= 0:
        return False
    if int(section.get("inserted_words") or 0) != 0:
        return False
    source_text = (section.get("source_text") or "").strip()
    if not source_text:
        return False
    if section.get("covers_full_sentence"):
        return False
    if _contains_sentence_break(source_text):
        return False
    return True


def _rebuild_result(base_result, chapter_results):
    all_invalid_sections = []
    for chapter in chapter_results:
        chapter["replacement_chunks"] = _merge_replacement_chunks(
            chapter.get("invalid_sections") or [],
            base_result["chunk_size"],
        )
        chapter["invalid_chunk_count"] = len(chapter["replacement_chunks"])
        chapter["missing_words"] = sum(section["missing_words"] for section in chapter.get("invalid_sections") or [])
        chapter["inserted_words"] = sum(section["inserted_words"] for section in chapter.get("invalid_sections") or [])
        all_invalid_sections.extend(chapter.get("invalid_sections") or [])

    replacement_chunks = _merge_replacement_chunks(all_invalid_sections, base_result["chunk_size"])
    base_result["chapters"] = chapter_results
    base_result["missing_words"] = sum(chapter["missing_words"] for chapter in chapter_results)
    base_result["inserted_words"] = sum(chapter["inserted_words"] for chapter in chapter_results)
    base_result["invalid_sections"] = all_invalid_sections
    base_result["invalid_section_count"] = len(all_invalid_sections)
    base_result["replacement_chunks"] = replacement_chunks
    base_result["invalid_chunk_count"] = len(replacement_chunks)
    return base_result


def build_attribution_classifier(client, model_name, system_prompt, user_prompt_template, max_tokens=256):
    def classify(payload):
        user_prompt = (user_prompt_template or "")
        replacements = {
            "{chapter}": payload.get("chapter_title") or "",
            "{left_context}": payload.get("left_context") or "",
            "{missing_text}": payload.get("source_text") or "",
            "{right_context}": payload.get("right_context") or "",
        }
        for placeholder, value in replacements.items():
            user_prompt = user_prompt.replace(placeholder, value)
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0,
            top_p=1,
            max_tokens=max_tokens,
        )
        choice = response.choices[0] if getattr(response, "choices", None) else None
        text = ((choice.message.content if choice and getattr(choice, "message", None) else "") or "").strip()
        normalized = text.upper()
        if _AFFIRMATIVE_RE.search(text) and not _NEGATIVE_RE.search(text):
            return True, text, normalized
        if _NEGATIVE_RE.search(text) and not _AFFIRMATIVE_RE.search(text):
            return False, text, normalized
        return False, text, normalized

    return classify


def run_script_sanity_check(source_document, script_document, chunk_size, attribution_resolver=None, known_phrase_decisions=None):
    source_chapters = []
    for index, chapter in enumerate(source_document.get("chapters") or [], start=1):
        source_chapters.append({
            "index": index,
            "title": (chapter.get("title") or "").strip(),
            "text": chapter.get("text") or "",
        })

    script_chapters = _group_script_chapters(script_document.get("entries") or [])
    chapter_pairs = _pair_chapters(source_chapters, script_chapters)

    chapter_results = []
    all_invalid_sections = []

    for pair in chapter_pairs:
        kind = pair["kind"]

        if kind == "paired":
            chapter_result = _diff_paired_chapter(pair["source"], pair["script"])
            chapter_result["kind"] = "paired"
            chapter_result["source_title"] = pair["source"].get("title") or ""
            chapter_result["script_title"] = pair["script"].get("title") or ""
        elif kind == "missing_chapter":
            section = _build_chapter_only_section(kind, pair["source"])
            chapter_result = {
                "kind": kind,
                "chapter_index": pair["source"].get("index"),
                "chapter_title": pair["source"].get("title") or "",
                "source_title": pair["source"].get("title") or "",
                "script_title": "",
                "source_word_count": len(_tokenize_with_positions(pair["source"].get("text") or "")),
                "script_word_count": 0,
                "missing_words": section["missing_words"],
                "inserted_words": 0,
                "invalid_sections": [section],
            }
        else:
            section = _build_chapter_only_section(kind, pair["script"])
            chapter_result = {
                "kind": kind,
                "chapter_index": pair["script"].get("index"),
                "chapter_title": pair["script"].get("title") or "",
                "source_title": "",
                "script_title": pair["script"].get("title") or "",
                "source_word_count": 0,
                "script_word_count": len(_tokenize_with_positions(pair["script"].get("text") or "")),
                "missing_words": 0,
                "inserted_words": section["inserted_words"],
                "invalid_sections": [section],
            }

        chapter_result["replacement_chunks"] = _merge_replacement_chunks(
            chapter_result["invalid_sections"],
            chunk_size,
        )
        chapter_result["invalid_chunk_count"] = len(chapter_result["replacement_chunks"])
        chapter_results.append(chapter_result)
        all_invalid_sections.extend(chapter_result["invalid_sections"])

    result = {
        "checked_at": time.time(),
        "source_type": source_document.get("type") or "text",
        "source_title": source_document.get("title") or source_document.get("book_title") or "",
        "chunk_size": int(chunk_size),
        "source_chapter_count": len(source_chapters),
        "script_chapter_count": len(script_chapters),
        "missing_words": sum(result["missing_words"] for result in chapter_results),
        "inserted_words": sum(result["inserted_words"] for result in chapter_results),
        "invalid_sections": all_invalid_sections,
        "invalid_section_count": len(all_invalid_sections),
        "replacement_chunks": _merge_replacement_chunks(all_invalid_sections, chunk_size),
        "invalid_chunk_count": len(_merge_replacement_chunks(all_invalid_sections, chunk_size)),
        "chapters": chapter_results,
        "raw_missing_words": sum(result["missing_words"] for result in chapter_results),
        "raw_inserted_words": sum(result["inserted_words"] for result in chapter_results),
        "raw_invalid_section_count": len(all_invalid_sections),
        "attribution_pruned_words": 0,
        "attribution_pruned_sections": 0,
        "attribution_candidates": 0,
        "attribution_cache_hits": 0,
        "attribution_model_queries": 0,
        "attribution_decisions": [],
    }

    if not attribution_resolver and not known_phrase_decisions:
        return result

    phrase_decisions = {}
    if isinstance(known_phrase_decisions, dict):
        phrase_decisions.update(known_phrase_decisions)

    candidate_map = {}
    for _, section in _iter_sections(chapter_results):
        if not _should_classify_attribution(section):
            continue
        phrase_key = _normalize_phrase_key(section.get("source_text"))
        if not phrase_key:
            continue
        section["phrase_key"] = phrase_key
        candidate_map.setdefault(phrase_key, {
            "phrase_key": phrase_key,
            "source_text": _normalize_spacing(section.get("source_text")),
            "chapter_title": section.get("chapter_title") or "",
            "left_context": _normalize_spacing(section.get("left_context")),
            "right_context": _normalize_spacing(section.get("right_context")),
            "missing_words": int(section.get("missing_words") or 0),
            "decision": None,
            "source": None,
            "reply": "",
        })

    result["attribution_candidates"] = len(candidate_map)

    for phrase_key, payload in candidate_map.items():
        cached = phrase_decisions.get(phrase_key)
        if isinstance(cached, dict) and cached.get("decision") in ("accepted", "rejected"):
            payload["decision"] = cached["decision"]
            payload["source"] = "cache"
            payload["reply"] = cached.get("reply", "")
            result["attribution_cache_hits"] += 1
            continue
        if attribution_resolver is None:
            continue
        accepted, reply, normalized_reply = attribution_resolver(payload)
        payload["decision"] = "accepted" if accepted else "rejected"
        payload["source"] = "model"
        payload["reply"] = reply
        phrase_decisions[phrase_key] = {
            "phrase": payload["source_text"],
            "decision": payload["decision"],
            "reply": normalized_reply[:500],
            "checked_at": time.time(),
        }
        result["attribution_model_queries"] += 1

    pruned_keys = {
        phrase_key
        for phrase_key, payload in candidate_map.items()
        if payload.get("decision") == "accepted"
    }
    result["attribution_decisions"] = list(candidate_map.values())

    if not pruned_keys:
        result["attribution_phrase_decisions"] = phrase_decisions
        return result

    updated_chapters = []
    pruned_words = 0
    pruned_sections = 0
    for chapter in chapter_results:
        filtered_sections = []
        for section in chapter.get("invalid_sections") or []:
            phrase_key = section.get("phrase_key") or _normalize_phrase_key(section.get("source_text"))
            if phrase_key in pruned_keys and _should_classify_attribution(section):
                pruned_words += int(section.get("missing_words") or 0)
                pruned_sections += 1
                continue
            filtered_sections.append(section)
        updated_chapter = dict(chapter)
        updated_chapter["invalid_sections"] = filtered_sections
        updated_chapters.append(updated_chapter)

    result["attribution_pruned_words"] = pruned_words
    result["attribution_pruned_sections"] = pruned_sections
    result["attribution_phrase_decisions"] = phrase_decisions
    return _rebuild_result(result, updated_chapters)
