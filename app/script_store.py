import copy
import json
import re


_WORD_RE = re.compile(r"\w+", re.UNICODE)


def _clean_string(value):
    return str(value or "").strip()


def clean_dictionary_entries(entries):
    cleaned = []
    if not isinstance(entries, list):
        return cleaned

    for entry in entries:
        if not isinstance(entry, dict):
            continue
        source = _clean_string(entry.get("source"))
        alias = _clean_string(entry.get("alias"))
        if not source or not alias:
            continue
        cleaned.append({"source": source, "alias": alias})

    return cleaned


def normalize_script_document(data):
    if isinstance(data, list):
        return {
            "entries": data,
            "dictionary": [],
        }

    if isinstance(data, dict):
        entries = data.get("entries")
        if not isinstance(entries, list):
            entries = []
        return {
            "entries": entries,
            "dictionary": clean_dictionary_entries(data.get("dictionary", [])),
        }

    return {"entries": [], "dictionary": []}


def load_script_document(path):
    with open(path, "r", encoding="utf-8") as f:
        return normalize_script_document(json.load(f))


def save_script_document(path, entries=None, dictionary=None):
    try:
        current = load_script_document(path)
    except FileNotFoundError:
        current = {"entries": [], "dictionary": []}

    document = {
        "entries": current["entries"] if entries is None else entries,
        "dictionary": current["dictionary"] if dictionary is None else clean_dictionary_entries(dictionary),
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(document, f, indent=2, ensure_ascii=False)

    return document


def _match_case(alias, matched_text):
    if not matched_text:
        return alias
    if matched_text.isupper():
        return alias.upper()
    if matched_text.islower():
        return alias.lower()
    words = re.findall(r"[A-Za-z]+", matched_text)
    if words and all(word[:1].isupper() and word[1:].islower() for word in words):
        return alias.title()
    if matched_text[:1].isupper():
        return alias[:1].upper() + alias[1:]
    return alias


def _entry_pattern(source):
    escaped = re.escape(source)
    if _WORD_RE.search(source):
        return re.compile(rf"(?<!\w){escaped}(?!\w)", re.IGNORECASE)
    return re.compile(escaped, re.IGNORECASE)


def apply_dictionary_to_text(text, dictionary_entries):
    current_text = text or ""
    counts = []
    placeholders = {}
    placeholder_index = 0

    for entry in clean_dictionary_entries(dictionary_entries):
        pattern = _entry_pattern(entry["source"])
        replacements = 0

        def replace(match):
            nonlocal replacements, placeholder_index
            replacements += 1
            token = f"\uE000{placeholder_index}\uE001"
            placeholder_index += 1
            placeholders[token] = _match_case(entry["alias"], match.group(0))
            return token

        current_text = pattern.sub(replace, current_text)
        counts.append(replacements)

    for token, replacement in placeholders.items():
        current_text = current_text.replace(token, replacement)

    return current_text, counts


def build_dictionary_preview_counts(dictionary_entries, texts):
    counts = [0] * len(clean_dictionary_entries(dictionary_entries))
    for text in texts or []:
        _, text_counts = apply_dictionary_to_text(text or "", dictionary_entries)
        for index, value in enumerate(text_counts):
            counts[index] += value
    return counts


def apply_dictionary_to_chunks(chunks, dictionary_entries):
    transformed_chunks = copy.deepcopy(chunks)
    replacements_by_id = {}

    for chunk in transformed_chunks:
        transformed_text, _ = apply_dictionary_to_text(chunk.get("text", ""), dictionary_entries)
        chunk["text"] = transformed_text
        replacements_by_id[chunk.get("id")] = transformed_text

    return transformed_chunks, replacements_by_id
