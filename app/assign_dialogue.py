#!/usr/bin/env python3
"""
Assign speakers to dialogue paragraphs using LLM tool use.

For each paragraph tagged has_dialogue=True in paragraphs.json, this script:
  - Builds a context window centred on that paragraph (chunk_size * 0.8 chars)
  - Extracts the quoted dialogue lines from the paragraph
  - Makes a single chat.completions call with the identify_dialogue tool forced
  - Stores the returned speaker name in the paragraph, or marks it as an error

Results are written back into paragraphs.json atomically.
"""
import json
import os
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI

TASK_PROGRESS_PREFIX = "__TASK_PROGRESS__:"

# Matches straight (") or curly (\u201c / \u201d) double-quotes enclosing ≥2 chars.
QUOTE_RE = re.compile(r'["\u201c]([^"\u201d]{2,})["\u201d]', re.DOTALL)

IDENTIFY_DIALOGUE_TOOL = {
    "type": "function",
    "function": {
        "name": "identify_dialogue",
        "description": "Record the name of the character who speaks the dialogue in this paragraph.",
        "parameters": {
            "type": "object",
            "properties": {
                "speaker": {
                    "type": "string",
                    "description": (
                        "Name of the character speaking. Use an existing name from the "
                        "known-characters list when one fits, or supply a new name if none does. "
                        "Use NARRATOR only for clearly unvoiced internal thoughts inside quotes."
                    ),
                }
            },
            "required": ["speaker"],
        },
    },
}

DEFAULT_SYSTEM_PROMPT = (
    "You are a dialogue attribution specialist helping to build an audiobook script. "
    "Your sole task is to identify who speaks a given piece of dialogue based on the surrounding narrative. "
    "You MUST call the identify_dialogue tool with your answer. Do not respond in plain text."
)


def _normalize_speaker_name(raw_speaker: str) -> str:
    """Normalize a model-returned speaker name to canonical form.

    Rules:
    - Convert non-alphanumeric characters to spaces.
    - Drop all leading characters until the first letter.
    - Reject names with no letters.
    - Collapse whitespace.
    - Apply front capitalization per token.
    """
    raw = str(raw_speaker or "").strip()
    if not raw:
        return ""

    # Keep only letters/numbers; convert everything else to spaces.
    sanitized = "".join(ch if ch.isalnum() else " " for ch in raw)
    if not sanitized.strip():
        return ""

    # First character must be a letter: drop leading non-letter chars.
    first_letter_idx = -1
    for i, ch in enumerate(sanitized):
        if ch.isalpha():
            first_letter_idx = i
            break
    if first_letter_idx < 0:
        return ""
    sanitized = sanitized[first_letter_idx:]

    tokens = [token for token in sanitized.split() if token]
    if not tokens:
        return ""

    # Reject names that still contain no letters across all tokens.
    if not any(any(c.isalpha() for c in token) for token in tokens):
        return ""

    # Front capitalization: each token in Name-Case.
    normalized_tokens = []
    for token in tokens:
        head = token[0].upper()
        tail = token[1:].lower() if len(token) > 1 else ""
        normalized_tokens.append(head + tail)

    return " ".join(normalized_tokens)


def _log(msg: str):
    print(msg, flush=True)


def _progress(current: int, total: int, message: str):
    print(
        TASK_PROGRESS_PREFIX + json.dumps({"current": current, "total": total, "message": message}),
        flush=True,
    )


def build_context_window(paragraphs: list, target_idx: int, budget: int) -> str:
    """
    Centre paragraphs[target_idx] in a block of at most `budget` characters.
    Fills left and right sides alternately from adjacent paragraphs until the
    budget is exhausted or there are no more paragraphs to add.
    """
    target_text = paragraphs[target_idx]["text"]
    left_parts: list[str] = []
    right_parts: list[str] = []
    used = len(target_text)
    left = target_idx - 1
    right = target_idx + 1

    while used < budget:
        added = False
        if left >= 0:
            t = paragraphs[left]["text"]
            if used + len(t) <= budget:
                left_parts.insert(0, t)
                used += len(t)
                left -= 1
                added = True
        if right < len(paragraphs):
            t = paragraphs[right]["text"]
            if used + len(t) <= budget:
                right_parts.append(t)
                used += len(t)
                right += 1
                added = True
        if not added:
            break

    return "\n\n".join(left_parts + [target_text] + right_parts)


def extract_dialogue_lines(text: str) -> list[str]:
    """Return all quoted strings found in `text`."""
    return QUOTE_RE.findall(text)


def main():
    if len(sys.argv) < 3:
        _log("Usage: assign_dialogue.py <paragraphs_path> <config_path> [--retry-errors <N>]")
        sys.exit(1)

    paragraphs_path = sys.argv[1]
    config_path = sys.argv[2]

    retry_errors_mode = False
    retry_max = 0
    for i, arg in enumerate(sys.argv[3:], start=3):
        if arg == "--retry-errors" and i + 1 < len(sys.argv):
            retry_errors_mode = True
            try:
                retry_max = max(0, int(sys.argv[i + 1]))
            except (ValueError, IndexError):
                retry_max = 0

    # ── Load paragraphs ────────────────────────────────────────────────────────
    try:
        with open(paragraphs_path, "r", encoding="utf-8") as f:
            paragraphs_doc = json.load(f)
    except Exception as e:
        _log(f"ERROR: Could not load paragraphs file: {e}")
        sys.exit(1)

    paragraphs = paragraphs_doc.get("paragraphs", [])
    if not paragraphs:
        _log("ERROR: paragraphs.json contains no paragraphs. Run 'Process Paragraphs' first.")
        sys.exit(1)

    # ── Load config ────────────────────────────────────────────────────────────
    config: dict = {}
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
    except Exception as e:
        _log(f"WARNING: Could not load config ({e}). Using defaults.")

    llm_cfg = config.get("llm") or {}
    gen_cfg = config.get("generation") or {}
    prompts = config.get("prompts") or {}

    base_url   = llm_cfg.get("base_url", "http://localhost:1234/v1")
    api_key    = llm_cfg.get("api_key", "local")
    model_name = llm_cfg.get("model_name", "local-model")
    chunk_size = int(gen_cfg.get("chunk_size") or 3000)
    context_budget = int(chunk_size * 0.8)
    max_tokens = int(gen_cfg.get("max_tokens") or 2048)

    system_prompt = (prompts.get("dialogue_identification_system_prompt") or "").strip() or DEFAULT_SYSTEM_PROMPT

    tts_cfg = config.get("tts") or {}
    workers = max(1, int(tts_cfg.get("parallel_workers", 1) or 1))

    _log(f"Model: {model_name}  |  Context budget: {context_budget} chars  |  Max tokens: {max_tokens}  |  Workers: {workers}")

    client = OpenAI(base_url=base_url, api_key=api_key, timeout=600)

    # ── Retry-errors mode ──────────────────────────────────────────────────────
    if retry_errors_mode:
        error_ids = set(paragraphs_doc.get("dialogue_errors", []))
        error_paras = [(i, p) for i, p in enumerate(paragraphs)
                       if p.get("dialogue_error") and p["id"] in error_ids]
        _log(f"=== Error Correction: retrying {len(error_paras)} paragraph(s) with up to {retry_max} attempt(s) each (workers={workers}) ===")

        known_speakers: list[str] = [
            p.get("speaker") for p in paragraphs
            if p.get("speaker") and not p.get("dialogue_error")
        ]

        retry_lock = threading.Lock()
        fixed_ref = [0]

        def retry_one(idx_para):
            idx, para = idx_para
            para_id = para["id"]
            dialogue_lines = extract_dialogue_lines(para["text"])
            if not dialogue_lines:
                _log(f"SKIP {para_id}: no quoted text — cannot retry")
                return

            context_text = build_context_window(paragraphs, idx, context_budget)
            with retry_lock:
                char_list = ", ".join(known_speakers) if known_speakers else "(none yet)"
            dialogue_block = "\n".join(f'- "{line}"' for line in dialogue_lines)
            user_msg = (
                f"PASSAGE CONTEXT:\n{context_text}\n\n"
                f"KNOWN CHARACTERS SO FAR: {char_list}\n\n"
                f"DIALOGUE IN THE TARGET PARAGRAPH:\n{dialogue_block}\n\n"
                "Use the identify_dialogue tool to name the speaker of this dialogue. "
                "If none of the known characters fits, supply the new character's name."
            )

            speaker = ""
            for attempt in range(1, retry_max + 1):
                try:
                    response = client.chat.completions.create(
                        model=model_name,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user",   "content": user_msg},
                        ],
                        tools=[IDENTIFY_DIALOGUE_TOOL],
                        tool_choice="required",
                        temperature=0.1,
                        max_tokens=max_tokens,
                    )
                    msg = response.choices[0].message
                    tool_calls = getattr(msg, "tool_calls", None)
                    if tool_calls:
                        raw_args = tool_calls[0].function.arguments
                        args = json.loads(raw_args)
                        speaker = _normalize_speaker_name(args.get("speaker"))
                    else:
                        reasoning = getattr(msg, "reasoning_content", None) or ""
                        m = re.search(
                            r"<parameter=speaker>\s*(.*?)\s*</parameter>",
                            reasoning,
                            re.DOTALL | re.IGNORECASE,
                        )
                        if m:
                            speaker = _normalize_speaker_name(m.group(1))
                        if not speaker:
                            _log(f"RETRY {para_id} attempt {attempt}/{retry_max}: model returned no speaker")
                except Exception as e:
                    _log(f"RETRY {para_id} attempt {attempt}/{retry_max}: API error — {e}")

                if speaker:
                    break

            with retry_lock:
                if speaker:
                    para["speaker"] = speaker
                    para["dialogue_error"] = False
                    if speaker not in known_speakers:
                        known_speakers.append(speaker)
                        _log(f"New character: {speaker}")
                    error_ids.discard(para_id)
                    fixed_ref[0] += 1
                    _log(f"FIXED {para_id}: speaker = {speaker}")
                else:
                    _log(f"FAILED {para_id}: still no speaker after {retry_max} attempt(s)")

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(retry_one, item) for item in error_paras]
            for future in as_completed(futures):
                future.result()

        paragraphs_doc["dialogue_errors"] = list(error_ids)
        _atomic_write(paragraphs_path, paragraphs_doc)
        _log(f"Error correction done. Fixed: {fixed_ref[0]}/{len(error_paras)}. Remaining errors: {len(error_ids)}.")
        return

    # ── Filter dialogue paragraphs ─────────────────────────────────────────────
    dialogue_paras = [(i, p) for i, p in enumerate(paragraphs) if p.get("has_dialogue")]
    total = len(dialogue_paras)

    if total == 0:
        _log("No dialogue paragraphs found. Nothing to assign.")
        paragraphs_doc["dialogue_assignment_complete"] = True
        paragraphs_doc["dialogue_errors"] = []
        _atomic_write(paragraphs_path, paragraphs_doc)
        return

    _log(f"Assigning speakers to {total} dialogue paragraphs (workers={workers})...")

    known_speakers: list[str] = []
    errors: list[str] = []
    lock = threading.Lock()
    done_ref = [0]

    def process_para(idx_para):
        idx, para = idx_para
        para_id = para["id"]

        context_text = build_context_window(paragraphs, idx, context_budget)
        dialogue_lines = extract_dialogue_lines(para["text"])
        if not dialogue_lines:
            # has_dialogue was set but no quoted text found — treat as error
            with lock:
                para["speaker"] = None
                para["dialogue_error"] = True
                errors.append(para_id)
                done_ref[0] += 1
                _log(f"ERROR {para_id}: tagged as dialogue but no quoted text found")
                _maybe_progress(done_ref[0], total)
            return

        with lock:
            char_list = ", ".join(known_speakers) if known_speakers else "(none yet)"

        dialogue_block = "\n".join(f'- "{line}"' for line in dialogue_lines)
        user_msg = (
            f"PASSAGE CONTEXT:\n{context_text}\n\n"
            f"KNOWN CHARACTERS SO FAR: {char_list}\n\n"
            f"DIALOGUE IN THE TARGET PARAGRAPH:\n{dialogue_block}\n\n"
            "Use the identify_dialogue tool to name the speaker of this dialogue. "
            "If none of the known characters fits, supply the new character's name."
        )

        speaker = ""
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_msg},
                ],
                tools=[IDENTIFY_DIALOGUE_TOOL],
                tool_choice="required",
                temperature=0.1,
                max_tokens=max_tokens,
            )
            msg = response.choices[0].message
            tool_calls = getattr(msg, "tool_calls", None)
            if tool_calls:
                raw_args = tool_calls[0].function.arguments
                args = json.loads(raw_args)
                speaker = _normalize_speaker_name(args.get("speaker"))
            else:
                # Some models (e.g. Nemotron) put the tool call in reasoning_content
                # as XML instead of populating tool_calls. Parse it from there.
                reasoning = getattr(msg, "reasoning_content", None) or ""
                m = re.search(
                    r"<parameter=speaker>\s*(.*?)\s*</parameter>",
                    reasoning,
                    re.DOTALL | re.IGNORECASE,
                )
                if m:
                    speaker = _normalize_speaker_name(m.group(1))
                if not speaker:
                    _log(f"ERROR {para_id}: model did not return a speaker")
                    _log(f"--- REQUEST THAT FAILED (system) ---\n{system_prompt}\n--- REQUEST THAT FAILED (user) ---\n{user_msg}\n--- END REQUEST ---")
        except Exception as e:
            _log(f"ERROR {para_id}: API call failed — {e}")

        with lock:
            if speaker:
                para["speaker"] = speaker
                para["dialogue_error"] = False
                if speaker not in known_speakers:
                    known_speakers.append(speaker)
                    _log(f"New character: {speaker}")
            else:
                para["speaker"] = None
                para["dialogue_error"] = True
                if para_id not in errors:
                    errors.append(para_id)
            done_ref[0] += 1
            _maybe_progress(done_ref[0], total)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(process_para, item) for item in dialogue_paras]
        for future in as_completed(futures):
            future.result()

    # ── Save results ───────────────────────────────────────────────────────────
    paragraphs_doc["dialogue_assignment_complete"] = True
    paragraphs_doc["dialogue_errors"] = errors
    _atomic_write(paragraphs_path, paragraphs_doc)

    assigned = total - len(errors)
    _log(f"Done. Assigned: {assigned}/{total}. Errors: {len(errors)}.")
    if errors:
        _log(f"Paragraphs with errors: {', '.join(errors)}")


def _maybe_progress(done: int, total: int):
    if done % 10 == 0 or done == total:
        _progress(done, total, f"Assigned {done}/{total} dialogue paragraphs...")


def _atomic_write(path: str, data: dict):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


if __name__ == "__main__":
    main()
