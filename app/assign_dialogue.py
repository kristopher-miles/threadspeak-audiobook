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
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI

TASK_PROGRESS_PREFIX = "__TASK_PROGRESS__:"

# Matches straight (") or curly (\u201c / \u201d) double-quotes enclosing ≥2 chars.
QUOTE_RE = re.compile(r'["\u201c]([^"\u201d]{2,})["\u201d]', re.DOTALL)

IDENTIFY_DIALOGUE_TOOL = {
    "type": "function",
    "function": {
        "name": "identify_dialogue",
        "description": "Record the name of the character who speaks this quoted text.",
        "parameters": {
            "type": "object",
            "properties": {
                "speaker": {
                    "type": "string",
                    "description": (
                        "Name of the character speaking. Use an existing name from the "
                        "known-characters list when one fits, or supply a new name if none does. "
                        "Return NARRATOR if the quoted text is a scare quote, figure of speech, "
                        "or clearly not spoken aloud by a character."
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
    - Return "NARRATOR" unchanged for scare-quote/non-dialogue markers.
    - Convert non-alphanumeric characters to spaces.
    - Drop all leading characters until the first letter.
    - Reject names with no letters.
    - Collapse whitespace.
    - Apply front capitalization per token.
    """
    raw = str(raw_speaker or "").strip()
    if not raw:
        return ""

    # Preserve the NARRATOR sentinel regardless of case/spacing.
    if raw.upper() == "NARRATOR":
        return "NARRATOR"

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


def _call_identify_dialogue(client, model_name: str, system_prompt: str, user_msg: str, max_tokens: int) -> tuple:
    """
    Stream a single identify_dialogue tool call and return (speaker, raw_response).
    Exits as soon as the first complete tool call JSON is received so local
    models cannot loop and re-emit the same call repeatedly.
    Returns (normalized speaker name, raw_response), or ("", raw_response) on failure.
    """
    tool_call_args = ""
    reasoning_content = ""
    text_content = ""

    stream = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_msg},
        ],
        tools=[IDENTIFY_DIALOGUE_TOOL],
        tool_choice="required",
        parallel_tool_calls=False,
        temperature=0.1,
        max_tokens=max_tokens,
        stream=True,
    )
    for chunk in stream:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        # Accumulate reasoning_content for models (e.g. Nemotron) that emit XML there.
        rc = getattr(delta, "reasoning_content", None)
        if rc:
            reasoning_content += rc
        if delta.content:
            text_content += delta.content
        if delta.tool_calls:
            frag = delta.tool_calls[0].function.arguments
            if frag:
                tool_call_args += frag
        # Stop as soon as we have parseable JSON — don't let the model loop.
        if tool_call_args:
            try:
                args = json.loads(tool_call_args)
                stream.close()
                return (_normalize_speaker_name(args.get("speaker")), tool_call_args)
            except json.JSONDecodeError:
                pass

    # Fallback: try parsing whatever tool-call JSON we accumulated.
    if tool_call_args:
        try:
            args = json.loads(tool_call_args)
            return (_normalize_speaker_name(args.get("speaker")), tool_call_args)
        except json.JSONDecodeError:
            pass

    # Final fallback: Nemotron/XML models emit the call in reasoning_content.
    if reasoning_content:
        m = re.search(
            r"<parameter=speaker>\s*(.*?)\s*</parameter>",
            reasoning_content,
            re.DOTALL | re.IGNORECASE,
        )
        if m:
            return (_normalize_speaker_name(m.group(1)), reasoning_content)

    return ("", tool_call_args or reasoning_content or text_content)


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

    base_url   = llm_cfg.get("base_url", "http://localhost:1234/v1").rstrip("/")
    if not base_url.endswith("/v1"):
        base_url += "/v1"
    api_key    = llm_cfg.get("api_key", "local")
    model_name = llm_cfg.get("model_name", "local-model")
    chunk_size = int(gen_cfg.get("chunk_size") or 3000)
    context_budget = int(chunk_size * 0.8)
    max_tokens = int(gen_cfg.get("max_tokens") or 2048)

    system_prompt = (prompts.get("dialogue_identification_system_prompt") or "").strip() or DEFAULT_SYSTEM_PROMPT

    workers = max(1, int(llm_cfg.get("llm_workers", 1) or 1))

    _log(f"Model: {model_name}  |  Context budget: {context_budget} chars  |  Max tokens: {max_tokens}  |  Workers: {workers}")

    client = OpenAI(base_url=base_url, api_key=api_key, timeout=600)

    # ── Retry-errors mode ──────────────────────────────────────────────────────
    if retry_errors_mode:
        error_ids = set(paragraphs_doc.get("dialogue_errors", []))
        error_paras = [(i, p) for i, p in enumerate(paragraphs)
                       if p.get("dialogue_error") and p["id"] in error_ids]
        _log(f"=== Error Correction: retrying {len(error_paras)} paragraph(s) with up to {retry_max} attempt(s) each (workers={workers}) ===")

        seen_retry: set[str] = set()
        known_speakers: list[str] = []
        for p in paragraphs:
            if not p.get("dialogue_error"):
                for s in (p.get("speakers") or []):
                    if s and s != "NARRATOR" and s not in seen_retry:
                        seen_retry.add(s)
                        known_speakers.append(s)

        retry_lock = threading.Lock()
        fixed_ref = [0]

        def retry_one(idx_para):
            idx, para = idx_para
            para_id = para["id"]
            quotes = extract_dialogue_lines(para["text"])
            if not quotes:
                _log(f"SKIP {para_id}: no quoted text — cannot retry")
                return

            context_text = build_context_window(paragraphs, idx, context_budget)
            with retry_lock:
                char_list = ", ".join(known_speakers) if known_speakers else "(none yet)"

            current_speakers = list(para.get("speakers") or [""] * len(quotes))
            current_errors = list(para.get("quote_errors") or [True] * len(quotes))
            # Pad to match current quote count (text may have changed)
            while len(current_speakers) < len(quotes):
                current_speakers.append("")
                current_errors.append(True)

            for qi, quote_text in enumerate(quotes):
                if not current_errors[qi]:
                    continue  # Already succeeded
                user_msg = (
                    f"PASSAGE CONTEXT:\n{context_text}\n\n"
                    f"KNOWN CHARACTERS SO FAR: {char_list}\n\n"
                    f"QUOTED TEXT:\n\"{quote_text}\"\n\n"
                    "Use the identify_dialogue tool to name the speaker of this quoted text. "
                    "If none of the known characters fits, supply the new character's name. "
                    "If this is a scare quote, figure of speech, or clearly not spoken aloud by a character, "
                    "return NARRATOR as the speaker."
                )
                speaker = ""
                for attempt in range(1, retry_max + 1):
                    try:
                        speaker, raw = _call_identify_dialogue(client, model_name, system_prompt, user_msg, max_tokens)
                        if not speaker:
                            raw_tail = raw[-1024:] if len(raw) > 1024 else raw
                            job = f"{para_id} q{qi}"
                            _log(f"Begin model returned no speaker: {job} : Assign Dialogue\n{quote_text}\n{raw_tail}\nEnd model returned no speaker: {job} : Assign Dialogue")
                    except Exception as e:
                        raw_tail = str(e)[-1024:]
                        job = f"{para_id} q{qi}"
                        _log(f"Begin API call failed: {job} : Assign Dialogue\n{quote_text}\n{raw_tail}\nEnd API call failed: {job} : Assign Dialogue")
                    if speaker:
                        break

                with retry_lock:
                    if speaker:
                        current_speakers[qi] = speaker
                        current_errors[qi] = False
                        if speaker != "NARRATOR" and speaker not in known_speakers:
                            known_speakers.append(speaker)
                            _log(f"New character: {speaker}")
                        _log(f"FIXED {para_id} q{qi}: speaker = {speaker}")
                    else:
                        _log(f"FAILED {para_id} q{qi}: still no speaker after {retry_max} attempt(s)")

            with retry_lock:
                para["speakers"] = current_speakers
                para["quote_errors"] = current_errors
                para["dialogue_error"] = any(current_errors)
                if not para["dialogue_error"]:
                    error_ids.discard(para_id)
                    fixed_ref[0] += 1

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(retry_one, item) for item in error_paras]
            for future in as_completed(futures):
                future.result()

        paragraphs_doc["dialogue_errors"] = list(error_ids)
        _atomic_write(paragraphs_path, paragraphs_doc)
        _log(f"Error correction done. Fixed: {fixed_ref[0]}/{len(error_paras)}. Remaining errors: {len(error_ids)}.")
        return

    # ── Filter dialogue paragraphs — skip any already successfully assigned ──────
    all_dialogue_paras = [(i, p) for i, p in enumerate(paragraphs) if p.get("has_dialogue")]
    dialogue_paras = [
        (i, p) for i, p in all_dialogue_paras
        if not p.get("speakers") or p.get("dialogue_error")
    ]
    total = len(dialogue_paras)
    already_done = len(all_dialogue_paras) - total

    if not all_dialogue_paras:
        _log("No dialogue paragraphs found. Nothing to assign.")
        paragraphs_doc["dialogue_assignment_complete"] = True
        paragraphs_doc["dialogue_errors"] = []
        _atomic_write(paragraphs_path, paragraphs_doc)
        return

    if total == 0:
        _log("All dialogue paragraphs already assigned. Nothing to do.")
        all_errors = [p["id"] for _, p in all_dialogue_paras if p.get("dialogue_error")]
        paragraphs_doc["dialogue_assignment_complete"] = True
        paragraphs_doc["dialogue_errors"] = all_errors
        _atomic_write(paragraphs_path, paragraphs_doc)
        return

    if already_done:
        _log(f"Resuming: {already_done} already assigned, {total} remaining.")
    _log(f"Assigning speakers to {total} dialogue paragraphs (workers={workers})...")

    # Pre-populate known_speakers from any paragraphs already successfully assigned.
    seen_speakers: set[str] = set()
    known_speakers: list[str] = []
    for _, p in all_dialogue_paras:
        if p.get("dialogue_error"):
            continue
        for s in (p.get("speakers") or []):
            if s and s != "NARRATOR" and s not in seen_speakers:
                seen_speakers.add(s)
                known_speakers.append(s)

    errors: list[str] = []
    lock = threading.Lock()
    done_ref = [0]
    task_start_time = time.time()

    def process_para(idx_para):
        idx, para = idx_para
        para_id = para["id"]

        context_text = build_context_window(paragraphs, idx, context_budget)
        quotes = extract_dialogue_lines(para["text"])
        if not quotes:
            # has_dialogue was set but no quoted text found — treat as error
            with lock:
                para["speakers"] = []
                para["quote_errors"] = []
                para["dialogue_error"] = True
                errors.append(para_id)
                done_ref[0] += 1
                _log(f"ERROR {para_id}: tagged as dialogue but no quoted text found")
                _maybe_progress(done_ref[0], total, task_start_time)
                if done_ref[0] % 10 == 0:
                    _atomic_write(paragraphs_path, paragraphs_doc)
            return

        with lock:
            char_list = ", ".join(known_speakers) if known_speakers else "(none yet)"

        speakers = []
        quote_errors = []
        for qi, quote_text in enumerate(quotes):
            user_msg = (
                f"PASSAGE CONTEXT:\n{context_text}\n\n"
                f"KNOWN CHARACTERS SO FAR: {char_list}\n\n"
                f"QUOTED TEXT:\n\"{quote_text}\"\n\n"
                "Use the identify_dialogue tool to name the speaker of this quoted text. "
                "If none of the known characters fits, supply the new character's name. "
                "If this is a scare quote, figure of speech, or clearly not spoken aloud by a character, "
                "return NARRATOR as the speaker."
            )
            speaker = ""
            try:
                speaker, raw = _call_identify_dialogue(client, model_name, system_prompt, user_msg, max_tokens)
                if not speaker:
                    raw_tail = raw[-1024:] if len(raw) > 1024 else raw
                    job = f"{para_id} q{qi}"
                    _log(f"Begin model returned no speaker: {job} : Assign Dialogue\n{quote_text}\n{raw_tail}\nEnd model returned no speaker: {job} : Assign Dialogue")
            except Exception as e:
                raw_tail = str(e)[-1024:]
                job = f"{para_id} q{qi}"
                _log(f"Begin API call failed: {job} : Assign Dialogue\n{quote_text}\n{raw_tail}\nEnd API call failed: {job} : Assign Dialogue")
            speakers.append(speaker)
            quote_errors.append(not bool(speaker))

        with lock:
            para["speakers"] = speakers
            para["quote_errors"] = quote_errors
            para["dialogue_error"] = any(quote_errors)
            for s in speakers:
                if s and s != "NARRATOR" and s not in known_speakers:
                    known_speakers.append(s)
                    _log(f"New character: {s}")
            if para.get("dialogue_error") and para_id not in errors:
                errors.append(para_id)
            done_ref[0] += 1
            _maybe_progress(done_ref[0], total, task_start_time)
            if done_ref[0] % 10 == 0:
                _atomic_write(paragraphs_path, paragraphs_doc)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(process_para, item) for item in dialogue_paras]
        for future in as_completed(futures):
            future.result()

    # ── Save results ───────────────────────────────────────────────────────────
    # Collect errors from all dialogue paragraphs (including any from prior runs).
    all_errors = [p["id"] for _, p in all_dialogue_paras if p.get("dialogue_error")]
    paragraphs_doc["dialogue_assignment_complete"] = True
    paragraphs_doc["dialogue_errors"] = all_errors
    _atomic_write(paragraphs_path, paragraphs_doc)

    assigned = total - len(errors)
    _log(f"Done. Assigned: {assigned}/{total}. Errors: {len(errors)}.")
    if all_errors:
        _log(f"Paragraphs with errors: {', '.join(all_errors)}")


def _format_eta(start_time: float, done: int, total: int) -> str:
    if done <= 0 or total <= done:
        return ""
    elapsed = time.time() - start_time
    remaining_seconds = (elapsed / done) * (total - done)
    if remaining_seconds < 60:
        return f" (~{int(remaining_seconds)}s remaining)"
    minutes = int(remaining_seconds // 60)
    if minutes >= 60:
        return f" (~{minutes // 60}h {minutes % 60}m remaining)"
    return f" (~{minutes}m {int(remaining_seconds % 60)}s remaining)"


def _dots(done: int, total: int) -> str:
    filled = min(10, int(done / total * 10)) if total > 0 else 0
    return "[" + "•" * filled + "·" * (10 - filled) + "]"


def _maybe_progress(done: int, total: int, start_time: float):
    if done % 10 == 0 or done == total:
        eta = _format_eta(start_time, done, total)
        _progress(done, total, f"{_dots(done, total)} Assigned {done}/{total} dialogue paragraphs...{eta}")


def _atomic_write(path: str, data: dict):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


if __name__ == "__main__":
    main()
