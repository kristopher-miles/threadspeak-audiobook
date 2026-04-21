#!/usr/bin/env python3
"""
Assign speakers to dialogue paragraphs using adaptive structured LLM output.

For each paragraph tagged has_dialogue=True in paragraphs.json, this script:
  - Builds a context window centred on that paragraph (chunk_size * 0.8 chars)
  - Extracts the quoted dialogue lines from the paragraph
  - Requests a structured speaker attribution response (tool mode when available, JSON fallback otherwise)
  - Stores the returned speaker name in the paragraph, or marks it as an error

Results are written back into the project SQLite store.
"""
import argparse
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from llm import DIALOGUE_SPEAKER_CONTRACT, LLMClientFactory, LLMRuntimeConfig, get_llm_gateway
from stdio_utils import configure_utf8_stdio
from scripts.legacy_cli_project import infer_project_root, import_project_document_from_path
from script_provider import open_project_script_store

configure_utf8_stdio()

TASK_PROGRESS_PREFIX = "__TASK_PROGRESS__:"
_LLM_CLIENT_FACTORY = LLMClientFactory()
_STRUCTURED_LLM_SERVICE = get_llm_gateway()

# Matches straight (") or curly (\u201c / \u201d) double-quotes enclosing ≥2 chars.
QUOTE_RE = re.compile(r'["\u201c]([^"\u201d]{2,})["\u201d]', re.DOTALL)

DEFAULT_SYSTEM_PROMPT = (
    "You are a dialogue attribution specialist helping to build an audiobook script. "
    "Your sole task is to identify who speaks a given piece of dialogue based on the surrounding narrative."
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


def apply_narrated_dialogue_assignments(paragraphs_doc: dict) -> tuple[int, int]:
    """Assign every detected quote in dialogue paragraphs to NARRATOR."""
    paragraphs = paragraphs_doc.get("paragraphs") or []
    dialogue_para_count = 0
    quote_count = 0

    for para in paragraphs:
        if not para.get("has_dialogue"):
            continue
        dialogue_para_count += 1
        quotes = extract_dialogue_lines(str(para.get("text") or ""))
        para["speakers"] = ["NARRATOR"] * len(quotes)
        para["quote_errors"] = [False] * len(quotes)
        para["dialogue_error"] = False
        quote_count += len(quotes)

    paragraphs_doc["dialogue_assignment_complete"] = True
    paragraphs_doc["dialogue_errors"] = []
    return dialogue_para_count, quote_count


def _extract_speaker_from_text(text: str) -> str:
    raw = str(text or "").strip()
    if not raw:
        return ""
    try:
        payload = json.loads(raw)
    except Exception:
        payload = None
    if isinstance(payload, dict):
        return str(payload.get("speaker") or "").strip()
    match = re.search(r'"speaker"\s*:\s*"((?:[^"\\]|\\.)*)"', raw)
    if match:
        return match.group(1).strip()
    return raw.splitlines()[0].strip()


def _call_identify_dialogue(client, runtime: LLMRuntimeConfig, system_prompt: str, user_msg: str, max_tokens: int) -> tuple:
    """
    Resolve dialogue speaker through adaptive tool/json structured mode.
    Returns (normalized speaker name, raw_response, llm_mode, tool_call_observed),
    or ("", raw_response, llm_mode, tool_call_observed) on failure.
    """
    result = _STRUCTURED_LLM_SERVICE.run(
        client=client,
        runtime=runtime,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg},
        ],
        contract=DIALOGUE_SPEAKER_CONTRACT,
        temperature=0.1,
        max_tokens=max_tokens,
        use_streaming_tool=True,
        reasoning_parameter_name="speaker",
    )
    payload = result.parsed if isinstance(result.parsed, dict) else None
    raw_speaker = str((payload or {}).get("speaker") or "").strip()
    if not raw_speaker and result.mode == "json":
        raw_speaker = _extract_speaker_from_text(result.text)
    speaker = _normalize_speaker_name(raw_speaker)
    return (
        speaker,
        result.raw_payload or result.text,
        result.mode,
        bool(result.tool_call_observed),
    )


def main():
    parser = argparse.ArgumentParser(description="Assign dialogue speakers to persisted paragraphs.")
    parser.add_argument("paragraphs_path", nargs="?")
    parser.add_argument("config_path")
    parser.add_argument("--project-root", dest="project_root")
    parser.add_argument("--retry-errors", dest="retry_errors", type=int, default=0)
    parser.add_argument("--narrated", action="store_true")
    args = parser.parse_args()

    paragraphs_path = args.paragraphs_path
    config_path = args.config_path
    project_root = str(args.project_root or "").strip() or infer_project_root(paragraphs_path)
    retry_max = max(0, int(args.retry_errors or 0))
    retry_errors_mode = retry_max > 0

    if not project_root and not paragraphs_path:
        _log("Usage: assign_dialogue.py <paragraphs_path> <config_path> OR --project-root <root>")
        sys.exit(1)

    if paragraphs_path and not args.project_root:
        _log(f"Legacy file-mode detected; importing paragraphs into project store at {project_root}")
        try:
            import_project_document_from_path(
                project_root,
                "paragraphs",
                paragraphs_path,
                reason="legacy_assign_dialogue_import",
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
        _log("ERROR: persisted paragraph state contains no paragraphs. Run 'Process Paragraphs' first.")
        sys.exit(1)

    if args.narrated:
        dialogue_para_count, quote_count = apply_narrated_dialogue_assignments(paragraphs_doc)
        _persist_paragraphs_doc(paragraphs_path, project_root, paragraphs_doc)
        _log(
            "Narrated mode enabled. "
            f"Assigned NARRATOR to {quote_count} quote(s) across {dialogue_para_count} dialogue paragraph(s)."
        )
        return

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

    runtime = LLMRuntimeConfig.from_dict(
        llm_cfg,
        default_base_url="http://localhost:1234/v1",
        default_model_name="local-model",
        default_timeout=600.0,
    )
    model_name = runtime.model_name
    chunk_size = int(gen_cfg.get("chunk_size") or 3000)
    context_budget = int(chunk_size * 0.8)
    max_tokens = int(gen_cfg.get("max_tokens") or 2048)

    system_prompt = (prompts.get("dialogue_identification_system_prompt") or "").strip() or DEFAULT_SYSTEM_PROMPT

    workers = runtime.llm_workers

    _log(f"Model: {model_name}  |  Context budget: {context_budget} chars  |  Max tokens: {max_tokens}  |  Workers: {workers}")

    client = _LLM_CLIENT_FACTORY.create_client(runtime)

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
                    "Name the speaker of this quoted text. "
                    "If none of the known characters fits, supply the new character's name. "
                    "If this is a scare quote, figure of speech, or clearly not spoken aloud by a character, "
                    "return NARRATOR as the speaker."
                )
                speaker = ""
                for attempt in range(1, retry_max + 1):
                    try:
                        speaker, raw, llm_mode, llm_tool_call_observed = _call_identify_dialogue(
                            client, runtime, system_prompt, user_msg, max_tokens
                        )
                        _log(f"LLM telemetry: llm_mode={llm_mode} tool_call_observed={llm_tool_call_observed}")
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
        _persist_paragraphs_doc(paragraphs_path, project_root, paragraphs_doc)
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
        _persist_paragraphs_doc(paragraphs_path, project_root, paragraphs_doc)
        return

    if total == 0:
        _log("All dialogue paragraphs already assigned. Nothing to do.")
        all_errors = [p["id"] for _, p in all_dialogue_paras if p.get("dialogue_error")]
        paragraphs_doc["dialogue_assignment_complete"] = True
        paragraphs_doc["dialogue_errors"] = all_errors
        _persist_paragraphs_doc(paragraphs_path, project_root, paragraphs_doc)
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
                    _persist_paragraphs_doc(paragraphs_path, project_root, paragraphs_doc)
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
                "Name the speaker of this quoted text. "
                "If none of the known characters fits, supply the new character's name. "
                "If this is a scare quote, figure of speech, or clearly not spoken aloud by a character, "
                "return NARRATOR as the speaker."
            )
            speaker = ""
            try:
                speaker, raw, llm_mode, llm_tool_call_observed = _call_identify_dialogue(
                    client, runtime, system_prompt, user_msg, max_tokens
                )
                _log(f"LLM telemetry: llm_mode={llm_mode} tool_call_observed={llm_tool_call_observed}")
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
                _persist_paragraphs_doc(paragraphs_path, project_root, paragraphs_doc)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(process_para, item) for item in dialogue_paras]
        for future in as_completed(futures):
            future.result()

    # ── Save results ───────────────────────────────────────────────────────────
    # Collect errors from all dialogue paragraphs (including any from prior runs).
    all_errors = [p["id"] for _, p in all_dialogue_paras if p.get("dialogue_error")]
    paragraphs_doc["dialogue_assignment_complete"] = True
    paragraphs_doc["dialogue_errors"] = all_errors
    _persist_paragraphs_doc(paragraphs_path, project_root, paragraphs_doc)

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


def _persist_paragraphs_doc(paragraphs_path: str, project_root: str | None, data: dict):
    store = open_project_script_store(project_root)
    try:
        store.replace_project_document("paragraphs", data, reason="assign_dialogue", wait=True)
    finally:
        store.stop()


if __name__ == "__main__":
    main()
