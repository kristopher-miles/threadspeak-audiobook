#!/usr/bin/env python3
"""
Extract emotional temperament (mood/delivery) for all paragraphs using adaptive structured LLM output.

Three passes are performed:

  Pass 1 — Narrator tone, narration-only paragraphs (has_dialogue=False):
    Sends full paragraph text. Stores result in para['tone'].

  Pass 2 — Narrator tone, dialogue paragraphs (has_dialogue=True):
    Sends paragraph with all quoted dialogue replaced by [dialogue ignored].
    Instructs the model to describe only the narration delivery.
    Stores result in para['tone'].

  Pass 3 — Dialogue mood, dialogue paragraphs only:
    Sends only the extracted dialogue lines.
    Instructs the model to describe delivery as spoken by the identified speaker.
    Stores result in para['dialogue_mood'].

All results are written back into the project SQLite store after each pass.
"""
import argparse
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from llm import LLMClientFactory, LLMRuntimeConfig, SENTIMENT_MOOD_CONTRACT, get_llm_gateway
from stdio_utils import configure_utf8_stdio
from scripts.legacy_cli_project import infer_project_root, import_project_document_from_path
from script_provider import open_project_script_store
from source_document import is_structural_silence_text

configure_utf8_stdio()

TASK_PROGRESS_PREFIX = "__TASK_PROGRESS__:"
_LLM_CLIENT_FACTORY = LLMClientFactory()
_STRUCTURED_LLM_SERVICE = get_llm_gateway()

# Matches straight (") or curly (\u201c / \u201d) double-quotes enclosing ≥2 chars.
QUOTE_RE = re.compile(r'["\u201c][^"\u201d]{2,}["\u201d]', re.DOTALL)
WORD_RE = re.compile(r"\b\w+\b", re.UNICODE)

DEFAULT_SYSTEM_PROMPT = (
    "You are a narrative tone specialist helping to produce an audiobook. "
    "Your sole task is to identify the emotional sentiment and delivery style "
    "of a given paragraph of prose."
)


def is_structural_silence_paragraph(para: dict) -> bool:
    if not isinstance(para, dict):
        return False
    if para.get("is_structural_silence") is True:
        return True
    return is_structural_silence_text(para.get("text") or "")


def _log(msg: str):
    print(msg, flush=True)


def _progress(current: int, total: int, message: str):
    print(
        TASK_PROGRESS_PREFIX + json.dumps({"current": current, "total": total, "message": message}),
        flush=True,
    )


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


def count_words(text: str) -> int:
    return len(WORD_RE.findall(text or ""))


def build_temperament_context(paragraphs: list, target_idx: int, budget: int, minimum_words: int) -> str:
    """
    Build a backward-only context block for temperament extraction.

    The target paragraph is always the final paragraph in the returned block.
    If the target paragraph already meets the minimum word target, use it alone.
    Otherwise prepend earlier paragraphs until the minimum word target is met or
    another paragraph would exceed the character budget.
    """
    target_text = paragraphs[target_idx]["text"]
    parts = [target_text]
    used_chars = len(target_text)
    used_words = count_words(target_text)

    if used_words >= minimum_words:
        return target_text

    for idx in range(target_idx - 1, -1, -1):
        if is_structural_silence_paragraph(paragraphs[idx]):
            continue
        previous_text = paragraphs[idx]["text"]
        separator_chars = 2 if parts else 0
        if used_chars + separator_chars + len(previous_text) > budget:
            break
        parts.insert(0, previous_text)
        used_chars += separator_chars + len(previous_text)
        used_words += count_words(previous_text)
        if used_words >= minimum_words:
            break

    return "\n\n".join(parts)


def strip_dialogue(text: str) -> str:
    """Replace all quoted dialogue runs with [dialogue ignored]."""
    return QUOTE_RE.sub("[dialogue ignored]", text)


def extract_dialogue_only(text: str) -> str:
    """Return only the quoted dialogue portions joined by newlines."""
    return "\n".join(QUOTE_RE.findall(text))


def split_paragraphs_for_temperament(paragraphs: list) -> tuple[list[tuple[int, dict]], list[tuple[int, dict]]]:
    narration = []
    dialogue = []
    for idx, para in enumerate(paragraphs or []):
        if is_structural_silence_paragraph(para):
            continue
        if para.get("has_dialogue"):
            dialogue.append((idx, para))
        else:
            narration.append((idx, para))
    return narration, dialogue


def _extract_mood_from_text(text: str) -> str:
    raw = str(text or "").strip()
    if not raw:
        return ""
    try:
        payload = json.loads(raw)
    except Exception:
        payload = None
    if isinstance(payload, dict):
        return str(payload.get("mood") or "").strip()
    match = re.search(r'"mood"\s*:\s*"((?:[^"\\]|\\.)*)"', raw)
    if match:
        return match.group(1).strip()
    return raw.splitlines()[0].strip()


def call_sentiment(client, runtime: LLMRuntimeConfig, system_prompt: str, user_msg: str, max_tokens: int) -> tuple:
    """
    Call the LLM through adaptive tool/json structured mode.
    Returns (mood, raw_response, llm_mode, tool_call_observed),
    or ("", raw_response, llm_mode, tool_call_observed) on failure.
    """
    result = _STRUCTURED_LLM_SERVICE.run(
        client=client,
        runtime=runtime,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg},
        ],
        contract=SENTIMENT_MOOD_CONTRACT,
        temperature=0.1,
        max_tokens=max_tokens,
        use_streaming_tool=True,
        reasoning_parameter_name="mood",
    )
    payload = result.parsed if isinstance(result.parsed, dict) else None
    mood = str((payload or {}).get("mood") or "").strip()
    if not mood and result.mode == "json":
        mood = _extract_mood_from_text(result.text)
    return (
        mood,
        result.raw_payload or result.text,
        result.mode,
        bool(result.tool_call_observed),
    )


def retry_failed_temperament_errors(
    paragraphs_doc: dict,
    *,
    paragraphs_path: str | None,
    project_root: str | None,
    client,
    runtime: LLMRuntimeConfig,
    system_prompt: str,
    max_tokens: int,
    context_budget: int,
    temperament_words: int,
    retry_max: int,
) -> None:
    paragraphs = paragraphs_doc.get("paragraphs", [])
    all_narration_paras, all_dialogue_paras = split_paragraphs_for_temperament(paragraphs)
    tone_error_ids = {
        str(item).strip()
        for item in (paragraphs_doc.get("temperament_errors") or [])
        if str(item).strip()
    }
    dialogue_mood_error_ids = {
        str(item).strip()
        for item in (paragraphs_doc.get("dialogue_mood_errors") or [])
        if str(item).strip()
    }
    if not tone_error_ids:
        tone_error_ids = {
            str(para.get("id") or "").strip()
            for para in paragraphs
            if isinstance(para, dict) and para.get("temperament_error") and str(para.get("id") or "").strip()
        }
    if not dialogue_mood_error_ids:
        dialogue_mood_error_ids = {
            str(para.get("id") or "").strip()
            for para in paragraphs
            if isinstance(para, dict) and para.get("dialogue_mood_error") and str(para.get("id") or "").strip()
        }

    retry_narration_paras = [
        (idx, para) for idx, para in all_narration_paras
        if para.get("temperament_error") and para.get("id") in tone_error_ids
    ]
    retry_dialogue_tone_paras = [
        (idx, para) for idx, para in all_dialogue_paras
        if para.get("temperament_error") and para.get("id") in tone_error_ids
    ]
    retry_dialogue_mood_paras = [
        (idx, para) for idx, para in all_dialogue_paras
        if para.get("dialogue_mood_error") and para.get("id") in dialogue_mood_error_ids
    ]

    if not retry_narration_paras and not retry_dialogue_tone_paras and not retry_dialogue_mood_paras:
        _log("Error correction requested, but no temperament errors remain.")
        _checkpoint_write(paragraphs_path, project_root, paragraphs_doc, complete=True)
        return

    _log(
        "=== Error Correction: "
        f"{len(retry_narration_paras) + len(retry_dialogue_tone_paras)} tone paragraph(s), "
        f"{len(retry_dialogue_mood_paras)} dialogue-mood paragraph(s), "
        f"up to {retry_max} attempt(s) each ==="
    )

    def resolve_tone(context_text: str, target_text: str, *, narrator_only: bool) -> str:
        if narrator_only:
            user_msg = (
                f"PASSAGE CONTEXT:\n{context_text}\n\n"
                f"TARGET PARAGRAPH:\n{target_text}\n\n"
                "Identify the spoken emotional sentiment and delivery of the given narration "
                "portion of the paragraph. Ignore any dialogue when describing the emotional delivery:"
            )
        else:
            user_msg = (
                f"PASSAGE CONTEXT:\n{context_text}\n\n"
                f"TARGET PARAGRAPH:\n{target_text}\n\n"
                "Identify the spoken emotional sentiment and delivery of the given paragraph."
            )
        mood = ""
        for _attempt in range(1, retry_max + 1):
            try:
                mood, raw, _llm_mode, _tool_call_observed = call_sentiment(
                    client, runtime, system_prompt, user_msg, max_tokens
                )
                if not mood:
                    raw_tail = raw[-1024:] if len(raw) > 1024 else raw
                    _log(
                        "Begin model did not return a mood during retry\n"
                        f"{target_text}\n{raw_tail}\n"
                        "End model did not return a mood during retry"
                    )
            except Exception as e:
                raw_tail = str(e)[-1024:]
                _log(f"Begin API call failed during retry\n{target_text}\n{raw_tail}\nEnd API call failed during retry")
            if mood:
                return mood
        return ""

    for idx, para in retry_narration_paras:
        para_id = para["id"]
        context_text = build_temperament_context(paragraphs, idx, context_budget, temperament_words)
        mood = resolve_tone(context_text, para["text"], narrator_only=False)
        if mood:
            para["tone"] = mood
            para["temperament_error"] = False
            _log(f"FIXED {para_id}: narrator tone = {mood}")
        else:
            para["tone"] = ""
            para["temperament_error"] = True
            _log(f"FAILED {para_id}: narrator tone still missing after {retry_max} attempt(s)")
        _checkpoint_write(paragraphs_path, project_root, paragraphs_doc, complete=False)

    for idx, para in retry_dialogue_tone_paras:
        para_id = para["id"]
        context_text = build_temperament_context(paragraphs, idx, context_budget, temperament_words)
        narration_text = strip_dialogue(para["text"])
        mood = resolve_tone(context_text, narration_text, narrator_only=True)
        if mood:
            para["tone"] = mood
            para["temperament_error"] = False
            _log(f"FIXED {para_id}: narrator tone = {mood}")
        else:
            para["tone"] = ""
            para["temperament_error"] = True
            _log(f"FAILED {para_id}: narrator tone still missing after {retry_max} attempt(s)")
        _checkpoint_write(paragraphs_path, project_root, paragraphs_doc, complete=False)

    for idx, para in retry_dialogue_mood_paras:
        para_id = para["id"]
        speakers_list = para.get("speakers") or []
        context_text = build_temperament_context(paragraphs, idx, context_budget, temperament_words)
        quotes = QUOTE_RE.findall(para["text"])
        narration_text = strip_dialogue(para["text"])

        if not quotes:
            para["dialogue_moods"] = []
            para["quote_mood_errors"] = []
            para["dialogue_mood_error"] = True
            _log(f"FAILED {para_id}: no dialogue text found during mood retry")
            _checkpoint_write(paragraphs_path, project_root, paragraphs_doc, complete=False)
            continue

        moods = list(para.get("dialogue_moods") or [""] * len(quotes))
        mood_errors = list(para.get("quote_mood_errors") or [True] * len(quotes))
        while len(moods) < len(quotes):
            moods.append("")
        while len(mood_errors) < len(quotes):
            mood_errors.append(True)

        for qi, quote_str in enumerate(quotes):
            if not mood_errors[qi]:
                continue
            speaker = (speakers_list[qi] if qi < len(speakers_list) else "").strip() or "Unknown Speaker"
            inner = quote_str.strip('"\u201c\u201d')

            if speaker == "NARRATOR":
                user_msg = (
                    f"PASSAGE CONTEXT:\n{context_text}\n\n"
                    f"TARGET PARAGRAPH:\n{narration_text}\n\n"
                    "Identify the spoken emotional sentiment and delivery of the given narration "
                    "portion of the paragraph. Ignore any dialogue when describing the emotional delivery:"
                )
            else:
                user_msg = (
                    f"PASSAGE CONTEXT:\n{context_text}\n\n"
                    f"Identify the correct delivery emotion of the following dialogue as spoken by {speaker}.\n\n"
                    f"DIALOGUE:\n{inner}"
                )

            mood = ""
            for _attempt in range(1, retry_max + 1):
                try:
                    mood, raw, _llm_mode, _tool_call_observed = call_sentiment(
                        client, runtime, system_prompt, user_msg, max_tokens
                    )
                    if not mood:
                        raw_tail = raw[-1024:] if len(raw) > 1024 else raw
                        _log(
                            f"Begin model did not return a dialogue mood during retry: {para_id} q{qi}\n"
                            f"{inner}\n{raw_tail}\n"
                            f"End model did not return a dialogue mood during retry: {para_id} q{qi}"
                        )
                except Exception as e:
                    raw_tail = str(e)[-1024:]
                    _log(
                        f"Begin API call failed during dialogue mood retry: {para_id} q{qi}\n"
                        f"{inner}\n{raw_tail}\n"
                        f"End API call failed during dialogue mood retry: {para_id} q{qi}"
                    )
                if mood:
                    break

            if mood:
                moods[qi] = mood
                mood_errors[qi] = False
                _log(f"FIXED {para_id} q{qi}: dialogue mood = {mood}")
            else:
                _log(f"FAILED {para_id} q{qi}: dialogue mood still missing after {retry_max} attempt(s)")

        para["dialogue_moods"] = moods
        para["quote_mood_errors"] = mood_errors
        para["dialogue_mood_error"] = any(mood_errors)
        _checkpoint_write(paragraphs_path, project_root, paragraphs_doc, complete=False)

    _checkpoint_write(paragraphs_path, project_root, paragraphs_doc, complete=True)
    remaining_tone_errors = paragraphs_doc.get("temperament_errors") or []
    remaining_dialogue_mood_errors = paragraphs_doc.get("dialogue_mood_errors") or []
    _log(
        "Error correction complete. "
        f"Remaining tone errors: {len(remaining_tone_errors)}. "
        f"Remaining dialogue mood errors: {len(remaining_dialogue_mood_errors)}."
    )


def main():
    parser = argparse.ArgumentParser(description="Extract temperament into persisted paragraphs.")
    parser.add_argument("paragraphs_path", nargs="?")
    parser.add_argument("config_path")
    parser.add_argument("--project-root", dest="project_root")
    parser.add_argument("--retry-errors", dest="retry_errors", type=int, default=0)
    args = parser.parse_args()

    paragraphs_path = args.paragraphs_path
    config_path = args.config_path
    project_root = str(args.project_root or "").strip() or infer_project_root(paragraphs_path)
    retry_max = max(0, int(args.retry_errors or 0))
    if not project_root and not paragraphs_path:
        _log("Usage: extract_temperament.py <paragraphs_path> <config_path> OR --project-root <root>")
        sys.exit(1)

    if paragraphs_path and not args.project_root:
        _log(f"Legacy file-mode detected; importing paragraphs into project store at {project_root}")
        try:
            import_project_document_from_path(
                project_root,
                "paragraphs",
                paragraphs_path,
                reason="legacy_extract_temperament_import",
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
    temperament_words = int(gen_cfg.get("temperament_words") or 150)

    system_prompt = (prompts.get("temperament_extraction_system_prompt") or "").strip() or DEFAULT_SYSTEM_PROMPT

    workers = runtime.llm_workers

    _log(
        f"Model: {model_name}  |  Context budget: {context_budget} chars  |  "
        f"Temperament words: {temperament_words}  |  Max tokens: {max_tokens}  |  Workers: {workers}"
    )

    client = _LLM_CLIENT_FACTORY.create_client(runtime)

    if retry_max > 0:
        retry_failed_temperament_errors(
            paragraphs_doc,
            paragraphs_path=paragraphs_path,
            project_root=project_root,
            client=client,
            runtime=runtime,
            system_prompt=system_prompt,
            max_tokens=max_tokens,
            context_budget=context_budget,
            temperament_words=temperament_words,
            retry_max=retry_max,
        )
        return

    all_narration_paras, all_dialogue_paras = split_paragraphs_for_temperament(paragraphs)

    tone_errors: list[str] = []
    dialogue_mood_errors: list[str] = []

    _checkpoint_write(paragraphs_path, project_root, paragraphs_doc, complete=False)

    # ══════════════════════════════════════════════════════════════════════════
    # PASS 1 — Narrator tone for narration-only paragraphs
    # ══════════════════════════════════════════════════════════════════════════
    # Skip paragraphs that already have a valid tone from a prior run.
    narration_paras = [
        (i, p) for i, p in all_narration_paras
        if not p.get("tone") or p.get("temperament_error")
    ]
    total1 = len(narration_paras)
    already1 = len(all_narration_paras) - total1
    if already1:
        _log(f"[Pass 1/3] Resuming: {already1} already done, {total1} remaining.")
    _log(f"[Pass 1/3] Narrator tone for {total1} narration-only paragraphs (workers={workers})...")

    lock1 = threading.Lock()
    done1 = [0]
    pass1_start = time.time()

    def process_pass1(idx_para):
        idx, para = idx_para
        para_id = para["id"]
        context_text = build_temperament_context(paragraphs, idx, context_budget, temperament_words)

        user_msg = (
            f"PASSAGE CONTEXT:\n{context_text}\n\n"
            f"TARGET PARAGRAPH:\n{para['text']}\n\n"
            "Identify the spoken emotional sentiment and delivery of the given paragraph."
        )

        mood = ""
        try:
            mood, raw, llm_mode, llm_tool_call_observed = call_sentiment(
                client, runtime, system_prompt, user_msg, max_tokens
            )
            if not mood:
                raw_tail = raw[-1024:] if len(raw) > 1024 else raw
                _log(f"Begin model did not return a mood: {para_id} : Extract Temperament\n{para['text']}\n{raw_tail}\nEnd model did not return a mood: {para_id} : Extract Temperament")
        except Exception as e:
            raw_tail = str(e)[-1024:]
            _log(f"Begin API call failed: {para_id} : Extract Temperament\n{para['text']}\n{raw_tail}\nEnd API call failed: {para_id} : Extract Temperament")

        with lock1:
            if mood:
                para["tone"] = mood
                para["temperament_error"] = False
                _log(f"[{para_id}] Narrator tone: {mood}")
            else:
                para["tone"] = ""
                para["temperament_error"] = True
                if para_id not in tone_errors:
                    tone_errors.append(para_id)
            done1[0] += 1
            if done1[0] % 10 == 0 or done1[0] == total1:
                eta = _format_eta(pass1_start, done1[0], total1)
                _progress(done1[0], total1, f"{_dots(done1[0], total1)} [Pass 1/3] Narrator tone: {done1[0]}/{total1}{eta}...")
            _checkpoint_write(paragraphs_path, project_root, paragraphs_doc, complete=False)

    if total1 > 0:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(process_pass1, item) for item in narration_paras]
            for future in as_completed(futures):
                future.result()

    _log(f"[Pass 1/3] Done. {total1 - len(tone_errors)}/{total1} succeeded.")
    _checkpoint_write(paragraphs_path, project_root, paragraphs_doc, complete=False)

    # ══════════════════════════════════════════════════════════════════════════
    # PASS 2 — Narrator tone for dialogue paragraphs (dialogue stripped out)
    # ══════════════════════════════════════════════════════════════════════════
    # Skip paragraphs that already have a valid tone from a prior run.
    dialogue_paras_p2 = [
        (i, p) for i, p in all_dialogue_paras
        if not p.get("tone") or p.get("temperament_error")
    ]
    total2 = len(dialogue_paras_p2)
    already2 = len(all_dialogue_paras) - total2
    if already2:
        _log(f"[Pass 2/3] Resuming: {already2} already done, {total2} remaining.")
    _log(f"[Pass 2/3] Narrator tone for {total2} dialogue paragraphs (dialogue text removed, workers={workers})...")

    lock2 = threading.Lock()
    done2 = [0]
    pass2_start = time.time()

    def process_pass2(idx_para):
        idx, para = idx_para
        para_id = para["id"]
        context_text = build_temperament_context(paragraphs, idx, context_budget, temperament_words)
        narration_text = strip_dialogue(para["text"])

        user_msg = (
            f"PASSAGE CONTEXT:\n{context_text}\n\n"
            f"TARGET PARAGRAPH:\n{narration_text}\n\n"
            "Identify the spoken emotional sentiment and delivery of the given narration "
            "portion of the paragraph. Ignore any dialogue when describing the emotional delivery:"
        )

        mood = ""
        try:
            mood, raw, llm_mode, llm_tool_call_observed = call_sentiment(
                client, runtime, system_prompt, user_msg, max_tokens
            )
            if not mood:
                raw_tail = raw[-1024:] if len(raw) > 1024 else raw
                _log(f"Begin model did not return a narrator mood: {para_id} : Extract Temperament\n{narration_text}\n{raw_tail}\nEnd model did not return a narrator mood: {para_id} : Extract Temperament")
        except Exception as e:
            raw_tail = str(e)[-1024:]
            _log(f"Begin API call failed: {para_id} : Extract Temperament\n{narration_text}\n{raw_tail}\nEnd API call failed: {para_id} : Extract Temperament")

        with lock2:
            if mood:
                para["tone"] = mood
                para["temperament_error"] = False
                _log(f"[{para_id}] Narrator tone: {mood}")
            else:
                para["tone"] = ""
                para["temperament_error"] = True
                if para_id not in tone_errors:
                    tone_errors.append(para_id)
            done2[0] += 1
            if done2[0] % 10 == 0 or done2[0] == total2:
                eta = _format_eta(pass2_start, done2[0], total2)
                _progress(done2[0], total2, f"{_dots(done2[0], total2)} [Pass 2/3] Narrator tone (dialogue): {done2[0]}/{total2}{eta}...")
            _checkpoint_write(paragraphs_path, project_root, paragraphs_doc, complete=False)

    if total2 > 0:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(process_pass2, item) for item in dialogue_paras_p2]
            for future in as_completed(futures):
                future.result()

    _log(f"[Pass 2/3] Done. {total2 - sum(1 for e in tone_errors if e in [p['id'] for _, p in dialogue_paras_p2])}/{total2} succeeded.")
    _checkpoint_write(paragraphs_path, project_root, paragraphs_doc, complete=False)

    # ══════════════════════════════════════════════════════════════════════════
    # PASS 3 — Dialogue mood per individual quote in dialogue paragraphs
    # ══════════════════════════════════════════════════════════════════════════
    # Skip paragraphs that already have a valid dialogue_moods array from a prior run.
    dialogue_paras_p3 = [
        (i, p) for i, p in all_dialogue_paras
        if not p.get("dialogue_moods") or p.get("dialogue_mood_error")
    ]
    total3 = len(dialogue_paras_p3)
    already3 = len(all_dialogue_paras) - total3
    if already3:
        _log(f"[Pass 3/3] Resuming: {already3} already done, {total3} remaining.")
    _log(f"[Pass 3/3] Dialogue mood per quote for {total3} dialogue paragraphs (workers={workers})...")

    lock3 = threading.Lock()
    done3 = [0]
    pass3_start = time.time()

    def process_pass3(idx_para):
        idx, para = idx_para
        para_id = para["id"]
        speakers_list = para.get("speakers") or []
        context_text = build_temperament_context(paragraphs, idx, context_budget, temperament_words)
        quotes = QUOTE_RE.findall(para["text"])  # full match strings including outer quotes

        if not quotes:
            _log(f"WARNING {para_id}: no dialogue text found, skipping dialogue mood")
            with lock3:
                para["dialogue_moods"] = []
                para["quote_mood_errors"] = []
                para["dialogue_mood_error"] = True
                if para_id not in dialogue_mood_errors:
                    dialogue_mood_errors.append(para_id)
                done3[0] += 1
                if done3[0] % 10 == 0 or done3[0] == total3:
                    eta = _format_eta(pass3_start, done3[0], total3)
                    _progress(done3[0], total3, f"{_dots(done3[0], total3)} [Pass 3/3] Dialogue mood: {done3[0]}/{total3}{eta}...")
                _checkpoint_write(paragraphs_path, project_root, paragraphs_doc, complete=False)
            return

        moods = []
        mood_errors = []
        narration_text = strip_dialogue(para["text"])

        for qi, quote_str in enumerate(quotes):
            speaker = (speakers_list[qi] if qi < len(speakers_list) else "").strip() or "Unknown Speaker"
            inner = quote_str.strip('"\u201c\u201d')

            if speaker == "NARRATOR":
                # Scare quote — derive narrator delivery from the surrounding narration
                user_msg = (
                    f"PASSAGE CONTEXT:\n{context_text}\n\n"
                    f"TARGET PARAGRAPH:\n{narration_text}\n\n"
                    "Identify the spoken emotional sentiment and delivery of the given narration "
                    "portion of the paragraph. Ignore any dialogue when describing the emotional delivery:"
                )
            else:
                user_msg = (
                    f"PASSAGE CONTEXT:\n{context_text}\n\n"
                    f"Identify the correct delivery emotion of the following dialogue as spoken by {speaker}.\n\n"
                    f"DIALOGUE:\n{inner}"
                )

            mood = ""
            try:
                mood, raw, llm_mode, llm_tool_call_observed = call_sentiment(
                    client, runtime, system_prompt, user_msg, max_tokens
                )
                if not mood:
                    raw_tail = raw[-1024:] if len(raw) > 1024 else raw
                    job = f"{para_id} q{qi}"
                    _log(f"Begin model did not return a dialogue mood: {job} : Extract Temperament\n{inner}\n{raw_tail}\nEnd model did not return a dialogue mood: {job} : Extract Temperament")
            except Exception as e:
                raw_tail = str(e)[-1024:]
                job = f"{para_id} q{qi}"
                _log(f"Begin API call failed: {job} : Extract Temperament\n{inner}\n{raw_tail}\nEnd API call failed: {job} : Extract Temperament")

            moods.append(mood)
            mood_errors.append(not bool(mood))

        with lock3:
            para["dialogue_moods"] = moods
            para["quote_mood_errors"] = mood_errors
            para["dialogue_mood_error"] = any(mood_errors)
            if para.get("dialogue_mood_error"):
                if para_id not in dialogue_mood_errors:
                    dialogue_mood_errors.append(para_id)
            else:
                if para_id in dialogue_mood_errors:
                    dialogue_mood_errors.remove(para_id)
            for qi, mood in enumerate(moods):
                if mood:
                    spk = (speakers_list[qi] if qi < len(speakers_list) else "Unknown Speaker")
                    _log(f"[{para_id}] Dialogue mood q{qi} ({spk}): {mood}")
            done3[0] += 1
            if done3[0] % 10 == 0 or done3[0] == total3:
                eta = _format_eta(pass3_start, done3[0], total3)
                _progress(done3[0], total3, f"[Pass 3/3] Dialogue mood: {done3[0]}/{total3}{eta}...")
            _checkpoint_write(paragraphs_path, project_root, paragraphs_doc, complete=False)

    if total3 > 0:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(process_pass3, item) for item in dialogue_paras_p3]
            for future in as_completed(futures):
                future.result()

    # ── Save final results ─────────────────────────────────────────────────────
    # Collect errors from all paragraphs (includes any carried over from prior runs).
    all_tone_errors = [p["id"] for p in paragraphs if p.get("temperament_error")]
    all_dialogue_mood_errors = [p["id"] for p in paragraphs if p.get("dialogue_mood_error")]
    _checkpoint_write(paragraphs_path, project_root, paragraphs_doc, complete=True)

    _log(f"[Pass 3/3] Done. {total3 - len(dialogue_mood_errors)}/{total3} succeeded.")
    _log(
        f"All passes complete. "
        f"Tone errors: {len(all_tone_errors)}. "
        f"Dialogue mood errors: {len(all_dialogue_mood_errors)}."
    )
    if all_tone_errors:
        _log(f"Tone error paragraphs: {', '.join(all_tone_errors)}")
    if all_dialogue_mood_errors:
        _log(f"Dialogue mood error paragraphs: {', '.join(all_dialogue_mood_errors)}")


def _atomic_write(path: str, data: dict):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def _persist_paragraphs_doc(path: str | None, project_root: str | None, paragraphs_doc: dict):
    store = open_project_script_store(project_root)
    try:
        store.replace_project_document("paragraphs", paragraphs_doc, reason="extract_temperament", wait=True)
    finally:
        store.stop()


def _checkpoint_write(path: str | None, project_root: str | None, paragraphs_doc: dict, complete: bool):
    paragraphs = paragraphs_doc.get("paragraphs", [])
    paragraphs_doc["temperament_extraction_complete"] = complete
    paragraphs_doc["temperament_errors"] = [
        p["id"] for p in paragraphs if p.get("temperament_error")
    ]
    paragraphs_doc["dialogue_mood_errors"] = [
        p["id"] for p in paragraphs if p.get("dialogue_mood_error")
    ]
    _persist_paragraphs_doc(path, project_root, paragraphs_doc)


if __name__ == "__main__":
    main()
