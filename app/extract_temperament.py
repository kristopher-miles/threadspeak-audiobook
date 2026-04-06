#!/usr/bin/env python3
"""
Extract emotional temperament (mood/delivery) for all paragraphs using LLM tool use.

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

All results are written back into paragraphs.json atomically after each pass.
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
QUOTE_RE = re.compile(r'["\u201c][^"\u201d]{2,}["\u201d]', re.DOTALL)

IDENTIFY_SENTIMENT_TOOL = {
    "type": "function",
    "function": {
        "name": "identify_sentiment",
        "description": "Record the emotional delivery mood for this paragraph.",
        "parameters": {
            "type": "object",
            "properties": {
                "mood": {
                    "type": "string",
                    "description": (
                        "A single concise sentence describing how the reader should emotionally "
                        "deliver this paragraph (e.g. 'Read with quiet melancholy and a slow, "
                        "deliberate pace'). Must be one sentence only."
                    ),
                }
            },
            "required": ["mood"],
        },
    },
}

DEFAULT_SYSTEM_PROMPT = (
    "You are a narrative tone specialist helping to produce an audiobook. "
    "Your sole task is to identify the emotional sentiment and delivery style "
    "of a given paragraph of prose. "
    "You MUST call the identify_sentiment tool with your answer. Do not respond in plain text."
)


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
    return f" (~{int(remaining_seconds // 60)}m {int(remaining_seconds % 60)}s remaining)"


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


def strip_dialogue(text: str) -> str:
    """Replace all quoted dialogue runs with [dialogue ignored]."""
    return QUOTE_RE.sub("[dialogue ignored]", text)


def extract_dialogue_only(text: str) -> str:
    """Return only the quoted dialogue portions joined by newlines."""
    return "\n".join(QUOTE_RE.findall(text))


def call_sentiment(client, model_name: str, system_prompt: str, user_msg: str, max_tokens: int) -> str:
    """
    Call the LLM with identify_sentiment tool forced.
    Streams the response and exits as soon as the first complete tool call JSON arrives,
    preventing local models from looping and re-emitting the same tool call.
    Returns the mood string, or empty string on failure (caller handles logging).
    """
    tool_call_args = ""
    reasoning_content = ""

    stream = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_msg},
        ],
        tools=[IDENTIFY_SENTIMENT_TOOL],
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
        if delta.tool_calls:
            frag = delta.tool_calls[0].function.arguments
            if frag:
                tool_call_args += frag
        # Stop as soon as we have parseable JSON — don't let the model loop.
        if tool_call_args:
            try:
                args = json.loads(tool_call_args)
                stream.close()
                return (args.get("mood") or "").strip()
            except json.JSONDecodeError:
                pass

    # Fallback: try parsing whatever tool-call JSON we accumulated.
    if tool_call_args:
        try:
            args = json.loads(tool_call_args)
            return (args.get("mood") or "").strip()
        except json.JSONDecodeError:
            pass

    # Final fallback: Nemotron/XML models emit the call in reasoning_content.
    if reasoning_content:
        m = re.search(r"<parameter=mood>\s*(.*?)\s*</parameter>", reasoning_content, re.DOTALL | re.IGNORECASE)
        if m:
            return m.group(1).strip()

    return ""


def main():
    if len(sys.argv) < 3:
        _log("Usage: extract_temperament.py <paragraphs_path> <config_path>")
        sys.exit(1)

    paragraphs_path = sys.argv[1]
    config_path = sys.argv[2]

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

    system_prompt = (prompts.get("temperament_extraction_system_prompt") or "").strip() or DEFAULT_SYSTEM_PROMPT

    tts_cfg = config.get("tts") or {}
    workers = max(1, int(tts_cfg.get("parallel_workers", 1) or 1))

    _log(f"Model: {model_name}  |  Context budget: {context_budget} chars  |  Max tokens: {max_tokens}  |  Workers: {workers}")

    client = OpenAI(base_url=base_url, api_key=api_key, timeout=600)

    all_narration_paras = [(i, p) for i, p in enumerate(paragraphs) if not p.get("has_dialogue")]
    all_dialogue_paras  = [(i, p) for i, p in enumerate(paragraphs) if p.get("has_dialogue")]

    tone_errors: list[str] = []
    dialogue_mood_errors: list[str] = []

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
        context_text = build_context_window(paragraphs, idx, context_budget)

        user_msg = (
            f"PASSAGE CONTEXT:\n{context_text}\n\n"
            f"TARGET PARAGRAPH:\n{para['text']}\n\n"
            "Use the identify_sentiment tool to identify the spoken emotional "
            "sentiment and delivery of the given paragraph."
        )

        mood = ""
        try:
            mood = call_sentiment(client, model_name, system_prompt, user_msg, max_tokens)
            if not mood:
                _log(f"ERROR {para_id}: model did not return a mood")
        except Exception as e:
            _log(f"ERROR {para_id}: API call failed — {e}")

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
                _progress(done1[0], total1, f"[Pass 1/3] Narrator tone: {done1[0]}/{total1}{eta}...")
            if done1[0] % 10 == 0:
                _atomic_write(paragraphs_path, paragraphs_doc)

    if total1 > 0:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(process_pass1, item) for item in narration_paras]
            for future in as_completed(futures):
                future.result()

    _log(f"[Pass 1/3] Done. {total1 - len(tone_errors)}/{total1} succeeded.")
    _atomic_write(paragraphs_path, paragraphs_doc)

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
        context_text = build_context_window(paragraphs, idx, context_budget)
        narration_text = strip_dialogue(para["text"])

        user_msg = (
            f"PASSAGE CONTEXT:\n{context_text}\n\n"
            f"TARGET PARAGRAPH:\n{narration_text}\n\n"
            "Identify the spoken emotional sentiment and delivery of the given narration "
            "portion of the paragraph. Ignore any dialogue when describing the emotional delivery:"
        )

        mood = ""
        try:
            mood = call_sentiment(client, model_name, system_prompt, user_msg, max_tokens)
            if not mood:
                _log(f"ERROR {para_id}: model did not return a narrator mood")
        except Exception as e:
            _log(f"ERROR {para_id}: API call failed — {e}")

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
                _progress(done2[0], total2, f"[Pass 2/3] Narrator tone (dialogue): {done2[0]}/{total2}{eta}...")
            if done2[0] % 10 == 0:
                _atomic_write(paragraphs_path, paragraphs_doc)

    if total2 > 0:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(process_pass2, item) for item in dialogue_paras_p2]
            for future in as_completed(futures):
                future.result()

    _log(f"[Pass 2/3] Done. {total2 - sum(1 for e in tone_errors if e in [p['id'] for _, p in dialogue_paras_p2])}/{total2} succeeded.")
    _atomic_write(paragraphs_path, paragraphs_doc)

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
        context_text = build_context_window(paragraphs, idx, context_budget)
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
                    _progress(done3[0], total3, f"[Pass 3/3] Dialogue mood: {done3[0]}/{total3}{eta}...")
                if done3[0] % 10 == 0:
                    _atomic_write(paragraphs_path, paragraphs_doc)
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
                mood = call_sentiment(client, model_name, system_prompt, user_msg, max_tokens)
                if not mood:
                    _log(f"ERROR {para_id} q{qi}: model did not return a dialogue mood")
            except Exception as e:
                _log(f"ERROR {para_id} q{qi}: API call failed — {e}")

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
            if done3[0] % 10 == 0:
                _atomic_write(paragraphs_path, paragraphs_doc)

    if total3 > 0:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(process_pass3, item) for item in dialogue_paras_p3]
            for future in as_completed(futures):
                future.result()

    # ── Save final results ─────────────────────────────────────────────────────
    # Collect errors from all paragraphs (includes any carried over from prior runs).
    all_tone_errors = [p["id"] for p in paragraphs if p.get("temperament_error")]
    all_dialogue_mood_errors = [p["id"] for p in paragraphs if p.get("dialogue_mood_error")]
    paragraphs_doc["temperament_extraction_complete"] = True
    paragraphs_doc["temperament_errors"] = all_tone_errors
    paragraphs_doc["dialogue_mood_errors"] = all_dialogue_mood_errors
    _atomic_write(paragraphs_path, paragraphs_doc)

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


if __name__ == "__main__":
    main()
