import os
import sys
import json
import re
import argparse
from openai import OpenAI
from review_prompts import REVIEW_SYSTEM_PROMPT, REVIEW_USER_PROMPT
from generate_script import clean_json_string, repair_json_array, salvage_json_entries
from script_store import load_script_document, save_script_document
from task_checkpoint import build_signature, clear_checkpoint, load_checkpoint, save_checkpoint


CHECKPOINT_PATH = os.path.join(os.path.dirname(__file__), "..", "script_review_checkpoint.json")


def _is_section_break(text):
    """Check if text looks like a chapter heading or section title."""
    stripped = text.strip()
    # "CHAPTER ONE", "CHAPTER II", "Chapter Three", etc.
    if re.match(r'(?i)^chapter\b', stripped):
        return True
    # All-caps short text = likely a title ("A SCANDAL IN BOHEMIA", "THE RED-HEADED LEAGUE")
    if stripped == stripped.upper() and len(stripped) < 80 and stripped.isascii():
        return True
    return False


def merge_consecutive_narrators(entries, max_merged_length=800):
    """Merge consecutive NARRATOR entries that share the same instruct value.

    Skips merging across section/chapter breaks. Caps merged text at
    max_merged_length characters to avoid creating overly long TTS entries.
    """
    if not entries:
        return entries, 0

    merged = []
    merges = 0
    i = 0
    while i < len(entries):
        entry = entries[i]
        entry_chapter = entry.get("chapter")

        if entry.get("speaker") != "NARRATOR" or _is_section_break(entry.get("text", "")):
            merged.append(entry)
            i += 1
            continue

        # Start a narrator run — accumulate consecutive NARRATORs with same instruct
        combined_text = entry["text"]
        instruct = entry.get("instruct", "")
        run_count = 1
        j = i + 1

        while j < len(entries):
            next_entry = entries[j]
            if next_entry.get("speaker") != "NARRATOR":
                break
            if next_entry.get("instruct", "") != instruct:
                break
            if next_entry.get("chapter") != entry_chapter:
                break
            if _is_section_break(next_entry.get("text", "")):
                break
            candidate = combined_text + " " + next_entry["text"]
            if len(candidate) > max_merged_length:
                break
            combined_text = candidate
            run_count += 1
            j += 1

        merged_entry = {
            "speaker": "NARRATOR",
            "text": combined_text,
            "instruct": instruct
        }
        if entry_chapter:
            merged_entry["chapter"] = entry_chapter
        merged.append(merged_entry)
        if run_count > 1:
            merges += run_count - 1
        i = j

    return merged, merges


def build_review_batches(entries, batch_size):
    batches = []
    current_batch = []
    current_chapter = None

    for entry in entries:
        entry_chapter = entry.get("chapter") or None
        if current_batch and (entry_chapter != current_chapter or len(current_batch) >= batch_size):
            batches.append({
                "entries": current_batch,
                "chapter": current_chapter,
            })
            current_batch = []

        if not current_batch:
            current_chapter = entry_chapter
        current_batch.append(entry)

    if current_batch:
        batches.append({
            "entries": current_batch,
            "chapter": current_chapter,
        })

    return batches


def apply_batch_chapter(corrected_entries, chapter):
    if not chapter:
        for entry in corrected_entries:
            entry.pop("chapter", None)
        return corrected_entries

    for entry in corrected_entries:
        entry["chapter"] = chapter
    return corrected_entries


def review_batch(client, model_name, batch_entries, batch_num, total_batches,
                 previous_tail=None, source_context=None, max_retries=2,
                 system_prompt=None, user_prompt_template=None,
                 max_tokens=8000, temperature=0.4, top_p=0.8, top_k=20,
                 min_p=0, presence_penalty=0.0, banned_tokens=None):
    """Send a batch of script entries through the LLM for review and correction."""
    sys_prompt = system_prompt or REVIEW_SYSTEM_PROMPT
    usr_template = user_prompt_template or REVIEW_USER_PROMPT

    # Build context
    context_parts = []
    context_parts.append(f"Batch {batch_num} of {total_batches}.")

    if previous_tail:
        context_parts.append("\nPrevious batch ended with:")
        for entry in previous_tail:
            context_parts.append(json.dumps(entry, ensure_ascii=False))

    # Mode 2 future: inject source text
    if source_context:
        context_parts.append(f"\nORIGINAL SOURCE TEXT (for reference):\n{source_context}")

    context = "\n".join(context_parts)
    batch_json = json.dumps(batch_entries, indent=2, ensure_ascii=False)
    user_prompt = usr_template.format(context=context, batch=batch_json)

    for attempt in range(max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=temperature,
                top_p=top_p,
                presence_penalty=presence_penalty,
                max_tokens=max_tokens,
                extra_body={
                    k: v for k, v in {
                        "top_k": top_k,
                        "min_p": min_p,
                        "banned_tokens": banned_tokens if banned_tokens else None,
                    }.items() if v is not None
                }
            )

            choice = response.choices[0]
            text = choice.message.content.strip()
            finish_reason = choice.finish_reason
            usage = getattr(response, 'usage', None)

            # Log raw response
            log_dir = os.path.join(os.path.dirname(__file__), "..", "logs")
            os.makedirs(log_dir, exist_ok=True)
            log_path = os.path.join(log_dir, "review_responses.log")
            with open(log_path, "a", encoding="utf-8") as lf:
                lf.write(f"\n{'='*80}\n")
                lf.write(f"BATCH {batch_num}/{total_batches} | attempt {attempt + 1} | finish_reason={finish_reason}\n")
                if usage:
                    lf.write(f"tokens: prompt={getattr(usage, 'prompt_tokens', '?')} completion={getattr(usage, 'completion_tokens', '?')}\n")
                lf.write(f"{'─'*80}\n")
                lf.write(text)
                lf.write(f"\n{'='*80}\n")

            print(f"  finish_reason={finish_reason}", end="")
            if usage:
                print(f" | tokens: prompt={getattr(usage, 'prompt_tokens', '?')} completion={getattr(usage, 'completion_tokens', '?')}", end="")
            print()

            if finish_reason == "length":
                print(f"  WARNING: Response was truncated (hit max_tokens={max_tokens}). Consider increasing max_tokens or reducing batch size.")

        except Exception as e:
            print(f"Error calling LLM API (attempt {attempt + 1}): {e}")
            if attempt < max_retries:
                continue
            return None

        # Clean and parse JSON response
        json_text = clean_json_string(text)

        if not json_text:
            print(f"Warning: Could not find JSON array in batch {batch_num} response (attempt {attempt + 1})")
            if attempt < max_retries:
                print("Retrying...")
                continue
            print(f"Response preview: {text[:300]}...")
            return None

        entries = repair_json_array(json_text)

        if entries and len(entries) > 0:
            if attempt > 0:
                print(f"  Succeeded on retry {attempt + 1}")
            return entries

        print(f"Warning: Could not parse batch {batch_num} response as JSON (attempt {attempt + 1})")

        if attempt < max_retries:
            print("Retrying...")

        # Last resort
        salvaged = salvage_json_entries(json_text)
        if salvaged:
            print(f"Regex-salvaged {len(salvaged)} entries from malformed response")
            return salvaged

    return None


def normalize_text(text):
    """Normalize text for comparison: lowercase, collapse whitespace, strip punctuation."""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def check_text_loss(original_entries, corrected_entries, threshold=0.95):
    """Check if corrected entries lost significant text from the original.

    Returns (passed, original_text, corrected_text, ratio).
    passed is True if the corrected text covers at least `threshold` of the original words.
    """
    orig_words = []
    for e in original_entries:
        orig_words.extend(normalize_text(e.get("text", "")).split())

    corr_words = []
    for e in corrected_entries:
        corr_words.extend(normalize_text(e.get("text", "")).split())

    if not orig_words:
        return True, "", "", 1.0

    # Check what fraction of original words appear in corrected text
    orig_joined = " ".join(orig_words)
    corr_joined = " ".join(corr_words)

    # Simple word-level coverage: count original words that appear in corrected
    orig_word_set = set(orig_words)
    corr_word_set = set(corr_words)

    # For a more robust check, compare total word counts
    # (a dropped sentence means fewer total words)
    ratio = len(corr_words) / len(orig_words) if orig_words else 1.0

    passed = ratio >= threshold
    return passed, orig_joined, corr_joined, ratio


def diff_entries(original, corrected):
    """Compare original and corrected entries, return a summary dict."""
    stats = {
        "text_changed": 0,
        "speaker_changed": 0,
        "instruct_changed": 0,
        "entries_original": len(original),
        "entries_corrected": len(corrected),
    }

    # Compare entry-by-entry up to the shorter length
    compare_len = min(len(original), len(corrected))
    for i in range(compare_len):
        orig = original[i]
        corr = corrected[i]
        if orig.get("text") != corr.get("text"):
            stats["text_changed"] += 1
        if orig.get("speaker") != corr.get("speaker"):
            stats["speaker_changed"] += 1
        if orig.get("instruct") != corr.get("instruct"):
            stats["instruct_changed"] += 1

    return stats


def flatten_pending_batches(batches, start_index):
    pending = []
    for batch_info in batches[start_index:]:
        pending.extend(batch_info["entries"])
    return pending


def main():
    parser = argparse.ArgumentParser(description="Review and fix annotated audiobook script")
    parser.add_argument("--source", help="Path to original source text for comparison (mode 2, not yet implemented)")
    args = parser.parse_args()

    # Locate annotated_script.json
    script_path = os.path.join(os.path.dirname(__file__), "..", "annotated_script.json")
    if not os.path.exists(script_path):
        print("Error: annotated_script.json not found. Generate a script first.")
        sys.exit(1)

    script_document = load_script_document(script_path)
    checkpoint = load_checkpoint(CHECKPOINT_PATH)
    checkpoint_original_entries = checkpoint.get("original_entries") if isinstance(checkpoint, dict) else None
    entries = checkpoint_original_entries if isinstance(checkpoint_original_entries, list) else script_document["entries"]

    print(f"Loaded {len(entries)} script entries for review")

    # Load source text if provided (mode 2 prep)
    source_text = None
    if args.source:
        if os.path.exists(args.source):
            with open(args.source, "r", encoding="utf-8") as f:
                source_text = f.read()
            print(f"Loaded source text: {len(source_text)} chars")
        else:
            print(f"Warning: Source file not found: {args.source}")

    # Load config
    config_path = os.path.join(os.path.dirname(__file__), "config.json")
    config = {}
    if os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
        except Exception as e:
            print(f"Warning: Failed to load config.json: {e}")
    else:
        print("Warning: config.json not found. Using defaults.")

    llm_config = config.get("llm", {})
    base_url = llm_config.get("base_url", "http://localhost:11434/v1").rstrip("/")
    if not base_url.endswith("/v1"):
        base_url += "/v1"
    api_key = llm_config.get("api_key", "local")
    model_name = llm_config.get("model_name", "local-model")

    # Load custom review prompts or use defaults from review_prompts.txt
    prompts_config = config.get("prompts", {})
    review_sys = prompts_config.get("review_system_prompt") or REVIEW_SYSTEM_PROMPT
    review_usr = prompts_config.get("review_user_prompt") or REVIEW_USER_PROMPT

    generation_config = config.get("generation", {})
    batch_size = generation_config.get("review_batch_size", 25)
    max_tokens = generation_config.get("max_tokens", 8000)
    temperature = generation_config.get("temperature", 0.4)
    top_p = generation_config.get("top_p", 0.8)
    top_k = generation_config.get("top_k", 20)
    min_p = generation_config.get("min_p", 0)
    presence_penalty = generation_config.get("presence_penalty", 0.0)
    banned_tokens = generation_config.get("banned_tokens", [])

    print(f"Connecting to: {base_url}")
    print(f"Using model: {model_name}")
    print(f"Batch size: {batch_size} entries, Max tokens: {max_tokens}")
    if banned_tokens:
        print(f"Banned tokens: {banned_tokens}")

    client = OpenAI(base_url=base_url, api_key=api_key)

    # Split entries into batches
    batches = build_review_batches(entries, batch_size)

    total_batches = len(batches)
    print(f"Split into {total_batches} batches of ~{batch_size} entries")

    chunks_path = os.path.join(os.path.dirname(__file__), "..", "chunks.json")
    if os.path.exists(chunks_path):
        os.remove(chunks_path)
        print("Cleared old chunks.json")

    signature = build_signature({
        "task": "script_review",
        "entries": entries,
        "dictionary": script_document.get("dictionary", []),
        "review": {
            "batch_size": batch_size,
            "chapter_boundaries": [
                batch_info["chapter"]
                for batch_info in batches
            ],
        },
    })

    all_corrected = []
    total_stats = {
        "text_changed": 0,
        "speaker_changed": 0,
        "instruct_changed": 0,
        "entries_added": 0,
        "entries_removed": 0,
        "batches_failed": 0,
    }

    previous_tail = None
    previous_chapter = None
    start_batch_index = 0

    if checkpoint and checkpoint.get("task") == "script_review" and checkpoint.get("signature") == signature:
        checkpoint_corrected = checkpoint.get("all_corrected")
        checkpoint_stats = checkpoint.get("total_stats")
        checkpoint_batch_index = int(checkpoint.get("next_batch_index") or 0)
        if isinstance(checkpoint_corrected, list) and isinstance(checkpoint_stats, dict) and 0 < checkpoint_batch_index <= total_batches:
            all_corrected = checkpoint_corrected
            total_stats.update(checkpoint_stats)
            previous_tail = checkpoint.get("previous_tail")
            previous_chapter = checkpoint.get("previous_chapter")
            start_batch_index = checkpoint_batch_index
            print(f"Resuming prior review progress from batch {start_batch_index + 1}/{total_batches}")
        else:
            clear_checkpoint(CHECKPOINT_PATH)
    elif checkpoint:
        print("Discarding stale review checkpoint because the script or batch layout changed")
        clear_checkpoint(CHECKPOINT_PATH)

    if start_batch_index > 0:
        save_script_document(
            script_path,
            entries=all_corrected + flatten_pending_batches(batches, start_batch_index),
            dictionary=script_document["dictionary"],
        )

    for i, batch_info in enumerate(batches[start_batch_index:], start_batch_index + 1):
        batch = batch_info["entries"]
        batch_chapter = batch_info["chapter"]
        print(f"\nReviewing batch {i}/{total_batches} ({len(batch)} entries)...")
        if batch_chapter:
            print(f"  Chapter: {batch_chapter}")

        if batch_chapter != previous_chapter:
            previous_tail = None

        corrected = review_batch(
            client, model_name, batch, i, total_batches,
            previous_tail=previous_tail,
            source_context=None,  # Mode 2: would pass source text chunk here
            system_prompt=review_sys,
            user_prompt_template=review_usr,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            presence_penalty=presence_penalty,
            banned_tokens=banned_tokens
        )

        if corrected is None:
            print(f"  FAILED — keeping original entries for batch {i}")
            all_corrected.extend(batch)
            total_stats["batches_failed"] += 1
            previous_tail = batch[-2:] if len(batch) >= 2 else batch
            previous_chapter = batch_chapter
            save_script_document(
                script_path,
                entries=all_corrected + flatten_pending_batches(batches, i),
                dictionary=script_document["dictionary"],
            )
            save_checkpoint(
                CHECKPOINT_PATH,
                task="script_review",
                signature=signature,
                payload={
                    "original_entries": entries,
                    "total_batches": total_batches,
                    "next_batch_index": i,
                    "all_corrected": all_corrected,
                    "total_stats": total_stats,
                    "previous_tail": previous_tail,
                    "previous_chapter": previous_chapter,
                },
            )
            continue

        corrected = apply_batch_chapter(corrected, batch_chapter)

        # Text-loss safety check
        passed, orig_text, corr_text, ratio = check_text_loss(batch, corrected)
        if not passed:
            print(f"  WARNING: Text loss detected! Word ratio: {ratio:.2f} (threshold: 0.95)")
            print(f"  Original words: {len(orig_text.split())}, Corrected words: {len(corr_text.split())}")
            print(f"  Keeping original entries for batch {i} to prevent data loss.")
            all_corrected.extend(batch)
            total_stats["batches_failed"] += 1
            previous_tail = batch[-2:] if len(batch) >= 2 else batch
            previous_chapter = batch_chapter
            save_script_document(
                script_path,
                entries=all_corrected + flatten_pending_batches(batches, i),
                dictionary=script_document["dictionary"],
            )
            save_checkpoint(
                CHECKPOINT_PATH,
                task="script_review",
                signature=signature,
                payload={
                    "original_entries": entries,
                    "total_batches": total_batches,
                    "next_batch_index": i,
                    "all_corrected": all_corrected,
                    "total_stats": total_stats,
                    "previous_tail": previous_tail,
                    "previous_chapter": previous_chapter,
                },
            )
            continue

        # Diff stats
        stats = diff_entries(batch, corrected)
        entry_diff = len(corrected) - len(batch)

        if entry_diff > 0:
            total_stats["entries_added"] += entry_diff
        elif entry_diff < 0:
            total_stats["entries_removed"] += abs(entry_diff)

        total_stats["text_changed"] += stats["text_changed"]
        total_stats["speaker_changed"] += stats["speaker_changed"]
        total_stats["instruct_changed"] += stats["instruct_changed"]

        changes = stats["text_changed"] + stats["speaker_changed"] + stats["instruct_changed"]
        if changes > 0 or entry_diff != 0:
            print(f"  Changes: {stats['text_changed']} text, {stats['speaker_changed']} speaker, {stats['instruct_changed']} instruct", end="")
            if entry_diff > 0:
                print(f", +{entry_diff} entries (splits)")
            elif entry_diff < 0:
                print(f", {entry_diff} entries (merges)")
            else:
                print()
        else:
            print(f"  No changes")

        all_corrected.extend(corrected)
        previous_tail = corrected[-2:] if len(corrected) >= 2 else corrected
        previous_chapter = batch_chapter
        save_script_document(
            script_path,
            entries=all_corrected + flatten_pending_batches(batches, i),
            dictionary=script_document["dictionary"],
        )
        save_checkpoint(
            CHECKPOINT_PATH,
            task="script_review",
            signature=signature,
            payload={
                "original_entries": entries,
                "total_batches": total_batches,
                "next_batch_index": i,
                "all_corrected": all_corrected,
                "total_stats": total_stats,
                "previous_tail": previous_tail,
                "previous_chapter": previous_chapter,
            },
        )

    # Post-processing: merge consecutive NARRATOR entries with same instruct
    merge_narrators_enabled = generation_config.get("merge_narrators", False)
    narrator_merges = 0
    if merge_narrators_enabled:
        pre_merge_count = len(all_corrected)
        all_corrected, narrator_merges = merge_consecutive_narrators(all_corrected, max_merged_length=800)
        if narrator_merges > 0:
            print(f"\nPost-processing: merged {narrator_merges} consecutive narrator entries "
                  f"({pre_merge_count} -> {len(all_corrected)} entries)")
    else:
        print("\nNarrator merging: disabled (enable in Setup > Advanced)")

    # Write corrected script
    save_script_document(script_path, entries=all_corrected, dictionary=script_document["dictionary"])
    clear_checkpoint(CHECKPOINT_PATH)

    # Delete chunks.json so editor regenerates
    if os.path.exists(chunks_path):
        os.remove(chunks_path)
        print("Cleared old chunks.json")

    # Final summary
    total_changes = (total_stats["text_changed"] + total_stats["speaker_changed"] +
                     total_stats["instruct_changed"] + total_stats["entries_added"] +
                     total_stats["entries_removed"] + narrator_merges)

    print(f"\n{'='*60}")
    print(f"Review complete: {len(entries)} -> {len(all_corrected)} entries")
    print(f"  Text changed:    {total_stats['text_changed']}")
    print(f"  Speaker changed: {total_stats['speaker_changed']}")
    print(f"  Instruct changed:{total_stats['instruct_changed']}")
    print(f"  Entries added:   {total_stats['entries_added']}")
    print(f"  Entries removed: {total_stats['entries_removed']}")
    print(f"  Narrators merged:{narrator_merges}")
    if total_stats["batches_failed"] > 0:
        print(f"  Batches failed:  {total_stats['batches_failed']}")
    print(f"  Total changes:   {total_changes}")
    print(f"{'='*60}")

    if total_changes == 0:
        print("No issues found -- script looks clean.")
    else:
        print(f"Fixed {total_changes} issues across {total_batches} batches.")

    print(f"Output saved to: {script_path}")
    print("Task review completed successfully.")


if __name__ == "__main__":
    main()
