import os

_PROMPTS_FILE = os.path.join(os.path.dirname(__file__), "..", "default_prompts.txt")


def load_default_prompts():
    """Read default_prompts.txt from disk and return (system_prompt, user_prompt).

    Re-reads on every call so edits are picked up without restarting the app.
    """
    try:
        with open(_PROMPTS_FILE, "r", encoding="utf-8") as f:
            raw = f.read()
    except FileNotFoundError:
        raise RuntimeError(
            f"default_prompts.txt not found at {os.path.abspath(_PROMPTS_FILE)}. "
            "This file is required for LLM prompt defaults."
        )

    parts = raw.split("---SEPARATOR---", maxsplit=1)
    if len(parts) != 2:
        raise RuntimeError(
            "default_prompts.txt is malformed: expected exactly one '---SEPARATOR---' delimiter."
        )

    return parts[0].strip(), parts[1].strip()


# Cached at import time — used by generate_script.py (subprocess, fresh each run)
DEFAULT_SYSTEM_PROMPT, DEFAULT_USER_PROMPT = load_default_prompts()


DEFAULT_TEMPERAMENT_EXTRACTION_PROMPT = """\
You are a narrative tone specialist helping to produce an audiobook. \
Your sole task is to identify the emotional sentiment and delivery style \
of a given paragraph of prose.

You will receive:
  - A passage of text from the book (context window centred on the target paragraph)
  - The target paragraph (or dialogue lines) you must evaluate

Rules:
  - Respond with a single concise sentence describing how the reader should emotionally \
deliver the paragraph (e.g. "Read with quiet melancholy and a slow, deliberate pace").
  - You MUST call the identify_sentiment tool with your answer. Do not respond in plain text.\
"""

DEFAULT_DIALOGUE_IDENTIFICATION_PROMPT = """\
You are a dialogue attribution specialist helping to build an audiobook script. \
Your sole task is to identify who speaks a given piece of dialogue based on the surrounding narrative.

You will receive:
  - A passage of text from the book (context window centred on the target paragraph)
  - A list of character names already encountered in the story
  - The quoted dialogue lines from the target paragraph

Rules:
  - One paragraph = one speaker. Do not split attribution.
  - If a known character fits the context clues (e.g. "he said", "Sarah replied", speech patterns), \
use their exact name from the list.
  - If the dialogue belongs to a character not yet seen, provide that character's name.
  - Use NARRATOR only for clearly unvoiced internal thoughts printed inside quotation marks \
(a rare edge case).
  - You MUST call the identify_dialogue tool with your answer. Do not respond in plain text.\
"""
